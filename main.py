import os
import logging
import tempfile
import asyncio
import base64
import uuid
import redis
import json
import speech_recognition as sr
import urllib.parse
import gtts
import aiohttp
import google.generativeai as genai
import google.generativeai.types as generation_types
import re
import bleach
import requests

from twelvedata import TDClient
from deep_translator import GoogleTranslator
from keywords import complex_keywords
from collections import Counter
from typing import Optional, List, Dict
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pydub import AudioSegment
from langdetect import detect
from PIL import Image
from io import BytesIO
from aiohttp import FormData
from datetime import datetime, timedelta
from together import Together
from typing import List, Dict
from typing import Union, Tuple
from stopwords import stop_words
from google.generativeai.types import generation_types
from googleapiclient.discovery import build
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Konfigurasi logger
logging.basicConfig(
    level=logging.INFO,  # Atur level logging ke INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Format pesan log
    handlers=[
        logging.StreamHandler()  # Output log ke console
    ]
)

# Inisialisasi logger
logger = logging.getLogger(__name__)

# Konstanta untuk batasan ukuran file
MAX_AUDIO_SIZE = 200 * 1024 * 1024  # 200MB

# Konfigurasi Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")  # Gunakan nilai default jika tidak ada di environment
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    redis_available = True
    logger.info(f"Koneksi Redis berhasil ke: {REDIS_URL}")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Gagal terhubung ke Redis di {REDIS_URL}: {e}")
    redis_client = None
    redis_available = False
except Exception as e:
    logger.error(f"Error tak terduga saat inisiasi Redis: {e}")
    redis_client = None
    redis_available = False

def check_required_settings():
    missing_vars = []
    required_vars = {
        'TELEGRAM_TOKEN': 'Token Telegram Bot',
        'MISTRAL_API_KEY': 'API Key Mistral',
        'GOOGLE_API_KEY': 'API Key Google',
        'TOGETHER_API_KEY': 'API Key Together',
        'IMGFOTO_API_KEY': 'API Key ImgFoto.host'
    }

    for var_name, var_desc in required_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"{var_desc} ({var_name})")
            logger.error(f"Error: {var_name} tidak ditemukan!")

    if missing_vars:
        print("Error: Variabel environment yang diperlukan tidak ditemukan:")
        for var in missing_vars:
            print(f"- {var}")
        return False

    logger.info("Semua konfigurasi yang diperlukan tersedia")
    return True

def sanitize_input(text: str) -> str:
    allowed_tags = ['b', 'i']
    allowed_attributes = {}
    cleaned_text = bleach.clean(text, tags=allowed_tags, attributes=allowed_attributes)
    return cleaned_text

# Environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")

async def chat_with_gemini(messages: List[Dict[str, str]]) -> str:
    try:
        # Konversi format pesan ke struktur Gemini yang valid
        history = []
        for msg in messages:
            # Sesuai dokumentasi: role harus "user" atau "model"
            role = "user" if msg["role"] == "user" else "model"
            history.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        # Mulai chat dengan riwayat yang sesuai
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40
        }
        chat = gemini_model.start_chat(history=history, generation_config=generation_config)

        # Kirim pesan terakhir
        response = chat.send_message(messages[-1]["content"])

        return response.text if response else "Maaf, tidak ada respons"

    except Exception as e:
        logger.error(f"Error in chat_with_gemini: {str(e)}")
        return "Terjadi kesalahan saat memproses permintaan"

# Konstanta konfigurasi
CHUNK_DURATION = 30  # Durasi chunk dalam detik
SPEECH_RECOGNITION_TIMEOUT = 30  # Timeout untuk speech recognition dalam detik
MAX_RETRIES = 5  # Jumlah maksimal percobaan untuk API calls
RETRY_DELAY = 5  # Delay antara percobaan ulang dalam detik
CONVERSATION_TIMEOUT = 36600  # 3600 detik = 1 jam
MAX_CONCURRENT_SESSIONS = 1000
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
MAX_CONVERSATION_MESSAGES_SIMPLE = 10
MAX_CONVERSATION_MESSAGES_MEDIUM = 50
MAX_CONVERSATION_MESSAGES_COMPLEX = 100
MAX_REQUESTS_PER_MINUTE = 15
client = Together()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Statistik penggunaan dipindahkan ke Redis

class AudioProcessingError(Exception):
    """Custom exception untuk error pemrosesan audio"""
    pass

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    welcome_text = """Halo! Saya PAIDI asisten Anda. Saya dapat:
    - Memproses pesan suara (termasuk yang panjang)
    - Menanggapi dengan suara
    - Membantu dengan berbagai tugas
    - Memproses gambar
    - Generate gambar (gunakan /gambar atau /image diikuti dengan prompt)
    - pengingat (gunakan /ingatkan 5 belanja *mengingatkan anda 5 menit kedepan untuk belanja)

    Kirim saya pesan atau catatan suara untuk memulai!"""
    await update.message.reply_text(welcome_text)

def split_message(text: str, max_length: int = 4096) -> List[str]:
    parts = []
    while len(text) > max_length:
        split_index = text.rfind("\n", 0, max_length)
        if split_index == -1:
            split_index = text.rfind(" ", 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index].strip())
        text = text[split_index:].strip()
    parts.append(text)
    return parts

async def get_bbands(symbol: str, interval: str = "1h", start_date: str = None, end_date: str = None) -> Optional[Dict]:
    """
    Mengambil data Bollinger Bands (BBANDS) dari TwelveData API.
    """
    from datetime import datetime, timedelta
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not api_key:
        logger.error("TWELVEDATA_API_KEY tidak ditemukan di environment variables.")
        return None

    url = f"https://api.twelvedata.com/bbands?symbol={symbol}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={api_key}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Cek apakah respons sukses

            data = response.json()
            if data.get("status") == "ok":
                return data.get("values", [{}])[0]  # Ambil data terbaru
            else:
                logger.error(f"Gagal mengambil data BBANDS: {data.get('message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error saat mengambil data BBANDS (percobaan {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)  # Tunggu sebelum mencoba lagi
            else:
                return None
        except Exception as e:
            logger.error(f"Error tak terduga saat mengambil data BBANDS: {e}")
            return None
    return None

async def get_macd(symbol: str, interval: str = "1h", start_date: str = None, end_date: str = None) -> Optional[Dict]:
    """
    Mengambil data MACD dari TwelveData API.
    """
    # Implementation of get_macd remains the same as in the provided code
    from datetime import datetime, timedelta
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not api_key:
        logger.error("TWELVEDATA_API_KEY tidak ditemukan di environment variables.")
        return None

    url = f"https://api.twelvedata.com/macd?symbol={symbol}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={api_key}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Cek apakah respons sukses

            data = response.json()
            if data.get("status") == "ok":
                values = data.get("values", [{}])[0]  # Ambil data terbaru
                return { # Return all MACD values
                    "macd": values.get('macd'),
                    "macd_signal": values.get('macd_signal'),
                    "macd_histogram": values.get('macd_histogram'),
                }
            else:
                logger.error(f"Gagal mengambil data MACD: {data.get('message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error saat mengambil data MACD (percobaan {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)  # Tunggu sebelum mencoba lagi
            else:
                return None
        except Exception as e:
            logger.error(f"Error tak terduga saat mengambil data MACD: {e}")
            return None
    return None

async def get_vwap(symbol: str, interval: str = "1h", start_date: str = None, end_date: str = None) -> Optional[Dict]:
    """
    Mengambil data VWAP dari TwelveData API.
    """
    # Implementation of get_vwap remains the same as in the provided code
    from datetime import datetime, timedelta
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not api_key:
        logger.error("TWELVEDATA_API_KEY tidak ditemukan di environment variables.")
        return None

    url = f"https://api.twelvedata.com/vwap?symbol={symbol}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={api_key}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Cek apakah respons sukses

            data = response.json()
            if data.get("status") == "ok":
                return data.get("values", [{}])[0]  # Ambil data terbaru
            else:
                logger.error(f"Gagal mengambil data VWAP: {data.get('message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error saat mengambil data VWAP (percobaan {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)  # Tunggu sebelum mencoba lagi
            else:
                return None
        except Exception as e:
            logger.error(f"Error tak terduga saat mengambil data VWAP: {e}")
            return None
async def get_rsi(symbol: str, interval: str = "1h", start_date: str = None, end_date: str = None) -> Optional[Dict]:
    """
    Mengambil data RSI dari TwelveData API.
    """
    from datetime import datetime, timedelta
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not api_key:
        logger.error("TWELVEDATA_API_KEY tidak ditemukan di environment variables.")
        return None

    url = f"https://api.twelvedata.com/rsi?symbol={symbol}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={api_key}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Cek apakah respons sukses

            data = response.json()
            if data.get("status") == "ok":
                return data.get("values", [{}])[0]  # Ambil data terbaru
            else:
                logger.error(f"Gagal mengambil data RSI: {data.get('message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error saat mengambil data RSI (percobaan {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)  # Tunggu sebelum mencoba lagi
            else:
                return None
        except Exception as e:
            logger.error(f"Error tak terduga saat mengambil data RSI: {e}")
            return None
    return None
    return None

async def get_stock_data(symbol: str, interval: str = "1h", outputsize: int = 30, start_date: str = None, end_date: str = None) -> Optional[Dict]:
    # Implementation of get_stock_data remains the same as in the provided code
    from functools import partial
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    td = TDClient(apikey=os.getenv("TWELVEDATA_API_KEY"))

    for attempt in range(MAX_RETRIES):
        try:
            time_series_func = partial(
                td.time_series,
                symbol=symbol,
                interval=interval,
                outputsize=outputsize,
                start_date=start_date,
                end_date=end_date,
                timezone="Asia/Bangkok"
            )

            loop = asyncio.get_event_loop()
            ts = await loop.run_in_executor(None, time_series_func)
            data = await loop.run_in_executor(None, ts.as_json)

            if data:
                logger.info(f"Data saham: {data}")
                return data
            return None
        except Exception as e:
            logger.error(f"Error fetching stock data (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
            else:
                return None
    return None


async def get_stock_data_with_indicators(symbol: str) -> Optional[Dict]:
    """
    Mengambil data saham beserta indikator teknis (BBANDS, MACD, VWAP).
    """
    # Implementation of get_stock_data_with_indicators remains the same as in the provided code
    try:
        bbands = await get_bbands(symbol)
        macd = await get_macd(symbol)
        vwap = await get_vwap(symbol)

        rsi = await get_rsi(symbol)
        stock_data = {
            "bbands": bbands,
            "macd": macd,
            "vwap": vwap,
            "rsi": rsi
        }
        return stock_data
    except Exception as e:
        logger.error(f"Error fetching stock data with indicators: {str(e)}")
        return None

def format_technical_indicators(stock_data: Dict) -> str:
    """
    Format semua indikator teknis dan data historis dalam bentuk yang mudah dibaca oleh Gemini.
    """
    # Implementation of format_technical_indicators remains the same as in the provided code
    if not stock_data:
        return "Tidak ada data indikator teknis yang tersedia."

    historical_data = ""
    if isinstance(stock_data, list):
        for entry in stock_data:
            historical_data += (
                f"Tanggal: {entry.get('datetime', 'Tidak tersedia')}\n"
                f"  - Open: {entry.get('open', 'Tidak tersedia')}\n"
                f"  - Close: {entry.get('close', 'Tidak tersedia')}\n"
                f"  - High: {entry.get('high', 'Tidak tersedia')}\n"
                f"  - Low: {entry.get('low', 'Tidak tersedia')}\n"
                #f"  - Volume: {entry.get('volume', 'Tidak tersedia')}\n\n"
            )

    bbands = stock_data.get('bbands')
    macd = stock_data.get('macd')
    vwap = stock_data.get('vwap')
    rsi = stock_data.get('rsi')

    indicators = (
        f"1. **Bollinger Bands (BBANDS):**\n"
        f"   - Upper Band: {bbands.get('upper_band', 'Tidak tersedia') if bbands else 'Tidak tersedia'}\n"
        f"   - Middle Band: {bbands.get('middle_band', 'Tidak tersedia') if bbands else 'Tidak tersedia'}\n"
        f"   - Lower Band: {bbands.get('lower_band', 'Tidak tersedia') if bbands else 'Tidak tersedia'}\n\n"
        f"2. **Moving Average Convergence Divergence (MACD):** (Note: Signal and Histogram values may not always be available from the API)\n"
        f"   - MACD: {macd.get('macd', 'Tidak tersedia') if macd else 'Tidak tersedia'}\n"
        f"   - Signal: {macd.get('macd_signal', 'Tidak tersedia') if macd else 'Tidak tersedia'}\n"
        f"   - Histogram: {macd.get('macd_histogram', 'Tidak tersedia') if macd else 'Tidak tersedia'}\n"
        f"3. **Volume Weighted Average Price (VWAP):** {vwap.get('vwap', 'Tidak tersedia') if vwap else 'Tidak tersedia'}\n"
        f"4. **Relative Strength Index (RSI):** {rsi.get('rsi', 'Tidak tersedia') if rsi else 'Tidak tersedia'}\n"
    )

    return historical_data + indicators

def format_historical_data(historical_data: List[Dict]) -> str:
    """
    Format data historis saham dalam bentuk yang mudah dibaca oleh Gemini.
    """
    # Implementation of format_historical_data remains the same as in the provided code
    formatted_data = ""
    for entry in historical_data:
        formatted_data += (
            f"Tanggal: {entry.get('datetime', 'Tidak tersedia')}\n"
            f"  - Open: {entry.get('open', 'Tidak tersedia')}\n"
            f"  - Close: {entry.get('close', 'Tidak tersedia')}\n"
            f"  - High: {entry.get('high', 'Tidak tersedia')}\n"
            f"  - Low: {entry.get('low', 'Tidak tersedia')}\n"
            #f"  - Volume: {entry.get('volume', 'Tidak tersedia')}\n\n"
        )
    return formatted_data


async def handle_stock_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    processing_msg = None
    try:
        session = json.loads(redis_client.get(f"session:{chat_id}"))
        message_text = update.message.text or ""
        symbol = message_text.replace("/harga", "").strip()

        if not symbol:
            await update.message.reply_text("Mohon berikan simbol saham. Contoh: /harga AAPL")
            return

        processing_msg = await update.message.reply_text("🔄 Sedang mengambil dan menganalisis data saham...")

        stock_data = await get_stock_data_with_indicators(symbol)
        if not stock_data or not isinstance(stock_data, dict):
            await update.message.reply_text("Maaf, tidak dapat mengambil data saham. Silakan coba lagi.")
            return

        historical_data = await get_stock_data(symbol)
        if not historical_data:
            await update.message.reply_text("Maaf, tidak dapat mengambil data historis saham. Silakan coba lagi.")
            return

        stock_info = (
            f"Data untuk {symbol}:\n"
            f"{format_technical_indicators(stock_data)}\n"
            f"Data Historis:\n"
            f"{format_historical_data(historical_data)}"
        )
        prompt = (
            f"Berikut adalah data terkini untuk pasangan mata uang {symbol}:\n{stock_info}\n\n"
             "Lakukan analisis mendalam terhadap performa pasangan mata uang ini dengan cakupan berikut:\n\n"
             "1. **Tren Harga:**\n"
             "   - Apakah terdapat tren bullish atau bearish dalam jangka pendek dan panjang?\n"
            "   - Apakah Market Structure tren bullish atau bearish ?\n"
             "   - Identifikasi pola harga yang menonjol seperti support, resistance, dan breakout.\n\n"
              "2. **Indikator Teknis:**\n"
              "   - Analisis pergerakan menggunakan Bollinger Bands untuk melihat volatilitas.\n"
             "   - MACD untuk mengidentifikasi momentum tren.\n"
             "   - Evaluasi penggunaan VWAP untuk menentukan nilai harga yang wajar.\n\n"
             "3. **Saran:**\n"
             "   - Berdasarkan analisis di atas, apakah ini waktu yang tepat untuk **buy** atau **sell**?\n"
              "   - Berikan alasan berbasis data yang kuat, termasuk potensi entry dan exit level.\n\n"
             "Gunakan bahasa yang profesional namun tetap mudah dipahami oleh trader forex dengan berbagai tingkat pengalaman."
            )

        response = await process_with_gemini([{"role": "user", "content": prompt}])

        if response:
            filtered_response = await filter_text(response)
            response_parts = split_message(filtered_response)

            for part in response_parts:
                await update.message.reply_text(part)

            logger.info(f"Menambahkan analisis saham ke sesi untuk chat_id {chat_id}: {filtered_response[:100]}...") # Log before adding to session
            # Tambahkan respons asisten ke sesi dan update session setelah mengirim semua parts ke user
            session['messages'].append({"role": "assistant", "content": filtered_response})
            await update_session(chat_id, {"role": "assistant", "content": filtered_response})
            logger.info(f"Selesai menambahkan analisis saham ke sesi untuk chat_id {chat_id}") # Log after adding to session
        else:
            await update.message.reply_text("Maaf, terjadi kesalahan saat memproses data saham.")

    except Exception as e:
        logger.error(f"Error in handle_stock_request: {e}")
        await update.message.reply_text("Terjadi kesalahan saat memproses permintaan saham.")

    finally:
        if processing_msg:
            await processing_msg.delete()


async def determine_conversation_complexity(messages: List[Dict[str, str]], session: Dict, previous_complexity: str = "simple") -> str:
    # Implementation of determine_conversation_complexity remains the same as in the provided code
    user_messages = [msg.get('content', '') for msg in messages if msg.get('role') == 'user']
    user_text = " ".join(user_messages).lower()
    latest_message = user_messages[-1] if user_messages else ""
    has_complex_keywords = any(keyword in latest_message.lower() for keyword in complex_keywords)

    logger.info(f"Pesan terbaru: {latest_message}")
    logger.info(f"Apakah mengandung kata kunci kompleks? {has_complex_keywords}")

    if previous_complexity == "complex":
        if not has_complex_keywords:
            logger.info(f"Kompleksitas turun dari complex ke medium karena pesan terbaru tidak mengandung kata kunci kompleks.")
            return "medium"
        else:
            logger.info(f"Kompleksitas tetap complex karena pesan terbaru mengandung kata kunci kompleks.")
            return "complex"

    elif previous_complexity == "medium":
        if not has_complex_keywords:
            logger.info(f"Kompleksitas turun dari medium ke simple karena pesan terbaru tidak mengandung kata kunci kompleks.")
            return "simple"
        else:
            logger.info(f"Kompleksitas tetap medium karena pesan terbaru mengandung kata kunci kompleks.")
            return "medium"

    else:  # previous_complexity == "simple"
        if has_complex_keywords:
            logger.info(f"Kompleksitas naik dari simple ke complex karena pesan terbaru mengandung kata kunci kompleks.")
            return "complex"
        elif session.get('message_counter', 0) > 3:
            logger.info(f"Kompleksitas naik dari simple ke medium karena jumlah pesan > 3.")
            return "medium"
        else:
            logger.info(f"Kompleksitas tetap simple karena tidak ada kata kunci kompleks dan jumlah pesan <= 3.")
            return "simple"


async def set_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Implementation of set_reminder remains the same as in the provided code
    try:
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Format: /ingatkan <waktu> <pesan>")
            return

        time_str = args[0]
        message = " ".join(args[1:])

        try:
            time_seconds = int(time_str) * 60
        except ValueError:
            await update.message.reply_text("Format waktu tidak valid. Gunakan angka (misalnya, 5).")
            return

        context.job_queue.run_once(
            callback=send_reminder,
            when=time_seconds,
            chat_id=update.message.chat_id,
            data=message
        )

        await update.message.reply_text(f"Pengingat diatur untuk {time_str} menit lagi.")
    except Exception as e:
        logger.error(f"Error setting reminder: {str(e)}")
        await update.message.reply_text("Maaf, terjadi kesalahan saat mengatur pengingat.")

async def send_reminder(context: ContextTypes.DEFAULT_TYPE):
    # Implementation of send_reminder remains the same as in the provided code
    job = context.job
    await context.bot.send_message(chat_id=job.chat_id, text=f"⏰ Pengingat: {job.data}")

async def encode_image(image_source) -> str:
    """Encode an image file or BytesIO object to base64 string."""
    # Implementation of encode_image remains the same as in the provided code
    try:
        if isinstance(image_source, BytesIO):
            image_source.seek(0)
            return base64.b64encode(image_source.read()).decode('utf-8')
        elif isinstance(image_source, str):
            with open(image_source, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            raise TypeError("Invalid image source type. Expected str or BytesIO.")
    except Exception as e:
        logger.exception("Error encoding image")
        raise

async def check_rate_limit(user_id: int) -> bool:
    # Implementation of check_rate_limit remains the same as in the provided code
    key = f"rate_limit:{user_id}"
    count = redis_client.get(key)
    if count and int(count) > MAX_REQUESTS_PER_MINUTE:
        return False
    redis_client.incr(key)
    redis_client.expire(key, 60)
    return True

async def translate_to_english(text: str) -> str:
    """
    Menerjemahkan teks ke Bahasa Inggris menggunakan deep-translator.
    """
    # Implementation of translate_to_english remains the same as in the provided code
    try:
        translation = GoogleTranslator(source='auto', target='en').translate(text)
        return translation
    except Exception as e:
        logger.error(f"Error translating text to English: {str(e)}")
        return text

async def generate_image(update: Update, prompt: str) -> Optional[str]:
    # Implementation of generate_image remains the same as in the provided code
    try:
        english_prompt = await translate_to_english(prompt)
        logger.info(f"Original prompt: {prompt}, Translated prompt: {english_prompt}")
        client = Together()

        try:
            response = client.images.generate(
                prompt=english_prompt,
                model="black-forest-labs/FLUX.1-schnell-Free",
                width=1440,
                height=960,
                steps=4,
                n=1,
                guidance_scale=9.5,
                response_format="b64_json"
            )
            if response and hasattr(response, 'data') and len(response.data) > 0:
                return response.data[0].b64_json

            logger.error("No image data in response")
            return None

        except Exception as api_error:
            logger.error(f"Together API error: {str(api_error)}")
            return None

    except Exception as e:
        logger.exception("Error in generate_image")
        return None


async def handle_generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Implementation of handle_generate_image remains the same as in the provided code
    try:
        message_text = update.message.text or ""
        is_group = update.message.chat.type in ["group", "supergroup"]
        bot_username = context.bot.username.lower()

        if is_group:
            if f"@{bot_username}" not in message_text.lower():
                logger.info(f"Pesan di grup tanpa mention yang valid diabaikan. Pesan: {message_text}")
                return
            message_text = message_text.replace(f"@{bot_username}", "").strip()

        prompt = re.sub(r'^/(?:gambar|image)\s*', '', message_text).strip()

        if not prompt:
            await update.message.reply_text("Mohon berikan prompt untuk menghasilkan gambar. Contoh: /gambar pemandangan gunung")
            return

        processing_msg = await update.message.reply_text("🔄 Sedang menghasilkan gambar...")

        try:
            image_data = await generate_image(update, prompt)

            if image_data:
                image_bytes = base64.b64decode(image_data)
                with BytesIO(image_bytes) as bio:
                    bio.seek(0)
                    await update.message.reply_photo(photo=bio)
            else:
                await update.message.reply_text("Maaf, gagal menghasilkan gambar. Silakan coba lagi.")

        finally:
            await processing_msg.delete()

    except Exception as e:
        logger.error(f"Error dalam handle_generate_image: {e}")
        await update.message.reply_text("Terjadi kesalahan saat menghasilkan gambar.")


async def process_image_with_gemini(image_bytes: BytesIO, prompt: str = None) -> Tuple[Optional[str], Optional[str]]:
    # Implementation of process_image_with_gemini remains the same as in the provided code
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        image = Image.open(image_bytes)
        default_prompt = "Deskripsikan gambar ini sedetail mungkin dalam Bahasa Indonesia. Sebutkan objek yang ada di gambar, warna, bentuk, dan karakteristik penting lainnya. Analisis secara komprehensif."
        default_response = model.generate_content([default_prompt, image])
        filtered_default_response_text = await filter_text(default_response.text) if default_response.text else None

        user_prompt = prompt if prompt else default_prompt
        user_response = model.generate_content([user_prompt, image])
        filtered_user_response_text = await filter_text(user_response.text) if user_response.text else None

        return filtered_user_response_text, filtered_default_response_text

    except Exception as e:
        logger.exception("Error in processing image with Gemini")
        return "Terjadi kesalahan saat memproses gambar dengan Gemini.", None

async def search_google(query: str) -> List[str]:
    # Implementation of search_google remains the same as in the provided code
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        cse_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")

        if not api_key or not cse_id:
            logger.error("API Key atau CSE ID Google belum diatur di environment variables.")
            return []

        service = build("customsearch", "v1", developerKey=api_key, static_discovery=False)
        res = service.cse().list(q=query, cx=cse_id).execute()
        results = res.get("items", [])
        return [result["link"] for result in results]
    except Exception as e:
        logger.exception(f"Error saat mencari di Google: {e}")
        return []

async def process_with_gemini(messages: List[Dict[str, str]], session: Optional[Dict] = None, complexity: str = "simple") -> Optional[str]:
    # Implementation of process_with_gemini remains the same as in the provided code
    try:
        history = []
        for msg in messages:
            history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })

        system_instruction = None
        if messages and "system" in messages[0]["role"]:
            system_instruction = messages[0]["content"]
            history = history[1:]
        
        if complexity == "simple":
            system_instruction = "Berikan respons dalam Bahasa Indonesia jelas. Ingat konteks percakapan."
        elif complexity == "medium":
            system_instruction = "Berikan respons yang Relevan dan jelas dalam Bahasa Indonesia. Ingat konteks percakapan."
        elif complexity == "complex":
            system_instruction = "Berikan respons dalam bahasa indonesia yang detail dan komprehensif dengan analisis mendalam. Ingat konteks percakapan."

        model = genai.GenerativeModel(
            "gemini-2.0-flash-thinking-exp-01-21",
            system_instruction=system_instruction
        ) if system_instruction else gemini_model

        logger.info(f"History: {history}")
        chat = model.start_chat(history=history)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, chat.send_message, messages[-1]["content"])
        return await filter_text(response.text)

    except generation_types.BlockedPromptException as e:
        logger.error(f"Prompt diblokir: {str(e)}")
        return "Pertanyaan Anda mengandung konten yang tidak diizinkan"

    except generation_types.StopCandidateException as e:
        logger.error(f"Gemini RECITATION error: {e}")
        return "Terjadi kesalahan dalam memproses permintaan"

    except asyncio.TimeoutError:
        logger.error("Gemini timeout.")
        return "Permintaan timeout, silakan coba lagi nanti"

    except Exception as e:
        logger.exception(f"Error processing Gemini request: {e}")
        return "Terjadi kesalahan saat memproses permintaan"


async def process_image_with_pixtral_multiple(image_path: str, prompt: str = None, repetitions: int = 2) -> List[str]:
    # Implementation of process_image_with_pixtral_multiple remains the same as in the provided code
    try:
        base64_image = await encode_image(image_path)
        results = []

        async def single_request():
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }
            user_prompt = prompt if prompt else "Apa isi gambar ini? singkat padat Jelas Bahasa Indonesia."
            data = {
                "model": "pixtral-large-latest",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ]
            }

            max_retries = 3
            retry_delay = 2

            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://api.mistral.ai/v1/chat/completions",
                            headers=headers,
                            json=data
                        ) as response:
                            if response.status == 429:
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay * (attempt + 1))
                                    continue
                            response.raise_for_status()
                            result = await response.json()
                            return result['choices'][0]['message']['content']
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    logger.exception("Error in single request after all retries")
                    return "Terjadi kesalahan dalam analisis ini."

        for i in range(repetitions):
            result = await single_request()
            results.append(result)
            if i < repetitions - 1:
                await asyncio.sleep(1)

        return results

    except Exception as e:
        logger.exception("Error in processing image with Pixtral multiple")
        return ["Terjadi kesalahan saat memproses gambar."] * repetitions


async def process_voice_to_text(update: Update) -> Optional[str]:
    """
    Proses file suara menjadi teks dengan optimasi untuk Railway.
    """
    # Implementation of process_voice_to_text remains the same as in the provided code
    try:
        logger.info("Memulai pemrosesan pesan suara...")
        voice_file = await update.message.voice.get_file()

        with BytesIO() as ogg_bytes, BytesIO() as wav_bytes:
            ogg_data = await voice_file.download_as_bytearray()
            ogg_bytes.write(ogg_data)
            ogg_bytes.seek(0)

            audio = AudioSegment.from_ogg(ogg_bytes)
            audio = (audio
                    .set_channels(1)
                    .set_frame_rate(16000)
                    .normalize())

            audio.export(wav_bytes, format='wav')
            wav_bytes.seek(0)

            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.dynamic_energy_ratio = 1.5

            with sr.AudioFile(wav_bytes) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.record(source)

                try:
                    text = recognizer.recognize_google(
                        audio_data,
                        language="id-ID",
                        show_all=False
                    )
                    logger.info("Transkripsi berhasil")
                    return text
                except sr.UnknownValueError:
                    logger.warning("Google Speech Recognition tidak dapat memahami audio")
                    return None
                except sr.RequestError as e:
                    logger.error(f"Tidak dapat meminta hasil dari Google Speech Recognition; {e}")
                    return None

    except Exception as e:
        logger.exception("Error dalam pemrosesan audio")
        raise


# Potong audio besar
def split_audio_to_chunks(audio_path: str, chunk_duration: int = 60) -> List[str]:
    # Implementation of split_audio_to_chunks remains the same as in the provided code
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    for i in range(0, len(audio), chunk_duration * 1000):
        chunk_path = f"{audio_path}_{i // (chunk_duration * 1000)}.wav"
        audio[i:i + chunk_duration * 1000].export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def get_max_conversation_messages(complexity: str) -> int:
    """
    Mengembalikan batas pesan berdasarkan kompleksitas percakapan.
    """
    # Implementation of get_max_conversation_messages remains the same as in the provided code
    if complexity == "simple":
        return MAX_CONVERSATION_MESSAGES_SIMPLE
    elif complexity == "medium":
        return MAX_CONVERSATION_MESSAGES_MEDIUM
    elif complexity == "complex":
        return MAX_CONVERSATION_MESSAGES_COMPLEX
    else:
        return MAX_CONVERSATION_MESSAGES_MEDIUM

async def filter_text(text: str) -> str:
    """Filter untuk menghapus karakter tertentu seperti asterisks (*) dan #, serta kata 'Mistral'"""
    # Implementation of filter_text remains the same as in the provided code
    filtered_text = text.replace("*", "").replace("#", "").replace("Mistral AI", "PAIDI").replace("oleh Google", "PAIDI").replace("Mistral", "PAIDI").replace("Tentu, ", "")
    return filtered_text.strip()

async def process_with_mistral(messages: List[Dict[str, str]]) -> Optional[str]:
    # Implementation of process_with_mistral remains the same as in the provided code
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    messages.insert(0, {"role": "system", "content": "Pastikan semua respons diberikan dalam Bahasa Indonesia Yang Mudah Di pahami."})
    data = {
        "model": "mistral-large-latest",
        "messages": messages,
        "max_tokens": 10000,
        "temperature": 0.7,
        "top_p": 0.95
    }

    backoff_delay = RETRY_DELAY
    max_backoff = 60

    for attempt in range(MAX_RETRIES):
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 429:
                        logger.warning(f"Rate limit exceeded on attempt {attempt + 1}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(backoff_delay)
                            backoff_delay = min(backoff_delay * 2, max_backoff)
                            continue

                    response.raise_for_status()
                    json_response = await response.json()

                    if 'choices' in json_response and json_response['choices']:
                        content = json_response['choices'][0]['message']['content']
                        return await filter_text(content)
                    else:
                        logger.error(f"Invalid response format from Mistral API: {json_response}")
                        return "Terjadi kesalahan format respons dari server."

        except aiohttp.ClientError as e:
            logger.error(f"Percobaan {attempt + 1} gagal karena error HTTP: {str(e)}")
            if "Too Many Requests" in str(e):
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, max_backoff)
                    continue
        except asyncio.TimeoutError:
            logger.error(f"Percobaan {attempt + 1} gagal karena timeout")
        except json.JSONDecodeError as e:
            logger.error(f"Gagal mendecode respons JSON pada percobaan {attempt + 1}: {str(e)}")
        except Exception as e:
            logger.exception(f"Percobaan {attempt + 1} gagal dengan error: {str(e)}")

        if attempt < MAX_RETRIES - 1:
            await asyncio.sleep(backoff_delay)
            backoff_delay = min(backoff_delay * 2, max_backoff)
            logger.info(f"Menunggu {backoff_delay} detik sebelum percobaan berikutnya...")

    return "Maaf, server tidak merespons setelah beberapa percobaan. Mohon coba lagi nanti."


async def send_voice_response(update: Update, text: str):
    # Implementation of send_voice_response remains the same as in the provided code
    temp_file = None
    try:
        tts = gtts.gTTS(text, lang="id")
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tts.save(temp_file.name)

        with open(temp_file.name, 'rb') as voice_file:
            await update.message.reply_voice(voice=voice_file)
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Implementation of handle_voice remains the same as in the provided code
    try:
        chat_id = update.message.chat_id

        if not redis_client.exists(f"session:{chat_id}"):
            await initialize_session(chat_id)

        if update.message.voice.file_size > MAX_AUDIO_SIZE:
            await update.message.reply_text("Maaf, file audio terlalu besar (maksimal 20MB)")
            return

        await update_bot_statistics("voice_messages")
        await context.bot.send_chat_action(chat_id=update.message.chat_id, action="record_voice")
        processing_msg = await update.message.reply_text("Sedang memproses pesan suara Anda...")

        try:
            text = await process_voice_to_text(update)
            if text:
                text_parts = split_message(text)
                for part in text_parts:
                    await update.message.reply_text(f"Teks hasil transkripsi suara Anda:\n{part}")

                session = json.loads(redis_client.get(f"session:{chat_id}"))
                processed_messages = []
                for msg in session['messages']:
                    role = "user" if msg["role"] in ["user", "assistant"] else "model"
                    processed_messages.append({"role": role, "content": msg["content"]})

                response = await process_with_gemini(processed_messages)

                if response:
                    session['messages'].append({"role": "model", "content": response})
                    await update_session(chat_id, {"role": "model", "content": response})

                    filtered_response = await filter_text(response)
                    response_parts = split_message(filtered_response)
                    for part in response_parts:
                        await update.message.reply_text(part)
                    await send_voice_response(update, filtered_response)
            else:
                await update.message.reply_text("Maaf, saya tidak dapat mengenali suara dengan jelas. Mohon coba lagi.")

        finally:
            await processing_msg.delete()

    except Exception as e:
        await update_bot_statistics("errors")
        logger.exception("Error dalam handle_voice")
        await update.message.reply_text("Maaf, terjadi kesalahan dalam pemrosesan suara.")


async def upload_image_to_imgfoto(image_bytes: bytes) -> Optional[str]:
    """Upload image to ImgFoto.host and return the URL"""
    # Implementation of upload_image_to_imgfoto remains the same as in the provided code
    try:
        IMGFOTO_API_KEY = os.getenv('IMGFOTO_API_KEY')
        if not IMGFOTO_API_KEY:
            logger.error("IMGFOTO_API_KEY tidak ditemukan di environment variables")
            return None

        url = "https://imgfoto.host/api/1/upload"
        headers = {
            'X-API-Key': IMGFOTO_API_KEY,
            'Accept': 'application/json'
        }
        form = aiohttp.FormData()
        form.add_field(
            'source',
            image_bytes,
            filename='image.jpg',
            content_type='image/jpeg'
        )
        form.add_field('format', 'json')
        form.add_field('expiration', 'P1W')

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=form, headers=headers) as response:
                logger.info(f"ImgFoto response status: {response.status}")
                response_text = await response.text()
                logger.info(f"ImgFoto response body: {response_text}")

                if response.status == 200:
                    try:
                        result = await response.json()
                        if result.get('status_code') == 200 and result.get('success'):
                            image_url = result.get('image', {}).get('url')
                            if image_url:
                                logger.info(f"Upload successful. URL: {image_url}")
                                return image_url
                            else:
                                logger.error("No image URL in successful response")
                        else:
                            logger.error(f"Upload failed: {result.get('status_txt')}")
                    except Exception as e:
                        logger.error(f"Error parsing ImgFoto response: {str(e)}")
                else:
                    logger.error(f"ImgFoto upload failed with status {response.status}. Response: {response_text}")

    except Exception as e:
        logger.error(f"Error uploading to ImgFoto: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    return None


async def get_google_image_search_url(image_url: str) -> str:
    """Generate Google Lens search URL"""
    # Implementation of get_google_image_search_url remains the same as in the provided code
    encoded_url = urllib.parse.quote(image_url)
    return f"https://lens.google.com/uploadbyurl?url={encoded_url}"


async def search_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /carigambar command"""
    # Implementation of search_image_command remains the same as in the provided code
    try:
        if not os.getenv('IMGFOTO_API_KEY'):
            await update.message.reply_text(
                "❌ Fitur pencarian gambar belum dikonfigurasi.\n"
                "Mohon hubungi admin untuk mengatur IMGFOTO_API_KEY."
            )
            return

        if not update.message.reply_to_message or not update.message.reply_to_message.photo:
            await update.message.reply_text(
                "Cara penggunaan:\n"
                "1. Reply ke gambar yang ingin dicari\n"
                "2. Ketik: /carigambar"
            )
            return

        processing_msg = await update.message.reply_text("🔄 Sedang memproses pencarian gambar...")

        try:
            photo = update.message.reply_to_message.photo[-1]
            photo_file = await photo.get_file()
            logger.info(f"Processing image: size={photo.file_size} bytes, file_id={photo.file_id}")
            photo_bytes = await photo_file.download_as_bytearray()
            logger.info(f"Downloaded image size: {len(photo_bytes)} bytes")

            image_url = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    image_url = await upload_image_to_imgfoto(photo_bytes)
                    if image_url:
                        logger.info(f"Successfully uploaded to ImgFoto on attempt {attempt + 1}")
                        break
                    else:
                        logger.warning(f"Upload attempt {attempt + 1} failed, retrying...")

                    if attempt < max_retries - 1:
                        delay = min(10, (2 ** attempt))
                        logger.info(f"Waiting {delay} seconds before retry")
                        await asyncio.sleep(delay)
                except Exception as e:
                    logger.error(f"Error during upload attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        continue
                    raise

            if not image_url:
                error_msg = (
                    "❌ Gagal mengupload gambar ke ImgFoto.\n"
                    "Silakan coba lagi dalam beberapa saat."
                )
                logger.error("All upload attempts to ImgFoto failed")
                await update.message.reply_text(error_msg)
                return

            google_search_url = await get_google_image_search_url(image_url)

            await update.message.reply_text(
                "🔍 Hasil pencarian gambar:\n\n"
                f"📸 Link gambar: {image_url}\n\n"
                f"🌐 Cari dengan Google Lens:\n{google_search_url}\n\n"
                "ℹ️ Klik link Google Lens di atas untuk melihat hasil pencarian gambar serupa"
            )

        finally:
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception as e:
                    logger.error(f"Error deleting processing message: {str(e)}")

    except Exception as e:
        error_message = f"Error in search_image_command: {str(e)}"
        logger.exception(error_message)
        await update.message.reply_text(
            "❌ Terjadi kesalahan saat memproses gambar.\n"
            "Pastikan:\n"
            "1. Gambar yang di-reply masih tersedia\n"
            "2. Ukuran gambar tidak terlalu besar (max 10MB)\n"
            "3. Format gambar didukung (JPG, PNG)\n"
            "4. Koneksi internet stabil\n"
            "\nSilakan coba lagi nanti."
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Implementation of handle_photo remains the same as in the provided code
    chat_id = update.message.chat_id
    chat_type = update.message.chat.type
    caption = update.message.caption or ""

    user_id = update.message.from_user.id
    if not await check_rate_limit(user_id):
        await update.message.reply_text("Anda telah melebihi batas permintaan. Mohon tunggu beberapa saat.")
        return

    if chat_type in ["group", "supergroup"]:
        if f"@{context.bot.username}" not in caption:
            logger.info("Gambar di grup diabaikan karena tidak ada mention.")
            return

    try:
        if not redis_client.exists(f"session:{chat_id}"):
            await initialize_session(chat_id)

        session = json.loads(redis_client.get(f"session:{chat_id}"))
        logger.info(f"Sesi saat ini: {session}")

        await update_bot_statistics("photo_messages")
        processing_msg = await update.message.reply_text("Sedang menganalisa gambar...🔍🧐")

        photo_file = await update.message.photo[-1].get_file()

        with BytesIO() as temp_file:
            photo_bytes = await photo_file.download_as_bytearray()
            temp_file.write(photo_bytes)
            temp_file.seek(0)

            prompt = caption.replace(f"@{context.bot.username}", "").strip() if caption else None

            if not prompt:
                prompt = "Apa isi gambar ini? Berikan deskripsi detail dalam Bahasa Indonesiadan termasuk penjelasan yang komprehensif."
                logger.info(f"Prompt yang digunakan: {prompt}")
            else:
                prompt += " jawab dalam Bahasa Indonesia."

            gemini_result = await process_image_with_gemini(temp_file, prompt=prompt)

            if gemini_result and isinstance(gemini_result, tuple):
                filtered_user_response, filtered_default_response = await asyncio.gather(
                    filter_text(gemini_result[0] or ""),
                    filter_text(gemini_result[1] or "")
                )

                result_to_send = filtered_user_response if filtered_user_response else filtered_default_response
                result_parts = split_message(result_to_send)
                for part in result_parts:
                    await update.message.reply_text(f"Analisa:\n{part}")

                if not prompt and filtered_default_response:
                    session['messages'].append({
                        "role": "assistant",
                        "content": filtered_default_response
                    })
                elif filtered_user_response:
                    session['messages'].append({
                        "role": "assistant",
                        "content": filtered_user_response
                    })
                session['last_image_analysis'] = filtered_user_response if filtered_user_response else filtered_default_response
                await update_session(chat_id, {"role": "assistant", "content": filtered_user_response if filtered_user_response else filtered_default_response})
            elif isinstance(gemini_result, str) and gemini_result:
                await update.message.reply_text(f"Analisa:\n{gemini_result}")
                session['messages'].append({
                    "role": "assistant",
                    "content": gemini_result
                })
                session['last_image_analysis'] = filtered_result
                await update_session(chat_id, {"role": "assistant", "content": filtered_result})
            else:
                await update.message.reply_text("Maaf, tidak dapat menganalisa gambar. Silakan coba lagi.")

        await processing_msg.delete()

    except Exception as e:
        logger.exception("Error dalam proses analisis gambar dengan Gemini")
        await update.message.reply_text("Terjadi kesalahan saat memproses gambar.")


async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Implementation of handle_mention remains the same as in the provided code
    chat_id = update.message.chat_id
    chat_type = update.message.chat.type
    message_text = update.message.text or update.message.caption or ""
    is_reply = bool(update.message.reply_to_message)
    replied_message = None

    if chat_type in ["group", "supergroup"]:
        should_process = False

        if f'@{context.bot.username}' in message_text:
            message_text = message_text.replace(f'@{context.bot.username}', '').strip()
            should_process = True

        elif is_reply and update.message.reply_to_message.from_user.id == context.bot.id:
            should_process = True
            replied_message = update.message.reply_to_message.text

        if should_process and message_text:
            sanitized_text = sanitize_input(message_text)

            if sanitized_text.lower().startswith(('/gambar', '/image')):
                await handle_generate_image(update, context)
                return

            if not redis_client.exists(f"session:{chat_id}"):
                await initialize_session(chat_id)

            session = json.loads(redis_client.get(f"session:{chat_id}"))

            if not is_reply and await should_reset_context(chat_id, sanitized_text):
                await initialize_session(chat_id)
                session = json.loads(redis_client.get(f"session:{chat_id}"))

            if replied_message:
                sanitized_text = f"Dalam konteks pesan sebelumnya: '{replied_message}', {sanitized_text}"

            session['messages'].append({"role": "user", "content": sanitized_text})
            await update_session(chat_id, {"role": "user", "content": sanitized_text})

            context_window = MAX_CONVERSATION_MESSAGES_COMPLEX if is_reply else 10
            response = await process_with_smart_context(session['messages'][-context_window:])

            if response:
                filtered_response = await filter_text(response)
                session['messages'].append({"role": "assistant", "content": filtered_response})
                await update_session(chat_id, {"role": "assistant", "content": filtered_response})

                response_parts = split_message(filtered_response)
                for part in response_parts:
                    await update.message.reply_text(part)
        else:
            logger.info("Pesan di grup tanpa mention yang valid diabaikan.")


async def initialize_session(chat_id: int) -> None:
    # Implementation of initialize_session remains the same as in the provided code
    session = {
        'messages': [],
        'message_counter': 0,
        'last_update': datetime.now().timestamp(),
        'conversation_id': str(uuid.uuid4()),
        'complexity': 'simple'
    }
    redis_client.set(f"session:{chat_id}", json.dumps(session))
    redis_client.expire(f"session:{chat_id}", CONVERSATION_TIMEOUT)
    logger.info(f"Sesi direset untuk chat_id {chat_id}.")


async def update_session(chat_id: int, message: Dict[str, str]) -> None:
    # Implementation of update_session remains the same as in the provided code
    session_json = redis_client.get(f"session:{chat_id}")
    if session_json:
        session = json.loads(session_json)
    else:
        session = {
            'messages': [],
            'message_counter': 0,
            'last_update': datetime.now().timestamp(),
            'complexity': 'simple'
        }

    if 'message_counter' not in session:
        session['message_counter'] = 0

    previous_complexity = session.get('complexity', 'simple')
    new_complexity = await determine_conversation_complexity(session['messages'], session, previous_complexity)
    session['complexity'] = new_complexity

    if new_complexity == "simple" and previous_complexity == "medium":
        logger.info(f"Transisi dari medium ke simple, reset counter pesan untuk chat_id {chat_id}.")
        session['message_counter'] = 0

    session['message_counter'] += 1

    if previous_complexity != new_complexity:
        logger.info(f"Perubahan kompleksitas percakapan untuk chat_id {chat_id}: {previous_complexity} -> {new_complexity}")

    session['messages'].append(message)
    session['last_update'] = datetime.now().timestamp()

    redis_client.set(f"session:{chat_id}", json.dumps(session))
    redis_client.expire(f"session:{chat_id}", CONVERSATION_TIMEOUT)


async def process_with_smart_context(messages: List[Dict[str, str]]) -> Optional[str]:
    # Implementation of process_with_smart_context remains the same as in the provided code
    try:
        try:
            response = await asyncio.wait_for(process_with_gemini(messages), timeout=30)
            if response:
                logger.info("Menggunakan respons dari Gemini.")
                return response
        except generation_types.StopCandidateException as e:
            logger.error(f"Gemini RECITATION error: {e}")
        except asyncio.TimeoutError:
            logger.warning("Gemini timeout, beralih ke Mistral.")

        try:
            response = await asyncio.wait_for(process_with_mistral(messages), timeout=10)
            if response:
                logger.info("Menggunakan respons dari Mistral.")
                return response
        except asyncio.TimeoutError:
            logger.error("Mistral timeout.")

        logger.error("Semua model gagal memproses pesan.")
        return None
    except Exception as e:
        logger.exception(f"Error dalam pemrosesan konteks cerdas: {e}")
        return None


async def extract_relevant_keywords(messages: List[Dict[str, str]], top_n: int = 5) -> List[str]:
    # Implementation of extract_relevant_keywords remains the same as in the provided code
    context_text = " ".join([msg.get('content', '') for msg in messages])
    words = re.findall(r'\b\w+\b', context_text.lower())
    words = [re.sub(r'[^\w\s]', '', word) for word in words]
    words = [word for word in words if len(word) >= 3]
    words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    word_counts = Counter(stemmed_words)
    relevant_keywords = [word for word, count in word_counts.most_common(top_n)]
    logger.info(f"Extracted relevant keywords: {relevant_keywords}")
    return relevant_keywords


def is_same_topic(last_message: str, current_message: str, context_messages: List[Dict[str, str]], threshold: int = 2) -> bool:
    # Implementation of is_same_topic remains the same as in the provided code
    relevant_keywords = extract_relevant_keywords(context_messages)
    last_keywords = [word for word in relevant_keywords if word in last_message.lower()]
    current_keywords = [word for word in relevant_keywords if word in current_message.lower()]
    common_keywords = set(last_keywords) & set(current_keywords)
    logger.info(f"Last keywords: {last_keywords}")
    logger.info(f"Current keywords: {current_keywords}")
    logger.info(f"Common keywords: {common_keywords}")
    return len(common_keywords) >= threshold


def is_related_to_context(current_message: str, context_messages: List[Dict[str, str]]) -> bool:
    # Implementation of is_related_to_context remains the same as in the provided code
    relevant_keywords = extract_relevant_keywords(context_messages)
    return any(keyword in current_message.lower() for keyword in relevant_keywords)


async def should_reset_context(chat_id: int, message: str) -> bool:
    # Implementation of should_reset_context remains the same as in the provided code
    try:
        session_json = redis_client.get(f"session:{chat_id}")
        if not session_json:
            logger.info(f"Tidak ada sesi untuk chat_id {chat_id}, reset konteks.")
            return True

        session = json.loads(session_json)
        last_update = session.get('last_update', 0)
        current_time = datetime.now().timestamp()
        time_diff = current_time - last_update

        if time_diff > CONVERSATION_TIMEOUT:
            logger.info(f"Reset konteks untuk chat_id {chat_id} karena timeout (percakapan terlalu lama).")
            redis_client.delete(f"session:{chat_id}")
            return True

        reset_keywords = ['halo', 'hai', 'hi', 'hello', 'permisi', 'terima kasih', 'terimakasih', 'sip', 'tengkiuw', 'reset', 'mulai baru', 'clear']
        normalized_message = message.lower().strip()

        if any(keyword in normalized_message for keyword in reset_keywords):
            logger.info(f"Reset konteks untuk chat_id {chat_id} karena pesan mengandung kata kunci reset: {message}")
            redis_client.delete(f"session:{chat_id}")
            return True

        if 'message_counter' not in session:
            session['message_counter'] = 0

        complexity = await determine_conversation_complexity(session['messages'], session)
        max_messages = get_max_conversation_messages(complexity)

        if len(session['messages']) > max_messages:
            logger.info(f"Reset konteks untuk chat_id {chat_id} karena percakapan terlalu panjang (jumlah pesan: {len(session['messages'])}).")
            return True

        logger.info(f"Tidak perlu reset konteks untuk chat_id {chat_id}.")
        return False
    except redis.RedisError as e:
        logger.error(f"Redis Error saat memeriksa konteks untuk chat_id {chat_id}: {str(e)}")
        return True
    except Exception as e:
        logger.error(f"Error tak terduga saat memeriksa konteks untuk chat_id {chat_id}: {str(e)}")
        return True


async def reset_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /reset. Mereset sesi percakapan pengguna."""
    # Implementation of reset_session remains the same as in the provided code
    chat_id = update.message.chat_id
    try:
        redis_client.delete(f"session:{chat_id}")
        await update.message.reply_text("Sesi percakapan Anda telah direset. Mulai percakapan baru sekarang!")
    except Exception as e:
        logger.error(f"Gagal mereset sesi untuk chat_id {chat_id}: {str(e)}")
        await update.message.reply_text("Maaf, terjadi kesalahan saat mereset sesi.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, message_text: Optional[str] = None):
    # Implementation of handle_text remains the same as in the provided code
    if not message_text:
        message_text = update.message.text or ""

    sanitized_text = sanitize_input(message_text)
    user_id = update.message.from_user.id
    if not await check_rate_limit(user_id):
        await update.message.reply_text("Anda telah melebihi batas permintaan. Mohon tunggu beberapa saat.")
        return

    chat_id = update.message.chat_id
    chat_type = update.message.chat.type
    is_reply = bool(update.message.reply_to_message)
    replied_message = None

    session_json = redis_client.get(f"session:{chat_id}")
    if not session_json:
        await initialize_session(chat_id)
        session = {'messages': [], 'last_update': datetime.now().timestamp()}
    else:
        session = json.loads(session_json)

    if chat_type == "private" and is_reply and update.message.reply_to_message.from_user.id == context.bot.id:
        replied_message = update.message.reply_to_message.text
        sanitized_text = f"Dalam konteks pesan sebelumnya: '{replied_message}', {sanitized_text}"

    if not is_reply and await should_reset_context(chat_id, sanitized_text):
        await initialize_session(chat_id)
        session = json.loads(redis_client.get(f"session:{chat_id}"))

    session['messages'].append({"role": "user", "content": sanitized_text})
    await update_session(chat_id, {"role": "user", "content": sanitized_text})

    full_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in session['messages']
    ]

    response = await process_with_gemini(full_history)

    if response:
        filtered_response = await filter_text(response)
        session['messages'].append({"role": "assistant", "content": filtered_response})
        await update_session(chat_id, {"role": "assistant", "content": filtered_response})

        response_parts = split_message(filtered_response)
        for part in response_parts:
            await update.message.reply_text(part)
    else:
        await update.message.reply_text("Maaf, terjadi kesalahan dalam memproses pesan Anda.")



async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Implementation of handle_message remains the same as in the provided code
    await update_bot_statistics("total_messages")

    if update.message.text:
        await update_bot_statistics("text_messages")
        await handle_text(update, context)
    elif update.message.voice:
        await update_bot_statistics("voice_messages")
        await handle_voice(update, context)
    elif update.message.photo:
        await update_bot_statistics("photo_messages")
        await handle_photo(update, context)
    else:
        logger.info("Pesan dengan tipe yang tidak didukung diabaikan.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /help"""
    # Implementation of help_command remains the same as in the provided code
    help_text = """
🤖 **PAIDI Bot - Panduan Penggunaan** 🤖

Berikut adalah daftar perintah yang tersedia:

/start - Memulai percakapan dengan bot.
/help - Menampilkan panduan penggunaan bot.
/stats - Menampilkan statistik penggunaan bot.
/reset - Mereset sesi percakapan Anda.
/carigambar - Mencari gambar serupa menggunakan Google Lens.
/gambar <prompt> - Generate gambar berdasarkan prompt.
/reminder <waktu> <pesan> - Mengatur pengingat (contoh: /reminder 5 Beli susu).

**Fitur Lain:**
- Menerima pesan teks, suara, dan gambar.
- Menganalisis sentimen pesan Anda.
- Menanggapi dengan suara.

Kirim saya pesan atau catatan suara untuk memulai!
    """
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def track_message_statistics(update: Update):
    # Implementation of track_message_statistics remains the same as in the provided code
    user_id = update.message.from_user.id
    message_type = "text"
    if update.message.voice:
        message_type = "voice"
    elif update.message.photo:
        message_type = "photo"

    redis_client.hincrby(f"user:{user_id}:stats", message_type, 1)
    redis_client.hincrby(f"user:{user_id}:stats", "total_messages", 1)


async def get_user_statistics(user_id: int) -> Dict[str, int]:
    # Implementation of get_user_statistics remains the same as in the provided code
    stats = redis_client.hgetall(f"user:{user_id}:stats")
    return {k: int(v) for k, v in stats.items()}


async def get_bot_statistics() -> Dict[str, int]:
    """Mengambil statistik bot dari Redis."""
    # Implementation of get_bot_statistics remains the same as in the provided code
    stats = {
        "total_messages": 0,
        "voice_messages": 0,
        "text_messages": 0,
        "photo_messages": 0,
        "errors": 0
    }
    for key in stats.keys():
        value = redis_client.get(f"bot_stats:{key}")
        if value:
            stats[key] = int(value)
    return stats


async def update_bot_statistics(metric: str, increment: int = 1):
    """Update statistik bot di Redis."""
    # Implementation of update_bot_statistics remains the same as in the provided code
    redis_client.incrby(f"bot_stats:{metric}", increment)


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /stats. Menampilkan statistik penggunaan bot."""
    # Implementation of stats remains the same as in the provided code
    chat_id = update.message.chat_id
    stats = await get_bot_statistics()
    stats_message = (
        "📊 **Statistik Bot PAIDI** 📊\n\n"
        f"Total Pesan Diterima: {stats.get('total_messages', 0)}\n"
        f"Pesan Suara: {stats.get('voice_messages', 0)}\n"
        f"Pesan Teks: {stats.get('text_messages', 0)}\n"
        f"Pesan Foto: {stats.get('photo_messages', 0)}\n"
        f"Error: {stats.get('errors', 0)}\n"
        "\nStatistik ini mencerminkan penggunaan bot secara keseluruhan."
    )
    await update.message.reply_text(stats_message, parse_mode="Markdown")


def main():
    # Implementation of main remains the same as in the provided code
    logger.info("Memulai inisialisasi bot...")
    if not check_required_settings():
        logger.critical("Konfigurasi tidak lengkap!")
        return

    try:
        logger.info("Inisialisasi Gemini...")
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            logger.info("Gemini initialized successfully with GOOGLE_API_KEY")
        else:
            logger.warning("GOOGLE_API_KEY tidak ditemukan, fitur tidak akan berfungsi")
        logger.info("Gemini initialization completed.")

        logger.info("Inisialisasi aplikasi bot Telegram...")
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        logger.info("Menambahkan handlers...")
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stats", stats))
        application.add_handler(CommandHandler("reset", reset_session))
        application.add_handler(CommandHandler("carigambar", search_image_command))
        application.add_handler(CommandHandler("ingatkan", set_reminder))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("gambar", handle_generate_image))
        application.add_handler(CommandHandler("harga", handle_stock_request))
        application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE, handle_message))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(MessageHandler(
            (filters.TEXT | filters.CAPTION) &
            (filters.Entity("mention") | filters.REPLY),
            handle_mention
        ))
        logger.info("Handlers ditambahkan.")

        logger.info("Mulai menjalankan bot polling...")
        try:
            application.run_polling(timeout=30)
        except Exception as e:
            logger.critical(f"Gagal menjalankan bot: {e}", exc_info=True)
            raise
        logger.info("Bot polling dihentikan.")

    except Exception as e:
        logger.critical(f"Error fatal saat menjalankan bot: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    asyncio.run(main())
