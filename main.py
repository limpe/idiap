import os
import logging
import io
import tempfile
import asyncio
from typing import List, Dict
from datetime import datetime

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import speech_recognition as sr
from pydub import AudioSegment
import gtts
import aiohttp

# Konfigurasi logging untuk memudahkan debugging dan monitoring
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables untuk menyimpan kunci API yang diperlukan
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Kelas untuk mengelola berita
class NewsManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        
    async def get_top_headlines(self, country: str = 'id', category: str = None) -> List[Dict]:
        params = {
            'country': country,
            'apiKey': self.api_key,
            'pageSize': 5
        }
        
        if category:
            params['category'] = category
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/top-headlines", params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get('articles', [])
        except Exception as e:
            logger.error(f"Error saat mengambil berita: {e}")
            return []

    def format_news_response(self, articles: List[Dict]) -> str:
        if not articles:
            return "Maaf, tidak ada berita yang tersedia saat ini."
            
        response = "📰 Berita Terkini:\n\n"
        for i, article in enumerate(articles, 1):
            published = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
            response += f"{i}. {article['title']}\n"
            response += f"   {article['description']}\n" if article.get('description') else ""
            response += f"   Sumber: {article['source']['name']}\n"
            response += f"   Waktu: {published.strftime('%d %B %Y %H:%M')}\n\n"
        
        return response

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """Halo! Saya asisten Anda. Saya dapat:
    - Memproses pesan suara
    - Menanggapi dengan suara
    - Memberikan berita terkini (/berita)
    - Membantu dengan berbagai tugas

    Kirim pesan atau catatan suara untuk memulai!"""
    await update.message.reply_text(welcome_text)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        text = await process_voice_to_text(update)
        if text is None:
            return

        await update.message.reply_text(f"Anda berkata: {text}")

        response = await process_with_mistral(text)
        if response:
            await update.message.reply_text(f"Respon: {response}")
            await send_voice_response(update, response)
        else:
            await update.message.reply_text("Maaf, terjadi kesalahan dalam memproses permintaan Anda.")

    except Exception as e:
        logger.exception(f"Error tak terduga di handle_voice: {e}")
        await update.message.reply_text(f"Maaf, terjadi kesalahan dalam memproses pesan suara: {e}")

async def process_voice_to_text(update: Update):
    try:
        logger.info("Memproses pesan suara...")
        voice_file = await update.message.voice.get_file()
        voice_bytes = await voice_file.download_as_bytearray()

        try:
            audio = AudioSegment.from_ogg(io.BytesIO(voice_bytes))
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
        except Exception as e:
            logger.warning(f"Gagal konversi ke WAV: {e}, mencoba memproses langsung.")
            wav_io = io.BytesIO(voice_bytes)

        r = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            try:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data, language="id-ID")
                logger.info(f"Teks hasil Speech Recognition: {text}")
                return text
            except sr.UnknownValueError:
                logger.warning("Speech Recognition: Tidak mengerti.")
                await update.message.reply_text("Maaf, saya tidak mengerti apa yang Anda katakan.")
                return None
            except sr.RequestError as e:
                logger.error(f"Speech Recognition error: {e}")
                await update.message.reply_text(f"Maaf, terjadi masalah dengan layanan Speech Recognition: {e}")
                return None
    except Exception as e:
        logger.exception(f"Error dalam memproses audio: {e}")
        return None

async def process_with_mistral(text):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "pixtral-large-latest",
        "messages": [{"role": "user", "content": text}]
    }
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data
            ) as response:
                response.raise_for_status()
                json_response = await response.json()
                return json_response['choices'][0]['message']['content']
    except aiohttp.ClientTimeout:
        logger.error("Timeout saat memanggil Chat API")
        return "Maaf, permintaan ke Chat API terlalu lama. Coba lagi nanti."
    except aiohttp.ClientError as e:
        logger.error(f"Error aiohttp: {e}")
        return "Maaf, terjadi kesalahan saat berkomunikasi dengan Chat API."
    except Exception as e:
        logger.exception(f"Error tak terduga pada process_with_mistral: {e}")
        return "Terjadi kesalahan yang tidak terduga."

async def send_voice_response(update: Update, text):
    logger.info("Membuat respon suara...")
    tts = gtts.gTTS(text, lang="id")
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
        try:
            tts.save(fp.name)
            with open(fp.name, 'rb') as voice_file:
                await update.message.reply_voice(voice=voice_file)
            logger.info("Respon suara dikirim.")
        except Exception as e:
            logger.exception(f"Error saat membuat/mengirim file MP3: {e}")
            await update.message.reply_text("Terjadi kesalahan saat mengirim respon suara.")
        finally:
            os.remove(fp.name)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        logger.info(f"Menerima pesan teks: {update.message.text}")
        response = await process_with_mistral(update.message.text)
        if response:
            logger.info(f"Respon untuk teks: {response}")
            await update.message.reply_text(response)
            await send_voice_response(update, response)
        else:
            await update.message.reply_text("Maaf, terjadi kesalahan dalam memproses permintaan Anda.")
    except Exception as e:
        logger.exception(f"Error tak terduga di handle_text: {e}")
        await update.message.reply_text(f"Maaf, terjadi kesalahan: {e}")

async def handle_news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        news_manager = NewsManager(NEWS_API_KEY)
        
        args = context.args
        category = args[0].lower() if args else None
        
        valid_categories = ['business', 'entertainment', 'health', 'science', 'sports', 'technology']
        if category and category not in valid_categories:
            categories_text = ", ".join(valid_categories)
            await update.message.reply_text(
                f"Kategori tidak valid. Silakan pilih dari kategori berikut:\n{categories_text}"
            )
            return

        articles = await news_manager.get_top_headlines(category=category)
        response = news_manager.format_news_response(articles)
        
        await update.message.reply_text(response)
        await send_voice_response(update, response)
        
    except Exception as e:
        logger.exception(f"Error dalam handle_news_command: {e}")
        await update.message.reply_text("Maaf, terjadi kesalahan saat mengambil berita.")

def main():
    # Inisialisasi aplikasi
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Tambahkan handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("berita", handle_news_command))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("JARVIS sedang berjalan...")
    application.run_polling()

if __name__ == '__main__':
    main()
