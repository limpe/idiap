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
import re


from keywords import complex_keywords


from collections import Counter
from typing import Optional, List, Dict
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pydub import AudioSegment
from langdetect import detect
from groq import Groq
from PIL import Image
from io import BytesIO
from aiohttp import FormData
from datetime import datetime, timedelta
from together import Together
from typing import List, Dict


# Konstanta untuk batasan ukuran file
MAX_AUDIO_SIZE = 20 * 1024 * 1024  # 20MB

def check_required_settings():
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN tidak ditemukan!")
        return False
    if not MISTRAL_API_KEY:
        print("Error: MISTRAL_API_KEY tidak ditemukan!")
        return False
    # Tambahkan ini
    if not TOGETHER_API_KEY:
        print("Error: TOGETHER_API_KEY tidak ditemukan!")
        return False
    return True

def sanitize_input(text: str) -> str:
    # Remove potentially dangerous characters
    return re.sub(r'[<>"\';&]', '', text)

# Konfigurasi logging dengan format yang lebih detail
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')  # Menyimpan log ke file
    ]
)
logger = logging.getLogger(__name__)


# Environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


# Konstanta konfigurasi
CHUNK_DURATION = 30  # Durasi chunk dalam detik
SPEECH_RECOGNITION_TIMEOUT = 30  # Timeout untuk speech recognition dalam detik
MAX_RETRIES = 5  # Jumlah maksimal percobaan untuk API calls
RETRY_DELAY = 5  # Delay antara percobaan ulang dalam detik
CONVERSATION_TIMEOUT = 28800  # Durasi percakapan dalam detik
MAX_CONCURRENT_SESSIONS = 1000
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
MAX_CONVERSATION_MESSAGES_SIMPLE = 10
MAX_CONVERSATION_MESSAGES_MEDIUM = 50
MAX_CONVERSATION_MESSAGES_COMPLEX = 100
MAX_REQUESTS_PER_MINUTE = 10

# Statistik penggunaan
bot_statistics = {
    "total_messages": 0,
    "voice_messages": 0,
    "text_messages": 0,
    "photo_messages": 0,
    "errors": 0
}

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

    Kirim saya pesan atau catatan suara untuk memulai!"""
    await update.message.reply_text(welcome_text)

def split_message(text: str, max_length: int = 4096) -> List[str]:
    """Memecah teks panjang menjadi beberapa bagian sesuai batas Telegram."""
    parts = []
    while len(text) > max_length:
        # Cari posisi pemotongan terdekat (misalnya, setelah baris baru atau spasi)
        split_index = text.rfind("\n", 0, max_length)
        if split_index == -1:
            split_index = text.rfind(" ", 0, max_length)
        if split_index == -1:  # Jika tidak ada baris baru atau spasi, potong langsung
            split_index = max_length

        # Tambahkan bagian ke daftar
        parts.append(text[:split_index].strip())
        text = text[split_index:].strip()

    # Tambahkan sisa teks
    parts.append(text)
    return parts

def determine_conversation_complexity(messages: List[Dict[str, str]]) -> str:
    # Hitung jumlah pesan
    num_messages = len(messages)

    # Cek kata kunci tertentu untuk menentukan kompleksitas
    for message in messages:
        if any(keyword in message['content'].lower() for keyword in complex_keywords):
            return "complex"

    # Tentukan kompleksitas berdasarkan jumlah pesan
    if num_messages > 15:
        return "complex"
    elif num_messages > 5:
        return "medium"
    else:
        return "simple"

async def encode_image(image_source) -> str:
    """Encode an image file or BytesIO object to base64 string."""
    try:
        if isinstance(image_source, BytesIO):
            image_source.seek(0)  # Pastikan pointer berada di awal file
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
    key = f"rate_limit:{user_id}"
    count = redis_client.get(key)
    if count and int(count) > MAX_REQUESTS_PER_MINUTE:
        return False
    redis_client.incr(key)
    redis_client.expire(key, 60)
    return True

async def generate_image(update: Update, prompt: str) -> Optional[str]:
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "black-forest-labs/FLUX.1-schnell-Free",
            "prompt": prompt,
            "width": 1440,
            "height": 960,
            "steps": 4,
            "samples": 1,
            "cfg_scale": 7.5,
            "n": 1,
            "nsfw": True,  # Mengizinkan konten NSFW
            "response_format": "b64_json",
            "allow_nsfw": True  # Parameter tambahan untuk NSFW
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.together.xyz/v1/images/generations",
                headers=headers,
                json=data,
                timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error generating image: {error_text}")
                    
                    # Tangani error NSFW
                    if "NSFW content" in error_text:
                        await update.message.reply_text("Maaf, konten terdeteksi sebagai NSFW. Coba dengan prompt yang berbeda.")
                    return None

                result = await response.json()
                if 'data' in result and len(result['data']) > 0:
                    return result['data'][0]['b64_json']
                return None

    except Exception as e:
        logger.exception("Error in generate_image")
        return None

async def process_image_with_gemini(image_bytes: BytesIO, prompt: str = None) -> Optional[str]:
    try:
        # Inisialisasi model Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Konversi BytesIO ke PIL Image
        image = Image.open(image_bytes)

        # Gunakan prompt default jika tidak ada prompt yang diberikan
        user_prompt = prompt if prompt else "Apa isi gambar ini? Berikan deskripsi detail dalam Bahasa Indonesia."

        # Proses gambar dengan Gemini
        response = model.generate_content([user_prompt, image])

        # Kembalikan teks hasil analisis
        return response.text

    except Exception as e:
        logger.exception("Error in processing image with Gemini")
        return "Terjadi kesalahan saat memproses gambar dengan Gemini."

async def process_with_gemini(messages: List[Dict[str, str]]) -> Optional[str]:
    try:
        # Konversi format pesan ke format yang diterima Gemini
        gemini_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                continue  # Skip system messages
            gemini_messages.append({"role": msg['role'], "parts": [msg['content']]})

        # Mulai chat dengan Gemini
        chat = gemini_model.start_chat(history=gemini_messages)
        
        # Kirim pesan terakhir ke Gemini
        last_message = messages[-1]['content']
        response = chat.send_message(last_message)
        
        return response.text

    except Exception as e:
        logger.exception("Error in processing with Gemini")
        # Fallback ke Mistral jika Gemini gagal
        try:
            return await process_with_mistral(messages)
        except:
            return None


async def process_image_with_pixtral_multiple(image_path: str, prompt: str = None, repetitions: int = 2) -> List[str]:
    try:
        base64_image = await encode_image(image_path)
        results = []
        
        async def single_request():
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }

            # Use custom prompt if provided, otherwise use default
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
            retry_delay = 2  # seconds

            for attempt in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://api.mistral.ai/v1/chat/completions",
                            headers=headers,
                            json=data
                        ) as response:
                            if response.status == 429:  # Too Many Requests
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

        # Proses requests secara sequential dengan delay
        for i in range(repetitions):
            result = await single_request()
            results.append(result)
            if i < repetitions - 1:  # Tidak perlu delay setelah request terakhir
                await asyncio.sleep(1)  # Delay 1 detik antara requests

        return results

    except Exception as e:
        logger.exception("Error in processing image with Pixtral multiple")
        return ["Terjadi kesalahan saat memproses gambar."] * repetitions

async def process_voice_to_text(update: Update) -> Optional[str]:
    """Proses file suara menjadi teks dengan optimasi untuk Railway"""
    try:
        logger.info("Memulai pemrosesan pesan suara...")
        voice_file = await update.message.voice.get_file()

        # Gunakan BytesIO untuk mengurangi penggunaan disk
        with BytesIO() as ogg_bytes, BytesIO() as wav_bytes:
            # Download file langsung ke memory
            ogg_data = await voice_file.download_as_bytearray()
            ogg_bytes.write(ogg_data)
            ogg_bytes.seek(0)
            
            # Preprocessing audio dalam memory
            audio = AudioSegment.from_ogg(ogg_bytes)
            audio = (audio
                    .set_channels(1)  # Convert to mono
                    .set_frame_rate(16000)  # Set to 16kHz
                    .normalize())  # Normalize volume
            
            # Export ke WAV dalam memory
            audio.export(wav_bytes, format='wav')
            wav_bytes.seek(0)
            
            # Recognize dengan multiple engines
            recognizer = sr.Recognizer()
            
            # Konfigurasi recognizer
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.dynamic_energy_adjustment_damping = 0.15
            recognizer.dynamic_energy_ratio = 1.5
            
            with sr.AudioFile(wav_bytes) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                audio_data = recognizer.record(source)
                
                # Try Google Speech Recognition
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
    if complexity == "simple":
        return MAX_CONVERSATION_MESSAGES_SIMPLE
    elif complexity == "medium":
        return MAX_CONVERSATION_MESSAGES_MEDIUM
    elif complexity == "complex":
        return MAX_CONVERSATION_MESSAGES_COMPLEX
    else:
        return MAX_CONVERSATION_MESSAGES_MEDIUM  # Default

async def filter_text(text: str) -> str:
    """Filter untuk menghapus karakter tertentu seperti asterisks (*) dan #, serta kata 'Mistral'"""
    filtered_text = text.replace("*", "").replace("#", "").replace("Mistral AI", "PAIDI").replace("oleh Google", "PAIDI").replace("Mistral", "PAIDI").replace("Tentu, ", "")
    return filtered_text.strip()

async def process_with_mistral(messages: List[Dict[str, str]]) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Tambahkan instruksi sistem agar respon default dalam Bahasa Indonesia
    messages.insert(0, {"role": "system", "content": "Pastikan semua respons diberikan dalam Bahasa Indonesia Yang Mudah Di pahami."})

    data = {
        "model": "mistral-large-latest",
        "messages": messages,
        "max_tokens": 10000
    }

    for attempt in range(MAX_RETRIES):
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    response.raise_for_status()
                    json_response = await response.json()

                    if 'choices' in json_response and json_response['choices']:
                        content = json_response['choices'][0]['message']['content']
                        return await filter_text(content)

        except aiohttp.ClientError as e:
            logger.error(f"Percobaan {attempt + 1} gagal karena error HTTP: {str(e)}")
        except asyncio.TimeoutError:
            logger.error(f"Percobaan {attempt + 1} gagal karena timeout")
        except Exception as e:
            logger.error(f"Percobaan {attempt + 1} gagal: {str(e)}")

        if attempt < MAX_RETRIES - 1:
            logger.info(f"Menunggu {RETRY_DELAY} detik sebelum percobaan berikutnya...")
            await asyncio.sleep(RETRY_DELAY)

    return "Maaf, server tidak merespons setelah beberapa percobaan. Mohon coba lagi nanti."

async def send_voice_response(update: Update, text: str):
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
    try:
        chat_id = update.message.chat_id

        # Periksa apakah sesi sudah ada
        if not redis_client.exists(f"session:{chat_id}"):
            await initialize_session(chat_id)

        # Periksa ukuran file audio
        if update.message.voice.file_size > MAX_AUDIO_SIZE:
            await update.message.reply_text("Maaf, file audio terlalu besar (maksimal 20MB)")
            return

        # Update statistik
        bot_statistics["total_messages"] += 1
        bot_statistics["voice_messages"] += 1

        # Tampilkan indikator "recording voice"
        await context.bot.send_chat_action(chat_id=update.message.chat_id, action="record_voice")

        # Kirim pesan "Sedang memproses pesan suara Anda..."
        processing_msg = await update.message.reply_text("Sedang memproses pesan suara Anda...")

        try:
            # Proses pesan suara menjadi teks
            text = await process_voice_to_text(update)
            if text:
                await update.message.reply_text(f"Teks hasil transkripsi suara Anda:\n{text}")

                # Ambil sesi dari Redis
                session = json.loads(redis_client.get(f"session:{chat_id}"))

                # Tambahkan pesan pengguna ke sesi
                session['messages'].append({"role": "user", "content": text})
                await update_session(chat_id, {"role": "user", "content": text})

                # Proses pesan dengan Mistral
                mistral_messages = session['messages'][-50:]
                response = await process_with_mistral(mistral_messages)

                if response:
                    # Tambahkan respons asisten ke sesi
                    session['messages'].append({"role": "assistant", "content": response})
                    await update_session(chat_id, {"role": "assistant", "content": response})

                    # Filter dan kirim respons
                    filtered_response = await filter_text(response)
                    await update.message.reply_text(filtered_response)
                    await send_voice_response(update, filtered_response)
            else:
                await update.message.reply_text("Maaf, saya tidak dapat mengenali suara dengan jelas. Mohon coba lagi.")

        finally:
            # Hapus pesan "Sedang memproses..."
            await processing_msg.delete()

    except Exception as e:
        # Tangani error
        bot_statistics["errors"] += 1
        logger.exception("Error dalam handle_voice")
        await update.message.reply_text("Maaf, terjadi kesalahan dalam pemrosesan suara.")

async def upload_image_to_telegraph(image_bytes: bytes) -> Optional[str]:
    """Upload image to Telegraph and return the URL"""
    try:
        # Buat form data dengan content-type yang benar
        form = aiohttp.FormData()
        form.add_field(
            'file', 
            image_bytes, 
            filename='image.jpg', 
            content_type='image/jpeg'
        )
        
        # Gunakan timeout yang lebih lama
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {
            'Accept': 'application/json'
        }
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                'https://telegra.ph/upload', 
                data=form,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result and isinstance(result, list) and len(result) > 0:
                        return f"https://telegra.ph{result[0]['src']}"
                    else:
                        logger.error(f"Invalid response from Telegraph: {result}")
                else:
                    logger.error(f"Telegraph upload failed with status {response.status}")
                    
    except Exception as e:
        logger.error(f"Error uploading to Telegraph: {str(e)}")
    return None

async def get_google_image_search_url(image_url: str) -> str:
    """Generate Google Lens search URL"""
    encoded_url = urllib.parse.quote(image_url)
    return f"https://lens.google.com/uploadbyurl?url={encoded_url}"

async def search_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /carigambar command"""
    try:
        # Cek apakah ada reply ke gambar
        if not update.message.reply_to_message or not update.message.reply_to_message.photo:
            await update.message.reply_text(
                "Cara penggunaan:\n"
                "1. Reply ke gambar yang ingin dicari\n"
                "2. Ketik: /carigambar"
            )
            return

        processing_msg = await update.message.reply_text("🔄 Sedang memproses pencarian gambar...")

        try:
            # Ambil gambar dengan resolusi tertinggi
            photo = update.message.reply_to_message.photo[-1]
            photo_file = await photo.get_file()

            # Download gambar
            photo_bytes = await photo_file.download_as_bytearray()
            
            # Coba upload ke Telegraph dengan retry
            telegraph_url = None
            max_retries = 3
            
            for attempt in range(max_retries):
                telegraph_url = await upload_image_to_telegraph(photo_bytes)
                if telegraph_url:
                    break
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Tunggu 1 detik sebelum retry
            
            if not telegraph_url:
                await update.message.reply_text(
                    "❌ Gagal mengupload gambar. Silakan coba lagi dalam beberapa saat."
                )
                return

            # Generate dan kirim URL Google Lens
            google_search_url = await get_google_image_search_url(telegraph_url)
            await update.message.reply_text(
                "🔍 Hasil pencarian gambar:\n\n"
                f"🌐 Cari dengan Google Lens:\n{google_search_url}\n\n"
                "ℹ️ Klik link di atas untuk melihat hasil pencarian gambar serupa di Google"
            )

        finally:
            # Hapus pesan processing
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception:
                    pass

    except Exception as e:
        logger.exception("Error in search_image_command")
        await update.message.reply_text(
            "❌ Terjadi kesalahan. Silakan coba lagi nanti."
        )
        
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    chat_type = update.message.chat.type
    caption = update.message.caption or ""

    # Check rate limit
    user_id = update.message.from_user.id
    if not await check_rate_limit(user_id):
        await update.message.reply_text("Anda telah melebihi batas permintaan. Mohon tunggu beberapa saat.")
        return

    # Periksa apakah ini di grup
    if chat_type in ["group", "supergroup"]:
        if f"@{context.bot.username}" not in caption:
            logger.info("Gambar di grup diabaikan karena tidak ada mention.")
            return  # Abaikan jika tidak ada mention di grup

    try:
        # Periksa apakah sesi sudah ada di Redis
        if not redis_client.exists(f"session:{chat_id}"):
            await initialize_session(chat_id)

        # Ambil sesi dari Redis
        session = json.loads(redis_client.get(f"session:{chat_id}"))

        # Update statistik
        bot_statistics["total_messages"] += 1
        bot_statistics["photo_messages"] += 1

        # Kirim pesan "Sedang menganalisa gambar..."
        processing_msg = await update.message.reply_text("Sedang menganalisa gambar...🔍🧐")

        # Ambil file gambar
        photo_file = await update.message.photo[-1].get_file()

        # Proses gambar menggunakan BytesIO
        with BytesIO() as temp_file:
            photo_bytes = await photo_file.download_as_bytearray()
            temp_file.write(photo_bytes)
            temp_file.seek(0)

            # Siapkan prompt berdasarkan caption
            prompt = caption.replace(f"@{context.bot.username}", "").strip() if caption else None

            # Jika tidak ada prompt, gunakan prompt default dalam Bahasa Indonesia
            if not prompt:
                prompt = "Apa isi gambar ini? Berikan deskripsi detail dalam Bahasa Indonesia."
            else:
                # Tambahkan instruksi untuk merespons dalam Bahasa Indonesia
                prompt += " jawab dalam Bahasa Indonesia."

            # Proses gambar dengan Gemini
            gemini_result = await process_image_with_gemini(temp_file, prompt=prompt)

            if gemini_result:
                # Filter hasil analisis
                filtered_result = await filter_text(gemini_result)

                # Pecah hasil analisis jika terlalu panjang
                result_parts = split_message(filtered_result)
                for part in result_parts:
                    await update.message.reply_text(f"Analisa:\n{part}")

                # Simpan hasil analisis ke sesi
                session['messages'].append({
                    "role": "user",
                    "content": f"[User mengirim gambar]" + (f" dengan pertanyaan: {prompt}" if prompt else "")
                })
                session['messages'].append({
                    "role": "assistant",
                    "content": filtered_result
                })
                session['last_image_analysis'] = filtered_result
                await update_session(chat_id, {"role": "assistant", "content": filtered_result})
            else:
                await update.message.reply_text("Maaf, tidak dapat menganalisa gambar. Silakan coba lagi.")

        # Hapus pesan "Sedang menganalisa..."
        await processing_msg.delete()

    except Exception as e:
        # Tangani error
        logger.exception("Error dalam proses analisis gambar dengan Gemini")
        await update.message.reply_text("Terjadi kesalahan saat memproses gambar.")

        
async def cleanup_sessions(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Memulai proses pembersihan sesi yang tidak aktif")
    try:
        for key in redis_client.scan_iter("session:*"):
            session = json.loads(redis_client.get(key))
            last_update = session.get('last_update', 0)

            if datetime.now().timestamp() - last_update > CONVERSATION_TIMEOUT:
                redis_client.delete(key)
                logger.info(f"Sesi dengan kunci {key} telah dihapus karena tidak aktif")
    except redis.RedisError as e:
        logger.error(f"Gagal membersihkan sesi: {str(e)}")

async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan yang di-mention atau reply di grup."""
    chat_type = update.message.chat.type

    # Hanya proses jika di grup dan ada mention atau reply ke bot
    if chat_type in ["group", "supergroup"]:
        should_process = False
        message_text = update.message.text or update.message.caption or ""

        # Cek mention
        if f'@{context.bot.username}' in message_text:
            message_text = message_text.replace(f'@{context.bot.username}', '').strip()
            should_process = True

        # Cek reply
        elif update.message.reply_to_message and \
             update.message.reply_to_message.from_user.id == context.bot.id:
            should_process = True

        if should_process and message_text:
            # Cek jika pesan mengandung perintah /gambar atau /image
            if message_text.lower().startswith(('/gambar', '/image')):
                await handle_text(update, context, message_text)
                return

            # Lanjutkan pemrosesan pesan biasa
            chat_id = update.message.chat_id

            # Periksa apakah sesi Redis sudah ada
            if not redis_client.exists(f"session:{chat_id}"):
                await initialize_session(chat_id)

            # Reset konteks jika diperlukan
            if await should_reset_context(chat_id, message_text):
                await initialize_session(chat_id)

            # Ambil sesi dari Redis
            session = json.loads(redis_client.get(f"session:{chat_id}"))

            # Tambahkan pesan pengguna ke sesi
            session['messages'].append({"role": "user", "content": message_text})
            await update_session(chat_id, {"role": "user", "content": message_text})

            # Cek apakah pesan terkait dengan konteks sebelumnya
            if session['messages'] and not is_related_to_context(message_text, session['messages']):
                await update.message.reply_text("Sepertinya pertanyaan Anda tidak terkait dengan topik sebelumnya. Mari kita kembali ke topik sebelumnya.")
                # Tampilkan konteks terakhir
                last_context = session['messages'][-1]['content']
                await update.message.reply_text(f"Topik terakhir: {last_context}")
                return

            # Proses pesan dengan konteks cerdas
            response = await process_with_smart_context(session['messages'][-10:])
            
            if response:
                # Filter hasil respons sebelum dikirim ke pengguna
                filtered_response = await filter_text(response)

                # Tambahkan respons asisten ke sesi
                session['messages'].append({"role": "assistant", "content": filtered_response})
                await update_session(chat_id, {"role": "assistant", "content": filtered_response})

                # Pecah respons jika terlalu panjang
                response_parts = split_message(filtered_response)
                for part in response_parts:
                    await update.message.reply_text(part)
        else:
            logger.info("Pesan di grup tanpa mention yang valid diabaikan.")


async def initialize_session(chat_id: int) -> None:
    """Inisialisasi sesi baru di Redis"""
    try:
        session = {
            'messages': [],
            'last_update': datetime.now().timestamp(),
            'conversation_id': str(uuid.uuid4())
        }

        # Simpan sesi ke Redis sebagai JSON
        redis_client.set(f"session:{chat_id}", json.dumps(session))
        redis_client.expire(f"session:{chat_id}", CONVERSATION_TIMEOUT)
        logger.info(f"Sesi berhasil dibuat untuk chat_id {chat_id}")
    except redis.RedisError as e:
        logger.error(f"Gagal membuat sesi untuk chat_id {chat_id}: {str(e)}")
        raise Exception("Gagal menginisialisasi sesi.")


async def process_with_smart_context(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Proses pesan dengan konteks yang lebih cerdas.
    """
    # Ekstrak kata kunci yang relevan dari histori percakapan
    relevant_keywords = extract_relevant_keywords(messages)
    
    # Tambahkan informasi penting ke konteks
    if relevant_keywords:
        messages.insert(0, {"role": "system", "content": f"Informasi penting: {', '.join(relevant_keywords)}"})
    
    # Proses pesan dengan model AI (Gemini atau Mistral)
    response = await process_with_gemini(messages)
    if not response:
        response = await process_with_mistral(messages)
    
    return response
def extract_relevant_keywords(messages: List[Dict[str, str]], top_n: int = 5) -> List[str]:
    context_text = " ".join([msg['content'] for msg in messages])
    words = re.findall(r'\b\w+\b', context_text.lower())
    word_counts = Counter(words)
    common_words = word_counts.most_common(top_n)

    stop_words = {
        "saya", "anda", "di", "yang", "dan", "apa", "berapa", "bagaimana", "adalah", "nama",
        "ini", "itu", "untuk", "dengan", "pada", "dalam", "atau", "dari", "ke", "sebuah",
        "tidak", "bisa", "akan", "jika", "karena", "oleh", "ada", "sudah", "kami", "kita",
        "mereka", "jadi", "saat", "lagi", "juga", "semua", "lebih", "kurang", "perlu",
        "hanya", "sangat", "sama", "setiap", "ketika", "bahwa", "dapat", "tetapi",
        "seperti", "kemudian", "tanpa", "apakah", "siapa", "dimana", "mengapa",
        "kapan", "dengan", "oleh", "agar", "supaya", "yaitu", "walaupun", "meskipun",
        "tersebut", "maupun", "namun", "selain", "setelah", "sebelum", "saat", "karena"
    }

    relevant_keywords = [word for word, count in common_words if word not in stop_words]
    return relevant_keywords

def is_same_topic(last_message: str, current_message: str, context_messages: List[Dict[str, str]]) -> bool:
    relevant_keywords = extract_relevant_keywords(context_messages)
    last_keywords = [word for word in relevant_keywords if word in last_message.lower()]
    current_keywords = [word for word in relevant_keywords if word in current_message.lower()]
    return bool(set(last_keywords) & set(current_keywords))

def is_related_to_context(current_message: str, context_messages: List[Dict[str, str]]) -> bool:
    relevant_keywords = extract_relevant_keywords(context_messages)
    return any(keyword in current_message.lower() for keyword in relevant_keywords)

async def should_reset_context(chat_id: int, message: str) -> bool:
    try:
        session_json = redis_client.get(f"session:{chat_id}")
        if not session_json:
            return True

        session = json.loads(session_json)
        last_update = session.get('last_update', 0)
        current_time = datetime.now().timestamp()
        time_diff = current_time - last_update

        # Reset jika percakapan sudah terlalu lama
        if time_diff > CONVERSATION_TIMEOUT:
            return True

        # Reset jika pesan mengandung kata kunci awal percakapan
        keywords = ['halo', 'hai', 'hi', 'hello', 'permisi', '?']
        starts_with_keyword = any(message.lower().startswith(keyword) for keyword in keywords)
        if starts_with_keyword:
            return True

        # Tentukan kompleksitas percakapan
        complexity = determine_conversation_complexity(session['messages'])

        # Dapatkan batas pesan berdasarkan kompleksitas
        max_messages = get_max_conversation_messages(complexity)

        # Reset jika percakapan sudah terlalu panjang
        if len(session['messages']) > max_messages:
            return True

        # Reset jika terjadi perubahan topik
        if session['messages']:
            last_message = session['messages'][-1]['content']
            if not is_same_topic(last_message, message, session['messages']):
                return True

        return False
    except redis.RedisError as e:
        logger.error(f"Redis Error saat memeriksa konteks untuk chat_id {chat_id}: {str(e)}")
        return True

async def update_session(chat_id: int, message: Dict[str, str]) -> None:
    try:
        logger.info(f"Memulai pembaruan sesi untuk chat_id {chat_id}")
        session_json = redis_client.get(f"session:{chat_id}")
        
        if session_json:
            session = json.loads(session_json)
            logger.info(f"Data sesi ditemukan untuk chat_id {chat_id}")
        else:
            await initialize_session(chat_id)
            session = {'messages': [], 'last_update': datetime.now().timestamp()}
            logger.info(f"Sesi baru diinisialisasi untuk chat_id {chat_id}")

        # Tambahkan pesan baru ke sesi
        session['messages'].append(message)

        # Tentukan kompleksitas percakapan
        complexity = determine_conversation_complexity(session['messages'])

        # Dapatkan batas pesan berdasarkan kompleksitas
        max_messages = get_max_conversation_messages(complexity)

        # Batasi jumlah pesan dalam sesi
        if len(session['messages']) > max_messages:
            session['messages'] = session['messages'][-max_messages:]
            logger.info(f"Pesan di sesi untuk chat_id {chat_id} dibatasi hingga {max_messages} pesan")

        # Simpan sesi ke Redis
        redis_client.set(f"session:{chat_id}", json.dumps(session))
        redis_client.expire(f"session:{chat_id}", CONVERSATION_TIMEOUT)
        logger.info(f"Sesi berhasil diperbarui untuk chat_id {chat_id}")
    except redis.RedisError as e:
        logger.error(f"Gagal memperbarui sesi untuk chat_id {chat_id}: {str(e)}")
        raise Exception("Gagal memperbarui sesi.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, message_text: Optional[str] = None):
    # Jika message_text tidak diberikan, ambil dari update.message
    if not message_text:
        message_text = update.message.text or ""

    # Sanitasi input
    sanitized_text = sanitize_input(message_text)

    # Cek rate limit
    user_id = update.message.from_user.id
    if not await check_rate_limit(user_id):
        await update.message.reply_text("Anda telah melebihi batas permintaan. Mohon tunggu beberapa saat.")
        return

    # Cek jika pesan mengandung perintah /gambar atau /image
    if sanitized_text.lower().startswith(('/gambar', '/image')):
        # Extract the prompt
        prompt = sanitized_text.split(' ', 1)[1] if len(sanitized_text.split(' ', 1)) > 1 else None

        if not prompt:
            await update.message.reply_text("Mohon berikan prompt untuk generate gambar. Contoh: /gambar kucing lucu")
            return

        processing_msg = await update.message.reply_text("Sedang membuat gambar...")

        try:
            image_data = await generate_image(update, prompt)
            if image_data:
                # Convert base64 to image
                image_bytes = base64.b64decode(image_data)
                bio = BytesIO(image_bytes)
                bio.seek(0)

                # Send the image
                await update.message.reply_photo(
                    photo=bio,
                    caption=f"Hasil generate gambar untuk prompt: {prompt}"
                )
            else:
                await update.message.reply_text("Maaf, terjadi kesalahan saat membuat gambar.")
        except Exception as e:
            logger.exception("Error generating image")
            await update.message.reply_text("Maaf, terjadi kesalahan saat membuat gambar.")
        finally:
            await processing_msg.delete()
        return

    # Tambahkan statistik
    bot_statistics["total_messages"] += 1
    bot_statistics["text_messages"] += 1

    # Ambil chat_id dari update
    chat_id = update.message.chat_id

    # Periksa apakah sesi sudah ada di Redis
    session_json = redis_client.get(f"session:{chat_id}")
    if not session_json:
        await initialize_session(chat_id)
        session = {'messages': [], 'last_update': datetime.now().timestamp()}
    else:
        session = json.loads(session_json)

    # Tambahkan pesan pengguna ke sesi
    session['messages'].append({"role": "user", "content": sanitized_text})
    await update_session(chat_id, {"role": "user", "content": sanitized_text})

    # Proses pesan dengan konteks cerdas
    response = await process_with_smart_context(session['messages'][-10:])
    
    if response:
        # Filter hasil respons sebelum dikirim ke pengguna
        filtered_response = await filter_text(response)

        # Tambahkan respons asisten ke sesi
        session['messages'].append({"role": "assistant", "content": filtered_response})
        await update_session(chat_id, {"role": "assistant", "content": filtered_response})

        # Pecah respons jika terlalu panjang
        response_parts = split_message(filtered_response)
        for part in response_parts:
            await update.message.reply_text(part)
        
async def reset_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    redis_client.delete(f"session:{chat_id}")
    await update.message.reply_text("Sesi percakapan Anda telah direset.")

def main():
    if not check_required_settings():
        print("Bot tidak bisa dijalankan karena konfigurasi tidak lengkap")
        return

    try:
        # Inisialisasi application
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stats", stats))
        application.add_handler(CommandHandler("reset", reset_session))
        application.add_handler(CommandHandler("carigambar", search_image_command))

        # Message handlers dengan prioritas
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        
        # Handler baru untuk text dengan mention di grup
        application.add_handler(MessageHandler(
            (filters.TEXT | filters.CAPTION) & 
            (filters.Entity("mention") | filters.REPLY), 
            handle_mention
        ))
        
        # Handler baru untuk chat pribadi
        application.add_handler(MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE,
            handle_text
        ))

        # Cleanup session setiap jam
        application.job_queue.run_repeating(cleanup_sessions, interval=3600, first=10)

        application.run_polling()

    except Exception as e:
        logger.critical(f"Error fatal saat menjalankan bot: {e}")
        raise

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):  # <-- Tambahkan ini
    user_id = update.message.from_user.id
    current_time = datetime.now()

    # Cek kapan terakhir pengguna mengirim pesan
    last_message_time = redis_client.get(f"last_message_time_{user_id}")
    if last_message_time:
        last_message_time = datetime.fromtimestamp(float(last_message_time))
        if current_time - last_message_time < timedelta(seconds=5):  # Batasan: 1 pesan per 5 detik
            await update.message.reply_text("Anda mengirim pesan terlalu cepat. Mohon tunggu beberapa detik.")
            return

    # Update waktu terakhir pengguna mengirim pesan
    redis_client.set(f"last_message_time_{user_id}", current_time.timestamp())

    # Lanjutkan pemrosesan pesan
    await handle_text(update, context)

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active_sessions = len(list(redis_client.scan_iter("session:*")))
    stats_message = (
        f"Statistik Bot:\n"
        f"- Total Pesan: {bot_statistics['total_messages']}\n"
        f"- Pesan Suara: {bot_statistics['voice_messages']}\n"
        f"- Pesan Teks: {bot_statistics['text_messages']}\n"
        f"- Kesalahan: {bot_statistics['errors']}\n"
        f"- Sesi Aktif: {active_sessions}"
    )
    await update.message.reply_text(stats_message)

if __name__ == '__main__':
    asyncio.run(main())
