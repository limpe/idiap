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
import bleach
import requests



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
MAX_AUDIO_SIZE = 20 * 1024 * 1024  # 20MB

def check_required_settings():
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN tidak ditemukan!")
        return False
    if not MISTRAL_API_KEY:
        print("Error: MISTRAL_API_KEY tidak ditemukan!")
        return False
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY tidak ditemukan!")
        return False
    if not TOGETHER_API_KEY:
        print("Error: TOGETHER_API_KEY tidak ditemukan!")
        return False
    return True

def sanitize_input(text: str) -> str:
    # Daftar tag dan atribut yang diizinkan
    allowed_tags = ['b', 'i']  # Mengizinkan tag <b> dan <i>
    allowed_attributes = {}  # Tidak mengizinkan atribut apa pun

    # Membersihkan teks menggunakan Bleach
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
client = Together()

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
    - pengingat (gunakan /ingatkan 5 belanja *mengingatkan anda 5 menit kedepan untuk belanja)

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
    
async def determine_conversation_complexity(messages: List[Dict[str, str]]) -> str:
    """
    Menentukan kompleksitas percakapan berdasarkan input pengguna.
    """
    # Ambil hanya pesan dari pengguna (role 'user')
    user_messages = [msg.get('content', '') for msg in messages if msg.get('role') == 'user']
    
    # Gabungkan semua pesan pengguna menjadi satu teks
    user_text = " ".join(user_messages).lower()
    
    # Cek kata kunci kompleks dalam input pengguna
    if any(keyword in user_text for keyword in complex_keywords):
        logger.info(f"Kata kunci kompleks ditemukan dalam input pengguna: {user_text}")
        logger.info("Kompleksitas ditingkatkan ke: complex (karena kata kunci kompleks).")
        return "complex"
    
    # Hitung jumlah pesan pengguna
    num_user_messages = len(user_messages)
    logger.info(f"Jumlah pesan pengguna dalam percakapan: {num_user_messages}")
    
    # Tentukan kompleksitas berdasarkan jumlah pesan pengguna
    if num_user_messages > 15:
        logger.info("Kompleksitas ditingkatkan ke: complex (jumlah pesan pengguna > 15).")
        return "complex"
    elif num_user_messages > 5:
        logger.info("Kompleksitas ditingkatkan ke: medium (jumlah pesan pengguna > 5).")
        return "medium"
    else:
        logger.info("Kompleksitas tetap: simple (jumlah pesan pengguna <= 5).")
        return "simple"

async def set_reminder(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Parse waktu dan pesan dari perintah pengguna
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Format: /ingatkan <waktu> <pesan>")
            return

        time_str = args[0]
        message = " ".join(args[1:])

        # Konversi waktu ke detik
        try:
            time_seconds = int(time_str) * 60  # Misalnya, /reminder 5 Pesan
        except ValueError:
            await update.message.reply_text("Format waktu tidak valid. Gunakan angka (misalnya, 5).")
            return

        # Jadwalkan pengingat
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
    job = context.job
    await context.bot.send_message(chat_id=job.chat_id, text=f"⏰ Pengingat: {job.data}")

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

async def translate_to_english(text: str) -> str:
    """
    Menerjemahkan teks ke Bahasa Inggris menggunakan deep-translator.
    """
    try:
        translation = GoogleTranslator(source='auto', target='en').translate(text)
        return translation
    except Exception as e:
        logger.error(f"Error translating text to English: {str(e)}")
        return text  # Kembalikan teks asli jika terjemahan gagal

async def generate_image(update: Update, prompt: str) -> Optional[str]:
    try:
        # Terjemahkan prompt ke Bahasa Inggris (jika diperlukan)
        english_prompt = await translate_to_english(prompt)
        logger.info(f"Original prompt: {prompt}, Translated prompt: {english_prompt}")

        # Inisialisasi Together client
        client = Together()
        
        try:
            # Generate gambar menggunakan Together API
            response = client.images.generate(
                prompt=english_prompt,
                model="black-forest-labs/FLUX.1-schnell-Free",
                width=1440,
                height=960,
                steps=4,
                n=1,
                response_format="b64_json"
            )
            
            # Ambil base64 string dari response
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

# Langkah 3: Update fungsi handle_generate_image
async def handle_generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Pisahkan command dan pesan
        message_text = update.message.text or ""
        is_group = update.message.chat.type in ["group", "supergroup"]
        bot_username = context.bot.username.lower()
        
        # Penanganan untuk grup
        if is_group:
            # Cek apakah ada mention bot
            if f"@{bot_username}" not in message_text.lower():
                logger.info(f"Pesan di grup tanpa mention yang valid diabaikan. Pesan: {message_text}")
                return

            # Hapus mention bot dari pesan
            message_text = message_text.replace(f"@{bot_username}", "").strip()
        
        # Ekstrak prompt dari pesan
        # Hapus command /gambar atau /image dari awal pesan
        prompt = re.sub(r'^/(?:gambar|image)\s*', '', message_text).strip()
        
        if not prompt:
            await update.message.reply_text("Mohon berikan prompt untuk menghasilkan gambar. Contoh: /gambar pemandangan gunung")
            return

        # Kirim pesan "Sedang memproses..."
        processing_msg = await update.message.reply_text("🔄 Sedang menghasilkan gambar...")

        try:
            # Panggil fungsi untuk menghasilkan gambar
            image_data = await generate_image(update, prompt)

            if image_data:
                # Decode base64 string ke bytes
                image_bytes = base64.b64decode(image_data)
                # Kirim gambar menggunakan BytesIO
                with BytesIO(image_bytes) as bio:
                    bio.seek(0)
                    await update.message.reply_photo(photo=bio)
            else:
                await update.message.reply_text("Maaf, gagal menghasilkan gambar. Silakan coba lagi.")

        finally:
            # Hapus pesan "Sedang memproses..."
            await processing_msg.delete()

    except Exception as e:
        logger.error(f"Error dalam handle_generate_image: {e}")
        await update.message.reply_text("Terjadi kesalahan saat menghasilkan gambar.")

async def process_image_with_gemini(image_bytes: BytesIO, prompt: str = None) -> Optional[str]:
    try:
        # Inisialisasi model Gemini
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Konversi BytesIO ke PIL Image
        image = Image.open(image_bytes)

        # Gunakan prompt default jika tidak ada prompt yang diberikan
        user_prompt = prompt if prompt else "Apa isi gambar ini? Berikan deskripsi detail dalam Bahasa Indonesia dan termasuk penjelasan yang komprehensif."

        # Proses gambar dengan Gemini
        response = model.generate_content([user_prompt, image])

        # Kembalikan teks hasil analisis
        return response.text

    except Exception as e:
        logger.exception("Error in processing image with Gemini")
        return "Terjadi kesalahan saat memproses gambar dengan Gemini."

async def process_with_gemini(messages: List[Dict[str, str]]) -> Optional[str]:
    try:
        # Tentukan kompleksitas percakapan berdasarkan input pengguna
        complexity = await determine_conversation_complexity(messages)
        
        # Cek apakah pesan sistem sudah ada di awal percakapan
        if not any(msg.get('parts', [{}])[0].get('text', '').startswith("Berikan respons") for msg in messages):
            # Tambahkan instruksi sistem berdasarkan kompleksitas
            if complexity == "simple":
                system_message = {
                    "role": "user",
                    "parts": [{"text": "Berikan respons singkat, jelas, dan point pentingnya saja dalam Bahasa Indonesia."}]
                }
            elif complexity == "medium":
                system_message = {
                    "role": "user",
                    "parts": [{"text": "Berikan respons yang lebih detail dalam Bahasa Indonesia, namun tetap mudah dipahami."}]
                }
            elif complexity == "complex":
                system_message = {
                    "role": "user",
                    "parts": [{"text": "Berikan respons yang sangat detail dan mendalam dalam Bahasa Indonesia, termasuk penjelasan yang komprehensif."}]
                }
            
            # Sisipkan instruksi sistem di awal pesan
            messages.insert(0, system_message)

        # Konversi format pesan ke format yang diterima Gemini
        gemini_messages = []
        for msg in messages:
            if 'parts' in msg and isinstance(msg['parts'], list):
                gemini_messages.append(msg)
            elif 'content' in msg:
                gemini_messages.append({
                    "role": msg['role'],
                    "parts": [{"text": msg['content']}]
                })
            else:
                logger.error(f"Pesan tidak valid: {msg}")
                continue

        # Mulai chat dengan Gemini
        chat = gemini_model.start_chat(history=gemini_messages)
        
        # Ambil pesan terakhir dari pengguna
        last_message = messages[-1]
        user_message = last_message.get('content', '') if 'content' in last_message else last_message.get('parts', [{}])[0].get('text', '')

        # Cek apakah pesan mengandung kata kunci "sumber youtube" atau "sumber terkait"
        if "sumber youtube" in user_message.lower() or "sumber terkait" in user_message.lower():
            # Gunakan google_search untuk mencari sumber terkait
            response = gemini_model.generate_content(
                contents=user_message,
                tools={"google_search": {
                    "dynamic_retrieval_config": {
                        "mode": "unspecified",
                        "dynamic_threshold": 0.06
                    }
                }}
            )
        else:
            # Proses pesan seperti biasa
            try:
                response = chat.send_message(user_message)
            except generation_types.StopCandidateException as e:
                logger.error(f"Error RECITATION: {e}")
                return "Maaf, saya tidak dapat memberikan respons untuk pesan ini karena batasan keamanan."
            except Exception as e:
                logger.exception(f"Error saat mengirim pesan ke Gemini: {e}")
                return "Maaf, terjadi kesalahan saat memproses pesan Anda."

        # Kembalikan teks respons
        return response.text

    except Exception as e:
        logger.exception("Error in processing with Gemini")
        return "Maaf, terjadi kesalahan internal. Mohon coba lagi nanti."

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
    """
    Proses file suara menjadi teks dengan optimasi untuk Railway.

    Args:
        update (Update): Objek update dari Telegram yang berisi pesan suara.

    Returns:
        Optional[str]: Teks hasil transkripsi suara, atau None jika gagal.
    """
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
                # Pecah teks hasil transkripsi jika terlalu panjang
                text_parts = split_message(text)
                for part in text_parts:
                    await update.message.reply_text(f"Teks hasil transkripsi suara Anda:\n{part}")

                # Ambil sesi dari Redis
                session = json.loads(redis_client.get(f"session:{chat_id}"))

                # Tambahkan pesan pengguna ke sesi
                session['messages'].append({"role": "user", "content": text})
                await update_session(chat_id, {"role": "user", "content": text})

                # Proses pesan dengan Gemini
                response = await process_with_gemini(session['messages'])

                if response:
                    # Tambahkan respons asisten ke sesi
                    session['messages'].append({"role": "assistant", "content": response})
                    await update_session(chat_id, {"role": "assistant", "content": response})

                    # Filter dan kirim respons
                    filtered_response = await filter_text(response)
                    response_parts = split_message(filtered_response)
                    for part in response_parts:
                        await update.message.reply_text(part)
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
        logger.info(f"Sesi saat ini: {session}")
        
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
                prompt = "Apa isi gambar ini? Berikan deskripsi detail dalam Bahasa Indonesiadan termasuk penjelasan yang komprehensif."
                logger.info(f"Prompt yang digunakan: {prompt}")
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
                session['last_image_analysis'] = filtered_result  # Simpan hasil analisis terakhir
                await update_session(chat_id, {"role": "assistant", "content": filtered_result})
            else:
                await update.message.reply_text("Maaf, tidak dapat menganalisa gambar. Silakan coba lagi.")

        # Hapus pesan "Sedang menganalisa..."
        await processing_msg.delete()

    except Exception as e:
        # Tangani error
        logger.exception("Error dalam proses analisis gambar dengan Gemini")
        await update.message.reply_text("Terjadi kesalahan saat memproses gambar.")

async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    chat_type = update.message.chat.type
    message_text = update.message.text or update.message.caption or ""

    # Hanya proses jika di grup dan ada mention atau reply ke bot
    if chat_type in ["group", "supergroup"]:
        should_process = False

        # Cek mention
        if f'@{context.bot.username}' in message_text:
            message_text = message_text.replace(f'@{context.bot.username}', '').strip()
            should_process = True

        # Cek reply
        elif update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
            should_process = True

        if should_process and message_text:
            # Sanitasi input teks
            sanitized_text = sanitize_input(message_text)

            # Cek apakah pesan mengandung perintah /gambar atau /image
            if sanitized_text.lower().startswith(('/gambar', '/image')):
                await handle_generate_image(update, context)  # Panggil handler untuk /gambar
                return

            # Lanjutkan pemrosesan pesan biasa
            # Periksa apakah sesi sudah ada di Redis
            if not redis_client.exists(f"session:{chat_id}"):
                await initialize_session(chat_id)

            # Ambil sesi dari Redis
            session = json.loads(redis_client.get(f"session:{chat_id}"))

            # Reset konteks jika diperlukan
            if await should_reset_context(chat_id, sanitized_text):
                await initialize_session(chat_id)

            # Tambahkan pesan pengguna ke sesi
            session['messages'].append({"role": "user", "content": sanitized_text})
            await update_session(chat_id, {"role": "user", "content": sanitized_text})

            # Proses pesan dengan konteks cerdas
            response = await process_with_smart_context(session['messages'][-10:])  # Ambil 10 pesan terakhir

            if response:
                # Filter hasil respons
                filtered_response = await filter_text(response)

                # Tambahkan respons asisten ke sesi
                session['messages'].append({"role": "assistant", "content": filtered_response})
                await update_session(chat_id, {"role": "assistant", "content": filtered_response})

                # Kirim respons ke pengguna
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
        redis_client.set(f"session:{chat_id}", json.dumps(session))
        redis_client.expire(f"session:{chat_id}", CONVERSATION_TIMEOUT)
        logger.info(f"Sesi berhasil dibuat untuk chat_id {chat_id}")
    except redis.RedisError as e:
        logger.error(f"Gagal membuat sesi untuk chat_id {chat_id}: {str(e)}")
        raise Exception("Gagal menginisialisasi sesi.")

async def update_session(chat_id: int, message: Dict[str, str]) -> None:
    try:
        session_json = redis_client.get(f"session:{chat_id}")
        if session_json:
            session = json.loads(session_json)
        else:
            await initialize_session(chat_id)
            session = {'messages': [], 'last_update': datetime.now().timestamp()}

        # Tambahkan pesan baru ke sesi
        session['messages'].append(message)
        session['last_update'] = datetime.now().timestamp()

        # Simpan sesi ke Redis dan atur TTL
        redis_client.set(f"session:{chat_id}", json.dumps(session))
        redis_client.expire(f"session:{chat_id}", CONVERSATION_TIMEOUT)
    except redis.RedisError as e:
        logger.error(f"Gagal memperbarui sesi untuk chat_id {chat_id}: {str(e)}")
        raise Exception("Gagal memperbarui sesi.")


async def process_with_smart_context(messages: List[Dict[str, str]]) -> Optional[str]:
    try:
        # Coba Gemini terlebih dahulu
        try:
            response = await asyncio.wait_for(process_with_gemini(messages), timeout=10)
            if response:
                logger.info("Menggunakan respons dari Gemini.")
                return response
        except generation_types.StopCandidateException as e:
            logger.error(f"Gemini RECITATION error: {e}")
        except asyncio.TimeoutError:
            logger.warning("Gemini timeout, beralih ke Mistral.")
        
        # Jika Gemini gagal, coba Mistral
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
    
def extract_relevant_keywords(messages: List[Dict[str, str]], top_n: int = 5) -> List[str]:
    # Gabungkan semua pesan menjadi satu teks
    context_text = " ".join([msg.get('content', '') for msg in messages])
    
    # Ekstrak kata-kata dari teks
    words = re.findall(r'\b\w+\b', context_text.lower())
    word_counts = Counter(words)

    # Filter kata-kata yang relevan (bukan stop words)
    relevant_keywords = [word for word, count in word_counts.most_common(top_n) if word not in stop_words]
    
    # Log untuk debugging
    logger.info(f"Extracted relevant keywords: {relevant_keywords}")
    
    return relevant_keywords

def is_same_topic(last_message: str, current_message: str, context_messages: List[Dict[str, str]], threshold: int = 0) -> bool:
    # Ekstrak kata kunci relevan dari konteks percakapan
    relevant_keywords = extract_relevant_keywords(context_messages)
    
    # Ekstrak kata kunci dari pesan terakhir dan pesan saat ini
    last_keywords = [word for word in relevant_keywords if word in last_message.lower()]
    current_keywords = [word for word in relevant_keywords if word in current_message.lower()]

    # Temukan kata kunci yang sama antara pesan terakhir dan pesan saat ini
    common_keywords = set(last_keywords) & set(current_keywords)
    
    # Log untuk debugging
    logger.info(f"Last keywords: {last_keywords}")
    logger.info(f"Current keywords: {current_keywords}")
    logger.info(f"Common keywords: {common_keywords}")
    
    # Kembalikan True jika jumlah kata kunci yang sama memenuhi threshold
    return len(common_keywords) >= threshold

def is_related_to_context(current_message: str, context_messages: List[Dict[str, str]]) -> bool:
    relevant_keywords = extract_relevant_keywords(context_messages)
    return any(keyword in current_message.lower() for keyword in relevant_keywords)

async def should_reset_context(chat_id: int, message: str) -> bool:
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
            return True

        keywords = ['halo', 'hai', 'hi', 'hello', 'permisi', 'terima kasih', 'terimakasih', 'sip']
        if any(message.lower().startswith(keyword) for keyword in keywords):
            logger.info(f"Reset konteks untuk chat_id {chat_id} karena pesan mengandung kata kunci awal: {message}")
            return True

        complexity = await determine_conversation_complexity(session['messages'])
        max_messages = get_max_conversation_messages(complexity)
        if len(session['messages']) > max_messages:
            logger.info(f"Reset konteks untuk chat_id {chat_id} karena percakapan terlalu panjang (jumlah pesan: {len(session['messages'])}).")
            return True

        if session['messages']:
            last_message = session['messages'][-1]['content']
            if not is_same_topic(last_message, message, session['messages']):
                logger.info(f"Reset konteks untuk chat_id {chat_id} karena perubahan topik.")
                return True

        logger.info(f"Tidak perlu reset konteks untuk chat_id {chat_id}.")
        return False
    except redis.RedisError as e:
        logger.error(f"Redis Error saat memeriksa konteks untuk chat_id {chat_id}: {str(e)}")
        return True


async def reset_session(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /reset. Mereset sesi percakapan pengguna."""
    chat_id = update.message.chat_id
    try:
        # Hapus sesi dari Redis
        redis_client.delete(f"session:{chat_id}")
        await update.message.reply_text("Sesi percakapan Anda telah direset. Mulai percakapan baru sekarang!")
    except Exception as e:
        logger.error(f"Gagal mereset sesi untuk chat_id {chat_id}: {str(e)}")
        await update.message.reply_text("Maaf, terjadi kesalahan saat mereset sesi.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, message_text: Optional[str] = None):
    if not message_text:
        message_text = update.message.text or ""

    sanitized_text = sanitize_input(message_text)

    user_id = update.message.from_user.id
    if not await check_rate_limit(user_id):
        await update.message.reply_text("Anda telah melebihi batas permintaan. Mohon tunggu beberapa saat.")
        return

    chat_id = update.message.chat_id
    session_json = redis_client.get(f"session:{chat_id}")
    if not session_json:
        await initialize_session(chat_id)
        session = {'messages': [], 'last_update': datetime.now().timestamp()}
    else:
        session = json.loads(session_json)

    if await should_reset_context(chat_id, sanitized_text):
        await initialize_session(chat_id)
        session = {'messages': [], 'last_update': datetime.now().timestamp()}

    session['messages'].append({"role": "user", "content": sanitized_text})
    await update_session(chat_id, {"role": "user", "content": sanitized_text})

    if 'last_image_analysis' in session:
        session['messages'].append({
            "role": "assistant",
            "content": session['last_image_analysis']
        })

    try:
        response = await process_with_gemini(session['messages'])
        
        if response:
            filtered_response = await filter_text(response)
            session['messages'].append({"role": "assistant", "content": filtered_response})
            await update_session(chat_id, {"role": "assistant", "content": filtered_response})

            response_parts = split_message(filtered_response)
            for part in response_parts:
                await update.message.reply_text(part)
                
    except Exception as e:
        logger.exception("Error dalam pemrosesan pesan")
        await update.message.reply_text("Maaf, terjadi kesalahan dalam pemrosesan pesan.")


def main():
    if not check_required_settings():
        print("Bot tidak bisa dijalankan karena konfigurasi tidak lengkap")
        return

    try:
        # Initialize Gemini
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            logger.info("Gemini initialized successfully with GOOGLE_API_KEY")
        else:
            logger.warning("GOOGLE_API_KEY tidak ditemukan, fitur tidak akan berfungsi")

        # Initialize application
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stats", stats))
        application.add_handler(CommandHandler("reset", reset_session))
        application.add_handler(CommandHandler("carigambar", search_image_command))
        application.add_handler(CommandHandler("ingatkan", set_reminder))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("gambar", handle_generate_image))  # Tambahkan handler untuk /gambar

        # Message handlers
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(MessageHandler(
            (filters.TEXT | filters.CAPTION) & 
            (filters.Entity("mention") | filters.REPLY), 
            handle_mention
        ))
        application.add_handler(MessageHandler(
            filters.TEXT & filters.ChatType.PRIVATE,
            handle_message
        ))

        # Run bot
        application.run_polling()

    except Exception as e:
        logger.critical(f"Error fatal saat menjalankan bot: {e}")
        raise

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    current_time = datetime.now()

    last_message_time = redis_client.get(f"last_message_time_{user_id}")
    if last_message_time:
        last_message_time = datetime.fromtimestamp(float(last_message_time))
        if current_time - last_message_time < timedelta(seconds=5):
            await update.message.reply_text("Anda mengirim pesan terlalu cepat. Mohon tunggu beberapa detik.")
            return

    redis_client.set(f"last_message_time_{user_id}", current_time.timestamp())

    # Handle different message types
    if update.message.text:
        await handle_text(update, context)
    elif update.message.voice:
        await handle_voice(update, context)
    elif update.message.photo:
        await handle_photo(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /help"""
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
    user_id = update.message.from_user.id
    message_type = "text"
    if update.message.voice:
        message_type = "voice"
    elif update.message.photo:
        message_type = "photo"

    # Simpan statistik ke Redis
    redis_client.hincrby(f"user:{user_id}:stats", message_type, 1)
    redis_client.hincrby(f"user:{user_id}:stats", "total_messages", 1)

async def get_user_statistics(user_id: int) -> Dict[str, int]:
    stats = redis_client.hgetall(f"user:{user_id}:stats")
    return {k: int(v) for k, v in stats.items()}

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_stats = await get_user_statistics(user_id)
    stats_message = (
        f"Statistik Pengguna:\n"
        f"- Total Pesan: {user_stats.get('total_messages', 0)}\n"
        f"- Pesan Teks: {user_stats.get('text', 0)}\n"
        f"- Pesan Suara: {user_stats.get('voice', 0)}\n"
        f"- Pesan Gambar: {user_stats.get('photo', 0)}"
    )
    await update.message.reply_text(stats_message)

if __name__ == '__main__':
    asyncio.run(main())
