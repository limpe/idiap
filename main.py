import os
import logging
import tempfile
import asyncio
import base64
import uuid
from typing import Optional, List, Dict

from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from pydub import AudioSegment
import gtts
import aiohttp
from langdetect import detect
from groq import Groq
from PIL import Image
from io import BytesIO
from aiohttp import FormData

# Konstanta untuk batasan ukuran file
MAX_AUDIO_SIZE = 20 * 1024 * 1024  # 20MB

def check_required_settings():
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN tidak ditemukan!")
        return False
    if not MISTRAL_API_KEY:
        print("Error: MISTRAL_API_KEY tidak ditemukan!")
        return False
    return True

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

# Konstanta konfigurasi
CHUNK_DURATION = 30  # Durasi chunk dalam detik
SPEECH_RECOGNITION_TIMEOUT = 30  # Timeout untuk speech recognition dalam detik
MAX_RETRIES = 5  # Jumlah maksimal percobaan untuk API calls
RETRY_DELAY = 5  # Delay antara percobaan ulang dalam detik
MAX_CONVERSATION_MESSAGES = 10
CONVERSATION_TIMEOUT = 1000  #Durasi percakapan dalam detik 

# Dictionary untuk menyimpan histori percakapan
#user_sessions: Dict[int, List[Dict[str, str]]] = {}

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
    welcome_text = """Halo! Saya asisten Anda. Saya dapat:
    - Memproses pesan suara (termasuk yang panjang)
    - Menanggapi dengan suara
    - Membantu dengan berbagai tugas
    - Memproses gambar

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

async def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.exception("Error encoding image")
        raise

async def process_image_with_pixtral_multiple(image_path: str, repetitions: int = 2) -> List[str]:
    """Process image using Pixtral model multiple times with rate limiting."""
    try:
        base64_image = await encode_image(image_path)
        results = []
        
        async def single_request():
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "pixtral-12b-2409",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Apa isi gambar ini? Mohon jelaskan dengan detail dalam Bahasa Indonesia."
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
    """Proses file suara menjadi teks dengan penanganan error"""
    try:
        logger.info("Memulai pemrosesan pesan suara...")

        # Unduh file suara
        voice_file = await update.message.voice.get_file()

        # Gunakan `with` untuk file sementara
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=True) as temp_ogg:
            await voice_file.download_to_drive(temp_ogg.name)
            logger.info(f"File suara didownload ke {temp_ogg.name}")

            # Konversi OGG ke WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
                audio = AudioSegment.from_ogg(temp_ogg.name).set_channels(1).set_frame_rate(16000)
                audio.export(temp_wav.name, format='wav')
                logger.info("Konversi file OGG ke WAV berhasil")

                # Transkripsi suara
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_wav.name) as source:
                    text = recognizer.recognize_google(recognizer.record(source), language="id-ID")
                    logger.info("Transkripsi selesai")
                    return text

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

async def filter_text(text: str) -> str:
    """Filter untuk menghapus karakter tertentu seperti asterisks (*) dan #, serta kata 'Mistral'"""
    filtered_text = text.replace("*", "").replace("#", "").replace("Mistral AI", "PAIDI")
    return filtered_text.strip()

async def process_with_mistral(messages: List[Dict[str, str]]) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Tambahkan instruksi sistem agar respon default dalam Bahasa Indonesia
    messages.insert(0, {"role": "system", "content": "Pastikan semua respons diberikan dalam Bahasa Indonesia."})

    data = {
        "model": "pixtral-large-latest",
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
        if update.message.voice.file_size > MAX_AUDIO_SIZE:
            await update.message.reply_text("Maaf, file audio terlalu besar (maksimal 20MB)")
            return
            
        bot_statistics["total_messages"] += 1
        bot_statistics["voice_messages"] += 1

        chat_id = update.message.chat_id

        processing_msg = await update.message.reply_text("Sedang memproses pesan suara Anda...")
        text = await process_voice_to_text(update)

        if text:
            text = await filter_text(text)
            await update.message.reply_text(f"Teks hasil transkripsi suara Anda: \n{text}")

            if chat_id not in user_sessions:
                user_sessions[chat_id] = []

            user_sessions[chat_id].append({"role": "user", "content": text})
            mistral_messages = user_sessions[chat_id][-10:]
            response = await process_with_mistral(mistral_messages)

            if response:
                user_sessions[chat_id].append({"role": "assistant", "content": response})
                response = await filter_text(response)
                await update.message.reply_text(response)
                await send_voice_response(update, response)

        await processing_msg.delete()

    except AudioProcessingError as e:
        bot_statistics["errors"] += 1
        await update.message.reply_text(f"Maaf, terjadi kesalahan: {str(e)}")
    except Exception as e:
        bot_statistics["errors"] += 1
        logger.exception("Error dalam handle_voice")
        await update.message.reply_text("Maaf, terjadi kesalahan.")
        
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk memproses foto menggunakan Pixtral dengan multiple analisis."""
    chat_type = update.message.chat.type
    
    # Periksa apakah bot perlu merespons
    should_respond = False
    if chat_type == "private":
        should_respond = True
    elif chat_type in ["group", "supergroup"]:
        # Cek mention dalam caption atau reply ke bot
        if (update.message.caption and f'@{context.bot.username}' in update.message.caption) or \
           (update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id):
            should_respond = True
    
    if not should_respond:
        logger.info("Gambar di grup tanpa mention diabaikan.")
        return

    try:
        bot_statistics["total_messages"] += 1
        bot_statistics["photo_messages"] += 1

        # Kirim pesan sedang memproses
        processing_msg = await update.message.reply_text("Sedang menganalisa gambar...")

        # Download dan proses gambar
        photo_file = await update.message.photo[-1].get_file()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            await photo_file.download_to_drive(temp_file.name)
            
            # Proses dengan Pixtral (2 analisis)
            try:
                results = await process_image_with_pixtral_multiple(temp_file.name)
                
                # Hapus pesan processing
                await processing_msg.delete()
                
                # Kirim hasil analisis
                if results and any(results):
                    await update.message.reply_text("Hasil Analisa Gambar:")
                    for i, result in enumerate(results, 1):
                        if result and result.strip():
                            await update.message.reply_text(f"Analisis {i}:\n{result}")
                else:
                    await update.message.reply_text("Maaf, tidak dapat menganalisa gambar. Silakan coba lagi.")
            
            except Exception as e:
                logger.error(f"Error dalam proses analisis gambar: {str(e)}")
                await update.message.reply_text("Terjadi kesalahan saat menganalisa gambar. Silakan coba lagi.")
            
            finally:
                # Cleanup
                if os.path.exists(temp_file.name):
                    os.remove(temp_file.name)

    except Exception as e:
        bot_statistics["errors"] += 1
        logger.exception("Error dalam handle_photo")
        await update.message.reply_text("Maaf, terjadi kesalahan saat memproses gambar.")
        if processing_msg:
            await processing_msg.delete()
        
async def cleanup_sessions(context: ContextTypes.DEFAULT_TYPE):
    """Bersihkan sesi yang sudah tidak aktif"""
    current_time = asyncio.get_event_loop().time()
    
    for chat_id in list(user_sessions.keys()):
        if current_time - user_sessions[chat_id]['last_update'] > CONVERSATION_TIMEOUT:
            del user_sessions[chat_id]

async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan yang di-mention atau reply di grup"""
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
            await handle_text(update, context, message_text)
        else:
            logger.info("Pesan di grup tanpa mention yang valid diabaikan.")

# Struktur data baru untuk user_sessions
user_sessions: Dict[int, Dict] = {}

async def initialize_session(chat_id: int) -> None:
    """Inisialisasi sesi baru untuk chat"""
    # Batasi jumlah sesi aktif
    if len(user_sessions) >= MAX_CONCURRENT_SESSIONS:
        # Hapus sesi pengguna paling lama berdasarkan waktu 'last_update'
        oldest_session = min(user_sessions.items(), key=lambda x: x[1]['last_update'])[0]
        del user_sessions[oldest_session]
        logger.info(f"Sesi pengguna {oldest_session} dihapus untuk mengosongkan ruang.")

    # Inisialisasi sesi baru
    user_sessions[chat_id] = {
        'messages': [],
        'last_update': asyncio.get_event_loop().time(),
        'conversation_id': str(uuid.uuid4())
    }


async def should_reset_context(chat_id: int, message: str) -> bool:
    """Tentukan apakah konteks perlu direset"""
    if chat_id not in user_sessions:
        return True
        
    current_time = asyncio.get_event_loop().time()
    time_diff = current_time - user_sessions[chat_id]['last_update']
    
    keywords = ['halo', 'hai', 'hi', 'hello', 'permisi', '?']
    starts_with_keyword = any(message.lower().startswith(keyword) for keyword in keywords)
    
    return time_diff > CONVERSATION_TIMEOUT or starts_with_keyword

async def update_session(chat_id: int, message: Dict[str, str]) -> None:
    """Update sesi chat dengan pesan baru"""
    if chat_id not in user_sessions:
        await initialize_session(chat_id)
    
    session = user_sessions[chat_id]
    session['messages'].append(message)
    session['last_update'] = asyncio.get_event_loop().time()
    
    if len(session['messages']) > MAX_CONVERSATION_MESSAGES:
        session['messages'] = session['messages'][-MAX_CONVERSATION_MESSAGES:]

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE, message: Optional[str] = None):
    """Handler untuk pesan teks dengan manajemen konteks yang lebih baik"""
    chat_type = update.message.chat.type
    chat_id = update.message.chat_id

    # Tentukan pesan yang akan diproses
    if chat_type in ["group", "supergroup"]:
        if context.bot.username in update.message.text:
            message = update.message.text.replace(f'@{context.bot.username}', '').strip()
        else:
            logger.info("Pesan di grup tanpa mention diabaikan.")
            return
    else:
        if not message:
            message = update.message.text.strip()

    # Update statistik
    bot_statistics["total_messages"] += 1
    bot_statistics["text_messages"] += 1

    # Cek apakah perlu reset konteks
    if await should_reset_context(chat_id, message):
        await initialize_session(chat_id)
        
    # Filter dan proses pesan
    message = await filter_text(message)
    await update_session(chat_id, {"role": "user", "content": message})
    
    # Ambil pesan-pesan dalam sesi untuk diproses
    session_messages = user_sessions[chat_id]['messages']
    response = await process_with_mistral(session_messages)

    if response:
        await update_session(chat_id, {"role": "assistant", "content": response})
        response = await filter_text(response)
        await update.message.reply_text(response)

def main():
    if not check_required_settings():
        print("Bot tidak bisa dijalankan karena konfigurasi tidak lengkap")
        return

    try:
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stats", stats))

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

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats_message = (
    f"Statistik Bot:\n"
    f"- Total Pesan: {bot_statistics['total_messages']}\n"
    f"- Pesan Suara: {bot_statistics['voice_messages']}\n"
    f"- Pesan Teks: {bot_statistics['text_messages']}\n"
    f"- Kesalahan: {bot_statistics['errors']}\n"
    f"- Sesi Aktif: {len(user_sessions)}"
)
    await update.message.reply_text(stats_message)

if __name__ == '__main__':
    asyncio.run(main())
