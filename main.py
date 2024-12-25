import os
import logging
import tempfile
import asyncio
import base64
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

# Dictionary untuk menyimpan histori percakapan
user_sessions: Dict[int, List[Dict[str, str]]] = {}

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
    
def encode_image(image_path):
    """Encode an image file to a base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

async def process_image_with_groq(image_path: str) -> str:
    try:
        base64_image = encode_image(image_path)
        client = Groq()

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Apa isi gambar ini? Mohon jawab dalam Bahasa Indonesia."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.exception("Error in processing image with Groq")
        return "Terjadi kesalahan saat memproses gambar."
        
async def process_image_with_groq_multiple(image_path: str, repetitions: int = 5) -> List[str]:
    """Proses gambar ke Groq API beberapa kali secara paralel."""
    try:
        base64_image = encode_image(image_path)
        client = Groq()

        async def single_request():
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Apa isi gambar ini? Mohon jawab dalam Bahasa Indonesia."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    model="llama-3.2-90b-vision-preview",
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                logger.exception("Error in single request to Groq")
                return "Error processing this request."

        # Jalankan `repetitions` permintaan secara paralel
        results = await asyncio.gather(*[single_request() for _ in range(repetitions)])
        return results

    except Exception as e:
        logger.exception("Error in processing image with Groq multiple")
        return ["Terjadi kesalahan saat memproses gambar."] * repetitions

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for processing image uploads"""
    try:
        if not update.message.photo:
            await update.message.reply_text("Please send a valid image.")
            return

        bot_statistics["total_messages"] += 1

        # Download image to temporary file
        photo_file = await update.message.photo[-1].get_file()
        temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        await photo_file.download_to_drive(temp_image_path)

        # Process image with Groq
        result = await process_image_with_groq(temp_image_path)
        await update.message.reply_text(f"Hasil Analisa Gambar: {result}")

        # Cleanup temporary file
        os.remove(temp_image_path)
    except Exception as e:
        bot_statistics["errors"] += 1
        logger.exception("Error in handle_image")
        await update.message.reply_text("An error occurred while processing the image.")

async def process_voice_to_text(update: Update) -> Optional[str]:
    """Proses file suara menjadi teks dengan penanganan error"""
    temp_files = []  # Track temporary files for cleanup

    try:
        logger.info("Memulai pemrosesan pesan suara...")
        voice_file = await update.message.voice.get_file()

        # Buat file sementara untuk audio OGG
        temp_ogg = tempfile.NamedTemporaryFile(suffix='.ogg', delete=False)
        temp_files.append(temp_ogg.name)
        await voice_file.download_to_drive(temp_ogg.name)
        logger.info(f"File suara didownload ke {temp_ogg.name}")

        # Buat file sementara untuk WAV
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_wav.name)

        audio = AudioSegment.from_ogg(temp_ogg.name).set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav.name, format='wav')
        logger.info("Konversi file OGG ke WAV berhasil")

        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav.name) as source:
            text_chunks = []
            duration_seconds = len(audio) / 1000.0
            total_chunks = int(duration_seconds / CHUNK_DURATION) + 1

            for i in range(total_chunks):
                offset = i * CHUNK_DURATION
                chunk_duration = min(CHUNK_DURATION, duration_seconds - offset)

                if chunk_duration <= 0:
                    break

                audio_chunk = recognizer.record(source, duration=chunk_duration)
                for attempt in range(MAX_RETRIES):
                    try:
                        chunk_text = recognizer.recognize_google(audio_chunk, language="id-ID")
                        text_chunks.append(chunk_text)
                        break
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        if attempt == MAX_RETRIES - 1:
                            raise AudioProcessingError(f"Error pada speech recognition: {e}")

            if not text_chunks:
                raise AudioProcessingError("Tidak ada teks yang berhasil dikenali")

            final_text = " ".join(text_chunks)
            logger.info("Pemrosesan suara selesai")
            return final_text

    except Exception as e:
        logger.exception("Error dalam pemrosesan audio")
        raise AudioProcessingError(f"Gagal memproses audio: {str(e)}")

    finally:
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Gagal menghapus file sementara: {temp_file}")

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

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan teks"""
    chat_type = update.message.chat.type  # Periksa tipe chat (grup atau pribadi)

    if chat_type in ["group", "supergroup"]:  # Jika chat di grup
        if context.bot.username in update.message.text:  # Periksa mention
            # Ambil teks tanpa mention
            text = update.message.text.replace(f'@{context.bot.username}', '').strip()
        else:
            logger.info("Pesan di grup tanpa mention diabaikan.")
            return  # Abaikan pesan tanpa mention
    else:  # Jika chat pribadi
        text = update.message.text.strip()  # Ambil seluruh teks

    bot_statistics["total_messages"] += 1
    bot_statistics["text_messages"] += 1

    chat_id = update.message.chat_id
    text = await filter_text(text)

    if chat_id not in user_sessions:
        user_sessions[chat_id] = []

    user_sessions[chat_id].append({"role": "user", "content": text})
    mistral_messages = user_sessions[chat_id][-10:]
    response = await process_with_mistral(mistral_messages)

    if response:
        user_sessions[chat_id].append({"role": "assistant", "content": response})
        response = await filter_text(response)
        await update.message.reply_text(response)
        

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
    """Handler untuk memproses gambar dengan beberapa analisis paralel."""
    chat_type = update.message.chat.type  # Periksa tipe chat (grup atau pribadi)

    # Periksa mention di caption atau reply
    mention_found = False
    caption = ""
    
    if chat_type in ["group", "supergroup"]:
        if update.message.caption and context.bot.username in update.message.caption:
            mention_found = True
            caption = update.message.caption.replace(f'@{context.bot.username}', '').strip()
        elif update.message.reply_to_message and update.message.reply_to_message.caption and context.bot.username in update.message.reply_to_message.caption:
            mention_found = True
            caption = update.message.reply_to_message.caption.replace(f'@{context.bot.username}', '').strip()
    else:  # Chat pribadi
        caption = update.message.caption or ""

    if not mention_found and chat_type in ["group", "supergroup"]:
        logger.info("Gambar di grup tanpa mention diabaikan.")
        return

    try:
        bot_statistics["total_messages"] += 1
        bot_statistics["photo_messages"] += 1

        # Unduh file gambar
        photo_file = await update.message.photo[-1].get_file()
        temp_image_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
        await photo_file.download_to_drive(temp_image_path)

        # Proses gambar dengan analisis paralel (5 kali)
        results = await process_image_with_groq_multiple(temp_image_path, repetitions=5)

        # Kirim setiap hasil analisis sebagai pesan terpisah
        for i, result in enumerate(results):
            await update.message.reply_text(f"Analisis {i+1}:\n{result}")

        # Bersihkan file sementara
        os.remove(temp_image_path)

    except Exception as e:
        bot_statistics["errors"] += 1
        logger.exception("Error dalam handle_photo")
        await update.message.reply_text("Maaf, terjadi kesalahan saat memproses gambar.")

async def cleanup_sessions(context: ContextTypes.DEFAULT_TYPE):
    """Bersihkan sesi lama untuk menghemat memori"""
    for chat_id in list(user_sessions.keys()):
        if len(user_sessions[chat_id]) > 300:
            user_sessions[chat_id] = user_sessions[chat_id][-100:]

async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk memproses mention di grup."""
    if context.bot.username in update.message.text:
        message = update.message.text.replace(f'@{context.bot.username}', '').strip()
        if message:
            # Memanggil handler teks dengan pesan yang di-mention
            await handle_text(update, context)

def main():
    if not check_required_settings():
        print("Bot tidak bisa dijalankan karena konfigurasi tidak lengkap")
        return

    try:
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stats", stats))

        # Message handlers
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

        # Mention handler untuk teks
        application.add_handler(MessageHandler(filters.TEXT & filters.Entity("mention"), handle_mention))

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
        f"- Kesalahan: {bot_statistics['errors']}"
    )
    await update.message.reply_text(stats_message)

if __name__ == '__main__':
    asyncio.run(main())
