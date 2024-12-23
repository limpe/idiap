import os
import logging
import tempfile
import asyncio
from typing import Optional, List, Dict

from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from pydub import AudioSegment
import gtts
import aiohttp
from langdetect import detect
from PIL import Image
from io import BytesIO

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
                        return await filter_text(json_response['choices'][0]['message']['content'])

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
    try:
        bot_statistics["total_messages"] += 1
        bot_statistics["text_messages"] += 1

        chat_id = update.message.chat_id
        text = await filter_text(update.message.text)

        # Deteksi bahasa teks
        detected_language = detect(text)
        await update.message.reply_text(f"Bahasa yang terdeteksi: {detected_language}")

        if chat_id not in user_sessions:
            user_sessions[chat_id] = []

        user_sessions[chat_id].append({"role": "user", "content": text})
        mistral_messages = user_sessions[chat_id][-10:]
        response = await process_with_mistral(mistral_messages)

        if response:
            user_sessions[chat_id].append({"role": "assistant", "content": response})
            response = await filter_text(response)
            await update.message.reply_text(response)

    except Exception as e:
        bot_statistics["errors"] += 1
        logger.exception("Error dalam handle_text")
        await update.message.reply_text("Maaf, terjadi kesalahan.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
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
    try:
        bot_statistics["total_messages"] += 1
        bot_statistics["photo_messages"] += 1

        chat_id = update.message.chat_id
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()

        # Mengirim gambar ke Mistral API
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
        }

        form = aiohttp.FormData()
        form.add_field('file', image_bytes, filename='image.png', content_type='image/png')

        for attempt in range(MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        "https://api.mistral.ai/v1/vision/analyze",
                        headers=headers,
                        data=form
                    ) as response:
                        response.raise_for_status()
                        json_response = await response.json()

                        if 'description' in json_response:
                            image_description = json_response['description']
                            break

            except aiohttp.ClientError as e:
                logger.error(f"Percobaan {attempt + 1} gagal karena error HTTP: {str(e)}")
            except asyncio.TimeoutError:
                logger.error(f"Percobaan {attempt + 1} gagal karena timeout")
            except Exception as e:
                logger.error(f"Percobaan {attempt + 1} gagal: {str(e)}")

            if attempt < MAX_RETRIES - 1:
                logger.info(f"Menunggu {RETRY_DELAY} detik sebelum percobaan berikutnya...")
                await asyncio.sleep(RETRY_DELAY)

        if chat_id not in user_sessions:
            user_sessions[chat_id] = []

        user_sessions[chat_id].append({"role": "user", "content": image_description})
        mistral_messages = user_sessions[chat_id][-10:]
        response = await process_with_mistral(mistral_messages)

        if response:
            user_sessions[chat_id].append({"role": "assistant", "content": response})
            response = await filter_text(response)
            await update.message.reply_text(response)

    except Exception as e:
        bot_statistics["errors"] += 1
        logger.exception("Error dalam handle_photo")
        await update.message.reply_text("Maaf, terjadi kesalahan.")
        image_description = "Tidak dapat memproses gambar."

        if chat_id not in user_sessions:
            user_sessions[chat_id] = []

        user_sessions[chat_id].append({"role": "user", "content": image_description})
        mistral_messages = user_sessions[chat_id][-10:]
        response = await process_with_mistral(mistral_messages)

        if response:
            user_sessions[chat_id].append({"role": "assistant", "content": response})
            response = await filter_text(response)
            await update.message.reply_text(response)

async def cleanup_sessions(context: ContextTypes.DEFAULT_TYPE):
    """Bersihkan sesi lama untuk menghemat memori"""
    for chat_id in list(user_sessions.keys()):
        if len(user_sessions[chat_id]) > 300:
            user_sessions[chat_id] = user_sessions[chat_id][-100:]

def main():
    try:
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

        # Statistik command
        async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
            stats_message = (
                f"Statistik Bot:\n"
                f"- Total Pesan: {bot_statistics['total_messages']}\n"
                f"- Pesan Suara: {bot_statistics['voice_messages']}\n"
                f"- Pesan Teks: {bot_statistics['text_messages']}\n"
                f"- Pesan Gambar: {bot_statistics['photo_messages']}\n"
                f"- Kesalahan: {bot_statistics['errors']}"
            )
            await update.message.reply_text(stats_message)

        application.add_handler(CommandHandler("stats", stats))

        # Memastikan JobQueue diaktifkan
        application.job_queue.run_repeating(cleanup_sessions, interval=3600, first=10)

        application.run_polling()

    except Exception as e:
        logger.critical(f"Error fatal saat menjalankan bot: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main())
