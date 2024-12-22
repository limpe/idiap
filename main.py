import os
import logging
import tempfile
import asyncio
from typing import Optional, List

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import speech_recognition as sr
from pydub import AudioSegment
import gtts
import aiohttp

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
MAX_RETRIES = 3  # Jumlah maksimal percobaan untuk API calls

class AudioProcessingError(Exception):
    """Custom exception untuk error pemrosesan audio"""
    pass

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    welcome_text = """Halo! Saya asisten Anda. Saya dapat:
    - Memproses pesan suara (termasuk yang panjang)
    - Menanggapi dengan suara
    - Membantu dengan berbagai tugas

    Kirim saya pesan atau catatan suara untuk memulai!"""
    await update.message.reply_text(welcome_text)

async def process_voice_to_text(update: Update) -> Optional[str]:
    """
    Memproses file suara menjadi teks dengan penanganan file yang lebih baik
    dan pemrosesan chunk untuk file panjang.
    """
    temp_files = []  # Track temporary files for cleanup

    try:
        logger.info("Memulai pemrosesan pesan suara...")
        voice_file = await update.message.voice.get_file()
        
        # Buat temporary file untuk audio OGG
        temp_ogg = tempfile.NamedTemporaryFile(suffix='.ogg', delete=False)
        temp_files.append(temp_ogg.name)
        
        # Download file suara
        await voice_file.download_to_drive(temp_ogg.name)
        logger.info(f"File suara didownload ke {temp_ogg.name}")

        # Buat temporary file untuk WAV
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_wav.name)

        # Konversi OGG ke WAV dengan parameter optimal
        audio = AudioSegment.from_ogg(temp_ogg.name)
        audio = audio.set_channels(1).set_frame_rate(16000)  # Optimize untuk speech recognition
        audio.export(temp_wav.name, format='wav')
        logger.info("Konversi ke WAV selesai")

        # Inisialisasi speech recognizer
        recognizer = sr.Recognizer()
        recognizer.operation_timeout = SPEECH_RECOGNITION_TIMEOUT

        text_chunks: List[str] = []
        
        # Hitung durasi menggunakan pydub
        duration_seconds = len(audio) / 1000.0  # Convert milliseconds to seconds
        total_chunks = int(duration_seconds / CHUNK_DURATION) + 1
        
        logger.info(f"Durasi audio: {duration_seconds:.2f} detik, akan diproses dalam {total_chunks} chunk")
        
        with sr.AudioFile(temp_wav.name) as source:
            for i in range(total_chunks):
                offset = i * CHUNK_DURATION
                chunk_duration = min(CHUNK_DURATION, duration_seconds - offset)
                
                if chunk_duration <= 0:
                    break
                
                # Record dari posisi yang tepat
                audio_chunk = recognizer.record(source, duration=chunk_duration)
                
                # Coba recognize dengan beberapa percobaan
                for attempt in range(MAX_RETRIES):
                    try:
                        chunk_text = recognizer.recognize_google(
                            audio_chunk,
                            language="id-ID"
                        )
                        text_chunks.append(chunk_text)
                        logger.info(f"Chunk {i+1}/{total_chunks} berhasil diproses")
                        break
                    except sr.UnknownValueError:
                        logger.warning(f"Chunk {i+1}: Suara tidak terdeteksi")
                        continue
                    except sr.RequestError as e:
                        if attempt == MAX_RETRIES - 1:
                            raise AudioProcessingError(f"Error pada speech recognition: {e}")
                        logger.warning(f"Attempt {attempt+1} failed, retrying...")
                        await asyncio.sleep(1)  # Brief delay before retry

        if not text_chunks:
            raise AudioProcessingError("Tidak ada teks yang berhasil dikenali")

        final_text = " ".join(text_chunks)
        logger.info("Pemrosesan suara selesai")
        return final_text

    except Exception as e:
        logger.exception("Error dalam pemrosesan audio")
        raise AudioProcessingError(f"Gagal memproses audio: {str(e)}")

    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.debug(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

async def process_with_mistral(text: str) -> Optional[str]:
    """
    Processes text using the Mistral API with improved error handling and request formatting.
    
    Args:
        text: The input text to be processed
        
    Returns:
        Optional[str]: The processed response or an error message
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Ensure the text is not empty and is properly formatted
    if not text or not text.strip():
        return "Maaf, tidak ada teks yang dapat diproses."
    
    # Prepare the request data with the correct model name
    data = {
        "model": "mistral-large-latest",  # Changed from pixtral-large-latest
        "messages": [
            {
                "role": "user",
                "content": text.strip()
            }
        ],
        "max_tokens": 1000  # Add reasonable limit
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
                    # Log the response status for debugging
                    logger.info(f"Mistral API response status: {response.status}")
                    
                    if response.status == 400:
                        error_body = await response.text()
                        logger.error(f"Mistral API 400 error: {error_body}")
                        return "Maaf, terjadi kesalahan dalam memformat permintaan ke AI."
                    
                    response.raise_for_status()
                    json_response = await response.json()
                    
                    # Verify the response structure
                    if 'choices' in json_response and json_response['choices']:
                        return json_response['choices'][0]['message']['content']
                    else:
                        logger.error(f"Unexpected response structure: {json_response}")
                        return "Maaf, respons dari AI tidak sesuai format yang diharapkan."

        except aiohttp.ClientTimeout as e:
            logger.warning(f"Timeout pada attempt {attempt + 1}: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                return "Maaf, server sedang sibuk. Mohon coba lagi nanti."
            await asyncio.sleep(1)
            
        except aiohttp.ClientError as e:
            logger.error(f"Error koneksi API: {str(e)}")
            return "Maaf, terjadi masalah koneksi dengan server AI."
            
        except Exception as e:
            logger.exception("Error tak terduga dalam process_with_mistral")
            return "Maaf, terjadi kesalahan yang tidak terduga saat memproses permintaan Anda."

    return "Maaf, server tidak merespons setelah beberapa percobaan. Mohon coba lagi nanti."

async def send_voice_response(update: Update, text: str):
    """
    Mengirim respons suara dengan penanganan error yang lebih baik
    dan pembersihan file temporary yang konsisten.
    """
    temp_file = None
    try:
        logger.info("Membuat respons suara...")
        tts = gtts.gTTS(text, lang="id")
        
        # Gunakan tempfile untuk manajemen file yang lebih aman
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tts.save(temp_file.name)
        
        # Kirim respons suara
        with open(temp_file.name, 'rb') as voice_file:
            await update.message.reply_voice(voice=voice_file)
        logger.info("Respons suara berhasil dikirim")

    except Exception as e:
        logger.exception("Error dalam pembuatan/pengiriman respons suara")
        await update.message.reply_text(
            "Maaf, terjadi kesalahan saat membuat respons suara. "
            "Berikut respons dalam bentuk teks: " + text
        )

    finally:
        # Bersihkan file temporary
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception as e:
                logger.warning(f"Gagal menghapus file temporary: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan suara dengan penanganan error yang lebih baik"""
    try:
        # Beri tahu user bahwa pesannya sedang diproses
        processing_msg = await update.message.reply_text(
            "Sedang memproses pesan suara Anda..."
        )

        # Proses voice message
        text = await process_voice_to_text(update)
        if text:
            await update.message.reply_text(f"Anda berkata: {text}")
            
            # Proses dengan Mistral
            response = await process_with_mistral(text)
            if response:
                await update.message.reply_text(response)
                await send_voice_response(update, response)
        
        # Hapus pesan "sedang memproses"
        await processing_msg.delete()

    except AudioProcessingError as e:
        await update.message.reply_text(f"Maaf, terjadi kesalahan: {str(e)}")
    except Exception as e:
        logger.exception("Error tak terduga dalam handle_voice")
        await update.message.reply_text(
            "Maaf, terjadi kesalahan yang tidak terduga. "
            "Mohon coba lagi nanti."
        )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan teks"""
    try:
        logger.info(f"Menerima pesan teks: {update.message.text}")
        
        # Beri tahu user bahwa pesannya sedang diproses
        processing_msg = await update.message.reply_text(
            "Sedang memproses pesan Anda..."
        )

        # Proses dengan Mistral
        response = await process_with_mistral(update.message.text)
        if response:
            await update.message.reply_text(response)
            await send_voice_response(update, response)

        # Hapus pesan "sedang memproses"
        await processing_msg.delete()

    except Exception as e:
        logger.exception("Error dalam handle_text")
        await update.message.reply_text(
            "Maaf, terjadi kesalahan dalam memproses pesan Anda. "
            "Mohon coba lagi nanti."
        )

def main():
    """Fungsi utama untuk menjalankan bot"""
    try:
        # Buat aplikasi dengan error handling yang lebih baik
        application = Application.builder().token(TELEGRAM_TOKEN).build()

        # Tambahkan handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice))
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_text
        ))

        # Mulai bot
        logger.info("Bot mulai berjalan...")
        application.run_polling()

    except Exception as e:
        logger.critical(f"Error fatal saat menjalankan bot: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main())
