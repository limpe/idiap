import os
import logging
import tempfile
import asyncio
from typing import Optional, List, Dict

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

# Dictionary untuk menyimpan histori percakapan
user_sessions: Dict[int, List[Dict[str, str]]] = {}

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
    """Proses file suara menjadi teks dengan penanganan error"""
    temp_files = []  # Daftar file sementara untuk dibersihkan

    try:
        logger.info("Memulai pemrosesan pesan suara...")
        voice_file = await update.message.voice.get_file()

        # Buat file sementara untuk audio OGG
        temp_ogg = tempfile.NamedTemporaryFile(suffix='.ogg', delete=False)
        temp_files.append(temp_ogg.name)

        # Download file suara
        await voice_file.download_to_drive(temp_ogg.name)
        logger.info(f"File suara didownload ke {temp_ogg.name}")

        # Buat file sementara untuk WAV
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_files.append(temp_wav.name)

        # Konversi OGG ke WAV
        audio = AudioSegment.from_ogg(temp_ogg.name)
        audio.export(temp_wav.name, format='wav')
        logger.info("Konversi file OGG ke WAV berhasil")

        # Proses WAV dengan SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav.name) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID")
            logger.info(f"Text dari suara: {text}")
            return text

    except Exception as e:
        logger.exception("Error dalam pemrosesan suara")
        raise AudioProcessingError(f"Gagal memproses audio: {str(e)}")

    finally:
        # Hapus file sementara
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Gagal menghapus file sementara: {temp_file}")

async def process_with_mistral(messages: List[Dict[str, str]]) -> Optional[str]:
    """
    Processes text using the Mistral API with context support.
    
    Args:
        messages: A list of message dictionaries containing chat history
        
    Returns:
        Optional[str]: The processed response or an error message
    """
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

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
                    if response.status == 400:
                        error_body = await response.text()
                        logger.error(f"Mistral API 400 error: {error_body}")
                        return "Maaf, terjadi kesalahan dalam memformat permintaan ke AI."
                    
                    response.raise_for_status()
                    json_response = await response.json()

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

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan teks dengan konteks percakapan"""
    try:
        chat_id = update.message.chat_id
        text = update.message.text

        # Tambahkan pesan ke histori percakapan pengguna
        if chat_id not in user_sessions:
            user_sessions[chat_id] = []
        
        user_sessions[chat_id].append({"role": "user", "content": text})

        # Gunakan maksimal 10 pesan terakhir untuk menghemat sumber daya
        mistral_messages = user_sessions[chat_id][-10:]
        response = await process_with_mistral(mistral_messages)

        if response:
            user_sessions[chat_id].append({"role": "assistant", "content": response})
            await update.message.reply_text(response)

    except Exception as e:
        logger.exception("Error dalam handle_text")
        await update.message.reply_text(
            "Maaf, terjadi kesalahan dalam memproses pesan Anda. "
            "Mohon coba lagi nanti."
        )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan suara dengan konteks percakapan"""
    try:
        chat_id = update.message.chat_id

        # Beri tahu pengguna bahwa pesannya sedang diproses
        processing_msg = await update.message.reply_text(
            "Sedang memproses pesan suara Anda..."
        )

        # Proses voice message menjadi teks
        text = await process_voice_to_text(update)
        if text:
            # Tambahkan teks yang dihasilkan ke histori percakapan pengguna
            if chat_id not in user_sessions:
                user_sessions[chat_id] = []
            
            user_sessions[chat_id].append({"role": "user", "content": text})

            # Gunakan maksimal 10 pesan terakhir untuk menghemat sumber daya
            mistral_messages = user_sessions[chat_id][-10:]
            response = await process_with_mistral(mistral_messages)

            if response:
                # Tambahkan respons AI ke histori percakapan
                user_sessions[chat_id].append({"role": "assistant", "content": response})

                # Kirim respons ke pengguna dalam bentuk teks dan suara
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

def main():
    """Fungsi utama untuk menjalankan bot"""
    try:
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
