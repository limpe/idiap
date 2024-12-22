import os
import logging
import io
import tempfile

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import speech_recognition as sr
from pydub import AudioSegment
import gtts

# Konfigurasi logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """Halo! Saya asisten Anda. Saya dapat:
    - Memproses pesan suara
    - Menanggapi dengan suara
    - Membantu dengan berbagai tugas

    Kirim saya pesan atau catatan suara untuk memulai!"""
    await update.message.reply_text(welcome_text)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
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
                audio = r.record(source)
                text = r.recognize_google(audio, language="id-ID")
            except sr.UnknownValueError:
                await update.message.reply_text("Maaf, saya tidak mengerti apa yang Anda katakan.")
                return
            except sr.RequestError as e:
                await update.message.reply_text(f"Maaf, terjadi masalah dengan layanan Speech Recognition: {e}")
                return
            except Exception as e: # Tangkap exception umum untuk speech recognition
                logger.error(f"Error pada speech recognition: {e}")
                await update.message.reply_text("Terjadi kesalahan saat mengenali ucapan.")
                return

        await update.message.reply_text(f"Anda berkata: {text}")

        response = process_with_mistral(text)
        await update.message.reply_text(f"Respon Mistral: {response}")

        tts = gtts.gTTS(response, lang="id")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            try:
                tts.save(fp.name)
                await update.message.reply_voice(voice=open(fp.name, 'rb'))
            except Exception as e:
                logger.error(f"Gagal membuat respon suara: {e}")
                await update.message.reply_text("Terjadi kesalahan saat membuat respon suara.")
            finally:
                os.remove(fp.name)

    except Exception as e:
        logger.error(f"Error dalam handle_voice: {e}")
        await update.message.reply_text(f"Maaf, terjadi kesalahan dalam memproses pesan suara: {e}")

def process_with_mistral(text):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "pixtral-large-latest", # atau model lain yang Anda inginkan
        "messages": [{"role": "user", "content": text}]
    }
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10 # Tambahkan timeout untuk mencegah hang
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error memanggil Mistral API: {e}")
        return f"Terjadi kesalahan saat berkomunikasi dengan Mistral API: {e}"
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Format respon Mistral tidak sesuai: {e} - {response.text if 'response' in locals() else 'Tidak ada respon'}") #pengecekan response
        return "Terjadi kesalahan dalam memproses respon dari Mistral API."

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = process_with_mistral(update.message.text)
        await update.message.reply_text(response)

        tts = gtts.gTTS(response, lang="id")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            try:
                tts.save(fp.name)
                await update.message.reply_voice(voice=open(fp.name, 'rb'))
            except Exception as e:
                logger.error(f"Gagal membuat respon suara: {e}")
                await update.message.reply_text("Terjadi kesalahan saat membuat respon suara.")
            finally:
                os.remove(fp.name)
    except Exception as e:
        logger.error(f"Error dalam handle_text: {e}")
        await update.message.reply_text(f"Maaf, terjadi kesalahan: {e}")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("JARVIS sedang berjalan...")
    application.run_polling()

if __name__ == '__main__':
    main()
