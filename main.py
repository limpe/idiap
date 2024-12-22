from pydub import AudioSegment
import os
import logging
import io
import tempfile
import pydub.utils


from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import speech_recognition as sr
from pydub import AudioSegment
import gtts


# Konfigurasi logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
pydub.utils.register_audio_codec("ffmpeg", "/usr/bin/ffmpeg")

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
        logger.info("Memproses pesan suara...")
        voice_file = await update.message.voice.get_file()
        logger.info(f"File suara didapatkan: {voice_file.file_path}")
        voice_bytes = await voice_file.download_as_bytearray()
        logger.info(f"File suara diunduh ({len(voice_bytes)} bytes).")

        try:
            audio = AudioSegment.from_ogg(io.BytesIO(voice_bytes))
            wav_io = io.BytesIO()
            audio.export(wav_io, format='wav')
            wav_io.seek(0)
            logger.info("Konversi ke WAV berhasil.")
        except Exception as e:
            logger.warning(f"Gagal konversi ke WAV: {e}, mencoba memproses langsung.")
            wav_io = io.BytesIO(voice_bytes)

        r = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            try:
                audio_data = r.record(source)
                logger.info("Merekam audio untuk Speech Recognition...")
                text = r.recognize_google(audio_data, language="id-ID")
                logger.info(f"Teks hasil Speech Recognition: {text}")
            except sr.UnknownValueError:
                logger.warning("Speech Recognition: Tidak mengerti.")
                await update.message.reply_text("Maaf, saya tidak mengerti apa yang Anda katakan.")
                return
            except sr.RequestError as e:
                logger.error(f"Speech Recognition error: {e}")
                await update.message.reply_text(f"Maaf, terjadi masalah dengan layanan Speech Recognition: {e}")
                return
            except Exception as e:
                logger.exception("Error tak terduga pada Speech Recognition")
                await update.message.reply_text("Terjadi kesalahan saat mengenali ucapan.")
                return

        await update.message.reply_text(f"Anda berkata: {text}")

        logger.info("Memanggil Mistral API...")
        response = process_with_mistral(text)
        logger.info(f"Respon dari Mistral: {response}")
        await update.message.reply_text(f"Respon Mistral: {response}")

        logger.info("Membuat respon suara...")
        tts = gtts.gTTS(response, lang="id")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            try:
                tts.save(fp.name)
                logger.info(f"File MP3 disimpan di: {fp.name}")
                with open(fp.name, 'rb') as voice_file:
                    await update.message.reply_voice(voice=voice_file)
                logger.info("Respon suara dikirim.")
            except FileNotFoundError:
                logger.error(f"File MP3 tidak ditemukan: {fp.name}")
                await update.message.reply_text("Terjadi kesalahan saat mengirim respon suara.")
            except Exception as e:
                logger.exception(f"Error saat membuat/mengirim file MP3: {e}")
                await update.message.reply_text("Terjadi kesalahan saat mengirim respon suara.")
            finally:
                os.remove(fp.name)

    except Exception as e:
        logger.exception(f"Error tak terduga di handle_voice: {e}")
        await update.message.reply_text(f"Maaf, terjadi kesalahan dalam memproses pesan suara: {e}")

# Fungsi untuk memproses audio
def process_audio(input_file, output_file):
    try:
        # Membaca file audio
        sound = AudioSegment.from_mp3(input_file) # Atau format audio lainnya

        # Melakukan beberapa operasi pada audio (contoh)
        sound = sound + 3  # Menambah volume 3dB

        # Menyimpan file audio yang telah diproses
        sound.export(output_file, format="wav")

        print(f"File audio berhasil diproses dan disimpan di {output_file}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Main program (bagian yang dieksekusi saat script dijalankan)
if __name__ == "__main__":
    input_file = "audio.mp3"  # Ganti dengan nama file input Anda
    output_file = "output.wav" # Ganti dengan nama file output yang diinginkan

    # Pastikan file input ada
    if os.path.exists(input_file):
      process_audio(input_file, output_file)
    else:
      print(f"File {input_file} tidak ditemukan")

def process_with_mistral(text):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": text}]
    }
    try:
        logger.info(f"Mengirim request ke Mistral dengan teks: {text}")
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        json_response = response.json()
        logger.info(f"Respon JSON dari Mistral: {json_response}")
        return json_response['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error memanggil Mistral API: {e}")
        if 'response' in locals() and response is not None:
            logger.error(f"Response text from mistral : {response.text}")
        return f"Terjadi kesalahan saat berkomunikasi dengan Mistral API: {e}"
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Format respon Mistral tidak sesuai: {e}")
        if 'json_response' in locals(): # Indentasi yang benar
            logger.error(f"Full JSON Response : {json_response}")
        return "Terjadi kesalahan dalam memproses respon dari Mistral API."
    except Exception as e:
        logger.exception(f"Error tak terduga pada process_with_mistral: {e}")
        return "Terjadi kesalahan yang tidak terduga."


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        logger.info(f"Menerima pesan teks: {update.message.text}")
        response = process_with_mistral(update.message.text)
        logger.info(f"Respon Mistral untuk teks: {response}")

        await update.message.reply_text(response)

        logger.info("Membuat respon suara untuk teks...")
        tts = gtts.gTTS(response, lang="id")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            try:
                tts.save(fp.name)
                logger.info(f"File MP3 untuk teks disimpan di: {fp.name}")
                with open(fp.name, 'rb') as voice_file:
                    await update.message.reply_voice(voice=voice_file)
                logger.info("Respon suara untuk teks dikirim.")
            except FileNotFoundError:
                logger.error(f"File MP3 untuk teks tidak ditemukan: {fp.name}")
                await update.message.reply_text("Terjadi kesalahan saat mengirim respon suara.")
            except Exception as e:
                logger.exception(f"Error saat membuat/mengirim file MP3 untuk teks: {e}")
                await update.message.reply_text("Terjadi kesalahan saat mengirim respon suara.")
            finally:
                os.remove(fp.name)

    except Exception as e:
        logger.exception(f"Error tak terduga di handle_text: {e}")
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
