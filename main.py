import telegram
import speech_recognition as sr
import requests
import gTTS
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN") # Ambil dari variabel lingkungan
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY") # Ambil dari variabel lingkungan
MISTRAL_API_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

bot = telegram.Bot(token=BOT_TOKEN)

def handle_message(update, context):
    # ... (Kode untuk menangani pesan teks dan suara, sama seperti sebelumnya) ...
def main():
    # ... (Kode untuk menjalankan bot, sama seperti sebelumnya) ...

if __name__ == '__main__':
    main()