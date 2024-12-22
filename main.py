import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import speech_recognition as sr
from pydub import AudioSegment
import io
import gtts
import tempfile

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
        # Get voice message
        voice_file = await update.message.voice.get_file()
        
        # Download voice file
        voice_bytes = await voice_file.download_as_bytearray()
        
        # Convert to wav using pydub
        audio = AudioSegment.from_ogg(io.BytesIO(voice_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        wav_io.seek(0)
        
        # Convert to text using speech_recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
        # Send text version
        await update.message.reply_text(f"You said: {text}")
        
        # Process with Mistral
        response = process_with_mistral(text)
        
        # Convert response to voice
        tts = gtts.gTTS(response)
        
        # Save temporarily and send
        with tempfile.NamedTemporaryFile(suffix='.mp3') as fp:
            tts.save(fp.name)
            await update.message.reply_voice(voice=open(fp.name, 'rb'))
            
    except Exception as e:
        await update.message.reply_text(f"Maaf, saya tidak bisa memproses pesan suara itu: {str(e)}")

def process_with_mistral(text):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "pixtral-large-latest",
        "messages": [{"role": "user", "content": text}]
    }
    
    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    return response.json()['choices'][0]['message']['content']

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        response = process_with_mistral(update.message.text)
        
        # Send text response
        await update.message.reply_text(response)
        
        # Convert to voice and send
        tts = gtts.gTTS(response)
        with tempfile.NamedTemporaryFile(suffix='.mp3') as fp:
            tts.save(fp.name)
            await update.message.reply_voice(voice=open(fp.name, 'rb'))
            
    except Exception as e:
        await update.message.reply_text(f"Sorry, an error occurred: {str(e)}")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("Lagi proses...")
    application.run_polling()

if __name__ == '__main__':
    main()
