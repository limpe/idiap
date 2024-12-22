import telegram
import requests # Untuk berinteraksi dengan Mistral.ai API

# Ganti dengan token dan API key Anda
BOT_TOKEN = "7644424168:AAEiozQG2CXHI4cFV1sTh2kidhbxobCe3sk"
MISTRAL_API_KEY = "ocFb4UFEr0OLSJdj7VBShdFFuH5Wjlf9"

bot = telegram.Bot(token=BOT_TOKEN)

def handle_message(update, context):
    user_message = update.message.text

    # Kirim pesan ke Mistral.ai API
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": user_message
    }
    response = requests.post("MISTRAL_API_ENDPOINT", headers=headers, json=data) # Ganti dengan endpoint Mistral.ai yang sesuai

    ai_response = response.json().get("response", "Maaf, ada kesalahan.") # Ekstrak respon dari Mistral.ai
    update.message.reply_text(ai_response)

def main():
    updater = telegram.Updater(BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(telegram.MessageHandler(telegram.Filters.text & ~telegram.Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
