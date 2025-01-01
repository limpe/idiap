async def handle_mention(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan yang di-mention atau reply di grup."""
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
            # Jika perintah pembuatan gambar
            if message_text.lower().startswith(('/gambar', '/image')):
                await handle_text(update, context, message_text)
                return

            chat_id = update.message.chat_id

            # Periksa apakah sesi Redis sudah ada
            if not redis_client.exists(f"session:{chat_id}"):
                await initialize_session(chat_id)

            # Reset konteks jika diperlukan
            if await should_reset_context(chat_id, message_text):
                await initialize_session(chat_id)

            # Proses teks
            session = json.loads(redis_client.get(f"session:{chat_id}"))
            session['messages'].append({"role": "user", "content": message_text})
            redis_client.set(f"session:{chat_id}", json.dumps(session))

            # Cek apakah pesan terkait dengan konteks sebelumnya
            if session['messages'] and not is_related_to_context(message_text, session['messages']):
                await update.message.reply_text("Sepertinya pertanyaan Anda tidak terkait dengan topik sebelumnya. Mari kita kembali ke topik sebelumnya.")
                # Tampilkan konteks terakhir
                last_context = session['messages'][-1]['content']
                await update.message.reply_text(f"Topik terakhir: {last_context}")
                return

            # Proses pesan dengan konteks cerdas
            response = await process_with_smart_context(session['messages'][-10:])
            
            if response:
                # Filter hasil respons sebelum dikirim ke pengguna
                filtered_response = await filter_text(response)
                session['messages'].append({"role": "assistant", "content": filtered_response})
                redis_client.set(f"session:{chat_id}", json.dumps(session))

                # Pecah respons jika terlalu panjang
                response_parts = split_message(filtered_response)
                for part in response_parts:
                    await update.message.reply_text(part)
        else:
            logger.info("Pesan di grup tanpa mention yang valid diabaikan.")
