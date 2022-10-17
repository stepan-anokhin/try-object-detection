import logging
import os.path
import tempfile

import httpx
import orjson
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from try_object_detection.detection import DetectionResults
from try_object_detection.workflow.detect_objects import detect_file

logger = logging.getLogger(__name__)


async def start(update: Update, context) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Only you can help yourself!")


async def echo(update: Update, context) -> None:
    """Echo the user message."""
    prompt = update.message.text
    async with httpx.AsyncClient() as client:
        response = await client.post('https://pelevin.gpt.dobro.ai/generate/',
                                     json={"prompt": f"Ты говоришь: {prompt}. Мой ответ: ", "length": 60},
                                     timeout=60000)
    logger.info(response)
    data = orjson.loads(response.content)
    await update.message.reply_text(data["replies"][0])


def description(results: DetectionResults) -> str:
    """Get caption for detection results."""
    counts = {}
    for cls in results.classes:
        label = results.labels[cls - 1]
        counts[label] = counts.get(label, 0) + 1
    caption = "Найдено:\n"
    for label, count in counts.items():
        caption += f"{label}: {count}\n"
    return caption


async def detect(update: Update, context) -> None:
    """Detect objects on photo the user message."""
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input")
        output_path = os.path.join(tmpdir, "output.jpg")
        await file.download(input_path)
        results = detect_file(input_path, output_path)
        caption = description(results)
        if results.classes:
            with open(output_path, "rb") as output_file:
                await update.message.reply_photo(output_file, quote=True, caption=caption)
        else:
            await update.message.reply_text("Ничего не разобрать...")


def main() -> None:
    """Start the bot."""
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, detect))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == '__main__':
    main()
