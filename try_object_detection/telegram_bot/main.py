import asyncio
import functools
import logging
import os.path
import random
import re
import tempfile
from typing import Tuple

import cv2
import httpx
import orjson
from telegram import ForceReply, Update
from telegram.constants import ChatType
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from try_object_detection.detection import DetectionResults
from try_object_detection.detection.mobile_net import load_model
from try_object_detection.workflow import run as prepare_data
from try_object_detection.workflow.detect_objects import detect_file

logger = logging.getLogger(__name__)


@functools.cache
def preload_model() -> cv2.dnn_DetectionModel:
    """Preload ObjectDetection model."""
    prepared = prepare_data()
    return load_model(prepared.model_file, prepared.config_path)


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


def is_reply_to_me(update: Update, context) -> bool:
    return (
            update.message.reply_to_message and
            update.message.reply_to_message.from_user.id == context.bot.id
    )


def mentions_me(update: Update, context) -> bool:
    return (
            re.match(fr'^(.*[\W\s])?[Бб]от([\W\s].*)?$', update.message.text) or
            context.bot.name in update.message.text or
            context.bot.first_name in update.message.text
    )


def should_answer(update: Update, context) -> bool:
    """Check if bot should answer to text message."""
    return (
            update.effective_chat.type == ChatType.PRIVATE or
            is_reply_to_me(update, context) or
            mentions_me(update, context) or
            random.random() < 0.05
    )


async def echo(update: Update, context) -> None:
    """Echo the user message."""
    if not should_answer(update, context):
        return
    prompt = update.message.text
    async with httpx.AsyncClient() as client:
        response = await client.post('https://pelevin.gpt.dobro.ai/generate/',
                                     json={"prompt": f"Ты говоришь: {prompt}. Мой ответ: ", "length": 60},
                                     timeout=60000)
    logger.info(response)
    data = orjson.loads(response.content)
    await update.message.reply_text(data["replies"][0])


def detected_person(results: DetectionResults) -> bool:
    """Check if persons are detected."""
    labels = {results.labels[cls - 1] for cls in results.classes if 1 <= cls <= len(results.labels)}
    return "person" in labels


def description(results: DetectionResults) -> Tuple[str, bool]:
    """Get caption for detection results."""
    counts = {}
    for cls in results.classes:
        if 1 <= cls <= len(results.labels):
            label = results.labels[cls - 1]
            counts[label] = counts.get(label, 0) + 1
    caption = "Найдено:\n"
    for label, count in counts.items():
        caption += f"{label}: {count}\n"
    return caption, bool(counts)


async def detect(update: Update, context) -> None:
    """Detect objects on photo the user message."""
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input")
        output_path = os.path.join(tmpdir, "output.jpg")
        await file.download(input_path)
        results = detect_file(input_path, output_path)
        caption, detected_known_objects = description(results)
        if detected_known_objects:
            with open(output_path, "rb") as output_file:
                await update.message.reply_photo(output_file, quote=True, caption=caption)
        else:
            await update.message.reply_text("Ничего не разобрать...")


async def capture_cam(update: Update, context) -> None:
    """Handle /cam command."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    with tempfile.TemporaryDirectory() as tempdir:
        frame_path = os.path.join(tempdir, "frame.jpg")
        output_path = os.path.join(tempdir, "output.jpg")
        cv2.imwrite(frame_path, frame)
        results = detect_file(frame_path, output_path)
        caption, detected_known_objects = description(results)
        with open(output_path, "rb") as output_file:
            await update.message.reply_photo(output_file, quote=True, caption=caption)


async def alarm(update: Update, context):
    """Handle /alarm command"""
    cap = cv2.VideoCapture(0)

    async def watch_cam():
        while True:
            await asyncio.sleep(1)
            ret, frame = cap.read()
            with tempfile.TemporaryDirectory() as tempdir:
                frame_path = os.path.join(tempdir, "frame.jpg")
                output_path = os.path.join(tempdir, "output.jpg")
                cv2.imwrite(frame_path, frame)
                results = detect_file(frame_path, output_path)
                if detected_person(results):
                    caption = "Тревога! Кто-то в комнате!"
                    with open(output_path, "rb") as output_file:
                        await update.message.reply_photo(output_file, quote=True, caption=caption)

    asyncio.create_task(watch_cam())


def main() -> None:
    """Start the bot."""
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cam", capture_cam))
    application.add_handler(CommandHandler("alarm", alarm))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, detect))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == '__main__':
    main()
