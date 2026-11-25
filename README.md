import asyncio
import io
import os
import threading
from pathlib import Path

import discord
from discord.ext import commands
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="$", intents=intents)

_yolo_model = None
_yolo_lock = threading.Lock()
_image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
_max_image_size = 10 * 1024 * 1024


def recognize_image(image_data):
    global _yolo_model

    with Image.open(io.BytesIO(image_data)) as image:
        image = image.convert("RGB")

    with _yolo_lock:
        if _yolo_model is None:
            _yolo_model = YOLO("yolo11n.pt")
        result = _yolo_model(image, verbose=False)[0]

    if result.boxes is None:
        return []

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        detections.append((_yolo_model.names[class_id], confidence))
    return detections


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


@bot.command()
async def hello(ctx):
    await ctx.send(f"Hi! I am a bot {bot.user}!")


@bot.command()
async def heh(ctx, count_heh=5):
    await ctx.send("he" * count_heh)


@bot.command()
async def check(ctx):
    attachments = [
        attachment
        for attachment in ctx.message.attachments
        if (attachment.content_type or "").startswith("image/")
        or Path(attachment.filename).suffix.lower() in _image_extensions
    ]

    if not attachments:
        await ctx.send("Envie uma imagem junto com o comando `$check`.")
        return

    for attachment in attachments:
        if attachment.size > _max_image_size:
            await ctx.send(f"A imagem `{attachment.filename}` excede o limite de 10 MB.")
            continue

        try:
            image_data = await attachment.read()
            detections = await asyncio.to_thread(recognize_image, image_data)
        except UnidentifiedImageError:
            await ctx.send(f"Não consegui abrir `{attachment.filename}` como imagem.")
            continue

        if not detections:
            await ctx.send(f"Não reconheci objetos em `{attachment.filename}`.")
            continue

        detection_text = "\n".join(
            f"- {name} ({confidence:.0%})"
            for name, confidence in detections[:10]
        )
        await ctx.send(f"**{attachment.filename}**\n{detection_text}")


token = os.getenv("DISCORD_BOT_TOKEN")
if not token:
    raise RuntimeError("Defina a variável de ambiente DISCORD_BOT_TOKEN.")

bot.run(token)
