import base64
import uuid
from datetime import datetime, timezone
from io import BytesIO

import boto3
import requests
from PIL import Image

BACKEND_URL = "http://35.225.27.48:5000/generate"


def bytes2images(images):
    pil_images = []
    for image in images:
        image = Image.open(BytesIO(base64.b64decode(image)))
        pil_images.append(image)
    return pil_images


prompt = "A mature poodle toy."
response = requests.post(
    BACKEND_URL,
    json={
        "prompt": prompt,
    }
)
response_json = response.json()
images = bytes2images(response_json.pop("images"))
for i, image in enumerate(images):
    image.save(f"test_{i}.png")
