import requests
from io import BytesIO
import base64
import ray
from PIL import Image

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=5900, stdoutToServer=True, stderrToServer=True)


@ray.remote
def send_query(json):
    resp = requests.post("http://127.0.0.1:8511/hello", json=json)
    return resp.json()


json = {'prompt': ['A mature poodle toy.', 'A mature poodle toy.', 'A mature poodle toy.', 'A mature poodle toy.'], 'negative_prompt': ['ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy', 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy', 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy', 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy'], 'seed': 1163782286, 'steps': 50, 'gs': 9, 'num_samples': 4}

responses = ray.get([send_query.remote(json) for _ in range(10)])

images = responses[0].json()["images"]
for image in images:
    image = Image.open(BytesIO(base64.b64decode(image)))
    image.save("test.png")
