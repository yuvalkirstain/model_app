import base64
import io
import random
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
import jax
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from diffusers import FlaxDPMSolverMultistepScheduler, FlaxStableDiffusionPipeline
from pydantic import BaseModel


MODEL_ID = "stabilityai/stable-diffusion-2-1"
N_STEPS = 25
GS = 9.0
NEG_PROMPT = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

app = FastAPI()

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind
print(f"Found {num_devices} JAX devices of type {device_type}.")
assert "TPU" in device_type, "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"

scheduler, scheduler_params = FlaxDPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    scheduler=scheduler,
    revision="bf16",
    dtype=jnp.bfloat16,
)
params["scheduler"] = scheduler_params


class GenImage(BaseModel):
    prompt: str
    num_samples: int = 4
    user_id: Optional[str] = None
    negative_prompt: Optional[str] = NEG_PROMPT


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def create_key(seed=0):
    return jax.random.PRNGKey(seed)


def images2bytes(images):
    images_bytes = []
    for image in images:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        images_bytes.append(base64.b64encode(img_byte_arr.getvalue()))
    return images_bytes


@app.post("/generate")
def generate_image(gen_image: GenImage):
    prompt = gen_image.prompt
    print(f"Prompt: {prompt}")

    prompt = [prompt] * jax.device_count()
    prompt_ids = pipeline.prepare_inputs(prompt)
    negative_prompt = [gen_image.negative_prompt] * jax.device_count()
    neg_prompt_ids = pipeline.prepare_inputs(negative_prompt)

    p_params = replicate(params)
    prompt_ids = shard(prompt_ids)
    neg_prompt_ids = shard(neg_prompt_ids)

    seed = random.randint(0, 2147483647)
    rng = create_key(seed)
    rng = jax.random.split(rng, jax.device_count())

    print("Generating images...")
    start = datetime.now()
    images = pipeline(
        prompt_ids=prompt_ids,
        params=p_params,
        prng_seed=rng,
        num_inference_steps=N_STEPS,
        guidance_scale=GS,
        neg_prompt_ids=neg_prompt_ids,
        jit=True
    )[0]
    gen_time = datetime.now() - start
    print(f"Generated {len(images)} images in {gen_time}")

    images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
    images = pipeline.numpy_to_pil(images)

    image_bytes = images2bytes(images)
    num_images = len(images)
    return {
        "user_id": gen_image.user_id,
        "prompt": [gen_image.prompt] * num_images,
        "negative_prompt": [gen_image.negative_prompt] * num_images,
        "seed": seed,
        "gs": GS,
        "steps": N_STEPS,
        "idx": [i for i in range(num_images)],
        "num_generated": num_images,
        "scheduler_cls": scheduler.__class__.__name__,
        "model_id": MODEL_ID,
        "images": image_bytes,
    }


@app.get("/")
async def root():
    return {"message": "Hello World", "num_devices": num_devices, "device_type": device_type}

# Dummy to do jit compile before the user
generate_image(GenImage(prompt="a dog"))
print("Finished")
