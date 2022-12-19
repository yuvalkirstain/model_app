import os
import random
from typing import List, Dict
import base64
import socket
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler
import torch
from starlette.requests import Request
from ray import serve
import io
import time


auth_token = os.getenv("auth_token")
MODEL_ID = "stabilityai/stable-diffusion-2-1"
IMAGE_RES = 768
SCHEDULER_CLS = DPMSolverMultistepScheduler
STEPS = 25
GS = 9



@serve.deployment(route_prefix="/hello", num_replicas=4, ray_actor_options={"num_cpus": 4, "num_gpus": 1})
class TextToImgModel:
    def __init__(self):
        scheduler = SCHEDULER_CLS.from_pretrained(MODEL_ID, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            use_auth_token=auth_token,
            revision="fp16",
            torch_dtype=torch.float16,
            scheduler=scheduler
        )
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_vae_slicing()
        self.pipe.safety_checker = None


    def generate(self, prompt: List[str], negative_prompt: List[str], seed: int, steps: int, gs: float) -> Dict:
        print(f"Generating image for {prompt}")
        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
        start = time.time()
        with torch.inference_mode():
            images = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                negative_prompt=negative_prompt,
                guidance_scale=gs,
                generator=generator,
                height=IMAGE_RES,
                width=IMAGE_RES
            ).images
        end = time.time()
        print(f"Generation tool {end - start}")
        images_bytes = []
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            images_bytes.append(base64.b64encode(img_byte_arr.getvalue()))
        return {"images": images_bytes}

    @serve.batch(max_batch_size=2, batch_wait_timeout_s=2)
    async def handle_batch(self, inputs_lst: List[Dict]) -> List[Dict]:
        seed = random.randint(0, 2147483647)
        batched_inputs = {
            "prompt": [prompt for inputs in inputs_lst for prompt in inputs["prompt"]],
            "negative_prompt": [negative_prompt for inputs in inputs_lst for negative_prompt in inputs["negative_prompt"]],
            "seed": seed,
            "steps": STEPS,
            "gs": GS
        }
        total_images = len(batched_inputs["prompt"])
        print(f"Num examples: {len(inputs_lst)}\nNum prompts: {total_images}")

        results = self.generate(**batched_inputs)
        batch_response = []
        start_idx = 0
        for inputs in inputs_lst:
            response = {}
            num_samples = inputs["num_samples"]
            response["user_id"] = inputs["user_id"]
            response["prompt"] = inputs["prompt"][start_idx:start_idx+num_samples]
            response["negative_prompt"] = inputs["negative_prompt"][start_idx:start_idx+num_samples]
            response["seed"] = seed
            response["gs"] = GS
            response["steps"] = STEPS
            response["idx"] = [start_idx + i for i in range(num_samples)]
            response["num_generated"] = total_images
            response["scheduler_cls"] = SCHEDULER_CLS.__name__
            response["model_id"] = MODEL_ID
            response["images"] = results["images"][start_idx:start_idx+num_samples]
            batch_response.append(response)
            start_idx += num_samples
        return batch_response

    async def __call__(self, http_request: Request) -> Dict:
        print(f"Received request {http_request.json()}")
        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        print("Your Computer Name is:" + hostname)
        print("Your Computer IP Address is:" + IPAddr)
        inputs = await http_request.json()
        return await self.handle_batch(inputs)
        # return self.generate(**inputs)

# ray.init(address='ray://132.67.247.206:10001')
model = TextToImgModel.bind()

# model.pipe.to("cuda")
# model.pipe.enable_xformers_memory_efficient_attention()
# translation = model.generate("A mature poodle toy.")
# print(translation)
