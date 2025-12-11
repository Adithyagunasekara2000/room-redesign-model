import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
).to(device)


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
).to(device)

pipe.enable_xformers_memory_efficient_attention()


def estimate_depth(image: Image.Image) -> Image.Image:
    """
    Convert the input RGB image to a pseudo depth map using a simple method.
    In a more advanced version, you’d use a real depth estimation model,
    but this keeps the example lightweight.
    """
    # Convert PIL to numpy
    img = np.array(image.convert("RGB"))
    # Convert to grayscale as a fake "depth" approximation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    depth = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
   
    depth_image = Image.fromarray(depth)
    return depth_image


def predict(image, prompt: str = "modern interior design, bright, stylish, high quality"):
    """
    image: file-like object from Replicate (input room photo)
    prompt: text prompt defining the new style
    """
   
    init_image = load_image(image).convert("RGB")

  
    depth_map = estimate_depth(init_image)

    result = pipe(
        prompt=prompt,
        image=init_image,
        control_image=depth_map,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
    )

   
    return result.images[0]
