from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Load image
image_path = "your_image.png"  # Replace with your own
image = Image.open(image_path).convert("RGB")

# Set the model (you can also try llava-hf/llava-1.5-13b-hf or others)
model_id = "llava-hf/llava-1.5-7b-hf"