import os
import base64
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import torch

def model_setup():
    #image embedding
    print(os.path.abspath("data/images"))
    clip_model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_model_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

    image_folder = "./data/images"
    bad_files, recovered_files, total_files = 0, 0, 0

    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        total_files += 1
        is_valid = False

        try:
            with Image.open(image_path) as img:
                img.verify()
                is_valid = True
        except Exception:
            try:
                with open(image_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                print(f"\n {filename} contents preview:")
                print(repr(content[:200])) 
                image_data = base64.b64decode(content)
                image = Image.open(BytesIO(image_data)).convert("RGB")
                image.save(image_path)
                recovered_files += 1
                is_valid = True
            except Exception as e:
                print(f"{filename} is invalid and could not be recovered: {e}")
                bad_files += 1
        if is_valid:
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_embeds = clip_model.get_image_features(**inputs)
                print(f" {filename} → Embedding shape: {image_embeds.shape}")
            except Exception as embed_error:
                print(f"Failed to embed {filename}: {embed_error}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    


    

    






    # Set the model (you can also try llava-hf/llava-1.5-13b-hf or others)
    # model_id = "llava-hf/llava-1.5-7b-hf"

    # model = LlavaForConditionalGeneration.from_pretrained(
    #     model_id, torch_dtype=torch.float16, device_map="auto"
    # )
print(model_setup())