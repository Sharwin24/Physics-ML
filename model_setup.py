import os
import base64
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import torch
import json

def model_setup():
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

    json_file = 'data/PhysUnivBench_en_unified.json'
    #iterate through json file and tokenize every entry
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        texts = [str(entry) for entry in data]
    #tokenize all texts at once
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    print(tokenized_texts['input_ids'].shape)
    

    


    

    






  
print(model_setup())