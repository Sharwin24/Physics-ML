
from typing import Optional
from transformers import AutoProcessor, LLavaForConditionalGeneration
from PIL import Image
import torch


class LLaVAInterface:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"LLaVA Model {self.model_name} initialized on {self.device}")

    def load_model(self):
        # Placeholder for model loading logic
        print(f"Loading model {self.model_name} on {self.device}")
        return LLavaForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def generate_response(self, prompt: str, image_path: Optional[str] = None) -> str:
        if image_path:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(text=prompt, images=image,
                                    return_tensors="pt").to(self.device, torch.float16)
        else:
            inputs = self.processor(text=prompt, return_tensors="pt").to(
                self.device, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=200
        )
        response = self.processor.batch_decode(
            output, skip_special_tokens=True)[0]
        return response
