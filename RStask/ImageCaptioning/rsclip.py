from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel

class RS_CLIP:
    # 初始化模型和处理器
    # 模型和处理器的参数应该根据具体需求进行调整
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
        self.model = CLIPModel.from_pretrained(
            "flax-community/clip-rsicd-v2", torch_dtype=self.torch_dtype).to(self.device)
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = 'A satellite image of ' + self.processor.decode(out[0], skip_special_tokens=True)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions
