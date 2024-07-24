import torch
from PIL import Image
from transformers import  Blip2Processor, Blip2ForConditionalGeneration
from transformers import  BlipProcessor, BlipForConditionalGeneration

class RS_BLIP:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = BlipProcessor.from_pretrained("/home/mars/cyh_ws/LLM/models/rs-blip-gray")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "/home/mars/cyh_ws/LLM/models/rs-blip-gray", torch_dtype=self.torch_dtype).to(self.device)
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = 'A satellite image of ' + self.processor.decode(out[0], skip_special_tokens=True)
        # captions = 'A satellite image of many ships in port of Singapore.'

        # captions = 'A satellite image of basketball court near by zhejiang lab.'
        # captions = 'A satellite image of bridge near by zhejiang lab.'
        # captions = 'A satellite image of ​​farmland near by zhejiang lab.'
        # captions = '这张遥感图像显示了之江实验室附近有农田。'


        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions