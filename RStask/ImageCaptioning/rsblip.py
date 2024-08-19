import torch
import torchvision

from PIL import Image
from transformers import  Blip2Processor, Blip2ForConditionalGeneration
from transformers import  BlipProcessor, BlipForConditionalGeneration
from translate import Translator

class RS_BLIP:
    def __init__(self, device):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_path = '/home/mars/cyh_ws/LLM/models/rs-blip/' 
        self.processor = BlipProcessor.from_pretrained(self.model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=self.torch_dtype).to(self.device)
    


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
    
    def inference_app(self, raw_image):
    
        
        inputs = self.processor(Image.open(raw_image), return_tensors="pt").to(self.device, self.torch_dtype)
        # inputs = self.processor(raw_image, return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions_app = 'A satellite image of ' + self.processor.decode(out[0], skip_special_tokens=True)
        print(type(captions_app))
        # captions_CN = Translator(from_lang="EN-US",to_lang="ZH").translate(str(captions_app))
        # captions_CN = '这张遥感图像显示了' + Translator(from_lang="EN-US", to_lang="ZH").translate(captions_app)
        # captions = 'A satellite image of many ships in port of Singapore.'
        # captions_CN = '这张遥感图像显示了' + Translator(from_lang="EN-US",to_lang="ZH").translate(self.processor.decode(out[0], skip_special_tokens=True))
        # captions = self.processor.decode(out[0], skip_special_tokens=True)
        # captions = 'A satellite image of basketball court near by zhejiang lab.'
        # captions = 'A satellite image of bridge near by zhejiang lab.'
        # captions = 'A satellite image of ​​farmland near by zhejiang lab.'
        # captions = '这张遥感图像显示了之江实验室附近有农田。'


        print(f"\nProcessed ImageCaptioning, Input Image: {raw_image}, Output Text: {captions_app}")
        return captions_app
    