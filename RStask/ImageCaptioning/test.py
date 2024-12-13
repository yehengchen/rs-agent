import torch
import torchvision

from PIL import Image
from transformers import  BlipProcessor, BlipForConditionalGeneration
from translate import Translator
from thop import profile

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

    def count_parameters(self):

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return total_params

    def count_flops(self):

        input = torch.randn(1, 3, 224, 224)
        flops, params = profile(self.model, inputs=(input, ))
        
        return flops, params

if __name__ == '__main__':
    rs_blip = RS_BLIP('cuda')
    image_path = './8a59_fire_detection_.png'
    rs_blip.inference(image_path)
    # print(f"\nModel Parameters: {rs_blip.count_parameters()}")
    total_params = rs_blip.count_parameters()
    print(f"\nModel Parameters: {total_params / 1000000000:.2f} B")
    
    # flops, params = rs_blip.count_flops()
    # print(f"\nModel FLOPS: {flops/1000000000:.2f} GFLOPS")
    # print(f"\nModel Parameters: {params/1000000:.2f} M")
    