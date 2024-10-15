from ResNetScene import ResNetAID
from thop import profile, clever_format
from torchvision import models

def run_classification(image_path):
    model = models.resnet34(pretrained=False, num_classes=30)
    
    # model = ResNetAID('cuda:0')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # cls = model.predict(image_path)

    return  total_params

if __name__ == "__main__":

    image_path = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/rs-ship.jpg'
    total_params = run_classification(image_path)
    # print(cls_rslt)
    print(f"\nModel Parameters: {total_params / 1000000000:.2f} B")

