import logging
from skimage import  io
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from RStask.LanduseSegmentation.unet import *
import os

def preprocess_1(img, device):
    img = img / 255

    img = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float32)
    img_tensor = torch.unsqueeze(img, 0)
    # img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = img_tensor.to(device)

    return img_tensor

class Unet_seg(nn.Module):
    def __init__(self, device):
        super(Unet_seg, self).__init__()
        self.model=unet(n_channels=3, n_classes=6)
        self.device = device
        self.model.load_state_dict(torch.load('/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/checkpoints/unet_3ch_GID_5_seg.pt'))
        self.model.float().eval().to(device)

        self.category = ['building','farmland', 'forest', 'meadow', 'water', 'ignored']
        self.color_bar=[[255, 0, 0],[0, 255, 0],[0, 255, 255],[255, 255, 0],[0, 0, 255],[0, 0, 0]]
        self.mean, self.std = torch.tensor([123.675, 116.28, 103.53]).reshape((1, 3, 1, 1)), torch.tensor(
            [58.395, 57.12, 57.375]).reshape((1, 3, 1, 1))
    def visualize(self,pred,cls):
        vis=np.zeros([pred.shape[0],pred.shape[1],3]).astype(np.uint8)
        if len(cls)>1:
            for i in range(len(self.category)):
                vis[:,:,0][pred==i]=self.color_bar[i][0]
                vis[:,:,1][pred == i] = self.color_bar[i][1]
                vis[:,:,2][pred == i] = self.color_bar[i][2]
        else:
            idx=cls[0]
            vis[:, :, 0][pred == idx] = self.color_bar[idx][0]
            vis[:, :, 1][pred == idx] = self.color_bar[idx][1]
            vis[:, :, 2][pred == idx] = self.color_bar[idx][2]
        return vis


    def inference(self,image_path, det_prompt, updated_image_path):
        det_prompt=det_prompt.strip()
        try:
            or_image = io.imread(image_path)
            image = torch.from_numpy(io.imread(image_path))
            image = (image.permute(2, 0, 1).unsqueeze(0) - self.mean) / self.std
        except:
            print('Image format error!')
            return ('Category ',det_prompt,' do not suuport!','The expected input category include Building, Road, Water, Barren, Forest, Farmland, Landuse.')
        
        with torch.no_grad():
            b, c, h, w = image.shape
            pred = self.model(image.to(self.device))
            pred = F.interpolate(pred, (h, w), mode='bilinear')
        pred = pred.argmax(1).cpu().squeeze().int().numpy()
        if det_prompt.lower() == 'landuse':
            pred_vis = self.visualize(pred, self.category)
        elif det_prompt.lower() in [i.lower() for i in self.category]:
            idx=[i.lower() for i in self.category].index(det_prompt.strip().lower())
            pred_vis = self.visualize(pred, [idx])
        else:
            print('Category ',det_prompt,' do not suuport!')
            return ('Category ',det_prompt,' do not suuport!','The expected input category include Building, Road, Water, Barren, Forest, Farmland, Landuse.')
        
        pred = Image.fromarray(pred_vis.astype(np.uint8))
        result = cv2.addWeighted(or_image, 0.5, pred_vis, 0.5, 0)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # pred = cv2.addWeighted(image, 0.5, pred_vis, 0.5, 0)
        cv2.imwrite(updated_image_path, result)
        # pred.save(updated_image_path)
        print(f"\nProcessed Landuse Segmentation, Input Image: {image_path+','+det_prompt}, Output: {updated_image_path}")
        
        return det_prompt #+' segmentation result in '+updated_image_path

    def inference_app(self,image_path, updated_image_path):
        det_prompt= 'landuse'
        or_image = io.imread(image_path)
        image = torch.from_numpy(or_image)
        image = (image.permute(2, 0, 1).unsqueeze(0) - self.mean) / self.std
        with torch.no_grad():
            b, c, h, w = image.shape
            pred = self.model(image.to(self.device))
            pred = F.interpolate(pred, (h, w), mode='bilinear')
        pred = pred.argmax(1).cpu().squeeze().int().numpy()
        if det_prompt.lower() == 'landuse':
            pred_vis = self.visualize(pred, self.category)
        elif det_prompt.lower() in [i.lower() for i in self.category]:
            idx=[i.lower() for i in self.category].index(det_prompt.strip().lower())
            pred_vis = self.visualize(pred, [idx])
        else:
            print('Category ',det_prompt,' do not suuport!')
            return ('Category ',det_prompt,' do not suuport!','The expected input category include Building, Road, Water, Barren, Forest, Farmland, Landuse.')

        print(f"\nProcessed Landuse Segmentation, Input Image: {image_path+','+det_prompt}, Output: {updated_image_path}")

        result = cv2.addWeighted(or_image, 0.5, pred_vis, 0.5, 0)
        return result, image.shape


if __name__=='__main__':
    net=Unet_seg()
    print(sum(p.numel() for p in net.parameters()))
    x=torch.ones((2,3,512,512))
    output=net(x)
    print(output.shape)