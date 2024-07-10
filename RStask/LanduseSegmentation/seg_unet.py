import logging
from skimage import  io
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2


def preprocess_1(img, device):
    img = img / 255

    img = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float32)
    img_tensor = torch.unsqueeze(img, 0)
    # img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = img_tensor.to(device)

    return img_tensor




def main(seg_img_dir, seg_ckpt_path):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    mapping = {0: (255, 0, 0),  # built-up
            1: (0, 255, 0),  # farmland
            2: (0, 255, 255),  # forest
            3: (255, 255, 0),  # meadow
            4: (0, 0, 255),  # water
            5: (0, 0, 0)}  # ignored
    seg_model = UNet(in_channels=3, out_channels=6)
    seg_model.load_state_dict(torch.load(seg_ckpt_path))
    seg_model.float().eval().to(device)

    seg_img_name = os.listdir(seg_img_dir)[0]
    seg_img_path = f'{seg_img_dir}/{seg_img_name}'
    seg_img = cv2.imread(seg_img_path)
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)  # train with pillow images(RGB), but inference with opencv(BGR)

    with torch.no_grad():
        seg_img_tensor = preprocess_1(seg_img, device)
        seg_output = seg_model(seg_img_tensor)

    probs = torch.softmax(seg_output, dim=1).squeeze(0)

    height, width = probs.shape[1:3]

    # probs -> class index -> RGB image
    seg_cls_idx = torch.argmax(probs, dim=0)
    seg_pred = torch.zeros(height, width, 3, dtype=torch.uint8)

    for k in mapping:
        idx = (seg_cls_idx == torch.tensor(k, dtype=torch.uint8))
        validx = (idx == 1)
        seg_pred[validx, :] = torch.tensor(mapping[k], dtype=torch.uint8)

    seg_pred = seg_pred.squeeze().cpu().numpy()
    seg_pred = cv2.resize(seg_pred, (seg_img.shape[0], seg_img.shape[1]))
    seg_pred = cv2.cvtColor(seg_pred, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{output_dir}/seg_out.png', seg_pred)

    print('Segmentation Done!')


def inference(self,image_path, det_prompt,updated_image_path):
    det_prompt=det_prompt.strip()
    image = torch.from_numpy(io.imread(image_path))
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

    pred = Image.fromarray(pred_vis.astype(np.uint8))
    pred.save(updated_image_path)
    print(f"\nProcessed Landuse Segmentation, Input Image: {image_path+','+det_prompt}, Output: {updated_image_path}")
    return det_prompt+' segmentation result in '+updated_image_path

if __name__=='__main__':
    net=main()
    print(sum(p.numel() for p in net.parameters()))
    x=torch.ones((2,3,512,512))
    output=net(x)
    print(output.shape)