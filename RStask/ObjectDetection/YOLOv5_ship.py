# from RStask.ObjectDetection.models.common import DetectMultiBackend
from RStask.ObjectDetection.models.yolov5s import *
import os
import sys
import torch
from skimage import io
import numpy as np
import torchvision
import cv2
from PIL import Image
import time

def preprocess(img, device):
    img = img / 255

    # img = torch.from_numpy(img).to(torch.float32)
    img_tensor = torch.unsqueeze(img, 0)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = img_tensor.to(device)

    return img_tensor


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

class YoloDetection_ship:
    def __init__(self, device):
        self.device = device
        # self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.category = ["ship"]

        self.det_model = yolov5s(num_classes=len(self.category), slice=False)
        # self.det_model.load_state_dict(torch.load('/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/checkpoints/yolov5_best.pt')['model'].state_dict())
        self.det_model.load_state_dict(torch.load('/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/checkpoints/v5s_ship.pt')['model'].state_dict())
        self.det_model.float().fuse().eval().to(self.device)
    # def __init__(self, device):
        # self.device = device
        # try:
        #     self.model = DetectMultiBackend('/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/checkpoints/v5s_ship.pt', device=torch.device(device), dnn=False, fp16=False)
        # except:
        #     self.model = DetectMultiBackend('/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/checkpoints/v5s_ship.pt', device=torch.device(device), dnn=False,fp16=False)
        # self.category = ["ship"]

    def inference(self, image_path, det_prompt,updated_image_path):

        det_img = torch.from_numpy(io.imread(image_path))
        print('det_img_shape',det_img.shape)
        
        h, w = det_img.shape
        with torch.no_grad():
            det_img_tensor = preprocess(det_img, self.device)
            det_output, _ = self.det_model(det_img_tensor)
            det_pred = non_max_suppression(det_output, conf_thres=0.1, iou_thres=0.2, labels=[], multi_label=True,
                                             agnostic=False)[0]
            detections = det_pred.clone()
            detections = detections[det_pred[:, 4] > 0.75]
            detections_box = (detections[:, :4] / (640 / h)).int().cpu().numpy()
            detection_classes = detections[:, 5].int().cpu().numpy()
        if len(detection_classes) > 0:
            det = np.zeros((h, w, 3))
            for i in range(len(detections_box)):
                x1, y1, x2, y2 = detections_box[i]
                det[y1:y2, x1:x2] = detection_classes[i] + 1

            self.visualize(image_path,updated_image_path,detections)
            print(
                f"\nProcessed Object Detection, Input Image: {image_path}, Output Bounding box: {updated_image_path},Output text: {'Object Detection Done'}")
            return  det_prompt+' object detection result in '+updated_image_path
        # visualize
    def visualize(self,image_path, newpic_path,detections):
        font = cv2.FONT_HERSHEY_SIMPLEX
        im = io.imread(image_path)
        boxes = detections.int().cpu().numpy()
        for i in range(len(boxes)):
            cv2.rectangle(im, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 255), 2)
            cv2.rectangle(im, (boxes[i][0], boxes[i][1] - 15), (boxes[i][0] + 45, boxes[i][1] - 2), (0, 0, 255),thickness=-1)
            cv2.putText(im, self.category[boxes[i][-1]], (boxes[i][0], boxes[i][1] - 2), font, 0.5, (255, 255, 255),1)
        Image.fromarray(im.astype(np.uint8)).save(newpic_path)
        with open(newpic_path[:-4]+'.txt','w') as f:
            for i in range(len(boxes)):
                f.write(str(list(boxes[i,:4]))[1:-1]+', '+self.category[boxes[i][-1]]+'\n')
        # cv2.imshow("Object detection", im)

   