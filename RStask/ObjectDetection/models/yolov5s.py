import time

import torch
import torch.nn as nn
import numpy as np
import os, sys, math, warnings
from copy import deepcopy
from collections import OrderedDict


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        m.anchors[:] = m.anchors.flip(0)


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def convert_pt(src_path: str, dst_path: str, yolov5_path='../yolov5_v61', device=torch.device('cpu'), num_classes=1):
    sys.path.insert(0, yolov5_path)
    src_best_pt = torch.load(src_path, device)

    model_src = src_best_pt['ema'] if src_best_pt['ema']!=None else src_best_pt['model']
    model_src_dict = model_src.state_dict()

    model_dst = yolov5s(num_classes=num_classes)
    model_dst_dict = model_dst.state_dict()
    model_dst.load_state_dict(model_dst_dict,)

    model_dst_dict_list = list(model_dst_dict)
    for ind, (key, value) in enumerate(model_src_dict.items()):
        model_dst_dict[model_dst_dict_list[ind]] = value

    model_dst.load_state_dict(model_dst_dict)
    ckpt = {'epoch': src_best_pt['epoch'],
            'best_fitness': src_best_pt['best_fitness'],
            'training_results': None,
            'model': deepcopy(model_dst),
            'optimizer': src_best_pt['optimizer'] }
    torch.save(ckpt,dst_path)


class Slice(nn.Module):
    # Slice image of index (row, col)
    def __init__(self, n_row=2, n_col=2, gap=0.3):
        super().__init__()
        self.n_row = n_row
        self.n_col = n_col
        self.gap = gap

    def forward(self, x: torch.Tensor):  # x(b,c,w,h) -> y(4b,c,w/2,h/2)
        temp1 = x[:, :, 0: 640, 0: 640]
        temp2 = x[:, :, 0: 640, 448: 1088]
        temp3 = x[:, :, 448: 1088, 0: 640]
        temp4 = x[:, :, 448: 1088, 448: 1088]
        # temp2 = x[:, :, 0: 640, 320: 960]
        # temp3 = x[:, :, 320: 960, 0: 640]
        # temp4 = x[:, :, 320: 960, 320: 960]
        return torch.cat([temp1, temp2, temp3, temp4], 0)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
    

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
    

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
    

class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=1, anchors:list=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], 
                 ch:list=[128, 256, 512], inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers, 3
        self.na = len(anchors[0]) // 2  # number of anchors, 6//2=3
        self.grid = [torch.zeros(1)] * self.nl  # init grid, [tensor([0.]), tensor([0.]), tensor([0.])]
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid, [tensor([0.]), tensor([0.]), tensor([0.])]
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment), True
        
        

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        # if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
        #     yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        # else:
        warnings.filterwarnings('ignore')
        yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        # stack: Concatenates a sequence of tensors along a new dimension. 
        # expand: Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid
    

class yolov5s(nn.Module):
    def __init__(self, ch_in=1, num_classes=20,
                 anchors = [[5,4, 8,7, 16,8],        # P3/8
                            [12,14, 25,13, 20,23],       # P4/16
                            [40,21, 30,40, 69,51]],  # P5/32
                 slice=False, n_row=2, n_col=2, gap=0.0
                ):

        super().__init__()
        self.slice = slice
        self.n_row = n_row
        self.n_col = n_col
        self.gap = gap
        """
        depth_multiple: 0.33  # model depth multiple
        width_multiple: 0.50  # layer channel multiple
        """
        self.model = nn.Sequential(OrderedDict([
            ('slice', Slice(n_row, n_col, gap)),
            # backbone
            ('P1', Conv(ch_in, 32, 6, 2, 2)),            # 0-P1/2
            ('P2', Conv(32, 64, 3, 2)),                  # 1-P2/4
            ('C3_1', C3(64, 64, 1)),
            ('P3', Conv(64, 128, 3, 2)),                 # 3-P3/8
            ('C3_2', C3(128, 128, 2)),
            ('P4', Conv(128, 256, 3, 2)),                # 5-P4/16
            ('C3_3', C3(256, 256, 3)),
            ('P5', Conv(256, 512, 3,2)),                 # 7-P5/32
            ('C3_4', C3(512, 512, 1)),
            ('sppf', SPPF(512, 512, 5)),                 # 9

            # head
            # up
            ('conv_for_up1', Conv(512, 256, 1, 1)),      # 10 large
            ('up1', nn.Upsample(scale_factor=2)),
            ('cat1', Concat(dimension=1)),               # 12 cat backbone P4
            ('C3_5', C3(512, 256, 1, False)),            # 13

            ('conv_for_up2', Conv(256, 128, 1, 1)),      # 14 medium
            ('up2', nn.Upsample(scale_factor=2)),     
            ('cat2', Concat(dimension=1)),               # 16 cat backbone P3
            ('C3_6', C3(256, 128, 1, False)),            # 17 small
            # down
            ('conv_for_down1', Conv(128, 128, 3, 2)),
            ('cat3', Concat(dimension=1)),               # 19 cat head P4
            ('C3_7', C3(256, 256, 1, False)),            # 20 medium

            ('conv_for_down2', Conv(256, 256, 3, 2)),
            ('cat4', Concat(dimension=1)),               # 22 cat head P5 
            ('C3_8', C3(512, 512, 1, False)),            # 23 large

            ('detect', Detect(nc=num_classes, anchors=anchors, ch=[128, 256, 512], inplace=True)) # 24 detect
        ]))

        s = 256  # 2x min stride
        self.detect_index = -1
        self.model[self.detect_index].stride = torch.tensor([8, 16, 32])  # forward
        self.model[self.detect_index].anchors /= self.model[-1].stride.view(-1, 1, 1)
        check_anchor_order(self.model[self.detect_index])
        self.stride = self.model[self.detect_index].stride
        self._initialize_biases()
        
        # Init weights, biases
        initialize_weights(self)


    def forward(self, x: torch.Tensor):
        if self.slice:
            return self.forward_one(self.model[0](x))
        else: 
            return self.forward_one(x)


    def forward_one(self, x):
        # backbone
        x_0   = self.model[1](x)                # 0-P1/2
        x_1   = self.model[2](x_0)              # 1-P2/4
        x_2   = self.model[3](x_1)
        x_3   = self.model[4](x_2)              # 3-P3/8
        x_4   = self.model[5](x_3)
        x_5   = self.model[6](x_4)              # 5-P4/16
        x_6   = self.model[7](x_5)
        x_7   = self.model[8](x_6)              # 7-P5/32
        x_8   = self.model[9](x_7)
        x_9   = self.model[10](x_8)              # 9

        # head
        # up
        x_10  = self.model[11](x_9)             # 10 large
        x_11  = self.model[12](x_10)
        x_12  = self.model[13]([x_11, x_6])     # 12 cat backbone P4
        x_13  = self.model[14](x_12)            # 13

        x_14  = self.model[15](x_13)            # 14 medium
        x_15  = self.model[16](x_14)
        x_16  = self.model[17]([x_15, x_4])     # 16 cat backbone P3
        x_17  = self.model[18](x_16)            # 17 small

        # down
        x_18  = self.model[19](x_17)
        x_19  = self.model[20]([x_18, x_14])    # 19 cat head P4
        x_20  = self.model[21](x_19)            # 20 medium

        x_21  = self.model[22](x_20)
        x_22  = self.model[23]([x_21, x_10])    # 22 cat head P5 
        x_23  = self.model[24](x_22)            # 23 large

        out   = self.model[25]([x_17, x_20, x_23]) # 24 detect
        return out

    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[self.detect_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            with torch.no_grad():
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def _print_biases(self):
        m = self.model[self.detect_index]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))


    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self


    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[self.detect_index]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


if __name__=='__main__':
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    device = torch.device('cuda:0')
    print(device)
    convert_pt(src_path='../yolov5_v61/runs/train/dota2_clsall_512_256/weights/best.pt',
               dst_path='./weights/yolov5s-dota2_clsall_224.pt',
               num_classes=18,
               device=device)
    # model_path = "/home/nvidia/yxq_workspace/code/weights/yolov5s-ch1-ship.pt"

    det_model = Model(num_classes=1, slice=False)
    print(det_model)
    # det_model.load_state_dict(torch.load(model_path)['model'].state_dict())
    det_model.float().fuse().eval().to(device)

    imgsz = 1024
    img = torch.rand((64,1,imgsz,imgsz)).to(device)
    # model = yolov5s(num_classes=18, slice=False)
    #
    # model.load_state_dict(torch.load('./weights/yolov5s-dota2_clsall_224.pt')['model'].state_dict())
    # model.float().fuse().eval().to(device)
    t1 = time.time()
    result = det_model(img)
    print(time.time() - t1)
    # print(det_model)
    print(result[0].shape, [result[1][i].shape for i in range(len(result[1]))])
    
