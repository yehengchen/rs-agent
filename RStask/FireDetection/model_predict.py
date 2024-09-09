# -*- coding: utf-8 -*-
# @File : train.py
# @Author : Kaicheng Yang
# @Time : 2022/01/26 11:03:11
import os
import torch

from model_define import DTransformer
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
import numpy as np
import scipy.io as scio
# np.seterr(divide='ignore',invalid='ignore')
# # from scheduler import cosine_lr
from torchvision import  transforms
# logging.basicConfig(level=logging.NOTSET)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from osgeo import gdal


def argsoutput():
    parser = argparse.ArgumentParser()
    # Optimizer parameters
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.5e-5")
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument("--beta1", type=float, default=0.99, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument(
        "--warmup", type=int, default=500, help="Number of steps to warmup for."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Number of steps to warmup for.")
    parser.add_argument("--epoches", type=int, default=50, help="Number of steps to warmup for.")
    # Vit params
    parser.add_argument("--output", default='./output', type=str)
    parser.add_argument("--vit_model", default='./Vit_weights/imagenet21k+imagenet2012_ViT-B_16-224.pth', type=str)
    parser.add_argument("--load", type=bool, default=False, help="Load pretrained model")
    parser.add_argument("--image_size", type=int, default=224, help="input image size", choices=[224, 384])
    parser.add_argument("--num-classes", type=int, default=2, help="number of classes in dataset")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--emb_dim", type=int, default=768)
    parser.add_argument("--mlp_dim", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--attn_dropout_rate", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--output_path", type=str,
                        default='./checkpoints',
                        help="output path")
    parser.add_argument("--savename", type=str, default='Fire_detec_1024_v2.pt', help="save file name")


    args = parser.parse_args()

    return args

def applyPCA(X, numComponents=3):
    #_, _, X = read_img(filename)
    X = X.transpose((1, 2, 0))  # generage image w, h, c
    #print('x shape: ', X.shape)
    newX = np.reshape(X, (-1, X.shape[2]))
    #print('newX.shape:', newX.shape)
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))  # reduction image w, h, 3
    #print('applypca: ', newX.shape)
    return newX


class Custodataload(Dataset):

    def __init__(self,filepath, transform):
        super(Custodataload, self).__init__()
        self.transform = transform


        img_path = np.load(filepath, allow_pickle=True)


        self.train = img_path

        print('data length is: ', len(self.train))
        print('loadimg done!')

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        imgfile = self.train[idx]
        img_mat = gdal.Open(imgfile).ReadAsArray()
        if 'normal' in imgfile:
            label = 1
        else:
            label = 0
        #print(img_mat.shape)
        if img_mat.shape[2] == 3:
            #print('img_mat shape: ', img_mat.shape)
            #img = img_mat.transpose((2, 0, 1))
            img = img_mat
            img = Image.fromarray(img)

        elif img_mat.shape[0] == 3:
            #print('img_mat shape: ', img_mat.shape)
            img = img_mat.transpose((1, 2, 0))
            #img = img_mat
            img = Image.fromarray(img)


        else:
            img = applyPCA(img_mat)
            # print('pca imgshape: ', img.shape)
            img = Image.fromarray(np.uint8(img * 255))

        img = self.transform(img)

        return img, label

def Disaster2(filepath):
    transform = transforms.Compose([
        # transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        # transforms.RandomRotation(30),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    # dataset for train
    traindataset = Custodataload(filepath=filepath, transform=transform, )

    # dataset for eval
    evaldataset = traindataset


    return evaldataset


def evaluate_1(data_loader, mainmodel, device):

    mainmodel.eval()
    mainmodel.cuda()

    prob_arr = []
    for step, batch in enumerate(data_loader):
        images = batch[0]

        images = images.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = mainmodel(images)



        prob_out = torch.nn.functional.softmax(output, dim=1)

        print(prob_out)

        prob_arr.extend(prob_out.detach().cpu().numpy())

    #np.save('fire_prob.npy', prob_arr)

    return prob_arr




def get_prob(model):

    ## predicted image paths
    filepath = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/fire.jpeg'

    ## saved prob file
    savename = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/fire_reslt.jpeg'


    temp_val_dataset = Disaster2(filepath)


    sampler_val = torch.utils.data.SequentialSampler(temp_val_dataset)


    data_loader_val = torch.utils.data.DataLoader(
        temp_val_dataset, sampler=sampler_val,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        shuffle=False

    )


    eval_prob = evaluate_1(data_loader_val, model, device)

    np.save(savename,eval_prob)



def eval_process(args):
    n_gpu = 1
    mode = 'eval'

    if mode=='eval':
        args.load = False


    model = DTransformer(args)
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # trained model weight
    loadmodel = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/checkpoints/Fire_detec_1024_v2.pt'
    model_path = os.path.join(args.output_path, loadmodel)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    # torch.save(model.state_dict(), model_path)

    # eval
    logging.info('**************************** start to evaluate *******************************')
    model.eval()

    ## get each label prob
    get_prob(model)


if __name__ == '__main__':
    args = argsoutput()
    eval_process(args)




