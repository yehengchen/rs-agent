import os
import json
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model_define import DTransformer
import numpy as np
from osgeo import gdal
import cv2

data_transform = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Resize((224, 224)),
    # transforms.RandomRotation(30),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def vit_predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # load image
    # img_path = "/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/zjlab_nearby.png"
    # img_path = "/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/colorado1024/3209.tif"
    img_path = "/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/colorado1024/1526.tif"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # img_pil = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    # print(x.shape)
    img = data_transform(img)
    # expand batch dimension
    # x = torch.squeeze(img, dim=2).to(device)
    # # x = torch.squeeze(0).to(device)
    # print(x.shape)

    img = torch.unsqueeze(img, dim=0).to(device)
    
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = DTransformer(args)
    model.to(device)
    # load model weights
    model_weight_path = "/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/checkpoints/Fire_detec_1024_v2.pt"
    model.load_state_dict(torch.load(model_weight_path, map_location=lambda storage, loc: storage))
    model.eval()

    prob_arr = []
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device, non_blocking=True))).cpu()
        # predict = torch.softmax(output, dim=0)
        predict = torch.nn.functional.softmax(output, dim=0)

        predict_cla = torch.argmax(predict).numpy()
    
    prob_arr.extend(predict.detach().cpu().numpy())
    # npyfilename = 'fire_prob.npy'
    np.save('./fire_prob.npy', prob_arr)
    data = np.load('./fire_prob.npy')


    if predict[predict_cla].numpy() <0.5:
        print_res = "class: {}   prob: {:.3}".format('no match',
                                                     predict[predict_cla].numpy())
    else:
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

    im = Image.open(img_path)
    x = data_transform(im).to(device)

    att_mat = model(x.unsqueeze(0))
    # att_mat = torch.stack(att_mat).squeeze(1)
    att_mat = torch.stack([att_mat]).squeeze(1).to(device)
    att_mat = torch.mean(att_mat, dim=1)

    residual_att = torch.eye(att_mat.size(-1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    
    mask = v[0].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]

    if class_indict[str(predict_cla)] == 'Fire':
        red_mask = Image.new('RGB', im.size, (255, 0, 0))
        red_mask = np.array(red_mask)
        # red_mask = v[0].reshape(grid_size, grid_size).detach().numpy()
        red_mask = cv2.resize(red_mask / red_mask.max(), im.size)
        result = (red_mask * im).astype("uint8")
    
    else:
        red_mask = Image.new('RGB', im.size, (255, 255, 255))
        red_mask = np.array(red_mask)
        # red_mask = v[0].reshape(grid_size, grid_size).detach().numpy()
        red_mask = cv2.resize(red_mask / red_mask.max(), im.size)
        result = (red_mask * im).astype("uint8")


    # print(type(mask))
    # result = (mask * im).astype("uint8")
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(im)
    _ = ax2.imshow(result)

    # probs = torch.nn.Softmax(dim=-1)(logits)
    # top5 = torch.argsort(probs, dim=-1, descending=True)
    print("Prediction Label and Attention Map!\n")
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    
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
    CropSize = 1024

    #重叠率
    RepetitionRate = 0.0
    vit_predict()