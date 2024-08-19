# RS-Chat: Solving Remote Sensing Tasks with LLM and Visual Models
Introduction
----
### Requirements
```
pip install -r requirements.txt
```

### Run
```
# Suported language: English, Chinese

python rschat.py --image_dir ./image/rs-ship.jpg --language English
```

### Updates
- [2024.07.05]: 
	Add Llama3-8b-Instruct support, supporting English.
  
- [2024.07.06]: 
	Update the LLM model to Llama3-8b-Chinese, base ,supporting EN/CH.

- [2024.07.10]: 
  	Update the image captioning model to RS_BLIP, model has been finetuned on the rsicd dataset.

- [2024.07.15]: 
	Add PAN image object detection and counting support.``` {--img_ch} 1 for PAN/Gray, 3 for MS/RGB```

- [2024.07.17]: 
	Add Landuse Segmentation (Unet) support.



### Supported Model
| Model | Hugging Face  | github     |
| :--------: | :--------: | :--------: |
|Llama-3-8B-Instruct | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | [llama3](https://github.com/meta-llama/llama3) |

### Supported Function
| Function |    Description  | Method | Pretrain Dataset     | Model Weights     |
| :--------: | :--------: | :--------: | :--------: | :--------: |
| Image Captioning | Describe the remote sensing image | [BLIP](https://icml.cc/virtual/2022/spotlight/16016) | [BLIP Dataset](https://icml.cc/virtual/2022/spotlight/16016)| [weight(github)](https://github.com/salesforce/BLIP) |
| Scene Classification | Classify the type of scene | [ResNet](https://arxiv.org/abs/1512.03385) | [AID Dataset](http://www.captain-whu.com/project/AID/)|[weight(Google)](https://drive.google.com/file/d/1f-WES6fTGGa5W9BcDPMVhGk3Foc4p9Or/view?usp=drive_link) [weight(Baidu)](https://pan.baidu.com/s/1yNgUQKieZBEJZ0axzN4tiw?pwd=RSGP) |
| Object Detection | Detect RS object from image | [YOLO v5](https://zenodo.org/badge/latestdoi/264818686) | [DOTA](http://captain.whu.edu.cn/DOTAweb)| [weight(Google)](https://drive.google.com/file/d/1Hb7XA6gZxNam8y8nxs2p6EqJ-XaG1o5Y/view?usp=drive_link) [weight(Baidu)](https://pan.baidu.com/s/1XTG-MLxx5_D0OO6M80OP1A?pwd=RSGP) |
| Instance Segmentation | Extract Instance Mask of certain object | [SwinTransformer+UperNet](https://github.com/open-mmlab/mmsegmentation) | [iSAID](https://captain-whu.github.io/iSAID/index)| [weight(Google)](https://drive.google.com/file/d/165jeD0oi6fSpvWrpgfVBbzUOsyHN0xEq/view?usp=drive_link) [weight(Baidu)](https://pan.baidu.com/s/1Tv6BCt68L2deY_wMVZizgg?pwd=RSGP)|
| Landuse Classification | Extract Pixel-wise Landuse Classification | [HRNet](https://github.com/HRNet) | [LoveDA](https://github.com/Junjue-Wang/LoveDA)| [weight(Google)](https://drive.google.com/file/d/1fRyEpb7344S4Y5F2Q4EBO3fXVT4kXaft/view?usp=drive_link) [weight(Baidu)](https://pan.baidu.com/s/1m6yOXbT6cKGqJ64z86u7fQ?pwd=RSGP) |
| Object Counting | Count the number of certain object in an image | [YOLO v5](https://zenodo.org/badge/latestdoi/264818686) | [DOTA](http://captain.whu.edu.cn/DOTAweb)| Same as Object Detection |
| Edge Detection | Extract edge of remote sensing image | Canny |None| None |

 More funtions to be updated~

### Citation

Please cite the repo if you use the data or code in this repo.

```
@article{RS ChatGPT,
	title = {Remote Sensing ChatGPT: Solving Remote Sensing Tasks with ChatGPT and Visual Models},
	shorttitle = {Remote Sensing ChatGPT},
	doi = {10.48550/ARXIV.2401.09083},
	author = {Guo, Haonan and Su, Xin and Wu, Chen and Du, Bo and Zhang, Liangpei and Li, Deren},
	year = {2024},
}

```

## Acknowledgments
- [Visual ChatGPT](https://github.com/microsoft/TaskMatrix)
- [YOLOv5](https://github.com/hukaixuan19970627/yolov5_obb)
- [BLIP](https://github.com/salesforce/BLIP)