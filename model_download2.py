import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('Gurveer05/blip-image-captioning-base-rscid-finetuned', cache_dir='/home/mars/cyh_ws/LLM/models', revision='master')
