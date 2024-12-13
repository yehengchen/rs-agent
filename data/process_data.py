import os
import sys
import re
import uuid
import cv2
import time
import numpy as np
from skimage import io
import warnings
warnings.filterwarnings("ignore")
import logging

def pre_embeding_file(chatbot):
    language = 'Chinese'
    if language == "English":
        message = "Image uploaded successfully, processing, please wait ..."
    else:
        message = "图像上传成功，正在处理中，请耐心等待..."

    return chatbot + [[message, None]]

def applydata_(chatbot):  
    message = "图像描述生成中，请耐心等待..."
    return chatbot + [[message, None]]

def is_use_database(chatbot,use_database):
    if use_database == "是":
        message = "使用知识库中...."
    else:
        message = "取消使用知识库"
    return chatbot + [[message, None]]

def image_format(raw_image):
    folder_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    folder_path = os.path.join('image', folder_name) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    image_filename = os.path.join(folder_path, f"{str(uuid.uuid4())[:8]}.png")
    # image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
    img = io.imread(raw_image)
    width, height = img.shape[1], img.shape[0]
    ratio = min(4096 / width, 4096 / height)

    if ratio < 1:
        width_new, height_new = (round(width * ratio), round(height * ratio))
    else:
        width_new, height_new = width, height
    width_new = int(np.round(width_new / 64.0)) * 64
    height_new = int(np.round(height_new / 64.0)) * 64

    if width_new != width or height_new != height:
        img = cv2.resize(img, (width_new, height_new))
        print(f"======>Auto Resizing Image from {height, width} to {height_new, width_new}...")
    else:
        print(f"======>Auto Renaming Image...")
    io.imsave(image_filename, img.astype(np.uint8))
    return image_filename

def image_list(folder_path):
    file_list = os.listdir(folder_path)
    image_path_list = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        
        if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.tif'):
            image_path_list.append(file_path)

    print(image_path_list)
    return image_path_list

def is_image(path):
    try:
        img = cv2.imread(path)
        is_img = True
    except IOError:
        is_img = False
    
    return is_img

def process_inputs(inputs):
    global image_path, det_prompt
    pattern = r"(^image[^,]*),\s+([^\n]*)\n"
    match = re.search(pattern, inputs)
    
    image_path = inputs.split(",")[0].strip()
    det_prompt = inputs.split(",")[1].strip()

def process_inputs(inputs):
    global image_path, det_prompt
    pattern = r"(^image[^,]*),\s+([^\n]*)\n"
    match = re.search(pattern, inputs)
    try:
        image_path = inputs.split(",")[0].strip()
        det_prompt = inputs.split(",")[1].strip()
    except IndexError as e:
        print(f"Error: {e}")

    path_match = re.search(r'image_path=(image/[\w.]+)', image_path)
    
    if path_match:
        image_path = path_match.group(1)
        logging.debug(f"image_path: {image_path}")
    else:
        logging.debug('No match found')

    if match:
        image_path = match.group(1)
        det_prompt = match.group(2)
        logging.debug(f"image_path: {image_path}, det_prompt: {det_prompt}")
    else:
        logging.debug('no match\n')

    if is_image(image_path):
        image_path = image_path
        logging.debug(f"Valid image found: {image_path}")
    else:
        logging.debug('No image found')
    
    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)
    
    det_prompt = remove_punctuation(det_prompt)

    return image_path, det_prompt

def extract_numbers(input_string):
    numbers = re.findall(r'\d+', input_string)
    if len(numbers) > 0:
        numbers = str(numbers[0])
    else:
        numbers = None

    return numbers

def replace_all_numbers(input_string, replacement):
    result = re.sub(r'\d+', replacement, input_string)

    return result

def input_highlight(prompt):
    sys.stdout.write(Fore.LIGHTBLUE_EX + Style.BRIGHT + prompt + Style.RESET_ALL)
    sys.stdout.flush()
    txt = input()

    return txt

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    
    return decorator

def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}.png'.replace('__', '_')

    return os.path.join(head, new_file_name)