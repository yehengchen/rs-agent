import os
import sys
import re
import uuid
import argparse
import inspect
import cv2
import time
import torch
import numpy as np

from skimage import io
from colorama import Fore, Back, Style
from LLM import LLaMA3_LLM, LLaMA3_1_LLM, Qwen2_LLM
# from langchain.chat_models import ChatOpenAI
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.agents import AgentType, AgentExecutor, create_structured_chat_agent

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from RStask.ObjectDetection.models.yolov5s import *
from RStask.LanduseSegmentation.unet import *

from Prefix import RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX, RS_CHATGPT_PREFIX_CN, RS_CHATGPT_FORMAT_INSTRUCTIONS_CN, RS_CHATGPT_SUFFIX_CN, VISUAL_CHATGPT_PREFIX_CN, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN, VISUAL_CHATGPT_SUFFIX_CN
from RStask import ImageEdgeFunction, CaptionFunction, LanduseFunction, LanduseFunction_Unet, DetectionFunction, DetectionFunction_ship, CountingFuncnction, \
    CountingFuncnction_ship, SceneFunction, InstanceFunction, CaptionFunction_RS_BLIP, CaptionFunction3
import gradio as gr
from gradio import ChatMessage
import warnings
warnings.filterwarnings("ignore")

os.environ['GRADIO_TEMP_DIR'] = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/tmp'
os.makedirs('image', exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def image_format(raw_image):
    image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
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
    _, ext = os.path.splitext(path)
    if ext.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.gif'):
        is_img = True
    else:
        is_img = False
    
    try:
        cv2.imread(image_path)
    except:
        is_img = False
        print("Image not found")
    
    return is_img

def process_inputs(inputs):
    global image_path, det_prompt
    pattern = r"(^image[^,]*),\s+([^\n]*)\n"
    match = re.search(pattern, inputs)
    
    image_path = inputs.split(",")[0].strip()
    det_prompt = inputs.split(",")[1].strip()

    path_match = re.search(r'image_path=(image/[\w.]+)', image_path)
    
    
    if path_match:
        image_path = path_match.group(1)
        print(image_path)
    else:
        print('No match found')

    if match:
        image_path = match.group(1)
        det_prompt = match.group(2)
    else:
        print('no match\n')

    if is_image(image_path):
        image_path = image_path
    else:
        print('No image found')

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

# def get_new_image_name(org_img_name, func_name="update"):
#     head_tail = os.path.split(org_img_name)
#     head = head_tail[0]
#     tail = head_tail[1]
#     name_split = tail.split('.')[0].split('_')
#     this_new_uuid = str(uuid.uuid4())[:4]
#     if len(name_split) == 1:
#         most_org_file_name = name_split[0]
#     else:
#         assert len(name_split) == 4
#         most_org_file_name = name_split[3]
#     recent_prev_file_name = name_split[0]
#     new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
#     return os.path.join(head, new_file_name)


def show_image(img_path):
    img = cv2.imread(img_path)
    reimg = cv2.resize(img, (640, 640))
    cv2.imshow('image', reimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class EdgeDetection:
    def __init__(self, device):
        print("Initializing Edge Detection Function")
        self.func = ImageEdgeFunction()

    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the remote sensing image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the  edge of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        global updated_image_path

        updated_image_path = get_new_image_name(inputs, func_name="edge")
        self.func.inference(inputs, updated_image_path) 

        return updated_image_path
    
class ObjectCounting:
    print("Initializing ObjectCounting Function")
    def __init__(self, device):
        self.func = CountingFuncnction(device)

    @prompts(name="Count object",
             description="useful when you want to count the number of the  object in the image. "
                         "like: how many ships are there in the image? or count the number of bridges"
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be counted Must be Arabic numerals")
    
    def inference(self, inputs):
        # image_path, det_prompt = inputs.split(",")
        image_path, det_prompt = process_inputs(inputs)
        log_text = self.func.inference(image_path, det_prompt)
        # cv2.putText(image_path, log_text, (640, 640))
        # show_image(image_path)
        global count_num
        count_num = extract_numbers(log_text)

        return log_text


class InstanceSegmentation:
    def __init__(self, device):
        print("Initializing InstanceSegmentation")
        self.func = InstanceFunction(device)

    @prompts(name="Instance Segmentation for Remote Sensing Image",
             description="useful when you want to apply man-made instance segmentation for the image. The expected input category include plane, ship or ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle, helicopter, roundabout, soccer ball field, and swimming pool."
                         "like: extract ship from this image, "
                         "or predict the ship in this image, or extract tennis court from this image, segment harbor from this image, Extract the vehicle in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from plane, or ship, or storage tank, or baseball diamond, or tennis court, or basketball court, or ground track field, or harbor, or bridge, or vehicle, or helicopter, or roundabout, or soccer ball field, or  swimming pool. "
            
            # description="当您想要对图像进行人造实例分割时，这个工具非常有用。预期的输入类别包括飞机、船/船只、存储罐、棒球场、网球场、篮球场、田径场、港口、桥梁、车辆、直升机、环形交叉路口、足球场和游泳池。"
            #             "例如：从这张图像提取船，"
            #             "或者预测这张图像中的船只，"
            #             "或者从这张图像提取网球场，"
            #             "或者分割这张图像中的港口，"
            #             "从这张图像提取车辆。"
            #             "这个工具的输入应该是一个逗号分隔的字符串，代表图像路径和类别文本，"
            #             "从飞机、船只、存储罐、棒球场、网球场、篮球场、田径场、港口、桥梁、车辆、直升机、环形交叉路口、足球场和游泳池中选择。"

            )
    def inference(self, inputs):
        global updated_image_path
       
        # image_path, det_prompt = inputs.split(",")
        image_path, det_prompt = process_inputs(inputs)
        updated_image_path = get_new_image_name(image_path, func_name="instance_" + det_prompt)
        text = self.func.inference(image_path, det_prompt, updated_image_path)

        # text = "Category do not suuport. Please try again."

        return text

class SceneClassification:
    def __init__(self, device):
        print("Initializing SceneClassification")
        self.func = SceneFunction(device)

    @prompts(name="Scene Classification for Remote Sensing Image",
             description="useful when you want to know the type of scene or function for the image. "
                         "like: what is the category of this image?, "
                         "or classify the scene of this image, or predict the scene category of this image, or what is the function of this image. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, inputs):
        # output_txt = self.func.inference(inputs)
        output_txt = self.func.inference(inputs.split('\n')[0])
        return output_txt

class LandUseSegmentation:
    def __init__(self, device):
        print("Initializing LandUseSegmentation")
        self.func = LanduseFunction_Unet(device)

    @prompts(name="Land Use Segmentation for Remote Sensing Image",
            description="useful when you want to apply land use gegmentation for the image. The expected input category include Building, Road, Water, Barren, Forest, Farmland, Landuse."
                         "like: generate landuse map from this image, "
                         "or predict the landuse on this image, or extract building from this image, segment roads from this image, Extract the water bodies in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from Lnad Use, or Building, or Road, or Water, or Barren, or Forest, or Farmland, or Landuse."
    
            # description="当您想要对图像进行土地使用分割时，这个工具非常有用。预期的输入类别包括建筑、道路、水、荒地、森林、农田和土地使用。"
            #             "例如：从这张图像生成土地使用图，"
            #             "或者在这张图像上预测土地使用，"
            #             "或者从这张图像提取建筑，从这张图像分割道路，"
            #             "从这张图像提取水体。"
            #             "这个工具的输入应该是一个逗号分隔的字符串，代表图像路径和类别文本，"
            #             "从土地使用、建筑、道路、水、荒地、森林、农田和土地使用中选择。"
        )
    def inference(self, inputs):
        global updated_image_path


        image_path, det_prompt = process_inputs(inputs)
        # image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="landuse")

        text = self.func.inference(image_path, det_prompt, updated_image_path)
        
        # img = cv2.imread(image_path)
        # mask = cv2.imread(updated_image_path)
        # result = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        # cv2.imwrite(updated_image_path, result)

        # cv2.imshow('mask_rslt', result)

        return text

class ObjectDetection:
    def __init__(self, device):
        print("Initializing ObjectDetection")
        self.func = DetectionFunction(device)
             
    @prompts(name="Detect the given object",
             description="useful when you only want to detect the bounding box of the certain objects in the picture according to the given text."
                         "like: detect the ship, or can you locate an object for me."
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found"
    
            # description="当您只想根据给定的文本检测图片中特定对象的范围框时，这个工具非常有用。"
            #             "例如：检测船只，或者定位一个目标。"
            #             "这个工具的输入应该是一个逗号分隔的字符串，代表图像路径和要找到的对象的文本描述。"
            )
    def inference(self, inputs):
        global updated_image_path

        image_path, det_prompt = process_inputs(inputs)
        # image_path, det_prompt = inputs.split(",")

        updated_image_path = get_new_image_name(image_path, func_name="detection_" + det_prompt.replace(' ', '_'))
        log_text = self.func.inference(image_path, det_prompt, updated_image_path)


        return log_text

class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.func = CaptionFunction_RS_BLIP(device)

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        captions = self.func.inference(image_path.split('\n')[0])
        # captions = 'A satellite image of many ships in the port of Singapore.'
        # print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")

        return captions

class RSChat:
    def __init__(self, load_dict):
        print(f"Initializing RSChat, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for RSChat")
        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))

        # self.llm = ChatOpenAI(api_key=openai_key, base_url=proxy_url, model_name=gpt_name,temperature=0)
        # self.llm = LLaMA3_LLM(mode_name_or_path="/home/zjlab/Meta-Llama-3-8B-Instruct")
        # self.llm = LLaMA3_LLM(mode_name_or_path="/home/zjlab/Llama-3-8B-Chinese")
        self.llm = Qwen2_LLM(mode_name_or_path="/home/mars/cyh_ws/LLM/models/Qwen2-7B-Instruct")
        # self.llm = LLaMA3_1_LLM(mode_name_or_path="/home/mars/cyh_ws/LLM/models/Llama3.1-8B-Chinese-Chat")
        # self.llm = LLaMA3_LLM(mode_name_or_path="/home/zjlab/llama3___1-8b-instruct-dpo-zh")

        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output', return_messages=True)

    def initialize(self, language):
        self.memory.clear()  # clear previous history
        if language == 'English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX           
            place = 'Enter text and press enter, or upload an image'
            label_clear = 'Clear'
            label_submit = 'Submit'

        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_CHATGPT_PREFIX_CN, RS_CHATGPT_FORMAT_INSTRUCTIONS_CN, RS_CHATGPT_SUFFIX_CN
            # PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX_CN, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN,  VISUAL_CHATGPT_SUFFIX_CN
            place = '输入文字并回车，'
            label_clear = '清除'
            label_submit = '提交'

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True, 
            max_iter=5,
            allow_delegation=False,
            stop=["\nObservation:", "\n\tObservation:"],
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX,}, 
            handle_parsing_errors=True)

        return gr.update(visible=True), gr.update(visible=False), gr.update(
                placeholder=place), gr.update(value=label_clear)
    
    def run_no_image(self, text, state):
        try:
            res = self.agent({"input": text.strip()})
            res['output'] = res['output'].replace("\\", "/")
            res['intermediate_steps'] = 'No'
            # print('2!#@!#!@res', res)
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
        
        response = res['output']
        state = state + [(text, response)]
        return state
    
    def run_text(self, text, state):
        try:
            res = self.agent({"input": text.strip()})
            res['output'] = res['output'].replace("\\", "/")

        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
        
        # print("input text:",text, "\ncurrent state:",state)
        # res = self.agent({"input": text.strip()})
        # res['output'] = res['output'].replace("\\", "/")
        intermediate_step_list = [[], [], []]
        observations = []
        re_pattern = re.compile(
            r"Thought:\s*(?P<thought>.*?)\n"
            r"Action:\s*(?P<action>.*?)\n"
            r"Action Input:\s*(?P<action_input>.*)"
        )
        for i, step in enumerate(res['intermediate_steps']):
            matched = re_pattern.search(step[0].log)
            if matched:
                thought = matched.group("thought")
                action = matched.group("action")
                action_input = matched.group("action_input")
                intermediate_step_list[0].append("🦜Thought: " + thought)
                intermediate_step_list[1].append("🛠️Action: " + action)
                intermediate_step_list[2].append("🛠️Action Input: " + action_input)
                observations.append(res['intermediate_steps'][i][1])
            else:
                continue

        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        # print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
        #       f"Current Memory: {self.agent.memory.buffer}")
        return state, intermediate_step_list, observations

    def run_image(self, image_dir, language, state, txt=None):
        folder_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        folder_path = os.path.join('image', folder_name) 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        image_filename = os.path.join(folder_path, f"{str(uuid.uuid4())[:8]}.png")
        img = io.imread(image_dir)
        width, height = img.shape[1],img.shape[0]
        ratio = min(4096 / width, 4096 / height)

        if ratio < 1:
            width_new, height_new = (round(width * ratio), round(height * ratio))
        else:
            width_new, height_new =width,height 
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        
        if width_new!=width or height_new!=height:
            img = cv2.resize(img,(width_new, height_new))
            print(f"======>Auto Resizing Image from {height,width} to {height_new,width_new}...")
        else:
            print(f"======>Auto Renaming Image...")
        io.imsave(image_filename, img.astype(np.uint8))
        description = self.models['ImageCaptioning'].inference(image_filename)
        
        if language == 'English':
            Human_prompt = f' Provide a remote sensing image named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
            AI_prompt = "Received."
        else:
            Human_prompt = f' 提供一张遥感图片名为 {image_filename} 。它的英文描述是: {description}。 这些信息帮助你理解这个图像，但是你应该使用工具来完成以下的任务，而不是直接从英文描述中想象。 如果你明白了, 说 \"收到\". \n'
            AI_prompt = "收到。"

        self.memory.chat_memory.add_user_message(Human_prompt)
        self.memory.chat_memory.add_ai_message(AI_prompt)
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        
        # if language == 'English':
        #     print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\nCurrent Memory: {self.agent.memory.buffer}")
        # else:
        #     print(f"\n正在处理图像: {image_filename}\n当前状态: {state}\n当前记忆: {self.agent.memory.buffer}")
        
        # print(img.shape)
        state = self.run_text(f'{txt} {image_filename} ', state)

        return state
    
    def run_image_gradio(self, image, language, state=[], txt=None):
        # description = None
        
        global description
        description = self.models['ImageCaptioning'].inference(image)
        print(description)
        if language == 'English':
            Human_prompt = f' Provide a remote sensing image named {image}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
            AI_prompt = "Received."
        else:
            Human_prompt = f'\nHuman: 提供一张名为 {image}的图片。它的描述是: {description}。 这些信息帮助你理解这个图像，但是你应该使用工具来完成下面的任务，而不是直接从我的描述中想象。 如果你明白了, 说 \"收到\". \n'
            AI_prompt = "收到。"
        
        self.memory.chat_memory.add_user_message(Human_prompt)
        self.memory.chat_memory.add_ai_message(AI_prompt)

        # state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]

        state = state + [(f"![](file={image})*{image}*", AI_prompt)]
        if language == 'English':
            print(
                f"\nProcessed run_image, Input image: {image}\nCurrent state: {state}\nCurrent Memory: {self.agent.memory.buffer}")
        else:
            print(f"\n正在处理图像: {image}\n当前状态: {state}\n当前记忆: {self.agent.memory.buffer}")
        state, intermediate_step_list, observations = self.run_text(f'{txt} {image} ', state)
        return state, intermediate_step_list, observations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_key', type=str, required=False)
    parser.add_argument('--language', type=str, default='Chinese',choices=['English','Chinese'])
    parser.add_argument('--load', type=str,
                        help='Image Captioning is basic models that is required. You can select from [ImageCaptioning,ObjectDetection,LandUseSegmentation,InstanceSegmentation,ObjectCounting,SceneClassification,EdgeDetection]',
                        default="ImageCaptioning_cuda:0,SceneClassification_cuda:0,ObjectDetection_cuda:0,LandUseSegmentation_cuda:0,InstanceSegmentation_cuda:0,ObjectCounting_cuda:0,EdgeDetection_cpu")
    args = parser.parse_args()

    language = args.language
    count_num = ""
    updated_image_path = ""
    det_prompt = ""
    image_path = ""
    state = []

    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    agent = RSChat(load_dict=load_dict)
    agent.initialize(args.language)

    bot_avatar = "/home/mars/cyh_ws/LLM/robot_ikon.png"
    user_avatar = "/home/mars/cyh_ws/LLM/zjlab.jpg"   
    folder_path = "/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/app_img/"
    examples_img = image_list(folder_path)
    root_path = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/'

    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px} img {max-height: 100% !important}", theme = gr.themes.Default(text_size='lg')) as demo:


        gr.Markdown(
        """

        ![Imgur](https://www.zhejianglab.org/static/img/white_logo.3926fa67.png)

        <div align=center>

        # 🤖 RS-Agent 

        </div>
        
        """)

        with gr.Accordion("Image Prosessor"):
        
            with gr.Tab("Image Captioning"):      
                caption = gr.Interface(fn=CaptionFunction_RS_BLIP(device='cpu').inference_app,
                            inputs=gr.Image(label="Upload image", type="filepath"),
                            outputs=[gr.Textbox(label="Image Caption")],
                            title="Image Captioning",
                            description="Caption any image using the BLIP model",
                            allow_flagging="never",
                            examples=examples_img
                            )
            
            with gr.Tab("Object Detection"):

                gr.Interface(fn=DetectionFunction(device='cpu').inference_app,
                            inputs=[gr.Image(label="Upload image", type="filepath")],
                            outputs=[gr.Image(label="Object Detection", type="filepath"), gr.Textbox(label="Imgae Shape", type="text")],
                            title="Object Detection",
                            description="Object Detection remote-sensing image using the YOLOv5 model",
                            allow_flagging="never",
                            examples=examples_img
                            )
                
            with gr.Tab("Instance Segmentation"):

                gr.Interface(fn=InstanceFunction(device='cpu').inference_app,
                            inputs=gr.Image(label="Upload image", type="filepath"),
                            outputs=[gr.Image(label="Instance Segmentation", type="filepath"), gr.Textbox(label="Imgae Shape", type="text")],
                            title="Instance Segmentation",
                            description="Instance Segmentation remote-sensing image using the swint model",
                            allow_flagging="never",
                            examples=examples_img
                            )
                
            with gr.Tab("Landuse Segmentation"):
                
                gr.Interface(fn=LanduseFunction(device='cpu').inference_app,
                            inputs=gr.Image(label="Upload image", type="filepath"),
                            outputs=[gr.Image(label="Landuse Segmentation", type="filepath"), gr.Textbox(label="Imgae Shape", type="text")],
                            title="Landuse Segmentation",
                            description="Landuse Segmentation remote-sensing image using the Unet model",
                            allow_flagging="never",
                            examples=examples_img
                        )
            
        with gr.Accordion("RS-Agent with Llama3"):
            # lang = gr.Radio(choices=['Chinese', 'English'], value=None,label='Language')
            image_input = gr.State(None)
            processed_image_output = gr.State(None)
            # image_input = gr.State(None)
            # processed_image_output = gr.State(None)
            # chatbot = gr.State(None)
            # imagebox = textbox = gr.Textbox()
            
            examples_img_list = [i.split(',') for i in examples_img]
            print('dsadasdas',examples_img_list)
            with gr.Row():

                with gr.Column(scale=1): 
                    image_input = gr.Image(type="filepath", label="Upload image")
                    examples = gr.Examples(examples_img_list, image_input)
                    # examples=image_list(folder_path)
                with gr.Column(scale=1):
                    # chatbot_display = chatbot
                    processed_image_output = gr.Image(type="filepath", label="处理后的图片")
            
            with gr.Row():
                chatbot = gr.Chatbot(
                    value=[[None, "你好，我是【之江天绘】智能遥感图像小助手🤖，有什么我可以帮助你的吗？请先上传图像🖼️！"]],
                    placeholder="<strong>🤖 RS-Agent</strong><br> assist with a wide range of remote sensing image related tasks",
                    label="RS-Agent",
                    height=600,
                    avatar_images=(user_avatar, bot_avatar),
                    scale=3,
                    layout="bubble",
                )
            
            with gr.Row() as input_raw:

                # btn = gr.UploadButton(label="🖼️",file_types=["image"], scale=0.05)
                msg = gr.Textbox(lines=1, label="Chat Message", placeholder="输入您的指令, 可通过Shift+回车⏎换行", scale=3.95)
                submit_button = gr.Button("Submit")
                clear = gr.Button("Clear")
            
            def user(user_message, history):
                user_message = user_message.replace("\n", "")
                return user_message, history + [[user_message, None]]


            def bot(image_input, message, history):
                history[-1][1] = ""
                out_img = None
                global updated_image_path
                if image_input is not None:
                    # history[-1][1] = image_input
                    processed_image = image_format(image_input)
                    current_state, thought_process, observations = agent.run_image_gradio(processed_image, args.language, [], message)
                    observation_pattern = re.compile(r".*?(image.*?\.png)")
                    for j in range(len(thought_process[0])):
                        for i in range(3):
                            history[-1][1] = history[-1][1] + thought_process[i][j] + "\n"
                        if j < len(observations):
                            history[-1][1] = history[-1][1] + "💭Observation " + str(j) + ": " + observations[j] + "\n"
                            # print('#########', observations[j])
                            match = observation_pattern.search(observations[j])
                            if match:
                                out_img = match.group(1)
                        history += [[None, None]]
                        history[-1][1] = ''
                    
                    if is_image(updated_image_path):
                        out_img = updated_image_path
                    else:
                        out_img = None

                    final_thought = "\n" + "🧠Agent Response:\n" + current_state[1][1]
                    # print(final_thought)
                    # print(history[-1][1])  
                    
                    # if count_num != None:
                    #     final_thought = replace_all_numbers(final_thought, count_num) 
                    #     print(f'\033[1m\033[36m {final_thought}\033[0m')
                    # else:
                    #     print(f'\033[1m\033[36m {final_thought}\033[0m')

                    history[-1][1] = final_thought
                    updated_image_path = ""
                else:
                    # gr.Warning("Warning! Please upload an image first.", duration=5)
                    response = agent.run_no_image(message, history)
                    # print('!@#!#!@#sate', response)
                    history[-1][1] = response[-1][1] + '\n请先上传图像🖼️，再对图像进行提问!'
                    out_img = None
                return history, out_img


            def clear_uploaded_image():
                image_input.value = ''
                processed_image_output.value = ''
                return
            # lang.change(agent.initialize, [lang], [input_raw, lang, msg, clear])
            
            # btn.upload(handle_image_upload, [image_input, state, msg], [uploaded_image, chatbot, state, msg])

            # msg.submit(user, [image_input, chatbot], [image_input, chatbot], queue=False)
            
            response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [image_input, msg, chatbot], [chatbot, processed_image_output]
            )
            msg.submit(lambda: "", None, msg)
            # response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
            


            submit_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [image_input, msg, chatbot], [chatbot, processed_image_output])
            submit_button.click(agent.memory.clear)
            submit_button.click(lambda: "", None, msg)
            # image_input_show = '![]' +image_input
            image_input.upload(user, [image_input, chatbot], [image_input, chatbot], queue=False).then(bot, [image_input, msg, chatbot], [chatbot, processed_image_output])
            
            clear.click(lambda: None, None, chatbot, queue=False)
            clear.click(agent.memory.clear)
            # clear.click(lambda: [], None, state)
            clear.click(lambda: None, None, image_input, queue=False)
            clear.click(lambda: None, None, processed_image_output, queue=False)
            # clear.click(lambda: None, None, image_path, queue=False)
            clear.click(lambda:None, "", updated_image_path)
            clear.click(clear_uploaded_image, None)
            clear.click(lambda: None, "", det_prompt, queue=False)
            # init.click(lambda: None, None, agent.initialize(args.language), queue=False)
    
    gr.close_all()
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0", server_port=8080)