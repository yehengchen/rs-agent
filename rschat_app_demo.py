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
import yaml
import random
import pandas as pd
# import PyPDF2
from tqdm import tqdm
from skimage import io
import warnings
warnings.filterwarnings("ignore")
import logging

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
# from langchain.vectorstores import Milvus
from langchain_community.vectorstores import Milvus

from RStask.ObjectDetection.models.yolov5s import *
from RStask.LanduseSegmentation.unet import *
from RStask.FireDetection.fire_det import *

from Prefix import RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX, RS_CHATGPT_PREFIX_CN, RS_CHATGPT_FORMAT_INSTRUCTIONS_CN, RS_CHATGPT_SUFFIX_CN, VISUAL_CHATGPT_PREFIX_CN, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN, VISUAL_CHATGPT_SUFFIX_CN
from RStask import ImageEdgeFunction, CaptionFunction, LanduseFunction, LanduseFunction_Unet, DetectionFunction, DetectionFunction_ship, CountingFuncnction, \
    CountingFuncnction_ship, SceneFunction, InstanceFunction, CaptionFunction_RS_BLIP, CaptionFunction3, FireFunction
import gradio as gr

from process_data import *

os.environ['GRADIO_TEMP_DIR'] = '/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/tmp'
os.makedirs('image', exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class EdgeDetection:
    def __init__(self, device):
        print("Initializing Edge Detection Function")
        self.func = ImageEdgeFunction()

    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the remote sensing image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the  edge of this image. "
                         "The input to this tool should be a string, representing the image_path"
            )
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
                         "representing the image_path, the text description of the object to be counted Must be Arabic numerals"
            )
    
    def inference(self, inputs):
        # image_path, det_prompt = inputs.split(",")
        image_path, det_prompt = process_inputs(inputs)
        log_text = self.func.inference(image_path, det_prompt)
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
            )
    def inference(self, inputs):
        global updated_image_path
        # image_path, det_prompt = inputs.split(",")
        image_path, det_prompt = process_inputs(inputs)
        updated_image_path = get_new_image_name(image_path, func_name="instance_" + det_prompt)
        text = self.func.inference(image_path, det_prompt, updated_image_path)

        return text

class SceneClassification:
    def __init__(self, device):
        print("Initializing SceneClassification")
        self.func = SceneFunction(device)

    @prompts(name="Scene Classification for Remote Sensing Image",
             description="useful when you want to know the type of scene or function for the image. "
                         "like: what is the category of this image? "
                         "or classify the scene of this image, or predict the scene category of this image, or what is the function of this image. "
                         "The input to this tool should be a string, representing the image_path. "
            )
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

            )
    def inference(self, inputs):
        global updated_image_path
        image_path, det_prompt = process_inputs(inputs)
        # image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="landuse")
        text = self.func.inference(image_path, det_prompt, updated_image_path)

        return text

class ObjectDetection:
    def __init__(self, device):
        print("Initializing ObjectDetection")
        self.func = DetectionFunction(device)
             
    @prompts(name="Detect the given object",
             description="useful when you only want to detect the bounding box of the certain objects in the picture according to the given text."
                         "like: detect the ship, or can you locate an object for me. how many plane are there in the image? or count the number of bridges."
                         "The input to this tool should be a comma separated string of two,"
                         "representing the image_path, the text description of the object to be found"
            )

    def inference(self, inputs):
        global updated_image_path
        image_path, det_prompt = process_inputs(inputs)
        # image_path, det_prompt = inputs.split(",")

        updated_image_path = get_new_image_name(image_path, func_name="detection_" + det_prompt.replace(' ', '_'))
        log_text = self.func.inference(image_path, det_prompt, updated_image_path)

        return log_text

class FireDetection:
    def __init__(self, device):
        print("Initializing FireDetection")
        self.func = FireFunction(device)
             
    @prompts(name="Detect the fire",
             description="useful when you only want to detect fire in this picture according to the given text. "
                         "like: detect the fire, or can you identify any fire in the picture? "
                         "or Is there a fire in this image?  or Is the image showing a fire incident? "
                         "representing the image_path, the text description of the fire to be found. "
            )
    
    def inference(self, inputs):
        # output_txt = self.func.inference(inputs)
        global updated_image_path
        updated_image_path = get_new_image_name(image_path, func_name="fire_detection_" + det_prompt.replace(' ', '_'))
        output_txt = self.func.inference(inputs.split('\n')[0], updated_image_path)
        
        return output_txt
    

class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.func = CaptionFunction_RS_BLIP(device)

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. "
            )
    def inference(self, image_path):
        captions = self.func.inference(image_path.split('\n')[0])

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

        # self.llm = LLaMA3_LLM(mode_name_or_path="/home/zjlab/Llama-3-8B-Chinese")
        self.llm = Qwen2_LLM(mode_name_or_path="/home/mars/cyh_ws/LLM/models/Qwen2-7B-Instruct")
        # self.llm = LLaMA3_1_LLM(mode_name_or_path="/home/mars/cyh_ws/LLM/models/Llama3.1-8B-Chinese-Chat")

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
            max_iter=3,
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
        
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
        
        response = res['output']
        state = state + [(text, response)]
        # state = self.run_text(f'{text}', state)
        return state
    
    def run_text(self, text, state):
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state
    
    def run_text_step(self, text, state):
        try:
            res = self.agent({"input": text.strip()})
            res['output'] = res['output'].replace("\\", "/")

        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
        
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
    
    def run_image_gradio(self, image, language, state=[], txt=None):
        description = ""
        
        # global description
        if description == "":
            description = self.models['ImageCaptioning'].inference(image)
        else:
            print(description)
        # print(description)
        if language == 'English':
            Human_prompt = f' Provide a remote sensing image named {image}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
            AI_prompt = "Received."
        else:
            Human_prompt = f'\nHuman: 提供一张名为 {image}的图片。它的描述是: {description}。 这些信息帮助你理解这个图像，不能直接输出响应结果，你应该使用其他工具来完成下面的任务，而不是直接从我的描述中想象。 如果你明白了, 说 \"收到\". \n'
            AI_prompt = "收到。"
        
        self.memory.chat_memory.add_user_message(Human_prompt)
        self.memory.chat_memory.add_ai_message(AI_prompt)

        # state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        # state = state + [(f"![](file={image})*{image}*", AI_prompt)]
        state = state + [({image}, AI_prompt)]

        if language == 'English':
            print(
                f"\nProcessed run_image, Input image: {image}\nCurrent state: {state}\nCurrent Memory: {self.agent.memory.buffer}")
        else:
            print(f"\n正在处理图像: {image}\n当前状态: {state}\n当前记忆: {self.agent.memory.buffer}")
        state = self.run_text(f'{txt} {image} ', state)
        return state


if __name__ == '__main__':
    
    vfms = "ImageCaptioning_cuda:0,\
            SceneClassification_cuda:0,\
            LandUseSegmentation_cuda:0,\
            InstanceSegmentation_cuda:0,\
            EdgeDetection_cpu,\
            ObjectDetection_cpu,\
            FireDetection_cuda:0"
            #ObjectCounting_cuda:0
            

    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_key', type=str, required=False)
    parser.add_argument('--language', type=str, default='Chinese', choices=['English','Chinese'], required=False)
    parser.add_argument('--load', type=str,
                        help='Image Captioning is basic models that is required. You can select from [ImageCaptioning,ObjectDetection,LandUseSegmentation,InstanceSegmentation,ObjectCounting,SceneClassification,EdgeDetection,FireDetection]',
                        default=vfms) 
    
    args = parser.parse_args()
    language = args.language
    
    count_num = ""
    updated_image_path = ""
    det_prompt = ""
    image_path = ""
    state =  gr.State([])
    image_input = gr.State(None)
    processed_image_output = gr.State(None)
    processed_image_state = gr.State(None)
    embedding_model_state = gr.State()
    milvus_books = None
    milvus_books_state = gr.State(milvus_books)
    trash = gr.State()

    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    agent = RSChat(load_dict=load_dict)
    agent.initialize(args.language)

    bot_avatar = "/home/mars/cyh_ws/LLM/robot_ikon.png"
    user_avatar = "/home/mars/cyh_ws/LLM/Remote-Sensing-Chat/image/assistant.png"
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

        with gr.Accordion("Image Prosessor", open=False):
        
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

            with gr.Tab("Fire Detection"):
                
                gr.Interface(fn=FireFunction(device='cpu').inference_app,
                            inputs=gr.Image(label="Upload image", type="filepath"),
                            outputs=[gr.Image(label="Fire Detection", type="filepath"), gr.Textbox(label="Imgae Shape", type="text")],
                            title="Fire Detection",
                            description="Fire Detection remote-sensing image using the Vit model",
                            allow_flagging="never",
                            examples=examples_img
                        )
                

        with gr.Accordion("RS-Agent with LLM"):

            # lang = gr.Radio(choices=['Chinese', 'English'], value=None,label='Language')
            
            examples_img_list = [i.split(',') for i in examples_img]
            with gr.Row():
                with gr.Column(scale=1): 
                    image_input = gr.Image(type="filepath", label="上传图像")
                    submit_img_button = gr.Button("上传图像")
                                        
                with gr.Column(scale=1):
                    processed_image_output = gr.Image(type="filepath", label="图像解析")
                    # processed_image_state = gr.Textbox(image_input, label="处理后的图像信息")
                    
            examples = gr.Examples(examples_img_list, image_input, label="遥感图像示例")

            with gr.Row():
                chatbot = gr.Chatbot(
                    value=[[None, "你好，我是【之江天绘】智能遥感图像助手🤖，有什么我可以帮助你的吗？可以先上传图片🖼️，再进行提问！"]],
                    placeholder="<strong>🤖 RS-Agent</strong><br> assist with a wide range of remote sensing image related tasks",
                    label="RS-Agent",
                    height=600,
                    avatar_images=(user_avatar, bot_avatar),
                    scale=3,
                    layout="bubble",
                )
            
            with gr.Row() as input_raw:
                # btn = gr.UploadButton(label="🖼️",file_types=["image"], scale=0.05)
                msg = gr.Textbox(interactive=True, lines=1, show_label=False, placeholder="输入您的指令, 可通过Shift+回车⏎换行", scale=3.95)
                submit_button = gr.Button("发送")
                clear = gr.Button("刷新")
            
            def user(user_message, history):
                user_message = user_message.replace("\n", "")
                return user_message, history + [[user_message, None]]


            def bot(image_input, message, history):
                history[-1][1] = ""
                out_img = None
                global updated_image_path
                # global processed_image
                if image_input:
                    # history[-1][1] = image_input
                    global processed_image
                    processed_image = image_format(image_input)
                    current_state = agent.run_image_gradio(processed_image, args.language, [], message)                    
                  
                    if is_image(updated_image_path):
                        out_img = updated_image_path
                    else:
                        out_img = ""

                    # final_thought = "\n" + "🧠之江天绘遥感智能体:\n" + current_state[1][1]
                    final_thought = "\n" + current_state[1][1]

                    history[-1][1] = final_thought
                    updated_image_path = ""
                    if out_img == "":
                        out_img = processed_image
                else:
                    # gr.Warning("Warning! Please upload an image first.", duration=5)
                    response = agent.run_no_image(message, history)
                    history[-1][1] = response[-1][1] + '\n`💡提示：可以上传图片🖼️，并对图片进行提问🤔，也可以让我介绍一下功能`'
                    out_img = None
            
                return history, out_img


            response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [image_input, msg, chatbot], [chatbot, processed_image_output]
            )
            msg.submit(lambda: "", None, msg)
            # response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)
            
            submit_img_button.click(pre_embeding_file, inputs=[chatbot], outputs=[chatbot]).then(
                bot, [image_input, msg, chatbot], [chatbot, processed_image_output])
            submit_img_button.click(lambda: "", None, msg)
            submit_img_button.click(agent.memory.clear)

            submit_button.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [image_input, msg, chatbot], [chatbot, processed_image_output]   
            )     
            
            submit_button.click(agent.memory.clear)
            submit_button.click(lambda: "", None, msg)
            
            clear.click(agent.memory.clear)
            clear.click(lambda: None, None, chatbot, queue=False)
            clear.click(lambda: None, None, image_input, queue=False)
            clear.click(lambda: None, None, processed_image_output, queue=False)
            clear.click(lambda:None, "", updated_image_path)
            clear.click(lambda: None, "", det_prompt, queue=False)
    
    gr.close_all()
    demo.queue()
    demo.launch(share=True, server_name="0.0.0.0", server_port=8088)