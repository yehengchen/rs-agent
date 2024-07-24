import torch
import gradio as gr
from gradio import ChatMessage
from threading import Thread
from transformers.agents import ReactCodeAgent, agent_types
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from rschat import RSChat
from rschat import *
from rschat import ObjectDetection
import numpy as np
import cv2
from RStask import ImageEdgeFunction, CaptionFunction, LanduseFunction, LanduseFunction_Unet, DetectionFunction, DetectionFunction_ship, CountingFuncnction, \
    CountingFuncnction_ship, SceneFunction, InstanceFunction, CaptionFunction_RS_BLIP, CaptionFunction3
from typing import Generator


device = "cuda"  # the device to load the model onto


bot_avatar = "/home/mars/cyh_ws/LLM/zjlab.jpg"           # 聊天机器人头像位置
user_avatar = "/home/mars/cyh_ws/LLM/zjlab.jpg"           # 用户头像位置
model_path = "/home/zjlab/Llama-3-8B-Chinese"   # 已下载的模型位置

# 存储全局的历史对话记录，Llama3支持系统prompt，所以这里默认设置！
llama3_chat_history = [
    {"role": "system", "content": "You are a helpful assistant trained by MetaAI! But you are running with DataLearnerAI Code."}
]

# 初始化所有变量，用于载入模型
tokenizer = None
streamer = None
model = None
terminators = None

# def sepia(image):
#     image_path, det_prompt = process_inputs(inputs)
#     updated_image_path = get_new_image_name(image_path, func_name="detection_" + det_prompt.replace(' ', '_'))
#     log_text = DetectionFunction.inference(image_path, det_prompt, updated_image_path)
#     sepia_img = cv2.imread(updated_image_path)
#     # output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return sepia_img

def sepia(input_img):
    #处理图像
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

def init_model():
    """初始化模型，载入本地模型
    """
    global tokenizer, model, streamer, terminators
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )


with gr.Blocks(theme="gstaff/xkcd",
                    css="#box { height: 420px; overflow-y: scroll !important}",
) as demo:
    gr.Markdown(
    """
    # RS-Agent
    copyright@ZJLab

    """)
    # step1: 载入模型
    init_model()
    gr.Interface(sepia, inputs=gr.Image(), outputs="image")
    
    # captioner
    # gr.Interface(fn=captioner,
    #                 inputs=[gr.Image(label="Upload image", type="pil")],
    #                 outputs=[gr.Textbox(label="Caption")],
    #                 title="Image Captioning with BLIP",
    #                 description="Caption any image using the BLIP model",
    #                 allow_flagging="never")

    # step2: 初始化gradio的chatbot应用，并添加按钮等信息
    chatbot = gr.Chatbot(
        height=900,
        avatar_images=(user_avatar, bot_avatar)
    )

    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    # clear_history
    def clear_history():
        global llama3_chat_history
        llama3_chat_history = []

    # respond
    def respond(message, chat_history):

        global llama3_chat_history, tokenizer, model, streamer

        llama3_chat_history.append({"role": "user", "content": message})

        # 使用Llama3自带的聊天模板，格式化对话记录
        history_str = tokenizer.apply_chat_template(
            llama3_chat_history,
            tokenize=False,
            add_generation_prompt=True
        )

        # tokenzier
        inputs = tokenizer(history_str, return_tensors='pt').to(device)

        chat_history.append([message, ""])

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
            num_beams=1,
            do_sample=True,
            top_p=0.8,
            temperature=0.3,
            eos_token_id=terminators
        )

        # 启动线程，用以监控流失输出结果
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            chat_history[-1][1] += new_text
            yield "", chat_history

        llama3_chat_history.append(
            {"role": "assistant", "content": chat_history[-1][1]}
        )

    # 点击clear按钮，触发历史记录clear
    clear.click(clear_history)
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=7860)