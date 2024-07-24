import torch
import gradio as gr
from gradio import ChatMessage
from threading import Thread
from transformers.agents import ReactCodeAgent, agent_types
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
# from rschat import RSChat
from rschat import *
# from rschat import ObjectDetection
from LLM import LLaMA3_LLM
import numpy as np
import cv2
import inspect
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from RStask import ImageEdgeFunction, CaptionFunction, LanduseFunction, LanduseFunction_Unet, DetectionFunction, DetectionFunction_ship, CountingFuncnction, \
    CountingFuncnction_ship, SceneFunction, InstanceFunction, CaptionFunction_RS_BLIP, CaptionFunction3
from Prefix import RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX, RS_CHATGPT_PREFIX_CN, RS_CHATGPT_FORMAT_INSTRUCTIONS_CN, RS_CHATGPT_SUFFIX_CN
from typing import Generator
from typing import Any, Dict, List, Optional, Tuple


device = "cuda"  # the device to load the model onto

bot_avatar = "/home/mars/cyh_ws/LLM/zjlab.jpg"           # 聊天机器人头像位置
user_avatar = "/home/mars/cyh_ws/LLM/zjlab.jpg"           # 用户头像位置
model_path = "/home/zjlab/Llama-3-8B-Chinese"   # 已下载的模型位置

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

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    
    return decorator

def pull_message(step_log: dict):
    if step_log.get("rationale"):
        yield ChatMessage(
            role="assistant", content=step_log["rationale"]
        )
    if step_log.get("tool_call"):
        used_code = step_log["tool_call"]["tool_name"] == "code interpreter"
        content = step_log["tool_call"]["tool_arguments"]
        if used_code:
            content = f"```py\n{content}\n```"
        yield ChatMessage(
            role="assistant",
            metadata={"title": f"🛠️ Used tool {step_log['tool_call']['tool_name']}"},
            content=content,
        )
    if step_log.get("observation"):
        yield ChatMessage(
            role="assistant", content=f"```\n{step_log['observation']}\n```"
        )
    if step_log.get("error"):
        yield ChatMessage(
            role="assistant",
            content=str(step_log["error"]),
            metadata={"title": "💥 Error"},
        )

def stream_from_transformers_agent(
    agent: ReactCodeAgent, prompt: str
) -> Generator[ChatMessage, None, ChatMessage | None]:
    """Runs an agent with the given prompt and streams the messages from the agent as ChatMessages."""

    class Output:
        output: agent_types.AgentType | str = None
    
    step_log = None
    for step_log in agent.run(prompt, stream=True):
        if isinstance(step_log, dict):
            for message in pull_message(step_log):
                print("message", message)
                yield message


    Output.output = step_log
    if isinstance(Output.output, agent_types.AgentText):
        yield ChatMessage(
            role="assistant", content=f"**Final answer:**\n```\n{Output.output.to_string()}\n```")  # type: ignore
    elif isinstance(Output.output, agent_types.AgentImage):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "image/png"},  # type: ignore
        )
    elif isinstance(Output.output, agent_types.AgentAudio):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "audio/wav"},  # type: ignore
        )
    else:
        return ChatMessage(role="assistant", content=Output.output)


class RSChat_app:
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

  
        self.llm = LLaMA3_LLM(mode_name_or_path="/home/zjlab/Llama-3-8B-Chinese")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output', return_messages=True)

    def initialize(self, language):
        self.memory.clear()  # clear previous history
        if language == 'English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_CHATGPT_PREFIX_CN, RS_CHATGPT_FORMAT_INSTRUCTIONS_CN, RS_CHATGPT_SUFFIX_CN
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True, stop=["\nObservation:", "\n\tObservation:"],
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX}, 
            handle_parsing_errors=True)

    def run_text(self, text, state): 
        try:
            res = self.agent({"input": text.strip()})
            res['output'] = res['output'].replace("\\", "/")

        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            
        # print(res)
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])        

        state = state + [(text, response)]

        # print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
        #       f"Current Memory: {self.agent.memory.buffer}")
        return state

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
        
        Human_prompt = f' Provide a remote sensing image named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
        AI_prompt = "Received."


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
    
    def _reset_chat(self) -> Tuple[str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        # clear agent history
        self.agent.reset()
        return "", "", ""  # clear textboxes    
    
    def _generate_response(
        self, chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Generate the response from agent, and capture the stdout of the
        ReActAgent's thoughts.
        """

    def run(self, *args: Any, **kwargs: Any) -> Any:
        demo = gr.Blocks(
            theme="gstaff/xkcd",
            css="#box { height: 420px; overflow-y: scroll !important}",
        )
        with demo:
            gr.Markdown(
            """
            # RS-Agent
            copyright@ZJLab

            """)
    
            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Message History",
                    scale=3,
                )
                console = gr.HTML(elem_id="box")
            with gr.Row():
                message = gr.Textbox(label="Write A Message", scale=4)
                clear = gr.ClearButton()

                # gr.Interface(sepia, inputs=gr.Image(), outputs="image")
            Human_prompt = f' Provide a remote sensing image named . The description is: . This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
            AI_prompt = "Received."
            message.submit(
                self.run_text,
                [message, chat_window],
                [message, chat_window],
                queue=False,
            ).then(
                self.run_text,
                chat_window,
                [chat_window, console],
            )
            clear.click(self._reset_chat, None, [message, chat_window, console])

        demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":

    load = "ImageCaptioning_cuda:0,SceneClassification_cuda:0,ObjectDetection_cuda:0,LandUseSegmentation_cuda:0,InstanceSegmentation_cuda:0,ObjectCounting_cuda:0,EdgeDetection_cpu"
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in load.split(',')}
    bot = RSChat_app(load_dict=load_dict).run()
    bot.initialize('English')
    print('RSChat initialization done, you can now chat with RSChat~')


    # demo.launch(server_name="0.0.0.0", server_port=7860)