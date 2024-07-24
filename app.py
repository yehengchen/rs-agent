from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from LLM import LLaMA3_LLM
from RStask.ObjectDetection.models.yolov5s import *
from RStask.LanduseSegmentation.unet import *

from Prefix import RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX, RS_CHATGPT_PREFIX_CN, RS_CHATGPT_FORMAT_INSTRUCTIONS_CN, RS_CHATGPT_SUFFIX_CN
from RStask import ImageEdgeFunction, CaptionFunction, LanduseFunction, LanduseFunction_Unet, DetectionFunction, DetectionFunction_ship, CountingFuncnction, \
    CountingFuncnction_ship, SceneFunction, InstanceFunction, CaptionFunction_RS_BLIP, CaptionFunction3

# from langchain_openai import ChatOpenAI
from gradio import ChatMessage
import gradio as gr

from dotenv import load_dotenv

llm = LLaMA3_LLM(mode_name_or_path="/home/zjlab/Llama-3-8B-Chinese")  
memory = ConversationBufferMemory(memory_key="chat_history", output_key='output', return_messages=True)
# tools = Tool(name=func.name, description=func.description, func=func)

tools = load_tools(["ObjectCounting"])

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")
# print(prompt.messages) -- to see the prompt

PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_CHATGPT_PREFIX, RS_CHATGPT_FORMAT_INSTRUCTIONS, RS_CHATGPT_SUFFIX
agent = initialize_agent(
            tools,
            llm,
            agent="conversational-react-description",
            verbose=True,
            memory= memory,
            return_intermediate_steps=True, stop=["\nObservation:", "\n\tObservation:"],
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX}, 
            handle_parsing_errors=True)


agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)

async def interact_with_langchain_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    async for chunk in agent_executor.astream(
        {"input": prompt}
    ):
        if "steps" in chunk:
            for step in chunk["steps"]:
                messages.append(ChatMessage(role="assistant", content=step.action.log,
                                  metadata={"title": f"🛠️ Used tool {step.action.tool}"}))
                yield messages
        if "output" in chunk:
            messages.append(ChatMessage(role="assistant", content=chunk["output"]))
            yield messages


with gr.Blocks() as demo:
    gr.Markdown("# Chat with a LangChain Agent 🦜⛓️ and see its thoughts 💭")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/141/parrot_1f99c.png",
        ),
    )
    input = gr.Textbox(lines=1, label="Chat Message")
    input.submit(interact_with_langchain_agent, [input_2, chatbot_2], [chatbot_2])

demo.launch()