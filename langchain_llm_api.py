import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn, json, datetime
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from langchain.llms.base import LLM
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain import FewShotPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import IPython
import sentence_transformers
from typing import Optional, List, Mapping, Any
import numpy as np
import json
import requests

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

app = FastAPI()

# 允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["http://127.0.0.1:5173"],
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatGLM(LLM):
    history: list[dict[str, str]] = []
    #history = []
    
    def __init__(self):
        super().__init__()
        self.history = []
    
    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        data={
            'text': prompt,
            'history': self.history
        }
        url = "http://10.16.22.110:8080/chat/"
        # response = requests.post(url, json=data)
        # if response.status_code != 200:
        #     try:
        #         error_message = response.json().get('error', 'Unknown error')
        #     except ValueError:
        #         error_message = response.text or 'Unknown error'
        #     return f"Error {response.status_code}: {error_message}"
        # resp = response.json()
        # if stop is not None:
        #     response = enforce_stop_tokens(response, stop)
        # self.history = resp['history']
        # return resp['response']
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # 如果响应状态码不是200，引发异常
            resp = response.json()
            if stop is not None:
                # 注意：这里看起来要执行的操作可能和你的逻辑有出入，确保 enforce_stop_tokens 函数存在并正确实现
                response = enforce_stop_tokens(response, stop)
            # self.history = resp['history']
            return resp['response']
        except requests.exceptions.HTTPError as http_err:
            # 返回HTTP错误信息和状态码
            error_details = response.text  # 获取原始响应文本作为错误详情
            return f"HTTP error {response.status_code}: {error_details}"
        except Exception as err:
            # 捕获所有其他异常类型
            return f"Other error occurred: {err}"
        
class UserAnswer(BaseModel):
    text: str
    history: list[dict[str, str]]
    user_description: list

class UserDescription(BaseModel):
    user_description: list

llm = ChatGLM()

EMBEDDING_MODEL = "/mnt/nfs/hejj/Xorbits/bge-large-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device='cuda')
scales_db = FAISS.load_local("/mnt/nfs/hejj/fastchatlog/various_psychological_scales_db",embeddings=embeddings)

#测试用例，检查接口是否有问题
@app.get("/test_chat/")
async def test_chat():
    # 构造要发送到8000端口服务的数据
    data = {
        "text": "你好",
        "history": []
    }
    url = "http://10.16.22.110:8080/chat/"

    try:
        # 向8000端口的服务发送请求
        response = requests.post(url, json=data)
        response.raise_for_status()  # 确保响应状态码是200

        # 如果成功，返回响应数据
        return response.json()
    except requests.RequestException as e:
        # 如果发生错误（包括网络错误、响应状态码不是200等），抛出HTTP异常
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/description/")
async def chat(query: UserAnswer):
    #需要中转一下变量格式，我也不确定为什么
    #单独存放历史记录
    format_conversion = query.history,
    format_conversion = format_conversion[0]

    #将历史记录加入模型自身的历史记录中辅助提取特征
    llm.history = format_conversion
    # create our examples
    examples = [
        {
            "query": "近期的生活对我来说真的很困难。每天都感到无法摆脱一种沉重的情绪，就像有一块巨石压在心头一样。我失去了对生活的兴趣，无法体验到以往的乐趣。\
    即使是最简单的日常任务也变得艰难，我感到疲惫不堪，几乎没有动力去做什么事情。",
            "answer": "持续的情绪低落、兴趣丧失、精力不足、以及丧失对日常任务的动力。"
        }, {
            "query": "孤独和孤立感困扰着我，虽然我周围有人，但我感到与世界疏离。与家人和朋友的交往变得越来越少，因为我觉得无法表达自己的情感，\
    也不想成为别人的负担。我常常感到无助和无望，觉得生活毫无意义。",
            "answer": "孤独感、社交孤立、情感表达困难、无助感以及对生活的无望感。"
        }
    ]
    
    # create a example template
    example_template = """
    状况描述: {query}
    症状: {answer}
    """
    
    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables = ["query", "answer"],
        template = example_template
    )
    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """ 根据最后的状况描述，参考以下示例，简短的概括最后的状况描述的症状，不需要提出建议也不需要做诊断。\
    如果该回复与心理健康没有关系，请说 “请按照要求回答问题”，答案请使用中文。
    示例："""
    # and the suffix our user input and output indicator
    suffix = """
    状况描述: {query}
    症状: """
    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples = examples,
        example_prompt = example_prompt,
        prefix = prefix,
        suffix = suffix,
        input_variables = ["query"],
        example_separator = ""
    )
    #将提取到的情感特征加入到情感特征集中
    query.user_description.append(llm(few_shot_prompt_template.format(query = query.text)))
    
    # create our examples
    question_examples = [
        {
            "query": "持续的情绪低落、兴趣丧失、精力不足、以及丧失对日常任务的动力。",
            "answer": "我很抱歉听到您正在经历这些困难。除了情绪低落和丧失兴趣外，您是否还经历了其他的情感问题，\
    比如焦虑、紧张或者恐惧？这些症状是否影响到您的社交活动或与家人朋友的交往？"
        }, {
            "query": "孤独感、社交孤立、情感表达困难、无助感以及对生活的无望感。",
            "answer": "您提到了孤独感、社交孤立、情感表达困难、无助感以及对生活的无望感。这些情绪问题可能相互关联。\
    您能详细描述一下，这些情绪是如何影响您的日常生活和情感状态的吗？是否有特定的情境或触发因素会加重这些情感困扰？"
        }
    ]
    
    # create a example template
    question_example_template = """
    状况描述: {query}
    问题: {answer}
    """
    
    # create a prompt example from above template
    question_example_prompt = PromptTemplate(
        input_variables = ["query", "answer"],
        template = question_example_template
    )
    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    question_prefix = """ 根据最后的状况描述，结合该描述向用户进一步提出问题，以便更深入的了解用户的心理状况。提问方式参考示例，不要涉及具体的心理疾病。语境和逻辑须符合日常对话。
    示例："""
    # and the suffix our user input and output indicator
    question_suffix = """
    状况描述: {query}
    问题: """
    # now create the few shot prompt template
    question_few_shot_prompt_template = FewShotPromptTemplate(
        examples = question_examples,
        example_prompt = question_example_prompt,
        prefix = question_prefix,
        suffix = question_suffix,
        input_variables = ["query"],
        example_separator = ""
    )

    # 将用户与机器人直接的历史对话作为附加信息加入
    llm.history = format_conversion
    robot_question = llm(question_few_shot_prompt_template.format(query = "、".join(query.user_description[-1])))
    # 将新一轮的用户描述以及机器人根据用户特征生成的问题加入格式化的历史对话中，返回给前端
    format_conversion.append({
        "role": "user",
        "content": query.text
    })
    format_conversion.append({
        "role": "assistant",
        "content": robot_question
    })

    #将历史记录置空以免影响后续对话
    llm.history = []
    
    return {
        "response": robot_question,
        "history": format_conversion,
        "user_description": query.user_description
    }

@app.post("/estimation/")
async def root(query: UserDescription):
    # 使用join方法将列表项拼接为一个字符串，可以指定一个分隔符（如果需要的话）
    result_string = "".join(query.user_description)
    
    # Build prompt
    diseases_template = """<指令>根据最后的状况描述，以及你现有的知识。分析用户的精神状况与其他心理疾病症状的相似性。\
    从[抑郁症,焦虑症,双相情感障碍,无心理疾病]四个结果中选择一个最有可能的结果作为判断，不需要附加其他内容，只返回判断结果即可。
    </指令>
    <示例>
    用户精神状况：具体内容。
    可能的心理疾病：双相情感障碍
    
    用户精神状况：具体内容。
    可能的心理疾病：焦虑症
    </示例>
    
    用户精神状况：{context}
    可能的心理疾病："""
     
    diseases_prompt = PromptTemplate(input_variables=["context"], template=diseases_template)
    diseases_query = diseases_prompt.format(context = result_string)
    llm.history = []
    diseases_result = llm(diseases_query)
    llm.history = []
    similarDocs = scales_db.similarity_search(diseases_result, include_metadata=True, k=1)
    return {
        "diseases_result": diseases_result,
        "scale": similarDocs[0].page_content
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
