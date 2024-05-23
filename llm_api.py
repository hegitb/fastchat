import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, json, datetime
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

app = FastAPI()

class Query(BaseModel):
    text: str
    history: list[dict[str, str]]

#chatGLM的调用方式
# path = "/mnt/nfs/penglc/nlp_model/ZhipuAI/chatglm2-6b"
# tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
# model.eval()

# @app.post("/chat/")
# async def chat(query: Query):
#     response, history = model.chat(tokenizer, query.text, query.history, max_length=2048)
#     return {
#         "response": response,
#         "history": history
#     }

#baichuan的调用方式
path = "/mnt/nfs/hejj/baichuan-inc/Baichuan2-13B-Chat"
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(path)
model.eval()

@app.post("/chat/")
async def chat(query: Query):
    query.history.append({"role": "user", "content": query.text})
    response = model.chat(tokenizer, query.history)
    # 手动构造历史对话，所以此处未返回
    return {
        "response": response,
        # "history": query.history
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
