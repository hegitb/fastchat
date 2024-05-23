# fastchat
#本项目用于搭建基于baichuan大语言模型的心理测评系统



#代码启动方法：
uvicorn llm_api:app --reload --host 0.0.0.0 --port 8010  #启动baichuan2-13b模型
uvicorn langchain_llm_api:app --reload --host 0.0.0.0 --port 8001 #启动api接口

