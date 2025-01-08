"""
pip install python-dotenv
pip install langchain
pip install langchain-openai
"""

import base64
import json
import re
import requests
import traceback

from model_configurations import get_model_configuration

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

history_store = {}

def get_history_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history_store:
        history_store[session_id] = InMemoryChatMessageHistory()
    return history_store[session_id]

def get_image(path):
    with open(path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_llm():
    return AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

@tool
def get_holidays(country, year, month) -> json:
    '''獲取指定國家某月份的節假日資訊。country 是國家代碼(ISO 3166-1 alpha-2)。例如 'TW' 代表台灣。year 是年。month 是月'''
    response = requests.get("https://calendarific.com/api/v2/holidays", 
                            params={
                                "api_key": "uRoB9IJhzoK7C17HxKquy4lq6AVIeSfO",
                                "country": country,
                                "year": year,
                                "month": month,
                                "language": "zh"
                            })
    if response.status_code == 200:
        try:
            holidays = response.json()['response']['holidays']
            return {
                "Result": [
                    {
                        "date": holiday['date']['iso'],
                        "name": holiday['name']
                    }
                    for holiday in holidays
                ]
            }
        except Exception as e:
            print(f"Error: {e}\n{response.json()}")
    else:
        print(f"Request error：{response.status_code}\n{response.text}")
    return json.loads('{"Result": {}}')

def format_json(data):
    return json.dumps(data, indent=4, ensure_ascii=False)

holidays_json_format = '{{"Result": [{{ "date": "yyyy-MM-dd", "name": "節日名稱" }}, {{ "date": "yyyy-MM-dd", "name": "節日名稱" }}] }}'

def generate_hw01(question):
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', '妳是一個專門回答在特定國家的某個月份有哪些節假日的專家'),
        ('system', f'每次只須回答一個節日，答案請用此 JSON 格式呈現:{holidays_json_format}'), 
        ('human', '{input}')
    ])
    response = create_llm().invoke(prompt_template.format_prompt(input=question).to_messages())
    return format_json(JsonOutputParser().invoke(response))
    
def generate_hw02(question):
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', '妳是一個專門回答在特定國家的某個月份有哪些節假日的專家'),
        ('system', f'回答的所有節日，並用繁體中文回答節日名稱，答案請用此 JSON 格式呈現:{holidays_json_format}'), 
        ('human', '{input}'),
        ('human', '{agent_scratchpad}'),
    ])
    tools = [get_holidays]
    response = AgentExecutor(
        agent=create_tool_calling_agent(create_llm(), tools, prompt_template),
        tools=tools,
        verbose=True
    ).invoke({'input': question})
    return format_json(JsonOutputParser().parse(response['output']))
    
def generate_hw03(question2, question3):
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', '妳是一個專門回答在特定國家的某個月份有哪些節假日的專家'),
        MessagesPlaceholder(variable_name='history'),
        ('human', '{input}'),
        ('human', '{agent_scratchpad}'),
    ])
    tools = [get_holidays]
    config = {'configurable': {'session_id': 'hw03'}}
    llm_with_history = RunnableWithMessageHistory(
        AgentExecutor(
            agent=create_tool_calling_agent(create_llm(), tools, prompt_template),
            tools=tools,
            verbose=True
        ),
        input_messages_key='input',
        history_messages_key='history',
        get_session_history=get_history_by_session_id
    )
    parser = JsonOutputParser()
    response = llm_with_history.invoke(
        {'input': f'{question2}\n回答的所有節日，並用繁體中文回答節日名稱，答案請用此 JSON 格式呈現:{holidays_json_format}'},
        config=config
    )
    # print(parser.invoke(response['output']))
    response = llm_with_history.invoke(
        {'input': f'{question3}\n 回答是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false，並說明做出此回答的理由，答案請用此 JSON 格式呈現:{{ "Result": {{ "add": true, "reason": "理由描述" }} }}'},
        config=config
    )
    return format_json(parser.invoke(response['output']))

def generate_hw04(question):
    llm = create_llm()
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', '妳是一個專門回答圖片中分數的專家，每次只須回答一個分數，答案請用如下的 JSON 格式呈現:{{ "Result": {{ "score": 5498 }} }}'),
        ('human', '{input}')
    ])
    messages = prompt_template.format_prompt(input=question).to_messages()
    messages.append(HumanMessage([{ 
        'type': 'image_url', 
        'image_url': {'url': f'data:image/jpeg;base64,{get_image("./baseball.png")}'}
    }]))
    response = llm.invoke(messages)
    return format_json(JsonOutputParser().invoke(response))
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

# print(generate_hw01('2024年台灣10月紀念日有哪些?'))
# print(generate_hw02('2024年台灣10月紀念日有哪些?'))
# print(generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？'))
# print(generate_hw04('請問中華台北的積分是多少'))