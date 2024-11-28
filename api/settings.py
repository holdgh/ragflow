#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
from datetime import date
from enum import IntEnum, Enum
import rag.utils.es_conn
import rag.utils.infinity_conn

import rag.utils
from rag.nlp import search
from graphrag import search as kg_search
from api.utils import get_base_config, decrypt_database_config
from api.constants import RAG_FLOW_SERVICE_NAME
# 读取环境变量LIGHTEN【在dockerfile中有定义】的值，默认值为0
LIGHTEN = int(os.environ.get('LIGHTEN', "0"))

LLM = None
LLM_FACTORY = None
LLM_BASE_URL = None
CHAT_MDL = ""
EMBEDDING_MDL = ""
RERANK_MDL = ""
ASR_MDL = ""
IMAGE2TEXT_MDL = ""
API_KEY = None
PARSERS = None
HOST_IP = None
HOST_PORT = None
SECRET_KEY = None

DATABASE_TYPE = os.getenv("DB_TYPE", 'mysql')
DATABASE = decrypt_database_config(name=DATABASE_TYPE)

# authentication
AUTHENTICATION_CONF = None

# client
CLIENT_AUTHENTICATION = None
HTTP_APP_KEY = None
GITHUB_OAUTH = None
FEISHU_OAUTH = None

DOC_ENGINE = None
docStoreConn = None

retrievaler = None
kg_retrievaler = None


def init_settings():
    """
    从配置文件service_conf.yaml和环境变量中获取配置信息，初始化大模型、请求信息、认证信息、存储和检索工具对象
    """
    # 以global声明当前方法体内的变量名为全局变量
    global LLM, LLM_FACTORY, LLM_BASE_URL
    # 读取大模型的配置字典
    LLM = get_base_config("user_default_llm", {})
    LLM_FACTORY = LLM.get("factory", "Tongyi-Qianwen")
    LLM_BASE_URL = LLM.get("base_url")

    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, IMAGE2TEXT_MDL
    if not LIGHTEN:
        # 如果lighten为0，则采用默认大模型配置。
        default_llm = {
            "Tongyi-Qianwen": {
                "chat_model": "qwen-plus",
                "embedding_model": "text-embedding-v2",
                "image2text_model": "qwen-vl-max",
                "asr_model": "paraformer-realtime-8k-v1",
            },
            "OpenAI": {
                "chat_model": "gpt-3.5-turbo",
                "embedding_model": "text-embedding-ada-002",
                "image2text_model": "gpt-4-vision-preview",
                "asr_model": "whisper-1",
            },
            "Azure-OpenAI": {
                "chat_model": "gpt-35-turbo",
                "embedding_model": "text-embedding-ada-002",
                "image2text_model": "gpt-4-vision-preview",
                "asr_model": "whisper-1",
            },
            "ZHIPU-AI": {
                "chat_model": "glm-3-turbo",
                "embedding_model": "embedding-2",
                "image2text_model": "glm-4v",
                "asr_model": "",
            },
            "Ollama": {
                "chat_model": "qwen-14B-chat",
                "embedding_model": "flag-embedding",
                "image2text_model": "",
                "asr_model": "",
            },
            "Moonshot": {
                "chat_model": "moonshot-v1-8k",
                "embedding_model": "",
                "image2text_model": "",
                "asr_model": "",
            },
            "DeepSeek": {
                "chat_model": "deepseek-chat",
                "embedding_model": "",
                "image2text_model": "",
                "asr_model": "",
            },
            "VolcEngine": {
                "chat_model": "",
                "embedding_model": "",
                "image2text_model": "",
                "asr_model": "",
            },
            "BAAI": {
                "chat_model": "",
                "embedding_model": "BAAI/bge-large-zh-v1.5",
                "image2text_model": "",
                "asr_model": "",
                "rerank_model": "BAAI/bge-reranker-v2-m3",
            }
        }

        if LLM_FACTORY:
            # LLM_FACTORY非空时，从default中选取相应的char_model、asr_model和image2text_model
            CHAT_MDL = default_llm[LLM_FACTORY]["chat_model"] + f"@{LLM_FACTORY}"
            ASR_MDL = default_llm[LLM_FACTORY]["asr_model"] + f"@{LLM_FACTORY}"
            IMAGE2TEXT_MDL = default_llm[LLM_FACTORY]["image2text_model"] + f"@{LLM_FACTORY}"
        # 选取BAAI的embedding和rerank模型
        EMBEDDING_MDL = default_llm["BAAI"]["embedding_model"] + "@BAAI"
        RERANK_MDL = default_llm["BAAI"]["rerank_model"] + "@BAAI"

    global API_KEY, PARSERS, HOST_IP, HOST_PORT, SECRET_KEY
    API_KEY = LLM.get("api_key", "")
    PARSERS = LLM.get(
        "parsers",
        "naive:General,qa:Q&A,resume:Resume,manual:Manual,table:Table,paper:Paper,book:Book,laws:Laws,presentation:Presentation,picture:Picture,one:One,audio:Audio,knowledge_graph:Knowledge Graph,email:Email")
    # 获取配置文件service_conf.yaml中的IP和端口
    HOST_IP = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
    HOST_PORT = get_base_config(RAG_FLOW_SERVICE_NAME, {}).get("http_port")
    # 获取配置文件service_conf.yaml中的secret_key、authentication、oauth
    SECRET_KEY = get_base_config(
        RAG_FLOW_SERVICE_NAME,
        {}).get("secret_key", str(date.today()))

    global AUTHENTICATION_CONF, CLIENT_AUTHENTICATION, HTTP_APP_KEY, GITHUB_OAUTH, FEISHU_OAUTH
    # authentication
    AUTHENTICATION_CONF = get_base_config("authentication", {})

    # client
    CLIENT_AUTHENTICATION = AUTHENTICATION_CONF.get(
        "client", {}).get(
        "switch", False)
    HTTP_APP_KEY = AUTHENTICATION_CONF.get("client", {}).get("http_app_key")
    GITHUB_OAUTH = get_base_config("oauth", {}).get("github")
    FEISHU_OAUTH = get_base_config("oauth", {}).get("feishu")

    global DOC_ENGINE, docStoreConn, retrievaler, kg_retrievaler
    # 从环境变量中获取DOC_ENGINE信息，以此来初始化全局变量docStoreConn对象【存储工具，是ES还是infinity】
    # 根据docker/.env文件可知，采用ES存储
    DOC_ENGINE = os.environ.get('DOC_ENGINE', "elasticsearch")
    if DOC_ENGINE == "elasticsearch":
        docStoreConn = rag.utils.es_conn.ESConnection()
    elif DOC_ENGINE == "infinity":
        docStoreConn = rag.utils.infinity_conn.InfinityConnection()
    else:
        raise Exception(f"Not supported doc engine: {DOC_ENGINE}")
    # 初始化全局变量retrievaler和kg_retrievaler【检索工具】
    retrievaler = search.Dealer(docStoreConn)
    kg_retrievaler = kg_search.KGSearch(docStoreConn)


class CustomEnum(Enum):
    @classmethod
    def valid(cls, value):
        try:
            cls(value)
            return True
        except BaseException:
            return False

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def names(cls):
        return [member.name for member in cls.__members__.values()]


class RetCode(IntEnum, CustomEnum):
    SUCCESS = 0
    NOT_EFFECTIVE = 10
    EXCEPTION_ERROR = 100
    ARGUMENT_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    CONNECTION_ERROR = 105
    RUNNING = 106
    PERMISSION_ERROR = 108
    AUTHENTICATION_ERROR = 109
    UNAUTHORIZED = 401
    SERVER_ERROR = 500
    FORBIDDEN = 403
    NOT_FOUND = 404
