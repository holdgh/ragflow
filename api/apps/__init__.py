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
import sys
import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from flask import Blueprint, Flask
from werkzeug.wrappers.request import Request
from flask_cors import CORS
from flasgger import Swagger
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer

from api.db import StatusEnum
from api.db.db_models import close_connection
from api.db.services import UserService
from api.utils import CustomJSONEncoder, commands

from flask_session import Session
from flask_login import LoginManager
from api import settings
from api.utils.api_utils import server_error_response
from api.constants import API_VERSION

__all__ = ["app"]

Request.json = property(lambda self: self.get_json(force=True, silent=True))

app = Flask(__name__)

# Add this at the beginning of your file to configure Swagger UI
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,  # Include all endpoints
            "model_filter": lambda tag: True,  # Include all models
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}

swagger = Swagger(
    app,
    config=swagger_config,
    template={
        "swagger": "2.0",
        "info": {
            "title": "RAGFlow API",
            "description": "",
            "version": "1.0.0",
        },
        "securityDefinitions": {
            "ApiKeyAuth": {"type": "apiKey", "name": "Authorization", "in": "header"}
        },
    },
)

CORS(app, supports_credentials=True, max_age=2592000)
app.url_map.strict_slashes = False
app.json_encoder = CustomJSONEncoder
# 全局异常设置
app.errorhandler(Exception)(server_error_response)

## convince for dev and debug
# app.config["LOGIN_DISABLED"] = True
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
# 最大请求长度128M
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024)
)

Session(app)
login_manager = LoginManager()
login_manager.init_app(app)
# 重置密码和重置邮箱
commands.register_commands(app)


def search_pages_path(pages_dir):
    """
    pages_dir：Windows环境下入参为WindowsPath对象
    逻辑：获取所有具备以下特点的路径：
        1、以_app.py结尾且不以.开头的路径
        2、所有以sdk结尾目录下，所有python脚本的路径
    """
    # pages_dir.glob("*_app.py")获取pages_dir目录下，所有以_app.py结尾且不以.开头的路径【Windows环境下入参为WindowsPath对象】列表
    app_path_list = [
        path for path in pages_dir.glob("*_app.py") if not path.name.startswith(".")
    ]
    # pages_dir.glob("*sdk/*.py")获取pages_dir目录中所有以sdk结尾目录下，所有python脚本的路径【Windows环境下入参为WindowsPath对象】列表
    api_path_list = [
        path for path in pages_dir.glob("*sdk/*.py") if not path.name.startswith(".")
    ]
    # 合并一起，返回合并结果
    app_path_list.extend(api_path_list)
    return app_path_list


def register_page(page_path):
    path = f"{page_path}"
    # page_path.stem获取路径page_path的文件名，形如：WindowsPath('D:/project/AI/ragflow/api/apps/sdk/dataset.py')-->dataset
    # 'chat_app'.rstrip('_app')-->'chat'
    page_name = page_path.stem.rstrip("_app")
    # page_path.parts获取路径中每一级别的目录或文件名，形如：WindowsPath('D:/project/AI/ragflow/api/apps/sdk/chat.py')-->('D:\\', 'project', 'AI', 'ragflow', 'api', 'apps', 'sdk', 'chat.py')
    # page_path.parts[page_path.parts.index("api"): -1]取parts中”从api开始到倒数第二元素为止“的子元组
    # 综合：WindowsPath('D:/project/AI/ragflow/api/apps/sdk/chat.py')-->api.apps.sdk.chat
    module_name = ".".join(
        page_path.parts[page_path.parts.index("api"): -1] + (page_name,)
    )
    # 以module_name为'api.apps.sdk.chat'，page_path为WindowsPath('D:/project/AI/ragflow/api/apps/sdk/chat.py')作为示例，结果为：ModuleSpec(name='api.apps.sdk.chat', loader=<_frozen_importlib_external.SourceFileLoader object at 0x0000023A487758B0>, origin='D:\\project\\AI\\ragflow\\a
    # pi\\apps\\sdk\\chat.py')
    # importlib.util.spec_from_file_location(name, location, *, loader=None, submodule_search_locations=None) 其根据指向某个文件的路径创建一个 ModuleSpec 实例
    spec = spec_from_file_location(module_name, page_path)
    # 接上句示例参数，结果为：<module 'api.apps.sdk.chat' from 'D:\\project\\AI\\ragflow\\api\\apps\\sdk\\chat.py'>
    # 构建模块
    # importlib.util.module_from_spec(spec) 从 spec 中创建一个新的模块，之后就可以使 module 当 itertools 使用。
    page = module_from_spec(spec)
    # 为模块赋值属性：app,manager【蓝图对象】
    page.app = app
    page.manager = Blueprint(page_name, module_name)
    # 将模块赋值给sys的模块
    sys.modules[module_name] = page
    # spec.loader.exec_module(module) 执行某个模块
    spec.loader.exec_module(page)
    page_name = getattr(page, "page_name", page_name)
    # 设置路径前缀：路径包含sdk，则采用/api/API_VERSION，否则采用/API_VERSION/page_name
    url_prefix = (
        f"/api/{API_VERSION}" if "/sdk/" in path else f"/{API_VERSION}/{page_name}"
    )
    # 蓝图是一种组织 Flask 应用程序的方式，将应用程序划分为多个模块并使其易于管理
    # 访问接口的方式：http://ip:port/url_prefix/蓝图中的接口路径
    # 具体接口路径中的注解@manager.route中的manager就是蓝图变量的名称
    app.register_blueprint(page.manager, url_prefix=url_prefix)
    return url_prefix

# 效果：Path('D:\\project\\AI\\ragflow\\readme.txt').parent-->WindowsPath('D:/project/AI/ragflow')
pages_dir = [
    # WindowsPath('D:\project\AI\ragflow\api\')
    Path(__file__).parent,
    # WindowsPath('D:\project\AI\ragflow\api\apps')
    Path(__file__).parent.parent / "api" / "apps",
    # WindowsPath('D:\project\AI\ragflow\api\apps\sdk')
    Path(__file__).parent.parent / "api" / "apps" / "sdk",
]
# 在代码根目录下、代码根目录/api/apps目录下、代码根目录/api/apps/sdk目录下，搜索所有具备以下特点的路径：
#         1、以_app.py结尾且不以.开头的路径
#         2、所有以sdk结尾目录下，所有python脚本的路径
# 登记上述路径模块到app中
client_urls_prefix = [
    register_page(path) for dir in pages_dir for path in search_pages_path(dir)
]


@login_manager.request_loader
def load_user(web_request):
    jwt = Serializer(secret_key=settings.SECRET_KEY)
    # 获取请求中的令牌信息
    authorization = web_request.headers.get("Authorization")
    if authorization:
        # 令牌非空时
        try:
            # 依据令牌和逻辑未删除状态查询用户信息
            access_token = str(jwt.loads(authorization))
            user = UserService.query(
                access_token=access_token, status=StatusEnum.VALID.value
            )
            if user:
                # 用户信息非空，则返回用户id
                return user[0]
            else:
                return None
        except Exception:
            logging.exception("load_user got exception")
            return None
    else:
        return None


@app.teardown_request
def _db_close(exc):
    close_connection()
