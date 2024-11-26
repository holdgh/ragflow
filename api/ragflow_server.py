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

import logging
from api.utils.log_utils import initRootLogger
initRootLogger("ragflow_server")
for module in ["pdfminer"]:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.WARNING)
for module in ["peewee"]:
    module_logger = logging.getLogger(module)
    module_logger.handlers.clear()
    module_logger.propagate = True

import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from werkzeug.serving import run_simple
from api import settings
from api.apps import app
from api.db.runtime_config import RuntimeConfig
from api.db.services.document_service import DocumentService
from api import utils

from api.db.db_models import init_database_tables as init_web_db
from api.db.init_data import init_web_data
from api.versions import get_ragflow_version
from api.utils import show_configs


def update_progress():
    while True:
        time.sleep(3)
        try:
            # 每3秒执行一次逻辑
            # 逻辑：对未完成处理的文档依据其任务列表进行处理
            DocumentService.update_progress()
        except Exception:
            logging.exception("update_progress exception")


if __name__ == '__main__':
    # 打印启动日志
    logging.info(r"""
        ____   ___    ______ ______ __               
       / __ \ /   |  / ____// ____// /____  _      __
      / /_/ // /| | / / __ / /_   / // __ \| | /| / /
     / _, _// ___ |/ /_/ // __/  / // /_/ /| |/ |/ / 
    /_/ |_|/_/  |_|\____//_/    /_/ \____/ |__/|__/                             

    """)
    logging.info(
        f'RAGFlow version: {get_ragflow_version()}'
    )
    logging.info(
        f'project base: {utils.file_utils.get_project_base_directory()}'
    )
    # 打印api目录下的constants.py中的常量配置信息
    show_configs()
    # 依据配置文件service_conf.yaml和环境变量初始化配置信息
    settings.init_settings()

    # init db 初始化数据库--尚未理清
    init_web_db()
    init_web_data()
    # init runtime config
    import argparse

    parser = argparse.ArgumentParser()
    # store_true参数，加上即为true，不加则为false
    parser.add_argument(
        "--version", default=False, help="RAGFlow version", action="store_true"
    )
    parser.add_argument(
        "--debug", default=False, help="debug mode", action="store_true"
    )
    args = parser.parse_args()
    if args.version:
        # version参数为true时，打印版本信息
        print(get_ragflow_version())
        sys.exit(0)

    RuntimeConfig.DEBUG = args.debug
    if RuntimeConfig.DEBUG:
        # debug参数为true时，打印调试提示信息
        logging.info("run on debug mode")

    # 初始化版本和服务连接信息
    RuntimeConfig.init_env()
    RuntimeConfig.init_config(JOB_SERVER_HOST=settings.HOST_IP, HTTP_PORT=settings.HOST_PORT)
    # 异步处理任务：对未完成处理的文档依据其任务列表进行处理
    thread = ThreadPoolExecutor(max_workers=1)
    thread.submit(update_progress)

    # start http server
    # 启动后端接口服务，具体见api/apps/__init__.py中的app对象
    try:
        logging.info("RAGFlow HTTP server start...")
        run_simple(
            hostname=settings.HOST_IP,
            port=settings.HOST_PORT,
            application=app,
            threaded=True,
            use_reloader=RuntimeConfig.DEBUG,
            use_debugger=RuntimeConfig.DEBUG,
        )
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGKILL)
