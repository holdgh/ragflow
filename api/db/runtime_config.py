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
from api.versions import get_ragflow_version
from .reload_config_base import ReloadConfigBase


class RuntimeConfig(ReloadConfigBase):
    DEBUG = None
    WORK_MODE = None
    HTTP_PORT = None
    JOB_SERVER_HOST = None
    JOB_SERVER_VIP = None
    ENV = dict()
    SERVICE_DB = None
    LOAD_CONFIG_MANAGER = False

    @classmethod
    def init_config(cls, **kwargs):
        for k, v in kwargs.items():
            # hasattr()函数是Python内置函数之一，用于判断对象是否具有指定的属性或方法。它接受两个参数：对象和属性或方法的名称。函数返回一个布尔值，如果对象具有指定的属性或方法，则返回True，否则返回False。
            if hasattr(cls, k):
                # 当前类具有属性k时，则为k属性设置值v
                setattr(cls, k, v)

    @classmethod
    def init_env(cls):
        # 更新ENV中的版本信息
        cls.ENV.update({"version": get_ragflow_version()})

    @classmethod
    def load_config_manager(cls):
        cls.LOAD_CONFIG_MANAGER = True

    @classmethod
    def get_env(cls, key):
        return cls.ENV.get(key, None)

    @classmethod
    def get_all_env(cls):
        return cls.ENV

    @classmethod
    def set_service_db(cls, service_db):
        cls.SERVICE_DB = service_db
