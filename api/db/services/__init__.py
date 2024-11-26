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
import pathlib
import re
from .user_service import UserService


def duplicate_name(query_func, **kwargs):
    """
    功能：对name做“在文件后缀前追加版本号'(数字序列)'的去重【判断标准：查询不到数据】”处理
    逻辑：递归追加版本号，直到查询数据不存在，则返回不存在的name
    优点：适配各种类似逻辑，方法模板化
    """
    # 获取入参name字段
    fnm = kwargs["name"]
    # 依据条件查询数据
    objs = query_func(**kwargs)
    # 查询数据为空，直接返回name
    if not objs: return fnm
    # 获取文件后缀
    ext = pathlib.Path(fnm).suffix #.jpg
    # 去除文件后缀
    nm = re.sub(r"%s$"%ext, "", fnm)
    # 正则查询nm中“以(数字序列)结尾”
    r = re.search(r"\(([0-9]+)\)$", nm)
    c = 0
    if r:
        # r非空时，说明nm“以(数字序列)结尾”
        # 此处r.group(0)会返回'(数字序列)'，r.group(1)会返回'数字序列'
        c = int(r.group(1))
        # 去除'(数字序列)'
        nm = re.sub(r"\([0-9]+\)$", "", nm)
    # 此处逻辑：nm中不存在“以(数字序列)结尾”，则追加版本号(1)，存在时，则在原来基础上追加1作为版本号
    c += 1
    # 用新的版本号重命名nm
    nm = f"{nm}({c})"
    # 追加后缀名
    if ext: nm += f"{ext}"
    # 设置下次递归处理的name参数为nm
    kwargs["name"] = nm
    # 递归处理
    return duplicate_name(query_func, **kwargs)

