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
from flask import request
from api import settings
from api.db import StatusEnum
from api.db.services.dialog_service import DialogService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import  TenantLLMService
from api.db.services.user_service import TenantService
from api.utils import get_uuid
from api.utils.api_utils import get_error_data_result, token_required
from api.utils.api_utils import get_result



@manager.route('/chats', methods=['POST'])
@token_required
def create(tenant_id):
    """
    功能：创建聊天助手【关联知识库、提示prompt、检索参数、大模型及超参数】
    逻辑：
        1、检验知识库入参
        2、检验大模型和用户入参
        3、提示prompt处理
        4、助理记录其他字段初始化
        5、保存助理记录
        6、构造返回值【多为字段转换逻辑】
    """
    # ============检验知识库入参-start============
    req=request.json
    ids= req.get("dataset_ids")
    if not ids:
        # 知识库id列表为空
        return get_error_data_result(message="`dataset_ids` is required")
    for kb_id in ids:
        kbs = KnowledgebaseService.accessible(kb_id=kb_id,user_id=tenant_id)
        if not kbs:
            # 当前用户不存在当前知识库
            return get_error_data_result(f"You don't own the dataset {kb_id}")
        kbs = KnowledgebaseService.query(id=kb_id)
        kb = kbs[0]
        if kb.chunk_num == 0:
            # 当前知识库没有解析完成的文件【在task_executor中解析处理完成，会将文件分块数量维护到对应的文档和知识库记录】
            return get_error_data_result(f"The dataset {kb_id} doesn't own parsed file")
    kbs = KnowledgebaseService.get_by_ids(ids)
    embd_count = list(set([kb.embd_id for kb in kbs]))
    if len(embd_count) != 1:
        # 一个知识库只能有一个embedding模型
        return get_result(message='Datasets use different embedding models."',code=settings.RetCode.AUTHENTICATION_ERROR)
    req["kb_ids"] = ids
    # ============检验知识库入参-end============
    # ============检验大模型和用户入参-start============
    # llm
    llm = req.get("llm")
    if llm:
        if "model_name" in llm:
            req["llm_id"] = llm.pop("model_name")
            # 校验大模型服务是否存在
            if not TenantLLMService.query(tenant_id=tenant_id,llm_name=req["llm_id"],model_type="chat"):
                return get_error_data_result(f"`model_name` {req.get('llm_id')} doesn't exist")
        req["llm_setting"] = req.pop("llm")
    # 校验用户是否存在
    e, tenant = TenantService.get_by_id(tenant_id)
    if not e:
        return get_error_data_result(message="Tenant not found!")
    # ============检验大模型和用户入参-end============
    # ============提示prompt处理-start============
    # prompt
    prompt = req.get("prompt")
    key_mapping = {"parameters": "variables",
                   "prologue": "opener",
                   "quote": "show_quote",
                   "system": "prompt",
                   "rerank_id": "rerank_model",
                   "vector_similarity_weight": "keywords_similarity_weight"}
    # 对于similarity_threshold，官方demo解释：我们使用混合相似度得分来评估两行文本之间的距离。 它是加权关键词相似度和向量余弦相似度。 如果查询和块之间的相似度小于此阈值，则该块将被过滤掉。
    # 对于vector_similarity_weight，官方demo解释：我们使用混合相似性评分来评估两行文本之间的距离。它是加权关键字相似性和矢量余弦相似性或rerank得分（0〜1）。两个权重的总和为1.0。
    key_list = ["similarity_threshold", "vector_similarity_weight", "top_n", "rerank_id"]
    if prompt:
        # 入参中的提示prompt非空时
        # 基于key_mapping替换提示prompt入参字段名
        for new_key, old_key in key_mapping.items():
            if old_key in prompt:
                prompt[new_key] = prompt.pop(old_key)
        # 基于key_list提取prompt中的指定字段到req中
        for key in key_list:
            if key in prompt:
                req[key] = prompt.pop(key)
        # 将入参中的提示prompt字段名置为prompt_config
        req["prompt_config"] = req.pop("prompt")
    # ============提示prompt处理-end============
    # ============助理记录其他字段初始化-start============
    # init
    # 随机id
    req["id"] = get_uuid()
    # 助理描述
    req["description"] = req.get("description", "A helpful Assistant")
    # 助理头像
    req["icon"] = req.get("avatar", "")
    # 官方demo对top_n参数的解释：并非所有相似度得分高于“相似度阈值”的块都会被提供给大语言模型。 LLM 只能看到这些“Top N”块
    req["top_n"] = req.get("top_n", 6)
    # TODO 在页面入参中找不到top_k参数
    req["top_k"] = req.get("top_k", 1024)
    # 校验rerank模型是否存在
    req["rerank_id"] = req.get("rerank_id", "")
    if req.get("rerank_id"):
        if not TenantLLMService.query(tenant_id=tenant_id,llm_name=req.get("rerank_id"),model_type="rerank"):
            return get_error_data_result(f"`rerank_model` {req.get('rerank_id')} doesn't exist")
    # 入参中没有大模型时，取用户记录的默认大模型
    if not req.get("llm_id"):
        req["llm_id"] = tenant.llm_id
    # 检验助理名称是否传值，是否在数据库中已存在
    if not req.get("name"):
        return get_error_data_result(message="`name` is required.")
    if DialogService.query(name=req["name"], tenant_id=tenant_id, status=StatusEnum.VALID.value):
        return get_error_data_result(message="Duplicated chat name in creating chat.")
    # tenant_id
    # 校验用户id是否传值
    if req.get("tenant_id"):
        return get_error_data_result(message="`tenant_id` must not be provided.")
    # TODO 此处多余？
    req["tenant_id"] = tenant_id
    # prompt more parameter
    # 设置提示词
    default_prompt = {
        "system": """You are an intelligent assistant. Please summarize the content of the knowledge base to answer the question. Please list the data in the knowledge base and answer in detail. When all knowledge base content is irrelevant to the question, your answer must include the sentence "The answer you are looking for is not found in the knowledge base!" Answers need to consider chat history.
      Here is the knowledge base:
      {knowledge}
      The above is the knowledge base.""",
        "prologue": "Hi! I'm your assistant, what can I do for you?",
        "parameters": [
            {"key": "knowledge", "optional": False}
        ],
        "empty_response": "Sorry! No relevant content was found in the knowledge base!"
    }
    # 提示词涉及：系统提示词、提示语、关键参数、无法回答时的默认回复
    key_list_2 = ["system", "prologue", "parameters", "empty_response"]
    # 未设置提示配置时，采用默认提示配置
    if "prompt_config" not in req:
        req['prompt_config'] = {}
    for key in key_list_2:
        temp = req['prompt_config'].get(key)
        if not temp:
            # 未设置提示配置的指定项时，采用默认提示配置的指定项
            req['prompt_config'][key] = default_prompt[key]
    # 处理关键参数
    for p in req['prompt_config']["parameters"]:
        # 过滤可选的关键参数
        if p["optional"]:
            continue
        # 如果系统提示词中没有【不可选，即必有】关键参数字段“类似，{knowledge}”，则抛出异常
        if req['prompt_config']["system"].find("{%s}" % p["key"]) < 0:
            return get_error_data_result(
                message="Parameter '{}' is not used".format(p["key"]))
    # ============助理记录其他字段初始化-end============
    # save
    # 保存助理记录
    if not DialogService.save(**req):
        return get_error_data_result(message="Fail to new a chat!")
    # response
    # ===========构造返回值【多为字段转换逻辑】-start===========
    # 查询助理记录
    e, res = DialogService.get_by_id(req["id"])
    if not e:
        return get_error_data_result(message="Fail to new a chat!")
    res = res.to_json()
    renamed_dict = {}
    # 基于key_mapping转换字段名
    for key, value in res["prompt_config"].items():
        new_key = key_mapping.get(key, key)
        renamed_dict[new_key] = value
    # 收集转换结果到prompt字段
    res["prompt"] = renamed_dict
    # 删除原始字段
    del res["prompt_config"]
    new_dict = {"similarity_threshold": res["similarity_threshold"],
                "keywords_similarity_weight": res["vector_similarity_weight"],
                "top_n": res["top_n"],
                "rerank_model": res['rerank_id']}
    # 将new_dict保存或更新到res的prompt中
    res["prompt"].update(new_dict)
    # 删除上述new_dict涉及的原始字段
    for key in key_list:
        del res[key]
    # 其他字段转换
    res["llm"] = res.pop("llm_setting")
    res["llm"]["model_name"] = res.pop("llm_id")
    del res["kb_ids"]
    res["dataset_ids"] = req["dataset_ids"]
    res["avatar"] = res.pop("icon")
    # ===========构造返回值【多为字段转换逻辑】-end===========
    return get_result(data=res)

@manager.route('/chats/<chat_id>', methods=['PUT'])
@token_required
def update(tenant_id,chat_id):
    if not DialogService.query(tenant_id=tenant_id, id=chat_id, status=StatusEnum.VALID.value):
        return get_error_data_result(message='You do not own the chat')
    req =request.json
    ids = req.get("dataset_ids")
    if "show_quotation" in req:
        req["do_refer"]=req.pop("show_quotation")
    if "dataset_ids" in req:
        if not ids:
            return get_error_data_result("`datasets` can't be empty")
        if ids:
            for kb_id in ids:
                kbs = KnowledgebaseService.accessible(kb_id=chat_id, user_id=tenant_id)
                if not kbs:
                    return get_error_data_result(f"You don't own the dataset {kb_id}")
                kbs = KnowledgebaseService.query(id=kb_id)
                kb = kbs[0]
                if kb.chunk_num == 0:
                    return get_error_data_result(f"The dataset {kb_id} doesn't own parsed file")
            kbs = KnowledgebaseService.get_by_ids(ids)
            embd_count=list(set([kb.embd_id for kb in kbs]))
            if len(embd_count) != 1 :
                return get_result(
                    message='Datasets use different embedding models."',
                    code=settings.RetCode.AUTHENTICATION_ERROR)
            req["kb_ids"] = ids
    llm = req.get("llm")
    if llm:
        if "model_name" in llm:
            req["llm_id"] = llm.pop("model_name")
            if not TenantLLMService.query(tenant_id=tenant_id,llm_name=req["llm_id"],model_type="chat"):
                return get_error_data_result(f"`model_name` {req.get('llm_id')} doesn't exist")
        req["llm_setting"] = req.pop("llm")
    e, tenant = TenantService.get_by_id(tenant_id)
    if not e:
        return get_error_data_result(message="Tenant not found!")
    if req.get("rerank_model"):
        if not TenantLLMService.query(tenant_id=tenant_id,llm_name=req.get("rerank_model"),model_type="rerank"):
            return get_error_data_result(f"`rerank_model` {req.get('rerank_model')} doesn't exist")
    # prompt
    prompt = req.get("prompt")
    key_mapping = {"parameters": "variables",
                   "prologue": "opener",
                   "quote": "show_quote",
                   "system": "prompt",
                   "rerank_id": "rerank_model",
                   "vector_similarity_weight": "keywords_similarity_weight"}
    key_list = ["similarity_threshold", "vector_similarity_weight", "top_n", "rerank_id"]
    if prompt:
        for new_key, old_key in key_mapping.items():
            if old_key in prompt:
                prompt[new_key] = prompt.pop(old_key)
        for key in key_list:
            if key in prompt:
                req[key] = prompt.pop(key)
        req["prompt_config"] = req.pop("prompt")
    e, res = DialogService.get_by_id(chat_id)
    res = res.to_json()
    if "name" in req:
        if not req.get("name"):
            return get_error_data_result(message="`name` is not empty.")
        if req["name"].lower() != res["name"].lower() \
                and len(
            DialogService.query(name=req["name"], tenant_id=tenant_id, status=StatusEnum.VALID.value)) > 0:
            return get_error_data_result(message="Duplicated chat name in updating dataset.")
    if "prompt_config" in req:
        res["prompt_config"].update(req["prompt_config"])
        for p in res["prompt_config"]["parameters"]:
            if p["optional"]:
                continue
            if res["prompt_config"]["system"].find("{%s}" % p["key"]) < 0:
                return get_error_data_result(message="Parameter '{}' is not used".format(p["key"]))
    if "llm_setting" in req:
        res["llm_setting"].update(req["llm_setting"])
    req["prompt_config"] = res["prompt_config"]
    req["llm_setting"] = res["llm_setting"]
    # avatar
    if "avatar" in req:
        req["icon"] = req.pop("avatar")
    if "dataset_ids" in req:
        req.pop("dataset_ids")
    if not DialogService.update_by_id(chat_id, req):
        return get_error_data_result(message="Chat not found!")
    return get_result()


@manager.route('/chats', methods=['DELETE'])
@token_required
def delete(tenant_id):
    req = request.json
    if not req:
        ids=None
    else:
        ids=req.get("ids")
    if not ids:
        id_list = []
        dias=DialogService.query(tenant_id=tenant_id,status=StatusEnum.VALID.value)
        for dia in dias:
            id_list.append(dia.id)
    else:
        id_list=ids
    for id in id_list:
        if not DialogService.query(tenant_id=tenant_id, id=id, status=StatusEnum.VALID.value):
            return get_error_data_result(message=f"You don't own the chat {id}")
        temp_dict = {"status": StatusEnum.INVALID.value}
        DialogService.update_by_id(id, temp_dict)
    return get_result()

@manager.route('/chats', methods=['GET'])
@token_required
def list_chat(tenant_id):
    id = request.args.get("id")
    name = request.args.get("name")
    chat = DialogService.query(id=id,name=name,status=StatusEnum.VALID.value,tenant_id=tenant_id)
    if not chat:
        return get_error_data_result(message="The chat doesn't exist")
    page_number = int(request.args.get("page", 1))
    items_per_page = int(request.args.get("page_size", 30))
    orderby = request.args.get("orderby", "create_time")
    if request.args.get("desc") == "False" or request.args.get("desc") == "false":
        desc = False
    else:
        desc = True
    chats = DialogService.get_list(tenant_id,page_number,items_per_page,orderby,desc,id,name)
    if not chats:
        return get_result(data=[])
    list_assts = []
    renamed_dict = {}
    key_mapping = {"parameters": "variables",
                   "prologue": "opener",
                   "quote": "show_quote",
                   "system": "prompt",
                   "rerank_id": "rerank_model",
                   "vector_similarity_weight": "keywords_similarity_weight",
                   "do_refer":"show_quotation"}
    key_list = ["similarity_threshold", "vector_similarity_weight", "top_n", "rerank_id"]
    for res in chats:
        for key, value in res["prompt_config"].items():
            new_key = key_mapping.get(key, key)
            renamed_dict[new_key] = value
        res["prompt"] = renamed_dict
        del res["prompt_config"]
        new_dict = {"similarity_threshold": res["similarity_threshold"],
                    "keywords_similarity_weight": res["vector_similarity_weight"],
                    "top_n": res["top_n"],
                    "rerank_model": res['rerank_id']}
        res["prompt"].update(new_dict)
        for key in key_list:
            del res[key]
        res["llm"] = res.pop("llm_setting")
        res["llm"]["model_name"] = res.pop("llm_id")
        kb_list = []
        for kb_id in res["kb_ids"]:
            kb = KnowledgebaseService.query(id=kb_id)
            if not kb :
                return get_error_data_result(message=f"Don't exist the kb {kb_id}")
            kb_list.append(kb[0].to_json())
        del res["kb_ids"]
        res["datasets"] = kb_list
        res["avatar"] = res.pop("icon")
        list_assts.append(res)
    return get_result(data=list_assts)