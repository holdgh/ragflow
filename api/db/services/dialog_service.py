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
import binascii
import os
import json
import re
from copy import deepcopy
from timeit import default_timer as timer
import datetime
from datetime import timedelta
from api.db import LLMType, ParserType,StatusEnum
from api.db.db_models import Dialog, Conversation,DB
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMService, TenantLLMService, LLMBundle
from api import settings
from rag.app.resume import forbidden_select_fields4resume
from rag.nlp.search import index_name
from rag.utils import rmSpace, num_tokens_from_string, encoder
from api.utils.file_utils import get_project_base_directory


class DialogService(CommonService):
    model = Dialog

    @classmethod
    @DB.connection_context()
    def get_list(cls, tenant_id,
                 page_number, items_per_page, orderby, desc, id , name):
        chats = cls.model.select()
        if id:
            chats = chats.where(cls.model.id == id)
        if name:
            chats = chats.where(cls.model.name == name)
        chats = chats.where(
              (cls.model.tenant_id == tenant_id)
            & (cls.model.status == StatusEnum.VALID.value)
        )
        if desc:
            chats = chats.order_by(cls.model.getter_by(orderby).desc())
        else:
            chats = chats.order_by(cls.model.getter_by(orderby).asc())

        chats = chats.paginate(page_number, items_per_page)

        return list(chats.dicts())


class ConversationService(CommonService):
    model = Conversation

    @classmethod
    @DB.connection_context()
    def get_list(cls,dialog_id,page_number, items_per_page, orderby, desc, id , name):
        sessions = cls.model.select().where(cls.model.dialog_id ==dialog_id)
        if id:
            sessions = sessions.where(cls.model.id == id)
        if name:
            sessions = sessions.where(cls.model.name == name)
        if desc:
            sessions = sessions.order_by(cls.model.getter_by(orderby).desc())
        else:
            sessions = sessions.order_by(cls.model.getter_by(orderby).asc())

        sessions = sessions.paginate(page_number, items_per_page)

        return list(sessions.dicts())


def message_fit_in(msg, max_length=4000):
    def count():
        nonlocal msg
        tks_cnts = []
        for m in msg:
            tks_cnts.append(
                {"role": m["role"], "count": num_tokens_from_string(m["content"])})
        total = 0
        for m in tks_cnts:
            total += m["count"]
        return total

    c = count()
    if c < max_length:
        return c, msg

    msg_ = [m for m in msg[:-1] if m["role"] == "system"]
    msg_.append(msg[-1])
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg

    ll = num_tokens_from_string(msg_[0]["content"])
    l = num_tokens_from_string(msg_[-1]["content"])
    if ll / (ll + l) > 0.8:
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[:max_length - l])
        msg[0]["content"] = m
        return max_length, msg

    m = msg_[1]["content"]
    m = encoder.decode(encoder.encode(m)[:max_length - l])
    msg[1]["content"] = m
    return max_length, msg


def llm_id2llm_type(llm_id):
    llm_id = llm_id.split("@")[0]
    fnm = os.path.join(get_project_base_directory(), "conf")
    llm_factories = json.load(open(os.path.join(fnm, "llm_factories.json"), "r"))
    for llm_factory in llm_factories["factory_llm_infos"]:
        for llm in llm_factory["llm"]:
            if llm_id == llm["llm_name"]:
                return llm["model_type"].strip(",")[-1]


def chat(dialog, messages, stream=True, **kwargs):
    """
    功能：聊天对话，回答用户问题
    逻辑：
        1、获取问答逻辑所需数据
            - 获取大模型记录及max_tokens【输入和输出的总和】
            - 获取知识库和检索器
            - 获取文档id列表【附件列表】
            - 获取embedding模型和chat模型
            - 获取助理记录的提示配置【TODO tts模型是干啥的？】和知识库的解析配置【field_map字段】
        2、依据field_map字段，分情况问答处理：
            - 具备field_map字段时，TODO 执行基于SQL的检索回答逻辑
            - 不具备field_map字段时，执行基于检索器的检索回答逻辑
                1、助理记录提示配置的关键参数处理【不可选时，对话入参必传校验；可选时，从系统提示词中去除关键参数占位符】
                2、基于对话历史和多轮对话设置，进行问题重构。接着获取rerank模型
                3、基于助理记录提示配置的关键参数是否包含knowledge，分情况处理检索
                    - 关键参数不包含knowledge时，设置检索内容为空
                    - 关键参数包含knowledge时，进行以下处理：
                        1、对当前问题进行关键词提取【由助理记录提示配置中的keyword是否为true来决定】，并将关键词追加到当前问题中
                        2、基于--用户问题【可能含有关键字】、embedding模型、用户id列表、助理记录关联的知识库id列表及其他配置、附件列表、rerank模型--TODO 进行检索
                4、判断检索结果是否为空：
                    - 检索不到内容，基于助理记录提示配置的空回复，进行回答
                    - 检索内容非空：
                        1、基于检索内容和重构问题构造提示词，并进行关于模型token数量限制的处理，同步计算输出token限制数量。
                        2、基于chat模型和提示词【检索内容和用户问题】、对话历史【计算token数量限制时，考虑了对话历史】、助理记录大模型配置进行问答【区分流式和非流式】
    """
    # ===========获取大模型-start==========
    # 断言messages最后一个元素为用户问题
    assert messages[-1]["role"] == "user", "The last content of this conversation is not from user."
    st = timer()
    # 获取大模型记录
    tmp = dialog.llm_id.split("@")
    fid = None
    llm_id = tmp[0]
    if len(tmp)>1: fid = tmp[1]

    llm = LLMService.query(llm_name=llm_id) if not fid else LLMService.query(llm_name=llm_id, fid=fid)
    if not llm:
        llm = TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id) if not fid else \
            TenantLLMService.query(tenant_id=dialog.tenant_id, llm_name=llm_id, llm_factory=fid)
        if not llm:
            raise LookupError("LLM(%s) not found" % dialog.llm_id)
        # 默认最大token数
        max_tokens = 8192
    else:
        max_tokens = llm[0].max_tokens
    # ===========获取大模型-end==========
    # ===========获取知识库和检索器-start==========
    # 获取知识库记录
    kbs = KnowledgebaseService.get_by_ids(dialog.kb_ids)
    # 要求所选知识库使用统一的embedding模型
    embd_nms = list(set([kb.embd_id for kb in kbs]))
    if len(embd_nms) != 1:
        yield {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}
        return {"answer": "**ERROR**: Knowledge bases use different embedding models.", "reference": []}
    # 根据知识库的解析类型获取检索器
    is_kg = all([kb.parser_id == ParserType.KG for kb in kbs])
    retr = settings.retrievaler if not is_kg else settings.kg_retrievaler
    # ===========获取知识库和检索器-end==========
    # ===========获取附件列表【文档记录id列表】-start==========
    # 取包含当前问题和2个历史对话的问题 TODO 【这个放的位置有些靠前，当前系统代码有一些地方于此类似，可优化】
    questions = [m["content"] for m in messages if m["role"] == "user"][-3:]
    # 取对话入参中的文档记录id列表赋予附件列表
    attachments = kwargs["doc_ids"].split(",") if "doc_ids" in kwargs else None
    if "doc_ids" in messages[-1]:
        # TODO 用户问题含有doc_ids字段时
        # 将用户问题的doc_ids字段值赋予attachments【附件列表】
        attachments = messages[-1]["doc_ids"]
        # 收集其他message中的附件数据
        for m in messages[:-1]:
            if "doc_ids" in m:
                attachments.extend(m["doc_ids"])
    # ===========获取附件列表【文档记录id列表】-end==========
    # ===========获取embedding模型和chat模型-start==========
    # 获取embedding模型
    embd_mdl = LLMBundle(dialog.tenant_id, LLMType.EMBEDDING, embd_nms[0])
    if not embd_mdl:
        raise LookupError("Embedding model(%s) not found" % embd_nms[0])
    # 获取chat模型
    if llm_id2llm_type(dialog.llm_id) == "image2text":
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.IMAGE2TEXT, dialog.llm_id)
    else:
        chat_mdl = LLMBundle(dialog.tenant_id, LLMType.CHAT, dialog.llm_id)
    # ===========获取embedding模型和chat模型-end==========
    # ===========获取助理记录中的提示配置【提示配置中含有tts项时获取tts模型】和所选知识库解析配置中的filed_map-start==========
    # 获取助理记录的提示配置数据
    prompt_config = dialog.prompt_config
    # 获取所选知识库解析配置parser_config中的filed_map数据
    field_map = KnowledgebaseService.get_field_map(dialog.kb_ids)
    tts_mdl = None
    if prompt_config.get("tts"):
        # 助理记录提示配置中含有tts项时，获取tts模型
        tts_mdl = LLMBundle(dialog.tenant_id, LLMType.TTS)
    # ===========获取助理记录中的提示配置【提示配置中含有tts项时获取tts模型】和所选知识库解析配置中的filed_map-end==========
    # ===========field_map非空时，采用SQL检索基于chat模型回答问题-start==========
    # try to use sql if field mapping is good to go
    if field_map:
        # 所选知识库解析配置中含有filed_map数据时，采用SQL检索数据，基于chat模型回答问题-
        logging.debug("Use SQL to retrieval:{}".format(questions[-1]))
        ans = use_sql(questions[-1], field_map, dialog.tenant_id, chat_mdl, prompt_config.get("quote", True))
        if ans:
            yield ans
            return
    # ===========field_map非空时，采用SQL检索基于chat模型回答问题-end==========
    # ======助理记录提示配置的关键参数处理【不可选时，对话入参必传校验；可选时，从系统提示词中去除关键参数占位符】-start======
    # 遍历处理助理记录的提示配置关键参数
    for p in prompt_config["parameters"]:
        if p["key"] == "knowledge":
            # 跳过关键参数knowledge
            continue
        # 要求提示配置中的关键参数字段必须【不可选时，必传】在对话入参中有所传值
        if p["key"] not in kwargs and not p["optional"]:
            raise KeyError("Miss parameter: " + p["key"])
        if p["key"] not in kwargs:
            # 去除提示配置中系统提示词中的可选关键参数
            prompt_config["system"] = prompt_config["system"].replace(
                "{%s}" % p["key"], " ")
    # ======助理记录提示配置的关键参数处理【不可选时，对话入参必传校验；可选时，从系统提示词中去除关键参数占位符】-end======
    # ========基于对话历史和多轮对话设置，进行问题重构-start========
    if len(questions) > 1 and prompt_config.get("refine_multiturn"):
        # 问题列表超过1个问题【说明有对话历史存在】且助理支持多轮对话时，基于对话历史，对当前问题进行重构，生成语义独立的新问题
        questions = [full_question(dialog.tenant_id, dialog.llm_id, messages)]
    else:
        # 助理不支持多轮对话或问题列表只有一个时，只取当前问题
        questions = questions[-1:]
    # ========基于对话历史和多轮对话设置，进行问题重构-end========
    refineQ_tm = timer()
    keyword_tm = timer()
    # 获取rerank模型
    rerank_mdl = None
    if dialog.rerank_id:
        rerank_mdl = LLMBundle(dialog.tenant_id, LLMType.RERANK, dialog.rerank_id)
    # TODO 经过重构问题后，这里是干啥？
    for _ in range(len(questions) // 2):
        questions.append(questions[-1])
    # ============基于助理记录提示配置的关键参数是否包含knowledge，分情况处理检索-start=============
    if "knowledge" not in [p["key"] for p in prompt_config["parameters"]]:
        # 助理记录提示配置的关键参数列表中不含knowledge时，设置知识库检索结果为空
        kbinfos = {"total": 0, "chunks": [], "doc_aggs": []}
    else:
        # 助理记录提示配置的关键参数列表中有knowledge时
        # 对于用户问题进行关键字提取
        if prompt_config.get("keyword", False):
            # TODO 官方demo环境中，助理记录的提示配置字段没有keyword字段
            # 助理记录提示配置中设置了keyword为true时，对问题进行关键字提取，并追加到问题【这里可以理解为检索知识库的查询条件】列表中
            questions[-1] += keyword_extraction(chat_mdl, questions[-1])
            keyword_tm = timer()
        # 对知识库的用户id做去重
        tenant_ids = list(set([kb.tenant_id for kb in kbs]))
        # 基于--用户问题【可能含有关键字】、embedding模型、用户id列表、助理记录关联的知识库id列表及其他配置、附件列表、rerank模型--进行检索
        kbinfos = retr.retrieval(" ".join(questions), embd_mdl, tenant_ids, dialog.kb_ids, 1, dialog.top_n,
                                        dialog.similarity_threshold,
                                        dialog.vector_similarity_weight,
                                        doc_ids=attachments,
                                        top=dialog.top_k, aggs=False, rerank_mdl=rerank_mdl)
    # ============基于助理记录提示配置的关键参数是否包含knowledge，分情况处理检索-end=============
    # ============检索不到内容时的空回复-start=============
    # 依据检索结果构造知识库检索内容列表
    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]
    logging.debug(
        "{}->{}".format(" ".join(questions), "\n->".join(knowledges)))
    retrieval_tm = timer()
    # 检索不到内容且助理记录提示配置中设置了空响应，则回复空响应设置内容
    # 官方demo环境对空响应的解释：如果在知识库中没有检索到用户的问题，它将使用它作为答案。 如果您希望 LLM 在未检索到任何内容时提出自己的意见，请将此留空。
    if not knowledges and prompt_config.get("empty_response"):
        empty_res = prompt_config["empty_response"]
        yield {"answer": empty_res, "reference": kbinfos, "audio_binary": tts(tts_mdl, empty_res)}
        return {"answer": prompt_config["empty_response"], "reference": kbinfos}
    # ============检索不到内容时的空回复-end=============
    # =========基于检索内容和重构问题构造提示词，并进行关于模型token数量限制的处理，同步计算输出token限制数量-start==========
    # 将检索内容放入对话入参中的knowledge字段【这里就是关键参数knowledge的作用】
    kwargs["knowledge"] = "\n\n------\n\n".join(knowledges)
    # 获取助理记录中的大模型设置
    gen_conf = dialog.llm_setting
    # 基于检索结果构造系统提示词，追加到chat模型输入中
    msg = [{"role": "system", "content": prompt_config["system"].format(**kwargs)}]
    # 追加对话历史和当前问题到chat大模型输入中
    msg.extend([{"role": m["role"], "content": re.sub(r"##\d+\$\$", "", m["content"])}
                for m in messages if m["role"] != "system"])
    # 对于chat模型输入做基于最大token【97%占比】数量限制的适配处理
    used_token_count, msg = message_fit_in(msg, int(max_tokens * 0.97))
    assert len(msg) >= 2, f"message_fit_in has bug: {msg}"
    # 构造提示词【考虑对msg的截取处理，这里对截取的内容做了重新收集。这里保留了完整的当前用户问题，说明截取操作避开了对当前用户问题的处理】
    # 系统提示词
    prompt = msg[0]["content"]
    # 用户问题【重构后的】
    prompt += "\n\n### Query:\n%s" % " ".join(questions)
    # 设置chat模型生成答案的最大token数量【取【助理记录大模型配置的生成最大token数】与【大模型最大token数-输入已用token数】的最小值】
    if "max_tokens" in gen_conf:
        gen_conf["max_tokens"] = min(
            gen_conf["max_tokens"],
            max_tokens - used_token_count)
    # =========基于检索内容和重构问题构造提示词，并进行关于模型token数量限制的处理，同步计算输出token限制数量-end==========
    def decorate_answer(answer):
        nonlocal prompt_config, knowledges, kwargs, kbinfos, prompt, retrieval_tm
        refs = []
        if knowledges and (prompt_config.get("quote", True) and kwargs.get("quote", True)):
            answer, idx = retr.insert_citations(answer,
                                                       [ck["content_ltks"]
                                                        for ck in kbinfos["chunks"]],
                                                       [ck["vector"]
                                                        for ck in kbinfos["chunks"]],
                                                       embd_mdl,
                                                       tkweight=1 - dialog.vector_similarity_weight,
                                                       vtweight=dialog.vector_similarity_weight)
            idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
            recall_docs = [
                d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
            if not recall_docs: recall_docs = kbinfos["doc_aggs"]
            kbinfos["doc_aggs"] = recall_docs

            refs = deepcopy(kbinfos)
            for c in refs["chunks"]:
                if c.get("vector"):
                    del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        done_tm = timer()
        prompt += "\n\n### Elapsed\n  - Refine Question: %.1f ms\n  - Keywords: %.1f ms\n  - Retrieval: %.1f ms\n  - LLM: %.1f ms" % (
            (refineQ_tm - st) * 1000, (keyword_tm - refineQ_tm) * 1000, (retrieval_tm - keyword_tm) * 1000,
            (done_tm - retrieval_tm) * 1000)
        return {"answer": answer, "reference": refs, "prompt": prompt}

    # 基于chat模型和提示词【检索内容和用户问题】、对话历史、助理记录大模型配置进行问答【区分流式和非流式】
    if stream:
        last_ans = ""
        answer = ""
        for ans in chat_mdl.chat_streamly(prompt, msg[1:], gen_conf):
            answer = ans
            delta_ans = ans[len(last_ans):]
            if num_tokens_from_string(delta_ans) < 16:
                continue
            last_ans = answer
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
        delta_ans = answer[len(last_ans):]
        if delta_ans:
            yield {"answer": answer, "reference": {}, "audio_binary": tts(tts_mdl, delta_ans)}
        yield decorate_answer(answer)
    else:
        answer = chat_mdl.chat(prompt, msg[1:], gen_conf)
        logging.debug("User: {}|Assistant: {}".format(
            msg[-1]["content"], answer))
        res = decorate_answer(answer)
        res["audio_binary"] = tts(tts_mdl, answer)
        yield res


def use_sql(question, field_map, tenant_id, chat_mdl, quota=True):
    sys_prompt = "你是一个DBA。你需要这对以下表的字段结构，根据用户的问题列表，写出最后一个问题对应的SQL。"
    user_promt = """
表名：{}；
数据库表字段说明如下：
{}

问题如下：
{}
请写出SQL, 且只要SQL，不要有其他说明及文字。
""".format(
        index_name(tenant_id),
        "\n".join([f"{k}: {v}" for k, v in field_map.items()]),
        question
    )
    tried_times = 0

    def get_table():
        nonlocal sys_prompt, user_promt, question, tried_times
        sql = chat_mdl.chat(sys_prompt, [{"role": "user", "content": user_promt}], {
            "temperature": 0.06})
        logging.debug(f"{question} ==> {user_promt} get SQL: {sql}")
        sql = re.sub(r"[\r\n]+", " ", sql.lower())
        sql = re.sub(r".*select ", "select ", sql.lower())
        sql = re.sub(r" +", " ", sql)
        sql = re.sub(r"([;；]|```).*", "", sql)
        if sql[:len("select ")] != "select ":
            return None, None
        if not re.search(r"((sum|avg|max|min)\(|group by )", sql.lower()):
            if sql[:len("select *")] != "select *":
                sql = "select doc_id,docnm_kwd," + sql[6:]
            else:
                flds = []
                for k in field_map.keys():
                    if k in forbidden_select_fields4resume:
                        continue
                    if len(flds) > 11:
                        break
                    flds.append(k)
                sql = "select doc_id,docnm_kwd," + ",".join(flds) + sql[8:]

        logging.debug(f"{question} get SQL(refined): {sql}")
        tried_times += 1
        return settings.retrievaler.sql_retrieval(sql, format="json"), sql

    tbl, sql = get_table()
    if tbl is None:
        return None
    if tbl.get("error") and tried_times <= 2:
        user_promt = """
        表名：{}；
        数据库表字段说明如下：
        {}

        问题如下：
        {}

        你上一次给出的错误SQL如下：
        {}

        后台报错如下：
        {}

        请纠正SQL中的错误再写一遍，且只要SQL，不要有其他说明及文字。
        """.format(
            index_name(tenant_id),
            "\n".join([f"{k}: {v}" for k, v in field_map.items()]),
            question, sql, tbl["error"]
        )
        tbl, sql = get_table()
        logging.debug("TRY it again: {}".format(sql))

    logging.debug("GET table: {}".format(tbl))
    if tbl.get("error") or len(tbl["rows"]) == 0:
        return None

    docid_idx = set([ii for ii, c in enumerate(
        tbl["columns"]) if c["name"] == "doc_id"])
    docnm_idx = set([ii for ii, c in enumerate(
        tbl["columns"]) if c["name"] == "docnm_kwd"])
    clmn_idx = [ii for ii in range(
        len(tbl["columns"])) if ii not in (docid_idx | docnm_idx)]

    # compose markdown table
    clmns = "|" + "|".join([re.sub(r"(/.*|（[^（）]+）)", "", field_map.get(tbl["columns"][i]["name"],
                                                                        tbl["columns"][i]["name"])) for i in
                            clmn_idx]) + ("|Source|" if docid_idx and docid_idx else "|")

    line = "|" + "|".join(["------" for _ in range(len(clmn_idx))]) + \
           ("|------|" if docid_idx and docid_idx else "")

    rows = ["|" +
            "|".join([rmSpace(str(r[i])) for i in clmn_idx]).replace("None", " ") +
            "|" for r in tbl["rows"]]
    rows = [r for r in rows if re.sub(r"[ |]+", "", r)]
    if quota:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    else:
        rows = "\n".join([r + f" ##{ii}$$ |" for ii, r in enumerate(rows)])
    rows = re.sub(r"T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+Z)?\|", "|", rows)

    if not docid_idx or not docnm_idx:
        logging.warning("SQL missing field: " + sql)
        return {
            "answer": "\n".join([clmns, line, rows]),
            "reference": {"chunks": [], "doc_aggs": []},
            "prompt": sys_prompt
        }

    docid_idx = list(docid_idx)[0]
    docnm_idx = list(docnm_idx)[0]
    doc_aggs = {}
    for r in tbl["rows"]:
        if r[docid_idx] not in doc_aggs:
            doc_aggs[r[docid_idx]] = {"doc_name": r[docnm_idx], "count": 0}
        doc_aggs[r[docid_idx]]["count"] += 1
    return {
        "answer": "\n".join([clmns, line, rows]),
        "reference": {"chunks": [{"doc_id": r[docid_idx], "docnm_kwd": r[docnm_idx]} for r in tbl["rows"]],
                      "doc_aggs": [{"doc_id": did, "doc_name": d["doc_name"], "count": d["count"]} for did, d in
                                   doc_aggs.items()]},
        "prompt": sys_prompt
    }


def relevant(tenant_id, llm_id, question, contents: list):
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    prompt = """
        You are a grader assessing relevance of a retrieved document to a user question. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        No other words needed except 'yes' or 'no'.
    """
    if not contents:return False
    contents = "Documents: \n" + "   - ".join(contents)
    contents = f"Question: {question}\n" + contents
    if num_tokens_from_string(contents) >= chat_mdl.max_length - 4:
        contents = encoder.decode(encoder.encode(contents)[:chat_mdl.max_length - 4])
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": contents}], {"temperature": 0.01})
    if ans.lower().find("yes") >= 0: return True
    return False


def rewrite(tenant_id, llm_id, question):
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    prompt = """
        You are an expert at query expansion to generate a paraphrasing of a question.
        I can't retrieval relevant information from the knowledge base by using user's question directly.     
        You need to expand or paraphrase user's question by multiple ways such as using synonyms words/phrase, 
        writing the abbreviation in its entirety, adding some extra descriptions or explanations, 
        changing the way of expression, translating the original question into another language (English/Chinese), etc. 
        And return 5 versions of question and one is from translation.
        Just list the question. No other words are needed.
    """
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": question}], {"temperature": 0.8})
    return ans


def keyword_extraction(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer. 
Task: extract the most important keywords/phrases of a given piece of text content.
Requirements: 
  - Summarize the text content, and give top {topn} important keywords/phrases.
  - The keywords MUST be in language of the given piece of text content.
  - The keywords are delimited by ENGLISH COMMA.
  - Keywords ONLY in output.

### Text Content 
{content}

"""
    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Output: "}
    ]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple): kwd = kwd[0]
    if kwd.find("**ERROR**") >=0: return ""
    return kwd


def question_proposal(chat_mdl, content, topn=3):
    prompt = f"""
Role: You're a text analyzer. 
Task:  propose {topn} questions about a given piece of text content.
Requirements: 
  - Understand and summarize the text content, and propose top {topn} important questions.
  - The questions SHOULD NOT have overlapping meanings.
  - The questions SHOULD cover the main content of the text as much as possible.
  - The questions MUST be in language of the given piece of text content.
  - One question per line.
  - Question ONLY in output.

### Text Content 
{content}

"""
    msg = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Output: "}
    ]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = chat_mdl.chat(prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple): kwd = kwd[0]
    if kwd.find("**ERROR**") >= 0: return ""
    return kwd


def full_question(tenant_id, llm_id, messages):
    """
    功能：基于对话历史，对当前问题进行重构，生成语义独立的新问题【有异常时，返回当前用户问题即可】
    """
    # 依据llm_id选择chat模型
    if llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    conv = []
    for m in messages:
        # 过滤掉非用户和非助理message【收集用户和助理message】
        if m["role"] not in ["user", "assistant"]: continue
        conv.append("{}: {}".format(m["role"].upper(), m["content"]))
    conv = "\n".join(conv)
    # 今天、昨天和明天
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + timedelta(days=1)).isoformat()
    prompt = f"""
Role: A helpful assistant

Task and steps: 
    1. Generate a full user question that would follow the conversation.
    2. If the user's question involves relative date, you need to convert it into absolute date based on the current date, which is {today}. For example: 'yesterday' would be converted to {yesterday}.
    
Requirements & Restrictions:
  - Text generated MUST be in the same language of the original user's question.
  - If the user's latest question is completely, don't do anything, just return the original question.
  - DON'T generate anything except a refined question.

######################
-Examples-
######################

# Example 1
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
###############
Output: What's the name of Donald Trump's mother?

------------
# Example 2
## Conversation
USER: What is the name of Donald Trump's father?
ASSISTANT:  Fred Trump.
USER: And his mother?
ASSISTANT:  Mary Trump.
User: What's her full name?
###############
Output: What's the full name of Donald Trump's mother Mary Trump?

------------
# Example 3
## Conversation
USER: What's the weather today in London?
ASSISTANT:  Cloudy.
USER: What's about tomorrow in Rochester?
###############
Output: What's the weather in Rochester on {tomorrow}?
######################

# Real Data
## Conversation
{conv}
###############
    """
    # 注意上述提示词中的conv位置和example内容，意为：基于对话历史，对当前问题进行重构，生成语义独立的问题
    # 利用chat模型回答问题
    ans = chat_mdl.chat(prompt, [{"role": "user", "content": "Output: "}], {"temperature": 0.2})
    # chat模型回答中没有error时返回答案；有error时，返回用户当前问题
    return ans if ans.find("**ERROR**") < 0 else messages[-1]["content"]


def tts(tts_mdl, text):
    if not tts_mdl or not text: return
    bin = b""
    for chunk in tts_mdl.tts(text):
        bin += chunk
    return binascii.hexlify(bin).decode("utf-8")


def ask(question, kb_ids, tenant_id):
    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    embd_nms = list(set([kb.embd_id for kb in kbs]))

    is_kg = all([kb.parser_id == ParserType.KG for kb in kbs])
    retr = settings.retrievaler if not is_kg else settings.kg_retrievaler

    embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embd_nms[0])
    chat_mdl = LLMBundle(tenant_id, LLMType.CHAT)
    max_tokens = chat_mdl.max_length

    kbinfos = retr.retrieval(question, embd_mdl, tenant_id, kb_ids, 1, 12, 0.1, 0.3, aggs=False)
    knowledges = [ck["content_with_weight"] for ck in kbinfos["chunks"]]

    used_token_count = 0
    for i, c in enumerate(knowledges):
        used_token_count += num_tokens_from_string(c)
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            break

    prompt = """
    Role: You're a smart assistant. Your name is Miss R.
    Task: Summarize the information from knowledge bases and answer user's question.
    Requirements and restriction:
      - DO NOT make things up, especially for numbers.
      - If the information from knowledge is irrelevant with user's question, JUST SAY: Sorry, no relevant information provided.
      - Answer with markdown format text.
      - Answer in language of user's question.
      - DO NOT make things up, especially for numbers.
      
    ### Information from knowledge bases
    %s
    
    The above is information from knowledge bases.
     
    """%"\n".join(knowledges)
    msg = [{"role": "user", "content": question}]

    def decorate_answer(answer):
        nonlocal knowledges, kbinfos, prompt
        answer, idx = retr.insert_citations(answer,
                                           [ck["content_ltks"]
                                            for ck in kbinfos["chunks"]],
                                           [ck["vector"]
                                            for ck in kbinfos["chunks"]],
                                           embd_mdl,
                                           tkweight=0.7,
                                           vtweight=0.3)
        idx = set([kbinfos["chunks"][int(i)]["doc_id"] for i in idx])
        recall_docs = [
            d for d in kbinfos["doc_aggs"] if d["doc_id"] in idx]
        if not recall_docs: recall_docs = kbinfos["doc_aggs"]
        kbinfos["doc_aggs"] = recall_docs
        refs = deepcopy(kbinfos)
        for c in refs["chunks"]:
            if c.get("vector"):
                del c["vector"]

        if answer.lower().find("invalid key") >= 0 or answer.lower().find("invalid api") >= 0:
            answer += " Please set LLM API-Key in 'User Setting -> Model Providers -> API-Key'"
        return {"answer": answer, "reference": refs}

    answer = ""
    for ans in chat_mdl.chat_streamly(prompt, msg, {"temperature": 0.1}):
        answer = ans
        yield {"answer": answer, "reference": {}}
    yield decorate_answer(answer)

