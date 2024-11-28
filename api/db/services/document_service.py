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
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from io import BytesIO

from peewee import fn

from api.db.db_utils import bulk_insert_into_db
from api import settings
from api.utils import current_timestamp, get_format_time, get_uuid
from graphrag.mind_map_extractor import MindMapExtractor
from rag.settings import SVR_QUEUE_NAME
from rag.utils.storage_factory import STORAGE_IMPL
from rag.nlp import search, rag_tokenizer

from api.db import FileType, TaskStatus, ParserType, LLMType
from api.db.db_models import DB, Knowledgebase, Tenant, Task, UserTenant
from api.db.db_models import Document
from api.db.services.common_service import CommonService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db import StatusEnum
from rag.utils.redis_conn import REDIS_CONN


class DocumentService(CommonService):
    model = Document

    @classmethod
    @DB.connection_context()
    def get_list(cls, kb_id, page_number, items_per_page,
                 orderby, desc, keywords, id, name):
        docs = cls.model.select().where(cls.model.kb_id == kb_id)
        if id:
            docs = docs.where(
                cls.model.id == id)
        if name:
            docs = docs.where(
                cls.model.name == name
            )
        if keywords:
            docs = docs.where(
                fn.LOWER(cls.model.name).contains(keywords.lower())
            )
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        docs = docs.paginate(page_number, items_per_page)
        count = docs.count()
        return list(docs.dicts()), count

    @classmethod
    @DB.connection_context()
    def get_by_kb_id(cls, kb_id, page_number, items_per_page,
                     orderby, desc, keywords):
        if keywords:
            docs = cls.model.select().where(
                (cls.model.kb_id == kb_id),
                (fn.LOWER(cls.model.name).contains(keywords.lower()))
            )
        else:
            docs = cls.model.select().where(cls.model.kb_id == kb_id)
        count = docs.count()
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        docs = docs.paginate(page_number, items_per_page)

        return list(docs.dicts()), count

    @classmethod
    @DB.connection_context()
    def insert(cls, doc):
        if not cls.save(**doc):
            raise RuntimeError("Database error (Document)!")
        e, doc = cls.get_by_id(doc["id"])
        if not e:
            raise RuntimeError("Database error (Document retrieval)!")
        e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
        if not KnowledgebaseService.update_by_id(
                kb.id, {"doc_num": kb.doc_num + 1}):
            raise RuntimeError("Database error (Knowledgebase)!")
        return doc

    @classmethod
    @DB.connection_context()
    def remove_document(cls, doc, tenant_id):
        settings.docStoreConn.delete({"doc_id": doc.id}, search.index_name(tenant_id), doc.kb_id)
        cls.clear_chunk_num(doc.id)
        return cls.delete_by_id(doc.id)

    @classmethod
    @DB.connection_context()
    def get_newly_uploaded(cls):
        fields = [
            cls.model.id,
            cls.model.kb_id,
            cls.model.parser_id,
            cls.model.parser_config,
            cls.model.name,
            cls.model.type,
            cls.model.location,
            cls.model.size,
            Knowledgebase.tenant_id,
            Tenant.embd_id,
            Tenant.img2txt_id,
            Tenant.asr_id,
            cls.model.update_time]
        docs = cls.model.select(*fields) \
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id)) \
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id)) \
            .where(
            cls.model.status == StatusEnum.VALID.value,
            ~(cls.model.type == FileType.VIRTUAL.value),
            cls.model.progress == 0,
            cls.model.update_time >= current_timestamp() - 1000 * 600,
            cls.model.run == TaskStatus.RUNNING.value) \
            .order_by(cls.model.update_time.asc())
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def get_unfinished_docs(cls):
        """
        查询未处理完成文档信息
        条件：
            - status=='1'
            - type!='virtual'
            - 0<process<1
        问题：在上传文件操作中，初始化文档信息时，数据表document的process字段默认值为0【因此不在ragflow_server中update_process的处理范围之内】，status字段默认值为'1'，run字段默认值为'0'。那么，在什么时候文档满足未处理完成的查询条件呢？注意ragflow系统，文档解析是手动触发的，也即有专门的接口触发文档解析操作，推测生成了任务.答：我的推测正确，在手动触发解析时，最终更新文档记录的progress为0-1之间的正小数
        """
        fields = [cls.model.id, cls.model.process_begin_at, cls.model.parser_config, cls.model.progress_msg,
                  cls.model.run]
        docs = cls.model.select(*fields) \
            .where(
            cls.model.status == StatusEnum.VALID.value,
            ~(cls.model.type == FileType.VIRTUAL.value),
            cls.model.progress < 1,
            cls.model.progress > 0)
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def increment_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duation):
        # 将文档解析数据维护到文档记录中
        num = cls.model.update(token_num=cls.model.token_num + token_num,
                               chunk_num=cls.model.chunk_num + chunk_num,
                               process_duation=cls.model.process_duation + duation).where(
            cls.model.id == doc_id).execute()
        if num == 0:
            raise LookupError(
                "Document not found which is supposed to be there")
        # 将文档解析数据更新知识库中
        num = Knowledgebase.update(
            token_num=Knowledgebase.token_num +
                      token_num,
            chunk_num=Knowledgebase.chunk_num +
                      chunk_num).where(
            Knowledgebase.id == kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def decrement_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duation):
        num = cls.model.update(token_num=cls.model.token_num - token_num,
                               chunk_num=cls.model.chunk_num - chunk_num,
                               process_duation=cls.model.process_duation + duation).where(
            cls.model.id == doc_id).execute()
        if num == 0:
            raise LookupError(
                "Document not found which is supposed to be there")
        num = Knowledgebase.update(
            token_num=Knowledgebase.token_num -
                      token_num,
            chunk_num=Knowledgebase.chunk_num -
                      chunk_num
        ).where(
            Knowledgebase.id == kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def clear_chunk_num(cls, doc_id):
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't fine document in database."

        num = Knowledgebase.update(
            token_num=Knowledgebase.token_num -
                      doc.token_num,
            chunk_num=Knowledgebase.chunk_num -
                      doc.chunk_num,
            doc_num=Knowledgebase.doc_num - 1
        ).where(
            Knowledgebase.id == doc.kb_id).execute()
        return num

    @classmethod
    @DB.connection_context()
    def get_tenant_id(cls, doc_id):
        docs = cls.model.select(
            Knowledgebase.tenant_id).join(
            Knowledgebase, on=(
                    Knowledgebase.id == cls.model.kb_id)).where(
            cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def get_knowledgebase_id(cls, doc_id):
        docs = cls.model.select(cls.model.kb_id).where(cls.model.id == doc_id)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["kb_id"]

    @classmethod
    @DB.connection_context()
    def get_tenant_id_by_name(cls, name):
        docs = cls.model.select(
            Knowledgebase.tenant_id).join(
            Knowledgebase, on=(
                    Knowledgebase.id == cls.model.kb_id)).where(
            cls.model.name == name, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def accessible(cls, doc_id, user_id):
        docs = cls.model.select(
            cls.model.id).join(
            Knowledgebase, on=(
                    Knowledgebase.id == cls.model.kb_id)
        ).join(UserTenant, on=(UserTenant.tenant_id == Knowledgebase.tenant_id)
               ).where(cls.model.id == doc_id, UserTenant.user_id == user_id).paginate(0, 1)
        docs = docs.dicts()
        if not docs:
            return False
        return True

    @classmethod
    @DB.connection_context()
    def accessible4deletion(cls, doc_id, user_id):
        docs = cls.model.select(
            cls.model.id).join(
            Knowledgebase, on=(
                    Knowledgebase.id == cls.model.kb_id)
        ).where(cls.model.id == doc_id, Knowledgebase.created_by == user_id).paginate(0, 1)
        docs = docs.dicts()
        if not docs:
            return False
        return True

    @classmethod
    @DB.connection_context()
    def get_embd_id(cls, doc_id):
        docs = cls.model.select(
            Knowledgebase.embd_id).join(
            Knowledgebase, on=(
                    Knowledgebase.id == cls.model.kb_id)).where(
            cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return
        return docs[0]["embd_id"]

    @classmethod
    @DB.connection_context()
    def get_doc_id_by_doc_name(cls, doc_name):
        fields = [cls.model.id]
        doc_id = cls.model.select(*fields) \
            .where(cls.model.name == doc_name)
        doc_id = doc_id.dicts()
        if not doc_id:
            return
        return doc_id[0]["id"]

    @classmethod
    @DB.connection_context()
    def get_thumbnails(cls, docids):
        fields = [cls.model.id, cls.model.kb_id, cls.model.thumbnail]
        return list(cls.model.select(
            *fields).where(cls.model.id.in_(docids)).dicts())

    @classmethod
    @DB.connection_context()
    def update_parser_config(cls, id, config):
        e, d = cls.get_by_id(id)
        if not e:
            raise LookupError(f"Document({id}) not found.")

        def dfs_update(old, new):
            for k, v in new.items():
                if k not in old:
                    old[k] = v
                    continue
                if isinstance(v, dict):
                    assert isinstance(old[k], dict)
                    dfs_update(old[k], v)
                else:
                    old[k] = v

        dfs_update(d.parser_config, config)
        cls.update_by_id(id, {"parser_config": d.parser_config})

    @classmethod
    @DB.connection_context()
    def get_doc_count(cls, tenant_id):
        docs = cls.model.select(cls.model.id).join(Knowledgebase,
                                                   on=(Knowledgebase.id == cls.model.kb_id)).where(
            Knowledgebase.tenant_id == tenant_id)
        return len(docs)

    @classmethod
    @DB.connection_context()
    def begin2parse(cls, docid):
        # 更新文档记录：progress【随机正小数0.0****， 0-1之间】、progress_msg、process_begin_at
        cls.update_by_id(
            docid, {"progress": random.random() * 1 / 100.,
                    "progress_msg": "Task is queued...",
                    "process_begin_at": get_format_time()
                    })

    @classmethod
    @DB.connection_context()
    def update_progress(cls):
        """
        1、查询未处理完成的文档列表
        2、逐一处理：
            - 1、查询当前文档任务列表，为空则处理下一个文档
            - 2、遍历任务列表，统计完成状态finished、prg【处理成功数量？】,处理信息列表msg和处理失败数量bad
            - 3、根据任务列表上述统计字段，更新文档处理相关字段：process_duation、run、progress、progress_msg
        思考：
            1、【在task_executor处理任务后，不是有对progress的更新吗？task_executor中更新的是任务记录的progress】为什么要有专门的progress更新操作？对于涉及生成raptor任务的doc记录而言，这里更新了doc记录的progress_msg字段【含有'------ RAPTOR -------'】
        """
        # 查询未处理完成的文档信息
        docs = cls.get_unfinished_docs()
        for d in docs:
            try:
                # 查询数据库任务信息
                tsks = Task.query(doc_id=d["id"], order_by=Task.create_time)
                if not tsks:
                    # 任务信息为空，则处理下一个文档
                    continue
                msg = []
                prg = 0
                finished = True
                bad = 0
                # 查询当前文档信息
                e, doc = DocumentService.get_by_id(d["id"])
                status = doc.run  # TaskStatus.RUNNING.value
                # 遍历当前文档的任务列表，统计完成状态finished、prg【收集任务处理进度？该指标不是整数】,处理信息列表和处理失败数量
                for t in tsks:
                    if 0 <= t.progress < 1:
                        # 0 表示未处理完成
                        finished = False
                    # 不小于0，则累计；否则不累计
                    prg += t.progress if t.progress >= 0 else 0
                    # 收集处理信息，去重
                    if t.progress_msg not in msg:
                        msg.append(t.progress_msg)
                    # 收集处理失败数量，-1表示处理失败
                    if t.progress == -1:
                        bad += 1
                prg /= len(tsks)
                if finished and bad:
                    # 若当前文档处理完成且存在处理失败的任务，则将prg置为-1，并将文档状态run字段置为4【任务失败】
                    prg = -1
                    status = TaskStatus.FAIL.value
                elif finished:
                    # 若处理完成且不存在失败任务【说明只有当非raptor任务处理完成时，才会生成raptor任务】
                    if d["parser_config"].get("raptor", {}).get("use_raptor") and d["progress_msg"].lower().find(
                            " raptor") < 0:
                        # 若文档parser_config字段的raptor字段存在非空字段use_raptor且文档progress_msg字段的小写模式不存在raptor字符串，则：
                        # 1、创建指定文档的“组织树检索的递归抽象处理”任务，并插入数据库，插入redis消息队列rag_flow_svr_queue，redis插入失败，则抛出异常
                        # 2、重新设置文档的prg值，并在文档progress_msg字段追加"------ RAPTOR -------"，表示文档存在raptor任务
                        queue_raptor_tasks(d)
                        prg = 0.98 * len(tsks) / (len(tsks) + 1)
                        msg.append("------ RAPTOR -------")
                    else:
                        # 若文档parser_config字段的raptor字段不存在非空字段use_raptor或文档progress_msg字段的小写模式不存在raptor字符串，则将文档状态run字段置为3【任务完成】
                        status = TaskStatus.DONE.value
                # 更新文档处理相关字段：process_duation、run、progress、progress_msg
                msg = "\n".join(msg)
                info = {
                    "process_duation": datetime.timestamp(
                        datetime.now()) -
                                       d["process_begin_at"].timestamp(),
                    "run": status}
                if prg != 0:
                    info["progress"] = prg
                if msg:
                    info["progress_msg"] = msg
                cls.update_by_id(d["id"], info)
            except Exception as e:
                if str(e).find("'0'") < 0:
                    logging.exception("fetch task exception")

    @classmethod
    @DB.connection_context()
    def get_kb_doc_count(cls, kb_id):
        return len(cls.model.select(cls.model.id).where(
            cls.model.kb_id == kb_id).dicts())

    @classmethod
    @DB.connection_context()
    def do_cancel(cls, doc_id):
        try:
            _, doc = DocumentService.get_by_id(doc_id)
            return doc.run == TaskStatus.CANCEL.value or doc.progress < 0
        except Exception:
            pass
        return False


def queue_raptor_tasks(doc):
    """
    创建指定文档的“组织树检索的递归抽象处理”任务，并插入数据库，插入redis消息队列rag_flow_svr_queue，redis插入失败，则抛出异常
    """
    def new_task():
        nonlocal doc
        return {
            "id": get_uuid(),
            "doc_id": doc["id"],
            "from_page": 0,
            "to_page": -1,
            # 组织树检索的递归抽象处理
            "progress_msg": "Start to do RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)."
        }
    # 创建任务，并插入任务
    task = new_task()
    bulk_insert_into_db(Task, [task], True)
    # 【注意任务没有type字段，这里是为了追加type字段，标识当前任务是raptor任务，后续处理任务时，选用raptor任务处理逻辑】设置任务type字段为raptor，并将任务放入redis消息队列rag_flow_svr_queue中【若插入队列不成功，则抛出异常】
    task["type"] = "raptor"
    assert REDIS_CONN.queue_product(SVR_QUEUE_NAME, message=task), "Can't access Redis. Please check the Redis' status."


def doc_upload_and_parse(conversation_id, file_objs, user_id):
    from rag.app import presentation, picture, naive, audio, email
    from api.db.services.dialog_service import ConversationService, DialogService
    from api.db.services.file_service import FileService
    from api.db.services.llm_service import LLMBundle
    from api.db.services.user_service import TenantService
    from api.db.services.api_service import API4ConversationService

    e, conv = ConversationService.get_by_id(conversation_id)
    if not e:
        e, conv = API4ConversationService.get_by_id(conversation_id)
    assert e, "Conversation not found!"

    e, dia = DialogService.get_by_id(conv.dialog_id)
    kb_id = dia.kb_ids[0]
    e, kb = KnowledgebaseService.get_by_id(kb_id)
    if not e:
        raise LookupError("Can't find this knowledgebase!")

    embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING, llm_name=kb.embd_id, lang=kb.language)

    err, files = FileService.upload_document(kb, file_objs, user_id)
    assert not err, "\n".join(err)

    def dummy(prog=None, msg=""):
        pass

    FACTORY = {
        ParserType.PRESENTATION.value: presentation,
        ParserType.PICTURE.value: picture,
        ParserType.AUDIO.value: audio,
        ParserType.EMAIL.value: email
    }
    parser_config = {"chunk_token_num": 4096, "delimiter": "\n!?;。；！？", "layout_recognize": False}
    exe = ThreadPoolExecutor(max_workers=12)
    threads = []
    doc_nm = {}
    for d, blob in files:
        doc_nm[d["id"]] = d["name"]
    for d, blob in files:
        kwargs = {
            "callback": dummy,
            "parser_config": parser_config,
            "from_page": 0,
            "to_page": 100000,
            "tenant_id": kb.tenant_id,
            "lang": kb.language
        }
        threads.append(exe.submit(FACTORY.get(d["parser_id"], naive).chunk, d["name"], blob, **kwargs))

    for (docinfo, _), th in zip(files, threads):
        docs = []
        doc = {
            "doc_id": docinfo["id"],
            "kb_id": [kb.id]
        }
        for ck in th.result():
            d = deepcopy(doc)
            d.update(ck)
            md5 = hashlib.md5()
            md5.update((ck["content_with_weight"] +
                        str(d["doc_id"])).encode("utf-8"))
            d["id"] = md5.hexdigest()
            d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
            d["create_timestamp_flt"] = datetime.now().timestamp()
            if not d.get("image"):
                docs.append(d)
                continue

            output_buffer = BytesIO()
            if isinstance(d["image"], bytes):
                output_buffer = BytesIO(d["image"])
            else:
                d["image"].save(output_buffer, format='JPEG')

            STORAGE_IMPL.put(kb.id, d["id"], output_buffer.getvalue())
            d["img_id"] = "{}-{}".format(kb.id, d["id"])
            del d["image"]
            docs.append(d)

    parser_ids = {d["id"]: d["parser_id"] for d, _ in files}
    docids = [d["id"] for d, _ in files]
    chunk_counts = {id: 0 for id in docids}
    token_counts = {id: 0 for id in docids}
    es_bulk_size = 64

    def embedding(doc_id, cnts, batch_size=16):
        nonlocal embd_mdl, chunk_counts, token_counts
        vects = []
        for i in range(0, len(cnts), batch_size):
            vts, c = embd_mdl.encode(cnts[i: i + batch_size])
            vects.extend(vts.tolist())
            chunk_counts[doc_id] += len(cnts[i:i + batch_size])
            token_counts[doc_id] += c
        return vects

    idxnm = search.index_name(kb.tenant_id)
    try_create_idx = True

    _, tenant = TenantService.get_by_id(kb.tenant_id)
    llm_bdl = LLMBundle(kb.tenant_id, LLMType.CHAT, tenant.llm_id)
    for doc_id in docids:
        cks = [c for c in docs if c["doc_id"] == doc_id]

        if parser_ids[doc_id] != ParserType.PICTURE.value:
            mindmap = MindMapExtractor(llm_bdl)
            try:
                mind_map = json.dumps(mindmap([c["content_with_weight"] for c in docs if c["doc_id"] == doc_id]).output,
                                      ensure_ascii=False, indent=2)
                if len(mind_map) < 32: raise Exception("Few content: " + mind_map)
                cks.append({
                    "id": get_uuid(),
                    "doc_id": doc_id,
                    "kb_id": [kb.id],
                    "docnm_kwd": doc_nm[doc_id],
                    "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", doc_nm[doc_id])),
                    "content_ltks": "",
                    "content_with_weight": mind_map,
                    "knowledge_graph_kwd": "mind_map"
                })
            except Exception as e:
                logging.exception("Mind map generation error")

        vects = embedding(doc_id, [c["content_with_weight"] for c in cks])
        assert len(cks) == len(vects)
        for i, d in enumerate(cks):
            v = vects[i]
            d["q_%d_vec" % len(v)] = v
        for b in range(0, len(cks), es_bulk_size):
            if try_create_idx:
                if not settings.docStoreConn.indexExist(idxnm, kb_id):
                    settings.docStoreConn.createIdx(idxnm, kb_id, len(vects[0]))
                try_create_idx = False
            settings.docStoreConn.insert(cks[b:b + es_bulk_size], idxnm, kb_id)

        DocumentService.increment_chunk_num(
            doc_id, kb.id, token_counts[doc_id], chunk_counts[doc_id], 0)

    return [d["id"] for d, _ in files]