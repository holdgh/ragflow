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
import random

from api.db.db_utils import bulk_insert_into_db
from deepdoc.parser import PdfParser
from peewee import JOIN
from api.db.db_models import DB, File2Document, File
from api.db import StatusEnum, FileType, TaskStatus
from api.db.db_models import Task, Document, Knowledgebase, Tenant
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.utils import current_timestamp, get_uuid
from deepdoc.parser.excel_parser import RAGFlowExcelParser
from rag.settings import SVR_QUEUE_NAME
from rag.utils.storage_factory import STORAGE_IMPL
from rag.utils.redis_conn import REDIS_CONN


class TaskService(CommonService):
    model = Task

    @classmethod
    @DB.connection_context()
    def get_task(cls, task_id):
        fields = [
            cls.model.id,
            cls.model.doc_id,
            cls.model.from_page,
            cls.model.to_page,
            cls.model.retry_count,
            Document.kb_id,
            Document.parser_id,
            Document.parser_config,
            Document.name,
            Document.type,
            Document.location,
            Document.size,
            Knowledgebase.tenant_id,
            Knowledgebase.language,
            Knowledgebase.embd_id,
            Tenant.img2txt_id,
            Tenant.asr_id,
            Tenant.llm_id,
            cls.model.update_time]
        docs = cls.model.select(*fields) \
            .join(Document, on=(cls.model.doc_id == Document.id)) \
            .join(Knowledgebase, on=(Document.kb_id == Knowledgebase.id)) \
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id)) \
            .where(cls.model.id == task_id)
        docs = list(docs.dicts())
        if not docs: return None

        msg = "\nTask has been received."
        prog = random.random() / 10.
        if docs[0]["retry_count"] >= 3:
            msg = "\nERROR: Task is abandoned after 3 times attempts."
            prog = -1

        cls.model.update(progress_msg=cls.model.progress_msg + msg,
                         progress=prog,
                         retry_count=docs[0]["retry_count"]+1
                         ).where(
            cls.model.id == docs[0]["id"]).execute()

        if docs[0]["retry_count"] >= 3: return None

        return docs[0]

    @classmethod
    @DB.connection_context()
    def get_ongoing_doc_name(cls):
        with DB.lock("get_task", -1):
            docs = cls.model.select(*[Document.id, Document.kb_id, Document.location, File.parent_id]) \
                .join(Document, on=(cls.model.doc_id == Document.id)) \
                .join(File2Document, on=(File2Document.document_id == Document.id), join_type=JOIN.LEFT_OUTER) \
                .join(File, on=(File2Document.file_id == File.id), join_type=JOIN.LEFT_OUTER) \
                .where(
                    Document.status == StatusEnum.VALID.value,
                    Document.run == TaskStatus.RUNNING.value,
                    ~(Document.type == FileType.VIRTUAL.value),
                    cls.model.progress < 1,
                    cls.model.create_time >= current_timestamp() - 1000 * 600
                )
            docs = list(docs.dicts())
            if not docs: return []

            return list(set([(d["parent_id"] if d["parent_id"] else d["kb_id"], d["location"]) for d in docs]))

    @classmethod
    @DB.connection_context()
    def do_cancel(cls, id):
        """
        功能：依据id查询任务，查询任务中关联的文档记录：若文档记录run字段为'2'或者progress字段小于0，则返回true
        """
        try:
            task = cls.model.get_by_id(id)
            _, doc = DocumentService.get_by_id(task.doc_id)
            return doc.run == TaskStatus.CANCEL.value or doc.progress < 0
        except Exception:
            pass
        return False

    @classmethod
    @DB.connection_context()
    def update_progress(cls, id, info):
        if os.environ.get("MACOS"):
            if info["progress_msg"]:
                cls.model.update(progress_msg=cls.model.progress_msg + "\n" + info["progress_msg"]).where(
                    cls.model.id == id).execute()
            if "progress" in info:
                cls.model.update(progress=info["progress"]).where(
                    cls.model.id == id).execute()
            return

        with DB.lock("update_progress", -1):
            if info["progress_msg"]:
                cls.model.update(progress_msg=cls.model.progress_msg + "\n" + info["progress_msg"]).where(
                    cls.model.id == id).execute()
            if "progress" in info:
                cls.model.update(progress=info["progress"]).where(
                    cls.model.id == id).execute()


def queue_tasks(doc: dict, bucket: str, name: str):
    """
    功能：基于文档记录，分情况创建任务列表，批量保存任务列表，更新文档记录progress字段为0-1之间的正小数【触发启动脚本ragflow_server的update_progress】，将任务记录依次放入redis中的消息队列
    思考：
        1、此处生成的任务记录不含type字段设置，也即没有raptor
    """
    def new_task():
        return {
            "id": get_uuid(),
            "doc_id": doc["id"]
        }
    # 初始化任务列表
    tsks = []
    # 分类创建任务
    if doc["type"] == FileType.PDF.value:
        # pdf类型的文件
        # 从minio获取文件
        file_bin = STORAGE_IMPL.get(bucket, name)
        # 注意parser_config来源于知识库记录中的parser_config字段
        # 从文档记录parser_config字段获取layout识别标识
        do_layout = doc["parser_config"].get("layout_recognize", True)
        # 获取pdf文件总页数
        pages = PdfParser.total_page_number(doc["name"], file_bin)
        # 从文档记录parser_config字段获取单任务对应的总页数，默认为12，也即一个任务处理12页pdf
        page_size = doc["parser_config"].get("task_page_size", 12)
        if doc["parser_id"] == "paper":
            # 文档记录parser_id字段为paper时
            # 设置单任务对应的pdf页数默认为22
            page_size = doc["parser_config"].get("task_page_size", 22)
        if doc["parser_id"] in ["one", "knowledge_graph"] or not do_layout:
            # 文档记录parser_id字段为one或knowledge_graph或不进行layout时
            # 设置单任务对应的pdf页数为10亿，也即一个任务处理整个pdf文档
            page_size = 10 ** 9
        # 从文档记录parser_config字段获取页码范围
        page_ranges = doc["parser_config"].get("pages") or [(1, 10 ** 5)]
        for s, e in page_ranges:
            s -= 1
            s = max(0, s)
            e = min(e - 1, pages)
            for p in range(s, e, page_size):
                # 每个任务对应处理page_size页pdf
                task = new_task()
                task["from_page"] = p
                task["to_page"] = min(p + page_size, e)
                tsks.append(task)

    elif doc["parser_id"] == "table":
        # 文档记录parser_id为table【根据上传文件操作可知，此时对应知识库记录的parser_id为table】
        # 从minio获取文件
        file_bin = STORAGE_IMPL.get(bucket, name)
        # 获取表格文件【excel、csv、txt】的行数
        rn = RAGFlowExcelParser.row_number(doc["name"], file_bin)
        for i in range(0, rn, 3000):
            # 每个任务处理3000行数据
            task = new_task()
            task["from_page"] = i
            task["to_page"] = min(i + 3000, rn)
            tsks.append(task)
    else:
        # 其他情况，直接创建任务
        tsks.append(new_task())
    # 将任务列表批量保存到数据库
    bulk_insert_into_db(Task, tsks, True)
    # 更新文档记录：progress【随机正小数，0-1之间，0.0****】
    DocumentService.begin2parse(doc["id"])
    # 将任务列表，依次放入redis中的消息队列
    for t in tsks:
        assert REDIS_CONN.queue_product(SVR_QUEUE_NAME, message=t), "Can't access Redis. Please check the Redis' status."
