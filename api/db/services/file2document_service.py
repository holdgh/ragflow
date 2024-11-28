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
from datetime import datetime

from api.db import FileSource
from api.db.db_models import DB
from api.db.db_models import File, File2Document
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.utils import current_timestamp, datetime_format, get_uuid


class File2DocumentService(CommonService):
    model = File2Document

    @classmethod
    @DB.connection_context()
    def get_by_file_id(cls, file_id):
        objs = cls.model.select().where(cls.model.file_id == file_id)
        return objs

    @classmethod
    @DB.connection_context()
    def get_by_document_id(cls, document_id):
        objs = cls.model.select().where(cls.model.document_id == document_id)
        return objs

    @classmethod
    @DB.connection_context()
    def insert(cls, obj):
        if not cls.save(**obj):
            raise RuntimeError("Database error (File)!")
        e, obj = cls.get_by_id(obj["id"])
        if not e:
            raise RuntimeError("Database error (File retrieval)!")
        return obj

    @classmethod
    @DB.connection_context()
    def delete_by_file_id(cls, file_id):
        return cls.model.delete().where(cls.model.file_id == file_id).execute()

    @classmethod
    @DB.connection_context()
    def delete_by_document_id(cls, doc_id):
        return cls.model.delete().where(cls.model.document_id == doc_id).execute()

    @classmethod
    @DB.connection_context()
    def update_by_file_id(cls, file_id, obj):
        obj["update_time"] = current_timestamp()
        obj["update_date"] = datetime_format(datetime.now())
        num = cls.model.update(obj).where(cls.model.id == file_id).execute()
        e, obj = cls.get_by_id(cls.model.id)
        return obj

    @classmethod
    @DB.connection_context()
    def get_storage_address(cls, doc_id=None, file_id=None):
        """
        功能：获取file2Document记录，据此获取文件记录，以获取【本地来源的文件返回parent_id和location】知识库id和minio中的文件地址
        """
        # 查询file2Document记录【文档id非空，用文档id查；文档id为空，用文件id查】
        if doc_id:
            f2d = cls.get_by_document_id(doc_id)
        else:
            f2d = cls.get_by_file_id(file_id)
        if f2d:
            # file2Document非空时
            # 查询文件记录
            file = File.get_by_id(f2d[0].file_id)
            if not file.source_type or file.source_type == FileSource.LOCAL:
                # 文件记录source_type字段为空或为空字符串，直接返回文件记录的parent_id和location字段
                return file.parent_id, file.location
            # 取file2Document记录中的文档id
            doc_id = f2d[0].document_id
        # 断言文档id非空
        assert doc_id, "please specify doc_id"
        # 查询文档记录
        e, doc = DocumentService.get_by_id(doc_id)
        # 返回文档记录中的知识库id和location字段【上传文件操作中赋值为minio中的文件地址】
        return doc.kb_id, doc.location
