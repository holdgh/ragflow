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
import re
import os
from concurrent.futures import ThreadPoolExecutor

from flask_login import current_user
from peewee import fn

from api.db import FileType, KNOWLEDGEBASE_FOLDER_NAME, FileSource, ParserType
from api.db.db_models import DB, File2Document, Knowledgebase
from api.db.db_models import File, Document
from api.db.services import duplicate_name
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.utils import get_uuid
from api.utils.file_utils import filename_type, thumbnail_img
from rag.utils.storage_factory import STORAGE_IMPL


class FileService(CommonService):
    model = File

    @classmethod
    @DB.connection_context()
    def get_by_pf_id(cls, tenant_id, pf_id, page_number, items_per_page,
                     orderby, desc, keywords):
        if keywords:
            files = cls.model.select().where(
                (cls.model.tenant_id == tenant_id),
                (cls.model.parent_id == pf_id),
                (fn.LOWER(cls.model.name).contains(keywords.lower())),
                ~(cls.model.id == pf_id)
            )
        else:
            files = cls.model.select().where((cls.model.tenant_id == tenant_id),
                                             (cls.model.parent_id == pf_id),
                                             ~(cls.model.id == pf_id)
                                             )
        count = files.count()
        if desc:
            files = files.order_by(cls.model.getter_by(orderby).desc())
        else:
            files = files.order_by(cls.model.getter_by(orderby).asc())

        files = files.paginate(page_number, items_per_page)

        res_files = list(files.dicts())
        for file in res_files:
            if file["type"] == FileType.FOLDER.value:
                file["size"] = cls.get_folder_size(file["id"])
                file['kbs_info'] = []
                children = list(cls.model.select().where(
                    (cls.model.tenant_id == tenant_id),
                    (cls.model.parent_id == file["id"]),
                    ~(cls.model.id == file["id"]),
                ).dicts())
                file["has_child_folder"] = any(value["type"] == FileType.FOLDER.value for value in children)                       
                continue
            kbs_info = cls.get_kb_id_by_file_id(file['id'])
            file['kbs_info'] = kbs_info

        return res_files, count

    @classmethod
    @DB.connection_context()
    def get_kb_id_by_file_id(cls, file_id):
        kbs = (cls.model.select(*[Knowledgebase.id, Knowledgebase.name])
               .join(File2Document, on=(File2Document.file_id == file_id))
               .join(Document, on=(File2Document.document_id == Document.id))
               .join(Knowledgebase, on=(Knowledgebase.id == Document.kb_id))
               .where(cls.model.id == file_id))
        if not kbs: return []
        kbs_info_list = []
        for kb in list(kbs.dicts()):
            kbs_info_list.append({"kb_id": kb['id'], "kb_name": kb['name']})
        return kbs_info_list

    @classmethod
    @DB.connection_context()
    def get_by_pf_id_name(cls, id, name):
        file = cls.model.select().where((cls.model.parent_id == id) & (cls.model.name == name))
        if file.count():
            e, file = cls.get_by_id(file[0].id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
            return file
        return None

    @classmethod
    @DB.connection_context()
    def get_id_list_by_id(cls, id, name, count, res):
        if count < len(name):
            file = cls.get_by_pf_id_name(id, name[count])
            if file:
                res.append(file.id)
                return cls.get_id_list_by_id(file.id, name, count + 1, res)
            else:
                return res
        else:
            return res

    @classmethod
    @DB.connection_context()
    def get_all_innermost_file_ids(cls, folder_id, result_ids):
        subfolders = cls.model.select().where(cls.model.parent_id == folder_id)
        if subfolders.exists():
            for subfolder in subfolders:
                cls.get_all_innermost_file_ids(subfolder.id, result_ids)
        else:
            result_ids.append(folder_id)
        return result_ids

    @classmethod
    @DB.connection_context()
    def create_folder(cls, file, parent_id, name, count):
        if count > len(name) - 2:
            return file
        else:
            file = cls.insert({
                "id": get_uuid(),
                "parent_id": parent_id,
                "tenant_id": current_user.id,
                "created_by": current_user.id,
                "name": name[count],
                "location": "",
                "size": 0,
                "type": FileType.FOLDER.value
            })
            return cls.create_folder(file, file.id, name, count + 1)

    @classmethod
    @DB.connection_context()
    def is_parent_folder_exist(cls, parent_id):
        parent_files = cls.model.select().where(cls.model.id == parent_id)
        if parent_files.count():
            return True
        cls.delete_folder_by_pf_id(parent_id)
        return False

    @classmethod
    @DB.connection_context()
    def get_root_folder(cls, tenant_id):
        """
        功能：查询特定用户的文件记录【字典形式】
            存在则返回第一个文件记录；不存在则初始化一条文件记录并保存返回初始化的文件记录
        """
        # 依据条件“用户id && 文件记录parent_id==id”，查询文件记录列表
        for file in cls.model.select().where((cls.model.tenant_id == tenant_id),
                                        (cls.model.parent_id == cls.model.id)
                                        ):
            # 直接返回第一个文件记录【字典形式】
            return file.to_dict()
        # 数据库中不存在特定用户的文件记录时
        # 初始化一条文件记录，保存至数据库，并返回文件记录【字典形式】
        file_id = get_uuid()
        file = {
            "id": file_id,
            "parent_id": file_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": "/",
            "type": FileType.FOLDER.value,
            "size": 0,
            "location": "",
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def get_kb_folder(cls, tenant_id):
        # 先查root_file，再查folder_file，查到即返回
        for root in cls.model.select().where(
                (cls.model.tenant_id == tenant_id), (cls.model.parent_id == cls.model.id)):
            for folder in cls.model.select().where(
                    (cls.model.tenant_id == tenant_id), (cls.model.parent_id == root.id),
                    (cls.model.name == KNOWLEDGEBASE_FOLDER_NAME)):
                return folder.to_dict()
        assert False, "Can't find the KB folder. Database init error."

    @classmethod
    @DB.connection_context()
    def new_a_file_from_kb(cls, tenant_id, name, parent_id, ty=FileType.FOLDER.value, size=0, location=""):
        # 查询特定知识库的kb_folder_file记录，存在则直接返回，不存在则创建保存并会返回
        for file in cls.query(tenant_id=tenant_id, parent_id=parent_id, name=name):
            return file.to_dict()
        file = {
            "id": get_uuid(),
            "parent_id": parent_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": name,
            "type": ty,
            "size": size,
            "location": location,
            "source_type": FileSource.KNOWLEDGEBASE
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def init_knowledgebase_docs(cls, root_id, tenant_id):
        """
        入参：
            root_id：文件id
            tenant_id：用户id
        功能：确保“name为'.knowledgebase' && parent_id为root_id”的文件记录存在
        结果：
            1、确保“name为'.knowledgebase' && parent_id为root_id”的文件记录存在
            2、对于当前用户的每个知识库而言，对应的【tenant_id, name【kb.name】, parent_id【folder["id"]】】文件记录存在
            3、对于当前用户的每个知识库的每个文档而言，对应文档id的file2Document记录【且有文件记录与其关联】存在
        推论：
            1、用户--多个知识库
            2、用户文件记录结构
                - 存在一条“文件记录parent_id==id”的文件记录【该文件记录的id记为’root_id‘】
                - 存在一条“name为'.knowledgebase' && parent_id为root_id”的文件记录【该文件记录的id记为’folder_id‘】
                - 对于当前用户的每个知识库而言，对应存在一条“name为kb.name, parent_id为folder_id”的文件记录【该文件记录的id记为’kb_folder_id‘】
                - 对于当前用户的每个知识库的每个文档而言，对应存在一条“id为当前文档id”的file2Document记录【且有文件记录【parent_id为kb_folder_id】与其关联】
        人话推论：
            - 1个user-->1个root_file【特点：parent_id==id】-->1个folder_file【特点：parent_id==root_id】-->多个kb_folder_file【一个知识库一个，特点：parent_id==folder_id】
            - 1个知识库-->TODO 推测多个【文档-file2Document】-file【特点：parent_id==kb_folder_id】
        """
        # 依据条件“name为'.knowledgebase' && parent_id为root_id”查询文件记录列表
        for _ in cls.model.select().where((cls.model.name == KNOWLEDGEBASE_FOLDER_NAME)\
                                          & (cls.model.parent_id == root_id)):
            # 存在，则直接返回
            return
        # 不存在“name为'.knowledgebase' && parent_id为root_id”的文件记录时
        # 创建一个“name为'.knowledgebase' && parent_id为root_id”的文件记录
        folder = cls.new_a_file_from_kb(tenant_id, KNOWLEDGEBASE_FOLDER_NAME, root_id)
        # 依据条件“特定用户id”查询知识库记录列表，遍历处理
        for kb in Knowledgebase.select(*[Knowledgebase.id, Knowledgebase.name]).where(Knowledgebase.tenant_id==tenant_id):
            # 创建【tenant_id, name【kb.name】, parent_id【folder["id"]】】文件记录
            kb_folder = cls.new_a_file_from_kb(tenant_id, kb.name, folder["id"])
            # 依据条件“特定知识库id”查询文档列表，遍历处理
            for doc in DocumentService.query(kb_id=kb.id):
                # 基于当前知识库的当前文档记录创建并保存文件记录【存在特定文档记录id的file2Document记录时，直接返回】
                # 先后顺序：知识库-文档-file2Document
                FileService.add_file_from_kb(doc.to_dict(), kb_folder["id"], tenant_id)

    @classmethod
    @DB.connection_context()
    def get_parent_folder(cls, file_id):
        file = cls.model.select().where(cls.model.id == file_id)
        if file.count():
            e, file = cls.get_by_id(file[0].parent_id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
        else:
            raise RuntimeError("Database error (File doesn't exist)!")
        return file

    @classmethod
    @DB.connection_context()
    def get_all_parent_folders(cls, start_id):
        parent_folders = []
        current_id = start_id
        while current_id:
            e, file = cls.get_by_id(current_id)
            if file.parent_id != file.id and e:
                parent_folders.append(file)
                current_id = file.parent_id
            else:
                parent_folders.append(file)
                break
        return parent_folders

    @classmethod
    @DB.connection_context()
    def insert(cls, file):
        if not cls.save(**file):
            raise RuntimeError("Database error (File)!")
        e, file = cls.get_by_id(file["id"])
        if not e:
            raise RuntimeError("Database error (File retrieval)!")
        return file

    @classmethod
    @DB.connection_context()
    def delete(cls, file):
        return cls.delete_by_id(file.id)

    @classmethod
    @DB.connection_context()
    def delete_by_pf_id(cls, folder_id):
        return cls.model.delete().where(cls.model.parent_id == folder_id).execute()

    @classmethod
    @DB.connection_context()
    def delete_folder_by_pf_id(cls, user_id, folder_id):
        try:
            files = cls.model.select().where((cls.model.tenant_id == user_id)
                                             & (cls.model.parent_id == folder_id))
            for file in files:
                cls.delete_folder_by_pf_id(user_id, file.id)
            return cls.model.delete().where((cls.model.tenant_id == user_id)
                                            & (cls.model.id == folder_id)).execute(),
        except Exception:
            logging.exception("delete_folder_by_pf_id")
            raise RuntimeError("Database error (File retrieval)!")

    @classmethod
    @DB.connection_context()
    def get_file_count(cls, tenant_id):
        files = cls.model.select(cls.model.id).where(cls.model.tenant_id == tenant_id)
        return len(files)

    @classmethod
    @DB.connection_context()
    def get_folder_size(cls, folder_id):
        size = 0

        def dfs(parent_id):
            nonlocal size
            for f in cls.model.select(*[cls.model.id, cls.model.size, cls.model.type]).where(
                    cls.model.parent_id == parent_id, cls.model.id != parent_id):
                size += f.size
                if f.type == FileType.FOLDER.value:
                    dfs(f.id)

        dfs(folder_id)
        return size

    @classmethod
    @DB.connection_context()
    def add_file_from_kb(cls, doc, kb_folder_id, tenant_id):
        # 存在特定文档id的file2Document记录时，直接返回
        for _ in File2DocumentService.get_by_document_id(doc["id"]): return
        # 不存在特定文档id的file2Document记录时
        # 初始化文件记录
        file = {
            "id": get_uuid(),
            "parent_id": kb_folder_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": doc["name"],
            "type": doc["type"],
            "size": doc["size"],
            "location": doc["location"],
            "source_type": FileSource.KNOWLEDGEBASE
        }
        # 保存文件记录
        cls.save(**file)
        # 保存为file2Document记录
        File2DocumentService.save(**{"id": get_uuid(), "file_id": file["id"], "document_id": doc["id"]})
    
    @classmethod
    @DB.connection_context()
    def move_file(cls, file_ids, folder_id):
        try:
            cls.filter_update((cls.model.id << file_ids, ), { 'parent_id': folder_id })
        except Exception:
            logging.exception("move_file")
            raise RuntimeError("Database error (File move)!")

    @classmethod
    @DB.connection_context()
    def upload_document(self, kb, file_objs, user_id):
        """
        入参介绍：
            kb：知识库记录
            file_objs：文件列表
            user_id：用户id
        逻辑介绍：
            1、查询用户文件结构【不存在，则创建】
                - 1个user-->1个root_file【特点：parent_id==id】-->1个folder_file【特点：parent_id==root_id】-->多个kb_folder_file【一个知识库一个，特点：parent_id==folder_id】
                - 1个知识库-->多个{文档-file2Document-file【特点：parent_id==kb_folder_id】}
            2、遍历处理文件：
                - 检验文件数据
                - 保存文件至minio
                - 将特定类型的文件转为图片，并将图片至minio
                - 保存文档记录、file2Document记录、文件记录至数据库
        出参：
            err：处理异常的文件名及异常信息
            files：对应入参文件列表的文档记录列表
        """
        # =========查询用户文件结构-start=========
        # 获取特定用户的root文件记录【字典形式，一个】
        root_folder = self.get_root_folder(user_id)
        # 获取id
        pf_id = root_folder["id"]
        # 初始化知识库
        self.init_knowledgebase_docs(pf_id, user_id)
        # 获取特定用户的知识库folder_file文件记录
        kb_root_folder = self.get_kb_folder(user_id)
        # 获取特定用户的特定知识库的kb_folder_file文件记录
        kb_folder = self.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])
        # =========查询用户文件结构-end=========
        err, files = [], []
        # 遍历文件列表，逐个保存处理
        for file in file_objs:
            try:
                # ==========检验文件数据-start==========
                # 校验文档数据量最大限制
                MAX_FILE_NUM_PER_USER = int(os.environ.get('MAX_FILE_NUM_PER_USER', 0))
                if MAX_FILE_NUM_PER_USER > 0 and DocumentService.get_doc_count(kb.tenant_id) >= MAX_FILE_NUM_PER_USER:
                    raise RuntimeError("Exceed the maximum file number of a free user!")
                # 对文件名做追加版本号的去重处理
                filename = duplicate_name(
                    DocumentService.query,
                    name=file.filename,
                    kb_id=kb.id)
                # 获取文件类型
                filetype = filename_type(filename)
                # 文件类型不支持检查
                if filetype == FileType.OTHER.value:
                    raise RuntimeError("This type of file has not been supported yet!")
                # ==========检验文件数据-end==========
                # ==========保存文件至minio-start==========
                # 文件位置
                location = filename
                # 当前项目采用minio存储方式，以知识库id作为bucket名称
                # 基于location设置minio存储的文件名【通过不断追加'_'，直至文件名不存在，来保证唯一性】
                while STORAGE_IMPL.obj_exist(kb.id, location):
                    location += "_"
                # 读取文件数据
                blob = file.read()
                # 将文件数据blob以文件名location放入minio的文件桶kb.id中
                STORAGE_IMPL.put(kb.id, location, blob)
                # ==========保存文件至minio-end==========
                # ==========将特定类型的文件转为图片，并将图片至minio-start==========
                # 随机生成文档id
                doc_id = get_uuid()
                # 将文件转化为图片【目前支持将pdf|jpg|jpeg|png|tif|gif|icon|ico|webp|ppt|pptx文件转为png图片，对于pdf和ppt及pptx, TODO 只处理第一页？】，并放入minio
                img = thumbnail_img(filename, blob)
                thumbnail_location = ''
                if img is not None:
                    # 图片文件名含有随机生成的文档id，因此这里没有进行唯一性处理
                    thumbnail_location = f'thumbnail_{doc_id}.png'
                    STORAGE_IMPL.put(kb.id, thumbnail_location, img)
                # ==========将特定类型的文件转为图片，并将图片至minio-end==========
                # ==========保存文档记录、file2Document记录、文件记录至数据库-start==========
                # 创建文档记录
                doc = {
                    "id": doc_id,
                    "kb_id": kb.id,
                    # 根据文件类型设置parser_id字段【非‘图片、音频、ppt、邮箱类型’时，取知识库的parser_id字段】
                    "parser_id": self.get_parser(filetype, filename, kb.parser_id),
                    "parser_config": kb.parser_config,
                    "created_by": user_id,
                    "type": filetype,
                    "name": filename,
                    "location": location,
                    "size": len(blob),
                    "thumbnail": thumbnail_location
                }
                # 数据表document的process字段默认值为0【因此不在ragflow_server中update_process的处理范围之内】，status字段默认值为'1'，run字段默认值为'0'
                # 插入文档记录
                DocumentService.insert(doc)
                # 保存文件记录【文档-file2Document-file，三者此时一一对应】
                FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
                # ==========保存文档记录、file2Document记录、文件记录至数据库-end==========
                # 收集文档记录和文件数据到files中
                files.append((doc, blob))
            except Exception as e:
                # 收集异常数据--文件名：异常信息
                err.append(file.filename + ": " + str(e))

        return err, files

    @staticmethod
    def parse_docs(file_objs, user_id):
        from rag.app import presentation, picture, naive, audio, email

        def dummy(prog=None, msg=""):
            pass

        FACTORY = {
            ParserType.PRESENTATION.value: presentation,
            ParserType.PICTURE.value: picture,
            ParserType.AUDIO.value: audio,
            ParserType.EMAIL.value: email
        }
        parser_config = {"chunk_token_num": 16096, "delimiter": "\n!?;。；！？", "layout_recognize": False}
        exe = ThreadPoolExecutor(max_workers=12)
        threads = []
        for file in file_objs:
            kwargs = {
                "lang": "English",
                "callback": dummy,
                "parser_config": parser_config,
                "from_page": 0,
                "to_page": 100000,
                "tenant_id": user_id
            }
            filetype = filename_type(file.filename)
            blob = file.read()
            threads.append(exe.submit(FACTORY.get(FileService.get_parser(filetype, file.filename, ""), naive).chunk, file.filename, blob, **kwargs))

        res = []
        for th in threads:
            res.append("\n".join([ck["content_with_weight"] for ck in th.result()]))

        return "\n\n".join(res)

    @staticmethod
    def get_parser(doc_type, filename, default):
        if doc_type == FileType.VISUAL:
            return ParserType.PICTURE.value
        if doc_type == FileType.AURAL:
            return ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            return ParserType.PRESENTATION.value
        if re.search(r"\.(eml)$", filename):
            return ParserType.EMAIL.value
        return default