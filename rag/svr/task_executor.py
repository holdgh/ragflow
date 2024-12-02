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
import sys
from api.utils.log_utils import initRootLogger
# 根据命令行参数的个数来设置消费者编号：个数小于2取值'0'，否则取第二个命令行参数作为消费者编号
# 见入口脚本docker/entrypoint.sh，参数个数为1个，因此取值为'0'
CONSUMER_NO = "0" if len(sys.argv) < 2 else sys.argv[1]
initRootLogger(f"task_executor_{CONSUMER_NO}")
for module in ["pdfminer"]:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.WARNING)
for module in ["peewee"]:
    module_logger = logging.getLogger(module)
    module_logger.handlers.clear()
    module_logger.propagate = True

from datetime import datetime
import json
import os
import hashlib
import copy
import re
import sys
import time
import threading
from functools import partial
from io import BytesIO
from multiprocessing.context import TimeoutError
from timeit import default_timer as timer

import numpy as np

from api.db import LLMType, ParserType
from api.db.services.dialog_service import keyword_extraction, question_proposal
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import LLMBundle
from api.db.services.task_service import TaskService
from api.db.services.file2document_service import File2DocumentService
from api import settings
from api.db.db_models import close_connection
from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one, audio, \
    knowledge_graph, email
from rag.nlp import search, rag_tokenizer
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from rag.settings import DOC_MAXIMUM_SIZE, SVR_QUEUE_NAME
from rag.utils import rmSpace, num_tokens_from_string
from rag.utils.redis_conn import REDIS_CONN, Payload
from rag.utils.storage_factory import STORAGE_IMPL

BATCH_SIZE = 64

FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: knowledge_graph
}
# task_consumer_0
CONSUMER_NAME = "task_consumer_" + CONSUMER_NO
PAYLOAD: Payload | None = None
BOOT_AT = datetime.now().isoformat()
PENDING_TASKS = 0
LAG_TASKS = 0

mt_lock = threading.Lock()
DONE_TASKS = 0
FAILED_TASKS = 0
CURRENT_TASK = None


def set_progress(task_id, from_page=0, to_page=-1, prog=None, msg="Processing..."):
    """
    功能：处理任务过程中，设置任务记录的progress字段和progress_msg字段的值，任务关闭时，执行payload.ack操作
    """
    global PAYLOAD
    if prog is not None and prog < 0:
        # prog为负数时，设置error msg
        msg = "[ERROR]" + msg
    # 查询当前任务是否关闭
    cancel = TaskService.do_cancel(task_id)
    if cancel:
        # 任务关闭时，设置msg【关闭】，并将prog置为-1
        msg += " [Canceled]"
        prog = -1

    if to_page > 0:
        # 终止页码大于0时
        if msg:
            # msg非空时，设置msg【第from_page + 1页到第to_page + 1页：msg】
            msg = f"Page({from_page + 1}~{to_page + 1}): " + msg
    d = {"progress_msg": msg}
    if prog is not None:
        # prog非空时，将prog放入d中的progress字段
        d["progress"] = prog
    try:
        # 用d更新当前任务的progress和progress_msg字段
        TaskService.update_progress(task_id, d)
    except Exception:
        logging.exception(f"set_progress({task_id}) got exception")
    # TODO 为啥此处要关闭数据库连接？
    close_connection()
    if cancel:
        # 任务关闭时
        if PAYLOAD:
            # 在collect方法中有对全局变量PAYLOAD的赋值，负载数据非空时，消费者向队列回复"我收到了数据【msg_id】"
            PAYLOAD.ack()
            PAYLOAD = None
        os._exit(0)


def collect():
    """
    功能：获取指定队列指定组指定消费者的一个未关闭任务记录
    思考：
        - 1个队列【这里设定了一个队列rag_flow_svr_queue】-->多个组【这里设定了一个组rag_flow_svr_task_broker】-->多个消费者【这里设定一个消费者task_consumer_0】
        - 一个消费者-->多个数据【这里视为pendings，每一个pending对应一个msg_id】
        - 1个队列-->多个msg_id【根据手动触发操作可知，msg就是task记录】
        - 1个msg_id-->【多个】msg【这里取第一个msg构造负载数据】
        - 1个msg-->1个payload-->1个task记录
        - 1个doc记录-->多个task记录
        问题：pendings从哪里来？msg_id与task的关系？
        回答：在ragflow_server.py中的update_progress操作处，存在创建任务【仅限raptor任务】并插入redis数据的操作。在手动触发解析操作中，最终把生成的任务列表依次放入了redis中的消息队列【任务就是队列中的msg】
    """
    global CONSUMER_NAME, PAYLOAD, DONE_TASKS, FAILED_TASKS
    try:
        # 从redis中获取指定队列的指定组的指定消费者的一个【unacked，未处理的数据】负载数据【组不存在或者没有数据，直接返回】
        PAYLOAD = REDIS_CONN.get_unacked_for(CONSUMER_NAME, SVR_QUEUE_NAME, "rag_flow_svr_task_broker")
        if not PAYLOAD:
            # 上述负载为空时，从redis中获取指定队列的指定组的指定消费者的一个负载数据【组不存在，则创建组】
            PAYLOAD = REDIS_CONN.queue_consumer(SVR_QUEUE_NAME, "rag_flow_svr_task_broker", CONSUMER_NAME)
        if not PAYLOAD:
            # 仍取不到负载数据，则睡1秒并返回none
            time.sleep(1)
            return None
    except Exception:
        logging.exception("Get task event from queue exception")
        return None
    # 获取负载中的消息
    msg = PAYLOAD.get_message()
    if not msg:
        # 负载中的消息为空，则直接返回none
        return None
    # 判断负载消息对应的文档记录是否已关闭
    if TaskService.do_cancel(msg["id"]):
        # 关闭，完成任务数加1，并返回none
        with mt_lock:
            DONE_TASKS += 1
        logging.info("Task {} has been canceled.".format(msg["id"]))
        return None
    # 依据负载消息获取对应的任务记录【注意get_task方法查询结果含有任务、文档、知识库、用户的各种字段】
    task = TaskService.get_task(msg["id"])
    if not task:
        # 任务记录为空，则完成任务数加1，并返回none
        with mt_lock:
            DONE_TASKS += 1
        logging.warning("{} empty task!".format(msg["id"]))
        return None
    if msg.get("type", "") == "raptor":
        # 负载消息中的type字段为raptor时，为任务记录添加task_type字段并设置值为raptor
        task["task_type"] = "raptor"
    # 返回任务记录【负载存在，且任务未关闭，任务存在】
    return task


def get_storage_binary(bucket, name):
    return STORAGE_IMPL.get(bucket, name)


def build(row):
    """
    功能：依据当前任务对应文档记录的解析器类型获取解析器，对相应文件做分块处理，得到文件分块数据，遍历处理文件分块数据【追加一些基础字段，图片处理【有则存至minio，并将minio图片信息追加至文件分块数据，保证当前分块能找到图片】】，收集到结果中并返回文件分块数据列表
    """
    # 校验文档尺寸【size字段来源，见get_task方法】是否超出文档最大限制
    if row["size"] > DOC_MAXIMUM_SIZE:
        # 超出限制，更新任务记录，直接返回空列表
        set_progress(row["id"], prog=-1, msg="File size exceeds( <= %dMb )" %
                                             (int(DOC_MAXIMUM_SIZE / 1024 / 1024)))
        return []
    # 设置偏函数callback
    callback = partial(
        set_progress,
        row["id"],
        row["from_page"],
        row["to_page"])
    # 获取解析器【依据文档记录的解析类型parser_id】
    chunker = FACTORY[row["parser_id"].lower()]
    try:
        st = timer()
        # 获取文件所在文件桶名称和文件在minio中的绝对路径名
        bucket, name = File2DocumentService.get_storage_address(doc_id=row["doc_id"])
        # 获取文件数据
        binary = get_storage_binary(bucket, name)
        logging.info(
            "From minio({}) {}/{}".format(timer() - st, row["location"], row["name"]))
    except TimeoutError:
        # 超时异常，更新到任务记录
        callback(-1, "Internal server error: Fetch file from minio timeout. Could you try it again.")
        logging.exception(
            "Minio {}/{} got timeout: Fetch file from minio timeout.".format(row["location"], row["name"]))
        raise
    except Exception as e:
        # 其他异常，更新到任务记录
        if re.search("(No such file|not found)", str(e)):
            callback(-1, "Can not find file <%s> from minio. Could you try it again?" % row["name"])
        else:
            callback(-1, "Get file from minio: %s" % str(e).replace("'", ""))
        logging.exception("Chunking {}/{} got exception".format(row["location"], row["name"]))
        raise

    try:
        # 对于文件进行解析【入参：文件名、文件、起始页、终止页、知识库语言、任务回调函数callback、知识库id、知识库解析配置、用户id】
        # 解析得到文件分块数据
        cks = chunker.chunk(row["name"], binary=binary, from_page=row["from_page"],
                            to_page=row["to_page"], lang=row["language"], callback=callback,
                            kb_id=row["kb_id"], parser_config=row["parser_config"], tenant_id=row["tenant_id"])
        logging.info("Chunking({}) {}/{} done".format(timer() - st, row["location"], row["name"]))
    except Exception as e:
        callback(-1, "Internal server error while chunking: %s" %
                 str(e).replace("'", ""))
        logging.exception("Chunking {}/{} got exception".format(row["location"], row["name"]))
        raise
    # 遍历处理解析器得到的文件分块数据，收集到docs
    docs = []
    # 为每个文件分块数据追加文档id和知识库id
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": str(row["kb_id"])
    }
    el = 0
    for ck in cks:
        d = copy.deepcopy(doc)
        # 将ck中的数据保存【d中不存在的字段，保存到d中】或更新【d中存在的字段，更新为ck相应字段的值】到d中
        d.update(ck)
        md5 = hashlib.md5()
        md5.update((ck["content_with_weight"] +
                    str(d["doc_id"])).encode("utf-8"))
        d["id"] = md5.hexdigest()
        d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.now().timestamp()
        if not d.get("image"):
            # 当前文件分块数据没有图片，则去除image字段
            _ = d.pop("image", None)
            # 设置各种图片相关字段数据，并将结果收集到docs，处理下一个文件分块数据
            d["img_id"] = ""
            d["page_num_list"] = json.dumps([])
            d["position_list"] = json.dumps([])
            d["top_list"] = json.dumps([])
            docs.append(d)
            continue
        # 当前文件分块数据中有图片时，将图片保存至minio
        try:
            output_buffer = BytesIO()
            if isinstance(d["image"], bytes):
                # image字段值为bytes类型时，将其写入output_buffer中
                output_buffer = BytesIO(d["image"])
            else:
                # image字段值非bytes类型时，将其以JPEG格式保存到output_buffer中
                d["image"].save(output_buffer, format='JPEG')

            st = timer()
            # 将图片文件保存至minio【桶：知识库id，文件名：文件分块id，图片数据】
            STORAGE_IMPL.put(row["kb_id"], d["id"], output_buffer.getvalue())
            el += timer() - st
        except Exception:
            logging.exception(
                "Saving image of chunk {}/{}/{} got exception".format(row["location"], row["name"], d["_id"]))
            raise
        # 当前文件分块数据中有图片时，设置图片相关字段数据
        # 图片id设置为“文件桶名-图片名”
        d["img_id"] = "{}-{}".format(row["kb_id"], d["id"])
        # 删除文件分块中的image字段
        del d["image"]
        # 将当前文件分块数据收集到docs中
        docs.append(d)
    logging.info("MINIO PUT({}):{}".format(row["name"], el))
    # 知识库parser_config中设置auto_keywords非0时，基于chat模型，对每个文件分块进行关键字提取，并将提取结果设置到相应文件分块数据中
    # 官方demo环境解释：在查询此类关键词时，为每个块提取 N 个关键词以提高其排名得分。在“系统模型设置”中设置的 LLM 将消耗额外的 token。您可以在块列表中查看结果。
    if row["parser_config"].get("auto_keywords", 0):
        st = timer()
        callback(msg="Start to generate keywords for every chunk ...")
        chat_mdl = LLMBundle(row["tenant_id"], LLMType.CHAT, llm_name=row["llm_id"], lang=row["language"])
        for d in docs:
            d["important_kwd"] = keyword_extraction(chat_mdl, d["content_with_weight"],
                                                    row["parser_config"]["auto_keywords"]).split(",")
            d["important_tks"] = rag_tokenizer.tokenize(" ".join(d["important_kwd"]))
        callback(msg="Keywords generation completed in {:.2f}s".format(timer() - st))

    # 知识库parser_config中设置auto_questions非0时，基于chat模型，对每个文件分块进行question生成，并将question数据设置到相应文件分块数据中
    # 官方demo环境解释：在查询此类问题时，为每个块提取 N 个问题以提高其排名得分。在“系统模型设置”中设置的 LLM 将消耗额外的 token。您可以在块列表中查看结果。如果发生错误，此功能不会破坏整个分块过程，除了将空结果添加到原始块。
    if row["parser_config"].get("auto_questions", 0):
        st = timer()
        callback(msg="Start to generate questions for every chunk ...")
        chat_mdl = LLMBundle(row["tenant_id"], LLMType.CHAT, llm_name=row["llm_id"], lang=row["language"])
        for d in docs:
            qst = question_proposal(chat_mdl, d["content_with_weight"], row["parser_config"]["auto_questions"])
            d["content_with_weight"] = f"Question: \n{qst}\n\nAnswer:\n" + d["content_with_weight"]
            qst = rag_tokenizer.tokenize(qst)
            if "content_ltks" in d:
                d["content_ltks"] += " " + qst
            if "content_sm_ltks" in d:
                d["content_sm_ltks"] += " " + rag_tokenizer.fine_grained_tokenize(qst)
        callback(msg="Question generation completed in {:.2f}s".format(timer() - st))
    # 返回【处理当前任务对应起始页和终止页范围内的文件数据】文件分块数据列表
    return docs


def init_kb(row, vector_size: int):
    idxnm = search.index_name(row["tenant_id"])
    return settings.docStoreConn.createIdx(idxnm, row["kb_id"], vector_size)


def embedding(docs, mdl, parser_config=None, callback=None):
    if parser_config is None:
        parser_config = {}
    batch_size = 32
    # TODO 对文件分块中的标题内容进行正则处理，并收集到tts中
    # 将文件分块内容中的表格类标签替换为空格【例如：'<table>123</table>'-->' 123 '】，并收集到cnts中
    tts, cnts = [rmSpace(d["title_tks"]) for d in docs if d.get("title_tks")], [
        re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", d["content_with_weight"]) for d in docs]
    tk_count = 0
    if len(tts) == len(cnts):
        # 如果标题列表和内容列表等长
        # 对标题列表分批进行embedding，收集结果到tts_中
        tts_ = np.array([])
        for i in range(0, len(tts), batch_size):
            vts, c = mdl.encode(tts[i: i + batch_size])
            if len(tts_) == 0:
                tts_ = vts
            else:
                tts_ = np.concatenate((tts_, vts), axis=0)
            tk_count += c
            callback(prog=0.6 + 0.1 * (i + 1) / len(tts), msg="")
        tts = tts_
    # 对内容列表分批进行embedding，收集结果到cnts_
    cnts_ = np.array([])
    for i in range(0, len(cnts), batch_size):
        vts, c = mdl.encode(cnts[i: i + batch_size])
        if len(cnts_) == 0:
            cnts_ = vts
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)
        tk_count += c
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    title_w = float(parser_config.get("filename_embd_weight", 0.1))
    # 【标题列表和内容列表等长】对标题和内容的embedding结果进行加权求和，得到最终embedding结果
    vects = (title_w * tts + (1 - title_w) *
             cnts) if len(tts) == len(cnts) else cnts
    # embedding结果应该和文件分块的长度一致，也即一个文件分块，经过embedding处理，得到一个embedding结果
    assert len(vects) == len(docs)
    vector_size = 0
    # 将embedding结果对应追加到文件分块的”q_embedding结果长度_vec“字段中
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        vector_size = len(v)
        d["q_%d_vec" % len(v)] = v
    return tk_count, vector_size


def run_raptor(row, chat_mdl, embd_mdl, callback=None):
    """
    功能：进行raptor处理，得到当前raptor操作得到的分块数据列表【还没有保存】、当前raptor操作得到的分块数据列表的总token数量、embedding输出向量的尺寸
    逻辑：
        1、查询当前文档已保存的分块数据【两个字段：content_with_weight，vctr_nm】列表
        2、基于已保存的分块数据列表【为什么有已保存的，因为raptor任务在非raptor任务之后由update_progress操作生成，非raptor任务会先生成分块数据并保存】、chat模型、embedding模型和任务记录及set_progress操作，进行raptor处理，得到分块数据的两个字段：content_with_weight，vctr_nm
        3、基于任务记录和raptor结果，构造【未保存的】分块数据列表及其content_with_weight的总token数量和embedding模型输出向量尺寸
    """
    # 首先获取embedding模型的示例输出维数
    vts, _ = embd_mdl.encode(["ok"])
    vector_size = len(vts[0])
    # 'q_%d_vec' % 72-->'q_72_vec'
    vctr_nm = "q_%d_vec" % vector_size
    # 初始化分块列表
    chunks = []
    # es检索当前任务记录对应文档【doc_id】的分块数据【两个字段：content_with_weight，vctr_nm】列表【最多1024条】
    for d in settings.retrievaler.chunk_list(row["doc_id"], row["tenant_id"], [str(row["kb_id"])],
                                             fields=["content_with_weight", vctr_nm]):
        # 收集【已保存的】分块数据【(文件分块的文本内容,文件分块的embedding结果)】
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))
    # 创建一个“组织树检索的递归抽象处理”对象
    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"]
    )
    # 【已保存的】分块尺寸
    original_length = len(chunks)
    # TODO 进行“组织树检索的递归抽象处理”
    raptor(chunks, row["parser_config"]["raptor"]["random_seed"], callback)
    # 下述为chunk数据字段的一部分
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"])
    }
    # 当前raptor操作得到的chunk数据列表
    res = []
    # 当前raptor操作得到的chunk数据列表的总的token数量
    tk_count = 0
    # 遍历当前经过raptor处理的分块数据
    for content, vctr in chunks[original_length:]:
        d = copy.deepcopy(doc)
        md5 = hashlib.md5()
        md5.update((content + str(d["doc_id"])).encode("utf-8"))
        d["id"] = md5.hexdigest()
        d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.now().timestamp()
        d[vctr_nm] = vctr.tolist()
        d["content_with_weight"] = content
        d["content_ltks"] = rag_tokenizer.tokenize(content)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
        res.append(d)
        tk_count += num_tokens_from_string(content)
    # 返回：当前raptor操作得到的分块数据列表【还没有保存】、当前raptor操作得到的分块数据列表的总token数量、embedding输出向量的尺寸
    return res, tk_count, vector_size


def do_handle_task(r):
    """
    功能：处理任务
    逻辑：
        1、设置任务处理回调函数【用于更新任务处理情况】和获取embedding模型
        2、根据任务的task_type字段是否为raptor进行分情况处理：
            - raptor处理：
                1、查询当前文档已保存的分块数据【两个字段：content_with_weight，vctr_nm】列表
                2、基于已保存的分块数据列表【为什么有已保存的，因为raptor任务在非raptor任务之后由update_progress操作生成，非raptor任务会先生成分块数据并保存】、chat模型、embedding模型和任务记录及set_progress操作，进行raptor处理，得到分块数据的两个字段：content_with_weight，vctr_nm
                3、基于任务记录和raptor结果，构造【未保存的】分块数据列表及其content_with_weight的总token数量和embedding模型输出向量尺寸
            - 非raptor处理：
                1、依据当前任务对应文档记录的解析器类型获取解析器，对相应文件做分块处理，得到文件分块数据
                2、遍历处理文件分块数据【追加一些基础字段，图片处理【有则存至minio，并将minio图片信息追加至文件分块数据，保证当前分块能找到图片】】，收集到结果中并返回文件分块数据列表
                3、对文件分块数据列表进行embedding处理
        3、批量保存当前任务对应范围内的文件解析数据至es存储
        4、将当前任务的解析结果更新到对应文档记录和对应知识库
    新语法点：偏函数
        背景：函数在执行时，要带上所有必要的参数进行调用。但是，有时参数可以在函数被调用之前提前获知。这种情况下，一个函数有一个或多个参数预先就能用上，以便函数能用更少的参数进行调用。示例如下：
        from functools import partial

        def mod( n, m ):
          return n % m

        mod_by_100 = partial( mod, 100 )

        print mod( 100, 7 )  # 2
        print mod_by_100( 7 )  # 2，这里使用偏函数mod_by_100，相当于固定了函数mod的第一个参数值为100，作为一个新的函数使用，等价于下述定义：
        def mod_by_100( m ):
          return 100 % m
    """
    # partial偏函数：偏函数是将所要承载的函数作为partial()函数的第一个参数，原函数的各个参数依次作为partial()函数后续的参数，除非使用关键字参数。
    callback = partial(set_progress, r["id"], r["from_page"], r["to_page"])
    try:
        # 获取embedding模型
        embd_mdl = LLMBundle(r["tenant_id"], LLMType.EMBEDDING, llm_name=r["embd_id"], lang=r["language"])
    except Exception as e:
        callback(-1, msg=str(e))
        raise
    if r.get("task_type", "") == "raptor":
        # 见collect方法，给任务记录追加task_type的条件：负载消息中的type字段为raptor
        # 对于raptor任务，获取chat模型，进行run_raptor操作
        try:
            # 获取chat模型
            chat_mdl = LLMBundle(r["tenant_id"], LLMType.CHAT, llm_name=r["llm_id"], lang=r["language"])
            # 基于chat模型和embedding模型，处理任务，并利用callback更新任务及通知redis该任务已处理
            # 得到：当前raptor操作得到的分块数据列表【还没有保存】、当前raptor操作得到的分块数据列表的总token数量、embedding输出向量的尺寸
            cks, tk_count, vector_size = run_raptor(r, chat_mdl, embd_mdl, callback)
        except Exception as e:
            callback(-1, msg=str(e))
            raise
    else:
        # 任务记录中没有task_type字段或该字段值非'raptor'时
        st = timer()
        # 进行任务拆分
        cks = build(r)
        logging.info("Build chunks({}): {}".format(r["name"], timer() - st))
        if cks is None:
            return
        if not cks:
            callback(1., "No chunk! Done!")
            return
        # TODO: exception handler
        ## set_progress(r["did"], -1, "ERROR: ")
        callback(
            msg="Finished slicing files ({} chunks in {:.2f}s). Start to embedding the content.".format(len(cks), timer() - st)
        )
        st = timer()
        try:
            # 对经过build得到的文件分块数据列表cls进行embedding处理【方法中，对cks的元素作了embedding后的替换处理，因此不用返回cks】
            tk_count, vector_size = embedding(cks, embd_mdl, r["parser_config"], callback)
        except Exception as e:
            callback(-1, "Embedding error:{}".format(str(e)))
            logging.exception("run_rembedding got exception")
            tk_count = 0
            raise
        logging.info("Embedding elapsed({}): {:.2f}".format(r["name"], timer() - st))
        callback(msg="Finished embedding (in {:.2f}s)! Start to build index!".format(timer() - st))
    # logging.info(f"task_executor init_kb index {search.index_name(r["tenant_id"])} embd_mdl {embd_mdl.llm_name}
    # vector length {vector_size}")
    # 批量保存文档解析数据到es存储中
    init_kb(r, vector_size)
    chunk_count = len(set([c["id"] for c in cks]))
    st = timer()
    es_r = ""
    es_bulk_size = 4
    for b in range(0, len(cks), es_bulk_size):
        es_r = settings.docStoreConn.insert(cks[b:b + es_bulk_size], search.index_name(r["tenant_id"]), r["kb_id"])
        if b % 128 == 0:
            callback(prog=0.8 + 0.1 * (b + 1) / len(cks), msg="")
    logging.info("Indexing elapsed({}): {:.2f}".format(r["name"], timer() - st))
    if es_r:
        callback(-1,
                 "Insert chunk error, detail info please check log file. Please also check Elasticsearch/Infinity "
                 "status!")
        settings.docStoreConn.delete({"doc_id": r["doc_id"]}, search.index_name(r["tenant_id"]), r["kb_id"])
        logging.error('Insert chunk error: ' + str(es_r))
        raise Exception('Insert chunk error: ' + str(es_r))

    if TaskService.do_cancel(r["id"]):
        settings.docStoreConn.delete({"doc_id": r["doc_id"]}, search.index_name(r["tenant_id"]), r["kb_id"])
        return

    callback(msg="Indexing elapsed in {:.2f}s.".format(timer() - st))
    callback(1., "Done!")
    # 将文档解析数据维护到文档记录中，同步更新知识库的文档解析数据
    DocumentService.increment_chunk_num(
        r["doc_id"], r["kb_id"], tk_count, chunk_count, 0)
    logging.info(
        "Chunk doc({}), token({}), chunks({}), elapsed:{:.2f}".format(
            r["id"], tk_count, len(cks), timer() - st))


def handle_task():
    global PAYLOAD, mt_lock, DONE_TASKS, FAILED_TASKS, CURRENT_TASK
    # 获取指定队列指定组指定消费者的一个未关闭任务记录
    task = collect()
    if task:
        try:
            # 处理任务，完成处理时，完成任务数加1
            logging.info(f"handle_task begin for task {json.dumps(task)}")
            with mt_lock:
                CURRENT_TASK = copy.deepcopy(task)
            do_handle_task(task)
            with mt_lock:
                DONE_TASKS += 1
                CURRENT_TASK = None
            logging.info(f"handle_task done for task {json.dumps(task)}")
        except Exception:
            # 处理任务异常时，失败任务数加1
            with mt_lock:
                FAILED_TASKS += 1
                CURRENT_TASK = None
            logging.exception(f"handle_task got exception for task {json.dumps(task)}")
    if PAYLOAD:
        # 在collect方法中有对全局变量PAYLOAD的赋值，PAYLOAD非空时，消费者向队列回复"我收到了数据【msg_id】"
        PAYLOAD.ack()
        PAYLOAD = None


def report_status():
    """
    功能：对于task_consumer_0，每30秒插入一次心跳数据【每次插入时，清理半小时前的心跳数据】
    问题：TODO 队列rag_flow_svr_queue中的组rag_flow_svr_task_broker，从何而来？这里查询了组rag_flow_svr_task_broker中的等大任务数和掉队任务数，不存在组rag_flow_svr_task_broker时，两个任务数取默认值0
    """
    global CONSUMER_NAME, BOOT_AT, PENDING_TASKS, LAG_TASKS, mt_lock, DONE_TASKS, FAILED_TASKS, CURRENT_TASK
    # 将{'TASKEXE':'task_consumer_0'}添加到redis
    REDIS_CONN.sadd("TASKEXE", CONSUMER_NAME)
    while True:
        try:
            now = datetime.now()
            # 队列名称：rag_flow_svr_queue，组名：rag_flow_svr_task_broker
            # 从redis查询指定队列queue的指定组group_name，不存在则返回none
            group_info = REDIS_CONN.queue_info(SVR_QUEUE_NAME, "rag_flow_svr_task_broker")
            if group_info is not None:
                # 组非空，取组中的pending【等待中】任务数量和lag【掉队的】任务数量
                PENDING_TASKS = int(group_info["pending"])
                LAG_TASKS = int(group_info["lag"])

            with mt_lock:
                # 语法：字典转json字符串
                # 构造一个心跳数据：消费者名称'task_consumer_0'、当时时间、task_executor启动时间、等待任务数、掉队任务数、完成任务数【初始化0】、失败任务数【初始化0】、当前任务【初始化none】
                heartbeat = json.dumps({
                    "name": CONSUMER_NAME,
                    "now": now.isoformat(),
                    "boot_at": BOOT_AT,
                    "pending": PENDING_TASKS,
                    "lag": LAG_TASKS,
                    "done": DONE_TASKS,
                    "failed": FAILED_TASKS,
                    "current": CURRENT_TASK,
                })
            # 将对应消费者的心跳数据以有序集合的形式放入redis，score取当前时间戳【利用zrangebyscore取值时，按照score从小到大排序，也即先放入的先取出】
            REDIS_CONN.zadd(CONSUMER_NAME, heartbeat, now.timestamp())
            logging.info(f"{CONSUMER_NAME} reported heartbeat: {heartbeat}")
            # 统计距今30分钟以前的对应CONSUMER_NAME的心跳数量
            expired = REDIS_CONN.zcount(CONSUMER_NAME, 0, now.timestamp() - 60 * 30)
            if expired > 0:
                # 如果距今30分钟前，存在心跳，则将这些心跳清除
                REDIS_CONN.zpopmin(CONSUMER_NAME, expired)
        except Exception:
            logging.exception("report_status got exception")
        # 两次循环之间睡30秒
        time.sleep(30)


def main():
    """
    任务的声明周期：
        1、手动触发解析操作产生任务列表并放入redis中的指定队列
        2、collect方法获取任务
        3、根据task_type分情况处理任务
            - task_type为raptor时，选用chat模型，经run_raptor逻辑，走后续保存逻辑
            - 其他，经build逻辑，选用embedding模型，走后续保存逻辑
    备注：
        1、task_type字段的由来：
            - doc记录中存在parser_config【取值为知识库记录的parser_config字段，见上传文件upload_document】和progress_msg字段，在启动脚本ragflow_server中的update_progress步骤中有以下逻辑：
                if d["parser_config"].get("raptor", {}).get("use_raptor") and d["progress_msg"].lower().find(
                            " raptor") < 0:
                    # 若文档parser_config字段的raptor字段存在非空字段use_raptor且文档progress_msg字段的小写模式不存在raptor字符串，则：
                    # 1、创建指定文档的“组织树检索的递归抽象处理”任务，并插入数据库，插入redis消息队列rag_flow_svr_queue，redis插入失败，则抛出异常
                    queue_raptor_tasks(d)
            - 在queue_raptor_tasks方法中，有以下操作：
                # 【注意任务没有type字段，这里是为了追加type字段，标识当前任务是raptor任务，后续处理任务时，选用raptor任务处理逻辑】设置任务type字段为raptor，并将任务放入redis消息队列rag_flow_svr_queue中【若插入队列不成功，则抛出异常】
                task["type"] = "raptor"
                assert REDIS_CONN.queue_product(SVR_QUEUE_NAME, message=task), "Can't access Redis. Please check the Redis' status."
            - 在collect方法中，有以下操作：
                if msg.get("type", "") == "raptor":
                    # 负载消息中的type字段为raptor时，为任务记录添加task_type字段并设置值为raptor
                    task["task_type"] = "raptor"
    """
    # 依据配置文件service_conf.yaml和环境变量初始化配置信息
    settings.init_settings()
    # 逻辑见report_status，对于task_consumer_0，每30秒插入一次心跳数据【每次插入时，清理半小时前的心跳数据】
    background_thread = threading.Thread(target=report_status)
    background_thread.daemon = True
    background_thread.start()

    while True:
        # 处理任务【文件处理】
        handle_task()


if __name__ == "__main__":
    main()
