源码结构：
api-后端api
web-前端页面
deepdoc-文件解析
conf-配置信息

业务：文件上传，手动触发解析，问答逻辑

核心：文件解析与问答

文件解析：
    接口：/v1/document/run
    处理逻辑：
    1、api/db/services/task_service.py中的queue_tasks方法【创建多个异步任务】--根据文件内容拆分为多个任务，通过redis消息队列进行暂存，之后进行离线异步处理
    2、消息队列的消费模块rag/svr/task_executor.py中的main方法--
        - 调用collect方法从消息队列中获取任务
        - 接下来每个任务依次调用build方法进行文件解析
            - 根据parser_id选择合适的解析器组【解析器组包含pdf、word等格式的文件解析，一个解析器组可以理解为一个使用场景】
            - 以默认的native类型为例深入对应的chunk方法实现，位于rag/app/native.py中，该方法包含了docx、pdf、xlsx、md等格式的解析，以pdf为例：
                - 解析器继承自deepdoc/parser/pdf_parser.py中的RAGFlowPdfParser
                - 基于PyPDF2实现打开pdf文件，基于pdfplumber实现表格数据的提取【相对PyMuPDF速度更慢，但是处理更精细】
                - 使用的OCR模型为/InfiniFlow/deepdoc，在解析中额外加载了一个XGB模型InfiniFlow/text_concat_xgb_v1.0用于内容提取
                - 从解析效果看，RAGFlow可以将解析后的文本块与原始文档中的原始位置关联起来。目前只有RAGFlow实现了该功能。
                - 注意1：在ragflow的文件预处理中，具有一些数据【版权内容，参考信息等】的清理操作【在很多地方出现特殊处理，难以拆分出明确的预处理逻辑】，例如在deepdoc/vision/layout_recognizer.py
                - 注意2：ragflow将提取的内容分为普通的文本和表格，分别对其进行tokenize，方便进行检索
        - 调用embedding方法进行向量化
        - 调用ELASTICSEARCH.bulk方法写入ES存储

问答：
    接口：/v1/conversation/completion【实际对话逻辑在api/db/services/dialog_service.py中的chat方法完成】
    处理逻辑：
        1、ragflow中的检索为混合检索，实现文本检索和向量检索，混合检索依赖ES实现【将混合检索操作转换为复杂的查询条件，利用ES-dsl进行复杂查询构造，然后直接提交给ES即可】
        2、对检索结果进行重排【rag/nlp/search.py中的rerank方法】。重排是基于文本匹配得分和向量匹配得分混合进行排序，默认文本匹配的权重为0.3，向量匹配的权重为0.7。基于混合的相似得分进行过滤和重排，默认混合得分低于0.2的会被过滤掉
        3、重排之后的逻辑与qanything类似，基于大模型token数量限制的文档处理和构造提示词给到大模型进行回答。