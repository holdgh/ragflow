文件解析处理-基于redis消息队列的离线异步处理
消息队列的消费模块rag/svr/task_executor.py中的main方法--
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