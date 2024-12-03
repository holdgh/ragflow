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
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED, wait
from threading import Lock
from typing import Tuple
import umap
import numpy as np
from sklearn.mixture import GaussianMixture

from rag.utils import truncate


class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    def __init__(self, max_cluster, llm_model, embd_model, prompt, max_token=512, threshold=0.1):
        self._max_cluster = max_cluster
        self._llm_model = llm_model
        self._embd_model = embd_model
        self._threshold = threshold
        self._prompt = prompt
        self._max_token = max_token

    def _get_optimal_clusters(self, embeddings: np.ndarray, random_state:int):
        """
        功能：获取高斯混合模型聚类拟合embedding数据误差最小的聚类数量【从1到max_clusters，遍历拟合收集误差，择最小误差选择聚类数量】
        """
        # 设置最大聚类总数
        max_clusters = min(self._max_cluster, len(embeddings))
        # 利用等差数列设置聚类数目列表，例如np.arange(1,10)-->array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            # 获取当前聚类数量的高斯混合模型
            gm = GaussianMixture(n_components=n, random_state=random_state)
            # 利用当前高斯混合模型拟合embedding数据
            gm.fit(embeddings)
            # 收集当前高斯混合模型的拟合误差
            bics.append(gm.bic(embeddings))
        # 选取拟合误差最小的聚类数量作为返回结果
        optimal_clusters = n_clusters[np.argmin(bics)]
        return optimal_clusters

    def __call__(self, chunks: Tuple[str, np.ndarray], random_state, callback=None):
        """
        功能：raptor任务处理【递归处理】
        逻辑：
            1、获取当前chunks的起止索引[start, end-1]【当end-start<=1时,循环终止】
            2、summarize处理：对chunks中索引在ck_idx【索引列表】的内容进行基于大模型和知识库记录parser_config中提示词的处理【摘要总结】后，接着对大模型输出做embedding处理，并将embedding结果和大模型输出追加到chunks中
            3、更新起止索引：
                start = end
                end = len(chunks) # 由于chunks经过summarize处理后，追加了新的元组元素【(文本内容，文本内容的embedding结果)】，因此这里的索引更新，正常情况下满足end-start>0
            关键逻辑说明：ck_idx的获取方式
                - 当end=start+2时，ck_idx为[start, start+1]
                - 当end>start+2时，对chunks[start:end]，采用降维聚类的方式，获取对应类目的ck_idx【比如聚类有5个，则对应执行5次summarize逻辑，对应的ck_idx为属于相应类目的文件分块索引】
        入参：
            chunks：已经保存的文件分块解析数据【(文件分块的文本内容,文件分块的embedding结果)，元组列表】列表【TODO 此来源于普通任务的处理】
            random_state：row["parser_config"]["raptor"]["random_seed"]--来源于任务对应文档记录中的parser_config【来源于知识库记录】字段
            callback：任务处理回调函数，用于维护处理任务过程的处理情况
        """
        layers = [(0, len(chunks))]
        start, end = 0, len(chunks)
        # 还未完成普通任务时【见ragflow_server中的update_progress操作，完成普通任务时，才会判断解析配置数据生成raptor任务】，直接返回
        if len(chunks) <= 1: return
        # 过滤掉embedding结果为空的数据
        chunks = [(s, a) for s, a in chunks if len(a) > 0]

        def summarize(ck_idx, lock):
            """
            功能：对chunks中索引在ck_idx【索引列表】的内容进行基于大模型和知识库记录parser_config中提示词的处理后，接着对大模型输出做embedding处理，并将embedding结果和大模型输出追加到chunks中
            """
            nonlocal chunks
            try:
                # 获取文件分块索引范围内的文本内容列表
                texts = [chunks[i][0] for i in ck_idx]
                # 获取平均文本内容token允许长度
                len_per_chunk = int((self._llm_model.max_length - self._max_token)/len(texts))
                # 对文本内容进行基于token允许长度的截断处理，只取token允许长度范围内的文本内容
                cluster_content = "\n".join([truncate(t, max(1, len_per_chunk)) for t in texts])
                # self_prompt来源于当前任务对应文档记录的parser_config【来源于知识库】中的提示词
                """
                官方demo环境提示词初始配置如下：
                '''
                请总结以下段落。 小心数字，不要编造。 段落如下：
                    {cluster_content}
                以上就是你需要总结的内容。
                '''
                由上述提示词可知，这里是对文件分块列表做摘要总结。结合对embedding数据降维聚类，可知raptor的逻辑：
                    1、对文件分块解析结果列表【[start，end)范围内的】做摘要总结。
                        - 总览：【元组列表，(文本内容，文本内容的embedding结果)，可以将embedding结果分两个方面看待：表征当前分块内容的语义；可看作是文本内容的索引【体现在，对embedding数据降维聚类，取同一类的文本内容，进行summarize操作】】
                        - 分步：
                            - 提取当前范围内的embedding结果列表，进行降维聚类处理，得到n个类别
                            - 对当前范围内的embedding结果列表，做类别预测，选出同一类的索引，以取到相应的【同一类别的】文本内容列表，做summarize处理【文本摘要和embedding】，追加到文本分块解析结果列表中【后续更新start和end时，是对新得到这些文本分块解析结果做同步骤1的处理】
                    2、更新start和end，处理新得到的那些文本分块解析结果
                    3、直到start和end相差小于2时【也即第1步最多生成1个(摘要总结内容，摘要总结内容的embedding结果)】，raptor逻辑结束
                """
                cnt = self._llm_model.chat("You're a helpful assistant.",
                                             [{"role": "user", "content": self._prompt.format(cluster_content=cluster_content)}],
                                             {"temperature": 0.3, "max_tokens": self._max_token}
                                             )
                # 去除大模型输出中的“(······\n由于长度的原因，回答被截断了，要继续吗？|For the content length reason, it stopped, continue?)”
                cnt = re.sub("(······\n由于长度的原因，回答被截断了，要继续吗？|For the content length reason, it stopped, continue?)", "", cnt)
                logging.debug(f"SUM: {cnt}")
                # 进行embedding操作
                embds, _ = self._embd_model.encode([cnt])
                with lock:
                    # embedding结果为空时，直接返回
                    if not len(embds[0]): return
                    # 将大模型输出和其embedding结果收集到chunks中
                    # 这里的append操作使得__call__方法中的start和end索引更新变得有效
                    chunks.append((cnt, embds[0]))
            except Exception as e:
                logging.exception("summarize got exception")
                return e

        labels = []
        while end - start > 1:
            # 收集当前所有【start:end】文件分块的embedding结果
            embeddings = [embd for _, embd in chunks[start: end]]
            if len(embeddings) == 2:
                # 当前只有两个文件分块时
                summarize([start, start+1], Lock())
                if callback:
                    callback(msg="Cluster one layer: {} -> {}".format(end-start, len(chunks)-end))
                labels.extend([0,0])
                layers.append((end, len(chunks)))
                # 更新下一次处理的起止索引
                start = end
                end = len(chunks)
                continue
            # 取当前所有文件分块的8成
            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            """
             UMAP参数解释
             UMAP(n_neighbors=100, # 邻近参数，根据想探索的数据大小确定数值。
               n_components=3, # 嵌入的空间维度，default=2
               metric='euclidean', # 计算高维空间中距离的度量方式
               n_epochs=1000, # default None,用于优化低维嵌入的训练轮数。较大的值会产生更准确的嵌入结果
               learning_rate=1.0, # default 1.0, 嵌入优化的初始学习率。
               init='spectral', # default 'spectral',它表示低维嵌入的初始化方式。可选的选项有：{'spectral', 'random', 一个初始嵌入位置的NumPy数组}。
               min_dist=0.1, # default 0.1, 它表示嵌入点之间的有效最小距离。
               spread=1.0, # default 1.0, 它表示嵌入点的有效尺度。与min_dist结合使用，决定了嵌入点的聚集程度。
               low_memory=False, # default False, 对于某些数据集，最近邻计算可能会消耗大量内存。如果发现UMAP由于内存限制而失败，请考虑将此选项设置为True。
               set_op_mix_ratio=1.0, # default 1.0, 此参数的值应介于0.0和1.0之间；值为1.0将使用纯模糊并集，而值为0.0将使用纯模糊交集。
               local_connectivity=1, # default 1, 它表示所需的局部连接性，即在局部级别上应该假设连接的最近邻数量。
               repulsion_strength=1.0, # default 1.0, 它是在低维嵌入优化中应用于负样本的加权值。
               negative_sample_rate=5, # default 5, 增加此值将导致应用更大的斥力力量，增加优化成本，但略微提高准确性。
               transform_queue_size=4.0, # default 4.0, 较大的值将导致较慢的性能，但更准确的最近邻评估。
               a=None, # 更具体的参数，用于控制嵌入。如果为None，则这些值将根据"min_dist"和"spread"的确定自动设置。
               b=None, # default None, 用于控制嵌入。如果为None，则这些值将根据"min_dist"和"spread"的确定自动设置。
               random_state=42, # default: None, random_state是随机数生成器使用的种子。
               metric_kwds=None, # default None) 传递给度量函数的参数，例如Minkowski距离的"p"值。
               angular_rp_forest=False, # default False,是否使用角度随机投影森林来初始化近似最近邻搜索。
               target_n_neighbors=-1, # default -1,用于构建目标简单集的最近邻数量。如果设置为-1，则使用n_neighbors的值。
               transform_seed=42, # default 42,用于转换操作的随机种子，用于处理转换过程中的随机性。
               verbose=False, # default False,控制日志输出的详细程度。
               unique=False, # default False, 控制在进行嵌入之前是否对数据的行进行唯一化处理。如果设置为True，则会删除重复的行，保留唯一的行。
              )
            """
            # umap 一种非线性降维和可视化算法。它通过构建数据点之间的邻近关系图，并利用图的拓扑结构进行流形近似和优化。可用在高维度的数据处理中，使其可视化
            # TODO 这里降维的作用：提取主要特征，减少后续高斯混合模型的计算量？
            reduced_embeddings = umap.UMAP(
                n_neighbors=max(2, n_neighbors), n_components=min(12, len(embeddings)-2), metric="cosine"
            ).fit_transform(embeddings)
            # 获取最佳聚类数量【高斯混合模型算法【适用于数据分布不满足明显的“圆形”聚类场景】与k均值算法【适用于近似圆形的聚类场景】作用类似，聚类】
            n_clusters = self._get_optimal_clusters(reduced_embeddings, random_state)
            # 获取
            if n_clusters == 1:
                # 只有一个聚类时，直接初始化reduced_embeddings属于这个聚类中心的概率为0【相当于没有聚类，或者只有一个聚类】
                lbls = [0 for _ in range(len(reduced_embeddings))]
            else:
                # 多于一个聚类时【取reduced_embeddings属于所有聚类中心的概率的超过阈值的第一个概率】
                # 获取最佳聚类数量的高斯混合模型
                gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
                # 拟合embedding数据
                gm.fit(reduced_embeddings)
                # 预测给定数据属于每个聚类中心的后验概率【即输入数据属于每个聚类的概率】
                probs = gm.predict_proba(reduced_embeddings)
                # 将概率不大于阈值的置为array([], dtype=int64)，大于阈值的置为array([0], dtype=int64)；将这些空列表和[0]收集到列表lbls中
                lbls = [np.where(prob > self._threshold)[0] for prob in probs]
                # TODO 这里会因索引越界报错？【空列表时，没法取第一个元素】
                lbls = [lbl[0] if isinstance(lbl, np.ndarray) else lbl for lbl in lbls]
            lock = Lock()
            with ThreadPoolExecutor(max_workers=12) as executor:
                threads = []
                # [1,n_clusters)之间的正整数数列
                for c in range(n_clusters):
                    # 获取属于第c个聚类的文件分块索引
                    ck_idx = [i+start for i in range(len(lbls)) if lbls[i] == c]
                    threads.append(executor.submit(summarize, ck_idx, lock))
                wait(threads, return_when=ALL_COMPLETED)
                logging.debug(str([t.result() for t in threads]))

            assert len(chunks) - end == n_clusters, "{} vs. {}".format(len(chunks) - end, n_clusters)
            labels.extend(lbls)
            layers.append((end, len(chunks)))
            # 更新raptor任务处理情况
            if callback:
                callback(msg="Cluster one layer: {} -> {}".format(end-start, len(chunks)-end))
            # 更新起止索引
            start = end
            end = len(chunks)

