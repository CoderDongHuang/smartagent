"""
混合检索器：结合BM25和向量检索
提升召回率和排序质量
"""
import logging
from typing import List, Tuple, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from models import KnowledgeResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器 - 结合BM25和向量相似度"""

    def __init__(
        self,
        vectorstore: Chroma,
        embeddings: Embeddings,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        top_k: int = 5
    ):
        """
        初始化混合检索器

        Args:
            vectorstore: Chroma向量存储实例
            embeddings: 嵌入模型
            bm25_weight: BM25权重
            vector_weight: 向量检索权重
            top_k: 返回结果数量
        """
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.top_k = top_k

        # BM25索引相关
        self.bm25 = None
        self.corpus = []
        self.doc_ids = []
        self._initialized = False

    def initialize_bm25(self):
        """从向量库提取文档并初始化BM25索引"""
        logger.info("初始化BM25索引...")

        # 从Chroma获取所有文档
        try:
            all_docs = self.vectorstore.get()

            if not all_docs or 'documents' not in all_docs:
                logger.warning("向量库为空，无法初始化BM25")
                return

            self.corpus = all_docs['documents']
            self.doc_ids = all_docs['ids']

            # 中文分词预处理（简单按空格分词，生产环境可用jieba）
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]

            # 创建BM25索引
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._initialized = True

            logger.info(f"BM25索引初始化完成，共{len(self.corpus)}个文档")
            logger.debug(f"BM25文档IDs示例: {self.doc_ids[:3]}")

        except Exception as e:
            logger.error(f"BM25索引初始化失败: {e}")
            raise

    def bm25_search(self, query: str, k: int = None) -> List[Tuple[str, float]]:
        """
        BM25检索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            [(文档ID, 分数)] 列表，按分数降序
        """
        if not self._initialized:
            logger.warning("BM25未初始化，执行初始化")
            self.initialize_bm25()

        k = k or self.top_k * 2

        # 对查询分词
        query_tokens = query.lower().split()

        # BM25评分
        scores = self.bm25.get_scores(query_tokens)

        logger.debug(f"BM25原始分数: {scores}")

        # 获取Top-K分数和索引
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            # 即使分数为0也返回（因为中文分词可能不准确）
            doc_id = self.doc_ids[idx]
            score = float(scores[idx])
            # 给所有文档一个基础分数，避免全0
            if score == 0:
                score = 0.1  # 基础分数
            results.append((doc_id, score))

        logger.debug(f"BM25检索返回 {len(results)} 个结果")
        if results:
            logger.debug(f"BM25 IDs: {[r[0] for r in results[:3]]}")
        return results

    def vector_search(self, query: str, k: int = None) -> List[Tuple[str, float]]:
        """
        向量相似度检索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            [(文档ID, 距离分数)] 列表
        """
        k = k or self.top_k * 2

        try:
            # 使用Chroma进行向量检索
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )

            # 转换格式
            formatted_results = []
            for doc, distance in results:
                similarity_score = 1.0 / (1.0 + distance)

                # 关键修复：从metadata中获取id字段
                doc_id = doc.metadata.get('id')

                # 如果metadata中没有id，尝试其他方式
                if not doc_id:
                    # 通过文档内容在corpus中查找对应的ID
                    if self.corpus and doc.page_content in self.corpus:
                        idx = self.corpus.index(doc.page_content)
                        if idx < len(self.doc_ids):
                            doc_id = self.doc_ids[idx]

                    # 最后的备选方案
                    if not doc_id:
                        doc_id = f"doc_{abs(hash(doc.page_content)) % 10000}"

                formatted_results.append((doc_id, similarity_score))

            logger.debug(f"向量检索返回 {len(formatted_results)} 个结果")
            if formatted_results:
                logger.debug(f"Vector IDs: {[r[0] for r in formatted_results[:3]]}")
            return formatted_results

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-Max归一化分数到[0,1]区间
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [0.5] * len(scores)

        normalized = [(s - min_score) / (max_score - min_score) for s in scores]
        return normalized

    def hybrid_search(self, query: str) -> List[KnowledgeResult]:
        """
        混合检索主方法

        Args:
            query: 查询文本

        Returns:
            知识库结果列表，按混合分数排序
        """
        try:
            # 1. 执行两种检索
            bm25_results = self.bm25_search(query)
            vector_results = self.vector_search(query)

            logger.debug(f"BM25结果数: {len(bm25_results)}, Vector结果数: {len(vector_results)}")

            # 详细调试：打印IDs
            if bm25_results:
                logger.info(f"BM25返回IDs: {[r[0] for r in bm25_results[:3]]}")
            if vector_results:
                logger.info(f"Vector返回IDs: {[r[0] for r in vector_results[:3]]}")

            # 检查是否有共同的ID
            bm25_ids = set([r[0] for r in bm25_results])
            vector_ids = set([r[0] for r in vector_results])
            common_ids = bm25_ids & vector_ids
            logger.info(f"BM25 IDs集合大小: {len(bm25_ids)}")
            logger.info(f"Vector IDs集合大小: {len(vector_ids)}")
            logger.info(f"共同IDs数量: {len(common_ids)}")

            # 2. 构建文档ID到结果的映射
            bm25_dict = {doc_id: score for doc_id, score in bm25_results}
            vector_dict = {doc_id: score for doc_id, score in vector_results}


            # 3. 合并所有文档ID
            all_doc_ids = set(bm25_dict.keys()) | set(vector_dict.keys())

            if not all_doc_ids:
                logger.warning("两种检索均未返回结果")
                return []

            # 4. 归一化分数
            bm25_scores = self.normalize_scores(list(bm25_dict.values()))
            vector_scores = self.normalize_scores(list(vector_dict.values()))

            bm25_normalized = dict(zip(bm25_dict.keys(), bm25_scores))
            vector_normalized = dict(zip(vector_dict.keys(), vector_scores))

            # 5. 计算混合分数
            hybrid_scores = {}
            for doc_id in all_doc_ids:
                bm25_score = bm25_normalized.get(doc_id, 0.0)
                vector_score = vector_normalized.get(doc_id, 0.0)

                hybrid_score = (
                    self.bm25_weight * bm25_score +
                    self.vector_weight * vector_score
                )
                hybrid_scores[doc_id] = hybrid_score

            # 6. 按混合分数排序
            sorted_doc_ids = sorted(
                hybrid_scores.keys(),
                key=lambda x: hybrid_scores[x],
                reverse=True
            )

            # 7. 获取Top-K结果并构建返回对象
            final_results = []
            for doc_id in sorted_doc_ids[:self.top_k]:
                # 从向量库获取完整文档信息
                try:
                    docs = self.vectorstore.get(ids=[doc_id])

                    if docs and docs['documents']:
                        content = docs['documents'][0]
                        metadata = docs['metadatas'][0] if docs['metadatas'] else {}

                        result = KnowledgeResult(
                            content=content,
                            source=metadata.get('source', 'unknown'),
                            score=hybrid_scores[doc_id],
                            metadata=metadata
                        )
                        final_results.append(result)
                except Exception as e:
                    logger.warning(f"获取文档 {doc_id} 失败: {e}")
                    continue

            logger.info(f"混合检索返回 {len(final_results)} 个结果")
            return final_results

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        添加文档到向量库并更新BM25索引
        """
        try:
            # 添加到Chroma向量库
            ids = [f"doc_{i}" for i in range(len(documents))]
            self.vectorstore.add_texts(
                texts=documents,
                metadatas=metadatas,
                ids=ids
            )

            # 重新初始化BM25索引
            self.initialize_bm25()

            logger.info(f"成功添加 {len(documents)} 个文档")

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        添加文档到向量库并更新BM25索引
        """
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.add_documents_with_ids(documents, metadatas, ids)

    def add_documents_with_ids(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """
        添加文档到向量库并更新BM25索引（带自定义IDs）
        """
        try:
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]

            # 添加到Chroma向量库
            self.vectorstore.add_texts(
                texts=documents,
                metadatas=metadatas,
                ids=ids
            )

            # 重新初始化BM25索引
            self.initialize_bm25()

            logger.info(f"成功添加 {len(documents)} 个文档")

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

