"""
RAG知识库检索引擎
集成Chroma向量数据库和混合检索器
"""
import os
import logging

# 必须在导入其他模块之前设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import CHROMA_PERSIST_DIR
from hybrid_retriever import HybridRetriever
from models import KnowledgeResult

logger = logging.getLogger(__name__)


class KnowledgeBaseEngine:
    """知识库检索引擎"""

    def __init__(
            self,
            persist_directory: str = None,
            collection_name: str = "customer_service_kb",
            embedding_model: str = "shibing624/text2vec-base-chinese",
            bm25_weight: float = 0.3,
            vector_weight: float = 0.7,
            top_k: int = 5,
            use_china_mirror: bool = True
    ):
        """
        初始化知识库引擎

        Args:
            persist_directory: Chroma持久化目录
            collection_name: 集合名称
            embedding_model: 嵌入模型名称（默认使用中文模型）
            bm25_weight: BM25权重
            vector_weight: 向量权重
            top_k: 返回结果数量
            use_china_mirror: 是否使用国内镜像
        """
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self.top_k = top_k

        # 记录使用的镜像
        logger.info(f"使用HuggingFace镜像: {os.environ.get('HF_ENDPOINT')}")

        # 初始化嵌入模型
        logger.info(f"加载嵌入模型: {embedding_model}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 初始化或加载Chroma向量库
        self.vectorstore = self._init_vectorstore()

        # 初始化混合检索器
        self.hybrid_retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            embeddings=self.embeddings,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            top_k=top_k
        )

        logger.info("知识库引擎初始化完成")

    def _init_vectorstore(self) -> Chroma:
        """初始化向量存储"""
        try:
            # 确保持久化目录存在
            os.makedirs(self.persist_directory, exist_ok=True)

            # 加载或创建Chroma向量库
            vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

            logger.info(f"向量库加载完成，路径: {self.persist_directory}")
            return vectorstore

        except Exception as e:
            logger.error(f"向量库初始化失败: {e}")
            raise

    def search(self, query: str) -> List[KnowledgeResult]:
        """
        检索知识库

        Args:
            query: 查询文本

        Returns:
            相关知识列表
        """
        try:
            logger.info(f"检索知识库: {query}")
            results = self.hybrid_retriever.hybrid_search(query)

            if results:
                logger.info(f"找到 {len(results)} 条相关知识")
            else:
                logger.warning("未找到相关知识")

            return results

        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            raise

    def add_knowledge(
        self,
        documents: List[str],
        sources: List[str] = None,
        categories: List[str] = None
    ):
        """
        添加知识到知识库

        Args:
            documents: 文档内容列表
            sources: 来源列表
            categories: 分类列表
        """
        try:
            # 构建元数据（id与Chroma的ids保持一致）
            metadatas = []
            ids = []
            for i in range(len(documents)):
                doc_id = f"doc_{i}"
                metadata = {
                    'source': sources[i] if sources else 'manual_input',
                    'category': categories[i] if categories else 'general',
                    'id': doc_id  # 使用相同的ID
                }
                metadatas.append(metadata)
                ids.append(doc_id)

            # 添加文档（传入统一的ids）
            self.hybrid_retriever.add_documents_with_ids(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"成功添加 {len(documents)} 条知识")

        except Exception as e:
            logger.error(f"添加知识失败: {e}")
            raise


    def load_from_files(self, file_paths: List[str]):
        """
        从文件加载知识（支持txt, md等文本文件）

        Args:
            file_paths: 文件路径列表
        """
        documents = []
        sources = []

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 可以按段落分割，这里简单整个文件作为一条知识
                    documents.append(content)
                    sources.append(file_path)
            except Exception as e:
                logger.error(f"读取文件 {file_path} 失败: {e}")

        if documents:
            self.add_knowledge(documents, sources)
            logger.info(f"从 {len(file_paths)} 个文件加载知识完成")

    def clear_knowledge_base(self):
        """清空知识库"""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = self._init_vectorstore()
            logger.warning("知识库已清空")
        except Exception as e:
            logger.error(f"清空知识库失败: {e}")
            raise

    def get_knowledge_count(self) -> int:
        """获取知识库文档数量"""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return count
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

