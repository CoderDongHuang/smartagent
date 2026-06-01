"""
系统初始化脚本
"""
import logging
from config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD,
    CHROMA_PERSIST_DIR
)
from knowledge_base import KnowledgeBaseEngine
from ticket_system import TicketSystemAPI
from conversation_manager import ConversationManager
from agent import CustomerServiceAgent

logger = logging.getLogger(__name__)


def init_systems():
    """初始化所有系统组件"""
    logger.info("=" * 50)
    logger.info("开始初始化系统组件")
    logger.info("=" * 50)

    # 1. 初始化知识库
    logger.info("1. 初始化知识库...")
    knowledge_base = KnowledgeBaseEngine(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="customer_service_kb",
        bm25_weight=0.3,
        vector_weight=0.7,
        top_k=5
    )
    logger.info("✓ 知识库初始化完成")

    # 2. 初始化工单系统
    logger.info("2. 初始化工单系统...")
    ticket_system = TicketSystemAPI()
    logger.info("✓ 工单系统初始化完成")

    # 3. 初始化对话管理器
    logger.info("3. 初始化对话管理器...")
    conversation_manager = ConversationManager(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        ttl=3600
    )
    logger.info("✓ 对话管理器初始化完成")

    # 4. 初始化Agent
    logger.info("4. 初始化智能客服Agent...")
    agent = CustomerServiceAgent(
        knowledge_base=knowledge_base,
        ticket_system=ticket_system,
        conversation_manager=conversation_manager,
        llm_model="deepseek-v4-flash",
        temperature=0.7
    )
    logger.info("✓ Agent初始化完成")

    logger.info("=" * 50)
    logger.info("所有系统组件初始化完成")
    logger.info("=" * 50)

    return {
        'knowledge_base': knowledge_base,
        'ticket_system': ticket_system,
        'conversation_manager': conversation_manager,
        'agent': agent
    }
