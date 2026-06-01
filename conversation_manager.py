"""
Redis对话历史管理器
负责存储和检索多轮对话上下文
"""
import json
import time
import logging
from typing import List, Optional
from redis import Redis
from models import ChatMessage

logger = logging.getLogger(__name__)


class ConversationManager:
    """对话历史管理器"""

    def __init__(self, host='localhost', port=6379, db=0, password=None, ttl=3600):
        """
        初始化Redis连接

        Args:
            host: Redis主机
            port: Redis端口
            db: Redis数据库编号
            password: Redis密码
            ttl: 会话过期时间(秒)，默认1小时
        """
        try:
            self.redis_client = Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            self.ttl = ttl
            # 测试连接
            self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
            raise

    def _get_conversation_key(self, conversation_id: str) -> str:
        """生成Redis键名"""
        return f"conversation:{conversation_id}"

    def save_message(self, conversation_id: str, message: ChatMessage):
        """
        保存单条消息到对话历史

        Args:
            conversation_id: 会话ID
            message: 聊天消息对象
        """
        key = self._get_conversation_key(conversation_id)
        message_dict = message.dict()
        if not message.timestamp:
            message_dict['timestamp'] = time.time()

        # 使用列表存储消息
        self.redis_client.rpush(key, json.dumps(message_dict))
        # 设置过期时间
        self.redis_client.expire(key, self.ttl)

        logger.debug(f"消息已保存到会话 {conversation_id}")

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[ChatMessage]:
        """
        获取对话历史

        Args:
            conversation_id: 会话ID
            limit: 返回最近的消息数量

        Returns:
            聊天消息列表
        """
        key = self._get_conversation_key(conversation_id)

        # 获取最后N条消息
        messages = self.redis_client.lrange(key, -limit, -1)

        chat_messages = []
        for msg_json in messages:
            try:
                msg_dict = json.loads(msg_json)
                chat_messages.append(ChatMessage(**msg_dict))
            except Exception as e:
                logger.warning(f"解析消息失败: {e}")
                continue

        return chat_messages

    def clear_conversation(self, conversation_id: str):
        """
        清空指定会话的历史记录

        Args:
            conversation_id: 会话ID
        """
        key = self._get_conversation_key(conversation_id)
        self.redis_client.delete(key)
        logger.info(f"会话 {conversation_id} 已清空")

    def get_all_conversations(self) -> List[str]:
        """
        获取所有活跃会话ID

        Returns:
            会话ID列表
        """
        keys = self.redis_client.keys("conversation:*")
        return [key.split(":")[1] for key in keys]

    def format_history_for_llm(self, conversation_id: str, limit: int = 10) -> str:
        """
        格式化对话历史为LLM可用的字符串格式

        Args:
            conversation_id: 会话ID
            limit: 消息数量限制

        Returns:
            格式化的对话历史字符串
        """
        messages = self.get_conversation_history(conversation_id, limit)

        formatted_lines = []
        for msg in messages:
            role_label = "用户" if msg.role == "user" else "助手"
            formatted_lines.append(f"{role_label}: {msg.content}")

        return "\n".join(formatted_lines)
