"""
LangChain Agent核心逻辑
集成RAG、Function Calling和对话管理
"""
import uuid
import logging
import os
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from knowledge_base import KnowledgeBaseEngine
from ticket_system import TicketSystemAPI
from conversation_manager import ConversationManager
from models import (
    QueryType, AgentResponse, KnowledgeResult,
    TicketRequest, TicketResponse, ChatMessage
)

logger = logging.getLogger(__name__)


class CustomerServiceAgent:
    """智能客服Agent"""

    def __init__(
        self,
        knowledge_base: KnowledgeBaseEngine,
        ticket_system: TicketSystemAPI,
        conversation_manager: ConversationManager,
        llm_model: str = "deepseek-chat",
        temperature: float = 0.7
    ):
        """
        初始化Agent

        Args:
            knowledge_base: 知识库引擎
            ticket_system: 工单系统API
            conversation_manager: 对话管理器
            llm_model: LLM模型名称
            temperature: 温度参数
        """
        self.knowledge_base = knowledge_base
        self.ticket_system = ticket_system
        self.conversation_manager = conversation_manager

        # 初始化LLM
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["OPENAI_API_BASE"] = OPENAI_BASE_URL

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature
        )

        # 定义工具
        self.tools = self._create_tools()

        # 创建Agent
        self.agent_executor = self._create_agent()

        logger.info("智能客服Agent初始化完成")

    def _create_tools(self):
        """创建Agent可用的工具"""

        @tool
        def search_knowledge_base(query: str) -> str:
            """在知识库中搜索相关信息"""
            results = self.knowledge_base.search(query)
            if results:
                formatted = "\n\n".join([
                    f"[来源: {r.source}]\n{r.content}"
                    for r in results[:3]
                ])
                return formatted
            else:
                return "未在知识库中找到相关信息"

        @tool
        def create_support_ticket(
            user_id: str,
            issue_title: str,
            issue_description: str,
            category: str = "general",
            priority: str = "medium"
        ) -> str:
            """当知识库无法解决问题时，创建客服工单"""
            request = TicketRequest(
                user_id=user_id,
                issue_title=issue_title,
                issue_description=issue_description,
                category=category,
                priority=priority
            )
            response = self.ticket_system.create_ticket(request)
            return (
                f"工单已创建！\n"
                f"工单号: {response.ticket_id}\n"
                f"状态: {response.status}\n"
                f"消息: {response.message}"
            )

        return [search_knowledge_base, create_support_ticket]

    def _create_agent(self):
        """创建LangChain 1.x Agent（使用LangGraph）"""

        # 使用 LangGraph 的 ReAct Agent
        from langgraph.prebuilt import create_react_agent

        # 创建 ReAct Agent
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools
        )

        return agent

    def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        conversation_id: str = None
    ) -> AgentResponse:
        """
        处理用户查询

        Args:
            query: 用户问题
            user_id: 用户ID
            conversation_id: 会话ID（可选，不提供则创建新的）

        Returns:
            Agent响应
        """
        try:
            # 如果没有会话ID，创建新的
            if not conversation_id:
                conversation_id = f"conv_{uuid.uuid4().hex[:12]}"

            logger.info(f"处理查询 [会话: {conversation_id}, 用户: {user_id}]")
            logger.info(f"问题: {query}")

            # 保存用户消息
            user_message = ChatMessage(role="user", content=query)
            self.conversation_manager.save_message(conversation_id, user_message)

            # 获取对话历史
            chat_history = self.conversation_manager.format_history_for_llm(
                conversation_id,
                limit=5
            )

            # 先进行知识库检索
            kb_results = self.knowledge_base.search(query)

            # 判断是否需要创建工单（基于检索分数）
            needs_ticket = False
            if not kb_results or all(r.score < 0.3 for r in kb_results):
                needs_ticket = True
                logger.info("知识库匹配度低，需要创建工单")

            # 构建Agent输入
            messages = [
                ("system", f"""你是一个专业的智能客服助手。

你可以使用以下工具：
- search_knowledge_base: 搜索知识库
- create_support_ticket: 创建工单

当前对话历史：
{chat_history}

请专业、友好地回答用户问题。"""),
                ("human", query)
            ]

            # 执行Agent
            result = self.agent_executor.invoke({
                "messages": messages
            })

            # 提取回复内容
            if isinstance(result, dict) and "messages" in result:
                answer = result["messages"][-1].content
            else:
                answer = str(result)

            # 确定查询类型和构建响应
            if needs_ticket:
                # 自动创建工单
                ticket_request = TicketRequest(
                    user_id=user_id,
                    issue_title=query[:50],
                    issue_description=query,
                    category="technical_support",
                    priority="medium"
                )
                ticket_response = self.ticket_system.create_ticket(ticket_request)

                response = AgentResponse(
                    query_type=QueryType.TICKET_REQUIRED,
                    answer=f"{answer}\n\n已为您创建工单: {ticket_response.ticket_id}",
                    ticket=ticket_response,
                    conversation_id=conversation_id
                )
            else:
                response = AgentResponse(
                    query_type=QueryType.KNOWLEDGE_BASE,
                    answer=answer,
                    sources=kb_results,
                    conversation_id=conversation_id
                )

            # 保存助手回复
            assistant_message = ChatMessage(
                role="assistant",
                content=response.answer
            )
            self.conversation_manager.save_message(
                conversation_id,
                assistant_message
            )

            logger.info(f"查询处理完成 [会话: {conversation_id}]")
            return response

        except Exception as e:
            logger.error(f"处理查询失败: {e}", exc_info=True)
            raise

    def reset_conversation(self, conversation_id: str):
        """重置会话"""
        self.conversation_manager.clear_conversation(conversation_id)
        logger.info(f"会话已重置: {conversation_id}")
