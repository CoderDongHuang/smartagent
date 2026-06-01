"""
数据模型定义
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """查询类型枚举"""
    KNOWLEDGE_BASE = "knowledge_base"  # 知识库可回答
    TICKET_REQUIRED = "ticket_required"  # 需要创建工单


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="角色: user/assistant/system")
    content: str = Field(..., description="消息内容")
    timestamp: Optional[float] = None


class KnowledgeResult(BaseModel):
    """知识库检索结果"""
    content: str = Field(..., description="检索到的知识内容")
    source: str = Field(..., description="知识来源")
    score: float = Field(default=0.0, description="相关性分数")
    metadata: Optional[dict] = Field(default=None, description="元数据")


class TicketRequest(BaseModel):
    """工单请求模型"""
    user_id: str = Field(..., description="用户ID")
    issue_title: str = Field(..., description="问题标题")
    issue_description: str = Field(..., description="问题详细描述")
    category: str = Field(default="general", description="问题分类")
    priority: str = Field(default="medium", description="优先级: low/medium/high")


class TicketResponse(BaseModel):
    """工单响应模型"""
    ticket_id: str = Field(..., description="工单ID")
    status: str = Field(..., description="工单状态")
    message: str = Field(..., description="返回消息")
    created_at: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent响应模型"""
    query_type: QueryType = Field(..., description="查询类型")
    answer: str = Field(..., description="回复内容")
    sources: Optional[List[KnowledgeResult]] = Field(None, description="知识来源")
    ticket: Optional[TicketResponse] = Field(None, description="工单信息")
    conversation_id: Optional[str] = Field(None, description="会话ID")
