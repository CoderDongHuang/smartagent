"""
FastAPI Web接口
提供RESTful API服务
"""
import os
# 必须在导入其他模块之前设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from config import APP_HOST, APP_PORT, LOG_LEVEL
from models import AgentResponse
from agent import CustomerServiceAgent
from init_systems import init_systems

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="智能客服Agent API",
    description="基于LangChain + RAG + Function Calling的智能客服系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化系统组件
systems = init_systems()
knowledge_base = systems['knowledge_base']
ticket_system = systems['ticket_system']
conversation_manager = systems['conversation_manager']
agent = systems['agent']


# 请求模型
class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None


class ResetConversationRequest(BaseModel):
    """重置会话请求"""
    conversation_id: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    kb_document_count: int


# API路由
@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "message": "智能客服Agent API",
        "docs": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """健康检查"""
    try:
        doc_count = knowledge_base.get_knowledge_count()
        return HealthResponse(
            status="healthy",
            kb_document_count=doc_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=AgentResponse, tags=["Chat"])
async def chat(request: QueryRequest):
    """
    发送消息给智能客服

    Args:
        request: 查询请求对象

    Returns:
        Agent响应
    """
    try:
        response = agent.process_query(
            query=request.query,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        logger.error(f"处理查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/reset", tags=["Conversation"])
async def reset_conversation(request: ResetConversationRequest):
    """
    重置指定会话的对话历史

    Args:
        request: 重置会话请求

    Returns:
        操作结果
    """
    try:
        agent.reset_conversation(request.conversation_id)
        return {"message": f"会话 {request.conversation_id} 已重置"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}/history", tags=["Conversation"])
async def get_conversation_history(conversation_id: str):
    """
    获取会话历史

    Args:
        conversation_id: 会话ID

    Returns:
        对话历史列表
    """
    try:
        history = conversation_manager.get_conversation_history(conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages": [msg.dict() for msg in history]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动服务: {APP_HOST}:{APP_PORT}")
    uvicorn.run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        log_level=LOG_LEVEL.lower()
    )
