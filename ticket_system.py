"""
工单系统API工具
模拟真实的工单创建和管理
"""
import uuid
import logging
from datetime import datetime
from typing import Optional
from models import TicketRequest, TicketResponse

logger = logging.getLogger(__name__)


class TicketSystemAPI:
    """工单系统API客户端"""

    def __init__(self, api_base_url: str = None, api_key: str = None):
        """
        初始化工单系统API

        Args:
            api_base_url: API基础URL（实际项目中配置）
            api_key: API密钥
        """
        self.api_base_url = api_base_url or "https://api.ticket-system.com"
        self.api_key = api_key

        # 模拟工单存储（实际应调用真实API）
        self.tickets = {}

        logger.info("工单系统API初始化完成")

    def create_ticket(self, request: TicketRequest) -> TicketResponse:
        """
        创建工单

        Args:
            request: 工单请求对象

        Returns:
            工单响应对象
        """
        try:
            # 生成工单ID
            ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

            # 在实际项目中，这里应该调用真实的工单系统API
            # 例如：requests.post(f"{self.api_base_url}/tickets", json=request.dict())

            # 模拟创建工单
            ticket_data = {
                'ticket_id': ticket_id,
                'user_id': request.user_id,
                'issue_title': request.issue_title,
                'issue_description': request.issue_description,
                'category': request.category,
                'priority': request.priority,
                'status': 'open',
                'created_at': datetime.now().isoformat()
            }

            # 存储到本地（模拟）
            self.tickets[ticket_id] = ticket_data

            logger.info(f"工单创建成功: {ticket_id}")

            return TicketResponse(
                ticket_id=ticket_id,
                status='open',
                message=f"工单已创建，我们将尽快处理您的问题",
                created_at=ticket_data['created_at']
            )

        except Exception as e:
            logger.error(f"创建工单失败: {e}")
            raise

    def get_ticket_status(self, ticket_id: str) -> Optional[dict]:
        """
        查询工单状态

        Args:
            ticket_id: 工单ID

        Returns:
            工单信息字典
        """
        try:
            # 实际项目中应调用真实API
            ticket = self.tickets.get(ticket_id)

            if ticket:
                return {
                    'ticket_id': ticket['ticket_id'],
                    'status': ticket['status'],
                    'created_at': ticket['created_at']
                }
            else:
                logger.warning(f"工单不存在: {ticket_id}")
                return None

        except Exception as e:
            logger.error(f"查询工单状态失败: {e}")
            raise

    def close_ticket(self, ticket_id: str, resolution: str = None) -> bool:
        """
        关闭工单

        Args:
            ticket_id: 工单ID
            resolution: 解决方案描述

        Returns:
            是否成功关闭
        """
        try:
            ticket = self.tickets.get(ticket_id)

            if not ticket:
                logger.warning(f"工单不存在: {ticket_id}")
                return False

            # 更新工单状态
            ticket['status'] = 'closed'
            ticket['resolution'] = resolution
            ticket['closed_at'] = datetime.now().isoformat()

            logger.info(f"工单已关闭: {ticket_id}")
            return True

        except Exception as e:
            logger.error(f"关闭工单失败: {e}")
            return False

    def simulate_real_api_call(self, request: TicketRequest) -> TicketResponse:
        """
        模拟真实API调用（展示如何集成真实工单系统）

        在实际项目中，应该这样调用：

        import requests

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post(
            f'{self.api_base_url}/api/v1/tickets',
            json={
                'userId': request.user_id,
                'title': request.issue_title,
                'description': request.issue_description,
                'category': request.category,
                'priority': request.priority
            },
            headers=headers
        )

        if response.status_code == 201:
            data = response.json()
            return TicketResponse(
                ticket_id=data['id'],
                status=data['status'],
                message=data.get('message', '工单创建成功')
            )
        else:
            raise Exception(f"API调用失败: {response.text}")
        """
        # 这里直接使用create_ticket
        return self.create_ticket(request)
