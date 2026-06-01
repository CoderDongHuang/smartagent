"""
测试脚本 - 演示完整的对话流程
"""
import logging
from init_systems import init_systems

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_customer_service():
    """测试智能客服系统"""
    logger.info("=" * 60)
    logger.info("开始测试智能客服系统")
    logger.info("=" * 60)

    # 初始化系统
    systems = init_systems()
    agent = systems['agent']

    # 测试场景1：知识库可以回答的问题
    logger.info("\n【测试1】知识库问答")
    response = agent.process_query(
        query="如何重置密码？",
        user_id="user_001"
    )
    logger.info(f"查询类型: {response.query_type}")
    logger.info(f"回复: {response.answer[:200]}...")
    if response.sources:
        logger.info(f"知识来源: {response.sources[0].source}")
    logger.info(f"会话ID: {response.conversation_id}")

    # 继续同一会话
    logger.info("\n【测试2】多轮对话")
    response2 = agent.process_query(
        query="那支付方式有哪些呢？",
        user_id="user_001",
        conversation_id=response.conversation_id
    )
    logger.info(f"回复: {response2.answer[:200]}...")

    # 测试场景2：需要创建工单的问题
    logger.info("\n【测试3】需要工单的场景")
    response3 = agent.process_query(
        query="我的账户被莫名锁定了，需要紧急解封！",
        user_id="user_002"
    )
    logger.info(f"查询类型: {response3.query_type}")
    logger.info(f"回复: {response3.answer}")
    if response3.ticket:
        logger.info(f"工单号: {response3.ticket.ticket_id}")
        logger.info(f"工单状态: {response3.ticket.status}")

    # 测试场景3：复杂问题
    logger.info("\n【测试4】复杂问题处理")
    response4 = agent.process_query(
        query="我想退款但是找不到退款按钮，而且我的订单已经超过了7天，这种情况还能退吗？",
        user_id="user_003"
    )
    logger.info(f"查询类型: {response4.query_type}")
    logger.info(f"回复: {response4.answer[:300]}...")

    logger.info("\n" + "=" * 60)
    logger.info("测试完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_customer_service()
