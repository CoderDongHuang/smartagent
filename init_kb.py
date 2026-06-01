"""
初始化知识库脚本
运行此脚本将示例数据添加到知识库
"""
import logging
from init_systems import init_systems
from sample_data import SAMPLE_KNOWLEDGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """初始化知识库"""
    logger.info("开始初始化知识库...")

    # 初始化系统
    systems = init_systems()
    kb = systems['knowledge_base']

    # 准备数据
    documents = [item['content'] for item in SAMPLE_KNOWLEDGE]
    sources = [item['source'] for item in SAMPLE_KNOWLEDGE]
    categories = [item['category'] for item in SAMPLE_KNOWLEDGE]

    # 添加到知识库
    logger.info(f"正在添加 {len(documents)} 条知识...")
    kb.add_knowledge(
        documents=documents,
        sources=sources,
        categories=categories
    )

    # 显示统计
    count = kb.get_knowledge_count()
    logger.info(f"知识库初始化完成！共有 {count} 条知识")

    # 测试检索
    logger.info("\n测试检索功能...")
    test_queries = ["怎么重置密码", "支持哪些支付方式", "如何退款"]

    for query in test_queries:
        logger.info(f"\n查询: {query}")
        results = kb.search(query)
        for i, result in enumerate(results[:2], 1):
            logger.info(f"  结果{i}: 分数={result.score:.3f}, 来源={result.source}")


if __name__ == "__main__":
    main()
