from mcp.server.fastmcp import FastMCP

from sqlalchemy import text
from db import SessionLocal
import sys
from pathlib import Path

# 加入项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(BASE_DIR)
)

from rag.retriever import search_policy

mcp = FastMCP("customer-service")


@mcp.tool()
def get_order(order_id: str) -> str:
    """根据订单号查询订单信息"""

    db = SessionLocal()
    try:
        sql = text(
            """
            SELECT
                order_no,
                customer_id,
                product_name,
                status,
                tracking_no,
                created_at
            FROM orders
            WHERE order_no = :order_id
            """
        )

        result = db.execute(
            sql,
            {
                "order_id": order_id
            }
        )

        order = result.fetchone()

        if order is None:
            return f"没有找到订单 {order_id}"
        return (
            f"订单号：{order.order_no}\n"
            f"客户ID：{order.customer_id}\n"
            f"商品：{order.product_name}\n"
            f"状态：{order.status}\n"
            f"物流单号：{order.tracking_no or '暂无'}\n"
            f"创建时间：{order.created_at}"
        )


    except Exception as e:
        return f"查询订单失败：{e}"

    finally:
        db.close()


@mcp.tool()
def get_customer(customer_id: str) -> str:
    """根据客户 ID 查询客户信息"""

    customers = {
        "C001": "张三，VIP客户",
        "C002": "李四，普通客户",
    }

    return customers.get(
        customer_id,
        f"没有找到客户 {customer_id}"
    )


@mcp.tool()
def get_logistics(tracking_no: str) -> str:
    """
    根据物流单号查询物流信息
    """

    db = SessionLocal()

    try:
        sql = text(
            """
            SELECT
                tracking_no,
                status,
                location,
                update_time
            FROM logistics
            WHERE tracking_no = :tracking_no
            """
        )

        result = db.execute(
            sql,
            {
                "tracking_no": tracking_no
            }
        )

        logistics = result.fetchone()

        if logistics is None:
            return f"没有找到物流 {tracking_no}"

        return (
            f"物流单号：{logistics.tracking_no}\n"
            f"状态：{logistics.status}\n"
            f"当前位置：{logistics.location}\n"
            f"更新时间：{logistics.update_time}"
        )


    finally:
        db.close()


@mcp.tool()
def query_policy(question: str) -> str:
    """
查询企业知识库。

适用于：
- 退款规则
- 售后流程
- 商品使用说明
- 服务政策

不适用于：
- 查询订单状态
- 查询物流
- 查询客户信息
"""
    print(
        "进入query_policy:",
        question
    )

    result = search_policy(question)

    print(
        "RAG返回:",
        result
    )

    return str(result)


if __name__ == "__main__":
    mcp.run()
