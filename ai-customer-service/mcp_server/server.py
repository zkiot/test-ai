from typing import Any

from mcp.server.fastmcp import FastMCP

from sqlalchemy import text

from rag.rule_engine import evaluate_policy
from .db import SessionLocal
import sys
from pathlib import Path
import logging

from rag.retriever import search_policy

# 加入项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(BASE_DIR)
)

import sys

if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")

if sys.stderr:
    sys.stderr.reconfigure(encoding="utf-8")
mcp = FastMCP("customer-service")
logger = logging.getLogger(__name__)


@mcp.tool()
def get_order(order_id: str) -> str:
    """根据订单号查询订单信息"""

    db = SessionLocal()
    try:
        sql = text(
            """
            SELECT order_no,
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
            SELECT tracking_no,
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
    logger.info(
        "[query_policy] question=%s",
        question
    )

    try:

        result = search_policy(question)

        logger.info(
            "[query_policy] search_policy result=%s",
            result
        )

        return result

    except Exception as e:

        logger.exception(
            "[query_policy] 查询政策失败"
        )

        return f"政策查询失败：{str(e)}"


@mcp.tool()
def evaluate_refund_policy(
        question: str,
        context: dict[str, Any]
) -> dict:
    logger.info(
        "[evaluate_refund_policy] question=%s",
        question
    )

    logger.info(
        "[evaluate_refund_policy] context=%s",
        context
    )

    try:

        result = evaluate_policy(
            question,
            context
        )

        logger.info(
            "[evaluate_refund_policy] result=%s",
            result
        )

        return result

    except Exception as e:

        logger.exception(
            "[evaluate_refund_policy] failed"
        )

        return {
            "matched": False,
            "status": "error",
            "message": str(e),
        }


# @mcp.tool()
# def evaluate_refund(
#         order_id: str,
#         question: str
# ) -> dict[str, Any]:
#     logger.info(
#         "[evaluate_refund] order_id=%s question=%s",
#         order_id,
#         question
#     )
#
#     try:
#         # ============================================
#         # 1. 查询订单
#         # ============================================
#
#         order = get_order(order_id)
#
#         logger.info(
#             "[evaluate_refund] order=%s",
#             order
#         )
#
#         # ============================================
#         # 2. 查询物流
#         # ============================================
#
#         logistics = get_logistics(
#             order_id
#         )
#
#         logger.info(
#             "[evaluate_refund] logistics=%s",
#             logistics
#         )
#
#         # ============================================
#         # 3. 查询客户
#         # ============================================
#
#         customer_id = extract_customer_id(order)
#
#         customer = None
#
#         if customer_id:
#             customer = get_customer(
#                 customer_id
#             )
#
#         # ============================================
#         # 4. 组装业务上下文
#         # ============================================
#
#         context = {
#             "order_status": extract_order_status(order),
#             "days_after_received": calculate_received_days(
#                 logistics
#             ),
#             "used": extract_used(order),
#             "opened": extract_opened(order),
#             "customer_level": extract_customer_level(
#                 customer
#             )
#         }
#
#         logger.info(
#             "[evaluate_refund] context=%s",
#             context
#         )
#
#         # ============================================
#         # 5. RAG + Rule Engine
#         # ============================================
#
#         result = evaluate_policy(
#             question,
#             context
#         )
#
#         return {
#             "order_id": order_id,
#             "context": context,
#             "result": result
#         }
#
#     except Exception as e:
#
#         logger.exception(
#             "[evaluate_refund] failed"
#         )
#
#         return {
#             "order_id": order_id,
#             "matched": False,
#             "status": "error",
#             "message": str(e)
#         }


if __name__ == "__main__":
    mcp.run()
