from mcp.server.fastmcp import FastMCP

from sqlalchemy import text
from db import SessionLocal


mcp = FastMCP("customer-service")


@mcp.tool()
def get_order(order_id: str) -> str:
    """根据订单号查询订单信息"""

    db = SessionLocal()

    try:
        sql = text("""
                   SELECT order_no,
                          customer_id,
                          product_name,
                          status,
                          tracking_no
                   FROM orders
                   WHERE order_no = :order_id
                   """)

        result = db.execute(
            sql,
            {
                "order_id": order_id
            }
        )

        order = result.fetchone()

        if not order:
            return f"没有找到订单 {order_id}"

        return (
            f"订单号：{order.order_no}\n"
            f"客户：{order.customer_id}\n"
            f"商品：{order.product_name}\n"
            f"状态：{order.status}\n"
            f"物流单号：{order.tracking_no or '暂无'}"
        )

    except Exception as e:
        return f"查询订单失败：{str(e)}"

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


if __name__ == "__main__":
    mcp.run()
