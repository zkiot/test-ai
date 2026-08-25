from db import SessionLocal
from sqlalchemy import text


db = SessionLocal()

try:
    result = db.execute(
        text("""
            SELECT
                order_no,
                product_name,
                status,
                tracking_no
            FROM orders
            WHERE order_no = :order_id
        """),
        {
            "order_id": "10001"
        }
    )

    order = result.fetchone()

    print(order)

finally:
    db.close()