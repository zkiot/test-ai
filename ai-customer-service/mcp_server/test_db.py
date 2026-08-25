from db import SessionLocal
from sqlalchemy import text


db = SessionLocal()

try:
    result = db.execute(
        text("SELECT 1")
    )

    print(result.fetchone())

finally:
    db.close()