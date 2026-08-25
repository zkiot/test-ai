import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

load_dotenv()


def get_mysql_uri() -> str:

    host = os.getenv("MEMORY_MYSQL_HOST", "localhost")
    port = os.getenv("MEMORY_MYSQL_PORT", "3306")
    user = os.getenv("MEMORY_MYSQL_USER", "root")
    password = os.getenv("MEMORY_MYSQL_PASSWORD") or os.getenv("PWD")
    database = os.getenv(
        "MEMORY_MYSQL_DB",
        "ai_customer_service"
    )

    if not password:
        raise RuntimeError(
            "MEMORY_MYSQL_PASSWORD/PWD 未配置"
        )

    # 密码可能包含 @ / : / / 等特殊字符
    user = quote_plus(user)
    password = quote_plus(password)

    return (
        f"mysql://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


def create_checkpointer():

    db_uri = get_mysql_uri()

    return AIOMySQLSaver.from_conn_string(
        db_uri
    )