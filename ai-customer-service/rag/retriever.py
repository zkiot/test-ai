from pathlib import Path
import os
import logging

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import json

# =========================================================
# 基础配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_PATH = str(
    BASE_DIR / "data" / "chroma_db"
)

# 距离越小越相关
MAX_DISTANCE = 0.75
DISTANCE_THRESHOLD = 0.70
# =========================================================
# 日志
# =========================================================

logger = logging.getLogger(__name__)

# =========================================================
# Embedding
# =========================================================

logger.info("正在初始化 Embedding 模型...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)

logger.info("Embedding 模型初始化完成")

# =========================================================
# Chroma
# =========================================================

db = Chroma(
    persist_directory=VECTOR_PATH,
    embedding_function=embeddings
)

logger.info("Chroma 初始化完成")


# =========================================================
# RAG 检索
# =========================================================

def search_policy(question: str) -> str:
    logger.info(
        "[search_policy] question=%s",
        question
    )

    results = db.similarity_search_with_score(
        question,
        k=3
    )

    if not results:
        return {
            "matched": False,
            "reason": "未查询到相关业务规则",
        }

    doc, distance = results[0]

    logger.info(
        "[search_policy] rule_id=%s distance=%.4f",
        doc.metadata.get("rule_id"),
        distance
    )

    logger.info(
        "[search_policy] metadata=%s",
        doc.metadata
    )

    if distance > 0.6:
        return {
            "matched": False,
            "reason": "未查询到足够相关的业务规则",
            "distance": round(float(distance), 4),
        }

    conditions = doc.metadata.get("conditions", "{}")

    if isinstance(conditions, str):
        conditions = json.loads(conditions)

    logger.info(
        "[search_policy] conditions=%s",
        conditions
    )

    return {
        "matched": True,
        "rule_id": doc.metadata.get("rule_id"),
        "category": doc.metadata.get("category"),
        "policy": doc.metadata.get("policy"),
        "action": doc.metadata.get("action"),
        "conditions": conditions,
        "distance": round(float(distance), 4),
    }



# =========================================================
# 测试
# =========================================================

if __name__ == "__main__":

    test_questions = [
        "发货以后还能取消吗？",
        "快递已经在路上了可以退款吗？",
        "收到货了还能退吗？",
        "买回来没用过可以退款吗？",
        "拆开包装还能退款吗？",
        "VIP退货会不会处理得更快？",
        "普通会员有优先售后吗？",
        "退款需要什么条件？",
        "已经签收8天还能退款吗？",
        "我想买苹果手机",
    ]

    for question in test_questions:
        print("\n")
        print("=" * 60)
        print(f"问题：{question}")
        print("=" * 60)

        result = search_policy(question)

        print(result)
