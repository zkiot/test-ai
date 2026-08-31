from pathlib import Path
import os
import shutil

import json

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from docs.refund_rules import REFUND_RULES


BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_PATH = BASE_DIR / "data" / "chroma_db"


def create_vector_store():

    documents = []

    for rule in REFUND_RULES:

        # 用于向量检索的文本
        text = "\n".join(
            rule["questions"]
        )

        document = Document(
            page_content="\n".join(rule["questions"]),
            metadata={
                "rule_id": rule["rule_id"],
                "category": rule["category"],
                "policy": rule["policy"],
                "action": rule["action"],
                "conditions": json.dumps(
                    rule["conditions"],
                    ensure_ascii=False
                )
            }
        )

        documents.append(document)

        print("\n" + "=" * 60)
        print(rule["rule_id"])
        print("category:", rule["category"])
        print("questions:")

        for question in rule["questions"]:
            print("-", question)

        print("policy:", rule["policy"])
        print("action:", rule["action"])

    print(
        f"\n规则数量：{len(documents)}"
    )

    # 删除旧数据库
    if VECTOR_PATH.exists():

        print(
            f"\n删除旧向量库：{VECTOR_PATH}"
        )

        shutil.rmtree(VECTOR_PATH)

    print(
        "\n正在加载 Embedding 模型..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    )

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(VECTOR_PATH)
    )

    print(
        "\n向量库创建完成"
    )

    print(
        "collection count=",
        db._collection.count()
    )


if __name__ == "__main__":
    create_vector_store()