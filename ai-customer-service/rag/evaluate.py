from retriever import db
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


TEST_CASES = [

    {
        "question": "已经发货还能退款吗？",
        "expected": "3."
    },

    {
        "question": "签收以后还能退款吗？",
        "expected": "1."
    },

    {
        "question": "商品拆封了怎么办？",
        "expected": "2."
    },

    {
        "question": "VIP有什么售后权益？",
        "expected": "4."
    },

    {
        "question": "已经收到商品7天内可以退款吗？",
        "expected": "1."
    },

    {
        "question": "物流还在运输中可以取消订单吗？",
        "expected": "3."
    },
]


def evaluate(k=3):

    total = len(TEST_CASES)
    hit = 0

    print("\n")
    print("=" * 70)
    print("RAG 检索评测")
    print("=" * 70)

    for case in TEST_CASES:

        question = case["question"]
        expected = case["expected"]

        results = db.similarity_search_with_score(
            question,
            k=k
        )

        print("\n问题：", question)
        print("期望规则：", expected)

        matched = False

        for index, (doc, distance) in enumerate(results, start=1):

            content = doc.page_content

            print(
                f"Top {index} | "
                f"distance={distance:.4f} | "
                f"{content}"
            )

            if expected in content:
                matched = True

        if matched:
            hit += 1
            print("结果：✅ 命中")
        else:
            print("结果：❌ 未命中")

    recall = hit / total

    print("\n")
    print("=" * 70)
    print(f"Top-{k} Recall: {recall:.2%}")
    print(f"命中：{hit}/{total}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate(k=3)