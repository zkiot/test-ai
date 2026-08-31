import logging

from rag.retriever import search_policy

logger = logging.getLogger(__name__)


def match_rule(
        rule: dict,
        context: dict
) -> dict:
    conditions = rule.get("conditions", {})

    # =====================================================
    # 1. 订单状态
    # =====================================================

    if "order_status" in conditions:

        expected = conditions["order_status"]
        actual = context.get("order_status")

        if actual is None:
            return {
                "matched": False,
                "status": "missing_context",
                "field": "order_status",
                "message": "缺少订单状态",
            }

        if actual != expected:
            return {
                "matched": False,
                "status": "condition_not_met",
                "field": "order_status",
                "expected": expected,
                "actual": actual,
                "message": (
                    f"订单状态不满足规则："
                    f"需要 {expected}，实际为 {actual}"
                ),
            }

    # =====================================================
    # 2. 是否使用
    # =====================================================

    if "used" in conditions:

        expected = conditions["used"]
        actual = context.get("used")

        if actual is None:
            return {
                "matched": False,
                "status": "missing_context",
                "field": "used",
                "message": "缺少商品是否使用的信息"
            }

        if actual != expected:
            return {
                "matched": False,
                "status": "condition_not_met",
                "field": "used",
                "expected": expected,
                "actual": actual,
                "message": "商品使用状态不满足退款条件"
            }

    # =====================================================
    # 3. 是否拆封
    # =====================================================

    if "opened" in conditions:

        expected = conditions["opened"]
        actual = context.get("opened")

        if actual is None:
            return {
                "matched": False,
                "status": "missing_context",
                "field": "opened",
                "message": "缺少商品是否拆封的信息",
            }

        if actual != expected:
            return {
                "matched": False,
                "status": "condition_not_met",
                "field": "opened",
                "expected": expected,
                "actual": actual,
                "message": "商品拆封状态不满足规则",
            }

    # =====================================================
    # 4. VIP
    # =====================================================

    if "customer_level" in conditions:

        expected = conditions["customer_level"]
        actual = context.get("customer_level")

        if actual is None:
            return {
                "matched": False,
                "status": "missing_context",
                "field": "customer_level",
                "message": "缺少客户等级信息",
            }

        if actual != expected:
            return {
                "matched": False,
                "status": "condition_not_met",
                "field": "customer_level",
                "expected": expected,
                "actual": actual,
                "message": "客户等级不满足规则",
            }

    # =====================================================
    # 5. 签收天数
    # =====================================================

    if "days_after_received_max" in conditions:

        max_days = conditions[
            "days_after_received_max"
        ]

        actual_days = context.get(
            "days_after_received"
        )

        if actual_days is None:
            return {
                "matched": False,
                "status": "missing_context",
                "field": "days_after_received",
                "message": "缺少签收天数",
            }

        if actual_days > max_days:
            return {
                "matched": False,
                "status": "condition_not_met",
                "field": "days_after_received",
                "expected_max": max_days,
                "actual": actual_days,
                "message": (
                    f"已签收 {actual_days} 天，"
                    f"超过 {max_days} 天退款期限"
                ),
            }

    # =====================================================
    # 全部条件满足
    # =====================================================

    return {
        "matched": True,
        "status": "matched",
        "message": "满足业务规则",
    }


def evaluate_policy(
        question: str,
        context: dict
) -> dict:
    logger.info(
        "[evaluate_policy] question=%s context=%s",
        question,
        context
    )

    # -----------------------------------------------------
    # 第一步：RAG 找规则
    # -----------------------------------------------------

    rule = search_policy(question)

    if not rule.get("matched"):
        return {
            "matched": False,
            "status": "rule_not_found",
            "message": rule.get(
                "reason",
                "未找到相关业务规则"
            ),
        }

    # -----------------------------------------------------
    # 第二步：执行规则
    # -----------------------------------------------------

    result = match_rule(
        rule,
        context
    )

    # -----------------------------------------------------
    # 第三步：组合结果
    # -----------------------------------------------------

    return {
        "matched": result["matched"],
        "status": result["status"],
        "rule_id": rule["rule_id"],
        "policy": rule["policy"],
        "action": rule["action"],
        "message": result["message"],
    }
if __name__ == "__main__":

    # =============================
    # 测试1：已发货
    # =============================

    result = evaluate_policy(
        "发货以后还能取消吗？",
        {
            "order_status": "已发货"
        }
    )

    print("\n测试1：")
    print(result)


    # =============================
    # 测试2：签收5天
    # =============================

    result = evaluate_policy(
        "签收以后还能退款吗？",
        {
            "order_status": "已签收",
            "days_after_received": 5,
            "used": False
        }
    )

    print("\n测试2：")
    print(result)


    # =============================
    # 测试3：签收8天
    # =============================

    result = evaluate_policy(
        "已经签收8天还能退款吗？",
        {
            "order_status": "已签收",
            "days_after_received": 8,
            "used": False
        }
    )

    print("\n测试3：")
    print(result)
