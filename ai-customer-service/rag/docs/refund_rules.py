REFUND_RULES = [
    {
        "rule_id": "refund_001",
        "category": "refund",
        "questions": [
            "签收以后还能退款吗？",
            "收到商品7天内可以退款吗？",
            "7天无理由退款吗？",
            "签收7天内能退货吗？",
            "商品收到后还能退款吗？",
            "买回来没用过可以退款吗？",
            "退款需要什么条件？",
            "退款条件是什么？",
            "申请退款需要满足什么条件？",
            "什么情况下可以退款？",
            "退款有什么要求？",
        ],
        "policy": "商品签收7天内，如果商品未使用，可以申请退款。",
        "conditions": {
            "order_status": "已签收",
            "days_after_received_max": 7,
            "used": False,
        },
        "action": "允许申请退款",
    },

    {
        "rule_id": "refund_002",
        "category": "refund",
        "questions": [
            "商品拆封了怎么办？",
            "拆封后还能退款吗？",
            "商品已经拆封还能退吗？",
            "拆开包装还能退款吗？",
            "拆封商品怎么处理？",
        ],
        "policy": "已拆封商品，需要经过质量检测。",
        "conditions": {
            "opened": True,
        },
        "action": "需要质量检测",
    },

    {
        "rule_id": "refund_003",
        "category": "refund",
        "questions": [
            "已经发货还能退款吗？",
            "已发货可以取消吗？",
            "物流运输中可以取消订单吗？",
            "发货后还能取消订单吗？",
            "发货以后还能取消吗？",
            "快递已经在路上了可以退款吗？",
        ],
        "policy": "已发货订单不能直接取消，需要收到商品后申请售后。",
        "conditions": {
            "order_status": "已发货",
        },
        "action": "不能直接取消，需要收到商品后申请售后",
    },

    {
        "rule_id": "refund_004",
        "category": "vip",
        "questions": [
            "VIP有什么售后权益？",
            "VIP退款有什么特殊政策？",
            "VIP会员售后有什么优惠？",
            "VIP客户退款是不是优先？",
            "VIP售后是不是更快？",
            "VIP退货会不会处理得更快？",
        ],
        "policy": "VIP客户享受优先售后处理。",
        "conditions": {
            "customer_level": "VIP",
        },
        "action": "优先售后处理",
    },
]