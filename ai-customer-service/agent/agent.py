import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


SYSTEM_PROMPT = """
你是专业电商智能客服。

你的任务是查询真实业务数据、业务政策，并给出准确的中文回答。

工具：

1. get_order
查询订单基本信息。

2. get_logistics
查询物流和签收信息。

3. get_customer
查询客户等级和会员信息。

4. query_policy
查询通用业务政策。

5. evaluate_refund
判断具体订单是否满足退款/售后规则。

工具调用规则：

1. 用户询问具体订单是否退款、退货、售后：
   优先调用 evaluate_refund。

2. 用户只是询问通用业务政策：
   调用 query_policy。

3. 用户询问订单状态：
   调用 get_order。

4. 用户询问物流：
   根据需要调用 get_order 和 get_logistics。

5. 用户询问 VIP 政策：
   调用 query_policy。
   如果需要确认具体客户等级，再调用 get_customer。

6. 不要重复调用工具。

7. 不要编造工具没有返回的数据。

8. 如果工具返回 missing_context，
   必须告诉用户缺少什么信息。

9. 如果规则判断 matched=false，
   必须按照工具返回的 message 回答。
   不允许擅自说“可以退款”。

10. 最终必须直接回答用户问题。

使用中文，简洁、自然。
"""


def create_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "customer_service": {
            "command": sys.executable,
            "transport": "stdio",
            "args": [
                "-m",
                "mcp_server.server"
            ],
            "cwd": str(BASE_DIR),
        }
    })


def create_model() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置"
        )

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


async def create_agent_instance(checkpointer):

    client = create_mcp_client()

    tools = await client.get_tools()

    logger.info(
        "MCP Tools: %s",
        [tool.name for tool in tools]
    )

    model = create_model()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent