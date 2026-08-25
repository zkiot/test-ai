import asyncio
import sys
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

server_path = (
        BASE_DIR
        /
        "mcp_server"
        /
        "server.py"
)


async def main():
    client = MultiServerMCPClient(
        {
            "customer_service": {
                "command": sys.executable,
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print("当前Tools:")
    for tool in tools:
        print("-", tool.name)

    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
        model_kwargs={
            "parallel_tool_calls": False
        }
    )

    model_with_tools = model.bind_tools(
        tools,
        parallel_tool_calls=False
    )
    agent = create_agent(
        model=model_with_tools,
        tools=tools,
        debug=True
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": """
你是智能客服。

工具规则：

1. 用户询问订单：
必须调用 get_order

2. 用户询问退款：
必须：
先 get_order
再 query_policy

3. 所有工具调用完成后：
必须立即停止调用工具，
生成最终中文客服回复。

禁止：
- 重复调用工具
- 只描述查询过程
- 不回答用户

最终回答格式：

订单情况：
退款判断：
处理建议：
"""
                },
                {
                    "role": "user",
                    "content": "订单10001还能退款吗？"
                }
            ]
        }, config={
            "recursion_limit": 5
        }
    )

    for msg in result["messages"]:
        print("\n====")
        print(type(msg).__name__)
        print(msg.content)

    print("\n最终回答:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
