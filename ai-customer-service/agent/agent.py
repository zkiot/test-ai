import asyncio
import sys
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "customer_service": {
                "command": sys.executable,
                "args": ["../mcp_server/server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    tools = await client.get_tools()

    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0
    )

    agent = create_agent(
        model,
        tools
    )
    model.bind_tools(tools)

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "帮我查询一下订单10001"
                }
            ]
        }
    )
    # result = model.invoke([
    #     {
    #         "role": "user",
    #         "content": "帮我查询一下订单10001"
    #     }
    # ]
    # )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
