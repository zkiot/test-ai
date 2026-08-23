# tool_use.py
# Day 7～8：Tool Use + Function Calling 详解

from dotenv import load_dotenv
import os
import json
import random
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ================================================================
# 第一部分：定义工具（Tool）
# 结合你的运维平台背景，定义一套运维工具集
# ================================================================

@tool
def get_server_status(server_ip: str) -> str:
    """
    查询服务器实时状态，包括CPU、内存、磁盘使用率。
    当用户询问某台服务器状态时使用此工具。
    :param server_ip: 服务器IP地址，如 192.168.1.100
    """
    # 模拟真实监控数据（实际项目对接监控平台API）
    mock_data = {
        "192.168.1.100": {"cpu": 92, "memory": 78, "disk": 65, "status": "告警"},
        "192.168.1.101": {"cpu": 35, "memory": 52, "disk": 40, "status": "正常"},
        "192.168.1.102": {"cpu": 8,  "memory": 20, "disk": 30, "status": "正常"},
    }
    data = mock_data.get(server_ip, {
        "cpu": random.randint(10, 95),
        "memory": random.randint(20, 90),
        "disk": random.randint(10, 80),
        "status": "正常"
    })
    return json.dumps({
        "server_ip": server_ip,
        "cpu_usage": f"{data['cpu']}%",
        "memory_usage": f"{data['memory']}%",
        "disk_usage": f"{data['disk']}%",
        "status": data["status"],
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, ensure_ascii=False)


@tool
def query_cmdb(resource_type: str, keyword: str) -> str:
    """
    查询CMDB配置管理数据库，获取资产信息。
    当用户询问服务器归属、负责人、业务系统等信息时使用。
    :param resource_type: 资源类型，如 server/database/middleware
    :param keyword: 搜索关键词，如IP地址、主机名、业务名称
    """
    mock_cmdb = {
        "192.168.1.100": {
            "hostname": "prod-web-01",
            "owner": "张三",
            "team": "电商业务组",
            "business": "订单系统",
            "env": "生产环境",
            "idc": "上海机房A区",
            "spec": "8核16G"
        },
        "192.168.1.101": {
            "hostname": "prod-web-02",
            "owner": "李四",
            "team": "用户中心组",
            "business": "用户系统",
            "env": "生产环境",
            "idc": "上海机房A区",
            "spec": "4核8G"
        }
    }
    result = mock_cmdb.get(keyword, {"error": f"未找到 {keyword} 的CMDB信息"})
    return json.dumps(result, ensure_ascii=False)


@tool
def create_alert_ticket(
    server_ip: str,
    alert_type: str,
    severity: str,
    description: str
) -> str:
    """
    创建告警工单，通知相关负责人处理。
    当发现服务器异常需要人工介入时使用。
    :param server_ip: 告警服务器IP
    :param alert_type: 告警类型，如 CPU告警/内存告警/磁盘告警
    :param severity: 严重程度：P0/P1/P2/P3
    :param description: 告警详细描述
    """
    ticket_id = f"INC{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return json.dumps({
        "ticket_id": ticket_id,
        "server_ip": server_ip,
        "alert_type": alert_type,
        "severity": severity,
        "description": description,
        "status": "已创建",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": f"工单 {ticket_id} 已创建，已通知相关负责人"
    }, ensure_ascii=False)


@tool
def get_alert_history(server_ip: str, days: int = 7) -> str:
    """
    查询服务器历史告警记录。
    当需要了解服务器告警趋势时使用。
    :param server_ip: 服务器IP地址
    :param days: 查询最近几天，默认7天
    """
    mock_history = [
        {"time": "2026-08-18 14:23:00", "type": "CPU告警", "value": "89%", "duration": "15分钟"},
        {"time": "2026-08-17 09:11:00", "type": "内存告警", "value": "85%", "duration": "30分钟"},
        {"time": "2026-08-15 22:45:00", "type": "CPU告警", "value": "95%", "duration": "5分钟"},
    ]
    return json.dumps({
        "server_ip": server_ip,
        "query_days": days,
        "alert_count": len(mock_history),
        "alerts": mock_history
    }, ensure_ascii=False)


@tool
def execute_inspection(server_ip: str, inspection_type: str = "basic") -> str:
    """
    对服务器执行巡检任务，检查系统健康状态。
    :param server_ip: 服务器IP地址
    :param inspection_type: 巡检类型：basic基础巡检/full全量巡检
    """
    results = {
        "system": "正常",
        "process": "正常",
        "network": "正常",
        "security": "发现2个高危端口未关闭（8080, 9090）",
        "log": "发现 ERROR 日志 156 条（最近24小时）",
        "backup": "最近备份时间：2026-08-18 02:00（正常）"
    }
    return json.dumps({
        "server_ip": server_ip,
        "inspection_type": inspection_type,
        "inspection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "score": 72,
        "issues": ["存在高危端口", "ERROR日志偏多"]
    }, ensure_ascii=False)


# ================================================================
# 第二部分：创建 Agent
# ================================================================
def create_ops_agent():
    """创建智能运维 Agent"""

    tools = [
        get_server_status,
        query_cmdb,
        create_alert_ticket,
        get_alert_history,
        execute_inspection
    ]

    system_prompt = SystemMessage(content="""你是一个专业的智能运维助手，可以调用以下工具帮助运维工程师处理问题：

        工具能力：
        - get_server_status：查询服务器实时指标（CPU/内存/磁盘）
        - query_cmdb：查询资产信息（负责人/业务/机房）
        - create_alert_ticket：创建告警工单通知相关人员
        - get_alert_history：查询历史告警记录
        - execute_inspection：执行服务器巡检

        工作原则：
        1. 先获取信息，再给出判断
        2. 发现异常主动建议创建工单
        3. 回答要简洁，关键数据用数字标注
        4. 如果问题复杂，分步骤说明处理流程
        """)

    # create_react_agent 返回一个可直接 invoke 的 LangGraph 编译图
    return create_agent(llm, tools=tools, system_prompt=system_prompt)


# ================================================================
# 第三部分：各种场景演示
# ================================================================

def demo_single_tool():
    """场景1：单工具调用"""
    print("\n" + "=" * 60)
    print("📌 场景1：单工具调用 - 查询服务器状态")
    print("=" * 60)

    agent = create_ops_agent()
    result = agent.invoke(
        {"messages": [("user", "帮我查一下 192.168.1.100 这台服务器的状态")]},
        config={"recursion_limit": 10}
    )
    print(f"\n🤖 最终回答：{result['messages'][-1].content}")


def demo_multi_tool():
    """场景2：多工具串联调用"""
    print("\n" + "=" * 60)
    print("📌 场景2：多工具串联 - 自动排查并创建工单")
    print("=" * 60)

    agent = create_ops_agent()
    result = agent.invoke(
        {"messages": [("user", """
        192.168.1.100 CPU告警了，请帮我：
        1. 查询当前服务器状态
        2. 查询这台服务器是谁负责的
        3. 查询最近7天的告警历史
        4. 如果情况严重，创建一个P1级别的告警工单
        """)]},
        config={"recursion_limit": 10}
    )
    print(f"\n🤖 最终回答：{result['messages'][-1].content}")

    # 从消息列表中提取工具调用步骤
    tool_calls = [
        tc
        for msg in result['messages']
        if isinstance(msg, AIMessage)
        for tc in (msg.tool_calls or [])
    ]
    print(f"\n📋 调用了 {len(tool_calls)} 个工具：")
    for i, tc in enumerate(tool_calls, 1):
        print(f"  第{i}步：调用 {tc['name']}({tc['args']})")


def demo_inspection():
    """场景3：主动巡检"""
    print("\n" + "=" * 60)
    print("📌 场景3：执行巡检并生成报告")
    print("=" * 60)

    agent = create_ops_agent()
    result = agent.invoke(
        {"messages": [("user", "对 192.168.1.100 执行一次全量巡检，告诉我巡检结果和处理建议")]},
        config={"recursion_limit": 10}
    )
    print(f"\n🤖 最终回答：{result['messages'][-1].content}")


def demo_with_memory():
    """场景4：带记忆的多轮 Agent 对话"""
    print("\n" + "=" * 60)
    print("📌 场景4：带记忆的多轮对话 Agent")
    print("=" * 60)

    agent = create_ops_agent()
    chat_history = []

    conversations = [
        "帮我查一下 192.168.1.100 的状态",
        "这台服务器是谁负责的？",          # 考验上下文：不需要再说IP
        "帮它创建一个告警工单，P1级别",     # 考验上下文：知道是哪台服务器
    ]

    for user_input in conversations:
        print(f"\n👤 用户：{user_input}")

        result = agent.invoke(
            {"messages": chat_history + [("user", user_input)]},
            config={"recursion_limit": 10}
        )

        response = result["messages"][-1].content
        print(f"🤖 AI：{response}")

        # 手动维护多轮历史
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))


# ================================================================
# 工具调用原理演示（不用 Agent，手动控制）
# ================================================================
def demo_raw_tool_call():
    """演示 Function Calling 底层原理"""
    print("\n" + "=" * 60)
    print("📌 底层原理：Function Calling 手动控制")
    print("=" * 60)

    # 1. 给 LLM 绑定工具
    tools = [get_server_status, query_cmdb]
    llm_with_tools = llm.bind_tools(tools)

    # 2. 发送消息，LLM 决定是否调用工具
    messages = [HumanMessage(content="查一下192.168.1.101的状态")]
    response = llm_with_tools.invoke(messages)

    print(f"LLM 响应类型：{type(response)}")
    print(f"是否有工具调用：{bool(response.tool_calls)}")

    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n🔧 LLM 决定调用工具：{tool_call['name']}")
            print(f"   传入参数：{tool_call['args']}")

            # 3. 手动执行工具
            tool_map = {
                "get_server_status": get_server_status,
                "query_cmdb": query_cmdb
            }
            tool_result = tool_map[tool_call['name']].invoke(tool_call['args'])
            print(f"   工具返回：{tool_result}")

            # 4. 把工具结果传回给 LLM 生成最终答案
            messages.append(response)
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call['id']
            ))

        # 5. LLM 基于工具结果生成最终回答
        final_response = llm_with_tools.invoke(messages)
        print(f"\n🤖 最终回答：{final_response.content}")


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 7～8：Tool Use + Function Calling 示例")
    print("\n选择场景：")
    print("  1. 单工具调用（查询服务器状态）")
    print("  2. 多工具串联（自动排查+创建工单）")
    print("  3. 执行巡检并生成报告")
    print("  4. 带记忆的多轮对话 Agent")
    print("  5. 底层原理（手动控制工具调用）")
    print("  0. 运行全部")

    choice = input("\n请输入编号：").strip()

    demos = {
        "1": demo_single_tool,
        "2": demo_multi_tool,
        "3": demo_inspection,
        "4": demo_with_memory,
        "5": demo_raw_tool_call,
    }

    if choice == "0":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("运行默认场景2：多工具串联")
        demo_multi_tool()