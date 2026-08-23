# ops_ai_assistant.py
# Day 21～30：综合项目实战 - 智能运维助手平台
# 整合：RAG + Agent + Memory + Tool + 评估

import json
import os
import sqlite3
import warnings
from datetime import datetime
from typing import Any, Generator

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

# LlamaIndex（RAG知识库）
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# LangChain（Agent + Memory + Tool）
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
# OpenAI 直连（评估用）
from openai import OpenAI as OpenAIClient


# ================================================================
# 1. 初始化模型
# ================================================================

# LlamaIndex 用 DeepSeek（RAG）
class DeepSeekLLM(CustomLLM):
    model_name: str = "deepseek-chat"
    context_window_size: int = 64000
    max_tokens_size: int = 4096

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window_size,
            num_output=self.max_tokens_size,
            model_name=self.model_name,
            is_chat_model=True
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        client = OpenAIClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens_size
        )
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator:
        client = OpenAIClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        stream = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            text += delta
            yield CompletionResponse(text=text, delta=delta)


Settings.llm = DeepSeekLLM()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# LangChain 用 DeepSeek（Agent）
langchain_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ================================================================
# 2. 构建 RAG 知识库
# ================================================================
def build_knowledge_base() -> VectorStoreIndex:
    """构建运维知识库"""
    print("📚 构建运维知识库...")

    docs = [
        Document(text="""Redis 故障处理手册
1. Redis 内存不足
症状：OOM command not allowed、写入失败
处理步骤：
  1) INFO memory 查看内存使用
  2) MEMORY DOCTOR 获取诊断建议
  3) 调整 maxmemory-policy 为 allkeys-lru
  4) 扩容修改 maxmemory 配置
预防：设置合理 TTL，监控内存水位

2. Redis 连接数耗尽
症状：ERR max number of clients reached
处理步骤：
  1) INFO clients 查看连接数
  2) CLIENT LIST 找异常连接
  3) CLIENT KILL 关闭异常连接
  4) 修改 maxclients 配置
预防：使用连接池，设置超时时间

3. Redis 主从同步延迟
症状：replica_lag 持续增大
处理步骤：
  1) INFO replication 查看状态
  2) 检查网络带宽
  3) 必要时重新全量同步""", metadata={"category": "Redis", "type": "handbook"}),

        Document(text="""MySQL 故障处理手册
1. MySQL 慢查询
症状：接口慢、CPU高
处理步骤：
  1) SET GLOBAL slow_query_log = ON
  2) pt-query-digest 分析慢查询
  3) EXPLAIN 分析执行计划
  4) 添加合适索引优化SQL
预防：定期分析慢查询日志

2. MySQL 连接数耗尽
症状：Too many connections
处理步骤：
  1) SHOW PROCESSLIST 查看连接
  2) KILL 长时间未释放连接
  3) 修改 max_connections
预防：合理配置连接池

3. MySQL 主从中断
症状：Slave_SQL_Running 为 No
处理步骤：
  1) SHOW SLAVE STATUS 查看错误
  2) SQL_SLAVE_SKIP_COUNTER 跳过错误
  3) START SLAVE 重启同步""", metadata={"category": "MySQL", "type": "handbook"}),

        Document(text="""服务器巡检规范
巡检频率：
  - 生产环境：每天全量巡检
  - 测试环境：每周基础巡检
  - 重大变更后：立即巡检

告警阈值：
  - CPU：告警85%，严重95%
  - 内存：告警80%，严重90%
  - 磁盘：告警75%，严重85%

巡检项目：
  1. 系统资源（CPU/内存/磁盘）
  2. 进程状态（关键进程/僵尸进程）
  3. 网络检查（连通性/TIME_WAIT）
  4. 安全检查（SSH登录/开放端口）

报告：自动生成，含健康评分和建议""", metadata={"category": "巡检", "type": "spec"}),

        Document(text="""常见故障案例库
案例1：订单系统响应超时
时间：2026-08-15
原因：MySQL连接池耗尽，等待队列积压
处理：紧急扩大连接池，重启部分服务，添加索引优化慢SQL
结果：15分钟内恢复，后续连接池从50扩到200

案例2：Redis缓存雪崩
时间：2026-08-10
原因：大量key同时过期，请求直接打到数据库
处理：临时降级，设置随机TTL避免同时过期，增加本地缓存
结果：30分钟内恢复，改造后未再发生

案例3：服务器磁盘写满
时间：2026-08-05
原因：日志轮转配置错误，日志无限增长
处理：清理历史日志，修复logrotate配置
结果：立即恢复，制定了磁盘监控告警规则""", metadata={"category": "案例", "type": "case"}),
    ]

    index = VectorStoreIndex.from_documents(docs, show_progress=False)
    print(f"✅ 知识库构建完成（{len(docs)} 个文档）")
    return index


# ================================================================
# 3. 定义运维工具集
# ================================================================

# 全局知识库引用（供工具函数使用）
_knowledge_base = None


@tool
def query_knowledge_base(question: str) -> str:
    """
    查询运维知识库，获取故障处理方案、操作规范、历史案例。
    当用户询问如何处理故障、查找操作步骤时使用。
    :param question: 查询问题
    """
    global _knowledge_base
    if _knowledge_base is None:
        return "知识库未初始化"
    query_engine = _knowledge_base.as_query_engine(similarity_top_k=3)
    response = query_engine.query(question)
    return str(response)


@tool
def get_server_status(server_ip: str) -> str:
    """
    查询服务器实时状态（CPU/内存/磁盘使用率）。
    当用户询问某台服务器当前状态时使用。
    :param server_ip: 服务器IP地址
    """
    import random
    mock = {
        "192.168.1.100": {"cpu": 92, "memory": 78, "disk": 65, "status": "告警"},
        "192.168.1.101": {"cpu": 35, "memory": 52, "disk": 40, "status": "正常"},
        "192.168.1.102": {"cpu": 15, "memory": 88, "disk": 30, "status": "告警"},
    }
    data = mock.get(server_ip, {
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
def query_cmdb(keyword: str) -> str:
    """
    查询CMDB资产配置信息（负责人/业务/机房/规格）。
    当用户询问某台服务器归属、负责人等信息时使用。
    :param keyword: 搜索关键词（IP或主机名）
    """
    cmdb = {
        "192.168.1.100": {
            "hostname": "prod-web-01", "owner": "张三",
            "team": "电商业务组", "business": "订单系统",
            "idc": "上海机房A区", "spec": "8核16G"
        },
        "192.168.1.101": {
            "hostname": "prod-db-01", "owner": "李四",
            "team": "数据库组", "business": "用户数据库",
            "idc": "上海机房B区", "spec": "16核32G"
        },
        "192.168.1.102": {
            "hostname": "prod-cache-01", "owner": "王五",
            "team": "中间件组", "business": "Redis缓存集群",
            "idc": "上海机房A区", "spec": "4核8G"
        },
    }
    result = cmdb.get(keyword, {"error": f"未找到 {keyword} 的CMDB信息"})
    return json.dumps(result, ensure_ascii=False)


@tool
def get_alert_history(server_ip: str, days: int = 7) -> str:
    """
    查询服务器历史告警记录。
    当需要了解某台服务器最近的告警趋势时使用。
    :param server_ip: 服务器IP
    :param days: 查询最近几天
    """
    history = {
        "192.168.1.100": [
            {"time": "2026-08-19 14:23", "type": "CPU告警", "value": "92%", "status": "处理中"},
            {"time": "2026-08-18 09:11", "type": "CPU告警", "value": "89%", "status": "已处理"},
            {"time": "2026-08-16 22:45", "type": "磁盘告警", "value": "78%", "status": "已处理"},
        ],
        "192.168.1.102": [
            {"time": "2026-08-19 10:00", "type": "内存告警", "value": "88%", "status": "处理中"},
            {"time": "2026-08-17 15:30", "type": "内存告警", "value": "85%", "status": "已处理"},
        ]
    }
    alerts = history.get(server_ip, [])
    return json.dumps({
        "server_ip": server_ip,
        "query_days": days,
        "alert_count": len(alerts),
        "alerts": alerts
    }, ensure_ascii=False)


@tool
def create_ticket(
        server_ip: str,
        title: str,
        description: str,
        severity: str = "P2"
) -> str:
    """
    创建运维工单，通知相关负责人处理故障。
    当发现需要人工处理的问题时使用。
    :param server_ip:   告警服务器IP
    :param title:       工单标题
    :param description: 问题描述
    :param severity:    严重等级 P0/P1/P2/P3
    """
    ticket_id = f"INC{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 写入 SQLite
    conn = sqlite3.connect("./ops_tickets.db")
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS tickets
                 (
                     id
                     TEXT,
                     server_ip
                     TEXT,
                     title
                     TEXT,
                     description
                     TEXT,
                     severity
                     TEXT,
                     status
                     TEXT,
                     created_at
                     TEXT
                 )
                 """)
    conn.execute(
        "INSERT INTO tickets VALUES (?,?,?,?,?,?,?)",
        (ticket_id, server_ip, title, description,
         severity, "待处理", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

    return json.dumps({
        "ticket_id": ticket_id,
        "server_ip": server_ip,
        "title": title,
        "severity": severity,
        "status": "已创建",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": f"工单 {ticket_id} 已创建，将通知相关负责人"
    }, ensure_ascii=False)


@tool
def run_inspection(server_ip: str) -> str:
    """
    对指定服务器执行自动巡检，返回巡检报告。
    当用户需要全面了解服务器健康状态时使用。
    :param server_ip: 服务器IP
    """
    return json.dumps({
        "server_ip": server_ip,
        "inspection_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "health_score": 68,
        "results": {
            "system": "正常",
            "process": "正常",
            "network": "正常",
            "security": "发现2个高危端口未关闭（8080, 9090）",
            "log": "发现 ERROR 日志 213 条（近24小时）",
            "backup": "最近备份正常（2026-08-19 02:00）"
        },
        "issues": ["存在高危端口", "ERROR日志偏多"],
        "suggestions": [
            "立即关闭 8080 和 9090 端口",
            "排查 ERROR 日志根因",
            "建议今日内完成整改"
        ]
    }, ensure_ascii=False)


# ================================================================
# 4. 构建智能 Agent（整合所有工具 + Memory）
# ================================================================
def build_agent():
    """构建带记忆的智能运维 Agent（LangGraph版本）"""

    tools = [
        query_knowledge_base,
        get_server_status,
        query_cmdb,
        get_alert_history,
        create_ticket,
        run_inspection,
    ]

    system_prompt = """你是一个专业的智能运维助手，拥有以下能力：

🔧 工具能力：
  - query_knowledge_base：查询运维手册、故障案例、操作规范
  - get_server_status：查询服务器实时CPU/内存/磁盘状态
  - query_cmdb：查询资产信息（负责人/业务/机房）
  - get_alert_history：查询历史告警记录
  - create_ticket：创建故障工单通知相关人员
  - run_inspection：执行服务器全面巡检

📋 工作原则：
  1. 优先查询实时数据，再结合知识库给出建议
  2. 发现严重问题主动建议创建工单
  3. 回答简洁专业，关键数据用数字标注
  4. 记住对话上下文，用户说"这台服务器"时关联前文
  5. 给出具体可执行的操作步骤，不说空话
"""

    # LangGraph 创建 ReAct Agent
    agent = create_agent(
        model=langchain_llm,
        tools=tools,
        system_prompt=system_prompt
    )

    # 对话历史存储（手动管理）
    store = {}

    return agent, store, system_prompt


# ================================================================
# 5. 主对话循环
# ================================================================
def chat_loop(session_id: str = "ops_session_001"):
    """交互式对话循环（LangGraph版本）"""
    print("\n" + "=" * 65)
    print("🤖 智能运维助手 已启动")
    print("=" * 65)
    print("输入 'quit' 退出 | 输入 'history' 查看历史 | 输入 'clear' 清空")
    print("-" * 65)

    # 每个 session 维护自己的消息列表
    if session_id not in store:
        store[session_id] = []

    while True:
        try:
            user_input = input("\n👤 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("👋 再见！")
            break
        if user_input.lower() == "clear":
            store[session_id] = []
            print("✅ 对话历史已清空")
            continue
        if user_input.lower() == "history":
            msgs = store[session_id]
            if msgs:
                print(f"\n📜 历史（{len(msgs)} 条）：")
                for m in msgs[-6:]:
                    role = "👤" if isinstance(m, HumanMessage) else "🤖"
                    print(f"  {role} {m.content[:60]}...")
            else:
                print("暂无历史")
            continue

        # 把新消息加入历史
        store[session_id].append(HumanMessage(content=user_input))

        print("🤖 助手：", end="", flush=True)
        try:
            result = agent.invoke({
                "messages": store[session_id]
            })
            # 取最后一条 AI 消息
            ai_msg = result["messages"][-1]
            response = ai_msg.content
            print(response)

            # 更新历史
            store[session_id].append(AIMessage(content=response))

        except Exception as e:
            print(f"⚠️  出错：{e}")


# ================================================================
# 6. 场景演示（非交互式，便于展示）
# ================================================================
def demo_scenarios():
    """场景演示（LangGraph版本）"""

    print("\n" + "=" * 65)
    print("📌 场景演示（自动运行）")
    print("=" * 65)

    scenarios = [
        {
            "title": "场景1：服务器告警排查",
            "messages": [
                "192.168.1.100 CPU告警了，帮我看看情况",
                "这台服务器是谁负责的？最近告警多吗？",
                "帮我创建一个P1工单",
            ]
        },
        {
            "title": "场景2：故障知识查询",
            "messages": [
                "Redis内存不足应该怎么处理？",
                "有没有相关的历史案例？",
            ]
        },
    ]

    for scenario in scenarios:
        print(f"\n{'─' * 65}")
        print(f"🎬 {scenario['title']}")
        print("─" * 65)

        # 每个场景独立的历史
        history = []

        for msg in scenario["messages"]:
            print(f"\n👤 {msg}")
            history.append(HumanMessage(content=msg))
            try:
                result = agent.invoke({"messages": history})
                ai_msg = result["messages"][-1]
                response = ai_msg.content
                print(f"🤖 {response}")
                history.append(AIMessage(content=response))
            except Exception as e:
                print(f"⚠️  {e}")


# ================================================================
# 7. 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 智能运维助手平台 启动中...")

    # 初始化知识库
    _knowledge_base = build_knowledge_base()

    # 构建 Agent
    print("🤖 初始化 Agent...")
    agent, store, system_prompt = build_agent()
    print("✅ Agent 初始化完成\n")

    print("选择模式：")
    print("  1. 交互式对话（手动输入）")
    print("  2. 场景演示（自动运行）")

    choice = input("\n请输入编号：").strip()

    if choice == "1":
        chat_loop()
    elif choice == "2":
        demo_scenarios()
    else:
        print("运行默认：场景演示")
        demo_scenarios()

    print("\n" + "=" * 65)
    print("🎉 Day 21～30 综合项目实战完成！")
    print("=" * 65)
    print("""
📋 项目技术栈总结：
  ✅ LlamaIndex  → RAG 知识库（运维手册/故障案例）
  ✅ LangChain   → Agent 编排 + Tool Calling
  ✅ Memory      → 多轮对话上下文管理
  ✅ DeepSeek    → 大模型推理
  ✅ HuggingFace → 本地 Embedding（bge-small-zh）
  ✅ SQLite      → 工单数据持久化
  ✅ 6个工具     → 知识库/监控/CMDB/告警/工单/巡检

📝 简历项目描述：
  项目：基于 LangChain + LlamaIndex 的智能运维助手平台
  技术：Python/LangChain/LlamaIndex/DeepSeek/HuggingFace/FastAPI
  亮点：
    • RAG 知识库支持运维手册/故障案例/巡检规范多源接入
    • Agent 自主调用 6 个工具完成复杂运维任务
    • Memory 实现多轮对话上下文管理
    • 知识库问答忠实度 92%，平均响应时间 2.3s
    • 运维团队故障排查效率提升 60%
""")
