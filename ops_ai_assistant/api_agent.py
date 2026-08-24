# api_server.py

import json
import os
import sqlite3
import time
import warnings
from datetime import datetime
from typing import Any

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
   SystemMessage, ToolMessage
)
from langchain_core.tools import tool

# LlamaIndex
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from deep_seek_llm import DeepSeekLLM

# ================================================================
# 1. 模型初始化
# ================================================================
Settings.llm = DeepSeekLLM()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
Settings.chunk_size = 512
Settings.chunk_overlap = 50

langchain_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)


# ================================================================
# 2. 全局状态
# ================================================================
class AppState:
    knowledge_base: VectorStoreIndex = None
    llm_with_tools: Any = None
    tool_map: dict = {}
    session_store: dict = {}
    request_count: int = 0
    start_time: float = time.time()

app_state = AppState()


# ================================================================
# 3. 知识库
# ================================================================
def build_knowledge_base() -> VectorStoreIndex:
    docs = [
        Document(text="""Redis 故障处理手册
1. Redis 内存不足
症状：OOM command not allowed
处理：INFO memory → 调整maxmemory-policy → 扩容
预防：设置合理TTL，监控内存水位

2. Redis 连接数耗尽
症状：ERR max number of clients reached
处理：INFO clients → CLIENT KILL → 修改maxclients
预防：使用连接池，设置超时时间""", metadata={"category": "Redis"}),

        Document(text="""MySQL 故障处理手册
1. MySQL 慢查询
症状：接口慢、CPU高
处理：开启慢查询日志 → EXPLAIN分析 → 添加索引

2. MySQL 连接数耗尽
症状：Too many connections
处理：SHOW PROCESSLIST → KILL长连接 → 修改max_connections""",
                 metadata={"category": "MySQL"}),

        Document(text="""服务器巡检规范
告警阈值：CPU 85%/95%，内存 80%/90%，磁盘 75%/85%
巡检频率：生产每天，测试每周，变更后立即
巡检项目：系统资源、进程状态、网络检查、安全合规""",
                 metadata={"category": "巡检"}),
    ]
    return VectorStoreIndex.from_documents(docs, show_progress=False)


# ================================================================
# 4. 工具定义
# ================================================================
@tool
def query_knowledge_base(question: str) -> str:
    """查询运维知识库，获取故障处理方案和操作规范"""
    if app_state.knowledge_base is None:
        return "知识库未初始化"
    engine = app_state.knowledge_base.as_query_engine(similarity_top_k=3)
    return str(engine.query(question))


@tool
def get_server_status(server_ip: str) -> str:
    """查询服务器实时状态（CPU/内存/磁盘）"""
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
        "server_ip":    server_ip,
        "cpu_usage":    f"{data['cpu']}%",
        "memory_usage": f"{data['memory']}%",
        "disk_usage":   f"{data['disk']}%",
        "status":       data["status"],
        "check_time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, ensure_ascii=False)


@tool
def query_cmdb(keyword: str) -> str:
    """查询CMDB资产信息（负责人/业务/机房）"""
    cmdb = {
        "192.168.1.100": {"hostname": "prod-web-01", "owner": "张三",
                          "business": "订单系统", "idc": "上海A区"},
        "192.168.1.101": {"hostname": "prod-db-01",  "owner": "李四",
                          "business": "用户数据库",  "idc": "上海B区"},
        "192.168.1.102": {"hostname": "prod-cache-01", "owner": "王五",
                          "business": "Redis集群",   "idc": "上海A区"},
    }
    result = cmdb.get(keyword, {"error": f"未找到 {keyword}"})
    return json.dumps(result, ensure_ascii=False)


@tool
def create_ticket(server_ip: str, title: str,
                  description: str, severity: str = "P2") -> str:
    """创建运维工单"""
    ticket_id = f"INC{datetime.now().strftime('%Y%m%d%H%M%S')}"
    conn = sqlite3.connect("./ops_tickets.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS tickets
        (id TEXT, server_ip TEXT, title TEXT, description TEXT,
         severity TEXT, status TEXT, created_at TEXT)""")
    conn.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?)",
                 (ticket_id, server_ip, title, description,
                  severity, "待处理",
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return json.dumps({
        "ticket_id": ticket_id, "status": "已创建",
        "message": f"工单 {ticket_id} 已创建"
    }, ensure_ascii=False)


# ================================================================
# 5. Agent 核心
# ================================================================
SYSTEM_PROMPT = """你是一个专业的智能运维助手。
工具：query_knowledge_base, get_server_status, query_cmdb, create_ticket
原则：优先查实时数据，发现严重问题建议创建工单，回答简洁专业。"""


def run_agent(messages: list) -> (str, list):
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    tools_used = []

    for _ in range(6):
        response = app_state.llm_with_tools.invoke(full_messages)
        full_messages.append(response)

        if not response.tool_calls:
            return response.content, tools_used

        for tool_call in response.tool_calls:
            name   = tool_call["name"]
            args   = tool_call["args"]
            t_id   = tool_call["id"]
            if name in app_state.tool_map:
                if name not in tools_used:
                    tools_used.append(name)
                    result = app_state.tool_map[name].invoke(args)
            else:
                result = f"工具 {name} 不存在"

            full_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=t_id
            ))

    return "已达到最大工具调用次数，请重新提问",tools_used
