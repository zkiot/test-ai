# multi_source_rag.py
# Day 13～14：多数据源接入

import os
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
import warnings
import json
import sqlite3
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI as OpenAIClient
from typing import Any, Generator

# ================================================================
# 自定义 DeepSeek LLM（复用上节方案）
# ================================================================
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


# 全局配置
Settings.llm = DeepSeekLLM()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
Settings.chunk_size = 512
Settings.chunk_overlap = 50


# ================================================================
# 准备各种数据源
# ================================================================
def prepare_all_sources():
    """准备多种格式的测试数据"""
    os.makedirs("./data/txt",  exist_ok=True)
    os.makedirs("./data/json", exist_ok=True)
    os.makedirs("./data/csv",  exist_ok=True)

    # ① TXT 文件
    with open("./data/txt/redis_handbook.txt", "w", encoding="utf-8") as f:
        f.write("""Redis 故障处理手册

1. Redis 内存不足
症状：OOM command not allowed、写入失败
处理步骤：
  1) 执行 INFO memory 查看内存使用
  2) 调整 maxmemory-policy 为 allkeys-lru
  3) 扩容或清理过期 key
预防：设置合理 TTL，监控内存水位

2. Redis 连接数耗尽
症状：ERR max number of clients reached
处理步骤：
  1) 执行 INFO clients 查看连接数
  2) CLIENT KILL 关闭异常连接
  3) 修改 maxclients 配置
预防：使用连接池，设置超时时间
""")

    with open("./data/txt/mysql_handbook.txt", "w", encoding="utf-8") as f:
        f.write("""MySQL 故障处理手册

1. MySQL 慢查询
症状：接口响应慢、CPU高
处理步骤：
  1) 开启慢查询日志
  2) EXPLAIN 分析执行计划
  3) 添加合适索引
预防：定期分析慢查询日志

2. MySQL 连接数耗尽
症状：Too many connections
处理步骤：
  1) SHOW PROCESSLIST 查看连接
  2) KILL 长时间未释放连接
  3) 修改 max_connections
预防：合理配置连接池
""")

    # ② JSON 文件（模拟 CMDB 数据）
    cmdb_data = [
        {
            "ip": "192.168.1.100",
            "hostname": "prod-web-01",
            "owner": "张三",
            "team": "电商业务组",
            "business": "订单系统",
            "env": "生产环境",
            "idc": "上海机房A区",
            "spec": "8核16G",
            "os": "CentOS 7.9",
            "status": "正常"
        },
        {
            "ip": "192.168.1.101",
            "hostname": "prod-db-01",
            "owner": "李四",
            "team": "数据库组",
            "business": "用户数据库",
            "env": "生产环境",
            "idc": "上海机房B区",
            "spec": "16核32G",
            "os": "CentOS 7.9",
            "status": "正常"
        },
        {
            "ip": "192.168.1.102",
            "hostname": "prod-cache-01",
            "owner": "王五",
            "team": "中间件组",
            "business": "Redis缓存集群",
            "env": "生产环境",
            "idc": "上海机房A区",
            "spec": "4核8G",
            "os": "Ubuntu 20.04",
            "status": "告警"
        }
    ]
    with open("./data/json/cmdb.json", "w", encoding="utf-8") as f:
        json.dump(cmdb_data, f, ensure_ascii=False, indent=2)

    # ③ CSV 文件（模拟告警历史）
    with open("./data/csv/alert_history.csv", "w", encoding="utf-8") as f:
        f.write("时间,服务器IP,告警类型,告警值,持续时长,处理状态\n")
        f.write("2026-08-18 14:23,192.168.1.100,CPU告警,89%,15分钟,已处理\n")
        f.write("2026-08-18 09:11,192.168.1.101,内存告警,85%,30分钟,已处理\n")
        f.write("2026-08-17 22:45,192.168.1.102,Redis内存,92%,5分钟,处理中\n")
        f.write("2026-08-17 16:30,192.168.1.100,磁盘告警,78%,持续,待处理\n")
        f.write("2026-08-16 08:00,192.168.1.101,慢查询,500ms,2小时,已处理\n")

    # ④ SQLite 数据库（模拟运维工单）
    conn = sqlite3.connect("./data/ops_tickets.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id TEXT PRIMARY KEY,
            server_ip TEXT,
            title TEXT,
            description TEXT,
            severity TEXT,
            status TEXT,
            owner TEXT,
            created_at TEXT,
            resolved_at TEXT
        )
    """)
    tickets = [
        ("INC001", "192.168.1.100", "CPU持续高负载",
         "订单系统服务器CPU使用率持续超过90%，经排查发现是慢SQL导致，已优化索引",
         "P1", "已解决", "张三", "2026-08-18 14:30", "2026-08-18 16:00"),
        ("INC002", "192.168.1.102", "Redis内存不足",
         "Redis缓存集群内存使用率达92%，已调整maxmemory-policy为allkeys-lru并扩容",
         "P1", "处理中", "王五", "2026-08-17 22:50", None),
        ("INC003", "192.168.1.101", "MySQL主从延迟",
         "发现主从延迟超过30秒，排查为从库IO压力大，已迁移部分读流量",
         "P2", "已解决", "李四", "2026-08-16 10:00", "2026-08-16 14:00"),
    ]
    cursor.executemany(
        "INSERT OR REPLACE INTO tickets VALUES (?,?,?,?,?,?,?,?,?)",
        tickets
    )
    conn.commit()
    conn.close()

    print("✅ 所有数据源准备完成：")
    print("  📄 TXT  → ./data/txt/（运维手册）")
    print("  📋 JSON → ./data/json/（CMDB数据）")
    print("  📊 CSV  → ./data/csv/（告警历史）")
    print("  🗄️  DB   → ./data/ops_tickets.db（工单数据）")


# ================================================================
# 数据源1：TXT 文件（SimpleDirectoryReader 直接读）
# ================================================================
def load_from_txt() -> list:
    print("\n📄 加载 TXT 文件...")
    docs = SimpleDirectoryReader(
        "./data/txt",
        recursive=True
    ).load_data()
    print(f"  加载了 {len(docs)} 个文档")
    return docs


# ================================================================
# 数据源2：JSON 文件（手动解析转 Document）
# ================================================================
def load_from_json() -> list:
    print("\n📋 加载 JSON 文件（CMDB数据）...")
    with open("./data/json/cmdb.json", "r", encoding="utf-8") as f:
        cmdb_data = json.load(f)

    docs = []
    for server in cmdb_data:
        # 把每条记录转成自然语言文本
        text = f"""
服务器配置信息：
- IP地址：{server['ip']}
- 主机名：{server['hostname']}
- 负责人：{server['owner']}（{server['team']}）
- 所属业务：{server['business']}
- 运行环境：{server['env']}
- 所在机房：{server['idc']}
- 硬件规格：{server['spec']}
- 操作系统：{server['os']}
- 当前状态：{server['status']}
"""
        doc = Document(
            text=text,
            metadata={
                "source": "CMDB",
                "ip": server["ip"],
                "owner": server["owner"],
                "business": server["business"]
            }
        )
        docs.append(doc)

    print(f"  加载了 {len(docs)} 条CMDB记录")
    return docs


# ================================================================
# 数据源3：CSV 文件（手动解析转 Document）
# ================================================================
def load_from_csv() -> list:
    print("\n📊 加载 CSV 文件（告警历史）...")
    docs = []

    with open("./data/csv/alert_history.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()

    headers = lines[0].strip().split(",")

    for line in lines[1:]:
        values = line.strip().split(",")
        record = dict(zip(headers, values))

        text = (
            f"告警记录：{record['时间']} 服务器 {record['服务器IP']} "
            f"发生{record['告警类型']}，"
            f"当前值 {record['告警值']}，"
            f"持续 {record['持续时长']}，"
            f"处理状态：{record['处理状态']}"
        )

        doc = Document(
            text=text,
            metadata={
                "source": "告警历史",
                "server_ip": record["服务器IP"],
                "alert_type": record["告警类型"],
                "status": record["处理状态"]
            }
        )
        docs.append(doc)

    print(f"  加载了 {len(docs)} 条告警记录")
    return docs


# ================================================================
# 数据源4：SQLite 数据库（查询转 Document）
# ================================================================
def load_from_database() -> list:
    print("\n🗄️  加载数据库（运维工单）...")
    conn = sqlite3.connect("./data/ops_tickets.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tickets")
    rows = cursor.fetchall()
    conn.close()

    docs = []
    for row in rows:
        ticket_id, server_ip, title, desc, severity, status, owner, created, resolved = row
        text = f"""
运维工单记录：
- 工单号：{ticket_id}
- 标题：{title}
- 关联服务器：{server_ip}
- 严重等级：{severity}
- 处理状态：{status}
- 负责人：{owner}
- 创建时间：{created}
- 解决时间：{resolved or '未解决'}
- 详细描述：{desc}
"""
        doc = Document(
            text=text,
            metadata={
                "source": "工单系统",
                "ticket_id": ticket_id,
                "server_ip": server_ip,
                "severity": severity
            }
        )
        docs.append(doc)

    print(f"  加载了 {len(docs)} 条工单记录")
    return docs


# ================================================================
# 合并所有数据源，构建统一索引
# ================================================================
def build_unified_index():
    print("\n" + "=" * 60)
    print("📌 构建统一知识库索引")
    print("=" * 60)

    persist_dir = "./multi_source_index"

    if os.path.exists(persist_dir):
        print("📂 发现已有索引，直接加载...")
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        index = load_index_from_storage(storage_context)
        print("✅ 索引加载完成")
        return index

    # 从4个数据源加载文档
    all_docs = []
    all_docs.extend(load_from_txt())
    all_docs.extend(load_from_json())
    all_docs.extend(load_from_csv())
    all_docs.extend(load_from_database())

    print(f"\n📦 共加载 {len(all_docs)} 个文档，开始构建索引...")

    # 构建统一向量索引
    index = VectorStoreIndex.from_documents(
        all_docs,
        show_progress=True
    )

    # 持久化
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"\n✅ 索引构建完成，已保存到 {persist_dir}/")

    return index


# ================================================================
# 统一查询入口
# ================================================================
def demo_unified_query(index):
    print("\n" + "=" * 60)
    print("📌 多数据源统一查询演示")
    print("=" * 60)

    query_engine = index.as_query_engine(
        similarity_top_k=4,
        response_mode="compact"
    )

    questions = [
        # 查运维手册（TXT）
        "Redis 内存不足应该怎么处理？",
        # 查 CMDB（JSON）
        "192.168.1.102 这台服务器是谁负责的？",
        # 查告警历史（CSV）
        "最近有哪些服务器发生过告警？",
        # 查工单系统（DB）
        "INC001 工单的处理结果是什么？",
        # 跨数据源综合查询
        "王五负责的服务器最近有什么问题？"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        response = query_engine.query(q)
        print(f"💡 {response}")

        # 显示检索来源
        sources = set(
            node.metadata.get("source", node.metadata.get("file_name", "未知"))
            for node in response.source_nodes
        )
        print(f"📚 来源：{', '.join(sources)}")


# ================================================================
# 按数据源分类查询（精细控制）
# ================================================================
def demo_filtered_query(index):
    print("\n" + "=" * 60)
    print("📌 按数据源过滤查询")
    print("=" * 60)

    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    # 只查 CMDB 数据源
    print("\n🔍 只查 CMDB 数据：")
    filters = MetadataFilters(filters=[
        MetadataFilter(key="source", value="CMDB")
    ])
    cmdb_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters
    )
    response = cmdb_engine.query("上海机房A区有哪些服务器？")
    print(f"💡 {response}")

    # 只查工单系统
    print("\n🔍 只查工单数据：")
    filters2 = MetadataFilters(filters=[
        MetadataFilter(key="source", value="工单系统")
    ])
    ticket_engine = index.as_query_engine(
        similarity_top_k=3,
        filters=filters2
    )
    response2 = ticket_engine.query("有哪些P1级别的工单？")
    print(f"💡 {response2}")


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 13～14：多数据源接入")

    # 准备数据
    prepare_all_sources()

    # 构建统一索引
    index = build_unified_index()

    print("\n选择演示：")
    print("  1. 多数据源统一查询")
    print("  2. 按数据源过滤查询")
    print("  0. 运行全部")

    choice = input("\n请输入编号：").strip()

    if choice == "1" or choice == "0":
        demo_unified_query(index)

    if choice == "2" or choice == "0":
        demo_filtered_query(index)

    print("\n✅ Day 13～14 完成！下一步：Day 15～16 高级检索策略")