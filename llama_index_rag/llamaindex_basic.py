# llamaindex_basic.py
# Day 11～12：LlamaIndex 基础详解
import os
import warnings
from typing import Any, Generator

from llama_index.core.llms.callbacks import llm_completion_callback

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI as OpenAIClient

load_dotenv()

# ================================================================
# 自定义 DeepSeek LLM（彻底绕过模型名校验）
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

# ================================================================
# 全局配置（LlamaIndex 推荐方式）
# ================================================================
# LLM：通过 LangChain 桥接，彻底绕过模型名校验
Settings.llm = DeepSeekLLM()

# Settings.llm = LangChainLLM(llm=_langchain_llm)
# Settings.embed_model = OpenAIEmbedding(
#     model="deepseek-embedding",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base=os.getenv("DEEPSEEK_BASE_URL")
# )
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)

Settings.chunk_size = 512        # 文本块大小
Settings.chunk_overlap = 50      # 重叠大小


# ================================================================
# 准备测试文档（复用 Day9 的文档）
# ================================================================
def prepare_docs():
    os.makedirs("./docs", exist_ok=True)

    docs_content = {
        "redis_handbook.txt": """
Redis 故障处理手册

1. Redis 内存不足
症状：OOM command not allowed、写入失败
处理步骤：
  1) 执行 INFO memory 查看内存使用情况
  2) 执行 MEMORY DOCTOR 获取诊断建议
  3) 调整 maxmemory-policy 为 allkeys-lru
  4) 扩容：修改 maxmemory 配置
预防措施：设置合理的 TTL，定期清理无用 key

2. Redis 主从同步延迟
症状：主从数据不一致、replica_lag 持续增大
处理步骤：
  1) 执行 INFO replication 查看同步状态
  2) 检查网络带宽是否拥塞
  3) 必要时重新全量同步
预防措施：监控 replication_backlog

3. Redis 连接数耗尽
症状：ERR max number of clients reached
处理步骤：
  1) 执行 INFO clients 查看当前连接数
  2) 执行 CLIENT LIST 找到异常连接
  3) 修改 maxclients 配置
预防措施：使用连接池，设置合理超时
""",
        "mysql_handbook.txt": """
MySQL 故障处理手册

1. MySQL 慢查询
症状：接口响应慢、数据库CPU高
处理步骤：
  1) 开启慢查询日志：SET GLOBAL slow_query_log = ON
  2) 分析慢查询：使用 pt-query-digest 工具
  3) 执行 EXPLAIN 分析执行计划
  4) 添加合适的索引
预防措施：定期分析慢查询日志

2. MySQL 主从同步中断
症状：Slave_IO_Running 或 Slave_SQL_Running 为 No
处理步骤：
  1) SHOW SLAVE STATUS 查看错误信息
  2) 如果是重复键错误：SET GLOBAL SQL_SLAVE_SKIP_COUNTER=1
  3) 执行 START SLAVE 重启同步
预防措施：设置 expire_logs_days，监控同步延迟

3. MySQL 连接数耗尽
症状：Too many connections 错误
处理步骤：
  1) SHOW PROCESSLIST 查看当前连接
  2) KILL 长时间未释放的连接
  3) 修改 max_connections 参数
预防措施：合理配置连接池大小
""",
        "inspection_spec.txt": """
服务器巡检规范

巡检频率：
  - 生产环境：每天执行一次全量巡检
  - 测试环境：每周执行一次基础巡检
  - 重大变更后：立即执行全量巡检

巡检项目清单：

1. 系统资源
  - CPU 使用率：告警阈值 85%，严重阈值 95%
  - 内存使用率：告警阈值 80%，严重阈值 90%
  - 磁盘使用率：告警阈值 75%，严重阈值 85%

2. 进程检查
  - 检查关键进程是否存活
  - 检查僵尸进程数量（超过10个告警）

3. 网络检查
  - 检查网络连通性
  - 检查 TIME_WAIT 连接数（超过5000告警）

4. 安全检查
  - 检查 SSH 登录失败次数（超过100次/小时告警）
  - 检查开放端口变化
  - 检查高危漏洞补丁状态

巡检报告：
  每次巡检完成后自动生成报告
  报告包含：健康评分、问题列表、处理建议
"""
    }

    for filename, content in docs_content.items():
        filepath = f"./docs/{filename}"
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    print("✅ 测试文档准备完成")


# ================================================================
# 方式1：极简 5 行代码 RAG
# ================================================================
def demo_simple_rag():
    print("\n" + "=" * 60)
    print("📌 方式1：极简 RAG（5行代码）")
    print("=" * 60)

    # 第1步：加载文档
    documents = SimpleDirectoryReader("./docs").load_data()
    print(f"加载了 {len(documents)} 个文档")

    # 第2步：建立索引（自动完成切分+向量化）
    index = VectorStoreIndex.from_documents(documents)

    # 第3步：创建查询引擎
    query_engine = index.as_query_engine()

    # 第4步：查询
    questions = [
        "Redis 内存不足怎么处理？",
        "MySQL 慢查询如何排查？",
        "服务器 CPU 告警阈值是多少？"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        response = query_engine.query(q)
        print(f"💡 {response}")

    return index


# ================================================================
# 方式2：自定义配置（更精细的控制）
# ================================================================
def demo_custom_config():
    print("\n" + "=" * 60)
    print("📌 方式2：自定义配置（精细控制）")
    print("=" * 60)

    # 自定义切分器
    splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=30,
        paragraph_separator="\n\n"
    )

    # 加载文档
    documents = SimpleDirectoryReader("./docs").load_data()

    # 手动切分成 Node
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"文档数量：{len(documents)}")
    print(f"切分后 Node 数量：{len(nodes)}")
    print(f"平均 Node 大小：{sum(len(n.text) for n in nodes) // len(nodes)} 字符")

    # 预览前3个 Node
    print("\n前3个 Node 预览：")
    for i, node in enumerate(nodes[:3]):
        print(f"\n  【Node {i+1}】")
        print(f"  ID: {node.node_id[:8]}...")
        print(f"  内容: {node.text[:80]}...")
        print(f"  来源: {node.metadata.get('file_name', '未知')}")

    # 基于 Node 建立索引
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(
        similarity_top_k=3,           # 检索 Top3
        response_mode="compact"       # 压缩模式，减少 Token
    )

    response = query_engine.query("Redis 连接数耗尽的解决方案？")
    print(f"\n❓ Redis 连接数耗尽的解决方案？")
    print(f"💡 {response}")

    # 查看来源节点
    print(f"\n📚 来源节点（{len(response.source_nodes)} 个）：")
    for i, node in enumerate(response.source_nodes):
        print(f"  [{i+1}] 相似度: {node.score:.4f} | "
              f"来源: {node.metadata.get('file_name', '未知')}")
        print(f"       内容: {node.text[:60]}...")

    return index


# ================================================================
# 方式3：索引持久化（避免重复向量化）
# ================================================================
def demo_persistence():
    print("\n" + "=" * 60)
    print("📌 方式3：索引持久化（生产环境必备）")
    print("=" * 60)

    persist_dir = "./llamaindex_storage"

    if os.path.exists(persist_dir):
        # 直接加载已有索引
        print("📂 发现已有索引，直接加载...")
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir
        )
        index = load_index_from_storage(storage_context)
        print("✅ 索引加载完成，无需重新向量化！")
    else:
        # 首次创建并持久化
        print("🔨 首次创建索引...")
        documents = SimpleDirectoryReader("./docs").load_data()
        index = VectorStoreIndex.from_documents(documents)

        # 持久化到磁盘
        index.storage_context.persist(persist_dir=persist_dir)
        print(f"✅ 索引已保存到 {persist_dir}/")

    # 使用索引
    query_engine = index.as_query_engine()
    response = query_engine.query("巡检报告包含哪些内容？")
    print(f"\n❓ 巡检报告包含哪些内容？")
    print(f"💡 {response}")


# ================================================================
# 方式4：手动构建 Document（代码动态生成知识库）
# ================================================================
def demo_manual_documents():
    print("\n" + "=" * 60)
    print("📌 方式4：手动构建 Document（动态知识库）")
    print("=" * 60)

    # 模拟从数据库查出的运维数据
    server_data = [
        {
            "ip": "192.168.1.100",
            "hostname": "prod-web-01",
            "owner": "张三",
            "business": "订单系统",
            "spec": "8核16G",
            "idc": "上海机房A区"
        },
        {
            "ip": "192.168.1.101",
            "hostname": "prod-db-01",
            "owner": "李四",
            "business": "用户数据库",
            "spec": "16核32G",
            "idc": "上海机房B区"
        },
        {
            "ip": "192.168.1.102",
            "hostname": "prod-cache-01",
            "owner": "王五",
            "business": "Redis缓存集群",
            "spec": "4核8G",
            "idc": "上海机房A区"
        }
    ]

    # 把数据库记录转成 Document
    documents = []
    for server in server_data:
        text = f"""
服务器信息：
IP地址：{server['ip']}
主机名：{server['hostname']}
负责人：{server['owner']}
所属业务：{server['business']}
配置规格：{server['spec']}
所在机房：{server['idc']}
"""
        doc = Document(
            text=text,
            metadata={
                "ip": server["ip"],
                "owner": server["owner"],
                "business": server["business"]
            }
        )
        documents.append(doc)

    print(f"从数据库构建了 {len(documents)} 个 Document")

    # 建立索引
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 自然语言查询 CMDB！
    questions = [
        "192.168.1.101 是谁负责的？",
        "订单系统的服务器在哪个机房？",
        "上海机房A区有哪些服务器？"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        response = query_engine.query(q)
        print(f"💡 {response}")


# ================================================================
# 方式5：不同 response_mode 对比
# ================================================================
def demo_response_modes():
    print("\n" + "=" * 60)
    print("📌 方式5：不同 response_mode 对比")
    print("=" * 60)

    documents = SimpleDirectoryReader("./docs").load_data()
    index = VectorStoreIndex.from_documents(documents)

    question = "服务器巡检需要检查哪些项目？"
    print(f"❓ 问题：{question}\n")

    modes = {
        "compact":        "压缩模式（默认，节省Token）",
        "refine":         "精炼模式（逐个节点精炼答案，质量高）",
        "tree_summarize": "树形摘要（大量文档时推荐）",
        "simple_summarize": "简单摘要（速度最快）"
    }

    for mode, description in modes.items():
        print(f"【{mode}】{description}")
        try:
            engine = index.as_query_engine(response_mode=mode)
            response = engine.query(question)
            print(f"回答：{str(response)[:120]}...")
        except Exception as e:
            print(f"跳过：{e}")
        print()


# ================================================================
# LangChain vs LlamaIndex 对比演示
# ================================================================
def demo_comparison():
    print("\n" + "=" * 60)
    print("📌 LangChain vs LlamaIndex 核心差异")
    print("=" * 60)

    print("""
┌─────────────────┬──────────────────┬──────────────────┐
│   对比项         │   LangChain       │   LlamaIndex      │
├─────────────────┼──────────────────┼──────────────────┤
│ 代码量           │ 较多，流程明确    │ 极简，5行搞定     │
│ RAG 能力         │ 够用              │ 更强，专为RAG设计 │
│ 数据接入         │ 较丰富            │ 更丰富            │
│ 检索策略         │ 基础              │ 多种高级策略      │
│ Agent 能力       │ 更强              │ 较弱              │
│ 灵活性           │ 高               │ 中等              │
│ 学习成本         │ 中等              │ 低               │
│ 适合场景         │ 复杂编排          │ RAG问答系统       │
└─────────────────┴──────────────────┴──────────────────┘

实际项目推荐搭配：
  LlamaIndex  → 负责 RAG 知识库检索
  LangChain   → 负责 Agent 编排和工具调用
  两者结合    → 企业级 AI 应用最佳实践
""")


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 11～12：LlamaIndex 基础")

    prepare_docs()

    print("\n选择演示：")
    print("  1. 极简 RAG（5行代码）")
    print("  2. 自定义配置（精细控制）")
    print("  3. 索引持久化（生产必备）")
    print("  4. 手动构建 Document（动态知识库）")
    print("  5. 不同 response_mode 对比")
    print("  6. LangChain vs LlamaIndex 对比")
    print("  0. 运行全部")

    choice = input("\n请输入编号：").strip()

    demos = {
        "1": demo_simple_rag,
        "2": demo_custom_config,
        "3": demo_persistence,
        "4": demo_manual_documents,
        "5": demo_response_modes,
        "6": demo_comparison,
    }

    if choice == "0":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("运行默认：极简 RAG")
        demo_simple_rag()

    print("\n✅ Day 11～12 完成！下一步：Day 13～14 多数据源接入")