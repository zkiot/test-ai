# knowledge_graph.py
# Day 17～18：知识图谱（Knowledge Graph）

import os
import warnings

from llama_index.core.schema import TextNode

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    KnowledgeGraphIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI as OpenAIClient
from typing import Any, Generator


# ================================================================
# DeepSeek LLM（复用）
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
# 准备运维知识文档
# ================================================================
def prepare_documents() -> list:
    docs_text = [
        """
        张三负责订单系统。
        订单系统部署在服务器192.168.1.100上。
        服务器192.168.1.100位于上海机房A区。
        订单系统使用MySQL数据库存储订单数据。
        订单系统使用Redis缓存热点数据。
        MySQL数据库部署在服务器192.168.1.101上。
        Redis缓存部署在服务器192.168.1.102上。
        李四负责MySQL数据库的运维。
        王五负责Redis缓存集群的运维。
        """,
        """
        服务器192.168.1.100配置为8核16G。
        服务器192.168.1.100运行CentOS 7.9操作系统。
        服务器192.168.1.101配置为16核32G。
        服务器192.168.1.102配置为4核8G。
        上海机房A区包含服务器192.168.1.100和192.168.1.102。
        上海机房B区包含服务器192.168.1.101。
        """,
        """
        Redis内存不足故障的处理方式是调整maxmemory-policy。
        Redis连接数耗尽故障的处理方式是修改maxclients配置。
        MySQL慢查询故障的处理方式是添加索引和优化SQL。
        MySQL连接数耗尽故障的处理方式是修改max_connections。
        CPU告警的阈值是85%，严重阈值是95%。
        内存告警的阈值是80%，严重阈值是90%。
        磁盘告警的阈值是75%，严重阈值是85%。
        """,
        """
        自动巡检平台负责监控所有服务器。
        自动巡检平台每天对生产环境执行全量巡检。
        CMDB系统存储所有服务器的配置信息。
        告警系统负责接收和分发告警通知。
        告警系统会通知对应服务器的负责人。
        工单系统记录所有故障处理过程。
        """
    ]
    return [Document(text=text) for text in docs_text]


# ================================================================
# 方式1：自动提取三元组构建知识图谱
# ================================================================
def demo_auto_kg():
    print("\n" + "=" * 60)
    print("📌 方式1：自动提取三元组（LLM自动构建）")
    print("=" * 60)

    documents = prepare_documents()

    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    print("🔨 构建知识图谱（LLM自动提取三元组）...")
    print("   格式：（实体A，关系，实体B）\n")

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=8,
        include_embeddings=True,
        show_progress=True
    )

    print("\n✅ 知识图谱构建完成！")

    print("\n📊 提取到的三元组（部分）：")
    triplets = get_all_triplets(graph_store)
    for i, (subj, rel, obj) in enumerate(triplets[:15]):
        print(f"  ({subj})  --[{rel}]-->  ({obj})")
    print(f"  ... 共 {len(triplets)} 个三元组")

    return kg_index, graph_store

# ================================================================
# 工具函数：兼容新版 SimpleGraphStore 获取三元组
# ================================================================
def get_all_triplets(graph_store) -> list:
    """从 SimpleGraphStore 内部结构获取所有三元组"""
    triplets = []
    try:
        # 新版：_data.graph_dict
        data = graph_store._data
        for subj, relations in data.graph_dict.items():
            for rel, obj_list in relations.items():
                for obj in obj_list:
                    triplets.append((subj, rel, obj))
    except AttributeError:
        try:
            # 兼容旧版
            triplets = graph_store.get_triplets()
        except Exception:
            print("⚠️  无法获取三元组，图存储结构可能已变化")
    return triplets

# ================================================================
# 方式2：手动构建知识图谱（精确控制）
# ================================================================
def demo_manual_kg():
    print("\n" + "=" * 60)
    print("📌 方式2：手动构建知识图谱（精确控制）")
    print("=" * 60)

    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    kg_index = KnowledgeGraphIndex(
        nodes=[],
        storage_context=storage_context,
        include_embeddings=True
    )

    triplets = [
        # 人员 → 负责 → 系统
        ("张三",          "负责",     "订单系统"),
        ("李四",          "负责",     "MySQL数据库"),
        ("王五",          "负责",     "Redis缓存集群"),
        # 系统 → 部署于 → 服务器
        ("订单系统",       "部署于",   "192.168.1.100"),
        ("MySQL数据库",    "部署于",   "192.168.1.101"),
        ("Redis缓存集群",  "部署于",   "192.168.1.102"),
        # 系统 → 依赖 → 其他系统
        ("订单系统",       "依赖",     "MySQL数据库"),
        ("订单系统",       "依赖",     "Redis缓存集群"),
        # 服务器 → 位于 → 机房
        ("192.168.1.100", "位于",     "上海机房A区"),
        ("192.168.1.101", "位于",     "上海机房B区"),
        ("192.168.1.102", "位于",     "上海机房A区"),
        # 服务器 → 规格
        ("192.168.1.100", "规格为",   "8核16G"),
        ("192.168.1.101", "规格为",   "16核32G"),
        ("192.168.1.102", "规格为",   "4核8G"),
        # 故障 → 处理方式
        ("Redis内存不足",  "处理方式", "调整maxmemory-policy"),
        ("MySQL慢查询",    "处理方式", "添加索引优化SQL"),
        ("CPU告警",        "告警阈值", "85%"),
    ]

    for subj, rel, obj in triplets:
        node = TextNode(text=f"{subj} {rel} {obj}")
        kg_index.upsert_triplet_and_node((subj, rel, obj), node=node)

    print(f"✅ 手动插入 {len(triplets)} 个三元组")
    print("\n📊 图谱结构预览：")
    for subj, rel, obj in triplets[:8]:
        print(f"  ({subj})  --[{rel}]-->  ({obj})")

    return kg_index


# ================================================================
# 知识图谱查询
# ================================================================
def demo_kg_query(kg_index, title="知识图谱查询"):
    print(f"\n{'=' * 60}")
    print(f"📌 {title}")
    print("=" * 60)

    # 关键词检索
    print("\n🔍 模式1：关键词检索（图遍历）")
    keyword_engine = kg_index.as_query_engine(
        include_text=False,
        retriever_mode="keyword",
        response_mode="tree_summarize"
    )
    kw_questions = [
        "张三负责什么系统？",
        "订单系统依赖哪些组件？",
        "192.168.1.100 在哪个机房？",
    ]
    for q in kw_questions:
        print(f"\n  ❓ {q}")
        try:
            resp = keyword_engine.query(q)
            print(f"  💡 {resp}")
        except Exception as e:
            print(f"  ⚠️  {e}")

    # 混合检索
    print("\n🔍 模式2：混合检索（图遍历 + 向量语义）")
    hybrid_engine = kg_index.as_query_engine(
        include_text=True,
        retriever_mode="hybrid",
        similarity_top_k=3,
        response_mode="compact"
    )
    hybrid_questions = [
        "如果订单系统出现故障，影响范围是什么？",
        "王五管的服务器规格是多少？",
        "Redis 内存不足应该找谁处理？",
    ]
    for q in hybrid_questions:
        print(f"\n  ❓ {q}")
        try:
            resp = hybrid_engine.query(q)
            print(f"  💡 {resp}")
        except Exception as e:
            print(f"  ⚠️  {e}")


# ================================================================
# 知识图谱 vs 向量检索 对比
# ================================================================
def demo_kg_vs_vector(kg_index):
    print("\n" + "=" * 60)
    print("📌 知识图谱 vs 向量检索 对比")
    print("=" * 60)

    documents = prepare_documents()
    vector_index = VectorStoreIndex.from_documents(documents)
    vector_engine = vector_index.as_query_engine(similarity_top_k=3)
    kg_engine = kg_index.as_query_engine(
        include_text=True,
        retriever_mode="hybrid",
        similarity_top_k=3
    )

    questions = [
        ("关系推理", "张三负责的系统部署在哪台服务器上？"),
        ("关系推理", "上海机房A区有哪些服务器？"),
        ("语义理解", "服务器内存快满了该怎么办？"),
    ]

    for q_type, q in questions:
        print(f"\n❓ 【{q_type}】{q}")
        try:
            kg_resp = kg_engine.query(q)
            print(f"  🕸️  知识图谱：{str(kg_resp)[:150]}")
        except Exception as e:
            print(f"  🕸️  知识图谱：查询出错 - {e}")
        vec_resp = vector_engine.query(q)
        print(f"  📦 向量检索：{str(vec_resp)[:150]}")

    print("""
┌────────────────┬─────────────────┬─────────────────┐
│   对比项        │   知识图谱        │   向量检索        │
├────────────────┼─────────────────┼─────────────────┤
│ 关系推理        │ ✅ 强            │ ❌ 弱            │
│ 语义理解        │ ❌ 弱            │ ✅ 强            │
│ 多跳推理        │ ✅ 支持          │ ❌ 不支持         │
│ 模糊匹配        │ ❌ 弱            │ ✅ 强            │
│ 构建成本        │ 高               │ 低               │
│ 适合场景        │ 实体关系复杂      │ 文档语义问答      │
└────────────────┴─────────────────┴─────────────────┘
""")


# ================================================================
# 图谱可视化（文本版）
# ================================================================
def demo_visualize(graph_store):
    print("\n" + "=" * 60)
    print("📌 知识图谱可视化（文本版）")
    print("=" * 60)

    triplets = get_all_triplets(graph_store)
    graph_dict = {}
    for subj, rel, obj in triplets:
        if subj not in graph_dict:
            graph_dict[subj] = []
        graph_dict[subj].append((rel, obj))

    print("\n🕸️  图谱结构：\n")
    for entity, relations in list(graph_dict.items())[:8]:
        print(f"  【{entity}】")
        for rel, obj in relations:
            print(f"    └─[{rel}]─→ {obj}")
        print()


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 17～18：知识图谱")

    print("\n选择演示：")
    print("  1. 自动提取三元组（LLM构建）")
    print("  2. 手动构建知识图谱")
    print("  3. 知识图谱 vs 向量检索对比")
    print("  0. 运行全部（推荐）")

    choice = input("\n请输入编号：").strip()

    if choice == "1":
        kg_index, graph_store = demo_auto_kg()
        demo_visualize(graph_store)
        demo_kg_query(kg_index, "自动图谱查询")

    elif choice == "2":
        kg_index = demo_manual_kg()
        demo_kg_query(kg_index, "手动图谱查询")

    elif choice == "3":
        kg_index = demo_manual_kg()
        demo_kg_vs_vector(kg_index)

    elif choice == "0":
        print("\n▶ 第一部分：自动提取知识图谱")
        kg_index_auto, graph_store = demo_auto_kg()
        demo_visualize(graph_store)
        demo_kg_query(kg_index_auto, "自动图谱查询")

        print("\n▶ 第二部分：手动构建知识图谱")
        kg_index_manual = demo_manual_kg()
        demo_kg_query(kg_index_manual, "手动图谱查询")

        print("\n▶ 第三部分：与向量检索对比")
        demo_kg_vs_vector(kg_index_auto)

    else:
        print("运行默认：手动构建 + 查询")
        kg_index = demo_manual_kg()
        demo_kg_query(kg_index, "手动图谱查询")

    print("\n✅ Day 17～18 完成！下一步：Day 19～20 RAG 评估")