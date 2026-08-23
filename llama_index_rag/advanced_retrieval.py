# advanced_retrieval.py
# Day 15～16：高级检索策略

import os
import warnings
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
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
Settings.chunk_size = 256
Settings.chunk_overlap = 30


# ================================================================
# 准备测试知识库
# ================================================================
def build_index() -> VectorStoreIndex:
    docs_content = [
        ("redis",   """Redis 故障处理手册
1. Redis 内存不足
症状：OOM command not allowed
处理：INFO memory 查看 → 调整 maxmemory-policy → 扩容
预防：设置合理 TTL，监控内存水位

2. Redis 连接数耗尽
症状：ERR max number of clients reached
处理：INFO clients → CLIENT KILL → 修改 maxclients
预防：连接池 + 超时时间

3. Redis 主从同步延迟
症状：replica_lag 持续增大
处理：INFO replication → 检查网络 → 必要时重新全量同步
预防：监控 replication_backlog"""),

        ("mysql",   """MySQL 故障处理手册
1. MySQL 慢查询
症状：接口慢、CPU高
处理：开启慢查询日志 → EXPLAIN 分析 → 添加索引
预防：定期分析慢查询日志

2. MySQL 连接数耗尽
症状：Too many connections
处理：SHOW PROCESSLIST → KILL 长连接 → 修改 max_connections
预防：合理配置连接池

3. MySQL 主从同步中断
症状：Slave_SQL_Running 为 No
处理：SHOW SLAVE STATUS → 跳过错误 → START SLAVE
预防：监控同步延迟，定期备份"""),

        ("inspect", """服务器巡检规范
巡检频率：
- 生产环境：每天全量巡检
- 测试环境：每周基础巡检

巡检阈值：
- CPU 告警 85%，严重 95%
- 内存 告警 80%，严重 90%
- 磁盘 告警 75%，严重 85%

巡检项目：系统资源、进程状态、网络连通性、安全合规
巡检报告：自动生成，包含健康评分和处理建议"""),

        ("cmdb",    """CMDB 资产信息
服务器 192.168.1.100：
  主机名：prod-web-01，负责人：张三
  业务：订单系统，机房：上海A区，规格：8核16G

服务器 192.168.1.101：
  主机名：prod-db-01，负责人：李四
  业务：用户数据库，机房：上海B区，规格：16核32G

服务器 192.168.1.102：
  主机名：prod-cache-01，负责人：王五
  业务：Redis缓存集群，机房：上海A区，规格：4核8G"""),
    ]

    documents = [
        Document(text=content, metadata={"category": cat})
        for cat, content in docs_content
    ]

    print("🔨 构建索引...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("✅ 索引构建完成\n")
    return index


# ================================================================
# 策略1：基础向量检索（对照组）
# ================================================================
def demo_basic_retrieval(index):
    print("\n" + "=" * 60)
    print("📌 策略1：基础向量检索（对照组）")
    print("=" * 60)

    query_engine = index.as_query_engine(similarity_top_k=2)

    question = "Redis 内存不足和连接数耗尽分别怎么处理？"
    print(f"❓ {question}")
    response = query_engine.query(question)
    print(f"💡 {response}")
    print(f"📚 检索节点数：{len(response.source_nodes)}")


# ================================================================
# 策略2：调整 Top-K（控制检索数量）
# ================================================================
def demo_topk_retrieval(index):
    print("\n" + "=" * 60)
    print("📌 策略2：Top-K 对比（检索数量影响）")
    print("=" * 60)

    question = "服务器巡检需要检查哪些项目，阈值是多少？"

    for k in [1, 3, 5]:
        print(f"\n【Top-{k}】")
        engine = index.as_query_engine(similarity_top_k=k)
        response = engine.query(question)
        print(f"检索节点：{len(response.source_nodes)} 个")
        print(f"回答长度：{len(str(response))} 字符")
        print(f"回答预览：{str(response)[:100]}...")


# ================================================================
# 策略3：相似度阈值过滤
# ================================================================
def demo_similarity_filter(index):
    print("\n" + "=" * 60)
    print("📌 策略3：相似度阈值过滤（过滤低质量结果）")
    print("=" * 60)

    # 相似度后处理器：过滤低于阈值的节点
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5  # 先取5个
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor]  # 过滤相似度 < 0.5 的节点
    )

    questions = [
        "Redis 内存不足怎么处理？",   # 知识库有，相似度高
        "Python 如何连接数据库？",      # 知识库没有，相似度低会被过滤
    ]

    for q in questions:
        print(f"\n❓ {q}")
        response = engine.query(q)
        filtered_count = len(response.source_nodes)
        print(f"过滤后节点数：{filtered_count}")
        if filtered_count > 0:
            print(f"💡 {response}")
        else:
            print("💡 知识库中未找到相关内容（已被相似度过滤）")


# ================================================================
# 策略4：混合检索（向量 + 关键词）
# ================================================================
def demo_hybrid_retrieval(index):
    print("\n" + "=" * 60)
    print("📌 策略4：混合检索（向量 + BM25关键词）")
    print("=" * 60)

    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever

        # 从索引获取所有节点
        nodes = list(index.docstore.docs.values())

        # 向量检索器
        vector_retriever = VectorIndexRetriever(
            index=index, similarity_top_k=3
        )

        # BM25 关键词检索器
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=3
        )

        # 融合检索器（RRF 算法合并两路结果）
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            similarity_top_k=3,
            num_queries=1,        # 不生成扩展查询
            mode="reciprocal_rerank",  # RRF 重排序
            use_async=False
        )

        engine = RetrieverQueryEngine(retriever=fusion_retriever)

        question = "INFO memory 命令怎么用？"
        print(f"❓ {question}")
        print("（这个问题包含专有命令，关键词检索更有优势）")
        response = engine.query(question)
        print(f"💡 {response}")

    except ImportError:
        print("⚠️  需要安装：pip install llama-index-retrievers-bm25")
        print("跳过混合检索演示，改用纯向量检索：")
        engine = index.as_query_engine(similarity_top_k=3)
        response = engine.query("INFO memory 命令怎么用？")
        print(f"💡 {response}")


# ================================================================
# 策略5：查询重写（Query Rewriting）
# ================================================================
def demo_query_rewriting(index):
    print("\n" + "=" * 60)
    print("📌 策略5：查询重写（提升检索召回率）")
    print("=" * 60)

    from llama_index.core.retrievers import QueryFusionRetriever

    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

    # QueryFusionRetriever 会自动生成多个查询变体
    fusion_retriever = QueryFusionRetriever(
        retrievers=[retriever],
        similarity_top_k=3,
        num_queries=3,        # 自动生成3个查询变体
        use_async=False
    )

    engine = RetrieverQueryEngine(retriever=fusion_retriever)

    # 模糊/口语化的问题
    question = "服务器连不上数据库怎么办"
    print(f"❓ 原始问题（口语化）：{question}")
    print("🔄 LLM 自动生成多个查询变体，扩大检索范围...")
    response = engine.query(question)
    print(f"💡 {response}")


# ================================================================
# 策略6：重排序（Reranking）
# ================================================================
def demo_reranking(index):
    print("\n" + "=" * 60)
    print("📌 策略6：自定义重排序（按相关性二次排序）")
    print("=" * 60)

    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from typing import List, Optional

    # 自定义重排序器：关键词命中加分
    class KeywordBoostReranker(BaseNodePostprocessor):
        keywords: list = []

        def __init__(self, keywords: list):
            super().__init__()
            self.keywords = keywords

        @classmethod
        def class_name(cls) -> str:
            return "KeywordBoostReranker"

        def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None
        ) -> List[NodeWithScore]:
            for node in nodes:
                text = node.node.text.lower()
                # 命中关键词则提升分数
                boost = sum(
                    0.1 for kw in self.keywords
                    if kw.lower() in text
                )
                node.score = (node.score or 0) + boost
            # 按新分数降序排列
            return sorted(nodes, key=lambda x: x.score or 0, reverse=True)

    # 使用自定义重排序器
    reranker = KeywordBoostReranker(keywords=["内存", "OOM", "maxmemory"])
    postprocessor = SimilarityPostprocessor(similarity_cutoff=0.3)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor, reranker]
    )

    question = "内存相关的故障怎么处理？"
    print(f"❓ {question}")
    response = engine.query(question)
    print(f"💡 {response}")
    print(f"\n📊 重排序后 Top3 节点分数：")
    for i, node in enumerate(response.source_nodes[:3]):
        print(f"  [{i+1}] 分数：{node.score:.4f} | "
              f"{node.node.text[:50]}...")


# ================================================================
# 策略对比总结
# ================================================================
def demo_strategy_comparison(index):
    print("\n" + "=" * 60)
    print("📌 策略对比：同一问题不同检索策略效果")
    print("=" * 60)

    question = "连接数耗尽怎么处理？"
    print(f"❓ 测试问题：{question}\n")

    strategies = {
        "基础检索 Top2": index.as_query_engine(similarity_top_k=2),
        "基础检索 Top5": index.as_query_engine(similarity_top_k=5),
        "相似度过滤0.5": RetrieverQueryEngine(
            retriever=VectorIndexRetriever(index=index, similarity_top_k=5),
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
        ),
    }

    for name, engine in strategies.items():
        print(f"【{name}】")
        response = engine.query(question)
        print(f"  节点数：{len(response.source_nodes)}")
        print(f"  回答：{str(response)[:120]}...")
        print()


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 15～16：高级检索策略")

    index = build_index()

    print("\n选择演示：")
    print("  1. 基础向量检索（对照组）")
    print("  2. Top-K 对比")
    print("  3. 相似度阈值过滤")
    print("  4. 混合检索（向量+BM25）")
    print("  5. 查询重写")
    print("  6. 自定义重排序")
    print("  7. 策略对比")
    print("  0. 运行全部")

    choice = input("\n请输入编号：").strip()

    demos = {
        "1": lambda: demo_basic_retrieval(index),
        "2": lambda: demo_topk_retrieval(index),
        "3": lambda: demo_similarity_filter(index),
        "4": lambda: demo_hybrid_retrieval(index),
        "5": lambda: demo_query_rewriting(index),
        "6": lambda: demo_reranking(index),
        "7": lambda: demo_strategy_comparison(index),
    }

    if choice == "0":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("运行默认：策略对比")
        demo_strategy_comparison(index)

    print("\n✅ Day 15～16 完成！下一步：Day 17～18 知识图谱")