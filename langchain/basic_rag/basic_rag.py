# basic_rag.py
# Day 9～10：基础 RAG（检索增强生成）详解

from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ============ 初始化模型 ============
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0
)

# Embedding 模型（把文字转成向量）
# 国内可用：智谱/阿里/本地 bge 模型
# 这里用 OpenAI 兼容接口示范
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# ============ 准备测试文档 ============
def create_sample_docs():
    """创建测试用的运维知识库文档"""
    os.makedirs("./docs", exist_ok=True)

    # 文档1：Redis 故障处理手册
    with open("./docs/redis_handbook.txt", "w", encoding="utf-8") as f:
        f.write("""
Redis 故障处理手册

1. Redis 内存不足
症状：OOM command not allowed、写入失败
处理步骤：
  1) 执行 INFO memory 查看内存使用情况
  2) 执行 MEMORY DOCTOR 获取诊断建议
  3) 清理过期 key：执行 DEBUG SLEEP 0
  4) 调整 maxmemory-policy 为 allkeys-lru
  5) 扩容：修改 maxmemory 配置或迁移到更大实例
预防措施：设置合理的 TTL，定期清理无用 key，监控内存水位

2. Redis 主从同步延迟
症状：主从数据不一致、replica_lag 持续增大
处理步骤：
  1) 执行 INFO replication 查看同步状态
  2) 检查网络带宽：主从之间网络是否拥塞
  3) 检查从库负载：是否处理了大量读请求
  4) 必要时重新全量同步：REPLICAOF NO ONE 再重新配置
预防措施：监控 replication_backlog，避免从库读压力过大

3. Redis 连接数耗尽
症状：ERR max number of clients reached
处理步骤：
  1) 执行 INFO clients 查看当前连接数
  2) 执行 CLIENT LIST 找到异常连接
  3) 执行 CLIENT KILL 关闭异常连接
  4) 修改 maxclients 配置
  5) 检查应用侧连接池配置是否合理
预防措施：使用连接池，设置合理的连接超时时间
""")

    # 文档2：MySQL 故障处理手册
    with open("./docs/mysql_handbook.txt", "w", encoding="utf-8") as f:
        f.write("""
MySQL 故障处理手册

1. MySQL 慢查询
症状：接口响应慢、数据库CPU高
处理步骤：
  1) 开启慢查询日志：SET GLOBAL slow_query_log = ON
  2) 分析慢查询：使用 pt-query-digest 工具
  3) 执行 EXPLAIN 分析执行计划
  4) 添加合适的索引
  5) 优化 SQL 语句（避免全表扫描、减少回表）
预防措施：定期分析慢查询日志，压测前做执行计划检查

2. MySQL 主从同步中断
症状：Slave_IO_Running 或 Slave_SQL_Running 为 No
处理步骤：
  1) SHOW SLAVE STATUS\\G 查看错误信息
  2) 如果是 1062 重复键错误：SET GLOBAL SQL_SLAVE_SKIP_COUNTER=1
  3) 如果是主库 binlog 已清理：需要重新全量同步
  4) 执行 START SLAVE 重启同步
预防措施：设置 expire_logs_days，定期备份，监控同步延迟

3. MySQL 连接数耗尽
症状：Too many connections 错误
处理步骤：
  1) SHOW PROCESSLIST 查看当前连接
  2) 找到长时间未释放的连接并 KILL
  3) 修改 max_connections 参数
  4) 检查应用连接池配置
预防措施：合理配置连接池大小，设置连接超时时间
""")

    # 文档3：服务器巡检规范
    with open("./docs/inspection_spec.txt", "w", encoding="utf-8") as f:
        f.write("""
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
  - 系统负载：1分钟负载不超过CPU核数的2倍

2. 进程检查
  - 检查关键进程是否存活
  - 检查僵尸进程数量（超过10个告警）
  - 检查进程文件描述符使用情况

3. 网络检查
  - 检查网络连通性
  - 检查 TIME_WAIT 连接数（超过5000告警）
  - 检查带宽使用率（超过80%告警）

4. 安全检查
  - 检查 SSH 登录失败次数（超过100次/小时告警）
  - 检查开放端口变化
  - 检查系统用户变化
  - 检查高危漏洞补丁状态

巡检报告：
  每次巡检完成后自动生成报告，发送给负责人
  报告包含：健康评分、问题列表、处理建议
""")

    print("✅ 测试文档创建完成：./docs/ 目录下共3个文件")


# ================================================================
# 第一步：文档加载
# ================================================================
def demo_document_loading():
    print("\n" + "=" * 60)
    print("📌 第一步：文档加载")
    print("=" * 60)

    # 方式1：加载单个文件
    loader = TextLoader("./docs/redis_handbook.txt", encoding="utf-8")
    docs = loader.load()
    print(f"单文件加载：{len(docs)} 个文档")
    print(f"内容预览：{docs[0].page_content[:100]}...")
    print(f"元数据：{docs[0].metadata}")

    # 方式2：加载整个目录
    dir_loader = DirectoryLoader(
        "./docs",
        glob="**/*.txt",           # 匹配所有 txt 文件
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    all_docs = dir_loader.load()
    print(f"\n目录加载：{len(all_docs)} 个文档")
    for doc in all_docs:
        print(f"  • {doc.metadata['source']} "
              f"（{len(doc.page_content)} 字符）")

    return all_docs


# ================================================================
# 第二步：文档切分（Chunking）
# ================================================================
def demo_text_splitting(docs):
    print("\n" + "=" * 60)
    print("📌 第二步：文档切分（Chunking）")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # 每块最多300字符
        chunk_overlap=50,    # 相邻块重叠50字符（防止信息截断）
        separators=["\n\n", "\n", "。", "，", " "]  # 按优先级切分
    )

    chunks = splitter.split_documents(docs)

    print(f"切分前：{len(docs)} 个文档")
    print(f"切分后：{len(chunks)} 个文本块")
    print(f"平均块大小：{sum(len(c.page_content) for c in chunks) // len(chunks)} 字符")

    # 展示切分效果
    print(f"\n前3个文本块预览：")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n  【块 {i+1}】（{len(chunk.page_content)} 字符）")
        print(f"  {chunk.page_content[:80]}...")

    return chunks


# ================================================================
# 第三步：向量化 + 存储
# ================================================================
def demo_vectorstore(chunks):
    print("\n" + "=" * 60)
    print("📌 第三步：向量化 + 存入向量数据库")
    print("=" * 60)

    print("正在向量化文本块...")

    # 创建向量数据库（本地持久化）
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"   # 持久化路径
    )

    print(f"✅ 向量化完成，共存储 {vectorstore._collection.count()} 个向量")

    # 演示相似度搜索
    print("\n🔍 相似度搜索演示：")
    query = "Redis 内存不足怎么处理"
    results = vectorstore.similarity_search_with_score(query, k=3)

    for i, (doc, score) in enumerate(results):
        print(f"\n  【结果 {i+1}】相似度分数：{score:.4f}")
        print(f"  来源：{doc.metadata.get('source', '未知')}")
        print(f"  内容：{doc.page_content[:100]}...")

    return vectorstore


# ================================================================
# 第四步：构建 RAG Chain
# ================================================================
def demo_rag_chain(vectorstore):
    print("\n" + "=" * 60)
    print("📌 第四步：构建 RAG Chain")
    print("=" * 60)

    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",   # 相似度搜索
        search_kwargs={"k": 3}      # 返回最相关的3个文本块
    )

    # RAG Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的运维知识助手。
        请严格基于以下检索到的知识库内容回答问题。
        如果知识库中没有相关信息，直接说"知识库中暂无此信息"，不要编造答案。

        知识库内容：
        {context}
        """),
        ("human", "{input}")
    ])

    # 构建 RAG Chain（两种写法）

    # 写法1：官方推荐（create_retrieval_chain）
    doc_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    # 测试问题
    test_questions = [
        "Redis 内存不足应该怎么处理？",
        "MySQL 主从同步中断了，怎么排查？",
        "服务器巡检的频率是怎么规定的？",
        "CPU 使用率超过多少会触发告警？",
        "Python 怎么连接 Redis？"  # 知识库中没有的内容
    ]

    for question in test_questions:
        print(f"\n❓ 问题：{question}")
        result = rag_chain.invoke({"input": question})
        print(f"💡 回答：{result['answer']}")

        # 显示检索到的来源
        sources = set(doc.metadata.get('source', '') for doc in result['context'])
        print(f"📚 来源：{', '.join(sources)}")

    return rag_chain


# ================================================================
# 第五步：LCEL 写法（更灵活）
# ================================================================
def demo_lcel_rag(vectorstore):
    print("\n" + "=" * 60)
    print("📌 第五步：LCEL 写法 RAG（更灵活）")
    print("=" * 60)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        """把检索到的文档列表格式化成字符串"""
        return "\n\n".join([
            f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}"
            for doc in docs
        ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """基于以下知识库内容回答问题，不知道就说不知道：
        {context}"""),
        ("human", "{question}")
    ])

    # LCEL 写法：更直观的管道
    rag_chain = (
        {
            "context": retriever | format_docs,  # 检索 → 格式化
            "question": RunnablePassthrough()     # 问题直接透传
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 流式输出
    print("❓ 问题：MySQL 慢查询如何排查？")
    print("💡 回答（流式输出）：", end="", flush=True)
    for chunk in rag_chain.stream("MySQL 慢查询如何排查？"):
        print(chunk, end="", flush=True)
    print()


# ================================================================
# 第六步：加载已有向量库（持久化的价值）
# ================================================================
def demo_load_existing():
    print("\n" + "=" * 60)
    print("📌 第六步：加载已有向量库（无需重新向量化）")
    print("=" * 60)

    # 直接加载已持久化的向量库，不需要重新处理文档
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    count = vectorstore._collection.count()
    print(f"✅ 成功加载向量库，共 {count} 个向量")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "基于知识库回答：{context}"),
        ("human", "{input}")
    ])

    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )

    result = chain.invoke({"input": "磁盘使用率超过多少需要告警？"})
    print(f"❓ 问题：磁盘使用率超过多少需要告警？")
    print(f"💡 回答：{result['answer']}")


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 9～10：基础 RAG 完整流程")

    # 第一次运行：创建测试文档
    if not os.path.exists("./docs"):
        create_sample_docs()

    print("\n选择演示：")
    print("  1. 完整流程（文档加载→切分→向量化→RAG问答）")
    print("  2. 只演示 RAG 问答（需先运行过1）")
    print("  3. LCEL 写法 + 流式输出")
    print("  0. 运行全部")

    choice = input("\n请输入编号：").strip()

    if choice == "1" or choice == "0":
        # 完整流程
        all_docs = demo_document_loading()
        chunks = demo_text_splitting(all_docs)
        vectorstore = demo_vectorstore(chunks)
        rag_chain = demo_rag_chain(vectorstore)

    if choice == "2":
        # 加载已有向量库
        demo_load_existing()

    if choice == "3" or choice == "0":
        # LCEL 写法
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        demo_lcel_rag(vectorstore)

    if choice == "0":
        demo_load_existing()
    print("\n✅ Day 9～10 完成！")
    print("下一步：Day 11～12 LlamaIndex 基础")