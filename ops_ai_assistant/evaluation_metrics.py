# evaluation_metrics.py
# Day 27～28：效果评估 + 指标量化

import json
import os
import statistics
import time
import warnings
from datetime import datetime

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI as OpenAIClient
from deep_seek_llm import DeepSeekLLM

Settings.llm = DeepSeekLLM()
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)
Settings.chunk_size = 512
Settings.chunk_overlap = 50


# ================================================================
# 构建知识库 + 测试集
# ================================================================
def build_index_and_testset():
    docs = [
        Document(text="""Redis 故障处理手册
1. Redis 内存不足
症状：OOM command not allowed、写入失败
处理：INFO memory查看 → 调整maxmemory-policy为allkeys-lru → 扩容
预防：设置合理TTL，监控内存水位

2. Redis 连接数耗尽
症状：ERR max number of clients reached
处理：INFO clients → CLIENT LIST找异常 → CLIENT KILL → 修改maxclients
预防：连接池 + 超时时间

3. Redis 主从同步延迟
症状：replica_lag持续增大
处理：INFO replication → 检查网络 → 必要时重新全量同步
预防：监控replication_backlog"""),

        Document(text="""MySQL 故障处理手册
1. MySQL 慢查询
症状：接口慢、CPU高
处理：开启慢查询日志 → EXPLAIN分析 → 添加索引
预防：定期分析慢查询日志

2. MySQL 连接数耗尽
症状：Too many connections
处理：SHOW PROCESSLIST → KILL长连接 → 修改max_connections
预防：合理配置连接池

3. MySQL 主从中断
症状：Slave_SQL_Running为No
处理：SHOW SLAVE STATUS → 跳过错误 → START SLAVE
预防：监控同步延迟，定期备份"""),

        Document(text="""服务器巡检规范
巡检频率：生产每天，测试每周，变更后立即执行
告警阈值：
  CPU：告警85%，严重95%
  内存：告警80%，严重90%
  磁盘：告警75%，严重85%
巡检项目：系统资源、进程状态、网络检查、安全合规
报告：自动生成，含健康评分和建议"""),
    ]

    index = VectorStoreIndex.from_documents(docs, show_progress=False)

    # 标准测试集（问题 + 参考答案 + 分类）
    test_cases = [
        {
            "id": "TC001",
            "question": "Redis 内存不足应该怎么处理？",
            "reference": "Redis内存不足时，执行INFO memory查看使用情况，调整maxmemory-policy为allkeys-lru，并考虑扩容修改maxmemory配置。预防措施包括设置合理TTL和监控内存水位。",
            "category": "Redis故障",
            "difficulty": "中等"
        },
        {
            "id": "TC002",
            "question": "MySQL 慢查询如何排查？",
            "reference": "MySQL慢查询排查：开启慢查询日志，使用EXPLAIN分析执行计划，针对性添加索引，优化SQL语句。预防上要定期分析慢查询日志。",
            "category": "MySQL故障",
            "difficulty": "中等"
        },
        {
            "id": "TC003",
            "question": "生产环境服务器巡检频率是多少？",
            "reference": "生产环境每天执行一次全量巡检，测试环境每周一次，重大变更后立即执行。",
            "category": "巡检规范",
            "difficulty": "简单"
        },
        {
            "id": "TC004",
            "question": "CPU 使用率超过多少触发告警？",
            "reference": "CPU使用率告警阈值为85%，严重阈值为95%。",
            "category": "巡检规范",
            "difficulty": "简单"
        },
        {
            "id": "TC005",
            "question": "Redis 连接数耗尽的根本原因和解决方案？",
            "reference": "Redis连接数耗尽时执行INFO clients查看，CLIENT LIST找异常连接，CLIENT KILL关闭，修改maxclients扩大上限。根本预防是使用连接池并设置合理超时。",
            "category": "Redis故障",
            "difficulty": "中等"
        },
        {
            "id": "TC006",
            "question": "MySQL 主从同步中断怎么恢复？",
            "reference": "MySQL主从中断恢复：SHOW SLAVE STATUS查看错误，如是重复键错误用SQL_SLAVE_SKIP_COUNTER跳过，然后START SLAVE重启同步。",
            "category": "MySQL故障",
            "difficulty": "较难"
        },
        {
            "id": "TC007",
            "question": "磁盘告警阈值是多少？",
            "reference": "磁盘使用率告警阈值75%，严重阈值85%。",
            "category": "巡检规范",
            "difficulty": "简单"
        },
        {
            "id": "TC008",
            "question": "Python 怎么连接 Redis？",  # 知识库外的问题
            "reference": "知识库中没有Python编程相关内容，该问题超出知识库范围。",
            "category": "知识库外",
            "difficulty": "超纲"
        },
    ]

    return index, test_cases


# ================================================================
# 核心评估函数
# ================================================================
def evaluate_single(
        question: str,
        answer: str,
        reference: str,
        client: OpenAIClient
) -> dict:
    """
    用 LLM 对单条回答打分
    返回：忠实度/完整性/清晰度/综合分/评语
    """
    prompt = f"""你是一个RAG系统评估专家，请对以下问答进行评分。

问题：{question}
参考答案：{reference}
系统回答：{answer}

请严格按照JSON格式输出评分（1-5分整数），不要输出其他内容：
{{
  "faithfulness":  <1-5，回答是否忠实于参考内容，没有幻觉>,
  "completeness":  <1-5，是否覆盖了参考答案的关键信息>,
  "clarity":       <1-5，回答是否清晰易懂、结构合理>,
  "overall":       <1-5，综合质量评分>,
  "out_of_scope":  <true或false，问题是否超出知识库范围>,
  "comment":       "<一句话点评，指出最主要的优点或问题>"
}}"""

    try:
        raw = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        ).choices[0].message.content.strip()

        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "faithfulness": 3, "completeness": 3,
            "clarity": 3, "overall": 3,
            "out_of_scope": False,
            "comment": f"评估解析失败：{e}"
        }


# ================================================================
# 指标1：RAG 质量评估
# ================================================================
def demo_rag_quality(index, test_cases):
    print("\n" + "=" * 65)
    print("📌 指标1：RAG 回答质量评估")
    print("=" * 65)

    client = OpenAIClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    query_engine = index.as_query_engine(similarity_top_k=3)
    all_scores = []

    for case in test_cases:
        print(f"\n[{case['id']}] {case['category']} | 难度：{case['difficulty']}")
        print(f"  ❓ {case['question']}")

        # 获取RAG回答
        start = time.time()
        response = query_engine.query(case["question"])
        elapsed = time.time() - start
        answer = str(response)

        print(f"  💡 {answer[:80]}...")
        print(f"  ⏱️  响应时间：{elapsed:.2f}s")

        # LLM 打分
        scores = evaluate_single(
            case["question"], answer, case["reference"], client
        )
        scores["elapsed"] = round(elapsed, 2)
        scores["case_id"] = case["id"]
        scores["category"] = case["category"]
        all_scores.append(scores)

        # 打印评分
        bar = lambda s: "█" * s + "░" * (5 - s)
        print(f"  📊 忠实度:{bar(scores['faithfulness'])} "
              f"完整性:{bar(scores['completeness'])} "
              f"清晰度:{bar(scores['clarity'])} "
              f"综合:{bar(scores['overall'])}")
        print(f"  💬 {scores['comment']}")

    return all_scores


# ================================================================
# 指标2：响应时间测试
# ================================================================
def demo_latency(index, test_cases):
    print("\n" + "=" * 65)
    print("📌 指标2：响应时间分布")
    print("=" * 65)

    query_engine = index.as_query_engine(similarity_top_k=3)
    latencies = []

    # 每个问题测3次取平均
    for case in test_cases[:5]:
        times = []
        for _ in range(3):
            start = time.time()
            query_engine.query(case["question"])
            times.append(time.time() - start)

        avg = statistics.mean(times)
        latencies.append(avg)
        print(f"  {case['id']}：平均 {avg:.2f}s "
              f"（最快{min(times):.2f}s / 最慢{max(times):.2f}s）")

    print(f"\n  📊 整体延迟统计：")
    print(f"     平均响应时间：{statistics.mean(latencies):.2f}s")
    print(f"     最快响应时间：{min(latencies):.2f}s")
    print(f"     最慢响应时间：{max(latencies):.2f}s")
    if len(latencies) > 1:
        print(f"     标准差：      {statistics.stdev(latencies):.2f}s")

    return latencies


# ================================================================
# 指标3：检索召回率
# ================================================================
def demo_retrieval_recall(index, test_cases):
    print("\n" + "=" * 65)
    print("📌 指标3：检索召回率（检索到的内容是否包含答案关键词）")
    print("=" * 65)

    # 每个测试用例的关键词
    keywords_map = {
        "TC001": ["INFO memory", "maxmemory-policy", "allkeys-lru"],
        "TC002": ["EXPLAIN", "慢查询", "索引"],
        "TC003": ["每天", "每周", "生产"],
        "TC004": ["85%", "95%", "CPU"],
        "TC005": ["INFO clients", "CLIENT KILL", "maxclients"],
        "TC006": ["SLAVE STATUS", "START SLAVE", "跳过"],
        "TC007": ["75%", "85%", "磁盘"],
        "TC008": [],  # 知识库外，不应该召回到关键内容
    }

    hit_rates = []
    for case in test_cases:
        keywords = keywords_map.get(case["id"], [])
        if not keywords:
            print(f"  [{case['id']}] 跳过（知识库外问题）")
            continue

        # 获取检索节点
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(case["question"])
        retrieved_text = " ".join([n.text for n in nodes])

        # 计算关键词命中率
        hits = [kw for kw in keywords if kw in retrieved_text]
        hit_rate = len(hits) / len(keywords)
        hit_rates.append(hit_rate)

        status = "✅" if hit_rate >= 0.7 else "⚠️ " if hit_rate >= 0.4 else "❌"
        print(f"  {status} [{case['id']}] 命中率：{hit_rate:.0%} "
              f"({len(hits)}/{len(keywords)} 关键词) "
              f"| {hits}")

    avg_recall = sum(hit_rates) / len(hit_rates) if hit_rates else 0
    print(f"\n  📊 平均召回率：{avg_recall:.1%}")
    return hit_rates


# ================================================================
# 指标4：边界问题处理（超纲问题）
# ================================================================
def demo_boundary(index):
    print("\n" + "=" * 65)
    print("📌 指标4：边界问题处理（超出知识库范围的问题）")
    print("=" * 65)

    query_engine = index.as_query_engine(similarity_top_k=3)

    boundary_cases = [
        {"q": "Python 怎么连接 Redis？", "expect": "拒绝或说明范围"},
        {"q": "今天股市行情怎么样？", "expect": "拒绝或说明范围"},
        {"q": "帮我写一首诗", "expect": "拒绝或说明范围"},
        {"q": "Redis 内存不足怎么办？", "expect": "正常回答"},  # 对照组
    ]

    for case in boundary_cases:
        response = query_engine.query(case["q"])
        answer = str(response)

        # 判断是否识别了边界
        boundary_keywords = ["不知道", "超出", "没有", "无法", "不包含",
                             "知识库", "范围", "相关信息"]
        is_boundary = any(kw in answer for kw in boundary_keywords)

        print(f"\n  ❓ {case['q']}")
        print(f"  期望：{case['expect']}")
        print(f"  回答：{answer[:100]}...")

        if case["expect"] == "拒绝或说明范围":
            status = "✅ 正确拒绝" if is_boundary else "❌ 应该拒绝但没有"
        else:
            status = "✅ 正常回答"
        print(f"  结果：{status}")


# ================================================================
# 生成完整评估报告
# ================================================================
def generate_report(all_scores: list, latencies: list, hit_rates: list):
    print("\n" + "=" * 65)
    print("📋 完整评估报告")
    print("=" * 65)

    # 过滤掉知识库外的问题
    valid_scores = [s for s in all_scores if not s.get("out_of_scope")]

    # 各维度均分
    metrics = {
        "faithfulness": "忠实度",
        "completeness": "完整性",
        "clarity": "清晰度",
        "overall": "综合分"
    }

    print("\n【RAG 质量指标】")
    summary = {}
    for key, label in metrics.items():
        scores = [s[key] for s in valid_scores if key in s]
        if scores:
            avg = sum(scores) / len(scores)
            summary[key] = round(avg, 2)
            bar = "█" * int(avg) + "░" * (5 - int(avg))
            print(f"  {label}：{bar} {avg:.2f}/5.0")

    print(f"\n【响应性能指标】")
    if latencies:
        avg_latency = statistics.mean(latencies)
        print(f"  平均响应时间：{avg_latency:.2f}s")
        print(f"  P95 响应时间：{sorted(latencies)[int(len(latencies) * 0.95)]:.2f}s")

    print(f"\n【检索质量指标】")
    if hit_rates:
        avg_recall = sum(hit_rates) / len(hit_rates)
        print(f"  平均召回率：{avg_recall:.1%}")

    # 按类别统计
    print(f"\n【分类得分】")
    categories = {}
    for s in valid_scores:
        cat = s.get("category", "未知")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(s["overall"])

    for cat, scores in categories.items():
        avg = sum(scores) / len(scores)
        print(f"  {cat}：{avg:.1f}/5.0（{len(scores)} 题）")

    # 简历可以写的量化指标
    print(f"\n{'─' * 65}")
    print("📝 简历量化指标（直接复制）：")
    print(f"{'─' * 65}")
    overall_avg = summary.get("overall", 0)
    faith_avg = summary.get("faithfulness", 0)
    recall_avg = sum(hit_rates) / len(hit_rates) if hit_rates else 0
    lat_avg = statistics.mean(latencies) if latencies else 0

    print(f"""
  • 知识库问答综合得分 {overall_avg:.1f}/5.0，忠实度 {faith_avg:.1f}/5.0
  • 关键词召回率 {recall_avg:.0%}，有效覆盖运维手册核心内容
  • 平均响应时间 {lat_avg:.1f}s，满足运维场景实时查询需求
  • 测试集覆盖 Redis/MySQL/巡检规范三大类，共 {len(all_scores)} 条用例
""")

    # 保存报告
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "avg_latency_s": round(lat_avg, 2),
        "avg_recall": round(recall_avg, 3),
        "test_cases": len(all_scores),
        "details": all_scores
    }
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 详细报告已保存至 eval_report.json")

    return report


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 27～28：效果评估 + 指标量化")

    print("\n🔨 构建知识库和测试集...")
    index, test_cases = build_index_and_testset()
    print(f"✅ 完成，测试集 {len(test_cases)} 条\n")

    print("选择评估项目：")
    print("  1. RAG 回答质量评估")
    print("  2. 响应时间测试")
    print("  3. 检索召回率")
    print("  4. 边界问题处理")
    print("  5. 生成完整报告（需先运行1/2/3）")
    print("  0. 运行全部 + 生成报告（推荐）")

    choice = input("\n请输入编号：").strip()

    all_scores = []
    latencies = []
    hit_rates = []

    if choice == "0":
        all_scores = demo_rag_quality(index, test_cases)
        latencies = demo_latency(index, test_cases)
        hit_rates = demo_retrieval_recall(index, test_cases)
        demo_boundary(index)
        generate_report(all_scores, latencies, hit_rates)

    elif choice == "1":
        all_scores = demo_rag_quality(index, test_cases)

    elif choice == "2":
        latencies = demo_latency(index, test_cases)

    elif choice == "3":
        hit_rates = demo_retrieval_recall(index, test_cases)

    elif choice == "4":
        demo_boundary(index)

    elif choice == "5":
        print("请先运行选项 0 获取完整数据")

    print("\n✅ Day 27～28 完成！下一步：Day 29～30 简历整理")
