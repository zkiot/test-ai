# rag_evaluation.py
# Day 19～20：RAG 评估体系

import os
import warnings
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:10808"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import json
from typing import Any, Generator
from openai import OpenAI as OpenAIClient
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BatchEvalRunner,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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
# 准备知识库 + 测试集
# ================================================================
def build_index_and_testset():
    """构建知识库索引和评估测试集"""

    docs = [
        Document(text="""Redis 故障处理手册
1. Redis 内存不足
症状：OOM command not allowed、写入失败
处理步骤：
  1) 执行 INFO memory 查看内存使用情况
  2) 执行 MEMORY DOCTOR 获取诊断建议
  3) 调整 maxmemory-policy 为 allkeys-lru
  4) 扩容：修改 maxmemory 配置
预防措施：设置合理的 TTL，定期清理无用 key，监控内存水位

2. Redis 连接数耗尽
症状：ERR max number of clients reached
处理步骤：
  1) 执行 INFO clients 查看当前连接数
  2) 执行 CLIENT LIST 找到异常连接
  3) 执行 CLIENT KILL 关闭异常连接
  4) 修改 maxclients 配置
预防措施：使用连接池，设置合理的连接超时时间"""),

        Document(text="""MySQL 故障处理手册
1. MySQL 慢查询
症状：接口响应慢、数据库CPU高
处理步骤：
  1) 开启慢查询日志：SET GLOBAL slow_query_log = ON
  2) 分析慢查询：使用 pt-query-digest 工具
  3) 执行 EXPLAIN 分析执行计划
  4) 添加合适的索引
  5) 优化 SQL 语句
预防措施：定期分析慢查询日志，压测前做执行计划检查

2. MySQL 主从同步中断
症状：Slave_SQL_Running 为 No
处理步骤：
  1) SHOW SLAVE STATUS 查看错误信息
  2) 如果是重复键错误：SET GLOBAL SQL_SLAVE_SKIP_COUNTER=1
  3) 执行 START SLAVE 重启同步
预防措施：设置 expire_logs_days，监控同步延迟"""),

        Document(text="""服务器巡检规范
巡检频率：
  - 生产环境：每天执行一次全量巡检
  - 测试环境：每周执行一次基础巡检
  - 重大变更后：立即执行全量巡检

巡检阈值：
  - CPU 使用率：告警阈值 85%，严重阈值 95%
  - 内存使用率：告警阈值 80%，严重阈值 90%
  - 磁盘使用率：告警阈值 75%，严重阈值 85%

巡检项目：
  1. 系统资源（CPU/内存/磁盘/负载）
  2. 进程检查（关键进程存活、僵尸进程）
  3. 网络检查（连通性、TIME_WAIT数量）
  4. 安全检查（SSH登录失败、开放端口、漏洞补丁）

巡检报告：自动生成，包含健康评分、问题列表、处理建议"""),
    ]

    index = VectorStoreIndex.from_documents(docs, show_progress=False)

    # 评估测试集：问题 + 标准答案
    test_cases = [
        {
            "question": "Redis 内存不足应该怎么处理？",
            "reference": "Redis内存不足时，需要执行INFO memory查看内存使用，调整maxmemory-policy为allkeys-lru，并考虑扩容修改maxmemory配置。预防措施包括设置合理TTL和监控内存水位。"
        },
        {
            "question": "MySQL 慢查询如何排查？",
            "reference": "MySQL慢查询排查步骤：开启慢查询日志，使用pt-query-digest分析，执行EXPLAIN查看执行计划，添加合适索引，优化SQL语句。"
        },
        {
            "question": "生产环境服务器巡检频率是多少？",
            "reference": "生产环境每天执行一次全量巡检，测试环境每周执行一次基础巡检，重大变更后立即执行全量巡检。"
        },
        {
            "question": "CPU 告警阈值是多少？",
            "reference": "CPU使用率告警阈值为85%，严重阈值为95%。"
        },
        {
            "question": "Redis 连接数耗尽怎么解决？",
            "reference": "Redis连接数耗尽时，执行INFO clients查看连接数，CLIENT LIST找到异常连接，CLIENT KILL关闭异常连接，修改maxclients配置。预防使用连接池。"
        },
        {
            "question": "Python 如何爬取网页数据？",  # 知识库中没有的问题
            "reference": "知识库中没有关于Python爬虫的内容。"
        },
    ]

    return index, test_cases


# ================================================================
# 评估维度1：忠实度（Faithfulness）
# 回答是否忠实于检索到的上下文，不编造内容
# ================================================================
def demo_faithfulness_eval(index, test_cases):
    print("\n" + "=" * 60)
    print("📌 评估维度1：忠实度（Faithfulness）")
    print("回答是否完全基于检索内容，没有幻觉")
    print("=" * 60)

    evaluator = FaithfulnessEvaluator(llm=Settings.llm)
    query_engine = index.as_query_engine(similarity_top_k=3)

    results = []
    for case in test_cases[:4]:
        question = case["question"]
        response = query_engine.query(question)

        eval_result = evaluator.evaluate_response(response=response)

        result = {
            "question":  question,
            "answer":    str(response)[:100] + "...",
            "passing":   eval_result.passing,
            "score":     eval_result.score,
            "feedback":  eval_result.feedback or "无"
        }
        results.append(result)

        status = "✅ 通过" if eval_result.passing else "❌ 未通过"
        print(f"\n❓ {question}")
        print(f"   忠实度：{status}（分数：{eval_result.score}）")
        print(f"   反馈：{eval_result.feedback or '无'}")

    passing_rate = sum(1 for r in results if r["passing"]) / len(results)
    print(f"\n📊 忠实度通过率：{passing_rate:.0%}（{sum(1 for r in results if r['passing'])}/{len(results)}）")
    return results


# ================================================================
# 评估维度2：相关性（Relevancy）
# 检索到的内容是否与问题相关
# ================================================================
def demo_relevancy_eval(index, test_cases):
    print("\n" + "=" * 60)
    print("📌 评估维度2：相关性（Relevancy）")
    print("检索到的上下文是否与问题相关")
    print("=" * 60)

    evaluator = RelevancyEvaluator(llm=Settings.llm)
    query_engine = index.as_query_engine(similarity_top_k=3)

    results = []
    for case in test_cases:
        question = case["question"]
        response = query_engine.query(question)

        eval_result = evaluator.evaluate_response(
            query=question,
            response=response
        )

        result = {
            "question": question,
            "passing":  eval_result.passing,
            "score":    eval_result.score,
        }
        results.append(result)

        status = "✅ 相关" if eval_result.passing else "❌ 不相关"
        print(f"\n❓ {question}")
        print(f"   相关性：{status}（分数：{eval_result.score}）")

    passing_rate = sum(1 for r in results if r["passing"]) / len(results)
    print(f"\n📊 相关性通过率：{passing_rate:.0%}（{sum(1 for r in results if r['passing'])}/{len(results)}）")
    return results


# ================================================================
# 评估维度3：正确性（Correctness）
# 回答是否与标准答案一致
# ================================================================
def demo_correctness_eval(index, test_cases):
    print("\n" + "=" * 60)
    print("📌 评估维度3：正确性（Correctness）")
    print("回答是否与标准答案一致（需要人工标注答案）")
    print("=" * 60)

    evaluator = CorrectnessEvaluator(llm=Settings.llm)
    query_engine = index.as_query_engine(similarity_top_k=3)

    results = []
    for case in test_cases[:4]:
        question  = case["question"]
        reference = case["reference"]
        response  = query_engine.query(question)

        eval_result = evaluator.evaluate_response(
            query=question,
            response=response,
            reference=reference
        )

        result = {
            "question":  question,
            "score":     eval_result.score,
            "passing":   eval_result.passing,
            "feedback":  eval_result.feedback or "无"
        }
        results.append(result)

        score = eval_result.score or 0
        bar = "█" * int(score) + "░" * (5 - int(score))
        print(f"\n❓ {question}")
        print(f"   正确性评分：{bar} {score:.1f}/5.0")
        print(f"   反馈：{eval_result.feedback or '无'}")

    avg_score = sum(r["score"] or 0 for r in results) / len(results)
    print(f"\n📊 平均正确性评分：{avg_score:.2f}/5.0")
    return results


# ================================================================
# 自定义评估器（不依赖 LlamaIndex 内置）
# ================================================================
def demo_custom_eval(index, test_cases):
    print("\n" + "=" * 60)
    print("📌 自定义评估器（更灵活，直接调用 LLM）")
    print("=" * 60)

    client = OpenAIClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    query_engine = index.as_query_engine(similarity_top_k=3)

    def evaluate_answer(question: str, answer: str, reference: str) -> dict:
        """自定义评估：让 LLM 打分"""
        prompt = f"""请评估以下RAG系统的回答质量，从三个维度打分（1-5分）：

问题：{question}
参考答案：{reference}
系统回答：{answer}

请严格按照以下JSON格式输出，不要有其他内容：
{{
  "faithfulness": <1-5的整数，回答是否忠实于参考内容>,
  "completeness": <1-5的整数，回答是否完整覆盖了关键信息>,
  "clarity":      <1-5的整数，回答是否清晰易懂>,
  "overall":      <1-5的整数，综合评分>,
  "comment":      "<一句话评语>"
}}"""

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()

        try:
            # 清理可能的 markdown 代码块
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception:
            return {
                "faithfulness": 0, "completeness": 0,
                "clarity": 0, "overall": 0,
                "comment": "解析失败"
            }

    all_scores = []
    for case in test_cases[:4]:
        question  = case["question"]
        reference = case["reference"]
        response  = query_engine.query(question)
        answer    = str(response)

        scores = evaluate_answer(question, answer, reference)
        all_scores.append(scores)

        print(f"\n❓ {question}")
        print(f"   忠实度：{'█' * scores['faithfulness']}{'░' * (5-scores['faithfulness'])} {scores['faithfulness']}/5")
        print(f"   完整性：{'█' * scores['completeness']}{'░' * (5-scores['completeness'])} {scores['completeness']}/5")
        print(f"   清晰度：{'█' * scores['clarity']}{'░' * (5-scores['clarity'])} {scores['clarity']}/5")
        print(f"   综合分：{'█' * scores['overall']}{'░' * (5-scores['overall'])} {scores['overall']}/5")
        print(f"   评语：  {scores['comment']}")

    # 汇总报告
    print(f"\n{'=' * 60}")
    print("📊 评估汇总报告")
    print("=" * 60)
    metrics = ["faithfulness", "completeness", "clarity", "overall"]
    labels  = ["忠实度", "完整性", "清晰度", "综合分"]
    for metric, label in zip(metrics, labels):
        avg = sum(s[metric] for s in all_scores) / len(all_scores)
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {label}：{bar} {avg:.2f}/5.0")

    return all_scores


# ================================================================
# 批量评估 + 生成报告
# ================================================================
def demo_batch_eval(index, test_cases):
    print("\n" + "=" * 60)
    print("📌 批量评估 + 生成完整报告")
    print("=" * 60)

    client = OpenAIClient(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    query_engine = index.as_query_engine(similarity_top_k=3)

    report = {
        "total":    len(test_cases),
        "results":  [],
        "summary":  {}
    }

    all_scores = []
    for i, case in enumerate(test_cases):
        question  = case["question"]
        reference = case["reference"]

        print(f"  评估中 [{i+1}/{len(test_cases)}]：{question[:30]}...")

        # 获取 RAG 回答
        response  = query_engine.query(question)
        answer    = str(response)

        # 自定义评分
        prompt = f"""评估RAG回答，输出JSON：
{{"faithfulness":<1-5>,"completeness":<1-5>,"clarity":<1-5>,"overall":<1-5>,"comment":"<评语>"}}
问题：{question}
参考：{reference}
回答：{answer[:300]}
只输出JSON，不要其他内容。"""

        raw = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        ).choices[0].message.content.strip()

        try:
            raw = raw.replace("```json", "").replace("```", "").strip()
            scores = json.loads(raw)
        except Exception:
            scores = {"faithfulness": 3, "completeness": 3,
                      "clarity": 3, "overall": 3, "comment": "解析失败"}

        all_scores.append(scores)
        report["results"].append({
            "question": question,
            "answer":   answer[:200],
            "scores":   scores
        })

    # 计算汇总指标
    metrics = ["faithfulness", "completeness", "clarity", "overall"]
    report["summary"] = {
        m: round(sum(s[m] for s in all_scores) / len(all_scores), 2)
        for m in metrics
    }

    # 打印完整报告
    print(f"\n{'=' * 60}")
    print("📋 完整评估报告")
    print("=" * 60)
    print(f"测试用例总数：{report['total']}")
    print(f"\n各维度平均分：")
    label_map = {
        "faithfulness": "忠实度",
        "completeness": "完整性",
        "clarity":      "清晰度",
        "overall":      "综合分"
    }
    for metric, avg in report["summary"].items():
        label = label_map.get(metric, metric)
        bar   = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {label}：{bar} {avg}/5.0")

    # 保存报告
    with open("rag_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细报告已保存至 rag_eval_report.json")

    return report


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 19～20：RAG 评估")

    print("\n🔨 构建知识库和测试集...")
    index, test_cases = build_index_and_testset()
    print(f"✅ 知识库构建完成，测试集 {len(test_cases)} 条\n")

    print("选择演示：")
    print("  1. 忠实度评估（Faithfulness）")
    print("  2. 相关性评估（Relevancy）")
    print("  3. 正确性评估（Correctness）")
    print("  4. 自定义评估器（推荐）")
    print("  5. 批量评估 + 生成报告")
    print("  0. 运行全部")

    choice = input("\n请输入编号：").strip()

    demos = {
        "1": lambda: demo_faithfulness_eval(index, test_cases),
        "2": lambda: demo_relevancy_eval(index, test_cases),
        "3": lambda: demo_correctness_eval(index, test_cases),
        "4": lambda: demo_custom_eval(index, test_cases),
        "5": lambda: demo_batch_eval(index, test_cases),
    }

    if choice == "0":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("运行默认：自定义评估器")
        demo_custom_eval(index, test_cases)

    print("\n✅ Day 19～20 完成！")
    print("🎉 第二阶段 LlamaIndex 全部完成！")
    print("⏳ 下一步：Day 21～30 综合项目实战")