# resume_optimizer.py

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# ============ 初始化大模型 ============
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# ============ 定义输出结构 ============
class ResumeOptimization(BaseModel):
    overall_score: int = Field(description="简历整体评分 0-100")
    issues: List[str] = Field(description="发现的问题列表")
    optimized_resume: str = Field(description="优化后的完整简历内容")
    improvements: List[str] = Field(description="具体改进说明列表")
    interview_tips: List[str] = Field(description="面试建议列表")


# ============ 第一步：分析简历 ============
analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个拥有10年经验的资深HR和职业规划顾问。
    请从以下维度分析简历：
    1. 内容完整性（基本信息、教育背景、工作经历、项目经验、技能）
    2. 描述是否量化（有没有数据支撑，如提升30%、负责5人团队）
    3. 关键词匹配（是否包含岗位相关技术关键词）
    4. 语言表达（是否简洁专业，有无错别字）
    5. 整体结构（排版逻辑是否清晰）

    请用中文回答，分析要具体，指出真实问题。"""),
    ("human", "请分析这份简历：\n\n{resume}")
])
analyze_chain = analyze_prompt | llm

# ============ 第二步：优化简历 ============
optimize_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深简历优化专家，专注于互联网/IT行业。

    优化原则：
    1. 工作经历用 STAR 法则（情境→任务→行动→结果）
    2. 技术描述要量化（如：优化接口响应时间从500ms降至100ms）
    3. 突出与目标岗位相关的技能和经验
    4. 语言精炼，每条描述不超过2行
    5. 项目经验按重要程度排序
    6. 技能部分分类清晰（后端/前端/数据库/中间件/工具）

    目标岗位：{target_job}
    分析结果：{analysis}

    {format_instructions}"""),
    ("human", "请优化这份简历：\n\n{resume}")
])

# ============ 构建完整 Chain ============
parser = JsonOutputParser(pydantic_object=ResumeOptimization)
optimize_prompt_with_format = optimize_prompt.partial(
    format_instructions=parser.get_format_instructions()
)
optimize_chain = optimize_prompt_with_format | llm | parser


def optimize_resume(resume: str, target_job: str = "Java后端工程师") -> dict:
    """
    简历优化主函数
    :param resume: 原始简历文本
    :param target_job: 目标岗位
    :return: 优化结果
    """
    print("📋 第一步：正在分析简历...")
    analysis = analyze_chain.invoke({"resume": resume})

    print("✨ 第二步：正在优化简历...")
    result = optimize_chain.invoke({
        "resume": resume,
        "target_job": target_job,
        "analysis": analysis.content
    })

    return result


def print_result(result: dict):
    """格式化打印结果"""
    print("\n" + "=" * 60)
    print("📊 简历优化报告")
    print("=" * 60)

    print(f"\n🎯 整体评分：{result['overall_score']}/100")

    print("\n❌ 发现的问题：")
    for i, issue in enumerate(result['issues'], 1):
        print(f"  {i}. {issue}")

    print("\n✅ 具体改进说明：")
    for i, improvement in enumerate(result['improvements'], 1):
        print(f"  {i}. {improvement}")

    print("\n💡 面试建议：")
    for i, tip in enumerate(result['interview_tips'], 1):
        print(f"  {i}. {tip}")

    print("\n" + "=" * 60)
    print("📄 优化后的简历：")
    print("=" * 60)
    print(result['optimized_resume'])


# ============ 测试用简历 ============
test_resume = """
姓名：张三
电话：138xxxxxxxx
邮箱：zhangsan@example.com

教育背景：
江苏大学 计算机科学与技术 本科 2018-2022

工作经历：
2022.7 - 至今  ABC科技公司  Java开发工程师
- 负责后端开发
- 写代码
- 修bug
- 参与项目开发

2021.6 - 2021.9  XYZ公司  实习生
- 学习Java
- 做了一些功能

项目经验：
1. 订单管理系统
用Spring Boot做的订单系统，有增删改查功能

2. 用户管理系统
做了用户的登录注册功能

技能：
Java, MySQL, Spring Boot, Redis
"""

if __name__ == "__main__":
    result = optimize_resume(
        resume=test_resume,
        target_job="Java后端工程师"
    )
    print_result(result)
