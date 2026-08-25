# tech_solution_generator.py
# 技术方案生成器 - 基于 LangChain + DeepSeek

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# ============ 初始化大模型 ============
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.3  # 技术方案需要严谨，降低随机性
)


# ============ 定义输出结构 ============
class TechStack(BaseModel):
    category: str = Field(description="技术分类：如后端/前端/数据库/中间件/部署")
    name: str = Field(description="技术名称")
    reason: str = Field(description="选型理由")
    alternatives: List[str] = Field(description="备选方案")


class ArchitectureLayer(BaseModel):
    layer_name: str = Field(description="层次名称：如接入层/业务层/数据层")
    components: List[str] = Field(description="该层包含的组件")
    description: str = Field(description="该层的职责说明")


class ImplementationStep(BaseModel):
    phase: str = Field(description="阶段名称：如第一期/第二期")
    duration: str = Field(description="预计耗时：如2周/1个月")
    tasks: List[str] = Field(description="该阶段任务列表")
    deliverables: List[str] = Field(description="该阶段交付物")


class RiskItem(BaseModel):
    risk: str = Field(description="风险描述")
    probability: str = Field(description="发生概率：高/中/低")
    impact: str = Field(description="影响程度：高/中/低")
    mitigation: str = Field(description="应对措施")


class TechSolution(BaseModel):
    project_name: str = Field(description="项目名称")
    background: str = Field(description="项目背景和问题描述")
    objectives: List[str] = Field(description="项目目标列表")
    tech_stack: List[TechStack] = Field(description="技术选型列表")
    architecture: List[ArchitectureLayer] = Field(description="架构设计分层")
    implementation_steps: List[ImplementationStep] = Field(description="实施步骤")
    risks: List[RiskItem] = Field(description="风险分析")
    summary: str = Field(description="方案总结")


# ============ Step 1：需求分析 Chain ============
analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深架构师，拥有10年以上大型系统设计经验。
    请对用户的需求进行深入分析，从以下维度展开：
    1. 核心业务场景和用户痛点
    2. 功能性需求（系统需要做什么）
    3. 非功能性需求（性能/可用性/扩展性/安全性）
    4. 技术挑战和难点
    5. 约束条件（团队规模/时间/预算/现有技术栈）

    分析要具体、专业，为后续技术选型和架构设计做铺垫。"""),
    ("human", """
    需求描述：{requirement}
    团队规模：{team_size}
    预计用户量：{user_scale}
    现有技术栈：{existing_stack}
    """)
])
analyze_chain = analyze_prompt | llm | StrOutputParser()

# ============ Step 2：并行生成技术方案 ============
parser = JsonOutputParser(pydantic_object=TechSolution)

solution_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个顶级技术架构师，请基于需求分析结果生成完整的技术方案。

    方案要求：
    1. 技术选型要结合团队现有技术栈，避免过度引入新技术
    2. 架构设计要分层清晰，职责明确
    3. 实施步骤要可落地，按优先级分期交付
    4. 风险分析要诚实，不能只说优点
    5. 整体方案要在成本、性能、可维护性之间取得平衡

    {format_instructions}"""),
    ("human", """
    原始需求：{requirement}
    需求分析：{analysis}
    团队规模：{team_size}
    预计用户量：{user_scale}
    现有技术栈：{existing_stack}
    重点关注：{focus_points}

    请生成完整技术方案。
    """)
]).partial(format_instructions=parser.get_format_instructions())

solution_chain = solution_prompt | llm | parser

# ============ Step 3：生成架构图描述 ============
diagram_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个技术文档专家，请根据技术方案生成架构图的文字描述。
    用 ASCII 图形表示系统架构，要清晰直观。"""),
    ("human", """
    技术方案摘要：{solution_summary}
    请画出系统架构图（ASCII风格）
    """)
])
diagram_chain = diagram_prompt | llm | StrOutputParser()


# ============ 主函数 ============
def generate_tech_solution(
        requirement: str,
        team_size: str = "5人后端团队",
        user_scale: str = "日活1万",
        existing_stack: str = "Java/Spring Boot/MySQL/Redis",
        focus_points: str = "性能、可维护性、快速交付"
) -> dict:
    """
    生成技术方案
    :param requirement:     需求描述
    :param team_size:       团队规模
    :param user_scale:      用户规模
    :param existing_stack:  现有技术栈
    :param focus_points:    重点关注方向
    :return: 完整技术方案
    """

    print("🔍 第一步：正在分析需求...")
    analysis = analyze_chain.invoke({
        "requirement": requirement,
        "team_size": team_size,
        "user_scale": user_scale,
        "existing_stack": existing_stack
    })
    print("✅ 需求分析完成\n")

    print("⚙️  第二步：正在生成技术方案...")
    solution = solution_chain.invoke({
        "requirement": requirement,
        "analysis": analysis,
        "team_size": team_size,
        "user_scale": user_scale,
        "existing_stack": existing_stack,
        "focus_points": focus_points
    })
    print("✅ 技术方案生成完成\n")

    print("🏗️  第三步：正在生成架构图...")
    diagram = diagram_chain.invoke({
        "solution_summary": f"""
        项目：{solution['project_name']}
        架构层次：{[layer['layer_name'] for layer in solution['architecture']]}
        核心技术：{[tech['name'] for tech in solution['tech_stack']]}
        """
    })
    print("✅ 架构图生成完成\n")

    return {
        "analysis": analysis,
        "solution": solution,
        "diagram": diagram
    }


# ============ 格式化打印 ============
def print_solution(result: dict):
    solution = result["solution"]
    diagram = result["diagram"]

    print("\n" + "=" * 65)
    print(f"📋 技术方案：{solution['project_name']}")
    print("=" * 65)

    # 背景
    print(f"\n📌 项目背景\n{solution['background']}")

    # 目标
    print(f"\n🎯 项目目标")
    for i, obj in enumerate(solution['objectives'], 1):
        print(f"  {i}. {obj}")

    # 技术选型
    print(f"\n🔧 技术选型")
    print(f"  {'分类':<10} {'技术':<20} {'选型理由':<30} {'备选'}")
    print(f"  {'-' * 75}")
    for tech in solution['tech_stack']:
        alts = "/".join(tech['alternatives']) if tech['alternatives'] else "无"
        print(f"  {tech['category']:<10} {tech['name']:<20} "
              f"{tech['reason']:<30} {alts}")

    # 架构设计
    print(f"\n🏗️  架构设计")
    for layer in solution['architecture']:
        print(f"\n  【{layer['layer_name']}】")
        print(f"  职责：{layer['description']}")
        print(f"  组件：{' | '.join(layer['components'])}")

    # 架构图
    print(f"\n📐 架构图")
    for line in diagram.split('\n'):
        print(f"  {line}")

    # 实施步骤
    print(f"\n📅 实施计划")
    for step in solution['implementation_steps']:
        print(f"\n  ▶ {step['phase']}（{step['duration']}）")
        print(f"    任务：")
        for task in step['tasks']:
            print(f"      • {task}")
        print(f"    交付物：")
        for deliverable in step['deliverables']:
            print(f"      ✓ {deliverable}")

    # 风险分析
    print(f"\n⚠️  风险分析")
    print(f"  {'风险':<30} {'概率':<6} {'影响':<6} {'应对措施'}")
    print(f"  {'-' * 70}")
    for risk in solution['risks']:
        print(f"  {risk['risk']:<30} {risk['probability']:<6} "
              f"{risk['impact']:<6} {risk['mitigation']}")

    # 总结
    print(f"\n📝 方案总结\n{solution['summary']}")
    print("\n" + "=" * 65)


# ============ 保存到文件 ============
def save_solution(result: dict, filename: str = "tech_solution.md"):
    solution = result["solution"]
    diagram = result["diagram"]

    content = f"""# {solution['project_name']} 技术方案

## 一、项目背景
{solution['background']}

## 二、项目目标
{chr(10).join(f'- {obj}' for obj in solution['objectives'])}

## 三、技术选型
| 分类 | 技术 | 选型理由 | 备选方案 |
|------|------|--------|--------|
{chr(10).join(f"| {t['category']} | {t['name']} | {t['reason']} | {'/'.join(t['alternatives'])} |" for t in solution['tech_stack'])}

## 四、架构设计
{chr(10).join(f"### {layer['layer_name']}{chr(10)}**职责**：{layer['description']}{chr(10)}**组件**：{' | '.join(layer['components'])}" for layer in solution['architecture'])}

## 五、架构图
```
{diagram}
```

## 六、实施计划
{chr(10).join(f"### {step['phase']}（{step['duration']}）{chr(10)}**任务**：{chr(10)}{chr(10).join(f'- {t}' for t in step['tasks'])}{chr(10)}**交付物**：{chr(10)}{chr(10).join(f'- {d}' for d in step['deliverables'])}" for step in solution['implementation_steps'])}

## 七、风险分析
| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|--------|
{chr(10).join(f"| {r['risk']} | {r['probability']} | {r['impact']} | {r['mitigation']} |" for r in solution['risks'])}

## 八、方案总结
{solution['summary']}
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ 技术方案已保存到 {filename}")


# ============ 测试入口 ============
if __name__ == "__main__":
    # 示例需求（结合你的运维平台背景）
    requirement = """
    我们公司目前运维团队管理着200台服务器，现有以下痛点：
    1. 服务器巡检依赖人工，每次巡检需要2天，效率低
    2. 告警信息分散在多个系统，缺乏统一视图
    3. 历史故障案例没有沉淀，同类问题反复出现
    4. 新人上手慢，查不到运维文档和操作规范

    希望建设一个 AI 智能运维助手，能够：
    - 自动执行服务器巡检并生成报告
    - 统一接收和处理告警
    - 基于历史故障库智能推荐解决方案
    - 支持自然语言查询运维知识库
    """

    result = generate_tech_solution(
        requirement=requirement,
        team_size="3人后端 + 1人前端",
        user_scale="内部使用，50名运维人员",
        existing_stack="Java/Spring Boot/MySQL/Redis/Vue3",
        focus_points="快速落地、与现有系统集成、AI能力引入"
    )

    # 打印结果
    print_solution(result)

    # 保存到 Markdown 文件
    save_solution(result, "tech_solution.md")
