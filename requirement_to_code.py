# requirement_to_code.py
# 串联 Chain 实战：需求分析 → 自动生成代码

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# ============ 初始化大模型 ============
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.2   # 生成代码要严谨，调低随机性
)


# ============ 定义结构 ============

# 需求分析结构
class RequirementAnalysis(BaseModel):
    module_name: str = Field(description="模块名称")
    entities: List[str] = Field(description="涉及的实体/数据对象列表")
    api_list: List[str] = Field(description="需要实现的接口列表，如：POST /user/add")
    business_rules: List[str] = Field(description="业务规则和约束条件")
    tech_points: List[str] = Field(description="技术要点，如：需要分页/需要缓存/需要事务")
    complexity: str = Field(description="复杂度评估：简单/中等/复杂")


# 生成代码结构
class GeneratedCode(BaseModel):
    entity: str = Field(description="实体类代码（Java）")
    mapper: str = Field(description="Mapper 接口代码")
    mapper_xml: str = Field(description="Mapper XML SQL代码")
    service: str = Field(description="Service 接口代码")
    service_impl: str = Field(description="ServiceImpl 实现类代码")
    controller: str = Field(description="Controller 代码")
    dto: str = Field(description="DTO 请求/响应对象代码")


# ============ Chain 1：需求分析 ============
analyze_parser = JsonOutputParser(pydantic_object=RequirementAnalysis)

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深 Java 后端架构师。
    请对用户需求进行详细分析，提取关键信息用于后续代码生成。

    分析维度：
    1. 识别核心实体（对应数据库表）
    2. 拆分 API 接口列表（RESTful 风格）
    3. 提炼业务规则（校验逻辑、状态流转等）
    4. 标注技术要点（分页/缓存/事务/并发等）
    5. 评估整体复杂度

    {format_instructions}"""),
    ("human", """
    需求描述：{requirement}
    技术栈：{tech_stack}
    """)
]).partial(format_instructions=analyze_parser.get_format_instructions())

analyze_chain = analyze_prompt | llm | analyze_parser


# ============ Chain 2：生成代码 ============
code_parser = JsonOutputParser(pydantic_object=GeneratedCode)

code_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深 Java 后端开发工程师，请基于需求分析结果生成完整代码。

    代码规范：
    1. 实体类：使用 Lombok（@Data/@Builder），字段加 Swagger 注解
    2. Mapper：继承 BaseMapper，复杂查询写 XML
    3. Service：接口 + 实现类分离，加事务注解
    4. Controller：RESTful 风格，统一返回 Result<T>
    5. DTO：请求用 XxxRequest，响应用 XxxResponse，加参数校验注解
    6. 代码要有注释，关键逻辑加说明

    技术栈：{tech_stack}

    {format_instructions}"""),
    ("human", """
    原始需求：{requirement}

    需求分析结果：
    - 模块名称：{module_name}
    - 核心实体：{entities}
    - API列表：{api_list}
    - 业务规则：{business_rules}
    - 技术要点：{tech_points}

    请生成完整的 Spring Boot 代码。
    """)
]).partial(format_instructions=code_parser.get_format_instructions())

code_chain = code_prompt | llm | code_parser


# ============ 串联两个 Chain ============
def build_full_chain(tech_stack: str):
    """
    构建完整的串联 Chain
    流程：需求输入 → 需求分析 → 代码生成
    """

    def analyze_and_generate(inputs: dict) -> dict:
        requirement = inputs["requirement"]

        # Step 1：需求分析
        print("🔍 第一步：正在分析需求...")
        analysis: RequirementAnalysis = analyze_chain.invoke({
            "requirement": requirement,
            "tech_stack": tech_stack
        })
        print(f"✅ 需求分析完成，识别到 {len(analysis['entities'])} 个实体，"
              f"{len(analysis['api_list'])} 个接口\n")

        # Step 2：基于分析结果生成代码
        print("⚙️  第二步：正在生成代码...")
        code: GeneratedCode = code_chain.invoke({
            "requirement": requirement,
            "tech_stack": tech_stack,
            "module_name": analysis["module_name"],
            "entities": "\n".join(analysis["entities"]),
            "api_list": "\n".join(analysis["api_list"]),
            "business_rules": "\n".join(analysis["business_rules"]),
            "tech_points": "\n".join(analysis["tech_points"])
        })
        print("✅ 代码生成完成\n")

        return {
            "analysis": analysis,
            "code": code
        }

    return RunnableLambda(analyze_and_generate)


# ============ 打印需求分析结果 ============
def print_analysis(analysis: dict):
    print("\n" + "=" * 60)
    print("📋 需求分析结果")
    print("=" * 60)
    print(f"模块名称：{analysis['module_name']}")
    print(f"复杂度：  {analysis['complexity']}")

    print(f"\n📦 核心实体（{len(analysis['entities'])}个）")
    for e in analysis['entities']:
        print(f"  • {e}")

    print(f"\n🔌 API 接口（{len(analysis['api_list'])}个）")
    for api in analysis['api_list']:
        print(f"  • {api}")

    print(f"\n📏 业务规则")
    for rule in analysis['business_rules']:
        print(f"  • {rule}")

    print(f"\n⚙️  技术要点")
    for point in analysis['tech_points']:
        print(f"  • {point}")


# ============ 打印代码 ============
def print_code(code: dict):
    sections = [
        ("📄 DTO（请求/响应对象）",  code["dto"]),
        ("📄 Entity（实体类）",      code["entity"]),
        ("📄 Mapper 接口",           code["mapper"]),
        ("📄 Mapper XML",            code["mapper_xml"]),
        ("📄 Service 接口",          code["service"]),
        ("📄 ServiceImpl 实现类",    code["service_impl"]),
        ("📄 Controller",            code["controller"]),
    ]

    for title, content in sections:
        print(f"\n{'=' * 60}")
        print(title)
        print("=" * 60)
        print(content)


# ============ 保存代码到文件 ============
def save_code(analysis: dict, code: dict, output_dir: str = "./generated"):
    os.makedirs(output_dir, exist_ok=True)

    # 根据模块名生成文件名前缀
    module = analysis["module_name"]  # 如 "User"

    files = {
        f"{module}Request.java":     code["dto"],
        f"{module}.java":            code["entity"],
        f"{module}Mapper.java":      code["mapper"],
        f"{module}Mapper.xml":       code["mapper_xml"],
        f"{module}Service.java":     code["service"],
        f"{module}ServiceImpl.java": code["service_impl"],
        f"{module}Controller.java":  code["controller"],
    }

    for filename, content in files.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✅ 已保存：{filepath}")

    print(f"\n🎉 共生成 {len(files)} 个文件，保存在 {output_dir}/ 目录")


# ============ 主函数 ============
def run(requirement: str, tech_stack: str = "Spring Boot / MyBatis / MySQL / Redis"):
    full_chain = build_full_chain(tech_stack)
    result = full_chain.invoke({"requirement": requirement})

    analysis = result["analysis"]
    code = result["code"]

    # 打印分析结果
    print_analysis(analysis)

    # 打印代码
    print_code(code)

    # 保存到文件
    print(f"\n{'=' * 60}")
    print("💾 保存代码文件")
    print("=" * 60)
    save_code(analysis, code)

    return result


# ============ 测试入口 ============
if __name__ == "__main__":

    # 示例需求（结合你的运维平台背景）
    requirement = """
    开发一个服务器资产管理模块，具体需求如下：

    1. 服务器信息管理：
       - 新增服务器（IP地址、主机名、机房、负责人、系统类型、配置规格）
       - 编辑和删除服务器信息
       - 按IP/主机名/机房/负责人搜索，支持分页

    2. 服务器状态管理：
       - 服务器有状态：正常/维护中/已下线
       - 状态变更需要记录变更时间和操作人

    3. 数据校验：
       - IP地址格式校验
       - 同一IP不能重复录入

    4. 性能要求：
       - 列表查询支持分页，每页最多  100条
       - 单台服务器详情查询走Redis缓存，TTL 5分钟
    """

    run(
        requirement=requirement,
        tech_stack="Spring Boot 3 / MyBatis-Plus / MySQL 8 / Redis / Lombok / Swagger3"
    )