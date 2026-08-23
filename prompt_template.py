from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)


# 定义输出结构
class CodeReview(BaseModel):
    score: int = Field(description="代码质量评分 0-100")
    issues: List[str] = Field(description="发现的问题列表")
    suggestions: List[str] = Field(description="改进建议列表")
    summary: str = Field(description="总体评价")


parser = JsonOutputParser(pydantic_object=CodeReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个资深Java代码审查专家。
    请从以下维度审查代码：
    1. 代码规范性
    2. 性能问题
    3. 安全漏洞
    4. 可维护性

    {format_instructions}"""),
    ("human", "请审查以下代码：\n```java\n{code}\n```")
]).partial(format_instructions=parser.get_format_instructions())

review_chain = prompt | llm | parser

# 测试
code = """
public List<User> getAllUsers() {
    List<User> users = new ArrayList<>();
    Connection conn = DriverManager.getConnection(url, user, pwd);
    Statement stmt = conn.createStatement();
    ResultSet rs = stmt.executeQuery("select * from user");
    while(rs.next()) {
        users.add(new User(rs.getString("name")));
    }
    return users;
}
"""

result = review_chain.invoke({"code": code})
print(f"评分：{result['score']}/100")
print(f"问题：")
for issue in result['issues']:
    print(f"  ❌ {issue}")
print(f"建议：")
for suggestion in result['suggestions']:
    print(f"  ✅ {suggestion}")
print(f"总结：{result['summary']}")
