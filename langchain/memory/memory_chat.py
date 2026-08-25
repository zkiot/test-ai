# memory_chat.py
# Day 5～6：Memory 多轮对话详解

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7
)


# ================================================================
# 方式一：最简单的 Memory（全量历史）
# ================================================================
def demo_basic_memory():
    print("\n" + "=" * 60)
    print("📌 方式一：基础 Memory（记住全部历史）")
    print("=" * 60)

    # session 存储（实际项目可换成 Redis）
    # key = session_id，value = ChatMessageHistory
    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        """根据 session_id 获取对话历史，不存在则创建"""
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # 构建带 Memory 的 Prompt
    # MessagesPlaceholder 是历史消息的占位符
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的 Java 技术顾问，记住用户的信息来个性化回答。"),
        MessagesPlaceholder(variable_name="chat_history"),  # 历史消息插入位置
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    # 包装成带 Memory 的 Chain
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",           # 用户输入的 key
        history_messages_key="chat_history"   # 历史消息的 key
    )

    # 同一个 session_id = 同一个对话上下文
    config = {"configurable": {"session_id": "user_001"}}

    # 多轮对话测试
    conversations = [
        "我叫张三，是一个有3年经验的Java后端工程师",
        "我主要用Spring Boot和MySQL，最近在学Redis",
        "我叫什么名字？我有几年经验？",        # 测试是否记住信息
        "基于我的背景，推荐我下一步学什么？"   # 测试是否能联系上下文
    ]

    for msg in conversations:
        print(f"\n👤 用户：{msg}")
        response = chain_with_memory.invoke(
            {"input": msg},
            config=config
        )
        print(f"🤖 AI：{response}")

    # 查看历史消息
    print(f"\n📜 历史消息记录（共 {len(store['user_001'].messages)} 条）：")
    for i, msg in enumerate(store["user_001"].messages):
        role = "👤 用户" if isinstance(msg, HumanMessage) else "🤖 AI"
        content_preview = msg.content[:40] + "..." if len(msg.content) > 40 else msg.content
        print(f"  [{i+1}] {role}：{content_preview}")


# ================================================================
# 方式二：滑动窗口 Memory（只记最近 N 轮）
# ================================================================
def demo_window_memory():
    print("\n" + "=" * 60)
    print("📌 方式二：滑动窗口 Memory（只记最近3轮）")
    print("=" * 60)

    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    def get_windowed_history(session_id: str) -> InMemoryChatMessageHistory:
        """只返回最近 3 轮（6条消息：3个Human + 3个AI）"""
        full_history = get_session_history(session_id)
        windowed = InMemoryChatMessageHistory()
        # 取最近 6 条消息
        recent_messages = full_history.messages[-6:]
        for msg in recent_messages:
            windowed.add_message(msg)
        return windowed

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个运维助手，只记住最近3轮对话。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain_with_window = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_windowed_history,   # 使用滑动窗口版本
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    config = {"configurable": {"session_id": "ops_session"}}

    conversations = [
        "第1轮：服务器A IP是192.168.1.1",
        "第2轮：服务器B IP是192.168.1.2",
        "第3轮：服务器C IP是192.168.1.3",
        "第4轮：服务器D IP是192.168.1.4",
        "还记得服务器A的IP吗？"   # 第1轮已滑出窗口，应该不记得了
    ]

    for msg in conversations:
        print(f"\n👤 用户：{msg}")
        response = chain_with_window.invoke(
            {"input": msg},
            config=config
        )
        print(f"🤖 AI：{response}")


# ================================================================
# 方式三：多用户隔离（不同 session_id）
# ================================================================
def demo_multi_user():
    print("\n" + "=" * 60)
    print("📌 方式三：多用户隔离（不同用户独立对话）")
    print("=" * 60)

    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个个性化技术助手，记住每个用户的信息。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain_with_memory = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    # 用户A 的对话
    print("\n--- 用户A（张三）的对话 ---")
    config_a = {"configurable": {"session_id": "user_zhangsan"}}
    chain_with_memory.invoke({"input": "我是张三，专注Java后端"}, config=config_a)
    resp_a = chain_with_memory.invoke({"input": "我叫什么？专注什么方向？"}, config=config_a)
    print(f"🤖 AI（对张三）：{resp_a}")

    # 用户B 的对话（完全独立）
    print("\n--- 用户B（李四）的对话 ---")
    config_b = {"configurable": {"session_id": "user_lisi"}}
    chain_with_memory.invoke({"input": "我是李四，专注前端Vue3"}, config=config_b)
    resp_b = chain_with_memory.invoke({"input": "我叫什么？专注什么方向？"}, config=config_b)
    print(f"🤖 AI（对李四）：{resp_b}")

    # 验证隔离性
    print("\n--- 验证隔离性 ---")
    resp_isolation = chain_with_memory.invoke(
        {"input": "你还记得李四吗？"},
        config=config_a  # 张三的 session
    )
    print(f"🤖 AI（张三的session问李四）：{resp_isolation}")
    # 预期：AI 不知道李四是谁，因为是不同 session


# ================================================================
# 方式四：实战 - 智能运维对话助手 + 忘记/总结命令
# ================================================================
def  build_ops_summary(messages):
    """把历史消息总结成一个简短的运维摘要"""
    if not messages:
        return "当前没有任何对话记录，无法总结。"

    text = "\n".join(
        f"{('用户' if isinstance(msg, HumanMessage) else '助手')}: {msg.content}"
        for msg in messages
    )

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业的运维助手。请基于以下对话内容输出简洁总结：
        1. 服务器/环境信息
        2. 已发现的问题
        3. 已做的排查与诊断
        4. 结论或后续建议

        总结要求：简洁、专业、可直接用于复盘。"""),
        ("human", "{history}")
    ])

    return (summary_prompt | llm | StrOutputParser()).invoke({"history": text})


def demo_ops_assistant():
    print("\n" + "=" * 60)
    print("📌 方式四：实战 - 智能运维对话助手")
    print("=" * 60)
    print("💡 额外支持：‘忘掉刚才的内容’、‘总结一下我们聊了什么’")

    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的智能运维助手。

        你的能力：
        1. 记住本次对话中提到的服务器信息
        2. 记住用户的操作历史
        3. 根据上下文给出连续性建议
        4. 如果用户说"刚才那台服务器"，你要能关联上下文

        回答要简洁专业，重要信息用数字标注。"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    ops_chain = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    config = {"configurable": {"session_id": "ops_001"}}

    def handle_special_command(user_input: str):
        """处理特殊命令，返回 (handled, response)"""
        text = user_input.strip()

        # 1. 忘记：清空对话历史
        if any(keyword in text for keyword in ["忘掉刚才的内容", "忘掉刚才", "清空历史", "忘记刚才", "清空对话", "重置对话"]):
            history = get_session_history(config["configurable"]["session_id"])
            history.clear()
            return True, "已清空本次对话历史，后续记忆已重置。"

        # 2. 总结：输出摘要
        if any(keyword in text for keyword in ["总结一下我们聊了什么", "总结一下", "聊了什么", "总结一下这次对话"]):
            history = get_session_history(config["configurable"]["session_id"])
            return True, build_ops_summary(history.messages)

        return False, ""

    # 模拟真实运维对话
    conversations = [
        "服务器 192.168.1.100 CPU突然飙到95%",
        "top命令看了下，是java进程占用高，PID是12345",
        "jstack看了下，发现大量线程在等待数据库连接",
        "数据库连接池配置是最大50个连接",
        "基于刚才的分析，给我一个完整的排查和解决方案",
        "总结一下我们聊了什么",
        "忘掉刚才的内容",
        "现在服务器 10.0.0.8 访问缓慢，请帮我排查",
        "总结一下我们聊了什么"
    ]

    for msg in conversations:
        print(f"\n👤 运维工程师：{msg}")

        handled, special_response = handle_special_command(msg)
        if handled:
            print(f"🤖 运维助手：{special_response}")
            continue

        response = ops_chain.invoke(
            {"input": msg},
            config=config
        )
        print(f"🤖 运维助手：{response}")


# ================================================================
# 方式五：手动管理历史（最灵活）
# ================================================================
def demo_manual_history():
    print("\n" + "=" * 60)
    print("📌 方式五：手动管理历史（完全控制）")
    print("=" * 60)

    # 手动维护消息列表
    history = InMemoryChatMessageHistory()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个Java面试辅导老师"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm | StrOutputParser()

    def chat(user_input: str) -> str:
        """手动管理历史的对话函数"""
        # 调用 Chain（传入当前历史）
        response = chain.invoke({
            "input": user_input,
            "chat_history": history.messages
        })

        # 手动更新历史
        history.add_user_message(user_input)
        history.add_ai_message(response)

        return response

    # 手动控制：可以在任意时机清空/修改历史
    conversations = [
        "我在准备Java面试，帮我出一道关于线程池的题",
        "我的回答：线程池有核心线程数和最大线程数...",
        "我答的怎么样？有什么遗漏？"
    ]

    for msg in conversations:
        print(f"\n👤 用户：{msg}")
        response = chat(msg)
        print(f"🤖 AI：{response}")

    # 随时可以查看/清空历史
    print(f"\n📜 当前历史消息数：{len(history.messages)}")
    history.clear()
    print(f"🗑️  清空后历史消息数：{len(history.messages)}")


# ================================================================
# 运行所有示例
# ================================================================
if __name__ == "__main__":
    print("🚀 Day 5～6：Memory 多轮对话示例")
    print("选择要运行的示例：")
    print("  1. 基础 Memory（全量历史）")
    print("  2. 滑动窗口 Memory")
    print("  3. 多用户隔离")
    print("  4. 实战运维对话助手")
    print("  5. 手动管理历史")
    print("  0. 运行全部")

    choice = input("\n请输入编号：").strip()


    demos = {
        "1": demo_basic_memory,
        "2": demo_window_memory,
        "3": demo_multi_user,
        "4": demo_ops_assistant,
        "5": demo_manual_history,
    }

    if choice == "0":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    else:
        print("❌ 无效输入，运行默认示例（方式四：运维助手）")
        demo_ops_assistant()