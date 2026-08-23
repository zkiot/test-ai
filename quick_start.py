# 用 DeepSeek 替代 OpenAI（省钱！）
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from config import Config

# 加载 .env 文件（放在代码最开头！）
load_dotenv()

# 读取配置
config = Config()
deepseek_key = config.DEEPSEEK_API_KEY
deepseek_base_url = config.DEEPSEEK_BASE_URL
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=deepseek_key,
    base_url=deepseek_base_url
)

response = llm.invoke("你好，介绍一下自己")
print(response.content)
