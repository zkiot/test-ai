# 完整的配置管理（推荐写法）
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # 大模型配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    # 向量数据库
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # 调试模式
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    @classmethod
    def validate(cls):
        """启动时检查必要配置"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError("❌ DEEPSEEK_API_KEY 未配置，请检查 .env 文件")
        print("✅ 配置加载成功")

# 使用
config = Config()
config.validate()