# LlamaIndex 核心概念对照表
LlamaIndex 概念        对应 LangChain 概念
─────────────────────────────────────────
Document          ≈   Document
Node              ≈   chunk（文本块）
VectorStoreIndex  ≈   Chroma.from_documents()
QueryEngine       ≈   RAG Chain
Settings          ≈   全局配置（LlamaIndex独有）
StorageContext    ≈   持久化管理

# 五种方式核心差异
方式	适用场景	代码量
极简5行	快速原型、演示	极少
自定义配置	精细控制切分策略	中等
索引持久化	生产环境必用	中等
手动Document	数据库/API动态数据	中等
response_mode	不同质量/速度需求	少
# response_mode 选型建议
compact        → 日常使用，默认选择
refine         → 对答案质量要求高
tree_summarize → 文档量大（50个以上）
simple_summarize → 追求速度，快速回答

# 依 赖
pip install llama-index
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai

pip install llama-index-embeddings-huggingface
pip install sentence-transformers   # HuggingFace 模型依赖

# 高级检索策略
c

python llamaindex_basic.py
# 输入 1 看极简版（最直观）
# 输入 4 看动态CMDB查询（最实用）