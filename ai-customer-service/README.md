# 智能客服 Agent：LangGraph + MCP + RAG

第一版先不接复杂的数据库，先把 MCP 的完整链路跑通。

# 项目最终结构
ai-customer-service/
├── mcp_server/
│   └── server.py          # MCP Server
│
├── agent/
│   └── agent.py           # LangGraph Agent
│
├── rag/
│   ├── loader.py
│   ├── vector_store.py
│   └── retriever.py
│
├── api/
│   └── main.py            # FastAPI
│
├── requirements.txt
└── README.md

# 创建虚拟环境：
python -m venv .venv

# 激活虚拟环境：
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 安装依赖：
pip install -r requirements.txt