# api_server.py
# Day 24～26：FastAPI 封装智能运维助手

import json
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv

load_dotenv()

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage
)

from api_agent import (app_state, query_knowledge_base, build_knowledge_base, get_server_status, query_cmdb,
                       create_ticket, langchain_llm, run_agent, SYSTEM_PROMPT)


# ================================================================
# 6. FastAPI 应用
# ================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 初始化智能运维助手...")
    app_state.knowledge_base = build_knowledge_base()
    tools = [query_knowledge_base, get_server_status, query_cmdb, create_ticket]
    app_state.tool_map = {t.name: t for t in tools}
    app_state.llm_with_tools = langchain_llm.bind_tools(tools)
    print("✅ 初始化完成！")
    yield
    print("👋 服务关闭")


app = FastAPI(
    title="智能运维助手 API",
    description="基于 LangChain + LlamaIndex 的智能运维问答平台",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# 7. 数据模型
# ================================================================
class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="会话ID")
    message: str = Field(..., description="用户消息")


class ChatResponse(BaseModel):
    session_id: str
    message: str
    elapsed_ms: int
    history_len: int


class TicketListResponse(BaseModel):
    total: int
    tickets: List[dict]


# ================================================================
# 8. 路由
# ================================================================

@app.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - app_state.start_time),
        "request_count": app_state.request_count,
        "session_count": len(app_state.session_store),
        "knowledge_base": "ready" if app_state.knowledge_base else "not ready"
    }


@app.post("/chat", response_model=ChatResponse, summary="普通对话")
async def chat(request: ChatRequest):
    app_state.request_count += 1
    start = time.time()

    sid = request.session_id
    if sid not in app_state.session_store:
        app_state.session_store[sid] = []
    history = app_state.session_store[sid]

    history.append(HumanMessage(content=request.message))

    try:
        response_text = run_agent(history)
        history.append(AIMessage(content=response_text))
        return ChatResponse(
            session_id=sid,
            message=response_text,
            elapsed_ms=int((time.time() - start) * 1000),
            history_len=len(history)
        )
    except Exception as e:
        history.pop()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", summary="流式对话（SSE）")
async def chat_stream(request: ChatRequest):
    app_state.request_count += 1
    sid = request.session_id
    if sid not in app_state.session_store:
        app_state.session_store[sid] = []
    history = app_state.session_store[sid]
    history.append(HumanMessage(content=request.message))

    def generate():
        full_response = ""
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
        try:
            for chunk in app_state.llm_with_tools.stream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'content': chunk.content}, ensure_ascii=False)}\n\n"
            history.append(AIMessage(content=full_response))
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/session/{session_id}", summary="获取会话历史")
async def get_session(session_id: str):
    history = app_state.session_store.get(session_id, [])
    messages = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    return {"session_id": session_id, "messages": messages, "total": len(messages)}


@app.delete("/session/{session_id}", summary="清空会话")
async def clear_session(session_id: str):
    if session_id in app_state.session_store:
        app_state.session_store[session_id] = []
    return {"message": f"会话 {session_id} 已清空"}


@app.get("/sessions", summary="所有会话列表")
async def list_sessions():
    sessions = [
        {"session_id": sid, "message_count": len(msgs)}
        for sid, msgs in app_state.session_store.items()
    ]
    return {"total": len(sessions), "sessions": sessions}


@app.post("/knowledge/query", summary="直接查询知识库")
async def query_knowledge(question: str):
    if app_state.knowledge_base is None:
        raise HTTPException(status_code=503, detail="知识库未初始化")
    engine = app_state.knowledge_base.as_query_engine(similarity_top_k=3)
    response = engine.query(question)
    sources = list(set(
        node.metadata.get("category", "未知")
        for node in response.source_nodes
    ))
    return {"question": question, "answer": str(response), "sources": sources}


@app.get("/tickets", response_model=TicketListResponse, summary="工单列表")
async def list_tickets(limit: int = 10):
    try:
        conn = sqlite3.connect("./ops_tickets.db")
        rows = conn.execute(
            "SELECT id, server_ip, title, severity, status, created_at "
            "FROM tickets ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        tickets = [
            {"id": r[0], "server_ip": r[1], "title": r[2],
             "severity": r[3], "status": r[4], "created_at": r[5]}
            for r in rows
        ]
        return TicketListResponse(total=len(tickets), tickets=tickets)
    except Exception:
        return TicketListResponse(total=0, tickets=[])


@app.get("/stats", summary="系统统计")
async def get_stats():
    return {
        "uptime_seconds": int(time.time() - app_state.start_time),
        "total_requests": app_state.request_count,
        "active_sessions": len(app_state.session_store),
        "total_messages": sum(len(v) for v in app_state.session_store.values()),
        "tools_available": list(app_state.tool_map.keys()),
    }


# ================================================================
# 9. 主入口
# ================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
