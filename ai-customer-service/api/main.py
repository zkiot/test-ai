import logging
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.agent import create_agent_instance
from agent.checkpointer import create_checkpointer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户消息",
    )

    thread_id: str | None = Field(
        default=None,
        description="会话ID，不传则自动创建",
    )


class ChatResponse(BaseModel):

    thread_id: str
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("正在初始化 Agent...")

    async with create_checkpointer() as checkpointer:

        # 第一次运行必须 setup
        await checkpointer.setup()

        logger.info(
            "MySQL Checkpointer 初始化完成"
        )

        logger.info(
            "正在初始化 Agent..."
        )

        agent = await create_agent_instance(
            checkpointer
        )

        app.state.agent = agent

        logger.info(
            "Agent 初始化完成"
        )

        yield

        logger.info(
            "FastAPI 正在关闭..."
        )


app = FastAPI(
    title="AI Customer Service",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():

    return {
        "status": "ok"
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest
):

    thread_id = (
        request.thread_id
        or str(uuid.uuid4())
    )

    logger.info(
        "chat thread_id=%s message=%s",
        thread_id,
        request.message,
    )

    agent = app.state.agent

    try:

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request.message,
                    }
                ]
            },
            {
                "configurable": {
                    "thread_id": thread_id
                }
            },
        )

        messages = result.get(
            "messages",
            []
        )

        if not messages:
            raise RuntimeError(
                "Agent 没有返回消息"
            )

        answer = messages[-1].content

        return ChatResponse(
            thread_id=thread_id,
            answer=answer,
        )

    except Exception as e:

        logger.exception(
            "Agent 执行失败"
        )

        raise HTTPException(
            status_code=500,
            detail="智能客服处理失败",
        ) from e
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )