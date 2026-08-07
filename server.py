"""
FastAPI Web 入口
================
python server.py  →  http://localhost:8000
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.app import get_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="PaperMate — 论文知识库 Agent")
agent = get_agent()


class ChatRequest(BaseModel):
    message: str


def _sse_event(data: dict) -> str:
    """格式化 SSE 事件"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    """对话入口（非流式，兼容保留）"""
    try:
        answer = agent.ask(req.message)
        return {"reply": answer}
    except Exception as e:
        logger.error(f"chat error: {e}")
        return JSONResponse(status_code=500, content={"reply": f"⚠️ 出错了: {e}"})


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话（SSE）"""
    async def generate():
        try:
            for event in agent.ask_stream(req.message):
                yield _sse_event(event)
        except Exception as e:
            logger.error(f"stream error: {e}")
            yield _sse_event({"type": "token", "content": f"\n⚠️ 出错了: {e}"})
            yield _sse_event({"type": "done", "messages": None})
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/status")
async def status():
    """知识库状态"""
    return agent.get_stats()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传 PDF → 临时区 → 自动解析 → 通知 Agent"""
    content = await file.read()
    result = agent.upload_pdf(file.filename, content)

    if result.get("status") != "ok":
        return JSONResponse(status_code=400, content=result)

    # 自动解析，对齐 CLI 行为
    parse_result = agent.tool("parse_document", {"file_ref": result["filename"]})
    if parse_result.get("success"):
        data = parse_result["data"]
        agent.notify_document(
            data["file_ref"], data["title"],
            data.get("authors", "Unknown"), data["chunks"],
        )
        result["parse"] = {
            "title": data["title"],
            "authors": data.get("authors", ""),
            "chunks": data["chunks"],
        }
    else:
        result["parse"] = {"error": parse_result.get("error", {}).get("message", "解析失败")}

    return result


@app.post("/ingest")
async def ingest(file_ref: str):
    """将临时上传的文档入库"""
    result = agent.tool("ingest_document", {"file_ref": file_ref})
    return result


@app.get("/docs")
async def docs():
    """可用文档列表"""
    return agent.get_available_docs()


@app.get("/history")
async def history():
    """获取对话历史（供页面刷新后恢复）"""
    return {"messages": agent.get_history()}


# 静态前端（首页）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    # 启动前清空临时上传区（方案 A）
    agent.clear_uploads()
    uvicorn.run(app, host="127.0.0.1", port=8000)

