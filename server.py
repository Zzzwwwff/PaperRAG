"""
FastAPI Web 入口
================
python server.py  →  http://localhost:8000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.app import get_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="PaperMate — 论文知识库 Agent")
agent = get_agent()


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """对话入口"""
    try:
        answer = agent.ask(req.message)
        return {"reply": answer}
    except Exception as e:
        logger.error(f"chat error: {e}")
        return JSONResponse(status_code=500, content={"reply": f"⚠️ 出错了: {e}"})


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


# 静态前端（首页）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    # 启动前清空临时上传区（方案 A）
    agent.clear_uploads()
    uvicorn.run(app, host="127.0.0.1", port=8000)

