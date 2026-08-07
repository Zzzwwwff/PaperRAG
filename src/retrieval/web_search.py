"""
联网搜索
========
Tavily Search API 封装。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from tavily import TavilyClient
from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

logger = logging.getLogger(__name__)

_client_cache = None  # 缓存上次搜索结果，供 save_web 使用


def search_web(query, max_results=None):
    """Tavily 联网搜索，返回 [{title, url, content, score}, ...]"""
    global _client_cache
    if max_results is None:
        max_results = TAVILY_MAX_RESULTS
    if not TAVILY_API_KEY:
        return {"success": False, "error": {"code": "WEB_API_ERROR",
                "message": "Tavily API key not configured"}}

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        result = client.search(query, max_results=max_results,
                               search_depth="advanced")
        hits = []
        for r in result.get("results", []):
            hits.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            })
        _client_cache = hits
        logger.info(f"Web search: {len(hits)} results")
        return {"success": True, "data": hits}
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {"success": False, "error": {"code": "WEB_API_ERROR",
                "message": str(e)}}


def get_last_web_results():
    """获取上次缓存的搜索结果"""
    return _client_cache or []


# ===== quick test =====
if __name__ == "__main__":
    r = search_web("phase noise oscillator")
    if r["success"]:
        for h in r["data"][:3]:
            print(f"[{h['title'][:50]}] {h['content'][:80]}")
    else:
        print("search failed:", r["error"]["message"])

