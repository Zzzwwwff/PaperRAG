"""
ReAct Agent 循环引擎
====================
Think → Act → Observe 循环，通过 Function Calling 与 DeepSeek 交互。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE,
    MAX_ROUNDS, LLM_TIMEOUT, SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE,
    MAX_HISTORY_ROUNDS, MAX_HISTORY_TOKENS,
)
from src.tools.registry import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _trim_messages(messages):
    """滑动窗口裁剪"""
    if len(messages) <= MAX_HISTORY_ROUNDS * 3 + 1:
        return messages
    system = [messages[0]]
    recent = messages[-(MAX_HISTORY_ROUNDS * 3):]
    trimmed = system + recent
    est = sum(len(str(m)) // 3 for m in trimmed)
    while est > MAX_HISTORY_TOKENS and len(trimmed) > 2:
        trimmed.pop(1)
        est = sum(len(str(m)) // 3 for m in trimmed)
    return trimmed


def run_agent(user_input, messages=None):
    """
    ReAct 循环主入口。返回 (answer_text, messages)。

    Args:
        user_input: 用户输入的文本
        messages: 对话历史（首次调用传 None）

    Returns:
        (answer, updated_messages)
    """
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = _trim_messages(messages)
    messages.append({"role": "user", "content": user_input})

    for round_num in range(1, MAX_ROUNDS + 1):
        logger.info(f"--- Round {round_num} ---")

        try:
            resp = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"⚠️ LLM 服务异常: {e}", messages

        msg = resp.choices[0].message

        # 无 tool_calls → 直接回答
        if not msg.tool_calls:
            answer = msg.content or ""
            messages.append({"role": "assistant", "content": answer})
            logger.info(f"Answer ({round_num} rounds)")
            return answer, messages

        # 有 tool_calls → 执行工具
        messages.append(msg)
        tool_results = []

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            logger.info(f"Act: {name}({str(args)[:80]})")
            result = execute_tool(name, args)

            status = "✓" if result.get("success") else "✗"
            err = result.get("error", {}).get("code", "")
            logger.info(f"Observe: {status} {err}")

            tool_results.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        # 工具结果已返回给 LLM（含错误信息），让它基于已有知识继续回答
        # 不再短路 —— 由 LLM 判断是重试、换工具，还是直接基于知识回答
        continue

    # 达到最大轮次
    answer = "⚠️ 已达到搜索上限，请尝试更具体的问题。"
    messages.append({"role": "assistant", "content": answer})
    return answer, messages


def build_rag_prompt(query, hits):
    """组装 RAG Prompt（供外部直接调用）"""
    parts = []
    for i, h in enumerate(hits, 1):
        src = h.get("metadata", {})
        parts.append(
            f"[来源{i}: {src.get('paper_title', '')}, {src.get('section', '')}]\n"
            f"{h['text']}"
        )
    return RAG_PROMPT_TEMPLATE.format(
        context="\n\n---\n\n".join(parts), question=query
    )


# ===== quick test =====
if __name__ == "__main__":
    ans, _ = run_agent("Hello, what phase noise models do you know?")
    print(f"\n=== Answer ===\n{ans[:500]}")

