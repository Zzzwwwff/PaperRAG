"""
ReAct Agent 循环引擎
====================
Think → Act → Observe 循环，通过 Function Calling 与 DeepSeek 交互。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
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


def _safe_messages(msgs):
    """确保消息列表全部为可 JSON 序列化的 dict"""
    out = []
    for m in msgs:
        if isinstance(m, dict):
            out.append(m)
        else:
            # ChatCompletionMessage 等对象 → dict
            role = getattr(m, "role", "")
            content = getattr(m, "content", None)
            tc = getattr(m, "tool_calls", None)
            entry = {"role": role, "content": content}
            if tc:
                entry["tool_calls"] = [
                    {"id": t.id, "type": "function",
                     "function": {"name": t.function.name, "arguments": t.function.arguments}}
                    for t in tc
                ]
            elif hasattr(m, "tool_call_id"):
                entry["tool_call_id"] = m.tool_call_id
            out.append(entry)
    return out


def _trim_messages(messages):
    """滑动窗口裁剪，保证不拆散 tool_calls/tool 配对"""
    if len(messages) <= MAX_HISTORY_ROUNDS * 3 + 1:
        return messages
    system = [messages[0]]
    recent = messages[-(MAX_HISTORY_ROUNDS * 3):]
    trimmed = system + recent
    est = sum(len(str(m)) // 3 for m in trimmed)
    i = 1  # 从 system 之后开始
    while est > MAX_HISTORY_TOKENS and i < len(trimmed) - 1:
        role = trimmed[i].get("role") if isinstance(trimmed[i], dict) else getattr(trimmed[i], "role", "")
        # 如果是带 tool_calls 的 assistant，连后面的 tool 消息一起跳过
        if role == "assistant" and (isinstance(trimmed[i], dict) and trimmed[i].get("tool_calls") or hasattr(trimmed[i], "tool_calls") and getattr(trimmed[i], "tool_calls", None)):
            skip = 1
            while i + skip < len(trimmed):
                r = trimmed[i + skip].get("role") if isinstance(trimmed[i + skip], dict) else getattr(trimmed[i + skip], "role", "")
                if r == "tool":
                    skip += 1
                else:
                    break
            # 弹出整个 tool 调用块
            for _ in range(skip):
                if i < len(trimmed):
                    trimmed.pop(i)
        else:
            trimmed.pop(i)
        est = sum(len(str(m)) // 3 for m in trimmed)
    return trimmed


def run_agent_stream(user_input, messages=None):
    """
    流式 ReAct 循环。生成器，逐个产出事件 dict:
      {"type": "tool", "name": "...", "success": True/False}
      {"type": "token", "content": "..."}      ← 回答的逐 token 增量
      {"type": "done", "messages": [...]}
    用法: for event in run_agent_stream("你好"): ...
    """
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages = _trim_messages(messages)
    messages.append({"role": "user", "content": user_input})

    # 本轮工具调用计数（防死循环）
    tool_call_counts = {}

    for round_num in range(1, MAX_ROUNDS + 1):
        try:
            resp = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT,
                stream=True,
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            yield {"type": "token", "content": f"\n⚠️ LLM 服务异常: {e}"}
            yield {"type": "done", "messages": _safe_messages(messages)}
            return

        # 收集流式响应
        tool_calls_buf = {}
        content_parts = []
        finish_reason = None

        for chunk in resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                yield {"type": "token", "content": delta.content}
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_buf:
                        tool_calls_buf[idx] = {"id": "", "name": "", "args": ""}
                    if tc.id:
                        tool_calls_buf[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_buf[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_buf[idx]["args"] += tc.function.arguments
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # 组装 tool_calls
        tool_calls = []
        for idx in sorted(tool_calls_buf):
            t = tool_calls_buf[idx]
            if t["id"]:
                tool_calls.append({
                    "id": t["id"],
                    "type": "function",
                    "function": {"name": t["name"], "arguments": t["args"]},
                })

        # 无 tool_calls → 直接回答完成
        if not tool_calls:
            answer = "".join(content_parts)
            messages.append({"role": "assistant", "content": answer})
            yield {"type": "done", "messages": _safe_messages(messages)}
            return

        # 有 tool_calls → 执行工具（转为纯 dict）
        messages.append({
            "role": "assistant",
            "content": "".join(content_parts) or None,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            logger.info(f"Act: {name}({str(args)[:80]})")
            result = execute_tool(name, args)
            logger.info(f"Observe: {'✓' if result.get('success') else '✗'}")

            yield {"type": "tool", "name": name,
                   "success": result.get("success", False)}
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })

        # —— 防死循环：检测相同参数重复调用（parse_document 除外）——
        for tc in tool_calls:
            name = tc["function"]["name"]
            if name == "parse_document":
                continue  # 允许重复解析
            args_key = tc["function"]["arguments"][:80]
            dup_key = f"{name}:{args_key}"
            tool_call_counts[dup_key] = tool_call_counts.get(dup_key, 0) + 1

        dup_tools = [k for k, c in tool_call_counts.items() if c >= 2]
        if dup_tools:
            logger.warning(f"死循环检测(stream): {dup_tools}，强制回答")
            answer = "⚠️ 检测到重复搜索，请尝试更具体的问题。"
            messages.append({"role": "assistant", "content": answer})
            yield {"type": "token", "content": answer}
            yield {"type": "done", "messages": _safe_messages(messages)}
            return

        if round_num >= MAX_ROUNDS - 1:
            messages.append({
                "role": "system",
                "content": "【系统提示】这是最后一轮，你必须基于现有信息直接回答用户，不要再调用工具。"
            })

    # 达到最大轮次
    answer = "⚠️ 已达到搜索上限。"
    messages.append({"role": "assistant", "content": answer})
    yield {"type": "token", "content": answer}
    yield {"type": "done", "messages": _safe_messages(messages)}


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

