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

    # 本轮工具调用计数（防死循环）
    tool_call_counts = {}

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

        # —— 防死循环：统计本轮工具调用次数 ——
        for tc in msg.tool_calls:
            name = tc.function.name
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

        # 同一工具连续调用 ≥3 次 → 强制终止并回答
        dup_tools = [t for t, c in tool_call_counts.items() if c >= 3]
        if dup_tools:
            logger.warning(f"死循环检测: {dup_tools} 已调用 {tool_call_counts} 次，强制回答")
            answer = "⚠️ 检测到重复搜索，请尝试更具体的问题（如指定论文名称、作者等）。"
            messages.append({"role": "assistant", "content": answer})
            return answer, messages

        # 最后一轮仍调用工具 → 提醒 LLM 该回答了
        if round_num >= MAX_ROUNDS - 1:
            messages.append({
                "role": "system",
                "content": "【系统提示】这是最后一轮，你必须基于现有信息直接回答用户，不要再调用工具。"
            })

        continue

    # 达到最大轮次
    answer = "⚠️ 已达到搜索上限，请尝试更具体的问题。"
    messages.append({"role": "assistant", "content": answer})
    return answer, messages


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
            yield {"type": "done", "messages": messages}
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
            yield {"type": "done", "messages": messages}
            return

        # 有 tool_calls → 执行工具
        from openai.types.chat.chat_completion_message import ChatCompletionMessage
        msg = ChatCompletionMessage(
            role="assistant", content="".join(content_parts) or None,
            tool_calls=tool_calls,
        )
        messages.append(msg)

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

        # —— 防死循环：统计本轮工具调用次数 ——
        for tc in tool_calls:
            name = tc["function"]["name"]
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

        dup_tools = [t for t, c in tool_call_counts.items() if c >= 3]
        if dup_tools:
            logger.warning(f"死循环检测(stream): {dup_tools} 已调用 {tool_call_counts} 次，强制回答")
            answer = "⚠️ 检测到重复搜索，请尝试更具体的问题。"
            messages.append({"role": "assistant", "content": answer})
            yield {"type": "token", "content": answer}
            yield {"type": "done", "messages": messages}
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
    yield {"type": "done", "messages": messages}


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

