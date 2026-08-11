"""
CLI 交互入口
============
python main.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from src.app import get_agent

logger = logging.getLogger(__name__)

# ===== 终端颜色 =====
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

BANNER = f"""
{BOLD}{CYAN}  📚 PaperMate 论文知识库 Agent{RESET}
  {DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
  {DIM}输入问题对话 | /help 帮助 | /quit 退出{RESET}
"""

HELP_TEXT = f"""
{YELLOW}直接输入问题与 Agent 对话{RESET}
  {BOLD}命令:{RESET}
  {GREEN}/help{RESET}     显示帮助
  {GREEN}/status{RESET}   查看知识库状态
  {GREEN}/upload <路径>{RESET}  上传 PDF
  {GREEN}/quit{RESET}     退出
"""


def main():
    agent = get_agent()
    print(BANNER)

    session = PromptSession()
    style = Style.from_dict({"prompt": "#4a90d9 bold"})

    while True:
        try:
            user_input = session.prompt([("class:prompt", "\n> ")], style=style).strip()
        except (EOFError, KeyboardInterrupt):
            agent.clear_uploads()
            print(f"\n{GREEN}👋 再见{RESET}")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            agent.clear_uploads()
            print(f"{GREEN}👋 再见{RESET}")
            break
        elif user_input == "/help":
            print(HELP_TEXT)
            continue
        elif user_input == "/status":
            stats = agent.get_stats()
            print(f"  {CYAN}📚 知识库:{RESET} {BOLD}{stats.get('total_papers', 0)}{RESET} 篇论文, "
                  f"{BOLD}{stats.get('total_chunks', 0)}{RESET} 个片段")
            print(f"  {CYAN}📂 会话上传:{RESET} {BOLD}{stats.get('uploaded_files', 0)}{RESET} 篇")
            print(f"  {DIM}{'─' * 46}{RESET}")
            for p in stats.get("papers", [])[:5]:
                print(f"    {DIM}·{RESET} {p[:60]}")
            if len(stats.get("papers", [])) > 5:
                print(f"    {DIM}... 共 {len(stats['papers'])} 篇{RESET}")
            continue
        elif user_input.startswith("/upload"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"  {YELLOW}用法: /upload 文件路径{RESET}")
                continue
            path = Path(parts[1])
            if not path.exists():
                print(f"  {RED}❌ 文件不存在: {path}{RESET}")
                continue
            result = agent.upload_pdf(path.name, path.read_bytes())
            print(f"  {GREEN}✅ 已上传:{RESET} {result['filename']}")

            # 上传后立即解析，让用户看到 Agent 在干什么
            print(f"  {DIM}📖 正在解析论文...{RESET}")
            parse_result = agent.tool("parse_document", {"file_ref": result["filename"]})
            if parse_result.get("success"):
                data = parse_result["data"]
                print(f"  {GREEN}✅ 解析完成:{RESET} 《{data['title']}》")
                print(f"     {DIM}作者: {data.get('authors', '未知')} | 共 {data['chunks']} 个段落{RESET}")
                # 注入文档上下文，让 Agent 记住这篇论文
                agent.notify_document(
                    data["file_ref"], data["title"],
                    data.get("authors", "Unknown"), data["chunks"],
                )
                print(f"  {CYAN}💡 现在可以问: \"这篇论文作者是谁\"、\"分析这篇论文\"{RESET}")
            else:
                err = parse_result.get("error", {})
                print(f"  {YELLOW}⚠️ 解析未完成:{RESET} {err.get('message', '未知错误')}")
            continue

        # 正常对话（流式输出）
        print(f"  {DIM}🤔 思考中...{RESET}")
        answer_parts = []
        for event in agent.ask_stream(user_input):
            if event["type"] == "tool":
                mark = "✅" if event.get("success") else "⚠️"
                print(f"  {DIM}{mark} 调用工具: {event['name']}{RESET}")
            elif event["type"] == "token":
                answer_parts.append(event["content"])
                print(event["content"], end="", flush=True)
        print(f"\n  {DIM}{'─' * 46}{RESET}")


if __name__ == "__main__":
    main()


