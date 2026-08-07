"""
CLI 交互入口
============
python main.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
from src.app import get_agent

logger = logging.getLogger(__name__)

HELP_TEXT = """
📚 论文知识库 Agent
────────────────────
直接输入问题与 Agent 对话
命令:
  /help     显示帮助
  /status   查看知识库状态
  /upload   /upload 文件路径   上传PDF
  /quit     退出
"""


def main():
    agent = get_agent()
    print(HELP_TEXT)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("👋 再见")
            break
        elif user_input == "/help":
            print(HELP_TEXT)
            continue
        elif user_input == "/status":
            stats = agent.get_stats()
            print(f"  知识库: {stats.get('total_papers', 0)} 篇论文, "
                  f"{stats.get('total_chunks', 0)} 个片段")
            print(f"  会话上传: {stats.get('uploaded_files', 0)} 篇")
            continue
        elif user_input.startswith("/upload"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("  用法: /upload 文件路径")
                continue
            path = Path(parts[1])
            if not path.exists():
                print(f"  ❌ 文件不存在: {path}")
                continue
            result = agent.upload_pdf(path.name, path.read_bytes())
            print(f"  ✅ 已上传: {result['filename']}")
            continue

        # 正常对话
        print("  🤔 思考中...")
        answer = agent.ask(user_input)
        print(f"\n{answer}")


if __name__ == "__main__":
    main()

