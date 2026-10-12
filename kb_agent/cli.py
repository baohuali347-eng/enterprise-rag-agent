"""命令行问答入口。

用法: uv run python -m kb_agent.cli
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from kb_agent.graph.builder import ask


def main() -> None:
    print("=" * 56)
    print("  企业知识库问答 Agent（LangChain + LangGraph + DeepSeek）")
    print("  输入 quit 退出")
    print("=" * 56)
    history: list = []
    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "退出"):
            break
        history.append(HumanMessage(content=query))
        result = ask(query, history=history)
        answer = result["answer"]
        print(f"\n助手: {answer}")
        if result["sources"]:
            print("\n来源:")
            for s in result["sources"]:
                print(f"  - {s['title']} ({s['url']})")
        history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()
