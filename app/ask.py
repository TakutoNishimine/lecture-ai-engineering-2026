"""第1回: 素のLLMへの質問と、会話履歴を保持する対話。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --chat "私の好きな色は青です。"
    uv run python -m app.ask --chat   # 最初の質問から入力する

--chat は同じチャットを再利用し、質問と回答の履歴を次のリクエストに渡す。
履歴は実行中のメモリにだけ保持し、終了すると消える。
講座の用語集を検索する機能はまだない（第3回で追加する）。
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response) -> None:
    """返却されたトークン数を表示する。未提供の値は推測しない。"""
    usage = response.usage_metadata
    if usage is None:
        print("[tokens] 入力: 未取得 / 出力: 未取得")
        return
    prompt = usage.prompt_token_count
    output = usage.candidates_token_count
    print(
        f"[tokens] 入力: {prompt if prompt is not None else '未取得'} / "
        f"出力: {output if output is not None else '未取得'}"
    )


def ask(question: str) -> str:
    """従来どおり、履歴なしで1回だけ質問する。"""
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
        print_usage(response)
        return response.text or "（テキストの回答がありません）"


def chat(first_question: str | None = None) -> None:
    """同じチャットに続けて質問し、SDKで質問・回答の履歴を保持する。"""
    print("続けて質問できます。空行、/exit、/quit、Ctrl+Dで終了します。")
    with genai.Client(api_key=get_api_key()) as client:
        session = client.chats.create(model=MODEL)
        question = first_question
        while True:
            if question is None:
                try:
                    question = input("[質問] ")
                except (EOFError, KeyboardInterrupt):
                    print("\n会話を終了しました。")
                    return
            else:
                print(f"[質問] {question}")
            question = question.strip()
            if not question or question.lower() in {"/exit", "/quit"}:
                print("会話を終了しました。")
                return
            response = session.send_message(question)
            print(response.text or "（テキストの回答がありません）")
            print_usage(response)
            print()
            question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat(args.question)
        else:
            question = args.question if args.question is not None else DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
