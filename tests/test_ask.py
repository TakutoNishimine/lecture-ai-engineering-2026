"""実際のGoogle SDKを使い、HTTP応答だけを差し替える（APIキー不要）。"""

import json

import httpx
import pytest
from google import genai

from app import ask


@pytest.fixture
def requests(monkeypatch):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {
                "role": "model", "parts": [{"text": "青ですね。"}]
            }, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4},
        })

    client_class = genai.Client

    def create_client(**kwargs):
        return client_class(**kwargs, http_options={
            "client_args": {"transport": httpx.MockTransport(respond)}
        })

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setattr(ask.genai, "Client", create_client)
    return payloads


def test_chat_sends_previous_user_and_model_messages(requests, monkeypatch, capsys):
    questions = iter(["私の好きな色は？", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))
    ask.chat("私の好きな色は青です。")
    assert len(requests) == 2
    contents = requests[1]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[0]["parts"][0]["text"] == "私の好きな色は青です。"
    assert contents[1]["parts"][0]["text"] == "青ですね。"
    assert contents[2]["parts"][0]["text"] == "私の好きな色は？"
    assert "入力: 10 / 出力: 4" in capsys.readouterr().out


def test_single_questions_do_not_share_history(requests):
    assert ask.ask("最初の質問") == "青ですね。"
    ask.ask("別の質問")
    assert len(requests[1]["contents"]) == 1
    assert requests[1]["contents"][0]["parts"][0]["text"] == "別の質問"


@pytest.mark.parametrize("ending", ["", "  ", "/exit", "/quit"])
def test_exit_without_api_request(requests, monkeypatch, ending):
    monkeypatch.setattr("builtins.input", lambda _: ending)
    ask.chat()
    assert requests == []


@pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
def test_input_interruption_exits_cleanly(requests, monkeypatch, interruption):
    def interrupt(_):
        raise interruption
    monkeypatch.setattr("builtins.input", interrupt)
    ask.chat()
    assert requests == []


def test_usage_missing_and_zero(capsys):
    from google.genai import types
    ask.print_usage(types.GenerateContentResponse())
    assert "入力: 未取得 / 出力: 未取得" in capsys.readouterr().out
    ask.print_usage(types.GenerateContentResponse(usage_metadata={
        "prompt_token_count": 0, "candidates_token_count": None,
    }))
    assert "入力: 0 / 出力: 未取得" in capsys.readouterr().out


def test_main_default_question(requests, monkeypatch):
    monkeypatch.setattr("sys.argv", ["app.ask"])
    ask.main()
    assert requests[0]["contents"][0]["parts"][0]["text"] == ask.DEFAULT_QUESTION


def test_main_api_error(requests, monkeypatch, capsys):
    from google.genai import errors
    monkeypatch.setattr("sys.argv", ["app.ask", "--chat", "質問"])
    def fail(_):
        raise errors.ClientError(429, {"error": {"message": "quota", "code": 429}})
    monkeypatch.setattr(ask, "chat", fail)
    with pytest.raises(SystemExit) as exc:
        ask.main()
    assert exc.value.code == 1
    assert "レート制限に達しました" in capsys.readouterr().out
