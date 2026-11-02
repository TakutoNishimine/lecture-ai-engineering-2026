"""保存した回答原文から、公式ルールに従って集計を再確認する。API/GPU不要。"""
import json
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    notebook = load(ROOT / "baseline.ipynb")
    with tempfile.TemporaryDirectory() as scratch:
        data = {"json": json, "WORK": Path(scratch)}
        exec("".join(notebook["cells"][7]["source"]), data)
    results = {}
    for label in ["baseline", "epochs10"]:
        folder = ROOT / "results" / label
        result = load(folder / "result.json")
        results[label] = result
        for phase in ["before", "after"]:
            evaluation = load(folder / f"eval_{phase}.json")
            rows = evaluation["rows"]
            assert len(rows) == 21
            assert [r["question"] for r in rows] == [e["question"] for e in data["test"]]
            for row in rows:
                try:
                    obj = json.loads(row["raw"])
                except ValueError:
                    obj = None
                fmt = (isinstance(obj, dict) and set(obj) == {"answer", "needs_review"}
                       and isinstance(obj["answer"], str) and isinstance(obj["needs_review"], bool))
                loose = obj
                if not fmt:
                    match = re.search(r"\{.*\}", row["raw"], re.S)
                    try:
                        loose = json.loads(match.group(0)) if match else None
                    except ValueError:
                        loose = None
                gold = data["gold"][row["question"]]
                content = (isinstance(loose, dict) and loose.get("needs_review") == gold["needs_review"]
                           and all(w in loose.get("answer", "") for w in gold["must_include"]))
                assert row["format_ok"] == fmt
                assert row["content_ok"] == content
                assert row["both_ok"] == (fmt and content)
            for metric, flag in [("format_rate", "format_ok"), ("content_rate", "content_ok"), ("both_rate", "both_ok")]:
                rate = sum(r[flag] for r in rows) / len(rows)
                assert abs(evaluation["summary"][metric] - rate) < 1e-12
                assert abs(result[phase][metric] - rate) < 1e-12
        assert result["training_log"][-1]["step"] == result["config"]["num_train_epochs"] * 2
        print(label, result["after"], "final_logged_loss", result["final_logged_loss"], "seconds", result["training_seconds"])
    diffs = [key for key, value in results["baseline"]["config"].items()
             if results["epochs10"]["config"][key] != value]
    assert diffs == ["num_train_epochs"], diffs
    before0 = load(ROOT / "results/baseline/eval_before.json")["rows"]
    before1 = load(ROOT / "results/epochs10/eval_before.json")["rows"]
    assert before0 == before1, "初期モデルの出力に違いがある"
    print("OK: 2条件・84回答、集計値、設定差分、初期モデル出力を検証")


if __name__ == "__main__":
    verify()
