from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_openai_finetune import read_job_id, retrieve_job_status
from evaluate_finetuned_model import evaluate_examples, parse_model_action
from start_openai_finetune import start_finetune


def make_example(action: str = "reveal", x: int = 1, y: int = 2) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Board:\n. . .\n. 1 .\n. . .\n\nChoose the next move."},
            {"role": "assistant", "content": json.dumps({"action": action, "x": x, "y": y})},
        ]
    }


class FakeFiles:
    def __init__(self) -> None:
        self.created: list[tuple[str, str]] = []

    def create(self, file, purpose: str):
        self.created.append((Path(file.name).name, purpose))
        return SimpleNamespace(id=f"file-{len(self.created)}", purpose=purpose, filename=Path(file.name).name)


class FakeJobs:
    def __init__(self) -> None:
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id="ftjob-123", status="validating_files", fine_tuned_model=None)

    def retrieve(self, job_id: str):
        return SimpleNamespace(id=job_id, status="succeeded", fine_tuned_model="ft:gpt-test", error=None)


class FakeClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.fine_tuning = SimpleNamespace(jobs=FakeJobs())


class FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class OpenAIWorkflowScriptTests(unittest.TestCase):
    def test_start_finetune_uploads_files_creates_job_and_saves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.jsonl"
            val_path = tmp_path / "val.jsonl"
            output_path = tmp_path / "job.json"
            train_path.write_text(json.dumps(make_example()) + "\n", encoding="utf-8")
            val_path.write_text(json.dumps(make_example("flag", 0, 0)) + "\n", encoding="utf-8")

            client = FakeClient()
            metadata = start_finetune(
                client=client,
                train_path=train_path,
                val_path=val_path,
                model="gpt-test",
                output_path=output_path,
                suffix="minesweeper",
                seed=42,
            )

            self.assertEqual(client.files.created, [("train.jsonl", "fine-tune"), ("val.jsonl", "fine-tune")])
            self.assertEqual(client.fine_tuning.jobs.create_kwargs["training_file"], "file-1")
            self.assertEqual(client.fine_tuning.jobs.create_kwargs["validation_file"], "file-2")
            self.assertEqual(client.fine_tuning.jobs.create_kwargs["model"], "gpt-test")
            self.assertEqual(client.fine_tuning.jobs.create_kwargs["suffix"], "minesweeper")
            self.assertEqual(client.fine_tuning.jobs.create_kwargs["seed"], 42)
            self.assertEqual(metadata["fine_tuning_job_id"], "ftjob-123")
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))["training_file_id"], "file-1")

    def test_check_finetune_reads_job_id_and_retrieves_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "job.json"
            metadata_path.write_text(json.dumps({"fine_tuning_job_id": "ftjob-123"}), encoding="utf-8")

            client = FakeClient()
            status = retrieve_job_status(client, read_job_id(metadata_path))

            self.assertEqual(status["id"], "ftjob-123")
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(status["fine_tuned_model"], "ft:gpt-test")
            self.assertIsNone(status["error"])

    def test_parse_model_action_accepts_json_fences(self) -> None:
        self.assertEqual(
            parse_model_action('```json\n{"action":"flag","x":1,"y":2}\n```'),
            {"action": "flag", "x": 1, "y": 2},
        )

    def test_evaluate_examples_reports_match_metrics_and_invalid_json(self) -> None:
        completions = FakeChatCompletions(
            [
                '{"action":"reveal","x":1,"y":2}',
                '{"action":"flag","x":0,"y":0}',
                'not json',
            ]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        examples = [
            make_example("reveal", 1, 2),
            make_example("reveal", 0, 0),
            make_example("flag", 2, 2),
        ]

        result = evaluate_examples(client, model="ft:gpt-test", examples=examples)

        self.assertEqual(result.total_examples, 3)
        self.assertEqual(result.exact_matches, 1)
        self.assertEqual(result.action_matches, 1)
        self.assertEqual(result.coordinate_matches, 2)
        self.assertEqual(result.invalid_json, 1)
        self.assertEqual(completions.calls[0]["model"], "ft:gpt-test")
        self.assertEqual(completions.calls[0]["max_completion_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
