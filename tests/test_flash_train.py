import json
from pathlib import Path

from muscriptor import flash_train


def _fake_run(monkeypatch, tmp_path: Path, *, fresh: bool, with_state: bool):
    dataset = tmp_path / "musicnet"
    dataset.mkdir()
    base = tmp_path / "base.pt"
    base.write_bytes(b"checkpoint")
    workdir = tmp_path / "run"
    paths = flash_train.resolve_run_paths(workdir)
    paths.root.mkdir(parents=True)
    if with_state:
        paths.state.write_bytes(b"old-state")

    captured = {}

    def fake_finetune(base_checkpoint, output, **kwargs):
        captured["base"] = Path(base_checkpoint)
        captured["output"] = Path(output)
        captured.update(kwargs)
        Path(output).write_bytes(b"model")
        Path(kwargs["metrics_path"]).write_text('{"f1": 0.5}\n', encoding="utf-8")
        Path(kwargs["state_output"]).write_bytes(b"new-state")
        return {"f1": 0.5, "best_step": 2}

    monkeypatch.setattr(flash_train, "finetune_multidataset", fake_finetune)
    result = flash_train.run_training(
        workdir=workdir,
        musicnet=dataset,
        base_checkpoint=base,
        steps=3,
        fresh=fresh,
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    return paths, captured, result, manifest


def test_resolve_run_paths_uses_stable_names(tmp_path: Path):
    paths = flash_train.resolve_run_paths(tmp_path / "job")

    assert paths.model.name == "model.pt"
    assert paths.metrics.name == "metrics.json"
    assert paths.state.name == "training-state.pt"
    assert paths.manifest.name == "run.json"


def test_runner_auto_resumes_existing_state(monkeypatch, tmp_path: Path):
    paths, captured, result, manifest = _fake_run(
        monkeypatch, tmp_path, fresh=False, with_state=True
    )

    assert captured["resume_state"] == paths.state
    assert captured["state_output"] == paths.state
    assert captured["steps"] == 3
    assert result["f1"] == 0.5
    assert manifest["status"] == "completed"
    assert manifest["resume_state"] == str(paths.state)


def test_fresh_run_backs_up_old_state(monkeypatch, tmp_path: Path):
    paths, captured, _, manifest = _fake_run(
        monkeypatch, tmp_path, fresh=True, with_state=True
    )

    assert captured["resume_state"] is None
    backup = Path(manifest["backed_up_state"])
    assert backup.is_file()
    assert backup.read_bytes() == b"old-state"
    assert paths.state.read_bytes() == b"new-state"


def test_runner_requires_at_least_one_dataset(tmp_path: Path):
    try:
        flash_train.run_training(workdir=tmp_path / "job")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected dataset validation error")
