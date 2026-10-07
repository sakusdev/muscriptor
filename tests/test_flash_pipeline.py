from pathlib import Path

from muscriptor import flash_pipeline


def _comparison(*, f1_delta: float = 0.02, latency_ratio: float = 1.0):
    return {
        "delta": {
            "precision": 0.01,
            "recall": 0.02,
            "f1": f1_delta,
            "latency_ratio": latency_ratio,
        },
        "per_dataset": {
            "Fixture": {"delta": {"precision": 0.01, "recall": 0.02, "f1": f1_delta}}
        },
    }


def test_comparison_gate_rejects_dataset_regression():
    report = _comparison(f1_delta=-0.03)

    gate = flash_pipeline._comparison_gate(
        report,
        min_global_gain=-1.0,
        max_dataset_regression=0.02,
        max_latency_ratio=2.0,
    )

    assert not gate["passed"]
    assert any("Fixture" in failure for failure in gate["failures"])


def test_pipeline_installs_only_when_both_gates_pass(monkeypatch, tmp_path: Path):
    workdir = tmp_path / "run"
    destination = tmp_path / "installed" / "flash.pt"

    def fake_train(**kwargs):
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "model.pt").write_bytes(b"candidate")
        return {"f1": 0.8}

    def fake_compare(*args, output=None, **kwargs):
        report = _comparison()
        if output is not None:
            Path(output).write_text("{}\n", encoding="utf-8")
        return report

    monkeypatch.setattr(flash_pipeline, "run_training", fake_train)
    monkeypatch.setattr(flash_pipeline, "compare_checkpoints", fake_compare)
    monkeypatch.setattr(
        flash_pipeline,
        "evaluate_candidate",
        lambda *args, **kwargs: {"passed": True, "failures": []},
    )

    result = flash_pipeline.run_pipeline(
        workdir=workdir,
        musicnet=tmp_path,
        promote_to=destination,
        promote_benchmark_iterations=1,
    )

    assert result["passed"]
    assert destination.read_bytes() == b"candidate"
    assert (workdir / "comparison.json").is_file()
    assert (workdir / "promotion.json").is_file()
    assert (workdir / "pipeline.json").is_file()


def test_pipeline_does_not_install_failed_candidate(monkeypatch, tmp_path: Path):
    workdir = tmp_path / "run"
    destination = tmp_path / "installed.pt"

    def fake_train(**kwargs):
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "model.pt").write_bytes(b"candidate")
        return {"f1": 0.8}

    monkeypatch.setattr(flash_pipeline, "run_training", fake_train)
    monkeypatch.setattr(
        flash_pipeline,
        "compare_checkpoints",
        lambda *args, **kwargs: _comparison(f1_delta=-0.05),
    )
    monkeypatch.setattr(
        flash_pipeline,
        "evaluate_candidate",
        lambda *args, **kwargs: {"passed": True, "failures": []},
    )

    result = flash_pipeline.run_pipeline(
        workdir=workdir,
        musicnet=tmp_path,
        promote_to=destination,
        compare_max_dataset_regression=0.02,
        promote_benchmark_iterations=1,
    )

    assert not result["passed"]
    assert not destination.exists()


def test_comparison_gate_rejects_latency_ratio():
    gate = flash_pipeline._comparison_gate(
        _comparison(latency_ratio=1.5),
        min_global_gain=0.0,
        max_dataset_regression=0.02,
        max_latency_ratio=1.25,
    )

    assert not gate["passed"]
    assert any("latency ratio" in failure for failure in gate["failures"])
