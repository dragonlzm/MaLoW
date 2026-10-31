"""The experiment supervisor must never stop unrelated GPU processes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "text_summary_supervisor", Path(__file__).resolve().parents[1] / "scripts/run_text_summary_evaluation.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def test_refuses_unrelated_gpu_jobs_before_signalling(monkeypatch):
    monkeypatch.setattr(supervisor, "burner_pids", lambda binary: [10])
    monkeypatch.setattr(supervisor, "gpu_processes", lambda: {10, 20})
    signals = []
    monkeypatch.setattr(supervisor.os, "kill", lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match="unrelated"):
        supervisor.stop_burner(Path("/burner"))
    assert signals == []


@pytest.mark.parametrize("rollout_step_seconds", [10, 30])
def test_commands_only_differ_by_summary_flag_and_output(rollout_step_seconds):
    for benchmark in ("ovo", "streaming"):
        control = supervisor.command(benchmark, "out", 8, False, False, rollout_step_seconds)
        summary = supervisor.command(benchmark, "out", 8, True, False, rollout_step_seconds)
        assert summary == [*control, "--text-summary"]
        assert "--limit" not in summary
        assert summary[summary.index("--rollout-step-seconds") + 1] == str(rollout_step_seconds)


def test_actual_question_count_is_not_fixed_benchmark_weight(tmp_path):
    value = {key: {"accuracy_percent": 50, "num_examples": 1}
             for key in ("real", "omni", "sqa", "proactive")}
    value["weighted_average"] = {"accuracy_percent": 50, "total_questions": 4500}
    (tmp_path / "summary.json").write_text(json.dumps(value))
    assert supervisor.metrics(tmp_path, "streaming")["questions"] == 4


def test_memory_stats_counts_logged_normalizations_and_legacy_rows(tmp_path):
    rows = [{"kind": "summary", "seconds": 1., "tokens": 50},
            {"kind": "summary", "seconds": 2., "tokens": 60, "normalizations": []},
            {"kind": "summary", "seconds": 3., "tokens": 70,
             "normalizations": ["trailing_quote", "object_entry:events[0]"]}]
    (tmp_path / "text_summary_rank_00000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    stats = supervisor.memory_stats(tmp_path)
    assert stats["summaries"] == 3 and stats["normalized_summaries"] == 1
    assert stats["summary_tokens_max"] == 70


def test_ovo_smoke_preserves_all_forward_rounds_and_source(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    rounds = [json.dumps({"realtime": t, "type": 0}) for t in (8, 14, 20)]
    originals = {}
    for split in ("backward", "realtime", "forward"):
        rows = ([{"id": 1, "test_info": [json.dumps({"realtime": 30})]},
                 {"id": 2, "test_info": rounds}] if split == "forward" else
                [{"id": 1, "realtime": 5.}, {"id": 2, "realtime": 20.}, {"id": 3, "realtime": 100.}])
        source = tmp_path / f"data/ovo-annotations/{split}/test-00000-of-00001.parquet"
        source.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), source)
        originals[source] = source.read_bytes()
    output = supervisor.prepare_ovo_smoke_annotations(tmp_path)
    chosen = pq.read_table(output / "forward/test-00000-of-00001.parquet").to_pylist()
    assert chosen == [{"id": 2, "test_info": rounds}]
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("fail", [False, True])
def test_burner_starts_after_jobs_or_failure(monkeypatch, tmp_path, fail):
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "EVALUATIONS", tmp_path / "evaluations")
    events = []
    monkeypatch.setattr(supervisor, "gpu_processes", lambda: set(range(8)) if "burner started" in events else set())
    monkeypatch.setattr(supervisor, "burner_pids", lambda binary: [])
    monkeypatch.setattr(supervisor, "stop_burner", lambda binary: [10])
    monkeypatch.setattr(supervisor, "prepare_smoke_annotations", lambda root: root / "annotations")
    monkeypatch.setattr(supervisor, "prepare_ovo_smoke_annotations", lambda root: root / "ovo_annotations")
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(supervisor.signal, "signal", lambda *args: None)
    monkeypatch.setattr(supervisor, "metrics", lambda *args: {"overall": 50, "questions": 1})
    monkeypatch.setattr(supervisor, "memory_stats", lambda output: {"summaries": 1, "answers": 1})
    burner = tmp_path / "gpu_burn"
    burner.write_text("fake")
    monkeypatch.setattr("sys.argv", ["run", "--run-id", "smoke", "--smoke-only", "--burner", str(burner),
                                     "--rollout-step-seconds", "30"])
    class Process:
        pid = 100
        def wait(self, **kwargs):
            events.append("worker exited")
            return 1 if fail else 0
        def poll(self):
            return None
    def popen(command, **kwargs):
        events.append("burner started" if command[0] == str(burner) else "worker started")
        return Process()
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    if fail:
        with pytest.raises(RuntimeError, match="exited with code"):
            supervisor.main()
    else:
        supervisor.main()
    assert events[-2:] == ["worker exited", "burner started"]
    assert events.count("worker started") == (1 if fail else 4)
    state = json.loads((tmp_path / "outputs/text_sum_evaluation/smoke/status.json").read_text())
    assert [(job["benchmark"], job["summary"]) for job in state["jobs"]] == (
        [("ovo", True)] if fail else
        [("ovo", True), ("streaming", True), ("ovo", False), ("streaming", False)])
    for job in state["jobs"]:
        assert "-30s-" in job["name"]
        assert job["command"][job["command"].index("--rollout-step-seconds") + 1] == "30"


def test_resume_skips_completed_jobs_and_retries_failed_job(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    monkeypatch.setattr(supervisor, 'EVALUATIONS', tmp_path / 'evaluations')
    burner = tmp_path / 'burner'; burner.touch()
    monkeypatch.setattr(supervisor, 'gpu_processes', lambda: set())
    monkeypatch.setattr(supervisor, 'burner_pids', lambda b: [])
    monkeypatch.setattr(supervisor, 'stop_burner', lambda b: [])
    monkeypatch.setattr(supervisor.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(supervisor, 'prepare_smoke_annotations', lambda root, **kw: root / 'annotations')
    monkeypatch.setattr(supervisor, 'prepare_ovo_smoke_annotations', lambda root: root / 'ovo')
    monkeypatch.setattr(supervisor, 'validate_completed_job', lambda *a: None)
    monkeypatch.setattr(supervisor, 'metrics', lambda *a: {'questions': 1, 'overall': 50})
    monkeypatch.setattr(supervisor, 'memory_stats', lambda *a: {'summaries': 1, 'answers': 1})
    launched = []
    class Process:
        pid = 987
        def __init__(self, rc): self.rc = rc
        def wait(self, **kw): return self.rc
        def poll(self): return self.rc
    def popen(cmd, **kw):
        if cmd[0] == str(burner): return Process(0)
        launched.append(cmd)
        return Process(1 if len(launched) == 2 else 0)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', popen)
    monkeypatch.setattr('sys.argv', ['run', '--run-id', 'recover', '--smoke-only', '--burner', str(burner),
                                     '--checkpoint-progress', '--streaming-group-by-video', '--rollout-step-seconds', '30'])
    with pytest.raises(RuntimeError, match='exited'):
        supervisor.main()
    monkeypatch.setattr('sys.argv', ['run', '--run-id', 'recover', '--resume'])
    supervisor.main()
    assert [cmd[cmd.index('--benchmark') + 1] for cmd in launched] == ['ovo', 'streaming', 'streaming', 'ovo', 'streaming']
    state = json.loads((tmp_path / 'outputs/text_sum_evaluation/recover/status.json').read_text())
    assert state['status'] == 'complete' and len(state['attempts']) == 2
    assert len(state['jobs']) == 4
    assert len(state['jobs'][1]['attempt_history']) == 1
    assert '--streaming-group-by-video' in state['jobs'][1]['command']


def test_replay_queue_runs_lora_then_base_and_keeps_sources_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    monkeypatch.setattr(supervisor, 'EVALUATIONS', tmp_path / 'evaluations')
    burner=tmp_path/'burner';burner.touch()
    bundles=tmp_path/'bundles'
    for benchmark in ('ovo','streaming'):
        path=bundles/benchmark;path.mkdir(parents=True);(path/'manifest.json').write_text('{}')
    monkeypatch.setattr(supervisor, 'gpu_processes', lambda:set())
    monkeypatch.setattr(supervisor, 'burner_pids', lambda b:[])
    monkeypatch.setattr(supervisor, 'stop_burner', lambda b:[])
    monkeypatch.setattr(supervisor.signal, 'signal', lambda *a:None)
    monkeypatch.setattr(supervisor, 'validate_completed_job', lambda *a:None)
    monkeypatch.setattr(supervisor, 'metrics', lambda p,b:{'overall':50,'questions':3035 if b=='ovo' else 4500})
    monkeypatch.setattr(supervisor, 'memory_stats', lambda *a:{'summaries':1,'answers':1})
    class Process:
        pid=987
        def wait(self,**kw):return 0
        def poll(self):return 0
    monkeypatch.setattr(supervisor.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr('sys.argv',['run','--run-id','replay','--skip-smoke','--burner',str(burner),
                                   '--text-cache-root',str(bundles),'--rollout-step-seconds','30'])
    supervisor.main()
    root=tmp_path/'outputs/text_sum_evaluation/replay'
    state=json.loads((root/'status.json').read_text())
    assert [(j['benchmark'],j['base_only']) for j in state['jobs']]==[('ovo',False),('streaming',False),('ovo',True),('streaming',True)]
    for j in state['jobs']:
        assert j['summary'] and '--text-summary' in j['command']
        assert ('--text-cache-base-only' in j['command'])==j['base_only']
        assert j['command'][j['command'].index('--text-cache-miss')+1]==('generate' if j['base_only'] else 'error')
    assert 'base_replay' in json.loads((root/'comparison.json').read_text())['ovo']
