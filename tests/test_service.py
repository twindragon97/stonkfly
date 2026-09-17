"""Container supervision regressions; no dataset, exchange or real trades."""

import json
from types import SimpleNamespace

import pytest

from stonkfly import service
from stonkfly.diagnostics import SOURCE_CHANGED, public_diagnostic


@pytest.mark.parametrize("exit_code", [0, 1, 137])
def test_paper_exit_keeps_ui_alive_without_restarting(exit_code):
    dashboard_exits = iter([None, None, None, 12])
    dashboard = SimpleNamespace(poll=lambda: next(dashboard_exits))
    worker = SimpleNamespace(poll=lambda: exit_code)
    notifications, sleeps = [], []
    result = service.supervise(dashboard, worker, notifications.append, sleeps.append)
    assert result == 12
    assert notifications == [exit_code]
    assert len(sleeps) == 3  # UI kept alive after the worker exited.


def test_error_reason_allowlist_does_not_echo_secrets(tmp_path):
    expected = public_diagnostic({"reason": SOURCE_CHANGED})
    assert expected["code"] == "source_changed"
    assert "STONKFLY_RUN" in expected["message"]
    assert "secret-token" not in json.dumps(public_diagnostic({"reason": "SDK failed: secret-token"}))
    error = tmp_path / "error.json"
    error.write_text(json.dumps({"reason": SOURCE_CHANGED}))
    assert service.worker_diagnostic(tmp_path, 0) == expected
    assert service.worker_diagnostic(tmp_path, error.stat().st_mtime + 10)["code"] == "worker_stopped"


@pytest.mark.parametrize("prepared", [False, True])
def test_main_honors_run_directory_and_retains_error_state(tmp_path, monkeypatch, prepared):
    root = tmp_path / "root"
    data = root / "data"
    run = tmp_path / "custom-paper"
    run.mkdir()
    old_checkpoint = run / "brain-0.npz"
    old_checkpoint.write_bytes(b"existing checkpoint must stay intact")
    if prepared:
        for name in ("graph.npz", "annotations.feather", "normalized/neurons.feather"):
            path = data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setenv("STONKFLY_ROOT", str(root))
    monkeypatch.setenv("STONKFLY_DATA", str(data))
    monkeypatch.setenv("STONKFLY_RUN", str(run))
    monkeypatch.setattr(service.signal, "signal", lambda *args: None)
    launched, stopped = [], []
    class Process:
        returncode = 0
        def poll(self):
            return self.returncode
    def popen(args, **kwargs):
        launched.append((args, kwargs))
        if "run" in args:
            (run / "error.json").write_text(json.dumps({"reason": SOURCE_CHANGED}))
        return Process()
    monkeypatch.setattr(service.subprocess, "Popen", popen)
    monkeypatch.setattr(service, "stop_process", stopped.append)
    def supervise(dashboard, worker, notify):
        notify(1)
        assert json.loads((run / "service.json").read_text())["code"] == "source_changed"
        return 7
    monkeypatch.setattr(service, "supervise", supervise)
    assert service.main() == 7
    assert launched[0][0][-1] == str(run)
    assert launched[1][0][-1] == ("verify" if prepared else "prepare")
    assert launched[2][0][-2:] == ["--out", str(run)]
    assert all("--live" not in args and kwargs["start_new_session"] for args, kwargs in launched)
    assert len(stopped) == 3
    assert old_checkpoint.read_bytes() == b"existing checkpoint must stay intact"


def test_failed_dataset_does_not_launch_paper_or_kill_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("STONKFLY_ROOT", str(tmp_path))
    monkeypatch.setenv("STONKFLY_RUN", str(tmp_path / "paper"))
    monkeypatch.setenv("STONKFLY_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(service.signal, "signal", lambda *args: None)
    calls = []
    class Process:
        returncode = 1
        def poll(self):
            return 1
        def wait(self):
            calls.append("dashboard wait")
            return 5
    def popen(args, **kwargs):
        calls.append(args)
        return Process()
    monkeypatch.setattr(service.subprocess, "Popen", popen)
    monkeypatch.setattr(service, "stop_process", lambda _: None)
    assert service.main() == 5
    assert len(calls) == 3 and calls[-1] == "dashboard wait"
    assert json.loads((tmp_path / "paper/service.json").read_text())["state"] == "preparation_failed"


def test_already_exited_children_are_not_signalled(monkeypatch):
    def unexpected(*args):
        pytest.fail("Must not signal a possibly reused PID")
    monkeypatch.setattr(service.os, "killpg", unexpected, raising=False)
    process = SimpleNamespace(poll=lambda: 0, terminate=unexpected, pid=123)
    service.stop_process(process)


def test_supervision_with_real_local_child_processes():
    import subprocess
    import sys
    dashboard = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1); raise SystemExit(11)"])
    worker = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    exits = []
    try:
        assert service.supervise(dashboard, worker, exits.append) == 11
        assert exits == [7]
    finally:
        for process in (worker, dashboard):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
