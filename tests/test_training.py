"""Training orchestration tests use an explicit fake controller, never claim neural learning."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stonkfly.config import D, Settings
from stonkfly.historical import Candle, read_csv, validate
from stonkfly.training import TrainingConfig, replay


def candles(prices=(100, 110, 120, 90)):
    return [Candle(1609459200 + i * 86400, str(p), str(p + 10), str(p - 10), str(p))
            for i, p in enumerate(prices)]


class FakeController:
    def __init__(self, sides=("BUY", "SELL", "HOLD", "HOLD")):
        self.sides = iter(sides)
        self.signals = []
        self.frames = []
        self.resets = []
        self.brain = SimpleNamespace(reset=lambda **kwargs: self.resets.append(kwargs),
                                     memory=lambda: {"changed_edges": 0})

    def observe(self, frame, kind):
        self.signals.append(kind)
        self.frames.append(frame.copy())
        return {"side": next(self.sides), "memory": {"changed_edges": 0}}


@pytest.mark.parametrize("changes", [
    {"evaluation_year": 2020}, {"start_year": 2023}, {"epochs": 0},
    {"epochs": True}, {"neural_ms": 1}, {"neural_ms": float("nan")},
    {"interval": "1m"}, {"evaluation_year": 9999}, {"start_year": 2020.0},
])
def test_reject_invalid_protocol(changes):
    args = dict(start_year=2020, end_year=2022, evaluation_year=2023)
    args.update(changes)
    with pytest.raises(ValueError):
        TrainingConfig(**args)


def test_history_requires_complete_ordered_coverage(tmp_path):
    rows = candles()
    start, end = rows[0].timestamp, rows[-1].timestamp + 86400
    assert validate(rows, start, end, 86400) == rows
    for bad in [rows[:-1], rows[::-1], [rows[0], rows[0], *rows[2:]]]:
        with pytest.raises(ValueError):
            validate(bad, start, end, 86400)
    path = tmp_path / "history.csv"
    path.write_text("timestamp,open,high,low,close\n" + "\n".join(
        f"{r.timestamp},{r.open},{r.high},{r.low},{r.close}" for r in rows))
    assert read_csv(path, start, end, 86400) == rows


@pytest.mark.parametrize("prices", [("NaN", "100", "90", "99"), ("99", "100", "0", "99"), ("101", "100", "90", "99")])
def test_bad_ohlc(prices):
    with pytest.raises(ValueError):
        Candle(1, *prices)


def test_replay_next_open_and_delayed_fee_reward(tmp_path):
    controller = FakeController()
    out = tmp_path / "episode"
    result = replay(candles(), 86400, controller, Settings(), out, tmp_path / "STOP", lambda _: None)
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    fill = events[0]["execution"]
    assert D(fill["quote"]) / D(fill["base"]) == D(110) * D("1.0005") ** 2
    assert controller.signals[0] == "none"
    assert controller.signals[1] == "aversive"  # Same open/close; only costs.
    assert result["fills"] == 2
    assert result["decisions"] == {"BUY": 1, "SELL": 1, "HOLD": 2}
    assert controller.resets == [{"keep_memory": True}]
    import sqlite3
    with sqlite3.connect(out / "ledger.sqlite") as db:
        orders = db.execute("SELECT status,created,plan FROM orders ORDER BY created").fetchall()
    assert all(row[0] == "SETTLED" for row in orders)
    assert orders[0][1] == candles()[1].timestamp
    assert json.loads(orders[0][2])["side"] == "BUY"


def test_future_changes_do_not_change_first_observation(tmp_path):
    controllers = [FakeController(), FakeController()]
    histories = [candles(), candles((100, 500, 300, 200))]
    for i, (controller, history) in enumerate(zip(controllers, histories)):
        replay(history, 86400, controller, Settings(), tmp_path / str(i), tmp_path / "STOP", lambda _: None)
    assert np.array_equal(controllers[0].frames[0], controllers[1].frames[0])
    assert controllers[0].signals[0] == controllers[1].signals[0] == "none"


def test_no_position_sell_is_veto_not_replacement(tmp_path):
    result = replay(candles(), 86400, FakeController(("SELL",) * 4), Settings(),
                    tmp_path / "episode", tmp_path / "STOP", lambda _: None, True)
    assert result["fills"] == 0
    assert result["equity"] == "100"
    assert result["vetoes"] == 3


def test_stop_and_frozen_evaluation(tmp_path):
    controller = FakeController()
    replay(candles(), 86400, controller, Settings(learning=False), tmp_path / "frozen",
           tmp_path / "STOP", lambda _: None, True)
    assert controller.brain.weights_frozen is True
    (tmp_path / "STOP").touch()
    with pytest.raises(InterruptedError):
        replay(candles(), 86400, FakeController(), Settings(), tmp_path / "stopped",
               tmp_path / "STOP", lambda _: None)


def test_memory_transfer_validates_before_mutation(tmp_path):
    from stonkfly.neural.controller import FlyController
    controller = FlyController.__new__(FlyController)
    controller.memory_signature = lambda: {"format": "test-only"}
    resets = []
    controller.brain = SimpleNamespace(memory_u=np.zeros(2), memory_w=np.zeros(2),
        baseline_plastic=np.array([2., 3.]), weight=np.array([2., 99., 3.]),
        circuit={"edges": np.array([0, 2])}, reset=lambda: resets.append(True))
    controller.brain.memory_w[:] = -.1
    path = tmp_path / "memory.npz"
    controller.save_memory(path)
    controller.brain.memory_w[:] = 0
    controller.load_memory(path)
    np.testing.assert_allclose(controller.brain.weight, [1.8, 99., 2.7])
    assert len(resets) == 1
    np.savez(path, metadata=json.dumps({"format": "test-only"}), memory_u=np.zeros(2), memory_w=np.array([np.nan, 0.]))
    with pytest.raises(ValueError):
        controller.load_memory(path)
    assert len(resets) == 1


def test_jobs_reject_command_injection_and_simultaneous_start(tmp_path, monkeypatch):
    from dashboard.training_jobs import TrainingJobs
    jobs = TrainingJobs(Path.cwd(), tmp_path / "jobs", tmp_path / "cache")
    with pytest.raises((TypeError, ValueError)):
        jobs.start({"start_year": 2020, "end_year": 2021, "evaluation_year": 2022, "out": "elsewhere"})
    calls = []
    class Process:
        def __init__(self, args, **kwargs):
            calls.append(args)
        def poll(self):
            return None
    monkeypatch.setattr("dashboard.training_jobs.subprocess.Popen", Process)
    args = {"start_year": 2020, "end_year": 2021, "evaluation_year": 2022}
    jobs.start(args)
    with pytest.raises(ValueError, match="activo"):
        jobs.start(args)
    assert len(calls) == 1 and "--live" not in calls[0]
    jobs.stop()
    assert (jobs.job / "STOP").exists()


def test_download_pagination_cache_and_integrity(tmp_path, monkeypatch):
    from stonkfly import historical
    from urllib.parse import parse_qs, urlsplit
    from datetime import datetime
    requests = []
    class Response:
        def __init__(self, rows):
            self.rows = rows
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def read(self):
            return json.dumps(self.rows).encode()
    def request(req, timeout):
        query = parse_qs(urlsplit(req.full_url).query)
        start = int(datetime.fromisoformat(query["start"][0]).timestamp())
        end = int(datetime.fromisoformat(query["end"][0]).timestamp())
        requests.append((start, end))
        # API may return a preceding/outside bar and returns descending order.
        return Response([[t, 90, 110, 100, 101, 1] for t in range(end, start - 86401, -86400)])
    monkeypatch.setattr(historical, "urlopen", request)
    monkeypatch.setattr(historical.time, "sleep", lambda _: None)
    first = historical.fetch_year(2020, 86400, tmp_path)
    assert len(first) == 366 and len(requests) == 2
    assert historical.fetch_year(2020, 86400, tmp_path) == first and len(requests) == 2
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["candles"][0][1] = "99"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        historical.fetch_year(2020, 86400, tmp_path)


def test_training_orchestration_freezes_evaluation_and_saves_final_memory(tmp_path, monkeypatch):
    from stonkfly import training
    config = TrainingConfig(2020, 2021, 2022, epochs=2)
    monkeypatch.setattr("stonkfly.data.verify", lambda: {"test_double": True})
    monkeypatch.setattr(training, "fetch_year", lambda *args: candles())
    controller = FakeController()
    controller.s = Settings()
    saves = []
    def save_memory(path):
        saves.append(path)
        path.write_bytes(b"explicit orchestration test double, not neural memory")
    controller.save_memory = save_memory
    monkeypatch.setattr("stonkfly.neural.controller.FlyController", lambda _: controller)
    phases = []
    def fake_replay(rows, seconds, c, settings, out, stop, progress, frozen=False):
        phases.append((out.name, frozen, settings.learning, len(rows)))
        progress({"step": len(rows)})
        return {"test_double": True}
    monkeypatch.setattr(training, "replay", fake_replay)
    out = tmp_path / "training"
    training.train(config, out, tmp_path / "cache")
    assert phases == [("epoch-1", False, True, 8), ("epoch-2", False, True, 8),
                      ("evaluation-trained", True, False, 4), ("evaluation-untrained", True, False, 4)]
    assert len(saves) == 2  # No evaluation writes to the final trained memory.
    assert controller.resets == [{"keep_memory": False}]
    assert json.loads((out / "status.json").read_text())["state"] == "completed"
    assert json.loads((out / "report.json").read_text())["learning_validated"] is False
    with pytest.raises(ValueError, match="new training directory"):
        training.train(config, out, tmp_path / "cache")


def test_training_failure_is_visible(tmp_path, monkeypatch):
    from stonkfly.training import train
    def fail():
        raise FileNotFoundError("Prepare dataset first")
    monkeypatch.setattr("stonkfly.data.verify", fail)
    out = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        train(TrainingConfig(2020, 2021, 2022), out, tmp_path / "cache")
    assert json.loads((out / "status.json").read_text())["state"] == "failed"


def test_http_training_rejects_cross_origin_and_invalid_config(tmp_path, monkeypatch):
    import sys
    import threading
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    monkeypatch.setattr(sys, "argv", ["dashboard/server.py"])
    from dashboard import server
    from dashboard.training_jobs import TrainingJobs
    monkeypatch.setattr(server, "JOBS", TrainingJobs(Path.cwd(), tmp_path / "runs", tmp_path / "cache"))
    http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        connection = HTTPConnection("127.0.0.1", http.server_port)
        connection.request("GET", "/training.json")
        response = connection.getresponse()
        token = json.loads(response.read())["token"]
        for origin, supplied, expected in [("http://evil.invalid", token, 403),
                (f"http://127.0.0.1:{http.server_port}", "wrong", 403),
                (f"http://127.0.0.1:{http.server_port}", token, 400)]:
            connection.request("POST", "/training/start", "{}", {"Origin": origin,
                "Content-Type": "application/json", "X-Training-Token": supplied})
            response = connection.getresponse()
            assert response.status == expected
            response.read()
        assert server.JOBS.job is None
        connection.close()
    finally:
        http.shutdown()
        http.server_close()
        worker.join()
