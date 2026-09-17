"""Accelerated chronological replay using the complete, unchanged neural graph."""

import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .broker import PaperBroker
from .config import D, Settings
from .historical import INTERVALS, fetch_year, read_csv, year_start
from .ledger import Ledger
from .market import Quote
from .reinforcement import reinforcement
from .risk import Guard, Veto


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    start_year: int
    end_year: int
    evaluation_year: int
    interval: str = "1d"
    epochs: int = 1
    neural_ms: float = 500

    def __post_init__(self):
        current = datetime.now(timezone.utc).year
        if any(type(v) is not int for v in (self.start_year, self.end_year, self.evaluation_year, self.epochs)):
            raise ValueError("Years and epochs must be integers")
        if not 2009 <= self.start_year <= self.end_year < self.evaluation_year < current:
            raise ValueError("Use completed years: training must precede evaluation")
        if self.interval not in INTERVALS or not 1 <= self.epochs <= 20:
            raise ValueError("Choose 1h, 6h or 1d and 1–20 epochs")
        if isinstance(self.neural_ms, bool) or self.neural_ms not in (100, 250, 500):
            raise ValueError("Neural duration must be 100, 250 or 500 ms")


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def quote(price, timestamp):
    # BTC-USD is explicitly a price proxy for a simulated USDC account.
    # Synthetic spread = 10 bps total, execution impact = 5 bps per side.
    p = D(price)
    return Quote("BTC-USDC", p * D(".9995"), p * D("1.0005"), timestamp,
                 D(".00000001"), D(".01"), D(".01"), D("1"), D(".00000001"))


def replay(candles, seconds, controller, settings, out, stop, progress, frozen=False):
    """Observe closed bar t, execute only at bar t+1 open; reward on next mark."""
    from .display import market_frame

    out.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(out / "ledger.sqlite", settings, "paper")
    broker = PaperBroker(settings, ledger)
    guard = Guard(settings, ledger, stop)
    controller.brain.reset(keep_memory=True)
    controller.brain.weights_frozen = frozen
    history, curve = [], []
    peak, max_drawdown = D(settings.capital), D(0)
    fills, vetoes, rewards, aversive = 0, 0, 0, 0
    decisions = {"BUY": 0, "SELL": 0, "HOLD": 0}
    anchor = D(settings.capital)
    started = time.monotonic()
    try:
        with (out / "events.jsonl").open("w", encoding="utf-8") as events:
            for i, candle in enumerate(candles):
                if stop.exists():
                    raise InterruptedError("Training stopped; last completed epoch memory retained")
                now = candle.timestamp + seconds
                q = quote(candle.close, now)
                quotes = {q.product: q}
                equity = ledger.equity(quotes)
                kind, delta = reinforcement(equity, anchor, settings.reward_deadband)
                rewards += kind == "reward"
                aversive += kind == "aversive"
                anchor = equity  # Before the next fill, so its fee is rewarded next bar.
                peak = max(peak, equity)
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
                history.append(float(D(candle.close)))
                history = history[-120:]
                neural = controller.observe(market_frame(q.product, history, q.bid, q.ask), kind)
                side = neural["side"]
                decisions[side] += 1
                execution = {"status": "HOLD" if side == "HOLD" else "END_OF_EPISODE"}
                try:
                    guard.check(quotes, now)
                    if side != "HOLD" and i + 1 < len(candles):
                        nxt = candles[i + 1]
                        execution_quote = quote(nxt.open, nxt.timestamp)
                        # Adverse impact is declared and deterministic; never selects actions.
                        execution_quote = dataclasses.replace(execution_quote,
                            bid=execution_quote.bid * D(".9995"),
                            ask=execution_quote.ask * D("1.0005"))
                        plan = guard.plan(q.product, side, {q.product: execution_quote}, nxt.timestamp)
                        plan = ledger.reserve(plan, nxt.timestamp)  # Durable before fill.
                        execution = broker.execute(plan, lambda p: guard.before_submit(p, nxt.timestamp))
                        fills += 1
                except Veto as error:
                    execution = {"status": "VETO", "reason": str(error)}
                    vetoes += 1
                row = {"step": i + 1, "timestamp": now, "equity": str(equity),
                       "delta": str(delta), "neural": neural, "execution": execution}
                events.write(json.dumps(row, allow_nan=False) + "\n")
                # Small status files every second, no full graph checkpoint per observation.
                if i % max(1, len(candles) // 250) == 0 or i == len(candles) - 1:
                    curve.append({"timestamp": now, "equity": str(equity)})
                progress({"step": i + 1, "steps": len(candles), "market_time": now,
                          "equity": str(equity), "fills": fills, "vetoes": vetoes,
                          "side": side, "memory": neural["memory"], "curve": curve[-300:]})
        # Comparable passive benchmark: first actionable open, same spread/impact/fees.
        first = quote(candles[1].open, candles[1].timestamp)
        passive_size = D(settings.capital) / (first.ask * D("1.0005") * (1 + D(settings.paper_fee)))
        buy_hold = passive_size * quote(candles[-1].close, candles[-1].timestamp + seconds).bid
        return {"equity": str(equity), "return_pct": float((equity / D(settings.capital) - 1) * 100),
                "max_drawdown_pct": float(max_drawdown * 100), "fills": fills, "vetoes": vetoes,
                "decisions": decisions, "reward_pulses": rewards, "aversive_pulses": aversive,
                "buy_hold_equity": str(buy_hold), "cash_equity": settings.capital,
                "halted": ledger.get("halted"), "steps": len(candles),
                "elapsed_seconds": time.monotonic() - started, "curve": curve,
                "memory": controller.brain.memory()}
    finally:
        ledger.close()


def train(config, out, cache, csv_path=None):
    from .locking import exclusive

    out, cache = Path(out), Path(cache)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists():
        raise ValueError("Use a new training directory; partial runs are never silently resumed")
    status = {"state": "preparing", "config": dataclasses.asdict(config), "run": out.name,
              "started_at": time.time(), "learning_validated": False}
    status_path = out / "status.json"
    last_write = 0.0

    def update(values, force=False):
        nonlocal last_write
        status.update(values)
        if force or time.monotonic() - last_write >= 1:
            status["updated_at"] = time.time()
            atomic_json(status_path, status)
            last_write = time.monotonic()

    stop = out / "STOP"
    update({}, True)
    try:
        with exclusive(out.parent / "training-worker.lock"):
            from .data import verify
            update({"state": "loading"}, True)
            dataset = verify()
            seconds = INTERVALS[config.interval]
            update({"state": "downloading"}, True)
            years = list(range(config.start_year, config.end_year + 1)) + [config.evaluation_year]
            data = {}
            for year in years:
                if stop.exists():
                    raise InterruptedError("Training stopped")
                data[year] = (read_csv(csv_path, year_start(year), year_start(year + 1), seconds)
                              if csv_path else fetch_year(year, seconds, cache, update, stop.exists))
            training = [c for year in range(config.start_year, config.end_year + 1) for c in data[year]]
            evaluation = data[config.evaluation_year]
            if stop.exists():
                raise InterruptedError("Training stopped")
            from .neural.controller import FlyController
            update({"state": "loading"}, True)
            settings = Settings(neural_ms=config.neural_ms, pulse_ms=min(200, config.neural_ms))
            controller = FlyController(settings)
            provenance = {"config": dataclasses.asdict(config), "settings": dataclasses.asdict(settings),
                          "dataset": dataset, "source": "local CSV" if csv_path else "Coinbase Exchange BTC-USD",
                          "price_proxy": "BTC-USD prices used as BTC-USDC; no historical USDC basis modeled",
                          "data_sha256": hashlib.sha256(json.dumps({y: [dataclasses.asdict(c) for c in rows]
                                                 for y, rows in data.items()}, sort_keys=True).encode()).hexdigest(),
                          "source_sha256": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in Path(__file__).parent.rglob("*") if p.suffix in (".py", ".cpp")},
                          "fill": "next candle open; 10 bps spread plus 5 bps adverse impact/side; configured fee",
                          "timing": "One configured neural window per candle; not real-time physiology",
                          "selection": "Final epoch, never best-profit selection", "learning_validated": False}
            atomic_json(out / "provenance.json", provenance)
            total = len(training) * config.epochs + 2 * len(evaluation)
            completed = 0
            compute_start = time.monotonic()

            def progress(values):
                done = completed + values["step"]
                elapsed = time.monotonic() - compute_start
                rate = done / max(elapsed, 0.001)
                update({**values, "completed_steps": done, "total_steps": total,
                        "progress": done / total, "steps_per_second": rate,
                        "eta_seconds": (total - done) / rate})

            results = []
            for epoch in range(1, config.epochs + 1):
                update({"state": "training", "epoch": epoch}, True)
                results.append(replay(training, seconds, controller, settings,
                                      out / f"epoch-{epoch}", stop, progress))
                completed += len(training)
                controller.save_memory(out / "memory.npz")
                update({"epochs": results}, True)
            memory_sha = hashlib.sha256((out / "memory.npz").read_bytes()).hexdigest()
            frozen = dataclasses.replace(settings, learning=False)
            controller.s = frozen
            update({"state": "evaluating", "evaluation": "trained"}, True)
            trained = replay(evaluation, seconds, controller, frozen, out / "evaluation-trained", stop, progress, True)
            completed += len(evaluation)
            controller.brain.reset(keep_memory=False)
            update({"state": "evaluating", "evaluation": "untrained"}, True)
            baseline = replay(evaluation, seconds, controller, frozen, out / "evaluation-untrained", stop, progress, True)
            report = {"config": dataclasses.asdict(config), "epochs": results,
                      "trained": trained, "untrained": baseline, "memory_sha256": memory_sha,
                      "memory_file": "memory.npz", "learning_validated": False}
            atomic_json(out / "report.json", report)
            update({"state": "completed", "progress": 1, "eta_seconds": 0, "report": report}, True)
            return report
    except (InterruptedError, KeyboardInterrupt):
        update({"state": "stopped", "message": "Stopped; completed epoch memory is retained."}, True)
    except Exception as error:
        update({"state": "failed", "message": f"{type(error).__name__}: {error}"}, True)
        raise
