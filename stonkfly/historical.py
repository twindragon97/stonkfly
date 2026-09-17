"""Public BTC-USD candles, cached locally; never account or execution APIs."""

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import D

INTERVALS = {"1h": 3600, "6h": 21600, "1d": 86400}
SOURCE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: str
    high: str
    low: str
    close: str

    def __post_init__(self):
        o, h, l, c = (D(getattr(self, k)) for k in ("open", "high", "low", "close"))
        if type(self.timestamp) is not int or self.timestamp < 0 or not 0 < l <= min(o, c) <= max(o, c) <= h:
            raise ValueError("Invalid OHLC candle")


def year_start(year):
    return int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())


def validate(candles, start, end, seconds):
    """Require the entire requested UTC interval; never fill missing prices."""
    if len(candles) != (end - start) // seconds:
        raise ValueError("Historical coverage incomplete; choose another period/resolution or provide a complete CSV")
    for i, candle in enumerate(candles):
        if candle.timestamp != start + i * seconds:
            raise ValueError("Historical candles contain gaps, duplicates or unordered timestamps")
    return candles


def read_csv(path, start, end, seconds):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        candles = [Candle(int(r["timestamp"]), r["open"], r["high"], r["low"], r["close"])
                   for r in csv.DictReader(handle)]
    # Preserve source order so malformed/duplicated files fail validation.
    return validate([c for c in candles if start <= c.timestamp < end], start, end, seconds)


def fetch_year(year, seconds, cache, progress=lambda _: None, cancelled=lambda: False):
    start, end = year_start(year), year_start(year + 1)
    path = Path(cache) / f"coinbase-BTC-USD-{year}-{seconds}.json"
    if path.exists():
        record = json.loads(path.read_text())
        payload = json.dumps(record["candles"], separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Historical cache checksum mismatch")
        return validate([Candle(*c) for c in record["candles"]], start, end, seconds)
    found = {}
    for cursor in range(start, end, seconds * 299):
        if cancelled():
            raise InterruptedError("Training stopped")
        stop = min(end, cursor + seconds * 299)
        query = urlencode({"start": datetime.fromtimestamp(cursor, timezone.utc).isoformat(),
                           "end": datetime.fromtimestamp(stop, timezone.utc).isoformat(),
                           "granularity": seconds})
        request = Request(SOURCE + "?" + query, headers={"User-Agent": "Stonkfly historical research"})
        with urlopen(request, timeout=30) as response:
            rows = json.loads(response.read(), parse_float=str)
        if not isinstance(rows, list):
            raise ValueError("Unexpected historical response")
        for r in rows:
            candle = Candle(int(r[0]), str(r[3]), str(r[2]), str(r[1]), str(r[4]))
            if cursor <= candle.timestamp < stop:
                if candle.timestamp in found and found[candle.timestamp] != candle:
                    raise ValueError("Conflicting historical candles")
                found[candle.timestamp] = candle
        progress({"download_year": year, "download_fraction": (stop - start) / (end - start)})
        time.sleep(0.15)  # Public data request pacing only; replay never sleeps.
    candles = validate(sorted(found.values(), key=lambda c: c.timestamp), start, end, seconds)
    rows = [[c.timestamp, c.open, c.high, c.low, c.close] for c in candles]
    payload = json.dumps(rows, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps({"source": SOURCE, "sha256": hashlib.sha256(payload).hexdigest(), "candles": rows}))
    temporary.replace(path)
    return candles
