# Historical pretraining

Open **Entrenar con históricos** in the dashboard, or `/training`. Select the
first and last training years, a later evaluation year, candle resolution and
number of passes. The dashboard starts one background, paper-only worker and
shows progress, observed throughput, estimated remaining time, equity and memory
changes. It remains usable without the neural dataset; starting training requires
`python -m stonkfly prepare` and the existing Linux/macOS C++ environment (or Docker).

```sh
python dashboard/server.py
python -m stonkfly train --start-year 2020 --end-year 2023 --evaluation-year 2024 --interval 1d --epochs 3 --out runs/training/example
```

Completed UTC years only. The UI downloads public Coinbase Exchange **BTC-USD**
since 2016, subject to complete coverage. This is a declared price proxy for the
simulated BTC-USDC account, not historical USDC pricing. Cached OHLC data lives
under `data/historical/`; every load validates its checksum and complete sequence.
The [Coinbase candles API](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
limits requests to 300 candles and may omit intervals. Downloads use bounded
requests, reject missing bars, and never forward-fill prices. Network errors stop
the session. Completed cached years can be reused by a new session.

For other historical sources/years, the CLI accepts `--csv PATH`. Supply columns
`timestamp,open,high,low,close` with UTC Unix **seconds** at the start of each bar,
ascending timestamps and exact coverage for every requested training/evaluation
year. Prices must be finite and positive and obey OHLC bounds. Extra years are
ignored, but duplicates, gaps and out-of-order bars in requested years are errors.
The source must represent Bitcoin USD prices. Keep input files inside ignored
`data/`; no account credentials are needed.

## What is accelerated

Replay never waits for the market. Each completed candle advances one configured
neural observation window. At 1d, one year is 365/366 observations; at 1h, it is
8,760/8,784. This coarser experiment is not equivalent to replaying every minute.
All 166,700 neurons and 25,582,938 retained connections remain in the model.
There is no surrogate policy, graph pruning, optimizer choosing profitable actions,
or replacement strategy when a risk check vetoes an action.

The default 500 ms neural window matches the existing protocol. Optional 100/250 ms
windows change retinal adaptation, spike statistics and reinforcement timing; they
are experimental protocols, not equivalent acceleration. Native integration still
uses 0.1 ms ticks and rate-rule bins of at most 10 ms. Runtime depends on hardware
and activity; the UI measures throughput after loading. No minutes-scale speed or
profitable learning is promised. Initialization/download time and checkpoint writes
can change the remaining-time estimate.

## Episode and execution protocol

1. Start with 100 simulated units, no BTC and reset neural activity/traces. Across
   training passes, retain only the two learned efficacy states on existing edges.
2. Mark holdings at the current completed candle's synthetic bid. Compare equity
   with the previous pre-trade mark: fees and market movement enter the next
   engineered dopamine stimulus. No future prices enter the RGB chart or decoder.
3. Observe only historical closes up to this bar. Apply the unchanged fixed neural
   decoder. The first observation has a short chart; no future warm-up is supplied.
4. Execute a non-HOLD proposal at the **next bar's open**, with a total synthetic
   spread of 10 basis points, 5 basis points of adverse execution impact per side,
   and the existing 0.6% paper fee. The final bar only observes and marks, with no
   fill beyond the episode. There is no historical order-book/liquidity model.
5. Use Decimal amounts and existing sizing/risk limits. Cooldowns/daily limits use
   historical UTC time. Persist each intent before filling it with PaperBroker.
   A loss stop disables further orders for that episode; holdings remain exposed
   and continue to be marked through the end. A veto never chooses another action.

Episodes do not force liquidation at the end; final equity marks held BTC to bid.
Portfolio resets at the next episode are explicit experimental resets, not trades.
Full-graph checkpoints are not rewritten per candle: events are buffered, order
intents remain durable, and compact memory is saved atomically after each completed
training pass. No profit-based checkpoint selection occurs. Stop is cooperative at
observation/download boundaries. A partial pass cannot resume: start a new run;
the last completed pass's `memory.npz` remains intact. No synthetic results are
presented as outputs of the full neural model.

## Evaluation and using memory

After all passes, evaluate the final memory on the reserved later year with all
efficacies frozen, including passive decay. Repeat on the same year with initial
untrained efficacies and reset activity. Each evaluation starts a new 100-unit
portfolio. Reports include final marked equity, return, peak-to-trough drawdown,
fills, vetoes, stimuli, decisions and equity curves. Cash and buy-and-hold references
are shown; buy-and-hold invests all capital at the first actionable open with the
same costs, so its exposure differs from the neural agent's $10 order limit.
Inspect both baselines and trading activity: a HOLD-only policy is not proof of
learning. Repeated tuning against the reserved year invalidates its independence;
one year/one protocol is not a statistical validation of profitable learning.

Artifacts are in the selected ignored run folder: `status.json`, `provenance.json`,
`report.json`, `memory.npz`, and per-phase events/ledgers. Provenance records the
dataset verification, source/data hashes, settings and execution assumptions.

```sh
python -m stonkfly run --memory runs/training/example/memory.npz --out runs/paper-pretrained
# Later: resume that paper run without --memory.
python -m stonkfly run --out runs/paper-pretrained
```

Memory import checks graph, circuit, model/kernel configuration and efficacy
bounds. Only memory transfers, never cash, positions, orders or transient activity.
Initialization is limited to a fresh paper run and persisted before trading.
The UI prints the exact command after evaluation; it never activates live trading
or replaces the running paper worker automatically.

Training UI start/stop requests require a same-origin token. This prevents ordinary
cross-site form requests, not access by other users of the dashboard: keep it on
localhost or a trusted LAN, or put an authenticated proxy in front of it. Existing
paper monitoring remains read-only. In Docker, rebuild/use the image containing
these changes; the existing bind-mounted `data` and `runs` preserve training output.
Paper and training workers may run concurrently and share the available CPU/RAM.
