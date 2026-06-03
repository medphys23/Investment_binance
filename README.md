# Investment Binance

Local Binance market-analysis dashboard and always-on paper bot for USDC spot pairs.

This project is for research only. It does not use Binance account credentials, does not place real orders, does not manage withdrawals, and does not provide financial advice.

## What It Does

- Tracks `BNBUSDC`, `SUIUSDC`, `SOLUSDC`, `BTCUSDC`, and `ADAUSDC`.
- Uses public Binance spot market data, preferring `https://data-api.binance.vision` to avoid regional `451` failures.
- Shows EMA, Bollinger Bands, RSI, volume, Fibonacci levels, market structure, and probabilistic Elliott Wave reads.
- Compares key timeframes: `1h`, `4h`, `12h`, `1d`, and `1w`.
- Adds MACD, ADX/DMI trend strength, VWAP distance, a volatility regime read, and optional derivatives context (funding, taker buy/sell, long/short ratio, open interest) plus BTC correlation.
- Runs a local paper bot that can open, follow, and close simulated long and short trades.
- Stores worker state, signal features, paper trades, model runs, and policy state in SQLite.
- Learns from its own experience with a reinforcement-style batch policy that retrains every `200` closed trades and sets an advisory entry-probability threshold and size multiplier for the next batch.
- Tracks a simulated leverage tier for research only, capped at `10x`.
- Ships a phone-friendly, installable (PWA) Monitor view backed by worker-precomputed SQLite so the dashboard stays light on mobile.

## Install

```powershell
pip install -r requirements.txt
```

## Run The Dashboard

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open:

```text
http://127.0.0.1:8501
```

The dashboard has three tabs:

- **Monitor:** phone-friendly KPI cards (worker heartbeat, equity, drawdown, open/closed trades, win rate, policy version/threshold), a timeframe bias heatmap, current candidates, and the equity curve. Reads the worker's precomputed SQLite snapshot, so it is fast on mobile.
- **Market Signals:** charting, indicators, alerts, timeframe comparison, and BNB relationship analysis. Watchlist/matrix/comparison read the worker snapshot with a live fallback; only the selected chart is recomputed live.
- **Paper Bot & Trades:** worker heartbeat, paper candidates with advisory ML win-probability gauges, open/closed paper trades, performance charts (equity, drawdown, return distribution, rolling win rate, long-vs-short), the reinforcement-style policy panel, model calibration, and feature importance.

### Install to your phone home screen (PWA)

The app injects a web manifest and mobile meta tags, so once you can reach it from your phone (for example via a local tunnel) you can use the browser's **Add to Home Screen** to launch it like an app. Where to host or tunnel the dashboard for remote phone access is intentionally left as a later deployment decision.

## Run The Always-On Paper Worker

```powershell
python -m src.paper_worker
```

Or on Windows:

```powershell
.\scripts\run_paper_worker.ps1
```

The worker runs every 5 minutes. For a single test cycle:

```powershell
python -m src.paper_worker --once
```

### Start automatically at Windows logon

Register a per-user scheduled task (no admin required). The worker starts about 60 seconds after you sign in, restarts on failure, and logs to `data/worker.log`:

```powershell
.\scripts\register_paper_worker_startup.ps1
```

Start it immediately without rebooting:

```powershell
Start-ScheduledTask -TaskName InvestmentBinancePaperWorker
```

Check status:

```powershell
Get-ScheduledTask -TaskName InvestmentBinancePaperWorker
Get-Content .\data\worker.log -Tail 20
```

Remove auto-start:

```powershell
.\scripts\unregister_paper_worker_startup.ps1
```

The worker still stops when the PC is shut down or asleep. Auto-start only covers the next logon after boot.

## Data Storage

The worker writes to:

```text
data/paper_bot.sqlite
```

SQLite tables:

- `market_snapshots`
- `signal_features` (now includes MACD, ADX, VWAP distance, volatility regime, derivatives, and BTC correlation)
- `paper_trades` (now includes `p_win` and `policy_version` per trade)
- `model_runs`
- `policy_state` (reinforcement-style policy history: threshold, size multiplier, batch reward/win rate)
- `bot_state` (also holds the precomputed `dashboard_snapshot` and active policy controls)

The trained policy model is cached at `data/policy_model.pkl`. Generated local data, logs, and the model file are ignored by git.

## Paper Bot Rules

- Simulated long and short paper trades (no exchange execution).
- Bullish confluence opens `paper_long`; bearish confluence opens `paper_short`.
- Mixed or low-score setups stay `observe`.
- Starting paper equity: `10,000 USDC`.
- Risk per paper trade: `1%`.
- Max open paper trades: `3`.
- Position size is based on entry-to-invalidation distance.
- Trades close on target, invalidation, or timeout.

## Machine Learning (reinforcement-style batch policy)

The ML layer learns from the bot's own closed paper trades and feeds back into how it trades the next batch, always within the fixed risk and leverage rules.

- Cold start: stays pass-through until at least `50` closed paper trades exist (advisory only, no gating).
- Retrains every `200` closed trades (`PAPER_POLICY_BATCH`) instead of every cycle, using a time-ordered (walk-forward) split and `scikit-learn` probability calibration (`CalibratedClassifierCV`) over a `RandomForestClassifier`.
- Derives the next batch's advisory **entry-probability threshold** and **size multiplier** from the realized reward of the last batch (win rate, average return, profit factor). Doing more of what worked and less of what did not is the "reinforcement" element.
- Gating: a candidate must clear the calibrated win-probability threshold in addition to the existing confluence/confidence checks. The size multiplier only scales position size **down** within the `1%` risk ceiling.
- Honest framing: this is a periodic, reward-driven contextual policy (batched contextual-bandit style), not full online RL, and it never guarantees outcomes.
- Persists metrics and feature importance in `model_runs` and policy history in `policy_state`; shows status, calibration, and feature importance in the Paper Bot tab.

## Simulated Leverage Roadmap

The project stores a simulated leverage tier for research only.

- Starts at `1x`.
- Can increase slowly based on paper-trade performance.
- Maximum simulated leverage is `10x`.
- De-risks after drawdown or repeated invalidations.
- This does not execute Binance futures trades.

## Verification

```powershell
python -m unittest discover -s tests
python -m compileall app.py src tests
```
