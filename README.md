# Investment Binance

Local Binance market-analysis dashboard and always-on paper bot for USDC spot pairs.

This project is for research only. It does not use Binance account credentials, does not place real orders, does not manage withdrawals, and does not provide financial advice.

## What It Does

- Tracks `BNBUSDC`, `SUIUSDC`, `SOLUSDC`, `BTCUSDC`, and `ADAUSDC`.
- Uses public Binance spot market data, preferring `https://data-api.binance.vision` to avoid regional `451` failures.
- Shows EMA, Bollinger Bands, RSI, volume, Fibonacci levels, market structure, and probabilistic Elliott Wave reads.
- Compares key timeframes: `1h`, `4h`, `12h`, `1d`, and `1w`.
- Runs a local paper bot that can open, follow, and close simulated spot-long trades.
- Stores worker state, signal features, paper trades, and model runs in SQLite.
- Trains a batch ML model from closed paper trades once enough data exists.
- Tracks a simulated leverage tier for research only, capped at `10x`.

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

The dashboard has two tabs:

- **Market Signals:** charting, indicators, alerts, timeframe comparison, and BNB relationship analysis.
- **Paper Bot & Trades:** worker heartbeat, paper candidates, open/closed paper trades, risk metrics, simulated leverage tier, and ML model status.

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
- `signal_features`
- `paper_trades`
- `model_runs`
- `bot_state`

Generated local data is ignored by git.

## Paper Bot Rules

- Spot-style paper trades only.
- Opens only simulated long positions.
- Bearish setups are treated as `risk_off` or `avoid`, not shorts.
- Starting paper equity: `10,000 USDC`.
- Risk per paper trade: `1%`.
- Max open paper trades: `3`.
- Position size is based on entry-to-invalidation distance.
- Trades close on target, invalidation, or timeout.

## Machine Learning

The ML layer is batch-trained from closed paper trades.

- Skips training until at least `50` closed paper trades exist.
- Uses an auditable `scikit-learn` baseline model.
- Stores metrics and feature importance in `model_runs`.
- Shows model status in the Paper Bot tab.

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
