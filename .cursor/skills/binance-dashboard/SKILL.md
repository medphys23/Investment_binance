# Build or revise Binance technical signal dashboard

## Trigger

Use this workflow when the user asks to build, revise, rerun, or extend the Binance technical signal dashboard, including changes to symbols, timeframes, indicators, alerts, or market thesis panels.

## Preconditions

- Work from the repository root.
- Keep the app read-only and unauthenticated.
- Do not add trading execution, account API keys, withdrawals, or order management.
- Paper-trade tracking may write local SQLite files under `data/`, but it must use live Binance public market prices only.
- Do not require manual trade journal uploads for the active paper bot.
- Keep paper trades as spot-style longs only. Bearish setups must not open shorts.
- Keep Streamlit indicators and paper-bot trade state in separate dashboard tabs.
- Preserve `6h`, `8h`, and `12h` as first-class timeframes unless the user explicitly removes them.
- Use USDC quote pairs for tracked coins unless the user explicitly asks for a different quote asset.
- Keep `1h`, `4h`, `12h`, `1d`, and `1w` as the key comparison timeframes.

## Steps

1. Read `AGENTS.md` and `skills.md`.
2. Inspect `app.py`, `src/config.py`, `src/binance_client.py`, `src/indicators.py`, `src/signals.py`, `src/paper_worker.py`, `src/storage.py`, `src/paper_trading.py`, `src/ml.py`, and `tests/`.
3. Update configuration first for symbols, timeframes, indicator periods, or data limits.
4. Keep Binance access inside `src/binance_client.py` and use public REST endpoints only.
5. Use the configured spot base URL fallback chain; prefer `data-api.binance.vision` for spot market data because `api.binance.com` may return regional `451` restrictions.
6. Keep futures context optional and non-blocking because `/fapi` endpoints may be region-blocked.
7. Keep indicator math inside `src/indicators.py`.
8. Keep scenario, confluence, Elliott Wave, and alert logic inside `src/signals.py`.
9. Keep Streamlit layout and chart rendering inside `app.py`.
10. Add or update focused tests in `tests/` for changed indicator or signal behavior.

## Verification

```powershell
python -m unittest discover -s tests
python -m compileall app.py src tests
```

Optional UI smoke check:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## Iteration notes

- 2026-06-03: Initial V1 created as a local Streamlit dashboard with BNB/SUI/SOL/BTC/ADA, EMA, Bollinger Bands, RSI, volume, Fibonacci, probabilistic Elliott Wave, futures context, multi-timeframe matrix, and in-app alerts.
- 2026-06-03: Mirrored repo `skills.md` workflow into Cursor skill format.
- 2026-06-03: Fixed regional Binance `451` failure by preferring `data-api.binance.vision` for spot data and treating futures endpoints as optional when blocked.
- 2026-06-03: Switched tracked pairs from USDT to USDC and added key-timeframe comparison across `1h`, `4h`, `12h`, `1d`, and `1w`.
- 2026-06-03: Replaced manual trade journal upload idea with an active local paper-trade tracker that opens/follows/closes paper positions from live market data only.
- 2026-06-03: Added SQLite-backed always-on paper worker, batch ML learning, spot-long-only paper rules, and separate Market Signals / Paper Bot dashboard tabs.
