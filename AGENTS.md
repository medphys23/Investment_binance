# Investment - agent instructions

Inherits global rules from `~/.codex/AGENTS.md` and `~/.cursor/rules/`.

**Cursor mirror:** [`.cursor/rules/`](.cursor/rules/) - keep in sync with this file.

## Purpose

Local Binance technical signal dashboard for monitoring BNB, SUI, SOL, BTC, and ADA against USDC across multiple timeframes. The app analyzes public market data only and must never access exchange accounts, place trades, or automate real execution. Local paper-trade logs are allowed for research.

## Stack

- Python 3.11+
- Streamlit local dashboard: `streamlit run app.py`
- Always-on local paper worker: `python -m src.paper_worker`
- SQLite local storage: `data/paper_bot.sqlite`
- Binance public REST market data via `requests`; spot calls prefer `data-api.binance.vision` because `api.binance.com` may return regional `451` restrictions
- `pandas` / `numpy` for indicator calculations
- `plotly` for candlestick, volume, RSI, EMA, Bollinger, Fibonacci, and Elliott Wave overlays
- `scikit-learn` for the reinforcement-style batch policy learned from closed local paper trades (probability calibration + reward-driven controls)
- Worker precomputes a JSON `dashboard_snapshot` (in `bot_state`) so the Streamlit app reads SQLite instead of recomputing all symbols/timeframes
- Installable PWA Monitor view: `.streamlit/config.toml` theme plus an inline manifest/meta injection in `app.py`
- Tests use the Python standard-library `unittest`

## Uses from global catalog

- Python numerics: `numpy`, `pandas`
- Streamlit + charts: Streamlit dashboard with Plotly visualization
- Configuration and secrets: no API keys required; do not add account credentials
- Research / investment boundary: informational analytics only, not financial advice or trading automation

## Verification

Run these after changing app logic, indicators, Binance data access, or signal rules:

```powershell
python -m unittest discover -s tests
python -m compileall app.py src tests
```

For UI smoke checks:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Then open `http://127.0.0.1:8501` and verify the page loads with Binance market data.

## Repo skills catalog

Maintain [`skills.md`](skills.md) beside this file - document repeatable Codex workflows per `~/.codex/AGENTS.md`. **Cursor mirror:** `.cursor/skills/<name>/SKILL.md` when workflows exist.

## Repo-specific rules

- Keep Binance access read-only. Do not add authenticated Binance account endpoints, API keys, real trade placement, order management, withdrawals, or automated exchange execution.
- Paper trading may open/follow/close local simulated positions only when based on live Binance public market prices. Do not use synthetic market data or manual trade uploads for the active paper bot.
- Paper trades are simulated long and short positions from live public prices only (no exchange execution). Bearish confluence opens `paper_short`; bullish opens `paper_long`; unclear setups stay `observe`.
- The worker and dashboard share SQLite at `data/paper_bot.sqlite`; generated `data/` outputs must stay out of git.
- Simulated leverage is research-only, starts at `1x`, and must never exceed `10x`.
- The ML layer is a reinforcement-style batch policy: retrain every `200` closed trades (`PAPER_POLICY_BATCH`), cold-start pass-through before `50`, time-ordered split + calibration, and reward-driven advisory entry-probability threshold and size multiplier. The size multiplier only scales position size down within the `1%` risk ceiling; gating never overrides the leverage cap or scenario-based rules. Frame it as a reward-driven contextual policy, never a guarantee.
- Heavy analysis belongs in the worker (`src/analysis_engine.py` `gather_cycle_data` + `build_dashboard_snapshot`), persisted to SQLite; keep the dashboard reading the snapshot with a live fallback only for the single selected chart.
- New persisted data: extra `signal_features` columns (MACD, ADX, VWAP distance, volatility regime, funding, taker buy/sell, long/short ratio, open interest, BTC correlation), `paper_trades.p_win`/`policy_version`, and a `policy_state` table. Use the lightweight `ALTER TABLE` migration in `src/storage.py` when adding columns so existing databases keep working. The trained model caches at `data/policy_model.pkl` (gitignored).
- Keep predictions scenario-based: confidence, confluence, invalidation, and likely zones only. Do not present deterministic price guarantees.
- Treat Elliott Wave detection as probabilistic and explain invalidation/context where surfaced.
- Prioritize `6h`, `8h`, and `12h` timeframes as first-class regime filters.
- Use USDC quote pairs for tracked coins. Do not switch back to USDT unless the user explicitly asks.
- Keep the key comparison matrix focused on `1h`, `4h`, `12h`, `1d`, and `1w`.
- When adding symbols or indicators, update `src/config.py`, signal tests, and this file if verification changes.
- Keep generated caches and bytecode out of git.
- Spot market data must use the configured base URL fallback chain. Futures data is optional because `/fapi` endpoints may return regional `451` restrictions.

## Stack propagation

When you introduce a new library, skill, or tool here, update `~/.codex/AGENTS.md` and propagate to other repos per global policy.

## Git

- Do not commit unless the user asks.
