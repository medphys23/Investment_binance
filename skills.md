# Investment - Codex repeatable workflows

Dual canonical: mirror each workflow in `.cursor/skills/<name>/SKILL.md`. Add `### Heading` sections per workflow (see `~/.codex/AGENTS.md` -> Skill propagation -> Repo `skills.md`).

### Build or revise Binance technical signal dashboard

**Triggered by:**
- User asks to build, revise, rerun, or extend the Binance technical signal dashboard.
- User adds indicators, symbols, timeframes, alert rules, or market thesis panels.

**Preconditions:**
- Work from the repository root.
- Keep the app read-only and unauthenticated.
- Do not add trading execution, account API keys, withdrawals, or order management.
- Paper-trade tracking may write local SQLite files under `data/`, but it must use live Binance public market prices only.
- Do not require manual trade journal uploads for the active paper bot.
- Paper trades may open simulated longs (`paper_long`) or shorts (`paper_short`) from live public prices; mixed setups stay `observe`.
- Heavy analysis runs in the worker (`gather_cycle_data`), which persists a JSON `dashboard_snapshot` in `bot_state`; the dashboard reads that snapshot and only recomputes the single selected chart live.
- The reinforcement-style ML policy retrains every `200` closed trades (`PAPER_POLICY_BATCH`), with cold-start pass-through before `50`. It sets an advisory entry-probability threshold and a size multiplier within the `1%` risk ceiling and `10x` simulated-leverage cap. Keep it framed as a reward-driven contextual policy, never a guarantee.
- Keep Streamlit indicators, the mobile Monitor view, and paper-bot trade state in separate dashboard tabs (`Monitor`, `Market Signals`, `Paper Bot & Trades`). Keep the PWA manifest/meta injection in `app.py` and theme in `.streamlit/config.toml`.
- Preserve `6h`, `8h`, and `12h` as first-class timeframes unless the user explicitly removes them.
- Use USDC quote pairs for tracked coins unless the user explicitly asks for a different quote asset.
- Keep `1h`, `4h`, `12h`, `1d`, and `1w` as the key comparison timeframes.

**Steps:**
1. Read `AGENTS.md` and this workflow.
2. Inspect existing app structure before editing: `app.py`, `src/config.py`, `src/binance_client.py`, `src/indicators.py`, `src/signals.py`, `src/analysis_engine.py`, `src/paper_worker.py`, `src/storage.py`, `src/paper_trading.py`, `src/risk.py`, `src/strategy.py`, `src/ml.py`, and `tests/`.
3. Update configuration first for symbols, timeframes, indicator periods, or data limits.
4. Keep Binance access inside `src/binance_client.py` and use public REST endpoints only.
5. Use the configured spot base URL fallback chain; prefer `data-api.binance.vision` for spot market data because `api.binance.com` may return regional `451` restrictions.
6. Keep futures context optional and non-blocking because `/fapi` endpoints may be region-blocked.
7. Keep indicator math inside `src/indicators.py`; avoid mixing UI code with calculations.
8. Keep scenario, confluence, Elliott Wave, and alert logic inside `src/signals.py`.
9. Keep Streamlit layout and chart rendering inside `app.py`.
10. Add or update focused tests in `tests/` for changed indicator or signal behavior.
11. Start or refresh the Streamlit app only after verification passes.
12. For worker changes, run `python -m src.paper_worker --once` only when a live-data smoke test is needed.

**Verification:**

```powershell
python -m unittest discover -s tests
python -m compileall app.py src tests
```

Optional UI smoke check:

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

**Iteration notes:**
- 2026-06-03: Initial V1 created as a local Streamlit dashboard with BNB/SUI/SOL/BTC/ADA, EMA, Bollinger Bands, RSI, volume, Fibonacci, probabilistic Elliott Wave, futures context, multi-timeframe matrix, and in-app alerts.
- 2026-06-03: Added repo-level `AGENTS.md`, `skills.md`, and Cursor rule mirrors after user requested top-level Codex/skills structure.
- 2026-06-03: Fixed regional Binance `451` failure by preferring `data-api.binance.vision` for spot data and treating futures endpoints as optional when blocked.
- 2026-06-03: Switched tracked pairs from USDT to USDC and added key-timeframe comparison across `1h`, `4h`, `12h`, `1d`, and `1w`.
- 2026-06-03: Replaced manual trade journal upload idea with an active local paper-trade tracker that opens/follows/closes paper positions from live market data only.
- 2026-06-03: Added SQLite-backed always-on paper worker, batch ML learning, spot-long-only paper rules, and separate Market Signals / Paper Bot dashboard tabs.
- 2026-06-03: Enabled symmetric `paper_short` entries (mirror of `paper_long` confluence) for bear-market training.
- 2026-06-03: Added MACD/ADX/VWAP/volatility-regime/derivatives/BTC-correlation features; moved heavy analysis into the worker (`gather_cycle_data` + `dashboard_snapshot`) so the dashboard reads SQLite; refactored `ml.py` into a 200-trade reinforcement-style batch policy (time-ordered split + calibration + reward-driven threshold/size multiplier) wired into entry gating and sizing with cold-start pass-through; added new `signal_features`/`paper_trades` columns and a `policy_state` table; and added a mobile/PWA Monitor tab with Plotly visuals plus `.streamlit/config.toml`.
