"""Shared configuration for the Binance technical signal dashboard."""

SYMBOLS = ["BNBUSDC", "SUIUSDC", "SOLUSDC", "BTCUSDC", "ADAUSDC"]
DASHBOARD_TABS = ["Monitor", "Market Signals", "Paper Bot & Trades"]

TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"]
KEY_TIMEFRAMES = ["1h", "4h", "12h", "1d", "1w"]
INTERMEDIATE_TIMEFRAMES = ["6h", "8h", "12h"]

EMA_PERIODS = [9, 20, 50, 100, 200]
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14
REL_VOLUME_PERIOD = 20
FIB_LOOKBACK = 160
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
VWAP_PERIOD = 20
VOLATILITY_RANK_WINDOW = 120
DERIVATIVES_PERIOD = "1h"

BINANCE_SPOT_BASE_URLS = ["https://data-api.binance.vision", "https://api.binance.com"]
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"

DEFAULT_CANDLE_LIMIT = 500
MATRIX_CANDLE_LIMIT = 260

PAPER_DB_PATH = "data/paper_bot.sqlite"
PAPER_WORKER_INTERVAL_SECONDS = 300
PAPER_STARTING_EQUITY = 10_000.0
PAPER_RISK_PER_TRADE = 0.01
PAPER_MAX_OPEN_TRADES = 3
PAPER_MAX_HOLD_HOURS = 168
PAPER_MIN_ML_CLOSED_TRADES = 50
PAPER_ML_RETRAIN_NEW_TRADES = 20
SIMULATED_MAX_LEVERAGE = 10.0

# Reinforcement-style batch policy: learn from the last batch of closed trades
# to set how the bot navigates the next batch. The model never guarantees
# outcomes; it only adjusts an advisory entry threshold and size multiplier
# within the existing risk ceiling and simulated leverage cap.
PAPER_POLICY_BATCH = 200
PAPER_POLICY_MODEL_PATH = "data/policy_model.pkl"
PAPER_POLICY_BASE_THRESHOLD = 0.50
PAPER_POLICY_MIN_THRESHOLD = 0.40
PAPER_POLICY_MAX_THRESHOLD = 0.65
PAPER_POLICY_MIN_SIZE_MULTIPLIER = 0.25
PAPER_POLICY_MAX_SIZE_MULTIPLIER = 1.0
PAPER_POLICY_RECENT_WINDOW = 600
