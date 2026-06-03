import unittest

from src.config import DASHBOARD_TABS, KEY_TIMEFRAMES, SYMBOLS


class ConfigTest(unittest.TestCase):
    def test_symbols_use_usdc_quote_asset(self) -> None:
        self.assertEqual(SYMBOLS, ["BNBUSDC", "SUIUSDC", "SOLUSDC", "BTCUSDC", "ADAUSDC"])
        self.assertTrue(all(symbol.endswith("USDC") for symbol in SYMBOLS))

    def test_key_timeframes_are_locked_for_comparison(self) -> None:
        self.assertEqual(KEY_TIMEFRAMES, ["1h", "4h", "12h", "1d", "1w"])

    def test_dashboard_tabs_include_monitor_signals_and_trades(self) -> None:
        self.assertEqual(DASHBOARD_TABS, ["Monitor", "Market Signals", "Paper Bot & Trades"])


if __name__ == "__main__":
    unittest.main()
