import unittest
from unittest.mock import Mock, patch

import requests

from src.binance_client import BinanceClient


class BinanceClientTest(unittest.TestCase):
    def test_spot_get_falls_back_to_next_public_market_data_base_url(self) -> None:
        blocked = Mock()
        blocked.raise_for_status.side_effect = requests.HTTPError("451 restricted")

        ok = Mock()
        ok.raise_for_status.return_value = None
        ok.json.return_value = {"symbol": "BTCUSDT"}

        with patch("src.binance_client.requests.get", side_effect=[blocked, ok]) as request_get:
            result = BinanceClient()._get_spot("/api/v3/ticker/24hr", {"symbol": "BTCUSDT"})

        self.assertEqual(result, {"symbol": "BTCUSDT"})
        self.assertEqual(request_get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
