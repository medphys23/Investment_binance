"""Small read-only Binance REST client for public market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from .config import BINANCE_FUTURES_BASE_URL, BINANCE_SPOT_BASE_URLS


class BinanceClientError(RuntimeError):
    """Raised when Binance public market data cannot be fetched."""


@dataclass(frozen=True)
class BinanceClient:
    """Public Binance REST client with no account or trading capabilities."""

    timeout: float = 12.0

    def _get(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{base_url}{path}"
        try:
            response = requests.get(url, params=params or {}, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BinanceClientError(f"Binance request failed for {path}: {exc}") from exc
        return response.json()

    def _get_spot(self, path: str, params: dict[str, Any] | None = None) -> Any:
        errors: list[str] = []
        for base_url in BINANCE_SPOT_BASE_URLS:
            try:
                return self._get(base_url, path, params)
            except BinanceClientError as exc:
                errors.append(str(exc))
        raise BinanceClientError(f"Binance spot request failed for {path}: {' | '.join(errors)}")

    def spot_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        data = self._get_spot(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(data, columns=columns)
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df.drop(columns=["ignore"])

    def futures_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        data = self._get(
            BINANCE_FUTURES_BASE_URL,
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        df = pd.DataFrame(data, columns=columns)
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df.drop(columns=["ignore"])

    def spot_24h(self, symbol: str) -> dict[str, Any]:
        return self._get_spot("/api/v3/ticker/24hr", {"symbol": symbol})

    def futures_24h(self, symbol: str) -> dict[str, Any] | None:
        try:
            return self._get(BINANCE_FUTURES_BASE_URL, "/fapi/v1/ticker/24hr", {"symbol": symbol})
        except BinanceClientError:
            return None

    def book_ticker(self, symbol: str) -> dict[str, Any] | None:
        try:
            return self._get_spot("/api/v3/ticker/bookTicker", {"symbol": symbol})
        except BinanceClientError:
            return None

    def open_interest(self, symbol: str) -> dict[str, Any] | None:
        try:
            return self._get(BINANCE_FUTURES_BASE_URL, "/fapi/v1/openInterest", {"symbol": symbol})
        except BinanceClientError:
            return None

    def mark_price(self, symbol: str) -> dict[str, Any] | None:
        try:
            return self._get(BINANCE_FUTURES_BASE_URL, "/fapi/v1/premiumIndex", {"symbol": symbol})
        except BinanceClientError:
            return None

    def taker_buy_sell_volume(self, symbol: str, period: str, limit: int = 30) -> list[dict[str, Any]]:
        try:
            return self._get(
                BINANCE_FUTURES_BASE_URL,
                "/futures/data/takerlongshortRatio",
                {"symbol": symbol, "period": period, "limit": limit},
            )
        except BinanceClientError:
            return []

    def long_short_ratio(self, symbol: str, period: str, limit: int = 30) -> list[dict[str, Any]]:
        try:
            return self._get(
                BINANCE_FUTURES_BASE_URL,
                "/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": period, "limit": limit},
            )
        except BinanceClientError:
            return []
