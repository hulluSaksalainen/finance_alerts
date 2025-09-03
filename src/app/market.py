"""Module for retrieving stock market data."""
import time
import logging
from typing import Tuple
import yfinance as yf

logger = logging.getLogger("stock-alerts")

def get_open_and_last(ticker: str) -> Tuple[float, float]:
    """
    Retrieve today's opening price and the latest available price for a ticker.

    Strategy:
      1. Try intraday data with finer intervals ("1m", "5m", "15m").
         - Use the very first "Open" of the day.
         - Use the most recent "Close" (last candle).
         - Retry once per interval in case Yahoo delivers empty DataFrames.
      2. If no intraday data is available (e.g., market closed),
         fall back to daily interval ("1d").
    """
    attempt=0
    for interval in ("1m", "5m", "15m","1d"):
        try:
            for attempt in range(2):
                df = yf.Ticker(ticker).history(
                    period="1d", interval=interval, auto_adjust=False
                )
                if not df.empty:
                    open_today = float(df.iloc[0]["Open"])
                    last_price = float(df.iloc[-1]["Close"])
                    logger.debug(
                        "Intraday %s: interval=%s open=%.4f last=%.4f",
                        ticker, interval, open_today, last_price,
                    )
                    return open_today, last_price
                logger.debug(
                    "Empty intraday data (%s, %s), retrying once",
                    ticker, interval,
                )
                time.sleep(0.4)
        except (ValueError, KeyError, IndexError) as e:
            logger.warning("Unexpected error fetching intraday data for %s at interval %s: %s",
                ticker, interval, e)
            time.sleep(0.4)
        attempt+=1
        if df.empty and interval == "1d" and attempt >=2:
            raise RuntimeError(f"No data available for {ticker}")

    row = df.iloc[-1]
    open_today, last_price = float(row["Open"]), float(row["Close"])
    logger.debug(
        "Fallback daily data %s: open=%.4f last=%.4f",
        ticker, open_today, last_price,
    )
    return open_today, last_price
