"""Core application logic for stock price monitoring and notifications."""
import datetime as dt
from zoneinfo import ZoneInfo
import logging
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urlparse, parse_qs
import requests

from .market import get_open_and_last
from .ntfy import notify_ntfy
from .state import load_state, save_state
from .company import auto_keywords
from .news import fetch_headlines, build_query, filter_titles

logger = logging.getLogger("stock-alerts")

def _ticker_to_query(ticker: str, override_name: str | None = None) -> str:
    """
    Return a human-friendly query term for a ticker.

    Args:
        ticker: Raw ticker symbol (e.g., "AAPL").
        override_name: Optional override (e.g., "Apple").

    Returns:
        A display/query string; override_name if provided, else the ticker.
    """
    return ticker if override_name is None else override_name

def _ensure_https(u: str) -> str:
    """
    Ensure the given URL has a scheme. If missing, prefix with https://

    This helps when feeds provide bare domains or schemeless URLs.
    """
    # TO DO: Handle empty strings
    # TO DO: If u starts with http:// or https://, return u unchanged
    # TO DO: Otherwise, prefix u with "https://"
    if not u:
        return u
    elif u.startswith("http://") or u.startswith("https://"):
        return u
    else:
        return "https://" + u

def _extract_original_url(link: str, *, resolve_redirects: bool = True,
    timeout: float = 3.0) -> str:
    """
    Try to extract the original article URL from Google News redirect links.

    Strategy:
        1) If it's a news.google.com link and contains ?url=..., use that.
        2) Optionally resolve redirects via HEAD (fallback GET) to obtain the final URL.
        3) If all fails, return the input link.

    Args:
        link: Possibly a Google News RSS link.
        resolve_redirects: Whether to follow redirects to the final URL.
        timeout: Per-request timeout in seconds.

    Returns:
        A best-effort "clean" URL pointing to the original source.
    """
    # TO DO: Normalize link via _ensure_https
    link=_ensure_https(link)
    # TO DO: If link is a news.google.com URL, attempt to extract ?url= parameter
    if "news.google.com" in link:
        parsed = urlparse(link)
        qs = parse_qs(parsed.query)
        if "url" in qs and qs["url"]:
            return qs["url"][0]
    if not resolve_redirects:
        return link
    # TO DO: Optionally resolve redirects via HEAD or GET
    try:
        r = requests.head(link, allow_redirects=True, timeout=timeout)
    # TO DO: Return cleaned URL or fallback to original link
        r.raise_for_status()
        return r.url
    except requests.RequestException as e:
        logger.debug("Failed to resolve redirects for %s: %s", link, e)
        return link



def _domain(url: str) -> str:
    """
    Extract a pretty domain (strip leading 'www.') from a URL for compact display.
    """
    # TO DO: Parse the domain with urlparse
    r=urlparse(url)
    domain=r.netloc
    # TO DO: Strip leading "www." if present
    if domain.startswith("www."):
        domain=domain[4:]
    # TO DO: Return cleaned domain or original url on error
    return domain if domain else url

def _format_headlines(items: List[Dict[str, Any]]) -> str:
    """
    Build a compact Markdown block for headlines.

    - Web (ntfy web app): Markdown will be rendered (nice links)
    - Mobile (ntfy apps): Markdown shows as plain text, so we also include
      a short, real URL line that remains clickable on phones.

    Returns:
        A multi-line string ready to embed into the notification body.
    """
    # TO DO: Handle empty list case
    if not items:
        return ""
    # TO DO: Build Markdown lines with titles, sources and cleaned links
    lines = []
    for item in items:  # type: ignore
        title = item.get("title", "No Title")
        source = item.get("source", "Unknown Source")
        link = item.get("link", "")
        clean_link = _extract_original_url(link)
        domain = _domain(clean_link)
        lines.append(f"- [{title}]({clean_link}) ({source}, {domain})")
    # TO DO: Join lines with newline characters and return the result
    return "\n".join(lines)

def now_tz(tz: str) -> dt.datetime:
    """
    Get current date/time in a specific timezone (e.g., 'Europe/Berlin').

    Using timezone-aware datetimes avoids DST pitfalls and makes logging consistent.
    """
    # TO DO: Use dt.datetime.now with ZoneInfo to return timezone-aware datetime
    return dt.datetime.now(ZoneInfo(tz))

def is_market_hours(cfg_mh: dict) -> bool:
    """
    Heuristic market-hours check (simple window, no holidays).

    Args:
        cfg_mh: Market hours config with keys:
            - enabled (bool)
            - tz (str)
            - start_hour (int)
            - end_hour (int)
            - days_mon_to_fri_only (bool)

    Returns:
        True if within the configured hours, else False.
    """
    # TO DO: If checking is disabled, return True
    if not cfg_mh.get("enabled", True):
        return True
    # TO DO: Obtain current time via now_tz(cfg_mh["tz"])
    now = now_tz(cfg_mh.get("tz", "UTC"))
    # TO DO: Optionally limit to Monday–Friday
    if cfg_mh.get("days_mon_to_fri_only", True):
        if now.weekday() >= 5:
            return False
    # TO DO: Compare current hour with start_hour/end_hour
    start_hour = cfg_mh.get("start_hour", 0)
    end_hour = cfg_mh.get("end_hour", 23)
    return start_hour <= now.hour < end_hour

def run_once(
    tickers: List[str],
    threshold_pct: float,
    ntfy_server: str,
    ntfy_topic: str,
    state_file: Path,
    market_hours_cfg: dict,
    test_cfg: dict,
    news_cfg: dict,
) -> None:
    """
    Execute one monitoring cycle:
      - Check market hours (with optional test bypass)
      - For each ticker:
          * Fetch open & last price (intraday preferred)
          * Compute Δ% vs. open
          * Trigger ntfy push if |Δ%| ≥ threshold (with de-bounce via state file)
          * Optionally attach compact news headlines (with cleaned source URLs)

    Side effects:
      - Sends an HTTP POST to ntfy (unless dry_run)
      - Reads/writes the alert state JSON (anti-spam)
      - Writes logs according to logging setup
    """
    # TO DO: Log job start and determine market-hours eligibility
    logger.info("Starting monitoring cycle")
    in_market_hours = is_market_hours(market_hours_cfg)
    if not in_market_hours:
        if test_cfg.get("bypass_market_hours", False):
            logger.info("Outside market hours, but bypass enabled; continuing")
        else:
            logger.info("Outside market hours; exiting")
            return
    # TO DO: Load alert state from state_file
    state = load_state(state_file)
    dry_run = test_cfg.get("dry_run", False)
    force_delta_pct = test_cfg.get("force_delta_pct", None)
    # TO DO: Iterate over tickers and fetch open/last prices
    for ticker in tickers:
        # TO DO: Fetch open and last prices via get_open_and_last
        try:
            open_price, last_price = get_open_and_last(ticker)
        except (requests.RequestException, ValueError) as e:
            logger.warning("Failed to fetch prices for %s: %s", ticker, e)
            continue
        if open_price is None or last_price is None:
            logger.warning("Incomplete price data for %s; skipping", ticker)
            continue
        # TO DO: Compute Δ% and apply test override if provided
        delta_pct = ((last_price - open_price) / open_price * 100) if open_price != 0 else 0.0
        if force_delta_pct is not None:
            delta_pct = force_delta_pct
            logger.info("Test mode: forcing Δ%% for %s to %.2f%%", ticker, delta_pct)
        logger.info("Ticker %s: open=%.4f last=%.4f Δ%%=%.2f%%", ticker, open_price,
            last_price, delta_pct)
        # TO DO: Decide whether to send an alert based on threshold_pct and state
        alert_needed = abs(delta_pct) >= threshold_pct
        # TO DO: Decide whether to send alerts and prepare notification body
        already_alerted = state.get(ticker, {}).get("alerted", False)
        if alert_needed and not already_alerted:
            logger.info("Alert triggered for %s (Δ%%=%.2f%%)", ticker, delta_pct)
            # TO DO: Optionally fetch and format news headlines
            headlines_md = ""
            if news_cfg.get("enabled", False):
                override_name = news_cfg.get("override_name")
                query_term = _ticker_to_query(ticker, override_name)
                query = build_query(query_term, ticker)
                try:
                    articles = fetch_headlines(query, limit=news_cfg.get("max_items", 3))
                    filtered_articles = filter_titles(articles,
                        required_keywords=auto_keywords(ticker)[1])
                    # TO DO: Optionally fetch and format news headlines
                    headlines_md = _format_headlines(filtered_articles[:news_cfg.get("max_items",
                        3)])
                except (requests.RequestException, ValueError) as e:
                    logger.warning("Failed to fetch/filter news for %s: %s", ticker, e)
            # TO DO: Prepare notification title and message body
            direction = "up" if delta_pct > 0 else "down"
            # TO DO: Compute Δ% and apply test overrides if needed
            title = f"Stock Alert: {ticker} {'▲' if delta_pct > 0 else '▼'} {abs(delta_pct):.2f}%"
            message_lines = [
                f"Ticker: {ticker}",
                f"Open Price: {open_price:.4f}",
                f"Last Price: {last_price:.4f}",
                f"Change: {delta_pct:.2f}% ({direction})",
            ]
            if headlines_md:
                message_lines.append("\n**News Headlines:**\n" + headlines_md)
            message = "\n".join(message_lines)
            # TO DO: Send notification via notify_ntfy
            notify_ntfy(
                server=ntfy_server,
                topic=ntfy_topic,
                title=title,
                message=message,
                dry_run=dry_run,
                markdown=True,
                click_url=f"https://finance.yahoo.com/quote/{ticker}",
            )
            # TO DO: persist state via save_state
            state[ticker] = {"alerted": True, "last_alert_time": dt.datetime.utcnow().isoformat()}
            save_state(state_file, state)
        elif not alert_needed and already_alerted:
            logger.info("Resetting alert state for %s (Δ%%=%.2f%%)", ticker, delta_pct)
            state[ticker] = {"alerted": False}
        else:
            logger.info("No alert needed for %s (Δ%%=%.2f%%)", ticker, delta_pct)
