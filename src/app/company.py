"""Module for retrieving and caching company metadata."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import time
import yfinance as yf
logger = logging.getLogger("stock-alerts")
CACHE_FILE = Path("../../json/company_cache.json")

# TO DO # Common legal suffixes often found in company names (ADD MORE),
# which we remove to get a cleaner keyword (e.g., "Apple Inc." -> "Apple").
LEGAL_SUFFIXES = {
    "inc", "inc.","AG", "AG.", "SE", "SE.", "Ltd", "Ltd.", "LLC", "LLC.", "Corp", "Corp.",
    "Corporation", "Corporation.", "GmbH", "GmbH.", "PLC", "PLC.", "Co", "Co.", "S.A.",
    "S.A", "N.V.", "N.V", "AB", "AB.",
}

# TO DO Add class attributes like in the class description

@dataclass
class CompanyMeta:
    """
    Represents metadata about a company/ticker.
    
    Attributes:
        ticker (str): The full ticker symbol, e.g., "SAP.DE".
        name (Optional[str]): Cleaned company name without legal suffixes, e.g., "Apple".
        raw_name (Optional[str]): Original company name as returned by Yahoo Finance, e.g., 
            "Apple Inc.".
        source (str): Source of the name (e.g., "info.longName", "info.shortName", "fallback").
        base_ticker (str): Simplified ticker without suffixes, e.g., "SAP" for "SAP.DE".
    """
    ticker: str
    name: Optional[str] = None
    raw_name: Optional[str] = None
    source: str = ""
    base_ticker: str = ""

# TO DO Finish this function:

def _load_cache() -> Dict[str, Any]:
    """Load cached company metadata from JSON file."""
    if CACHE_FILE.exists():
        try:
            # Return content of file
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, IOError, json.JSONDecodeError):
            # Return empty dictionary
            return {}
    else:
        # Return empty dictionary
        return {}

def _save_cache(cache: Dict[str, Any]) -> None:
    """Save company metadata to local cache file."""
    exist_cache = _load_cache()
    exist_cache.update(cache)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")

# TO DO Finish the function logic
def _strip_legal_suffixes(name: str) -> str:
    """
    Remove common legal suffixes from a company name.

    Example:
        "Apple Inc." -> "Apple"
        "SAP SE" -> "SAP"
    """
    parts = [p.strip(",. ") for p in name.split()]
    while parts and parts[-1].lower() in LEGAL_SUFFIXES:
        # There is something missing
        parts.pop()
    return " ".join(parts) if parts else name.strip()

# TO DO Finish the function logic
def _base_ticker(symbol: str) -> str:
    """
    Extract the base ticker symbol.

    Examples:
        "SAP.DE" -> "SAP"
        "BRK.B"  -> "BRK"
        "^GDAXI" -> "^GDAXI" (indices remain unchanged)
    """
    if symbol.startswith("^"):  # Index tickers like ^GDAXI
        return symbol
    if "." in symbol:
        symbol = symbol.split(".")[0]
    elif "-" in symbol:
        symbol = symbol.split("-")[0]
    elif "/" in symbol:
        symbol = symbol.split("/")[0]
    return symbol

# TO DO Finish the try and except block
def _fetch_yf_info(symbol: str, retries: int = 2, delay: float = 0.4) -> Dict[str, Any]:
    """
    Fetch company information from Yahoo Finance.

    Args:
        symbol (str): Ticker symbol.
        retries (int): Number of retries if request fails.
        delay (float): Delay between retries in seconds.

    Returns:
        dict: Yahoo Finance info dictionary (may be empty if lookup fails).
    """
    last_exc = None
    for _ in range(retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if info:
                return info
        except (KeyError, ValueError, TypeError) as e:
            last_exc = e
            time.sleep(delay)
            logger.debug( "No company info %s: %s", ticker, last_exc )
    return {}


def get_company_meta(symbol: str) -> CompanyMeta:
    """
    Retrieve company metadata (name, base ticker, etc.) with caching and fallbacks.
    """
    # TO DO: Load the cache with _load_cache() and return early if the symbol exists
    cache = _load_cache()
    if symbol in cache:
        return CompanyMeta(**cache[symbol])

    # TO DO: Fetch raw company information via _fetch_yf_info
    info = _fetch_yf_info(symbol)

    # TO DO: Extract a potential company name from info ("longName", "shortName", "displayName")
    raw_name = info.get("longName") or info.get("shortName") or info.get("displayName")
    if not raw_name:
        raw_name = ""
    source = ("info.longName" if "longName" in info
        else ("info.shortName" if "shortName" in info
        else ("info.displayName" if "displayName" in info
        else "fallback")))

    # TO DO: Clean the extracted name with _strip_legal_suffixes and handle fallback to _base_ticker
    clean = _strip_legal_suffixes(raw_name) if raw_name else _base_ticker(symbol)


    # TO DO: Create a CompanyMeta instance and cache the result using _save_cache
    meta = CompanyMeta(
        ticker=symbol,
        name=clean if clean else None,
        raw_name=raw_name if raw_name else None,
        source=source,
        base_ticker=_base_ticker(symbol)
    )
    _save_cache({**cache, symbol: meta.__dict__})

    # TO DO: Save the constructed metadata back into the cache
    _save_cache(cache)
    return meta


def auto_keywords(symbol: str) -> Tuple[str, list[str]]:
    """
    Generate a company search keyword set based on symbol.
    """
    # TO DO: Fetch the CompanyMeta for the symbol
    meta = get_company_meta(symbol)

    # TO DO: Determine the display name and construct the keyword list
    name = meta.name if meta.name else symbol
    base = meta.base_ticker
    primary = base if base != symbol else ""
    req = [kw for kw in (name, primary, base) if kw]

    # To DO: Return the cleaned name and the list of required keywords
    return name, req
