"""Module for fetching and filtering news headlines."""
from __future__ import annotations
import datetime as dt
from typing import List, Dict, Iterable
from urllib.parse import quote_plus
import feedparser


def build_query(name: str, ticker: str) -> str:
    """
    Build a Google News search query for a company.
    """
    # TO DO: Return a query combining company name, ticker, and finance keywords
    query_string = f"""'{name}' OR '{ticker}'
        (stock OR stocks OR share OR shares OR market OR markets OR finance OR financial)"""
    return query_string

def filter_titles(items: List[Dict[str, str]],
        required_keywords: Iterable[str] = ()) -> List[Dict[str, str]]:
    """
    Filter news items so that only those containing required keywords
    in their title are kept.
    """
    # TO DO: If no required keywords, return items unchanged
    if not required_keywords:
        return items
    # TO DO: Otherwise, keep only items whose title contains any keyword (case-insensitive)
    lower_keywords = [kw.lower() for kw in required_keywords]
    filtered = [] # type: List[Dict[str, str]]
    for item in items:
        title = item.get("title", "").lower()
        if any(kw in title for kw in lower_keywords):
            filtered.append(item)
    return filtered

def _google_news_rss_url(query: str, lang: str = "de", country: str = "DE") -> str:
    """
    Build a Google News RSS URL for a given query.
    """
    # TO DO: Encode the query with quote_plus, append "when:12h"
    q = quote_plus(query)  # URL-encode query
    q += "+when:12h"
    # TO DO: Construct and return the final RSS URL
    url=f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={country}&ceid={country}:{lang}"
    return url


def fetch_headlines(
    query: str,
    limit: int = 2,
    news: List = None,
    lookback_hours: int = 12,
    lang: str = "de",
    country: str = "DE",
) -> tuple[list[dict[str, str]], list]:
    """
    Fetch latest headlines from Google News RSS for a given query.
    """
    # TO DO: Build the RSS URL via _google_news_rss_url and parse it with feedparser
    if news is None:
        news = []
    url = _google_news_rss_url(query, lang=lang, country=country)
    feed = feedparser.parse(url)
    entries = feed.entries
    results = []
    now = dt.datetime.utcnow()
    for entry in entries:
        published = entry.get("published_parsed")
        if not published:
            continue
        pub_dt = dt.datetime(*published[:6])
        age_hours = (now - pub_dt).total_seconds() / 3600
        # TO DO: Filter entries by publication time (lookback_hours) and collect title/source/link
        if age_hours > lookback_hours:
            continue
        title = entry.get("title", "No title")
        link = entry.get("link", "")
        if (link,pub_dt) in news:
            continue
        else:
            news.append((link,pub_dt))
        source = entry.get("source", {}).get("title", "Unknown source")
        results.append({
            "title": title,
            "link": link,
            "source": source,
            "published": pub_dt.isoformat() + "Z",
        })
        # TO DO: Stop after collecting 'limit' items
        if len(results) >= limit:
            break
    for (l,p) in news:
        if (now - p).total_seconds() / 3600 > lookback_hours:
            news.remove((l,p))
    return results, news
