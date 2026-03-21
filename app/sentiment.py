"""
Sentiment Analysis module for NeuCast.

Uses FinBERT (ProsusAI/finbert) to analyze financial news sentiment.
News are fetched from Yahoo Finance RSS / Google News RSS.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy-loaded transformer pipeline ──
_sentiment_pipeline = None
_FINBERT_MODEL = "ProsusAI/finbert"


def _get_pipeline():
    """Load FinBERT pipeline on first call (heavy, ~500MB)."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=_FINBERT_MODEL,
                tokenizer=_FINBERT_MODEL,
                device=-1,           # CPU (use 0 for GPU)
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
            return None
    return _sentiment_pipeline


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    url: str
    sentiment: str = "neutral"      # positive / negative / neutral
    score: float = 0.0              # confidence 0..1
    sentiment_value: float = 0.0    # -1..+1 numeric


@dataclass
class SentimentResult:
    news: list[NewsItem] = field(default_factory=list)
    avg_score: float = 0.0          # -1..+1
    positive_pct: float = 0.0       # 0..100
    negative_pct: float = 0.0
    neutral_pct: float = 0.0
    total_articles: int = 0
    signal: str = "neutral"         # bullish / bearish / neutral
    signal_strength: float = 0.0    # 0..1 abs(avg_score)


def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'").replace("&quot;", '"')
    return text.strip()


def fetch_news_yfinance(ticker: str, max_articles: int = 20) -> list[dict]:
    """Fetch news using yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news or []
        results = []
        for item in news[:max_articles]:
            title = item.get("title", "")
            if not title:
                continue
            results.append({
                "title": _clean_html(title),
                "source": item.get("publisher", "Unknown"),
                "published": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0)
                ).strftime("%Y-%m-%d %H:%M") if item.get("providerPublishTime") else "",
                "url": item.get("link", ""),
            })
        return results
    except Exception as e:
        logger.warning(f"yfinance news fetch failed: {e}")
        return []


def fetch_news_rss(ticker: str, max_articles: int = 15) -> list[dict]:
    """Fetch news from Google News RSS as fallback."""
    try:
        import urllib.request
        import xml.etree.ElementTree as ET

        url = (
            f"https://news.google.com/rss/search?"
            f"q={ticker}+stock+when:7d&hl=en&gl=US&ceid=US:en"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read()

        root = ET.fromstring(xml_data)
        results = []
        for item in root.findall(".//item")[:max_articles]:
            title = item.findtext("title", "")
            source = item.findtext("source", "Unknown")
            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            if title:
                results.append({
                    "title": _clean_html(title),
                    "source": _clean_html(source),
                    "published": pub_date[:16] if pub_date else "",
                    "url": link,
                })
        return results
    except Exception as e:
        logger.warning(f"RSS news fetch failed: {e}")
        return []


def analyze_sentiment(ticker: str, max_articles: int = 20) -> SentimentResult:
    """
    Fetch news for ticker and analyze sentiment with FinBERT.

    Returns SentimentResult with per-article scores and aggregate signal.
    """
    # Fetch news from multiple sources
    articles = fetch_news_yfinance(ticker, max_articles)
    if len(articles) < 5:
        rss_articles = fetch_news_rss(ticker, max_articles - len(articles))
        # Deduplicate by title similarity
        existing_titles = {a["title"].lower()[:50] for a in articles}
        for a in rss_articles:
            if a["title"].lower()[:50] not in existing_titles:
                articles.append(a)
                existing_titles.add(a["title"].lower()[:50])

    if not articles:
        return SentimentResult(signal="no_data")

    # Analyze with FinBERT
    pipe = _get_pipeline()
    news_items = []

    if pipe is not None:
        titles = [a["title"] for a in articles]
        try:
            results = pipe(titles, batch_size=8)
        except Exception as e:
            logger.warning(f"FinBERT inference failed: {e}")
            results = [{"label": "neutral", "score": 0.5}] * len(titles)
    else:
        # Fallback: keyword-based sentiment (no transformers installed)
        results = []
        positive_words = {
            "surge", "jump", "rally", "gain", "soar", "rise", "bull",
            "up", "high", "record", "profit", "beat", "exceed", "strong",
            "growth", "upgrade", "buy", "outperform", "positive", "boom",
        }
        negative_words = {
            "drop", "fall", "crash", "decline", "loss", "bear", "down",
            "low", "plunge", "sell", "miss", "weak", "cut", "risk",
            "fear", "warning", "downgrade", "recession", "crisis", "fail",
        }
        for a in articles:
            words = set(a["title"].lower().split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            if pos > neg:
                results.append({"label": "positive", "score": min(0.5 + pos * 0.1, 0.9)})
            elif neg > pos:
                results.append({"label": "negative", "score": min(0.5 + neg * 0.1, 0.9)})
            else:
                results.append({"label": "neutral", "score": 0.6})

    for article, result in zip(articles, results):
        label = result["label"].lower()
        confidence = result["score"]

        if label == "positive":
            sentiment_value = confidence
        elif label == "negative":
            sentiment_value = -confidence
        else:
            sentiment_value = 0.0

        news_items.append(NewsItem(
            title=article["title"],
            source=article["source"],
            published=article["published"],
            url=article["url"],
            sentiment=label,
            score=round(confidence, 3),
            sentiment_value=round(sentiment_value, 3),
        ))

    # Aggregate
    values = [n.sentiment_value for n in news_items]
    avg = float(np.mean(values)) if values else 0.0
    total = len(news_items)

    pos_count = sum(1 for n in news_items if n.sentiment == "positive")
    neg_count = sum(1 for n in news_items if n.sentiment == "negative")
    neu_count = total - pos_count - neg_count

    if avg > 0.15:
        signal = "bullish"
    elif avg < -0.15:
        signal = "bearish"
    else:
        signal = "neutral"

    return SentimentResult(
        news=news_items,
        avg_score=round(avg, 3),
        positive_pct=round(100 * pos_count / total, 1) if total else 0,
        negative_pct=round(100 * neg_count / total, 1) if total else 0,
        neutral_pct=round(100 * neu_count / total, 1) if total else 0,
        total_articles=total,
        signal=signal,
        signal_strength=round(abs(avg), 3),
    )
