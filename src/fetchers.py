"""Source fetchers. Each returns a list of dicts in the same shape:

{"id", "source", "title", "url", "summary", "score", "published_at"}
"""
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import feedparser
import requests
from dateutil import parser as date_parser

from src import config

logger = logging.getLogger(__name__)


def _get(url: str, **kwargs) -> requests.Response:
    headers = {"User-Agent": config.USER_AGENT, **kwargs.pop("headers", {})}
    response = requests.get(url, headers=headers, timeout=config.HTTP_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response


def _fetch_each(keys: Iterable, fetch_one: Callable, label: str) -> list[dict]:
    """Keep partial results when some requests fail; raise only if all of them do."""
    items, errors = [], []
    for key in keys:
        try:
            items.extend(fetch_one(key))
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("%s request failed for %s: %s", label, key, exc)
            errors.append(exc)
    if errors and not items:
        raise RuntimeError(f"all {label} requests failed; last error: {errors[-1]}")
    return items


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _fetch_arxiv_day(day: datetime, cutoff: datetime) -> list[dict]:
    categories = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    stamp = day.strftime("%Y%m%d")
    params = {
        "search_query": f"({categories}) AND submittedDate:[{stamp}0000 TO {stamp}2359]",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": config.ARXIV_RESULTS_PER_DAY,
    }
    feed = feedparser.parse(_get(config.ARXIV_API_URL, params=params).text)

    items = []
    for entry in feed.entries:
        # arXiv reports query errors as a normal feed entry.
        if "api/errors" in entry.get("id", ""):
            raise ValueError(_clean(entry.get("summary", "arXiv API error")))
        published = date_parser.parse(entry.published)
        if published < cutoff:
            continue
        # Drop the version suffix so a revised paper is not treated as new.
        arxiv_id = re.sub(r"v\d+$", "", entry.id.rsplit("/abs/", 1)[-1])
        items.append(
            {
                "id": f"arxiv:{arxiv_id}",
                "source": "arxiv",
                "title": _clean(entry.title),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "summary": _clean(entry.summary),
                "score": 0,
                "published_at": published.isoformat(),
            }
        )
    return items


def fetch_arxiv() -> list[dict]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=config.LOOKBACK_DAYS)
    days = [now - timedelta(days=n) for n in range(config.LOOKBACK_DAYS + 1)]

    def fetch_day(day: datetime) -> list[dict]:
        if day is not days[0]:
            time.sleep(config.ARXIV_REQUEST_DELAY_SECONDS)
        return _fetch_arxiv_day(day, cutoff)

    return _fetch_each(days, fetch_day, "arXiv")


def _fetch_huggingface_day(day: datetime) -> list[dict]:
    params = {"date": day.strftime("%Y-%m-%d"), "limit": 100}
    entries = _get(config.HUGGINGFACE_API_URL, params=params).json()

    papers = [entry["paper"] for entry in entries]
    popular = [p for p in papers if p.get("upvotes", 0) >= config.HUGGINGFACE_MIN_UPVOTES]
    popular.sort(key=lambda p: p["upvotes"], reverse=True)
    return [
        {
            "id": f"huggingface:{paper['id']}",
            "source": "huggingface",
            "title": _clean(paper["title"]),
            # Same URL as the arXiv fetcher, which is how main spots the overlap.
            "url": f"https://arxiv.org/abs/{paper['id']}",
            "summary": _clean(paper.get("summary", "")),
            "score": paper["upvotes"],
            "published_at": paper.get("publishedAt", ""),
        }
        for paper in popular[: config.HUGGINGFACE_TOP_PER_DAY]
    ]


def fetch_huggingface() -> list[dict]:
    now = datetime.now(timezone.utc)
    days = [now - timedelta(days=n) for n in range(config.LOOKBACK_DAYS + 1)]
    return _fetch_each(days, _fetch_huggingface_day, "Hugging Face")


def _get_reddit_token() -> str:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are not set")
    response = requests.post(
        config.REDDIT_TOKEN_URL,
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": config.USER_AGENT},
        timeout=config.HTTP_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _fetch_subreddit(subreddit: str, token: str) -> list[dict]:
    url = f"{config.REDDIT_API_URL}/r/{subreddit}/top"
    params = {"t": "month", "limit": config.REDDIT_LIMIT}
    headers = {"Authorization": f"Bearer {token}"}
    payload = _get(url, params=params, headers=headers).json()

    items = []
    for child in payload["data"]["children"]:
        post = child["data"]
        if post.get("stickied") or post.get("pinned") or post.get("distinguished"):
            continue
        if post.get("score", 0) < config.REDDIT_MIN_SCORE:
            continue
        published = datetime.fromtimestamp(post["created_utc"], timezone.utc)
        items.append(
            {
                "id": f"reddit:{post['id']}",
                "source": "reddit",
                "title": _clean(post["title"]),
                "url": f"https://www.reddit.com{post['permalink']}",
                "summary": _clean(post.get("selftext", "")),
                "score": post["score"],
                "published_at": published.isoformat(),
            }
        )
    return items


def fetch_reddit() -> list[dict]:
    token = _get_reddit_token()
    return _fetch_each(
        config.REDDIT_SUBREDDITS, lambda sub: _fetch_subreddit(sub, token), "Reddit"
    )


def _fetch_github_topic(topic: str) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS)
    params = {
        "q": f"topic:{topic} created:>{since:%Y-%m-%d}",
        "sort": "stars",
        "order": "desc",
        "per_page": config.GITHUB_RESULTS_PER_TOPIC,
    }
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = _get(config.GITHUB_API_URL, params=params, headers=headers).json()

    return [
        {
            "id": f"github:{repo['id']}",
            "source": "github",
            "title": repo["full_name"],
            "url": repo["html_url"],
            # The shared item shape has no language field, and the digest needs it.
            "summary": _clean(
                f"Primary language: {repo.get('language') or 'unknown'}. "
                f"{repo.get('description') or ''}"
            ),
            "score": repo["stargazers_count"],
            "published_at": repo["created_at"],
        }
        for repo in payload["items"]
    ]


def fetch_github() -> list[dict]:
    return _fetch_each(config.GITHUB_TOPICS, _fetch_github_topic, "GitHub")


FETCHERS = {
    "arxiv": fetch_arxiv,
    "huggingface": fetch_huggingface,
    "reddit": fetch_reddit,
    "github": fetch_github,
}
