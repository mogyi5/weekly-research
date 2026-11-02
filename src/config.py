"""Central settings for the digest pipeline."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "digest_cache.db"
DIGESTS_DIR = ROOT_DIR / "digests"

HTTP_TIMEOUT = 30
USER_AGENT = "python:biweekly-research-digest:v1.0 (research digest bot)"

LOOKBACK_DAYS = 14

# arXiv has no popularity signal and these categories get hundreds of papers a
# day, so each day of the window is sampled separately. A single newest-first
# query would only ever cover the last day or two.
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ["cs.AI", "stat.ML"]
ARXIV_RESULTS_PER_DAY = 25
ARXIV_REQUEST_DELAY_SECONDS = 3  # arXiv asks for one request every 3 seconds

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API_URL = "https://oauth.reddit.com"
REDDIT_SUBREDDITS = ["MachineLearning", "LocalLLaMA"]
REDDIT_LIMIT = 30
REDDIT_MIN_SCORE = 15

GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_TOPICS = ["machine-learning"]
GITHUB_RESULTS_PER_TOPIC = 30

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
LLM_MAX_ATTEMPTS = 3
LLM_RETRY_DELAY_SECONDS = 20
SNIPPET_CHARS = 300
MAX_DIGEST_ITEMS = 20
