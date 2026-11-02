"""Entry point: fetch, deduplicate, synthesize, save, record state."""
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# `python src/main.py` puts src/ on the path rather than the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, db, fetchers, synthesizer  # noqa: E402

logger = logging.getLogger("digest")


def collect_items() -> tuple[list[dict], list[str]]:
    """Run every fetcher, returning the items and the names of any that failed."""
    items: dict[str, dict] = {}
    failed = []
    for name, fetch in fetchers.FETCHERS.items():
        try:
            results = fetch()
        except Exception:
            logger.exception("%s fetcher failed", name)
            failed.append(name)
            continue
        logger.info("%s: fetched %d items", name, len(results))
        items.update((item["id"], item) for item in results)
    return list(items.values()), failed


def unique_by_url(items: list[dict]) -> list[dict]:
    """A paper can arrive from both arXiv and Hugging Face; keep the one with votes."""
    best: dict[str, dict] = {}
    for item in items:
        current = best.get(item["url"])
        if current is None or item["score"] > current["score"]:
            best[item["url"]] = item
    return list(best.values())


def write_digest(body: str, failed_sources: list[str], deferred: int = 0) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    config.DIGESTS_DIR.mkdir(parents=True, exist_ok=True)

    # A second run on the same day must not overwrite the first one's items,
    # which are already marked as seen and would never come back.
    path = config.DIGESTS_DIR / f"digest_{today}.md"
    run = 2
    while path.exists():
        path = config.DIGESTS_DIR / f"digest_{today}_{run}.md"
        run += 1

    text = f"# Research Digest: {today}\n\n{body.strip()}\n"
    notes = []
    if failed_sources:
        notes.append(f"Sources unavailable for this run: {', '.join(failed_sources)}.")
    if deferred:
        notes.append(f"{deferred} items could not be assessed and are left for the next run.")
    if notes:
        text += "\n---\n" + "\n\n".join(f"*{note}*" for note in notes) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # A mock digest in CI would mark real items as seen, losing them for good.
    if os.getenv("GITHUB_ACTIONS") and not synthesizer.has_api_key():
        logger.error("GEMINI_API_KEY secret is not set; refusing to publish a mock digest")
        return 1

    db.init_db()
    items, failed = collect_items()
    if len(failed) == len(fetchers.FETCHERS):
        logger.error("All fetchers failed")
        return 1

    new_items = [item for item in items if not db.is_item_seen(item["id"])]
    if not new_items:
        logger.info("No new items found; nothing to do")
        return 0
    candidates = unique_by_url(new_items)
    logger.info(
        "%d new items out of %d fetched, %d after removing overlaps",
        len(new_items), len(items), len(candidates),
    )

    # Nothing is marked as seen until the digest is saved, so a failed LLM call
    # leaves these items available to the next run.
    try:
        kept, assessed = synthesizer.filter_items(candidates)
        deferred = len(candidates) - len(assessed)
        if kept:
            selected = synthesizer.select_top(kept)
            logger.info("Kept %d items, using %d", len(kept), len(selected))
            path = write_digest(synthesizer.synthesize_digest(selected), failed, deferred)
            logger.info("Saved %s", path)
        else:
            logger.info("Filter discarded every assessed item; no digest written")
    except Exception:
        logger.exception("Synthesis failed; items left unmarked for the next run")
        return 1

    # Mark overlapping copies too, so the arXiv copy cannot resurface next run.
    assessed_urls = {item["url"] for item in assessed}
    seen = [item for item in new_items if item["url"] in assessed_urls]
    db.mark_items_seen(seen)
    logger.info("Marked %d items as seen, left %d for next run", len(seen), deferred)
    return 0


if __name__ == "__main__":
    sys.exit(main())
