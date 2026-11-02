# Bi-Weekly Research Digest

Runs on the 1st and 15th of each month via GitHub Actions. It collects recent items from arXiv, Reddit and GitHub, skips anything already recorded in `data/digest_cache.db`, filters and summarizes the rest with Gemini, and commits a report to `digests/digest_YYYY-MM-DD.md`.

## Setup

1. Push this repository to GitHub with `main` as the default branch.
2. Add a `GEMINI_API_KEY` repository secret under **Settings > Secrets and variables > Actions**. `GITHUB_TOKEN` is provided automatically.
3. Optional, for Reddit: add `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` secrets from a Reddit app. Reddit reviews new API access requests, so this may take time or be declined. Until these are set, digests run without Reddit and say so in the footer.
4. Start a first run from the **Actions** tab with **Run workflow**.

## Running locally

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
```

```bash
pytest
```

```bash
python src/main.py
```

Without `GEMINI_API_KEY`, the run writes a clearly labelled mock digest. **It still marks items as seen**, so don't commit a `data/digest_cache.db` from a mock run: those items would be skipped by the real pipeline. In GitHub Actions a missing key fails the run instead.

## Configuration

Sources, thresholds and limits are in `src/config.py`. Set `GEMINI_MODEL` in the environment to override the default model.

## Failure behaviour

| Situation | Result |
| --- | --- |
| No new items | Logs it and exits 0 with no digest |
| One source fails | Digest is written and names the missing source in its footer |
| All sources fail | Exits 1 |
| Gemini call fails | Exits 1; items stay unmarked for the next run |
| Second run on the same day | Writes `digest_YYYY-MM-DD_2.md` rather than overwriting |
