from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import httpx
import pytest
import requests
from google import genai
from google.genai import errors as genai_errors

from src import config, db, fetchers, main, synthesizer

ITEM_KEYS = {"id", "source", "title", "url", "summary", "score", "published_at"}


def make_item(item_id="arxiv:1", source="arxiv", score=0):
    return {
        "id": item_id,
        "source": source,
        "title": f"Title {item_id}",
        "url": f"https://example.com/{item_id}",
        "summary": "Summary",
        "score": score,
        "published_at": "2026-11-01T00:00:00+00:00",
    }


def json_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    return response


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "test.db")
    monkeypatch.setattr(config, "DIGESTS_DIR", tmp_path / "digests")
    monkeypatch.setattr(config, "ARXIV_REQUEST_DELAY_SECONDS", 0)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    db.init_db()


# --- Database ---


def test_unknown_item_is_not_seen():
    assert db.is_item_seen("arxiv:1") is False


def test_marked_item_is_seen_and_stored():
    db.mark_items_seen([make_item("arxiv:1")])
    assert db.is_item_seen("arxiv:1") is True
    assert db.is_item_seen("arxiv:2") is False


def test_marking_twice_does_not_raise():
    db.mark_items_seen([make_item("arxiv:1")])
    db.mark_items_seen([make_item("arxiv:1"), make_item("arxiv:2")])
    assert db.is_item_seen("arxiv:2") is True


# --- arXiv ---


def arxiv_feed(published, arxiv_id="2611.00001v2"):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <published>{published}</published>
    <title>A  Paper
      Title</title>
    <summary>An abstract.</summary>
  </entry>
</feed>"""


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_arxiv_parses_recent_paper_and_strips_version(monkeypatch):
    monkeypatch.setattr(config, "LOOKBACK_DAYS", 0)
    recent = datetime.now(timezone.utc) + timedelta(minutes=1)
    get = MagicMock(return_value=MagicMock(text=arxiv_feed(iso(recent))))
    monkeypatch.setattr(fetchers.requests, "get", get)

    items = fetchers.fetch_arxiv()

    assert len(items) == 1
    assert set(items[0]) == ITEM_KEYS
    assert items[0]["id"] == "arxiv:2611.00001"
    assert items[0]["title"] == "A Paper Title"
    assert items[0]["url"] == "https://arxiv.org/abs/2611.00001"
    query = get.call_args.kwargs["params"]["search_query"]
    assert "cat:cs.AI OR cat:stat.ML" in query
    assert "submittedDate:[" in query


def test_arxiv_samples_every_day_of_the_window(monkeypatch):
    recent = datetime.now(timezone.utc)
    get = MagicMock(return_value=MagicMock(text=arxiv_feed(iso(recent))))
    monkeypatch.setattr(fetchers.requests, "get", get)

    fetchers.fetch_arxiv()

    assert get.call_count == config.LOOKBACK_DAYS + 1


def test_arxiv_drops_papers_older_than_window(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=config.LOOKBACK_DAYS + 5)
    monkeypatch.setattr(
        fetchers.requests, "get", MagicMock(return_value=MagicMock(text=arxiv_feed(iso(old))))
    )
    assert fetchers.fetch_arxiv() == []


def test_arxiv_error_entry_raises_when_every_request_fails(monkeypatch):
    feed = arxiv_feed(iso(datetime.now(timezone.utc)), arxiv_id="x").replace(
        "http://arxiv.org/abs/x", "http://arxiv.org/api/errors#bad_query"
    )
    monkeypatch.setattr(
        fetchers.requests, "get", MagicMock(return_value=MagicMock(text=feed))
    )
    with pytest.raises(RuntimeError):
        fetchers.fetch_arxiv()


# --- Reddit ---


def reddit_post(post_id, score=100, **overrides):
    post = {
        "id": post_id,
        "title": f"Post {post_id}",
        "permalink": f"/r/MachineLearning/comments/{post_id}/",
        "selftext": "Body",
        "score": score,
        "created_utc": 1793577600,
        "stickied": False,
        "pinned": False,
        "distinguished": None,
    }
    return {"data": {**post, **overrides}}


@pytest.fixture
def reddit_auth(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "test-id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "test-secret")
    post = MagicMock(return_value=json_response({"access_token": "test-token"}))
    monkeypatch.setattr(fetchers.requests, "post", post)
    return post


def test_reddit_requests_app_only_token(reddit_auth, monkeypatch):
    monkeypatch.setattr(
        fetchers.requests,
        "get",
        MagicMock(return_value=json_response({"data": {"children": []}})),
    )

    fetchers.fetch_reddit()

    assert reddit_auth.call_args.kwargs["auth"] == ("test-id", "test-secret")
    assert reddit_auth.call_args.kwargs["data"] == {"grant_type": "client_credentials"}


def test_reddit_without_credentials_raises_before_any_request(monkeypatch):
    get = MagicMock()
    post = MagicMock()
    monkeypatch.setattr(fetchers.requests, "get", get)
    monkeypatch.setattr(fetchers.requests, "post", post)

    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID"):
        fetchers.fetch_reddit()

    get.assert_not_called()
    post.assert_not_called()


def test_reddit_skips_stickies_mod_posts_and_low_scores(reddit_auth, monkeypatch):
    monkeypatch.setattr(config, "REDDIT_SUBREDDITS", ["MachineLearning"])
    payload = {
        "data": {
            "children": [
                reddit_post("keep"),
                reddit_post("sticky", stickied=True),
                reddit_post("pinned", pinned=True),
                reddit_post("mod", distinguished="moderator"),
                reddit_post("low", score=config.REDDIT_MIN_SCORE - 1),
            ]
        }
    }
    get = MagicMock(return_value=json_response(payload))
    monkeypatch.setattr(fetchers.requests, "get", get)

    items = fetchers.fetch_reddit()

    assert [i["id"] for i in items] == ["reddit:keep"]
    assert set(items[0]) == ITEM_KEYS
    assert items[0]["url"].startswith("https://www.reddit.com/r/")
    assert get.call_args.args[0] == "https://oauth.reddit.com/r/MachineLearning/top"
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert "python" in get.call_args.kwargs["headers"]["User-Agent"]
    assert get.call_args.kwargs["params"]["t"] == "month"


def test_reddit_keeps_results_when_one_subreddit_fails(reddit_auth, monkeypatch):
    monkeypatch.setattr(config, "REDDIT_SUBREDDITS", ["blocked", "ok"])
    ok = json_response({"data": {"children": [reddit_post("a")]}})
    monkeypatch.setattr(
        fetchers.requests,
        "get",
        MagicMock(side_effect=[requests.ConnectionError("403"), ok]),
    )
    assert [i["id"] for i in fetchers.fetch_reddit()] == ["reddit:a"]


def test_reddit_raises_when_every_subreddit_fails(reddit_auth, monkeypatch):
    monkeypatch.setattr(
        fetchers.requests, "get", MagicMock(side_effect=requests.ConnectionError("403"))
    )
    with pytest.raises(RuntimeError):
        fetchers.fetch_reddit()


# --- GitHub ---


GITHUB_PAYLOAD = {
    "items": [
        {
            "id": 42,
            "full_name": "acme/tool",
            "html_url": "https://github.com/acme/tool",
            "description": "Does things",
            "language": "Rust",
            "stargazers_count": 512,
            "created_at": "2026-10-30T00:00:00Z",
        }
    ]
}


def test_github_maps_repo_and_sends_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    get = MagicMock(return_value=json_response(GITHUB_PAYLOAD))
    monkeypatch.setattr(fetchers.requests, "get", get)

    items = fetchers.fetch_github()

    assert set(items[0]) == ITEM_KEYS
    assert items[0]["id"] == "github:42"
    assert items[0]["score"] == 512
    assert "Primary language: Rust" in items[0]["summary"]
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
    assert "topic:machine-learning created:>" in get.call_args.kwargs["params"]["q"]


def test_github_omits_auth_header_without_token(monkeypatch):
    get = MagicMock(return_value=json_response(GITHUB_PAYLOAD))
    monkeypatch.setattr(fetchers.requests, "get", get)
    fetchers.fetch_github()
    assert "Authorization" not in get.call_args.kwargs["headers"]


# --- Synthesizer ---


def api_error(code):
    return genai_errors.APIError(code, {"error": {"code": code, "message": "x", "status": "x"}})


@pytest.fixture
def gemini(monkeypatch):
    """Returns the mocked generate_content and the mocked sleep."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = MagicMock()
    monkeypatch.setattr(genai, "Client", MagicMock(return_value=client))
    sleep = MagicMock()
    monkeypatch.setattr(synthesizer.time, "sleep", sleep)
    return client.models.generate_content, sleep


def models_called(generate):
    return [call.kwargs["model"] for call in generate.call_args_list]


def test_gemini_retries_with_growing_delays(gemini):
    generate, sleep = gemini
    generate.side_effect = [api_error(503), api_error(503), MagicMock(text="ok")]

    assert synthesizer._call_gemini("prompt") == "ok"

    assert models_called(generate) == [config.GEMINI_MODEL] * 3
    assert [call.args[0] for call in sleep.call_args_list] == [30, 60]


def test_gemini_falls_back_when_primary_stays_unavailable(gemini):
    generate, _ = gemini
    attempts = config.LLM_MAX_ATTEMPTS
    generate.side_effect = [api_error(503)] * attempts + [MagicMock(text="ok")]

    assert synthesizer._call_gemini("prompt") == "ok"

    assert models_called(generate) == [config.GEMINI_MODEL] * attempts + [
        config.GEMINI_FALLBACK_MODELS[0]
    ]


def test_gemini_raises_when_every_model_is_unavailable(gemini):
    generate, _ = gemini
    generate.side_effect = api_error(503)

    with pytest.raises(genai_errors.APIError):
        synthesizer._call_gemini("prompt")

    models = 1 + len(config.GEMINI_FALLBACK_MODELS)
    assert generate.call_count == config.LLM_MAX_ATTEMPTS * models


@pytest.mark.parametrize(
    "error", [httpx.RemoteProtocolError("dropped"), httpx.ReadTimeout("too slow")]
)
def test_gemini_retries_transport_errors(gemini, error):
    generate, _ = gemini
    generate.side_effect = [error, MagicMock(text="ok")]

    assert synthesizer._call_gemini("prompt") == "ok"
    assert generate.call_count == 2


def test_filter_sends_batches_and_merges_ranks(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "FILTER_BATCH_SIZE", 2)
    responses = iter(['["b", "a"]', '["d", "c"]', '["e"]'])
    call = MagicMock(side_effect=lambda *a, **k: next(responses))
    monkeypatch.setattr(synthesizer, "_call_gemini", call)

    kept = synthesizer.filter_items([make_item(i) for i in "abcde"])

    assert call.call_count == 3
    assert [i["id"] for i in kept] == ["b", "d", "e", "a", "c"]


def test_gemini_does_not_retry_client_errors(gemini):
    generate, sleep = gemini
    generate.side_effect = api_error(401)

    with pytest.raises(genai_errors.APIError):
        synthesizer._call_gemini("prompt")

    assert generate.call_count == 1
    sleep.assert_not_called()


def test_filter_uses_model_ranking_and_ignores_unknown_ids(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        synthesizer, "_call_gemini", lambda *a, **k: '["b", "made-up", "a", "b"]'
    )
    items = [make_item("a"), make_item("b"), make_item("spam")]
    assert [i["id"] for i in synthesizer.filter_items(items)] == ["b", "a"]


def test_select_top_balances_sources_and_respects_limit():
    ranked = [make_item(f"a{n}", "arxiv") for n in range(10)] + [
        make_item("r0", "reddit"),
        make_item("g0", "github"),
    ]
    selected = synthesizer.select_top(ranked, limit=5)
    assert [i["id"] for i in selected] == ["a0", "r0", "g0", "a1", "a2"]


def test_mock_digest_has_all_sections_and_is_labelled():
    digest = synthesizer.synthesize_digest([make_item("a", "arxiv")])
    assert "Mock digest" in digest
    for heading in (
        "Executive Pulse",
        "Tooling & Repositories",
        "Research & Papers",
        "Community Signals",
    ):
        assert f"## {heading}" in digest


# --- Orchestration ---


def use_fetchers(monkeypatch, **fakes):
    monkeypatch.setattr(fetchers, "FETCHERS", fakes)


def digests():
    return sorted(config.DIGESTS_DIR.glob("*.md")) if config.DIGESTS_DIR.exists() else []


def test_main_writes_digest_and_marks_items(monkeypatch):
    use_fetchers(monkeypatch, arxiv=lambda: [make_item("arxiv:1")])

    assert main.main() == 0

    assert len(digests()) == 1
    assert "Title arxiv:1" in digests()[0].read_text(encoding="utf-8")
    assert db.is_item_seen("arxiv:1")


def test_main_second_run_finds_nothing_new(monkeypatch):
    use_fetchers(monkeypatch, arxiv=lambda: [make_item("arxiv:1")])
    main.main()

    assert main.main() == 0
    assert len(digests()) == 1


def test_main_same_day_run_does_not_overwrite(monkeypatch):
    use_fetchers(monkeypatch, arxiv=lambda: [make_item("arxiv:1")])
    main.main()
    use_fetchers(monkeypatch, arxiv=lambda: [make_item("arxiv:2")])
    main.main()

    assert len(digests()) == 2
    assert "Title arxiv:1" in digests()[0].read_text(encoding="utf-8")


def test_main_excludes_seen_items_from_digest(monkeypatch):
    db.mark_items_seen([make_item("arxiv:old")])
    use_fetchers(monkeypatch, arxiv=lambda: [make_item("arxiv:old"), make_item("arxiv:new")])

    main.main()

    text = digests()[0].read_text(encoding="utf-8")
    assert "arxiv:new" in text
    assert "arxiv:old" not in text


def test_main_reports_failed_source_in_digest(monkeypatch):
    def broken():
        raise RuntimeError("blocked")

    use_fetchers(monkeypatch, arxiv=lambda: [make_item("arxiv:1")], reddit=broken)

    assert main.main() == 0
    assert "Sources unavailable for this run: reddit" in digests()[0].read_text(encoding="utf-8")


def test_main_fails_when_all_sources_fail(monkeypatch):
    def broken():
        raise RuntimeError("blocked")

    use_fetchers(monkeypatch, arxiv=broken, reddit=broken)
    assert main.main() == 1


def test_main_leaves_items_unmarked_when_synthesis_fails(monkeypatch):
    def broken(_items):
        raise RuntimeError("LLM down")

    use_fetchers(monkeypatch, arxiv=lambda: [make_item("arxiv:1")])
    monkeypatch.setattr(synthesizer, "synthesize_digest", broken)

    assert main.main() == 1
    assert not db.is_item_seen("arxiv:1")
    assert digests() == []


def test_main_refuses_mock_digest_in_ci(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    fetch = MagicMock(return_value=[make_item("arxiv:1")])
    use_fetchers(monkeypatch, arxiv=fetch)

    assert main.main() == 1
    fetch.assert_not_called()
    assert not db.is_item_seen("arxiv:1")
