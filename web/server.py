"""FastAPI backend for the CodebaseQA web chat UI. See PLAN.md §6 (extended: web UI)."""
from __future__ import annotations

import json
import queue
import sys
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from cqa.agent import Agent
from cqa.config import load_config
from cqa.gittools import list_refs
from cqa.providers import provider_status, set_runtime_key
from cqa.tools import list_dir, read_full_file

app = FastAPI(title="CodebaseQA")

_cfg = load_config()
_agent: Agent | None = None
_agent_lock = threading.Lock()

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = Agent(_cfg)
    return _agent


@app.get("/")
def index(theme: str | None = Query(None)):
    """Serve the SPA shell. An optional ?theme=light|dark is baked into the initial
    <html> tag server-side — avoids a flash-of-wrong-theme on first paint, and (as a
    side effect) makes headless screenshots deterministic without waiting on client JS."""
    if theme not in ("light", "dark"):
        return FileResponse(str(STATIC_DIR / "index.html"))
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace('<html lang="en">', f'<html lang="en" data-theme="{theme}">', 1)
    return HTMLResponse(html)


@app.get("/api/health")
def health():
    index_state_path = _cfg.storage.index_state_file
    indexed = index_state_path.exists()
    state = json.loads(index_state_path.read_text(encoding="utf-8")) if indexed else None
    return {
        "ok": True,
        "indexed": indexed,
        "chat_model": _cfg.models.chat_model,
        "embed_model": _cfg.models.embed_model,
        "index": state,
        "features": {
            "rerank": _cfg.retrieval.rerank.enabled,
            "analyze": _cfg.retrieval.analyze.enabled,
            "expand": _cfg.retrieval.expand.enabled,
            "contextual": bool(state and state.get("contextual")),
        },
    }


@app.get("/api/models")
def models():
    """Provider/model availability for the Settings panel — see PLAN.md §5.1 (extended)."""
    return provider_status()


@app.get("/api/modules")
def modules():
    return {"modules": get_agent().modules}


class CredentialRequest(BaseModel):
    provider: str
    api_key: str


@app.post("/api/credentials")
def set_credentials(req: CredentialRequest):
    """Store a pasted API key/token for a provider — in memory only for this
    server process (never written to disk, never sent in a URL)."""
    try:
        set_runtime_key(req.provider, req.api_key.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.delete("/api/credentials/{provider}")
def clear_credentials(provider: str):
    try:
        set_runtime_key(provider, None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.get("/api/source")
def source(path: str = Query(...)):
    result = read_full_file(_cfg, path)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/tree")
def tree(path: str = Query("")):
    result = list_dir(_cfg, path)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# --- Threads (conversation sidebar) -------------------------------------


class RenameRequest(BaseModel):
    title: str


@app.get("/api/threads")
def list_threads():
    return {"threads": get_agent().threads.list()}


@app.post("/api/threads")
def create_thread():
    return get_agent().threads.create()


@app.get("/api/threads/{thread_id}/messages")
def thread_messages(thread_id: str):
    return {"messages": get_agent().history(thread_id)}


@app.patch("/api/threads/{thread_id}")
def rename_thread(thread_id: str, req: RenameRequest):
    get_agent().threads.rename(thread_id, req.title)
    return {"ok": True}


@app.delete("/api/threads/{thread_id}")
def delete_thread(thread_id: str):
    get_agent().delete_thread(thread_id)
    return {"ok": True}


# --- SSE plumbing ---------------------------------------------------------


def _sse(make_events):
    """Run `make_events()` (a generator of {type, ...} dicts) on a worker thread and stream
    each event as SSE, event name = its `type`.

    GET + browser-native EventSource, rather than a POST fetch ReadableStream —
    more broadly compatible with proxies/sandboxes that buffer chunked POST bodies.
    """
    q: queue.Queue = queue.Queue()
    SENTINEL = object()

    def worker():
        try:
            for event in make_events():
                q.put(event)
        except Exception as e:  # noqa: BLE001
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    async def event_gen():
        while True:
            item = await run_in_threadpool(q.get)
            if item is SENTINEL:
                break
            yield {"event": item["type"], "data": json.dumps(item)}

    return EventSourceResponse(event_gen())


# --- Chat -----------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    thread_id: str | None = None
    provider: str = "ollama"
    model: str | None = None
    module: str | None = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Non-streaming: returns the full answer at once."""
    agent = get_agent()
    return agent.ask(
        req.question, thread_id=req.thread_id, provider=req.provider, model=req.model, module=req.module
    )


@app.get("/api/chat/stream")
async def chat_stream(
    question: str = Query(...),
    thread_id: str | None = Query(None),
    provider: str = Query("ollama"),
    model: str | None = Query(None),
    module: str | None = Query(None),
):
    """SSE stream of {type: analyze|retrieve|reason|tools|token|token_reset|final|error} events."""
    agent = get_agent()
    return _sse(lambda: agent.stream(question, thread_id=thread_id, provider=provider, model=model, module=module))


# --- Feedback + stats (#17, #18) ------------------------------------------


class FeedbackRequest(BaseModel):
    turn_id: str
    rating: int  # 1 = 👍, -1 = 👎, 0 = clear
    reason: str | None = None
    comment: str | None = None
    correct_paths: list[str] | None = None


@app.post("/api/feedback")
def post_feedback(req: FeedbackRequest):
    try:
        ok = get_agent().turns.set_feedback(
            req.turn_id, req.rating, reason=req.reason, comment=req.comment, correct_paths=req.correct_paths
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="unknown turn")
    return {"ok": True}


@app.get("/api/feedback/export")
def export_feedback():
    """JSON Lines of every rated turn (question, answer, rating, correction, retrieval stats)."""
    rows = get_agent().turns.export_feedback()
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    return Response(body, media_type="application/x-ndjson",
                    headers={"Content-Disposition": "attachment; filename=feedback.jsonl"})


@app.get("/api/stats")
def stats():
    return get_agent().turns.stats()


# --- Review mode (#11) ------------------------------------------------------

_reviews: "OrderedDict[str, object]" = OrderedDict()  # review_id -> ReviewPrep (last 20)


class ReviewPrepareRequest(BaseModel):
    base: str | None = None
    head: str | None = None
    diff: str | None = None  # pasted unified diff (POST because it's far too big for a query string)


@app.get("/api/review/refs")
def review_refs():
    info = list_refs(_cfg)
    info["default_base"] = _cfg.review.default_base
    info["default_head"] = _cfg.review.default_head
    return info


@app.post("/api/review/prepare")
async def review_prepare(req: ReviewPrepareRequest):
    """Analyse a git range or pasted diff (no LLM, fast) and return a summary the UI can
    show before the user commits to a full review."""
    try:
        prep = await run_in_threadpool(get_agent().prepare_review, req.base, req.head, req.diff)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    rid = uuid.uuid4().hex
    _reviews[rid] = prep
    while len(_reviews) > 20:
        _reviews.popitem(last=False)
    return {"review_id": rid, "title": prep.title, "stats": prep.stats, "warnings": prep.warnings}


@app.get("/api/review/stream")
async def review_stream(
    review_id: str = Query(...),
    thread_id: str | None = Query(None),
    provider: str = Query("ollama"),
    model: str | None = Query(None),
):
    prep = _reviews.get(review_id)
    if prep is None:
        raise HTTPException(status_code=404, detail="unknown or expired review_id — prepare it again")
    agent = get_agent()
    return _sse(lambda: agent.stream_review(prep, thread_id=thread_id, provider=provider, model=model))


# --- Tours (#12) -----------------------------------------------------------


@app.get("/api/tour/stream")
async def tour_stream(
    target: str = Query(...),
    thread_id: str | None = Query(None),
    provider: str = Query("ollama"),
    model: str | None = Query(None),
    refresh: bool = Query(False),
):
    agent = get_agent()

    def events():
        try:
            prep = agent.prepare_tour(target)
        except ValueError as e:
            yield {"type": "error", "message": str(e)}
            return
        yield {"type": "tour", "target": prep.target, "kind": prep.kind, "title": prep.title}
        yield from agent.stream_tour(prep, thread_id=thread_id, provider=provider, model=model, refresh=refresh)

    return _sse(events)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
