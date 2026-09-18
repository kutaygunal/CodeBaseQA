"""FastAPI backend for the CodebaseQA web chat UI. See PLAN.md §6 (extended: web UI)."""
from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from cqa.agent import Agent
from cqa.config import load_config
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
    get_agent().threads.delete(thread_id)
    return {"ok": True}


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
    """SSE stream of {type: retrieve|reason|tools|final|error} progress events.

    GET + browser-native EventSource, rather than a POST fetch ReadableStream —
    more broadly compatible with proxies/sandboxes that buffer chunked POST bodies.
    """
    agent = get_agent()
    q: queue.Queue = queue.Queue()
    SENTINEL = object()

    def worker():
        try:
            for event in agent.stream(
                question, thread_id=thread_id, provider=provider, model=model, module=module
            ):
                q.put(event)
        except Exception as e:  # noqa: BLE001
            q.put({"type": "error", "message": str(e)})
        finally:
            q.put(SENTINEL)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    async def event_gen():
        while True:
            item = await run_in_threadpool(q.get)
            if item is SENTINEL:
                break
            yield {"event": item["type"], "data": json.dumps(item)}

    return EventSourceResponse(event_gen())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=False)
