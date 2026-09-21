"""Load config/luxtrace.yaml into a typed, path-resolved object."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RepoConfig:
    root: Path
    include_dirs: list[str]
    ignore_dirs: set[str]
    cpp_extensions: set[str]
    text_extensions: set[str]


@dataclass
class ChunkingConfig:
    target_tokens: int
    max_tokens: int
    min_tokens: int
    overlap_ratio: float


@dataclass
class ModelsConfig:
    chat_model: str
    chat_model_fallback: str
    embed_model: str


@dataclass
class RerankConfig:
    enabled: bool = False
    model: str = "jinaai/jina-reranker-v1-turbo-en"
    # 0 = order purely by cross-encoder score. >0 = fuse the reranker's ordering with the
    # original RRF ordering (RRF again, this weight on the original) so a strong first-stage
    # hit can't be evicted by one noisy cross-encoder score.
    fuse_weight: float = 0.0


@dataclass
class AnalyzeConfig:
    """Query rewriting / HyDE / sub-question planner (one small LLM call)."""

    enabled: bool = False
    max_queries: int = 3
    max_sub_questions: int = 3
    # Questions with fewer words than this (and no follow-up context) skip the LLM call.
    min_words: int = 6
    use_hyde: bool = True


@dataclass
class ExpandConfig:
    """Call-graph / include-graph neighbor expansion after ranking."""

    enabled: bool = False
    seeds: int = 3
    max_extra: int = 4


@dataclass
class RetrievalConfig:
    top_k: int
    rrf_k: int
    min_hits: int
    rrf_min_score: float
    candidate_k: int = 20
    # Task instruction prepended to each *query* before dense embedding (documents are
    # kept unprefixed). Qwen3-Embedding is instruction-aware; non-empty measurably helps
    # retrieval (see PLAN.md §3). Leave empty for uniform nomic-style semantic matching.
    embed_query_instruction: str = ""
    rerank: RerankConfig = field(default_factory=RerankConfig)
    analyze: AnalyzeConfig = field(default_factory=AnalyzeConfig)
    expand: ExpandConfig = field(default_factory=ExpandConfig)


@dataclass
class ContextualConfig:
    enabled: bool = False
    model: str | None = None  # None -> models.chat_model
    concurrency: int = 6


@dataclass
class IngestConfig:
    contextual: ContextualConfig = field(default_factory=ContextualConfig)


@dataclass
class ReviewConfig:
    max_diff_chars: int = 60000
    max_tool_rounds: int = 5
    default_base: str = "HEAD~1"
    default_head: str = "HEAD"


@dataclass
class GraphConfig:
    max_tool_rounds: int


@dataclass
class StorageConfig:
    chroma_dir: Path
    index_state_file: Path
    collection_name: str


@dataclass
class Config:
    repo: RepoConfig
    chunking: ChunkingConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    graph: GraphConfig
    storage: StorageConfig
    ingest: IngestConfig = field(default_factory=IngestConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    raw: dict = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config" / "luxtrace.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    repo_root = (PROJECT_ROOT / raw["repo"]["root"]).resolve()

    ret = dict(raw["retrieval"])
    retrieval = RetrievalConfig(
        **{k: v for k, v in ret.items() if k not in ("rerank", "analyze", "expand")},
        rerank=RerankConfig(**(ret.get("rerank") or {})),
        analyze=AnalyzeConfig(**(ret.get("analyze") or {})),
        expand=ExpandConfig(**(ret.get("expand") or {})),
    )
    ing = raw.get("ingest") or {}
    ingest = IngestConfig(contextual=ContextualConfig(**(ing.get("contextual") or {})))
    review = ReviewConfig(**(raw.get("review") or {}))

    return Config(
        repo=RepoConfig(
            root=repo_root,
            include_dirs=raw["repo"]["include_dirs"],
            ignore_dirs=set(raw["repo"]["ignore_dirs"]),
            cpp_extensions=set(raw["repo"]["cpp_extensions"]),
            text_extensions=set(raw["repo"]["text_extensions"]),
        ),
        chunking=ChunkingConfig(**raw["chunking"]),
        models=ModelsConfig(**raw["models"]),
        retrieval=retrieval,
        graph=GraphConfig(**raw["graph"]),
        storage=StorageConfig(
            chroma_dir=(PROJECT_ROOT / raw["storage"]["chroma_dir"]).resolve(),
            index_state_file=(PROJECT_ROOT / raw["storage"]["index_state_file"]).resolve(),
            collection_name=raw["storage"]["collection_name"],
        ),
        ingest=ingest,
        review=review,
        raw=raw,
    )
