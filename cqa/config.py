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
class RetrievalConfig:
    top_k: int
    rrf_k: int
    min_hits: int
    rrf_min_score: float


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
    raw: dict = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config" / "luxtrace.yaml"
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    repo_root = (PROJECT_ROOT / raw["repo"]["root"]).resolve()

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
        retrieval=RetrievalConfig(**raw["retrieval"]),
        graph=GraphConfig(**raw["graph"]),
        storage=StorageConfig(
            chroma_dir=(PROJECT_ROOT / raw["storage"]["chroma_dir"]).resolve(),
            index_state_file=(PROJECT_ROOT / raw["storage"]["index_state_file"]).resolve(),
            collection_name=raw["storage"]["collection_name"],
        ),
        raw=raw,
    )
