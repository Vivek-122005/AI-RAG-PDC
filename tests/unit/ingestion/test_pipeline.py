from pathlib import Path

from agentic_rag.indexing.store import QdrantStore
from agentic_rag.ingestion.chunking import ChunkingConfig
from agentic_rag.ingestion.pipeline import ingest_path

CFG = ChunkingConfig(min_tokens=1)


def write_corpus(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "cats.md").write_text("# Cats\n\nCats purr when they are content and sleep a lot.\n")
    (root / "sub" / "dogs.txt").write_text("Dogs bark at strangers and love long walks.\n")
    (root / "ignored.bin").write_bytes(b"\x00\x01")


def test_ingest_is_idempotent_and_detects_changes(tmp_path: Path, store: QdrantStore) -> None:
    write_corpus(tmp_path)

    r1 = ingest_path(tmp_path, store, CFG)
    assert r1.results == {"cats.md": "added", "sub/dogs.txt": "added"}
    n = store.count()
    assert n == 2

    r2 = ingest_path(tmp_path, store, CFG)
    assert set(r2.results.values()) == {"unchanged"}
    assert store.count() == n

    (tmp_path / "cats.md").write_text(
        "# Cats\n\nCats hunt mice at night.\n\nThey also climb trees.\n"
    )
    r3 = ingest_path(tmp_path, store, CFG)
    assert r3.results["cats.md"] == "updated"
    assert r3.results["sub/dogs.txt"] == "unchanged"
    texts = [h.text for h in store.search("cats mice", limit=5)]
    assert any("mice" in t for t in texts)
    assert not any("purr" in t for t in texts)  # old chunks were replaced


def test_search_modes_and_citation(tmp_path: Path, store: QdrantStore) -> None:
    write_corpus(tmp_path)
    ingest_path(tmp_path, store, CFG)
    for mode in ("hybrid", "dense", "sparse"):
        hits = store.search("dogs bark", limit=1, mode=mode)
        assert hits[0].source == "sub/dogs.txt", mode
    hit = store.search("cats purr", limit=1)[0]
    assert hit.citation == "Cats § Cats"


def test_failed_file_does_not_abort_batch(tmp_path: Path, store: QdrantStore) -> None:
    (tmp_path / "good.txt").write_text("A perfectly fine document about tea.")
    (tmp_path / "bad.pdf").write_bytes(b"not really a pdf")
    r = ingest_path(tmp_path, store, CFG)
    assert r.results == {"bad.pdf": "failed", "good.txt": "added"}
