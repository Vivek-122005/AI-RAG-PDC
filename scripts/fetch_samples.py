"""Download a small public sample corpus (arXiv papers on RAG / self-correction).

The PDFs are not committed; run `make samples` to reproduce the dev corpus.
"""

from pathlib import Path

import httpx

SAMPLES = {
    "rag_lewis_2020.pdf": "https://arxiv.org/pdf/2005.11401",
    "self_rag_2023.pdf": "https://arxiv.org/pdf/2310.11511",
    "corrective_rag_2024.pdf": "https://arxiv.org/pdf/2401.15884",
    "ragas_2023.pdf": "https://arxiv.org/pdf/2309.15217",
}

DEST = Path(__file__).resolve().parent.parent / "corpus" / "samples"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for name, url in SAMPLES.items():
            path = DEST / name
            if path.exists():
                print(f"skip  {name}")
                continue
            r = client.get(url)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError(f"{url} did not return a PDF")
            path.write_bytes(r.content)
            print(f"saved {name} ({len(r.content) // 1024} KiB)")


if __name__ == "__main__":
    main()
