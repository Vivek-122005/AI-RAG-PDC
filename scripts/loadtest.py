"""Light load test for a running API: latency percentiles, throughput and error mix.

    uv run python scripts/loadtest.py --requests 20 --concurrency 2 --key demo-key

Sized for a laptop: the server answers one question at a time (API_MAX_CONCURRENT_JOBS),
so concurrency mostly measures queueing, rate limiting and the cache. Use --pipeline
baseline for a quicker run and --unique to defeat the answer cache.
"""

import argparse
import asyncio
import statistics
import time
from collections import Counter

import httpx

QUESTIONS = [
    "What three actions can CRAG trigger?",
    "What does the RAGAS framework measure?",
    "Which reflection tokens does Self-RAG use?",
    "What retriever does the original RAG paper use?",
    "How does CRAG decide that retrieved documents are incorrect?",
]


async def one(client: httpx.AsyncClient, question: str, pipeline: str) -> tuple[int, float, bool]:
    start = time.perf_counter()
    try:
        r = await client.post("/v1/query", json={"question": question, "pipeline": pipeline})
        cached = r.status_code == 200 and bool(r.json().get("cached"))
        return r.status_code, time.perf_counter() - start, cached
    except httpx.HTTPError:
        return 0, time.perf_counter() - start, False


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


async def main(args: argparse.Namespace) -> None:
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    timeout = httpx.Timeout(args.timeout)
    limit = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=timeout) as client:

        async def run(i: int) -> tuple[int, float, bool]:
            q = QUESTIONS[i % len(QUESTIONS)]
            if args.unique:
                q = f"{q} (run {i})"
            async with limit:
                return await one(client, q, args.pipeline)

        start = time.perf_counter()
        results = await asyncio.gather(*(run(i) for i in range(args.requests)))
        wall = time.perf_counter() - start

    codes = Counter(code for code, _, _ in results)
    ok = [s for code, s, _ in results if code == 200]
    print(f"requests {args.requests}  concurrency {args.concurrency}  wall {wall:.1f}s")
    print(f"status   {dict(sorted(codes.items()))}   (0 = connection error / timeout)")
    print(f"cached   {sum(c for _, _, c in results)}")
    if ok:
        print(
            f"latency  p50 {pct(ok, 50):.2f}s  p95 {pct(ok, 95):.2f}s  max {max(ok):.2f}s"
            f"  mean {statistics.fmean(ok):.2f}s"
        )
        print(f"through  {len(ok) / wall:.2f} answers/s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--key", default="", help="API key (blank for an open dev server)")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--pipeline", choices=["agent", "baseline"], default="baseline")
    p.add_argument("--unique", action="store_true", help="vary questions to bypass the cache")
    p.add_argument("--timeout", type=float, default=600.0)
    asyncio.run(main(p.parse_args()))
