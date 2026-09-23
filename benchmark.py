from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


STATE = {
    "subject": "決済障害",
    "body": "購入者全員が支払いできません。至急、技術担当に調査してほしいです。",
}


def questions(count: int) -> dict[str, dict[str, str]]:
    return {
        f"outage_{index}": {
            "type": "noul",
            "instructions": f"サービス障害が発生していますか？ 判定番号 {index}",
        }
        for index in range(count)
    }


def request(url: str, count: int) -> tuple[float, float]:
    payload = json.dumps({"state": STATE, "questions": questions(count)}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/decision",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req) as response:
        body = json.load(response)
    return (time.perf_counter() - started) * 1000, float(body["local_runtime"]["elapsed_ms"])


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8788")
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    print("batch  p50_server_ms  p95_server_ms  p50_e2e_ms  questions_per_sec")
    for count in (1, 5, 10, 50):
        for _ in range(args.warmups):
            request(args.url, count)
        samples = [request(args.url, count) for _ in range(args.runs)]
        client = [sample[0] for sample in samples]
        server = [sample[1] for sample in samples]
        p50 = statistics.median(server)
        print(
            f"{count:>5}  {p50:>13.2f}  {percentile(server, 0.95):>13.2f}"
            f"  {statistics.median(client):>10.2f}  {count / (p50 / 1000):>17.2f}"
        )

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(request, args.url, 1) for _ in range(args.concurrency)]
        concurrent = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    print(
        f"concurrent_1q={args.concurrency} wall_ms={elapsed * 1000:.2f} "
        f"requests_per_sec={args.concurrency / elapsed:.2f} "
        f"median_response_ms={statistics.median(value[0] for value in concurrent):.2f}"
    )


if __name__ == "__main__":
    main()

