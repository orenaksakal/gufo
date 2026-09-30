#!/usr/bin/env python3
"""Matched long-prefix OpenCode-style turn latency, including new checkpoints.

Keep the same C1 server/context/weights/toolchain in both arms. Each measured
branch appends a distinct approximately pp2048 coding request to one cached
tool history; exact repeated requests would skip the checkpoint under test.
"""

import argparse
import datetime
import json
import math
import mmap
from pathlib import Path
import runpy
import statistics
import urllib.request

helpers = runpy.run_path(str(Path(__file__).with_name("framework-opencode-check.py")))
call, digest, history = (helpers[name] for name in ("call", "digest", "history"))


def host_memory():
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {"available_bytes": int(fields["MemAvailable"].split()[0]) * 1024,
            "compact_stall": int(vm["compact_stall"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-UD-Q4_K_XL")
    parser.add_argument("--depth", type=int, default=32768)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--think", choices=("off", "on"), default="off")
    parser.add_argument("--resident-mib", type=int, default=0,
                        help="retain this much ordinary host RAM during turns (local server only)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.depth < 4096 or args.repetitions < 1 or args.resident_mib < 0:
        parser.error("depth must be at least 4096, repetitions positive and resident MiB nonnegative")
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    settings = {"depth": args.depth, "repetitions": args.repetitions, "output_budget": 128,
                "thinking": args.think == "on"}
    if args.resident_mib:
        settings["resident_mib"] = args.resident_mib
    if baseline and (not baseline.get("passed") or
                     {"thinking": False, **baseline["settings"]} != settings):
        parser.error("baseline must pass and use identical measurement settings")
    with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
        card = next(m for m in json.load(response)["data"] if m["id"] == args.model)
    context = card.get("context_length", 0)
    if context < args.depth + 4096:
        parser.error("server context must leave room for the appended turn")
    if baseline and baseline.get("context", context) != context:
        parser.error("baseline and candidate context capacities differ")
    common = {"model": args.model, "max_tokens": 128, "temperature": 0, "seed": 73,
              "top_k": 0, "top_p": 1, "tools": helpers["TOOLS"], "tool_choice": "none",
              "chat_template_kwargs": {"enable_thinking": settings["thinking"], "preserve_thinking": True}}
    # Keep calibration traffic/cache history identical in both arms.
    empty, _ = call(args.url, {**common, "messages": history(0), "max_tokens": 1, "cache_prompt": False})
    probe, _ = call(args.url, {**common, "messages": history(64), "max_tokens": 1, "cache_prompt": False})
    overhead = empty["usage"]["prompt_tokens"]
    blocks = math.ceil((args.depth - overhead) / ((probe["usage"]["prompt_tokens"] - overhead) / 64))
    if baseline and blocks != baseline["archive_blocks"]:
        raise RuntimeError("baseline and candidate prompt calibration differ")
    prefix = history(blocks)
    initial, elapsed = call(args.url, {**common, "messages": prefix, "cache_prompt": False})
    prompt_tokens = initial["usage"]["prompt_tokens"]
    if not args.depth <= prompt_tokens < args.depth + 128:
        raise RuntimeError("prefix outside calibrated depth tolerance")
    message = initial["choices"][0]["message"]
    if baseline and digest(message) != baseline["prefix_message_sha256"]:
        raise RuntimeError("prefix completion differs from baseline")
    report = {"schema": "gufo.framework-turn.v1", "settings": settings, "context": context,
              "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "archive_blocks": blocks, "prefix_tokens": prompt_tokens,
              "prefix_message_sha256": digest(message), "prefix_wall_ms": elapsed,
              "prefix_usage": initial["usage"], "warmups": {}, "cases": {}}
    # Controlled developer-workload pressure: pages are kept alive but idle
    # throughout the measured requests. Allocate after cold prefill, and leave
    # room for a fresh deep snapshot plus the server's other scratch buffers.
    resident = None
    if args.resident_mib:
        count = args.resident_mib << 20
        if host_memory()["available_bytes"] < count + (12 << 30):
            raise RuntimeError("resident-memory fixture needs 12 GiB free headroom")
        resident = mmap.mmap(-1, count)
        resident.madvise(mmap.MADV_NOHUGEPAGE)
        for offset in range(0, count, mmap.PAGESIZE):
            resident[offset] = 0x37
    report["host_before_turns"] = host_memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode, temperature in (("greedy", 0), ("sampled", 1)):
        samples = []
        for repetition in range(args.repetitions + 1):
            task = f"Coding turn {mode} {repetition}.\n" + (
                "Preserve existing behavior and every unrelated setting in this file.\n" * 96)
            task += "Copy this config.py verbatim in a Python code block. No commentary.\n```python\n" + prefix[-2]["content"] + "```"
            body = {**common, "messages": [*prefix, message, {"role": "user", "content": task}],
                    "temperature": temperature, "cache_prompt": True}
            if temperature:
                body.update(top_k=20, top_p=0.95)
            memory_before = host_memory()
            result, elapsed = call(args.url, body)
            memory_after = host_memory()
            usage = result["usage"]
            if usage["cached_tokens"] < prompt_tokens or usage["gufo"]["prefill_tokens"] == 0:
                raise RuntimeError("branch must reuse the deep prefix and prefill its new suffix")
            if usage["gufo"]["cache_snapshot_bytes"] == 0:
                raise RuntimeError("branch failed to exercise a new checkpoint")
            row = {"request_sha256": digest(body), "message_sha256": digest(result["choices"][0]["message"]),
                   "finish_reason": result["choices"][0]["finish_reason"], "usage": usage, "wall_ms": elapsed,
                   "host_available_bytes": memory_after["available_bytes"],
                   "host_compactions": memory_after["compact_stall"] - memory_before["compact_stall"]}
            if not repetition:
                report["warmups"][mode] = row
            if repetition:
                if baseline:
                    previous = baseline["cases"][mode]["samples"][repetition - 1]
                    row["exact_replay"] = all(row[k] == previous[k] for k in ("request_sha256", "message_sha256", "finish_reason")) and all(usage[k] == previous["usage"][k] for k in ("prompt_tokens", "completion_tokens", "cached_tokens"))
                samples.append(row)
                report["cases"][mode] = {"samples": samples,
                    "median_wall_ms": statistics.median(s["wall_ms"] for s in samples),
                    "median_snapshot_ms": statistics.median(s["usage"]["gufo"]["cache_snapshot_ms"] for s in samples)}
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{mode} {repetition}: cached={usage['cached_tokens']} "
                  f"pp={usage['gufo']['prefill_tokens']} tg={usage['completion_tokens']} "
                  f"snapshot={usage['gufo']['cache_snapshot_ms']:.1f}ms wall={elapsed:.1f}ms", flush=True)
            if baseline and repetition and not row["exact_replay"]:
                raise RuntimeError("branch completion differs from baseline")
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if resident is not None:
        resident.close()


if __name__ == "__main__":
    main()
