#!/usr/bin/env python3
"""Paired HTTP speed/byte-replay checks for the Framework Flash-Next fork.

Use the same model, capacity, draft ceiling, serving concurrency and toolchain
for both arms. Each case gets one untimed warmup; all timed requests bypass
prefix reuse. This is a regression check, not an independent model-quality eval.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.model_bench.llm import synthetic_text
from gufo.serving_bench import run_request


def cases():
    source = "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass Settings:\n"
    source += "".join(f"    worker_{i:02d}_timeout: int = {100 + i}\n" for i in range(72))
    source += "\ndef load_settings():\n    return Settings()\n"
    edit = (
        "Here is config.py. Change only worker_35_timeout to 999. Return the complete "
        "updated file in a Python code block, preserving every other line.\n\n```python\n"
        + source + "```"
    )
    return {
        "prose": (synthetic_text(777, 1680) + "\n\nExplain how virtual memory, page tables "
                  "and page faults work in a detailed technical essay.", 0.0),
        "copy-code": ("Copy this file exactly into a Python code block, with no commentary.\n```python\n"
                      + source + "```", 0.0),
        "edit-code": (edit, 0.0),
        "repetition": (synthetic_text(777, 1630) + "\n\nOutput the words red green blue "
                       "repeated 200 times. Only output the repeated words.", 0.0),
        "sampled-code": (edit, 1.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-UD-Q4_K_XL")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--cases", default=",".join(cases()))
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_tokens < 1:
        parser.error("repetitions and max-tokens must be positive")
    selected = args.cases.split(",")
    if len(set(selected)) != len(selected) or set(selected) - cases().keys():
        parser.error("cases must be distinct known case names")
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    report = {
        "schema": "gufo.framework-flash-next.v1", "label": args.label,
        "measured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "max_tokens": args.max_tokens, "repetitions": args.repetitions,
        "warmups_per_case": 1, "cache_prompt": False, "thinking": False,
        "cases": {},
    }
    if baseline and any(baseline[k] != report[k] for k in
                        ("schema", "max_tokens", "repetitions", "cache_prompt", "thinking")):
        parser.error("baseline measurement settings differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = True
    for name in selected:
        prompt, temperature = cases()[name]
        samples = []
        body = {"seed": 73, "chat_template_kwargs": {"enable_thinking": False}}
        if temperature:
            body.update(top_k=20, top_p=0.95)
        for repetition in range(args.repetitions + 1):
            result = run_request(
                base_url=args.base_url, model=args.model, prompt=prompt,
                max_tokens=args.max_tokens, temperature=temperature,
                timeout_seconds=600, client_id="framework-fork-bench",
                concurrency=1, repetition=repetition, request_index=0,
                case_id=name, cache_prompt=False, extra_body=body,
            ).public()
            if result["cached_prompt_tokens"] != 0:
                raise RuntimeError("cache bypass did not take effect")
            if result["completion_tokens"] != args.max_tokens:
                raise RuntimeError(f"{name}: fixture did not fill the fixed output budget")
            if repetition:
                samples.append(result)
        row = {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "temperature": temperature, "extra_body": body, "samples": samples,
            "median_decode_tps": statistics.median(s["decode_tokens_per_second"] for s in samples),
            "median_prefill_tps": statistics.median(s["prefill_tokens_per_second"] for s in samples),
            "median_wall_ms": statistics.median(s["wall_ms"] for s in samples),
        }
        if baseline:
            control = baseline["cases"][name]
            if any(row[k] != control[k] for k in ("prompt_sha256", "temperature", "extra_body")):
                raise RuntimeError(f"baseline input changed: {name}")
            fields = ("prompt_tokens", "completion_tokens", "completion_sha256")
            row["exact_replay"] = all(
                all(a[k] == b[k] for k in fields)
                for a, b in zip(samples, control["samples"], strict=True)
            )
            row["decode_gain_percent"] = 100 * (row["median_decode_tps"] / control["median_decode_tps"] - 1)
            passed = passed and row["exact_replay"]
        report["cases"][name] = row
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: {row['median_decode_tps']:.2f} decode tok/s; "
              f"{row['median_prefill_tps']:.2f} prefill tok/s; "
              f"replay={row.get('exact_replay', 'baseline')}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
