#!/usr/bin/env python3
"""Prepared C1/2/4/6/8 coding/prose regression on one fresh server per arm.

Use eight sessions, context 4096, MTP ceiling seven and thinking off. Every
session is prepared before each decode cohort; rates sum individual requests.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.serving_bench import PromptCase, reference_hashes_from_report, run_corpus_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-UD-Q4_K_XL")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-dirty", action="store_true")
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    fingerprint = json.loads(args.fingerprint.read_text())
    fingerprint = fingerprint.get("fingerprint", fingerprint)
    if (not isinstance(fingerprint.get("canonical"), dict) or
            not isinstance(fingerprint.get("fingerprintId"), str) or
            len(fingerprint["fingerprintId"]) != 64):
        parser.error("fingerprint must contain canonical inventory and a SHA256 ID")
    with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
        card = next(m for m in json.load(response)["data"] if m["id"] == args.model)
    context = card.get("context_length", 0)
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if context != 4096 or (baseline and
                           (not baseline.get("passed") or baseline.get("context") != context)):
        parser.error("both servers must use context 4096")
    fixtures = runpy.run_path(str(Path(__file__).with_name("framework-bench.py")))["cases"]()
    cases = [PromptCase(name, "framework", fixtures[name][0]) for name in ("prose", "edit-code")]
    common = dict(
        base_url=args.url, model=args.model, cases=cases,
        workload_id="framework-mixed-batch", max_tokens=128, temperature=0,
        concurrency_levels=[1, 2, 4, 6, 8], warmup_rounds=0,
        timeout_seconds=600, fingerprint=fingerprint, source_revision=args.source_revision,
        source_dirty=args.source_dirty,
        suite_bytes=json.dumps([c.__dict__ for c in cases], sort_keys=True).encode(),
        corpus_layout="distinct", cache_prompt=True, prefill_first=True,
        notes=["One fresh eight-session server per arm; context 4096, MTP cap seven, thinking off."])
    # Prompt preparation alone does not warm the wider speculative shapes.
    # Exercise complete tg128 cohorts at every width before measuring them.
    warmup = run_corpus_benchmark(**common, repetitions=1)
    hashes = reference_hashes_from_report(warmup)
    report = run_corpus_benchmark(**common, repetitions=3,
                                 reference={"source": "untimed-c1", "hashes": hashes})
    report["workload"]["fullDecodeWarmupRepetitions"] = 1
    report["warmup"] = {"completion_hashes": hashes,
                        "requests": {width: len(row["samples"])
                                     for width, row in warmup["results"].items()}}
    report["context"] = context
    report["source"]["buildMode"] = "cmake-release"
    if baseline and report["workload"] != baseline["workload"]:
        raise RuntimeError("baseline measurement settings or prompts differ")
    passed = True
    for width, row in report["results"].items():
        if any(s["completion_tokens"] != 128 for s in row["samples"]):
            raise RuntimeError(f"{width}: fixture did not fill tg128")
        if max(s["physical_execution_width"] for s in row["samples"]) != int(width[1:]):
            raise RuntimeError(f"{width}: server did not execute the requested batch width")
        passed = passed and row["completionExactness"]["exactRate"] == 1.0
        if baseline:
            fields = ("case_id", "prompt_tokens", "completion_tokens", "completion_sha256",
                      "cached_prompt_tokens")
            row["baseline_exact_replay"] = all(all(a[k] == b[k] for k in fields)
                for a, b in zip(row["samples"], baseline["results"][width]["samples"], strict=True))
            passed = passed and row["baseline_exact_replay"]
        print(f"{width}: {len(row['samples'])} measured requests; "
              f"replay={row.get('baseline_exact_replay', 'baseline')}", flush=True)
    report["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
