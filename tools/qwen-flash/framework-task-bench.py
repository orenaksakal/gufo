#!/usr/bin/env python3
"""Bounded executable edit tasks through the real OpenCode V2 CLI.

Run serially against an exclusive Gufo server. Each task gets a fresh repository,
private OpenCode server/config/database, ten model steps and a wall-time limit.
The grader runs the original tests against only the resulting source file.
These small fixtures are task checks, not a general coding-quality score.
"""

import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def fixtures():
    return {
        "config-bounds": {
            "prompt": "Fix normalize() in solution.py. Preserve unknown fields and do not mutate the input. "
                      "Trim name (default empty). Missing retries defaults to 3; accept integers and trimmed "
                      "decimal integer strings with an optional + or - sign, clamp them to [0, 10], "
                      "and reject booleans, floats, underscores and malformed strings "
                      "with ValueError. Read and run test_contract.py. Change only solution.py.",
            "source": "def normalize(record):\n"
                      "    record['name'] = record.get('name', '').strip()\n"
                      "    record['retries'] = max(0, int(record.get('retries', 3)))\n"
                      "    return record\n",
            "tests": "import unittest\nfrom solution import normalize\n\n"
                     "class Contract(unittest.TestCase):\n"
                     "    def test_defaults_and_preservation(self):\n"
                     "        data = {'name': '  Ada ', 'extra': [1, 2]}\n"
                     "        self.assertEqual(normalize(data), {'name': 'Ada', 'retries': 3, 'extra': [1, 2]})\n"
                     "        self.assertEqual(data, {'name': '  Ada ', 'extra': [1, 2]})\n"
                     "        self.assertEqual(normalize({}), {'name': '', 'retries': 3})\n"
                     "    def test_bounds(self):\n"
                     "        for value, expected in [(0, 0), (-5, 0), (11, 10), (' 7 ', 7), (' +7 ', 7), ('-2', 0), ('30', 10)]:\n"
                     "            with self.subTest(value=value):\n"
                     "                self.assertEqual(normalize({'retries': value})['retries'], expected)\n"
                     "    def test_invalid(self):\n"
                     "        for value in [True, False, 2.0, None, '2.5', 'bad', '', '+', '1_0', '0x10']:\n"
                     "            with self.subTest(value=value), self.assertRaises(ValueError):\n"
                     "                normalize({'retries': value})\n",
        },
        "lru-recency": {
            "prompt": "Fix LRU in solution.py. Successful get() and put() on an existing key make it most recent. "
                      "get() returns its default only for missing keys, including when a stored value is falsy. "
                      "Inserting beyond capacity evicts the least recent key. Zero capacity stores nothing. "
                      "Negative capacity raises ValueError. Read and run test_contract.py. Change only solution.py.",
            "source": "from collections import OrderedDict\n\n"
                      "class LRU:\n"
                      "    def __init__(self, capacity):\n"
                      "        self.capacity = capacity\n"
                      "        self.data = OrderedDict()\n"
                      "    def get(self, key, default=None):\n"
                      "        return self.data.get(key) or default\n"
                      "    def put(self, key, value):\n"
                      "        self.data[key] = value\n"
                      "        if len(self.data) > self.capacity:\n"
                      "            self.data.popitem()\n",
            "tests": "import unittest\nfrom solution import LRU\n\n"
                     "class Contract(unittest.TestCase):\n"
                     "    def test_read_and_eviction(self):\n"
                     "        c = LRU(2)\n"
                     "        c.put('a', 1); c.put('b', 2)\n"
                     "        self.assertEqual(c.get('a'), 1)\n"
                     "        c.put('c', 3)\n"
                     "        self.assertIsNone(c.get('b'))\n"
                     "        self.assertEqual(c.get('a'), 1)\n"
                     "        self.assertEqual(c.get('c'), 3)\n"
                     "    def test_update_recency(self):\n"
                     "        c = LRU(2)\n"
                     "        c.put('a', 1); c.put('b', 2); c.put('a', 4); c.put('c', 3)\n"
                     "        self.assertIsNone(c.get('b'))\n"
                     "        self.assertEqual(c.get('a'), 4)\n"
                     "    def test_falsy_and_missing(self):\n"
                     "        for value in [0, False, '', None]:\n"
                     "            c = LRU(1); c.put('a', value)\n"
                     "            self.assertEqual(c.get('a', 'missing'), value)\n"
                     "            self.assertEqual(c.get('b', 'missing'), 'missing')\n"
                     "    def test_capacity(self):\n"
                     "        c = LRU(0); c.put('a', 1)\n"
                     "        self.assertIsNone(c.get('a'))\n"
                     "        with self.assertRaises(ValueError): LRU(-1)\n",
        },
    }


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def grade(source, tests, directory):
    directory.mkdir()
    (directory / "solution.py").write_text(source)
    (directory / "test_contract.py").write_text(tests)
    try:
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "-v", "test_contract"],
                                cwd=directory, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return {"passed": False, "timed_out": True, "output": "Grader exceeded 20 seconds."}
    return {"passed": result.returncode == 0, "returncode": result.returncode,
            "output": result.stdout + result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/opencode"))
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="gufo-edit-tasks-", dir=args.work_root)).resolve()
    template = json.loads((Path(__file__).resolve().parents[2] / "deploy/framework/opencode.json").read_text())
    template["providers"]["gufo"]["settings"]["baseURL"] = args.url
    model = template["providers"]["gufo"]["models"]["flash-next"]
    model["limit"]["output"] = 2048
    model["body"].update(temperature=0, top_p=1, top_k=0, seed=73)
    model["body"]["chat_template_kwargs"]["enable_thinking"] = False
    template["agents"] = {"build": {"steps": 10}}
    report = {"schema": "gufo.framework-tasks.v1", "label": args.label,
              "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "Fresh small repositories; CLI startup, inference and agent tools timed. External grading excluded.",
              "settings": {"temperature": 0, "seed": 73, "thinking": False, "max_steps": 10,
                           "context": 196608, "output_limit": 2048, "timeout_seconds": args.timeout},
              "work_directory": str(root), "cases": {}}
    if not args.prepare_only:
        report["opencode_version"] = subprocess.check_output([args.opencode, "--version"], text=True).strip()
    for name, fixture in fixtures().items():
        work = root / name
        work.mkdir()
        (work / "solution.py").write_text(fixture["source"])
        (work / "test_contract.py").write_text(fixture["tests"])
        (work / "AGENTS.md").write_text("Complete the requested change in solution.py and run python3 -m unittest -v.\n")
        subprocess.run(["git", "init", "--quiet", str(work)], check=True)
        config = json.loads(json.dumps(template))
        config["permissions"] = [{"action": "*", "resource": "*", "effect": "deny"}] + [
            {"action": tool, "resource": "*", "effect": "allow"} for tool in ("read", "glob", "grep")
        ] + [{"action": "edit", "resource": "solution.py", "effect": "allow"},
             {"action": "shell", "resource": "python3 -m unittest*", "effect": "allow"}]
        config_root = work / ".config"
        (config_root / "opencode").mkdir(parents=True)
        (config_root / "opencode/opencode.json").write_text(json.dumps(config, indent=2) + "\n")
        before = grade(fixture["source"], fixture["tests"], root / (name + "-before"))
        if before["passed"]:
            raise RuntimeError("fixture must fail before editing: " + name)
        row = {"prompt_sha256": digest(fixture["prompt"]), "tests_sha256": digest(fixture["tests"]),
               "original_source_sha256": digest(fixture["source"]), "original_grade": before}
        report["cases"][name] = row
        if args.prepare_only:
            continue
        # A nested invocation must not inherit the parent session or its logical
        # working directory. OpenCode resolves the Location from PWD as well as
        # the process cwd; subprocess.Popen(cwd=...) alone leaves PWD unchanged.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("OPENCODE")}
        env.update(PWD=str(work), XDG_CONFIG_HOME=str(config_root),
                   OPENCODE_DB=str(work / ".opencode.db"))
        log_path = args.output.with_name(args.output.stem + "-" + name + ".jsonl")
        error_path = log_path.with_suffix(".stderr.log")
        start = time.monotonic()
        timed_out = False
        with log_path.open("w") as log, error_path.open("w") as errors:
            process = subprocess.Popen([args.opencode, "run", "--standalone", "--format", "json",
                                        "--model", "gufo/flash-next", "--agent", "build", "--auto",
                                        fixture["prompt"]], cwd=work, env=env, stdout=log, stderr=errors,
                                       start_new_session=True)
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        row.update(wall_ms=(time.monotonic() - start) * 1000, timed_out=timed_out,
                   returncode=process.returncode, events=str(log_path))
        events = [json.loads(line) for line in log_path.read_text().splitlines() if line.startswith("{")]
        tools = {event["part"]["id"]: event["part"] for event in events if event.get("type") == "tool_use"}
        row["tool_calls"] = len(tools)
        row["tool_counts"] = dict(Counter(part["tool"] for part in tools.values()))
        row["tool_errors"] = sum(part["state"]["status"] != "completed" for part in tools.values())
        row["model_steps"] = sum(event.get("type") == "step_start" for event in events)
        row["events_sha256"] = digest(log_path.read_text())
        final_source = (work / "solution.py").read_text() if (work / "solution.py").is_file() else ""
        row["final_source_sha256"] = digest(final_source)
        row["tests_unchanged"] = (work / "test_contract.py").read_text() == fixture["tests"]
        row["grade"] = grade(final_source, fixture["tests"], root / (name + "-grade"))
        row["passed"] = (not timed_out and process.returncode == 0 and row["tests_unchanged"]
                         and row["grade"]["passed"] and final_source != fixture["source"])
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, "passed=" + str(row["passed"]), "tools=" + str(row["tool_calls"]),
              "wall_ms=" + str(round(row["wall_ms"], 1)), flush=True)
    report["prepared_only"] = args.prepare_only
    report["passed"] = not args.prepare_only and all(row["passed"] for row in report["cases"].values())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if args.prepare_only or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
