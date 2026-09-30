#!/usr/bin/env python3
"""Check long tool histories, prefix reuse and OpenCode's streaming contract.

Run against an exclusive C1 server with --context 196608. Synthetic fixtures
exercise the HTTP contract; they are not an agent task-success benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request


TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the project.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"], "additionalProperties": False}}}]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source(blocks):
    return "\n".join(
        f"# archived module {i:06d}\n"
        f"def normalize_record_{i:06d}(record):\n"
        "    name = record.get('name', '').strip()\n"
        "    retries = max(0, int(record.get('retries', 3)))\n"
        "    return {'name': name, 'retries': retries, 'enabled': True}\n"
        for i in range(blocks))


def history(blocks):
    config = "class Settings:\n" + "".join(
        f"    worker_{i:02d}_timeout = {100 + i}\n" for i in range(64))
    return [
        {"role": "system", "content": "You are a coding assistant. Follow the user's instructions."},
        {"role": "user", "content": "Read the archived source and config.py before editing."},
        {"role": "assistant", "content": "", "reasoning_content": "I need the source and configuration.",
         "tool_calls": [{"id": "archive-read", "type": "function", "function": {
             "name": "read_file", "arguments": '{"path":"archive.py"}'}}]},
        {"role": "tool", "tool_call_id": "archive-read", "content": source(blocks)},
        {"role": "assistant", "content": "", "reasoning_content": "Now inspect the settings.",
         "tool_calls": [{"id": "config-read", "type": "function", "function": {
             "name": "read_file", "arguments": '{"path":"config.py"}'}}]},
        {"role": "tool", "tool_call_id": "config-read", "content": config},
        {"role": "user", "content": "Copy config.py verbatim in a Python code block. No commentary."},
    ]


def call(url, body):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json", "X-Client-ID": "framework-opencode-check"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        if not body.get("stream"):
            result = json.load(response)
        else:
            message = {"role": "assistant", "content": "", "reasoning_content": ""}
            calls, usage, finish, done = {}, None, None, False
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(raw)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta", {})
                    for field in ("content", "reasoning_content"):
                        message[field] += delta.get(field) or ""
                    for part in delta.get("tool_calls", []):
                        target = calls.setdefault(part["index"], {
                            "type": "function", "function": {"name": "", "arguments": ""}})
                        if part.get("id"):
                            target["id"] = part["id"]
                        for field in ("name", "arguments"):
                            target["function"][field] += part.get("function", {}).get(field) or ""
            if not done or usage is None or finish is None:
                raise RuntimeError("incomplete OpenAI-compatible stream")
            if calls:
                message["tool_calls"] = [calls[i] for i in sorted(calls)]
            result = {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}
    return result, (time.monotonic() - started) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next-UD-Q4_K_XL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if baseline and not baseline.get("passed"):
        parser.error("comparison needs a completed, passing baseline report")
    with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=10) as response:
        card = next(m for m in json.load(response)["data"] if m["id"] == args.model)
    if card.get("context_length", 0) < 196608:
        raise RuntimeError("server must reserve 196608 total context tokens")
    if baseline and card["context_length"] != baseline["context"]:
        raise RuntimeError("baseline and candidate context capacities differ")
    report = {"schema": "gufo.framework-opencode.v1", "context": card["context_length"], "cases": {}}
    common = {"model": args.model, "max_tokens": 128, "temperature": 0,
              "top_k": 0, "top_p": 1, "seed": 73, "tools": TOOLS, "tool_choice": "none",
              "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(name, body):
        result, elapsed = call(args.url, body)
        usage = result["usage"]
        message = result["choices"][0]["message"]
        # Call IDs are generated by the server; compare the semantic call instead.
        normalized = {**message, "tool_calls": [
            {"type": c["type"], "function": c["function"]} for c in message.get("tool_calls", [])]}
        row = {"request_sha256": digest(body), "message_sha256": digest(normalized),
               "usage": usage, "wall_ms": elapsed, "finish_reason": result["choices"][0]["finish_reason"]}
        mismatch = None
        if baseline:
            control = baseline["cases"][name]
            if name != "streamed-tool-result" and row["request_sha256"] != control["request_sha256"]:
                mismatch = "request"
            for field in ("message_sha256", "finish_reason"):
                if row[field] != control[field]:
                    mismatch = field
            for field in ("prompt_tokens", "completion_tokens"):
                if usage[field] != control["usage"][field]:
                    mismatch = field
            row["exact_replay"] = mismatch is None
        report["cases"][name] = row
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{name}: prompt={usage['prompt_tokens']} cached={usage['cached_tokens']} "
              f"prefill={usage['gufo']['prefill_tokens']} wall={elapsed:.1f}ms", flush=True)
        if mismatch:
            raise RuntimeError(f"{name}: baseline {mismatch} differs")
        return result

    if baseline:
        blocks = baseline["archive_blocks"]
    else:
        empty, _ = call(args.url, {**common, "messages": history(0), "max_tokens": 1, "cache_prompt": False})
        probe, _ = call(args.url, {**common, "messages": history(64), "max_tokens": 1, "cache_prompt": False})
        overhead = empty["usage"]["prompt_tokens"]
        per_block = (probe["usage"]["prompt_tokens"] - overhead) / 64
        blocks = math.ceil((133120 - overhead) / per_block)
    report["archive_blocks"] = blocks
    messages = history(blocks)
    first = record("long-cold", {**common, "messages": messages, "cache_prompt": False})
    if not 129024 <= first["usage"]["prompt_tokens"] <= 163840:
        raise RuntimeError("long fixture is outside the 126Ki–160Ki input budget")
    if first["usage"]["cached_tokens"] or first["usage"]["completion_tokens"] != 128:
        raise RuntimeError("cold fixture did not prefill its whole input and generate tg128")
    config = messages[-2]["content"]
    messages += [first["choices"][0]["message"], {"role": "user", "content":
        "Copy this config.py verbatim in a Python code block. No commentary.\n```python\n" + config + "```"}]
    for sampled in (False, True):
        body = {**common, "messages": messages, "cache_prompt": True}
        if sampled:
            body.update(temperature=1.0, top_k=20, top_p=0.95)
        name = "long-sampled" if sampled else "long-greedy"
        result = record(name, body)
        repeated = record(name + "-replay", body)
        if digest(result["choices"]) != digest(repeated["choices"]):
            raise RuntimeError(f"{name}: cached seeded replay differs")
        for item in (result, repeated):
            if item["usage"]["completion_tokens"] == 0 or not item["choices"][0]["message"].get("content"):
                raise RuntimeError(f"{name}: continuation produced no content")
            if item["usage"]["cached_tokens"] < 129024 or item["usage"]["gufo"]["prefill_tokens"] > 2048:
                raise RuntimeError(f"{name}: long conversation prefix was recomputed")

    tool_body = {**common, "messages": [{"role": "user", "content":
        "Use read_file to inspect config.py. Do not answer before calling the tool."}],
        "tool_choice": "required", "max_tokens": 512, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
    tool = record("streamed-thinking-tool", tool_body)
    assistant = tool["choices"][0]["message"]
    calls = assistant.get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"]["name"] != "read_file" or not assistant["reasoning_content"]:
        raise RuntimeError("stream did not include reasoning and one read_file call")
    if json.loads(calls[0]["function"]["arguments"]) != {"path": "config.py"}:
        raise RuntimeError("streamed tool arguments are not the expected JSON")
    followup = record("streamed-tool-result", {
        **common, "messages": [*tool_body["messages"], assistant,
            {"role": "tool", "tool_call_id": calls[0]["id"], "content": "deployment_marker = 'alder-73'"},
            {"role": "user", "content": "Reply with only the deployment_marker value from that file."}],
        "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 32})
    if "alder-73" not in followup["choices"][0]["message"]["content"]:
        raise RuntimeError("tool-result continuation lost the returned file content")
    structured = {**common, "tools": [], "messages": [
        {"role": "user", "content": "Return JSON with a single key ok and the value true."}],
        "max_tokens": 32, "response_format": {"type": "json_schema", "json_schema": {
            "name": "status", "strict": True, "schema": {"type": "object",
            "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}}}}
    answer = record("structured-json", structured)
    if json.loads(answer["choices"][0]["message"]["content"]) != {"ok": True}:
        raise RuntimeError("structured response failed semantic check")
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
