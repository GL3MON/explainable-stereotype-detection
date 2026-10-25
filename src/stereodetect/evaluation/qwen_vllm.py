"""Evaluate sentence-to-JSON StereoDetect via a thinking-enabled vLLM API."""

import argparse
import asyncio
import csv
import hashlib
import json
import time
from pathlib import Path

from stereodetect.models.qwen.json_task import (
    SYSTEM,
    detailed_report,
    load_rows,
    messages,
    parse_label,
)

FINAL_JSON_INSTRUCTION = (
    "FINAL ANSWER RULE: After </think>, immediately output exactly one valid JSON object "
    'containing only the keys "category", "target", and "label". '
    "Category and target must be strings; label must be an integer from 0 to 4. "
    "Start the final answer with { and finish with }. End the response immediately after }. "
    "If thinking is cut short, still output the JSON immediately. "
    "Do not continue the thinking text, explain, reconsider, add Markdown or code fences, "
    "or output a second answer."
)

THINKING_SYSTEM = SYSTEM.replace(
    "Do not include reasoning, Markdown, or any text outside the JSON object.",
    "Think briefly inside the thinking block. " + FINAL_JSON_INSTRUCTION,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def evaluate(args):
    import httpx
    from sklearn.metrics import classification_report

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.csv)
    total = len(rows)
    if args.limit:
        rows = rows[: args.limit]
    config = dict(
        vars(args),
        enable_thinking=True,
        system_prompt=THINKING_SYSTEM,
        user_prompt_suffix=FINAL_JSON_INSTRUCTION,
        input_columns=["Sentence"],
        scored_field="label",
        invalid_label=-1,
        dataset_rows=total,
        num_examples=len(rows),
        partial_evaluation=len(rows) < total,
        data_sha256=digest(args.csv),
        adapter_sha256=digest(Path(args.adapter) / "adapter_model.safetensors"),
    )
    config_path = output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Run configuration differs; choose a new output directory")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    records = {}
    completions = output / "completions.jsonl"
    if completions.exists():
        for line in completions.read_text().splitlines():
            record = json.loads(line)
            if record["index"] in records:
                raise ValueError("Duplicate saved example")
            records[record["index"]] = record
        for index, record in records.items():
            if index >= len(rows) or record["Sentence"] != rows[index]["Sentence"]:
                raise ValueError("Saved completions do not match dataset")
    tokenizer_data = json.loads((Path(args.adapter) / "tokenizer.json").read_text())
    thinking_end_id = next(
        t["id"] for t in tokenizer_data["added_tokens"] if t["content"] == "</think>"
    )
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), trust_env=False) as client:
        models = await client.get(args.base_url + "/models")
        models.raise_for_status()
        model_list = models.json()
        if args.model not in [m["id"] for m in model_list["data"]]:
            raise ValueError("Requested fine-tuned model is not served")
        (output / "served_models.json").write_text(json.dumps(model_list, indent=2) + "\n")
        with completions.open("a", encoding="utf-8") as handle:

            async def one(index, row):
                async with semaphore:
                    msg = messages(row)
                    msg[0]["content"] = THINKING_SYSTEM
                    msg[1]["content"] += "\n\n" + FINAL_JSON_INSTRUCTION
                    payload = {
                        "model": args.model,
                        "messages": msg,
                        "chat_template_kwargs": {"enable_thinking": True},
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "presence_penalty": args.presence_penalty,
                        "max_tokens": args.max_tokens,
                        "seed": 42,
                    }
                    if args.thinking_token_budget is not None:
                        payload["thinking_token_budget"] = args.thinking_token_budget
                        payload["return_token_ids"] = True
                    response = await client.post(args.base_url + "/chat/completions", json=payload)
                    response.raise_for_status()
                    raw = response.json()
                    choice = raw["choices"][0]
                    message = choice["message"]
                    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                    final = message.get("content") or ""
                    prediction = parse_label(final)
                    token_ids = choice.get("token_ids") or []
                    close_position = (
                        token_ids.index(thinking_end_id) if thinking_end_id in token_ids else None
                    )
                    record = {
                        "index": index,
                        **row,
                        "reasoning": reasoning,
                        "final_answer": final,
                        "prediction": prediction,
                        "correct": prediction == int(row["labels"]),
                        "invalid": prediction == -1,
                        "thinking_closed": close_position is not None
                        if args.thinking_token_budget is not None
                        else bool(final.strip()),
                        "reasoning_tokens_before_close": close_position,
                        "tokens_after_reasoning_close": len(token_ids) - close_position - 1
                        if close_position is not None
                        else None,
                        "truncated": choice["finish_reason"] == "length",
                        "finish_reason": choice["finish_reason"],
                        "generated_tokens": raw.get("usage", {}).get("completion_tokens", 0),
                        "response": raw,
                    }
                    records[index] = record
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    if len(records) % 50 == 0 or len(records) == len(rows):
                        print(
                            f"Evaluated {len(records)}/{len(rows)}; invalid={sum(r['invalid'] for r in records.values())}; truncated={sum(r['truncated'] for r in records.values())}; elapsed={time.perf_counter() - started:.1f}s",
                            flush=True,
                        )

            await asyncio.gather(*(one(i, row) for i, row in enumerate(rows) if i not in records))
    ordered = [records[i] for i in range(len(rows))]
    truth = [int(row["labels"]) for row in rows]
    predictions = [r["prediction"] for r in ordered]
    metrics = detailed_report(truth, predictions)
    report = dict(
        config,
        metrics=metrics,
        invalid_completions=sum(r["invalid"] for r in ordered),
        truncated_completions=sum(r["truncated"] for r in ordered),
        reasoning_completions=sum(bool(r["reasoning"].strip()) for r in ordered),
        thinking_closed_completions=sum(r["thinking_closed"] for r in ordered),
        max_reasoning_tokens_before_close=max(
            (
                r["reasoning_tokens_before_close"]
                for r in ordered
                if r.get("reasoning_tokens_before_close") is not None
            ),
            default=None,
        ),
        average_generated_tokens=sum(r["generated_tokens"] for r in ordered) / len(ordered),
        elapsed_seconds_this_invocation=time.perf_counter() - started,
    )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    text = classification_report(
        truth, predictions, labels=list(range(5)), digits=4, zero_division=0
    )
    text += f"\nAccuracy (all examples): {metrics['accuracy']:.6f}\nInvalid: {report['invalid_completions']}\nTruncated: {report['truncated_completions']}\n"
    (output / "classification_report.txt").write_text(text)
    fields = list(rows[0]) + [
        "prediction",
        "correct",
        "invalid",
        "thinking_closed",
        "truncated",
        "generated_tokens",
    ]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in ordered)
    print(text, flush=True)
    print("Saved", output / "evaluation.json", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--model", default="stereodetect-json")
    p.add_argument("--adapter", default="outputs/qwen35_2b_stereodetect_json/adapter")
    p.add_argument("--csv", default="data/test.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--thinking-token-budget", type=int, default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    if (
        args.thinking_token_budget is not None
        and not 0 <= args.thinking_token_budget < args.max_tokens
    ):
        p.error("Thinking budget must be nonnegative and below max-tokens")
    if (
        not 0 <= args.temperature <= 2
        or not 0 < args.top_p <= 1
        or not -2 <= args.presence_penalty <= 2
        or args.top_k < 1
    ):
        p.error("Invalid sampling parameters")
    if args.concurrency < 1 or args.max_tokens < 1 or (args.limit is not None and args.limit < 1):
        p.error("Concurrency, token budget, and limit must be positive")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
