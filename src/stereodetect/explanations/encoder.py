"""LIME and Kernel SHAP on aligned word occurrences, adapting HEARTS to five labels."""

import argparse
import csv
import html
import json
import math
import os
import time
from pathlib import Path

from stereodetect.labels import LABEL_NAMES
from stereodetect.models.encoders.scoring import ProbabilityScorer
from stereodetect.runtime import package_versions


def agreement(a, b):
    import numpy as np

    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return {
        "cosine": float(np.dot(a, b) / norm) if norm > 1e-12 else None,
        "pearson": float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 1e-12 and np.std(b) > 1e-12
        else None,
    }


def select_rows(rows, count, seed):
    import numpy as np

    if count == 0 or count >= len(rows):
        return list(range(len(rows)))
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(5):
        quota = count // 5 + int(label < count % 5)
        candidates = [i for i, r in enumerate(rows) if int(r["labels"]) == label]
        errors = [i for i in candidates if int(rows[i]["prediction"]) != label]
        correct = [i for i in candidates if int(rows[i]["prediction"]) == label]
        rng.shuffle(errors)
        rng.shuffle(correct)
        chosen = errors[: quota // 2] + correct[: quota - min(len(errors), quota // 2)]
        chosen += [i for i in errors + correct if i not in chosen][: max(0, quota - len(chosen))]
        selected.extend(chosen)
    return sorted(selected)


def explain_text(text, scorer, mask_token, seed, lime_samples, shap_samples):
    import numpy as np
    import shap
    from lime.lime_text import IndexedString, LimeTextExplainer

    indexed = IndexedString(text, split_expression=r"\W+", bow=False, mask_string=mask_token)
    n = indexed.num_words()
    if n == 0:
        raise ValueError("No explainable words")
    words = [indexed.word(i) for i in range(n)]
    probabilities = scorer([text])[0]
    predicted = int(probabilities.argmax())
    labels = tuple(dict.fromkeys([predicted, 1]))
    lime = LimeTextExplainer(
        class_names=LABEL_NAMES,
        split_expression=r"\W+",
        bow=False,
        mask_string=mask_token,
        feature_selection="none",
        random_state=seed,
    )
    local = lime.explain_instance(
        text, scorer, labels=labels, num_features=n, num_samples=lime_samples
    )

    def masked_scores(masks):
        return scorer(
            [indexed.inverse_removing(np.flatnonzero(np.asarray(mask) < 0.5)) for mask in masks]
        )

    np.random.seed(seed)
    kernel = shap.KernelExplainer(
        masked_scores, np.zeros((1, n)), feature_names=words, link="identity"
    )
    values = np.asarray(
        kernel.shap_values(np.ones((1, n)), nsamples=shap_samples, l1_reg=0, silent=True)
    )
    if values.shape != (1, n, 5):
        raise ValueError(f"Unexpected SHAP shape {values.shape}")
    expected = np.asarray(kernel.expected_value)
    result = {
        "predicted_label": predicted,
        "probabilities": probabilities.tolist(),
        "words": words,
        "word_character_starts": [int(indexed.string_position(i)[0]) for i in range(n)],
        "shap_expected_probabilities": expected.tolist(),
        "lime_stereotype_local_r2": float(local.score),
        "explanations": {},
    }
    k = max(1, math.ceil(n * 0.2))
    random_words = np.random.default_rng(seed).choice(n, k, replace=False)
    for label in labels:
        lime_values = np.zeros(n)
        for feature, weight in local.local_exp[label]:
            lime_values[feature] = weight
        shap_values = values[0, :, label]
        additivity_error = float(abs(expected[label] + shap_values.sum() - probabilities[label]))
        if additivity_error > 1e-4:
            raise RuntimeError(f"SHAP additivity failed: {additivity_error}")
        variants = [
            indexed.inverse_removing(np.argsort(-lime_values)[:k]),
            indexed.inverse_removing(np.argsort(-shap_values)[:k]),
            indexed.inverse_removing(random_words),
        ]
        changed = scorer(variants)[:, label]
        result["explanations"][str(label)] = {
            **agreement(lime_values, shap_values),
            "lime_values": lime_values.tolist(),
            "shap_values": shap_values.tolist(),
            "lime_intercept": float(local.intercept[label]),
            "shap_additivity_error": additivity_error,
            "masked_word_count": k,
            "lime_top20pct_probability_drop": float(probabilities[label] - changed[0]),
            "shap_top20pct_probability_drop": float(probabilities[label] - changed[1]),
            "random20pct_probability_drop": float(probabilities[label] - changed[2]),
        }
    return result


def highlight(text, words, starts, values):
    import numpy as np

    maximum = max(float(np.abs(values).max()), 1e-12)
    fragments, previous = [], 0
    for word, start, score in zip(words, starts, values):
        fragments.append(html.escape(text[previous:start]))
        color = "220,70,50" if score >= 0 else "45,105,200"
        opacity = abs(float(score)) / maximum * 0.55
        fragments.append(
            f'<span style="background:rgba({color},{opacity:.3f})" title="Contribution {score:+.5f}">{html.escape(word)}</span>'
        )
        previous = start + len(word)
    fragments.append(html.escape(text[previous:]))
    return "".join(fragments)


def save_plot(record, path):
    import matplotlib

    matplotlib.use("Agg")
    import textwrap

    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(record["explanations"])
    fig, axes = plt.subplots(1, len(labels), figsize=(8 * len(labels), 7), squeeze=False)
    for axis, label in zip(axes[0], labels):
        detail = record["explanations"][label]
        lime = np.array(detail["lime_values"])
        values = np.array(detail["shap_values"])
        indices = np.sort(np.argsort(np.abs(lime) + np.abs(values))[-20:])
        y = np.arange(len(indices))
        axis.barh(y - 0.18, lime[indices], 0.35, color="#c65a35", label="LIME")
        axis.barh(y + 0.18, values[indices], 0.35, color="#397ba6", label="Kernel SHAP")
        axis.set_yticks(y, [f"{record['words'][i]} [{i}]" for i in indices])
        axis.invert_yaxis()
        axis.axvline(0, color="black", linewidth=0.6)
        axis.set_xlabel("Contribution to class probability")
        axis.set_title(f"{label}: {LABEL_NAMES[int(label)]}")
        axis.legend()
    fig.suptitle(
        "\n".join(textwrap.wrap(record["sentence"], 100))
        + f"\nGold {record['gold_label']}; predicted {record['predicted_label']}; correct={record['correct']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/albert_stereodetect/model")
    parser.add_argument("--display-name", default="ALBERT")
    parser.add_argument(
        "--indices-json",
        default=None,
        help="Reuse selected_test_indices from another explanation run",
    )
    parser.add_argument("--wandb-run-name", default="albert-lime-shap-explanations")
    parser.add_argument("--predictions", default="outputs/albert_stereodetect/test_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/albert_stereodetect/explanations")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Stratified diagnostic sample; 0 explains every test example",
    )
    parser.add_argument("--lime-samples", type=int, default=1000)
    parser.add_argument("--shap-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="stereodetect")
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    if args.num_examples < 0 or min(args.lime_samples, args.shap_samples, args.batch_size) < 1:
        parser.error("Invalid sample counts or batch size")
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "explanations.jsonl").exists():
        parser.error("Explanations already exist; choose a new --output-dir")
    (output / "figures").mkdir(exist_ok=True)
    with open(args.predictions, newline="") as handle:
        rows = list(csv.DictReader(handle))
    selection = "Balanced by gold class, up to half errors per class; diagnostic sample, not representative test statistics"
    if args.indices_json:
        manifest = json.loads(Path(args.indices_json).read_text())
        selected = manifest["selected_test_indices"]
        if (
            manifest["test_set_size"] != len(rows)
            or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < len(rows) for i in selected)
        ):
            parser.error("Invalid or incompatible selection manifest")
        selection = f"Paired diagnostic sample reused from {args.indices_json}; originally stratified and error-enriched for ALBERT; not representative test statistics"
    else:
        selected = select_rows(rows, args.num_examples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model_type = AutoConfig.from_pretrained(args.model_dir, local_files_only=True).model_type
    model_kwargs = {"attn_implementation": "sdpa"} if model_type == "modernbert" else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, local_files_only=True, **model_kwargs
    )
    scorer = ProbabilityScorer(model, tokenizer, args.batch_size, args.max_length)
    config = {
        **vars(args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selection": selection,
        "selected_test_indices": selected,
        "test_set_size": len(rows),
        "feature_space": "Word occurrences; LIME split_expression=\\W+, bow=False; identical masking for SHAP",
        "mask_token": tokenizer.mask_token,
        "shap_background": "All words masked",
        "shap_link": "identity",
        "shap_l1_reg": 0,
        "lime_kernel_width": 25,
        "lime_feature_selection": "none",
        "explained_classes": "Predicted class and stereotype class 1",
        "precision": "BF16 autocast; probability output float32",
        "hearts_reference": "https://aclanthology.org/2025.ijcnlp-long.1.pdf",
        "packages": package_versions("torch", "vllm", "transformers", "shap", "lime", "slicer"),
        "limitations": [
            "Word masking can create unnatural text.",
            "Attribution agreement is not calibrated correctness confidence.",
            "This adapts the HEARTS method to five-label StereoDetect, not its original binary EMGSD experiment.",
            "No gold word-importance annotations are available for token-hit-rate evaluation.",
        ],
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    records, summaries, sections = [], [], []
    started = time.perf_counter()
    with (output / "explanations.jsonl").open("w", encoding="utf-8") as handle:
        for number, index in enumerate(selected, 1):
            row = rows[index]
            text = row["Sentence"]
            record = explain_text(
                text,
                scorer,
                tokenizer.mask_token,
                args.seed + index,
                args.lime_samples,
                args.shap_samples,
            )
            record.update(
                {
                    "test_index": index,
                    "sentence": text,
                    "category": row.get("Category"),
                    "target": row.get("Target"),
                    "gold_label": int(row["labels"]),
                    "correct": record["predicted_label"] == int(row["labels"]),
                    "training_evaluation_prediction": int(row["prediction"]),
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
            detail = record["explanations"][str(record["predicted_label"])]
            summaries.append(
                {
                    "test_index": index,
                    "gold_label": record["gold_label"],
                    "predicted_label": record["predicted_label"],
                    "correct": record["correct"],
                    **{
                        k: detail[k]
                        for k in (
                            "cosine",
                            "pearson",
                            "shap_additivity_error",
                            "lime_top20pct_probability_drop",
                            "shap_top20pct_probability_drop",
                            "random20pct_probability_drop",
                        )
                    },
                    "lime_stereotype_local_r2": record["lime_stereotype_local_r2"],
                }
            )
            figure = f"test_{index:04d}.png"
            save_plot(record, output / "figures" / figure)
            content = ""
            for label, detail in record["explanations"].items():
                content += f"<h3>Class {label}: {html.escape(LABEL_NAMES[int(label)])}</h3>"
                for method in ("lime", "shap"):
                    content += f"<p><b>{method.upper()}:</b> {highlight(text, record['words'], record['word_character_starts'], np.array(detail[method + '_values']))}</p>"
                content += f"<p>Cosine: {detail['cosine']}; Pearson: {detail['pearson']}</p>"
            sections.append(
                f"<details><summary>Test {index}: gold {record['gold_label']}, predicted {record['predicted_label']}; correct={record['correct']}</summary>"
                + content
                + f'<a href="figures/{figure}">Comparison figure</a></details>'
            )
            scorer.cache.clear()
            print(
                f"Explained {number}/{len(selected)}; test {index}; correct={record['correct']}; elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    def mean(key, subset):
        values = [r[key] for r in subset if r[key] is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "num_explained": len(records),
        "selection": config["selection"],
        "elapsed_seconds": time.perf_counter() - started,
        "model_forward_examples": scorer.forward_examples,
        "mean_cosine": mean("cosine", summaries),
        "mean_pearson": mean("pearson", summaries),
        "mean_lime_probability_drop": mean("lime_top20pct_probability_drop", summaries),
        "mean_shap_probability_drop": mean("shap_top20pct_probability_drop", summaries),
        "mean_random_probability_drop": mean("random20pct_probability_drop", summaries),
        "max_shap_additivity_error": max(r["shap_additivity_error"] for r in summaries),
        "undefined_cosine_count": sum(r["cosine"] is None for r in summaries),
        "undefined_pearson_count": sum(r["pearson"] is None for r in summaries),
        "prediction_differences_from_training_evaluation": sum(
            r["predicted_label"] != r["training_evaluation_prediction"] for r in records
        ),
    }
    for name, correct in [("correct", True), ("incorrect", False)]:
        subset = [r for r in summaries if r["correct"] == correct]
        summary[name] = {
            "num_examples": len(subset),
            "mean_cosine": mean("cosine", subset),
            "mean_pearson": mean("pearson", subset),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>{display_name} LIME and SHAP</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;line-height:1.8}details{border:1px solid #ddd;padding:16px;margin:12px 0}summary{cursor:pointer}span{padding:2px}</style>'
    page += "<h1>{display_name} LIME and SHAP explanations</h1><p>Red supports a class; blue opposes it. Expand an example to compare methods. Agreement is not calibrated correctness confidence.</p>"
    page += f"<p>{len(records)} diagnostic examples, balanced by gold label and enriched with errors. Full classifier metrics use all {len(rows)} test examples.</p>"
    page += "<p>" + html.escape(selection) + "</p>"
    page = page.replace("{display_name}", html.escape(args.display_name))
    (output / "index.html").write_text(page + "".join(sections) + "</html>")
    if args.wandb:
        import wandb

        run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, job_type="explain", config=config
        )
        run.log({k: v for k, v in summary.items() if isinstance(v, (float, int))})
        run.summary["correct"] = summary["correct"]
        run.summary["incorrect"] = summary["incorrect"]
        (output / "wandb_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url}, indent=2) + "\n"
        )
        run.finish()
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Explanation browser: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
