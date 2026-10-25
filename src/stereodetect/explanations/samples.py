"""Build a readable sample report with saved attributions and actual masking checks."""

import argparse
import csv
import html
import json
from pathlib import Path

from stereodetect.explanations.encoder import highlight
from stereodetect.labels import LABEL_NAMES
from stereodetect.models.encoders.scoring import ProbabilityScorer

NOTES = {
    1713: (
        "Clear stereotype cue",
        "Both methods rank businesslike above the other words for stereotype class 1. This is a correct prediction under the dataset annotation; a positive-sounding trait can still be classified as a stereotype.",
    ),
    377: (
        "Context changes word contributions",
        "Chatty strongly supports stereotype class 1, while amicable receives a negative contribution. Positive/negative attribution describes the chosen class score in this sentence, not whether the word itself sounds positive or negative.",
    ),
    1689: (
        "Anti-stereotype example",
        "The annotation and prediction are anti-stereotype, class 0. The main contributions come from Physicists and dumb. Attribution signs concern class 0, not stereotype class 1 or the moral meaning of dumb.",
    ),
    946: (
        "Explain a phrase together",
        "The large model correctly predicts neutral without target, class 2. Mechanical Turk is treated as a grouped feature below, alongside the original word-level scores. A social-group interpretation of Turk alone is not justified by these attributions.",
    ),
    346: (
        "Bias-class example",
        "The model agrees with the dataset label Bias, class 4. The group name and subjective framing can be investigated with the word scores and masking checks. Correct classification does not establish that every attribution is a valid linguistic reason.",
    ),
    219: (
        "Contrast with the bias example",
        "This short group-generalization sentence is annotated as stereotype, class 1. Compare it with the personal-opinion sentence in Test 346, which the dataset labels Bias. The report explains the learned distinction rather than endorsing either statement.",
    ),
    4: (
        "Correct prediction with diffuse attribution",
        "The historical sentence is correctly classified as neutral with target, class 3. Importance is spread across content and function words. This is a caution case: a correct prediction can still yield a difficult-to-interpret word explanation.",
    ),
    58: (
        "Explanation of a wrong prediction",
        "The dataset label is Bias, class 4; the model predicts anti-stereotype, class 0. Both explainers can agree strongly about the wrong decision. The good and Arabs contributions describe the model response, not evidence that its label is correct.",
    ),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", default="outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407"
    )
    parser.add_argument("--indices", default="1713,377,1689,946,346,219,4,58")
    args = parser.parse_args(argv)
    import numpy as np
    import shap
    from lime.lime_text import IndexedString, LimeTextExplainer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    run = Path(args.run_dir)
    output = run / "explanations" / "samples"
    output.mkdir(parents=True, exist_ok=True)
    saved = {
        r["test_index"]: r
        for r in map(
            json.loads, (run / "explanations" / "explanations.jsonl").read_text().splitlines()
        )
    }
    selected = [int(x) for x in args.indices.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(run / "model", local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        run / "model", local_files_only=True, attn_implementation="sdpa"
    )
    scorer = ProbabilityScorer(model, tokenizer, batch_size=128, max_length=128)
    sections = []
    records = []
    checks = []
    for index in selected:
        original = saved[index]
        text = original["sentence"]
        label = original["predicted_label"]
        indexed = IndexedString(
            text, split_expression=r"\W+", bow=False, mask_string=tokenizer.mask_token
        )
        n = indexed.num_words()
        probabilities = scorer([text])[0]
        assert int(probabilities.argmax()) == label, "Prediction changed during verification"
        lime = LimeTextExplainer(
            class_names=LABEL_NAMES,
            split_expression=r"\W+",
            bow=False,
            mask_string=tokenizer.mask_token,
            feature_selection="none",
            random_state=42 + index,
        )
        fit = lime.explain_instance(text, scorer, labels=(label,), num_features=n, num_samples=1000)
        variants = [indexed.inverse_removing(np.array([i])) for i in range(n)]
        masked_scores = scorer(variants)
        single = []
        for i in range(n):
            check = {
                "test_index": index,
                "word_index": i,
                "word": indexed.word(i),
                "masked_sentence": variants[i],
                "explained_label": label,
                "original_probability": float(probabilities[label]),
                "masked_probability": float(masked_scores[i, label]),
                "probability_drop": float(probabilities[label] - masked_scores[i, label]),
                "masked_predicted_label": int(masked_scores[i].argmax()),
            }
            single.append(check)
            checks.append(check)
        detail = dict(original["explanations"][str(label)])
        lime_values = np.zeros(n)
        for feature, value in fit.local_exp[label]:
            lime_values[feature] = value
        detail["lime_values"] = lime_values.tolist()
        top = np.argsort(-np.array(detail["shap_values"]))[:2].tolist()
        if index == 946:
            top = [0, 1]
        if index == 377:
            top = [3, 5]
        both = indexed.inverse_removing(np.array(top))
        both_prob = scorer([both])[0]
        interaction = {
            "word_a": indexed.word(top[0]),
            "word_b": indexed.word(top[1]),
            "full_probability": float(probabilities[label]),
            "mask_a_probability": float(masked_scores[top[0], label]),
            "mask_b_probability": float(masked_scores[top[1], label]),
            "mask_both_probability": float(both_prob[label]),
            "mask_both_prediction": int(both_prob.argmax()),
            "mask_both_sentence": both,
            "interaction_at_full_context": float(
                probabilities[label]
                - masked_scores[top[0], label]
                - masked_scores[top[1], label]
                + both_prob[label]
            ),
        }
        record = {
            "test_index": index,
            "sentence": text,
            "gold_label": original["gold_label"],
            "predicted_label": label,
            "probabilities": probabilities.tolist(),
            "correct": original["correct"],
            "lime_predicted_class_local_r2": float(fit.score),
            "lime_predicted_class_values": lime_values.tolist(),
            "single_word_mask_checks": single,
            "pair_mask_check": interaction,
            "saved_explanation": original,
        }
        grouped = None
        if index == 946:
            groups = [[0, 1]] + [[i] for i in range(2, n)]
            names = ["Mechanical Turk"] + [indexed.word(i) for i in range(2, n)]

            def group_scores(masks):
                strings = []
                for mask in masks:
                    removed = [
                        word for group, on in zip(groups, mask) if on < 0.5 for word in group
                    ]
                    strings.append(indexed.inverse_removing(np.array(removed, dtype=int)))
                return scorer(strings)

            np.random.seed(42 + index)
            kernel = shap.KernelExplainer(
                group_scores, np.zeros((1, len(groups))), feature_names=names, link="identity"
            )
            values = np.asarray(
                kernel.shap_values(np.ones((1, len(groups))), nsamples=512, l1_reg=0, silent=True)
            )
            error = abs(
                float(kernel.expected_value[label])
                + values[0, :, label].sum()
                - probabilities[label]
            )
            assert error < 1e-4
            grouped = {
                "group_names": names,
                "shap_values": values[0, :, label].tolist(),
                "expected_probability": float(kernel.expected_value[label]),
                "additivity_error": float(error),
                "interpretation": "Phrase groups change the coalition game; grouped attribution need not equal the sum of separate word attributions.",
            }
            record["grouped_shap"] = grouped
        records.append(record)
        title, note = NOTES.get(
            index,
            ("Explanation example", "Inspect word scores and actual masking responses together."),
        )
        heading = f"Test {index}: {title}"
        safe = html.escape
        content = f"<section><h2>{safe(heading)}</h2><blockquote>{safe(text)}</blockquote>"
        content += f"<p><b>Dataset:</b> {original['gold_label']} — {safe(LABEL_NAMES[original['gold_label']])}<br><b>Model:</b> {label} — {safe(LABEL_NAMES[label])}; softmax score {probabilities[label] * 100:.2f}%<br><b>Correct:</b> {original['correct']}</p>"
        content += f'<p class="note">{safe(note)}</p>'
        content += f"<p><b>LIME:</b> {highlight(text, original['words'], original['word_character_starts'], np.array(detail['lime_values']))}</p>"
        content += f"<p><b>SHAP:</b> {highlight(text, original['words'], original['word_character_starts'], np.array(detail['shap_values']))}</p>"
        content += "<table><tr><th>Word</th><th>LIME weight</th><th>SHAP contribution</th><th>Actual single-word masking drop</th></tr>"
        for i in np.argsort(-np.abs(np.array(detail["shap_values"]))):
            content += f"<tr><td>{safe(indexed.word(i))}</td><td>{detail['lime_values'][i]:+.4f}</td><td>{detail['shap_values'][i]:+.4f}</td><td>{single[i]['probability_drop'] * 100:+.2f} percentage points</td></tr>"
        content += "</table>"
        content += f"<p>LIME local fit R² for the <b>predicted class</b>: {fit.score:.3f}. Higher means a better fit on sampled perturbations; it is not correctness confidence.</p>"
        content += "<h3>Check interactions with other words held fixed</h3><table><tr><th>Input</th><th>Score for the original predicted class</th></tr>"
        for name, value in [
            ("Original sentence", interaction["full_probability"]),
            ("Mask " + interaction["word_a"], interaction["mask_a_probability"]),
            ("Mask " + interaction["word_b"], interaction["mask_b_probability"]),
            ("Mask both", interaction["mask_both_probability"]),
        ]:
            content += f"<tr><td>{safe(name)}</td><td>{value * 100:.2f}%</td></tr>"
        content += f"</table><p>Joint masking produces class {interaction['mask_both_prediction']}. The context-specific interaction contrast is {interaction['interaction_at_full_context']:+.4f}, calculated as p(full) − p(mask A) − p(mask B) + p(mask both). Nonzero values show non-additivity under these masks.</p>"
        if grouped:
            content += "<h3>Grouped SHAP: keep Mechanical Turk together</h3><table><tr><th>Feature group</th><th>Contribution to class 2 probability</th></tr>"
            for name, value in zip(grouped["group_names"], grouped["shap_values"]):
                content += f"<tr><td>{safe(name)}</td><td>{value:+.4f}</td></tr>"
            content += "</table><p>This allocation treats the phrase as one feature. It answers a different masking-game question from separate-word SHAP.</p>"
        content += f'<p><a href="../figures/test_{index:04d}.png">Original LIME/SHAP comparison figure</a></p></section>'
        sections.append(content)
        scorer.cache.clear()
        print(
            f"Prepared Test {index}; correct={record['correct']}; predicted-class LIME R²={fit.score:.3f}",
            flush=True,
        )
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    )
    with (output / "masking_checks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>ModernBERT-large explanation samples</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:20px;line-height:1.6}h1,h2,h3{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:6px;text-align:left}blockquote,.note{padding:12px;background:#f5f7f8}span{padding:2px}section{margin:30px 0;border-top:2px solid #ddd;padding-top:10px}@media print{section{break-before:page}tr{break-inside:avoid}table{font-size:10px}}</style>'
    page += "<h1>Readable explanation samples: ModernBERT-large</h1><p>These eight hand-selected illustrations come from the latest large-model run. They are examples for learning and diagnosis, not a representative benchmark or evidence of explanation correctness.</p>"
    page += '<div class="color-legend" style="border:1px solid #bbb;padding:12px;margin:12px 0;break-inside:avoid"><b>Highlight legend (LIME and SHAP)</b><p><span style="background:rgba(220,70,50,0.55);padding:4px 8px">Red: positive contribution (+)</span> Supports the class being explained.</p><p><span style="background:rgba(45,105,200,0.55);padding:4px 8px">Blue: negative contribution (−)</span> Opposes the class being explained.</p><p>Darker shading means a larger absolute contribution within that highlighted sentence. Pale or unshaded words have contributions close to zero. Colors refer to the predicted class shown for each sample; they do not mean correct/incorrect or good/bad. Color intensity is scaled separately for each explanation.</p></div>'
    page += "<p><b>Read the colors:</b> red supports the class being explained; blue opposes it. The model scores a whole sentence. LIME is an additive local approximation; SHAP distributes contributions across masked-word combinations. Neither score is an independent word probability.</p>"
    page += "<p>The actual single-word masking drop keeps all other words present. It can differ from SHAP because SHAP averages marginal effects across many other-word combinations. Pair masking and grouped SHAP below explicitly investigate interactions. Masked text can be unnatural, so these checks measure sensitivity rather than proving causal linguistic reasons.</p>"
    page += "<p>Gold labels are dataset annotations. Raw softmax scores are not calibrated correctness confidence. The separate full test report remains 93.33% accuracy and 93.39% macro F1.</p>"
    page += "".join(sections) + "</html>"
    (output / "index.html").write_text(page)
    print(f"Sample report: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
