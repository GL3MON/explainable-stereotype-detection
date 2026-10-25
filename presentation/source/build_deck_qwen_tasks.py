"""Revise the source-built deck while retaining its institutional template/style."""

from pathlib import Path
import json
import math
import shutil

HERE = Path(__file__).resolve().parent
ORIGINAL = (HERE / "build_deck.py").read_text()
PREFIX = ORIGINAL[:ORIGINAL.index("# 1. Title")]
PREFIX = PREFIX.replace("'deck_data.json'", "'deck_data_updated.json'")
PREFIX = PREFIX.replace("chart_data.add_series('Fine-tuned',", "chart_data.add_series('SFT',")
PREFIX = PREFIX.replace(
    "SLIDE_LOG.append({'number': number, 'heading': section, 'subtitle': subtitle})",
    "SLIDE_LOG.append({'number': number, 'heading': section, 'subtitle': subtitle, 'sources': list(sources), 'speaker': speaker})",
)
exec(compile(PREFIX, str(HERE / "build_deck.py"), "exec"), globals())

ROOT = HERE.parents[1]
OUTPUT = FOLDER / "revised_qwen_tasks_20261022"
OUTPUT.mkdir(exist_ok=True)
OUTPUT_STEM = "StereoDetect_BTP_Review_Qwen_Tasks_Corrected"
OUTPUT_PPTX = OUTPUT / f"{OUTPUT_STEM}.pptx"
LABELS_SHORT = ["0 Anti-stereotype", "1 Stereotype", "2 Neutral: no target", "3 Neutral: target", "4 Bias"]


def reuse(start, end):
    fragment = ORIGINAL[ORIGINAL.index(start):ORIGINAL.index(end)]
    fragment = fragment.replace(
        "Only the large variant is used throughout this deck.",
        "This architecture slide describes the saved large-model run; additional base-model experiments are reported in the appendix.",
    )
    for old, new in {
        "Qwen3.5-2B · Base versus fine-tuned label prediction": "C · Sentence + category + target → label · Base vs SFT",
        "Qwen3.5-2B · Base versus fine-tuned all-field generation": "B · Sentence → category + target + label · Base vs SFT",
        "Structured output · Category, target, label, and complete JSON": "B · Generated JSON · Per-field and all-fields accuracy",
    }.items():
        fragment = fragment.replace(old, new)
    exec(compile(fragment, str(HERE / "build_deck.py"), "exec"), globals())


def table_at(slide_obj, headers, values, widths, *, x=1.02, y=2.3, row_height=.44, size=16):
    graphic = slide_obj.shapes.add_table(len(values) + 1, len(headers), Inches(x), Inches(y),
                                         Inches(sum(widths)), Inches(row_height * (len(values) + 1)))
    result = graphic.table
    for index, width in enumerate(widths):
        result.columns[index].width = Inches(width)
    for row_index, values_row in enumerate([headers, *values]):
        for col, value in enumerate(values_row):
            cell = result.cell(row_index, col)
            cell.margin_left = cell.margin_right = Inches(.045)
            cell.margin_top = cell.margin_bottom = Inches(.022)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(PALE_BLUE if row_index == 0 else WHITE if row_index % 2 == 0 else "F8FAFC")
            cell.text = str(value)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = FONT
                paragraph.font.size = Pt(size)
                paragraph.font.bold = row_index == 0
                paragraph.font.color.rgb = rgb(NAVY if row_index == 0 else TEXT)
                paragraph.alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.CENTER
                paragraph.space_before = paragraph.space_after = Pt(0)
    return result


def report_rows(report, labels, names=None):
    return [[name, f"{report[str(label)]['precision'] * 100:.2f}",
             f"{report[str(label)]['recall'] * 100:.2f}",
             f"{report[str(label)]['f1-score'] * 100:.2f}",
             str(int(report[str(label)]["support"]))]
            for label, name in zip(labels, names or labels)]


def paired_reports(subtitle, cases, *, labels=None, names=None, footer=None):
    sources = [source for case in cases for source in case["sources"]]
    s = slide("Appendix · Classification Reports", subtitle, prototype=6, sources=sources,
              speaker="Full-test per-class precision, recall, F1 and support. P/R/F1 are percentages; N is gold support. All 1,738 rows are retained. Invalid or out-of-set predictions count as errors. Sources identify the saved prediction or report files.")
    xs = [1.02, 6.76] if len(cases) == 2 else [3.87]
    for x, case in zip(xs, cases):
        text(s, x, 2.08, 5.45, .6, case["title"], size=17, bold=True, color=NAVY)
        rows = report_rows(case["report"], labels or list(range(5)), names or LABELS_SHORT)
        table_at(s, ["Class", "P (%)", "R (%)", "F1 (%)", "N"], rows,
                 [2.3, .73, .73, .73, .96], x=x, y=2.86, row_height=.43, size=15)
        text(s, x, 5.63, 5.45, .7,
             f"Accuracy {case['accuracy'] * 100:.2f}%\nMacro F1 {case['macro_f1'] * 100:.2f}%", size=18, bold=True)
    source_line(s, footer or "Complete test split: 1,738 examples. P = precision; R = recall; N = gold support. All five labels are scored.")
    return s


def heat_color(value, limit=.70):
    neutral = (250, 250, 248)
    end = (43, 115, 185) if value < 0 else (214, 106, 31)
    weight = min(abs(value) / limit, 1.0)
    return "".join(f"{round(a + weight * (b - a)):02X}" for a, b in zip(neutral, end))


def heatmap_example(record, correct):
    explained = record["predicted_label"]
    detail = record["explanations"][str(explained)]
    s = slide("Demonstration", "ModernBERT-large · " + ("Correct prediction with word color maps" if correct else "A misclassified example with word color maps"),
              prototype=7, sources=[DATA["explanation_source"], "data/test.csv", "src/stereodetect/explanations/encoder.py"],
              speaker=f"Saved diagnostic test index {record['test_index']}. Exact word occurrences and attribution arrays are reused; no new explanation inference was run. Gold label {record['gold_label']}; model label {explained}. Both rows explain the predicted class {explained}. Blue denotes a negative contribution/weight for this class; orange denotes positive; near-white is close to zero. Both slides share a fixed symmetric scale -0.70 to +0.70. LIME surrogate weights and Kernel SHAP contributions are distinct quantities and not word probabilities. Attributions diagnose model sensitivity and do not certify correctness. Signed values are displayed to three decimals.")
    box(s, 1.02, 2.08, 11.25, .62, PALE_BLUE)
    text(s, 1.2, 2.22, 10.87, .35, f'“{record["sentence"]}”', size=22, bold=True)
    status = "Correct" if correct else "Misclassified"
    text(s, 1.06, 2.86, 11.12, .35,
         f"Gold {record['gold_label']} ({LABEL_NAMES_LOCAL[record['gold_label']]})   |   Predicted {explained} ({LABEL_NAMES_LOCAL[explained]})   |   {status}",
         size=17, color=GREEN if correct else "A04335", bold=True)
    words = record["words"]
    gap, left, available = .055, 2.69, 9.57
    weights = [max(4, len(word) + 1) for word in words]
    widths = [(available - gap * (len(words) - 1)) * weight / sum(weights) for weight in weights]
    for y, name, key in [(3.43, "LIME", "lime_values"), (4.30, "SHAP", "shap_values")]:
        text(s, 1.07, y + .17, 1.45, .35, name, size=20, bold=True, color=NAVY)
        x = left
        for word, width, value in zip(words, widths, detail[key]):
            tile = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
                                     Inches(x), Inches(y), Inches(width), Inches(.69))
            tile.adjustments[0] = .04
            tile.fill.solid()
            tile.fill.fore_color.rgb = rgb(heat_color(value))
            tile.line.color.rgb = rgb("D4DDE6")
            text(s, x + .025, y + .09, width - .05, .29, word, size=18 if len(words) < 6 else 16,
                 bold=True, align=PP_ALIGN.CENTER)
            text(s, x + .025, y + .43, width - .05, .2, f"{value:+.3f}", size=13,
                 align=PP_ALIGN.CENTER)
            x += width + gap
    insight = ("Largest support: businesslike. Masking the top word lowers the class score by 57.73 pp."
               if correct else "Largest support: Gentlemen. Strongest SHAP opposition: car. The predicted label is still wrong.")
    text(s, 1.06, 5.18, 11.12, .43, insight, size=17, bold=True)
    text(s, 1.06, 5.85, 3.2, .35, f"Blue: lowers class {explained} score", size=15, color="2B73B9")
    text(s, 9.12, 5.85, 3.15, .35, f"Orange: raises class {explained} score", size=15, color="A95015", align=PP_ALIGN.RIGHT)
    bar_x, bar_w, bar_y = 4.39, 4.25, 5.91
    for i in range(49):
        v = -.70 + 1.40 * i / 48
        tile = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(bar_x + bar_w * i / 49),
                                 Inches(bar_y), Inches(bar_w / 49 + .003), Inches(.18))
        tile.fill.solid()
        tile.fill.fore_color.rgb = rgb(heat_color(v))
        tile.line.fill.background()
    for x, label in [(bar_x - .12, "−0.70"), (bar_x + bar_w / 2 - .20, "0"), (bar_x + bar_w - .39, "+0.70")]:
        text(s, x, 6.13, .7, .2, label, size=11, color=GREY)
    text(s, 5.10, 5.61, 2.9, .23, "Near white = near zero", size=13, align=PP_ALIGN.CENTER, color=GREY)
    source_line(s, "Darker color = stronger attribution. Signs refer to the named model class, not sentiment or word probabilities.")
    return s


LABEL_NAMES_LOCAL = ["Anti-stereotype", "Stereotype", "Neutral without target term", "Neutral with target term", "Bias"]

# Preserve the title, introduction, literature review and motivation.
reuse("# 1. Title", "# 6. Problem statement")
SLIDE_LOG[0].update(sources=["presentation/BTP Presentation Template-3.pptx"], speaker="Identity placeholders and institutional template are preserved.")

s = slide("Problem Statement", "Three Qwen task contracts on the same held-out test split", prototype=4,
          sources=list(DATA["sources"].values()),
          speaker="There are three independent task adapters: sentence-only numeric labels, metadata-conditioned numeric labels, and sentence-only category/target/label JSON. Each is compared to the base checkpoint under its own prompt. All report the same held-out 1,738 rows. Metadata-conditioned results should not be read as input-matched rankings against sentence-only models.")
card(s, 1.02, 2.14, 3.6, 3.10, "A · Sentence-only labels", "Input: Sentence only\n\nOutput: one label, 0–4\n\nFive-label token scoring", PALE_GREEN, 20)
card(s, 4.85, 2.14, 3.6, 3.10, "B · Predict all fields", "Input: Sentence only\n\nOutput: category + target\n+ label JSON", PALE_PURPLE, 20)
card(s, 8.68, 2.14, 3.6, 3.10, "C · Label with metadata", "Input: Sentence + Category\n+ Target\n\nOutput: one label, 0–4", PALE_BLUE, 20)
text(s, 1.15, 5.55, 10.98, .8, "Each task has a Base and an SFT evaluation.\nMetadata means category + target; task B also predicts the label.", size=21, bold=True, align=PP_ALIGN.CENTER)
source_line(s, "All completed test evaluations use 1,738 examples. Category and target are never supplied to sentence-only tasks.")

reuse("# 7. Overall system", "# 12. Qwen3.5")
fragment = ORIGINAL[ORIGINAL.index("# 12. Qwen3.5"):ORIGINAL.index("# 13. Explainability")]
fragment = fragment.replace("Shared language backbone, two task adapters", "Shared language backbone, three task adapters")
fragment = fragment.replace("LoRA is trained separately for the label and JSON tasks; the adapters are not interchangeable.", "LoRA is trained separately for sentence-only labels, conditioned labels, and JSON; the three adapters are not interchangeable.")
fragment = fragment.replace("'Label task', 'Category + Target + Sentence → label', PALE_GREEN, 20", "'Two label adapters', 'Sentence → label\\nCategory + Target + Sentence → label', PALE_GREEN, 17")
fragment = fragment.replace("4.64, 5.46, 1.48", "4.54, 5.46, 1.58")
fragment = fragment.replace("4.64, 5.57, 1.48", "4.54, 5.57, 1.58")
exec(compile(fragment, str(HERE / "build_deck.py"), "exec"), globals())
reuse("# 13. Explainability pipeline", "# 14. Experimental setup")

s = slide("Experimental Results", "Dataset, training setup, and evaluation protocol", prototype=6,
          sources=["data/train.csv", "data/val.csv", "data/test.csv", DATA["sources"]["ft_sentence_labels"], "outputs/albert_stereodetect/run_config.json", "outputs/modernbert_large_lr3e5_warmup30_wd001_ep3_seed3407/run_config.json", "outputs/qwen35_2b_stereodetect_json/run_config.json"],
          speaker="The new sentence-only Qwen LoRA uses two T4 GPUs, batch 16 per GPU, global batch 32, three epochs and 570 optimizer steps. It trains in FP32 with LoRA rank 16, alpha 32 and dropout 0.05. The matching base evaluation uses vLLM BF16 on H100 MIG 0. Prompt, test rows and five-label argmax scoring match, but numerical precision and runtime differ. Validation macro F1 selected epoch 3; test was evaluated once after loading that adapter. Other Qwen adapters retain effective batch 16.")
for x, label, value, fill in [(1.02, "Training examples", "6,080", PALE_BLUE), (4.85, "Validation examples", "868", PALE_GREEN), (8.68, "Test examples", "1,738", PALE_PURPLE)]:
    metric(s, x, 2.1, 3.6, value, label, fill)
table(s, ["Saved training run", "Epochs", "Global batch", "Learning rate"], [
    ["ALBERT-base-v2", "6", "64", "2 × 10⁻⁵"],
    ["ModernBERT-large", "3", "32", "3 × 10⁻⁵"],
    ["Qwen · conditioned label / JSON", "3", "16", "2 × 10⁻⁴"],
    ["Qwen · sentence-only label (2×T4)", "3", "32", "2 × 10⁻⁴"],
], [4.35, 1.45, 2.65, 2.8], y=3.77, row_height=.48, size=17, highlight=(4,))
source_line(s, "New LoRA: 16 examples/GPU, 570 optimizer steps, FP32. Base: vLLM BF16. Validation macro F1 selects the adapter.")

s = slide("Experimental Results", "Model comparison on the 1,738-example test split", prototype=6,
          sources=[DATA["review_source"], *DATA["sources"].values()],
          speaker="The new sentence-only Qwen label result is 93.84% accuracy and 93.89% macro F1. The conditioned label adapter is a different experiment and receives gold category and target as input. Review-only baseline rows are preserved rather than replaced by later saved runs. Additional saved ModernBERT-base experiments and all full classification reports are included in the appendix. Precision/runtime differ across experiments, so this is descriptive rather than an engine-controlled model ranking.")
values = [
    ["all-MiniLM-L6-v2", "Review baseline", "77.33", "74.29"],
    ["Qwen3-Embedding-0.6B", "Review baseline", "91.08", "91.11"],
    ["ALBERT-base-v2", "Sentence only", percent(R["albert"]["accuracy"])[:-1], percent(R["albert"]["macro_f1"])[:-1]],
    ["ModernBERT-large · review", "Review run", "92.98", "93.05"],
    ["ModernBERT-large · saved", "Sentence only", percent(R["modernbert"]["accuracy"])[:-1], percent(R["modernbert"]["macro_f1"])[:-1]],
    ["Qwen SFT · A · sentence → label", "Sentence only", percent(R["ft_sentence_labels"]["accuracy"])[:-1], percent(R["ft_sentence_labels"]["macro_f1"])[:-1]],
    ["Qwen SFT · B · all fields", "Sentence only", percent(R["ft_json"]["accuracy"])[:-1], percent(R["ft_json"]["macro_f1"])[:-1]],
    ["Qwen SFT · C · metadata → label", "Sentence + metadata", percent(R["ft_labels"]["accuracy"])[:-1], percent(R["ft_labels"]["macro_f1"])[:-1]],
]
table(s, ["Model / run", "Input / source", "Accuracy (%)", "Macro F1 (%)"], values,
      [4.35, 3.35, 1.75, 1.8], y=2.08, row_height=.43, size=16, highlight=(6,))
source_line(s, "Qwen rows use the label metric. A/B/C are separate SFT adapters; Base/SFT pairs are on the next slide.")

TASKS = [
    {"id": "A", "task": "Sentence → label", "inputs": ["Sentence"], "outputs": ["label"], "base_key": "base_sentence_labels", "sft_key": "ft_sentence_labels"},
    {"id": "B", "task": "Sentence → category + target + label", "inputs": ["Sentence"], "outputs": ["category", "target", "label"], "base_key": "base_json", "sft_key": "ft_json"},
    {"id": "C", "task": "Sentence + category + target → label", "inputs": ["Sentence", "Category", "Target"], "outputs": ["label"], "base_key": "base_labels", "sft_key": "ft_labels"},
]
sources = [DATA['sources'][task[key]] for task in TASKS for key in ('base_key','sft_key')]
s = slide("Experimental Results", "Qwen3.5-2B · Three tasks × Base / SFT", prototype=6,
          sources=sources,
          speaker="Exactly six canonical full-test evaluations: three tasks, each with base and task-specific supervised fine-tuning via LoRA. Base is the same pretrained checkpoint under three prompts. Each SFT task uses its own adapter. Task B predicts category, target and label jointly; the summary scores only its label field. C receives gold category and target as input. We select the standalone test evaluations for B and C, and the completed Kaggle test evaluation for A. Earlier training-time test passes are repeated evaluations, not extra models. All runs disable thinking and use the same 1,738 test rows. Numerical engines and precision differ.")
rows=[]
for task in TASKS:
    for state,key in [('Base','base_key'),('SFT','sft_key')]:
        value=R[task[key]]
        rows.append([f"{task['id']} · {task['task']}",state,f"{value['accuracy']*100:.2f}",f"{value['macro_f1']*100:.2f}"])
table(s,["Input → predicted output", "Model state", "Label acc (%)", "Label macro F1 (%)"],rows,
      [5.55,1.25,2.15,2.3],y=2.12,row_height=.53,size=17,highlight=(2,4,6))
text(s,1.1,6.0,11.0,.4,"SFT = supervised fine-tuning with LoRA. One base checkpoint; three task-specific adapters.",size=18,bold=True)
source_line(s,"All scores here evaluate the label. Task B's category, target and joint exact-match scores follow separately.")
matrix={'model':'Qwen/Qwen3.5-2B','num_test_examples':1738,'enable_thinking':False,'tasks':[]}
for task in TASKS:
    item={k:v for k,v in task.items() if k not in ('base_key','sft_key')}
    for state,key in [('base','base_key'),('sft','sft_key')]:
        value=R[task[key]]
        item[state]={'evaluation':DATA['sources'][task[key]],'label_accuracy':value['accuracy'],'label_macro_f1':value['macro_f1']}
    matrix['tasks'].append(item)
(OUTPUT/'qwen_task_matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')

base, tuned = R["base_sentence_labels"], R["ft_sentence_labels"]
s = slide("Experimental Results", "A · Sentence → label · Base vs SFT", prototype=6,
          sources=[DATA["sources"]["base_sentence_labels"], DATA["sources"]["ft_sentence_labels"]],
          speaker="Both runs take only Sentence and select the largest logit among label tokens 0-4 with thinking disabled. Base has 403 correct, LoRA 1,631 correct out of 1,738. Accuracy rises from 23.19% to 93.84%, an observed gain of 70.66 percentage points. Macro F1 rises from 14.21% to 93.89%. Base uses vLLM BF16; the adapter uses PyTorch FP32, so this is not a precision/engine-controlled ablation.")
compare_chart(s, ["Accuracy", "Macro F1"], [base["accuracy"], base["macro_f1"]], [tuned["accuracy"], tuned["macro_f1"]])
metric(s, 8.73, 2.32, 3.53, f'+{(tuned["accuracy"] - base["accuracy"]) * 100:.2f} pp', "Observed accuracy gain")
card(s, 8.73, 3.97, 3.53, 2.12, "Sentence-only input", "No category/target input\nLabel-token argmax, 0–4\nAll 1,738 test examples", PALE_BLUE, 19)
source_line(s, "Base 23.19% accuracy / 14.21% macro F1; LoRA 93.84% / 93.89%. Base BF16; LoRA FP32; runtimes differ.")

reuse("# 17. Base vs task adaptation, JSON", "# 19. Per-class failure pattern")

s = slide("Experimental Results", "B · Generated JSON · Per-field accuracy and macro F1", prototype=6,
          sources=[DATA["report_manifest"], *DATA["report_sources"]],
          speaker="This supplements the exact-match field accuracy slide with full classification macro F1. Categories use five fixed normalized classes. Targets use the 87 exact gold target strings, preserving case and underscores. Predictions outside the gold class set are incorrect and appear in the companion reports; macro averages are over the explicit gold class set. The training-time and standalone JSON adapter evaluations have identical category and target prediction vectors and differ on one label. This table uses the standalone adapter evaluation.")
jbase, jft = DATA["json_runs"][0], DATA["json_runs"][2]
values = [[field.capitalize(), percent(jbase["fields"][field]["accuracy"])[:-1], percent(jft["fields"][field]["accuracy"])[:-1],
           percent(jbase["fields"][field]["macro_f1"])[:-1], percent(jft["fields"][field]["macro_f1"])[:-1]]
          for field in ("category", "label", "target")]
table(s, ["Field", "Base acc (%)", "LoRA acc (%)", "Base macro F1", "LoRA macro F1"], values,
      [2.15, 2.2, 2.2, 2.35, 2.35], y=2.30, row_height=.69, size=19)
for x, width, title, body, fill in [(1.02, 5.46, "Category", "Case/whitespace normalized; five classes", PALE_BLUE),
                                     (6.72, 5.57, "Target", "Exact match; 87 gold target strings", PALE_GREEN)]:
    box(s, x, 5.38, width, .95, fill)
    text(s, x + .18, 5.49, width - .36, .3, title, size=20, bold=True)
    text(s, x + .18, 5.94, width - .36, .28, body, size=16)
source_line(s, "Full per-class category, label and target reports are included in the appendix. Values are percentages.")

reuse("# 16. Base vs task adaptation, labels.", "# 17. Base vs task adaptation, JSON")

s = slide("Experimental Results", "Stereotype versus anti-stereotype remains the hardest boundary", prototype=6,
          sources=[DATA["sources"]["ft_sentence_labels"], DATA["sources"]["ft_json"], DATA["sources"]["modernbert"]],
          speaker="Across these three sentence-only experiments, classes 0 and 1 have lower F1 than the neutral and bias classes. For the new sentence-only Qwen LoRA, 46 anti-stereotypes become stereotypes and 36 stereotypes become anti-stereotypes: 82 of 107 errors. This is a confusion-matrix observation, not a linguistic causal explanation.")
reports = [R["modernbert"]["classification_report"], R["ft_sentence_labels"]["classification_report"], R["ft_json"]["classification_report"]]
values = [[LABELS_SHORT[i], str(DATA["class_support"][i]), *[f'{r[str(i)]["f1-score"] * 100:.2f}' for r in reports]] for i in range(5)]
table(s, ["Gold class", "N", "ModernBERT-large", "A · Qwen SFT", "B · Qwen SFT"], values,
      [3.45, .8, 2.4, 2.3, 2.3], y=2.18, row_height=.55, size=17)
box(s, 1.02, 5.83, 11.25, .55, PALE_AMBER)
text(s, 1.22, 5.94, 10.85, .34, "82 of 107 sentence-only Qwen label errors are stereotype ↔ anti-stereotype swaps.", size=20, bold=True)
source_line(s, "Per-class F1 (%), same test rows and sentence-only input. Different runtimes/precision; no controlled throughput claim.")

reuse("# 20. Demonstration", "# 21. Demonstration")
for record, correct in zip(DATA["explanation_examples"], (True, False)):
    heatmap_example(record, correct)

s = slide("Conclusion", "Task adaptation improves all three task setups", prototype=8,
          sources=[*DATA["sources"].values(), DATA["explanation_source"]],
          speaker="The new sentence-only label experiment improves from 23.19% to 93.84% accuracy under matched prompt, split and label-token scoring, with differing precision/runtime. The metadata-conditioned label adapter reaches 95.51%; its richer inputs prevent a direct input-matched ranking. Sentence-only JSON label accuracy is 93.61%, category accuracy 99.37%, target exact match 98.16%, and all-three-fields accuracy 92.87%. The explanation examples use ModernBERT-large, not Qwen, and include a misclassification to show why attribution does not certify correctness. Future work is proposed, not performed.")
card(s, 1.02, 2.1, 5.46, 3.96, "Three SFT task outcomes", "• A · Sentence → label: 93.84%\n• B · Sentence → all fields:\n   label 93.61%; joint 92.87%\n• C · Sentence + metadata → label: 95.51%\n\nColor maps show signed word influence.", PALE_GREEN, 21)
card(s, 6.72, 2.1, 5.57, 3.96, "Next phase", "• Review class 0 / class 1 errors\n• Match runtimes, precision and seeds\n• Evaluate calibration and new domains\n• Extend explanation diagnostics\n\nExplanations describe the model; they do not prove label correctness.", PALE_BLUE, 21)
source_line(s, "All main result slides use completed full-test evaluations; partial or validation results are kept separate.")

reuse("# 23–24. References", "# Drop the untouched template")
MAIN_SLIDES = len(SLIDE_LOG)

# Appendix: one canonical full-test evaluation per model/task.
short_names = ["A · Qwen Base · sentence → label", "A · Qwen SFT · sentence → label", "C · Qwen Base · metadata → label",
               "Repeated training-time evaluation", "C · Qwen SFT · metadata → label",
               "ALBERT-base-v2", "ModernBERT-base · original", "ModernBERT-base · retuned", "ModernBERT-large"]
s = slide("Appendix · Complete Test Results", "Label-only models · One canonical test evaluation per run", prototype=6,
          sources=[DATA["report_manifest"], DATA["report_sources"][0]],
          speaker="Four Qwen label-only evaluations are tasks A and C, each with Base and SFT. Four encoder runs are also retained. Task B generates JSON and appears in its own report pages. The earlier training-time pass of C uses the same SFT adapter and is preserved only in the companion TXT, not counted as another model. Every result uses all 1,738 test rows.")
rows = [[name, "Sentence" if i not in (2, 4) else "Sentence + metadata", percent(r["accuracy"])[:-1], percent(r["macro_f1"])[:-1]] for i, (name, r) in enumerate(zip(short_names, DATA["label_runs"])) if i != 3]
table(s, ["Saved experiment", "Inputs", "Accuracy (%)", "Macro F1 (%)"], rows,
      [4.6, 3.05, 1.8, 1.8], y=2.13, row_height=.40, size=15)
source_line(s, "Metadata = gold Category + Target. These are distinct experiments/evaluations; inspect inputs before comparing scores.")

for left_index, right_index in [(0, 1), (2, 4), (5, 8), (6, 7)]:
    cases = []
    for index in (left_index, right_index):
        if index is None:
            continue
        r = DATA["label_runs"][index]
        cases.append({"title": short_names[index], "report": r["classification_report"],
                      "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "sources": [r["source"]]})
    paired_reports("Label-only · Full per-class precision, recall and F1", cases)

base_json, train_json, ft_json = DATA["json_runs"]
shared_json_sources = [ft_json["source"]]
category_cases = []
for r, title, sources in [(base_json, "B · Qwen Base · JSON", [base_json["source"]]),
                          (ft_json, "B · Qwen SFT · JSON", shared_json_sources)]:
    category_cases.append({"title": title, "report": r["classification_reports"]["category"],
                           "accuracy": r["fields"]["category"]["accuracy"],
                           "macro_f1": r["fields"]["category"]["macro_f1"], "sources": sources})
paired_reports("JSON category · Five normalized classes", category_cases,
               labels=["gender", "race", "profession", "religion", "neutral"],
               names=["gender", "race", "profession", "religion", "neutral"],
               footer="Task B: only Sentence is input. Category, target and label are generated together by Base / SFT.")

for cases_source in [[(base_json, "B · Qwen Base · JSON"), (ft_json, "B · Qwen SFT · JSON")]]:
    cases = [{"title": title, "report": r["classification_reports"]["label"],
              "accuracy": r["fields"]["label"]["accuracy"], "macro_f1": r["fields"]["label"]["macro_f1"],
              "sources": [r["source"]]} for r, title in cases_source]
    paired_reports("B · Generated JSON · Label classification report", cases,
                   footer="This scores the label field from generated JSON; all-fields accuracy is a separate metric.")

targets = DATA["target_classes"]
for r, title, sources in [(base_json, "B · Qwen Base · JSON", [base_json["source"]]),
                          (ft_json, "B · Qwen SFT · JSON", shared_json_sources)]:
    report = r["classification_reports"]["target"]
    pages = math.ceil(len(targets) / 30)
    for page in range(pages):
        selected = targets[page * 30:(page + 1) * 30]
        s = slide("Appendix · Target Classification", f"{title} · Exact target strings ({page + 1} / {pages})", prototype=6,
                  sources=sources,
                  speaker=f"Per-target precision/recall/F1 on all 1,738 test examples. All 87 distinct gold target strings are listed across {pages} slides. Target matching is exact, including capitalization, underscores and whitespace. Gold classes alone define macro averages; out-of-set guesses remain wrong predictions. The training-time and standalone LoRA target vectors are identical, so one shared report represents both evaluations.")
        for x, subset in [(1.02, selected[:15]), (6.76, selected[15:])]:
            if subset:
                table_at(s, ["Gold target", "P (%)", "R (%)", "F1 (%)", "N"], report_rows(report, subset),
                         [2.5, .8, .8, .8, .55], x=x, y=2.12, row_height=.27, size=12)
        source_line(s, f"Exact target accuracy {r['fields']['target']['accuracy'] * 100:.2f}% · Macro F1 {r['fields']['target']['macro_f1'] * 100:.2f}% · Out-of-set predictions count as errors.")

s = slide("Appendix · Report Coverage", "Complete reports and diagnostic subsets are kept distinct", prototype=6,
          sources=[DATA["report_manifest"], *DATA["report_sources"]],
          speaker="The accompanying text files contain all nine completed label-only test evaluations and three validation class reports; and three completed JSON test evaluations, three complete validation evaluations, plus six partial/incomplete diagnostic runs. The 410-row thinking run is unfinished and is not a full-test result. No new classifier or explanation model was run while revising this deck.")
card(s, 1.02, 2.14, 5.46, 3.85, "Canonical test appendix", "• A / C: Base + SFT label reports\n• B: Base + SFT JSON field reports\n• 4 saved encoder runs\n• All 87 exact target classes\n\nOne test evaluation per model/task.\nSFT uses three separate LoRA adapters.", PALE_GREEN, 21)
card(s, 6.72, 2.14, 5.57, 3.85, "Companion TXT files", "Earlier training-time evaluations, validation reports and partial diagnostics remain in the TXT files.\n\nRepeated evaluations of an adapter are not additional models.\n\nNo model inference was rerun.", PALE_BLUE, 21)
source_line(s, "Companions: all_label_only_classification_reports.txt and all_category_label_target_classification_reports.txt")

# Drop only template prototypes. The original deck remains untouched.
for _ in PROTOTYPES:
    sid = PRS.slides._sldIdLst[0]
    PRS.part.drop_rel(sid.rId)
    PRS.slides._sldIdLst.remove(sid)

# Capture the authored text and its chosen bases before saving the generated PPTX.
claims = {"deck": OUTPUT_PPTX.name, "capture_mode": "generation-time from source-built revision",
          "slides": []}
for index, (slide_obj, info) in enumerate(zip(PRS.slides, SLIDE_LOG), 1):
    sources = info.get("sources") or ["presentation/BTP Presentation Template-3.pptx"]
    refs = [{"title": "Recorded source", "url": source} if source.startswith("http") else
            {"title": "Recorded local source", "path": source} for source in sources]
    entry = {"slide_number": index, "slide_title": info["subtitle"] or info["heading"], "claims": []}
    seen_text = set()
    for shape_index, shape in enumerate(slide_obj.shapes):
        if shape.has_text_frame and shape.text.strip():
            content = " ".join(shape.text.split())
            if content.isdigit() or content in seen_text:
                continue
            seen_text.add(content)
            key = f"s{index}-t{shape_index}"
            shape.name = "clara-claim:" + key
            basis = ([{"title": "Supplied institutional template", "path": "presentation/BTP Presentation Template-3.pptx"}]
                     if "Indian Institute of Information Technology Kottayam" in content or "[Student" in content or "[Guide" in content else refs)
            entry["claims"].append({"claim": content, "claim_key": key, "source_refs": basis})
        elif shape.has_table:
            for row_index, row in enumerate(shape.table.rows):
                content = " ".join(" ".join(cell.text.split()) for cell in row.cells)
                if content in seen_text:
                    continue
                seen_text.add(content)
                entry["claims"].append({"claim": content, "claim_key": f"s{index}-table{shape_index}-row{row_index}", "source_refs": refs})
    claims["slides"].append(entry)

PRS.core_properties.title = "StereoDetect — Updated BTP Review"
PRS.core_properties.subject = "Sentence-only LoRA results, full classification reports and signed explanation color maps"
PRS.core_properties.author = "StereoDetect Project Team"
PRS.core_properties.comments = "Updated copy of the source-built institutional deck. Numeric results and word scores come from saved artifacts."
PRS.save(OUTPUT_PPTX)
(OUTPUT / f"{OUTPUT_STEM}.claims.json").write_text(json.dumps(claims, indent=2) + "\n")
(OUTPUT / "slide_outline.json").write_text(json.dumps(SLIDE_LOG, indent=2) + "\n")

# Reopen and extract text/table rows independently for the deterministic drift check.
check = Presentation(OUTPUT_PPTX)
assert len(check.slides) == len(SLIDE_LOG) == 42, len(check.slides)
snapshot = {"deck": str(OUTPUT_PPTX), "slides": []}
for index, slide_obj in enumerate(check.slides, 1):
    texts = []
    for shape in slide_obj.shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= check.slide_width + Inches(.01), (index, shape.name)
        assert shape.top + shape.height <= check.slide_height + Inches(.01), (index, shape.name)
        if shape.has_text_frame and shape.text.strip():
            item = {"text": " ".join(shape.text.split())}
            if shape.name.startswith("clara-claim:"):
                item["claim_key"] = shape.name.removeprefix("clara-claim:")
            texts.append(item)
        elif shape.has_table:
            for row in shape.table.rows:
                texts.append({"text": " ".join(" ".join(cell.text.split()) for cell in row.cells)})
    snapshot["slides"].append({"slide_number": index, "texts": texts})
(OUTPUT / "current_deck_text.json").write_text(json.dumps(snapshot, indent=2) + "\n")
for source in DATA["report_sources"]:
    shutil.copy(ROOT / source, OUTPUT / Path(source).name)
print("Saved", OUTPUT_PPTX, "with", len(check.slides), "slides; main section:", MAIN_SLIDES)
