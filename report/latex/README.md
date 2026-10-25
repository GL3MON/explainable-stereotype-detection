# StereoDetect LaTeX project report

This report follows `report/Full_Project_Report_DL.docx`: institute/course cover,
contents, abstract and base-paper details, problem statement, architecture,
dataset, results, implementation/dataset links, and APA-style references.
The original Word document and presentations are preserved.

The cover includes Harissh Ragav Dhamodaran (2023BCD0059) and
Mugupuram Kavya Amrutha (2023BCS0221). Edit `metadata.tex` to change these
details or the project repository link.

## Compile

From this directory, use either:

```bash
tectonic main.tex
```

or a current TeX Live/MiKTeX installation:

```bash
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `main.pdf`. No separate BibTeX/Biber run is needed: APA-style
author--date references are maintained in `references.tex`. The Times-compatible
TeX Gyre Termes fonts are supplied by a standard TeX installation. Upload the
bundle to Overleaf and select **XeLaTeX** if compiling there.

All figures are vector PDFs, with PNG copies for convenient inspection.
`generated/*.tex` contains report tables; `generated/evidence.json` contains
recomputed metrics, the dataset audit, and SHA-256 values for source artifacts.
`supplementary/` contains the complete companion TXT reports, including earlier
evaluation passes and clearly labeled validation/partial diagnostic results.

## Experiment selection

Qwen uses exactly three task contracts, each with Base and SFT:

- A: Sentence → label.
- B: Sentence → category + target + label JSON.
- C: Sentence + gold category and target annotations → label.

The report calls the category and target fields **auxiliary annotations** and
describes task C as **annotation-conditioned label prediction**.

Each SFT task has a separate LoRA adapter. Task A uses the completed Kaggle
two-T4 result. Tasks B/C use their standalone full-test evaluations. Repeated
training-time test passes, validation reports and partial diagnostic runs are
not counted as additional canonical models. JSON label accuracy and all-fields
accuracy are distinct. Explanation figures belong to ModernBERT-large.
The report includes a four-panel A/C confusion-matrix comparison for Base and
SFT, with counts, row percentages, a shared color scale, and paired error changes.
Section 5 ends at 5.1; the References section follows it.

## Regenerate evidence and figures

`generate_assets.py` expects this directory to remain at `report/latex` inside
the original repository, with its saved data and output artifacts available.
It needs Python, NumPy and Matplotlib. It checks row alignment, recomputes
classification metrics, compares saved evaluation scores, and emits the tables
and figures. It does not perform training or model inference.

```bash
python generate_assets.py
```

Regeneration is optional: generated assets are included in the portable bundle.
The report records the dataset statistics and the original split assignments.
