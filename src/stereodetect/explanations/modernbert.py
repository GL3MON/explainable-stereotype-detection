"""ModernBERT defaults for the shared encoder explanation workflow."""

import sys

from stereodetect.explanations.encoder import main as explain


def main(argv=None):
    defaults = [
        "--model-dir",
        "outputs/modernbert_stereodetect/model",
        "--predictions",
        "outputs/modernbert_stereodetect/test_predictions.csv",
        "--output-dir",
        "outputs/modernbert_stereodetect/explanations",
        "--display-name",
        "ModernBERT",
        "--wandb-run-name",
        "modernbert-lime-shap-explanations",
        "--indices-json",
        "outputs/albert_stereodetect/explanations/run_config.json",
    ]
    return explain([*defaults, *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
