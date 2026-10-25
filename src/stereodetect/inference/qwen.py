"""Classify a sentence with the trained adapter, without generating reasoning."""

import argparse
import json

from stereodetect.models.qwen.labels import LABEL_NAMES, MODEL, messages, sentence_only_for_adapter


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence")
    parser.add_argument("--category")
    parser.add_argument("--target")
    parser.add_argument("--sentence-only", action="store_true")
    parser.add_argument("--adapter", default="outputs/qwen35_2b_stereodetect/adapter")
    args = parser.parse_args(argv)
    sentence_only = args.sentence_only or sentence_only_for_adapter(args.adapter)
    if not sentence_only and (args.category is None or args.target is None):
        parser.error("--category and --target are required for category/target-conditioned adapters")
    from stereodetect.runtime import configure_runtime

    configure_runtime()
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).to("cuda").eval()
    row = {"Sentence": args.sentence, "Category": args.category, "Target": args.target}
    inputs = tokenizer.apply_chat_template(
        messages(row, sentence_only=sentence_only),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(5)]
    with torch.inference_mode():
        scores = model(**inputs, logits_to_keep=1).logits[0, -1, ids].float()
        probabilities = scores.softmax(-1)
    label = int(scores.argmax())
    print(
        json.dumps(
            {
                "label": label,
                "label_name": LABEL_NAMES[label],
                "label_probabilities": probabilities.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
