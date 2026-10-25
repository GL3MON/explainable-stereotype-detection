"""Batched probability scoring for encoder explanations."""


class ProbabilityScorer:
    def __init__(self, model, tokenizer, batch_size=128, max_length=128, device="cuda"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.cache = {}
        self.forward_examples = 0

    def __call__(self, texts):
        import numpy as np
        import torch

        texts = [str(text) for text in texts]
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"
                ):
                    probabilities = (
                        self.model(**inputs).logits.float().softmax(dim=-1).cpu().numpy()
                    )
                if not np.isfinite(probabilities).all():
                    raise RuntimeError("Non-finite ALBERT probabilities")
                self.cache.update(zip(batch, probabilities))
                self.forward_examples += len(batch)
        return np.stack([self.cache[text] for text in texts])
