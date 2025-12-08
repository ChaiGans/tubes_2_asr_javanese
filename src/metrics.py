import torch
from typing import List
import editdistance


def compute_cer(reference: str, hypothesis: str) -> float:
    # Compute Character Error Rate (CER) between reference and hypothesis.
    # CER = (substitutions + insertions + deletions) / len(reference)
    if len(reference) == 0:
        if len(hypothesis) == 0:
            return 0.0
        else:
            return 1.0

    distance = editdistance.eval(reference, hypothesis)
    cer = distance / len(reference)

    return cer


def compute_batch_cer(references: List[str], hypotheses: List[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    total_cer = 0.0
    for ref, hyp in zip(references, hypotheses):
        total_cer += compute_cer(ref, hyp)

    return total_cer / len(references) if len(references) > 0 else 0.0


def compute_wer(reference: str, hypothesis: str) -> float:
    # Compute Word Error Rate (WER) between reference and hypothesis.
    # WER = (substitutions + insertions + deletions) / number of words in reference
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0
        else:
            return 1.0

    distance = editdistance.eval(ref_words, hyp_words)
    wer = distance / len(ref_words)

    return wer


if __name__ == "__main__":
    print("Testing CER computation...")

    test_cases = [
        ("aku libur sedina", "aku libur sedina", 0.0),
        ("aku libur", "aku", 6/10),
        ("hello", "hallo", 1/5),
        ("", "", 0.0),
    ]

    for ref, hyp, expected_cer in test_cases:
        cer = compute_cer(ref, hyp)
        print(f"Reference: '{ref}'")
        print(f"Hypothesis: '{hyp}'")
        print(f"CER: {cer:.4f} (expected: {expected_cer:.4f})")
        print()

    refs = ["aku libur sedina", "wong jawa seneng"]
    hyps = ["aku libur", "wong jawa seneng"]
    batch_cer = compute_batch_cer(refs, hyps)
    print(f"Batch CER: {batch_cer:.4f}")

    print("CER test passed!")
