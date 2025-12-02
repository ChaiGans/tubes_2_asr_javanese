"""
Evaluation Metrics for ASR: Character Error Rate (CER)
"""

import torch
from typing import List
import editdistance


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) between reference and hypothesis.
    
    CER = (substitutions + insertions + deletions) / len(reference)
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        CER as a float (0.0 = perfect, 1.0 = completely wrong)
    """
    if len(reference) == 0:
        if len(hypothesis) == 0:
            return 0.0
        else:
            return 1.0  # Or float('inf')
    
    # Compute edit distance
    distance = editdistance.eval(reference, hypothesis)
    
    # CER = edit distance / reference length
    cer = distance / len(reference)
    
    return cer


def compute_batch_cer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute average CER over a batch.
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
    
    Returns:
        Average CER
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    total_cer = 0.0
    for ref, hyp in zip(references, hypotheses):
        total_cer += compute_cer(ref, hyp)
    
    return total_cer / len(references) if len(references) > 0 else 0.0


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis.
    
    WER = (substitutions + insertions + deletions) / number of words in reference
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
    
    Returns:
        WER as a float
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0
        else:
            return 1.0
    
    # Compute edit distance on words
    distance = editdistance.eval(ref_words, hyp_words)
    
    # WER = edit distance / number of reference words
    wer = distance / len(ref_words)
    
    return wer


if __name__ == "__main__":
    print("Testing CER computation...")
    
    # Test cases
    test_cases = [
        ("aku libur sedina", "aku libur sedina", 0.0),  # Perfect match
        ("aku libur", "aku", 6/10),  # Missing word
        ("hello", "hallo", 1/5),  # One substitution
        ("", "", 0.0),  # Empty strings
    ]
    
    for ref, hyp, expected_cer in test_cases:
        cer = compute_cer(ref, hyp)
        print(f"Reference: '{ref}'")
        print(f"Hypothesis: '{hyp}'")
        print(f"CER: {cer:.4f} (expected: {expected_cer:.4f})")
        print()
    
    # Test batch CER
    refs = ["aku libur sedina", "wong jawa seneng"]
    hyps = ["aku libur", "wong jawa seneng"]
    batch_cer = compute_batch_cer(refs, hyps)
    print(f"Batch CER: {batch_cer:.4f}")
    
    print("CER test passed!")
