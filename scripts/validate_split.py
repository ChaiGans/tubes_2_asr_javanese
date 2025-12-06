"""
Validate the split_info.json file to ensure:
1. All utterance IDs are unique (no duplicates across train/val/test)
2. Test speakers are disjoint from train/val speakers
3. Provide a summary of the split statistics
"""

import json
from pathlib import Path
from collections import Counter


def extract_speaker(utt_id: str) -> str:
    """Extract speaker ID from utterance ID (e.g., 'speaker01_m_nn_utt01' -> 'speaker01')"""
    return utt_id.split("_")[0]


def extract_native_status(utt_id: str) -> str:
    """Extract native status from utterance ID (e.g., 'speaker01_m_nn_utt01' -> 'nn')"""
    parts = utt_id.split("_")
    if len(parts) >= 3:
        return parts[2]  # 'n' or 'nn'
    return "unknown"


def validate_split(split_info_path: str = "data/split_info.json"):
    """Validate the split and print a detailed summary."""
    
    # Load split info
    with open(split_info_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    
    train_utts = split_info["split"]["train"]
    val_utts = split_info["split"]["val"]
    test_utts = split_info["split"]["test"]
    
    print("=" * 70)
    print("SPLIT VALIDATION REPORT")
    print("=" * 70)
    
    # =========================================================================
    # 1. Check for duplicate utterances
    # =========================================================================
    print("\n[1] CHECKING FOR DUPLICATE UTTERANCES...")
    
    all_utts = train_utts + val_utts + test_utts
    utt_counts = Counter(all_utts)
    duplicates = {utt: count for utt, count in utt_counts.items() if count > 1}
    
    if duplicates:
        print(f"   ❌ FAIL: Found {len(duplicates)} duplicate utterances:")
        for utt, count in list(duplicates.items())[:10]:
            print(f"      - {utt}: appears {count} times")
        if len(duplicates) > 10:
            print(f"      ... and {len(duplicates) - 10} more")
    else:
        print("   ✅ PASS: All utterances are unique across splits")
    
    # =========================================================================
    # 2. Check speaker disjointness (test vs train/val)
    # =========================================================================
    print("\n[2] CHECKING SPEAKER DISJOINTNESS (test vs train/val)...")
    
    train_speakers = set(extract_speaker(utt) for utt in train_utts)
    val_speakers = set(extract_speaker(utt) for utt in val_utts)
    test_speakers = set(extract_speaker(utt) for utt in test_utts)
    
    train_val_speakers = train_speakers | val_speakers
    overlap = test_speakers & train_val_speakers

    overlap_train = train_speakers & val_speakers
    print(train_speakers - overlap_train)
    
    if overlap:
        print(f"   ❌ FAIL: {len(overlap)} test speakers also appear in train/val:")
        for spk in sorted(overlap):
            print(f"      - {spk}")
    else:
        print("   ✅ PASS: Test speakers are disjoint from train/val speakers")
    
    # =========================================================================
    # 3. Check train/val speaker overlap (utterance-level split allows this)
    # =========================================================================
    print("\n[3] CHECKING TRAIN/VAL SPEAKER OVERLAP...")
    
    train_val_overlap = train_speakers & val_speakers
    if train_val_overlap:
        print(f"   ℹ️ INFO: {len(train_val_overlap)} speakers appear in BOTH train and val")
        print("           (This is expected for utterance-level split)")
    else:
        print("   ℹ️ INFO: Train and val have completely disjoint speakers")
    
    # =========================================================================
    # 4. Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    
    total_utts = len(all_utts)
    print(f"\n📊 UTTERANCE COUNTS:")
    print(f"   Train:      {len(train_utts):>5} utterances ({100*len(train_utts)/total_utts:.1f}%)")
    print(f"   Validation: {len(val_utts):>5} utterances ({100*len(val_utts)/total_utts:.1f}%)")
    print(f"   Test:       {len(test_utts):>5} utterances ({100*len(test_utts)/total_utts:.1f}%)")
    print(f"   ─────────────────────────────")
    print(f"   Total:      {total_utts:>5} utterances")
    
    total_speakers = len(train_speakers | val_speakers | test_speakers)
    print(f"\n👤 SPEAKER COUNTS:")
    print(f"   Train:      {len(train_speakers):>3} speakers")
    print(f"   Validation: {len(val_speakers):>3} speakers")
    print(f"   Test:       {len(test_speakers):>3} speakers (speaker-disjoint)")
    print(f"   ─────────────────────────────")
    print(f"   Total unique: {total_speakers:>3} speakers")
    
    # Native/Non-native breakdown
    print(f"\n🌍 TEST SET NATIVE/NON-NATIVE BREAKDOWN:")
    test_native = [utt for utt in test_utts if extract_native_status(utt) == "n"]
    test_non_native = [utt for utt in test_utts if extract_native_status(utt) == "nn"]
    test_native_speakers = set(extract_speaker(utt) for utt in test_native)
    test_non_native_speakers = set(extract_speaker(utt) for utt in test_non_native)
    
    print(f"   Native speakers:     {len(test_native_speakers):>2} speakers ({len(test_native)} utterances)")
    print(f"   Non-native speakers: {len(test_non_native_speakers):>2} speakers ({len(test_non_native)} utterances)")
    
    # List test speakers
    print(f"\n📋 TEST SPEAKERS ({len(test_speakers)} total):")
    for spk in sorted(test_speakers):
        status = "native" if spk in test_native_speakers else "non-native"
        spk_utts = [utt for utt in test_utts if extract_speaker(utt) == spk]
        print(f"   - {spk} ({status}): {len(spk_utts)} utterances")
    
    print("\n" + "=" * 70)
    
    # Final verdict
    all_passed = (len(duplicates) == 0) and (len(overlap) == 0)
    if all_passed:
        print("✅ ALL VALIDATION CHECKS PASSED!")
    else:
        print("❌ SOME VALIDATION CHECKS FAILED!")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    import sys
    
    split_path = sys.argv[1] if len(sys.argv) > 1 else "data/split_info.json"
    validate_split(split_path)
