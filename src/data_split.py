"""
Speaker-disjoint data splitting for Javanese ASR

Implements the requirement:
- 20% speakers as test set (10% native + 10% non-native)
- 80% speakers for train+val
- From train+val speakers: 70% utterances for train, 10% for validation
- All splits are speaker-disjoint and reproducible
"""

import random
from typing import List, Dict, Tuple
from pathlib import Path
import json


def parse_speaker_info(utt_id: str) -> Tuple[str, str]:
    """
    Parse speaker ID and native status from utterance ID.
    
    Format: speaker01_m_nn_utt01
            speaker02_f_n_utt03
    
    Returns:
        (speaker_id, native_status)
        native_status: 'n' for native, 'nn' for non-native
    """
    parts = utt_id.split('_')
    if len(parts) >= 3:
        speaker_id = parts[0]  # e.g., 'speaker01'
        native_status = parts[2]  # e.g., 'n' or 'nn'
        return speaker_id, native_status
    else:
        raise ValueError(f"Invalid utterance ID format: {utt_id}")


def load_transcript_data(transcript_file: str) -> List[Dict]:
    """
    Load transcript data from CSV file.
    
    Returns:
        List of dicts with keys: 'utt_id', 'transcript', 'speaker', 'native_status'
    """
    data = []
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Skip header
            if line_idx == 0 and line.startswith('SentenceID'):
                continue
            
            # Parse CSV
            parts = line.split(',')
            if len(parts) >= 2:
                utt_id = parts[0].strip()
                transcript = parts[1].strip()
                
                if not transcript:
                    continue
                
                # Parse speaker info
                try:
                    speaker_id, native_status = parse_speaker_info(utt_id)
                    
                    data.append({
                        'utt_id': utt_id,
                        'transcript': transcript,
                        'speaker': speaker_id,
                        'native_status': native_status
                    })
                except ValueError as e:
                    print(f"Warning: Skipping invalid utterance ID: {utt_id}")
                    continue
    
    return data


def create_speaker_disjoint_split(
    transcript_file: str,
    test_speaker_ratio: float = 0.2,
    val_utterance_ratio: float = 0.1,
    seed: int = 42,
    save_split_info: bool = True,
    split_info_path: str = "data/split_info.json"
) -> Dict[str, List[str]]:
    """
    Create speaker-disjoint train/val/test split.
    
    Requirements:
    - Test: 20% of speakers (10% native, 10% non-native)
    - Train+Val: 80% of speakers
      - 70% of their utterances go to train
      - 10% of their utterances go to val
    
    Args:
        transcript_file: Path to transcript CSV
        test_speaker_ratio: Ratio of speakers for test (0.2 = 20%)
        val_utterance_ratio: Ratio of utterances for validation from train+val speakers
        seed: Random seed for reproducibility
        save_split_info: Whether to save split information to file
        split_info_path: Path to save split information
    
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing list of utterance IDs
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load all data
    print(f"Loading transcript data from {transcript_file}...")
    all_data = load_transcript_data(transcript_file)
    print(f"Loaded {len(all_data)} utterances")
    
    # Group by speaker and native status
    speakers_native = {}  # speaker_id -> list of utterances
    speakers_non_native = {}
    
    for item in all_data:
        speaker = item['speaker']
        native = item['native_status']
        
        if native == 'n':
            if speaker not in speakers_native:
                speakers_native[speaker] = []
            speakers_native[speaker].append(item)
        elif native == 'nn':
            if speaker not in speakers_non_native:
                speakers_non_native[speaker] = []
            speakers_non_native[speaker].append(item)
    
    print(f"\nSpeaker statistics:")
    print(f"  Native speakers: {len(speakers_native)}")
    print(f"  Non-native speakers: {len(speakers_non_native)}")
    print(f"  Total speakers: {len(speakers_native) + len(speakers_non_native)}")
    
    # Calculate number of speakers for test
    num_test_native = max(1, int(len(speakers_native) * test_speaker_ratio))
    num_test_non_native = max(1, int(len(speakers_non_native) * test_speaker_ratio))
    
    print(f"\nTest set allocation:")
    print(f"  Native speakers for test: {num_test_native} ({num_test_native/len(speakers_native)*100:.1f}%)")
    print(f"  Non-native speakers for test: {num_test_non_native} ({num_test_non_native/len(speakers_non_native)*100:.1f}%)")
    
    # Randomly select test speakers
    native_speaker_list = list(speakers_native.keys())
    non_native_speaker_list = list(speakers_non_native.keys())
    
    random.shuffle(native_speaker_list)
    random.shuffle(non_native_speaker_list)
    
    test_speakers_native = set(native_speaker_list[:num_test_native])
    test_speakers_non_native = set(non_native_speaker_list[:num_test_non_native])
    
    train_val_speakers_native = set(native_speaker_list[num_test_native:])
    train_val_speakers_non_native = set(non_native_speaker_list[num_test_non_native:])
    
    # Collect test utterances
    test_utterances = []
    for speaker in test_speakers_native:
        test_utterances.extend(speakers_native[speaker])
    for speaker in test_speakers_non_native:
        test_utterances.extend(speakers_non_native[speaker])
    
    # Collect train+val utterances
    train_val_utterances = []
    for speaker in train_val_speakers_native:
        train_val_utterances.extend(speakers_native[speaker])
    for speaker in train_val_speakers_non_native:
        train_val_utterances.extend(speakers_non_native[speaker])
    
    # Shuffle train+val utterances for random split
    random.shuffle(train_val_utterances)
    
    # Split train+val into train (70% of all data) and val (10% of all data)
    # Since test is 20%, train+val is 80%
    # We want train=70% and val=10% of total, so from train+val (80%):
    # train = 70/80 = 0.875 of train+val
    # val = 10/80 = 0.125 of train+val
    train_ratio_of_trainval = 0.875  # 70% / 80%
    
    num_train = int(len(train_val_utterances) * train_ratio_of_trainval)
    
    train_utterances = train_val_utterances[:num_train]
    val_utterances = train_val_utterances[num_train:]
    
    # Extract utterance IDs
    train_ids = [item['utt_id'] for item in train_utterances]
    val_ids = [item['utt_id'] for item in val_utterances]
    test_ids = [item['utt_id'] for item in test_utterances]
    
    # Print statistics
    total_utts = len(all_data)
    print(f"\nFinal split statistics:")
    print(f"  Train: {len(train_ids)} utterances ({len(train_ids)/total_utts*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} utterances ({len(val_ids)/total_utts*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} utterances ({len(test_ids)/total_utts*100:.1f}%)")
    print(f"  Total: {total_utts} utterances")
    
    # Verify speaker-disjoint
    train_speakers = set(item['speaker'] for item in train_utterances)
    val_speakers = set(item['speaker'] for item in val_utterances)
    test_speakers = test_speakers_native | test_speakers_non_native
    
    print(f"\nSpeaker-disjoint verification:")
    print(f"  Train speakers: {len(train_speakers)}")
    print(f"  Val speakers: {len(val_speakers)}")
    print(f"  Test speakers: {len(test_speakers)}")
    
    # Check for overlaps
    train_test_overlap = train_speakers & test_speakers
    val_test_overlap = val_speakers & test_speakers
    
    if len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        print(f"  WARNING: Speaker overlap detected!")
        print(f"    Train-Test overlap: {len(train_test_overlap)}")
        print(f"    Val-Test overlap: {len(val_test_overlap)}")
    else:
        print(f"  ✓ No speaker overlap between test and train/val sets")
        print(f"  ✓ Speaker-disjoint split verified!")
    
    # Note: train and val come from same speakers (per-utterance split)
    print(f"  Note: Train and val share {len(train_speakers & val_speakers)} speakers (per-utterance split)")
    
    split_dict = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    # Save split information
    if save_split_info:
        split_info = {
            'seed': seed,
            'test_speaker_ratio': test_speaker_ratio,
            'val_utterance_ratio': val_utterance_ratio,
            'test_speakers': {
                'native': list(test_speakers_native),
                'non_native': list(test_speakers_non_native)
            },
            'train_val_speakers': {
                'native': list(train_val_speakers_native),
                'non_native': list(train_val_speakers_non_native)
            },
            'split': split_dict
        }
        
        Path(split_info_path).parent.mkdir(parents=True, exist_ok=True)
        with open(split_info_path, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Split information saved to {split_info_path}")
    
    return split_dict


def load_split_info(split_info_path: str = "data/split_info.json") -> Dict:
    """
    Load previously saved split information.
    
    Returns:
        Dictionary with split information
    """
    with open(split_info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test the splitting
    split_dict = create_speaker_disjoint_split(
        transcript_file="data/transcripts.csv",
        test_speaker_ratio=0.2,
        val_utterance_ratio=0.1,
        seed=42,
        save_split_info=True,
        split_info_path="data/split_info.json"
    )
    
    print("\n" + "="*60)
    print("Split created successfully!")
    print("="*60)
