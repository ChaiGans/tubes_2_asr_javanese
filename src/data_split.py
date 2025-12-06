"""
Utterance-level split with simple native/non-native stratification.

- Test: ~10% native + ~10% non-native utterances (by count, capped by availability)
- Val:  ~10% of remaining utterances (unstratified)
- Train: the rest

Audio is validated (exists + readable). Only valid rows are kept in the split.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import pandas as pd
import soundfile as sf


def find_audio_path(utt_id: str, audio_dir: Path) -> Optional[Path]:
    """
    Try multiple filename variants to locate the audio file.
    """
    lower = audio_dir / f"{utt_id}.wav"
    cap = audio_dir / f"S{utt_id[1:]}.wav" if utt_id.startswith("speaker") else audio_dir / f"{utt_id.capitalize()}.wav"
    dash = audio_dir / f"{utt_id.replace('_', '-')}.wav"
    cap_dash = (
        audio_dir / f"S{utt_id[1:].replace('_', '-')}.wav"
        if utt_id.startswith("speaker")
        else audio_dir / f"{utt_id.capitalize().replace('_', '-')}.wav"
    )
    for p in (lower, cap, dash, cap_dash):
        if p.exists():
            return p.resolve()
    return None


def parse_gender_native(utt_id: str) -> Tuple[Optional[str], Optional[bool]]:
    """
    Parse gender/native tokens from an utterance ID.
    Expected format: speakerXX_m_nn_uttYY
    """
    parts = utt_id.split("_")
    gender = None
    native = None
    if len(parts) >= 3:
        gender_token = parts[1].lower()
        native_token = parts[2].lower()
        if gender_token == "m":
            gender = "male"
        elif gender_token == "f":
            gender = "female"
        if native_token == "n":
            native = True
        elif native_token == "nn":
            native = False
    return gender, native


def split_utterances(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified test split: 10% native + 10% non-native utterances.
    Validation: 10% of remaining utterances (unstratified).
    """
    total = len(df)
    target_test_native = int(round(total * 0.10))
    target_test_non_native = int(round(total * 0.10))
    target_val = int(round(total * 0.10))

    native_df = df[df["Native"] == True]
    non_native_df = df[df["Native"] == False]

    test_native = native_df.sample(n=min(target_test_native, len(native_df)), random_state=seed)
    test_non_native = non_native_df.sample(n=min(target_test_non_native, len(non_native_df)), random_state=seed)
    test_df = pd.concat([test_native, test_non_native], axis=0)
    test_indices = set(test_df.index)

    remain_df = df[~df.index.isin(test_indices)].copy()

    val_sample = remain_df.sample(n=min(target_val, len(remain_df)), random_state=seed)
    val_indices = set(val_sample.index)
    val_df = remain_df[remain_df.index.isin(val_indices)].copy()
    train_df = remain_df[~remain_df.index.isin(val_indices)].copy()

    return train_df, val_df, test_df


def _default_audio_dir(transcript_file: str) -> Path:
    """
    Infer audio directory relative to transcript file.
    """
    tpath = Path(transcript_file).resolve()
    return tpath.parent / "audio_input"


def create_speaker_disjoint_split(  # kept name for compatibility
    transcript_file: str,
    test_speaker_ratio: float = 0.2,  # unused (kept for signature compatibility)
    val_utterance_ratio: float = 0.1,  # unused (kept for signature compatibility)
    seed: int = 42,
    save_split_info: bool = True,
    split_info_path: str = "data/split_info.json",
    audio_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Build utterance-level split with native/non-native stratified test set.
    """
    audio_root = Path(audio_dir) if audio_dir else _default_audio_dir(transcript_file)

    print(f"Loading transcript data from {transcript_file}...")
    df = pd.read_csv(transcript_file)
    df = df.rename(columns={df.columns[0]: "SentenceID", df.columns[1]: "Transcript"})
    df["Transcript"] = df["Transcript"].fillna("").astype(str)
    df = df[df["Transcript"].str.strip() != ""]

    # Locate audio files
    df["AudioPath"] = df["SentenceID"].apply(lambda uid: find_audio_path(uid, audio_root))
    df = df.dropna(subset=["AudioPath"])

    # Validate audio readability
    valid_rows = []
    for _, row in df.iterrows():
        try:
            sf.info(row["AudioPath"])
            valid_rows.append(row)
        except Exception:
            continue

    if not valid_rows:
        raise ValueError("No valid audio files found for the provided transcripts and audio directory.")

    df = pd.DataFrame(valid_rows)

    # Parse gender/native flags
    df[["Gender", "Native"]] = df["SentenceID"].apply(lambda uid: pd.Series(parse_gender_native(uid)))
    df = df.dropna(subset=["Gender", "Native"])

    base_df = df[["SentenceID", "Transcript", "Gender", "Native", "AudioPath"]]

    train_df, val_df, test_df = split_utterances(base_df, seed=seed)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["Split"] = "train"
    val_df["Split"] = "val"
    test_df["Split"] = "test"

    out_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    print(f"Saved {len(out_df)} matched rows to {split_info_path}")
    split_counts = out_df["Split"].value_counts()
    for split_name in ["train", "val", "test"]:
        print(f"{split_name}: {split_counts.get(split_name, 0)}")

    split_dict = {
        "train": train_df["SentenceID"].tolist(),
        "val": val_df["SentenceID"].tolist(),
        "test": test_df["SentenceID"].tolist(),
    }

    if save_split_info:
        split_info = {
            "seed": seed,
            "method": "utterance_stratified_native",
            "test_native_frac": 0.10,
            "test_non_native_frac": 0.10,
            "val_frac": 0.10,
            "audio_dir": str(audio_root),
            "split": split_dict,
        }
        Path(split_info_path).parent.mkdir(parents=True, exist_ok=True)
        with open(split_info_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        # Also persist a CSV view similar to the user script.
        out_csv = Path(split_info_path).with_suffix(".csv")
        out_df.to_csv(out_csv, index=False)
        print(f"Split information saved to {split_info_path}")
        print(f"Split rows (with metadata) saved to {out_csv}")

    return split_dict


def load_split_info(split_info_path: str = "data/split_info.json") -> Dict:
    """
    Load previously saved split information.
    """
    with open(split_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    create_speaker_disjoint_split(
        transcript_file="data/transcripts.csv",
        seed=42,
        save_split_info=True,
        split_info_path="data/split_info.json",
        audio_dir="data/audio_input",
    )

    print("\n" + "=" * 60)
    print("Split created successfully!")
    print("=" * 60)
