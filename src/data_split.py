"""
Speaker-disjoint split with audio validation and flexible path matching.

Logic:
- Keep only utterances that have a readable WAV file.
- Parse gender/native from ID pattern: speakerXX_m_n_uttYY (m/f, n/nn).
- Test speakers: 10% native + 10% non-native (by speaker).
- From remaining speakers, assign train speakers until ~70% of total utterances is reached; the rest go to val.
- Save split lists to JSON (default: data/split_info.json) and an optional CSV view.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import json

import pandas as pd
import soundfile as sf


def _default_audio_dir(transcript_file: str) -> Path:
    """Infer audio directory relative to transcript file."""
    tpath = Path(transcript_file).resolve()
    return tpath.parent / "audio_input"


def find_audio_path(utt_id: str, audio_dir: Path) -> Optional[Path]:
    """
    Locate audio file for a given utterance ID by searching for <utt_id>.wav under audio_dir (recursive).
    Returns the first match if found, else None.
    """
    fname = f"{utt_id}.wav"
    matches = list(audio_dir.rglob(fname))
    if not matches:
        return None
    return matches[0].resolve()


def parse_gender_native(utt_id: str) -> Tuple[Optional[str], Optional[bool]]:
    """
    Parse gender/native tokens from utterance ID.
    Expected: speakerXX_m_n_uttYY or speakerXX_f_nn_uttYY, etc.
    Returns (gender_str, native_bool) or (None, None) if not parseable.
    """
    parts = utt_id.split("_")
    if len(parts) < 3:
        return None, None

    gender_token = parts[1].lower()
    native_token = parts[2].lower()

    gender = "male" if gender_token == "m" else "female" if gender_token == "f" else None
    native = True if native_token == "n" else False if native_token == "nn" else None

    if gender is None or native is None:
        return None, None

    return gender, native


def split_speakers(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Speaker-disjoint split:
      - Test speakers: 10% native + 10% non-native speakers.
      - Train speakers: enough from remaining to cover ~70% of utterances.
      - Val speakers: the rest of the remaining speakers.
    """
    rng = random.Random(seed)
    df = df.copy()
    df["Speaker"] = df["SentenceID"].apply(lambda x: x.split("_")[0])

    native_speakers = sorted(df[df["Native"] == True]["Speaker"].unique().tolist())
    non_native_speakers = sorted(df[df["Native"] == False]["Speaker"].unique().tolist())

    # Select test speakers
    n_native_test = max(1, int(len(native_speakers) * 0.10)) if native_speakers else 0
    n_non_native_test = max(1, int(len(non_native_speakers) * 0.10)) if non_native_speakers else 0

    test_native_spk = rng.sample(native_speakers, n_native_test) if n_native_test else []
    test_non_native_spk = rng.sample(non_native_speakers, n_non_native_test) if n_non_native_test else []
    test_speakers = set(test_native_spk + test_non_native_spk)

    # Remaining speakers for train+val
    remain_speakers = [s for s in df["Speaker"].unique() if s not in test_speakers]
    rng.shuffle(remain_speakers)

    total_rows = len(df)
    target_train_rows = int(total_rows * 0.70)
    train_speakers = set()
    current_total = 0

    for spk in remain_speakers:
        spk_rows = len(df[df["Speaker"] == spk])
        if current_total + spk_rows <= target_train_rows:
            train_speakers.add(spk)
            current_total += spk_rows

    val_speakers = set(remain_speakers) - train_speakers

    train_df = df[df["Speaker"].isin(train_speakers)].copy()
    val_df = df[df["Speaker"].isin(val_speakers)].copy()
    test_df = df[df["Speaker"].isin(test_speakers)].copy()

    return train_df, val_df, test_df


def create_speaker_disjoint_split(
    transcript_file: str,
    test_speaker_ratio: float = 0.2,  # kept for signature compatibility; actual split uses 10% native + 10% non-native speakers
    val_utterance_ratio: float = 0.1,  # unused
    seed: int = 42,
    save_split_info: bool = True,
    split_info_path: str = "data/split_info.json",
    audio_dir: Optional[str] = None,
    split_csv_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Build a speaker-disjoint split with audio validation.
    Returns a dict with utterance ID lists for train/val/test.
    """
    audio_root = Path(audio_dir) if audio_dir else _default_audio_dir(transcript_file)

    print(f"Loading transcript data from {transcript_file}...")
    df = pd.read_csv(transcript_file)
    df = df.rename(columns={df.columns[0]: "SentenceID", df.columns[1]: "Transcript"})
    df["Transcript"] = df["Transcript"].fillna("").astype(str)
    df = df[df["Transcript"].str.strip() != ""]

    # Map audio
    df["AudioPath"] = df["SentenceID"].apply(lambda uid: find_audio_path(uid, audio_root))
    missing_audio = df["AudioPath"].isna().sum()
    if missing_audio:
        print(f"Warning: {missing_audio} utterances have no matching audio file and will be dropped.")
    df = df.dropna(subset=["AudioPath"])

    # Validate audio readability
    valid_rows = []
    for _, row in df.iterrows():
        try:
            sf.info(str(row["AudioPath"]))
            valid_rows.append(row)
        except Exception:
            continue

    if not valid_rows:
        raise ValueError("No valid audio files found for the provided transcripts and audio directory.")

    df = pd.DataFrame(valid_rows)

    # Parse gender + native
    df[["Gender", "Native"]] = df["SentenceID"].apply(lambda uid: pd.Series(parse_gender_native(uid)))
    before_parse = len(df)
    df = df.dropna(subset=["Gender", "Native"])
    if len(df) == 0:
        raise ValueError("No utterances with parseable gender/native tokens (expected pattern: speakerXX_m_n_uttYY).")
    dropped = before_parse - len(df)
    if dropped:
        print(f"Dropped {dropped} rows due to unparsable gender/native tags.")

    base_df = df[["SentenceID", "Transcript", "Gender", "Native", "AudioPath"]]

    # Perform speaker-disjoint split
    train_df, val_df, test_df = split_speakers(base_df, seed=seed)

    # Label splits
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["Split"] = "train"
    val_df["Split"] = "val"
    test_df["Split"] = "test"

    out_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}, total={len(out_df)}")

    split_dict = {
        "train": train_df["SentenceID"].tolist(),
        "val": val_df["SentenceID"].tolist(),
        "test": test_df["SentenceID"].tolist(),
    }

    if save_split_info:
        split_info = {
            "seed": seed,
            "method": "speaker_disjoint",
            "test_native_frac": 0.10,
            "test_non_native_frac": 0.10,
            "target_train_frac": 0.70,
            "audio_dir": str(audio_root),
            "split": split_dict,
        }
        Path(split_info_path).parent.mkdir(parents=True, exist_ok=True)
        with open(split_info_path, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        print(f"Split information saved to {split_info_path}")

        # Optional CSV view
        out_csv = Path(split_csv_path) if split_csv_path else Path(split_info_path).with_suffix(".csv")
        out_df.to_csv(out_csv, index=False)
        print(f"Split rows (with metadata) saved to {out_csv}")

    return split_dict


def load_split_info(split_info_path: str = "data/split_info.json") -> Dict:
    """Load previously saved split information."""
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
