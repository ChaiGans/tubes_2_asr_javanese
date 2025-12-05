import os
import sys
import re
import soundfile as sf
from collections import defaultdict

# Filename format: speakerXX_m_n_uttYY.wav
PATTERN = re.compile(r"^speaker(\d{2})_(m|f)_(n|nn)_utt(\d{2})\.wav$")

EXPECTED_SPEAKERS = 70
EXPECTED_UTT_PER_SPK = 30


def analyze_audio_folder(folder):
    errors = []
    speaker_data = defaultdict(lambda: {"gender": None, "native": None, "utts": [], "durations": []})

    audio_durations = []

    all_files = os.listdir(folder)

    for fname in all_files:
        if not fname.lower().endswith(".wav"):
            continue

        match = PATTERN.match(fname)
        if not match:
            errors.append(f"Invalid filename format: {fname}")
            continue

        spk_id, gender, nat, utt = match.groups()

        # Read duration
        filepath = os.path.join(folder, fname)
        try:
            audio, sr = sf.read(filepath)
            duration_sec = len(audio) / sr
        except Exception as e:
            errors.append(f"Cannot read audio {fname}: {e}")
            continue

        # Store data
        speaker_data[spk_id]["gender"] = gender
        speaker_data[spk_id]["native"] = nat
        speaker_data[spk_id]["utts"].append(int(utt))
        speaker_data[spk_id]["durations"].append(duration_sec)
        audio_durations.append(duration_sec)

    # ====================== Demographic Counts ======================
    # Four buckets:
    # m-n, m-nn, f-n, f-nn
    demo_count = {
        "m_n": 0,
        "m_nn": 0,
        "f_n": 0,
        "f_nn": 0,
    }

    for info in speaker_data.values():
        g = info["gender"]
        n = info["native"]
        if g is None or n is None:
            continue
        key = f"{g}_{n}"
        if key in demo_count:
            demo_count[key] += 1

    # ====================== Missing Speaker / Utt Checks ======================
    expected_ids = {f"{i:02d}" for i in range(1, EXPECTED_SPEAKERS + 1)}
    found_ids = set(speaker_data.keys())

    missing_speakers = sorted(expected_ids - found_ids)
    extra_speakers = sorted(found_ids - expected_ids)

    # Missing utts per speaker
    missing_utts = {}
    for spk_id, info in speaker_data.items():
        expected_utts = {i for i in range(1, EXPECTED_UTT_PER_SPK + 1)}
        found_utts = set(info["utts"])
        missing = expected_utts - found_utts
        if missing:
            missing_utts[spk_id] = sorted(list(missing))

    # ====================== Duration Statistics ======================
    if audio_durations:
        total_duration = sum(audio_durations)
        avg_duration = total_duration / len(audio_durations)
        min_duration = min(audio_durations)
        max_duration = max(audio_durations)
    else:
        total_duration = avg_duration = min_duration = max_duration = 0

    total_audio = len(audio_durations)
    total_speakers = len(speaker_data)

    # ====================== OUTPUT ======================
    print("=====================================")
    print(" AUDIO FORMAT CHECK REPORT")
    print("=====================================")

    if errors:
        print("\n❌ Filename / audio errors:")
        for err in errors:
            print(" -", err)
    else:
        print("\n✔ No filename errors!")

    print("\n-------------------------------------")
    print(" SPEAKER CHECK")
    print("-------------------------------------")
    print(f"Expected speakers : {EXPECTED_SPEAKERS}")
    print(f"Found speakers    : {total_speakers}")

    if missing_speakers:
        print("\n❌ Missing speakers:")
        print(", ".join(missing_speakers))
    else:
        print("\n✔ No missing speakers")

    if extra_speakers:
        print("\n⚠ Extra speakers:")
        print(", ".join(extra_speakers))

    print("\n-------------------------------------")
    print(" UTTERANCE COMPLETENESS")
    print("-------------------------------------")
    if missing_utts:
        print("❌ Missing utterances:")
        for spk, miss in missing_utts.items():
            print(f" - speaker{spk}: missing {miss}")
    else:
        print("✔ All speakers have 30 utterances")

    print("\n-------------------------------------")
    print(" AUDIO DURATION STATISTICS")
    print("-------------------------------------")
    print(f"Total audio files : {total_audio}")
    print(f"Total duration    : {total_duration:.2f} sec ({total_duration/60:.2f} min)")
    print(f"Average duration  : {avg_duration:.2f} sec")
    print(f"Shortest clip     : {min_duration:.2f} sec")
    print(f"Longest clip      : {max_duration:.2f} sec")

    print("\n-------------------------------------")
    print(" DEMOGRAPHIC DISTRIBUTION")
    print("-------------------------------------")
    print(f"Male Native (m_n)         : {demo_count['m_n']}")
    print(f"Male Non-Native (m_nn)    : {demo_count['m_nn']}")
    print(f"Female Native (f_n)       : {demo_count['f_n']}")
    print(f"Female Non-Native (f_nn)  : {demo_count['f_nn']}")

    print("=====================================")
    print(" CHECK COMPLETE")
    print("=====================================")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_formatter_check.py <folder_name>")
        sys.exit(1)

    analyze_audio_folder(sys.argv[1])
