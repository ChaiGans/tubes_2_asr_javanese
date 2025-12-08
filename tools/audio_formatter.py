import os
import sys
import re
import json
import subprocess
import shutil
from pathlib import Path
import soundfile as sf

try:
    import imageio_ffmpeg
    FFMPEG_BUNDLED = True
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_BUNDLED = False
    FFMPEG_PATH = "ffmpeg"

PATTERN = re.compile(r"^speaker(\d{2})_(m|f)_(n|nn)_utt(\d{2})\.wav$")


def check_ffmpeg_available():
    if FFMPEG_BUNDLED:
        return True
    return shutil.which("ffmpeg") is not None


def fix_filename_format(fname: str):
    import unicodedata

    original = fname

    fname = ''.join(char for char in fname if unicodedata.category(char) not in ['Cf', 'Cc', 'Mn'])

    fname = fname.lower()

    fname = re.sub(r"speaker(\d)_", r"speaker0\1_", fname)

    fname = re.sub(r"utt(\d)\.wav$", r"utt0\1.wav", fname)
    fname = re.sub(r"utt(\d)\.wav$", r"utt0\1.wav", fname)

    return fname, (original != fname)


def convert_broken_wav(filepath: Path):
    fixed_path = filepath.with_suffix(".fixed.wav")

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(filepath),
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(fixed_path)
    ]

    try:
        print(f"  → Attempting to fix: {filepath.name}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

        if result.returncode != 0:
            print(f"  ✗ ffmpeg failed with code {result.returncode}")
            if fixed_path.exists():
                os.remove(fixed_path)
            return False

        audio, sr = sf.read(fixed_path)

        os.replace(fixed_path, filepath)
        print(f"  ✓ Successfully fixed: {filepath.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        if fixed_path.exists():
            os.remove(fixed_path)
        return False


def process_folder(folder: str, report_file: str):
    folder = Path(folder)
    report = {"renamed": [], "reencoded": [], "failed": []}

    ffmpeg_available = check_ffmpeg_available()
    if FFMPEG_BUNDLED:
        print(f"✓ Using bundled ffmpeg: {FFMPEG_PATH}\n")
    elif not ffmpeg_available:
        print("⚠ WARNING: ffmpeg not found!")
        print("  Some audio files may need re-encoding, which requires ffmpeg.")
        print("  Install via pip: pip install imageio-ffmpeg")
        print("  Or system-wide: winget install ffmpeg")
        print("  Continuing with filename fixes only...\n")
    else:
        print("✓ Using system ffmpeg\n")

    for fname in os.listdir(folder):
        path = folder / fname

        if not fname.lower().endswith(".wav"):
            continue

        new_name, changed = fix_filename_format(fname)
        new_path = folder / new_name

        if changed and new_name != fname:
            os.rename(path, new_path)
            report["renamed"].append({"from": fname, "to": new_name})
            path = new_path

        try:
            audio, sr = sf.read(path)
        except Exception:
            if not ffmpeg_available:
                report["failed"].append(new_path.name)
            else:
                ok = convert_broken_wav(path)
                if ok:
                    report["reencoded"].append(new_path.name)
                else:
                    report["failed"].append(new_path.name)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("=====================================")
    print(" AUDIO FORMATTER FINISHED")
    print("=====================================")
    print(f"Renamed files   : {len(report['renamed'])}")
    print(f"Re-encoded WAVs : {len(report['reencoded'])}")
    print(f"Failed files    : {len(report['failed'])}")
    print(f"Report saved to : {report_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python audio_formatter.py <folder> [report_file.json]")
        sys.exit(1)

    folder = sys.argv[1]
    report_file = sys.argv[2] if len(sys.argv) >= 3 else "fix_report.json"

    process_folder(folder, report_file)
