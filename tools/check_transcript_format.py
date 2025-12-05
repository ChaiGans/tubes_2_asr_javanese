import csv
import sys
import re

# Pattern: speakerXX_[m/f]_[n/nn]_uttXX
PATTERN = re.compile(r"^speaker(\d{2})_(m|f)_(n|nn)_utt(\d{2})$")

REQUIRED_COLUMNS = ["SentenceID", "Transcript", "Device"]

def validate_csv(filename):
    errors = []
    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check required columns
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            print(f"❌ Missing required columns: {missing}")
            return

        for i, row in enumerate(reader, start=2):  # line numbers (header = line 1)
            sid = row["SentenceID"].strip()

            # Validate SentenceID format
            match = PATTERN.match(sid)
            if not match:
                errors.append(
                    f"Line {i}: Invalid SentenceID `{sid}` — expected format "
                    "speakerXX_[m|f]_[n|nn]_uttXX"
                )
                continue

            speaker_id, gender, nat, utt = match.groups()

            # Additional optional checks:
            if not (speaker_id.isdigit() and utt.isdigit()):
                errors.append(f"Line {i}: IDs must be numeric — got `{sid}`")

    # Print results
    if errors:
        print("❌ Format errors found:")
        for err in errors:
            print(" -", err)
    else:
        print("✅ CSV format is valid!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_format.py <csv_file>")
        sys.exit(1)

    validate_csv(sys.argv[1])
