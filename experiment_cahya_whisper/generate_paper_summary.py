import csv
import os

def parse_float(val):
    try:
        return float(val)
    except:
        return 0.0

def parse_int(val):
    try:
        return int(val)
    except:
        return 0

def fmt_lr(lr_float):
    val = f"{lr_float:.0e}".replace("e-0", "e-")
    return val

def get_description(row, model_type, prev_row=None, index=0):
    batch = parse_int(row.get("batch_size", 0))
    epoch = parse_int(row.get("num_epochs", 0))
    lr = parse_float(row.get("lr", 0))
    lora = row.get("use_lora", "").upper() == "TRUE"

    desc = []

    if model_type == "Medium":
        base_batch = 2
        base_epoch = 10
        base_lr = 1e-5
        base_lora = True

        if batch == base_batch and epoch == base_epoch and abs(lr - base_lr) < 1e-9 and lora == base_lora:
            return "Asumsi Default Whisper Medium"

        if not lora and batch == 4 and epoch == 10 and abs(lr - base_lr) < 1e-9:
             return "Asumsi Default Whisper Tanpa LoRA"

        if abs(lr - base_lr) > 1e-9:
            if prev_row:
                prev_lr = parse_float(prev_row.get("lr", 0))
                if abs(lr - prev_lr) > 1e-9:
                     return f"Eksperimen Kenaikan LR ({fmt_lr(prev_lr)} ke {fmt_lr(lr)})"

            return f"Eksperimen Kenaikan LR dari default ({fmt_lr(base_lr)} ke {fmt_lr(lr)})"

        if batch != base_batch:
             if prev_row:
                 prev_batch = parse_int(prev_row.get("batch_size", 0))
                 if prev_batch != 0 and batch != prev_batch:
                     return f"Eksperimen Peningkatan batch size dari Eksperimen {index-1} ({prev_batch} ke {batch})"
             return f"Eksperimen Peningkatan batch size ({base_batch} ke {batch})"

        if epoch != base_epoch:
             return f"Eksperimen Perubahan Epoch ({base_epoch} ke {epoch})"

    elif model_type == "Small":
        base_batch = 4
        base_epoch = 5
        base_lr = 1e-5
        base_lora = False

        if batch == base_batch and epoch == base_epoch and abs(lr - base_lr) < 1e-9 and lora == base_lora:
            return "Asumsi Default Whisper Small"

        if lora:
            return "Eksperimen penggunaan LoRA pada model Small"

        if abs(lr - base_lr) > 1e-9:
             if prev_row:
                prev_lr = parse_float(prev_row.get("lr", 0))
                if abs(lr - prev_lr) > 1e-9:
                     return f"Eksperimen Peningkatan LR ({fmt_lr(prev_lr)} ke {fmt_lr(lr)})"
             return f"Eksperimen Peningkatan LR ({fmt_lr(base_lr)} ke {fmt_lr(lr)})"

        if batch != base_batch:
             if prev_row:
                 prev_batch = parse_int(prev_row.get("batch_size", 0))
                 if prev_batch != 0 and batch != prev_batch:
                     return f"Eksperimen peningkatan batch Size dari eksperimen {index-1} ({prev_batch} ke {batch})"
             return f"Eksperimen peningkatan batch Size ({base_batch} ke {batch})"

    diffs = []
    if batch != 0: diffs.append(f"Batch {batch}")
    if epoch != 0: diffs.append(f"Epoch {epoch}")
    if lr != 0: diffs.append(f"LR {fmt_lr(lr)}")
    if lora: diffs.append("LoRA")

    return "Eksperimen: " + ", ".join(diffs)

def generate_summary():
    input_file = "results.csv"
    output_file = "paper_summary.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading from {input_file}...")

    rows = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            if row.get("experiment_name"):
                 rows.append(row)

    processed_rows = []
    model_map = {
        "cahya/whisper-medium-id": "Medium",
        "cahya/whisper-small-id": "Small"
    }

    count = 1
    prev_row = None

    for row in rows:
        raw_model = row.get("model_name", "")
        tipe_whisper = model_map.get(raw_model, raw_model)

        description = get_description(row, tipe_whisper, prev_row, count)

        lora_str = row.get("use_lora", "").upper()
        if lora_str == "TRUE":
            lora_out = "Yes"
        elif lora_str == "FALSE":
            lora_out = "No"
        else:
            lora_out = lora_str

        wer = row.get("test_wer", "")
        cer = row.get("test_cer", "")
        time_min = row.get("training_time_minutes", "")

        new_row = {
            "No": count,
            "Tipe Whisper": tipe_whisper,
            "Tujuan Pengujian": description,
            "LoRA": lora_out,
            "Test WER": wer,
            "Test CER": cer,
            "Waktu Pelatihan (menit)": time_min
        }
        processed_rows.append(new_row)

        prev_row = row
        count += 1

    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        fieldnames = [
            "No",
            "Tipe Whisper",
            "Tujuan Pengujian",
            "LoRA",
            "Test WER",
            "Test CER",
            "Waktu Pelatihan (menit)"
        ]

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"Successfully wrote {len(processed_rows)} rows to {output_file}")

if __name__ == "__main__":
    generate_summary()
