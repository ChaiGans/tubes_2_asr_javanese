import os
import json
import csv
import re
import glob

def parse_notebook(filepath):
    # Dictionary to hold results with default empty values
    data = {
        "experiment_name": os.path.splitext(os.path.basename(filepath))[0],
        "model_name": "",
        "batch_size": "",
        "num_epochs": "",
        "lr": "",
        "use_lora": "",
        "trainable_params": "",
        "all_params": "",
        "trainable_percent": "",
        "test_wer": "",
        "test_cer": "",
        "training_time_minutes": "",
        "early_stopping": False,
        "early_stopping_epoch": "",
        "best_val_loss": ""
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return data

    source_text = ""
    output_text = ""

    # Aggregate all source code and output text
    for cell in nb.get('cells', []):
        cell_type = cell.get('cell_type')
        if cell_type == 'code':
            # Code source
            source = cell.get('source', [])
            if isinstance(source, str):
                source_text += "\n" + source
            else:
                source_text += "\n" + "".join(source)

            # Outputs
            for output in cell.get('outputs', []):
                # Valid text output formats
                if 'text' in output:
                    output_text += "\n" + "".join(output['text'])
                elif 'data' in output and 'text/plain' in output['data']:
                    output_text += "\n" + "".join(output['data']['text/plain'])
                # Stream outputs
                elif 'name' in output and output['name'] in ['stdout', 'stderr']:
                    if 'text' in output:
                         output_text += "\n" + "".join(output['text'])

    # --- 1. CONFIGURATION extraction (from Source) ---
    # Keys we want to extract from code assignments
    config_keys = ["model_name", "batch_size", "num_epochs", "lr", "use_lora"]

    for key in config_keys:
        # Regex explanation:
        # \b{key}     : Word boundary matching the key
        # \s*         : Optional whitespace
        # (?:...)?    : Optional non-capturing group for type hint (e.g., : int)
        # =           : Assignment operator
        # \s*         : Optional whitespace
        # ( ... )     : Capturing group for the value
        # [^#\n,\)]+  : Match chars that are NOT comment start, newline, comma, or closing parenthesis
        pattern = re.compile(rf"\b{key}\s*(?::\s*[\w\[\]]+\s*)?=\s*([^#\n,\)]+)")
        match = pattern.search(source_text)
        if match:
            raw_val = match.group(1).strip()
            # Clean quotes
            val = raw_val.replace('"', '').replace("'", "")
            data[key] = val

    # --- 2. METRICS extraction (from Output, mostly) ---

    # Trainable Params
    # Pattern: "trainable params: 9,437,184 || all params: 773,295,104"
    # Also support if users printed it differently, but this fits the example
    tp_match = re.search(r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)", output_text, re.IGNORECASE)
    if tp_match:
        try:
            t_params = int(tp_match.group(1).replace(',', ''))
            a_params = int(tp_match.group(2).replace(',', ''))
            data["trainable_params"] = t_params
            data["all_params"] = a_params
            if a_params > 0:
                data["trainable_percent"] = (t_params / a_params) * 100
        except:
            pass

    # Test WER / CER
    # Pattern: "Test WER: 0.7895" or "Test WER = 0.7895" or "WER: 0.7895"
    wer_match = re.search(r"(?:Test\s+)?WER\s*[:=]\s*([\d.]+)", output_text, re.IGNORECASE)
    if wer_match:
        data["test_wer"] = wer_match.group(1)

    cer_match = re.search(r"(?:Test\s+)?CER\s*[:=]\s*([\d.]+)", output_text, re.IGNORECASE)
    if cer_match:
        data["test_cer"] = cer_match.group(1)

    # Training Time
    # Pattern: "Total train+eval time: 30.09 min (1805.3 s)" or "Training time: 45.5 s"
    # Look for number followed by unit
    time_match = re.search(r"(?:Time|Training time|Total train\+eval time)\s*[:=]\s*([\d.]+)\s*(min|s|sec)", output_text, re.IGNORECASE)
    if time_match:
        try:
            val = float(time_match.group(1))
            unit = time_match.group(2).lower()
            if 's' in unit:
                data["training_time_minutes"] = val / 60.0
            else:
                data["training_time_minutes"] = val
        except:
            pass

    # Best Val Loss
    # Pattern: "Best val loss: 0.7788"
    bvl_match = re.search(r"Best val loss\s*[:=]\s*([\d.]+)", output_text, re.IGNORECASE)
    if bvl_match:
        data["best_val_loss"] = bvl_match.group(1)

    # Early Stopping
    # Pattern: "Early stopping at epoch 7"
    # If found, set True and extract epoch.
    es_match = re.search(r"Early stopping at epoch\s*(\d+)", output_text, re.IGNORECASE)
    if es_match:
        data["early_stopping"] = True
        data["early_stopping_epoch"] = es_match.group(1)
    else:
        # Check if matched "Stopped at epoch"
        stopped_match = re.search(r"Stopped at epoch\s*[:=]?\s*(\d+)", output_text, re.IGNORECASE)
        if stopped_match:
             data["early_stopping"] = True
             data["early_stopping_epoch"] = stopped_match.group(1)

    return data

def main():
    # Find all .ipynb files in current directory
    notebooks = glob.glob("*.ipynb")
    results = []

    # Define CSV columns
    fieldnames = [
        "experiment_name",
        "model_name",
        "batch_size",
        "num_epochs",
        "lr",
        "use_lora",
        "trainable_params",
        "all_params",
        "trainable_percent",
        "test_wer",
        "test_cer",
        "training_time_minutes",
        "early_stopping",
        "early_stopping_epoch",
        "best_val_loss"
    ]

    print(f"Found {len(notebooks)} notebooks. Processing...")

    for nb_file in notebooks:
        try:
            res = parse_notebook(nb_file)
            # Create a row dictionary for CSV
            row = {k: res.get(k, "") for k in fieldnames}
            results.append(row)
        except Exception as e:
            print(f"Failed to process {nb_file}: {e}")

    # Write to CSV
    output_file = "results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Successfully wrote results to {output_file}")

if __name__ == "__main__":
    main()
