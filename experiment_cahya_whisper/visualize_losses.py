import json
import re
import matplotlib.pyplot as plt
import glob
import os

# pattern to extract metrics
# Matches format like: "Epoch 1/10 | Train loss: 6.6481 | Val loss: 3.7092"
pattern = re.compile(r"Epoch\s+(\d+)/\d+\s+\|\s+Train loss:\s+([\d.]+)\s+\|\s+Val loss:\s+([\d.]+)")

def parse_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    data = []

    for cell in nb.get('cells', []):
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                    text_lines = output.get('text', [])
                    # text can be a list of strings or a single string
                    if isinstance(text_lines, str):
                        text_lines = [text_lines]

                    for line in text_lines:
                        # Handle case where line itself contains multiple lines
                        for subline in line.split('\n'):
                            match = pattern.search(subline)
                            if match:
                                epoch = int(match.group(1))
                                train_loss = float(match.group(2))
                                val_loss = float(match.group(3))
                                data.append((epoch, train_loss, val_loss))

    # Sort by epoch just in case, though usually sequential
    data.sort(key=lambda x: x[0])

    # Unzip
    if not data:
        return [], [], []

    epochs, train_losses, val_losses = zip(*data)
    return epochs, train_losses, val_losses

def main():
    notebooks = glob.glob("*.ipynb")
    print(f"Found {len(notebooks)} notebooks in current directory.")

    for nb_path in notebooks:
        print(f"Processing {nb_path}...")
        epochs, train_losses, val_losses = parse_notebook(nb_path)

        if not epochs:
            print(f"  No training data found in {nb_path}")
            continue

        print(f"  Found {len(epochs)} epochs: {epochs}")

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-')
        plt.plot(epochs, val_losses, label='Validation Loss', marker='o', linestyle='-')
        plt.title(f'Losses for {os.path.basename(nb_path)}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        output_file = nb_path.replace('.ipynb', '_loss_graph.png')
        plt.savefig(output_file)
        plt.close()
        print(f"  Saved graph to {output_file}")

if __name__ == "__main__":
    main()
