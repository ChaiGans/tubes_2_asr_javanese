from pathlib import Path
from src.vocab import Vocabulary, build_vocab_from_file
from src.text_preprocessor import TextPreprocessor
from config import Config
from src.utils import set_seed

# Load config
cfg = Config()

# Set seed for reproducibility
set_seed(cfg.seed)

vocab_path = Path(cfg.vocab_path)

# ===== Option 1: Use default preprocessing (recommended) =====
# This will automatically apply preprocessing with default settings:
# - Lowercase all text
# - Remove punctuation
# - Replace & with 'la' or 'dan' (randomized with seed=42)
print("=" * 80)
print("Building vocabulary WITH preprocessing (default)")
print("=" * 80)
print(f"Building vocabulary from {cfg.transcript_file}")
vocab = build_vocab_from_file(
    cfg.transcript_file, 
    save_path=str(vocab_path),
    use_preprocessing=True  # This is True by default
)
print(f"Vocabulary size: {len(vocab)}")
print(f"Saved to: {vocab_path}")

# ===== Option 2: Custom preprocessing settings =====
# If you want to control the preprocessing behavior (e.g., always use 'la' for '&')
print("\n" + "=" * 80)
print("Example: Custom preprocessing (always replace & with 'la')")
print("=" * 80)
custom_preprocessor = TextPreprocessor(
    seed=42,
    ampersand_replacement='la'  # 'la', 'dan', or 'random'
)
vocab_custom = build_vocab_from_file(
    cfg.transcript_file,
    preprocessor=custom_preprocessor,
    use_preprocessing=True
)
print(f"Vocabulary size with custom preprocessing: {len(vocab_custom)}")

# ===== Option 3: No preprocessing (keep original text) =====
# If you want to preserve original text including capitalization and punctuation
print("\n" + "=" * 80)
print("Example: WITHOUT preprocessing (original text)")
print("=" * 80)
vocab_no_preproc = build_vocab_from_file(
    cfg.transcript_file,
    use_preprocessing=False
)
print(f"Vocabulary size without preprocessing: {len(vocab_no_preproc)}")

# Show comparison
print("\n" + "=" * 80)
print("Vocabulary Size Comparison")
print("=" * 80)
print(f"With default preprocessing:  {len(vocab)} characters")
print(f"With custom preprocessing:   {len(vocab_custom)} characters")
print(f"Without preprocessing:       {len(vocab_no_preproc)} characters")
print("\nNote: Preprocessing typically reduces vocab size by removing punctuation")
print("      and normalizing case, which helps model performance.")
