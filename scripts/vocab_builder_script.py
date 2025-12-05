from pathlib import Path
from src.vocab import Vocabulary, build_vocab_from_file
from config import Config
from src.utils import set_seed

# Load config
cfg = Config()

# Set seed for reproducibility
# set_seed(cfg.seed)

vocab_path = Path(cfg.vocab_path)
print(f"Building vocabulary from {cfg.transcript_file}")
vocab = build_vocab_from_file(cfg.transcript_file, save_path=str(vocab_path))

print(f"Vocabulary size: {len(vocab)}")
