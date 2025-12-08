from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vocab import Vocabulary
from src.utils import read_transcript
from config import Config


def build_vocabularies():
    """Build and save both character and word-level vocabularies."""
    
    config = Config()

    print(f"Reading transcripts from {config.transcript_file}...")
    transcripts = read_transcript(config.transcript_file)
    print(f"Found {len(transcripts)} transcripts")
    
    print("\n" + "="*50)
    print("Building CHARACTER-level vocabulary...")
    print("="*50)
    vocab_char = Vocabulary(token_type="char")
    vocab_char.build_from_transcripts(transcripts)
    
    char_vocab_path = Path("data/vocab_char.json")
    vocab_char.save(str(char_vocab_path))
    
    print(f"\nCharacter vocabulary summary:")
    print(f"  Total tokens: {len(vocab_char)}")
    print(f"  Special tokens: {vocab_char.special_tokens}")
    
    sample_chars = [vocab_char.idx2token[i] for i in range(5, min(15, len(vocab_char)))]
    print(f"  Sample characters: {sample_chars}")
    
    print("\n" + "="*50)
    print("Building WORD-level vocabulary...")
    print("="*50)
    vocab_word = Vocabulary(token_type="word")
    vocab_word.build_from_transcripts(transcripts, min_freq=1)
    
    word_vocab_path = Path("data/vocab_word.json")
    vocab_word.save(str(word_vocab_path))
    
    print(f"\nWord vocabulary summary:")
    print(f"  Total tokens: {len(vocab_word)}")
    print(f"  Special tokens: {vocab_word.special_tokens}")
    
    sample_words = [vocab_word.idx2token[i] for i in range(5, min(15, len(vocab_word)))]
    print(f"  Sample words: {sample_words}")
    
    print("\n" + "="*50)
    print("Testing encoding/decoding...")
    print("="*50)
    
    test_text = transcripts[0] if transcripts else "aku seneng"
    print(f"\nTest text: '{test_text}'")
    
    char_encoded = vocab_char.encode(test_text)
    char_decoded = vocab_char.decode(char_encoded)
    print(f"\nCharacter-level:")
    print(f"  Encoded: {char_encoded[:20]}{'...' if len(char_encoded) > 20 else ''}")
    print(f"  Decoded: '{char_decoded}'")
    print(f"  Match: {test_text == char_decoded}")
    
    word_encoded = vocab_word.encode(test_text)
    word_decoded = vocab_word.decode(word_encoded)
    print(f"\nWord-level:")
    print(f"  Encoded: {word_encoded[:20]}{'...' if len(word_encoded) > 20 else ''}")
    print(f"  Decoded: '{word_decoded}'")
    print(f"  Match: {test_text == word_decoded}")
    
    print("\n" + "="*50)
    print("DONE! Vocabularies saved to:")
    print(f"  - {char_vocab_path.absolute()}")
    print(f"  - {word_vocab_path.absolute()}")
    print("="*50)
    
    return vocab_char, vocab_word


if __name__ == "__main__":
    build_vocabularies()
