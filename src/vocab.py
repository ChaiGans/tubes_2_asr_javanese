"""
Character-level Vocabulary for Javanese ASR
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from .text_preprocessor import TextPreprocessor


class Vocabulary:
    """
    Character-level vocabulary with special tokens.
    
    Special tokens:
        <pad>: Padding token (index 0)
        <sos>: Start of sequence (index 1)
        <eos>: End of sequence (index 2)
        <unk>: Unknown character (index 3)
        <blank>: CTC blank token (index 4)
    """
    
    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"
    BLANK = "<blank>"
    
    def __init__(self):
        # Fixed special tokens
        self.special_tokens = [self.PAD, self.SOS, self.EOS, self.UNK, self.BLANK]
        
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        
        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token
    
    @property
    def pad_idx(self) -> int:
        return self.char2idx[self.PAD]
    
    @property
    def sos_idx(self) -> int:
        return self.char2idx[self.SOS]
    
    @property
    def eos_idx(self) -> int:
        return self.char2idx[self.EOS]
    
    @property
    def unk_idx(self) -> int:
        return self.char2idx[self.UNK]
    
    @property
    def blank_idx(self) -> int:
        return self.char2idx[self.BLANK]
    
    def __len__(self) -> int:
        return len(self.char2idx)
    
    def build_from_transcripts(self, transcripts: List[str]) -> None:
        """
        Build vocabulary from a list of transcripts.
        
        Args:
            transcripts: List of text transcripts
        """
        # Collect all unique characters
        unique_chars = set()
        for text in transcripts:
            unique_chars.update(text)
        
        # Sort for consistency
        unique_chars = sorted(unique_chars)
        
        # Add to vocabulary (after special tokens)
        next_idx = len(self.special_tokens)
        for char in unique_chars:
            if char not in self.char2idx:
                self.char2idx[char] = next_idx
                self.idx2char[next_idx] = char
                next_idx += 1
        
        print(f"Built vocabulary with {len(self)} tokens ({len(unique_chars)} characters + {len(self.special_tokens)} special tokens)")
    
    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to indices.
        
        Args:
            text: Input text
            add_sos: Whether to prepend <sos> token
            add_eos: Whether to append <eos> token
        
        Returns:
            List of token indices
        """
        indices = []
        
        if add_sos:
            indices.append(self.sos_idx)
        
        for char in text:
            indices.append(self.char2idx.get(char, self.unk_idx))
        
        if add_eos:
            indices.append(self.eos_idx)
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode indices to text.
        
        Args:
            indices: List of token indices
            remove_special: Whether to remove special tokens
        
        Returns:
            Decoded text
        """
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, self.UNK)
            
            if remove_special and char in self.special_tokens:
                continue
            
            chars.append(char)
        
        return ''.join(chars)
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved vocabulary to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        vocab = cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab.char2idx = data['char2idx']
        vocab.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        
        print(f"Loaded vocabulary with {len(vocab)} tokens from {filepath}")
        return vocab


def build_vocab_from_file(
    transcript_file: str, 
    save_path: Optional[str] = None,
    preprocessor: Optional[TextPreprocessor] = None,
    use_preprocessing: bool = True
) -> Vocabulary:
    """
    Build vocabulary from transcript file.
    
    Expected format: CSV with columns "SentenceID,Transcript,Device"
    
    Args:
        transcript_file: Path to transcript file
        save_path: Optional path to save vocabulary JSON
        preprocessor: Optional TextPreprocessor instance. If None and use_preprocessing is True,
                     a default preprocessor will be created
        use_preprocessing: Whether to apply text preprocessing (default: True)
    
    Returns:
        Vocabulary object
    """
    transcripts = []
    
    # Create default preprocessor if needed
    if use_preprocessing and preprocessor is None:
        preprocessor = TextPreprocessor(seed=42, ampersand_replacement='random')
        print("Using default TextPreprocessor (seed=42, ampersand='random')")

    print(f"Reading transcripts from {transcript_file}")
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        # Skip header line
        next(f, None)
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by comma (CSV format)
            parts = line.split(',')
            if len(parts) >= 2:
                transcript = parts[1].strip()
                
                # Apply preprocessing if enabled
                if use_preprocessing and preprocessor:
                    transcript = preprocessor.preprocess(transcript)
                
                transcripts.append(transcript)
    
    print(f"Read {len(transcripts)} transcripts from {transcript_file}")
    if use_preprocessing:
        print(f"Applied text preprocessing: lowercase, punctuation removal, & → la/dan")
    
    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_from_transcripts(transcripts)
    
    # Save if requested
    if save_path:
        vocab.save(save_path)
    
    return vocab


if __name__ == "__main__":
    # Test vocabulary
    print("Testing Vocabulary...")
    
    # Sample transcripts
    transcripts = [
        "aku libur sedina merga ana banjir",
        "wong jawa seneng mangan nasi",
        "sekolah diliburake merga udan deres"
    ]
    
    # Build vocab
    vocab = Vocabulary()
    vocab.build_from_transcripts(transcripts)
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Special tokens: {vocab.special_tokens}")
    print(f"Sample characters: {list(vocab.char2idx.keys())[5:15]}")
    
    # Test encoding
    text = "aku seneng"
    encoded = vocab.encode(text, add_sos=True, add_eos=True)
    print(f"\nOriginal: '{text}'")
    print(f"Encoded: {encoded}")
    
    # Test decoding
    decoded = vocab.decode(encoded, remove_special=True)
    print(f"Decoded: '{decoded}'")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    vocab.save(temp_path)
    loaded_vocab = Vocabulary.load(temp_path)
    
    # Verify
    assert len(vocab) == len(loaded_vocab)
    assert vocab.char2idx == loaded_vocab.char2idx
    print("\nSave/load test passed!")
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("\nVocabulary test passed!")
