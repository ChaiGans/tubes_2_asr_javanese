"""
Vocabulary for Javanese ASR (Character or Word level)
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from config import Config
from src.utils import read_transcript

class Vocabulary:
    """
    Vocabulary with special tokens. Supports both Character and Word levels.
    
    Special tokens:
        <pad>: Padding token (index 0)
        <sos>: Start of sequence (index 1)
        <eos>: End of sequence (index 2)
        <unk>: Unknown token (index 3)
        <blank>: CTC blank token (index 4)
    """
    
    PAD = "<pad>"
    SOS = "<sos>"
    EOS = "<eos>"
    UNK = "<unk>"
    BLANK = "<blank>"
    
    def __init__(self, token_type: str = "char"):
        """
        Args:
            token_type: "char" or "word"
        """
        self.token_type = token_type
        
        # Fixed special tokens
        self.special_tokens = [self.PAD, self.SOS, self.EOS, self.UNK, self.BLANK]
        
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        
        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD]
    
    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.SOS]
    
    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS]
    
    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK]
    
    @property
    def blank_idx(self) -> int:
        return self.token2idx[self.BLANK]
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def build_from_transcripts(self, transcripts: List[str], min_freq: int = 1) -> None:
        """
        Build vocabulary from a list of transcripts.
        
        Args:
            transcripts: List of text transcripts
            min_freq: Minimum frequency for a word to be included (only for word vocab)
        """
        unique_tokens = set()
        token_counts = {}
        
        for text in transcripts:
            if self.token_type == "char":
                tokens = list(text)
            else:
                tokens = text.strip().split()
            
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
                unique_tokens.add(token)
        
        # Filter by frequency for word vocab
        if self.token_type == "word" and min_freq > 1:
            unique_tokens = {t for t in unique_tokens if token_counts[t] >= min_freq}
        
        # Sort for consistency
        sorted_tokens = sorted(unique_tokens)
        
        # Add to vocabulary (after special tokens)
        next_idx = len(self.special_tokens)
        for token in sorted_tokens:
            if token not in self.token2idx:
                self.token2idx[token] = next_idx
                self.idx2token[next_idx] = token
                next_idx += 1
        
        print(f"Built {self.token_type}-level vocabulary with {len(self)} tokens")
    
    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to indices.
        """
        indices = []
        
        if add_sos:
            indices.append(self.sos_idx)
        
        if self.token_type == "char":
            tokens = list(text)
        else:
            tokens = text.strip().split()
            
        for token in tokens:
            indices.append(self.token2idx.get(token, self.unk_idx))
        
        if add_eos:
            indices.append(self.eos_idx)
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode indices to text.
        """
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, self.UNK)
            
            if remove_special and token in self.special_tokens:
                continue
            
            tokens.append(token)
        
        if self.token_type == "char":
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            'token_type': self.token_type,
            'token2idx': self.token2idx,
            'idx2token': {str(k): v for k, v in self.idx2token.items()}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved vocabulary to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        vocab = cls(token_type=data.get('token_type', 'char'))
        vocab.token2idx = data['token2idx']
        vocab.idx2token = {int(k): v for k, v in data['idx2token'].items()}
        
        print(f"Loaded {vocab.token_type}-level vocabulary with {len(vocab)} tokens from {filepath}")
        return vocab


def build_vocab_from_file(transcript_file: str, save_path: Optional[str] = None, token_type: str = "char") -> Vocabulary:
    """
    Build vocabulary from transcript file.
    """
    transcripts = read_transcript(transcript_file)
    print(f"Read {len(transcripts)} transcripts from {transcript_file}")
    
    vocab = Vocabulary(token_type=token_type)
    vocab.build_from_transcripts(transcripts)
    
    if save_path:
        vocab.save(save_path)
    
    return vocab


if __name__ == "__main__":
    # Test vocabulary
    print("Testing Vocabulary...")

    config = Config()
    transcripts = read_transcript(config.transcript_file)    
    
    # Test Char Vocab
    print("\n--- Character Level ---")
    vocab_char = Vocabulary(token_type="char")
    vocab_char.build_from_transcripts(transcripts)
    print(f"Size: {len(vocab_char)}")
    encoded = vocab_char.encode("aku seneng")
    decoded = vocab_char.decode(encoded)
    print(f"Encoded 'aku seneng': {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Test Word Vocab
    print("\n--- Word Level ---")
    vocab_word = Vocabulary(token_type="word")
    vocab_word.build_from_transcripts(transcripts)
    print(f"Size: {len(vocab_word)}")
    encoded = vocab_word.encode("aku seneng")
    decoded = vocab_word.decode(encoded)
    print(f"Encoded 'aku seneng': {encoded}")
    print(f"Decoded: '{decoded}'")
    
    print("\nVocabulary test passed!")
