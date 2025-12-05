"""
Text Preprocessor for Javanese ASR
Handles text normalization, lowercasing, punctuation removal, and special character transformations.
"""

import re
import random
import string
from typing import List, Optional


class TextPreprocessor:
    """
    Text preprocessor for Javanese transcripts.
    
    Features:
    - Lowercase all text
    - Remove punctuation
    - Replace '&' with 'la' or 'dan' (randomized)
    - Normalize whitespace
    """
    
    def __init__(self, seed: Optional[int] = None, ampersand_replacement: str = 'random'):
        """
        Initialize text preprocessor.
        
        Args:
            seed: Random seed for reproducible '&' replacement (None for true randomness)
            ampersand_replacement: How to replace '&':
                - 'random': randomly choose between 'la' and 'dan' for each occurrence
                - 'la': always replace with 'la'
                - 'dan': always replace with 'dan'
        """
        self.seed = seed
        self.ampersand_replacement = ampersand_replacement
        
        if seed is not None:
            random.seed(seed)
        
        # Define punctuation to remove (keep spaces for word separation)
        # Include common punctuation marks
        self.punctuation = set(string.punctuation)
        # Add additional unicode punctuation commonly used
        self.punctuation.update(['"', '"', ''', ''', '–', '—', '…'])
    
    def _replace_ampersand(self, text: str) -> str:
        """
        Replace '&' with 'la' or 'dan'.
        
        Args:
            text: Input text
            
        Returns:
            Text with '&' replaced
        """
        if '&' not in text:
            return text
        
        if self.ampersand_replacement == 'la':
            return text.replace('&', 'la')
        elif self.ampersand_replacement == 'dan':
            return text.replace('&', 'dan')
        else:  # random
            # Replace each '&' occurrence one by one with random choice
            result = []
            for char in text:
                if char == '&':
                    result.append(random.choice(['la', 'dan']))
                else:
                    result.append(char)
            return ''.join(result)
    
    def _remove_punctuation(self, text: str) -> str:
        """
        Remove all punctuation from text.
        
        Args:
            text: Input text
            
        Returns:
            Text without punctuation
        """
        return ''.join(char if char not in self.punctuation else ' ' for char in text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace (collapse multiple spaces to single space, strip leading/trailing).
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        return text.strip()
    
    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.
        
        Pipeline:
        1. Replace '&' with 'la' or 'dan'
        2. Lowercase
        3. Remove punctuation
        4. Normalize whitespace
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Step 1: Replace ampersand
        text = self._replace_ampersand(text)
        
        # Step 2: Lowercase
        text = text.lower()
        
        # Step 3: Remove punctuation
        text = self._remove_punctuation(text)
        
        # Step 4: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]
    
    def __call__(self, text: str) -> str:
        """
        Make the preprocessor callable.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        return self.preprocess(text)


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing TextPreprocessor...\n")
    
    # Test samples from the dataset
    test_samples = [
        "Aku libur sedina merga ana banjir",
        "Aku, adhiku & ibu mangan sega goreng!",
        "Kelereng gede & cilik ana neng omah.",
        "\"Sugeng enjing\", kandhaku.",
        "Pemerintah tetep komitmen arep terus mbukak informasi & nglakoni langkah-langkah pencegahan",
        "Eri ngarep warga gelem kerja bareng & ndhukung pembangunan",
    ]
    
    print("=" * 80)
    print("TEST 1: Random replacement (different each time)")
    print("=" * 80)
    preprocessor_random = TextPreprocessor(seed=None, ampersand_replacement='random')
    
    for sample in test_samples:
        processed = preprocessor_random.preprocess(sample)
        print(f"Original  : {sample}")
        print(f"Processed : {processed}")
        print()
    
    print("=" * 80)
    print("TEST 2: Fixed replacement with 'la'")
    print("=" * 80)
    preprocessor_la = TextPreprocessor(ampersand_replacement='la')
    
    for sample in test_samples:
        processed = preprocessor_la.preprocess(sample)
        print(f"Original  : {sample}")
        print(f"Processed : {processed}")
        print()
    
    print("=" * 80)
    print("TEST 3: Fixed replacement with 'dan'")
    print("=" * 80)
    preprocessor_dan = TextPreprocessor(ampersand_replacement='dan')
    
    for sample in test_samples:
        processed = preprocessor_dan.preprocess(sample)
        print(f"Original  : {sample}")
        print(f"Processed : {processed}")
        print()
    
    print("=" * 80)
    print("TEST 4: Reproducible random replacement (with seed)")
    print("=" * 80)
    
    # Process twice with same seed - should get identical results
    preprocessor_seed1 = TextPreprocessor(seed=42, ampersand_replacement='random')
    preprocessor_seed2 = TextPreprocessor(seed=42, ampersand_replacement='random')
    
    sample = "Warga & pemerintah & tokoh masyarakat"
    result1 = preprocessor_seed1.preprocess(sample)
    result2 = preprocessor_seed2.preprocess(sample)
    
    print(f"Original        : {sample}")
    print(f"Processed (1st) : {result1}")
    print(f"Processed (2nd) : {result2}")
    print(f"Results match   : {result1 == result2}")
    
    print("\n✓ TextPreprocessor test completed!")
