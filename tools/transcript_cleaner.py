"""
Transcript Cleaner Tool for Javanese ASR

Cleans transcripts from CSV file:
- Lowercase all text
- Remove punctuation
- Replace '&' with 'la' or 'dan' (randomized)
- Normalize whitespace

Usage:
    python transcript_cleaner.py <input_csv_file> [output_csv_file]
    
If output file is not specified, creates <input>_cleaned.csv

Example:
    python transcript_cleaner.py data/transcripts.csv
    python transcript_cleaner.py data/transcripts.csv data/transcripts_clean.csv
"""

import sys
import csv
import random
import string
from pathlib import Path


class TranscriptCleaner:
    """Clean and normalize Javanese transcripts."""
    
    def __init__(self, seed=42, ampersand_mode='random'):
        """
        Initialize cleaner.
        
        Args:
            seed: Random seed for '&' replacement
            ampersand_mode: 'random', 'la', or 'dan'
        """
        self.seed = seed
        self.ampersand_mode = ampersand_mode
        
        if seed is not None:
            random.seed(seed)
        
        # Punctuation to remove
        self.punctuation = set(string.punctuation)
        self.punctuation.update(['"', '"', ''', ''', '–', '—', '…', '’', '“', '”', "T"])
    
    def clean(self, text):
        """
        Clean a single transcript.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Step 1: Replace '&' with 'la' or 'dan'
        if '&' in text:
            if self.ampersand_mode == 'la':
                text = text.replace('&', 'la')
            elif self.ampersand_mode == 'dan':
                text = text.replace('&', 'dan')
            else:  # random
                result = []
                for char in text:
                    if char == '&':
                        result.append(random.choice(['la', 'dan']))
                    else:
                        result.append(char)
                text = ''.join(result)
        
        # Step 2: Lowercase
        text = text.lower()
        
        # Step 3: Remove punctuation (replace with space to keep word boundaries)
        text = ''.join(char if char not in self.punctuation else ' ' for char in text)
        
        # Step 4: Normalize whitespace
        text = ' '.join(text.split())
        
        return text


def process_csv(input_file, output_file, seed=42, ampersand_mode='random'):
    """
    Process CSV file and clean transcripts.
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output CSV
        seed: Random seed
        ampersand_mode: How to replace '&'
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    cleaner = TranscriptCleaner(seed=seed, ampersand_mode=ampersand_mode)
    
    print("=" * 70)
    print("TRANSCRIPT CLEANER")
    print("=" * 70)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Seed:        {seed}")
    print(f"& mode:      {ampersand_mode}")
    print("-" * 70)
    
    cleaned_count = 0
    total_count = 0
    unchanged_count = 0
    
    # Read and process
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        rows.append(header)
        
        for row in reader:
            total_count += 1
            
            if len(row) >= 2:
                sentence_id = row[0]
                original_transcript = row[1]
                device = row[2] if len(row) > 2 else ""
                
                # Clean transcript
                cleaned_transcript = cleaner.clean(original_transcript)
                
                # Track changes
                if cleaned_transcript != original_transcript:
                    cleaned_count += 1
                    if cleaned_count <= 5:  # Show first 5 examples
                        print(f"\nExample {cleaned_count}:")
                        print(f"  Original: {original_transcript}")
                        print(f"  Cleaned:  {cleaned_transcript}")
                else:
                    unchanged_count += 1
                
                # Add to output
                rows.append([sentence_id, cleaned_transcript, device])
            else:
                # Keep malformed rows as-is
                rows.append(row)
    
    # Write cleaned CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)
    print(f"Total transcripts:  {total_count}")
    print(f"Cleaned:            {cleaned_count}")
    print(f"Unchanged:          {unchanged_count}")
    print(f"Output saved to:    {output_file}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Determine output file
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Create default output filename
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
    
    # Optional: custom seed and mode from args
    seed = 42
    ampersand_mode = 'random'
    
    if len(sys.argv) >= 4:
        try:
            seed = int(sys.argv[3])
        except ValueError:
            print("Warning: Invalid seed, using default (42)")
    
    if len(sys.argv) >= 5:
        if sys.argv[4] in ['random', 'la', 'dan']:
            ampersand_mode = sys.argv[4]
        else:
            print(f"Warning: Invalid mode '{sys.argv[4]}', using 'random'")
    
    process_csv(input_file, output_file, seed=seed, ampersand_mode=ampersand_mode)


if __name__ == "__main__":
    main()
