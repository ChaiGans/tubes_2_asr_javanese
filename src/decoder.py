"""
Inference Decoders: Greedy for ASR
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import heapq


class GreedyDecoder:
    """
    Greedy decoder for seq2seq ASR model.
    At each step, selects the token with maximum probability.
    
    Args:
        model: Seq2SeqASR model
        vocab: Vocabulary object
        max_len: Maximum decoding length
        device: Device to run on
    """
    def __init__(self, model, vocab, max_len: int = 200, device: str = 'cpu'):
        self.model = model
        self.vocab = vocab
        self.max_len = max_len
        self.device = device
    
    @torch.no_grad()
    def decode_batch(self, encoder_outputs: torch.Tensor, encoder_lengths: torch.Tensor) -> Tuple[List[List[int]], List[str]]:
        """
        Greedy decode a batch of utterances given encoder outputs.
        
        Args:
            encoder_outputs: [batch, time, dim]
            encoder_lengths: [batch]
            
        Returns:
            decoded_indices: List of list of token indices
            decoded_transcripts: List of decoded strings
        """
        self.model.eval()
        
        batch_size = encoder_outputs.size(0)
        encoder_dim = encoder_outputs.size(2)
        enc_time = encoder_outputs.size(1)
        
        # Initialize decoder state based on RNN type
        if self.model.decoder.rnn_type == "gru":
            decoder_state = torch.zeros(1, batch_size, self.model.decoder.decoder_dim, device=self.device)
        else:
            h0 = torch.zeros(1, batch_size, self.model.decoder.decoder_dim, device=self.device)
            c0 = torch.zeros(1, batch_size, self.model.decoder.decoder_dim, device=self.device)
            decoder_state = (h0, c0)
        
        # Initialize attention
        prev_attn = torch.ones(batch_size, enc_time, device=self.device) / enc_time
        
        # Initialize context
        prev_context = torch.zeros(batch_size, encoder_dim, device=self.device)
        
        # Create encoder mask
        mask = self._create_mask(enc_time, encoder_lengths)
        
        # Start with <sos> token
        prev_token = torch.full((batch_size,), self.vocab.sos_idx, dtype=torch.long, device=self.device)
        
        # Storage for decoded sequences
        decoded_indices = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        # Decode step by step
        for step in range(self.max_len):
            # One decoding step - using correct argument names from model.py
            logits, prev_context, decoder_state, prev_attn = self.model.decoder.forward_step(
                prev_token=prev_token,
                prev_context=prev_context,
                decoder_state=decoder_state,
                encoder_outputs=encoder_outputs,
                prev_attn=prev_attn,
                mask=mask
            )
            
            # Greedy selection: argmax
            predicted_tokens = logits.argmax(dim=-1)  # [batch]
            
            # Update sequences
            for i in range(batch_size):
                if not finished[i]:
                    token_id = predicted_tokens[i].item()
                    
                    # Stop if <eos> or <pad>
                    if token_id == self.vocab.eos_idx or token_id == self.vocab.pad_idx:
                        finished[i] = True
                    else:
                        decoded_indices[i].append(token_id)
            
            # Stop if all sequences finished
            if all(finished):
                break
            
            # Update prev_token
            prev_token = predicted_tokens
        
        # Convert to text
        transcripts = []
        for seq in decoded_indices:
            text = self.vocab.decode(seq, remove_special=True)
            transcripts.append(text)
        
        return decoded_indices, transcripts

    @torch.no_grad()
    def decode(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> List[str]:
        """
        Greedy decode a batch of utterances from features.
        """
        self.model.eval()
        features = features.to(self.device)
        feature_lengths = feature_lengths.to(self.device)
        
        # Encode
        encoder_outputs, encoder_lengths = self.model.encoder(features, feature_lengths)
        
        _, transcripts = self.decode_batch(encoder_outputs, encoder_lengths)
        return transcripts
    
    def _create_mask(self, max_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """Create mask from lengths."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < lengths.unsqueeze(1)).long()
        return mask

if __name__ == "__main__":
    print("Decoder classes created successfully!")
