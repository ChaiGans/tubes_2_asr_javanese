"""
Inference Decoders: Greedy and Beam Search for ASR
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


class BeamSearchDecoder:
    """
    Beam search decoder for seq2seq ASR model.
    
    Args:
        model: Seq2SeqASR model
        vocab: Vocabulary object
        beam_size: Beam size (5-10 typical)
        max_len: Maximum decoding length
        device: Device to run on
        length_penalty: Length penalty alpha (0.0 = no penalty, 1.0 = full penalty)
    """
    def __init__(
        self, 
        model, 
        vocab, 
        beam_size: int = 5,
        max_len: int = 200,
        device: str = 'cpu',
        length_penalty: float = 0.6
    ):
        self.model = model
        self.vocab = vocab
        self.beam_size = beam_size
        self.max_len = max_len
        self.device = device
        self.length_penalty = length_penalty
    
    @torch.no_grad()
    def decode_batch(self, encoder_outputs: torch.Tensor, encoder_lengths: torch.Tensor) -> Tuple[List[List[int]], List[str]]:
        """
        Beam search decode a batch of utterances given encoder outputs.
        """
        self.model.eval()
        
        batch_size = encoder_outputs.size(0)
        transcripts = []
        all_indices = []
        
        # Process each utterance separately (batch=1 beam search)
        for i in range(batch_size):
            enc_out = encoder_outputs[i:i+1]
            enc_len = encoder_lengths[i:i+1]
            
            indices, transcript = self._beam_search_single(enc_out, enc_len)
            all_indices.append(indices)
            transcripts.append(transcript)
        
        return all_indices, transcripts

    @torch.no_grad()
    def decode(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> List[str]:
        """
        Beam search decode a batch of utterances.
        """
        self.model.eval()
        features = features.to(self.device)
        feature_lengths = feature_lengths.to(self.device)
        
        # Encode
        encoder_outputs, encoder_lengths = self.model.encoder(features, feature_lengths)
        
        _, transcripts = self.decode_batch(encoder_outputs, encoder_lengths)
        return transcripts
    
    def _beam_search_single(self, encoder_outputs: torch.Tensor, encoder_lengths: torch.Tensor) -> Tuple[List[int], str]:
        """
        Beam search for a single utterance.
        """
        encoder_dim = encoder_outputs.size(2)
        enc_time = encoder_outputs.size(1)
        
        # Initialize decoder state based on RNN type
        if self.model.decoder.rnn_type == "gru":
            initial_state = torch.zeros(1, 1, self.model.decoder.decoder_dim, device=self.device)
        else:
            h0 = torch.zeros(1, 1, self.model.decoder.decoder_dim, device=self.device)
            c0 = torch.zeros(1, 1, self.model.decoder.decoder_dim, device=self.device)
            initial_state = (h0, c0)
            
        initial_context = torch.zeros(1, encoder_dim, device=self.device)
        initial_attn = torch.ones(1, enc_time, device=self.device) / enc_time
        
        mask = self._create_mask(enc_time, encoder_lengths)
        
        beams = [
            (0.0, [self.vocab.sos_idx], initial_state, initial_context, initial_attn, False)
        ]
        
        completed = []
        
        # Beam search loop
        for step in range(self.max_len):
            candidates = []
            
            for score, tokens, decoder_state, context, attn, is_finished in beams:
                if is_finished:
                    completed.append((score, tokens))
                    continue
                
                # Get last token
                prev_token = torch.tensor([tokens[-1]], dtype=torch.long, device=self.device)
                
                # Decode one step - using correct argument names
                logits, new_context, new_state, new_attn = self.model.decoder.forward_step(
                    prev_token=prev_token,
                    prev_context=context,
                    decoder_state=decoder_state,
                    encoder_outputs=encoder_outputs,
                    prev_attn=attn,
                    mask=mask
                )
                
                # Get log probabilities
                log_probs = F.log_softmax(logits, dim=-1)  # [1, vocab_size]
                
                # Get top-k tokens
                topk_log_probs, topk_indices = log_probs.topk(self.beam_size, dim=-1)
                topk_log_probs = topk_log_probs.squeeze(0)
                topk_indices = topk_indices.squeeze(0)
                
                # Expand beams
                for k in range(self.beam_size):
                    token_id = topk_indices[k].item()
                    token_log_prob = topk_log_probs[k].item()
                    
                    new_score = score + token_log_prob
                    new_tokens = tokens + [token_id]
                    
                    # Check if finished
                    if token_id == self.vocab.eos_idx or token_id == self.vocab.pad_idx:
                        # Apply length penalty
                        length_norm = len(new_tokens) ** self.length_penalty
                        normalized_score = new_score / length_norm
                        completed.append((normalized_score, new_tokens))
                    else:
                        candidates.append((
                            new_score, new_tokens, new_state, new_context, new_attn, False
                        ))
            
            # Keep top beam_size candidates
            if len(candidates) == 0:
                break
            
            # Sort by score (descending)
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:self.beam_size]
        
        # Add remaining beams to completed
        for score, tokens, _, _, _, _ in beams:
            length_norm = len(tokens) ** self.length_penalty
            normalized_score = score / length_norm
            completed.append((normalized_score, tokens))
        
        # Select best
        if len(completed) == 0:
            return [], ""
        
        completed.sort(key=lambda x: x[0], reverse=True)
        best_tokens = completed[0][1]
        
        # Decode to text (remove <sos> and <eos>)
        best_tokens_clean = [t for t in best_tokens if t != self.vocab.sos_idx and t != self.vocab.eos_idx]
        transcript = self.vocab.decode(best_tokens_clean, remove_special=True)
        
        return best_tokens_clean, transcript
    
    def _create_mask(self, max_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """Create mask from lengths."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < lengths.unsqueeze(1)).long()
        return mask


if __name__ == "__main__":
    print("Decoder classes created successfully!")
