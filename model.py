"""
LAS-style Seq2Seq ASR Model for Low-Resource Javanese
Contains: Pyramidal BiLSTM Encoder, Location-Sensitive Attention, LSTM Decoder with Input Feeding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PyramidalBiLSTMEncoder(nn.Module):
    """
    Pyramidal BiLSTM Encoder that reduces time resolution via concatenation of adjacent frames.
    
    Args:
        input_dim: Dimension of input features (e.g., 80 for log-mel)
        hidden_size: Hidden size per direction in each BiLSTM layer
        num_layers: Number of BiLSTM layers (2-3 recommended for small datasets)
        dropout: Dropout probability
        pyramid_levels: Number of pyramidal reductions (1 or 2)
    """
    def __init__(
        self, 
        input_dim: int = 80, 
        hidden_size: int = 128, 
        num_layers: int = 3,
        dropout: float = 0.3,
        pyramid_levels: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pyramid_levels = pyramid_levels
        
        # First LSTM layer
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(
            nn.LSTM(
                input_dim, 
                hidden_size, 
                num_layers=1, 
                bidirectional=True, 
                batch_first=True,
                dropout=0
            )
        )
        
        # Subsequent layers with pyramidal reduction
        current_input_dim = hidden_size * 2  # BiLSTM output
        for i in range(1, num_layers):
            # After pyramidal reduction, input dim is doubled (concatenation)
            if i <= pyramid_levels:
                lstm_input_dim = current_input_dim * 2
            else:
                lstm_input_dim = current_input_dim
                
            self.lstm_layers.append(
                nn.LSTM(
                    lstm_input_dim,
                    hidden_size,
                    num_layers=1,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0
                )
            )
            current_input_dim = hidden_size * 2
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * 2  # BiLSTM output dimension
        
    def _apply_pyramidal_reduction(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reduce time resolution by concatenating adjacent frames.
        
        Args:
            x: [batch, time, dim]
            lengths: [batch]
        Returns:
            reduced_x: [batch, time//2, dim*2]
            reduced_lengths: [batch]
        """
        batch_size, time, dim = x.size()
        
        # If odd length, pad the last frame
        if time % 2 != 0:
            pad = x[:, -1:, :]  # Repeat last frame
            x = torch.cat([x, pad], dim=1)
            time += 1
        
        # Reshape and concatenate: [batch, time//2, dim*2]
        x = x.reshape(batch_size, time // 2, dim * 2)
        
        # Update lengths (round up for odd lengths)
        lengths = (lengths + 1) // 2
        
        return x, lengths
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through pyramidal encoder.
        
        Args:
            x: Input features [batch, time, input_dim]
            lengths: Original lengths [batch]
        Returns:
            encoder_outputs: [batch, reduced_time, hidden_size*2]
            output_lengths: [batch]
        """
        for i, lstm in enumerate(self.lstm_layers):
            # Pack sequence for efficiency
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # LSTM forward
            x_packed, _ = lstm(x_packed)
            
            # Unpack
            x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
            
            # Apply dropout (not after last layer)
            if i < len(self.lstm_layers) - 1:
                x = self.dropout(x)
            
            # Apply pyramidal reduction after first N layers
            if i < self.pyramid_levels:
                x, lengths = self._apply_pyramidal_reduction(x, lengths)
        
        return x, lengths


class LocationSensitiveAttention(nn.Module):
    """
    Location-Sensitive Additive Attention (Chorowski et al.)
    
    Combines:
    - Previous attention weights (via 1D convolution)
    - Decoder state
    - Encoder hidden states
    
    Args:
        encoder_dim: Dimension of encoder outputs
        decoder_dim: Dimension of decoder hidden state
        attention_dim: Dimension of attention hidden layer
        num_filters: Number of filters in location convolution
        kernel_size: Kernel size for location convolution
    """
    def __init__(
        self,
        encoder_dim: int = 256,
        decoder_dim: int = 256,
        attention_dim: int = 128,
        num_filters: int = 10,
        kernel_size: int = 5
    ):
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        
        # Location-based convolution
        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        
        # Projection layers
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_location = nn.Linear(num_filters, attention_dim, bias=False)
        
        # Energy scalar
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(
        self, 
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        prev_attention_weights: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention context and weights.
        
        Args:
            decoder_state: [batch, decoder_dim]
            encoder_outputs: [batch, enc_time, encoder_dim]
            prev_attention_weights: [batch, enc_time]
            encoder_mask: [batch, enc_time] (1 for valid positions, 0 for padding)
        
        Returns:
            context: [batch, encoder_dim]
            attention_weights: [batch, enc_time]
        """
        batch_size, enc_time, _ = encoder_outputs.size()
        
        # Process location features: [batch, 1, enc_time] -> [batch, num_filters, enc_time]
        location_features = self.location_conv(prev_attention_weights.unsqueeze(1))
        # -> [batch, enc_time, num_filters]
        location_features = location_features.transpose(1, 2)
        
        # Project encoder outputs: [batch, enc_time, attention_dim]
        encoder_proj = self.W_encoder(encoder_outputs)
        
        # Project decoder state: [batch, attention_dim] -> [batch, 1, attention_dim]
        decoder_proj = self.W_decoder(decoder_state).unsqueeze(1)
        
        # Project location features: [batch, enc_time, attention_dim]
        location_proj = self.W_location(location_features)
        
        # Compute energy: e_j = v^T tanh(W_h h_j + W_s s + W_f f_j)
        # [batch, enc_time, attention_dim]
        energy_input = encoder_proj + decoder_proj + location_proj
        energy = self.v(torch.tanh(energy_input)).squeeze(-1)  # [batch, enc_time]
        
        # Apply mask if provided (set padding positions to -inf)
        if encoder_mask is not None:
            energy = energy.masked_fill(encoder_mask == 0, -1e9)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(energy, dim=1)  # [batch, enc_time]
        
        # Compute context vector: c = sum(alpha_j * h_j)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # [batch, encoder_dim]
        
        return context, attention_weights


class DecoderWithAttention(nn.Module):
    """
    LSTM Decoder with Input Feeding and Location-Sensitive Attention.
    
    At each step:
    1. Input = [embedding(y_{t-1}); context_{t-1}]
    2. LSTM forward to get new state s_t
    3. Attention to compute context_t
    4. Combine [s_t; context_t] for output projection
    
    Args:
        vocab_size: Size of output vocabulary
        embedding_dim: Dimension of character embeddings
        encoder_dim: Dimension of encoder outputs
        decoder_dim: Dimension of decoder LSTM hidden state
        attention_dim: Dimension of attention mechanism
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        encoder_dim: int = 256,
        decoder_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM with input feeding: input = [embedding; prev_context]
        self.lstm = nn.LSTM(
            embedding_dim + encoder_dim,
            decoder_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = LocationSensitiveAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )
        
        # Output projection: [decoder_state; context] -> vocab
        self.output_projection = nn.Sequential(
            nn.Linear(decoder_dim + encoder_dim, decoder_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, vocab_size)
        )
        
    def forward_step(
        self,
        prev_token: torch.Tensor,
        prev_context: torch.Tensor,
        decoder_state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        prev_attention_weights: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            prev_token: [batch] - previous token indices
            prev_context: [batch, encoder_dim] - previous context vector
            decoder_state: (h, c) - LSTM hidden and cell states
            encoder_outputs: [batch, enc_time, encoder_dim]
            prev_attention_weights: [batch, enc_time]
            encoder_mask: [batch, enc_time]
        
        Returns:
            logits: [batch, vocab_size]
            context: [batch, encoder_dim]
            new_decoder_state: (h, c)
            attention_weights: [batch, enc_time]
        """
        # Embed previous token: [batch, embedding_dim]
        embedded = self.embedding(prev_token)
        
        # Input feeding: concatenate embedding with previous context
        lstm_input = torch.cat([embedded, prev_context], dim=-1).unsqueeze(1)
        # [batch, 1, embedding_dim + encoder_dim]
        
        # LSTM forward
        lstm_output, new_decoder_state = self.lstm(lstm_input, decoder_state)
        lstm_output = lstm_output.squeeze(1)  # [batch, decoder_dim]
        
        # Attention: use the new LSTM hidden state
        context, attention_weights = self.attention(
            decoder_state=lstm_output,
            encoder_outputs=encoder_outputs,
            prev_attention_weights=prev_attention_weights,
            encoder_mask=encoder_mask
        )
        
        # Output projection: [lstm_output; context] -> logits
        combined = torch.cat([lstm_output, context], dim=-1)
        logits = self.output_projection(combined)  # [batch, vocab_size]
        
        return logits, context, new_decoder_state, attention_weights
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        
        Args:
            encoder_outputs: [batch, enc_time, encoder_dim]
            encoder_lengths: [batch]
            targets: [batch, max_target_len] - target token indices (includes <sos>, <eos>)
            teacher_forcing_ratio: Probability of using ground truth (1.0 = always)
        
        Returns:
            logits: [batch, max_target_len-1, vocab_size]
        """
        batch_size, enc_time, encoder_dim = encoder_outputs.size()
        max_target_len = targets.size(1)
        
        # Create encoder mask from lengths
        encoder_mask = self._create_mask(enc_time, encoder_lengths).to(encoder_outputs.device)
        
        # Initialize decoder state
        h0 = torch.zeros(1, batch_size, self.decoder_dim, device=encoder_outputs.device)
        c0 = torch.zeros(1, batch_size, self.decoder_dim, device=encoder_outputs.device)
        decoder_state = (h0, c0)
        
        # Initialize attention (uniform distribution)
        prev_attention_weights = torch.ones(batch_size, enc_time, device=encoder_outputs.device) / enc_time
        
        # Initialize context (zeros or mean of encoder outputs)
        prev_context = torch.zeros(batch_size, encoder_dim, device=encoder_outputs.device)
        
        # Storage for outputs
        outputs = []
        
        # First input is <sos> token (targets[:, 0])
        prev_token = targets[:, 0]
        
        # Decode for max_target_len - 1 steps (excluding <sos>)
        for t in range(max_target_len - 1):
            logits, prev_context, decoder_state, prev_attention_weights = self.forward_step(
                prev_token=prev_token,
                prev_context=prev_context,
                decoder_state=decoder_state,
                encoder_outputs=encoder_outputs,
                prev_attention_weights=prev_attention_weights,
                encoder_mask=encoder_mask
            )
            
            outputs.append(logits.unsqueeze(1))
            
            # Teacher forcing: use ground truth or predicted token
            if teacher_forcing_ratio >= 1.0 or torch.rand(1).item() < teacher_forcing_ratio:
                prev_token = targets[:, t + 1]  # Ground truth
            else:
                prev_token = logits.argmax(dim=-1)  # Predicted
        
        # Concatenate all outputs: [batch, max_target_len-1, vocab_size]
        outputs = torch.cat(outputs, dim=1)
        
        return outputs
    
    def _create_mask(self, max_len: int, lengths: torch.Tensor) -> torch.Tensor:
        """Create mask from lengths. Returns 1 for valid positions, 0 for padding."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < lengths.unsqueeze(1)).long()
        return mask


class Seq2SeqASR(nn.Module):
    """
    Main Seq2Seq ASR Model combining Encoder, Decoder, and optional CTC head.
    
    Args:
        vocab_size: Size of character vocabulary
        input_dim: Dimension of input features (e.g., 80 for log-mel)
        encoder_hidden_size: Hidden size per direction in encoder
        encoder_num_layers: Number of encoder layers
        decoder_dim: Dimension of decoder LSTM
        use_ctc: Whether to add CTC head for joint training
        ctc_weight: Weight for CTC loss (lambda), 0.3 recommended
    """
    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        encoder_hidden_size: int = 128,
        encoder_num_layers: int = 3,
        decoder_dim: int = 256,
        attention_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        use_ctc: bool = False,
        ctc_weight: float = 0.3
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.use_ctc = use_ctc
        self.ctc_weight = ctc_weight
        
        # Encoder
        self.encoder = PyramidalBiLSTMEncoder(
            input_dim=input_dim,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_num_layers,
            dropout=dropout,
            pyramid_levels=2
        )
        
        encoder_dim = self.encoder.output_dim  # 2 * encoder_hidden_size
        
        # Decoder with attention
        self.decoder = DecoderWithAttention(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        # Optional CTC head
        if use_ctc:
            self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        else:
            self.ctc_head = None
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Training forward pass.
        
        Args:
            features: [batch, time, input_dim]
            feature_lengths: [batch]
            targets: [batch, max_target_len] - includes <sos> and <eos>
            target_lengths: [batch]
        
        Returns:
            attention_logits: [batch, max_target_len-1, vocab_size]
            ctc_logits: [batch, enc_time, vocab_size] or None
        """
        # Encode
        encoder_outputs, encoder_lengths = self.encoder(features, feature_lengths)
        
        # Decode with attention
        attention_logits = self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            targets=targets,
            teacher_forcing_ratio=1.0
        )
        
        # CTC head (if enabled)
        ctc_logits = None
        if self.use_ctc:
            ctc_logits = self.ctc_head(encoder_outputs)  # [batch, enc_time, vocab_size]
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)
        
        return attention_logits, ctc_logits
    
    def compute_loss(
        self,
        attention_logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        ctc_logits: Optional[torch.Tensor] = None,
        encoder_lengths: Optional[torch.Tensor] = None,
        pad_idx: int = 0,
        blank_idx: int = 4
    ) -> torch.Tensor:
        """
        Compute joint CTC + attention loss or pure attention loss.
        
        Args:
            attention_logits: [batch, max_target_len-1, vocab_size]
            targets: [batch, max_target_len] - includes <sos> and <eos>
            target_lengths: [batch]
            ctc_logits: [batch, enc_time, vocab_size] (log-softmax applied)
            encoder_lengths: [batch]
            pad_idx: Padding token index
            blank_idx: CTC blank token index
        
        Returns:
            total_loss: scalar
        """
        batch_size, max_target_len_minus_1, vocab_size = attention_logits.size()
        
        # Attention loss (cross-entropy)
        # Target for attention: tokens[1:] (exclude <sos>, predict <eos>)
        attention_targets = targets[:, 1:]  # [batch, max_target_len-1]
        
        # Flatten for cross-entropy
        attention_logits_flat = attention_logits.reshape(-1, vocab_size)
        attention_targets_flat = attention_targets.reshape(-1)
        
        # Cross-entropy (ignores padding)
        attention_loss = F.cross_entropy(
            attention_logits_flat,
            attention_targets_flat,
            ignore_index=pad_idx
        )
        
        # CTC loss (if enabled)
        if self.use_ctc and ctc_logits is not None:
            # CTC expects: [time, batch, vocab]
            ctc_logits_permuted = ctc_logits.permute(1, 0, 2)
            
            # CTC targets: remove <sos> and <eos>, keep only character sequence
            ctc_targets = []
            ctc_target_lengths = []
            for i in range(batch_size):
                # Assuming targets format: [<sos>, char1, char2, ..., <eos>, <pad>, ...]
                seq = targets[i, 1:target_lengths[i]-1]  # Remove <sos> and <eos>
                ctc_targets.append(seq)
                ctc_target_lengths.append(len(seq))
            
            ctc_targets = torch.cat(ctc_targets)
            ctc_target_lengths = torch.tensor(ctc_target_lengths, dtype=torch.long, device=targets.device)
            
            # CTC loss
            ctc_loss = F.ctc_loss(
                ctc_logits_permuted,
                ctc_targets,
                encoder_lengths,
                ctc_target_lengths,
                blank=blank_idx,
                reduction='mean',
                zero_infinity=True
            )
            
            # Joint loss
            total_loss = (1 - self.ctc_weight) * attention_loss + self.ctc_weight * ctc_loss
        else:
            total_loss = attention_loss
        
        return total_loss


if __name__ == "__main__":
    # Quick test
    print("Testing Seq2SeqASR model...")
    
    vocab_size = 50
    batch_size = 4
    time_steps = 200
    input_dim = 80
    max_target_len = 30
    
    # Create model
    model = Seq2SeqASR(
        vocab_size=vocab_size,
        input_dim=input_dim,
        encoder_hidden_size=128,
        encoder_num_layers=3,
        decoder_dim=256,
        use_ctc=True,
        ctc_weight=0.3
    )
    
    # Dummy data
    features = torch.randn(batch_size, time_steps, input_dim)
    feature_lengths = torch.tensor([200, 180, 150, 120])
    targets = torch.randint(0, vocab_size, (batch_size, max_target_len))
    targets[:, 0] = 1  # <sos>
    target_lengths = torch.tensor([30, 25, 20, 15])
    
    # Forward pass
    print("Running forward pass...")
    attention_logits, ctc_logits = model(features, feature_lengths, targets, target_lengths)
    
    print(f"Attention logits shape: {attention_logits.shape}")  # [batch, max_target_len-1, vocab_size]
    if ctc_logits is not None:
        print(f"CTC logits shape: {ctc_logits.shape}")  # [batch, enc_time, vocab_size]
    
    # Compute loss
    loss = model.compute_loss(
        attention_logits, targets, target_lengths, 
        ctc_logits, feature_lengths // 4  # Encoder reduces time by ~4x
    )
    print(f"Loss: {loss.item():.4f}")
    
    print("Model test passed!")
