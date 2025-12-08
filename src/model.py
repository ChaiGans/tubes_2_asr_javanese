"""
LAS-style Seq2Seq ASR Model for Low-Resource Javanese
Contains: Pyramidal/Standard BiLSTM Encoder, Location-Sensitive Attention, LSTM/GRU Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PyramidalBiLSTMEncoder(nn.Module):
    """
    Pyramidal BiLSTM Encoder that reduces time resolution via concatenation.
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
        
        self.lstm_layers = nn.ModuleList()
        
        # First layer
        self.lstm_layers.append(
            nn.LSTM(input_dim, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        )
        
        current_input_dim = hidden_size * 2
        for i in range(1, num_layers):
            if i <= pyramid_levels:
                lstm_input_dim = current_input_dim * 2
            else:
                lstm_input_dim = current_input_dim
                
            self.lstm_layers.append(
                nn.LSTM(lstm_input_dim, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
            )
            current_input_dim = hidden_size * 2
        
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_size * 2
        
    def _apply_pyramidal_reduction(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, time, dim = x.size()
        if time % 2 != 0:
            pad = x[:, -1:, :]
            x = torch.cat([x, pad], dim=1)
            time += 1
        x = x.reshape(batch_size, time // 2, dim * 2)
        lengths = (lengths + 1) // 2
        return x, lengths
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, lstm in enumerate(self.lstm_layers):
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            x_packed, _ = lstm(x_packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
            
            if i < len(self.lstm_layers) - 1:
                x = self.dropout(x)
            
            if i < self.pyramid_levels:
                x, lengths = self._apply_pyramidal_reduction(x, lengths)
        
        return x, lengths


class StandardBiLSTMEncoder(nn.Module):
    """
    Standard BiLSTM Encoder without time reduction.
    """
    def __init__(
        self, 
        input_dim: int = 80, 
        hidden_size: int = 128, 
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_size, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_dim = hidden_size * 2
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x_packed, _ = self.lstm(x_packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        return x, lengths


class LocationSensitiveAttention(nn.Module):
    """Location-Sensitive Attention Mechanism"""
    def __init__(self, encoder_dim, decoder_dim, attention_dim=128, num_filters=10, kernel_size=5):
        super().__init__()
        self.location_conv = nn.Conv1d(1, num_filters, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.W_encoder = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_location = nn.Linear(num_filters, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, decoder_state, encoder_outputs, prev_attention_weights, encoder_mask=None):
        # decoder_state: [batch, decoder_dim]
        # encoder_outputs: [batch, time, encoder_dim]
        
        location_features = self.location_conv(prev_attention_weights.unsqueeze(1)).transpose(1, 2)
        energy = self.v(torch.tanh(
            self.W_encoder(encoder_outputs) + 
            self.W_decoder(decoder_state).unsqueeze(1) + 
            self.W_location(location_features)
        )).squeeze(-1)
        
        if encoder_mask is not None:
            energy = energy.masked_fill(encoder_mask == 0, -1e9)
            
        attention_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class DecoderWithAttention(nn.Module):
    """Decoder with support for LSTM or GRU"""
    def __init__(
        self, vocab_size, embedding_dim=64, encoder_dim=256, decoder_dim=256, 
        attention_dim=128, dropout=0.3, rnn_type="lstm"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.rnn_type = rnn_type.lower()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        rnn_input_dim = embedding_dim + encoder_dim
        if self.rnn_type == "gru":
            self.rnn = nn.GRU(rnn_input_dim, decoder_dim, batch_first=True)
        else:
            self.rnn = nn.LSTM(rnn_input_dim, decoder_dim, batch_first=True)
            
        self.attention = LocationSensitiveAttention(encoder_dim, decoder_dim, attention_dim)
        
        self.output_projection = nn.Sequential(
            nn.Linear(decoder_dim + encoder_dim, decoder_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, vocab_size)
        )
        
    def forward_step(self, prev_token, prev_context, decoder_state, encoder_outputs, prev_attn, mask):
        embedded = self.embedding(prev_token)
        rnn_input = torch.cat([embedded, prev_context], dim=-1).unsqueeze(1)
        
        if self.rnn_type == "gru":
            rnn_output, new_state = self.rnn(rnn_input, decoder_state)
            # GRU state is just hidden state
            hidden_state = rnn_output.squeeze(1)
        else:
            rnn_output, new_state = self.rnn(rnn_input, decoder_state)
            # LSTM state is (h, c), we use h for attention
            hidden_state = rnn_output.squeeze(1)
            
        context, attn_weights = self.attention(hidden_state, encoder_outputs, prev_attn, mask)
        
        logits = self.output_projection(torch.cat([hidden_state, context], dim=-1))
                
        return logits, context, new_state, attn_weights

    def forward(self, encoder_outputs, encoder_lengths, targets, teacher_forcing_ratio=1.0):
        batch_size, enc_time, _ = encoder_outputs.size()
        max_len = targets.size(1)
        
        mask = (torch.arange(enc_time, device=encoder_outputs.device).unsqueeze(0) < encoder_lengths.unsqueeze(1)).long()
        
        # Init state
        if self.rnn_type == "gru":
            state = torch.zeros(1, batch_size, self.decoder_dim, device=encoder_outputs.device)
        else:
            h = torch.zeros(1, batch_size, self.decoder_dim, device=encoder_outputs.device)
            c = torch.zeros(1, batch_size, self.decoder_dim, device=encoder_outputs.device)
            state = (h, c)
            
        prev_attn = torch.ones(batch_size, enc_time, device=encoder_outputs.device) / enc_time
        prev_context = torch.zeros(batch_size, encoder_outputs.size(2), device=encoder_outputs.device)
        prev_token = targets[:, 0]
        
        outputs = []
        for t in range(max_len - 1):
            logits, prev_context, state, prev_attn = self.forward_step(
                prev_token, prev_context, state, encoder_outputs, prev_attn, mask
            )
            outputs.append(logits.unsqueeze(1))
            if torch.rand(1).item() < teacher_forcing_ratio:
                prev_token = targets[:, t + 1]
            else:
                prev_token = logits.argmax(dim=-1)
                
        return torch.cat(outputs, dim=1)


class Seq2SeqASR(nn.Module):
    """
    Configurable Seq2Seq ASR Model
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
        ctc_weight: float = 0.3,
        encoder_type: str = "pyramidal", # "pyramidal" or "standard"
        decoder_type: str = "lstm",      # "lstm" or "gru"
        pyramid_levels: int = 2          # Number of pyramid reduction layers
    ):
        super().__init__()
        self.use_ctc = use_ctc
        self.ctc_weight = ctc_weight
        self.encoder_type = encoder_type
        
        # Encoder
        if encoder_type == "pyramidal":
            self.encoder = PyramidalBiLSTMEncoder(
                input_dim, encoder_hidden_size, encoder_num_layers, dropout, pyramid_levels
            )
        else:
            self.encoder = StandardBiLSTMEncoder(
                input_dim, encoder_hidden_size, encoder_num_layers, dropout
            )
            
        encoder_dim = self.encoder.output_dim
        
        # Decoder
        self.decoder = DecoderWithAttention(
            vocab_size, embedding_dim, encoder_dim, decoder_dim, 
            attention_dim, dropout, rnn_type=decoder_type
        )
        
        # CTC
        if use_ctc:
            self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        else:
            self.ctc_head = None
            
    def forward(self, features, feature_lengths, targets, teacher_forcing_ratio=1.0):
        encoder_outputs, encoder_lengths = self.encoder(features, feature_lengths)
        attention_logits = self.decoder(encoder_outputs, encoder_lengths, targets, teacher_forcing_ratio)
        
        ctc_logits = None
        if self.use_ctc:
            ctc_logits = F.log_softmax(self.ctc_head(encoder_outputs), dim=-1)
        
        # Return encoder_lengths for CTC loss computation
        return attention_logits, ctc_logits, encoder_lengths

    def compute_loss(self, attention_logits, targets, target_lengths, ctc_logits=None, encoder_lengths=None, pad_idx=0, blank_idx=4):
        # Attention Loss
        attn_loss = F.cross_entropy(
            attention_logits.reshape(-1, attention_logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=pad_idx
        )
        
        # CTC Loss
        if self.use_ctc and ctc_logits is not None:
            batch_size = targets.size(0)
            ctc_targets = []
            ctc_target_lens = []
            for i in range(batch_size):
                seq = targets[i, 1:target_lengths[i]-1]
                ctc_targets.append(seq)
                ctc_target_lens.append(len(seq))
            
            ctc_targets = torch.cat(ctc_targets)
            ctc_target_lens = torch.tensor(ctc_target_lens, device=targets.device)
            
            ctc_loss = F.ctc_loss(
                ctc_logits.permute(1, 0, 2),
                ctc_targets,
                encoder_lengths,
                ctc_target_lens,
                blank=blank_idx,
                zero_infinity=True
            )
            
            return (1 - self.ctc_weight) * attn_loss + self.ctc_weight * ctc_loss
            
        return attn_loss
