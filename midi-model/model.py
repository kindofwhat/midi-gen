"""
MusicGPT model for MIDI music generation.

GPT-style transformer with position-in-event embedding for structural awareness.
"""

import torch
import torch.nn as nn


class MusicGPT(nn.Module):
    """
    GPT-style model for music generation with structural awareness.

    Key feature: Position-in-event embedding tells the model where it is
    within the 8-token event structure (positions 0-7).
    """

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 max_seq_len=1024, dropout=0.1, tokens_per_event=8):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tokens_per_event = tokens_per_event

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Position within event (0-7) embedding
        # This tells the model "you're at position 3 of an event"
        self.event_pos_emb = nn.Embedding(tokens_per_event, d_model)

        self.dropout = nn.Dropout(dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Sequence positions (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Position within event (0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, ...)
        event_positions = positions % self.tokens_per_event

        # Combine all embeddings
        x = self.token_emb(x) + self.pos_emb(positions) + self.event_pos_emb(event_positions)
        x = self.dropout(x)

        # Causal mask for autoregressive generation
        mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer (using decoder with memory=x for self-attention only)
        x = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)

        # Project to vocabulary
        logits = self.lm_head(x)

        return logits
