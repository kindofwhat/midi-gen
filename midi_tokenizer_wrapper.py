"""
Wrapper for SkyTNT's MIDI tokenizer.

This module provides a clean interface to use the proven tokenization
from midi-model with our own training pipeline.

The tokenizer converts MIDI to event sequences like:
  [NOTE, time1, time2, track, duration, channel, pitch, velocity]

This is much better than piano roll because:
1. Sparse representation (only note events, not empty frames)
2. Explicit structure (time, duration, pitch are separate tokens)
3. Proven to work for music generation
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

# Add midi-model to path
MIDI_MODEL_DIR = Path(__file__).parent / "midi-model"
sys.path.insert(0, str(MIDI_MODEL_DIR))

from midi_tokenizer import MIDITokenizerV2
from MIDI import midi2score, score2midi


class MIDITokenizerWrapper:
    """Wrapper around SkyTNT's tokenizer for easier use."""

    def __init__(self, version: str = "v2", optimise_midi: bool = True):
        """
        Initialize the tokenizer.

        Args:
            version: Tokenizer version ("v1" or "v2")
            optimise_midi: Whether to optimize/normalize MIDI data
        """
        self.tokenizer = MIDITokenizerV2() if version == "v2" else None
        if self.tokenizer:
            self.tokenizer.set_optimise_midi(optimise_midi)

        # Key properties for our model
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.max_token_seq = self.tokenizer.max_token_seq  # tokens per event

    def midi_file_to_tokens(self, midi_path: Path) -> Optional[np.ndarray]:
        """
        Convert a MIDI file to token sequence.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Array of shape (num_events, max_token_seq) or None if failed
        """
        try:
            with open(midi_path, 'rb') as f:
                midi_bytes = f.read()

            # MIDI bytes → score
            score = midi2score(midi_bytes)

            # Score → tokens
            tokens = self.tokenizer.tokenize(score, add_bos_eos=True)

            return np.array(tokens)
        except Exception as e:
            print(f"Error tokenizing {midi_path}: {e}")
            return None

    def tokens_to_midi_file(self, tokens: np.ndarray, output_path: Path) -> bool:
        """
        Convert token sequence back to MIDI file.

        Args:
            tokens: Array of shape (num_events, max_token_seq)
            output_path: Where to save MIDI

        Returns:
            True if successful
        """
        try:
            # Tokens → score
            score = self.tokenizer.detokenize(tokens)

            # Score → MIDI bytes
            midi_bytes = score2midi(score)

            with open(output_path, 'wb') as f:
                f.write(midi_bytes)

            return True
        except Exception as e:
            print(f"Error detokenizing to {output_path}: {e}")
            return False

    def flatten_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        Flatten 2D token array to 1D for simpler model input.

        The tokenizer outputs (num_events, max_token_seq) where each event
        has multiple tokens. We flatten to (num_events * max_token_seq,)
        for a standard sequence model.

        Padding tokens are preserved.
        """
        return tokens.flatten()

    def unflatten_tokens(self, flat_tokens: np.ndarray) -> np.ndarray:
        """Reshape flat tokens back to (num_events, max_token_seq)."""
        return flat_tokens.reshape(-1, self.max_token_seq)

    def get_event_info(self) -> dict:
        """Get information about event types and parameters."""
        return {
            "events": self.tokenizer.events,
            "event_parameters": self.tokenizer.event_parameters,
            "vocab_size": self.vocab_size,
            "max_token_seq": self.max_token_seq,
        }
