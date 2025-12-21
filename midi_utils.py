"""
Shared utilities for MIDI music generation.

This module contains all shared code between training and generation notebooks:
- Configuration constants
- Model architectures (Mamba, LSTM)
- MIDI processing functions
- Audio playback utilities
- Generation functions
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Positive class weight for BCEWithLogitsLoss to handle data imbalance
# Piano rolls are ~95% zeros, so we weight positive (note-on) samples more heavily
# Too low (1) = predicts silence, too high (20) = predicts noise
POS_WEIGHT = 5.0  # Balanced weight for positive class

MIDI_CONFIG = {
    "min_pitch": 21,       # A0 (lowest piano key)
    "max_pitch": 108,      # C8 (highest piano key)
    "fs": 16,              # Frames per second (temporal resolution)
    "velocity_bins": 32,   # Velocity quantization bins
    "max_duration": 60,    # Max clip duration in seconds
}

# Tick-based encoding config
TICKS_PER_BEAT = 4        # 16th note resolution (4 ticks = 1 beat)
MAX_DURATION_TICKS = 32   # Max note duration in ticks (2 bars)
DEFAULT_TEMPO = 120       # BPM for conversion

NUM_PITCHES = MIDI_CONFIG["max_pitch"] - MIDI_CONFIG["min_pitch"] + 1  # 88 piano keys
TICK_FEATURES = NUM_PITCHES * 2  # 88 pitches × 2 (velocity + duration) = 176
SEQUENCE_LENGTH = 64   # Input sequence length (ticks)
STRIDE = 16            # Sliding window stride
SEED = 42


def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("Using Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class MambaBlock(nn.Module):
    """
    Single Mamba block with selective state space.

    Key innovation: Input-dependent state transitions (selective SSM)
    Complexity: O(N) vs O(N²) for Transformers
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = torch.nn.functional.silu(x)

        y = self._selective_ssm(x)
        y = y * torch.nn.functional.silu(z)
        return self.out_proj(y)

    def _selective_ssm(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_inner = x.shape

        x_proj = self.x_proj(x)
        dt = x_proj[:, :, :1]
        B = x_proj[:, :, 1:self.d_state+1]
        C = x_proj[:, :, self.d_state+1:]

        dt = self.dt_proj(dt)
        dt = torch.nn.functional.softplus(dt)

        A = -torch.exp(self.A_log)
        dA = torch.einsum('bld,dn->bldn', dt, A)
        dA = torch.exp(dA)
        dB = torch.einsum('bld,bln->bldn', dt, B)

        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for i in range(seq_len):
            h = dA[:, i] * h + dB[:, i] * x[:, i:i+1, :].transpose(1, 2)
            y = torch.einsum('bdn,bn->bd', h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + x * self.D
        return y


class MusicMamba(nn.Module):
    """
    Mamba-based model for music generation.

    NOTE: Outputs LOGITS (no sigmoid) for BCEWithLogitsLoss compatibility.
    Apply sigmoid during inference.
    """

    def __init__(self, input_size: int = NUM_PITCHES,
                 d_model: int = 256,
                 d_state: int = 16,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': MambaBlock(d_model, d_state=d_state),
                'norm': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            })
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_size)
        )

    def forward(self, x: torch.Tensor,
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        x = self.input_proj(x)
        x = self.input_norm(x)

        for layer in self.layers:
            residual = x
            x = layer['mamba'](x)
            x = layer['dropout'](x)
            x = layer['norm'](x + residual)

        x = self.output_norm(x)
        return self.output_proj(x), None

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        return None


class MusicLSTM(nn.Module):
    """
    LSTM-based model for polyphonic music generation.

    NOTE: Outputs LOGITS (no sigmoid) for BCEWithLogitsLoss compatibility.
    """

    def __init__(self, input_size: int = NUM_PITCHES,
                 hidden_size: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.input_proj(x)
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.output_proj(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


# =============================================================================
# MIDI PROCESSING
# =============================================================================

def midi_to_piano_roll(midi_path: Path, config: dict = None,
                       trim_silence: bool = True) -> Optional[np.ndarray]:
    """
    Convert a MIDI file to a piano roll representation.

    Args:
        midi_path: Path to the MIDI file
        config: MIDI configuration dictionary
        trim_silence: If True, remove leading silence from the piano roll

    Returns:
        numpy array of shape (time_steps, num_pitches) with velocity values [0, 1]
    """
    if config is None:
        config = MIDI_CONFIG

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None

    end_time = min(midi.get_end_time(), config["max_duration"])
    if end_time < 1.0:
        return None

    piano_roll = midi.get_piano_roll(fs=config["fs"])
    max_frames = int(config["max_duration"] * config["fs"])
    piano_roll = piano_roll[config["min_pitch"]:config["max_pitch"]+1, :max_frames]
    piano_roll = piano_roll / 127.0
    piano_roll = np.clip(piano_roll, 0, 1)
    piano_roll = piano_roll.T  # Shape: (time_steps, num_pitches)

    if trim_silence:
        # Find the first frame with any note activity
        frame_activity = np.any(piano_roll > 0, axis=1)
        first_active = np.argmax(frame_activity)
        if first_active > 0 and frame_activity[first_active]:
            piano_roll = piano_roll[first_active:]

    return piano_roll.astype(np.float32)


def piano_roll_to_midi(piano_roll: np.ndarray, config: dict,
                       output_path: Path, velocity_threshold: float = 0.3) -> None:
    """
    Convert a piano roll back to a MIDI file.

    Args:
        piano_roll: Array of shape (time_steps, num_pitches) with values in [0, 1]
        config: MIDI configuration dictionary
        output_path: Path to save the MIDI file
        velocity_threshold: Minimum velocity to consider a note as "on"
    """
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    frame_duration = 1.0 / config["fs"]
    active_notes = {}

    for frame_idx, frame in enumerate(piano_roll):
        current_time = frame_idx * frame_duration

        for pitch_idx, velocity in enumerate(frame):
            midi_pitch = pitch_idx + config["min_pitch"]
            is_note_on = velocity > velocity_threshold

            if midi_pitch in active_notes:
                if not is_note_on:
                    start_time, start_vel = active_notes.pop(midi_pitch)
                    note = pretty_midi.Note(
                        velocity=int(start_vel * 127),
                        pitch=midi_pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)
            else:
                if is_note_on:
                    active_notes[midi_pitch] = (current_time, velocity)

    end_time = len(piano_roll) * frame_duration
    for midi_pitch, (start_time, velocity) in active_notes.items():
        note = pretty_midi.Note(
            velocity=int(velocity * 127),
            pitch=midi_pitch,
            start=start_time,
            end=end_time
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(str(output_path))


# =============================================================================
# TICK-BASED ENCODING (Alternative to Piano Roll)
# =============================================================================

def midi_to_tick_sequence(midi_path: Path, ticks_per_beat: int = TICKS_PER_BEAT,
                          max_ticks: int = None, trim_silence: bool = True
                          ) -> Optional[np.ndarray]:
    """
    Convert MIDI to tick-based sequence with velocity and duration.

    Each tick is a 16th note. Output shape: (num_ticks, 88, 2)
    - Channel 0: velocity (0 = no note, >0 = note velocity)
    - Channel 1: duration in ticks (how long the note lasts)

    Args:
        midi_path: Path to MIDI file
        ticks_per_beat: Resolution (4 = 16th notes)
        max_ticks: Maximum number of ticks (None = no limit)
        trim_silence: Remove leading silence

    Returns:
        Array of shape (num_ticks, 88, 2) or None if error
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Error loading {midi_path}: {e}")
        return None

    if midi.get_end_time() < 1.0:
        return None

    # Get tempo (use first tempo or default)
    tempo = DEFAULT_TEMPO
    if midi.get_tempo_changes()[1].size > 0:
        tempo = midi.get_tempo_changes()[1][0]

    # Calculate tick duration in seconds
    beat_duration = 60.0 / tempo  # seconds per beat
    tick_duration = beat_duration / ticks_per_beat  # seconds per tick

    # Calculate total ticks needed
    end_time = midi.get_end_time()
    total_ticks = int(np.ceil(end_time / tick_duration))

    if max_ticks is not None:
        total_ticks = min(total_ticks, max_ticks)

    # Initialize: (ticks, 88 pitches, 2 channels: velocity + duration)
    sequence = np.zeros((total_ticks, NUM_PITCHES, 2), dtype=np.float32)

    # Process all notes from all instruments
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            # Convert pitch to index (0-87)
            pitch_idx = note.pitch - MIDI_CONFIG["min_pitch"]
            if pitch_idx < 0 or pitch_idx >= NUM_PITCHES:
                continue

            # Convert time to tick
            start_tick = int(note.start / tick_duration)
            end_tick = int(note.end / tick_duration)

            if start_tick >= total_ticks:
                continue

            # Duration in ticks (capped)
            duration = min(end_tick - start_tick, MAX_DURATION_TICKS)
            duration = max(duration, 1)  # At least 1 tick

            # Velocity normalized to 0-1
            velocity = note.velocity / 127.0

            # Store at start tick
            # If there's already a note, keep the louder one
            if velocity > sequence[start_tick, pitch_idx, 0]:
                sequence[start_tick, pitch_idx, 0] = velocity
                sequence[start_tick, pitch_idx, 1] = duration / MAX_DURATION_TICKS  # Normalize duration

    # Trim leading silence
    if trim_silence:
        tick_activity = np.any(sequence[:, :, 0] > 0, axis=1)
        first_active = np.argmax(tick_activity)
        if first_active > 0 and tick_activity[first_active]:
            sequence = sequence[first_active:]

    return sequence


def tick_sequence_to_midi(sequence: np.ndarray, output_path: Path,
                          ticks_per_beat: int = TICKS_PER_BEAT,
                          tempo: float = DEFAULT_TEMPO,
                          velocity_threshold: float = 0.1) -> None:
    """
    Convert tick sequence back to MIDI file.

    Args:
        sequence: Array of shape (num_ticks, 88, 2) - velocity and duration
        output_path: Where to save MIDI
        ticks_per_beat: Resolution (must match encoding)
        tempo: BPM for output
        velocity_threshold: Minimum velocity to create note
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    beat_duration = 60.0 / tempo
    tick_duration = beat_duration / ticks_per_beat

    for tick_idx in range(len(sequence)):
        for pitch_idx in range(NUM_PITCHES):
            velocity = sequence[tick_idx, pitch_idx, 0]
            duration_norm = sequence[tick_idx, pitch_idx, 1]

            if velocity > velocity_threshold:
                midi_pitch = pitch_idx + MIDI_CONFIG["min_pitch"]
                start_time = tick_idx * tick_duration

                # Denormalize duration
                duration_ticks = max(1, int(duration_norm * MAX_DURATION_TICKS))
                end_time = start_time + (duration_ticks * tick_duration)

                note = pretty_midi.Note(
                    velocity=int(velocity * 127),
                    pitch=midi_pitch,
                    start=start_time,
                    end=end_time
                )
                piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(str(output_path))


def play_tick_sequence(sequence: np.ndarray, ticks_per_beat: int = TICKS_PER_BEAT,
                       tempo: float = DEFAULT_TEMPO, sample_rate: int = 22050):
    """Play a tick sequence as audio."""
    from IPython.display import Audio

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    beat_duration = 60.0 / tempo
    tick_duration = beat_duration / ticks_per_beat

    for tick_idx in range(len(sequence)):
        for pitch_idx in range(NUM_PITCHES):
            velocity = sequence[tick_idx, pitch_idx, 0]
            duration_norm = sequence[tick_idx, pitch_idx, 1]

            if velocity > 0.1:
                midi_pitch = pitch_idx + MIDI_CONFIG["min_pitch"]
                start_time = tick_idx * tick_duration
                duration_ticks = max(1, int(duration_norm * MAX_DURATION_TICKS))
                end_time = start_time + (duration_ticks * tick_duration)

                note = pretty_midi.Note(
                    velocity=int(velocity * 127),
                    pitch=midi_pitch,
                    start=start_time,
                    end=end_time
                )
                piano.notes.append(note)

    midi.instruments.append(piano)

    if _check_fluidsynth():
        audio_data = midi.fluidsynth(fs=sample_rate)
    else:
        audio_data = midi.synthesize(fs=sample_rate)

    return Audio(audio_data, rate=sample_rate)


# =============================================================================
# AUDIO PLAYBACK
# =============================================================================

def _check_fluidsynth() -> bool:
    """Check if FluidSynth is available."""
    try:
        import fluidsynth
        return True
    except ImportError:
        return False


def play_piano_roll(piano_roll: np.ndarray, config: dict = None,
                    sample_rate: int = 22050):
    """
    Convert piano roll to audio and create playable Audio widget.

    Returns:
        IPython Audio widget
    """
    from IPython.display import Audio

    if config is None:
        config = MIDI_CONFIG

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    frame_duration = 1.0 / config["fs"]
    velocity_threshold = 0.3
    active_notes = {}

    for frame_idx, frame in enumerate(piano_roll):
        current_time = frame_idx * frame_duration

        for pitch_idx, velocity in enumerate(frame):
            midi_pitch = pitch_idx + config["min_pitch"]
            is_note_on = velocity > velocity_threshold

            if midi_pitch in active_notes:
                if not is_note_on:
                    start_time, start_vel = active_notes.pop(midi_pitch)
                    note = pretty_midi.Note(
                        velocity=int(start_vel * 127),
                        pitch=midi_pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)
            else:
                if is_note_on:
                    active_notes[midi_pitch] = (current_time, velocity)

    end_time = len(piano_roll) * frame_duration
    for midi_pitch, (start_time, velocity) in active_notes.items():
        note = pretty_midi.Note(
            velocity=int(velocity * 127),
            pitch=midi_pitch,
            start=start_time,
            end=end_time
        )
        piano.notes.append(note)

    midi.instruments.append(piano)

    if _check_fluidsynth():
        audio_data = midi.fluidsynth(fs=sample_rate)
    else:
        audio_data = midi.synthesize(fs=sample_rate)

    return Audio(audio_data, rate=sample_rate)


def play_midi_file(midi_path: Path, sample_rate: int = 22050):
    """Load and play a MIDI file."""
    from IPython.display import Audio

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    if _check_fluidsynth():
        audio_data = midi.fluidsynth(fs=sample_rate)
    else:
        audio_data = midi.synthesize(fs=sample_rate)

    return Audio(audio_data, rate=sample_rate)


# =============================================================================
# GENERATION
# =============================================================================

@torch.no_grad()
def generate_music(model: nn.Module, seed_sequence: np.ndarray,
                   num_frames: int, temperature: float = 1.0,
                   top_k: int = 8, device: torch.device = None,
                   show_progress: bool = True) -> np.ndarray:
    """
    Generate music from a seed sequence using top-k sampling.

    Args:
        model: Trained model (outputs logits, not probabilities)
        seed_sequence: Initial sequence (seq_len, num_pitches)
        num_frames: Number of frames to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Maximum notes per frame (typical piano: 1-10 notes)
        device: Torch device
        show_progress: Show progress bar

    Returns:
        Generated piano roll (seed_len + num_frames, num_pitches) with values in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    current_seq = torch.FloatTensor(seed_sequence).unsqueeze(0).to(device)
    generated = list(seed_sequence)
    hidden = model.init_hidden(1, device)

    iterator = range(num_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating")

    for _ in iterator:
        logits, hidden = model(current_seq, hidden)
        next_logits = logits[0, -1].cpu().numpy()

        if temperature != 1.0:
            next_logits = next_logits / temperature

        next_probs = 1 / (1 + np.exp(-next_logits))

        # Top-k sampling: only keep the k most likely notes
        # This prevents the "noise everywhere" problem
        next_frame = np.zeros_like(next_probs)
        top_indices = np.argsort(next_probs)[-top_k:]  # Get top k indices

        for idx in top_indices:
            # Sample based on probability, but only from top-k candidates
            if np.random.random() < next_probs[idx]:
                next_frame[idx] = 0.7  # Fixed velocity for audibility

        generated.append(next_frame)

        next_tensor = torch.FloatTensor(next_frame).unsqueeze(0).unsqueeze(0).to(device)
        current_seq = torch.cat([current_seq[:, 1:, :], next_tensor], dim=1)

    return np.array(generated)


def create_random_seed(sequence_length: int = SEQUENCE_LENGTH,
                       num_pitches: int = NUM_PITCHES) -> np.ndarray:
    """Create a random seed with sparse notes for generation (piano roll)."""
    seed = np.zeros((sequence_length, num_pitches), dtype=np.float32)

    for i in range(0, sequence_length, 8):
        num_notes = random.randint(2, 4)
        pitches = random.sample(range(30, 60), num_notes)
        for p in pitches:
            velocity = random.uniform(0.5, 1.0)
            duration = random.randint(4, 12)
            for j in range(min(duration, sequence_length - i)):
                seed[i + j, p] = velocity

    return seed


def create_random_tick_seed(sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Create a random seed for tick-based generation.

    Returns: (seq_len, 88, 2) array with velocity and duration channels.
    """
    seed = np.zeros((sequence_length, NUM_PITCHES, 2), dtype=np.float32)

    # Add some random notes every few ticks
    for i in range(0, sequence_length, 4):  # Every beat
        num_notes = random.randint(1, 4)
        pitches = random.sample(range(20, 60), num_notes)  # Middle register
        for p in pitches:
            velocity = random.uniform(0.5, 0.9)
            duration = random.randint(2, 8) / MAX_DURATION_TICKS  # Normalized
            seed[i, p, 0] = velocity
            seed[i, p, 1] = duration

    return seed


@torch.no_grad()
def generate_tick_music(model: nn.Module, seed_sequence: np.ndarray,
                        num_ticks: int, temperature: float = 1.0,
                        top_k: int = 6, device: torch.device = None,
                        show_progress: bool = True) -> np.ndarray:
    """
    Generate tick-based music from a seed sequence.

    Args:
        model: Trained model (input/output: 176 dims = 88 pitches × 2 channels)
        seed_sequence: Shape (seq_len, 88, 2) or (seq_len, 176)
        num_ticks: Number of ticks to generate
        temperature: Sampling temperature
        top_k: Max notes per tick
        device: Torch device
        show_progress: Show progress bar

    Returns:
        Array of shape (seed_len + num_ticks, 88, 2)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Ensure seed is flattened: (seq_len, 176)
    if seed_sequence.ndim == 3:
        seed_flat = seed_sequence.reshape(len(seed_sequence), -1)
    else:
        seed_flat = seed_sequence

    current_seq = torch.FloatTensor(seed_flat).unsqueeze(0).to(device)
    generated_flat = list(seed_flat)
    hidden = model.init_hidden(1, device)

    iterator = range(num_ticks)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating")

    for _ in iterator:
        output, hidden = model(current_seq, hidden)
        next_logits = output[0, -1].cpu().numpy()  # Shape: (176,)

        if temperature != 1.0:
            next_logits = next_logits / temperature

        # Split into velocity (88) and duration (88) predictions
        vel_logits = next_logits[:NUM_PITCHES]
        dur_logits = next_logits[NUM_PITCHES:]

        # Convert velocity logits to probabilities
        vel_probs = 1 / (1 + np.exp(-vel_logits))

        # Top-k sampling for velocities
        next_vel = np.zeros(NUM_PITCHES, dtype=np.float32)
        next_dur = np.zeros(NUM_PITCHES, dtype=np.float32)

        top_indices = np.argsort(vel_probs)[-top_k:]

        # ALWAYS include top-2 notes (forced) to ensure music plays
        # Then probabilistically add more from top-k
        min_forced = 2  # Always include at least 2 notes per tick

        for rank, idx in enumerate(reversed(top_indices)):  # highest prob first
            if rank < min_forced:
                # Force include top notes
                next_vel[idx] = np.clip(vel_probs[idx] + 0.4, 0.5, 1.0)
                next_dur[idx] = np.clip(1 / (1 + np.exp(-dur_logits[idx])), 0.2, 1.0)
            else:
                # Probabilistic for the rest
                if np.random.random() < vel_probs[idx]:
                    next_vel[idx] = np.clip(vel_probs[idx] + 0.3, 0.4, 1.0)
                    next_dur[idx] = np.clip(1 / (1 + np.exp(-dur_logits[idx])), 0.1, 1.0)

        # Combine and append
        next_frame = np.concatenate([next_vel, next_dur])
        generated_flat.append(next_frame)

        # Update sequence
        next_tensor = torch.FloatTensor(next_frame).unsqueeze(0).unsqueeze(0).to(device)
        current_seq = torch.cat([current_seq[:, 1:, :], next_tensor], dim=1)

    # Reshape back to (ticks, 88, 2)
    result = np.array(generated_flat).reshape(-1, NUM_PITCHES, 2)
    return result


# =============================================================================
# DATASET
# =============================================================================

class MidiDataset(Dataset):
    """
    PyTorch Dataset for MIDI piano roll sequences.

    Creates overlapping sequences using a sliding window approach.
    """

    def __init__(self, piano_rolls: List[np.ndarray],
                 sequence_length: int = SEQUENCE_LENGTH,
                 stride: int = STRIDE):
        self.sequences = []
        self.targets = []

        for roll in piano_rolls:
            for i in range(0, len(roll) - sequence_length - 1, stride):
                seq = roll[i:i + sequence_length]
                target = roll[i + 1:i + sequence_length + 1]
                self.sequences.append(seq)
                self.targets.append(target)

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )


class TickDataset(Dataset):
    """
    PyTorch Dataset for tick-based MIDI sequences.

    Input shape per sequence: (seq_len, 88, 2) -> flattened to (seq_len, 176)
    """

    def __init__(self, tick_sequences: List[np.ndarray],
                 sequence_length: int = SEQUENCE_LENGTH,
                 stride: int = STRIDE):
        self.sequences = []
        self.targets = []

        for seq in tick_sequences:
            # Flatten (ticks, 88, 2) to (ticks, 176)
            flat = seq.reshape(len(seq), -1)

            for i in range(0, len(flat) - sequence_length - 1, stride):
                x = flat[i:i + sequence_length]
                y = flat[i + 1:i + sequence_length + 1]
                self.sequences.append(x)
                self.targets.append(y)

        if self.sequences:
            self.sequences = np.array(self.sequences)
            self.targets = np.array(self.targets)
        else:
            self.sequences = np.empty((0, sequence_length, TICK_FEATURES))
            self.targets = np.empty((0, sequence_length, TICK_FEATURES))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(checkpoint_path: Path, model_type: str = "mamba",
               device: torch.device = None) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        model_type: "mamba" or "lstm"
        device: Target device

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = get_device()

    if model_type == "mamba":
        model = MusicMamba(input_size=NUM_PITCHES).to(device)
    else:
        model = MusicLSTM(input_size=NUM_PITCHES).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded {model_type} model from epoch {checkpoint['epoch']+1}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    return model
