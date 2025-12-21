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
# This prevents the model from learning to predict "silence everywhere"
POS_WEIGHT = 20.0  # Weight for positive class (note-on events)

MIDI_CONFIG = {
    "min_pitch": 21,       # A0 (lowest piano key)
    "max_pitch": 108,      # C8 (highest piano key)
    "fs": 16,              # Frames per second (temporal resolution)
    "velocity_bins": 32,   # Velocity quantization bins
    "max_duration": 60,    # Max clip duration in seconds
}

NUM_PITCHES = MIDI_CONFIG["max_pitch"] - MIDI_CONFIG["min_pitch"] + 1  # 88 piano keys
SEQUENCE_LENGTH = 64   # Input sequence length
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
                   device: torch.device = None, show_progress: bool = True) -> np.ndarray:
    """
    Generate music from a seed sequence.

    Args:
        model: Trained model (outputs logits, not probabilities)
        seed_sequence: Initial sequence (seq_len, num_pitches)
        num_frames: Number of frames to generate
        temperature: Sampling temperature (higher = more random)
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
        binary_frame = (np.random.random(next_probs.shape) < next_probs).astype(np.float32)

        # Use fixed velocity (0.7) for notes that are "on", not the raw probability
        # This ensures notes are audible (above the 0.3 playback threshold)
        next_frame = binary_frame * 0.7

        generated.append(next_frame)

        next_tensor = torch.FloatTensor(next_frame).unsqueeze(0).unsqueeze(0).to(device)
        current_seq = torch.cat([current_seq[:, 1:, :], next_tensor], dim=1)

    return np.array(generated)


def create_random_seed(sequence_length: int = SEQUENCE_LENGTH,
                       num_pitches: int = NUM_PITCHES) -> np.ndarray:
    """Create a random seed with sparse notes for generation."""
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
