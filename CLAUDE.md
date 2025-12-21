# CLAUDE.md - MIDI Music Generation Project

This document provides guidance for AI assistants working with this codebase.

## Project Overview

**Purpose**: Deep learning-based polyphonic piano music generation from MIDI files
**Technology**: PyTorch, pretty_midi, Apple Silicon MPS optimization
**Dataset**: MAESTRO v3.0.0 (~200 hours of virtuosic piano performances)

## Architecture

### Models Available

| Model | Parameters | Description | Recommended Use |
|-------|------------|-------------|-----------------|
| **MusicMamba** | 1.8M | Selective State Space Model | Faster, better long-range memory |
| **MusicLSTM** | ~6M | Long Short-Term Memory | Simpler baseline |

Both models output **logits** (not probabilities) for compatibility with `BCEWithLogitsLoss` and mixed precision training. Apply sigmoid during inference.

### Pipeline Flow

```
MIDI Files → Piano Roll (88 keys × time) → Sequence Dataset → Model Training → Generation → MIDI Export
```

## Key Configuration

### MIDI Processing
```python
MIDI_CONFIG = {
    "min_pitch": 21,       # A0 (lowest piano key)
    "max_pitch": 108,      # C8 (highest piano key)
    "fs": 16,              # Frames per second
    "velocity_bins": 32,   # Velocity quantization
    "max_duration": 60,    # Max clip duration (seconds)
}
NUM_PITCHES = 88  # Standard piano range
```

### Training Parameters
```python
SEQUENCE_LENGTH = 64   # Input sequence length (frames)
STRIDE = 16            # Sliding window stride
BATCH_SIZE = 64        # Optimized for Apple Silicon MPS
LEARNING_RATE = 0.001
GRAD_CLIP = 1.0
NUM_EPOCHS = 25        # Balance for demo; 100+ for best results
```

## Directory Structure

```
intellij-nb/
├── CLAUDE.md                          # This file
├── midi_music_generation.ipynb        # Main notebook
├── midi_data/
│   ├── maestro-v3.0.0/               # MAESTRO dataset
│   │   ├── maestro-v3.0.0.csv        # Metadata with official splits
│   │   └── *.midi                    # Piano performance files
│   ├── best_model.pt                 # Trained model checkpoint
│   └── generated/                    # Generated MIDI outputs
│       ├── generated_music.mid
│       └── generated_temp_*.mid
└── sprec1@vertex.ti.bfh.ch/          # Remote copy (SSH)
```

## Development Guidelines

### Device Handling

The notebook automatically selects the optimal device:
```python
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')  # Apple Silicon
else:
    DEVICE = torch.device('cpu')
```

**Apple Silicon optimizations applied**:
- `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops
- `num_workers=0` (multiprocessing issues with Metal)
- `pin_memory=False` (not beneficial for MPS)
- Mixed precision via `torch.autocast(device_type='mps', dtype=torch.float16)`

### Model Modifications

When modifying models, remember:
1. **Output logits, not probabilities** - Use `BCEWithLogitsLoss` for training
2. **Apply sigmoid during inference** - In `generate_music()` function
3. **Return (output, hidden) tuple** - Even if hidden is `None` (Mamba)

### Data Split Strategy

Uses MAESTRO's official splits for proper evaluation:
- **Train**: 962 files (~16 hours)
- **Validation**: 137 files (~2.3 hours)
- **Test**: 177 files (~3 hours)

**Important**: No data leakage between splits (same piece never appears in multiple splits).

## Key Functions

### MIDI Processing

```python
def midi_to_piano_roll(midi_path, config) -> np.ndarray:
    """Convert MIDI to piano roll. Shape: (time_steps, 88)"""

def piano_roll_to_midi(piano_roll, config, output_path, velocity_threshold=0.3):
    """Convert piano roll back to MIDI file."""
```

### Generation

```python
@torch.no_grad()
def generate_music(model, seed_sequence, num_frames, temperature=1.0):
    """
    Autoregressive generation from seed.

    Args:
        temperature: 0.5-0.6 = coherent, 0.8 = balanced, 1.0+ = creative
    Returns:
        Piano roll of shape (seed_len + num_frames, 88)
    """
```

### Audio Playback

```python
def play_piano_roll(piano_roll, config, sample_rate=22050) -> Audio:
    """Convert to audio widget for notebook playback."""
    # Uses FluidSynth if available, otherwise basic sine synthesis
```

## Model Checkpoints

Saved at `midi_data/best_model.pt`:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'val_loss': float,
}
```

## Common Tasks

### Resume Training
```python
checkpoint = torch.load('midi_data/best_model.pt', map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Generate with Different Seeds
```python
# From training data
seed = piano_rolls_by_split["train"][idx][:SEQUENCE_LENGTH]

# Random sparse pattern
seed = generate_random_seed()  # See generate_from_random_seed()
```

### Adjust Generation Quality
- **Lower temperature (0.5-0.7)**: More repetitive but coherent
- **Higher temperature (1.0-1.5)**: More creative but potentially chaotic
- **Longer training (50-100+ epochs)**: Better overall quality

## Dependencies

```
torch>=2.0
numpy
pretty_midi
matplotlib
tqdm
requests (for dataset download)
pyfluidsynth (optional, for high-quality audio)
```

### FluidSynth Installation (for better audio)
```bash
# macOS
brew install fluid-synth && pip install pyfluidsynth

# Ubuntu
apt install fluidsynth && pip install pyfluidsynth
```

## Performance Notes

| Configuration | Expected Training Time | Quality |
|--------------|------------------------|---------|
| 25 epochs, full dataset | 2-4 hours | Decent |
| 50 epochs, full dataset | 4-8 hours | Good |
| 100+ epochs, full dataset | 8+ hours | Best |

With Apple Silicon MPS (M1/M2/M3), expect approximately:
- ~2 min per epoch with full dataset
- ~30s per epoch with limited dataset (50 files)

## Troubleshooting

### MPS Memory Issues
```python
# Reduce batch size
BATCH_SIZE = 32  # or 16

# Clear cache periodically
torch.mps.empty_cache()
```

### NaN Loss During Training
- Check if model outputs have exploded (use gradient clipping)
- Reduce learning rate
- Ensure `BCEWithLogitsLoss` is used (not `BCELoss` with sigmoid)

### Generated Music Sounds Random
- Train for more epochs (50+)
- Lower temperature (0.5-0.7)
- Check that best model was loaded properly
- Ensure seed sequence has meaningful content

## References

- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
- [Mamba Paper](https://arxiv.org/abs/2312.00752) - Linear-Time Sequence Modeling
- [Pure PyTorch Mamba](https://github.com/alxndrTL/mamba.py)
- [pretty_midi Documentation](https://craffel.github.io/pretty-midi/)