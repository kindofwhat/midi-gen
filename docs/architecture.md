# MusicGPT Architecture for MIDI Generation

This document explains the complete architecture for MIDI music generation, including tokenization, model architecture, training, and generation.

## Overview

The system treats music generation as a **language modeling task**: predict the next token given previous tokens.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MIDI File (.mid)                                                          │
│        │                                                                    │
│        ▼ midi_file_to_tokens()                                              │
│   ┌─────────────────────────────────────────┐                               │
│   │  Token Sequence                         │                               │
│   │  [BOS, event1, event2, ..., EOS]        │                               │
│   │  Each event = 8 tokens                  │                               │
│   └─────────────────────────────────────────┘                               │
│        │                                                                    │
│        ▼ flatten_tokens()                                                   │
│   ┌─────────────────────────────────────────┐                               │
│   │  Flat Sequence                          │                               │
│   │  [1, 0, 0, 0, 0, 0, 0, 0, 3, 9, ...]    │                               │
│   └─────────────────────────────────────────┘                               │
│        │                                                                    │
│        ▼ MusicGPT                                                           │
│   ┌─────────────────────────────────────────┐                               │
│   │  Next Token Prediction                  │                               │
│   │  P(token_t | token_1, ..., token_{t-1}) │                               │
│   └─────────────────────────────────────────┘                               │
│        │                                                                    │
│        ▼ unflatten_tokens()                                                 │
│   ┌─────────────────────────────────────────┐                               │
│   │  Token Sequence (num_events, 8)         │                               │
│   └─────────────────────────────────────────┘                               │
│        │                                                                    │
│        ▼ tokens_to_midi_file()                                              │
│   MIDI File (.mid)                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Part 1: MIDI Tokenization

### Why Tokenize MIDI?

MIDI files are binary event streams. Neural networks need numerical sequences. The tokenizer bridges this gap.

**Alternative approaches:**

| Approach | Pros | Cons |
|----------|------|------|
| Piano Roll (88×T matrix) | Simple, visual | Dense, slow, loses timing precision |
| Raw MIDI bytes | Complete info | Too low-level, variable length events |
| **Event Tokens** | Sparse, structured, proven | Requires careful design |

### Token Format

Each MIDI event becomes **8 tokens**:

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Event Token Structure (8 tokens per event)                               │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Position:  [0]      [1]     [2]     [3]    [4]      [5]     [6]    [7]   │
│  Meaning:   event    time1   time2   track  param1   param2  param3 param4│
│                                                                           │
│  Example (note event):                                                    │
│  [3, 9, 8, 147, 16, 188, 208, 265]                                        │
│   │  │  │   │    │    │    │    └─ duration (how long)                    │
│   │  │  │   │    │    │    └────── velocity (how loud)                    │
│   │  │  │   │    │    └─────────── pitch (which note: 60=C4)              │
│   │  │  │   │    └──────────────── channel (instrument group)             │
│   │  │  │   └───────────────────── track number                           │
│   │  │  └───────────────────────── time2 (fine: 0-15 sixteenth notes)     │
│   │  └──────────────────────────── time1 (coarse: delta from last event)  │
│   └─────────────────────────────── event_type (3 = "note")                │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Event Types

| ID | Event Type | Parameters | Purpose |
|----|------------|------------|---------|
| 3 | `note` | channel, pitch, velocity, duration | Musical notes |
| 4 | `patch_change` | channel, patch | Change instrument |
| 5 | `control_change` | channel, controller, value | Pedal, volume, etc. |
| 6 | `set_tempo` | bpm | Tempo changes |
| 7 | `time_signature` | nn, dd | Time signature |
| 8 | `key_signature` | sf, mi | Key signature |

### Vocabulary Breakdown (3406 tokens)

Token IDs are allocated sequentially in dictionary insertion order (see `midi-model/midi_tokenizer.py`):

```
Token ID ranges (MIDITokenizerV2):
├── 0            PAD (padding)
├── 1            BOS (begin of sequence)
├── 2            EOS (end of sequence)
├── 3-8          Event types (6 types: note, patch_change, control_change, set_tempo, time_signature, key_signature)
├── 9-136        time1 (128 values: coarse time, delta from previous event)
├── 137-152      time2 (16 values: fine time, 0-15 sixteenth notes)
├── 153-2200     duration (2048 values: note length in sixteenth notes)
├── 2201-2328    track (128 values: MIDI track number)
├── 2329-2344    channel (16 values: MIDI channel 0-15, channel 9 = drums)
├── 2345-2472    pitch (128 values: MIDI notes 0-127)
├── 2473-2600    velocity (128 values: note velocity 0-127)
├── 2601-2728    patch (128 values: GM instrument programs 0-127)
├── 2729-2856    controller (128 values: CC numbers 0-127)
├── 2857-2984    value (128 values: CC values 0-127)
├── 2985-3368    bpm (384 values: tempo 1-384 BPM)
├── 3369-3384    nn (16 values: time signature numerator 1-16)
├── 3385-3388    dd (4 values: time signature denominator 2^1 to 2^4)
├── 3389-3403    sf (15 values: key signature -7 to +7 sharps/flats, encoded as 0-14)
├── 3404-3405    mi (2 values: mode indicator, 0=major, 1=minor)
└── 3406         Total vocabulary size
```

### Time Encoding: Relative, Not Absolute

**Critical for understanding generation:**

```
Event 1: time1=0,  time2=0   → absolute = 0
Event 2: time1=2,  time2=8   → absolute = 0 + (2×16 + 8) = 40 sixteenths
Event 3: time1=1,  time2=4   → absolute = 40 + (1×16 + 4) = 60 sixteenths
         ↑
         DELTA from previous, not absolute time!
```

This is why **seed-based continuation is hard**: the model learns to generate time deltas from the previous event, not absolute positions.

## Part 2: MusicGPT Model Architecture

### Design Choice: Why Transformer?

| Architecture | Complexity | Long-range | Status |
|--------------|-----------|------------|--------|
| LSTM | O(N) | Poor | Baseline |
| Mamba (SSM) | O(N) | Good | Experimental |
| **Transformer** | O(N²) | Excellent | **Proven for music** |

For sequences of 512 tokens, O(N²) is acceptable. Transformers are battle-tested for music (MuseNet, Music Transformer, MusicLM).

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MusicGPT                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input: Token IDs (batch, seq_len=512)                         │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │  Token Embedding                        │                   │
│   │  nn.Embedding(3406, 256, padding_idx=0) │                   │
│   │  → (batch, 512, 256)                    │                   │
│   └─────────────────────────────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │  Positional Embedding                   │                   │
│   │  nn.Embedding(576, 256)                 │                   │
│   │  + (learned, not sinusoidal)            │                   │
│   └─────────────────────────────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │  Position-in-Event Embedding (NEW)      │                   │
│   │  nn.Embedding(8, 256)                   │                   │
│   │  position % 8 → tells model "you're at  │                   │
│   │  position 3 of an 8-token event"        │                   │
│   └─────────────────────────────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │  Dropout (0.1)                          │                   │
│   └─────────────────────────────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │  Transformer Decoder Block × 6          │                   │
│   │  (see detailed view below)              │                   │
│   └─────────────────────────────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │  Language Model Head                    │                   │
│   │  nn.Linear(256, 3406)                   │                   │
│   │  → (batch, 512, 3406) logits            │                   │
│   └─────────────────────────────────────────┘                   │
│                         │                                       │
│                         ▼                                       │
│   Output: Logits (batch, seq_len, vocab_size)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Transformer Decoder Block (×6)

```
┌─────────────────────────────────────────────────────────────────┐
│                  TransformerDecoderLayer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input x (batch, seq_len, d_model=256)                         │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────┐                      │
│   │  Multi-Head Self-Attention           │                      │
│   │  • 8 heads, head_dim = 32            │                      │
│   │  • Causal mask (can't see future)    │                      │
│   │  • Q, K, V all from same input       │                      │
│   └──────────────────────────────────────┘                      │
│              │                                                  │
│              ├──────────── + (residual)                         │
│              ▼                                                  │
│   ┌──────────────────────────────────────┐                      │
│   │  LayerNorm                           │                      │
│   └──────────────────────────────────────┘                      │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────┐                      │
│   │  Feed-Forward Network                │                      │
│   │  Linear(256 → 1024) + ReLU           │                      │
│   │  Dropout(0.1)                        │                      │
│   │  Linear(1024 → 256)                  │                      │
│   └──────────────────────────────────────┘                      │
│              │                                                  │
│              ├──────────── + (residual)                         │
│              ▼                                                  │
│   ┌──────────────────────────────────────┐                      │
│   │  LayerNorm                           │                      │
│   └──────────────────────────────────────┘                      │
│              │                                                  │
│              ▼                                                  │
│   Output (batch, seq_len, d_model=256)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Causal Masking

The model can only attend to **past and present** tokens, not future ones:

```
Query positions →
         0   1   2   3   4
Key   0 [✓] [✗] [✗] [✗] [✗]
pos   1 [✓] [✓] [✗] [✗] [✗]
↓     2 [✓] [✓] [✓] [✗] [✗]
      3 [✓] [✓] [✓] [✓] [✗]
      4 [✓] [✓] [✓] [✓] [✓]

✓ = can attend, ✗ = masked (set to -inf before softmax)
```

This ensures autoregressive generation: each token prediction only uses information from previous tokens.

### Parameter Count

```python
Component                              Parameters
─────────────────────────────────────────────────
Token Embedding (3406 × 256)           872,000
Position Embedding (576 × 256)         147,456
Transformer Blocks (6×):
  ├─ Self-Attention (4 × 256²)         262,144 × 6
  ├─ FFN (256×1024 + 1024×256)         524,288 × 6
  └─ LayerNorms (2 × 256 × 2)          1,024 × 6
LM Head (256 × 3406)                   872,000
─────────────────────────────────────────────────
Total                                  ~8.2M parameters
```

## Part 3: Training

### Task: Next Token Prediction

Given tokens `[t₁, t₂, ..., tₙ]`, predict `[t₂, t₃, ..., tₙ₊₁]`.

```
Input:  [BOS, 3, 9, 8, 147, 16, 188, 208, 265, 3, 9, 12, ...]
Target: [3, 9, 8, 147, 16, 188, 208, 265, 3, 9, 12, 150, ...]
         ↑
         Shifted by 1 position
```

### Loss Function: Cross-Entropy

```python
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
```

- Compares predicted probability distribution over 3406 tokens with actual next token
- Ignores PAD tokens (don't penalize predictions at padding positions)
- Reports as **perplexity** = exp(loss) for interpretability

### Dataset Creation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sliding Window Dataset                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original sequence (one MIDI file, ~9000 tokens):               │
│  [t₁, t₂, t₃, t₄, t₅, t₆, t₇, t₈, t₉, t₁₀, ...]                 │
│                                                                 │
│  SEQUENCE_LENGTH = 512                                          │
│  STRIDE = 256 (50% overlap)                                     │
│                                                                 │
│  Sample 1: [t₁ ... t₅₁₂]     (positions 0-511)                  │
│  Sample 2: [t₂₅₇ ... t₇₆₈]   (positions 256-767)                │
│  Sample 3: [t₅₁₃ ... t₁₀₂₄]  (positions 512-1023)               │
│  ...                                                            │
│                                                                 │
│  Total samples = (seq_len - 512) / 256 per file                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Configuration

```python
SEQUENCE_LENGTH = 512      # Context window
STRIDE = 256               # 50% overlap between samples
BATCH_SIZE = 32            # Samples per batch
LEARNING_RATE = 3e-4       # Initial LR
WEIGHT_DECAY = 0.01        # AdamW regularization
GRAD_CLIP = 1.0            # Gradient clipping
NUM_EPOCHS = 30            # Max epochs
EARLY_STOPPING = 5         # Patience for early stopping
```

### Learning Rate Schedule

```
ReduceLROnPlateau:
├─ Monitor: validation loss
├─ Factor: 0.5 (halve LR when plateau)
├─ Patience: 2 epochs
└─ Triggers when val_loss stops improving
```

### Overfitting Prevention

| Technique | Setting | Purpose |
|-----------|---------|---------|
| Dropout | 0.1 | Random neuron deactivation |
| Weight Decay | 0.01 | L2 regularization |
| Early Stopping | patience=5 | Stop when val_loss plateaus |
| More Data | 11k files | Best regularization |

## Part 4: Generation

### Autoregressive Sampling

```python
@torch.no_grad()
def generate(model, seed_tokens, max_events, temperature, top_k, top_p):
    generated = list(seed_tokens)  # Start with BOS or seed

    for _ in range(max_events * 8):  # 8 tokens per event
        # Get last 512 tokens as context
        context = generated[-SEQUENCE_LENGTH:]

        # Forward pass
        logits = model(context)[-1]  # Last position

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, top_k)
            logits[logits < top_k_logits[-1]] = -inf

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_probs = softmax(logits).sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum > top_p
            logits[mask] = -inf

        # Sample
        probs = softmax(logits)
        next_token = torch.multinomial(probs, 1)

        generated.append(next_token)

        if next_token == EOS_ID:
            break

    return generated
```

### Sampling Parameters

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `temperature=0.5` | Conservative, repetitive | Safe choice |
| `temperature=0.8` | Balanced creativity | **Default** |
| `temperature=1.0` | Original distribution | More varied |
| `temperature=1.2+` | Very creative, chaotic | Experimental |
| `top_k=30` | Only consider top 30 tokens | Reduces noise |
| `top_p=0.95` | Nucleus sampling | Alternative to top_k |

### Degeneration Detection

The model can "degenerate" - produce invalid tokens. We detect this:

```python
# Valid event types are 1-8 (BOS, EOS, note, patch_change, etc.)
# If event_type > 100, something went wrong
if generated_event[0] > 100:
    print("Degeneration detected!")
    break
```

### Constrained Decoding (Key Improvement)

Raw model output is often poor quality. We apply constraints during generation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Constrained Decoding                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CONTROL EVENT PENALTY (ctrl_penalty=10.0)                   │
│     - Reduces logits for control_change events (type=5)         │
│     - Prevents "pedal spam" (excessive sustain/volume events)   │
│                                                                 │
│  2. NOTE RATIO ENFORCEMENT (min_note_ratio=0.8)                 │
│     - Tracks ratio of note events vs total                      │
│     - If below 80%, boosts note logits (+3.0)                   │
│     - Suppresses metadata events (-2.0)                         │
│                                                                 │
│  3. PITCH REPETITION PENALTY (pitch_repeat_penalty=3.0)         │
│     - Remembers last N pitches (pitch_memory=12)                │
│     - Penalizes recently used pitches at position 5             │
│     - Forces melodic variety instead of same note               │
│                                                                 │
│  4. PAD TOKEN ENFORCEMENT                                       │
│     - Each event type uses specific number of tokens            │
│     - note=8, time_sig=6, tempo=5, etc.                         │
│     - Force PAD for unused positions                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Results with constraints:**

| Metric | Without | With Constraints |
|--------|---------|------------------|
| Unique pitches | 1 | 30+ |
| Note ratio | ~20% | 95%+ |
| Events before degeneration | 2 | 60-100 |
| Musical quality | Random noise | Recognizable music |

## Part 5: File Structure

```
midi-gen/
├── midi_tokenizer.py       # Unified tokenizer class
├── midi_training.ipynb     # Training notebook
├── midi_generation.ipynb   # Generation notebook (with constrained decoding)
├── midi_data/
│   ├── adl-piano-midi/     # Training data (~11k MIDI files)
│   ├── best_model_lm.pt    # Trained model checkpoint
│   └── generated/          # Generated outputs
├── midi-model/
│   ├── model.py            # MusicGPT model class (shared)
│   └── MIDI.py             # Low-level MIDI parser
└── docs/
    └── architecture.md     # This document
```

## Part 6: Checkpoint Format

```python
torch.save({
    'epoch': epoch,                      # Training epoch (best: 30)
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,                # Best validation loss (~0.73)
    'vocab_size': VOCAB_SIZE,            # 3406
    'sequence_length': SEQUENCE_LENGTH,  # 512
    'tokens_per_event': 8,               # Event structure (NEW)
}, 'best_model_lm.pt')
```

### Training Results

```
Dataset: 11,076 MIDI files (ADL Piano MIDI)
         8,709 training / 1,095 validation
         ~80M tokens total

Training: 35 epochs (early stopped at epoch 35)
          Best val_loss: 0.7274 (perplexity 2.1)
          Time: ~5 hours on CUDA
```

## Limitations & Known Issues

### 1. Seed-Based Continuation
The model can continue from seed sequences, starting with the seed then diverging:
- Use first 30-50 events as context
- Model continues in similar style
- Works best with seeds from training data genres

### 2. Long-Term Structure
512 tokens ≈ 64 events ≈ 4-8 bars of music. The model can't plan longer structures (verse-chorus, sonata form).

### 3. Degeneration
After 60-100 events, the model can produce invalid tokens:
- Detected by event_type > 20
- Stop generation and keep valid portion
- Constrained decoding delays but doesn't prevent this

### 4. Quality
The model produces recognizable music but:
- Limited harmonic sophistication
- Benefits from higher temperature for variety
- Constrained decoding is essential for usable output

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
2. [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) - GPT-2
3. [Music Transformer](https://arxiv.org/abs/1809.04281) - Transformers for music
4. [SkyTNT/midi-model](https://github.com/SkyTNT/midi-model) - Tokenizer source
5. [ADL Piano MIDI Dataset](https://github.com/lucasnfe/adl-piano-midi) - Training data
