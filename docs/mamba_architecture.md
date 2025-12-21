# Mamba Architecture for Music Generation

This document explains the Mamba-based model architecture used for MIDI music generation.

## Overview

Mamba is a **Selective State Space Model (SSM)** that provides an alternative to Transformers and LSTMs for sequence modeling. It achieves linear time complexity O(N) compared to Transformers' O(N²), while maintaining the ability to capture long-range dependencies.

**Key paper:** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MusicMamba                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input: Piano Roll (batch, seq_len, 88 pitches)               │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │  Input Projection (88 → 256)            │                  │
│   │  + LayerNorm                            │                  │
│   └─────────────────────────────────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │  MambaBlock #1                          │                  │
│   │  + Residual Connection                  │                  │
│   │  + LayerNorm + Dropout                  │                  │
│   └─────────────────────────────────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │  MambaBlock #2                          │                  │
│   │  + Residual Connection                  │                  │
│   │  + LayerNorm + Dropout                  │                  │
│   └─────────────────────────────────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │  MambaBlock #3                          │                  │
│   │  + Residual Connection                  │                  │
│   │  + LayerNorm + Dropout                  │                  │
│   └─────────────────────────────────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │  MambaBlock #4                          │                  │
│   │  + Residual Connection                  │                  │
│   │  + LayerNorm + Dropout                  │                  │
│   └─────────────────────────────────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────┐                  │
│   │  Output Projection                      │                  │
│   │  Linear(256→256) + GELU + Dropout       │                  │
│   │  Linear(256→88)                         │                  │
│   └─────────────────────────────────────────┘                  │
│                         │                                       │
│                         ▼                                       │
│   Output: Logits (batch, seq_len, 88 pitches)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## MambaBlock: The Core Component

Each MambaBlock is the fundamental repeating unit, analogous to a Transformer block or LSTM layer.

```
Input x (batch, seq_len, d_model=256)
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  1. INPUT PROJECTION                                         │
│     Linear: 256 → 1024 (expand × 2)                         │
│     Split into x (512) and z (512)                          │
│                                                              │
│     x = for SSM processing                                   │
│     z = for gating (controls output flow)                    │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  2. LOCAL CONTEXT (Conv1D)                                   │
│     Depthwise Conv1D: kernel_size=4                         │
│     Groups=512 (each channel independently)                  │
│     + SiLU activation                                        │
│                                                              │
│     Purpose: Capture local patterns before SSM               │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  3. SELECTIVE STATE SPACE MODEL (SSM)                        │
│     The key innovation of Mamba                              │
│                                                              │
│     For each timestep t:                                     │
│       • Compute Δt, Bt, Ct from input (selective!)          │
│       • Update hidden state: h_t = Ā·h_{t-1} + B̄·x_t        │
│       • Compute output: y_t = C_t·h_t                        │
│                                                              │
│     See detailed explanation below                           │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  4. GATED OUTPUT                                             │
│     y = y * SiLU(z)                                         │
│                                                              │
│     The gate z controls which information flows through      │
│     Similar to LSTM's output gate                            │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│  5. OUTPUT PROJECTION                                        │
│     Linear: 512 → 256                                       │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
Output (batch, seq_len, d_model=256)
```

## The Selective SSM: What Makes Mamba Special

### Traditional SSM (Fixed Parameters)

A State Space Model is defined by:
```
h_t = A·h_{t-1} + B·x_t    (state update)
y_t = C·h_t + D·x_t        (output)
```

Where:
- `h_t` = hidden state at time t
- `x_t` = input at time t
- `y_t` = output at time t
- `A, B, C, D` = learned parameters (fixed after training)

**Problem:** Fixed A, B, C means the model treats all inputs the same way. It can't "decide" what's important.

### Selective SSM (Input-Dependent Parameters)

Mamba makes B, C, and Δ (discretization step) **functions of the input**:

```
Δ_t, B_t, C_t = f(x_t)     ← Computed from current input!

Ā = exp(Δ_t · A)           ← Discretized A
B̄ = Δ_t · B_t              ← Discretized B

h_t = Ā·h_{t-1} + B̄·x_t    (state update - now input-dependent!)
y_t = C_t·h_t + D·x_t      (output)
```

### Why This Matters for Music

Consider a piano piece:
- **Important note (melody):** Model should remember it → Large Δ, strong B
- **Passing note:** Model can forget quickly → Small Δ, weak B
- **Chord change:** Need to update context → Different C to read state differently

The "selectivity" allows the model to dynamically decide:
- What to store in memory (controlled by B)
- How long to remember it (controlled by A and Δ)
- What to output (controlled by C)

## MambaBlock Code Walkthrough

This section traces through the actual implementation in `midi_utils.py` step by step.

### Step 1: Input Projection (Gating Setup)

```python
xz = self.in_proj(x)          # (batch, seq, d_model) → (batch, seq, d_inner*2)
x, z = xz.chunk(2, dim=-1)    # Split into: x (main path), z (gate)
```

The input is projected and split into two branches:
- `x` goes through the SSM processing
- `z` will gate the output later (similar to LSTM gates)

### Step 2: Local Convolution

```python
x = x.transpose(1, 2)              # → (batch, d_inner, seq) for Conv1d
x = self.conv1d(x)[:, :, :seq_len] # Depthwise conv, kernel=4
x = x.transpose(1, 2)              # → (batch, seq, d_inner)
x = silu(x)                        # Activation
```

This provides **local context mixing** (like seeing nearby frames) before the SSM processes the sequence.

### Step 3: Selective SSM - Parameter Computation

```python
# Project input to get SSM parameters (input-dependent!)
x_proj = self.x_proj(x)                    # → (batch, seq, d_state*2 + 1)
dt = x_proj[:, :, :1]                      # Δ (delta/time step) - 1 value
B  = x_proj[:, :, 1:d_state+1]             # B matrix - d_state values
C  = x_proj[:, :, d_state+1:]              # C matrix - d_state values
```

**This is the "selective" part**: B, C, and dt are **computed from the input** at each timestep. The model learns **when to update state and when to ignore input**.

### Step 4: Discretization

```python
dt = self.dt_proj(dt)              # Expand Δ to full dimension
dt = softplus(dt)                  # Ensure positive

A = -torch.exp(self.A_log)         # A is learned but fixed (not input-dependent)
dA = torch.einsum('bld,dn->bldn', dt, A)
dA = torch.exp(dA)                 # Discretized A: exp(Δ·A)
dB = torch.einsum('bld,bln->bldn', dt, B)  # Discretized B: Δ·B
```

This converts the continuous-time SSM to discrete time steps. The einsum operations efficiently compute:
- `dA[b,l,d,n] = exp(dt[b,l,d] * A[d,n])` - decay factor per state dimension
- `dB[b,l,d,n] = dt[b,l,d] * B[b,l,n]` - input contribution per state

### Step 5: The Recurrence (Sequential Scan)

```python
h = torch.zeros(batch, d_inner, d_state)  # Hidden state

for i in range(seq_len):
    # State update: h = dA * h + dB * x
    h = dA[:, i] * h + dB[:, i] * x[:, i:i+1, :].transpose(1, 2)

    # Output: y = C · h
    y = torch.einsum('bdn,bn->bd', h, C[:, i])
    ys.append(y)
```

**This is the heart of Mamba:**
- `dA[:, i] * h` → How much of the previous state to **keep** (forgetting factor)
- `dB[:, i] * x` → How much of the current input to **incorporate**
- `C[:, i]` → What part of the state to **read out**

### Step 6: Skip Connection

```python
y = y + x * self.D   # Direct input-to-output path (like ResNet skip)
```

The `D` parameter provides a direct path from input to output, helping gradient flow.

### Step 7: Gated Output

```python
y = y * silu(z)           # Gate with the z branch from step 1
return self.out_proj(y)   # Project back to d_model
```

The gate `z` (computed from the original input) controls which information flows through.

### Visual Summary

```
Input x (batch, seq, d_model)
         │
    ┌────┴────┐
    │  in_proj │ → Split
    └────┬────┘
    ┌────┴────┐
    x         z (gate)
    │         │
 Conv1d       │
    │         │
  SiLU        │
    │         │
 Selective    │
   SSM        │
    │         │
    ├── + D·x │
    │         │
    └────*────┘ (elementwise multiply with SiLU(z))
         │
     out_proj
         │
    Output (batch, seq, d_model)
```

### Intuition for Music Generation

For a music frame, the model learns:
- **High dt (large Δ)**: "This is important, remember it" (chord change, new phrase)
- **Low dt (small Δ)**: "Keep what I have, ignore this" (sustaining notes)
- **B matrix**: Controls **what aspects** of the input to incorporate into state
- **C matrix**: Controls **what aspects** of the state to read out

This selectivity is why Mamba can have better long-range memory than LSTMs despite being O(N).

## Parameter Dimensions

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `d_model` | 256 | Model hidden dimension |
| `d_state` | 16 | SSM state dimension (memory capacity) |
| `d_conv` | 4 | Convolution kernel size |
| `expand` | 2 | Expansion factor (d_inner = 512) |
| `n_layers` | 4 | Number of MambaBlocks |

**Total parameters:** ~1.8M

## Comparison with Other Architectures

| Aspect | Transformer | LSTM | Mamba |
|--------|-------------|------|-------|
| Complexity | O(N²) | O(N) | O(N) |
| Long-range | Excellent | Poor | Good |
| Parallelizable | Yes | No | Partial |
| Memory | Grows with N | Fixed | Fixed |
| Input-dependent | Attention | Gates | Selective SSM |

## Training Details

### Loss Function
Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`):
- Model outputs logits (no sigmoid)
- Loss function applies sigmoid internally
- More numerically stable

### Why No Mixed Precision on MPS?
The SSM's sequential state update:
```python
for i in range(seq_len):
    h = dA[:, i] * h + dB[:, i] * x[:, i:i+1, :].transpose(1, 2)
```

Contains:
- `torch.exp()` operations that can overflow in float16
- Cumulative multiplication that amplifies errors
- Long sequences compound the numerical instability

**Solution:** Use float32 for MPS, or add numerical clamping to the SSM.

## References

1. [Mamba Paper](https://arxiv.org/abs/2312.00752) - Gu & Dao, 2023
2. [Pure PyTorch Implementation](https://github.com/alxndrTL/mamba.py) - Basis for this code
3. [State Space Models Explained](https://srush.github.io/annotated-s4/) - Annotated S4
