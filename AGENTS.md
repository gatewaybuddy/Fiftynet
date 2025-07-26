# AGENTS.md

## 🧠 Purpose

This document defines the roles and responsibilities of each module in the FFT-Net hybrid transformer architecture. Each "agent" (module) plays a specific part in the pipeline of transforming input tokens into globally mixed and structured output via frequency-domain operations.

---

## 📦 Module Roles

### `rope.py` – ComplexRoPE
- **Role**: Adds rotary positional encoding using complex-valued log-scaled phase rotation.
- **Behavior**:
  - Converts real-valued input into complex format.
  - Applies log-frequency rotary embeddings to encode token position in phase.
  - Enables natural long-range positional generalization in the frequency domain.

---

### `nfo.py` – NeuralFourierOperator
- **Role**: Applies learnable modulation in the frequency domain.
- **Behavior**:
  - Multiplies each frequency vector by a learned complex-valued filter.
  - Modulates amplitude and phase across the token sequence globally.

---

### `fftnet_block.py` – FFTNetBlock
- **Role**: A single hybrid processing layer.
- **Behavior**:
  - Applies ComplexRoPE to input embeddings.
  - Performs FFT to move into frequency space.
  - Applies NeuralFourierOperator to modulate the spectrum.
  - Optionally applies inverse FFT to return to token space.
  - Follows with a feedforward MLP to introduce nonlinearity and projection.

---

### `model.py` – FFTNet
- **Role**: Full FFT-Net model stack.
- **Behavior**:
  - Embeds input token IDs into vector space.
  - Passes through multiple FFTNetBlocks.
  - Projects final output to token logits via linear head.

---

## 🎛️ Other Components

### `scripts/main.py`
- **Role**: Main execution script.
- **Behavior**:
  - Runs the model with synthetic or CLI-provided input.
  - Plots and saves the frequency spectrum of embeddings or intermediate stages.

---

## ⚙️ System Notes

- All modules respect a shared `device` (GPU or CPU).
- Mixed precision inference is enabled via `torch.cuda.amp.autocast()`.
- Project is modular and designed for extension (e.g., adding Wavelet or Attention blocks).

---

## 🧩 Interactions

```text
[input tokens]
    ↓
[embedding layer]
    ↓
[ComplexRoPE (rope.py)]
    ↓
[FFTNetBlock (fft_block.py)]
  ├─> [FFT + NeuralFourierOperator (nfo.py)]
  └─> [IFFT + MLP]
    ↓
[final output projection]
```
