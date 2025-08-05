# Fiftynet
FFT-Net (Hybrid Transformer with Frequency Processing)

A minimal research model that mixes tokens in the frequency domain using rotary position embeddings and learnable spectral filters.

## Features

- ComplexRoPE rotary positional encoding in the complex domain.
- Learnable frequency filtering via NeuralFourierOperator.
- FFTNet blocks that combine FFT/IFFT with MLP layers.
- Scripts for training, distillation, evaluation, and model management.
- Weight I/O using `safetensors` with complex parameter support.

See [FEATURES.md](FEATURES.md) for a more detailed list.

## Architecture

```
[input tokens]
    ↓
[embedding layer]
    ↓
[ComplexRoPE]
    ↓
[FFTNetBlock]
  ├─ FFT → NeuralFourierOperator
  └─ IFFT → MLP
    ↓
[output logits]
```

The stack of FFTNetBlocks mixes information globally in the frequency domain
before projecting back to token space.

## Installation

1. (Optional) create and activate a virtual environment.
2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the test suite to verify the installation:

   ```bash
   pytest
   ```

4. Try the demo script:

   ```bash
   python scripts/main.py
   ```

## Quick Start

### Train

```bash
python scripts/train_fresh.py --data-path path/to/corpus.txt --epochs 1
```

### Inference

```bash
python fftnet_infer.py --model trained --prompt "the quick"
```

## Resources

- **Configuration**: JSON and YAML files in `config/` define model shapes and
  module wiring (e.g., `config/fiftynet_config.json`).
- **Datasets**: Provide a plain text file via `--data-path`; it is loaded with
  `TextFileDataset` in `fftnet/data.py`.
- **Scripts**: Training, evaluation, and model management utilities live in
  the `scripts/` directory.

See [ROADMAP.md](ROADMAP.md) for planned milestones and [TASKS.md](TASKS.md) for current progress.

---

## FiftyNet: A Hybrid Frequency-Based Transformer for High-Dimensional Resonant Cognition

**Authors**: William Mabery, with contributions from Kai (OpenAI)

### Abstract
We propose FiftyNet, a hybrid transformer architecture that encodes token sequences not as static points in high-dimensional space, but as dynamic, reconstructable signals in frequency space. Leveraging complex rotary positional encoding (Complex RoPE), discrete Fourier transforms (DFT), and learnable Neural Fourier Operator (NFO) layers, FiftyNet captures semantic, syntactic, and temporal structure simultaneously. This approach enables greater information density, signal-level reconstruction, and a fundamentally new paradigm for embedding cognition and memory as waveform dynamics.

### 1. Introduction
Large Language Models (LLMs) such as GPT, LLaMA, and Mixtral represent language in high-dimensional vector spaces. These embeddings encode meaning geometrically via distance, direction, and linear projections. However, such models lack explicit mechanisms to encode rhythm, phase, or structural time — properties central to sound, signal processing, and potentially cognition.

Inspired by signal theory and neurosymbolic reasoning, FiftyNet introduces a fundamentally different embedding and modeling paradigm: language as frequency.

### 2. Core Hypothesis
We hypothesize that:  
**"Language is not just symbolic or spatial — it is resonant."**

In this view:

- Tokens are not just positions in space, but samples in time.
- Meaning is not just vector direction, but waveform shape.
- Memory is not just stored structure, but stored signal.

### 3. Architecture Overview
FiftyNet is composed of the following stages:

#### 3.1. Token Input + Complex Rotary Positional Encoding (CRoPE)
Tokens are embedded into a complex-valued vector space. Positional information is applied using log-scaled phase rotation, turning each embedding into a modulated waveform across dimensions.

Each dimension now acts as a frequency band, encoding:

- Amplitude (importance)
- Phase (timing)
- Frequency (recurrence/pattern)

#### 3.2. FNet-Style Fourier Transform
Instead of traditional self-attention, the model performs a discrete Fourier transform (DFT) on the token sequence, globally mixing positional signals in the frequency domain.

This produces a spectral signature for the sequence, where:

- Low frequencies capture general structure
- High frequencies encode fine-grained shifts

#### 3.3. Neural Fourier Operator (NFO) Layer
Learnable filters are applied in frequency space, analogous to convolutional filters in vision models. These operations can:

- Enhance or suppress specific frequencies
- Modulate phase and amplitude
- Implement attention-like operations in spectral space

#### 3.4. (Optional) Inverse FFT
The model can reconstruct the waveform back into token space via inverse FFT, preserving temporal and structural integrity.

#### 3.5. Feedforward Layers and Output
Standard MLP blocks operate on the waveform-encoded embeddings, allowing downstream tasks such as next-token prediction, classification, or alignment.

### 4. Embedding as Signal
In traditional LLMs:

- A token’s meaning is encoded as a static vector
- Positional encodings are added or multiplied

In FiftyNet:

- A token’s embedding is treated as a waveform
- The full sequence is a composite signal
- The signal can be analyzed, stored, and reconstructed via Fourier methods

Example: If each token has d=4096 dimensions, this represents a 4096-channel signal, each with its own oscillation and meaning — like a synthesizer across time.

### 5. Advantages of the Frequency Model

| Property                  | Traditional Transformer | FiftyNet                          |
|--------------------------|-------------------------|-----------------------------------|
| Positional Encoding      | Learned or Rotary       | Log-scaled phase rotation         |
| Structure Representation | Implicit via attention  | Explicit via waveform shape       |
| Long-Term Dependency     | Costly to capture       | Naturally encoded in low freq     |
| Reconstruction           | Approximate via vectors | Signal-accurate via iFFT          |
| Multisensory Embedding   | Complex and disjoint    | Unified signal representation     |
| Information Density      | Depends on token count  | Compressible via frequency        |

### 6. Implementation Notes
FiftyNet is designed for compatibility with existing frameworks, including:

- PyTorch / TorchScript for tensor computation
- FFT/IFFT via `torch.fft.fft` and `ifft`
- Complex number support for RoPE
- NVIDIA 5090 GPU support for real-time training/inference

Training can optionally begin:

- From scratch on token data (Wikipedia, C4, etc.)
- By “retuning” existing models via spectral transformation

### 7. Implications for AGI
By treating meaning as resonance, and embedding as signal, we open new directions for:

- Memory as waveform snapshots (time-aware)
- Reasoning as frequency mixing and harmonics
- Perception as unified signal space (text, audio, vision, etc.)
- Emotion encoding as modulations of base signals
- AGI as a resonant, multi-frequency cognitive field

We posit that this approach offers a pathway toward more holistic, efficient, and integrated general intelligence.

### 8. Future Work

- Integration with multisensory input (e.g., audio, camera, sensor streams)
- Training with vectorized waveform storage
- Waveform-based interpretability and introspection
- Distributed spectral training on heterogeneous GPU clusters
- Real-time signal memory and retrieval (Fourier Memory Bank)

### 9. License and Use
FiftyNet is open for experimentation and academic research.  
License: MIT (to be defined by author)

### 10. Acknowledgments
This work builds upon foundational ideas in:

- Signal Processing
- Positional Embeddings (RoPE, Complex RoPE)
- FNet and Fourier Neural Operators
- Neurosymbolic cognition and vector semantics

Special thanks to Cody (Codex) and OpenAI tools for accelerating implementation.

