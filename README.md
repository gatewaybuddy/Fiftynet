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
