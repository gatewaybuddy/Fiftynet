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

See [ROADMAP.md](ROADMAP.md) for planned milestones and [TASKS.md](TASKS.md) for current progress.
