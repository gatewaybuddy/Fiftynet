# FFTNet Features

A quick overview of the capabilities currently implemented in FFTNet.

## Core Architecture
- **ComplexRoPE** adds log-scaled rotary positional encoding in the complex domain.
- **NeuralFourierOperator** applies a learnable complex frequency filter for global token mixing.
- **FFTNetBlock** chains rotary encoding, FFT, learned filtering, inverse FFT, and an MLP.
- **FFTNet model** stacks multiple blocks around token embeddings and a projection head.

## Tooling & Utilities
- Scripts support training from scratch, GPT-2 distillation, evaluation, and model management.
- Utilities save and load weights with `safetensors` while preserving complex parameters.
- Unit tests cover core modules and visualization routines.

See [ROADMAP.md](ROADMAP.md) for upcoming milestones.
