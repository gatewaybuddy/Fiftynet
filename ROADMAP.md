# FFTNet Roadmap

## Completed
- ComplexRoPE applies log-scaled rotary position encoding in the complex domain.
- NeuralFourierOperator learns a complex frequency filter for global token mixing.
- FFTNetBlock chains rotary encoding, FFT, learned filtering, inverse FFT, and an MLP.
- FFTNet stacks multiple blocks with token embeddings and a projection head.
- Scripts support training from scratch, GPT-2 distillation, evaluation, and model management.
- Utilities save and load weights using `safetensors` while preserving complex parameters.
- Unit tests cover core modules and visualizations.

## Next Steps
1. **Data & Tokenization** – replace the dummy dataset with real corpora and configurable tokenizers.
2. **Configuration** – expand JSON/YAML configs to allow flexible block types and hyperparameters.
3. **Training** – add mixed-precision support, checkpointing, and richer logging.
4. **Evaluation** – integrate standard benchmarks and teacher-student similarity metrics.
5. **Deployment** – package inference as a CLI/API and publish example pretrained weights.
6. **Extensions** – explore alternative spectral operators (e.g., wavelets) or attention hybrids.
