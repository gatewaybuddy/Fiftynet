# FFTNet Roadmap

## Completed
- ComplexRoPE applies log-scaled rotary position encoding in the complex domain.
- NeuralFourierOperator learns a complex frequency filter for global token mixing.
- FFTNetBlock chains rotary encoding, FFT, learned filtering, inverse FFT, and an MLP.
- FFTNet stacks multiple blocks with token embeddings and a projection head.
- Scripts support training from scratch, GPT-2 distillation, evaluation, and model management.
- Utilities save and load weights using `safetensors` while preserving complex parameters.
- Unit tests cover core modules and visualizations.
- Data & tokenization: replaced the dummy dataset with real corpora and configurable tokenizers.
- Configuration: expanded JSON/YAML configs to allow flexible block types and hyperparameters.

## Next Steps
1. **Training** – add mixed-precision support, checkpointing, and richer logging.
2. **Evaluation** – integrate standard benchmarks and teacher-student similarity metrics.
3. **Deployment** – package inference as a CLI/API and publish example pretrained weights.
4. **Extensions** – explore alternative spectral operators (e.g., wavelets) or attention hybrids.
