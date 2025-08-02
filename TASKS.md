# Project Tasks

## Implemented
- [x] ComplexRoPE for complex rotary embeddings.
- [x] NeuralFourierOperator for frequency-domain filtering.
- [x] FFTNetBlock combining FFT-based processing with an MLP and residual connection.
- [x] FFTNet model that stacks blocks with token embeddings and a projection head.
- [x] Training scripts for fresh models and GPT-2 distillation.
- [x] Model persistence via `safetensors` with complex tensor handling.
- [x] Evaluation, visualization, and model management utilities.
- [x] Test suite covering core components and visualizations.

## TODO
- [ ] Replace `DummyWikiDataset` with modular data loaders and real corpora.
- [ ] Provide tokenizer and vocabulary management utilities.
- [ ] Extend configuration loading to parse block definitions from YAML.
- [ ] Enhance training loops with mixed precision, checkpointing, and early stopping.
- [ ] Package inference as a CLI/API and distribute example pretrained weights.
- [ ] Add benchmarks and performance regression tests.
- [ ] Expand README with usage examples and architecture details.
