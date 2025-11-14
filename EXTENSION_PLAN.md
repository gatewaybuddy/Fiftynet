# Fiftynet Extension Plan

This document outlines a comprehensive plan for extending Fiftynet's capabilities, organized by priority and area.

## Priority Levels
- **P0**: Critical for production readiness
- **P1**: High value, significantly enhances capabilities
- **P2**: Nice to have, research exploration
- **P3**: Future research directions

---

## 1. Architecture Extensions

### P1: Wavelet Transform Blocks
**Goal:** Provide alternative spectral representations beyond FFT.

**Rationale:**
- Wavelets capture localized frequency information (vs FFT's global view)
- Better for sequences with non-stationary patterns
- Can handle varying sequence lengths more naturally

**Implementation:**
```python
class WaveletBlock(nn.Module):
    """Multi-scale wavelet decomposition block"""
    def __init__(self, dim: int, wavelet_type: str = 'db4', levels: int = 3):
        # Discrete Wavelet Transform with learnable coefficients
        # Apply different processing to approximation vs detail coefficients
```

**Tasks:**
- [ ] Implement `WaveletBlock` using PyWavelets or custom implementation
- [ ] Add learnable coefficient modulation (like NFO for wavelets)
- [ ] Register in block registry
- [ ] Create config YAML for wavelet-based models
- [ ] Benchmark against FFT-based blocks
- [ ] Add tests: `tests/test_wavelet_block.py`

**Estimated effort:** 2-3 weeks

---

### P1: Hybrid Attention-Frequency Blocks
**Goal:** Combine attention mechanisms with frequency processing.

**Rationale:**
- Attention captures semantic relationships
- Frequency processing captures structural patterns
- Hybrid approach may get best of both worlds

**Architecture:**
```
Input → ComplexRoPE → Parallel:
                       ├─ FFT → NFO → IFFT
                       └─ Multi-Head Attention
                     → Concatenate → MLP → Output
```

**Implementation:**
```python
class HybridAttentionFFTBlock(nn.Module):
    """Combines frequency-domain and attention-based mixing"""
    def __init__(self, dim: int, num_heads: int = 4, use_fft: bool = True):
        # Parallel frequency and attention pathways
        # Learnable gate to balance contributions
```

**Tasks:**
- [ ] Implement `HybridAttentionFFTBlock`
- [ ] Add learnable gating mechanism to balance pathways
- [ ] Experiment with different fusion strategies (concat, add, gate)
- [ ] Compare memory/compute tradeoffs
- [ ] Add configuration support
- [ ] Add tests and benchmarks

**Estimated effort:** 2-3 weeks

---

### P2: Multi-Scale Frequency Processing
**Goal:** Process different frequency bands with different depths/parameters.

**Rationale:**
- Low frequencies (global structure) may need different processing than high frequencies (local details)
- Inspired by multi-scale vision architectures (U-Net, FPN)

**Architecture:**
```
Input → FFT → Split into frequency bands
              ├─ Low freq (0-25%) → Deep processing
              ├─ Mid freq (25-75%) → Medium processing
              └─ High freq (75-100%) → Light processing
            → Merge → IFFT → Output
```

**Tasks:**
- [ ] Implement frequency band splitting
- [ ] Create band-specific processing modules
- [ ] Design learnable band allocation
- [ ] Add cross-band interaction mechanisms
- [ ] Benchmark on long sequences

**Estimated effort:** 3-4 weeks

---

### P2: Learnable Frequency Selection
**Goal:** Let model learn which frequencies to emphasize.

**Implementation:**
```python
class AdaptiveFrequencyFilter(nn.Module):
    """Learns to select important frequency bands"""
    def __init__(self, seq_len: int, top_k: int):
        self.importance_scores = nn.Parameter(torch.randn(seq_len))
        self.top_k = top_k

    def forward(self, x_freq: torch.Tensor):
        # Select top-k frequencies based on learned scores
        # Apply processing only to selected frequencies
```

**Tasks:**
- [ ] Implement adaptive frequency selection
- [ ] Add sparsity regularization
- [ ] Visualize learned frequency importance
- [ ] Compare with full-spectrum processing

**Estimated effort:** 2 weeks

---

## 2. Training Enhancements

### P0: Checkpointing and Resume
**Goal:** Save and resume training from checkpoints.

**Current gap:** Training stops can't be resumed; progress is lost.

**Features:**
- Save optimizer state, scaler state, epoch, step
- Resume from latest checkpoint
- Configurable checkpoint frequency
- Automatic cleanup of old checkpoints

**Implementation:**
```bash
python scripts/train_fresh.py \
  --data-path corpus.txt \
  --checkpoint-dir checkpoints/exp1 \
  --checkpoint-every 1000 \
  --resume-from checkpoints/exp1/latest.pt
```

**Tasks:**
- [ ] Add checkpoint saving in training loop
- [ ] Implement `--resume-from` argument
- [ ] Save/restore optimizer and scaler state
- [ ] Add checkpoint rotation (keep last N)
- [ ] Test resume produces identical results
- [ ] Update documentation

**Estimated effort:** 1 week

---

### P0: Learning Rate Scheduling
**Goal:** Improve convergence with adaptive learning rates.

**Strategies:**
- Warmup: Linear increase for first N steps
- Cosine annealing: Smooth decay
- Plateau: Reduce on validation loss plateau
- One-cycle: Warmup then cosine decay

**Implementation:**
```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=args.lr,
    total_steps=num_steps,
    pct_start=0.1  # 10% warmup
)
```

**Tasks:**
- [ ] Add scheduler options: `--scheduler {warmup,cosine,plateau,onecycle}`
- [ ] Log learning rate in JSONL
- [ ] Plot LR curves in comparison script
- [ ] Compare convergence with/without scheduling
- [ ] Update training scripts

**Estimated effort:** 3-4 days

---

### P1: Gradient Clipping
**Goal:** Prevent gradient explosions during training.

**Implementation:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Tasks:**
- [ ] Add `--grad-clip` argument (default: 1.0)
- [ ] Log gradient norms before/after clipping
- [ ] Add gradient norm visualization
- [ ] Test on unstable training scenarios

**Estimated effort:** 2 days

---

### P1: Curriculum Learning
**Goal:** Train on progressively longer sequences.

**Rationale:**
- Easier to learn short-range patterns first
- Gradually introduce long-range dependencies
- May improve convergence and generalization

**Implementation:**
```python
# Start: seq_len=16 for 1000 steps
# Then: seq_len=32 for 1000 steps
# Then: seq_len=64 for 1000 steps
# Finally: seq_len=128 until convergence
```

**Tasks:**
- [ ] Implement sequence length scheduling
- [ ] Add `--curriculum` config file support
- [ ] Dynamically adjust dataset sequence length
- [ ] Track metrics per curriculum stage
- [ ] Compare with fixed-length training

**Estimated effort:** 1 week

---

### P2: Distributed Training
**Goal:** Multi-GPU and multi-node training.

**Technologies:**
- PyTorch DDP (DistributedDataParallel)
- DeepSpeed for large-scale training
- Horovod for multi-node

**Implementation:**
```bash
torchrun --nproc_per_node=4 scripts/train_fresh.py \
  --data-path corpus.txt \
  --distributed
```

**Tasks:**
- [ ] Add DDP wrapper around training
- [ ] Implement distributed data loading
- [ ] Add gradient synchronization
- [ ] Test on multi-GPU setup
- [ ] Add DeepSpeed integration (optional)
- [ ] Document distributed training setup

**Estimated effort:** 2-3 weeks

---

## 3. Inference Improvements

### P0: Sampling Strategies
**Goal:** Replace greedy decoding with better sampling methods.

**Methods:**
- **Nucleus (top-p)**: Sample from top cumulative probability p
- **Top-k**: Sample from top k tokens
- **Temperature**: Scale logits before sampling

**Implementation:**
```python
def sample_token(logits: torch.Tensor, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0):
    logits = logits / temperature
    if top_k > 0:
        # Keep only top k
    if top_p < 1.0:
        # Nucleus sampling
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
```

**Tasks:**
- [ ] Implement sampling utilities
- [ ] Add CLI arguments: `--temperature`, `--top-k`, `--top-p`
- [ ] Update `generate()` function
- [ ] Compare quality of different strategies
- [ ] Add tests

**Estimated effort:** 3-4 days

---

### P1: Beam Search
**Goal:** Higher quality generation via beam search.

**Implementation:**
```python
def beam_search(model, input_ids, beam_width=5, max_length=100):
    # Maintain top-k hypotheses
    # Expand and score each hypothesis
    # Return highest-scoring complete sequence
```

**Tasks:**
- [ ] Implement beam search decoder
- [ ] Add `--beam-width` argument
- [ ] Handle batch beam search
- [ ] Add length normalization
- [ ] Compare with greedy/sampling

**Estimated effort:** 1 week

---

### P1: Batch Inference
**Goal:** Process multiple prompts simultaneously.

**Current gap:** Inference only handles single prompts.

**Implementation:**
```bash
python fftnet_infer.py --model trained --batch-prompts prompts.txt
```

**Tasks:**
- [ ] Support batch input processing
- [ ] Implement dynamic padding
- [ ] Add batch size control
- [ ] Measure throughput improvement
- [ ] Update CLI interface

**Estimated effort:** 3 days

---

### P2: Streaming Generation
**Goal:** Stream tokens as they're generated (like ChatGPT).

**Use case:** Real-time applications, chatbots

**Implementation:**
```python
def generate_streaming(model, input_ids, max_tokens=100):
    for i in range(max_tokens):
        next_token = model.generate_one(input_ids)
        yield next_token
        input_ids = torch.cat([input_ids, next_token], dim=1)
```

**Tasks:**
- [ ] Implement streaming generator
- [ ] Add callback support for token processing
- [ ] Create demo with real-time display
- [ ] Measure latency per token

**Estimated effort:** 2-3 days

---

## 4. Evaluation & Analysis

### P0: Standard Benchmarks
**Goal:** Evaluate on established datasets.

**Benchmarks:**
- **WikiText-2/103**: Language modeling
- **Penn Treebank**: Classic LM benchmark
- **lambada**: Reading comprehension
- **HellaSwag**: Commonsense reasoning

**Implementation:**
```bash
python scripts/benchmark.py \
  --model weights/trained \
  --dataset wikitext-2 \
  --metric perplexity
```

**Tasks:**
- [ ] Create `scripts/benchmark.py`
- [ ] Integrate HuggingFace datasets
- [ ] Implement perplexity calculation
- [ ] Add multiple-choice accuracy for HellaSwag
- [ ] Generate comparison tables
- [ ] Document baseline results

**Estimated effort:** 1-2 weeks

---

### P0: Perplexity Metrics
**Goal:** Standard language modeling metric.

**Current gap:** Only track loss and accuracy.

**Implementation:**
```python
def compute_perplexity(model, dataset):
    total_loss = 0.0
    total_tokens = 0
    for batch in dataset:
        loss = model.compute_loss(batch)
        total_loss += loss * batch.numel()
        total_tokens += batch.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
```

**Tasks:**
- [ ] Add perplexity calculation to evaluation
- [ ] Report in training logs
- [ ] Add to comparison plots
- [ ] Document interpretation

**Estimated effort:** 2 days

---

### P1: Generation Quality Metrics
**Goal:** Measure generation quality beyond perplexity.

**Metrics:**
- **BLEU**: N-gram overlap with references
- **ROUGE**: Recall-oriented overlap
- **METEOR**: Semantic similarity
- **Self-BLEU**: Diversity measure

**Tasks:**
- [ ] Integrate `sacrebleu` or `evaluate` library
- [ ] Create generation evaluation script
- [ ] Collect reference completions
- [ ] Generate automated reports
- [ ] Compare sampling strategies

**Estimated effort:** 1 week

---

### P1: Frequency Analysis Tools
**Goal:** Deep insights into frequency-domain behavior.

**Visualizations:**
- Frequency spectrum heatmaps across layers
- Phase distribution plots
- Frequency band importance scores
- Temporal evolution of spectrum during training

**Implementation:**
```bash
python scripts/analyze_frequencies.py \
  --model weights/trained \
  --data-path test.txt \
  --output-dir analysis/
```

**Features:**
- Per-layer spectrum analysis
- Attention-like frequency attribution
- Frequency band ablation studies
- Interactive visualizations (Plotly)

**Tasks:**
- [ ] Create comprehensive analysis script
- [ ] Implement layer-wise spectrum extraction
- [ ] Add frequency ablation experiments
- [ ] Generate HTML reports
- [ ] Document interpretations

**Estimated effort:** 2 weeks

---

### P2: Interpretability Tools
**Goal:** Understand what the model learns.

**Tools:**
- Token attribution in frequency space
- Frequency band ablation impact
- Visualization of learned NFO filters
- Phase vs amplitude contribution analysis

**Tasks:**
- [ ] Implement frequency-domain attribution
- [ ] Create interactive visualizations
- [ ] Add filter visualization utilities
- [ ] Generate interpretation guides

**Estimated effort:** 2-3 weeks

---

## 5. Deployment & Packaging

### P0: REST API
**Goal:** Serve model via HTTP API.

**Technology:** FastAPI

**Implementation:**
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 50):
    tokens = model.generate(prompt, max_tokens)
    return {"text": tokenizer.decode(tokens)}
```

**Endpoints:**
- `POST /generate`: Text generation
- `POST /logits`: Get top-k predictions
- `GET /health`: Health check
- `GET /model-info`: Model metadata

**Tasks:**
- [ ] Create `api/server.py`
- [ ] Add request validation (Pydantic)
- [ ] Implement batch queuing
- [ ] Add rate limiting
- [ ] Create OpenAPI documentation
- [ ] Add deployment guide (Uvicorn, Gunicorn)

**Estimated effort:** 1 week

---

### P1: Docker Containerization
**Goal:** Easy deployment with Docker.

**Dockerfile:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0"]
```

**Tasks:**
- [ ] Create production Dockerfile
- [ ] Add docker-compose.yml
- [ ] Optimize image size (multi-stage build)
- [ ] Add GPU support (nvidia-docker)
- [ ] Document deployment
- [ ] Push to Docker Hub

**Estimated effort:** 3-4 days

---

### P1: ONNX Export
**Goal:** Export for inference optimization.

**Benefits:**
- Cross-platform deployment
- Potential speed improvements
- Integration with ONNX Runtime

**Implementation:**
```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=14
)
```

**Challenges:**
- Complex-valued operations may not have ONNX equivalents
- May need to rewrite using real-valued representations

**Tasks:**
- [ ] Test ONNX export compatibility
- [ ] Handle complex operations
- [ ] Validate exported model
- [ ] Benchmark ONNX Runtime performance
- [ ] Document export process

**Estimated effort:** 1-2 weeks

---

### P2: Model Quantization
**Goal:** Reduce model size and increase inference speed.

**Methods:**
- Dynamic quantization (INT8)
- Static quantization (calibration)
- Quantization-aware training

**Implementation:**
```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

**Tasks:**
- [ ] Test quantization compatibility with complex ops
- [ ] Implement quantization pipeline
- [ ] Measure accuracy vs size tradeoff
- [ ] Document quantization guide

**Estimated effort:** 1-2 weeks

---

### P2: PyPI Package
**Goal:** `pip install fiftynet`

**Structure:**
```
fiftynet/
  __init__.py
  model.py
  blocks/
    fft_block.py
    wavelet_block.py
  utils/
  cli/
setup.py
pyproject.toml
```

**Tasks:**
- [ ] Reorganize for package structure
- [ ] Create setup.py and pyproject.toml
- [ ] Add CLI entry points
- [ ] Write installation documentation
- [ ] Publish to PyPI
- [ ] Add version management

**Estimated effort:** 1 week

---

## 6. Data & Preprocessing

### P1: Multi-File Dataset Support
**Goal:** Train on multiple corpus files.

**Implementation:**
```bash
python scripts/train_fresh.py \
  --data-dir corpora/ \
  --file-pattern "*.txt"
```

**Features:**
- Concatenate or interleave files
- Weighted sampling across files
- Streaming from disk (for large corpora)

**Tasks:**
- [ ] Implement `MultiFileDataset`
- [ ] Add file discovery and loading
- [ ] Support weighted sampling
- [ ] Add streaming option
- [ ] Update training scripts

**Estimated effort:** 1 week

---

### P1: Streaming Data Loaders
**Goal:** Handle datasets larger than RAM.

**Technology:** HuggingFace `datasets` library with streaming

**Implementation:**
```python
from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.en", streaming=True)
```

**Tasks:**
- [ ] Integrate streaming datasets
- [ ] Add on-the-fly tokenization
- [ ] Implement shuffling for streaming
- [ ] Test on large corpora (C4, Pile)
- [ ] Document streaming setup

**Estimated effort:** 1 week

---

### P2: Data Augmentation
**Goal:** Increase training data diversity.

**Techniques:**
- Token dropout
- Random substitution with synonyms
- Back-translation
- Sentence shuffling

**Tasks:**
- [ ] Implement augmentation transforms
- [ ] Add to data pipeline
- [ ] Measure impact on generalization
- [ ] Make configurable

**Estimated effort:** 1-2 weeks

---

### P2: Multilingual Tokenizers
**Goal:** Support non-English languages.

**Implementation:**
- Use SentencePiece for subword tokenization
- Train on multilingual corpora
- Handle Unicode correctly

**Tasks:**
- [ ] Integrate SentencePiece
- [ ] Create multilingual training scripts
- [ ] Test on diverse languages
- [ ] Document usage

**Estimated effort:** 1 week

---

## 7. Research Extensions

### P2: Fourier Memory Bank
**Goal:** Store and retrieve waveform memories.

**Concept:**
- Store past sequences as frequency signatures
- Retrieve similar patterns via frequency matching
- Augment generation with retrieved context

**Architecture:**
```python
class FourierMemoryBank:
    def store(self, sequence: torch.Tensor, spectrum: torch.Tensor):
        # Store (sequence, spectrum) pairs

    def retrieve(self, query_spectrum: torch.Tensor, top_k: int = 5):
        # Find most similar spectra
        # Return corresponding sequences
```

**Tasks:**
- [ ] Implement memory bank with indexing
- [ ] Add similarity search (cosine, L2 in freq domain)
- [ ] Integrate with generation
- [ ] Benchmark retrieval quality
- [ ] Test on long-context tasks

**Estimated effort:** 3-4 weeks

---

### P3: Cross-Modal Frequency Encoding
**Goal:** Unified frequency representation for text, audio, images.

**Vision:**
- Text → token embeddings → frequency domain
- Audio → raw waveform → frequency domain
- Images → pixel patches → frequency domain
- Joint processing in shared frequency space

**Potential:**
- Multimodal understanding
- Cross-modal generation (text→audio, etc.)
- Unified AGI architecture

**Tasks:**
- [ ] Research cross-modal frequency alignment
- [ ] Implement audio encoder
- [ ] Implement vision encoder
- [ ] Design joint training objective
- [ ] Prototype multimodal model

**Estimated effort:** 2-3 months (research project)

---

### P3: Adaptive Frequency Resolution
**Goal:** Dynamically adjust frequency granularity.

**Concept:**
- Use coarse frequency resolution for simple patterns
- Use fine resolution for complex patterns
- Learn when to refine resolution

**Implementation:**
- Multi-resolution FFT (different window sizes)
- Learnable resolution selection
- Adaptive computational budget

**Tasks:**
- [ ] Design multi-resolution architecture
- [ ] Implement resolution selection mechanism
- [ ] Add computational efficiency metrics
- [ ] Test on varying complexity tasks

**Estimated effort:** 1-2 months (research project)

---

### P3: Phase-Aware Loss Functions
**Goal:** Directly optimize phase information.

**Current gap:** Cross-entropy only optimizes magnitude (logits)

**Approach:**
- Add loss term for phase alignment
- Encourage phase coherence across sequence
- Penalize phase jumps

**Tasks:**
- [ ] Design phase-aware loss
- [ ] Implement phase regularization
- [ ] Test impact on generation quality
- [ ] Analyze phase patterns in trained models

**Estimated effort:** 3-4 weeks

---

## Implementation Roadmap

### Phase 1: Production Readiness (1-2 months)
**Priority:** Get core features ready for real use

- [P0] Checkpointing and resume
- [P0] Learning rate scheduling
- [P0] Standard benchmarks
- [P0] Perplexity metrics
- [P0] Sampling strategies
- [P0] REST API

**Outcome:** Fiftynet ready for production experiments

---

### Phase 2: Architecture Expansion (2-3 months)
**Priority:** Explore architectural variants

- [P1] Wavelet blocks
- [P1] Hybrid attention-frequency blocks
- [P1] Gradient clipping
- [P1] Curriculum learning
- [P1] Beam search
- [P1] Generation quality metrics

**Outcome:** Multiple architecture options validated

---

### Phase 3: Deployment & Scale (1-2 months)
**Priority:** Enable large-scale deployment

- [P1] Docker containerization
- [P1] ONNX export
- [P1] Multi-file datasets
- [P1] Streaming data loaders
- [P2] Distributed training
- [P2] PyPI package

**Outcome:** Scalable, deployable system

---

### Phase 4: Research Exploration (3-6 months)
**Priority:** Push research boundaries

- [P2] Multi-scale frequency processing
- [P2] Fourier memory bank
- [P2] Frequency analysis tools
- [P2] Interpretability tools
- [P3] Cross-modal encoding
- [P3] Adaptive resolution
- [P3] Phase-aware losses

**Outcome:** Novel research contributions

---

## Success Metrics

### Technical Metrics
- **Perplexity**: < 30 on WikiText-2 (competitive with small transformers)
- **Inference speed**: > 100 tokens/sec on consumer GPU
- **Memory efficiency**: Train 100M param model on single 16GB GPU
- **Test coverage**: > 90% code coverage

### Research Metrics
- **Novel insights**: 2-3 papers on frequency-domain language modeling
- **Architectural variants**: 3-5 validated alternatives to pure FFT
- **Interpretability**: Clear explanations of learned frequency patterns

### Deployment Metrics
- **API latency**: < 100ms per request
- **Package adoption**: 100+ PyPI downloads/month
- **Documentation quality**: All features documented with examples

---

## Summary

This extension plan covers **60+ specific tasks** across 7 major areas:

1. **Architecture**: 4 new block types, multi-scale processing
2. **Training**: 6 enhancements for better convergence and scale
3. **Inference**: 4 improvements for quality and speed
4. **Evaluation**: 5 new metrics and analysis tools
5. **Deployment**: 5 packaging and serving solutions
6. **Data**: 4 preprocessing and loading improvements
7. **Research**: 4 novel research directions

**Total estimated effort:** 8-12 months for complete implementation

**Recommended starting point:** Phase 1 (Production Readiness) to establish solid foundation.
