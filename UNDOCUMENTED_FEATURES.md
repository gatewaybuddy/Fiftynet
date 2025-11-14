# Undocumented Features in Fiftynet

This document captures features that exist in the codebase but are not fully documented in the main README, FEATURES.md, or other documentation files.

## Training Features

### Mixed Precision Training
Both `train_fresh.py` and `train_distill.py` support mixed precision training via PyTorch's Automatic Mixed Precision (AMP).

**Usage:**
```bash
python scripts/train_fresh.py --data-path corpus.txt --mixed-precision
```

**Benefits:**
- Faster training on modern GPUs
- Reduced memory usage
- Maintained numerical stability via gradient scaling

**Implementation details:**
- Uses `torch.cuda.amp.GradScaler` for automatic gradient scaling
- Applies `autocast` context during forward passes
- Enabled via `--mixed-precision` flag (default: False)

### Early Stopping with Patience
Training scripts include early stopping to prevent overfitting.

**Configuration:**
```bash
python scripts/train_fresh.py --data-path corpus.txt --patience 3
```

**Behavior:**
- Monitors validation loss after each epoch
- Stops training if validation loss doesn't improve for N consecutive epochs
- Default patience: 3 epochs
- Saves best model checkpoint based on lowest validation loss

### JSONL Training Logs
All training runs generate detailed JSON Lines logs for analysis and visualization.

**Log location:** `logs/fresh_run.jsonl` or `logs/distill_run.jsonl`

**Log format:**
```json
{
  "step": 150,
  "epoch": 2,
  "loss": 0.4523,
  "accuracy": 0.8123,
  "timestamp": 1699564234.123
}
```

**Fields:**
- `step`: Global training step number
- `epoch`: Current epoch
- `loss`: Batch loss value
- `accuracy`: Batch accuracy
- `timestamp`: Unix timestamp for the log entry

### Reproducible Train/Val Splits
Control data splitting with a fixed seed for reproducibility.

**Usage:**
```bash
python scripts/train_fresh.py --data-path corpus.txt --split-seed 42
```

**Behavior:**
- Uses `torch.random_split` with specified seed
- Default split: 80% train, 20% validation
- Ensures identical splits across runs with same seed

### Configurable Batch Size and Learning Rate
Training scripts accept runtime hyperparameter configuration.

**Available options:**
```bash
python scripts/train_fresh.py \
  --data-path corpus.txt \
  --batch-size 32 \
  --lr 0.001 \
  --epochs 10
```

## Inference Features

### Multiple Inference Modes
`fftnet_infer.py` supports three distinct modes for different use cases.

#### 1. Text Generation Mode (default)
Generates and displays text continuation:
```bash
python fftnet_infer.py --model trained --prompt "the quick" --mode text
```

#### 2. Logits Inspection Mode
Shows top-k token predictions with scores:
```bash
python fftnet_infer.py --model trained --prompt "the quick" --mode logits --top-k 10
```

**Output example:**
```
brown: 2.4523
fox: 2.1234
dog: 1.9876
jumps: 1.5432
```

#### 3. Spectrum Visualization Mode
Displays frequency spectrum of embeddings:
```bash
python fftnet_infer.py --model trained --prompt "the quick" --mode spectrum
```

**Behavior:**
- Computes embeddings for generated sequence
- Applies FFT to visualize frequency content
- Saves plot as PNG or displays interactively

### Custom Tokenizer Support
Inference accepts custom trained tokenizers:
```bash
python fftnet_infer.py --tokenizer-path custom_tokenizer.json --prompt "hello"
```

### Configurable Generation Length
Control output length with `--max-new-tokens`:
```bash
python fftnet_infer.py --model trained --prompt "Once upon" --max-new-tokens 50
```

## Evaluation Features

### Teacher-Student Similarity Metrics
`scripts/evaluate.py` can compare model outputs to a teacher model (e.g., GPT-2).

**Usage:**
```bash
python scripts/evaluate.py \
  --model weights/trained \
  --data-path test.txt \
  --teacher-model gpt2
```

**Metrics computed:**
- Standard cross-entropy loss
- Token-level accuracy
- **Teacher similarity**: MSE between student and teacher logits
  - Lower values indicate closer alignment with teacher
  - Useful for evaluating distillation quality

### Spectrum Analysis in Evaluation
Evaluation script computes average frequency spectrum of model logits:

**Implementation:** `compute_spectrum()` in `evaluate.py`
- Processes batches from test dataset
- Applies FFT to logits along sequence dimension
- Averages magnitude across batches and vocab dimension
- Returns frequency-domain signature of model behavior

## Model Management Features

### Version Control for Models
`scripts/manage_models.py` provides a CLI for managing saved model versions.

**List all saved models:**
```bash
python scripts/manage_models.py --list
```

**Delete a model version:**
```bash
python scripts/manage_models.py --delete old_model
```

**Save a model version:**
```bash
python scripts/manage_models.py --save experiment_v1
```

**Load and inspect a model:**
```bash
python scripts/manage_models.py --load experiment_v1
```

**Storage format:**
- Models saved in `weights/` directory
- Each version has two files:
  - `{version}.safetensors`: Model weights
  - `{version}_config.json`: Model configuration

## Comparison and Analysis

### Training Run Comparison
`scripts/compare_runs.py` generates comparative visualizations.

**Basic comparison (loss curves only):**
```bash
python scripts/compare_runs.py \
  --fresh-log logs/fresh_run.jsonl \
  --distill-log logs/distill_run.jsonl
```

**Full comparison (with FFT spectrum analysis):**
```bash
python scripts/compare_runs.py \
  --fresh-log logs/fresh_run.jsonl \
  --distill-log logs/distill_run.jsonl \
  --fresh-model weights/fresh \
  --distill-model weights/distill \
  --data-path test.txt \
  --tokenizer-path tokenizer.json
```

**Output:**
- `training_comparison.png` with two subplots:
  1. **Training loss curves**: Step-by-step comparison
  2. **Logit FFT spectrum**: Frequency-domain behavior comparison

**Use cases:**
- Compare fresh training vs distillation
- Identify which approach converges faster
- Understand frequency-domain differences between models

### Logit Spectrum Analysis
Both `compare_runs.py` and `evaluate.py` include `average_logit_spectrum()`:

**What it does:**
- Samples batches from dataset
- Runs model to get logits
- Applies FFT along sequence dimension
- Averages magnitude across tokens and batches

**Why it matters:**
- Reveals which frequencies the model emphasizes
- Low frequencies → long-range patterns
- High frequencies → token-level patterns
- Can diagnose model behavior issues

## Visualization Features

### Agg Backend for Headless Environments
All visualization code uses Matplotlib's Agg backend:

```python
import matplotlib
matplotlib.use("Agg")
```

**Benefits:**
- Works on servers without display
- Compatible with SSH sessions
- Enables automated plotting in CI/CD

### Spectrum Plotting Utility
`fftnet/utils/visualization.py` provides `plot_embedding_spectrum()`:

**Features:**
- Plots average magnitude across tokens
- Supports both save and display modes
- Configurable for different embedding sources

**Usage in code:**
```python
from fftnet.utils.visualization import plot_embedding_spectrum

embeddings = model.embedding(tokens)
plot_embedding_spectrum(embeddings, save_path="spectrum.png")
```

## Benchmarking Infrastructure

### Performance Benchmarking
`benchmarks/` directory contains tools for measuring performance.

**Available benchmarks:**
- `benchmark_model.py`: Full model throughput and memory
- `benchmark_fftnet_block.py`: Individual block performance

**Benchmark utilities** (`benchmark_utils.py`):
- `benchmark_throughput()`: Measures iterations/second
- `benchmark_memory()`: Tracks peak memory usage
- Automatic CPU/GPU detection
- Warmup iterations for stable measurements

**Configuration:**
```python
# Example benchmark setup
model = FFTNet(vocab_size=5000, dim=64, num_blocks=2)
batch = torch.randint(0, 5000, (8, 128))
throughput = benchmark_throughput(model, batch, iterations=100)
memory = benchmark_memory(model, batch)
```

### GitHub Actions CI
`.github/workflows/benchmark.yml` runs benchmarks on every push/PR.

**What it does:**
- Runs all benchmark scripts
- Reports throughput and memory metrics
- Helps detect performance regressions
- Provides performance history

## Configuration System

### Dual Configuration Format
Models can be configured via both JSON and YAML:

**JSON** (`config/fiftynet_config.json`):
```json
{
  "vocab_size": 10,
  "dim": 4,
  "num_blocks": 2,
  "model_type": "fft"
}
```

**YAML** (`config/fiftynet_modules.yaml`):
```yaml
blocks:
  - type: FFTNetBlock
    name: block_0
  - type: FFTNetBlock
    name: block_1
```

### Block Registry Pattern
`fftnet/utils/config.py` includes a block registry for extensibility:

**Current blocks:**
- `FFTNetBlock`: Standard frequency-domain processing

**Extension mechanism:**
```python
# Future blocks can be added to registry:
BLOCK_REGISTRY = {
    'FFTNetBlock': FFTNetBlock,
    'WaveletBlock': WaveletBlock,  # Example
    'HybridAttentionBlock': HybridAttentionBlock,  # Example
}
```

**Benefits:**
- Easy to add new block types
- Configure architecture via YAML
- No code changes needed to swap blocks

## Data Loading

### TextFileDataset Features
`fftnet/data.py` provides sophisticated text loading:

**Features:**
- Loads plain text files of any size
- Tokenizes on-the-fly with SimpleTokenizer
- Creates sliding window sequences
- Returns `(input, target)` pairs for language modeling

**Configuration:**
```python
from fftnet.data import TextFileDataset

dataset = TextFileDataset(
    file_path="corpus.txt",
    tokenizer=tokenizer,
    seq_len=128  # Sequence length for training
)
```

**Behavior:**
- Reads entire file into memory
- Tokenizes once during initialization
- Creates overlapping sequences of length `seq_len`
- Target is input shifted by one token (next-token prediction)

## Tokenizer Features

### BPE Training from Text Iterators
`SimpleTokenizer` wraps HuggingFace tokenizers with custom training:

**Training from file:**
```python
from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer(vocab_size=5000)
tokenizer.train_from_iterator(
    text_iterator=open("corpus.txt"),
    vocab_size=5000
)
tokenizer.save("tokenizer.json")
```

**Serialization:**
- Saves to JSON format (HuggingFace format)
- Includes all BPE merge rules
- Portable across systems

### Vocabulary Size Control
Tokenizer vocab size is configurable and validated:
- Must be ≥ 256 (covers all bytes)
- Recommended: 1000-50000 depending on corpus size

## Storage Features

### Complex Parameter Handling
`fftnet/utils/storage.py` correctly handles complex-valued tensors:

**Challenge:**
- `safetensors` doesn't natively support complex dtypes
- FFTNet uses complex tensors in NeuralFourierOperator

**Solution:**
- Detects complex parameters during save
- Converts to real view: `tensor.view_as_real()`
- Stores metadata: `"_complex_meta": ["param1", "param2"]`
- Reconstructs complex view during load: `tensor.view_as_complex()`

**Example:**
```python
# NeuralFourierOperator.filters is complex-valued
save_model(model, "weights/trained", config)
loaded_model, cfg = load_model("weights/trained")
# loaded_model.filters is correctly restored as complex
```

### Config Bundling
Models are saved with their configuration:
- `{version}.safetensors`: Weights
- `{version}_config.json`: Complete model config

**Benefits:**
- Self-contained model artifacts
- No need to remember hyperparameters
- Easy to reproduce exact architecture

## Additional Command-Line Arguments

### Training Scripts
Undocumented flags in `train_fresh.py` and `train_distill.py`:

```bash
--seq-len 128           # Sequence length for training (default: 8)
--val-split 0.2         # Validation split ratio (default: 0.2)
--seed 42               # Global random seed for reproducibility
```

### Evaluation Script
Additional options in `evaluate.py`:

```bash
--batch-size 16         # Batch size for evaluation (default: 4)
--save-spectrum         # Save spectrum plot to file
```

### Compare Runs Script
Full argument list for `compare_runs.py`:

```bash
--seq-len 8             # Sequence length for spectrum analysis (default: 8)
--tokenizer-path PATH   # Custom tokenizer path (default: tokenizer.json)
```

## Summary

This document reveals **40+ undocumented features** across:
- **Training**: Mixed precision, early stopping, JSONL logs, reproducible splits
- **Inference**: Multiple modes, top-k logits, spectrum visualization
- **Evaluation**: Teacher similarity, spectrum analysis
- **Management**: Version control, comparison tools
- **Infrastructure**: Benchmarking, CI/CD, complex parameter handling

These features significantly enhance Fiftynet's capabilities beyond what's described in the main documentation.
