# Fiftynet Comprehensive Testing Plan

This document outlines a systematic testing strategy to ensure Fiftynet's correctness, robustness, and performance.

## Current Testing Status

### Existing Tests (10 files, ~500 lines)
- ✅ `test_model.py`: Basic model forward pass
- ✅ `test_fftnet_block.py`: Block forward pass
- ✅ `test_complex_rope.py`: Position encoding
- ✅ `test_neural_fourier_operator.py`: Frequency filtering
- ✅ `test_infer.py`: CLI and generation
- ✅ `test_tokenizer.py`: Tokenization
- ✅ `test_text_dataset.py`: Data loading
- ✅ `test_storage.py`: Model save/load
- ✅ `test_config.py`: Configuration loading
- ✅ `test_visualization.py`: Plotting utilities

### Coverage Gaps
- ❌ Training scripts (train_fresh.py, train_distill.py)
- ❌ Evaluation script
- ❌ Comparison and analysis scripts
- ❌ Model management utilities
- ❌ Edge cases (empty inputs, very long sequences)
- ❌ Numerical stability tests
- ❌ Performance regression tests
- ❌ Integration tests (full training → inference workflow)

---

## Testing Strategy

### Test Pyramid

```
        /\
       /  \     E2E Tests (10%)
      /____\
     /      \   Integration Tests (30%)
    /________\
   /          \ Unit Tests (60%)
  /__________\
```

**Philosophy:**
- **60% Unit tests**: Fast, isolated, comprehensive coverage
- **30% Integration tests**: Component interactions
- **10% E2E tests**: Full workflows, slower but realistic

---

## 1. Unit Testing Expansion

### 1.1 Core Architecture Tests

#### ComplexRoPE (`tests/test_complex_rope.py`)
**Current coverage:** Basic shape and dtype

**Additional tests needed:**
```python
def test_rope_position_independence():
    """Different positions should produce different encodings"""

def test_rope_log_scaling():
    """Verify log-scaled frequency distribution"""

def test_rope_invertibility():
    """Check if position info can be recovered"""

def test_rope_numerical_stability():
    """Test with extreme position indices"""

def test_rope_gradient_flow():
    """Ensure gradients propagate correctly"""
```

**Estimated tests to add:** 5-6

---

#### NeuralFourierOperator (`tests/test_neural_fourier_operator.py`)
**Current coverage:** Basic filtering

**Additional tests needed:**
```python
def test_nfo_complex_multiplication():
    """Verify complex-valued weight application"""

def test_nfo_frequency_selectivity():
    """Test learned filters can select frequencies"""

def test_nfo_amplitude_modulation():
    """Check amplitude scaling works correctly"""

def test_nfo_phase_modulation():
    """Check phase rotation works correctly"""

def test_nfo_parameter_initialization():
    """Verify weights initialized correctly"""

def test_nfo_gradient_magnitude():
    """Check gradients aren't too large/small"""
```

**Estimated tests to add:** 6-8

---

#### FFTNetBlock (`tests/test_fftnet_block.py`)
**Current coverage:** Forward pass shape

**Additional tests needed:**
```python
def test_block_residual_connections():
    """Verify residual path works correctly"""

def test_block_fft_invertibility():
    """FFT → IFFT should approximately recover input"""

def test_block_energy_preservation():
    """Check Parseval's theorem (energy conservation)"""

def test_block_sequence_length_flexibility():
    """Test with various sequence lengths"""

def test_block_batch_independence():
    """Verify batch elements processed independently"""

def test_block_determinism():
    """Same input should give same output"""
```

**Estimated tests to add:** 6-8

---

#### FFTNet Model (`tests/test_model.py`)
**Current coverage:** Basic forward pass

**Additional tests needed:**
```python
def test_model_vocab_size_handling():
    """Test with different vocab sizes"""

def test_model_layer_stacking():
    """Verify multiple blocks compose correctly"""

def test_model_parameter_count():
    """Check parameter count matches expected"""

def test_model_output_logits_range():
    """Verify logits are in reasonable range"""

def test_model_embedding_gradients():
    """Check gradients flow to embeddings"""

def test_model_overfitting_single_batch():
    """Model should overfit one batch (sanity check)"""
```

**Estimated tests to add:** 6-8

---

### 1.2 Utility Tests

#### Storage (`tests/test_storage.py`)
**Current coverage:** Basic save/load

**Additional tests needed:**
```python
def test_storage_complex_param_fidelity():
    """Complex params should match exactly after save/load"""

def test_storage_config_preservation():
    """Config should be identical after save/load"""

def test_storage_missing_file_handling():
    """Should raise clear error for missing files"""

def test_storage_corrupted_file_handling():
    """Handle corrupted safetensors gracefully"""

def test_storage_version_compatibility():
    """Test loading models from different versions"""

def test_storage_metadata_completeness():
    """All complex tensors should be in metadata"""
```

**Estimated tests to add:** 6-8

---

#### Configuration (`tests/test_config.py`)
**Current coverage:** Basic loading

**Additional tests needed:**
```python
def test_config_invalid_json():
    """Handle malformed JSON gracefully"""

def test_config_invalid_yaml():
    """Handle malformed YAML gracefully"""

def test_config_missing_required_fields():
    """Fail clearly on missing fields"""

def test_config_block_registry():
    """Verify all block types are registered"""

def test_config_dynamic_model_building():
    """Build model from config matches manual build"""

def test_config_unknown_block_type():
    """Handle unknown block types gracefully"""
```

**Estimated tests to add:** 6-8

---

#### Tokenizer (`tests/test_tokenizer.py`)
**Current coverage:** Basic encode/decode

**Additional tests needed:**
```python
def test_tokenizer_empty_string():
    """Handle empty input correctly"""

def test_tokenizer_special_characters():
    """Test Unicode, emojis, etc."""

def test_tokenizer_roundtrip_fidelity():
    """encode(decode(x)) should equal x"""

def test_tokenizer_vocab_size_enforcement():
    """Should respect vocab_size parameter"""

def test_tokenizer_unknown_token_handling():
    """Handle out-of-vocab gracefully"""

def test_tokenizer_batch_consistency():
    """Batch encode matches individual encodes"""
```

**Estimated tests to add:** 6-8

---

#### Data (`tests/test_text_dataset.py`)
**Current coverage:** Basic iteration

**Additional tests needed:**
```python
def test_dataset_empty_file():
    """Handle empty corpus file"""

def test_dataset_short_file():
    """Handle corpus shorter than seq_len"""

def test_dataset_sequence_boundaries():
    """Verify no off-by-one errors in sequences"""

def test_dataset_target_alignment():
    """Targets should be inputs shifted by 1"""

def test_dataset_reproducibility():
    """Same file should give same batches"""

def test_dataset_memory_efficiency():
    """Should not load file multiple times"""
```

**Estimated tests to add:** 6-8

---

#### Visualization (`tests/test_visualization.py`)
**Current coverage:** Basic plotting

**Additional tests needed:**
```python
def test_visualization_save_mode():
    """Should create file in save mode"""

def test_visualization_empty_input():
    """Handle empty embeddings gracefully"""

def test_visualization_spectrum_values():
    """Verify spectrum calculation correctness"""

def test_visualization_plot_dimensions():
    """Check plot has correct axes, labels"""

def test_visualization_agg_backend():
    """Verify Agg backend is used (no display)"""
```

**Estimated tests to add:** 5-6

---

### 1.3 Training Function Tests

**New file:** `tests/test_training.py`

```python
def test_train_loop_single_batch():
    """Train on one batch, verify loss decreases"""

def test_train_validation_split():
    """Verify train/val split is correct size"""

def test_train_mixed_precision():
    """Mixed precision should match regular training"""

def test_train_early_stopping_triggers():
    """Early stopping should trigger after patience"""

def test_train_optimizer_step():
    """Verify parameters update after optimizer step"""

def test_train_gradient_accumulation():
    """Test gradient accumulation if implemented"""

def test_train_logging_format():
    """JSONL logs should have required fields"""

def test_train_checkpoint_saving():
    """Checkpoints should be created at intervals"""
```

**Estimated tests to add:** 8-10

---

### 1.4 Inference Function Tests

**Expand:** `tests/test_infer.py`

**Additional tests needed:**
```python
def test_generate_determinism():
    """Same prompt should give same output (greedy)"""

def test_generate_max_tokens_respected():
    """Should generate exactly max_new_tokens"""

def test_generate_eos_handling():
    """Should stop at EOS token if encountered"""

def test_generate_batch_processing():
    """Batch generation should match individual"""

def test_cli_mode_text():
    """Text mode CLI output is correct"""

def test_cli_mode_logits():
    """Logits mode shows top-k correctly"""

def test_cli_mode_spectrum():
    """Spectrum mode creates visualization"""

def test_cli_missing_model():
    """Handle missing model file gracefully"""
```

**Estimated tests to add:** 8-10

---

## 2. Integration Testing

### 2.1 Training Pipeline Tests

**New file:** `tests/integration/test_training_pipeline.py`

```python
def test_full_training_workflow():
    """
    End-to-end training test:
    1. Create small corpus
    2. Train tokenizer
    3. Create dataset
    4. Train model for few steps
    5. Verify loss decreases
    6. Save model
    7. Load and verify
    """

def test_distillation_workflow():
    """
    Test distillation pipeline:
    1. Load teacher model (small GPT-2)
    2. Create student model
    3. Train with distillation loss
    4. Verify student learns from teacher
    """

def test_training_resume():
    """
    Test checkpoint resume:
    1. Train for N steps
    2. Save checkpoint
    3. Resume from checkpoint
    4. Verify continues correctly
    """

def test_validation_during_training():
    """
    Test validation loop:
    1. Train with validation set
    2. Verify validation runs each epoch
    3. Check validation metrics logged
    """
```

**Estimated tests:** 4-6

---

### 2.2 Inference Pipeline Tests

**New file:** `tests/integration/test_inference_pipeline.py`

```python
def test_train_then_infer():
    """
    Complete workflow:
    1. Train tiny model
    2. Save
    3. Load in inference script
    4. Generate text
    5. Verify output is coherent
    """

def test_model_versioning():
    """
    Test model management:
    1. Save multiple versions
    2. List versions
    3. Load specific version
    4. Delete old version
    """

def test_tokenizer_consistency():
    """
    Verify tokenizer used in training matches inference:
    1. Train with tokenizer A
    2. Save both model and tokenizer
    3. Load and infer
    4. Verify outputs decode correctly
    """
```

**Estimated tests:** 3-5

---

### 2.3 Data Pipeline Tests

**New file:** `tests/integration/test_data_pipeline.py`

```python
def test_corpus_to_dataset():
    """
    Full data pipeline:
    1. Create text corpus
    2. Train tokenizer
    3. Create dataset
    4. Iterate and verify batches
    """

def test_multi_file_loading():
    """
    Test loading from multiple files:
    1. Create multiple corpus files
    2. Load with pattern matching
    3. Verify all data included
    """

def test_large_corpus_streaming():
    """
    Test streaming for large files:
    1. Create large corpus (or mock it)
    2. Load with streaming
    3. Verify memory usage stays constant
    """
```

**Estimated tests:** 3-5

---

### 2.4 Config-to-Model Tests

**New file:** `tests/integration/test_config_system.py`

```python
def test_json_yaml_to_model():
    """
    Test full config pipeline:
    1. Load JSON config
    2. Load YAML modules
    3. Build model
    4. Verify architecture matches spec
    """

def test_custom_block_config():
    """
    Test custom blocks via config:
    1. Define custom block in YAML
    2. Register in registry
    3. Build model
    4. Verify custom block used
    """

def test_config_validation():
    """
    Test config validation:
    1. Invalid configs should fail gracefully
    2. Error messages should be clear
    """
```

**Estimated tests:** 3-5

---

## 3. End-to-End Testing

### 3.1 Complete Workflows

**New file:** `tests/e2e/test_complete_workflows.py`

```python
@pytest.mark.slow
def test_complete_training_cycle():
    """
    Full realistic workflow (10-15 min):
    1. Download small corpus (e.g., tiny Shakespeare)
    2. Train tokenizer (vocab=1000)
    3. Train model (dim=64, blocks=2, epochs=5)
    4. Evaluate on validation set
    5. Generate samples
    6. Verify perplexity < threshold
    7. Verify generation is coherent
    """

@pytest.mark.slow
def test_distillation_from_gpt2():
    """
    Test GPT-2 distillation (15-20 min):
    1. Load GPT-2 teacher
    2. Create student model
    3. Distill for few epochs
    4. Compare student vs teacher outputs
    5. Verify student learns something useful
    """

@pytest.mark.slow
def test_benchmark_workflow():
    """
    Test evaluation on standard benchmark:
    1. Download WikiText-2
    2. Train small model
    3. Evaluate perplexity
    4. Compare with baseline
    """
```

**Estimated tests:** 3-5

**Note:** These tests are marked `@pytest.mark.slow` and run separately from fast unit tests.

---

### 3.2 CLI Integration Tests

**New file:** `tests/e2e/test_cli.py`

```python
def test_train_fresh_cli():
    """Test train_fresh.py via subprocess"""
    result = subprocess.run([
        "python", "scripts/train_fresh.py",
        "--data-path", "test_corpus.txt",
        "--epochs", "1",
        "--batch-size", "4"
    ], capture_output=True)
    assert result.returncode == 0

def test_evaluate_cli():
    """Test evaluate.py via subprocess"""
    # Similar subprocess test

def test_compare_runs_cli():
    """Test compare_runs.py via subprocess"""
    # Similar subprocess test
```

**Estimated tests:** 5-8

---

## 4. Property-Based Testing

Use `hypothesis` library for property-based testing.

**New file:** `tests/property/test_properties.py`

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=1, max_value=1000),
       st.integers(min_value=4, max_value=128))
def test_rope_shape_property(batch_size, seq_len):
    """ComplexRoPE should work for any valid batch/seq size"""
    rope = ComplexRoPE(dim=16)
    x = torch.randn(batch_size, seq_len, 16)
    output = rope(x)
    assert output.shape == x.shape

@given(st.integers(min_value=10, max_value=10000),
       st.integers(min_value=2, max_value=512))
def test_model_vocab_size_property(vocab_size, dim):
    """Model should work for any valid vocab size and dim"""
    model = FFTNet(vocab_size=vocab_size, dim=dim, num_blocks=1)
    input_ids = torch.randint(0, vocab_size, (2, 10))
    logits = model(input_ids)
    assert logits.shape == (2, 10, vocab_size)

@given(st.text(min_size=1, max_size=1000))
def test_tokenizer_roundtrip_property(text):
    """Any text should roundtrip through tokenizer"""
    tokenizer = SimpleTokenizer(vocab_size=1000)
    # Train on some data first
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    # Might not be exact due to BPE, but should be similar
    assert len(decoded) > 0
```

**Estimated tests:** 10-15 properties

---

## 5. Performance Testing

### 5.1 Benchmark Tests

**Expand:** `benchmarks/benchmark_model.py` and `benchmarks/benchmark_fftnet_block.py`

**Additional metrics:**
```python
def test_benchmark_regression():
    """
    Ensure performance doesn't regress:
    1. Run benchmark
    2. Compare with baseline (stored in file)
    3. Fail if > 10% slower
    """

def test_benchmark_memory_growth():
    """
    Test for memory leaks:
    1. Run model 100 times
    2. Track memory usage
    3. Ensure no continuous growth
    """

def test_benchmark_batch_scaling():
    """
    Test batch size scaling:
    1. Benchmark with batch sizes 1, 2, 4, 8, 16
    2. Verify throughput scales appropriately
    3. Memory usage scales linearly
    """
```

**Estimated tests:** 5-8

---

### 5.2 Scalability Tests

**New file:** `tests/performance/test_scalability.py`

```python
@pytest.mark.slow
def test_large_model_training():
    """Test training with large model (100M+ params)"""

@pytest.mark.slow
def test_long_sequence_handling():
    """Test with very long sequences (1024, 2048 tokens)"""

def test_memory_efficiency_by_sequence_length():
    """
    Measure memory usage vs sequence length:
    - Should scale as O(seq_len) not O(seq_len^2)
    """

def test_compilation_speedup():
    """
    Test torch.compile speedup:
    1. Benchmark without compilation
    2. Benchmark with torch.compile
    3. Verify speedup
    """
```

**Estimated tests:** 4-6

---

## 6. Regression Testing

### 6.1 Output Consistency Tests

**New file:** `tests/regression/test_output_consistency.py`

```python
def test_model_output_consistency():
    """
    Ensure model outputs don't change unexpectedly:
    1. Load fixed weights
    2. Run inference on fixed input
    3. Compare output with stored reference
    4. Fail if outputs differ
    """

def test_training_reproducibility():
    """
    Verify training is reproducible:
    1. Train with fixed seed
    2. Train again with same seed
    3. Verify final losses match
    """
```

**Estimated tests:** 2-4

---

### 6.2 Known Issue Tests

**New file:** `tests/regression/test_known_issues.py`

```python
def test_issue_001_complex_gradient_bug():
    """
    Regression test for Issue #001:
    Complex gradients were None in early versions
    """
    # Test that complex params have gradients

def test_issue_002_tokenizer_empty_string():
    """
    Regression test for Issue #002:
    Tokenizer crashed on empty string
    """
    # Test empty string handling
```

**Pattern:** Add test for each bug fix to prevent recurrence.

---

## 7. Coverage Analysis

### 7.1 Current Coverage

**Goal:** Measure and improve code coverage

**Tools:**
- `pytest-cov` for coverage reports
- `coverage.py` for detailed analysis

**Commands:**
```bash
# Generate coverage report
pytest --cov=. --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

**Current estimated coverage:** ~40-50%

**Target coverage:** > 90%

---

### 7.2 Coverage Improvement Plan

**Priority areas:**
1. **Core modules** (model.py, fftnet_block.py, etc.): Target 95%+
2. **Utilities** (storage, config, visualization): Target 90%+
3. **Scripts** (training, evaluation): Target 70%+ (harder to test)
4. **Tests themselves**: N/A

**Action items:**
- [ ] Measure current coverage: `pytest --cov=.`
- [ ] Identify uncovered lines
- [ ] Write tests for uncovered code
- [ ] Add coverage threshold in CI: `--cov-fail-under=80`
- [ ] Gradually increase threshold

---

## 8. Test Infrastructure

### 8.1 Test Organization

**Directory structure:**
```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_model.py
│   ├── test_blocks.py
│   ├── test_utils.py
│   └── ...
├── integration/           # Component interaction tests
│   ├── test_training_pipeline.py
│   ├── test_inference_pipeline.py
│   └── ...
├── e2e/                   # End-to-end workflows
│   ├── test_complete_workflows.py
│   └── test_cli.py
├── performance/           # Benchmarks and scalability
│   ├── test_scalability.py
│   └── test_benchmarks.py
├── regression/            # Prevent known issues
│   ├── test_output_consistency.py
│   └── test_known_issues.py
├── property/              # Property-based tests
│   └── test_properties.py
├── fixtures/              # Shared test fixtures
│   ├── conftest.py
│   ├── sample_corpus.txt
│   └── reference_outputs.json
└── README.md              # Testing documentation
```

---

### 8.2 Shared Fixtures

**File:** `tests/fixtures/conftest.py`

```python
import pytest
import torch
from pathlib import Path

@pytest.fixture
def tiny_model():
    """Tiny model for fast testing"""
    return FFTNet(vocab_size=100, dim=16, num_blocks=1)

@pytest.fixture
def sample_tokenizer(tmp_path):
    """Trained tokenizer for testing"""
    tokenizer = SimpleTokenizer(vocab_size=100)
    corpus = "the quick brown fox jumps over the lazy dog"
    tokenizer.train_from_iterator([corpus], vocab_size=100)
    return tokenizer

@pytest.fixture
def sample_corpus(tmp_path):
    """Small text corpus for testing"""
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("This is a test corpus. " * 100)
    return corpus_path

@pytest.fixture
def trained_model(tmp_path, sample_corpus, sample_tokenizer):
    """Pre-trained tiny model for testing"""
    # Train for a few steps and return
    # Useful for inference tests
    ...
```

**Estimated fixtures:** 10-15

---

### 8.3 Test Utilities

**File:** `tests/utils.py`

```python
def assert_tensors_close(a, b, rtol=1e-5, atol=1e-8):
    """Assert tensors are approximately equal"""
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

def assert_complex_equal(a, b):
    """Assert complex tensors are equal"""
    assert_tensors_close(a.real, b.real)
    assert_tensors_close(a.imag, b.imag)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_overfitting_test(model, batch, steps=100):
    """
    Sanity check: model should overfit single batch
    Returns final loss (should be very low)
    """
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(steps):
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()
```

---

### 8.4 CI/CD Integration

**File:** `.github/workflows/tests.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov hypothesis
      - run: pytest tests/unit --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/integration

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/e2e -m slow --timeout=1800
```

**Additional CI checks:**
- [ ] Type checking with `mypy`
- [ ] Linting with `ruff` or `flake8`
- [ ] Format checking with `black`
- [ ] Documentation building

---

## 9. Test Execution Strategy

### 9.1 Test Suites

**Fast suite** (< 1 minute):
```bash
pytest tests/unit -v
```

**Medium suite** (< 5 minutes):
```bash
pytest tests/unit tests/integration -v
```

**Full suite** (< 30 minutes):
```bash
pytest -v
```

**Slow suite** (can be hours):
```bash
pytest -m slow -v
```

**Coverage suite**:
```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
```

---

### 9.2 Test Markers

Configure in `pytest.ini`:
```ini
[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests requiring GPU
    integration: marks integration tests
    e2e: marks end-to-end tests
    regression: marks regression tests
```

**Usage:**
```bash
# Run only fast tests
pytest -m "not slow"

# Run only GPU tests (if GPU available)
pytest -m gpu

# Skip integration and e2e
pytest -m "not integration and not e2e"
```

---

## 10. Testing Checklist

### Pre-Commit Checklist
- [ ] Run fast test suite: `pytest tests/unit`
- [ ] Check coverage of changed files
- [ ] Run linter: `ruff check .`
- [ ] Run formatter: `black --check .`
- [ ] Type check: `mypy .`

### Pre-PR Checklist
- [ ] Run full test suite: `pytest`
- [ ] Generate coverage report
- [ ] Coverage > 80% for new code
- [ ] All tests pass
- [ ] Add tests for new features
- [ ] Add regression tests for bug fixes
- [ ] Update documentation

### Release Checklist
- [ ] All tests pass (including slow)
- [ ] Benchmark tests pass
- [ ] No performance regression
- [ ] Coverage > 90% for core modules
- [ ] E2E tests pass
- [ ] Manual testing on sample corpora
- [ ] Documentation up to date

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
**Goal:** Improve unit test coverage to 80%+

- [ ] Add missing unit tests (50+ tests)
- [ ] Set up coverage measurement
- [ ] Create shared fixtures
- [ ] Organize test directory structure

**Deliverables:**
- 80%+ coverage on core modules
- Shared fixture library
- Test utilities module

---

### Phase 2: Integration (Week 3-4)
**Goal:** Add comprehensive integration tests

- [ ] Training pipeline tests (5-10 tests)
- [ ] Inference pipeline tests (5-10 tests)
- [ ] Data pipeline tests (5-10 tests)
- [ ] Config system tests (5-10 tests)

**Deliverables:**
- 20-40 integration tests
- CI running integration tests
- Test execution < 5 minutes

---

### Phase 3: E2E & Performance (Week 5-6)
**Goal:** Add end-to-end and performance tests

- [ ] Complete workflow tests (3-5 tests)
- [ ] CLI integration tests (5-8 tests)
- [ ] Performance benchmarks (5-8 tests)
- [ ] Scalability tests (4-6 tests)

**Deliverables:**
- E2E test suite
- Performance baselines established
- Benchmark CI workflow

---

### Phase 4: Advanced Testing (Week 7-8)
**Goal:** Property-based and regression testing

- [ ] Property-based tests (10-15 tests)
- [ ] Output consistency tests
- [ ] Regression test framework
- [ ] Coverage > 90%

**Deliverables:**
- Hypothesis tests
- Regression test suite
- > 90% overall coverage
- Complete testing documentation

---

## 12. Success Metrics

### Quantitative Metrics
- **Test count:** 150+ tests (currently ~30)
- **Coverage:** > 90% for core, > 80% overall
- **Test speed:** Unit tests < 1 min, full suite < 30 min
- **CI speed:** < 10 minutes for standard checks

### Qualitative Metrics
- **Confidence:** Deploy without fear of breaking things
- **Debugging speed:** Tests pinpoint issues quickly
- **Documentation:** Tests serve as usage examples
- **Maintenance:** Easy to add tests for new features

### Process Metrics
- **Pre-commit:** All developers run tests before committing
- **PR reviews:** Tests reviewed as carefully as code
- **Bug fixes:** Always accompanied by regression test
- **New features:** Always accompanied by tests

---

## Summary

This comprehensive testing plan covers:

1. **Unit testing:** 100+ new unit tests for complete coverage
2. **Integration testing:** 30+ tests for component interactions
3. **E2E testing:** 10+ tests for complete workflows
4. **Property-based:** 15+ property tests for edge cases
5. **Performance:** 10+ benchmarks and scalability tests
6. **Regression:** Framework to prevent bug recurrence
7. **Infrastructure:** CI/CD, fixtures, utilities

**Total estimated tests to add:** 150-200 tests

**Total estimated effort:** 6-8 weeks for complete implementation

**Current state:** ~30 tests, ~40-50% coverage
**Target state:** 180+ tests, > 90% coverage, robust CI/CD

**Recommended approach:** Implement in 4 phases over 2 months, prioritizing high-value unit tests first, then building up to integration and E2E tests.
