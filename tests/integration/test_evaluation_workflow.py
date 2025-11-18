"""Integration tests for evaluation workflows."""

import math
import tempfile
from pathlib import Path
import shutil

import pytest
import torch

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.storage import save_model, load_model
from tokenizer import SimpleTokenizer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def trained_model_and_tokenizer(temp_dir):
    """Create a small trained model for evaluation testing."""
    # Create corpus
    corpus_path = temp_dir / "corpus.txt"
    corpus_text = "The quick brown fox jumps over the lazy dog. " * 20
    corpus_path.write_text(corpus_text)

    # Create and train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train_from_iterator([corpus_text], vocab_size=100)
    tokenizer_path = temp_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Create model
    cfg = {
        "vocab_size": len(tokenizer),
        "dim": 16,
        "num_blocks": 1,
        "model_type": "fft"
    }

    model = FFTNet(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        num_blocks=cfg["num_blocks"]
    )

    # Save model
    model_path = temp_dir / "test_model"
    save_model(model, str(model_path), cfg)

    return model_path, tokenizer_path, corpus_path


def test_model_evaluation_basic(trained_model_and_tokenizer):
    """Test basic model evaluation."""
    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create test dataset
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=8)

    # Evaluate
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for x, y in dataset:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            logits = model(x)
            logits = logits.view(-1, cfg["vocab_size"])
            targets = y.view(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()

            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            count += targets.numel()

    avg_loss = total_loss / count
    accuracy = correct / count

    # Basic sanity checks
    assert avg_loss > 0, "Loss should be positive"
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"


def test_perplexity_calculation(trained_model_and_tokenizer):
    """Test perplexity calculation."""
    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create test dataset
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=8)

    # Evaluate
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in dataset:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            logits = model(x)
            logits = logits.view(-1, cfg["vocab_size"])
            targets = y.view(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()
            count += targets.numel()

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)

    # Perplexity sanity checks
    assert perplexity > 1, "Perplexity should be > 1"
    assert perplexity < 1e6, "Perplexity should be reasonable"
    assert math.isclose(math.log(perplexity), avg_loss, rel_tol=1e-5)


def test_spectrum_computation(trained_model_and_tokenizer):
    """Test frequency spectrum computation."""
    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create test dataset
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=8)

    # Compute spectrum
    model.eval()
    mags = []

    with torch.no_grad():
        for i, (x, _) in enumerate(dataset):
            if i >= 3:  # Just compute for a few samples
                break

            x = x.unsqueeze(0)
            logits = model(x)

            # Apply FFT
            freq = torch.fft.fft(logits, dim=1)
            mag = freq.abs().mean(dim=(0, 2))
            mags.append(mag)

    avg_spectrum = torch.stack(mags).mean(dim=0)

    # Spectrum sanity checks
    assert avg_spectrum.shape[0] == 8, "Spectrum should match sequence length"
    assert (avg_spectrum >= 0).all(), "Magnitudes should be non-negative"
    assert avg_spectrum.sum() > 0, "Spectrum should have non-zero energy"


def test_batch_evaluation(trained_model_and_tokenizer):
    """Test evaluation with batched data."""
    from torch.utils.data import DataLoader

    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create test dataset with batching
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Evaluate with batches
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            logits = logits.view(-1, cfg["vocab_size"])
            targets = y.view(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()
            count += targets.numel()

    avg_loss = total_loss / count

    # Should complete without errors
    assert avg_loss > 0


def test_evaluation_determinism(trained_model_and_tokenizer):
    """Test that evaluation is deterministic."""
    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create test dataset
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=8)

    # Evaluate twice
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate_once():
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, y in dataset:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                logits = model(x)
                logits = logits.view(-1, cfg["vocab_size"])
                targets = y.view(-1)

                loss = loss_fn(logits, targets)
                total_loss += loss.item() * targets.numel()
                count += targets.numel()

        return total_loss / count

    loss1 = evaluate_once()
    loss2 = evaluate_once()

    # Should be identical
    assert math.isclose(loss1, loss2, rel_tol=1e-6)


def test_evaluation_with_different_batch_sizes(trained_model_and_tokenizer):
    """Test that evaluation results are consistent across batch sizes."""
    from torch.utils.data import DataLoader

    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create test dataset
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=8)

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate_with_batch_size(batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, y in loader:
                logits = model(x)
                logits = logits.view(-1, cfg["vocab_size"])
                targets = y.view(-1)

                loss = loss_fn(logits, targets)
                total_loss += loss.item() * targets.numel()
                count += targets.numel()

        return total_loss / count

    # Evaluate with different batch sizes
    loss_batch1 = evaluate_with_batch_size(1)
    loss_batch2 = evaluate_with_batch_size(2)
    loss_batch4 = evaluate_with_batch_size(4)

    # Should be very close
    assert math.isclose(loss_batch1, loss_batch2, rel_tol=1e-5)
    assert math.isclose(loss_batch2, loss_batch4, rel_tol=1e-5)


def test_generation_quality_metrics(trained_model_and_tokenizer):
    """Test generation and basic quality metrics."""
    from fftnet_infer import generate

    model_path, tokenizer_path, _ = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    model.eval()

    # Generate text with different strategies
    prompt = "The quick"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    # Greedy generation
    generated_greedy, _ = generate(
        model, input_tensor, max_new_tokens=10, temperature=0.0
    )

    # Sampling generation
    torch.manual_seed(42)
    generated_sample, _ = generate(
        model, input_tensor, max_new_tokens=10, temperature=1.0
    )

    # Both should generate valid sequences
    assert generated_greedy.shape[1] == input_tensor.shape[1] + 10
    assert generated_sample.shape[1] == input_tensor.shape[1] + 10

    # Greedy should be deterministic
    torch.manual_seed(123)
    generated_greedy2, _ = generate(
        model, input_tensor, max_new_tokens=10, temperature=0.0
    )
    assert torch.equal(generated_greedy, generated_greedy2)


def test_model_comparison(trained_model_and_tokenizer, temp_dir):
    """Test comparing two models."""
    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load first model
    model1, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create second model (same architecture, different weights)
    model2 = FFTNet(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        num_blocks=cfg["num_blocks"]
    )

    # Save second model
    model2_path = temp_dir / "model2"
    save_model(model2, str(model2_path), cfg)

    # Evaluate both models on same data
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=8)
    loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate_model(model):
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, y in dataset:
                x = x.unsqueeze(0)
                y = y.unsqueeze(0)

                logits = model(x)
                logits = logits.view(-1, cfg["vocab_size"])
                targets = y.view(-1)

                loss = loss_fn(logits, targets)
                total_loss += loss.item() * targets.numel()
                count += targets.numel()

        return total_loss / count

    loss1 = evaluate_model(model1)
    loss2 = evaluate_model(model2)

    # Both should have positive loss
    assert loss1 > 0
    assert loss2 > 0

    # Different random weights should give different results
    # (unless extremely unlikely)
    assert not math.isclose(loss1, loss2, rel_tol=1e-3)


@pytest.mark.parametrize("seq_len", [4, 8, 16])
def test_evaluation_with_different_sequence_lengths(
    trained_model_and_tokenizer, seq_len
):
    """Test evaluation with different sequence lengths."""
    model_path, tokenizer_path, corpus_path = trained_model_and_tokenizer

    # Load model and tokenizer
    model, cfg = load_model(model_path)
    tokenizer = SimpleTokenizer.load(str(tokenizer_path))

    # Create dataset with specific sequence length
    dataset = TextFileDataset(str(corpus_path), tokenizer, seq_len=seq_len)

    # Evaluate
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataset):
            if i >= 5:  # Just test a few samples
                break

            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            logits = model(x)
            logits = logits.view(-1, cfg["vocab_size"])
            targets = y.view(-1)

            loss = loss_fn(logits, targets)
            total_loss += loss.item() * targets.numel()
            count += targets.numel()

    avg_loss = total_loss / count

    # Should work with all sequence lengths
    assert avg_loss > 0
