"""Integration tests for training workflows."""

import json
import tempfile
from pathlib import Path
import shutil

import pytest
import torch

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.config import load_config, build_model_from_config
from fftnet.utils.storage import save_model, load_model
from tokenizer import SimpleTokenizer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_corpus(temp_dir):
    """Create a small sample corpus for testing."""
    corpus_path = temp_dir / "corpus.txt"
    corpus_text = "The quick brown fox jumps over the lazy dog. " * 20
    corpus_path.write_text(corpus_text)
    return corpus_path


@pytest.fixture
def sample_tokenizer(temp_dir, sample_corpus):
    """Create and train a tokenizer on sample corpus."""
    tokenizer = SimpleTokenizer(vocab_size=100)
    corpus_text = sample_corpus.read_text()
    tokenizer.train_from_iterator([corpus_text], vocab_size=100)

    tokenizer_path = temp_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    return tokenizer_path


def test_full_training_workflow(temp_dir, sample_corpus, sample_tokenizer):
    """Test complete training workflow from scratch."""
    # Load tokenizer
    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))

    # Create tiny model config
    cfg = {
        "vocab_size": len(tokenizer),
        "dim": 16,
        "num_blocks": 1,
        "model_type": "fft"
    }

    # Build model
    model = FFTNet(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        num_blocks=cfg["num_blocks"]
    )

    # Create dataset
    dataset = TextFileDataset(str(sample_corpus), tokenizer, seq_len=8)
    assert len(dataset) > 0, "Dataset should not be empty"

    # Train for a few steps
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    for epoch in range(2):
        for i, (x, y) in enumerate(dataset):
            if i >= 5:  # Just train on 5 batches
                break

            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            optimizer.zero_grad()
            logits = model(x)
            logits = logits.view(-1, cfg["vocab_size"])
            targets = y.view(-1)
            loss = loss_fn(logits, targets)

            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

            loss.backward()
            optimizer.step()

    # Loss should decrease with training
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"

    # Save model
    save_path = temp_dir / "trained_model"
    save_model(model, str(save_path), cfg)

    # Verify saved files exist
    assert (save_path.parent / f"{save_path.name}.safetensors").exists()
    assert (save_path.parent / f"{save_path.name}_config.json").exists()

    # Load model back
    loaded_model, loaded_cfg = load_model(save_path)

    # Verify config matches
    assert loaded_cfg["vocab_size"] == cfg["vocab_size"]
    assert loaded_cfg["dim"] == cfg["dim"]

    # Verify model produces same output
    model.eval()
    loaded_model.eval()

    test_input = torch.randint(0, cfg["vocab_size"], (1, 8))
    with torch.no_grad():
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)

    torch.testing.assert_close(original_output, loaded_output, rtol=1e-4, atol=1e-4)


def test_checkpoint_save_and_resume(temp_dir, sample_corpus, sample_tokenizer):
    """Test checkpointing and resuming training."""
    from torch.cuda.amp import GradScaler

    # Setup
    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))
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

    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(enabled=False)

    # Save checkpoint
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir()

    checkpoint = {
        "epoch": 5,
        "step": 100,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": 0.5,
        "patience_cntr": 2,
        "config": cfg,
    }

    checkpoint_path = checkpoint_dir / "checkpoint_epoch_5_step_100.pt"
    torch.save(checkpoint, checkpoint_path)

    # Create new model and optimizer
    new_model = FFTNet(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        num_blocks=cfg["num_blocks"]
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    new_scaler = GradScaler(enabled=False)

    # Load checkpoint
    loaded_checkpoint = torch.load(checkpoint_path)
    new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    new_scaler.load_state_dict(loaded_checkpoint["scaler_state_dict"])

    # Verify state matches
    assert loaded_checkpoint["epoch"] == 5
    assert loaded_checkpoint["step"] == 100
    assert loaded_checkpoint["best_val"] == 0.5
    assert loaded_checkpoint["patience_cntr"] == 2

    # Verify model state matches
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        torch.testing.assert_close(p1, p2)


def test_training_with_validation_split(temp_dir, sample_corpus, sample_tokenizer):
    """Test training with train/validation split."""
    from torch.utils.data import random_split

    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))
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

    # Create dataset and split
    full_dataset = TextFileDataset(str(sample_corpus), tokenizer, seq_len=8)

    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    assert len(train_dataset) + len(val_dataset) == len(full_dataset)
    assert len(val_dataset) == val_size
    assert len(train_dataset) == train_size

    # Verify datasets are different
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0


def test_mixed_precision_training(temp_dir, sample_corpus, sample_tokenizer):
    """Test training with mixed precision."""
    from torch.cuda.amp import GradScaler, autocast

    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))
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

    device = torch.device("cpu")
    model.to(device)

    dataset = TextFileDataset(str(sample_corpus), tokenizer, seq_len=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(enabled=False)  # CPU doesn't support AMP, but we can test the API
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train one step with mixed precision API
    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    optimizer.zero_grad()

    with autocast(enabled=False):
        logits = model(x)
        logits = logits.view(-1, cfg["vocab_size"])
        targets = y.view(-1)
        loss = loss_fn(logits, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Should complete without errors
    assert loss.item() > 0


def test_gradient_clipping(temp_dir, sample_corpus, sample_tokenizer):
    """Test gradient clipping during training."""
    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))
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

    device = torch.device("cpu")
    model.to(device)

    dataset = TextFileDataset(str(sample_corpus), tokenizer, seq_len=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train one step
    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    optimizer.zero_grad()
    logits = model(x)
    logits = logits.view(-1, cfg["vocab_size"])
    targets = y.view(-1)
    loss = loss_fn(logits, targets)
    loss.backward()

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Should return a positive gradient norm
    assert grad_norm > 0

    optimizer.step()


def test_learning_rate_scheduling(temp_dir, sample_corpus, sample_tokenizer):
    """Test learning rate schedulers."""
    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=0.0001
    )

    initial_lr = optimizer.param_groups[0]['lr']

    # Step through scheduler
    for _ in range(5):
        scheduler.step()

    mid_lr = optimizer.param_groups[0]['lr']

    # LR should have changed
    assert mid_lr != initial_lr

    # Test OneCycle scheduler
    optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer2, max_lr=0.01, total_steps=20, pct_start=0.3
    )

    lrs = []
    for _ in range(20):
        lrs.append(optimizer2.param_groups[0]['lr'])
        scheduler2.step()

    # Should have warmup then decay
    assert max(lrs) > lrs[0]
    assert lrs[-1] < max(lrs)


def test_inference_after_training(temp_dir, sample_corpus, sample_tokenizer):
    """Test that trained model can generate text."""
    from fftnet_infer import generate

    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))
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

    model.eval()

    # Generate text
    prompt = "The quick"
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    generated, logits = generate(
        model,
        input_tensor,
        max_new_tokens=5,
        temperature=1.0,
        top_k=0,
        top_p=1.0
    )

    # Should generate more tokens than input
    assert generated.shape[1] > input_tensor.shape[1]

    # Decode output
    output_text = tokenizer.decode(generated[0].tolist())
    assert len(output_text) > len(prompt)


def test_early_stopping_logic(temp_dir):
    """Test early stopping patience counter."""
    patience = 3
    best_val = float('inf')
    patience_cntr = 0

    val_losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]  # Loss stops improving

    for val_loss in val_losses:
        if val_loss < best_val:
            best_val = val_loss
            patience_cntr = 0
        else:
            patience_cntr += 1
            if patience_cntr >= patience:
                break

    # Should have triggered early stopping
    assert patience_cntr >= patience
    assert best_val == 0.85


def test_jsonl_logging_format(temp_dir):
    """Test JSONL logging format."""
    log_path = temp_dir / "test_log.jsonl"

    # Write some log entries
    with log_path.open("w") as f:
        entries = [
            {"step": 1, "epoch": 1, "loss": 0.5, "accuracy": 0.8, "lr": 0.001},
            {"step": 2, "epoch": 1, "loss": 0.45, "accuracy": 0.82, "lr": 0.001},
            {"step": 3, "epoch": 1, "loss": 0.4, "accuracy": 0.85, "lr": 0.001},
        ]

        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Read back and verify
    with log_path.open("r") as f:
        lines = f.readlines()

    assert len(lines) == 3

    for line in lines:
        entry = json.loads(line)
        assert "step" in entry
        assert "epoch" in entry
        assert "loss" in entry
        assert "accuracy" in entry


@pytest.mark.parametrize("scheduler_type", ["cosine", "onecycle", "plateau"])
def test_different_schedulers(temp_dir, sample_corpus, sample_tokenizer, scheduler_type):
    """Test different learning rate schedulers."""
    tokenizer = SimpleTokenizer.load(str(sample_tokenizer))
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.0001
        )
    elif scheduler_type == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, total_steps=20, pct_start=0.1
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

    # Test that scheduler can be stepped
    if scheduler_type == "plateau":
        scheduler.step(0.5)  # Needs a metric
    else:
        scheduler.step()

    # Should not raise any errors
    assert True
