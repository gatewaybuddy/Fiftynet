import argparse
from pathlib import Path

import torch

from tokenizer import SimpleTokenizer
from fftnet.utils.visualization import plot_embedding_spectrum
from fftnet.utils.sampling import sample_top_k_top_p


from model import FFTNet
from fftnet.utils import storage
from fftnet.utils.config import load_config, build_model_from_config


def _tokenize(tokenizer: SimpleTokenizer, prompt: str) -> list[int]:
    """Convert prompt text to token IDs."""
    return tokenizer.encode(prompt)


def _decode(tokenizer: SimpleTokenizer, tokens: torch.Tensor) -> str:
    return tokenizer.decode(tokens.tolist())


@torch.no_grad()
def generate(
    model: FFTNet,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate tokens using the specified sampling strategy.

    Args:
        model: The FFTNet model
        input_ids: Input token IDs of shape (batch_size, seq_len)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens (0 to disable)
        top_p: Nucleus sampling threshold (1.0 to disable)

    Returns:
        Tuple of (generated_tokens, final_logits)
    """
    device = next(model.parameters()).device
    generated = input_ids.to(device)

    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]

        # Sample next token using specified strategy
        if temperature == 0 or (top_k == 0 and top_p == 1.0 and temperature == 1.0):
            # Greedy decoding
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            # Sample using temperature, top-k, and/or top-p
            next_token = sample_top_k_top_p(
                next_token_logits, k=top_k, p=top_p, temperature=temperature
            ).unsqueeze(-1)

        generated = torch.cat([generated, next_token], dim=1)

    logits = model(generated)
    return generated, logits


def main() -> None:
    parser = argparse.ArgumentParser(description="FFTNet inference")
    parser.add_argument("--model", help="Model version name to load", metavar="VERSION", nargs="?")
    parser.add_argument("--prompt", default="the quick", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=5, help="Number of tokens to generate")
    parser.add_argument("--mode", choices=["text", "logits", "spectrum"], default="text")
    parser.add_argument("--tokenizer-path", default="tokenizer.json", help="Tokenizer file")

    # Sampling strategy arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 for greedy, >1 for more random)",
    )
    parser.add_argument(
        "--top-k-sampling",
        type=int,
        default=0,
        dest="top_k_sampling",
        help="Top-k sampling: keep only top k tokens (0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling: cumulative probability threshold (1.0 to disable)",
    )

    # For backward compatibility with logits mode
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k predictions to show in logits mode",
    )

    args = parser.parse_args()

    tokenizer = SimpleTokenizer.load(args.tokenizer_path)

    if args.model:
        model, cfg = storage.load_model(Path("weights") / args.model)
    else:
        cfg = load_config("config/fiftynet_config.json", "config/fiftynet_modules.yaml")
        cfg["vocab_size"] = len(tokenizer)
        model = build_model_from_config(cfg)

    tokens = _tokenize(tokenizer, args.prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    # Generate with sampling strategy
    generated, logits = generate(
        model,
        input_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k_sampling,
        top_p=args.top_p,
    )

    if args.mode == "text":
        print(_decode(tokenizer, generated[0]))
    elif args.mode == "logits":
        last_logits = logits[0, -1]
        k = min(args.top_k, last_logits.size(0))
        values, indices = torch.topk(last_logits, k)
        for idx, val in zip(indices.tolist(), values.tolist()):
            word = tokenizer.decode([idx])
            print(f"{word}: {val:.4f}")
    else:  # spectrum
        embeddings = model.embedding(generated)
        plot_embedding_spectrum(embeddings)


if __name__ == "__main__":
    main()
