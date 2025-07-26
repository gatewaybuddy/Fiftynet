import torch
from model import FFTNet
from scripts.main import plot_embedding_spectrum


def main():
    model = FFTNet(vocab_size=100, dim=64, num_blocks=1)
    input_ids = torch.randint(0, 100, (1, 16))
    embeddings = model.embedding(input_ids)
    plot_embedding_spectrum(embeddings)
    logits = model(input_ids)
    print(logits.shape)


if __name__ == "__main__":
    main()
