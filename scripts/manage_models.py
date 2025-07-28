import argparse
from pathlib import Path
from typing import List

from model import FFTNet
from fftnet.utils.storage import save_model, load_model

WEIGHTS_DIR = Path('weights')


def list_models() -> List[str]:
    if not WEIGHTS_DIR.exists():
        return []
    names = {p.stem for p in WEIGHTS_DIR.glob('*.safetensors')}
    return sorted(names)


def delete_model(version: str) -> bool:
    removed = False
    weights = WEIGHTS_DIR / f"{version}.safetensors"
    config = WEIGHTS_DIR / f"{version}_config.json"
    if weights.exists():
        weights.unlink()
        removed = True
    if config.exists():
        config.unlink()
        removed = True
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description='Manage FFTNet model versions')
    parser.add_argument('--list', action='store_true', help='List saved versions')
    parser.add_argument('--save', metavar='VERSION', help='Save a new version')
    parser.add_argument('--load', metavar='VERSION', help='Load a version')
    parser.add_argument('--delete', metavar='VERSION', help='Delete a version')
    args = parser.parse_args()

    if args.list:
        versions = list_models()
        if versions:
            print('\n'.join(versions))
        else:
            print('No saved models')
        return

    if args.save:
        config = {'vocab_size': 10, 'dim': 16, 'num_blocks': 1}
        model = FFTNet(**config)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        save_model(model, str(WEIGHTS_DIR / args.save), config)
        print(f'Saved model as {args.save}')
        return

    if args.load:
        try:
            _, cfg = load_model(str(WEIGHTS_DIR / args.load))
        except FileNotFoundError:
            print(f'Model {args.load} not found')
        else:
            print(f'Loaded {args.load}: {cfg}')
        return

    if args.delete:
        if delete_model(args.delete):
            print(f'Deleted model {args.delete}')
        else:
            print(f'Model {args.delete} not found')
        return

    parser.print_help()


if __name__ == '__main__':
    main()
