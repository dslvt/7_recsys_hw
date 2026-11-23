"""Fit on train histories and evaluate on validation unless explicitly asked for test."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Protocol, Self

Catalog = dict[int, list[float]]
Examples = list[tuple[list[int], int]]


class Recommender(Protocol):
    def fit(self, catalog: Catalog, histories: list[list[int]], epochs: int = 1) -> Self: ...
    def recommend(self, history: list[int], k: int) -> list[int]: ...


class Popularity:
    """Count train interactions, including repeats; break ties by numeric ID."""
    def fit(self, catalog: Catalog, histories: list[list[int]], epochs: int = 1) -> Self:
        counts = Counter(item for history in histories for item in history)
        self.ranked = sorted(catalog, key=lambda item: (-counts[item], item))
        return self

    def recommend(self, history: list[int], k: int) -> list[int]:
        return self.ranked[:k]


def load_data(directory: Path, splits: tuple[str, ...]) -> tuple[Catalog, list[list[int]], dict[str, Examples]]:
    def read(name: str):
        with (directory / f'{name}.json').open(encoding='utf-8') as stream:
            return json.load(stream)

    catalog = {int(item): features for item, features in read('catalog').items()}
    histories = list(read('train').values())

    def split(name: str) -> Examples:
        return [(history, target) for history, target in read(name)]

    return catalog, histories, {name: split(name) for name in splits}


def evaluate(model: Recommender, examples: Examples) -> dict[str, float]:
    hits, gain = 0, 0.0
    for history, target in examples:
        ranked = model.recommend(history, k=10)[:10]
        if target in ranked:
            rank = 1 + ranked.index(target)
            hits += 1
            gain += 1 / math.log2(rank + 1)
    count = len(examples) or 1  # Empty splits report zeros.
    return {'Recall@10': hits / count, 'NDCG@10': gain / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('baseline', 'tiger'), default='baseline')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--split', choices=('val', 'test', 'both'), default='val')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    splits = ('val', 'test') if args.split == 'both' else (args.split,)
    catalog, histories, queries = load_data(args.data_dir, splits)
    model: Recommender
    if args.model == 'tiger':
        from submission import TIGER
        model = TIGER(seed=args.seed)
    else:
        model = Popularity()
    try:
        model.fit(catalog, histories, epochs=args.epochs)
        scores = {name: evaluate(model, examples) for name, examples in queries.items()}
    except NotImplementedError as exc:
        parser.exit(2, f'TIGER is unfinished: {exc}\n')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
