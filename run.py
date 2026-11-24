"""Evaluate a fitted or checkpoint-backed recommender on validation by default."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Protocol, Self

Catalog = dict[int, list[float]]
Examples = list[tuple[list[int], int]]


class Recommender(Protocol):
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


def evaluate(model: Recommender, examples: Examples, catalog: Catalog | None = None) -> dict[str, float]:
    hits_by_rank = Counter()
    for query, (history, target) in enumerate(examples, start=1):
        ranked = model.recommend(history, k=10)
        if (not isinstance(ranked, list) or len(ranked) > 10
                or any(type(item) is not int for item in ranked)
                or len(set(ranked)) != len(ranked)):
            raise ValueError(f'query {query}: recommend must return a list of up to 10 distinct integer IDs')
        if catalog is not None and (len(ranked) != min(10, len(catalog))
                                    or any(item not in catalog for item in ranked)):
            raise ValueError(f'query {query}: recommend must return min(10, catalog size) catalog IDs')
        if target in ranked:
            hits_by_rank[1 + ranked.index(target)] += 1
    count = len(examples) or 1  # Empty splits report zeros.
    scores = {}
    for k in (5, 10):
        ranks = range(1, k + 1)
        scores[f'Recall@{k}'] = sum(hits_by_rank[r] for r in ranks) / count
        scores[f'NDCG@{k}'] = math.fsum(hits_by_rank[r] / math.log2(r + 1) for r in ranks) / count
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--model', choices=('baseline', 'tiger', 'sasrec', 'bert4rec'), default='baseline')
    parser.add_argument('--data-dir', type=Path, default=root / 'data')
    parser.add_argument('--checkpoint-dir', type=Path, default=root / 'checkpoints')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--split', choices=('val', 'test', 'both'), default='val')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    splits = ('val', 'test') if args.split == 'both' else (args.split,)
    catalog, histories, queries = load_data(args.data_dir, splits)
    model: Recommender
    try:
        if args.model in ('sasrec', 'bert4rec'):
            from seq_baselines import load_baseline
            model = load_baseline(args.model, args.checkpoint_dir, catalog)
        elif args.model == 'tiger':
            from submission import TIGER
            model = TIGER(seed=args.seed).fit(catalog, histories, epochs=args.epochs)
        else:
            model = Popularity().fit(catalog, histories)
        scores = {name: evaluate(model, examples, catalog) for name, examples in queries.items()}
    except NotImplementedError as exc:
        parser.exit(2, f'{args.model} is unfinished: {exc}\n')
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(scores, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
