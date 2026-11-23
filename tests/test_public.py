"""Smoke tests for the supplied runner. Student algorithm examples are in API.md."""
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_public_api_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT))
    from submission import TIGER, beam_search, make_semantic_ids, residual_quantize

    model = TIGER(seed=42)
    assert all(callable(fn) for fn in (model.fit, model.recommend, residual_quantize,
                                       make_semantic_ids, beam_search))


def test_small_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT))
    from run import Popularity, evaluate

    model = Popularity().fit({11: [0.0], 2: [0.1]}, [[11, 11]])
    assert model.recommend([], 10) == [11, 2]
    assert model.recommend([11], 0) == []
    assert evaluate(model, []) == {'Recall@10': 0.0, 'NDCG@10': 0.0}


def test_baseline_cli(tmp_path: Path) -> None:
    # 17 wins by frequency; all other IDs tie and sort numerically. 11 misses.
    files = {
        'catalog': {str(i): [float(i)] for i in [17, *range(1, 12)]},
        'train': {'alice': [17, 17], 'bob': []},
        'val': [[[17], 17], [[], 1], [[], 11]],
        'test': [[[], 9]],  # rank 10 (inclusive)
    }
    for name, data in files.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(data), encoding='utf-8')
    output = subprocess.check_output(
        [sys.executable, str(ROOT / 'run.py'), '--model', 'baseline',
         '--data-dir', str(tmp_path), '--split', 'both'], text=True,
    )
    scores = json.loads(output)
    assert scores['val'] == pytest.approx({
        'Recall@10': 2 / 3, 'NDCG@10': (1 + 1 / math.log2(3)) / 3,
    })
    assert scores['test'] == pytest.approx({'Recall@10': 1, 'NDCG@10': 1 / math.log2(11)})
    (tmp_path / 'test.json').unlink()
    val_only = subprocess.check_output([sys.executable, str(ROOT / 'run.py'),
                                        '--data-dir', str(tmp_path)], text=True)
    assert json.loads(val_only) == {'val': scores['val']}
