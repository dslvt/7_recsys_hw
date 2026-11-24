"""Smoke tests for the supplied runner. Student algorithm examples are in API.md."""
import json
import math
from pathlib import Path
import shutil
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
    assert evaluate(model, []) == {'Recall@5': 0.0, 'NDCG@5': 0.0,
                                   'Recall@10': 0.0, 'NDCG@10': 0.0}


def test_one_ranking_for_both_cutoffs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT))
    from run import evaluate

    calls = []

    class Ranked:
        def recommend(self, history, k):
            calls.append((list(history), k))
            return list(range(1, k + 1))

    examples = [([5], 5), ([], 6), ([10], 10), ([11], 11)]
    assert evaluate(Ranked(), examples) == pytest.approx({
        'Recall@5': 1 / 4, 'NDCG@5': 1 / math.log2(6) / 4,
        'Recall@10': 3 / 4,
        'NDCG@10': sum(1 / math.log2(r + 1) for r in (5, 6, 10)) / 4,
    })
    assert calls == [(history, 10) for history, _ in examples]


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
        'Recall@5': 2 / 3, 'NDCG@5': (1 + 1 / math.log2(3)) / 3,
        'Recall@10': 2 / 3, 'NDCG@10': (1 + 1 / math.log2(3)) / 3,
    })
    assert scores['test'] == pytest.approx({'Recall@5': 0, 'NDCG@5': 0,
                                           'Recall@10': 1, 'NDCG@10': 1 / math.log2(11)})
    (tmp_path / 'test.json').unlink()
    val_only = subprocess.check_output([sys.executable, str(ROOT / 'run.py'),
                                        '--data-dir', str(tmp_path)], text=True)
    assert json.loads(val_only) == {'val': scores['val']}


def test_cli_rejects_unknown_model():
    result = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--model', 'unknown'],
                            text=True, capture_output=True)
    assert result.returncode == 2 and not result.stdout
    assert 'invalid choice' in result.stderr
    assert '{baseline,tiger,bert4rec}' in result.stderr


@pytest.mark.parametrize('custom_dir', [False, True])
def test_checkpoint_cli_loads_once_without_fit(tmp_path, custom_dir):
    name = 'bert4rec'
    # Run a copy so the temporary loader wins even if seq_baselines is installed.
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    shutil.copy(ROOT / 'run.py', bundle)
    directory = tmp_path / 'weights' if custom_dir else bundle / 'checkpoints'
    directory.mkdir()
    (directory / 'name').write_text(name)
    (bundle / 'seq_baselines.py').write_text('''
from pathlib import Path

def load_baseline(name, directory, catalog):
    assert isinstance(directory, Path) and (directory / 'name').read_text() == name
    assert catalog == {1: [0.1], 9: [0.9]}
    (directory / 'loaded').touch(exist_ok=False)
    class Frozen:
        def fit(self, *args, **kwargs):
            raise AssertionError('checkpoint model must not be fitted')
        def recommend(self, history, k):
            assert k == 10
            return [9, 1] if history else [1, 9]
    return Frozen()
''')
    for split, data in {'catalog': {'1': [0.1], '9': [0.9]}, 'train': {},
                        'val': [[[9], 9]], 'test': [[[], 9]]}.items():
        (tmp_path / f'{split}.json').write_text(json.dumps(data))
    command = [sys.executable, str(bundle / 'run.py'), '--model', name,
               '--data-dir', str(tmp_path), '--split', 'both']
    if custom_dir:
        command.extend(['--checkpoint-dir', str(directory)])
    scores = json.loads(subprocess.check_output(command, cwd=tmp_path, text=True))
    for split, rank in [('val', 1), ('test', 2)]:
        gain = 1 / math.log2(rank + 1)
        assert scores[split] == pytest.approx({'Recall@5': 1, 'NDCG@5': gain,
                                              'Recall@10': 1, 'NDCG@10': gain})


@pytest.mark.parametrize('ranking', [[1], [1, 1], [1, 3], [True, 2], [1, 2, 3]])
def test_runner_rejects_invalid_ranking(tmp_path, ranking):
    shutil.copy(ROOT / 'run.py', tmp_path)
    (tmp_path / 'submission.py').write_text(f'''
class TIGER:
    def __init__(self, seed): pass
    def fit(self, catalog, histories, epochs): return self
    def recommend(self, history, k): return {ranking!r}
''')
    for name, data in {'catalog': {'1': [0.1], '2': [0.2]}, 'train': {},
                       'val': [[[], 1]]}.items():
        (tmp_path / f'{name}.json').write_text(json.dumps(data))
    result = subprocess.run([sys.executable, str(tmp_path / 'run.py'), '--model', 'tiger',
                             '--data-dir', str(tmp_path)], text=True, capture_output=True)
    assert result.returncode != 0 and not result.stdout
    assert 'recommend' in result.stderr
