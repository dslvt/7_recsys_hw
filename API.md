# Public Python API

From this directory with Python 3.11:

```sh
python3.11 -m pip install -r requirements.txt
python3.11 -m pytest -q tests/test_public.py
python3.11 run.py --model baseline --data-dir data
python3.11 run.py --model tiger --data-dir data --seed 42 --epochs 1  # after the TODOs
```

**Runner.** Each invocation fits once on the catalog and train histories. By default only validation is loaded and scored. Use `--split test` or `--split both` after fixing the configuration. `--model` defaults to `baseline`, and the default data directory is `data/` next to `run.py`. Baseline counts all train interactions, including repeats; it ranks the entire catalog by decreasing count, breaking ties by ascending numeric item ID. Neither model should filter seen items. TIGER is a CPU PyTorch implementation task, not an alias for the baseline.

| File | JSON value | Runner conversion |
|---|---|---|
| `catalog.json` | Object: string item ID → list of floats (same feature dimension) | Convert keys to `int`; IDs need not be contiguous. |
| `train.json` | Object: string user ID → chronological list of integer item IDs | Pass the lists separately, without user keys. |
| `val.json`, `test.json` | Array of `[history, target]` pairs; history is a list of integer item IDs, target is one integer item ID | Evaluate each pair independently, including empty histories. |

For each target, let `r` be its one-based rank among the first ten recommendations. A hit contributes `1` to Recall@10 and `1/log2(r+1)` to NDCG@10; a miss contributes zero to both. Average over **all pairs** in the split. An empty split reports both scores as zero.

## Implement in `submission.py`

- `residual_quantize(x, codebooks) -> (codes, quantized)`: for floating CPU tensors `x[N,D]` and `codebooks[L,K,D]`, choose the nearest entry by squared L2 distance to each row's **current residual** at each level. Start at `x`, subtract each selected vector, and choose the lowest index on a tie. Return `torch.long` indices `[N,L]` and the sum of selected vectors `[N,D]` (not the remaining residual).
- `make_semantic_ids(item_codes: dict[int, tuple[int,...]]) -> dict[int, tuple[int,...]]`: group items with identical codes. Append the zero-based rank in ascending numeric item ID order **within each group**, including a `0` suffix for a singleton.
- `beam_search(step, valid_ids, beam_size, k) -> list[tuple[int,...]]`: `valid_ids` is a list of unique, equal-length, nonempty tuples of nonnegative local token indices. Start with `()` at score zero. `step(prefix: tuple[int,...])` returns 1-D logits over the **full** next-token vocabulary. Apply `log_softmax` before restricting to valid catalog continuations; never renormalize among valid children. Add each next token's log probability to the prefix score. At every depth pool all candidates, order by descending cumulative score then ascending lexicographic tuple, and retain at most `beam_size`. Return complete IDs in that order, up to `min(k, beam_size, len(valid_ids))`.
- `TIGER(seed: int = 42).fit(catalog: dict[int,list[float]], histories: list[list[int]], epochs: int = 1) -> self`: train the feature tokenizer and the history-to-semantic-ID model on CPU. Keep semantic IDs and their inverse for inference. The neural module classes in the starter are optional scaffolding, not public interfaces.
- `TIGER.recommend(history: list[int], k: int) -> list[int]`: after fit, return **exactly** `min(k, len(catalog))` distinct catalog item IDs in ranked order for nonnegative `k`. Support an empty history; seen items remain eligible. If decoding returns too few items, retry from the root with a wider beam, up to catalog size.

## Examples to try after implementing the TODOs

These are separate from the starter smoke tests.

```python
import torch
from submission import TIGER, beam_search, make_semantic_ids, residual_quantize

x = torch.tensor([[1.0], [3.0]])
books = torch.tensor([[[0.0], [2.0]], [[0.0], [1.0]]])
codes, q = residual_quantize(x, books)
assert codes.tolist() == [[0, 1], [1, 1]]  # first row ties at level 0
assert q.tolist() == [[1.0], [3.0]]
assert make_semantic_ids({9: (4,), 2: (4,), 5: (1,)}) == {
    2: (4, 0), 9: (4, 1), 5: (1, 0)}

# Prefix (0,) puts most probability on invalid token 1. Masking before
# normalization would incorrectly tie the two completions.
logits = {(): [0., 0.], (0,): [0., 8.], (1,): [0., 0.]}
assert beam_search(lambda p: torch.tensor(logits[p]), [(0, 0), (1, 0)], 2, 2) == [
    (1, 0), (0, 0)]

model = TIGER().fit({2: [0.0], 9: [1.0]}, [[2, 9], [9, 2]])
assert sorted(model.recommend([], 10)) == [2, 9]
```
