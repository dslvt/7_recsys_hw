# TIGER homework: MovieLens 100K next-item data

Distribute the code, not `data/`. Each student downloads their own copy from [GroupLens](https://grouplens.org/datasets/movielens/100k/). Use Python 3.7+; no pip packages, credentials, or random seed are needed.

```sh
# From the root of your cloned or unpacked starter:
python3 -B prepare.py
python3 -B -m unittest discover -s tests -p test_data.py -v
```

The first build downloads `ml-100k.zip` over HTTPS from GroupLens. Later builds verify and reuse `data/ml-100k.zip` offline. To use an already-downloaded original ZIP instead, run `python3 -B prepare.py --zip /path/to/ml-100k.zip`; this also populates the cache. All paths under `data/` are relative to `prepare.py`, not the caller's working directory.

The expected SHA-256 is `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229` (4,924,029 bytes; independently listed in the [TensorFlow Datasets checksum manifest](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/url_checksums/movielens.txt)). Downloaded, cached and explicitly supplied bytes must match before any ZIP member is read. If the cache is corrupt, supply the original with `--zip` or remove `data/ml-100k.zip` and rerun. Do not change the pin to accept a different file.

## Split and runner contract

Read all 100,000 rows of `u.data` as interactions, including one-star ratings. Do not threshold, deduplicate or use the provided random folds. For each user, order events by `(timestamp, item ID)` ascending, both numeric. An item-ID tie-break is deterministic; it does not imply finer timestamps. For a sequence `s` of at least three events, reserve its last two positions:

| Output | JSON contract | Verified size |
|---|---|---:|
| `data/train.json` | Object: string user ID → `s[:-2]`, a list of integer item IDs | 943 users; 98,114 events |
| `data/val.json` | Array of `[s[:-2], s[-2]]`, one pair per eligible user | 943 pairs |
| `data/test.json` | Array of `[s[:-1], s[-1]]`, one pair per eligible user | 943 pairs |
| `data/catalog.json` | Object: string item ID → 35 floats, one entry for every `u.item` row | 1,682 movies |

Both evaluation arrays use ascending numeric user ID order. Test history includes the observed validation target. Neither holdout is a training target for that user. Shorter sequences would stay entirely in train; all users in the pinned release qualify. IDs are original MovieLens IDs: look up an integer item as `catalog[str(item)]`.

Training histories are stored once. The downstream runner should iterate **all 97,171 nonempty training prefixes** per epoch, not just the last position per user:

```python
for seq in train.values():
    for i in range(1, len(seq)):
        history, target = seq[:i], seq[i]
        # Feed this pair to the training batch.
```

## Feature layout (35 floats)

- Coordinates 0 through 18: the 19 binary genre flags from `u.item` in source order, including `unknown` at index 0. Cast each to `0.0` or `1.0`.
- Coordinates 19 through 34: 16 title-token presence flags. Lowercase the raw title bytes using ASCII rules and take maximal `[a-z0-9]+` tokens, including any year. Set bucket `sha256(token).digest()[0] % 16` to `1.0`; repetitions and collisions count once. All other buckets are `0.0`.

No rating, timestamp, popularity, learned vocabulary, or split-dependent statistic enters the vector. JSON is compact ASCII with one trailing LF, including on Windows.

## Attribution and distribution

MovieLens 100K was collected by GroupLens Research at the University of Minnesota. In publications cite F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19. DOI: `10.1145/2827872`.

See `ml-100k/README` inside the ZIP for the full usage license. Redistribution requires separate permission; do not commit or share the archive, extracted records, or generated JSON. The terms require acknowledgment, prohibit implying endorsement, and require prior permission for commercial or revenue-bearing use.
