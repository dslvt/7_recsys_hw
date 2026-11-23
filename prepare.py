"""Build the TIGER next-item homework data from the pinned MovieLens 100K release."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
SHA256 = '50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229'
ARCHIVE_SIZE = 4924029
DATA = Path(__file__).resolve().parent / 'data'


def make_catalog(lines):
    catalog = {}
    for line in lines:
        fields = line.rstrip(b'\r\n').split(b'|')
        genres = [float(flag) for flag in fields[5:]]
        buckets = [0.0] * 16
        # Byte-oriented ASCII tokens are independent of locale and Unicode version.
        for token in re.findall(rb'[a-z0-9]+', fields[1].lower()):
            buckets[hashlib.sha256(token).digest()[0] % 16] = 1.0
        catalog[str(int(fields[0]))] = genres + buckets
    return catalog


def split_ratings(lines):
    events = defaultdict(list)
    for line in lines:
        user, item, _rating, timestamp = map(int, line.split())
        events[user].append((timestamp, item))

    train, val, test = {}, [], []
    for user in sorted(events):
        items = [item for _, item in sorted(events[user])]
        if len(items) >= 3:
            train[str(user)] = items[:-2]
            val.append([items[:-2], items[-2]])
            test.append([items[:-1], items[-1]])
        else:
            train[str(user)] = items
    return train, val, test


def build(directory, local_zip=None):
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / 'ml-100k.zip'
    source = local_zip if local_zip is not None else cache
    if local_zip is not None or cache.exists():
        with source.open('rb') as stream:
            raw = stream.read(ARCHIVE_SIZE + 1)
    else:
        print(f'Downloading {URL}')
        with urlopen(URL, timeout=60) as stream:
            raw = stream.read(ARCHIVE_SIZE + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise ValueError(f'SHA-256 mismatch: expected {SHA256}, got {digest}. '
                         'Supply an original GroupLens ZIP with --zip.')
    print(f'Verified SHA-256: {digest}')
    with ZipFile(io.BytesIO(raw)) as archive:
        with archive.open('ml-100k/u.item') as lines:
            catalog = make_catalog(lines)
        with archive.open('ml-100k/u.data') as lines:
            train, val, test = split_ratings(lines)

    with TemporaryDirectory(dir=directory) as tmp:
        stage = Path(tmp)
        if not cache.exists() or source.resolve() != cache.resolve():
            (stage / cache.name).write_bytes(raw)
        for name, value in [('catalog', catalog), ('train', train), ('val', val), ('test', test)]:
            text = json.dumps(value, separators=(',', ':'), allow_nan=False)
            (stage / f'{name}.json').write_bytes((text + '\n').encode('ascii'))
        for path in stage.iterdir():
            path.replace(directory / path.name)

    n = sum(map(len, train.values()))
    print(f'{len(catalog)} items, {len(train)} users, {n} train events, '
          f'{n - len(train)} train prefixes, {len(val)} validation and {len(test)} test targets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, help='use an original local ml-100k.zip without network access')
    args = parser.parse_args()
    try:
        build(DATA, args.zip)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
