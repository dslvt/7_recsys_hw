"""Offline data-layer tests. Run after prepare.py for the full release check."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import prepare

DATA = Path(prepare.__file__).parent / 'data'


class ArchiveTests(unittest.TestCase):
    def test_corrupt_local_zip_is_not_cached_or_parsed(self):
        DATA.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=DATA) as tmp:
            directory = Path(tmp)
            bad = directory / 'wrong.zip'
            bad.write_bytes(b'not MovieLens')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                prepare.build(directory, bad)
            self.assertEqual(list(directory.iterdir()), [bad])


class ReleaseTests(unittest.TestCase):
    @unittest.skipUnless((DATA / 'ml-100k.zip').is_file(), 'build first for real-release checks')
    def test_saved_json_matches_every_source_event(self):
        archive = DATA / 'ml-100k.zip'
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), prepare.SHA256)
        catalog, train, val, test = [json.loads((DATA / (name + '.json')).read_bytes())
                                     for name in ('catalog', 'train', 'val', 'test')]
        with ZipFile(archive) as z:
            ids = {line.split(b'|')[0].decode() for line in z.read('ml-100k/u.item').splitlines()}
            events = [tuple(map(int, line.split())) for line in z.read('ml-100k/u.data').splitlines()]
        self.assertEqual(set(catalog), ids)
        self.assertEqual(len(ids), 1682)
        self.assertTrue(all(len(v) == 35 and all(type(x) is float and x in (0.0, 1.0) for x in v)
                            for v in catalog.values()))
        self.assertEqual(len(events), 100000)
        expected = {}
        for uid, iid, _, ts in sorted(events, key=lambda e: (e[0], e[3], e[1])):
            expected.setdefault(str(uid), []).append(iid)
        self.assertEqual(set(train), set(expected))
        self.assertEqual((len(train), len(val), len(test)), (943, 943, 943))
        for index, (uid, items) in enumerate(expected.items()):
            self.assertEqual(train[uid], items[:-2])
            self.assertEqual(val[index], [items[:-2], items[-2]])
            self.assertEqual(test[index], [items[:-1], items[-1]])
            self.assertTrue(all(str(i) in ids for i in items))


class CatalogTests(unittest.TestCase):
    def test_catalog_uses_genre_order_and_stable_title_buckets(self):
        import prepare

        genres = [1.0, 0.0] + [0.0] * 16 + [1.0]
        row = b'9|Toy STORY! (1995)||||' + b'|'.join(str(int(x)).encode() for x in genres)
        buckets = [float(i in (4, 7, 15)) for i in range(16)]
        catalog = prepare.make_catalog([row])
        self.assertEqual(catalog, {'9': genres + buckets})
        self.assertTrue(all(type(value) is float for value in catalog['9']))


class SplitTests(unittest.TestCase):
    def test_numeric_ties_repeats_and_short_users(self):
        import prepare

        # In particular, item 10 follows 2, and a one-star event still counts.
        ratings = [b'11\t3\t4\t4', b'2\t10\t1\t2', b'11\t6\t1\t2',
                   b'2\t3\t3\t9', b'7\t9\t2\t1', b'2\t2\t1\t2',
                   b'11\t7\t1\t1', b'2\t2\t3\t1']
        train, val, test = prepare.split_ratings(ratings)
        self.assertEqual(train, {'2': [2, 2], '7': [9], '11': [7]})
        self.assertEqual(val, [[[2, 2], 10], [[7], 6]])
        self.assertEqual(test, [[[2, 2, 10], 3], [[7, 6], 3]])
