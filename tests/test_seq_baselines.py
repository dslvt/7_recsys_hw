"""Architecture and portable public checkpoint contracts on tiny invented catalogs."""
import pytest
import torch

from seq_baselines import SeqModel, load_baseline


def test_roundtrip_and_ties(tmp_path):
    ids = [2, 11, 123]
    model = SeqModel('bert4rec', ids, width=8, length=4, heads=2, layers=1).eval()
    for param in model.parameters():
        param.data.zero_()
    model.save(tmp_path)
    loaded = load_baseline('bert4rec', tmp_path, {i: [] for i in reversed(ids)})
    for hist in ([], [123, 2], [11] * 20):
        assert loaded.recommend(hist, 10) == ids
        assert loaded.recommend(hist, 2) == ids[:2]
    with pytest.raises(ValueError):
        load_baseline('bert4rec', tmp_path, {3: []})


def test_attention_direction_and_query():
    torch.manual_seed(7)
    model = SeqModel('bert4rec', [11, 22, 55], width=8, length=4, heads=2, layers=1).eval()
    x = torch.tensor([[1, 2, 3], [1, 2, 1]])
    h = model(x)
    assert not torch.equal(h[0, 0], h[1, 0])
    tokens = model.query_tokens([22] * 8)
    assert len(tokens) == 4
    assert tokens[-1] == model.mask


def test_fit_without_test_file(tmp_path):
    import json
    from train_baselines import fit, evaluate
    catalog = {2: [], 11: [], 123: []}
    val = [[[2, 11], 123], [[2], 2]]
    for key, obj in [('catalog', catalog), ('train', {'a': [2, 11, 123] * 25}), ('val', val)]:
        (tmp_path / (key + '.json')).write_text(json.dumps(obj))
    out = tmp_path / 'out'
    result = fit('bert4rec', tmp_path, out, epochs=2, seed=42, device='cpu')
    assert len(result['history']) == 2
    assert all(row['targets'] > 0 for row in result['history'])
    selected = max(result['history'], key=lambda r: r['val']['NDCG@10'])
    assert result['best_epoch'] == selected['epoch']
    assert evaluate(load_baseline('bert4rec', out, catalog), val) == selected['val']
