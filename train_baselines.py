"""Обучение и проверка BERT4Rec. При обучении файл test.json не нужен.

python train_baselines.py fit --device cuda --out new_run
python train_baselines.py test --out new_run  # после выбора весов по валидации
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from torch.nn import functional as F
from seq_baselines import NAMES, SeqModel, load_baseline

BATCH = 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(directory, name):
    return json.loads((directory / (name + '.json')).read_bytes())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def windows(model, histories):
    """Разбить каждую историю на непересекающиеся окна."""
    width = model.length
    rows = []
    for history in histories:
        seq = [model.index[i] for i in history]
        for start in range(0, len(seq), width):
            row = seq[start:start + width]
            rows.append(row + [0] * (width - len(row)))
    return torch.tensor(rows, dtype=torch.long)


def objective_inputs(model, rows):
    # Если в окне не выбран ни один фильм, скрываем последний.
    selected = (torch.rand(rows.shape, device=rows.device) < 0.2) & rows.ne(0)
    ends = rows.ne(0).sum(1) - 1
    empty = ~selected.any(1)
    selected[empty, ends[empty]] = True
    return rows.masked_fill(selected, model.mask), rows.masked_fill(~selected, 0)


def evaluate(model, examples):
    sums = {f'{metric}@{k}': 0.0 for k in (5, 10) for metric in ('Recall', 'NDCG')}
    model.eval()
    for start in range(0, len(examples), BATCH):
        batch = examples[start:start + BATCH]
        ranks = model.rank_batch([h for h, _ in batch], 10)
        for (_, target), recs in zip(batch, ranks):
            if target in recs:
                rank = recs.index(target) + 1
                for k in (5, 10):
                    if rank <= k:
                        sums[f'Recall@{k}'] += 1
                        sums[f'NDCG@{k}'] += 1 / math.log2(rank + 1)
    return {key: value / max(len(examples), 1) for key, value in sums.items()}


def fit(name, data, out, epochs=30, seed=42, device='cpu'):
    out.mkdir(parents=True, exist_ok=True)
    if (out / (name + '.pt')).exists() or epochs < 1:
        raise ValueError('use a fresh output directory and positive epochs')
    torch.manual_seed(seed)
    catalog = {int(i): v for i, v in read(data, 'catalog').items()}
    histories = list(read(data, 'train').values())
    val = read(data, 'val')
    model = SeqModel(name, sorted(catalog)).to(device)
    chunks = windows(model, histories).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    config = dict(name=name, seed=seed, epochs=epochs, batch=BATCH, lr=0.001, weight_decay=0.01,
                  optimizer='AdamW', grad_clip=1.0, bert_mask_probability=0.2,
                  architecture=model.config, selection='maximum validation NDCG@10; earliest epoch on tie')
    sources = [Path(__file__), Path(__file__).with_name('seq_baselines.py')]
    inputs = [data / (s + '.json') for s in ('train', 'catalog', 'val')]
    record = dict(config=config, sha256={p.name: digest(p) for p in sources + inputs},
                  device=device, torch=str(torch.__version__), python=platform.python_version(),
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
                  users=len(histories), events=sum(map(len, histories)), chunks=len(chunks),
                  catalog=len(catalog), validation_pairs=len(val), history=[])
    write(out / (name + '-plan.json'), {k: v for k, v in record.items() if k != 'history'})
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, count, steps = 0., 0, 0
        for ix in torch.randperm(len(chunks)).split(BATCH):
            x, y = objective_inputs(model, chunks[ix])
            active = y.ne(0)
            logits = model.logits(model(x)[active])
            loss = F.cross_entropy(logits, y[active] - 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            n = logits.shape[0]
            total += loss.item() * n
            count += n
            steps += 1
        metrics = evaluate(model, val)
        if metrics['NDCG@10'] > best:
            best = metrics['NDCG@10']
            record['best_epoch'] = epoch
            model.save(out)
        row = dict(epoch=epoch, loss=total / count, targets=count, steps=steps, val=metrics,
                   best_so_far=record['best_epoch'])
        record['history'].append(row)
        print(name, json.dumps(row, allow_nan=False), flush=True)
        write(out / (name + '-fit.json'), record)
    record['checkpoint_sha256'] = digest(out / (name + '.pt'))
    write(out / (name + '-fit.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fit', 'val', 'test'))
    parser.add_argument('--model', choices=NAMES, default='bert4rec')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    name = args.model
    if args.mode == 'fit':
        fit(name, args.data, args.out, args.epochs, args.seed, args.device)
    else:
        catalog = {int(i): v for i, v in read(args.data, 'catalog').items()}
        model = load_baseline(name, args.out, catalog).to(args.device)
        examples = read(args.data, args.mode)
        print(json.dumps(dict(model=name, split=args.mode, count=len(examples),
                              checkpoint_sha256=digest(args.out / (name + '.pt')),
                              split_sha256=digest(args.data / (args.mode + '.json')),
                              metrics=evaluate(model, examples))))


if __name__ == '__main__':
    main()
