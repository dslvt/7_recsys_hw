"""BERT4Rec. Токен 0 задаёт отступ, токены от 1 до N обозначают фильмы по порядку ID."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

NAMES = ('bert4rec',)


class SeqModel(nn.Module):
    def __init__(self, name, ids, width=64, length=50, heads=2, layers=2, dropout=0.2):
        super().__init__()
        if name not in NAMES or not ids or list(ids) != sorted(set(ids)):
            raise ValueError('expected a supported model and sorted unique catalog IDs')
        self.name, self.ids = name, list(ids)
        self.index = {item: i + 1 for i, item in enumerate(ids)}
        self.mask = len(ids) + 1
        self.config = dict(width=width, length=length, heads=heads, layers=layers, dropout=dropout)
        self.length = length
        self.items = nn.Embedding(len(ids) + 2, width, padding_idx=0)
        self.positions = nn.Embedding(length, width)
        self.input_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout, 'gelu',
                                       batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
        self.bias = nn.Parameter(torch.zeros(len(ids)))
        nn.init.normal_(self.items.weight, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def forward(self, tokens):
        length = tokens.size(1)
        pos = torch.arange(length, device=tokens.device)
        x = self.dropout(self.input_norm(self.items(tokens) + self.positions(pos)))
        pad = tokens.eq(0)
        for layer in self.blocks:
            x = layer(x, src_key_padding_mask=pad)
        return self.norm(x)

    def logits(self, hidden):
        return F.linear(self.head(hidden), self.items.weight[1:self.mask], self.bias)

    def query_tokens(self, history):
        tokens = [self.index[i] for i in history[-self.length:]]
        tokens.append(self.mask)
        return tokens[-self.length:]

    @torch.inference_mode()
    def rank_batch(self, histories, k):
        if k < 0:
            raise ValueError('k must be nonnegative')
        seq = [self.query_tokens(h) for h in histories]
        n = max(map(len, seq))
        device = self.items.weight.device
        tokens = torch.tensor([s + [0] * (n - len(s)) for s in seq], device=device)
        last = torch.tensor([len(s) - 1 for s in seq], device=device)
        hidden = self(tokens)[torch.arange(len(seq), device=device), last]
        order = self.logits(hidden).argsort(dim=-1, descending=True, stable=True)[:, :k]
        return [[self.ids[j] for j in row] for row in order.tolist()]

    def recommend(self, history: list[int], k: int) -> list[int]:
        return self.rank_batch([history], k)[0]

    def save(self, directory):
        path = Path(directory) / (self.name + '.pt')
        torch.save(dict(version=1, name=self.name, ids=self.ids, config=self.config,
                        state={k: v.detach().cpu() for k, v in self.state_dict().items()}), path)


def load_baseline(name: str, directory: Path, catalog: dict[int, list[float]]) -> SeqModel:
    if name not in NAMES:
        raise ValueError('unknown baseline')
    data = torch.load(Path(directory) / (name + '.pt'), weights_only=True, map_location='cpu')
    if data['version'] != 1 or data['name'] != name or data['ids'] != sorted(catalog):
        raise ValueError('incompatible checkpoint or item mapping')
    model = SeqModel(name, data['ids'], **data['config'])
    for value in data['state'].values():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError('invalid checkpoint tensor')
    model.load_state_dict(data['state'], strict=True)
    return model.eval()
