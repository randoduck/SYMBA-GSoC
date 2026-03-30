import json
from collections import Counter
from typing import List, Tuple, Dict


SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<C>', '<UNK>']


def build_vocab(token_lists: List[List[str]]) -> Tuple[List[str], Dict[str, int]]:
    counter     = Counter(t for seq in token_lists for t in seq)
    non_special = sorted(
        [t for t in counter if t not in SPECIAL_TOKENS],
        key=lambda t: -counter[t]
    )
    vocab = SPECIAL_TOKENS + non_special
    t2i   = {t: i for i, t in enumerate(vocab)}
    return vocab, t2i


def encode_sequence(tokens: List[str], t2i: Dict[str, int], max_len: int) -> List[int]:
    unk = t2i.get('<UNK>', t2i['<PAD>'])
    ids = [t2i.get(t, unk) for t in tokens][:max_len]
    return ids + [t2i['<PAD>']] * (max_len - len(ids))


def load_vocab(path: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    with open(path) as f:
        data = json.load(f)
    vocab = data['vocab']
    t2i   = {k: int(v) for k, v in data['t2i'].items()}
    i2t   = {v: k for k, v in t2i.items()}
    return vocab, t2i, i2t


def save_vocab(vocab: List[str], t2i: Dict[str, int], path: str) -> None:
    with open(path, 'w') as f:
        json.dump({'vocab': vocab, 't2i': t2i}, f, indent=2)
