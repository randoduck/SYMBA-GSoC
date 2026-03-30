import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, List


def train_one_epoch(encoder, decoder, dataloader: DataLoader,
                    criterion, optimiser, vocab_size: int,
                    pad_id: int, device, grad_clip: float = 1.0) -> float:
    encoder.train(); decoder.train()
    total_loss = 0.0; n_batches = 0

    for batch in dataloader:
        pts    = batch['points'].to(device)
        tids   = batch['encoded'].to(device)
        inp    = tids[:, :-1]
        target = tids[:, 1:]

        optimiser.zero_grad()
        emb    = encoder(pts)
        logits = decoder(emb, inp)
        loss   = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), grad_clip
        )
        optimiser.step()
        total_loss += loss.item(); n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_one_epoch(encoder, decoder, dataloader: DataLoader,
                   criterion, vocab_size: int,
                   pad_id: int, device) -> Tuple[float, float, float]:
    encoder.eval(); decoder.eval()
    total_loss  = 0.0; n_batches  = 0
    tok_correct = 0;   tok_total  = 0
    seq_correct = 0;   seq_total  = 0

    for batch in dataloader:
        pts    = batch['points'].to(device)
        tids   = batch['encoded'].to(device)
        inp    = tids[:, :-1]
        target = tids[:, 1:]
        emb    = encoder(pts)
        logits = decoder(emb, inp)
        loss   = criterion(logits.reshape(-1, vocab_size), target.reshape(-1))
        total_loss += loss.item(); n_batches += 1

        preds = logits.argmax(dim=-1)
        for i in range(preds.shape[0]):
            mask   = target[i] != pad_id
            n_tok  = mask.sum().item()
            n_corr = (preds[i][mask] == target[i][mask]).sum().item()
            tok_correct += n_corr; tok_total += n_tok
            if n_corr == n_tok: seq_correct += 1
            seq_total += 1

    return (total_loss / max(n_batches, 1),
            tok_correct / max(tok_total, 1),
            seq_correct / max(seq_total, 1))


@torch.no_grad()
def greedy_decode(emb: torch.Tensor, decoder,
                  bos_id: int, eos_id: int, pad_id: int,
                  i2t: dict, max_steps: int = 48,
                  device: str = 'cpu') -> List[str]:
    ids = [bos_id]
    for _ in range(max_steps):
        tids   = torch.tensor([ids], dtype=torch.long, device=device)
        logits = decoder(emb, tids)[0, -1]
        logits[pad_id] = float('-inf')
        nxt    = logits.argmax().item()
        ids.append(nxt)
        if nxt == eos_id: break

    toks = [i2t.get(t, f'<{t}>') for t in ids if t != pad_id]
    if '<EOS>' in toks: toks = toks[:toks.index('<EOS>')]
    if toks and toks[0] == '<BOS>': toks = toks[1:]
    return toks


def train_refiner_one_epoch(encoder, refiner, dataloader: DataLoader,
                            criterion, optimiser, vocab_size: int,
                            pad_id: int, mask_frac: float = 0.15,
                            device: str = 'cpu') -> float:
    encoder.eval(); refiner.train()
    total_loss = 0.0; n_batches = 0

    for pts, enc in dataloader:
        pts = pts.to(device); enc = enc.to(device)
        with torch.no_grad():
            emb = encoder(pts)

        B, T     = enc.shape
        mask_pos = []
        for b in range(B):
            non_pad   = (enc[b] != pad_id).nonzero(as_tuple=True)[0].tolist()
            n_mask    = max(1, int(len(non_pad) * mask_frac))
            positions = random.sample(non_pad, min(n_mask, len(non_pad)))
            mask_pos.append(positions)

        optimiser.zero_grad()
        logits = refiner(enc, emb, mask_pos)
        loss   = criterion(logits.reshape(-1, vocab_size), enc.reshape(-1))
        loss.backward(); optimiser.step()
        total_loss += loss.item(); n_batches += 1

    return total_loss / max(n_batches, 1)
