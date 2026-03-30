import torch
import torch.nn.functional as F
from typing import List


def postfix_grammar_mask(token_ids: List[int], step: int,
                         binary_set: set, unary_set: set, terminal_set: set,
                         bos_id: int, pad_id: int, eos_id: int,
                         vocab_size: int, max_seq: int) -> torch.Tensor:
    stack = 0
    for tid in token_ids:
        if tid in (bos_id, pad_id): continue
        if tid == eos_id:           break
        if tid in binary_set:       stack -= 1
        elif tid in unary_set:      stack += 0
        elif tid in terminal_set:   stack += 1

    remaining = max_seq - step - 2
    mask      = torch.zeros(vocab_size, dtype=torch.bool)

    if stack <= 0:
        for tid in terminal_set: mask[tid] = True
    elif stack == 1:
        mask[eos_id] = True
        for tid in terminal_set: mask[tid] = True
        if remaining > 1:
            for tid in unary_set: mask[tid] = True
    else:
        for tid in binary_set:   mask[tid] = True
        for tid in unary_set:    mask[tid] = True
        if remaining > stack:
            for tid in terminal_set: mask[tid] = True

    if not mask.any():
        for tid in terminal_set: mask[tid] = True
        mask[eos_id] = True

    return mask


def beam_search(emb: torch.Tensor, decoder,
                bos_id: int, eos_id: int, pad_id: int,
                binary_set: set, unary_set: set, terminal_set: set,
                vocab_size: int, max_seq: int,
                beam_size: int = 16, use_grammar: bool = True,
                device: str = 'cpu') -> List[List[int]]:
    seqs    = [[bos_id]]
    scores  = [0.0]
    done    = []
    done_sc = []

    for step in range(max_seq - 1):
        cands = []
        for seq, sc in zip(seqs, scores):
            if seq[-1] == eos_id:
                done.append(seq); done_sc.append(sc); continue
            if len(seq) >= max_seq - 1:
                done.append(seq + [eos_id]); done_sc.append(sc); continue

            tids = torch.tensor([seq], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = decoder(emb, tids)[0, -1]

            if use_grammar:
                gmask = postfix_grammar_mask(
                    seq, step, binary_set, unary_set, terminal_set,
                    bos_id, pad_id, eos_id, vocab_size, max_seq
                ).to(device)
                logits[~gmask] = float('-inf')

            logits[pad_id] = float('-inf')
            log_probs      = F.log_softmax(logits, dim=-1)
            n_expand       = min(beam_size * 2,
                                 int((log_probs > float('-inf')).sum().item()),
                                 vocab_size)
            if n_expand == 0:
                continue

            top_lp, top_ids = log_probs.topk(n_expand)
            for lp, ti in zip(top_lp.tolist(), top_ids.tolist()):
                if lp == float('-inf'): continue
                cands.append((sc + lp, seq + [ti]))

        if not cands:
            break

        cands.sort(key=lambda x: x[0] / max(len(x[1]), 1) ** 0.6, reverse=True)
        seqs   = [c[1] for c in cands[:beam_size]]
        scores = [c[0] for c in cands[:beam_size]]

    done    += seqs
    done_sc += scores

    if not done:
        return [[bos_id, eos_id]]

    ranked = sorted(range(len(done)),
                    key=lambda i: done_sc[i] / max(len(done[i]), 1) ** 0.6,
                    reverse=True)
    return [done[i] for i in ranked[:beam_size]]
