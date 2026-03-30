#!/usr/bin/env python3
import sys, os, json, random, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from models.encoder   import SineKANEncoder
from models.decoder   import ARDecoder
from models.birefiner import BiRefiner
from data.vocab       import load_vocab
from data.dataset     import load_real_clouds, collate_fn, RealDataset
from pipeline.beam_search import beam_search
from pipeline.bfgs        import best_candidate_r2
from pipeline.gp_wrapper  import run_pigp
from training.loops       import train_refiner_one_epoch

torch.manual_seed(42); np.random.seed(42); random.seed(42)
device = torch.device('cpu')
t_start = time.time()
print(f"Device: {device}")

vocab, t2i, i2t = load_vocab(VOCAB_PATH)
V       = len(vocab)
pad_id  = t2i['<PAD>']; bos_id = t2i['<BOS>']; eos_id = t2i['<EOS>']
MAX_SEQ = 50

with open(REAL_LABEL) as f: real_lab = json.load(f)
with open(REAL_CLOUD.replace('clouds/isha_data_clouds.json',
                              'clouds/isha_data_clouds.json')) as f:
    raw_clouds = json.load(f)

unk_id = t2i.get('<UNK>', pad_id)
def enc(toks): return ([t2i.get(t, unk_id) for t in toks] + [pad_id]*MAX_SEQ)[:MAX_SEQ]
for sp in ['train','val','test']:
    for e in real_lab[sp]: e['encoded'] = enc(e['tokens'])

cloud_lookup = load_real_clouds(REAL_CLOUD, MAX_D)
cloud_raw    = {e['filename']: e for e in raw_clouds}
all_entries  = real_lab['train'] + real_lab['val'] + real_lab['test']
var_lookup   = {e['filename']: e.get('variables', []) for e in all_entries}

BINARY_SET   = {t2i[t] for t in t2i if t.startswith('OP_')}
UNARY_SET    = {t2i[t] for t in t2i if t.startswith('FUNC_')}
TERMINAL_SET = {t2i[t] for t in t2i if t.startswith('VAR_') or
                t.startswith('CONST_') or t == '<C>'}

print(f"Vocab: {V}  |  Clouds: {len(cloud_lookup)}")

encoder = SineKANEncoder(MAX_D, EMBED_DIM).to(device)
encoder.load_state_dict(torch.load(ENC_PATH, map_location=device, weights_only=True))
decoder = ARDecoder(V, EMBED_DIM, GPT_DIM, N_LAYERS, N_HEADS, MAX_SEQ+10, DROPOUT).to(device)
decoder.load_state_dict(torch.load(DEC_PATH, map_location=device, weights_only=True))
encoder.eval(); decoder.eval()
print("Encoder + Decoder loaded.")

print("\nTraining BiRefiner...")
refiner = BiRefiner(V, EMBED_DIM, GPT_DIM, 2, 4, MAX_SEQ+10).to(device)

class RefDs(torch.utils.data.Dataset):
    def __init__(self, entries):
        self.items = [e for e in entries if e['filename'] in cloud_lookup]
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        e   = self.items[idx]
        pts = cloud_lookup[e['filename']][:N_SAMPLE]
        if pts.shape[0] < N_SAMPLE:
            pts = torch.cat([pts, torch.zeros(N_SAMPLE-pts.shape[0], pts.shape[1])], 0)
        return pts, torch.tensor(e['encoded'], dtype=torch.long)

def ref_collate(b): return torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b])
ref_dl   = DataLoader(RefDs(real_lab['train']+real_lab['val']), 8,
                      shuffle=True, collate_fn=ref_collate)
ref_crit = nn.CrossEntropyLoss(ignore_index=pad_id)
ref_opt  = torch.optim.AdamW(refiner.parameters(), lr=REFINER_LR, weight_decay=1e-4)

for ep in range(REFINER_EPOCHS):
    loss = train_refiner_one_epoch(encoder, refiner, ref_dl, ref_crit, ref_opt,
                                   V, pad_id, REFINER_MASK, str(device))
    if (ep+1) % 10 == 0:
        print(f"  Refiner ep {ep+1}/{REFINER_EPOCHS}: loss={loss:.4f}", flush=True)
refiner.eval()

def decode_ids(ids):
    toks = [i2t.get(t.item() if hasattr(t,'item') else int(t), '?') for t in ids]
    toks = [t for t in toks if t != '<PAD>']
    if '<EOS>' in toks: toks = toks[:toks.index('<EOS>')]
    if toks and toks[0] == '<BOS>': toks = toks[1:]
    return toks

import torch.nn.functional as F

def refine(seq_ids, emb):
    tids = torch.tensor([seq_ids + [pad_id]*(MAX_SEQ-len(seq_ids))],
                        dtype=torch.long, device=device)[:, :MAX_SEQ]
    for _ in range(REFINER_PASSES):
        with torch.no_grad(): logits = refiner(tids, emb)
        conf    = F.softmax(logits, -1).max(-1).values[0]
        non_pad = (tids[0] != pad_id).nonzero(as_tuple=True)[0].tolist()
        if not non_pad: break
        conf_vals = sorted([(p, conf[p].item()) for p in non_pad], key=lambda x: x[1])
        n_mask    = max(1, len(non_pad)//5)
        mask_pos  = [[c[0] for c in conf_vals[:n_mask]]]
        with torch.no_grad(): logits = refiner(tids, emb, mask_pos)
        for p in mask_pos[0]: tids[0,p] = logits.argmax(-1)[0,p]
    return decode_ids(tids[0].tolist())

test_entries = real_lab['test']
print(f"\n{'='*75}")
print(f"  PIPELINE: {len(test_entries)} test equations")
print(f"  Beam={BEAM_SIZE}  GP=pop{GP_POP}×{GP_GENS}gen")
print(f"{'='*75}\n")

results = []
for entry in test_entries:
    fn       = entry['filename']
    true_toks = [t for t in entry['tokens'] if t not in ('<BOS>','<EOS>','<PAD>')]
    encoded   = torch.tensor(entry['encoded'], dtype=torch.long).unsqueeze(0)

    if fn not in cloud_lookup: print(f"  {fn} SKIPPED"); continue

    pts      = cloud_lookup[fn][:N_SAMPLE].unsqueeze(0)
    var_names = var_lookup.get(fn, cloud_raw[fn].get('inputs', []))
    raw_pts   = cloud_raw.get(fn, {}).get('data', [])

    t0 = time.time()
    with torch.no_grad(): emb = encoder(pts)

    with torch.no_grad(): tf_logits = decoder(emb, encoded[:, :-1])
    tf_toks = decode_ids(tf_logits.argmax(-1).squeeze(0).tolist())

    beam_seqs  = beam_search(emb, decoder, bos_id, eos_id, pad_id,
                              BINARY_SET, UNARY_SET, TERMINAL_SET,
                              V, MAX_SEQ, BEAM_SIZE, use_grammar=True)
    beam_cands = [decode_ids(s) for s in beam_seqs]

    ref_cands = [refine(s, emb) for s in beam_seqs[:4]]

    all_cands = [tf_toks] + beam_cands[:5] + ref_cands[:4]
    _, _, bfgs_r2 = best_candidate_r2(all_cands, var_names, raw_pts, BFGS_RESTARTS)

    gp_r2, gp_expr = run_pigp(all_cands, var_names, cloud_raw[fn],
                               GP_POP, GP_GENS, GP_SEED_RATIO,
                               GP_TOURNAMENT, GP_CROSSOVER, GP_MUTATION, GP_ELITISM)

    dt = time.time() - t0
    best_r2 = max(bfgs_r2, gp_r2)
    results.append({'filename':fn, 'true':' '.join(true_toks[:8]),
                    'tf_acc': sum(p==t for p,t in zip(tf_toks,true_toks))/max(len(true_toks),1),
                    'beam_acc': sum(p==t for p,t in zip(beam_cands[0] if beam_cands else [],true_toks))/max(len(true_toks),1),
                    'bfgs_r2':bfgs_r2, 'gp_r2':gp_r2, 'gp_expr':gp_expr})

    status = 'PERFECT' if best_r2>=0.99 else ('GOOD' if best_r2>=0.5 else 'WEAK')
    print(f"  {fn:<15s}  TF={results[-1]['tf_acc']*100:5.1f}%  "
          f"BFGS={bfgs_r2:6.3f}  GP={gp_r2:6.3f}  [{status}]  ({dt:.0f}s)", flush=True)

N = len(results)
gp_r2s = [r['gp_r2'] for r in results]
best_r2s = [max(r['bfgs_r2'], r['gp_r2']) for r in results]
print(f"\n{'='*60}")
print(f"  FINAL RESULTS ({N} test equations)")
print(f"  GP  : mean={sum(max(g,0) for g in gp_r2s)/N:.3f}  "
      f">=0.5: {sum(g>=0.5 for g in gp_r2s)}/{N}  "
      f">=0.99: {sum(g>=0.99 for g in gp_r2s)}/{N}")
print(f"  Best: mean={sum(max(b,0) for b in best_r2s)/N:.3f}  "
      f">=0.5: {sum(b>=0.5 for b in best_r2s)}/{N}  "
      f">=0.99: {sum(b>=0.99 for b in best_r2s)}/{N}")
print(f"  Time: {(time.time()-t_start)/60:.1f} min")
print(f"{'='*60}")

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'results', 'pipeline_results.json')
with open(out_path, 'w') as f:
    json.dump({'n_test':N, 'results':results,
               'summary': {'gp_mean': sum(max(g,0) for g in gp_r2s)/N,
                            'gp_good': sum(g>=0.5 for g in gp_r2s),
                            'gp_great': sum(g>=0.99 for g in gp_r2s)}}, f, indent=2, default=str)
print(f"Results -> {out_path}")
