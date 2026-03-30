#!/usr/bin/env python3
import sys, os, json, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter

from config import *
from models.encoder   import SineKANEncoder
from models.decoder   import ARDecoder
from data.synthetic_gen import generate_synthetic
from data.vocab       import build_vocab, encode_sequence, save_vocab
from data.dataset     import SyntheticDataset, RealDataset, load_real_clouds, collate_fn
from training.loops   import train_one_epoch, eval_one_epoch, greedy_decode

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
t0_total = time.time()

print(f"\n{'='*60}")
print(f"  STEP 1: Generate {N_SYNTH:,} synthetic equations")
print(f"{'='*60}")
syn_eqs = generate_synthetic(N_SYNTH, N_POINTS, seed=42)

random_order = list(range(len(syn_eqs)))
import random; random.shuffle(random_order)
n_tr = int(len(syn_eqs)*0.80); n_vl = int(len(syn_eqs)*0.10)
syn_train = [syn_eqs[i] for i in random_order[:n_tr]]
syn_val   = [syn_eqs[i] for i in random_order[n_tr:n_tr+n_vl]]
syn_test  = [syn_eqs[i] for i in random_order[n_tr+n_vl:]]

print(f"\n{'='*60}"); print(f"  STEP 2: Build unified vocab"); print(f"{'='*60}")
with open(REAL_LABEL) as f: real_lab = json.load(f)

all_token_lists = [e['tokens'] for e in syn_eqs]
for sp in ['train','val','test']:
    for e in real_lab[sp]: all_token_lists.append(e['tokens'])

vocab, t2i = build_vocab(all_token_lists)
i2t    = {v: k for k, v in t2i.items()}
V      = len(vocab)
pad_id = t2i['<PAD>']; bos_id = t2i['<BOS>']; eos_id = t2i['<EOS>']
MAX_SEQ = max(max(len(e['tokens']) for e in syn_eqs),
              real_lab['metadata']['max_len'], 50)
print(f"  Vocab: {V}  MaxSeq: {MAX_SEQ}")

for e in syn_eqs:
    e['encoded'] = encode_sequence(e['tokens'], t2i, MAX_SEQ)
for sp in ['train','val','test']:
    for e in real_lab[sp]:
        e['encoded'] = encode_sequence(e['tokens'], t2i, MAX_SEQ)

save_vocab(vocab, t2i, VOCAB_PATH)
print(f"  Vocab saved -> {VOCAB_PATH}")

print(f"\n{'='*60}"); print(f"  STEP 3: Build models"); print(f"{'='*60}")
encoder = SineKANEncoder(MAX_D, EMBED_DIM, grid_size=5).to(device)
decoder = ARDecoder(V, EMBED_DIM, GPT_DIM, N_LAYERS, N_HEADS, MAX_SEQ+10, DROPOUT).to(device)
enc_p   = sum(p.numel() for p in encoder.parameters())
dec_p   = sum(p.numel() for p in decoder.parameters())
print(f"  Enc: {enc_p:,}  Dec: {dec_p:,}  Total: {enc_p+dec_p:,}")

print(f"\n{'='*60}")
print(f"  PHASE 1: Pre-train on {len(syn_train):,} synthetic equations")
print(f"{'='*60}")

syn_tr_ds = SyntheticDataset(syn_train, MAX_D, N_SAMPLE, train=True)
syn_vl_ds = SyntheticDataset(syn_val,   MAX_D, N_SAMPLE, train=False)
syn_TL    = DataLoader(syn_tr_ds, SYN_BATCH, shuffle=True,  collate_fn=collate_fn, drop_last=True)
syn_VL    = DataLoader(syn_vl_ds, SYN_BATCH, shuffle=False, collate_fn=collate_fn)
print(f"  {len(syn_TL)} batches/epoch")

crit  = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
opt   = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()),
                           lr=SYN_LR, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, SYN_EPOCHS, eta_min=1e-5)

best_val = float('inf'); patience_left = SYN_PATIENCE
syn_tr_h = []; syn_vl_h = []
print(f"\n{'Ep':>4} {'TrLoss':>8} {'VlLoss':>8} {'TokAcc':>7} {'SeqAcc':>7}")
print('-'*45)

for ep in range(SYN_EPOCHS):
    t_ep = time.time()
    tl = train_one_epoch(encoder, decoder, syn_TL, crit, opt, V, pad_id, device)
    vl, ta, sa = eval_one_epoch(encoder, decoder, syn_VL, crit, V, pad_id, device)
    sched.step(); syn_tr_h.append(tl); syn_vl_h.append(vl)
    tag = ''
    if vl < best_val:
        best_val = vl; patience_left = SYN_PATIENCE
        torch.save(encoder.state_dict(), ENC_PATH)
        torch.save(decoder.state_dict(), DEC_PATH)
        tag = ' *'
    else: patience_left -= 1
    print(f"{ep+1:4d} {tl:8.4f} {vl:8.4f} {ta*100:6.1f}% {sa*100:6.1f}%  "
          f"{time.time()-t_ep:.0f}s{tag}", flush=True)
    if patience_left <= 0 and ep >= 20:
        print(f"  Early stop at ep {ep+1}"); break

encoder.load_state_dict(torch.load(ENC_PATH, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(DEC_PATH, map_location=device, weights_only=True))
print("Phase 1 best weights loaded.")

print(f"\n{'='*60}"); print(f"  PHASE 2: Fine-tune on Feynman"); print(f"{'='*60}")
real_clouds = load_real_clouds(REAL_CLOUD, MAX_D)
real_tr_ds  = RealDataset(real_lab['train'], real_clouds, N_SAMPLE, train=True)
real_vl_ds  = RealDataset(real_lab['val'],   real_clouds, N_SAMPLE, train=False)
real_TL     = DataLoader(real_tr_ds, FT_BATCH, shuffle=True,  collate_fn=collate_fn)
real_VL     = DataLoader(real_vl_ds, FT_BATCH, shuffle=False, collate_fn=collate_fn)
print(f"  {len(real_tr_ds)} train / {len(real_vl_ds)} val")

crit  = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.05)
opt   = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()),
                           lr=FT_LR, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 10)

best_val = float('inf'); patience_left = FT_PATIENCE
ft_tr_h = []; ft_vl_h = []
print(f"\n{'Ep':>4} {'TrLoss':>8} {'VlLoss':>8} {'TokAcc':>7} {'SeqAcc':>7}")
print('-'*45)

for ep in range(FT_EPOCHS):
    t_ep = time.time()
    tl = train_one_epoch(encoder, decoder, real_TL, crit, opt, V, pad_id, device)
    vl, ta, sa = eval_one_epoch(encoder, decoder, real_VL, crit, V, pad_id, device)
    sched.step(vl); ft_tr_h.append(tl); ft_vl_h.append(vl)
    tag = ''
    if vl < best_val:
        best_val = vl; patience_left = FT_PATIENCE
        torch.save(encoder.state_dict(), ENC_PATH)
        torch.save(decoder.state_dict(), DEC_PATH)
        tag = ' *'
    else: patience_left -= 1
    print(f"{ep+1:4d} {tl:8.4f} {vl:8.4f} {ta*100:6.1f}% {sa*100:6.1f}%  "
          f"{time.time()-t_ep:.0f}s{tag}", flush=True)
    if patience_left <= 0 and ep >= 30:
        print(f"  Early stop at ep {ep+1}"); break

encoder.load_state_dict(torch.load(ENC_PATH, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(DEC_PATH, map_location=device, weights_only=True))

real_te_ds = RealDataset(real_lab['test'], real_clouds, N_SAMPLE, train=False)
encoder.eval(); decoder.eval()
tf_ok = 0; ar_ok = 0; n = 0
for idx in range(len(real_te_ds)):
    item = real_te_ds[idx]
    pts  = item['points'].unsqueeze(0).to(device)
    tids = item['encoded'].unsqueeze(0).to(device)
    true_toks = [i2t.get(t.item(),'?') for t in tids[0] if t.item() not in (pad_id,)]
    true_toks = [t for t in true_toks if t not in ('<BOS>','<EOS>','<PAD>')]
    with torch.no_grad():
        emb = encoder(pts)
        tf_logits = decoder(emb, tids[:, :-1])
    tf_toks = [i2t.get(t,'?') for t in tf_logits.argmax(-1).squeeze(0).tolist()
               if i2t.get(t,'?') not in ('<PAD>',)]
    ar_toks = greedy_decode(emb, decoder, bos_id, eos_id, pad_id, i2t, MAX_SEQ-2, str(device))
    tf_ok += (tf_toks == true_toks); ar_ok += (ar_toks == true_toks); n += 1

print(f"\n  Test: TF={tf_ok}/{n} ({tf_ok/n*100:.1f}%)  AR={ar_ok}/{n} ({ar_ok/n*100:.1f}%)")

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(syn_tr_h, label='Train'); axs[0].plot(syn_vl_h, label='Val')
axs[0].set_title(f'Phase 1: Synthetic ({len(syn_train):,} eqs)')
axs[0].legend(); axs[0].grid(True)
axs[1].plot(ft_tr_h, label='Train'); axs[1].plot(ft_vl_h, label='Val')
axs[1].set_title('Phase 2: Feynman Fine-tune')
axs[1].legend(); axs[1].grid(True)
plt.tight_layout(); plt.savefig(PLOT_PATH, dpi=120); plt.close()

print(f"\nWeights -> {ENC_PATH}, {DEC_PATH}")
print(f"Plot    -> {PLOT_PATH}")
print(f"Total time: {(time.time()-t0_total)/60:.1f} min")
