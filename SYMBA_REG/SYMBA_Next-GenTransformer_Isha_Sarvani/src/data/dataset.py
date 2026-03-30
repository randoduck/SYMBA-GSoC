import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List


def pad_and_normalise(cloud: np.ndarray, target_dim: int) -> np.ndarray:
    c = np.array(cloud, dtype=np.float32)
    if c.shape[1] < target_dim:
        c = np.pad(c, ((0, 0), (0, target_dim - c.shape[1])))
    elif c.shape[1] > target_dim:
        c = c[:, :target_dim]
    mu = c.mean(0); sd = c.std(0); sd[sd < 1e-8] = 1e-8
    return (c - mu) / sd


class SyntheticDataset(Dataset):
    def __init__(self, entries: List[dict], max_d: int = 10,
                 n_sample: int = 50, train: bool = True):
        self.entries  = entries
        self.max_d    = max_d
        self.n_sample = n_sample
        self.train    = train
        self._clouds  = {}
        for e in entries:
            norm = pad_and_normalise(e['cloud'], max_d)
            self._clouds[e['filename']] = torch.tensor(norm, dtype=torch.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e   = self.entries[idx]
        fn  = e['filename']
        cld = self._clouds[fn]
        N   = cld.shape[0]
        pts = cld[torch.randperm(N)[:self.n_sample]] if self.train else cld[:self.n_sample]
        if pts.shape[0] < self.n_sample:
            pad = torch.zeros(self.n_sample - pts.shape[0], pts.shape[1])
            pts = torch.cat([pts, pad], dim=0)
        return {
            'filename': fn,
            'points':   pts,
            'encoded':  torch.tensor(e['encoded'], dtype=torch.long),
        }


class RealDataset(Dataset):
    def __init__(self, entries: List[dict], cloud_lookup: Dict[str, torch.Tensor],
                 n_sample: int = 50, train: bool = True):
        self.items        = [e for e in entries if e['filename'] in cloud_lookup]
        self.cloud_lookup = cloud_lookup
        self.n_sample     = n_sample
        self.train        = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        e   = self.items[idx]
        fn  = e['filename']
        cld = self.cloud_lookup[fn]
        N   = cld.shape[0]
        pts = cld[torch.randperm(N)[:self.n_sample]] if self.train else cld[:self.n_sample]
        if pts.shape[0] < self.n_sample:
            pad = torch.zeros(self.n_sample - pts.shape[0], pts.shape[1])
            pts = torch.cat([pts, pad], dim=0)
        return {
            'filename': fn,
            'points':   pts,
            'encoded':  torch.tensor(e['encoded'], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        k: torch.stack([x[k] for x in batch])
           if isinstance(batch[0][k], torch.Tensor)
           else [x[k] for x in batch]
        for k in batch[0]
    }


def load_real_clouds(cloud_path: str, max_d: int = 10) -> Dict[str, torch.Tensor]:
    with open(cloud_path) as f:
        raw = json.load(f)
    return {
        e['filename']: torch.tensor(
            pad_and_normalise(e['data'], max_d), dtype=torch.float32
        )
        for e in raw
    }
