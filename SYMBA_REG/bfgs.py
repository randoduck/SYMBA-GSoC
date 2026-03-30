import numpy as np
from scipy.optimize import minimize
from typing import List

from .postfix_eval import eval_postfix


def bfgs_fit(tokens: List[str], var_names: List[str],
             raw_pts: list, n_restarts: int = 8,
             fit_n: int = 150) -> List[float]:
    nc = tokens.count('<C>')
    if nc == 0:
        return []

    pts = np.array(raw_pts[:fit_n])
    y   = pts[:, -1]
    X   = pts[:, :-1]

    def _loss(c):
        vd = {var_names[i]: X[:, i] for i in range(min(len(var_names), X.shape[1]))}
        try:
            yp = eval_postfix(tokens, vd, c)
            if yp is None or not np.all(np.isfinite(yp)):
                return 1e10
            return float(np.mean((yp - y) ** 2))
        except Exception:
            return 1e10

    best_c = np.ones(nc)
    best_l = float('inf')

    for _ in range(n_restarts):
        x0 = np.random.randn(nc) * 2
        try:
            r = minimize(_loss, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-6})
            if r.fun < best_l:
                best_l = r.fun; best_c = r.x
        except Exception:
            pass

    try:
        r2 = minimize(_loss, best_c, method='L-BFGS-B',
                      bounds=[(-100, 100)] * nc,
                      options={'maxiter': 1000})
        if r2.fun < best_l:
            best_c = r2.x
    except Exception:
        pass

    return best_c.tolist()


def r2_score(tokens: List[str], var_names: List[str],
             c_vals: List[float], raw_pts: list,
             eval_start: int = 150) -> float:
    pts = np.array(raw_pts[eval_start:]) if len(raw_pts) > eval_start \
          else np.array(raw_pts)
    if len(pts) == 0:
        return -10.0

    y  = pts[:, -1]; X = pts[:, :-1]
    vd = {var_names[i]: X[:, i] for i in range(min(len(var_names), X.shape[1]))}

    try:
        yp = eval_postfix(tokens, vd, c_vals)
        if yp is None or not np.all(np.isfinite(yp)):
            return -10.0
        ss_res = np.sum((y - yp) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(np.clip(1.0 - ss_res / (ss_tot + 1e-10), -10, 1))
    except Exception:
        return -10.0


def best_candidate_r2(candidates: List[List[str]], var_names: List[str],
                      raw_pts: list, n_restarts: int = 8):
    best_r2     = -10.0
    best_toks   = []
    best_consts = []

    for cand in candidates:
        if not cand or not raw_pts:
            continue
        c_vals = bfgs_fit(cand, var_names, raw_pts, n_restarts)
        r2     = r2_score(cand, var_names, c_vals, raw_pts)
        if r2 > best_r2:
            best_r2     = r2
            best_toks   = cand
            best_consts = c_vals

    return best_toks, best_consts, best_r2
