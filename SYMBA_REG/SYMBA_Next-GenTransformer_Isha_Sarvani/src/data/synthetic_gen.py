import random
import numpy as np
from typing import List, Optional


BINARY_OPS   = ['OP_add', 'OP_mul', 'OP_sub', 'OP_div', 'OP_pow']
UNARY_OPS    = ['FUNC_sin', 'FUNC_cos', 'FUNC_exp', 'FUNC_log',
                'FUNC_sqrt', 'FUNC_abs', 'FUNC_tanh']
NAMED_CONSTS = ['CONST_pi', 'CONST_E']
VAR_POOL     = [f'x{i}' for i in range(1, 11)]
BIN_WEIGHTS  = [0.30, 0.30, 0.15, 0.10, 0.15]
UNI_WEIGHTS  = [0.20, 0.20, 0.15, 0.15, 0.10, 0.05, 0.15]


class SynNode:
    def __init__(self, val: str, children=None, const_val=None):
        self.val       = val
        self.children  = children or []
        self.const_val = const_val

    def postfix(self) -> List[str]:
        if not self.children:
            return [self.val]
        tokens = []
        for ch in self.children:
            tokens.extend(ch.postfix())
        tokens.append(self.val)
        return tokens

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def evaluate(self, var_dict: dict) -> np.ndarray:
        v = self.val
        N = len(next(iter(var_dict.values())))
        if v == 'OP_add':   return self.children[0].evaluate(var_dict) + self.children[1].evaluate(var_dict)
        if v == 'OP_mul':   return self.children[0].evaluate(var_dict) * self.children[1].evaluate(var_dict)
        if v == 'OP_sub':   return self.children[0].evaluate(var_dict) - self.children[1].evaluate(var_dict)
        if v == 'OP_div':
            b = self.children[1].evaluate(var_dict)
            return self.children[0].evaluate(var_dict) / np.where(np.abs(b) < 1e-10, 1.0, b)
        if v == 'OP_pow':
            a = self.children[0].evaluate(var_dict)
            b = self.children[1].evaluate(var_dict)
            with np.errstate(all='ignore'):
                return np.sign(a) * (np.abs(a) + 1e-10) ** np.clip(b, -3, 3)
        if v == 'FUNC_sin':  return np.sin(self.children[0].evaluate(var_dict))
        if v == 'FUNC_cos':  return np.cos(self.children[0].evaluate(var_dict))
        if v == 'FUNC_exp':  return np.exp(np.clip(self.children[0].evaluate(var_dict), -15, 15))
        if v == 'FUNC_log':
            a = self.children[0].evaluate(var_dict)
            return np.log(np.abs(a) + 1e-10)
        if v == 'FUNC_sqrt': return np.sqrt(np.abs(self.children[0].evaluate(var_dict)))
        if v == 'FUNC_abs':  return np.abs(self.children[0].evaluate(var_dict))
        if v == 'FUNC_tanh': return np.tanh(self.children[0].evaluate(var_dict))
        if v == 'CONST_pi':  return np.full(N, np.pi)
        if v == 'CONST_E':   return np.full(N, np.e)
        if v == '<C>':       return np.full(N, self.const_val if self.const_val else 1.0)
        if v.startswith('VAR_'): return var_dict.get(v, np.ones(N))
        return np.ones(N)


def rand_leaf(var_tokens: List[str]) -> SynNode:
    r = random.random()
    if r < 0.55:
        return SynNode(random.choice(var_tokens))
    elif r < 0.82:
        c = round(random.uniform(-5, 5), 1)
        if abs(c) < 0.1:
            c = random.choice([-2, -1, 1, 2, 3])
        return SynNode('<C>', const_val=c)
    elif r < 0.93:
        return SynNode('CONST_pi')
    else:
        return SynNode('CONST_E')


def rand_tree(var_tokens: List[str], depth: int = 0,
              max_depth: Optional[int] = None) -> SynNode:
    if max_depth is None:
        max_depth = random.randint(1, 4)
    if depth >= max_depth or (depth > 0 and random.random() < 0.30):
        return rand_leaf(var_tokens)
    if random.random() < 0.60:
        op = random.choices(BINARY_OPS, weights=BIN_WEIGHTS, k=1)[0]
        return SynNode(op, [rand_tree(var_tokens, depth+1, max_depth),
                            rand_tree(var_tokens, depth+1, max_depth)])
    else:
        op = random.choices(UNARY_OPS, weights=UNI_WEIGHTS, k=1)[0]
        return SynNode(op, [rand_tree(var_tokens, depth+1, max_depth)])


def _valid_output(y: np.ndarray) -> bool:
    if y is None:                          return False
    if not np.all(np.isfinite(y)):         return False
    if np.std(y) < 1e-6:                   return False
    if np.max(np.abs(y)) > 1e6:            return False
    return True


def generate_synthetic(n: int = 10_000, n_points: int = 200,
                       seed: int = 42, verbose: bool = True) -> List[dict]:
    random.seed(seed); np.random.seed(seed)
    equations = []
    attempts  = 0

    while len(equations) < n and attempts < n * 30:
        attempts += 1
        nv       = random.randint(1, 6)
        raw_vars = random.sample(VAR_POOL[:max(nv+2, 6)], nv)
        raw_vars.sort()
        vtoks    = [f'VAR_{v}' for v in raw_vars]
        tree     = rand_tree(vtoks)
        pf       = tree.postfix()

        if not any(t.startswith('VAR_') for t in pf): continue
        sz = tree.size()
        if sz < 3 or sz > 30: continue

        X = {}; Xa = np.zeros((n_points, nv))
        for i, vn in enumerate(raw_vars):
            if random.random() < 0.3:
                vals = np.random.uniform(0.1, 8, n_points)
            elif random.random() < 0.5:
                vals = np.random.uniform(-3, 3, n_points)
            else:
                vals = np.random.uniform(-5, 5, n_points)
            X[f'VAR_{vn}'] = vals; Xa[:, i] = vals

        try:
            y = tree.evaluate(X)
        except Exception:
            continue

        if not _valid_output(y): continue

        cloud  = np.column_stack([Xa, y])
        tokens = ['<BOS>'] + pf + ['<EOS>']
        equations.append({
            'filename': f'synth_{len(equations):05d}',
            'tokens':   tokens,
            'cloud':    cloud,
            'n_vars':   nv,
        })

        if verbose and len(equations) % 2000 == 0:
            print(f"  Generated {len(equations):,}/{n:,}  ({attempts:,} attempts)")

    if verbose:
        print(f"  Done: {len(equations):,} equations, {attempts:,} attempts, "
              f"{(attempts-len(equations))/attempts*100:.0f}% rejected")
    return equations
