import numpy as np
from typing import List, Dict, Optional


BINARY_EVAL = {
    'OP_add': lambda a, b: a + b,
    'OP_mul': lambda a, b: a * b,
    'OP_sub': lambda a, b: a - b,
    'OP_div': lambda a, b: a / np.where(np.abs(b) < 1e-10, 1.0, b),
    'OP_pow': lambda a, b: np.sign(a) * (np.abs(a) + 1e-10) ** np.clip(b, -8, 8),
}

UNARY_EVAL = {
    'FUNC_sin':    np.sin,
    'FUNC_cos':    np.cos,
    'FUNC_exp':    lambda x: np.exp(np.clip(x, -30, 30)),
    'FUNC_log':    lambda x: np.log(np.abs(x) + 1e-10),
    'FUNC_sqrt':   lambda x: np.sqrt(np.abs(x)),
    'FUNC_abs':    np.abs,
    'FUNC_tanh':   np.tanh,
    'FUNC_arcsin': lambda x: np.arcsin(np.clip(x, -1, 1)),
    'FUNC_arccos': lambda x: np.arccos(np.clip(x, -1, 1)),
    'FUNC_arctan': np.arctan,
}

NAMED_CONSTS = {
    'CONST_pi': np.pi,
    'CONST_E':  np.e,
}


def eval_postfix(tokens: List[str], var_arrays: Dict[str, np.ndarray],
                 c_vals: List[float]) -> Optional[np.ndarray]:
    N     = max((len(v) for v in var_arrays.values()), default=1)
    stack = []
    c_idx = 0

    for tok in tokens:
        if tok in BINARY_EVAL:
            if len(stack) < 2:
                return np.full(N, np.nan)
            b, a = stack.pop(), stack.pop()
            try:
                stack.append(BINARY_EVAL[tok](a, b))
            except Exception:
                return np.full(N, np.nan)
        elif tok in UNARY_EVAL:
            if len(stack) < 1:
                return np.full(N, np.nan)
            a = stack.pop()
            try:
                stack.append(UNARY_EVAL[tok](a))
            except Exception:
                return np.full(N, np.nan)
        elif tok.startswith('VAR_'):
            stack.append(var_arrays.get(tok[4:], np.ones(N)))
        elif tok == '<C>':
            val = float(c_vals[c_idx]) if c_idx < len(c_vals) else 1.0
            c_idx += 1
            stack.append(np.full(N, val))
        elif tok in NAMED_CONSTS:
            stack.append(np.full(N, NAMED_CONSTS[tok]))
        elif tok in ('<BOS>', '<EOS>', '<PAD>'):
            continue
        else:
            stack.append(np.ones(N))

    return stack[0] if stack else np.full(N, np.nan)


def is_valid_postfix(tokens: List[str],
                     binary_set: set, unary_set: set,
                     terminal_set: set) -> bool:
    stack = 0
    for tok in tokens:
        if tok in ('<BOS>', '<PAD>'): continue
        if tok == '<EOS>':            break
        if tok in binary_set:         stack -= 1
        elif tok in unary_set:        stack += 0
        elif tok in terminal_set:     stack += 1
    return stack == 1
