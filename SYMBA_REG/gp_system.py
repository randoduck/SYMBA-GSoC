# gp_system.py
# ─────────────────────────────────────────────────────────────
# PIGP Seeding + Genetic Programming Evolution
# Takes beam search candidates → seeds GP → evolves → returns
# best symbolic equations.
#
# PIGP = Partially Informed Genetic Programming
#        (from ML4SCI 2024 paper — we add the feedback loop)
#
# GP core authored by Samyak Jha (GSoC 2025, ML4SCI SYMBA).
# Reused here with permission to avoid redundant reimplementation.
# ─────────────────────────────────────────────────────────────

import json
import random
import math
import numpy as np
from copy import deepcopy


# ── 1. EXPRESSION TREE NODE ───────────────────────────────────
class Node:
    """A node in a symbolic expression tree."""
    BINARY_OPS = {"OP_add", "OP_mul", "OP_pow"}
    UNARY_OPS  = {"FUNC_sin", "FUNC_cos", "FUNC_exp", "FUNC_log",
                  "FUNC_tanh", "FUNC_arcsin"}

    def __init__(self, value, children=None):
        self.value    = value                # token string e.g. "OP_mul"
        self.children = children or []

    def is_leaf(self):
        return len(self.children) == 0

    def is_binary(self):
        return self.value in self.BINARY_OPS

    def is_unary(self):
        return self.value in self.UNARY_OPS

    def depth(self):
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def clone(self):
        return deepcopy(self)

    def __repr__(self):
        if self.is_leaf():
            return self.value
        return f"{self.value}({', '.join(str(c) for c in self.children)})"


# ── 2. TOKEN SEQUENCE ↔ TREE CONVERSION ───────────────────────
def tokens_to_tree(tokens):
    """
    Convert prefix token list → expression tree.
    e.g. ["OP_mul", "VAR_q", "VAR_v"] → Node(OP_mul, [Node(VAR_q), Node(VAR_v)])
    """
    tokens = [t for t in tokens if t not in ("<BOS>", "<EOS>", "<PAD>")]
    idx    = [0]  # mutable index

    def parse():
        if idx[0] >= len(tokens):
            return Node("<C>")
        tok = tokens[idx[0]]
        idx[0] += 1

        if tok in Node.BINARY_OPS:
            left  = parse()
            right = parse()
            return Node(tok, [left, right])
        elif tok in Node.UNARY_OPS:
            arg   = parse()
            return Node(tok, [arg])
        else:
            return Node(tok)   # variable, constant, <C>

    try:
        tree = parse()
        return tree
    except Exception:
        return None


def tree_to_tokens(node):
    """Convert expression tree → prefix token list."""
    if node is None:
        return ["<C>"]
    if node.is_leaf():
        return [node.value]
    result = [node.value]
    for child in node.children:
        result.extend(tree_to_tokens(child))
    return result


# ── 3. TREE EVALUATION ────────────────────────────────────────
def evaluate_tree(node, var_values):
    """
    Evaluate expression tree given variable values dict.
    var_values: {"VAR_x": array, "VAR_y": array, ...}
    Returns numpy array of outputs, or None if error.
    """
    try:
        v = node.value

        if v == "OP_add":
            a = evaluate_tree(node.children[0], var_values)
            b = evaluate_tree(node.children[1], var_values)
            if a is None or b is None: return None
            return a + b

        elif v == "OP_mul":
            a = evaluate_tree(node.children[0], var_values)
            b = evaluate_tree(node.children[1], var_values)
            if a is None or b is None: return None
            return a * b

        elif v == "OP_pow":
            base = evaluate_tree(node.children[0], var_values)
            exp  = evaluate_tree(node.children[1], var_values)
            if base is None or exp is None: return None
            # safe power
            with np.errstate(all='ignore'):
                result = np.power(np.abs(base) + 1e-10, exp)
            return result

        elif v == "FUNC_sin":
            a = evaluate_tree(node.children[0], var_values)
            return np.sin(a) if a is not None else None

        elif v == "FUNC_cos":
            a = evaluate_tree(node.children[0], var_values)
            return np.cos(a) if a is not None else None

        elif v == "FUNC_exp":
            a = evaluate_tree(node.children[0], var_values)
            if a is None: return None
            return np.exp(np.clip(a, -50, 50))

        elif v == "FUNC_log":
            a = evaluate_tree(node.children[0], var_values)
            if a is None: return None
            return np.log(np.abs(a) + 1e-10)

        elif v == "FUNC_tanh":
            a = evaluate_tree(node.children[0], var_values)
            return np.tanh(a) if a is not None else None

        elif v == "FUNC_arcsin":
            a = evaluate_tree(node.children[0], var_values)
            if a is None: return None
            return np.arcsin(np.clip(a, -1, 1))

        elif v == "CONST_pi":
            return np.full_like(list(var_values.values())[0], math.pi)

        elif v == "<C>":
            # Learned constant — use 1.0 as default
            return np.ones_like(list(var_values.values())[0])

        elif v.startswith("VAR_"):
            return var_values.get(v, np.ones_like(list(var_values.values())[0]))

        else:
            return np.ones_like(list(var_values.values())[0])

    except Exception:
        return None


# ── 4. FITNESS FUNCTION ───────────────────────────────────────
def fitness(tree, data_entry):
    """
    Fitness = negative normalised MSE (higher is better).
    data_entry: from data_clouds.json
    """
    if tree is None:
        return -1e9

    inputs     = data_entry["inputs"]    # list of var names
    data       = np.array(data_entry["data"])  # (N, n_vars+1)
    n_vars     = len(inputs)

    X = data[:, :n_vars]
    y = data[:, n_vars]

    var_values = {f"VAR_{inputs[i]}": X[:, i] for i in range(n_vars)}

    pred = evaluate_tree(tree, var_values)
    if pred is None or not np.isfinite(pred).all():
        return -1e9

    # Normalised MSE
    mse  = np.mean((pred - y) ** 2)
    norm = np.var(y) + 1e-10
    r2   = 1.0 - mse / norm

    # Complexity penalty — shorter trees are better
    penalty = 0.001 * tree.size()

    return float(r2 - penalty)


# ── 5. RANDOM TREE GENERATION ─────────────────────────────────
def random_tree(variable_names, max_depth=4):
    """Generate a random expression tree."""
    ops_binary = list(Node.BINARY_OPS)
    ops_unary  = list(Node.UNARY_OPS)
    var_tokens = [f"VAR_{v}" for v in variable_names] + ["<C>", "CONST_pi"]

    def _grow(depth):
        if depth >= max_depth or (depth > 0 and random.random() < 0.4):
            return Node(random.choice(var_tokens))
        op = random.choice(ops_binary + ops_unary)
        if op in Node.BINARY_OPS:
            return Node(op, [_grow(depth+1), _grow(depth+1)])
        else:
            return Node(op, [_grow(depth+1)])

    return _grow(0)


# ── 6. GP OPERATORS ───────────────────────────────────────────
def subtree_mutation(tree, variable_names, max_depth=4):
    """Replace a random subtree with a new random tree."""
    tree  = tree.clone()
    nodes = _collect_nodes(tree)
    if not nodes:
        return tree
    target = random.choice(nodes)
    new    = random_tree(variable_names, max_depth=max_depth - target.depth())
    target.value    = new.value
    target.children = new.children
    return tree


def subtree_crossover(tree1, tree2):
    """Swap a random subtree from tree2 into tree1."""
    t1     = tree1.clone()
    t2     = tree2.clone()
    nodes1 = _collect_nodes(t1)
    nodes2 = _collect_nodes(t2)
    if not nodes1 or not nodes2:
        return t1
    n1 = random.choice(nodes1)
    n2 = random.choice(nodes2)
    n1.value    = n2.value
    n1.children = deepcopy(n2.children)
    return t1


def _collect_nodes(node, nodes=None):
    if nodes is None:
        nodes = []
    nodes.append(node)
    for c in node.children:
        _collect_nodes(c, nodes)
    return nodes


# ── 7. PIGP SEEDING ───────────────────────────────────────────
def pigp_seed_population(
    beam_candidates,   # list of dicts from beam_search.generate_candidates()
    variable_names,    # list of variable name strings
    pop_size    = 100,
    seed_ratio  = 0.3, # fraction of population seeded by beam search
):
    """
    PIGP: Partially Informed GP initialisation.
    - First (seed_ratio * pop_size) individuals = converted beam candidates
    - Rest = random trees

    This is the key idea from the ML4SCI 2024 paper.
    """
    population  = []
    n_seeded    = int(pop_size * seed_ratio)

    # ── seeded from beam search ──
    for cand in beam_candidates[:n_seeded]:
        tree = tokens_to_tree(cand["tokens"])
        if tree is not None:
            population.append(tree)

    # ── pad with random trees ──
    while len(population) < pop_size:
        population.append(random_tree(variable_names))

    print(f"Population: {len(population)} total | "
          f"{min(n_seeded, len(beam_candidates))} seeded from beam | "
          f"{len(population) - min(n_seeded, len(beam_candidates))} random")

    return population


# ── 8. GP EVOLUTION ───────────────────────────────────────────
def evolve(
    population,
    data_entry,
    variable_names,
    n_generations   = 50,
    tournament_size = 5,
    crossover_prob  = 0.7,
    mutation_prob   = 0.3,
    elitism         = 5,     # keep top-N unchanged each gen
):
    """
    Evolve the GP population using tournament selection,
    subtree crossover and subtree mutation.

    Returns (best_tree, best_fitness, history)
    """
    history = []

    # Initial fitness
    fitnesses = [fitness(t, data_entry) for t in population]

    for gen in range(n_generations):
        # ── elitism: carry top-N directly ──
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
        new_pop    = [t.clone() for _, t in sorted_pop[:elitism]]

        while len(new_pop) < len(population):
            op = random.random()

            if op < crossover_prob:
                # tournament select 2 parents
                p1 = _tournament(population, fitnesses, tournament_size)
                p2 = _tournament(population, fitnesses, tournament_size)
                child = subtree_crossover(p1, p2)
            else:
                # tournament select 1 parent + mutate
                p1    = _tournament(population, fitnesses, tournament_size)
                child = subtree_mutation(p1, variable_names)

            new_pop.append(child)

        population = new_pop
        fitnesses  = [fitness(t, data_entry) for t in population]

        best_fit   = max(fitnesses)
        best_tree  = population[fitnesses.index(best_fit)]
        history.append(best_fit)

        if gen % 10 == 0:
            print(f"  Gen {gen:3d} | Best fitness: {best_fit:.4f} | "
                  f"Expr: {' '.join(tree_to_tokens(best_tree)[:8])}...")

        # Early stopping — perfect fit
        if best_fit > 0.999:
            print(f"  ✓ Converged at generation {gen}")
            break

    best_fit  = max(fitnesses)
    best_tree = population[fitnesses.index(best_fit)]
    return best_tree, best_fit, history


def _tournament(population, fitnesses, k):
    """Tournament selection — pick best of k random individuals."""
    idxs    = random.sample(range(len(population)), min(k, len(population)))
    best_i  = max(idxs, key=lambda i: fitnesses[i])
    return population[best_i].clone()


# ── 9. FULL PIGP + GP PIPELINE FOR ONE EQUATION ───────────────
def run_gp_for_equation(
    beam_candidates,
    data_entry,
    pop_size       = 100,
    n_generations  = 50,
    seed_ratio     = 0.3,
):
    """
    Full PIGP + GP run for a single equation.
    Returns best tree, fitness, token list.
    """
    variable_names = data_entry["inputs"]

    # Seed population with beam search candidates
    population = pigp_seed_population(
        beam_candidates, variable_names,
        pop_size=pop_size, seed_ratio=seed_ratio
    )

    # Evolve
    best_tree, best_fit, history = evolve(
        population, data_entry, variable_names,
        n_generations=n_generations
    )

    best_tokens = tree_to_tokens(best_tree)

    return {
        "best_tree":    best_tree,
        "best_fitness": best_fit,
        "best_tokens":  best_tokens,
        "expression":   " ".join(best_tokens),
        "history":      history,
    }


# ── QUICK TEST ────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic test — no real data needed
    print("Testing GP system with synthetic data...")

    # Fake data: y = x1 * x2
    np.random.seed(42)
    N = 50
    x1 = np.random.uniform(0.1, 5.0, N)
    x2 = np.random.uniform(0.1, 5.0, N)
    y  = x1 * x2

    data_entry = {
        "inputs": ["x1", "x2"],
        "data":   np.column_stack([x1, x2, y]).tolist()
    }

    # Fake beam candidates (as if from beam search)
    beam_candidates = [
        {"tokens": ["OP_mul", "VAR_x1", "VAR_x2"], "score": -0.5},
        {"tokens": ["OP_add", "VAR_x1", "VAR_x2"], "score": -1.0},
    ]

    result = run_gp_for_equation(beam_candidates, data_entry, pop_size=50, n_generations=30)
    print(f"\nBest expression : {result['expression']}")
    print(f"Best fitness    : {result['best_fitness']:.4f}")
