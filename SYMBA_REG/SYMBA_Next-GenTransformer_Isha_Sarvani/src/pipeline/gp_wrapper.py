import random
from .gp_system import pigp_seed_population, evolve, tree_to_tokens


def run_pigp(candidates: list, var_names: list, data_entry: dict,
             pop_size: int = 100, n_generations: int = 80,
             seed_ratio: float = 0.4, tournament_size: int = 5,
             crossover_prob: float = 0.7, mutation_prob: float = 0.3,
             elitism: int = 5, verbose: bool = True):
    if not var_names or not data_entry.get('data'):
        return -10.0, ''

    gp_seeds = []
    for i, cand in enumerate(candidates[:6]):
        if cand:
            gp_seeds.append({'tokens': cand, 'score': -float(i)})

    all_token_names = list(set(t for cand in candidates for t in cand if cand))
    for _ in range(5):
        if candidates and candidates[0]:
            noisy = candidates[0][:]
            idx   = random.randint(0, max(len(noisy) - 1, 0))
            if all_token_names:
                noisy[idx] = random.choice(all_token_names)
            gp_seeds.append({'tokens': noisy, 'score': -5.0})

    try:
        population = pigp_seed_population(
            gp_seeds, var_names, pop_size=pop_size, seed_ratio=seed_ratio
        )
        best_tree, best_fit, _ = evolve(
            population, data_entry, var_names,
            n_generations=n_generations,
            tournament_size=tournament_size,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            elitism=elitism,
        )
        gp_r2   = best_fit + 0.001 * best_tree.size()
        gp_expr = ' '.join(tree_to_tokens(best_tree)[:12])
        return gp_r2, gp_expr

    except Exception as e:
        if verbose:
            print(f"    GP error: {e}")
        return -10.0, ''
