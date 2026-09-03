import random
import numpy as np

# =========================================================
# Genetic Algorithm (GA)
# =========================================================

def create_individual(bounds, rng):
    return [rng.randint(b[0], b[1]) for b in bounds]


def fitness(individual, evaluate, sim_time):
    return evaluate(individual, sim_duration=sim_time, headless=True)


def tournament_selection(population, scores, k=3):
    selected_idx = random.sample(range(len(population)), k)
    best_idx     = min(selected_idx, key=lambda i: scores[i])
    return population[best_idx][:]


def crossover(parent1, parent2, crossover_rate=0.5):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]
    point  = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(individual, bounds, mutation_rate=0.5):
    """
    mutation_rate=0.3 on 4 genes → ~76% chance at least one gene mutates.
    mutation_rate=0.1 on 4 genes → ~34% chance → most children are clones.

    step drawn from [-10,10] excluding 0 so every triggered mutation
    actually changes the value.
    """
    new_ind = individual[:]
    for i in range(len(new_ind)):
        if random.random() < mutation_rate:
            # non-zero step: pick from [-10,-1] or [1,10]
            step = random.choice([-1, 1]) * random.randint(1, 10)
            new_ind[i] = max(bounds[i][0],
                             min(bounds[i][1], new_ind[i] + step))
    return new_ind


def run_ga(
    evaluate,
    bounds,
    generations=15,
    population_size=10,
    sim_time=300,
    crossover_rate=0.5,
    mutation_rate=0.5,
    elite_size=1,
    target_fitness=None,
    seed=42
):
    """
    Parameters
    ----------
    evaluate        : callable – evaluate(green_times, sim_duration=, headless=)
    bounds          : list of (min, max) tuples, one per signal
    generations     : int
    population_size : int   – use ≥ 6 for meaningful diversity
    sim_time        : float – simulated seconds per evaluation
    elite_size      : int   – keep only the single best, not half the pool
                              (elite_size=2 with pop=4 means only 2 new
                               individuals per generation → near-zero diversity)
    mutation_rate   : float – 0.3 ensures most children are meaningfully different
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    random.seed(seed)

    population      = [create_individual(bounds, rng) for _ in range(population_size)]
    best_individual = None
    best_score      = float('inf')
    history         = []

    print("\n" + "=" * 50)
    print(f" GA  |  generations={generations}  pop={population_size}  "
          f"elite={elite_size}  mut={mutation_rate}  sim={sim_time}s")
    print("=" * 50)

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1}/{generations} ---")

        scores = []
        for i, individual in enumerate(population):
            try:
                score = fitness(individual, evaluate, sim_time)
                print(f"  {i+1:02d}: {individual}  →  {score:.4f}")
            except Exception as e:
                print(f"  {i+1:02d}: ERROR – {e}")
                score = float('inf')
            scores.append(score)

            if score < best_score:
                best_score      = score
                best_individual = individual[:]

        gen_best = min(scores)
        gen_avg  = sum(scores) / len(scores)
        print(f"  ── gen best={gen_best:.4f}  "
              f"global best={best_score:.4f}  "
              f"avg={gen_avg:.4f}")

        history.append({
            "gen": gen,
            "gen_best": gen_best,
            "global_best": best_score,
            "avg": gen_avg
        })

        if target_fitness is not None and best_score <= target_fitness:
            print("  Target fitness reached – stopping early.")
            break

        # Elitism: keep only the top `elite_size` individuals
        sorted_pop     = [ind for _, ind in sorted(zip(scores, population))]
        new_population = sorted_pop[:elite_size]

        # Fill the rest with crossover + mutation offspring
        while len(new_population) < population_size:
            p1 = tournament_selection(population, scores)
            p2 = tournament_selection(population, scores)
            c1, c2 = crossover(p1, p2, crossover_rate)
            c1 = mutate(c1, bounds, mutation_rate)
            c2 = mutate(c2, bounds, mutation_rate)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population

    print("\n" + "=" * 50)
    print(" GA COMPLETE")
    print(f" Best individual : {best_individual}")
    print(f" Best fitness    : {best_score:.4f}")
    print("=" * 50)

    return best_individual, best_score, history
