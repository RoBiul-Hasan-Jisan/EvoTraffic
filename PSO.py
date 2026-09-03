import numpy as np
import random

# =========================================================
# Particle Swarm Optimization (PSO)
# =========================================================

class Particle:
    def __init__(self, bounds, rng):
        self.position   = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        self.velocity   = np.array([rng.uniform(-5, 5)       for _ in bounds])
        self.best_pos   = np.copy(self.position)
        self.best_score = float('inf')
        self.score      = float('inf')

    def update_velocity(self, global_best_pos, w=0.7, c1=1.5, c2=1.5):
        """
        w < 1.0  → velocity shrinks each step → particles converge.  CORRECT
        w >= 1.0 → velocity grows → all particles fly to bounds
                   → all candidates clamped to corners
                   → scores become identical → flat history.          WRONG
        Standard convergent PSO: w=0.7, c1=c2=1.5.
        """
        r1 = random.random()
        r2 = random.random()
        cognitive     = c1 * r1 * (self.best_pos   - self.position)
        social        = c2 * r2 * (global_best_pos - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -10, 10)

    def update_position(self, bounds):
        self.position += self.velocity
        for i, (lo, hi) in enumerate(bounds):
            if self.position[i] < lo:
                self.position[i] = lo
                self.velocity[i] *= -0.5   # bounce: preserve exploration
            elif self.position[i] > hi:
                self.position[i] = hi
                self.velocity[i] *= -0.5


def run_pso(evaluate, iterations=15, swarm_size=10, sim_time=300):
    """
    Finds optimal green-signal timings using PSO.

    Parameters
    ----------
    evaluate   : callable – evaluate(green_times, sim_duration=, headless=)
    iterations : int      – PSO iterations
    swarm_size : int      – number of particles
    sim_time   : float    – simulated seconds per fitness call
    """
    bounds = [(5, 60)] * 4

    # Seeded for reproducible initial swarm positions
    rng = random.Random(42)
    np.random.seed(42)

    swarm             = [Particle(bounds, rng) for _ in range(swarm_size)]
    global_best_pos   = None
    global_best_score = float('inf')

    print("\n" + "!" * 50)
    print(f" PSO  |  iterations={iterations}  swarm={swarm_size}  sim={sim_time}s")
    print("!" * 50)

    history = []   # global-best score recorded after each full iteration

    for iteration in range(iterations):
        print(f"\n--- Iteration {iteration + 1}/{iterations} ---")
        iter_scores = []

        for p_idx, particle in enumerate(swarm):
            timings = [int(round(t)) for t in particle.position]
            particle.score = evaluate(timings, sim_duration=sim_time, headless=True)
            iter_scores.append(particle.score)

            print(f"  P{p_idx+1:02d}: {timings}  score={particle.score:.4f}")

            if particle.score < particle.best_score:
                particle.best_score = particle.score
                particle.best_pos   = np.copy(particle.position)

            if particle.score < global_best_score:
                global_best_score = particle.score
                global_best_pos   = np.copy(particle.position)

        print(f"  ── iter best={min(iter_scores):.4f}  "
              f"global best={global_best_score:.4f}  "
              f"avg={sum(iter_scores)/len(iter_scores):.4f}")

        # Move ALL particles AFTER evaluating the full swarm
        for particle in swarm:
            particle.update_velocity(global_best_pos)
            particle.update_position(bounds)

        iter_best = min(iter_scores)
        avg_fitness = sum(iter_scores) / len(iter_scores)

        history.append({
            "iter": iteration,
            "iter_best": iter_best,
            "global_best": global_best_score,
            "avg": avg_fitness
        })

    best_timings = [int(round(t)) for t in global_best_pos]

    print("\n" + "=" * 50)
    print(f" PSO COMPLETE")
    print(f" Best timings : {best_timings}")
    print(f" Best fitness : {global_best_score:.4f}")
    print("=" * 50)

    return best_timings, global_best_score, history
