# EvoTraffic — AI-Optimized Traffic Signal Control (PSO + GA)

A tick-based traffic-intersection simulator combined with two evolutionary
optimization algorithms — **Particle Swarm Optimization (PSO)** and a
**Genetic Algorithm (GA)** — used to find near-optimal green-light timings
for a four-way intersection under varying traffic conditions.

Fixed traffic-light timing plans can't react to real-time traffic. EvoTraffic
treats signal-timing selection as an optimization problem: each candidate
solution is a set of green-light durations, and a `pygame`-based simulation
scores that solution by running simulated traffic through the intersection
and measuring congestion.

---

## How It Works

1. **Simulation** (`simulation.py`) — a tick-accurate model of a 4-way
   intersection with vehicles (car, bus, truck, bike) spawning from four
   directions, queuing, and passing through a junction with conflict-aware
   right-of-way rules.
2. **Objective function** — each candidate signal-timing plan is scored by:

   ```
   J = Waiting Time + Queue Length + Number of Stops
   ```

   Lower `J` = better traffic flow. Every candidate is evaluated across
   **low / medium / high** traffic modes, and the final fitness is the
   average of the three:

   ```
   Fitness = (F_low + F_medium + F_high) / 3
   ```

3. **Optimizers** search the space of `[NS_green, EW_green, ...]` timings
   (bounded 5–60 seconds per phase) to minimize that fitness.

---

## Algorithms

### Particle Swarm Optimization (`PSO.py`)
Global-best PSO: a swarm of particles (candidate timing plans) moves through
the search space, pulled toward its own best-known position and the swarm's
global best.

```
v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
```

| Parameter | Value |
|---|---|
| Swarm size | 6 (small run) / 12 (big run) |
| Iterations | 8 (small) / 15 (big) |
| Inertia (w) | 0.7 |
| Cognitive (c1) | 1.5 |
| Social (c2) | 1.5 |
| Velocity limit | [-10, 10] |
| Bounds | [5, 60] s |

### Genetic Algorithm (`GA.py`)
Population-based search using tournament selection, single-point crossover,
step-based mutation, and elitism.

| Parameter | Value |
|---|---|
| Population size | 6 (small run) / 12 (big run) |
| Generations | 8 (small) / 15 (big) |
| Crossover rate | 0.9 |
| Mutation rate | 0.3 |
| Tournament size | 3 |
| Elite size | 1 |
| Bounds | [5, 60] s |

---

## Results Summary

| Metric | PSO (best) | GA (best) |
|---|---|---|
| Best fitness score | 50.68 | **49.41** |
| Best timings | `[52, 44, 6, 43]` | `[60, 5, 43, 20]` |

Averaged across low/medium/high traffic scenarios, **GA outperformed PSO**
on final fitness, waiting time, and queue length, while **PSO converged
faster** and was more stable in early iterations. See `results/*.json` for
the full run histories and `graphs.py` for the comparison plots
(PSO vs. baseline, GA vs. baseline, PSO vs. GA convergence, and final
comparison against the fixed-timing baseline).

---

## Project Structure

```
EvoTraffic-main/
├── main.py          # Entry point — orchestrates baseline, PSO, GA runs & plots
├── simulation.py     # Tick-based intersection simulation + evaluate() scoring function
├── PSO.py            # Particle Swarm Optimization implementation
├── GA.py              # Genetic Algorithm implementation
├── graphs.py          # Matplotlib comparison plots
├── images/           # Vehicle & signal sprites used by the pygame visualization
└── results/           # Saved JSON results (baseline, PSO, GA — small & big runs)
```

---

## Requirements

- Python 3.9+
- `pygame`
- `numpy`
- `matplotlib`

Install with:

```bash
pip install pygame numpy matplotlib
```

## Usage

Run the full pipeline (loads cached results from `results/` by default, or
re-runs the optimizers if `RUN_PSO` / `RUN_GA` are set to `True` in
`main.py`):

```bash
python main.py
```

This will:
1. Evaluate (or load) the fixed-timing baseline (`[10, 10, 10, 10]`).
2. Visualize the baseline in a `pygame` window.
3. Run or load PSO (small + big) results, visualize, and plot convergence.
4. Run or load GA (small + big) results, visualize, and plot convergence.
5. Plot a final PSO vs. GA vs. baseline comparison.

Results are cached as JSON in `results/` (`baseline.json`, `pso_small.json`,
`pso_big.json`, `ga_small.json`, `ga_big.json`) so repeated runs don't need
to re-simulate identical timing vectors.

## Key Findings

- **GA** achieved a better final fitness score across all three traffic
  scenarios, especially under high-traffic complexity, due to stronger
  exploration from crossover + mutation.
- **PSO** converged faster and was more stable early on, but its strong
  pull toward the global best reduced swarm diversity over time, causing it
  to get stuck in a less optimal region for complex, non-linear traffic
  scenarios.
