import os
import json
import functools
import pygame
from simulation import evaluate

try:
    from PSO import run_pso
    from GA import run_ga
    from graphs import (
        plot_pso_comparison,
        plot_ga_comparison,
        plot_optimizer_comparison_runs,
        plot_final_comparison,
    )
except ImportError:
    pass

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Caching ────────────────────────────────────────────────────────────────────
# Round green-times to 1 decimal place so near-identical PSO/GA positions
# hit the cache instead of spawning three full simulations each time.
_eval_cache: dict = {}

def robust_evaluate(
    green_times,
    sim_duration: int = 180,
    headless: bool = True,
) -> float:
    """
    Average fitness across low / medium / high traffic.

    Results are memoised: identical (rounded) green_times vectors are never
    re-simulated, which is a large win for PSO where particles frequently
    revisit near-identical positions.
    """
    key = tuple(round(t, 1) for t in green_times)
    if key in _eval_cache:
        return _eval_cache[key]

    scores = [
        evaluate(green_times, sim_duration, headless, mode)["fitness"]
        for mode in ("low", "medium", "high")
    ]
    result = sum(scores) / len(scores)
    _eval_cache[key] = result
    return result


# ── Persistence helpers ────────────────────────────────────────────────────────
def _result_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{name.lower()}.json")


def save_results(method_name: str, timings, score, history=None, extra=None):
    data = {
        "method": method_name,
        "timings": list(timings),
        "score": float(score),
    }
    if extra:
        data.update(extra)
    if history is not None:
        data["history"] = history

    path = _result_path(method_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅  {method_name} results saved → {path}")


def load_results(name: str) -> dict:
    """Load a previously saved result file and return its dict."""
    path = _result_path(name)
    with open(path) as f:
        return json.load(f)


def _evaluate_all_modes(times, sim_duration: int = 180):
    """Run evaluate() for all three traffic modes and return the dicts."""
    return {
        mode: evaluate(times, sim_duration, True, mode)
        for mode in ("low", "medium", "high")
    }


# ── Optimizer runner ───────────────────────────────────────────────────────────
def run_and_save_optimizer(
    name: str,
    run_fn,
    evaluate_fn,
    steps: int,
    population_size: int,
    sim_time: int,
    mode: str = "pso",
    bounds=None,
):
    print(f"\n▶  Running {name} …")

    if mode == "pso":
        times, score, history = run_fn(
            evaluate_fn,
            iterations=steps,
            swarm_size=population_size,
            sim_time=sim_time,
        )
    elif mode == "ga":
        if bounds is None:
            raise ValueError("bounds must be provided for GA mode")
        times, score, history = run_fn(
            evaluate_fn,
            bounds,
            generations=steps,
            population_size=population_size,
            sim_time=sim_time,
        )
    else:
        raise ValueError(f"Unknown optimizer mode: {mode!r}")

    per_mode = _evaluate_all_modes(times)
    save_results(name, times, score, history, extra=per_mode)
    return times, score, history


# ── Visualisation helpers ──────────────────────────────────────────────────────
def _visualise(times, label: str = "", modes=("low", "medium", "high"), duration: int = 60):
    if label:
        print(f"\n▶  Visualising {label} …")
    for mode in modes:
        evaluate(times, sim_duration=duration, headless=False, traffic_mode=mode)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  TRAFFIC OPTIMISATION SYSTEM  (PSO + GA, tick-based)")
    print("=" * 55)

    baseline_times = [10, 10, 10, 10]
    baseline_path = _result_path("baseline")

    # ── Baseline ──────────────────────────────────────────────────────
    if os.path.exists(baseline_path):
        print("\  Loading baseline …")
        baseline_data = load_results("baseline")
        baseline_times = baseline_data["timings"]
        # Stored under "fitness" key (legacy) or "score" for consistency.
        baseline_score = baseline_data.get("score", baseline_data.get("fitness"))
        print(f"    Score: {baseline_score:.4f}")
    else:
        print("\n▶  Running baseline (headless, 180 s, all traffic modes) …")
        per_mode = _evaluate_all_modes(baseline_times)
        baseline_score = sum(v["fitness"] for v in per_mode.values()) / len(per_mode)
        print(f"    Score: {baseline_score:.4f}")

        with open(baseline_path, "w") as f:
            json.dump(
                {
                    "timings": baseline_times,
                    "score": baseline_score,
                    **per_mode,
                },
                f,
                indent=4,
            )
        print("  baseline.json saved")

    # Visualise baseline — defer pygame.init() until we actually need it.
    pygame.init()
    _visualise(baseline_times, label="baseline")

    # ── PSO ───────────────────────────────────────────────────────────
    RUN_PSO = False

    if RUN_PSO:
        pso_small_times, pso_small_score, pso_small_history = run_and_save_optimizer(
            "PSO_SMALL", run_pso, robust_evaluate,
            steps=8, population_size=6, sim_time=200, mode="pso",
        )
        pso_big_times, pso_big_score, pso_big_history = run_and_save_optimizer(
            "PSO_BIG", run_pso, robust_evaluate,
            steps=15, population_size=12, sim_time=300, mode="pso",
        )
    else:
        print("\n  Loading PSO results …")
        pso_small_data = load_results("PSO_SMALL")
        pso_big_data   = load_results("PSO_BIG")

        pso_small_times   = pso_small_data["timings"]
        pso_small_score   = pso_small_data["score"]
        pso_small_history = pso_small_data.get("history", [])

        pso_big_times   = pso_big_data["timings"]
        pso_big_score   = pso_big_data["score"]
        pso_big_history = pso_big_data.get("history", [])

    _visualise(pso_small_times, label="PSO SMALL", modes=("medium",))
    plot_optimizer_comparison_runs(
        "PSO SMALL", pso_small_history,
        "PSO BIG",   pso_big_history,
    )
    plot_pso_comparison(
        [baseline_score] * len(pso_small_history),
        pso_small_history,
    )

    
    RUN_GA = False
    ga_bounds = [(5, 60)] * 4

    if RUN_GA:
        ga_small_times, ga_small_score, ga_small_history = run_and_save_optimizer(
            "GA_SMALL", run_ga, robust_evaluate,
            bounds=ga_bounds, steps=8, population_size=6, sim_time=200, mode="ga",
        )
        ga_big_times, ga_big_score, ga_big_history = run_and_save_optimizer(
            "GA_BIG", run_ga, robust_evaluate,
            bounds=ga_bounds, steps=15, population_size=12, sim_time=300, mode="ga",
        )
    else:
        print("\  Loading GA results …")
        ga_small_data = load_results("GA_SMALL")
        ga_big_data   = load_results("GA_BIG")

        ga_small_times   = ga_small_data["timings"]
        ga_small_score   = ga_small_data["score"]
        ga_small_history = ga_small_data.get("history", [])

        ga_big_times   = ga_big_data["timings"]
        ga_big_score   = ga_big_data["score"]
        ga_big_history = ga_big_data.get("history", [])

    _visualise(ga_small_times, label="GA SMALL", modes=("medium",))
    plot_optimizer_comparison_runs(
        "GA SMALL", ga_small_history,
        "GA BIG",   ga_big_history,
    )
    plot_ga_comparison(
        [baseline_score] * len(ga_small_history),
        ga_small_history,
    )

    # ── Final comparison ───────────────────────────────────────────────
    plot_final_comparison(
        baseline_score,
        [h["global_best"] for h in pso_big_history],
        [h["global_best"] for h in ga_big_history],
    )

    pygame.quit()


if __name__ == "__main__":
    main()