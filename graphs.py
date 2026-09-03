import matplotlib.pyplot as plt

def plot_pso_comparison(baseline, history):

    # ✅ extract global best curve
    y = [h["global_best"] for h in history]

    plt.figure()
    plt.plot(baseline, label="Baseline")
    plt.plot(y, label="PSO")

    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("PSO vs Baseline")
    plt.legend()
    plt.grid(True)
    plt.show()

    
def plot_ga_comparison(baseline, ga_history):

    y = [h["global_best"] for h in ga_history]

    plt.figure()
    plt.plot(baseline, linestyle='--', label="Baseline")
    plt.plot(y, marker='o', label="GA")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA vs Baseline")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_optimizer_comparison_runs(label1, history1, label2, history2):

    # helper function to extract data safely
    def extract(history):
        x = []
        y = []
        iter_best = []
        avg = []

        for h in history:
            # support both PSO (iter) and GA (gen)
            idx = h.get("iter", h.get("gen"))
            best_iter = h.get("iter_best", h.get("gen_best"))
            global_best = h.get("global_best")
            avg_val = h.get("avg")

            x.append(idx)
            y.append(global_best)
            iter_best.append(best_iter)
            avg.append(avg_val)

        return x, y, iter_best, avg

    # extract both histories
    x1, y1, iter_best1, avg1 = extract(history1)
    x2, y2, iter_best2, avg2 = extract(history2)

    # =========================================================
    # 📊 1. CONVERGENCE PLOT
    # =========================================================
    plt.figure(figsize=(10, 5))

    plt.plot(x1, y1, label=label1, marker='o')
    plt.plot(x2, y2, label=label2, marker='s')

    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Fitness Score")
    plt.title(f"Convergence: {label1} vs {label2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 📋 2. COMPARISON TABLE
    # =========================================================
    max_len = max(len(history1), len(history2))
    table_data = []

    for i in range(max_len):

        row = [
            i,

            # label1
            f"{iter_best1[i]:.2f}" if i < len(iter_best1) else "-",
            f"{y1[i]:.2f}" if i < len(y1) else "-",
            f"{avg1[i]:.2f}" if i < len(avg1) else "-",

            # label2
            f"{iter_best2[i]:.2f}" if i < len(iter_best2) else "-",
            f"{y2[i]:.2f}" if i < len(y2) else "-",
            f"{avg2[i]:.2f}" if i < len(avg2) else "-"
        ]

        table_data.append(row)

    plt.figure(figsize=(12, 6))
    plt.title(f"{label1} vs {label2} - History Table")
    plt.axis("off")

    plt.table(
        cellText=table_data,
        colLabels=[
            "Step",
            f"{label1} Best",
            f"{label1} Global",
            f"{label1} Avg",
            f"{label2} Best",
            f"{label2} Global",
            f"{label2} Avg",
        ],
        cellLoc="center",
        loc="center"
    )

    plt.tight_layout()
    plt.show()
    
def plot_final_comparison(baseline_score, pso_history, ga_history):
    plt.figure(figsize=(10, 5))

    # PSO curve
    plt.plot(pso_history, label="PSO", marker='o')

    # GA curve
    plt.plot(ga_history, label="GA", marker='s')

    # Baseline (flat line)
    plt.hlines(
        baseline_score,
        xmin=0,
        xmax=max(len(pso_history), len(ga_history)),
        colors='red',
        linestyles='dashed',
        label="Baseline"
    )

    plt.xlabel("Iteration / Generation")
    plt.ylabel("Fitness (Lower = Better or Higher = Better depending on your design)")
    plt.title("PSO vs GA vs Baseline Performance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()