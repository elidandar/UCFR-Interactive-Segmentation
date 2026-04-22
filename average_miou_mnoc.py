import re
import numpy as np
from collections import defaultdict
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


# =========================
# UTILITIES
# =========================
def safe_mean(vals):
    return np.mean(vals) if len(vals) > 0 else np.nan

def safe_std(vals):
    return np.std(vals) if len(vals) > 1 else 0.0


# =========================
# PARSER
# =========================
def parse_eval_logs(log_path):
    log_path = Path(log_path)
    text = log_path.read_text()

    results = defaultdict(lambda: {
        "noc": defaultdict(lambda: defaultdict(list)),
        "miou": defaultdict(lambda: defaultdict(list))
    })

    model_blocks = re.split(r"Eval results for model:\s*", text)

    for block in model_blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue

        model_name = lines[0].strip()

        for line in lines[1:]:
            if not line.strip().startswith("|"):
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]

            if len(parts) < 6:
                continue

            # Skip header row
            if parts[1].lower() == "dataset":
                continue
            dataset = parts[1]

            # Parse NoC
            try:
                results[model_name]["noc"]["80"][dataset].append(float(parts[2]))
                results[model_name]["noc"]["85"][dataset].append(float(parts[3]))
                results[model_name]["noc"]["90"][dataset].append(float(parts[4]))
                results[model_name]["noc"]["95"][dataset].append(float(parts[5]))
            except ValueError:
                pass

            # Parse mIoU
            #miou_matches = re.findall(r"mIoU@(\d+)=([\d.]+)%", line)
            miou_matches = re.findall(r"mIoU@(\d+)\s*=\s*([\d.]+)%", line)
            for click_str, val_str in miou_matches:
                click = int(click_str)
                val = float(val_str)
                results[model_name]["miou"][click][dataset].append(val)

    return results

# =========================
# FILTER MODELS
# =========================
def filter_models(results, keywords):
    return {
        model: data
        for model, data in results.items()
        if any(k in model for k in keywords)
    }

# =========================
# DATASET WINNERS (PER LEVEL)
# =========================
def dataset_winners(results, level):
    winners = {}
    models = list(results.keys())

    datasets = set()
    for model in models:
        datasets.update(results[model]["noc"][level].keys())

    for d in sorted(datasets):
        best_model = None
        best_val = float("inf")

        for model in models:
            vals = results[model]["noc"][level].get(d, [])
            mean_val = safe_mean(vals)

            if np.isnan(mean_val):
                continue

            if mean_val < best_val:
                best_val = mean_val
                best_model = model

        winners[d] = (best_model, best_val if best_model else np.nan)

    return winners

# =========================
# BALANCED SCORE (PER LEVEL)
# =========================

def balanced_score(results, level):
    scores = {}

    common_datasets = set.intersection(*[
        set(data["noc"][level].keys()) for data in results.values()
    ])

    for model, data in results.items():
        vals = [
            safe_mean(data["noc"][level][d])
            for d in common_datasets
            if d in data["noc"][level]
        ]
        scores[model] = safe_mean(vals)

    return scores



# =========================
# WIN COUNT (PER LEVEL)
# =========================
def win_count(results, level):
    winners = dataset_winners(results, level)
    counts = defaultdict(int)

    for d, (m, _) in winners.items():
        if m:
            counts[m] += 1
    return counts

# =========================
# PRINT (PER LEVEL)
# =========================
def print_results_per_level(results):
    for level in ["80", "85", "90", "95"]:
        print(f"\n\n==============================")
        print(f"=== RESULTS FOR NoC@{level} ===")
        print(f"==============================")

        # Winners
        print("\n--- Dataset-wise winners ---")
        winners = dataset_winners(results, level)
        for d, (m, v) in winners.items():
            if m:
                print(f"{d:10s}: {m} ({v:.2f})")
            else:
                print(f"{d:10s}: No data")

        # Balanced score
        print("\n--- Balanced Score ---")
        b_scores = balanced_score(results, level)
        sorted_models = sorted(b_scores.items(), key=lambda x: x[1])
        for rank, (m, s) in enumerate(sorted_models, 1):
            print(f"{rank}. {m}: {s:.3f}")

        best = min(
            b_scores,
            key=lambda x: b_scores[x] if not np.isnan(b_scores[x]) else np.inf
        )
        print(f"\n🏆 Best (balanced): {best}")

        # Win count
        print("\n--- Win Count ---")
        wc = win_count(results, level)
        for m, c in wc.items():
            print(f"{m}: {c} wins")

# =========================
# EXPORT
# =========================
def export_scores(results, export_dir):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export NoC
    noc_rows = []
    for model, data in results.items():
        for lvl in ["80", "85", "90", "95"]:
            for d, vals in data["noc"][lvl].items():
                noc_rows.append({
                    "Model": model,
                    "Dataset": d,
                    "Level": f"NoC@{lvl}",
                    "Mean": safe_mean(vals),
                    "Std": safe_std(vals)
                })
    pd.DataFrame(noc_rows).to_csv(export_dir / "dataset_noc_scores.csv", index=False)

    # Export mIoU
    miou_rows = []
    for model, data in results.items():
        for click, d_vals in data["miou"].items():
            for d, vals in d_vals.items():
                miou_rows.append({
                    "Model": model,
                    "Dataset": d,
                    "Click": click,
                    "Mean_mIoU": safe_mean(vals),
                    "Std_mIoU": safe_std(vals)
                })
    pd.DataFrame(miou_rows).to_csv(export_dir / "dataset_miou_scores.csv", index=False)

# =========================
# PLOTTING
# =========================
def plot_per_level(results, export_dir):
    export_dir = Path(export_dir)

    for level in ["80", "85", "90", "95"]:
        datasets = sorted({
            d for model in results
            for d in results[model]["noc"][level]
        })
        if not datasets: continue

        x = np.arange(len(datasets))
        n_models = len(results)
        width = 0.8 / n_models
    
        plt.figure(figsize=(12, 6))

        for i, (model, data) in enumerate(sorted(results.items())):
            means = []
            stds = []

            for d in datasets:
                vals = data["noc"][level].get(d, [])
                means.append(safe_mean(vals))
                stds.append(safe_std(vals))

            plt.bar(
                x + i * width,
                means,
                width,
                #yerr=stds,
                capsize=3,
                label=model
            )

            # for xi, mean in zip(x + i * width, means):
            #     if not np.isnan(mean):
            #         plt.text(
            #             xi,
            #             mean - 1.5,
            #             f"{mean:.1f}",
            #             ha='center',
            #             va='top',
            #             fontsize=6,
            #             color='black'
            #         )

        plt.xticks(x + width * (len(results) - 1) / 2, datasets, rotation=45)
        plt.ylabel(f"NoC@{level}")
        plt.title(f"Dataset-wise Comparison (NoC@{level})")
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(export_dir / f"comparison_noc_{level}.png", dpi=300)
        plt.close()

def plot_noc_curves_per_dataset(results, export_dir):
    export_dir = Path(export_dir)
    levels = ["80", "85", "90", "95"]
    x = [80, 85, 90, 95]

    datasets = sorted({
        d for model in results
        for lvl in levels
        for d in results[model]["noc"][lvl]
    })

    for dataset in datasets:
        plt.figure(figsize=(8, 5))

        for model, data in results.items():
            y = []
            for lvl in levels:
                vals = data["noc"][lvl].get(dataset, [])
                y.append(safe_mean(vals))

            if all(np.isnan(v) for v in y):
                continue

            plt.plot(x, y, marker='o', label=model)

        plt.xlabel("IoU Threshold (%)")
        plt.ylabel("NoC (lower is better)")
        plt.title(f"NoC Curve - {dataset}")
        plt.xticks(x)
        plt.grid(True)
        plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(export_dir / f"noc_curve_{dataset}.png", dpi=300)
        plt.close()


# =========================
# MIOU PRINTING FUNCTIONS
# =========================
def miou_dataset_winners(results, click):
    winners = {}
    models = list(results.keys())

    datasets = set()
    for model in models:
        datasets.update(results[model]["miou"].get(click, {}).keys())

    for d in sorted(datasets):
        best_model = None
        best_val = -float("inf") # HIGHER is better for mIoU

        for model in models:
            vals = results[model]["miou"].get(click, {}).get(d, [])
            mean_val = safe_mean(vals)

            if np.isnan(mean_val):
                continue

            if mean_val > best_val: # Look for maximum
                best_val = mean_val
                best_model = model

        winners[d] = (best_model, best_val if best_model else np.nan)

    return winners

def miou_balanced_score(results, click):
    scores = {}
    for model, data in results.items():
        dataset_means = []
        for d, vals in data["miou"].get(click, {}).items():
            m = safe_mean(vals)
            if not np.isnan(m):
                dataset_means.append(m)
        scores[model] = safe_mean(dataset_means)
    return scores

def miou_win_count(results, click):
    winners = miou_dataset_winners(results, click)
    counts = defaultdict(int)
    for d, (m, _) in winners.items():
        if m:
            counts[m] += 1
    return counts

def print_miou_results_per_click(results, target_clicks=[1, 5, 10, 20]):
    for click in target_clicks:
        print(f"\n\n==============================")
        print(f"=== RESULTS FOR mIoU @ {click} Clicks ===")
        print(f"==============================")

        # Winners
        print("\n--- Dataset-wise winners ---")
        winners = miou_dataset_winners(results, click)
        for d, (m, v) in winners.items():
            if m:
                print(f"{d:10s}: {m} ({v:.2f}%)")
            else:
                print(f"{d:10s}: No data")

        # Balanced score
        print("\n--- Balanced Score ---")
        b_scores = miou_balanced_score(results, click)
        sorted_models = sorted(b_scores.items(), key=lambda x: x[1])
        for rank, (m, s) in enumerate(sorted_models, 1):
            print(f"{rank}. {m}: {s:.3f}")

        # Best is MAX for mIoU
        valid_scores = {k: v for k, v in b_scores.items() if not np.isnan(v)}
        if valid_scores:
            best = max(valid_scores, key=valid_scores.get)
            print(f"\n🏆 Best (balanced): {best}")
        else:
            print(f"\n🏆 Best (balanced): N/A")

        # Win count
        print("\n--- Win Count ---")
        wc = miou_win_count(results, click)
        for m, c in wc.items():
            print(f"{m}: {c} wins")



def plot_miou_comparison_per_click(results, export_dir, target_clicks=[1, 5, 10, 20]):
    """
    Generates bar charts comparing mIoU across datasets for specific click counts.
    Creates files like: comparison_miou_click_5.png
    """
    export_dir = Path(export_dir)

    # Collect all unique datasets
    datasets = sorted({
        d for model in results
        for click in results[model]["miou"]
        for d in results[model]["miou"][click]
    })

    if not datasets: 
        return

    for click in target_clicks:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(datasets))
        n_models = len(results)
        width = 0.8 / n_models
        
        plotted_anything = False

        for i, (model, data) in enumerate(results.items()):
            means = []
            stds = []

            for d in datasets:
                vals = data["miou"].get(click, {}).get(d, [])
                means.append(safe_mean(vals))
                stds.append(safe_std(vals))
            
            # Only plot if there's actual data (not all NaNs)
            if not all(np.isnan(m) for m in means):
                # Replace NaNs with 0 for bar plotting purposes so it doesn't crash, 
                # or just let matplotlib handle it (usually ignores NaNs).
                #means_clean = [0 if np.isnan(m) else m for m in means]
                #plt.bar(x + i * width, means_clean, width, label=model)
                plt.bar(
                    x + i * width,
                    means,
                    width,
                    #yerr=stds,
                    capsize=3,
                    label=model
                )

                # for xi, mean in zip(x + i * width, means):
                #     if not np.isnan(mean):
                #         plt.text(
                #             xi,
                #             mean - 1.5,
                #             f"{mean:.1f}",
                #             ha='center',
                #             va='top',
                #             fontsize=6,
                #             color='black'
                #         )
                plotted_anything = True

        if plotted_anything:
            plt.xticks(x + width * (len(results) - 1) / 2, datasets, rotation=45)
            plt.ylabel(f"mIoU @ {click} Clicks (%)")
            plt.title(f"Dataset-wise Comparison (mIoU @ Click {click})")
            
            # Put legend outside the plot if it gets crowded, or keep it standard
            #plt.legend(fontsize=8)
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            plt.savefig(export_dir / f"comparison_miou_click_{click}.png", dpi=300)
        
        plt.close()


def plot_miou_curves_per_dataset(results, export_dir):
    export_dir = Path(export_dir)

    datasets = sorted({
        d for model in results
        for click in results[model]["miou"]
        for d in results[model]["miou"][click]
    })

    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        
        plotted_anything = False
        max_clicks_overall = 0

        for model, data in results.items():
            clicks = sorted(data["miou"].keys())
            if not clicks:
                continue

            x = []
            y = []
            for click in clicks:
                vals = data["miou"][click].get(dataset, [])
                mean_val = safe_mean(vals)
                if not np.isnan(mean_val):
                    x.append(click)
                    y.append(mean_val)

            if x and y:
                plt.plot(x, y, marker='o', markersize=4, label=model)
                plotted_anything = True
                max_clicks_overall = max(max_clicks_overall, max(x))

        plt.xlabel("Number of Clicks")
        plt.ylabel("mIoU (%)")
        plt.title(f"mIoU Improvement per Click - {dataset}")
        
        # Determine x-ticks safely (fallback to 20 if empty)
        if max_clicks_overall == 0:
            max_clicks_overall = 20
            
        plt.xticks(range(1, max_clicks_overall + 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Only call legend if we actually plotted something (Fixes the UserWarning)
        if plotted_anything:
            #plt.legend(fontsize=8)
            plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        plt.savefig(export_dir / f"miou_curve_{dataset}.png", dpi=300)
        plt.close()


# =========================
# RELATIVE IMPROVEMENT
# =========================
def plot_relative_improvement(
    results,
    export_dir,
    click=1,
    baseline="no_ucfr",
    paper_model="thresh_049_051:cf1_clk_True"
):
    export_dir = Path(export_dir)

    datasets = sorted({
        d for m in results
        for d in results[m]["miou"].get(click, {})
    })

    x = np.arange(len(datasets))
    n_models = len(results)
    width = 0.8 / n_models

    plt.figure(figsize=(12, 6))

    for i, (model, data) in enumerate(sorted(results.items())):
        if model == baseline:
            continue

        improvements = []
        sig_flags = []

        for d in datasets:
            base_vals = results[baseline]["miou"][click].get(d, [])
            vals = data["miou"][click].get(d, [])

            base = safe_mean(base_vals)
            val = safe_mean(vals)

            if np.isnan(base) or np.isnan(val):
                improvements.append(np.nan)
                continue

            improvements.append(val - base)

        bars = plt.bar(
            x + i * width,
            improvements,
            width,
            label=model,
            #alpha=1.0 if model == paper_model else 0.7
        )

        # for xi, val in zip(x + i * width, improvements):
        #     if np.isnan(val):
        #         continue

        #     plt.text(
        #         xi,
        #         val + (0.1 if val >= 0 else -0.1),
        #         f"{val:.2f}",
        #         ha='center',
        #         va='top',
        #         fontsize=6,
        #         color='black'
        #     )


    #plt.axhline(0, color='black', linestyle='--')
    plt.xticks(x + width, datasets, rotation=45)

    plt.ylabel("Δ mIoU vs baseline")
    plt.title(f"Relative Improvement (Click {click})")

    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(export_dir / f"relative_improvement_click_{click}.png", dpi=300)
    plt.close()



import numpy as np

# -------------------------
# SAFE HELPERS
# -------------------------
def safe_mean(vals):
    return np.mean(vals) if len(vals) > 0 else np.nan

def esc(x):
    """Escape LaTeX special characters"""
    return str(x).replace("_", r"\_")


def fmt(v, best):
    """format + bold best"""
    if np.isnan(v):
        return "--"
    txt = f"{v:.2f}"
    if not np.isnan(best) and v == best:
        txt = f"\\textbf{{{txt}}}"
    return txt


# -------------------------
# MAIN FUNCTION
# -------------------------
def generate_table(
    results,
    datasets,
    miou_clicks=[1, 5, 10, 20],
    noc_levels=[85, 90]
):

    models = sorted(results.keys())

    # keep only datasets that exist
    available = {
        d for m in results
        for lvl in results[m]["noc"]
        for d in results[m]["noc"][lvl]
    }
    datasets = [d for d in datasets if d in available]

    # -------------------------
    # compute best values
    # -------------------------
    best = {}

    for d in datasets:

        # mIoU (maximize)
        for c in miou_clicks:
            vals = []
            for m in models:
                v = safe_mean(results[m]["miou"].get(c, {}).get(d, []))
                if not np.isnan(v):
                    vals.append(v)
            best[(d, c)] = max(vals) if vals else np.nan

        # NoC (minimize)
        for lvl in noc_levels:
            vals = []
            for m in models:
                v = safe_mean(results[m]["noc"][str(lvl)].get(d, []))
                if not np.isnan(v):
                    vals.append(v)
            best[(d, lvl)] = min(vals) if vals else np.nan

    # -------------------------
    # table structure
    # -------------------------
    total_cols = len(miou_clicks) + len(noc_levels)

    table = r"""
\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{l""" + "c" * (len(datasets) * total_cols) + r"""}
\toprule
"""

    # -------------------------
    # Dataset header row
    # -------------------------
    table += "Model "

    for d in datasets:
        table += f"& \\multicolumn{{{total_cols}}}{{c}}{{{esc(d)}}} "

    table += r"\\"

    # cmidrules
    table += "\n"
    start = 2
    for d in datasets:
        end = start + total_cols - 1
        table += f"\\cmidrule(lr){{{start}-{end}}}"
        start = end + 1

    # -------------------------
    # Metric header row
    # -------------------------
    table += "\nModel "

    for _ in datasets:
        for c in miou_clicks:
            table += f"& mIoU$_{{{c}}}$ "
        for lvl in noc_levels:
            table += f"& NoC@{lvl} "

    table += r"\\ \midrule"
    table += "\n"

    # -------------------------
    # DATA ROWS
    # -------------------------
    for m in models:
        row = [f"\\texttt{{{esc(m)}}}"]

        for d in datasets:

            # mIoU block
            for c in miou_clicks:
                v = safe_mean(results[m]["miou"].get(c, {}).get(d, []))
                row.append(fmt(v, best[(d, c)]))

            # NoC block
            for lvl in noc_levels:
                v = safe_mean(results[m]["noc"][str(lvl)].get(d, []))
                row.append(fmt(v, best[(d, lvl)]))

        table += " & ".join(row) + r" \\" + "\n"

    # -------------------------
    # END TABLE
    # -------------------------
    table += r"""
\bottomrule
\end{tabular}
\caption{Dataset-wise comparison of mIoU (higher is better) and NoC (lower is better).}
\end{table*}
"""

    return table


def save_latex_table(latex_str, export_dir, filename="results_table.tex"):
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    file_path = export_dir / filename

    with open(file_path, "w") as f:
        f.write(latex_str)

    print(f"LaTeX table saved to: {file_path}")



# =========================
# MAIN
# =========================
if __name__ == "__main__":
    log_path = "./experiments/others/cf1_clk_True_cvpr_NoBRS_20_ious.txt"
    export_dir = Path("./experiments/others/thresh_comparison_results")

    results = parse_eval_logs(log_path)
    print("Available models and their datasets:\n")
    for model in results:
        print(model, sorted(results[model]["noc"]["80"].keys()))

    # focus models
    results = filter_models(results, [
        "no_ucfr",
        "thresh_049_051:cf1_clk_True",
        "thresh_05_05:cf1_clk_True",
        "thresh_045_055:cf1_clk_True",
        "thresh_04_06:cf1_clk_True"
    ])

    print_results_per_level(results)
    
    # Add this line here to print the mIoU terminal summaries:
    print_miou_results_per_click(results, target_clicks=[1, 5, 10, 20])
    plot_per_level(results, export_dir)
    plot_noc_curves_per_dataset(results, export_dir)
    plot_miou_curves_per_dataset(results, export_dir)
    plot_miou_comparison_per_click(results, export_dir, target_clicks=[1, 5, 10, 20])
    plot_relative_improvement(results, export_dir, click=1)

    export_scores(results, export_dir)

    latex_table = generate_table(
    results,
    datasets=["Berkeley", "SBD", "DAVIS"],
    miou_clicks=[1],
    noc_levels=[85, 90]
)

    save_latex_table(
        latex_table,
        export_dir,
        filename="main_results_table.tex"
    )