import json
import scipy.stats as stats
import matplotlib.pyplot as plt

# ------------------ Load JSON File ------------------
with open("savedModels/per_image_metrics.json", "r") as f:
    metrics = json.load(f)

# ------------------ Choose Models and Metric ------------------
model_a = "U_Net-FOCAL+DICE"
model_b = "U_Net-WBCE+DICE"
metric_name = ["dice", "f1", "iou"]

def get_scores(model_a,model_b, metric):
        # ------------------ Extract per-image metric lists ------------------
        print(f'------------------- Extracting per-image metrics for {metric.upper()} ------------------')
        # Extract per-image metric lists
        scores_a = metrics[model_a]["per_image_metrics"][metric]
        scores_b = metrics[model_b]["per_image_metrics"][metric]

        # ------------------ Sanity Checks ------------------
        assert len(scores_a) == len(scores_b), "Both models must have the same number of samples"
        print(f"Comparing {metric.upper()} scores for {len(scores_a)} images.")


        # ------------------ Hypothesis Testing ------------------

        # Null Hypothesis (H0): No difference between the models' scores
        # Alternative Hypothesis (H1): One model has statistically significantly different performance

        # 1. Paired t-test (parametric)
        t_stat, p_val_ttest = stats.ttest_rel(scores_a, scores_b)
        print(f"\nPaired t-test:")
        print(f"t-statistic = {t_stat:.4f}, p-value (exact) = {p_val_ttest:.10e}")

        # 2. Wilcoxon Signed-Rank Test (non-parametric)
        w_stat, p_val_wilcoxon = stats.wilcoxon(scores_a, scores_b)
        print(f"\nWilcoxon signed-rank test:")
        print(f"w-statistic = {w_stat:.4f}, p-value (exact) = {p_val_wilcoxon:.10e}")


        # ------------------ Interpretation ------------------
        alpha = 0.05
        if p_val_wilcoxon < alpha:
            print(f"\nResult: Statistically significant difference in {metric.upper()} scores (p < {alpha})")
        else:
            print(f"\nResult: No statistically significant difference in {metric.upper()} scores (p ≥ {alpha})")

for metric in metric_name:
    print(f"\n--- {metric.upper()} Comparison ---")
    get_scores(model_a, model_b, metric)
