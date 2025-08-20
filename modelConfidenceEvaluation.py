#!/usr/bin/env python3
# summarize_confidence.py
import json, os

SUMMARY_PATH = os.path.join("savedModels", "bayesian_eval_summary.json")
TOPK = None  # set to an int (e.g., 3) to only show top-K; keep None to show all

def confidence_from_var(var_mean: float) -> float:
    # Lower variance → higher confidence (simple monotonic mapping)
    return 1.0 / (1.0 + var_mean)

def main():
    if not os.path.exists(SUMMARY_PATH):
        print(f"[Error] File not found: {SUMMARY_PATH}")
        return

    with open(SUMMARY_PATH, "r") as f:
        summary = json.load(f)

    rows = []
    for model_id, res in summary.items():
        unc = res.get("uncertainty_summary", {})
        metrics = res.get("val_metrics_at_t", {})
        var_mean = float(unc.get("predictive_variance_mean", float("nan")))
        var_max  = float(unc.get("predictive_variance_max", float("nan")))
        dice     = float(metrics.get("dice", float("nan")))
        iou      = float(metrics.get("iou", float("nan")))
        auc      = float(metrics.get("auc", float("nan")))
        prec     = float(metrics.get("precision", float("nan")))
        rec      = float(metrics.get("recall", float("nan")))
        tstar    = float(res.get("best_threshold", float("nan")))
        conf     = confidence_from_var(var_mean)

        rows.append((model_id, var_mean, conf, dice, iou, auc, prec, rec, tstar, var_max))

    # Sort by lowest mean variance (tie-break: higher Dice)
    rows.sort(key=lambda x: (x[1], -x[3]))
    if TOPK is not None:
        rows = rows[:TOPK]

    print("Most confident models (lower mean variance first):")
    header = f"{'#':>2}  {'model':<30}  {'mean_var':>9}  {'conf':>10}  {'Dice':>6}  {'IoU':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}  {'t*':>5}"
    print(header)
    print("-" * len(header))
    for i, (mid, v, conf, dice, iou, auc, prec, rec, tstar, vmax) in enumerate(rows, 1):
        print(f"{i:>2}  {mid:<30}  {v:9.3g}  {conf:10.6f}  {dice:6.3f}  {iou:6.3f}  {auc:6.3f}  {prec:6.3f}  {rec:6.3f}  {tstar:5.2f}")

    if rows:
        best = rows[0]
        print("\nBest (by confidence):")
        print(f"  {best[0]}")
        print(f"  mean_var={best[1]:.6g}  confidence≈{best[2]:.6f}  Dice@t*={best[3]:.4f}  t*={best[8]:.2f}")

if __name__ == "__main__":
    main()


#python summarize_confidence.py
