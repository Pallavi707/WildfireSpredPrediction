import json
import scipy.stats as stats
import itertools

# ------------------ Load JSON File ------------------
with open("savedModels/test_results.json", "r") as f:
    metrics = json.load(f)

# ------------------ Configuration ------------------
alpha = 0.05
model_names = list(metrics.keys())
metric_names = list(metrics[model_names[0]]["per_image_metrics"].keys())
#metric_names = ['accuracy']

# ------------------ Column Names ------------------
headers = ["Model A", "Model B", "Metric", "t-stat", "p-value", "Significant"]

# ------------------ Build Table Data ------------------
table_data = []

for model_a, model_b in itertools.combinations(model_names, 2):
    for metric in metric_names:
        scores_a = metrics[model_a]["per_image_metrics"][metric]
        scores_b = metrics[model_b]["per_image_metrics"][metric]

        try:
            t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
            sig = "Yes" if p_val < alpha else "No"
            t_stat = f"{t_stat:.4f}"
            p_val = f"{p_val:.2e}"
        except Exception:
            t_stat, p_val, sig = "NA", "NA", "NA"

        table_data.append([model_a, model_b, metric, t_stat, p_val, sig])

# ------------------ Dynamic Column Widths ------------------
col_widths = [max(len(str(row[i])) for row in ([headers] + table_data)) for i in range(len(headers))]

# ------------------ Print Table ------------------
def print_row(row):
    print("| " + " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + " |")

def print_divider():
    print("+-" + "-+-".join("-" * w for w in col_widths) + "-+")

# Print Header
print_divider()
print_row(headers)
print_divider()

# Print Rows
for row in table_data:
    print_row(row)
    print_divider()
