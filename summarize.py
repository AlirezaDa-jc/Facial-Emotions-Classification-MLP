import json
import os

import pandas as pd

import config


def summarize_results():
    """Reads every results/<config_name>.json and builds one comparison table."""
    rows = []

    for fname in sorted(os.listdir(config.RESULTS_DIR)):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(config.RESULTS_DIR, fname), "r") as f:
            record = json.load(f)

        cfg = record["config"]
        rows.append({
            "config_name": record["config_name"],
            "hidden_sizes": str(cfg["hidden_sizes"]),
            "activation": cfg["activation"],
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "final_train_acc": record["train_accs"][-1],
            "final_val_acc": record["val_accs"][-1],
            "test_acc": record["test_acc"],
            "test_loss": record["test_loss"],
            "precision_macro": record["precision_macro"],
            "recall_macro": record["recall_macro"],
            "f1_macro": record["f1_macro"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by="test_acc", ascending=False).reset_index(drop=True)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "summary_table.csv")
    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved summary table to {csv_path}")

    return df


if __name__ == "__main__":
    summarize_results()
