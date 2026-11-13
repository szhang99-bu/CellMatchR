"""Summarize source-held-out Nissl recall candidate scores and tradeoffs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from finetune_cytodark import summarize
from recall_experiment import DEFAULT_OUTPUT


def measures(rows):
    totals = summarize(rows)
    p, r = totals["precision"], totals["recall"]
    totals["f2"] = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
    totals["f3"] = 10 * p * r / (9 * p + r) if 9 * p + r else 0.0
    return totals


def main() -> None:
    combinations = defaultdict(dict)
    for fold in range(4):
        files = sorted((DEFAULT_OUTPUT / f"fold_{fold}/scores").glob("*.json"))
        if len(files) != 16:
            raise RuntimeError(f"Fold {fold}: expected 16 score files, found {len(files)}")
        for path in files:
            data = json.loads(path.read_text())
            combinations[path.stem][fold] = data
    table = []
    for key, by_fold in combinations.items():
        if len(by_fold) != 4:
            raise RuntimeError(f"Incomplete fold set for {key}")
        rows = [row for fold in range(4) for row in by_fold[fold]["patches"]]
        macaque = [row for row in rows if row["group"] == "Macaque | Visual Cortex"]
        table.append({"key": key, "setting": by_fold[0]["setting"],
                      "overall": measures(rows), "macaque": measures(macaque)})
    table.sort(key=lambda item: item["overall"]["f3"], reverse=True)
    # Select settings for each target fold using scores from the *other* folds.
    crossfits = {}
    for metric in ("f2", "f3"):
        crossfit = []
        selected_rows = []
        for target in range(4):
            ranking = []
            for key, by_fold in combinations.items():
                training_rows = [row for fold in range(4) if fold != target
                                 for row in by_fold[fold]["patches"]]
                ranking.append((measures(training_rows)[metric], key))
            _, chosen = max(ranking)
            rows = combinations[chosen][target]["patches"]
            selected_rows.extend(rows)
            crossfit.append({"target_fold": target, "chosen": chosen,
                             "score": measures(rows)})
        crossfits[metric] = {
            "selection": crossfit, "overall": measures(selected_rows),
            "macaque": measures([row for row in selected_rows
                                  if row["group"] == "Macaque | Visual Cortex"]),
        }
    result = {"all_candidates": table, "crossfit_f2": crossfits["f2"]["selection"],
              "crossfit_overall": crossfits["f2"]["overall"],
              "crossfit_macaque": crossfits["f2"]["macaque"],
              "crossfit_f3": crossfits["f3"]["selection"],
              "crossfit_f3_overall": crossfits["f3"]["overall"],
              "crossfit_f3_macaque": crossfits["f3"]["macaque"],
              "note": "Every target fold's setting was selected using other folds only; "
                      "the candidate weights for a target fold never trained on its source fields."}
    (DEFAULT_OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for item in table[:10]:
        print(item["key"], "overall F3", item["overall"]["f3"],
              "macaque recall", item["macaque"]["recall"],
              "macaque FN", item["macaque"]["fn"])
    print("crossfit F3", result["crossfit_f3_overall"],
          result["crossfit_f3_macaque"])


if __name__ == "__main__":
    main()
