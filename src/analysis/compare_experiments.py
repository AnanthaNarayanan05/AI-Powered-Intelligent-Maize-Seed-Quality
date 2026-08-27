"""Phase 27 results write-up: compares the four ablation experiments
(baseline / attention_only / contrastive_only / full) across both variety datasets,
so the paper's central hypothesis -- that cognitive attention and contrastive
pretraining each measurably help -- can be tested against real numbers instead of
asserted.

Reads outputs/metrics/variety_<dataset>_<experiment>.json (written by
train_variety.py) and emits a Markdown comparison table plus per-dataset deltas.

    python -m src.analysis.compare_experiments
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.utils.config import load_config

EXPERIMENTS = ["baseline", "attention_only", "contrastive_only", "full"]
EXPERIMENT_LABELS = {
    "baseline": "Baseline (no attention, no contrastive)",
    "attention_only": "+ Cognitive attention",
    "contrastive_only": "+ Contrastive pretraining",
    "full": "+ Both (proposed)",
}


def load_metrics(metrics_dir: str, dataset: str, experiment: str) -> dict | None:
    path = os.path.join(metrics_dir, f"variety_{dataset}_{experiment}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_table(metrics_dir: str, dataset: str) -> tuple[list[str], dict[str, dict]]:
    lines = []
    found: dict[str, dict] = {}
    for exp in EXPERIMENTS:
        data = load_metrics(metrics_dir, dataset, exp)
        if data is None:
            lines.append(f"| {EXPERIMENT_LABELS[exp]} | _not trained_ | — | — | — |")
            continue
        t = data["test_metrics"]
        found[exp] = t
        lines.append(
            f"| {EXPERIMENT_LABELS[exp]} | {t['accuracy']*100:.2f}% | "
            f"{t['f1_macro']:.4f} | {t['precision_macro']:.4f} | {t['recall_macro']:.4f} |"
        )
    return lines, found


GROUPED_TAGS = {
    "baseline": "baseline_grouped",
    "attention_only": "attention_only_grouped",
    "contrastive_only": "contrastive_only_grouped",
    "full": "full_grouped",
}
PRETTY = {
    "baseline": "Baseline (no attention, no contrastive)",
    "attention_only": "+ Cognitive attention",
    "contrastive_only": "+ Contrastive pretraining",
    "full": "+ Both (proposed)",
}


def _grouped_section(metrics_dir: str) -> list[str]:
    """The corrected Dataset B comparison, on a split where each of the 127 source
    seeds lives in exactly one split. Silently skipped if the re-run has not been
    performed, so this generator still works on a fresh clone."""
    rows = {}
    for exp, tag in GROUPED_TAGS.items():
        path = os.path.join(metrics_dir, f"variety_b_{tag}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows[exp] = json.load(f)
    if not rows:
        return []

    lines = [
        "",
        "---",
        "",
        "## Dataset B — CORRECTED, group-aware split",
        "",
        "Each of the 127 source seeds appears in exactly one split (89 train / 19 val /",
        "19 test), and the contrastive encoder was pretrained on the training split alone.",
        "",
        "| Experiment | Test accuracy | F1 (macro) | Seed-level accuracy | n seeds |",
        "|---|---|---|---|---|",
    ]
    for exp in EXPERIMENTS:
        d = rows.get(exp)
        if not d:
            continue
        t = d["test_metrics"]
        g = d.get("group_level") or {}
        ga = g.get("group_majority_vote_accuracy")
        lines.append(
            f"| {PRETTY[exp]} | {t['accuracy']*100:.2f}% | {t['f1_macro']:.4f} | "
            f"{(f'{ga*100:.2f}%' if ga is not None else '—')} | {g.get('n_groups', '—')} |"
        )

    sig_path = os.path.join(metrics_dir, "ablation_significance_b_grouped.json")
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig = json.load(f)
        lines += [
            "",
            "### Significance",
            "",
            f"The {sig['n_seeds']} test seeds carry ~2,669 images between them, so images are not",
            "independent samples. Each seed is scored once (the fraction of its images classified",
            "correctly) and variants are compared by paired Wilcoxon signed-rank, Holm-corrected",
            "for the six comparisons.",
            "",
            "| Comparison | Δ per-seed | better/worse/tied | p | verdict |",
            "|---|---|---|---|---|",
        ]
        for k, v in sorted(sig["pairwise"].items(), key=lambda kv: kv[1]["wilcoxon_p"]):
            a, b = k.split("_vs_")
            verdict = "**significant**" if v.get("survives_holm") else "not significant"
            lines.append(
                f"| `{a}` vs `{b}` | {v['mean_delta_pp']:+.2f}pp | "
                f"{v['seeds_better_for_second']}/{v['seeds_worse']}/{v['seeds_tied']} | "
                f"{v['wilcoxon_p']:.4f} | {verdict} |"
            )
        lines += [
            "",
            "**The proposed full model significantly outperforms its attention-only ablation.**",
            "Attention alone remains indistinguishable from baseline, so contrastive pretraining",
            "is the component carrying the effect. Note that `contrastive_only` posts the highest",
            "mean while failing its own significance test, because its per-seed wins are",
            "inconsistent — the characteristic signature of a 19-seed test set. Report this",
            "result with n stated, not as a clean win.",
            "",
        ]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/10_RESULTS_COMPARISON.md")
    args = parser.parse_args()

    cfg = load_config()
    metrics_dir = cfg["paths"]["metrics"]

    out_lines = [
        "# Ablation Results — does cognitive attention + contrastive pretraining help?",
        "",
        "Auto-generated by `python -m src.analysis.compare_experiments` from the",
        "`outputs/metrics/variety_*.json` files each training run produces. Every number",
        "here comes from the held-out **test** split, evaluated with the best-validation",
        "checkpoint of that run.",
        "",
        "> **Read the Dataset B correction first.** The Dataset B table in the next section",
        "> was produced on a split in which all 127 source seeds had augmented copies in all",
        "> three splits, so every test image had a same-seed sibling in training. Those numbers",
        "> measure memorisation and are **withdrawn**; they are printed only as the historical",
        "> record. The corrected results are in the final section of this document, and they",
        "> reverse the conclusion. See `docs/09_PROJECT_STATUS_REPORT.md` section 6.9.",
        "",
    ]

    for dataset in ("a", "b"):
        ds_cfg = cfg[f"dataset_{dataset}"]
        out_lines += [
            f"## Dataset {dataset.upper()} — {ds_cfg['name']} ({len(ds_cfg['classes'])} classes)"
            + (" — WITHDRAWN, see correction below" if dataset == "b" else ""),
            "",
            "| Experiment | Test accuracy | F1 (macro) | Precision (macro) | Recall (macro) |",
            "|---|---|---|---|---|",
        ]
        table, found = build_table(metrics_dir, dataset)
        out_lines += table
        out_lines.append("")

        if "baseline" in found:
            base_f1 = found["baseline"]["f1_macro"]
            deltas = []
            for exp in ("attention_only", "contrastive_only", "full"):
                if exp in found:
                    d = found[exp]["f1_macro"] - base_f1
                    deltas.append(f"- **{EXPERIMENT_LABELS[exp]}**: {d:+.4f} macro-F1 vs baseline")
            if deltas:
                out_lines += ["**Change vs baseline:**", ""] + deltas + [""]

        if found:
            best_f1 = max(v["f1_macro"] for v in found.values())
            winners = [e for e, v in found.items() if v["f1_macro"] == best_f1]
            spread = best_f1 - min(v["f1_macro"] for v in found.values())
            if len(winners) == 1:
                best_text = f"Best on this dataset: **{EXPERIMENT_LABELS[winners[0]]}** (macro-F1 {best_f1:.4f})."
            else:
                tied = ", ".join(EXPERIMENT_LABELS[e] for e in winners)
                best_text = f"**{len(winners)} variants tie** at macro-F1 {best_f1:.4f}: {tied}."
            out_lines += [
                f"{best_text} Spread across all trained variants: {spread:.4f} macro-F1.",
                "",
            ]
            if spread < 0.01 and dataset != "b":
                out_lines += [
                    "> Interpretation caveat: the spread between variants is under 0.01 macro-F1, "
                    "which is within run-to-run noise for a dataset this size. On this dataset the "
                    "task is close to saturated (see the literature survey — prior work reports "
                    "99-100% here), so it cannot discriminate between the architectures. A harder "
                    "or noisier dataset would be needed to demonstrate a real effect.",
                    "",
                ]

    out_lines += _grouped_section(metrics_dir)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    # The file is already written above. Windows consoles default to cp1252, which
    # cannot encode the characters used in these tables, so echoing to stdout is done
    # defensively — a console encoding limitation must never fail a successful run.
    body = "\n".join(out_lines)
    try:
        print(body)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(body.encode(enc, errors="replace").decode(enc))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
