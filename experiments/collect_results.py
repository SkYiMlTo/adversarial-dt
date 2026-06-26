"""
Collect all experimental results and format as LaTeX tables.

Reads JSON result files from results/s1/ and results/s2/ and produces
LaTeX table code ready for insertion into the paper.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"


def format_pct(val):
    """Format a float as a percentage string."""
    return f"{val * 100:.1f}"


def collect_table1():
    """Format Table 1: Red-team evasion rate."""
    path = RESULTS_DIR / "s1" / "table1_evasion.json"
    if not path.exists():
        return "% Table 1: No data available\n"

    with open(path) as f:
        data = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Red-team evasion rate on S1 (fraction of 30 sessions with",
        r"no alarm during 60\,s of active fault). Budget $\epsilon = 0.6\,\Delta y$.}",
        r"\label{tab:redteam_results}",
        r"\small",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{Adversary} & \textbf{Fault} $\Delta y$ &",
        r"\textbf{CUSUM only} & \textbf{IWD$\vee$CUSUM} & \textbf{Sessions} \\",
        r"\midrule",
    ]

    for regime in ['whitebox', 'greybox']:
        regime_label = 'White-box (TCA)' if regime == 'whitebox' else 'Grey-box'
        for fault in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            key = str((regime, fault))
            if key in data:
                d = data[key]
                lines.append(
                    f"{regime_label:15s} & ${fault:.1f}\\sigma_\\eta$ & "
                    f"{format_pct(d['cusum_evasion_rate'])}\\% & "
                    f"{format_pct(d['combined_evasion_rate'])}\\% & "
                    f"{d['n_sessions']} \\\\"
                )
        if regime == 'whitebox':
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def collect_table3():
    """Format Table 3: SDS vs budget sweep."""
    path = RESULTS_DIR / "s1" / "table3_budget_sweep.json"
    if not path.exists():
        return "% Table 3: No data available\n"

    with open(path) as f:
        data = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Normalized SDS ($\overline{\mathrm{SDS}}$) versus plausibility",
        r"budget. S1, $\Delta y = 2\sigma_\eta$, $K = 200$. Mean $\pm$ std over 10 sessions.}",
        r"\label{tab:sds_budget}",
        r"\small",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"\textbf{Budget $\epsilon / \sigma_\eta$} &",
        r"\textbf{White-box $\overline{\mathrm{SDS}}$} &",
        r"\textbf{Grey-box $\overline{\mathrm{SDS}}$} \\",
        r"\midrule",
    ]

    for ratio in [0.25, 0.50, 0.75, 1.00, 1.50]:
        wb_key = str(('whitebox', ratio))
        gb_key = str(('greybox', ratio))

        # Handle both old (scalar) and new (dict) format
        wb_data = data.get(wb_key, 0.0)
        gb_data = data.get(gb_key, 0.0)

        if isinstance(wb_data, dict):
            wb_str = f"{wb_data['mean']:.3f}$\\pm${wb_data['std']:.3f}"
        else:
            wb_str = f"{wb_data:.3f}"

        if isinstance(gb_data, dict):
            gb_str = f"{gb_data['mean']:.3f}$\\pm${gb_data['std']:.3f}"
        else:
            gb_str = f"{gb_data:.3f}"

        lines.append(f"{ratio:.2f} & {wb_str} & {gb_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def collect_table5():
    """Format Table 5: Ablation."""
    path = RESULTS_DIR / "s1" / "table5_ablation.json"
    if not path.exists():
        return "% Table 5: No data available\n"

    with open(path) as f:
        data = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation of detection pipeline components.",
        r"S1, white-box TCA (full capability), $\epsilon = 0.75\,\sigma_\eta$.",
        r"Mean $\pm$ std over 10 sessions.}",
        r"\label{tab:iwd_ablation}",
        r"\small",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{TPR (\%)} & \textbf{FPR (\%)} \\",
        r"\midrule",
    ]

    for key, label in [
        ('cusum_only', 'CUSUM only'),
        ('iswt_only', 'ISWT (IWD) only'),
        ('combined', r'IWD$\vee$CUSUM (combined)'),
    ]:
        d = data.get(key, {})

        # Handle both old (no std) and new (with std) format
        tpr = d.get('tpr', 0)
        fpr = d.get('fpr', 0)
        tpr_std = d.get('tpr_std', None)
        fpr_std = d.get('fpr_std', None)

        if tpr_std is not None:
            tpr_str = f"{format_pct(tpr)}$\\pm${format_pct(tpr_std)}\\%"
            fpr_str = f"{format_pct(fpr)}$\\pm${format_pct(fpr_std)}\\%"
        else:
            tpr_str = f"{format_pct(tpr)}\\%"
            fpr_str = f"{format_pct(fpr)}\\%"

        lines.append(f"{label} & {tpr_str} & {fpr_str} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Collecting Results -> LaTeX Tables")
    print("=" * 70)

    tables = {
        'table1_evasion.tex': collect_table1(),
        'table3_budget.tex': collect_table3(),
        'table5_ablation.tex': collect_table5(),
    }

    for filename, content in tables.items():
        path = TABLES_DIR / filename
        with open(path, 'w') as f:
            f.write(content)
        print(f"  {filename}: written")

    print(f"\nAll tables saved to {TABLES_DIR}")


if __name__ == '__main__':
    main()
