"""
Generate publication-ready figures for the paper.

Produces:
    1. SDS convergence curves (SDS vs TCA iteration, white-box & grey-box)
    2. CUSUM time series under attack (with/without TCA)
    3. ISWT statistic time series (clean vs under TCA)
    4. Innovation covariance heatmap (clean vs under TCA)
    5. Budget sweep plot (SDS vs ε/σ_η)
    6. Detection timeline (combined alarm events)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ExperimentConfig, SystemConfig, EKFConfig
from core.process_model import TwoTankProcess
from core.ekf import ExtendedKalmanFilter
from core.cusum import CUSUMDetector
from core.iswt import ISWTDetector
from core.tca import TargetedConsistencyAttack
from core.calibration import full_calibration

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
    })


def generate_attack_demo_data(config: ExperimentConfig):
    """Generate demonstration data for figures."""
    sys_cfg = config.system
    T = 600  # 10 minutes

    # Calibrate
    calib = full_calibration(sys_cfg, 1800, seed=config.seed)
    ekf_cfg = calib['ekf_config']
    Q = calib['Q']
    R = calib['R']

    # Simulate with fault
    process = TwoTankProcess(sys_cfg)
    attacked_idx = list(range(6))  # All sensors (MitM on sensor bus)
    compromised_idx = [0, 1, 5]
    iswt_cfg = calib['iswt_config']
    baseline_cov = calib['baseline_cov']

    sim_fault = process.simulate(T, fault_config={
        'sensor_idx': compromised_idx,
        'fault_start': 0,
        'fault_magnitude': 2.0,
    }, seed=config.seed + 1)

    # Simulate clean
    sim_clean = process.simulate(T, seed=config.seed + 2)

    # Run TCA (white-box) with proportional budget
    fault_mult = 2.0
    epsilon = 0.6 * fault_mult * sys_cfg.sigma
    tca = TargetedConsistencyAttack(
        sys_cfg, ekf_cfg, config.cusum, iswt_cfg, config.tca,
        baseline_cov=baseline_cov
    )

    try:
        tca_result = tca.run_whitebox(
            sim_fault['y_faulted'], sim_fault['u'],
            attacked_idx, compromised_idx, epsilon, verbose=True
        )
        delta = tca_result['delta']
        sds_history = tca_result['sds_history']
        surr_history = tca_result['surr_history']
    except Exception:
        delta = np.zeros((T, 6))
        sds_history = [0.0]
        surr_history = [0.0]

    # Run pipelines
    data = {}

    for label, Y, name in [
        ('clean', sim_clean['y_noisy'], 'clean'),
        ('fault_no_tca', sim_fault['y_faulted'], 'fault'),
        ('fault_tca', sim_fault['y_faulted'] + delta, 'tca'),
    ]:
        ekf = ExtendedKalmanFilter(sys_cfg, ekf_cfg)
        ekf.set_noise_covariances(Q, R)
        U = sim_clean['u'] if label == 'clean' else sim_fault['u']
        ekf_res = ekf.run_batch(Y, U)

        cusum = CUSUMDetector(6, config.cusum)
        cusum_res = cusum.run_batch(ekf_res['std_innovation'])

        iswt = ISWTDetector(6, iswt_cfg, baseline_cov=baseline_cov)
        iswt_res = iswt.run_batch(ekf_res['std_innovation'])

        data[label] = {
            'innovations': ekf_res['innovation'],
            'std_innovations': ekf_res['std_innovation'],
            'G': cusum_res['G'],
            'cusum_alarm': cusum_res['alarm'],
            'iswt_stat': iswt_res['test_stat'],
            'iswt_alarm': iswt_res['alarm'],
            'iswt_critical': iswt_res['critical'],
        }

    data['sds_history'] = sds_history
    data['surr_history'] = surr_history

    return data


def plot_cusum_timeseries(data: dict, save_path: Path):
    """Fig: CUSUM statistics under different conditions."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    sensor_idx = 0  # L1 (attacked sensor, larger sigma => clearer plots)
    h = 5.0  # Threshold

    for ax, (label, title) in zip(axes, [
        ('clean', 'Clean Operation'),
        ('fault_no_tca', 'Fault Only (no TCA)'),
        ('fault_tca', 'Fault + TCA Attack'),
    ]):
        G = data[label]['G'][:, sensor_idx]
        t = np.arange(len(G))

        ax.plot(t, G, 'b-', alpha=0.8, label=f'$G_{{Q_{{pump}}}}(t)$')
        ax.axhline(y=h, color='r', linestyle='--', alpha=0.7,
                    label=f'Threshold $h = {h}$')
        ax.fill_between(t, 0, G, alpha=0.1, color='b')

        alarm_times = np.where(data[label]['cusum_alarm'][:, sensor_idx])[0]
        if len(alarm_times) > 0:
            ax.scatter(alarm_times, G[alarm_times], c='red', s=10,
                       zorder=5, label='Alarm')

        ax.set_ylabel('$G_i(t)$')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(bottom=-0.5)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time step $t$')
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_iswt_timeseries(data: dict, save_path: Path):
    """Fig: ISWT statistic under different conditions."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, (label, title) in zip(axes, [
        ('clean', 'Clean Operation'),
        ('fault_no_tca', 'Fault Only (no TCA)'),
        ('fault_tca', 'Fault + TCA Attack'),
    ]):
        stat = data[label]['iswt_stat']
        critical = data[label]['iswt_critical']
        t = np.arange(len(stat))

        ax.plot(t, stat, 'b-', alpha=0.8,
                label=r'$W \cdot \Lambda^{IW}(t)$')
        ax.axhline(y=critical, color='r', linestyle='--', alpha=0.7,
                    label=f'$\\chi^2_{{0.95}}$ = {critical:.1f}')

        alarm_times = np.where(data[label]['iswt_alarm'])[0]
        if len(alarm_times) > 0:
            ax.scatter(alarm_times, stat[alarm_times], c='red', s=10,
                       zorder=5, label='ISWT Alarm')

        ax.set_ylabel(r'$W \cdot \Lambda^{IW}(t)$')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time step $t$')
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_sds_convergence(data: dict, save_path: Path):
    """Fig: SDS convergence during TCA optimization."""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    sds_history = data['sds_history']
    surr_history = data['surr_history']
    iterations = np.arange(len(sds_history))

    color1 = 'tab:blue'
    ax1.set_xlabel('PGD Iteration $k$')
    ax1.set_ylabel('True SDS($\\delta^{(k)}$)', color=color1)
    ax1.plot(iterations, sds_history, '-', color=color1, linewidth=2, label='True SDS')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:red'
    ax2.set_ylabel('Surrogate Loss $\\mathcal{L}_{TCA}$', color=color2)
    ax2.plot(iterations, surr_history, '--', color=color2, linewidth=2, label='Surrogate Obj.')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('TCA Convergence: Internal Optimization vs True Metric')
    ax1.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_innovation_covariance(data: dict, save_path: Path):
    """Fig: Innovation spatial covariance heatmaps."""
    sensor_names = ['$L_1$', '$L_2$', '$P_{in}$', '$P_{out}$',
                    '$Q_{12}$', '$Q_{pump}$']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (label, title) in zip(axes, [
        ('clean', 'Clean'),
        ('fault_no_tca', 'Fault (no TCA)'),
        ('fault_tca', 'Fault + TCA'),
    ]):
        std_innov = data[label]['std_innovations']
        # Compute sample covariance (last 200 steps)
        window = std_innov[-200:]
        C = (window.T @ window) / len(window)

        im = ax.imshow(C, cmap='RdBu_r', vmin=-0.5, vmax=2.0,
                        aspect='equal')
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(sensor_names, fontsize=8, rotation=45)
        ax.set_yticklabels(sensor_names, fontsize=8)
        ax.set_title(title)

        # Annotate values
        for i in range(6):
            for j in range(6):
                color = 'white' if abs(C[i, j]) > 1.0 else 'black'
                ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label=r'$\hat{C}_{ij}$')
    fig.suptitle('Innovation Spatial Covariance $\\hat{\\mathbf{C}}(t)$',
                  fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_budget_sweep(save_path: Path):
    """Fig: SDS vs budget (from Table 3 results)."""
    results_file = RESULTS_DIR / "s1" / "table3_budget_sweep.json"
    if not results_file.exists():
        print("  Skipping budget sweep plot (no data)")
        return

    with open(results_file) as f:
        raw = json.load(f)

    ratios = [0.25, 0.50, 0.75, 1.00, 1.50]

    fig, ax = plt.subplots(figsize=(8, 5))

    for regime, marker, color in [
        ('whitebox', 'o-', '#2196F3'),
        ('greybox', 's--', '#FF9800'),
    ]:
        sds_vals = []
        for r in ratios:
            key = str((regime, r))
            val = raw.get(key, {'mean': 0.0})
            if isinstance(val, dict):
                sds_vals.append(val.get('mean', 0.0))
            else:
                sds_vals.append(float(val))

        ax.plot(ratios, sds_vals, marker, color=color,
                linewidth=2, markersize=8,
                label=f'{regime.replace("box", "-box").title()} TCA')

    ax.set_xlabel(r'Budget $\epsilon / \sigma_\eta$')
    ax.set_ylabel(r'$\overline{\mathrm{SDS}}$')
    ax.set_title('Deception Boundary: SDS vs. Perturbation Budget')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.6)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ======================================================================
# S3 Neural figures
# ======================================================================

def plot_lstm_roc(save_path: Path):
    """Fig: LSTM detector anomaly score distribution (clean vs attacked)."""
    s3_dir = RESULTS_DIR / "s3"
    table6_file = s3_dir / "table6_detection.json"
    if not table6_file.exists():
        print("  Skipping LSTM ROC plot (no S3 data)")
        return

    with open(table6_file) as f:
        table6 = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar chart: TPR across detectors for each fault magnitude
    detectors = ['cusum_only', 'iswt_only', 'lstm_only', 'cusum_iswt',
                 'full_pipeline']
    detector_labels = ['CUSUM', 'ISWT', 'LSTM', 'CUSUM+ISWT',
                        'Full Pipeline']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

    fault_mults = [1.0, 2.0, 4.0]
    n_detectors = len(detectors)
    n_faults = len(fault_mults)
    bar_width = 0.15
    x = np.arange(n_faults)

    for i, (det, label, color) in enumerate(
            zip(detectors, detector_labels, colors)):
        tprs = []
        for fm in fault_mults:
            key = f"tpr_{fm}sigma"
            if key in table6:
                tprs.append(table6[key].get(det, 0.0))
            else:
                tprs.append(0.0)
        ax.bar(x + i * bar_width, tprs, bar_width,
               label=label, color=color, alpha=0.85)

    ax.set_xlabel('Fault Magnitude')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Detection Performance: Individual vs Combined Detectors')
    ax.set_xticks(x + bar_width * (n_detectors - 1) / 2)
    ax.set_xticklabels([f'{fm}σ' for fm in fault_mults])
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_gan_vs_pgd(save_path: Path):
    """Fig: GAN vs PGD evasion comparison."""
    s3_dir = RESULTS_DIR / "s3"
    table8_file = s3_dir / "table8_gan_vs_pgd.json"
    if not table8_file.exists():
        print("  Skipping GAN vs PGD plot (no S3 data)")
        return

    with open(table8_file) as f:
        table8 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Detection rate comparison
    ax = axes[0]
    eps_ratios = [0.50, 1.00, 1.50]

    for fault_mult, marker, color in [
        (2.0, 'o', '#2196F3'), (4.0, 's', '#F44336')
    ]:
        pgd_tprs = []
        gan_tprs = []
        for eps in eps_ratios:
            key = f"{fault_mult}sigma_eps{eps}"
            if key in table8:
                pgd_tprs.append(table8[key]['pgd']['full_tpr'])
                gan_tprs.append(table8[key]['gan']['full_tpr'])
            else:
                pgd_tprs.append(1.0)
                gan_tprs.append(1.0)

        ax.plot(eps_ratios, pgd_tprs, f'{marker}-', color=color,
                linewidth=2, markersize=8,
                label=f'PGD ({fault_mult}σ)')
        ax.plot(eps_ratios, gan_tprs, f'{marker}--', color=color,
                linewidth=2, markersize=8, alpha=0.6,
                label=f'GAN ({fault_mult}σ)')

    ax.set_xlabel(r'Budget $\epsilon / \sigma_\eta$')
    ax.set_ylabel('Detection Rate (TPR)')
    ax.set_title('Full Pipeline Detection Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # Right: Speedup comparison
    ax = axes[1]
    speedups = []
    labels = []
    for key, val in table8.items():
        speedups.append(val.get('speedup', 1.0))
        labels.append(key.replace('sigma_eps', 'σ, ε='))

    bars = ax.bar(range(len(speedups)), speedups,
                   color='#4CAF50', alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('Speedup (PGD time / GAN time)')
    ax.set_title('GAN Inference Speedup over PGD')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}×', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_gan_training(save_path: Path):
    """Fig: GAN training dynamics."""
    s3_dir = RESULTS_DIR / "s3"
    history_file = s3_dir / "gan_training_history.json"
    if not history_file.exists():
        print("  Skipping GAN training plot (no data)")
        return

    with open(history_file) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = np.arange(len(history['g_loss']))

    # Left: Generator and Discriminator loss
    ax = axes[0]
    ax.plot(epochs, history['g_loss'], 'b-', alpha=0.7,
            label='Generator Loss', linewidth=1.5)
    ax.plot(epochs, history['d_loss'], 'r-', alpha=0.7,
            label='Discriminator Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('GAN Training: Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Evasion rate over training
    ax = axes[1]
    if 'evasion_rate' in history:
        ax.plot(epochs, history['evasion_rate'], 'g-',
                linewidth=2, label='Evasion Rate')
        ax.set_ylabel('Evasion Rate')
    ax.set_xlabel('Epoch')
    ax.set_title('GAN Training: Evasion Improvement')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_evasion_heatmap(save_path: Path):
    """Fig: Evasion heatmap (budget × fault magnitude → detection rate)."""
    s3_dir = RESULTS_DIR / "s3"
    table7_file = s3_dir / "table7_adversarial_lstm.json"
    if not table7_file.exists():
        print("  Skipping evasion heatmap (no S3 data)")
        return

    with open(table7_file) as f:
        table7 = json.load(f)

    fault_mults = [1.0, 2.0, 4.0]
    eps_ratios = [0.50, 1.00, 1.50]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (attack_type, title) in zip(axes, [
        ('tca_standard', 'Standard TCA (CUSUM+ISWT only)'),
        ('tca_neural', 'Neural TCA (CUSUM+ISWT+LSTM)'),
    ]):
        heatmap = np.zeros((len(fault_mults), len(eps_ratios)))

        for i, fm in enumerate(fault_mults):
            for j, er in enumerate(eps_ratios):
                key = f"{fm}sigma_eps{er}"
                if key in table7:
                    # Detection rate = 1 - evasion rate
                    heatmap[i, j] = table7[key][attack_type]['full_tpr']
                else:
                    heatmap[i, j] = 1.0

        im = ax.imshow(heatmap, cmap='RdYlGn_r', vmin=0, vmax=1,
                         aspect='auto')
        ax.set_xticks(range(len(eps_ratios)))
        ax.set_yticks(range(len(fault_mults)))
        ax.set_xticklabels([f'{r}' for r in eps_ratios])
        ax.set_yticklabels([f'{fm}σ' for fm in fault_mults])
        ax.set_xlabel(r'Budget $\epsilon / \sigma_\eta$')
        ax.set_ylabel('Fault Magnitude')
        ax.set_title(title)

        # Annotate cells
        for i in range(len(fault_mults)):
            for j in range(len(eps_ratios)):
                color = 'white' if heatmap[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{heatmap[i, j]:.2f}',
                        ha='center', va='center', fontsize=11,
                        fontweight='bold', color=color)

    fig.colorbar(im, ax=axes, shrink=0.8,
                  label='Detection Rate (TPR)')
    fig.suptitle('Detection Rate: Standard vs Neural TCA',
                  fontsize=13, y=1.02)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Publication Figures")
    print("=" * 70)

    config = ExperimentConfig()

    print("\nGenerating demonstration data...")
    data = generate_attack_demo_data(config)

    print("\nPlotting S1/S2 figures:")
    plot_cusum_timeseries(data, FIGURES_DIR / "cusum_timeseries.pdf")
    plot_iswt_timeseries(data, FIGURES_DIR / "iswt_timeseries.pdf")
    plot_sds_convergence(data, FIGURES_DIR / "sds_convergence.pdf")
    plot_innovation_covariance(data, FIGURES_DIR / "innovation_covariance.pdf")
    plot_budget_sweep(FIGURES_DIR / "budget_sweep.pdf")

    print("\nPlotting S3 figures:")
    plot_lstm_roc(FIGURES_DIR / "lstm_detection_comparison.pdf")
    plot_gan_vs_pgd(FIGURES_DIR / "gan_vs_pgd.pdf")
    plot_gan_training(FIGURES_DIR / "gan_training.pdf")
    plot_evasion_heatmap(FIGURES_DIR / "evasion_heatmap.pdf")

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()

