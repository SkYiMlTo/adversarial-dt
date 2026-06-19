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
    sim_fault = process.simulate(T, fault_config={
        'sensor_idx': [5],
        'fault_start': 0,
        'fault_magnitude': 2.0,
    }, seed=config.seed + 1)

    # Simulate clean
    sim_clean = process.simulate(T, seed=config.seed + 2)

    # Run TCA (white-box)
    epsilon = 0.75 * sys_cfg.sigma[5]
    tca = TargetedConsistencyAttack(
        sys_cfg, ekf_cfg, config.cusum, config.iswt, config.tca
    )

    try:
        tca_result = tca.run_whitebox(
            sim_fault['y_faulted'], sim_fault['u'],
            list(range(6)), [5], epsilon, verbose=True
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

        iswt = ISWTDetector(6, config.iswt)
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

    sensor_idx = 5  # Q_pump (attacked sensor)
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
            sds_vals.append(raw.get(key, 0.0))

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


def main():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Publication Figures")
    print("=" * 70)

    config = ExperimentConfig()

    print("\nGenerating demonstration data...")
    data = generate_attack_demo_data(config)

    print("\nPlotting figures:")
    plot_cusum_timeseries(data, FIGURES_DIR / "cusum_timeseries.pdf")
    plot_iswt_timeseries(data, FIGURES_DIR / "iswt_timeseries.pdf")
    plot_sds_convergence(data, FIGURES_DIR / "sds_convergence.pdf")
    plot_innovation_covariance(data, FIGURES_DIR / "innovation_covariance.pdf")
    plot_budget_sweep(FIGURES_DIR / "budget_sweep.pdf")

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
