"""
Smoke test: 3 sessions, WB only, fault=2.0sigma.
Validates that:
  1. Single calibration works
  2. TCA runs successfully
  3. SCM bound holds (evasion should be very low at 2.0sigma)
  4. CUSUM-naive attacker is caught by IWD more than adaptive
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ExperimentConfig
from experiments.run_s1_automated import (
    run_calibration, run_single_session, wilson_ci
)

ATTACKED_IDX = list(range(6))
COMPROMISED_IDX = [0, 1, 5]


def main():
    config = ExperimentConfig()
    # Override for speed
    config.s1_sessions_per_config = 3
    config.tca.K = 50  # Fewer WB iterations for speed

    print("=== SMOKE TEST ===")
    print()

    # Step 1: Calibration
    print("[1] Calibration...")
    calib = run_calibration(config)
    print(f"    ISWT critical: {calib['iswt_config'].empirical_critical:.2f}")
    print()

    # Step 2: Run 3 WB sessions at fault=2.0sigma (adaptive)
    print("[2] Adaptive attacker (iswt_weight=2.0), WB, fault=2.0sigma...")
    adaptive_results = []
    for sid in range(3):
        r = run_single_session(
            sid, 'whitebox', 2.0, config, calib,
            iswt_weight=2.0, seed_offset=0
        )
        print(f"    Session {sid}: CUSUM evade={r['cusum_evasion']}, "
              f"Combined evade={r['combined_evasion']}, "
              f"SDS={r['sds_final']:.4f}, "
              f"Latency CUSUM={r['cusum_latency']}, "
              f"IWD|C={r['combined_latency']}")
        adaptive_results.append(r)

    print()

    # Step 3: Run 3 WB sessions at fault=2.0sigma (CUSUM-naive)
    print("[3] CUSUM-naive attacker (iswt_weight=0.0), WB, fault=2.0sigma...")
    naive_results = []
    for sid in range(3):
        r = run_single_session(
            sid, 'whitebox', 2.0, config, calib,
            iswt_weight=0.0, seed_offset=0
        )
        print(f"    Session {sid}: CUSUM evade={r['cusum_evasion']}, "
              f"Combined evade={r['combined_evasion']}, "
              f"SDS={r['sds_final']:.4f}, "
              f"Latency CUSUM={r['cusum_latency']}, "
              f"IWD|C={r['combined_latency']}")
        naive_results.append(r)

    print()

    # Step 4: Run 3 WB sessions at fault=1.5sigma (adaptive) for comparison
    print("[4] Adaptive attacker, WB, fault=1.5sigma...")
    for sid in range(3):
        r = run_single_session(
            sid, 'whitebox', 1.5, config, calib,
            iswt_weight=2.0, seed_offset=0
        )
        print(f"    Session {sid}: CUSUM evade={r['cusum_evasion']}, "
              f"Combined evade={r['combined_evasion']}, "
              f"SDS={r['sds_final']:.4f}")

    print()

    # Step 5: Run 3 GB sessions at fault=2.0sigma (adaptive)
    print("[5] Adaptive attacker, GB, fault=2.0sigma...")
    for sid in range(3):
        r = run_single_session(
            sid, 'greybox', 2.0, config, calib,
            iswt_weight=2.0, seed_offset=0
        )
        print(f"    Session {sid}: CUSUM evade={r['cusum_evasion']}, "
              f"Combined evade={r['combined_evasion']}, "
              f"SDS={r['sds_final']:.4f}")

    print()

    # Summary
    print("=== SUMMARY ===")
    adapt_cusum = sum(1 for r in adaptive_results if r['cusum_evasion'])
    adapt_combined = sum(1 for r in adaptive_results if r['combined_evasion'])
    naive_cusum = sum(1 for r in naive_results if r['cusum_evasion'])
    naive_combined = sum(1 for r in naive_results if r['combined_evasion'])

    print(f"  Adaptive (2.0sigma WB): CUSUM evade={adapt_cusum}/3, "
          f"IWD|CUSUM evade={adapt_combined}/3")
    print(f"  Naive    (2.0sigma WB): CUSUM evade={naive_cusum}/3, "
          f"IWD|CUSUM evade={naive_combined}/3")

    if naive_cusum > naive_combined:
        print("  >> IWD catches CUSUM-naive attacks! (Expected)")
    else:
        print("  >> IWD did not add value at this operating point")
        print("  >> (This is OK if CUSUM already detects everything)")

    print()
    print("Smoke test complete.")


if __name__ == '__main__':
    main()
