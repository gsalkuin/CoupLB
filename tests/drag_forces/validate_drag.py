#!/usr/bin/env python3
"""
validate_drag.py — Validate CoupLB point-force drag test.

Checks:
  1. Terminal velocity reached (v_x converges to constant)
  2. No spurious transverse motion (v_y, v_z ≈ 0)
  3. Force balance: steady v_x implies IBM drag = applied force

Usage:
  python validate_drag.py log.lammps
"""
import sys
import re
import numpy as np


def parse_thermo(filename):
    """Extract thermo data (step, vx, vy, vz, vmag) from LAMMPS log."""
    data = []
    in_thermo = False
    with open(filename) as fh:
        for line in fh:
            s = line.strip()
            # Detect thermo header
            if s.startswith("Step") and "v_vx" in s:
                in_thermo = True
                continue
            if in_thermo:
                # End of thermo block
                if s.startswith("Loop") or s.startswith("print") or s.startswith("==="):
                    in_thermo = False
                    continue
                parts = s.split()
                if len(parts) >= 5:
                    try:
                        row = [float(x) for x in parts[:5]]
                        data.append(row)
                    except ValueError:
                        in_thermo = False
                        continue
    if not data:
        print(f"ERROR: no thermo data found in {filename}")
        sys.exit(1)
    return np.array(data)


def validate(filename):
    data = parse_thermo(filename)
    steps = data[:, 0]
    vx = data[:, 1]
    vy = data[:, 2]
    vz = data[:, 3]

    print(f"\n{'='*60}")
    print(f"  POINT-FORCE DRAG VALIDATION")
    print(f"{'='*60}")
    print(f"  Total steps: {int(steps[-1])}")
    print(f"  Data points: {len(data)}")

    # ---- Check 1: Terminal velocity convergence ----
    # Use last 20% of data to check if v_x is steady
    n_tail = max(len(vx) // 5, 3)
    vx_tail = vx[-n_tail:]
    vx_mean = np.mean(vx_tail)
    vx_std = np.std(vx_tail)

    # Relative fluctuation: should be < 1%
    rel_fluct = vx_std / abs(vx_mean) * 100 if abs(vx_mean) > 1e-15 else float('inf')

    print(f"\n  CHECK 1: Terminal velocity convergence")
    print(f"    v_x (last {n_tail} points): mean = {vx_mean:.6e}, std = {vx_std:.2e}")
    print(f"    Relative fluctuation: {rel_fluct:.4f}%")
    ok1 = rel_fluct < 1.0
    print(f"    -> {'PASS' if ok1 else 'FAIL'} (threshold: < 1%)")

    # ---- Check 2: No transverse motion ----
    vy_max = np.max(np.abs(vy))
    vz_max = np.max(np.abs(vz))
    transverse_ratio = max(vy_max, vz_max) / abs(vx_mean) * 100 if abs(vx_mean) > 1e-15 else float('inf')

    print(f"\n  CHECK 2: Spurious transverse motion")
    print(f"    max|v_y| = {vy_max:.2e}")
    print(f"    max|v_z| = {vz_max:.2e}")
    print(f"    Transverse / v_x = {transverse_ratio:.4f}%")
    ok2 = transverse_ratio < 1.0
    print(f"    -> {'PASS' if ok2 else 'FAIL'} (threshold: < 1% of v_x)")

    # ---- Check 3: Positive and physical ----
    print(f"\n  CHECK 3: Physical behavior")
    print(f"    v_x > 0: {vx_mean > 0}")
    print(f"    v_x monotonically increasing early: {np.all(np.diff(vx[:5]) > 0) if len(vx) > 5 else 'N/A'}")
    ok3 = vx_mean > 0
    print(f"    -> {'PASS' if ok3 else 'FAIL'}")

    # ---- Summary ----
    all_pass = ok1 and ok2 and ok3
    print(f"\n  {'='*60}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Terminal v_x = {vx_mean:.6e} LJ velocity units")
    print(f"  {'='*60}\n")

    # ---- Plot if matplotlib available ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(steps, vx, 'b-', lw=1)
        axes[0].axhline(vx_mean, color='r', ls='--', label=f'mean={vx_mean:.4e}')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('$v_x$')
        axes[0].set_title('Terminal velocity convergence')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(steps, vy, 'g-', lw=1, label='$v_y$')
        axes[1].plot(steps, vz, 'orange', lw=1, label='$v_z$')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Transverse velocity')
        axes[1].set_title('Spurious transverse motion')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(steps, np.sqrt(vx**2 + vy**2 + vz**2), 'k-', lw=1)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('$|v|$')
        axes[2].set_title('Speed magnitude')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('drag_validation.png', dpi=150)
        print("  Plot -> drag_validation.png\n")
    except ImportError:
        pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_drag.py log.lammps")
        sys.exit(1)
    validate(sys.argv[1])
