#!/usr/bin/env python3
"""
validate.py -- Compare CoupLB output against analytical solutions.

Usage:
  python validate.py poiseuille couplb_poiseuille.dat
  python validate.py couette    couplb_couette.dat
"""
import sys
import numpy as np

try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


def read_last_block(filename):
    steps = {}; current = None
    with open(filename) as fh:
        for line in fh:
            s = line.strip()
            if not s: continue
            if s.startswith('# step ='):
                current = int(s.split('=')[1]); steps[current] = []; continue
            if s.startswith('#') or current is None: continue
            parts = s.split()
            if len(parts) >= 6: steps[current].append([float(x) for x in parts])
    if not steps: print(f"ERROR: no data in {filename}"); sys.exit(1)
    last = max(steps.keys()); data = np.array(steps[last])
    print(f"Read step {last}, {len(data)} nodes"); return last, data


def validate_poiseuille(fname):
    Ny, nu, F = 32, 0.1, 1e-5
    step, data = read_last_block(fname)
    y, ux = data[:,2], data[:,4]
    ux_exact = F/(2*nu) * (y+0.5) * (Ny-0.5-y)
    err = np.abs(ux - ux_exact); ux_max = np.max(ux_exact)
    rel = np.max(err)/ux_max*100 if ux_max > 0 else float('inf')

    print(f"\n{'='*55}\n  POISEUILLE (step {step})\n{'='*55}")
    print(f"  u_max exact={ux_max:.6e}  LBM={np.max(ux):.6e}")
    print(f"  max|err|={np.max(err):.6e}  rel={rel:.4f}%  L2={np.sqrt(np.mean(err**2)):.6e}")
    print(f"  -> {'PASS' if rel<1 else 'MARGINAL' if rel<5 else 'FAIL'}\n{'='*55}\n")

    if HAS_PLT:
        fig,(a1,a2) = plt.subplots(1,2,figsize=(12,5))
        a1.plot(ux_exact,y,'b-',lw=2,label='Exact'); a1.plot(ux,y,'ro',ms=4,label='CoupLB')
        a1.set_xlabel('$u_x$'); a1.set_ylabel('$y$'); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(y,err,'k.-'); a2.set_xlabel('$y$'); a2.set_ylabel('|error|'); a2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig('poiseuille_validation.png',dpi=150)
        print("  Plot -> poiseuille_validation.png\n")


def validate_couette(fname):
    Ny, Uw = 32, 0.05
    step, data = read_last_block(fname)
    y, ux = data[:,2], data[:,4]
    ux_exact = Uw*(y+0.5)/Ny
    err = np.abs(ux-ux_exact); rel = np.max(err)/Uw*100

    print(f"\n{'='*55}\n  COUETTE (step {step})\n{'='*55}")
    print(f"  u_max exact={np.max(ux_exact):.6e}  LBM={np.max(ux):.6e}")
    print(f"  max|err|={np.max(err):.6e}  rel={rel:.4f}%  L2={np.sqrt(np.mean(err**2)):.6e}")
    print(f"  -> {'PASS' if rel<1 else 'MARGINAL' if rel<5 else 'FAIL'}\n{'='*55}\n")

    if HAS_PLT:
        fig,(a1,a2) = plt.subplots(1,2,figsize=(12,5))
        a1.plot(ux_exact,y,'b-',lw=2,label='Exact'); a1.plot(ux,y,'ro',ms=4,label='CoupLB')
        a1.set_xlabel('$u_x$'); a1.set_ylabel('$y$'); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(y,err,'k.-'); a2.set_xlabel('$y$'); a2.set_ylabel('|error|'); a2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig('couette_validation.png',dpi=150)
        print("  Plot -> couette_validation.png\n")


if __name__ == '__main__':
    if len(sys.argv) < 3: print("Usage: python validate.py {poiseuille|couette} <file>"); sys.exit(1)
    t, fn = sys.argv[1].lower(), sys.argv[2]
    if t == 'poiseuille': validate_poiseuille(fn)
    elif t == 'couette': validate_couette(fn)
    else: print(f"Unknown: {t}")
