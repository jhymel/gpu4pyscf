"""Speed + peak-VRAM benchmark for the tensor-free fused COSX K.

Compares, for the exchange-K build at a genuine 1000+ BF size:
  * RI-JK (gpu4pyscf density-fit get_k) -- the incumbent
  * unscreened streamed COSX (sgx_jk.get_k)
  * tensor-free fused all-L COSX (sgx_kfused.get_k_fused_direct), tol sweep

Reports wall time (K build) and peak GPU-pool memory (high-water mark after a
clean reset + device sync). Uses an init-guess density (memory/time probe of the
K build; no SCF convergence needed).

Run:  python -m gpu4pyscf.sgx.bench.bench_fused
"""
import argparse
import time
import numpy as np
import cupy as cp
from pyscf import gto


def _alkane(n_carbon):
    atoms = []
    cc = 1.54
    ang = np.deg2rad(112.0)
    dx = cc * np.sin(ang / 2)
    x = 0.0
    z = 0.0
    for i in range(n_carbon):
        atoms.append(["C", [x, 0.0, z]])
        atoms.append(["H", [x, 0.63, z + 0.63]])
        atoms.append(["H", [x, -0.63, z + 0.63]])
        x += dx
        z = 0.0 if z != 0.0 else 0.5
    atoms.append(["H", [-0.5, 0.0, 0.5]])
    atoms.append(["H", [x + 0.2, 0.0, 0.5]])
    return atoms


def _reset():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _run(fn, *a, warmup=True, **k):
    """Return (wall_seconds, peak_pool_MB) or ('OOM','OOM')."""
    pool = cp.get_default_memory_pool()
    _reset()
    try:
        if warmup:
            fn(*a, **k)
            cp.cuda.Stream.null.synchronize()
            _reset()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        peak = pool.total_bytes() / 1e6
        del out
        _reset()
        return round(dt, 3), round(peak, 1)
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError):
        _reset()
        return "OOM", "OOM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbons", type=int, default=30)
    ap.add_argument("--basis", type=str, default="def2-tzvp")
    ap.add_argument("--grid-level", type=int, default=1)
    ap.add_argument("--tols", type=float, nargs="+", default=[0.0, 1e-6, 1e-4])
    ap.add_argument("--skip-unscreened", action="store_true",
                    help="skip the slow streamed unscreened COSX row")
    args = ap.parse_args()

    from gpu4pyscf.dft.rks import RKS
    from gpu4pyscf.dft import gen_grid as ggen
    from gpu4pyscf.sgx.sgx_jk import get_k
    from gpu4pyscf.sgx.sgx_kfused import get_k_fused_direct

    mol = gto.M(atom=_alkane(args.carbons), basis=args.basis,
                unit="Angstrom", verbose=0)
    print(f"system C{args.carbons}/{args.basis}  nao={mol.nao} nbas={mol.nbas}",
          flush=True)

    mf = RKS(mol, xc="pbe0").density_fit()
    dm = cp.asarray(mf.get_init_guess(mol))
    g = ggen.Grids(mol)
    g.level = args.grid_level
    g.build()
    print(f"grid level {args.grid_level}: ngrids={g.coords.shape[0]}", flush=True)

    blk = max(64, int(256e6 / 8 / mol.nao ** 2))

    print(f"\n{'method':>28} {'time_s':>9} {'peak_MB':>9}", flush=True)

    t, m = _run(lambda: mf.get_jk(mol, dm, with_j=False, with_k=True)[1])
    print(f"{'RI-JK':>28} {str(t):>9} {str(m):>9}", flush=True)

    if not args.skip_unscreened:
        t, m = _run(get_k, mol, dm, g, ovlp_fit=True, blksize=blk, warmup=False)
        print(f"{'COSX unscreened (streamed)':>28} {str(t):>9} {str(m):>9}", flush=True)

    for tol in args.tols:
        t, m = _run(get_k_fused_direct, mol, dm, g, tol=tol, ovlp_fit=True)
        print(f"{'fused tol=' + ('0' if tol == 0 else f'{tol:.0e}'):>28} "
              f"{str(t):>9} {str(m):>9}", flush=True)


if __name__ == "__main__":
    main()
