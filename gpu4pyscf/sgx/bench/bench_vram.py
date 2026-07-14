"""Peak-VRAM comparison: unscreened COSX (varying blksize) vs RI-JK K build.

Answers: how low can COSX's peak GPU memory go via grid-block streaming
(blksize), and how does that compare to RI-JK's peak at a genuine 1000+ BF
size (the regime where RI-JK OOMs). Screening is OFF (unscreened COSX) -- the
VRAM lever measured here is grid-block streaming, which needs no screening.

Peak is measured as the CuPy default memory pool high-water mark
(total_bytes) after a clean free_all_blocks() reset, with a device sync so all
kernel allocations are captured. Each config runs a single K build on an
init-guess density (a memory probe; no SCF convergence needed).

Run:  python -m gpu4pyscf.sgx.bench.bench_vram
"""
import argparse
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


def _reset_pool():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _peak_mb(fn, *a, **k):
    """Peak default-pool bytes (MB) reached during fn, or 'OOM'."""
    pool = cp.get_default_memory_pool()
    _reset_pool()
    try:
        out = fn(*a, **k)
        cp.cuda.Stream.null.synchronize()
        peak = pool.total_bytes() / 1e6
        del out
        _reset_pool()
        return round(peak, 1)
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError):
        _reset_pool()
        return "OOM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbons", type=int, default=40)
    ap.add_argument("--basis", type=str, default="def2-tzvp")
    ap.add_argument("--grid-level", type=int, default=1)
    ap.add_argument("--blksizes", type=int, nargs="+", default=[4096, 1024, 256, 64])
    args = ap.parse_args()

    from gpu4pyscf.dft.rks import RKS
    from gpu4pyscf.dft import gen_grid as ggen
    from gpu4pyscf.sgx.sgx_jk import get_k

    mol = gto.M(atom=_alkane(args.carbons), basis=args.basis,
                unit="Angstrom", verbose=0)
    print(f"system: C{args.carbons}/{args.basis}  nao={mol.nao} nbas={mol.nbas}",
          flush=True)

    mf = RKS(mol, xc="pbe0").density_fit()
    dm = cp.asarray(mf.get_init_guess(mol))
    g = ggen.Grids(mol)
    g.level = args.grid_level
    g.build()
    print(f"grid level {args.grid_level}: ngrids={g.coords.shape[0]}", flush=True)

    # RI-JK K build (the incumbent; may OOM at this size)
    def rijk():
        return mf.get_jk(mol, dm, with_j=False, with_k=True)[1]

    print(f"\n{'method':>22} {'peak_MB':>10}", flush=True)
    print(f"{'RI-JK (get_k)':>22} {str(_peak_mb(rijk)):>10}", flush=True)
    for blk in args.blksizes:
        p = _peak_mb(get_k, mol, dm, g, ovlp_fit=True, blksize=blk)
        print(f"{'COSX blksize=' + str(blk):>22} {str(p):>10}", flush=True)


if __name__ == "__main__":
    main()
