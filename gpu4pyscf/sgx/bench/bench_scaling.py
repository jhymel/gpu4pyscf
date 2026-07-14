"""Scaling / memory benchmark: COSX (RI-J + seminumerical K) vs RI-JK.

Measures peak GPU memory and wall time for the exchange build across a ladder
of growing QM regions and basis sets. The motivating question is NOT raw speed
(our prototype COSX has no P-junction screening yet, so it is expected to be
slower per SCF than RI-JK) but MEMORY: whether COSX survives where RI-JK OOMs,
because COSX streams grid batches instead of materializing large RI exchange
tensors.

Run (from repo root):  python -m gpu4pyscf.sgx.bench.bench_scaling
"""
import argparse
import time

import numpy as np
import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_scf import cosx_fit


def _linear_alkane(n_carbon):
    """Build an all-trans alkane C_nH_{2n+2} as a size ladder (Angstrom)."""
    atoms = []
    cc = 1.54
    ang = np.deg2rad(112.0)
    dx = cc * np.sin(ang / 2)
    dz = cc * np.cos(ang / 2)
    x = 0.0
    z = 0.0
    for i in range(n_carbon):
        atoms.append(["C", [x, 0.0, z]])
        # crude H placement; geometry need not be optimal for a timing/memory probe
        atoms.append(["H", [x, 0.63, z + 0.63 if i % 2 == 0 else z - 0.63]])
        atoms.append(["H", [x, -0.63, z + 0.63 if i % 2 == 0 else z - 0.63]])
        x += dx
        z = 0.0 if z != 0.0 else 0.5
    # cap ends with extra H
    atoms.append(["H", [-0.5, 0.0, 0.5]])
    atoms.append(["H", [x + 0.2, 0.0, 0.5]])
    return atoms


def _peak_mem_mb():
    mempool = cp.get_default_memory_pool()
    return mempool.used_bytes() / 1e6, mempool.total_bytes() / 1e6


def _reset_mem():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _time_k_build(mf, dm, n=1):
    """Time a single K build (get_k or get_jk with_k=True). Returns (sec, K)."""
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        try:
            vk = mf.get_k(mf.mol, dm)
        except TypeError:
            vk = mf.get_jk(mf.mol, dm, with_j=False, with_k=True)[1]
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) / n, vk


def run_one(n_carbon, basis, xc="pbe0", grid_level=3):
    atoms = _linear_alkane(n_carbon)
    mol = gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)
    nao = mol.nao
    row = {"nC": n_carbon, "basis": basis, "nao": nao}

    # crude density for a K build (no need to converge for a memory/time probe)
    from gpu4pyscf.dft.rks import RKS

    dm = None

    # --- RI-JK ---
    _reset_mem()
    try:
        mf = RKS(mol, xc=xc).density_fit()
        dm = mf.get_init_guess(mol)
        t, _ = _time_k_build(mf, dm)
        used, total = _peak_mem_mb()
        row["rijk_sec"] = round(t, 4)
        row["rijk_mem_mb"] = round(total, 1)
    except (cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError):
        row["rijk_sec"] = None
        row["rijk_mem_mb"] = "OOM"
        _reset_mem()

    # --- COSX ---
    _reset_mem()
    try:
        mf2 = cosx_fit(RKS(mol, xc=xc).density_fit(), grid_level=grid_level)
        if dm is None:
            dm = mf2.get_init_guess(mol)
        t2, _ = _time_k_build(mf2, dm)
        used2, total2 = _peak_mem_mb()
        row["cosx_sec"] = round(t2, 4)
        row["cosx_mem_mb"] = round(total2, 1)
    except (cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError):
        row["cosx_sec"] = None
        row["cosx_mem_mb"] = "OOM"
        _reset_mem()

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbons", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    ap.add_argument("--bases", type=str, nargs="+", default=["def2-svp", "def2-tzvp"])
    ap.add_argument("--grid-level", type=int, default=3)
    args = ap.parse_args()

    print(f"{'nC':>3} {'basis':>10} {'nao':>5} "
          f"{'RIJK_s':>8} {'RIJK_MB':>9} {'COSX_s':>8} {'COSX_MB':>9}",
          flush=True)
    for basis in args.bases:
        for nc in args.carbons:
            r = run_one(nc, basis, grid_level=args.grid_level)
            print(f"{r['nC']:>3} {r['basis']:>10} {r['nao']:>5} "
                  f"{str(r.get('rijk_sec')):>8} {str(r.get('rijk_mem_mb')):>9} "
                  f"{str(r.get('cosx_sec')):>8} {str(r.get('cosx_mem_mb')):>9}",
                  flush=True)


if __name__ == "__main__":
    main()
