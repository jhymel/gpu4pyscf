"""Targeted OOM probe: find where RI-JK OOMs on this GPU, and confirm COSX
survives the same case (bounded, streamed memory).

Run (from repo root):  python -m gpu4pyscf.sgx.bench.oom_probe
"""
import sys
import time

import numpy as np
import cupy as cp
from pyscf import gto

sys.path.insert(0, ".")
from gpu4pyscf.sgx.bench.bench_scaling import _linear_alkane, _peak_mem_mb, _reset_mem
from gpu4pyscf.sgx.sgx_scf import cosx_fit


def probe(n_carbon, basis="def2-tzvp", xc="pbe0", grid_level=1, do_cosx=True):
    from gpu4pyscf.dft.rks import RKS

    mol = gto.M(atom=_linear_alkane(n_carbon), basis=basis, unit="Angstrom", verbose=0)
    nao = mol.nao
    print(f"nC={n_carbon} nao={nao} basis={basis}", flush=True)

    # RI-JK K build
    _reset_mem()
    try:
        mf = RKS(mol, xc=xc).density_fit()
        dm = mf.get_init_guess(mol)
        cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        _ = mf.get_jk(mol, dm, with_j=False, with_k=True)[1]
        cp.cuda.Stream.null.synchronize()
        _, tot = _peak_mem_mb()
        print(f"  RI-JK : OK  {time.perf_counter()-t0:6.2f}s  peak={tot:8.1f} MB", flush=True)
        rijk_ok = True
    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as e:
        print(f"  RI-JK : OOM ({type(e).__name__})", flush=True)
        rijk_ok = False
        _reset_mem()

    if do_cosx:
        _reset_mem()
        try:
            mf2 = cosx_fit(RKS(mol, xc=xc).density_fit(), grid_level=grid_level)
            dm2 = mf2.get_init_guess(mol)
            cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
            _ = mf2.get_jk(mol, dm2, with_j=False, with_k=True)[1]
            cp.cuda.Stream.null.synchronize()
            _, tot2 = _peak_mem_mb()
            print(f"  COSX  : OK  {time.perf_counter()-t0:6.2f}s  peak={tot2:8.1f} MB", flush=True)
        except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as e:
            print(f"  COSX  : OOM ({type(e).__name__})", flush=True)
            _reset_mem()

    return rijk_ok


if __name__ == "__main__":
    # Escalate RI-JK size until it OOMs (RI-JK is fast, so this is quick).
    # Skip the slow COSX build except at the crossover point of interest.
    for nc in [24, 32, 40, 48, 56]:
        rijk_ok = probe(nc, do_cosx=False)
        if not rijk_ok:
            print(f">>> RI-JK OOMs at nC={nc}; testing COSX survival here.", flush=True)
            probe(nc, do_cosx=True)
            break
