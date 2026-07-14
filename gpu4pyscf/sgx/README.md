# COSX / Seminumerical Exchange (`gpu4pyscf.sgx`) — development notes

GPU seminumerical (chain-of-spheres) exchange for GPU4PySCF: RI-J for Coulomb
paired with a grid-based exact-exchange (K) build (RIJCOSX). Layout mirrors
`pyscf/sgx/`.

## Status

Implemented and validated (see `tests/`):

- `sgx_jk.get_k` — COSX exchange matrix (K only), overlap-fitted, grid-batched.
- `sgx_scf.cosx_fit` — wrap a density-fitted `RKS` hybrid so exchange is COSX
  (RI-J kept for Coulomb). Full-range hybrids only (omega == 0) for now.
- `sgx_grad.get_k_energy_grad` — analytic nuclear gradient of the COSX exchange
  energy (orbital + integral response via `int3c1e_ip1`).
- QM/MM point-charge embedding composes cleanly (`mm_charge(cosx_fit(...))`).

**Not yet implemented:** P-junction (density-matrix) screening. Without it the
build is memory-favorable (streams grid batches; survives def2-TZVP cases that
OOM RI-JK) but **slower** than RI-JK. Screening is the next major piece.

## Validation summary

- K vs analytical DF-K: converges to the seminumerical floor with grid level.
- K vs CPU PySCF SGX: ~1e-8.
- RIJCOSX SCF energy vs CPU SGX SCF: ~1e-6 Eh.
- Analytic gradient vs finite difference: ~1e-6 (error -> 0 as grid refines).

## Testing during development (wheel-based env)

The compiled primitives (`int1e_grids`, etc.) ship in the installed
`gpu4pyscf-cuda12x` wheel, while this source tree is checked out at the matching
tag (v1.7.1). To test the pure-Python `sgx` package against the installed
compiled libs, copy `gpu4pyscf/sgx/` into the installed package and run pytest
with an external rootdir (avoids the source-tree lib loader):

```bash
SP=$(python -c "import gpu4pyscf, os; print(os.path.dirname(gpu4pyscf.__file__))")
cp -r gpu4pyscf/sgx "$SP"/
pytest --rootdir=/tmp "$SP"/sgx/tests/ -v
```

Once building GPU4PySCF from source (with compiled extensions), the tests run
in-tree normally: `pytest gpu4pyscf/sgx/tests/ -v`.

## Benchmarks

`bench/bench_scaling.py` (memory + time vs RI-JK across a size ladder) and
`bench/oom_probe.py` (find where RI-JK OOMs; confirm COSX survives).
