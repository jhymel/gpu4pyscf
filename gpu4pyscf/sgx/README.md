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

## Building from source (required for B2 kernel work)

B1 showed pure-Python screening cannot avoid building the dense
`(ngrids, nao, nao)` integral. Kernel-level screening (B2) requires a source
build with CUDA. Toolchain (conda-forge / pixi, CUDA 12.x):

```bash
pixi add cmake ninja "cuda-nvcc=12.*" gfortran \
         "cuda-cudart-dev=12.*" "cuda-nvrtc-dev=12.*" "cuda-cccl=12.*" libcublas-dev
```

Build `gpu4pyscf/lib` (add your GPU's arch; A10G = sm_86):

```bash
ENV=$(python -c "import sys,os;print(os.path.dirname(os.path.dirname(sys.executable)))")
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
cmake -B /tmp/g4build -S gpu4pyscf/lib -GNinja \
  -DCUDA_ARCHITECTURES='80-real;86-real' -DBUILD_LIBXC=OFF \
  -DCMAKE_CUDA_COMPILER=$ENV/bin/nvcc
cmake --build /tmp/g4build -j 8    # ~15-20 min, 138 targets; writes gpu4pyscf/lib/*.so
```

Then import the source tree (not the wheel) and test in-tree:

```bash
PYTHONPATH=$(pwd) python -m pytest gpu4pyscf/sgx/tests/ -q   # 24 passed
```

Verified: full build green on sm_86 (nvcc 12.9), all 24 sgx tests pass against
the from-source libraries.

## Testing during development (wheel-based env)

## Benchmarks

`bench/bench_scaling.py` (memory + time vs RI-JK across a size ladder) and
`bench/oom_probe.py` (find where RI-JK OOMs; confirm COSX survives).
