"""Seminumerical exchange (COSX / SGX) for GPU4PySCF.

Chain-of-spheres / pseudo-spectral exchange on the GPU: RI-J for Coulomb paired
with a grid-based seminumerical exact-exchange (K) build. Layout mirrors
``pyscf/sgx/``.

Public API:
    cosx_fit(mf, grid_level=..., blksize=...)  -- wrap a density-fitted RKS
        hybrid so exchange K is computed by COSX (RIJCOSX).
    sgx_jk.get_k(mol, dm, grids, ...)          -- the COSX exchange matrix.
    sgx_grad.get_k_energy_grad(...)            -- analytic nuclear gradient of E_K.

Prototype status (see the K-only / SCF / QMMM / gradient tests under tests/):
K build, RIJCOSX SCF, QM/MM point-charge embedding, and analytic gradients are
implemented and validated. P-junction (density-matrix) screening is not yet
implemented, so the current build is memory-favorable but not speed-competitive
with RI-JK.
"""
from gpu4pyscf.sgx.sgx_scf import cosx_fit

__all__ = ["cosx_fit"]
