"""TDD spec for B2.3: fused, streamed COSX K-direct CUDA kernel (l=0 first).

B2.3 computes K WITHOUT ever materializing the (ngrids, nao, nao) integral
tensor: per grid point, each shell-pair's analytic 3c-1e integral is computed
INLINE (Gaussian product + Boys(0) for s-shells) and contracted directly into
the K matrix, with density-based shell-pair screening. This is the only path
to both speed and low VRAM at 1000+ BF (B1/B2.2 could not avoid the dense
tensor; see COSX_B2_KERNEL_PLAN.md).

This first slice validates the l=0 (s-function) case end to end on an all-s
system, against the validated unscreened get_k (ovlp_fit path matched).

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_kfused.py -v
Requires a CUDA GPU + gpu4pyscf source build.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_jk import get_k
from gpu4pyscf.sgx.sgx_kfused import get_k_fused


def _mol_all_s():
    # sto-3g on H is a single s-shell per atom -> all-s, nao == natm.
    # 6 H atoms (even electron count) in a spread-out geometry.
    return gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.5; H 0.6 0.3 0.9; H 1.2 0.0 0.3; H 0.3 1.1 0.6",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )


def _dm(mol, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.rand(mol.nao, mol.nao)
    return a + a.T


def _grids(mol, level=1):
    from gpu4pyscf.dft import gen_grid as ggen

    g = ggen.Grids(mol)
    g.level = level
    g.build()
    return g


class TestFusedKSOnly:
    def test_all_s_system(self):
        mol = _mol_all_s()
        assert all(mol.bas_angular(i) == 0 for i in range(mol.nbas))

    def test_fused_matches_unscreened_no_screen(self):
        """Fused K-direct with screening OFF (tol<=0) must match the validated
        unscreened get_k on an all-s system."""
        mol = _mol_all_s()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(get_k_fused(mol, dm, g, tol=0.0, ovlp_fit=True)))
        np.testing.assert_allclose(k, k_ref, atol=1e-8, rtol=0)

    def test_fused_symmetric(self):
        mol = _mol_all_s()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k = cp.asnumpy(cp.asarray(get_k_fused(mol, dm, g, tol=0.0, ovlp_fit=True)))
        np.testing.assert_allclose(k, k.T, atol=1e-9, rtol=0)

    def test_fused_screened_close(self):
        """With safe-tol screening on, fused K stays close to unscreened."""
        mol = _mol_all_s()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(get_k_fused(mol, dm, g, tol=1e-8, ovlp_fit=True)))
        np.testing.assert_allclose(k, k_ref, atol=1e-4, rtol=0)

    def test_fused_no_dense_tensor_flag(self):
        """The fused path must report it never built the (ngrids,nao,nao) tensor
        (distinguishes it from the B1/B2.2 approach)."""
        mol = _mol_all_s()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k, stats = get_k_fused(mol, dm, g, tol=0.0, ovlp_fit=True, return_stats=True)
        assert stats.get("fused") is True
        assert "peak_intermediate_bytes" in stats
