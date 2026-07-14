"""TDD spec for per-grid-point (in-kernel) screening in the fused COSX K.

The fused kernel (GINTsgx_fused_k_gv) evaluates every kept shell-pair type at
every grid point. Per-grid-point screening adds a cheap early-exit inside the
kernel: for each (primitive shell-pair, grid point), if the integral-magnitude
bound (from the already-loaded pair prefactor eij/aij and |P-C|^2) is below the
screening tolerance, skip the expensive GINT_g1e recurrence + contraction. This
is where COSX's real speedup lives (validated: skippable fraction grows with
size, 8% at nao=106 -> 66% at nao=490 even at a safe 1e-10 bound).

Correctness must be preserved: screened result matches unscreened get_k within a
tolerance-controlled error, and tol<=0 is exact.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_kfused_screen.py -v
Requires a CUDA GPU + gpu4pyscf source build.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_jk import get_k
from gpu4pyscf.sgx.sgx_kfused import get_k_fused_direct


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


def _water(basis):
    return gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
                 basis=basis, unit="Angstrom", verbose=0)


class TestKernelScreening:
    def test_screened_matches_unscreened_svp(self):
        """Kernel screening at a safe tol preserves K vs unscreened get_k.

        Uses screen_tol=1e-10: with the density-matrix-weighted screen this
        skips substantial work while K error (~1e-7) stays well under atol.
        (At 1e-9 the DM screen skips more and error rises to ~2e-6, still
        chemically negligible but above this guard's budget — see the fused
        speed benchmark / investigation doc sec 9 for the tol/accuracy curve.)
        """
        mol = _water("def2-svp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_direct(mol, dm, g, screen_tol=1e-10, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-6, rtol=0)

    def test_screened_matches_unscreened_tzvp(self):
        """Up to d functions, kernel screening (screen_tol=1e-10) preserves K."""
        mol = _water("def2-tzvp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_direct(mol, dm, g, screen_tol=1e-10, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-5, rtol=0)

    def test_screen_tol_zero_is_exact(self):
        """screen_tol<=0 disables kernel screening -> exact vs get_k."""
        mol = _water("def2-svp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_direct(mol, dm, g, screen_tol=0.0, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-7, rtol=0)

    def test_error_grows_with_screen_tol(self):
        mol = _water("def2-svp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        errs = []
        for tol in (1e-10, 1e-6, 1e-3):
            k = cp.asnumpy(cp.asarray(
                get_k_fused_direct(mol, dm, g, screen_tol=tol, ovlp_fit=True)
            ))
            errs.append(np.abs(k - k_ref).max())
        assert all(errs[i] <= errs[i + 1] + 1e-12 for i in range(len(errs) - 1))
