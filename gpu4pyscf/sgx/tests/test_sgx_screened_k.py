"""TDD spec for B1: screened COSX K build (grid-point + shell screening).

B1 turns B0's measured sparsity into actual work reduction: per grid block, it
(1) drops grid points whose total weighted density is negligible, and (2)
restricts the integral/contraction to AO shells that survive density screening
for that block. Assembly runs on the compacted arrays (pure cupy gather/scatter,
no new CUDA). Result must match the unscreened get_k within a tolerance-controlled
error, and reduce to the unscreened result as tol -> 0.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_screened_k.py -v
Requires a CUDA GPU + gpu4pyscf.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_jk import get_k
from gpu4pyscf.sgx.sgx_screened import get_k_screened


def _mol(basis="def2-svp"):
    return gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
        basis=basis,
        unit="Angstrom",
        verbose=0,
    )


def _dm(mol):
    from gpu4pyscf.dft.rks import RKS

    mf = RKS(mol, xc="hf").density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    return cp.asnumpy(cp.asarray(mf.make_rdm1()))


def _grids(mol, level=3):
    from gpu4pyscf.dft import gen_grid as ggen

    g = ggen.Grids(mol)
    g.level = level
    g.build()
    return g


class TestScreenedKCorrectness:
    def test_zero_tol_matches_unscreened(self):
        """tol <= 0 disables screening: must equal unscreened get_k exactly
        (same grid, same overlap fitting)."""
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k_scr = cp.asnumpy(cp.asarray(
            get_k_screened(mol, dm, g, tol=0.0, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k_scr, k_ref, atol=1e-9, rtol=0)

    def test_safe_tol_small_error(self):
        """At a safe tol the screened K matches unscreened K tightly."""
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k_scr = cp.asnumpy(cp.asarray(
            get_k_screened(mol, dm, g, tol=1e-8, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k_scr, k_ref, atol=1e-5, rtol=0)

    def test_symmetric(self):
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k = cp.asnumpy(cp.asarray(
            get_k_screened(mol, dm, g, tol=1e-6, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k.T, atol=1e-10, rtol=0)

    def test_error_grows_with_tol(self):
        """Looser tol => larger (or equal) deviation from unscreened K."""
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        errs = []
        for tol in (1e-8, 1e-5, 1e-3):
            k = cp.asnumpy(cp.asarray(
                get_k_screened(mol, dm, g, tol=tol, ovlp_fit=True)
            ))
            errs.append(np.abs(k - k_ref).max())
        assert all(errs[i] <= errs[i + 1] + 1e-12 for i in range(len(errs) - 1))


class TestScreenedKReportsWork:
    def test_returns_work_stats_when_requested(self):
        """With return_stats=True, also return how much work was skipped so the
        benchmark can quantify the reduction."""
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k, stats = get_k_screened(
            mol, dm, g, tol=1e-5, ovlp_fit=True, return_stats=True
        )
        assert "frac_gridpoints_skipped" in stats
        assert "frac_shell_work_skipped" in stats
        assert 0.0 <= stats["frac_gridpoints_skipped"] <= 1.0
        assert 0.0 <= stats["frac_shell_work_skipped"] <= 1.0
