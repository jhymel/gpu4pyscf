"""TDD spec for B2.2a: block-skipping screened raw-integral COSX K driver.

This is the "prove the payoff" slice of B2.2. Instead of building the full
(ngrids, nao, nao) integral then masking (B1), this driver mirrors
get_int3c1e's per-shell-pair-TYPE block loop (over intopt.log_qs) and SKIPS
whole cp_ij_id blocks whose density-weighted Schwarz bound is below tol,
contracting each surviving block directly into K. Skipped blocks are never
computed -> real compute reduction (unlike B1). Pure Python/cupy driving the
existing CUDA integral kernel (no kernel edits).

Goal: does block-level skipping produce actual speedup vs unscreened at scale?
Correctness anchor: tol<=0 must equal the unscreened get_k exactly.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_kdirect.py -v
Requires a CUDA GPU + gpu4pyscf (source build).
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_jk import get_k
from gpu4pyscf.sgx.sgx_kdirect import get_k_blockscreened


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


class TestBlockScreenedK:
    def test_zero_tol_matches_unscreened(self):
        """tol <= 0 disables block skipping -> must equal unscreened get_k."""
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_blockscreened(mol, dm, g, tol=0.0, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-9, rtol=0)

    def test_safe_tol_small_error(self):
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_blockscreened(mol, dm, g, tol=1e-8, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-5, rtol=0)

    def test_symmetric(self):
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k = cp.asnumpy(cp.asarray(
            get_k_blockscreened(mol, dm, g, tol=1e-6, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k.T, atol=1e-10, rtol=0)

    def test_error_grows_with_tol(self):
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=3)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        errs = []
        for tol in (1e-8, 1e-5, 1e-3):
            k = cp.asnumpy(cp.asarray(
                get_k_blockscreened(mol, dm, g, tol=tol, ovlp_fit=True)
            ))
            errs.append(np.abs(k - k_ref).max())
        assert all(errs[i] <= errs[i + 1] + 1e-12 for i in range(len(errs) - 1))

    def test_reports_blocks_skipped(self):
        mol = _mol()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k, stats = get_k_blockscreened(
            mol, dm, g, tol=1e-3, ovlp_fit=True, return_stats=True
        )
        assert "frac_blocks_skipped" in stats
        assert 0.0 <= stats["frac_blocks_skipped"] <= 1.0
