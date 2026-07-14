"""TDD spec for B2.3 generalization: fused COSX K-direct for all angular momenta.

The s-only fused kernel (test_sgx_kfused.py) hand-rolled the s-shell integral.
This generalization builds the fused K on GPU4PySCF's existing general 3c-1e
device machinery (GINT_g1e + HRR), so it handles s/p/d/f uniformly: per
shell-pair-type block it computes the integral block via the validated
recurrence and contracts it directly into K (no dense (ngrids,nao,nao) tensor),
with density screening.

Validated against the unscreened get_k on s+p and s+p+d systems.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_kfused_general.py -v
Requires a CUDA GPU + gpu4pyscf source build.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_jk import get_k
from gpu4pyscf.sgx.sgx_kfused import get_k_fused_general


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


def _water_svp():
    # O + 2H, def2-svp -> s, p (and O has multiple s/p shells). Up to l=1.
    return gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
                 basis="def2-svp", unit="Angstrom", verbose=0)


def _water_tzvp():
    # def2-tzvp on O/H includes d functions -> tests up to l=2.
    return gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
                 basis="def2-tzvp", unit="Angstrom", verbose=0)


class TestFusedGeneralSP:
    def test_matches_unscreened_svp(self):
        """s+p system: fused-general (tol<=0) matches unscreened get_k."""
        mol = _water_svp()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_general(mol, dm, g, tol=0.0, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-7, rtol=0)

    def test_symmetric_svp(self):
        mol = _water_svp()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k = cp.asnumpy(cp.asarray(
            get_k_fused_general(mol, dm, g, tol=0.0, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k.T, atol=1e-9, rtol=0)


class TestFusedGeneralD:
    def test_matches_unscreened_tzvp(self):
        """s+p+d system: fused-general matches unscreened get_k (up to l=2)."""
        mol = _water_tzvp()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_general(mol, dm, g, tol=0.0, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-6, rtol=0)


class TestFusedGeneralScreened:
    def test_screened_close_svp(self):
        mol = _water_svp()
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_general(mol, dm, g, tol=1e-8, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-4, rtol=0)
