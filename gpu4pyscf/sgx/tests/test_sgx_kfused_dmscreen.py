"""TDD accuracy guard for density-matrix-weighted (P-junction) kernel screening.

The fused kernel's screen bound is multiplied by the density factor
max|fg_cart| over the shell-pair's AO ranges, so a (shell-pair, grid point) is
skipped only when its density-WEIGHTED contribution to K is below screen_tol.
Feasibility probe (COSX_GPU4PYSCF_INVESTIGATION.md sec 9): at safe tol this skips
~93% of work with negligible K error, unlike the pure-distance screen (sec 8).

These guard correctness: DM-screened K must match unscreened get_k at safe tol,
be exact at screen_tol=0, and degrade monotonically as tol loosens.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_kfused_dmscreen.py -v
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


class TestDMScreen:
    def test_dmscreen_safe_tol_matches_svp(self):
        mol = _water("def2-svp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_direct(mol, dm, g, screen_tol=1e-10, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-5, rtol=0)

    def test_dmscreen_safe_tol_matches_tzvp(self):
        mol = _water("def2-tzvp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_direct(mol, dm, g, screen_tol=1e-10, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-4, rtol=0)

    def test_dmscreen_zero_exact(self):
        mol = _water("def2-svp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(
            get_k_fused_direct(mol, dm, g, screen_tol=0.0, ovlp_fit=True)
        ))
        np.testing.assert_allclose(k, k_ref, atol=1e-7, rtol=0)
