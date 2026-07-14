"""TDD spec for the TRUE tensor-free all-L fused COSX K (get_k_fused_direct).

Unlike get_k_fused_general (which reuses GINTfill_int3c1e and holds a
(blk, nao, nao) Cartesian buffer), this path is genuinely tensor-free for ALL
angular momenta: a CUDA kernel computes each shell-pair's Cartesian integral via
GINT_g1e and accumulates the intermediate gv_cart[grid, v_cart] on the fly
(atomicAdd), so the largest intermediate scales as ngrids*nao_cart, NOT
ngrids*nao*nao. K = ao^T @ gv (+ overlap fit) on host.

Validated vs unscreened get_k on s+p (def2-svp) and s+p+d (def2-tzvp).

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_kfused_direct.py -v
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


class TestFusedDirectCorrectness:
    def test_matches_unscreened_svp(self):
        mol = _water("def2-svp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(get_k_fused_direct(mol, dm, g, tol=0.0, ovlp_fit=True)))
        np.testing.assert_allclose(k, k_ref, atol=1e-7, rtol=0)

    def test_matches_unscreened_tzvp(self):
        """s+p+d (up to l=2)."""
        mol = _water("def2-tzvp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k_ref = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
        k = cp.asnumpy(cp.asarray(get_k_fused_direct(mol, dm, g, tol=0.0, ovlp_fit=True)))
        np.testing.assert_allclose(k, k_ref, atol=1e-6, rtol=0)

    def test_symmetric(self):
        mol = _water("def2-svp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        k = cp.asnumpy(cp.asarray(get_k_fused_direct(mol, dm, g, tol=0.0, ovlp_fit=True)))
        np.testing.assert_allclose(k, k.T, atol=1e-9, rtol=0)


class TestFusedDirectIsTensorFree:
    def test_peak_intermediate_scales_ng_nao_not_squared(self):
        """The largest intermediate must scale ~ ngrids*nao_cart, i.e. be far
        below the dense ngrids*nao*nao tensor. Guards against a (blk,nao,nao)
        buffer sneaking back in."""
        mol = _water("def2-tzvp")
        dm = _dm(mol)
        g = _grids(mol, level=1)
        ng = g.coords.shape[0]
        nao = mol.nao
        _, stats = get_k_fused_direct(mol, dm, g, tol=0.0, ovlp_fit=True, return_stats=True)
        peak = stats["peak_intermediate_bytes"]
        dense = ng * nao * nao * 8
        # tensor-free intermediate should be at least ~10x smaller than dense
        assert peak < dense / 10, f"peak={peak} not << dense={dense}; tensor likely materialized"
        assert stats.get("tensor_free") is True
