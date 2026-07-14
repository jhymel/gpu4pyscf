"""TDD spec for the GPU COSX/SGX K-only prototype.

These tests define the interface and expected behavior of ``gpu4pyscf.sgx``. They are
written BEFORE the implementation (red), per the Dayhoff TDD workflow. The
implementer's job is to make them pass without modifying this file.

Run:  pytest gpu4pyscf/sgx/tests/ -v
Requires a CUDA GPU + gpu4pyscf.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_jk import get_k


def _water(basis="sto-3g"):
    return gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
        basis=basis,
        verbose=0,
    )


def _random_dm(nao, seed=0):
    rng = np.random.RandomState(seed)
    a = rng.rand(nao, nao)
    return a + a.T


def _gpu_grids(mol, level=1):
    from gpu4pyscf.dft import gen_grid as ggen

    g = ggen.Grids(mol)
    g.level = level
    g.build()
    return g


class TestKAssembly:
    """K assembly must be exact (machine precision) vs a direct contraction of
    the raw 3c-1e integral on the SAME grid, WITHOUT overlap fitting. This
    proves the grid-weighted chain-of-spheres assembly is correct, independent
    of grid-accuracy questions.
    """

    def test_k_matches_direct_contraction_no_ovlp_fit(self):
        from gpu4pyscf.gto.int3c1e import int1e_grids

        mol = _water()
        nao = mol.nao
        dm = _random_dm(nao)
        grids = _gpu_grids(mol, level=1)

        k = get_k(mol, dm, grids, ovlp_fit=False)
        k = cp.asnumpy(cp.asarray(k))

        # Direct reference on the same grid:
        #   K_uv = sum_g w_g * ao_gu * (sum_t gbn_gvt * (sum_j dm_tj w_g ao_gj))
        # Assembled block-free for a small system.
        from gpu4pyscf.dft import numint as gnumint

        ni = gnumint.NumInt()
        coords = grids.coords
        weights = cp.asarray(grids.weights)
        ao = cp.asarray(ni.eval_ao(mol, coords, deriv=0))          # (ng, nao)
        wao = ao * weights[:, None]
        dm_g = cp.asarray(dm)
        # fg[t,g] = sum_j dm_tj * wao_gj
        fg = cp.einsum("tj,gj->tg", dm_g, wao)
        gbn = cp.asarray(int1e_grids(mol, coords))                  # (ng, nao, nao)
        gv = cp.einsum("gvt,tg->vg", gbn, fg)                       # (nao, ng)
        vk = cp.einsum("gu,vg->uv", ao, gv)                         # (nao, nao)
        vk = (vk + vk.T) * 0.5
        ref = cp.asnumpy(vk)

        np.testing.assert_allclose(k, ref, atol=1e-10, rtol=0)

    def test_k_is_symmetric(self):
        mol = _water()
        dm = _random_dm(mol.nao)
        grids = _gpu_grids(mol, level=1)
        k = cp.asnumpy(cp.asarray(get_k(mol, dm, grids, ovlp_fit=True)))
        np.testing.assert_allclose(k, k.T, atol=1e-10, rtol=0)


class TestKPhysicsParityCPU:
    """With overlap fitting on, GPU COSX K must match PySCF's CPU SGX K on the
    same Becke grid to tight tolerance (same algorithm, different hardware).
    """

    def test_k_matches_cpu_sgx(self):
        from pyscf import sgx as cpu_sgx
        from pyscf.sgx.sgx_jk import get_jk_favork

        mol = _water()
        nao = mol.nao
        dm = _random_dm(nao)

        # CPU SGX K on its own grid (overlap fitting on by default).
        sgx = cpu_sgx.SGX(mol)
        sgx.grids = sgx.build_grids() if hasattr(sgx, "build_grids") else None
        # Fall back to the standard sgx_fit setup to get a grids object.
        mf = cpu_sgx.sgx_fit(__import__("pyscf").scf.RHF(mol))
        mf.with_df.build()
        sgxobj = mf.with_df
        _, vk_cpu = get_jk_favork(sgxobj, dm, hermi=1, with_j=False, with_k=True)

        # GPU COSX K on the SAME grid.
        grids = sgxobj.grids
        k_gpu = cp.asnumpy(cp.asarray(get_k(mol, dm, grids, ovlp_fit=True)))

        np.testing.assert_allclose(k_gpu, vk_cpu, atol=1e-8, rtol=0)


class TestKGridConvergence:
    """As the grid is refined, GPU COSX K must approach the analytical
    density-fit K (the physical target). This is a loose, monotone check.
    """

    def test_k_approaches_df_k_with_finer_grid(self):
        # COSX-K vs analytical DF-K drops sharply from the coarsest grid, then
        # plateaus at the intrinsic seminumerical floor (~3e-4 for this
        # molecule/basis, reached by ~level 1-2). So the *converging* regime is
        # level 0 -> level 2; comparing two already-converged levels is a
        # no-op (verified: level 1 and 3 both sit on the ~3.2e-4 floor).
        from gpu4pyscf.dft.rks import RKS

        mol = _water()
        dm = _random_dm(mol.nao)

        mf = RKS(mol, xc="hf").density_fit()
        vk_df = cp.asnumpy(cp.asarray(mf.get_k(mol, cp.asarray(dm))))

        errs = []
        for level in (0, 2):
            grids = _gpu_grids(mol, level=level)
            k = cp.asnumpy(cp.asarray(get_k(mol, dm, grids, ovlp_fit=True)))
            errs.append(np.abs(k - vk_df).max())

        assert errs[1] < errs[0], f"K did not converge with grid: {errs}"
        assert errs[1] < 5e-2, f"converged K error too large: {errs[1]}"
