"""TDD spec: analytic nuclear gradient of the COSX exchange energy.

Written before implementation (red). The COSX-K energy gradient has three
physical pieces (mirroring pyscf/sgx/grad/rhf.py):
  * orbital response   -- d(AO)/dA on the grid (eval_ao deriv=1)
  * integral response  -- d/dA and d/dC of the 3c-1e Coulomb integral
                          (int3c1e_ip1 / int3c1e_ip2)
  * grid response      -- Becke grid-weight derivatives (smoothness-critical)

Ground truth = finite differences of our OWN validated COSX exchange energy,
with the Becke grid REBUILT at each displaced geometry (so FD captures the grid
response automatically). Secondary check: agreement with PySCF CPU SGX analytic
gradient.

Interface under test:
    gpu4pyscf.sgx.sgx_grad.get_k_energy_grad(mol, dm, grid_level, hyb=1.0)
      -> (natm, 3) nuclear gradient of  E_K = -0.25 * hyb * einsum('ij,ij', dm, K)
         where K is the COSX exchange matrix (gpu4pyscf.sgx.sgx_jk.get_k).

We validate the gradient of a FIXED-density exchange energy functional
    E_K(R) = -0.25 * hyb * sum_ij dm_ij * K_ij(R)
with dm held constant (Pulay/density-response terms are separate and handled by
the SCF's Lagrangian; the seminumerical-specific contribution is dE_K/dR at
fixed dm, which is exactly what int3c1e_ip + grid response provide).

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_grad.py -v
Requires a CUDA GPU + gpu4pyscf.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_jk import get_k
from gpu4pyscf.sgx.sgx_grad import get_k_energy_grad


def _mol_at(coords_ang, basis="sto-3g"):
    atoms = [
        ["O", coords_ang[0]],
        ["H", coords_ang[1]],
        ["H", coords_ang[2]],
    ]
    return gto.M(atom=atoms, basis=basis, unit="Angstrom", verbose=0)


_GEOM = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.96],
        [0.93, 0.0, -0.24],
    ]
)


def _fixed_dm(mol):
    # A physically reasonable, geometry-independent-shape dm: converge a cheap
    # SCF once and freeze it. Using the SAME dm for energy + FD keeps the test
    # about the integral/grid gradient, not the density response.
    from gpu4pyscf.dft.rks import RKS

    mf = RKS(mol, xc="hf").density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    return cp.asnumpy(cp.asarray(mf.make_rdm1()))


def _cosx_k_energy(mol, dm, grid_level, hyb=1.0):
    from gpu4pyscf.dft import gen_grid as ggen

    g = ggen.Grids(mol)
    g.level = grid_level
    g.build()
    k = cp.asnumpy(cp.asarray(get_k(mol, dm, g, ovlp_fit=True)))
    return -0.25 * hyb * float(np.einsum("ij,ij", dm, k))


class TestCosxKGradient:
    def test_grad_matches_finite_difference(self):
        """Analytic COSX-K energy gradient must match central finite differences
        of the COSX-K energy (grid rebuilt per geometry) to good accuracy.
        """
        mol0 = _mol_at(_GEOM)
        dm = _fixed_dm(mol0)  # frozen density (numpy)
        grid_level = 3

        g_analytic = np.asarray(
            get_k_energy_grad(mol0, dm, grid_level=grid_level, hyb=1.0)
        )

        # Central finite differences on nuclear coordinates (Angstrom).
        h = 1e-3
        g_fd = np.zeros((3, 3))
        for a in range(3):
            for x in range(3):
                gp = _GEOM.copy(); gp[a, x] += h
                gm = _GEOM.copy(); gm[a, x] -= h
                ep = _cosx_k_energy(_mol_at(gp), dm, grid_level)
                em = _cosx_k_energy(_mol_at(gm), dm, grid_level)
                g_fd[a, x] = (ep - em) / (2 * h)

        # Convert analytic grad (Eh/Bohr) to Eh/Angstrom to match FD units.
        from pyscf.data.nist import BOHR  # Angstrom per Bohr
        g_analytic_ang = g_analytic / BOHR

        np.testing.assert_allclose(g_analytic_ang, g_fd, atol=2e-4, rtol=0)

    def test_grad_translational_invariance(self):
        """Sum of per-atom gradients must be ~zero (translational invariance)."""
        mol0 = _mol_at(_GEOM)
        dm = _fixed_dm(mol0)
        g = np.asarray(get_k_energy_grad(mol0, dm, grid_level=3, hyb=1.0))
        np.testing.assert_allclose(g.sum(axis=0), np.zeros(3), atol=1e-5)

    def test_grad_matches_cpu_sgx(self):
        """Cross-check against an INDEPENDENT implementation: finite differences
        of PySCF CPU SGX's exchange matrix K (same E_K definition, different K
        code path). This avoids coupling to CPU SGX's private analytic-gradient
        internals (which require driver-set state) while still validating the
        physics against a second implementation.
        """
        from pyscf import scf, sgx as cpu_sgx
        from pyscf.sgx.sgx_jk import get_jk_favork

        dm = _fixed_dm(_mol_at(_GEOM))

        def _cpu_k_energy(coords_ang, hyb=1.0):
            mol = _mol_at(coords_ang)
            mf = cpu_sgx.sgx_fit(scf.RHF(mol))
            mf.with_df.build()
            mf.with_df.grids.level = 3
            mf.with_df.grids.build()
            _, vk = get_jk_favork(mf.with_df, dm, hermi=1, with_j=False, with_k=True)
            return -0.25 * hyb * float(np.einsum("ij,ij", dm, vk))

        h = 1e-3
        g_cpu_fd = np.zeros((3, 3))
        for a in range(3):
            for x in range(3):
                gp = _GEOM.copy(); gp[a, x] += h
                gm = _GEOM.copy(); gm[a, x] -= h
                g_cpu_fd[a, x] = (_cpu_k_energy(gp) - _cpu_k_energy(gm)) / (2 * h)

        from pyscf.data.nist import BOHR

        g_gpu = np.asarray(get_k_energy_grad(_mol_at(_GEOM), dm, grid_level=3, hyb=1.0))
        g_gpu_ang = g_gpu / BOHR

        np.testing.assert_allclose(g_gpu_ang, g_cpu_fd, atol=3e-4, rtol=0)
