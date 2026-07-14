"""TDD spec: COSX exchange under QM/MM point-charge embedding.

Written before implementation (red). The QM/MM embedding (gpu4pyscf.qmmm
mm_charge / add_mm_charges) adds the MM point-charge attraction to get_hcore
only; it does not touch get_jk/get_k. So COSX-K (which we swap into get_jk)
should compose cleanly with embedding. This slice proves that end to end:
a full embedded RIJCOSX SCF converges and matches CPU PySCF SGX + mm_charge.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_qmmm.py -v
Requires a CUDA GPU + gpu4pyscf.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

from pyscf import gto

from gpu4pyscf.sgx.sgx_scf import cosx_fit


def _mol(basis="sto-3g"):
    return gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
        basis=basis,
        verbose=0,
    )


# A few MM point charges near the QM water (Angstrom), mimicking solvent.
_MM_COORDS = np.array(
    [
        [2.5, 0.0, 0.0],
        [-2.0, 1.5, 0.5],
        [0.5, -2.5, 1.0],
    ]
)
_MM_CHARGES = np.array([-0.8, 0.4, 0.4])


class TestCosxQMMM:
    def test_embedded_pbe0_matches_cpu_sgx(self):
        """Embedded GPU RIJCOSX SCF energy must match embedded CPU PySCF SGX
        SCF energy (same seminumerical algorithm + same MM embedding)."""
        from pyscf import dft, sgx as cpu_sgx, qmmm as cpu_qmmm

        mol = _mol()

        # CPU reference: mm_charge(sgx_fit(RKS(pbe0), pjs=True)).
        mf_cpu0 = cpu_sgx.sgx_fit(dft.RKS(mol, xc="pbe0"), pjs=True)
        mf_cpu = cpu_qmmm.mm_charge(mf_cpu0, _MM_COORDS, _MM_CHARGES, unit="Angstrom")
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()
        assert mf_cpu.converged
        grid_level = mf_cpu0.with_df.grids.level

        # GPU: mm_charge(cosx_fit(RKS(pbe0).density_fit())).
        from gpu4pyscf.dft.rks import RKS
        from gpu4pyscf import qmmm as gpu_qmmm

        mf_gpu0 = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=grid_level)
        mf_gpu = gpu_qmmm.mm_charge(mf_gpu0, _MM_COORDS, _MM_CHARGES, unit="Angstrom")
        mf_gpu.conv_tol = 1e-10
        e_gpu = mf_gpu.kernel()
        assert mf_gpu.converged

        assert abs(e_gpu - e_cpu) < 5e-5, f"E_gpu={e_gpu} E_cpu={e_cpu}"

    def test_embedding_shifts_energy(self):
        """Sanity: the MM charges must actually perturb the SCF energy, i.e.
        embedded COSX energy differs from bare COSX energy. Guards against the
        embedding being silently dropped when composed with the COSX wrapper.
        """
        from gpu4pyscf.dft.rks import RKS
        from gpu4pyscf import qmmm as gpu_qmmm

        mol = _mol()

        mf_bare = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=2)
        mf_bare.conv_tol = 1e-10
        e_bare = mf_bare.kernel()
        assert mf_bare.converged

        mf_emb0 = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=2)
        mf_emb = gpu_qmmm.mm_charge(mf_emb0, _MM_COORDS, _MM_CHARGES, unit="Angstrom")
        mf_emb.conv_tol = 1e-10
        e_emb = mf_emb.kernel()
        assert mf_emb.converged

        # The embedding should move the energy by a chemically meaningful amount.
        assert abs(e_emb - e_bare) > 1e-3, f"embedding had no effect: {e_emb} vs {e_bare}"

    def test_embedded_cosx_near_embedded_df(self):
        """Embedded COSX SCF energy should be within the seminumerical floor of
        the embedded analytical RI-JK SCF energy."""
        from gpu4pyscf.dft.rks import RKS
        from gpu4pyscf import qmmm as gpu_qmmm

        mol = _mol()

        mf_df0 = RKS(mol, xc="pbe0").density_fit()
        mf_df = gpu_qmmm.mm_charge(mf_df0, _MM_COORDS, _MM_CHARGES, unit="Angstrom")
        mf_df.conv_tol = 1e-10
        e_df = mf_df.kernel()
        assert mf_df.converged

        mf_cosx0 = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=3)
        mf_cosx = gpu_qmmm.mm_charge(mf_cosx0, _MM_COORDS, _MM_CHARGES, unit="Angstrom")
        mf_cosx.conv_tol = 1e-10
        e_cosx = mf_cosx.kernel()
        assert mf_cosx.converged

        assert abs(e_cosx - e_df) < 1e-3, f"E_cosx={e_cosx} E_df={e_df}"
