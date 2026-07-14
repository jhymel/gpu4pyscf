"""TDD spec for the RIJCOSX-style mean-field wrapper (RI-J + GPU COSX-K).

Written before implementation (red). The wrapper takes a GPU4PySCF RKS object,
keeps RI-J for Coulomb, and replaces exact-exchange K with the GPU COSX build
from gpu4pyscf.sgx.sgx_jk.get_k. Prototype scope: full-range hybrids only (omega=0).

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_scf.py -v
Requires a CUDA GPU + gpu4pyscf.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.sgx_scf import cosx_fit


def _mol(basis="sto-3g"):
    return gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
        basis=basis,
        verbose=0,
    )


class TestCosxSCF:
    def test_pbe0_energy_matches_cpu_sgx(self):
        """Full RKS(PBE0) SCF energy with RI-J + GPU COSX-K must match a CPU
        PySCF SGX-fitted RKS(PBE0) SCF (same seminumerical algorithm) to a
        tight tolerance. Both use dfj=True (RI-J) so only K differs from the
        analytical reference.
        """
        from pyscf import scf, dft, sgx as cpu_sgx

        mol = _mol()

        # CPU reference: RKS(PBE0) with sgx_fit(pjs=True) => RI-J + SGX-K.
        mf_cpu = cpu_sgx.sgx_fit(dft.RKS(mol, xc="pbe0"), pjs=True)
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()
        assert mf_cpu.converged

        # GPU: RKS(PBE0).density_fit() with COSX-K wrapper, same grids level.
        from gpu4pyscf.dft.rks import RKS

        mf_gpu = cosx_fit(RKS(mol, xc="pbe0").density_fit(),
                          grid_level=mf_cpu.with_df.grids.level)
        mf_gpu.conv_tol = 1e-10
        e_gpu = mf_gpu.kernel()
        assert mf_gpu.converged

        assert abs(e_gpu - e_cpu) < 5e-5, f"E_gpu={e_gpu} E_cpu={e_cpu}"

    def test_pbe0_energy_near_analytical_df(self):
        """COSX SCF energy should be close to the analytical RI-JK SCF energy
        (difference is the intrinsic seminumerical-K error, small)."""
        from gpu4pyscf.dft.rks import RKS

        mol = _mol()

        mf_df = RKS(mol, xc="pbe0").density_fit()
        mf_df.conv_tol = 1e-10
        e_df = mf_df.kernel()
        assert mf_df.converged

        mf_cosx = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=3)
        mf_cosx.conv_tol = 1e-10
        e_cosx = mf_cosx.kernel()
        assert mf_cosx.converged

        # sub-mHa agreement is the COSX design target
        assert abs(e_cosx - e_df) < 1e-3, f"E_cosx={e_cosx} E_df={e_df}"

    def test_rejects_range_separated_hybrid(self):
        """Prototype supports full-range hybrids only; RSH must raise clearly."""
        from gpu4pyscf.dft.rks import RKS

        mol = _mol()
        with pytest.raises((NotImplementedError, ValueError)):
            cosx_fit(RKS(mol, xc="wb97x").density_fit())
