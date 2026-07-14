"""TDD spec: end-to-end RIJCOSX SCF using the fused screened COSX-K path.

cosx_fit(mf, fused=True, screen_tol=...) wires get_k_fused_direct (tensor-free,
DM-screened, all-L) into the SCF as the exchange builder, keeping RI-J for
Coulomb. This is the usable end-to-end path: a full RKS hybrid SCF whose K comes
from the fused kernel.

Validated: fused RIJCOSX SCF energy matches the non-fused cosx_fit SCF (same
seminumerical algorithm) and is near the analytical RI-JK SCF.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_scf_fused.py -v
Requires a CUDA GPU + gpu4pyscf source build.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

from pyscf import gto

from gpu4pyscf.sgx.sgx_scf import cosx_fit


def _mol(basis="def2-svp"):
    return gto.M(atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
                 basis=basis, unit="Angstrom", verbose=0)


class TestFusedSCF:
    def test_fused_scf_matches_nonfused(self):
        """Fused-path RIJCOSX SCF energy matches the (streamed get_k) cosx_fit
        SCF at the same grid level -- same seminumerical algorithm, different K
        implementation."""
        from gpu4pyscf.dft.rks import RKS

        mol = _mol("def2-svp")

        mf_ref = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=1)
        mf_ref.conv_tol = 1e-9
        e_ref = mf_ref.kernel()
        assert mf_ref.converged

        mf = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=1,
                      fused=True, screen_tol=1e-10)
        mf.conv_tol = 1e-9
        e = mf.kernel()
        assert mf.converged

        assert abs(e - e_ref) < 1e-5, f"e_fused={e} e_ref={e_ref}"

    def test_fused_scf_near_analytical(self):
        """Fused RIJCOSX SCF energy is within the seminumerical floor of the
        analytical RI-JK SCF energy."""
        from gpu4pyscf.dft.rks import RKS

        mol = _mol("def2-svp")

        mf_df = RKS(mol, xc="pbe0").density_fit()
        mf_df.conv_tol = 1e-9
        e_df = mf_df.kernel()
        assert mf_df.converged

        mf = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=3,
                      fused=True, screen_tol=1e-10)
        mf.conv_tol = 1e-9
        e = mf.kernel()
        assert mf.converged

        assert abs(e - e_df) < 1e-3, f"e_fused={e} e_df={e_df}"

    def test_fused_scf_tzvp_d_functions(self):
        """Fused SCF works with d functions (def2-tzvp)."""
        from gpu4pyscf.dft.rks import RKS

        mol = _mol("def2-tzvp")

        mf_ref = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=1)
        mf_ref.conv_tol = 1e-9
        e_ref = mf_ref.kernel()

        mf = cosx_fit(RKS(mol, xc="pbe0").density_fit(), grid_level=1,
                      fused=True, screen_tol=1e-10)
        mf.conv_tol = 1e-9
        e = mf.kernel()
        assert mf.converged
        assert abs(e - e_ref) < 1e-5, f"e_fused={e} e_ref={e_ref}"
