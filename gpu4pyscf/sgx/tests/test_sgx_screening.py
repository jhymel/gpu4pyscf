"""TDD spec for B0: COSX P-junction screening estimators + accuracy harness.

This is the FEASIBILITY-GATE slice (Option B, stage B0). It does NOT speed
anything up. It provides:
  * a GPU density-matrix shell-pair screening mask, and
  * a harness that measures, for a given tolerance, both the fraction of work
    the mask would skip AND the resulting error in K vs the unscreened build.

The point is the tolerance -> (K error, fraction skippable) curve that decides
whether faithful screening (B1/B2) can deliver. Screening here is realized as
post-hoc masking of the density into the existing (validated) get_k, so we
measure accuracy impact without a fused kernel yet.

Run:  pytest gpu4pyscf/sgx/tests/test_sgx_screening.py -v
Requires a CUDA GPU + gpu4pyscf.
"""
import numpy as np
import pytest

pytest.importorskip("cupy")
pytest.importorskip("gpu4pyscf")

import cupy as cp
from pyscf import gto

from gpu4pyscf.sgx.screening import shell_dm_mask, screening_report


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


class TestShellDmMask:
    def test_mask_shape_and_symmetry(self):
        mol = _mol()
        dm = _dm(mol)
        mask = shell_dm_mask(mol, dm, tol=1e-6)
        mask = cp.asnumpy(cp.asarray(mask))
        assert mask.shape == (mol.nbas, mol.nbas)
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, mask.T)

    def test_tighter_tol_keeps_more(self):
        """Smaller tol => keep more shell-pairs (fewer skipped)."""
        mol = _mol()
        dm = _dm(mol)
        kept_tight = cp.asnumpy(cp.asarray(shell_dm_mask(mol, dm, tol=1e-10))).sum()
        kept_loose = cp.asnumpy(cp.asarray(shell_dm_mask(mol, dm, tol=1e-2))).sum()
        assert kept_tight >= kept_loose

    def test_zero_tol_keeps_all(self):
        """tol <= 0 must keep every shell-pair (no screening)."""
        mol = _mol()
        dm = _dm(mol)
        mask = cp.asnumpy(cp.asarray(shell_dm_mask(mol, dm, tol=0.0)))
        assert mask.all()


class TestScreeningReport:
    def test_report_fields(self):
        mol = _mol()
        dm = _dm(mol)
        from gpu4pyscf.dft import gen_grid as ggen

        g = ggen.Grids(mol); g.level = 1; g.build()
        rep = screening_report(mol, dm, g, tol=1e-6)
        for key in ("tol", "k_max_err", "frac_pairs_skipped"):
            assert key in rep
        assert 0.0 <= rep["frac_pairs_skipped"] <= 1.0
        assert rep["k_max_err"] >= 0.0

    def test_error_monotone_in_tol(self):
        """Looser tol => larger K error and more work skipped (monotone trends).
        This is the core feasibility signal.
        """
        mol = _mol()
        dm = _dm(mol)
        from gpu4pyscf.dft import gen_grid as ggen

        g = ggen.Grids(mol); g.level = 1; g.build()
        tols = [1e-10, 1e-6, 1e-3, 1e-1]
        reps = [screening_report(mol, dm, g, tol=t) for t in tols]
        errs = [r["k_max_err"] for r in reps]
        skips = [r["frac_pairs_skipped"] for r in reps]
        # error non-decreasing as tol loosens
        assert all(errs[i] <= errs[i + 1] + 1e-12 for i in range(len(errs) - 1))
        # skipped fraction non-decreasing as tol loosens
        assert all(skips[i] <= skips[i + 1] + 1e-12 for i in range(len(skips) - 1))

    def test_tight_tol_low_error(self):
        """At a chemically safe tol the K error must be small (screening is
        supposed to be nearly lossless)."""
        mol = _mol()
        dm = _dm(mol)
        from gpu4pyscf.dft import gen_grid as ggen

        g = ggen.Grids(mol); g.level = 3; g.build()
        rep = screening_report(mol, dm, g, tol=1e-8)
        assert rep["k_max_err"] < 1e-4
