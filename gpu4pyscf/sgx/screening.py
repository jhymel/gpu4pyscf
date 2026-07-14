"""B0: COSX P-junction (density-matrix) screening estimators + accuracy harness.

Feasibility-gate stage. Provides a GPU shell-pair DM screening mask and a
harness that measures the tolerance -> (K error, fraction of work skipped)
trade-off, to decide whether faithful screening (B1/B2) is worth building.

Reuses GPU4PySCF screening primitives (``condense('absmax')``) rather than
porting PySCF's C machinery. See tests/test_sgx_screening.py.
"""
from __future__ import annotations

import cupy as cp

from gpu4pyscf.lib.cupy_helper import condense
from gpu4pyscf.sgx.sgx_jk import get_k


def shell_dm_mask(mol, dm, tol=1e-8):
    """Boolean (nbas, nbas) mask of shell-pairs surviving DM screening.

    A shell-pair (I, J) is kept if the density-weighted magnitude bound for that
    pair exceeds ``tol``. ``tol <= 0`` keeps all pairs.

    Returns:
        (nbas, nbas) boolean array (cupy).
    """
    dm = cp.asarray(dm)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)

    ao_loc = mol.ao_loc_nr()
    ao_loc_cp = cp.asarray(ao_loc, dtype=cp.int32)

    dm_shl = condense('absmax', cp.abs(dm), ao_loc_cp)

    if tol <= 0:
        mask = cp.ones_like(dm_shl, dtype=bool)
    else:
        mask = dm_shl > tol

    mask = mask | mask.T
    return mask


def screening_report(mol, dm, grids, tol=1e-8):
    """Measure the accuracy/skippable-work trade-off for a screening tolerance.

    Realizes screening as post-hoc masking of the density into the unscreened
    COSX get_k (measures accuracy impact without a fused kernel), and reports:
        - tol
        - k_max_err          : max|K_screened - K_unscreened|
        - frac_pairs_skipped : fraction of shell-pairs the mask drops

    Returns:
        dict with the fields above.
    """
    dm_cp = cp.asarray(dm)
    if dm_cp.ndim == 3:
        dm_cp = dm_cp.sum(axis=0)

    mask = shell_dm_mask(mol, dm, tol)

    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    ao_mask = cp.zeros((nao, nao), dtype=dm_cp.dtype)
    mask_np = cp.asnumpy(mask)
    for i in range(mol.nbas):
        i0, i1 = ao_loc[i], ao_loc[i + 1]
        for j in range(mol.nbas):
            if mask_np[i, j]:
                j0, j1 = ao_loc[j], ao_loc[j + 1]
                ao_mask[i0:i1, j0:j1] = 1.0

    dm_screened = dm_cp * ao_mask

    _blk = max(128, int(256e6 / 8 / max(nao * nao, 1)))
    K_unscreened = get_k(mol, dm_cp, grids, ovlp_fit=True, blksize=_blk)
    K_screened = get_k(mol, dm_screened, grids, ovlp_fit=True, blksize=_blk)

    k_max_err = float(cp.asnumpy(cp.abs(K_unscreened - K_screened)).max())
    frac_pairs_skipped = float(1.0 - mask.sum() / mask.size)

    return {
        "tol": float(tol),
        "k_max_err": k_max_err,
        "frac_pairs_skipped": frac_pairs_skipped,
    }
