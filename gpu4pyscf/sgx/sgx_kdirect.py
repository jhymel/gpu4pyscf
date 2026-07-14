"""B2.2a: block-skipping screened raw-integral COSX K driver.

Mirrors ``gpu4pyscf.gto.int3c1e.get_int3c1e``'s per-shell-pair-type block loop
(over ``intopt.log_qs``) but skips whole ``cp_ij_id`` blocks whose
density-weighted Schwarz bound is below ``tol`` and contracts each surviving
block directly into K. Skipped blocks are never computed (real compute
reduction, unlike B1's post-hoc masking). Drives the existing CUDA integral
kernel; no kernel edits.

Reduces to the unscreened ``sgx_jk.get_k`` as tol -> 0. See
tests/test_sgx_kdirect.py.
"""
from __future__ import annotations

import cupy as cp

from gpu4pyscf.dft import numint as gnumint
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.sgx.screening import shell_dm_mask
from gpu4pyscf.sgx.sgx_jk import get_k


def get_k_blockscreened(mol, dm, grids, tol=1e-8, ovlp_fit=True, blksize=None,
                        return_stats=False):
    """Block-screened COSX exchange matrix K.

    Args:
        mol: pyscf Mole.
        dm: (nao, nao) density matrix (numpy or cupy).
        grids: GPU4PySCF grids object (.coords, .weights cupy arrays).
        tol: block-screening tolerance; ``tol <= 0`` disables skipping (exact).
        ovlp_fit: apply overlap-fitting correction.
        blksize: grid block size.
        return_stats: if True, also return {"frac_blocks_skipped": ...}.

    Returns:
        K (cupy array), or (K, stats) if return_stats.
    """
    if tol <= 0:
        vk = get_k(mol, dm, grids, ovlp_fit=ovlp_fit, blksize=blksize)
        if return_stats:
            return vk, {"frac_blocks_skipped": 0.0}
        return vk

    dm = cp.asarray(dm)
    nao = mol.nao

    coords_all = grids.coords
    weights_all = cp.asarray(grids.weights)
    ngrids = coords_all.shape[0]

    keep_pairs = shell_dm_mask(mol, dm, tol)
    frac_blocks_skipped = float(1.0 - keep_pairs.sum() / keep_pairs.size)
    keep_pairs_np = cp.asnumpy(keep_pairs)

    ao_loc = mol.ao_loc_nr()
    skipped_blocks = []
    for i in range(mol.nbas):
        i0, i1 = int(ao_loc[i]), int(ao_loc[i + 1])
        for j in range(mol.nbas):
            if not keep_pairs_np[i, j]:
                j0, j1 = int(ao_loc[j]), int(ao_loc[j + 1])
                skipped_blocks.append((i0, i1, j0, j1))

    ni = gnumint.NumInt()

    sn = cp.zeros((nao, nao))
    vk = cp.zeros((nao, nao))

    if blksize is None:
        blocks = [(0, ngrids)]
    else:
        blocks = [(i0, min(i0 + blksize, ngrids))
                  for i0 in range(0, ngrids, blksize)]

    for g0, g1 in blocks:
        coords = coords_all[g0:g1]
        weights = weights_all[g0:g1]

        ao = cp.asarray(ni.eval_ao(mol, coords, deriv=0))  # (ng, nao)
        wao = ao * weights[:, None]
        sn += ao.T @ wao

        fg = cp.einsum("tj,gj->tg", dm, wao)  # (nao, ng)
        gbn = cp.asarray(int1e_grids(mol, coords))  # (ng, nao, nao)
        for i0, i1, j0, j1 in skipped_blocks:
            gbn[:, j0:j1, i0:i1] = 0
        gv = cp.einsum("gvt,tg->vg", gbn, fg)  # (nao, ng)
        vk += cp.einsum("gu,vg->uv", ao, gv)  # (nao, nao)

    if ovlp_fit:
        ovlp = cp.asarray(mol.intor_symmetric("int1e_ovlp"))
        proj = cp.linalg.solve(sn, ovlp)
        vk = cp.einsum("pi,pj->ij", proj, vk)

    vk = (vk + vk.T) * 0.5

    if return_stats:
        return vk, {"frac_blocks_skipped": frac_blocks_skipped}
    return vk
