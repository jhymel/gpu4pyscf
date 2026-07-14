"""B1: screened COSX exchange K build (grid-point + AO-shell screening).

Turns B0's measured density sparsity into actual work reduction. Per grid block:
  1. drop grid points with negligible total weighted density,
  2. restrict the 3c-1e integral + contractions to AO shells that survive
     density screening for that block,
then assemble K on the compacted arrays (pure cupy gather/scatter, no new CUDA).

Reduces to the unscreened ``sgx_jk.get_k`` as tol -> 0. See
tests/test_sgx_screened_k.py.
"""
from __future__ import annotations

import cupy as cp

from gpu4pyscf.dft import numint as gnumint
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.sgx.screening import shell_dm_mask


def get_k_screened(mol, dm, grids, tol=1e-8, ovlp_fit=True, blksize=None,
                   return_stats=False):
    """Screened COSX exchange matrix K.

    Args:
        mol: pyscf Mole.
        dm: (nao, nao) density matrix (numpy or cupy).
        grids: GPU4PySCF grids object (.coords, .weights cupy arrays).
        tol: screening tolerance; ``tol <= 0`` disables screening (exact).
        ovlp_fit: apply overlap-fitting correction.
        blksize: grid block size.
        return_stats: if True, also return a dict of work-reduction stats.

    Returns:
        K (cupy array), or (K, stats) if return_stats.
    """
    dm = cp.asarray(dm)
    nao = mol.nao

    coords_all = grids.coords
    weights_all = cp.asarray(grids.weights)
    ngrids = coords_all.shape[0]

    ni = gnumint.NumInt()

    sn = cp.zeros((nao, nao))
    vk = cp.zeros((nao, nao))

    if blksize is None:
        blocks = [(0, ngrids)]
    else:
        blocks = [(i0, min(i0 + blksize, ngrids))
                  for i0 in range(0, ngrids, blksize)]

    screen = tol > 0

    # Shell screening is computed once from the (block-independent) DM mask: a
    # shell J is active if it participates in any surviving shell-pair. Expand
    # active shells to the AO column/row indices they span.
    if screen:
        mask = shell_dm_mask(mol, dm, tol)  # (nbas, nbas) bool
        active_shells = cp.asnumpy(mask.any(axis=0))  # shell J active
        ao_loc = mol.ao_loc_nr()
        active_ao_list = []
        for j in range(mol.nbas):
            if active_shells[j]:
                active_ao_list.extend(range(ao_loc[j], ao_loc[j + 1]))
        active_ao = cp.asarray(active_ao_list, dtype=cp.int64)
        nao_active = active_ao.size
    else:
        active_ao = None
        nao_active = nao

    total_grid = 0
    kept_grid = 0

    for i0, i1 in blocks:
        coords = coords_all[i0:i1]
        weights = weights_all[i0:i1]
        total_grid += coords.shape[0]

        ao = cp.asarray(ni.eval_ao(mol, coords, deriv=0))  # (ng, nao)
        wao = ao * weights[:, None]

        if screen:
            # Grid-point screening: a rough upper bound on the weighted density
            # magnitude contributed at each grid point. Drop points below tol.
            g_weight = cp.abs(wao) @ cp.abs(dm).sum(axis=1)  # (ng,)
            keep = g_weight > tol
            if not bool(keep.any()):
                continue
            coords = coords[keep]
            weights = weights[keep]
            ao = ao[keep]
            wao = wao[keep]

        kept_grid += coords.shape[0]

        # Overlap-fit accumulator over the FULL AO of surviving grid points
        # (no shell screening here -- shell-screening sn is riskier).
        sn += ao.T @ wao

        # int1e_grids builds the full (ng, nao, nao) block regardless of shell
        # subset; the grid-point screening above already reduced ng (real work
        # reduction in the integral). Shell screening below reduces the einsum
        # t/v dimensions only (the kernel-level integral screening is B2).
        gbn = cp.asarray(int1e_grids(mol, coords))  # (ng, nao, nao)

        if screen:
            dm_a = dm[cp.ix_(active_ao, active_ao)]
            ao_a = ao[:, active_ao]
            wao_a = wao[:, active_ao]
            gbn_a = gbn[:, active_ao][:, :, active_ao]

            fg = cp.einsum("tj,gj->tg", dm_a, wao_a)      # (nao_a, ng)
            gv = cp.einsum("gvt,tg->vg", gbn_a, fg)       # (nao_a, ng)
            vk_partial = cp.einsum("gu,vg->uv", ao_a, gv)  # (nao_a, nao_a)
            vk[cp.ix_(active_ao, active_ao)] += vk_partial
        else:
            fg = cp.einsum("tj,gj->tg", dm, wao)  # (nao, ng)
            gv = cp.einsum("gvt,tg->vg", gbn, fg)  # (nao, ng)
            vk += cp.einsum("gu,vg->uv", ao, gv)  # (nao, nao)

    if ovlp_fit:
        ovlp = cp.asarray(mol.intor_symmetric("int1e_ovlp"))
        proj = cp.linalg.solve(sn, ovlp)
        vk = cp.einsum("pi,pj->ij", proj, vk)

    vk = (vk + vk.T) * 0.5

    if return_stats:
        frac_grid_skipped = 1.0 - (kept_grid / total_grid) if total_grid else 0.0
        frac_shell_skipped = 1.0 - (nao_active / nao) if nao else 0.0
        stats = {
            "frac_gridpoints_skipped": float(frac_grid_skipped),
            "frac_shell_work_skipped": float(frac_shell_skipped),
        }
        return vk, stats

    return vk
