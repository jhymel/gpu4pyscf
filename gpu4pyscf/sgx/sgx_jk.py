"""Seminumerical (chain-of-spheres) exchange K on the GPU.

Port of the K-only path of ``pyscf.sgx.sgx_jk.get_jk_favork`` onto GPU4PySCF
primitives (``gpu4pyscf.gto.int3c1e.int1e_grids`` + ``gpu4pyscf.dft.numint``).

Prototype: K only, no P-junction screening. See tests/test_sgx_k.py for the
spec.
"""
from __future__ import annotations

import cupy as cp

from gpu4pyscf.dft import numint as gnumint
from gpu4pyscf.gto.int3c1e import int1e_grids


def get_k(mol, dm, grids, ovlp_fit=True, blksize=None):
    """Compute the COSX exchange matrix K for density matrix ``dm``.

    Args:
        mol: pyscf Mole.
        dm: (nao, nao) density matrix (numpy or cupy).
        grids: a GPU4PySCF grids object with ``.coords`` and ``.weights``
            (cupy arrays).
        ovlp_fit: apply the overlap-fitting correction (reduces grid aliasing).
        blksize: optional grid block size for memory control.

    Returns:
        K matrix (cupy array), symmetric, matching PySCF ``vk`` sign convention.
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
        blocks = [(i0, min(i0 + blksize, ngrids)) for i0 in range(0, ngrids, blksize)]

    for i0, i1 in blocks:
        coords = coords_all[i0:i1]
        weights = weights_all[i0:i1]

        ao = cp.asarray(ni.eval_ao(mol, coords, deriv=0))  # (ng, nao)
        wao = ao * weights[:, None]
        sn += ao.T @ wao

        fg = cp.einsum("tj,gj->tg", dm, wao)  # (nao, ng)
        gbn = cp.asarray(int1e_grids(mol, coords))  # (ng, nao, nao)
        gv = cp.einsum("gvt,tg->vg", gbn, fg)  # (nao, ng)
        vk += cp.einsum("gu,vg->uv", ao, gv)  # (nao, nao)

    if ovlp_fit:
        ovlp = cp.asarray(mol.intor_symmetric("int1e_ovlp"))
        proj = cp.linalg.solve(sn, ovlp)
        vk = cp.einsum("pi,pj->ij", proj, vk)

    vk = (vk + vk.T) * 0.5
    return vk
