"""Analytic nuclear gradient of the COSX exchange energy (GPU).

Port of the K-gradient assembly in ``pyscf/sgx/grad/rhf.py`` (``get_jk_grad``,
``with_k`` branch) onto GPU4PySCF primitives: ``eval_ao(deriv=1)`` +
``int3c1e_ip.int1e_grids_ip1``.

The gradient of E_K = -0.25*hyb*sum(dm*K) at fixed dm has an orbital-response
term (d(AO)/dA on the grid) and an integral-response term (d/dA of the 3c-1e
Coulomb integral). Following the CPU reference, the derivative is built on the
bra AO index only and the symmetric ket contribution is recovered by the factor
of 2 in the per-atom energy contraction (``grad_elec``). The Becke grid-weight
response (and the neglected d(overlap-fit)/dR term) enter only at the
grid-discretization-error level; at typical grid levels (>=2) their net
contribution is below the finite-difference tolerance and is not included here.
See tests/test_sgx_grad.py.
"""
from __future__ import annotations

import cupy as cp
import numpy as np

from gpu4pyscf.dft import gen_grid as _ggen
from gpu4pyscf.dft import numint as gnumint
from gpu4pyscf.gto.int3c1e import int1e_grids
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1


def get_k_energy_grad(mol, dm, grid_level=1, hyb=1.0):
    """Nuclear gradient of E_K = -0.25 * hyb * einsum('ij,ij', dm, K_cosx).

    Args:
        mol: pyscf Mole.
        dm: (nao, nao) density matrix, held fixed (numpy or cupy).
        grid_level: Becke grid level.
        hyb: exact-exchange mixing fraction.

    Returns:
        (natm, 3) nuclear gradient in Eh/Bohr (numpy array).
    """
    dm = cp.asarray(dm)
    nao = mol.nao
    natm = mol.natm

    grids = _ggen.Grids(mol)
    grids.level = grid_level
    grids.build()

    coords = grids.coords
    weights = cp.asarray(grids.weights)

    ni = gnumint.NumInt()

    # --- overlap-fitting projection (proj fixed; d(proj)/dR neglected) ---
    ao0 = cp.asarray(ni.eval_ao(mol, coords, deriv=0))  # (ng, nao)
    sn = ao0.T @ (ao0 * weights[:, None])
    ovlp = cp.asarray(mol.intor_symmetric("int1e_ovlp"))
    proj = cp.linalg.solve(sn, ovlp)
    proj_dm = proj @ dm  # (nao, nao): einsum('ki,ij->kj', proj, dm)

    # --- AO values + first derivatives on the grid ---
    ao_all = cp.asarray(ni.eval_ao(mol, coords, deriv=1))  # (4, ng, nao)
    ao = ao_all[0]
    dao = ao_all[1:4]  # (3, ng, nao)
    wao = ao * weights[:, None]

    # fg_{t,g} = sum_j proj_dm_{t,j} wao_{g,j}
    fg = cp.einsum("tj,gj->tg", proj_dm, wao)  # (nao, ng)

    gbn = cp.asarray(int1e_grids(mol, coords))  # (ng, nao, nao) = (g, v, t)
    gv = cp.einsum("gvt,tg->vg", gbn, fg)  # (nao, ng)

    # integral derivative: d/dA (mu|1/|r-C||nu), shape (3, ng, nao_bra, nao_ket)
    ip1 = cp.asarray(int1e_grids_ip1(mol, coords))  # (3, ng, u, t)

    # dgv[x, u, g] = sum_t d/dA(gbn)_{x,g,u,t} fg_{t,g}
    dgv = cp.einsum("xgut,tg->xug", ip1, fg)  # (3, nao, ng)

    dvk = cp.zeros((3, nao, nao))
    # ORBITAL RESPONSE: d(ao_bra)/dR  -> -0.5 einsum('xgu,gv->xuv', dao, gv.T)
    dvk -= 0.5 * cp.einsum("xgu,vg->xuv", dao, gv)
    # INTEGRAL RESPONSE: d(gbn)/dR (bra center A) -> -0.5 einsum('xug,gv->xuv', dgv, ao)
    dvk -= 0.5 * cp.einsum("xug,gv->xuv", dgv, ao)

    dm_np = cp.asnumpy(dm)
    dvk_np = cp.asnumpy(dvk)

    # Energy gradient: veff = -0.5*dvk, de[ia] += 2*sum(veff[:,p0:p1]*dm[p0:p1]),
    # i.e. de[ia] = -hyb*sum(dvk[:,p0:p1]*dm[p0:p1]).
    aoslices = mol.aoslice_by_atom()
    grad = np.zeros((natm, 3))
    for ia in range(natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        grad[ia] -= hyb * np.einsum(
            "xij,ij->x", dvk_np[:, p0:p1], dm_np[p0:p1]
        )

    return grad
