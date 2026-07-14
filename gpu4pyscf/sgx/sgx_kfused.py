"""B2.3: fused, streamed COSX K-direct build (l=0 first).

Computes K without materializing the (ngrids, nao, nao) integral tensor: a CUDA
kernel evaluates each shell-pair's analytic 3c-1e integral inline per grid point
and contracts directly into the K matrix, with density-based shell-pair
screening. Only path to both speed and low VRAM at 1000+ BF.

Current scope: l=0 (s-function) shells. See tests/test_sgx_kfused.py.
"""
from __future__ import annotations

import ctypes

import cupy as cp
import numpy as np

from gpu4pyscf.dft import numint as gnumint
from gpu4pyscf.gto.int3c1e import libgint
from gpu4pyscf.sgx.screening import shell_dm_mask

placeholder = None  # body grows via incremental edits


def _s_shell_primitive_data(mol):
    """Flatten per-s-shell primitive data for the fused kernel.

    For s-shells the AO index equals the shell index, so arrays are indexed by
    shell (== AO). Coefficients are folded with the spherical s-GTO primitive
    normalization ``(2*alpha/pi)^{3/4}`` so the inline integral matches
    ``int1e_grids``.

    Returns numpy arrays: centers (nbas,3), nprim (nbas,), prim_off (nbas,),
    exps (nprim_tot,), coefs (nprim_tot,).
    """
    if not all(mol.bas_angular(i) == 0 for i in range(mol.nbas)):
        raise NotImplementedError("get_k_fused currently supports s-shells only")

    atom_coords = mol.atom_coords()  # bohr
    centers = np.empty((mol.nbas, 3), dtype=np.float64)
    nprim = np.empty(mol.nbas, dtype=np.int32)
    prim_off = np.empty(mol.nbas, dtype=np.int32)

    exps_list = []
    coefs_list = []
    offset = 0
    for ish in range(mol.nbas):
        centers[ish] = atom_coords[mol.bas_atom(ish)]
        a = mol.bas_exp(ish)
        c = mol.bas_ctr_coeff(ish)[:, 0] * (2.0 * a / np.pi) ** 0.75
        nprim[ish] = a.size
        prim_off[ish] = offset
        offset += a.size
        exps_list.append(a)
        coefs_list.append(c)

    exps = np.concatenate(exps_list).astype(np.float64)
    coefs = np.concatenate(coefs_list).astype(np.float64)
    return centers, nprim, prim_off, exps, coefs


def get_k_fused(mol, dm, grids, tol=1e-8, ovlp_fit=True, return_stats=False):
    """Fused COSX exchange matrix K (no dense integral tensor).

    Args:
        mol: pyscf Mole (currently s-functions only).
        dm: (nao, nao) density matrix (numpy or cupy).
        grids: GPU4PySCF grids object (.coords, .weights cupy arrays).
        tol: shell-pair screening tolerance; ``tol <= 0`` disables screening.
        ovlp_fit: apply overlap-fitting correction.
        return_stats: if True, also return a stats dict (incl. "fused": True).

    Returns:
        K (cupy array), or (K, stats) if return_stats.
    """
    dm = cp.asarray(dm, dtype=cp.float64)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    nao = mol.nao

    coords = cp.asarray(grids.coords, dtype=cp.float64)
    weights = cp.asarray(grids.weights, dtype=cp.float64)
    ngrids = coords.shape[0]

    ni = gnumint.NumInt()
    ao = cp.asarray(ni.eval_ao(mol, coords, deriv=0), dtype=cp.float64)  # (ng, nao)

    # Overlap-fit accumulator sn = ao.T @ (ao * w), matching get_k.
    wao = ao * weights[:, None]
    sn = ao.T @ wao

    # Per-shell primitive data (s-shells: shell index == AO index).
    centers, nprim, prim_off, exps, coefs = _s_shell_primitive_data(mol)
    d_centers = cp.asarray(centers)
    d_nprim = cp.asarray(nprim)
    d_prim_off = cp.asarray(prim_off)
    d_exps = cp.asarray(exps)
    d_coefs = cp.asarray(coefs)

    # Shell-pair keep mask (for s-shells nbas == nao). tol <= 0 keeps all.
    if tol > 0:
        keep = shell_dm_mask(mol, dm, tol).astype(cp.int32)
        d_keep_ptr = keep.data.ptr
    else:
        keep = None
        d_keep_ptr = 0

    # Fused kernel: contracts directly into K (nao, nao), no (ng, nao, nao).
    vk = cp.zeros((nao, nao), dtype=cp.float64)
    dm_c = cp.ascontiguousarray(dm)
    ao_c = cp.ascontiguousarray(ao)

    stream = cp.cuda.get_current_stream()
    err = libgint.GINTsgx_fused_k_s(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(vk.data.ptr, ctypes.c_void_p),
        ctypes.cast(ao_c.data.ptr, ctypes.c_void_p),
        ctypes.cast(dm_c.data.ptr, ctypes.c_void_p),
        ctypes.cast(weights.data.ptr, ctypes.c_void_p),
        ctypes.cast(coords.data.ptr, ctypes.c_void_p),
        ctypes.cast(d_keep_ptr, ctypes.c_void_p),
        ctypes.cast(d_centers.data.ptr, ctypes.c_void_p),
        ctypes.cast(d_nprim.data.ptr, ctypes.c_void_p),
        ctypes.cast(d_prim_off.data.ptr, ctypes.c_void_p),
        ctypes.cast(d_exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(d_coefs.data.ptr, ctypes.c_void_p),
        ctypes.c_int(nao),
        ctypes.c_int(ngrids),
        ctypes.c_double(0.0))
    if err != 0:
        raise RuntimeError(f"GINTsgx_fused_k_s failed with code {err}")

    if ovlp_fit:
        ovlp = cp.asarray(mol.intor_symmetric("int1e_ovlp"))
        proj = cp.linalg.solve(sn, ovlp)
        vk = cp.einsum("pi,pj->ij", proj, vk)

    vk = (vk + vk.T) * 0.5

    if return_stats:
        # Largest intermediate never includes an (ngrids, nao, nao) term.
        peak = ao.nbytes + vk.nbytes
        stats = {"fused": True, "peak_intermediate_bytes": int(peak)}
        return vk, stats
    return vk


