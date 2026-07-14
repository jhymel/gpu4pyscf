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




def get_k_fused_general(mol, dm, grids, tol=1e-8, ovlp_fit=True,
                        return_stats=False):
    """Correct COSX exchange matrix K for ALL angular momenta (s/p/d/f).

    IMPORTANT: this is NOT the tensor-free fused kernel (that exists only for
    s-shells in ``get_k_fused``). It reuses the validated general
    ``GINTfill_int3c1e`` integral machinery, streams the grid in blocks, screens
    shell-pair-type blocks by density magnitude, and contracts each block into K.
    The transient buffer is ``(blk, nao, nao)`` (peak ~ blk*nao^2), so it is
    memory-favorable only relative to the FULL ``(ngrids, nao, nao)`` tensor
    when the grid is split into multiple blocks; it does not reach the s-only
    fused kernel's footprint. Its value is generality + correctness for d/f.

    Returns:
        K (cupy array), or (K, stats) if return_stats.
    """
    from pyscf import lib
    from pyscf.lib import c_null_ptr

    from gpu4pyscf.gto.int3c1e import VHFOpt
    from gpu4pyscf.lib.cupy_helper import cart2sph, get_avail_mem

    dm = cp.asarray(dm, dtype=cp.float64)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    nao = mol.nao

    coords_all = cp.asarray(grids.coords, dtype=cp.float64, order='C')
    weights_all = cp.asarray(grids.weights, dtype=cp.float64)
    ngrids = coords_all.shape[0]

    ni = gnumint.NumInt()
    ao_all = cp.asarray(ni.eval_ao(mol, coords_all, deriv=0), dtype=cp.float64)

    intopt = VHFOpt(mol)
    intopt.build(1e-13, aosym=True)
    omega = mol.omega

    dm_np = cp.asnumpy(dm)

    # Precompute per-block screening. A shell-pair-type block cp_ij_id is kept
    # if the max density magnitude over its AO ranges (times q bound) exceeds
    # tol. tol <= 0 keeps all blocks.
    n_blocks = len(intopt.log_qs)
    keep_block = [True] * n_blocks
    if tol > 0:
        for cp_ij_id in range(n_blocks):
            log_q_ij = intopt.log_qs[cp_ij_id]
            if len(log_q_ij) == 0:
                keep_block[cp_ij_id] = False
                continue
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
            # ao_loc here is sorted-order; map to mol order via _ao_idx.
            idx_i = intopt._ao_idx[i0:i1]
            idx_j = intopt._ao_idx[j0:j1]
            dm_max = np.abs(dm_np[np.ix_(idx_j, idx_i)]).max()
            q_max = float(np.exp(np.max(log_q_ij)))
            keep_block[cp_ij_id] = (q_max * dm_max) > tol
    n_kept = sum(1 for cp_ij_id in range(n_blocks)
                 if len(intopt.log_qs[cp_ij_id]) > 0 and keep_block[cp_ij_id])
    n_nonempty = sum(1 for cp_ij_id in range(n_blocks)
                     if len(intopt.log_qs[cp_ij_id]) > 0)
    frac_skipped = 1.0 - (n_kept / n_nonempty) if n_nonempty else 0.0

    # Grid blocking so the transient (blk, nao, nao) buffer stays bounded (a
    # few hundred MB at most), never the full (ngrids, nao, nao).
    cp.get_default_memory_pool().free_all_blocks()
    avail_mem = get_avail_mem()
    allowed_double = (avail_mem // 4) // 8
    per_grid_double = max(nao * nao, 1)
    ngrids_per_split = max(1, int(allowed_double // per_grid_double))
    ngrids_per_split = min(ngrids_per_split, ngrids)

    sn = cp.zeros((nao, nao))
    vk = cp.zeros((nao, nao))

    row, col = np.tril_indices(nao)

    for p0, p1 in lib.prange(0, ngrids, ngrids_per_split):
        blk = p1 - p0
        grids_slice = coords_all[p0:p1]
        ao_blk = ao_all[p0:p1]                     # (blk, nao)
        w_blk = weights_all[p0:p1]
        wao = ao_blk * w_blk[:, None]
        sn += ao_blk.T @ wao

        gbn_cart = cp.zeros((blk, nao, nao), order='C')
        stream = cp.cuda.get_current_stream()
        for cp_ij_id in range(n_blocks):
            log_q_ij = intopt.log_qs[cp_ij_id]
            if len(log_q_ij) == 0 or not keep_block[cp_ij_id]:
                continue
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            li = intopt.angular[cpi]
            lj = intopt.angular[cpj]

            nbins = 1
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)

            ci0, ci1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi + 1]
            cj0, cj1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj + 1]
            ncart_i = ci1 - ci0
            ncart_j = cj1 - cj0

            ao_offsets = np.array([ci0, cj0], dtype=np.int32)
            strides = np.array([ncart_i, ncart_i * ncart_j], dtype=np.int32)

            int3c_angular_slice = cp.zeros((blk, ncart_j, ncart_i), order='C')

            err = libgint.GINTfill_int3c1e(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(grids_slice.data.ptr, ctypes.c_void_p),
                ctypes.cast(c_null_ptr(), ctypes.c_void_p),
                ctypes.c_int(blk),
                ctypes.cast(int3c_angular_slice.data.ptr, ctypes.c_void_p),
                strides.ctypes.data_as(ctypes.c_void_p),
                ao_offsets.ctypes.data_as(ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(omega))
            if err != 0:
                raise RuntimeError('GINTfill_int3c1e failed')

            si0, si1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
            sj0, sj1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
            if not mol.cart:
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=1, ang=lj)
                int3c_angular_slice = cart2sph(int3c_angular_slice, axis=2, ang=li)

            gbn_cart[:, sj0:sj1, si0:si1] = int3c_angular_slice

        # Fill the symmetric (upper) triangle from the computed lower triangle.
        gbn_cart[:, row, col] = gbn_cart[:, col, row]
        gbn = intopt.unsort_orbitals(gbn_cart, axis=[1, 2])  # (blk, nao, nao)

        fg = cp.einsum("tj,gj->tg", dm, wao)        # (nao, blk)
        gv = cp.einsum("gvt,tg->vg", gbn, fg)        # (nao, blk)
        vk += cp.einsum("gu,vg->uv", ao_blk, gv)     # (nao, nao)

    if ovlp_fit:
        ovlp = cp.asarray(mol.intor_symmetric("int1e_ovlp"))
        proj = cp.linalg.solve(sn, ovlp)
        vk = cp.einsum("pi,pj->ij", proj, vk)

    vk = (vk + vk.T) * 0.5

    if return_stats:
        peak = int(ngrids_per_split) * nao * nao * 8
        stats = {
            # NOTE: this path is NOT the tensor-free fused kernel (that exists
            # only for s-shells in get_k_fused). It reuses GINTfill_int3c1e and
            # holds a transient (blk, nao, nao) Cartesian buffer per grid block,
            # so peak scales as blk*nao^2 and only shrinks when the grid is
            # split into multiple blocks. Correct for all angular momenta;
            # memory-favorable only vs the FULL (ngrids,nao,nao) when blk<ngrids.
            "fused": False,
            "streamed_blocked": True,
            "frac_blocks_skipped": float(frac_skipped),
            "peak_intermediate_bytes": int(peak),
        }
        return vk, stats
    return vk


def get_k_fused_direct(mol, dm, grids, tol=1e-8, ovlp_fit=True,
                       blksize=None, screen_tol=0.0, return_stats=False):
    """TRUE tensor-free fused COSX exchange K for ALL angular momenta.

    A CUDA kernel computes each shell-pair's Cartesian integral via GINT_g1e and
    accumulates the intermediate gv_cart[grid, v_cart] on the fly, so the largest
    intermediate scales as ngrids*nao_cart (NOT ngrids*nao*nao). K = ao^T @ gv
    (+ overlap fit) on host. Density shell-pair screening supported via ``tol``.

    Returns:
        K (cupy array), or (K, stats) if return_stats.
    """
    from gpu4pyscf.gto.int3c1e import VHFOpt

    dm = cp.asarray(dm, dtype=cp.float64)
    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    nao = mol.nao

    coords = cp.asarray(grids.coords, dtype=cp.float64, order='C')
    weights = cp.asarray(grids.weights, dtype=cp.float64)
    ngrids = coords.shape[0]

    ni = gnumint.NumInt()

    intopt = VHFOpt(mol)
    intopt.build(1e-13, aosym=True)
    omega = mol.omega

    C = intopt.cart2sph                    # (nao_cart, nao_sph) sorted
    nao_cart = C.shape[0]

    # Density-based shell-pair-type block screening (grid-independent).
    dm_np = cp.asnumpy(dm)
    n_blocks = len(intopt.log_qs)
    keep_block = [True] * n_blocks
    if tol > 0:
        for cp_ij_id in range(n_blocks):
            log_q_ij = intopt.log_qs[cp_ij_id]
            if len(log_q_ij) == 0:
                keep_block[cp_ij_id] = False
                continue
            cpi = intopt.cp_idx[cp_ij_id]
            cpj = intopt.cp_jdx[cp_ij_id]
            i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi + 1]
            j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj + 1]
            idx_i = intopt._ao_idx[i0:i1]
            idx_j = intopt._ao_idx[j0:j1]
            dm_max = np.abs(dm_np[np.ix_(idx_j, idx_i)]).max()
            q_max = float(np.exp(np.max(log_q_ij)))
            keep_block[cp_ij_id] = (q_max * dm_max) > tol
    n_nonempty = sum(1 for c in range(n_blocks) if len(intopt.log_qs[c]) > 0)
    n_kept = sum(1 for c in range(n_blocks)
                 if len(intopt.log_qs[c]) > 0 and keep_block[c])
    frac_skipped = 1.0 - (n_kept / n_nonempty) if n_nonempty else 0.0

    # Grid blocking: keep the tensor-free intermediate (nao_cart, blk) bounded.
    # Peak per chunk ~ nao_cart * blk (NOT nao_cart * ngrids, NOT nao*nao*ngrids).
    if blksize is None:
        # cap gv_cart+fg_cart per chunk near ~256 MB total.
        blksize = max(256, int(128e6 / 8 / max(nao_cart, 1)))

    sn = cp.zeros((nao, nao))
    vk = cp.zeros((nao, nao))
    stream = cp.cuda.get_current_stream()

    for p0 in range(0, ngrids, blksize):
        p1 = min(p0 + blksize, ngrids)
        blk = p1 - p0
        coords_blk = cp.ascontiguousarray(coords[p0:p1])
        w_blk = weights[p0:p1]
        ao_blk = cp.asarray(ni.eval_ao(mol, coords_blk, deriv=0), dtype=cp.float64)  # (blk, nao)
        wao = ao_blk * w_blk[:, None]
        sn += ao_blk.T @ wao

        # fg (sph mol) -> sph sorted -> cart sorted, (nao_cart, blk).
        fg_sph_mol = cp.einsum("tj,gj->tg", dm, wao)             # (nao, blk)
        fg_sph_sorted = intopt.sort_orbitals(fg_sph_mol, axis=[0])
        fg_cart = cp.ascontiguousarray(C @ fg_sph_sorted)         # (nao_cart, blk)

        gv_cart = cp.zeros((nao_cart, blk), dtype=cp.float64)     # the only big intermediate

        for cp_ij_id in range(n_blocks):
            log_q_ij = intopt.log_qs[cp_ij_id]
            if len(log_q_ij) == 0 or not keep_block[cp_ij_id]:
                continue
            nbins = 1
            bins_locs_ij = np.array([0, len(log_q_ij)], dtype=np.int32)
            err = libgint.GINTsgx_fused_k_gv(
                ctypes.cast(stream.ptr, ctypes.c_void_p),
                intopt.bpcache,
                ctypes.cast(coords_blk.data.ptr, ctypes.c_void_p),
                ctypes.cast(0, ctypes.c_void_p),          # charge_exponents = NULL
                ctypes.c_int(blk),
                ctypes.cast(gv_cart.data.ptr, ctypes.c_void_p),
                ctypes.cast(fg_cart.data.ptr, ctypes.c_void_p),
                bins_locs_ij.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbins),
                ctypes.c_int(cp_ij_id),
                ctypes.c_double(omega),
                ctypes.c_double(screen_tol))
            if err != 0:
                raise RuntimeError(f"GINTsgx_fused_k_gv failed with code {err}")

        gv_sph_sorted = C.T @ gv_cart                            # (nao_sph, blk)
        gv = intopt.unsort_orbitals(gv_sph_sorted, axis=[0])     # (nao, blk)
        vk += cp.einsum("gu,vg->uv", ao_blk, gv)                 # (nao, nao)

    peak_bytes = int(nao_cart * min(blksize, ngrids) * 8)

    if ovlp_fit:
        ovlp = cp.asarray(mol.intor_symmetric("int1e_ovlp"))
        proj = cp.linalg.solve(sn, ovlp)
        vk = cp.einsum("pi,pj->ij", proj, vk)

    vk = (vk + vk.T) * 0.5

    if return_stats:
        stats = {
            "tensor_free": True,
            "peak_intermediate_bytes": int(peak_bytes),
            "frac_blocks_skipped": float(frac_skipped),
        }
        return vk, stats
    return vk
