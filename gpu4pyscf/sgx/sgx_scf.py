"""RIJCOSX-style mean-field wrapper for GPU4PySCF.

Keeps RI-J (density fitting) for the Coulomb matrix and replaces exact-exchange
K with the GPU COSX build in ``gpu4pyscf.sgx.sgx_jk.get_k``. Prototype scope:
full-range hybrids only (omega == 0). See tests/test_sgx_scf.py for the spec.
"""
from __future__ import annotations


def cosx_fit(mf, grid_level=1, blksize=None):
    """Wrap a GPU4PySCF density-fitted RKS so exchange K is computed by COSX.

    Args:
        mf: a GPU4PySCF ``RKS(...).density_fit()`` mean-field object (hybrid xc).
        grid_level: Becke grid level for the COSX K grid.
        blksize: grid-block size for the COSX K build (streams the seminumerical
            integral in chunks to bound peak GPU memory). Defaults to a value
            that keeps the per-block (ngrids_blk, nao, nao) integral tensor
            within a few hundred MB.

    Returns:
        The same ``mf`` object with ``get_k`` overridden to use COSX-K.
    """
    import cupy as cp

    from gpu4pyscf.dft import gen_grid as ggen

    from gpu4pyscf.sgx.sgx_jk import get_k as cosx_get_k

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mf.mol.spin)
    if omega != 0:
        raise NotImplementedError(
            "COSX prototype supports full-range hybrids only (omega==0); "
            f"got range-separated xc {mf.xc!r}."
        )
    if hyb == 0:
        raise ValueError(
            f"COSX needs exact exchange, but xc {mf.xc!r} has no HF exchange (hyb==0)."
        )

    g = ggen.Grids(mf.mol)
    g.level = grid_level
    g.build()

    # Default block size: cap the per-block (nblk, nao, nao) float64 integral
    # tensor near ~256 MB so peak memory stays bounded as the system grows.
    nao = mf.mol.nao
    if blksize is None:
        blksize = max(128, int(256e6 / 8 / max(nao * nao, 1)))

    def _cosx_k(mol, dm, omega=None):
        if omega:
            raise NotImplementedError(
                "COSX prototype does not support screened-Coulomb (omega!=0) K"
            )
        return cosx_get_k(
            mol if mol is not None else mf.mol, cp.asarray(dm), g,
            ovlp_fit=True, blksize=blksize,
        )

    # RI-J for Coulomb, COSX for exchange. The density-fitted RKS get_veff
    # (gpu4pyscf.df.df_jk._DFHF.get_veff) obtains J and K together through
    # get_jk for hybrids, so overriding get_jk is what actually swaps K.
    def _cosx_get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True,
                     omega=None):
        if dm is None:
            dm = mf.make_rdm1()
        vj = None
        vk = None
        if with_j:
            vj = mf.with_df.get_jk(dm, hermi, True, False,
                                   mf.direct_scf_tol, omega)[0]
        if with_k:
            vk = _cosx_k(mol, dm, omega)
        return vj, vk

    def _cosx_get_k(mol=None, dm=None, hermi=1, omega=None):
        if dm is None:
            dm = mf.make_rdm1()
        return _cosx_k(mol, dm, omega)

    mf.get_jk = _cosx_get_jk
    mf.get_k = _cosx_get_k
    return mf
