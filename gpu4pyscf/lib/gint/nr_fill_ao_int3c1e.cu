/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "gint1e.h"
#include "cuda_alloc.cuh"
#include "cint2e.cuh"

#include "rys_roots.cu"
#include "g1e.cu"
#include "g1e_root_1.cu"
#include "g3c1e.cu"

static int GINTfill_int3c1e_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                  const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                  const double omega, const double* grid_points, const double* charge_exponents, const cudaStream_t stream)
{
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    int type_ijkl;
    switch (nrys_roots) {
    case 1:
        type_ijkl = (i_l << 2) | j_l;
        switch (type_ijkl) {
        case (0<<2)|0: GINTfill_int3c1e_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        case (1<<2)|0: GINTfill_int3c1e_kernel10<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "roots=1 type_ijkl %d\n", type_ijkl);
        }
        break;
    case 2: GINTfill_int3c1e_kernel_general<2, GSIZE2_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 3: GINTfill_int3c1e_kernel_general<3, GSIZE3_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 4: GINTfill_int3c1e_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 5: GINTfill_int3c1e_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    default:
        fprintf(stderr, "rys roots %d\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c1e_charge_contracted_tasks(double* output, const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                    const int stride_j, const int stride_ij, const int ao_offsets_i, const int ao_offsets_j,
                                                    const double omega, const double* grid_points, const double* charge_exponents,
                                                    const int n_charge_sum_per_thread, const cudaStream_t stream)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int ngrids = (offsets.ntasks_kl + n_charge_sum_per_thread - 1) / n_charge_sum_per_thread;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    const int type_ij = i_l * 10 + j_l;
    switch (type_ij) {
    case 00: GINTfill_int3c1e_charge_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 10: GINTfill_int3c1e_charge_contracted_kernel10<<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 11: GINTfill_int3c1e_charge_contracted_kernel_expanded<1, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 20: GINTfill_int3c1e_charge_contracted_kernel_expanded<2, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 21: GINTfill_int3c1e_charge_contracted_kernel_expanded<2, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 22: GINTfill_int3c1e_charge_contracted_kernel_expanded<2, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 30: GINTfill_int3c1e_charge_contracted_kernel_expanded<3, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 31: GINTfill_int3c1e_charge_contracted_kernel_expanded<3, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 32: GINTfill_int3c1e_charge_contracted_kernel_expanded<3, 2> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 40: GINTfill_int3c1e_charge_contracted_kernel_expanded<4, 0> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    case 41: GINTfill_int3c1e_charge_contracted_kernel_expanded<4, 1> <<<blocks, threads, 0, stream>>>(output, offsets, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
    default:
        const int nrys_roots = (i_l + j_l) / 2 + 1;
        switch (nrys_roots) {
        case 4: GINTfill_int3c1e_charge_contracted_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        case 5: GINTfill_int3c1e_charge_contracted_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(output, offsets, i_l, j_l, nprim_ij, stride_j, stride_ij, ao_offsets_i, ao_offsets_j, omega, grid_points, charge_exponents); break;
        default:
            fprintf(stderr, "type_ij = %d, nrys_roots = %d out of range\n", type_ij, nrys_roots);
            return 1;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int GINTfill_int3c1e_density_contracted_tasks(double* output, const double* density, const HermiteDensityOffsets hermite_density_offsets,
                                                     const BasisProdOffsets offsets, const int i_l, const int j_l, const int nprim_ij,
                                                     const double omega, const double* grid_points, const double* charge_exponents,
                                                     const int n_pair_sum_per_thread, const cudaStream_t stream)
{
    const int ntasks_ij = (offsets.ntasks_ij + n_pair_sum_per_thread - 1) / n_pair_sum_per_thread;
    const int ngrids = offsets.ntasks_kl;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    switch (i_l + j_l) {
    case  0: GINTfill_int3c1e_density_contracted_kernel00<<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  1: GINTfill_int3c1e_density_contracted_kernel10<<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  2: GINTfill_int3c1e_density_contracted_kernel_general< 2> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  3: GINTfill_int3c1e_density_contracted_kernel_general< 3> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  4: GINTfill_int3c1e_density_contracted_kernel_general< 4> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  5: GINTfill_int3c1e_density_contracted_kernel_general< 5> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  6: GINTfill_int3c1e_density_contracted_kernel_general< 6> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  7: GINTfill_int3c1e_density_contracted_kernel_general< 7> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    case  8: GINTfill_int3c1e_density_contracted_kernel_general< 8> <<<blocks, threads, 0, stream>>>(output, density, hermite_density_offsets, offsets, nprim_ij, omega, grid_points, charge_exponents); break;
    // Up to g + g = 8 now
    default:
        fprintf(stderr, "i_l + j_l = %d out of range\n", i_l + j_l);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

// B2.3: fused COSX exchange-K for s-shells (l=0). One thread per grid point.
// Computes the s-shell 3c-1e integral I_vt(g) = (v|1/|r-Cg||t) inline (same
// formula as GINTfill_int3c1e_density_contracted_kernel00) from raw per-shell
// primitive data, then contracts directly into K without ever materializing the
// (ngrids, nao, nao) tensor. For s-shells nbas == nao (1 AO per shell).
#define SGX_FUSED_MAX_NAO 64

__global__
static void GINTsgx_fused_k_s_kernel(double* K, const double* ao, const double* dm,
                                     const double* weights, const double* grid_points,
                                     const int* keep, const double* sh_center,
                                     const int* sh_nprim, const int* sh_prim_off,
                                     const double* prim_exp, const double* prim_coef,
                                     const int nao, const int ngrids, const double omega)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrids) {
        return;
    }

    const double w = weights[g];
    const double Cx = grid_points[g * 3 + 0];
    const double Cy = grid_points[g * 3 + 1];
    const double Cz = grid_points[g * 3 + 2];

    // fg[t] = sum_j dm[t,j] * w * ao[g,j]
    double fg[SGX_FUSED_MAX_NAO];
    for (int t = 0; t < nao; t++) {
        double acc = 0.0;
        for (int j = 0; j < nao; j++) {
            acc += dm[t * nao + j] * ao[g * nao + j];
        }
        fg[t] = acc * w;
    }

    // gv[v] = sum_t I_vt(g) * fg[t]
    for (int v = 0; v < nao; v++) {
        const int nv = sh_nprim[v];
        const int ov = sh_prim_off[v];
        const double Ax = sh_center[v * 3 + 0];
        const double Ay = sh_center[v * 3 + 1];
        const double Az = sh_center[v * 3 + 2];

        double gv = 0.0;
        for (int t = 0; t < nao; t++) {
            if (keep != NULL && keep[v * nao + t] == 0) {
                continue;
            }
            const int nt = sh_nprim[t];
            const int ot = sh_prim_off[t];
            const double Bx = sh_center[t * 3 + 0];
            const double By = sh_center[t * 3 + 1];
            const double Bz = sh_center[t * 3 + 2];

            const double ABx = Ax - Bx;
            const double ABy = Ay - By;
            const double ABz = Az - Bz;
            const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;

            double I_vt = 0.0;
            for (int ip = 0; ip < nv; ip++) {
                const double ai = prim_exp[ov + ip];
                const double ci = prim_coef[ov + ip];
                for (int jp = 0; jp < nt; jp++) {
                    const double aj = prim_exp[ot + jp];
                    const double cj = prim_coef[ot + jp];

                    const double aij = ai + aj;
                    const double eij = ci * cj * exp(-ai * aj / aij * AB2);
                    const double Px = (ai * Ax + aj * Bx) / aij;
                    const double Py = (ai * Ay + aj * By) / aij;
                    const double Pz = (ai * Az + aj * Bz) / aij;
                    const double PCx = Px - Cx;
                    const double PCy = Py - Cy;
                    const double PCz = Pz - Cz;

                    double a0 = aij;
                    const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
                    const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
                    a0 *= theta;

                    const double prefactor = 2.0 * M_PI / aij * eij * sqrt_theta;
                    const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
                    double eri = prefactor;
                    if (boys_input > 1e-14) {
                        const double sqrt_boys_input = sqrt(boys_input);
                        const double boys_0 = SQRTPIE4 / sqrt_boys_input * erf(sqrt_boys_input);
                        eri *= boys_0;
                    }
                    I_vt += eri;
                }
            }
            gv += I_vt * fg[t];
        }

        // K[u,v] += ao[g,u] * gv  for all u
        for (int u = 0; u < nao; u++) {
            const double contrib = ao[g * nao + u] * gv;
            if (contrib != 0.0) {
                atomicAdd(K + (u * nao + v), contrib);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TRUE tensor-free fused COSX exchange-K path for ALL angular momenta.
//
// Instead of writing each shell-pair's Cartesian 3c-1e integral into a
// (ngrids, nao, nao) tensor, this kernel contracts it on the fly into the
// intermediate gv_cart[v, g] (shape (nao_cart, ngrids), row-major):
//     gv[i, g] += I_{i,j}(g) * fg[j, g]            (bra i, ket j)
// and, for off-diagonal shell pairs (ish != jsh), also the transpose
//     gv[j, g] += I_{i,j}(g) * fg[i, g]
// so that the fully symmetric integral is accounted for while enumerating
// only the lower-triangle shell pairs (exactly the aosym pair list that
// GINTfill_int3c1e uses). gv_cart and fg_cart use absolute Cartesian
// sorted-AO indices (c_bpcache.ao_loc), so no ao_offsets are needed.
template <int NROOTS>
__device__
static void GINTcontract_gv_int3c1e(const double* g, double* gv_cart, const double* fg_cart,
                                    const int ish, const int jsh, const int i_grid,
                                    const int i_l, const int j_l, const int ngrids)
{
    const int* ao_loc = c_bpcache.ao_loc;

    const int i0 = ao_loc[ish];
    const int i1 = ao_loc[ish+1];
    const int j0 = ao_loc[jsh];
    const int j1 = ao_loc[jsh+1];

    const int *idx = c_idx;
    const int *idy = c_idx + TOT_NF;
    const int *idz = c_idx + TOT_NF * 2;

    const int g_size = NROOTS * (i_l + 1) * (j_l + 1);
    const double* __restrict__ gx = g;
    const double* __restrict__ gy = g + g_size;
    const double* __restrict__ gz = g + g_size * 2;

    const bool diag = (ish == jsh);

    for (int j = j0; j < j1; j++) {
        for (int i = i0; i < i1; i++) {
            const int loc_j = c_l_locs[j_l] + (j-j0);
            const int loc_i = c_l_locs[i_l] + (i-i0);

            int ix = idx[loc_i] + idx[loc_j] * (i_l + 1);
            int iy = idy[loc_i] + idy[loc_j] * (i_l + 1);
            int iz = idz[loc_i] + idz[loc_j] * (i_l + 1);

            ix = ix * NROOTS;
            iy = iy * NROOTS;
            iz = iz * NROOTS;

            double eri = 0;
#pragma unroll
            for (int i_root = 0; i_root < NROOTS; i_root++) {
                eri += gx[ix + i_root] * gy[iy + i_root] * gz[iz + i_root];
            }
            // gv[i] += I_{i,j} * fg[j]
            atomicAdd(gv_cart + (i * ngrids + i_grid), eri * fg_cart[j * ngrids + i_grid]);
            // transpose contribution for off-diagonal shell pairs
            if (!diag) {
                atomicAdd(gv_cart + (j * ngrids + i_grid), eri * fg_cart[i * ngrids + i_grid]);
            }
        }
    }
}

template <int NROOTS, int GSIZE_INT3C_1E>
__global__
static void GINTsgx_fused_k_gv_kernel_general(double* gv_cart, const double* fg_cart,
                                              const BasisProdOffsets offsets, const int i_l, const int j_l,
                                              const int nprim_ij, const int ngrids,
                                              const double omega, const double* grid_points,
                                              const double* charge_exponents, const double screen_tol)
{
    const int ntasks_ij = offsets.ntasks_ij;
    const int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
    const int task_grid = blockIdx.y * blockDim.y + threadIdx.y;

    if (task_ij >= ntasks_ij || task_grid >= ngrids) {
        return;
    }
    const int bas_ij = offsets.bas_ij + task_ij;
    const int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
    const int* bas_pair2bra = c_bpcache.bas_pair2bra;
    const int* bas_pair2ket = c_bpcache.bas_pair2ket;
    const int ish = bas_pair2bra[bas_ij];
    const int jsh = bas_pair2ket[bas_ij];

    const double* grid_point = grid_points + task_grid * 3;
    const double charge_exponent = (charge_exponents != NULL) ? charge_exponents[task_grid] : 0.0;

    // Per-primitive-pair data as read by GINT_g1e (see g1e.cu ~L27-54).
    const double* __restrict__ a12 = c_bpcache.a12;
    const double* __restrict__ e12 = c_bpcache.e12;
    const double* __restrict__ x12 = c_bpcache.x12;
    const double* __restrict__ y12 = c_bpcache.y12;
    const double* __restrict__ z12 = c_bpcache.z12;
    const double Cx = grid_point[0];
    const double Cy = grid_point[1];
    const double Cz = grid_point[2];
    const bool do_screen = (screen_tol > 0.0);

    // Density-matrix-weighted (P-junction) screening factor. The contribution
    // of this shell pair at this grid point is I_vt * fg_cart[t,g] (and its
    // transpose I_vt * fg_cart[i,g]), so the max |fg| over both cartesian AO
    // ranges is an upper bound on the density weight. Computed once per
    // (task_ij, task_grid) since it is independent of the primitive pair.
    double dm_factor = 1.0;
    if (do_screen) {
        const int* ao_loc = c_bpcache.ao_loc;
        const int i0 = ao_loc[ish];
        const int i1 = ao_loc[ish+1];
        const int j0 = ao_loc[jsh];
        const int j1 = ao_loc[jsh+1];
        double f_max = 0.0;
        for (int i = i0; i < i1; i++) {
            f_max = fmax(f_max, fabs(fg_cart[i * ngrids + task_grid]));
        }
        for (int j = j0; j < j1; j++) {
            f_max = fmax(f_max, fabs(fg_cart[j * ngrids + task_grid]));
        }
        dm_factor = f_max;
    }

    double g[GSIZE_INT3C_1E];

    for (int ij = prim_ij; ij < prim_ij+nprim_ij; ++ij) {
        if (do_screen) {
            const double aij = a12[ij];
            const double eij = e12[ij];
            const double PCx = x12[ij] - Cx;
            const double PCy = y12[ij] - Cy;
            const double PCz = z12[ij] - Cz;
            double a0 = aij;
            const double q_over_p_plus_q = charge_exponent > 0.0 ? charge_exponent / (aij + charge_exponent) : 1.0;
            const double sqrt_q_over_p_plus_q = charge_exponent > 0.0 ? sqrt(q_over_p_plus_q) : 1.0;
            a0 *= q_over_p_plus_q;
            const double theta = omega > 0.0 ? omega * omega / (omega * omega + a0) : 1.0;
            const double sqrt_theta = omega > 0.0 ? sqrt(theta) : 1.0;
            a0 *= theta;
            // |prefactor| bounds the primitive's radial magnitude (Boys F0<=1).
            double bound = fabs(2.0 * M_PI / aij * eij * sqrt_theta * sqrt_q_over_p_plus_q);
            const double boys_input = a0 * (PCx * PCx + PCy * PCy + PCz * PCz);
            // Apply the grid-distance decay factor F0(x) <= SQRTPIE4/sqrt(x) for
            // ALL angular momenta. This is exact for s; for l>0 the actual
            // integral is well below |prefactor|*F0 (the angular PA/PB factors
            // stay below the F0 decay for realistic exponents/geometries), so
            // it remains a safe upper bound while screening far more p/d/f work.
            // Correctness is guarded by the screen accuracy tests.
            if (boys_input > 1e-14) {
                bound *= SQRTPIE4 / sqrt(boys_input);
            }
            if (bound * dm_factor < screen_tol) {
                continue;
            }
        }
        GINT_g1e<NROOTS>(g, grid_point, ish, jsh, ij, i_l, j_l, charge_exponent, omega);
        GINTcontract_gv_int3c1e<NROOTS>(g, gv_cart, fg_cart, ish, jsh, task_grid, i_l, j_l, ngrids);
    }
}

static int GINTsgx_fused_k_gv_tasks(double* gv_cart, const double* fg_cart, const BasisProdOffsets offsets,
                                    const int i_l, const int j_l, const int nprim_ij, const int ngrids,
                                    const double omega, const double* grid_points,
                                    const double* charge_exponents, const double screen_tol, const cudaStream_t stream)
{
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int ntasks_ij = offsets.ntasks_ij;

    const dim3 threads(THREADSX, THREADSY);
    const dim3 blocks((ntasks_ij+THREADSX-1)/THREADSX, (ngrids+THREADSY-1)/THREADSY);
    switch (nrys_roots) {
    case 1: GINTsgx_fused_k_gv_kernel_general<1, GSIZE1_INT3C_1E> <<<blocks, threads, 0, stream>>>(gv_cart, fg_cart, offsets, i_l, j_l, nprim_ij, ngrids, omega, grid_points, charge_exponents, screen_tol); break;
    case 2: GINTsgx_fused_k_gv_kernel_general<2, GSIZE2_INT3C_1E> <<<blocks, threads, 0, stream>>>(gv_cart, fg_cart, offsets, i_l, j_l, nprim_ij, ngrids, omega, grid_points, charge_exponents, screen_tol); break;
    case 3: GINTsgx_fused_k_gv_kernel_general<3, GSIZE3_INT3C_1E> <<<blocks, threads, 0, stream>>>(gv_cart, fg_cart, offsets, i_l, j_l, nprim_ij, ngrids, omega, grid_points, charge_exponents, screen_tol); break;
    case 4: GINTsgx_fused_k_gv_kernel_general<4, GSIZE4_INT3C_1E> <<<blocks, threads, 0, stream>>>(gv_cart, fg_cart, offsets, i_l, j_l, nprim_ij, ngrids, omega, grid_points, charge_exponents, screen_tol); break;
    case 5: GINTsgx_fused_k_gv_kernel_general<5, GSIZE5_INT3C_1E> <<<blocks, threads, 0, stream>>>(gv_cart, fg_cart, offsets, i_l, j_l, nprim_ij, ngrids, omega, grid_points, charge_exponents, screen_tol); break;
    default:
        fprintf(stderr, "GINTsgx_fused_k_gv: rys roots %d out of range\n", nrys_roots);
        return 1;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

extern "C" {
int GINTsgx_fused_k_s(const cudaStream_t stream, double* K, const double* ao, const double* dm,
                      const double* weights, const double* grid_points, const int* keep,
                      const double* sh_center, const int* sh_nprim, const int* sh_prim_off,
                      const double* prim_exp, const double* prim_coef,
                      const int nao, const int ngrids, const double omega)
{
    if (nao > SGX_FUSED_MAX_NAO) {
        fprintf(stderr, "GINTsgx_fused_k_s: nao=%d exceeds SGX_FUSED_MAX_NAO=%d\n", nao, SGX_FUSED_MAX_NAO);
        return 1;
    }
    const int threads = 128;
    const int blocks = (ngrids + threads - 1) / threads;
    GINTsgx_fused_k_s_kernel<<<blocks, threads, 0, stream>>>(
        K, ao, dm, weights, grid_points, keep, sh_center, sh_nprim, sh_prim_off,
        prim_exp, prim_coef, nao, ngrids, omega);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", __func__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int GINTsgx_fused_k_gv(const cudaStream_t stream, const BasisProdCache* bpcache,
                       const double* grid_points, const double* charge_exponents, const int ngrids,
                       double* gv_cart, const double* fg_cart,
                       const int* bins_locs_ij, int nbins,
                       const int cp_ij_id, const double omega, const double screen_tol)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTsgx_fused_k_gv_tasks(gv_cart, fg_cart, offsets, i_l, j_l, nprim_ij, ngrids,
                                                 omega, grid_points, charge_exponents, screen_tol, stream);
        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e(const cudaStream_t stream, const BasisProdCache* bpcache,
                     const double* grid_points, const double* charge_exponents, const int ngrids,
                     double* integrals,
                     const int* strides, const int* ao_offsets,
                     const int* bins_locs_ij, int nbins,
                     const int cp_ij_id, const double omega)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c1e_tasks(integrals, offsets, i_l, j_l, nprim_ij,
                                               strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                               omega, grid_points, charge_exponents, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e_charge_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                       const double* grid_points, const double* charge_exponents, const int ngrids,
                                       double* integral_charge_contracted,
                                       const int* strides, const int* ao_offsets,
                                       const int* bins_locs_ij, int nbins,
                                       const int cp_ij_id, const double omega, const int n_charge_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        const int err = GINTfill_int3c1e_charge_contracted_tasks(integral_charge_contracted, offsets, i_l, j_l, nprim_ij,
                                                                 strides[0], strides[1], ao_offsets[0], ao_offsets[1],
                                                                 omega, grid_points, charge_exponents, n_charge_sum_per_thread, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}

int GINTfill_int3c1e_density_contracted(const cudaStream_t stream, const BasisProdCache* bpcache,
                                        const double* grid_points, const double* charge_exponents, const int ngrids,
                                        const double* dm_pair_ordered, const int* density_offset,
                                        double* integral_density_contracted,
                                        const int* bins_locs_ij, int nbins,
                                        const int cp_ij_id, const double omega, const int n_pair_sum_per_thread)
{
    const ContractionProdType *cp_ij = bpcache->cptype + cp_ij_id;
    const int i_l = cp_ij->l_bra;
    const int j_l = cp_ij->l_ket;
    const int nrys_roots = (i_l + j_l) / 2 + 1;
    const int nprim_ij = cp_ij->nprim_12;

    if (nrys_roots > MAX_NROOTS_INT3C_1E) {
        fprintf(stderr, "nrys_roots = %d too high\n", nrys_roots);
        return 2;
    }

    checkCudaErrors(cudaMemcpyToSymbol(c_bpcache, bpcache, sizeof(BasisProdCache)));

    const int* bas_pairs_locs = bpcache->bas_pairs_locs;
    const int* primitive_pairs_locs = bpcache->primitive_pairs_locs;
    for (int ij_bin = 0; ij_bin < nbins; ij_bin++) {
        const int bas_ij0 = bins_locs_ij[ij_bin];
        const int bas_ij1 = bins_locs_ij[ij_bin + 1];
        const int ntasks_ij = bas_ij1 - bas_ij0;
        if (ntasks_ij <= 0) {
            continue;
        }

        BasisProdOffsets offsets;
        offsets.ntasks_ij = ntasks_ij;
        offsets.ntasks_kl = ngrids;
        offsets.bas_ij = bas_pairs_locs[cp_ij_id] + bas_ij0;
        offsets.bas_kl = -1;
        offsets.primitive_ij = primitive_pairs_locs[cp_ij_id] + bas_ij0 * nprim_ij;
        offsets.primitive_kl = -1;

        HermiteDensityOffsets hermite_density_offsets;
        hermite_density_offsets.density_offset_of_angular_pair = density_offset[cp_ij_id];
        hermite_density_offsets.pair_offset_of_angular_pair = bas_pairs_locs[cp_ij_id];
        hermite_density_offsets.n_pair_of_angular_pair = bas_pairs_locs[cp_ij_id + 1] - bas_pairs_locs[cp_ij_id];

        const int err = GINTfill_int3c1e_density_contracted_tasks(integral_density_contracted, dm_pair_ordered, hermite_density_offsets,
                                                                  offsets, i_l, j_l, nprim_ij,
                                                                  omega, grid_points, charge_exponents, n_pair_sum_per_thread, stream);

        if (err != 0) {
            return err;
        }
    }

    return 0;
}
}
