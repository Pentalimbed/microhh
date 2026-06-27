/*
 * MicroHH
 * Copyright (c) 2011-2024 Chiel van Heerwaarden
 * Copyright (c) 2011-2024 Thijs Heus
 * Copyright (c) 2014-2024 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fftw3.h>
#include <cufft.h>
#include <iostream>
#include "master.h"
#include "grid.h"
#include "fields.h"
#include "pres.h"
#include "pres_2.h"
#include "defines.h"
#include "tools.h"
#include "constants.h"
#include "field3d_operators.h"
#include "stats.h"

// Kernel/CUDA launcher:
#include "pres_2_kernels.cuh"
#include "cuda_launcher.h"
#include "cuda_tiling.h"

namespace
{
    //template<typename TF> __global__
    //void pres_in_g(TF* __restrict__ p,
    //               TF* __restrict__ u ,  TF* __restrict__ v ,     TF* __restrict__ w ,
    //               TF* __restrict__ ut,  TF* __restrict__ vt,     TF* __restrict__ wt,
    //               const TF* __restrict__ dzi, TF* __restrict__ rhoref, TF* __restrict__ rhorefh,
    //               TF dxi, TF dyi, TF dti,
    //               const int jj, const int kk,
    //               const int jjp, const int kkp,
    //               const int imax, const int jmax, const int kmax,
    //               const int igc, const int jgc, const int kgc)
    //{
    //    const int ii = 1;
    //    const int i  = blockIdx.x*blockDim.x + threadIdx.x;
    //    const int j  = blockIdx.y*blockDim.y + threadIdx.y;
    //    const int k  = blockIdx.z;

    //    if (i < imax && j < jmax && k < kmax)
    //    {
    //        const int ijkp = i + j*jjp + k*kkp;
    //        const int ijk  = i+igc + (j+jgc)*jj + (k+kgc)*kk;

    //        p[ijkp] = rhoref [k+kgc]   * ( (ut[ijk+ii] + u[ijk+ii] * dti) - (ut[ijk] + u[ijk] * dti) ) * dxi
    //                + rhoref [k+kgc]   * ( (vt[ijk+jj] + v[ijk+jj] * dti) - (vt[ijk] + v[ijk] * dti) ) * dyi
    //              + ( rhorefh[k+kgc+1] * (  wt[ijk+kk] + w[ijk+kk] * dti)
    //                - rhorefh[k+kgc  ] * (  wt[ijk   ] + w[ijk   ] * dti) ) * dzi[k+kgc];
    //    }
    //}

    //template<typename TF> __global__
    //void pres_out_g(TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
    //                TF* __restrict__ p,
    //                const TF* __restrict__ dzhi, const TF dxi, const TF dyi,
    //                const int jj, const int kk,
    //                const int istart, const int jstart, const int kstart,
    //                const int iend, const int jend, const int kend)
    //{
    //    const int i  = blockIdx.x*blockDim.x + threadIdx.x + istart;
    //    const int j  = blockIdx.y*blockDim.y + threadIdx.y + jstart;
    //    const int k  = blockIdx.z + kstart;
    //    const int ii = 1;

    //    if (i < iend && j < jend && k < kend)
    //    {
    //        const int ijk = i + j*jj + k*kk;
    //        ut[ijk] -= (p[ijk] - p[ijk-ii]) * dxi;
    //        vt[ijk] -= (p[ijk] - p[ijk-jj]) * dyi;
    //        wt[ijk] -= (p[ijk] - p[ijk-kk]) * dzhi[k];
    //    }
    //}

    //template<typename TF> __global__
    //void solve_out_g(TF* __restrict__ p, TF* __restrict__ work3d,
    //                 const int jj, const int kk,
    //                 const int jjp, const int kkp,
    //                 const int istart, const int jstart, const int kstart,
    //                 const int imax, const int jmax, const int kmax)
    //{
    //    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    //    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    //    const int k = blockIdx.z;

    //    if (i < imax && j < jmax && k < kmax)
    //    {
    //        const int ijk  = i + j*jj + k*kk;
    //        const int ijkp = i+istart + (j+jstart)*jjp + (k+kstart)*kkp;

    //        p[ijkp] = work3d[ijk];

    //        if (k == 0)
    //            p[ijkp-kkp] = p[ijkp];
    //    }
    //}

    //template<typename TF> __global__
    //void solve_in_g(TF* __restrict__ p,
    //                TF* __restrict__ work3d, TF* __restrict__ b,
    //                TF* __restrict__ a, TF* __restrict__ c,
    //                const TF* __restrict__ dz, const TF* __restrict__ rhoref,
    //                TF* __restrict__ bmati, TF* __restrict__ bmatj,
    //                const int jj, const int kk,
    //                const int imax, const int jmax, const int kmax,
    //                const int kstart)
    //{
    //    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    //    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    //    const int k = blockIdx.z;

    //    if (i < imax && j < jmax && k < kmax)
    //    {
    //        const int ijk = i + j*jj + k*kk;

    //        // CvH this needs to be taken into account in case of an MPI run
    //        // iindex = mpi->mpicoordy * iblock + i;
    //        // jindex = mpi->mpicoordx * jblock + j;
    //        // b[ijk] = dz[k+kgc]*dz[k+kgc] * (bmati[iindex]+bmatj[jindex]) - (a[k]+c[k]);
    //        //  if(iindex == 0 && jindex == 0)

    //        b[ijk] = dz[k+kstart]*dz[k+kstart] * rhoref[k+kstart]*(bmati[i]+bmatj[j]) - (a[k]+c[k]);
    //        p[ijk] = dz[k+kstart]*dz[k+kstart] * p[ijk];

    //        if (k == 0)
    //        {
    //            // substitute BC's
    //            // ijk = i + j*jj;
    //            b[ijk] += a[0];
    //        }
    //        else if (k == kmax-1)
    //        {
    //            // for wave number 0, which contains average, set pressure at top to zero
    //            if (i == 0 && j == 0)
    //                b[ijk] -= c[k];
    //            // set dp/dz at top to zero
    //            else
    //                b[ijk] += c[k];
    //        }
    //    }
    //}

    //template<typename TF> __global__
    //void tdma_g(TF* __restrict__ a, TF* __restrict__ b, TF* __restrict__ c,
    //            TF* __restrict__ p, TF* __restrict__ work3d,
    //            const int jj, const int kk,
    //            const int imax, const int jmax, const int kmax)
    //{
    //    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    //    const int j = blockIdx.y*blockDim.y + threadIdx.y;

    //    if (i < imax && j < jmax)
    //    {
    //        const int ij = i + j*jj;

    //        TF work2d = b[ij];
    //        p[ij] /= work2d;

    //        for (int k=1; k<kmax; k++)
    //        {
    //            const int ijk = ij + k*kk;
    //            work3d[ijk] = c[k-1] / work2d;
    //            work2d = b[ijk] - a[k]*work3d[ijk];
    //            p[ijk] -= a[k]*p[ijk-kk];
    //            p[ijk] /= work2d;
    //        }

    //        for (int k=kmax-2; k>=0; k--)
    //        {
    //            const int ijk = ij + k*kk;
    //            p[ijk] -= work3d[ijk+kk]*p[ijk+kk];
    //        }
    //    }
    //}

    template<typename TF> __global__
    void calc_divergence_g(TF* __restrict__ u, TF* __restrict__ v, TF* __restrict__ w,
                           TF* __restrict__ div, const TF* __restrict__ dzi,
                           const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
                           TF dxi, TF dyi,
                           int jj, int kk, int istart, int jstart, int kstart,
                           int iend, int jend, int kend)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        const int k = blockIdx.z + kstart;
        const int ii = 1;

        if (i < iend && j < jend && k < kend)
        {
            const int ijk = i + j*jj + k*kk;
            div[ijk] = rhoref[k]*((u[ijk+ii]-u[ijk])*dxi + (v[ijk+jj]-v[ijk])*dyi)
                    + (rhorefh[k+1]*w[ijk+kk]-rhorefh[k]*w[ijk])*dzi[k];
        }
    }

    // ----------------------------------------------------------------------
    // Open-BC GPU kernels.
    // ----------------------------------------------------------------------

    // DCT along the i-direction: out[ki,j,k] = sum_i mat[ki,i] * in[i,j,k].
    // Operates on the no-ghost array of size (itot, jtot, kmax), i fastest.
    template<typename TF> __global__
    void dct_i_g(
            TF* const __restrict__ out, const TF* const __restrict__ in,
            const TF* const __restrict__ mat,
            const int itot, const int jtot, const int kmax)
    {
        const int ki = blockIdx.x*blockDim.x + threadIdx.x;
        const int j  = blockIdx.y*blockDim.y + threadIdx.y;
        const int k  = blockIdx.z;

        if (ki < itot && j < jtot && k < kmax)
        {
            const int row = ki*itot;
            const int base = j*itot + k*itot*jtot;
            TF sum = TF(0);
            for (int i=0; i<itot; ++i)
                sum += mat[row + i] * in[base + i];
            out[base + ki] = sum;
        }
    }

    // DCT along the j-direction: out[i,kj,k] = sum_j mat[kj,j] * in[i,j,k].
    template<typename TF> __global__
    void dct_j_g(
            TF* const __restrict__ out, const TF* const __restrict__ in,
            const TF* const __restrict__ mat,
            const int itot, const int jtot, const int kmax)
    {
        const int i  = blockIdx.x*blockDim.x + threadIdx.x;
        const int kj = blockIdx.y*blockDim.y + threadIdx.y;
        const int k  = blockIdx.z;

        if (i < itot && kj < jtot && k < kmax)
        {
            const int row = kj*jtot;
            const int base = i + k*itot*jtot;
            TF sum = TF(0);
            for (int j=0; j<jtot; ++j)
                sum += mat[row + j] * in[base + j*itot];
            out[base + kj*itot] = sum;
        }
    }

    // Zero the normal-velocity tendency at the lateral boundaries. Single rank
    // (USEMPI disabled in CUDA builds), so all four boundaries live on this rank.
    template<typename TF> __global__
    void zero_boundary_tend_g(
            TF* const __restrict__ ut, TF* const __restrict__ vt,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jj, const int kk)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        const int k = blockIdx.z + kstart;

        if (i < iend && j < jend && k < kend)
        {
            if (i == istart)   ut[istart + j*jj + k*kk] = TF(0);
            if (i == iend-1)   ut[iend   + j*jj + k*kk] = TF(0);
            if (j == jstart)   vt[i + jstart*jj + k*kk] = TF(0);
            if (j == jend-1)   vt[i + jend  *jj + k*kk] = TF(0);
        }
    }

    // Enforce Neumann (zero-gradient) pressure ghost cells at the lateral
    // boundaries, matching the CPU open-BC path.
    template<typename TF> __global__
    void pres_neumann_lateral_g(
            TF* const __restrict__ p,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int jjp, const int kkp)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        const int k = blockIdx.z + kstart;

        if (i < iend && j < jend && k < kend)
        {
            if (i == istart)   p[(istart-1) + j*jjp + k*kkp] = p[istart + j*jjp + k*kkp];
            if (i == iend-1)   p[iend       + j*jjp + k*kkp] = p[(iend-1) + j*jjp + k*kkp];
            if (j == jstart)   p[i + (jstart-1)*jjp + k*kkp] = p[i + jstart*jjp + k*kkp];
            if (j == jend-1)   p[i + jend      *jjp + k*kkp] = p[i + (jend-1)*jjp + k*kkp];
        }
    }
} // End namespace.

#ifdef USECUDA
template<typename TF>
void Pres_2<TF>::prepare_device()
{
    auto& gd = grid.get_grid_data();

    const int kmemsize = gd.kmax*sizeof(TF);
    const int imemsize = gd.itot*sizeof(TF);
    const int jmemsize = gd.jtot*sizeof(TF);

    const int ijmemsize = gd.imax*gd.jmax*sizeof(TF);

    bmati_g.allocate(gd.itot);
    bmatj_g.allocate(gd.jtot);
    a_g.allocate(gd.kmax);
    c_g.allocate(gd.kmax);
    work2d_g.allocate(gd.imax*gd.jmax);

    cuda_safe_call(cudaMemcpy(bmati_g,  bmati.data(),  imemsize,  cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy(bmatj_g,  bmatj.data(),  jmemsize,  cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy(a_g,      a.data(),      kmemsize,  cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy(c_g,      c.data(),      kmemsize,  cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy(work2d_g, work2d.data(), ijmemsize, cudaMemcpyHostToDevice));

    if (sw_openbc)
    {
        // Build the DCT matrices (REDFT10 forward / REDFT01 backward), matching
        // the host FFTW path. The backward matrices fold in the 1/(2N)
        // normalization that the CPU applies after the inverse transform.
        const TF pi = std::acos(TF(-1));

        auto build = [&](
                cuda_vector<TF>& fwd_g, cuda_vector<TF>& bwd_g, const int N)
        {
            std::vector<TF> fwd(N*N);
            std::vector<TF> bwd(N*N);
            const TF norm = TF(1) / (TF(2)*N);

            for (int k=0; k<N; ++k)
                for (int n=0; n<N; ++n)
                {
                    // REDFT10 (DCT-II): F[k,n] = 2 cos(pi k (2n+1) / (2N)).
                    fwd[k*N + n] = TF(2) * std::cos(pi*k*(2*n+1) / (TF(2)*N));
                    // REDFT01 (DCT-III): B[k,n] = 1 (n==0) else 2 cos(pi n (2k+1)/(2N)).
                    bwd[k*N + n] = (n == 0)
                            ? norm
                            : norm * TF(2) * std::cos(pi*n*(2*k+1) / (TF(2)*N));
                }

            fwd_g.allocate(N*N);
            bwd_g.allocate(N*N);
            cuda_safe_call(cudaMemcpy(fwd_g, fwd.data(), N*N*sizeof(TF), cudaMemcpyHostToDevice));
            cuda_safe_call(cudaMemcpy(bwd_g, bwd.data(), N*N*sizeof(TF), cudaMemcpyHostToDevice));
        };

        build(dctfi_g, dctbi_g, gd.itot);
        build(dctfj_g, dctbj_g, gd.jtot);
    }
    else
        make_cufft_plan();
}

template<typename TF>
void Pres_2<TF>::dct_forward(TF* const p, TF* const tmp)
{
    auto& gd = grid.get_grid_data();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;

    // i-direction DCT: in `p` -> out `tmp`.
    {
        dim3 grid((gd.itot+blocki-1)/blocki, (gd.jtot+blockj-1)/blockj, gd.kmax);
        dim3 block(blocki, blockj, 1);
        dct_i_g<TF><<<grid, block>>>(tmp, p, dctfi_g, gd.itot, gd.jtot, gd.kmax);
        cuda_check_error();
    }

    if (gd.jtot > 1)
    {
        // j-direction DCT: in `tmp` -> out `p`.
        dim3 grid((gd.itot+blocki-1)/blocki, (gd.jtot+blockj-1)/blockj, gd.kmax);
        dim3 block(blocki, blockj, 1);
        dct_j_g<TF><<<grid, block>>>(p, tmp, dctfj_g, gd.itot, gd.jtot, gd.kmax);
        cuda_check_error();
    }
    else
        cuda_safe_call(cudaMemcpy(p, tmp, gd.itot*gd.jtot*gd.kmax*sizeof(TF), cudaMemcpyDeviceToDevice));
}

template<typename TF>
void Pres_2<TF>::dct_backward(TF* const p, TF* const tmp)
{
    auto& gd = grid.get_grid_data();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;

    if (gd.jtot > 1)
    {
        // j-direction inverse DCT: in `p` -> out `tmp`.
        dim3 grid((gd.itot+blocki-1)/blocki, (gd.jtot+blockj-1)/blockj, gd.kmax);
        dim3 block(blocki, blockj, 1);
        dct_j_g<TF><<<grid, block>>>(tmp, p, dctbj_g, gd.itot, gd.jtot, gd.kmax);
        cuda_check_error();
    }
    else
        cuda_safe_call(cudaMemcpy(tmp, p, gd.itot*gd.jtot*gd.kmax*sizeof(TF), cudaMemcpyDeviceToDevice));

    // i-direction inverse DCT: in `tmp` -> out `p`.
    {
        dim3 grid((gd.itot+blocki-1)/blocki, (gd.jtot+blockj-1)/blockj, gd.kmax);
        dim3 block(blocki, blockj, 1);
        dct_i_g<TF><<<grid, block>>>(p, tmp, dctbi_g, gd.itot, gd.jtot, gd.kmax);
        cuda_check_error();
    }
}

template<typename TF>
void Pres_2<TF>::clear_device()
{
}

template<typename TF>
void Pres_2<TF>::exec(double dt, Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    // Open boundary conditions use a discrete cosine transform (Neumann) for the
    // pressure solve, which cuFFT has no native support for. We implement the DCT
    // (FFTW REDFT10 forward / REDFT01 backward) as a batched matrix multiply on
    // the device, and run the full solve on the GPU.
    if (sw_openbc)
    {
        const int blocki = gd.ithread_block;
        const int blockj = gd.jthread_block;
        const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
        const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);
        const TF dti = TF(1.)/dt;

        dim3 gridGPU (gridi,  gridj,  gd.kmax);
        dim3 blockGPU(blocki, blockj, 1);

        Grid_layout grid_layout_nogc = {
                0, gd.imax, 0, gd.jmax, 0, gd.kmax,
                1, gd.imax, gd.imax*gd.jmax};
        Grid_layout grid_layout_2d_nogc = {
                0, gd.imax, 0, gd.jmax, 0, 1,
                1, gd.imax, gd.imax*gd.jmax};
        Grid_layout grid_layout_int = {
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.istride, gd.jstride, gd.kstride};

        auto tmp1 = fields.get_tmp_g();
        auto tmp2 = fields.get_tmp_g();

        // Cyclic BCs on the velocity tendencies, then zero the normal-velocity
        // tendency at the lateral boundaries (open-BC specific).
        boundary_cyclic.exec_g(fields.mt.at("u")->fld_g);
        boundary_cyclic.exec_g(fields.mt.at("v")->fld_g);
        boundary_cyclic.exec_g(fields.mt.at("w")->fld_g);

        zero_boundary_tend_g<TF><<<gridGPU, blockGPU>>>(
                fields.mt.at("u")->fld_g, fields.mt.at("v")->fld_g,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells);
        cuda_check_error();

        // Build the RHS of the Poisson equation (no-ghost array in `p`).
        launch_grid_kernel<Pres_2_kernels::pres_in_g<TF>>(
                grid_layout_nogc,
                fields.sd.at("p")->fld_g.view(),
                fields.mp.at("u")->fld_g, fields.mp.at("v")->fld_g, fields.mp.at("w")->fld_g,
                fields.mt.at("u")->fld_g, fields.mt.at("v")->fld_g, fields.mt.at("w")->fld_g,
                gd.dzi_g, fields.rhoref_g, fields.rhorefh_g,
                gd.dxi, gd.dyi, TF(dti),
                gd.icells, gd.ijcells,
                gd.istart, gd.jstart, gd.kstart);

        // Forward DCT (REDFT10) in x and y.
        dct_forward(fields.sd.at("p")->fld_g, tmp1->fld_g);

        // Modified wavenumbers + boundary substitution.
        launch_grid_kernel<Pres_2_kernels::solve_in_g<TF>>(
                grid_layout_nogc,
                fields.sd.at("p")->fld_g.view(),
                tmp1->fld_g,
                tmp2->fld_g.view(),
                a_g, c_g, gd.dz_g,
                fields.rhoref_g,
                bmati_g, bmatj_g,
                gd.kstart, gd.kmax);

        // Tridiagonal solve in the vertical.
        launch_grid_kernel<Pres_2_kernels::tdma_g<TF>>(
                grid_layout_2d_nogc,
                a_g, tmp2->fld_g, c_g,
                fields.sd.at("p")->fld_g.view(),
                tmp1->fld_g.view(),
                gd.kmax);

        // Backward DCT (REDFT01, normalized) in y and x.
        dct_backward(fields.sd.at("p")->fld_g, tmp1->fld_g);

        // Put pressure back onto the ghosted grid + bottom zero-gradient BC.
        cuda_safe_call(cudaMemcpy(tmp1->fld_g, fields.sd.at("p")->fld_g, gd.ncells*sizeof(TF), cudaMemcpyDeviceToDevice));

        launch_grid_kernel<Pres_2_kernels::solve_out_g<TF>>(
                grid_layout_nogc,
                fields.sd.at("p")->fld_g.view(),
                tmp1->fld_g,
                gd.istart, gd.jstart, gd.kstart,
                gd.icells, gd.ijcells);

        // Cyclic ghost cells, then enforce lateral Neumann pressure ghost cells.
        boundary_cyclic.exec_g(fields.sd.at("p")->fld_g);

        pres_neumann_lateral_g<TF><<<gridGPU, blockGPU>>>(
                fields.sd.at("p")->fld_g,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells);
        cuda_check_error();

        // Subtract the pressure gradient from the velocity tendencies.
        launch_grid_kernel<Pres_2_kernels::pres_out_g<TF>>(
                grid_layout_int,
                fields.mt.at("u")->fld_g.view(),
                fields.mt.at("v")->fld_g.view(),
                fields.mt.at("w")->fld_g.view(),
                fields.sd.at("p")->fld_g,
                gd.dzhi_g, TF(1.)/gd.dx, TF(1.)/gd.dy);

        fields.release_tmp_g(tmp1);
        fields.release_tmp_g(tmp2);

        cudaDeviceSynchronize();
        stats.calc_tend(*fields.mt.at("u"), tend_name);
        stats.calc_tend(*fields.mt.at("v"), tend_name);
        stats.calc_tend(*fields.mt.at("w"), tend_name);
        return;
    }

    // Grid layout for KL/CL launches over interior, including ghost cells.
    Grid_layout grid_layout_int = {
            gd.istart, gd.iend,
            gd.jstart, gd.jend,
            gd.kstart, gd.kend,
            gd.istride,
            gd.jstride,
            gd.kstride};

    // Grid layout for KL/CL launches over interior, excluding ghost cells.
    Grid_layout grid_layout_nogc = {
            0, gd.imax,
            0, gd.jmax,
            0, gd.kmax,
            1,
            gd.imax,
            gd.imax*gd.jmax};

    // Grid layout for KL/CL launches over 2D, excluding ghost cells.
    Grid_layout grid_layout_2d_nogc = {
            0, gd.imax,
            0, gd.jmax,
            0, 1,
            1,
            gd.imax,
            gd.imax*gd.jmax};

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;
    const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
    const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);
    const TF dti = TF(1.)/dt;

    // 3D grid
    dim3 gridGPU (gridi,  gridj,  gd.kmax);
    dim3 blockGPU(blocki, blockj, 1);

    // 2D grid
    dim3 grid2dGPU (gridi,  gridj);
    dim3 block2dGPU(blocki, blockj);

    // Get two free tmp fields on gpu
    auto tmp1 = fields.get_tmp_g();
    auto tmp2 = fields.get_tmp_g();

    // calculate the cyclic BCs first
    boundary_cyclic.exec_g(fields.mt.at("u")->fld_g);
    boundary_cyclic.exec_g(fields.mt.at("v")->fld_g);
    boundary_cyclic.exec_g(fields.mt.at("w")->fld_g);

    launch_grid_kernel<Pres_2_kernels::pres_in_g<TF>>(
            grid_layout_nogc,
            fields.sd.at("p")->fld_g.view(),
            fields.mp.at("u")->fld_g,
            fields.mp.at("v")->fld_g,
            fields.mp.at("w")->fld_g,
            fields.mt.at("u")->fld_g,
            fields.mt.at("v")->fld_g,
            fields.mt.at("w")->fld_g,
            gd.dzi_g,
            fields.rhoref_g,
            fields.rhorefh_g,
            gd.dxi, gd.dyi,
            TF(dti),
            gd.icells, gd.ijcells,
            gd.istart, gd.jstart, gd.kstart);

    fft_forward(fields.sd.at("p")->fld_g, tmp1->fld_g, tmp2->fld_g);

    launch_grid_kernel<Pres_2_kernels::solve_in_g<TF>>(
            grid_layout_nogc,
            fields.sd.at("p")->fld_g.view(),
            tmp1->fld_g,
            tmp2->fld_g.view(),
            a_g, c_g, gd.dz_g,
            fields.rhoref_g,
            bmati_g, bmatj_g,
            gd.kstart, gd.kmax);

    // DOES NOT WORK (YET):
    launch_grid_kernel<Pres_2_kernels::tdma_g<TF>>(
            grid_layout_2d_nogc,
            a_g,
            tmp2->fld_g,
            c_g,
            fields.sd.at("p")->fld_g.view(),
            tmp1->fld_g.view(),
            gd.kmax);

    fft_backward(fields.sd.at("p")->fld_g, tmp1->fld_g, tmp2->fld_g);

    cuda_safe_call(cudaMemcpy(tmp1->fld_g, fields.sd.at("p")->fld_g, gd.ncells*sizeof(TF), cudaMemcpyDeviceToDevice));

    launch_grid_kernel<Pres_2_kernels::solve_out_g<TF>>(
            grid_layout_nogc,
            fields.sd.at("p")->fld_g.view(),
            tmp1->fld_g,
            gd.istart, gd.jstart, gd.kstart,
            gd.icells, gd.ijcells);

    boundary_cyclic.exec_g(fields.sd.at("p")->fld_g);

    launch_grid_kernel<Pres_2_kernels::pres_out_g<TF>>(
            grid_layout_int,
            fields.mt.at("u")->fld_g.view(),
            fields.mt.at("v")->fld_g.view(),
            fields.mt.at("w")->fld_g.view(),
            fields.sd.at("p")->fld_g,
            gd.dzhi_g, TF(1.)/gd.dx, TF(1.)/gd.dy);

    fields.release_tmp_g(tmp1);
    fields.release_tmp_g(tmp2);
    
    cudaDeviceSynchronize();
    stats.calc_tend(*fields.mt.at("u"), tend_name);
    stats.calc_tend(*fields.mt.at("v"), tend_name);
    stats.calc_tend(*fields.mt.at("w"), tend_name);
}

template<typename TF>
TF Pres_2<TF>::check_divergence()
{
    auto& gd = grid.get_grid_data();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;
    const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
    const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);

    dim3 gridGPU (gridi, gridj, gd.kcells);
    dim3 blockGPU(blocki, blockj, 1);

    auto divergence = fields.get_tmp_g();

    calc_divergence_g<TF><<<gridGPU, blockGPU>>>(
        fields.mp.at("u")->fld_g, fields.mp.at("v")->fld_g, fields.mp.at("w")->fld_g, divergence->fld_g,
        gd.dzi_g, fields.rhoref_g, fields.rhorefh_g, gd.dxi, gd.dyi,
        gd.icells, gd.ijcells,
        gd.istart,  gd.jstart, gd.kstart,
        gd.iend, gd.jend, gd.kend);
    cuda_check_error();

    TF divmax = field3d_operators.calc_max_g(divergence->fld_g);
    // TO-DO: add grid.get_max() or similar for future parallel versions

    fields.release_tmp_g(divergence);

    return divmax;
}
#endif


#ifdef FLOAT_SINGLE
template class Pres_2<float>;
#else
template class Pres_2<double>;
#endif
