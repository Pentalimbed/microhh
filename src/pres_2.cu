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
    inline void check_dct_cufft(const cufftResult result)
    {
        if (result != CUFFT_SUCCESS)
            throw std::runtime_error("cuFFT error while creating open-boundary DCT plan");
    }
    template<typename TF> cufftType cufft_r2c_type();
    template<> cufftType cufft_r2c_type<float>() { return CUFFT_R2C; }
    template<> cufftType cufft_r2c_type<double>() { return CUFFT_D2Z; }
    template<typename TF> cufftType cufft_c2r_type();
    template<> cufftType cufft_c2r_type<float>() { return CUFFT_C2R; }
    template<> cufftType cufft_c2r_type<double>() { return CUFFT_Z2D; }

    template<typename TF> void cufft_r2c(cufftHandle, TF*, void*);
    template<> void cufft_r2c<float>(cufftHandle plan, float* in, void* out)
    { cufftExecR2C(plan, in, static_cast<cufftComplex*>(out)); }
    template<> void cufft_r2c<double>(cufftHandle plan, double* in, void* out)
    { cufftExecD2Z(plan, in, static_cast<cufftDoubleComplex*>(out)); }
    template<typename TF> void cufft_c2r(cufftHandle, void*, TF*);
    template<> void cufft_c2r<float>(cufftHandle plan, void* in, float* out)
    { cufftExecC2R(plan, static_cast<cufftComplex*>(in), out); }
    template<> void cufft_c2r<double>(cufftHandle plan, void* in, double* out)
    { cufftExecZ2D(plan, static_cast<cufftDoubleComplex*>(in), out); }

    template<typename TF>
    __global__ void extend_x_g(
            TF* const out, const TF* const in,
            const int nx, const int ny, const int nz)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        const int z = blockIdx.z;
        if (x < 2*nx && y < ny && z < nz)
        {
            const int source_x = x < nx ? x : 2*nx-1-x;
            out[x+y*2*nx+z*2*nx*ny] = in[source_x+y*nx+z*nx*ny];
        }
    }

    template<typename TF, typename Complex>
    __global__ void extract_dct_x_g(
            TF* const out, const Complex* const in,
            const int nx, const int ny, const int nz)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        const int z = blockIdx.z;
        if (x < nx && y < ny && z < nz)
        {
            const TF angle = TF(M_PI)*x/(TF(2)*nx);
            const Complex value = in[x+y*(nx+1)+z*(nx+1)*ny];
            out[x+y*nx+z*nx*ny] = value.x*cos(angle)+value.y*sin(angle);
        }
    }

    template<typename TF, typename Complex>
    __global__ void prepare_idct_x_g(
            Complex* const out, const TF* const in,
            const int nx, const int ny, const int nz)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        const int z = blockIdx.z;
        if (x <= nx && y < ny && z < nz)
        {
            Complex value{};
            if (x < nx)
            {
                const TF angle = TF(M_PI)*x/(TF(2)*nx);
                const TF coefficient = in[x+y*nx+z*nx*ny];
                value.x = coefficient*cos(angle);
                value.y = coefficient*sin(angle);
            }
            out[x+y*(nx+1)+z*(nx+1)*ny] = value;
        }
    }

    template<typename TF>
    __global__ void extract_idct_x_g(
            TF* const out, const TF* const in,
            const int nx, const int ny, const int nz)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        const int z = blockIdx.z;
        if (x < nx && y < ny && z < nz)
            out[x+y*nx+z*nx*ny] = in[x+y*2*nx+z*2*nx*ny]/(TF(2)*nx);
    }

    template<typename TF>
    __global__ void extend_y_g(
            TF* const out, const TF* const in,
            const int nx, const int ny, const int z)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        if (x < nx && y < 2*ny)
        {
            const int source_y = y < ny ? y : 2*ny-1-y;
            out[x+y*nx+z*nx*2*ny] = in[x+source_y*nx+z*nx*ny];
        }
    }

    template<typename TF, typename Complex>
    __global__ void extract_dct_y_g(
            TF* const out, const Complex* const in,
            const int nx, const int ny, const int z)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        if (x < nx && y < ny)
        {
            const TF angle = TF(M_PI)*y/(TF(2)*ny);
            const Complex value = in[x+y*nx+z*nx*(ny+1)];
            out[x+y*nx+z*nx*ny] = value.x*cos(angle)+value.y*sin(angle);
        }
    }

    template<typename TF, typename Complex>
    __global__ void prepare_idct_y_g(
            Complex* const out, const TF* const in,
            const int nx, const int ny, const int z)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        if (x < nx && y <= ny)
        {
            Complex value{};
            if (y < ny)
            {
                const TF angle = TF(M_PI)*y/(TF(2)*ny);
                const TF coefficient = in[x+y*nx+z*nx*ny];
                value.x = coefficient*cos(angle);
                value.y = coefficient*sin(angle);
            }
            out[x+y*nx+z*nx*(ny+1)] = value;
        }
    }

    template<typename TF>
    __global__ void extract_idct_y_g(
            TF* const out, const TF* const in,
            const int nx, const int ny, const int z)
    {
        const int x = blockIdx.x*blockDim.x+threadIdx.x;
        const int y = blockIdx.y*blockDim.y+threadIdx.y;
        if (x < nx && y < ny)
            out[x+y*nx+z*nx*ny] = in[x+y*nx+z*nx*2*ny]/(TF(2)*ny);
    }

    template<typename TF>
    __global__ void zero_open_boundary_tendencies_g(
            TF* const ut, TF* const vt,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells,
            const bool west, const bool east,
            const bool south, const bool north)
    {
        const int n = blockIdx.x*blockDim.x+threadIdx.x;
        const int k = blockIdx.y+kstart;
        if (k >= kend) return;
        if (n < jend-jstart)
        {
            const int j = n+jstart;
            if (west) ut[istart+j*icells+k*ijcells] = TF(0);
            if (east) ut[iend+j*icells+k*ijcells] = TF(0);
        }
        if (n < iend-istart)
        {
            const int i = n+istart;
            if (south) vt[i+jstart*icells+k*ijcells] = TF(0);
            if (north) vt[i+jend*icells+k*ijcells] = TF(0);
        }
    }

    template<typename TF>
    __global__ void set_pressure_open_ghosts_g(
            TF* const p,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells,
            const bool west, const bool east,
            const bool south, const bool north)
    {
        const int n = blockIdx.x*blockDim.x+threadIdx.x;
        const int k = blockIdx.y+kstart;
        if (k >= kend) return;
        if (n < jend-jstart)
        {
            const int j = n+jstart;
            if (west) p[istart-1+j*icells+k*ijcells] = p[istart+j*icells+k*ijcells];
            if (east) p[iend+j*icells+k*ijcells] = p[iend-1+j*icells+k*ijcells];
        }
        if (n < iend-istart)
        {
            const int i = n+istart;
            if (south) p[i+(jstart-1)*icells+k*ijcells] = p[i+jstart*icells+k*ijcells];
            if (north) p[i+jend*icells+k*ijcells] = p[i+(jend-1)*icells+k*ijcells];
        }
    }
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
} // End namespace.

#ifdef USECUDA
template<typename TF>
void Pres_2<TF>::make_dct_plans()
{
    const auto& gd = grid.get_grid_data();
    const int rank = 1;

    const int nx = 2*gd.itot;
    int nx_size[] = {nx};
    int nx_real_embed[] = {nx};
    int nx_complex_embed[] = {gd.itot+1};
    check_dct_cufft(cufftPlanMany(&dct_x_forward_plan, rank, nx_size,
            nx_real_embed, 1, nx,
            nx_complex_embed, 1, gd.itot+1,
            cufft_r2c_type<TF>(), gd.jtot*gd.ktot));
    check_dct_cufft(cufftPlanMany(&dct_x_backward_plan, rank, nx_size,
            nx_complex_embed, 1, gd.itot+1,
            nx_real_embed, 1, nx,
            cufft_c2r_type<TF>(), gd.jtot*gd.ktot));

    const int ny = 2*gd.jtot;
    int ny_size[] = {ny};
    int ny_real_embed[] = {ny};
    int ny_complex_embed[] = {gd.jtot+1};
    check_dct_cufft(cufftPlanMany(&dct_y_forward_plan, rank, ny_size,
            ny_real_embed, gd.itot, 1,
            ny_complex_embed, gd.itot, 1,
            cufft_r2c_type<TF>(), gd.itot));
    check_dct_cufft(cufftPlanMany(&dct_y_backward_plan, rank, ny_size,
            ny_complex_embed, gd.itot, 1,
            ny_real_embed, gd.itot, 1,
            cufft_c2r_type<TF>(), gd.itot));
}

template<typename TF>
void Pres_2<TF>::dct_forward(TF* field)
{
    const auto& gd = grid.get_grid_data();
    const dim3 block(16, 16);
    const dim3 grid_x((2*gd.itot+15)/16, (gd.jtot+15)/16, gd.ktot);
    extend_x_g<TF><<<grid_x, block>>>(dct_x_real_g.data(), field, gd.itot, gd.jtot, gd.ktot);
    cufft_r2c<TF>(dct_x_forward_plan, dct_x_real_g, dct_x_complex_g.data());
    const dim3 extract_x((gd.itot+15)/16, (gd.jtot+15)/16, gd.ktot);
    if constexpr (std::is_same_v<TF, float>)
        extract_dct_x_g<TF, cufftComplex><<<extract_x, block>>>(field, reinterpret_cast<cufftComplex*>(dct_x_complex_g.data()), gd.itot, gd.jtot, gd.ktot);
    else
        extract_dct_x_g<TF, cufftDoubleComplex><<<extract_x, block>>>(field, reinterpret_cast<cufftDoubleComplex*>(dct_x_complex_g.data()), gd.itot, gd.jtot, gd.ktot);

    const dim3 grid_y((gd.itot+15)/16, (2*gd.jtot+15)/16);
    const dim3 extract_y((gd.itot+15)/16, (gd.jtot+15)/16);
    for (int k=0; k<gd.ktot; ++k)
    {
        extend_y_g<TF><<<grid_y, block>>>(dct_y_real_g.data(), field, gd.itot, gd.jtot, k);
        cufft_r2c<TF>(dct_y_forward_plan, dct_y_real_g.data()+k*gd.itot*2*gd.jtot,
                dct_y_complex_g.data()+2*k*gd.itot*(gd.jtot+1));
        if constexpr (std::is_same_v<TF, float>)
            extract_dct_y_g<TF, cufftComplex><<<extract_y, block>>>(field, reinterpret_cast<cufftComplex*>(dct_y_complex_g.data()), gd.itot, gd.jtot, k);
        else
            extract_dct_y_g<TF, cufftDoubleComplex><<<extract_y, block>>>(field, reinterpret_cast<cufftDoubleComplex*>(dct_y_complex_g.data()), gd.itot, gd.jtot, k);
    }
    cuda_check_error();
}

template<typename TF>
void Pres_2<TF>::dct_backward(TF* field)
{
    const auto& gd = grid.get_grid_data();
    const dim3 block(16, 16);
    const dim3 prepare_y((gd.itot+15)/16, (gd.jtot+1+15)/16);
    const dim3 extract_y((gd.itot+15)/16, (gd.jtot+15)/16);
    for (int k=0; k<gd.ktot; ++k)
    {
        if constexpr (std::is_same_v<TF, float>)
            prepare_idct_y_g<TF, cufftComplex><<<prepare_y, block>>>(reinterpret_cast<cufftComplex*>(dct_y_complex_g.data()), field, gd.itot, gd.jtot, k);
        else
            prepare_idct_y_g<TF, cufftDoubleComplex><<<prepare_y, block>>>(reinterpret_cast<cufftDoubleComplex*>(dct_y_complex_g.data()), field, gd.itot, gd.jtot, k);
        cufft_c2r<TF>(dct_y_backward_plan, dct_y_complex_g.data()+2*k*gd.itot*(gd.jtot+1),
                dct_y_real_g.data()+k*gd.itot*2*gd.jtot);
        extract_idct_y_g<TF><<<extract_y, block>>>(field, dct_y_real_g.data(), gd.itot, gd.jtot, k);
    }

    const dim3 prepare_x((gd.itot+1+15)/16, (gd.jtot+15)/16, gd.ktot);
    if constexpr (std::is_same_v<TF, float>)
        prepare_idct_x_g<TF, cufftComplex><<<prepare_x, block>>>(reinterpret_cast<cufftComplex*>(dct_x_complex_g.data()), field, gd.itot, gd.jtot, gd.ktot);
    else
        prepare_idct_x_g<TF, cufftDoubleComplex><<<prepare_x, block>>>(reinterpret_cast<cufftDoubleComplex*>(dct_x_complex_g.data()), field, gd.itot, gd.jtot, gd.ktot);
    cufft_c2r<TF>(dct_x_backward_plan, dct_x_complex_g, dct_x_real_g);
    const dim3 extract_x((gd.itot+15)/16, (gd.jtot+15)/16, gd.ktot);
    extract_idct_x_g<TF><<<extract_x, block>>>(field, dct_x_real_g.data(), gd.itot, gd.jtot, gd.ktot);
    cuda_check_error();
}

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
        dct_x_real_g.allocate(2*gd.itot*gd.jtot*gd.ktot);
        dct_x_complex_g.allocate(2*(gd.itot+1)*gd.jtot*gd.ktot);
        dct_y_real_g.allocate(gd.itot*2*gd.jtot*gd.ktot);
        dct_y_complex_g.allocate(2*gd.itot*(gd.jtot+1)*gd.ktot);
        make_dct_plans();
    }
    else
        make_cufft_plan();
}

template<typename TF>
void Pres_2<TF>::clear_device()
{
    if (dct_x_forward_plan) cufftDestroy(dct_x_forward_plan);
    if (dct_x_backward_plan) cufftDestroy(dct_x_backward_plan);
    if (dct_y_forward_plan) cufftDestroy(dct_y_forward_plan);
    if (dct_y_backward_plan) cufftDestroy(dct_y_backward_plan);
    dct_x_forward_plan = dct_x_backward_plan = 0;
    dct_y_forward_plan = dct_y_backward_plan = 0;
}

template<typename TF>
void Pres_2<TF>::exec(double dt, Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();
    const auto& md = master.get_MPI_data();

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

    if (sw_openbc)
    {
        const int length = std::max(gd.imax, gd.jmax);
        const dim3 block_edges(256);
        const dim3 grid_edges((length+255)/256, gd.kmax);
        zero_open_boundary_tendencies_g<TF><<<grid_edges, block_edges>>>(
                fields.mt.at("u")->fld_g, fields.mt.at("v")->fld_g,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells,
                md.mpicoordx == 0, md.mpicoordx == md.npx-1,
                md.mpicoordy == 0, md.mpicoordy == md.npy-1);
    }
    else
    {
        boundary_cyclic.exec_g(fields.mt.at("u")->fld_g);
        boundary_cyclic.exec_g(fields.mt.at("v")->fld_g);
        boundary_cyclic.exec_g(fields.mt.at("w")->fld_g);
    }

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

    if (sw_openbc)
        dct_forward(fields.sd.at("p")->fld_g);
    else
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

    launch_grid_kernel<Pres_2_kernels::tdma_g<TF>>(
            grid_layout_2d_nogc,
            a_g,
            tmp2->fld_g,
            c_g,
            fields.sd.at("p")->fld_g.view(),
            tmp1->fld_g.view(),
            gd.kmax);

    if (sw_openbc)
        dct_backward(fields.sd.at("p")->fld_g);
    else
        fft_backward(fields.sd.at("p")->fld_g, tmp1->fld_g, tmp2->fld_g);

    cuda_safe_call(cudaMemcpy(tmp1->fld_g, fields.sd.at("p")->fld_g, gd.ncells*sizeof(TF), cudaMemcpyDeviceToDevice));

    launch_grid_kernel<Pres_2_kernels::solve_out_g<TF>>(
            grid_layout_nogc,
            fields.sd.at("p")->fld_g.view(),
            tmp1->fld_g,
            gd.istart, gd.jstart, gd.kstart,
            gd.icells, gd.ijcells);

    if (sw_openbc)
    {
        const int length = std::max(gd.imax, gd.jmax);
        const dim3 block_edges(256);
        const dim3 grid_edges((length+255)/256, gd.kmax);
        set_pressure_open_ghosts_g<TF><<<grid_edges, block_edges>>>(
                fields.sd.at("p")->fld_g,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.ijcells,
                md.mpicoordx == 0, md.mpicoordx == md.npx-1,
                md.mpicoordy == 0, md.mpicoordy == md.npy-1);
    }
    else
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
