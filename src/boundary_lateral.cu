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

#include "boundary_lateral.h"
#include "master.h"
#include "grid.h"
#include "fields.h"
#include "stats.h"
#include "timeloop.h"
#include "constants.h"
#include "tools.h"

namespace
{
    // 3x3x3 (effectively 5-point horizontal) Laplacian used for the sponge
    // diffusion, identical to the CPU `diffusion_3x3x3()`.
    template<typename TF> __device__
    TF diffusion_3x3x3_g(
            const TF* const __restrict__ fld,
            const TF lbc_val,
            const int ijk,
            const int icells,
            const int ijcells)
    {
        const TF vc = lbc_val - fld[ijk];
        const TF v1 = lbc_val - fld[ijk-1];
        const TF v2 = lbc_val - fld[ijk+1];
        const TF v3 = lbc_val - fld[ijk-icells];
        const TF v4 = lbc_val - fld[ijk+icells];

        // NOTE: this matches the CPU kernel exactly (which uses `v3` twice).
        const TF fld_diff = v1 + v2 + v3 + v3 - TF(4) * vc;

        return fld_diff;
    }


    // Copy the lateral boundary condition arrays into the ghost cells of `fld`.
    template<typename TF, Lbc_location location> __global__
    void set_lbc_gcs_g(
            TF* const __restrict__ fld,
            const TF* const __restrict__ lbc,
            const int ngc, const int nsponge,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells, const int kcells)
    {
        const int jstride_out = icells;
        const int kstride_out = icells * jcells;

        if (location == Lbc_location::West || location == Lbc_location::East)
        {
            const int i = blockIdx.x*blockDim.x + threadIdx.x;
            const int j = blockIdx.y*blockDim.y + threadIdx.y;
            const int k = blockIdx.z*blockDim.z + threadIdx.z + kstart;

            if (i < ngc && j < jcells && k < kend)
            {
                const int jstride_lbc = ngc + nsponge;
                const int kstride_lbc = jstride_lbc * jcells;

                int ijk_in;
                int ijk_out;
                int ijk_gc;

                if (location == Lbc_location::West)
                {
                    ijk_in  = i + j*jstride_lbc + k*kstride_lbc;
                    ijk_out = i + j*jstride_out + k*kstride_out;
                    ijk_gc  = i + j*jstride_out + (kstart-1)*kstride_out;
                }
                else // East
                {
                    ijk_in  = (i+nsponge) + j*jstride_lbc + k*kstride_lbc;
                    ijk_out = (i+iend)     + j*jstride_out + k*kstride_out;
                    ijk_gc  = (i+iend)     + j*jstride_out + (kstart-1)*kstride_out;
                }

                fld[ijk_out] = lbc[ijk_in];

                // Set one ghost cell below the surface.
                if (k == kstart)
                    fld[ijk_gc] = fld[ijk_out];
            }
        }
        else // South / North
        {
            const int i = blockIdx.x*blockDim.x + threadIdx.x;
            const int j = blockIdx.y*blockDim.y + threadIdx.y;
            const int k = blockIdx.z*blockDim.z + threadIdx.z + kstart;

            if (i < icells && j < ngc && k < kend)
            {
                const int jstride_in = icells;
                const int kstride_lbc = jstride_in * (ngc + nsponge);

                int ijk_in;
                int ijk_out;
                int ijk_gc;

                if (location == Lbc_location::South)
                {
                    ijk_in  = i + j*jstride_in  + k*kstride_lbc;
                    ijk_out = i + j*jstride_out + k*kstride_out;
                    ijk_gc  = i + j*jstride_out + (kstart-1)*kstride_out;
                }
                else // North
                {
                    ijk_in  = i + (j+nsponge)*jstride_in  + k*kstride_lbc;
                    ijk_out = i + (j+jend)*jstride_out     + k*kstride_out;
                    ijk_gc  = i + (j+jend)*jstride_out     + (kstart-1)*kstride_out;
                }

                fld[ijk_out] = lbc[ijk_in];

                if (k == kstart)
                    fld[ijk_gc] = fld[ijk_out];
            }
        }
    }


    // Set the prescribed vertical velocity at the domain top.
    template<typename TF> __global__
    void set_wtop_g(
            TF* const __restrict__ w,
            const TF* const __restrict__ w_top_2d,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kend,
            const int icells, const int ijcells)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;

        if (i < iend && j < jend)
        {
            const int ij  = i + j*icells;
            const int ijk = i + j*icells + kend*ijcells;
            w[ijk] = w_top_2d[ij];
        }
    }


    // Enforce a Neumann BC of 0 on `w` at the lateral boundaries (extrapolation).
    template<typename TF, Lbc_location location> __global__
    void set_ghost_cell_w_g(
            TF* const __restrict__ a,
            const int istart, const int iend, const int igc,
            const int jstart, const int jend, const int jgc,
            const int kstart, const int kend,
            const int icells, const int ijcells)
    {
        if (location == Lbc_location::West || location == Lbc_location::East)
        {
            const int i = blockIdx.x*blockDim.x + threadIdx.x;
            const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
            const int k = blockIdx.z*blockDim.z + threadIdx.z + kstart;

            if (i < igc && j < jend && k < kend)
            {
                int ijk_d;
                int ijk_gc;
                if (location == Lbc_location::West)
                {
                    ijk_d  = (istart    ) + j*icells + k*ijcells;
                    ijk_gc = (istart-1-i) + j*icells + k*ijcells;
                }
                else // East
                {
                    ijk_d  = (iend-1) + j*icells + k*ijcells;
                    ijk_gc = (iend+i) + j*icells + k*ijcells;
                }
                a[ijk_gc] = a[ijk_d];
            }
        }
        else // South / North
        {
            const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
            const int j = blockIdx.y*blockDim.y + threadIdx.y;
            const int k = blockIdx.z*blockDim.z + threadIdx.z + kstart;

            if (i < iend && j < jgc && k < kend)
            {
                int ijk_d;
                int ijk_gc;
                if (location == Lbc_location::South)
                {
                    ijk_d  = i + (jstart    )*icells + k*ijcells;
                    ijk_gc = i + (jstart-1-j)*icells + k*ijcells;
                }
                else // North
                {
                    ijk_d  = i + (jend-1)*icells + k*ijcells;
                    ijk_gc = i + (jend+j)*icells + k*ijcells;
                }
                a[ijk_gc] = a[ijk_d];
            }
        }
    }


    // In-plane extrapolation of the four horizontal corner ghost columns.
    // Single-MPI-rank only (USEMPI is disabled in CUDA builds), so all four
    // corners live on this rank.
    template<typename TF> __global__
    void set_corner_ghost_extrap_g(
            TF* const __restrict__ fld,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells)
    {
        const int ii = 1;
        const int jj = icells;
        const int kk = ijcells;

        const int igc = istart;
        const int jgc = jstart;

        const int a = blockIdx.x*blockDim.x + threadIdx.x; // 0..igc-1
        const int b = blockIdx.y*blockDim.y + threadIdx.y; // 0..jgc-1
        const int k = blockIdx.z*blockDim.z + threadIdx.z + kstart;

        if (a < igc && b < jgc && k < kend)
        {
            const int di = a + 1;
            const int dj = b + 1;

            // South-west.
            {
                const int ijk0 = istart + jstart*jj + k*kk;
                const TF dfdi = fld[ijk0] - fld[ijk0-ii];
                const TF dfdj = fld[ijk0] - fld[ijk0-jj];
                const int ijk = (istart-di) + (jstart-dj)*jj + k*kk;
                fld[ijk] = fld[ijk0] - di*dfdi - dj*dfdj;
            }
            // South-east.
            {
                const int ijk0 = (iend-1) + jstart*jj + k*kk;
                const TF dfdi = fld[ijk0+ii] - fld[ijk0];
                const TF dfdj = fld[ijk0] - fld[ijk0-jj];
                const int ijk = ((iend-1)+di) + (jstart-dj)*jj + k*kk;
                fld[ijk] = fld[ijk0] + di*dfdi - dj*dfdj;
            }
            // North-west.
            {
                const int ijk0 = istart + (jend-1)*jj + k*kk;
                const TF dfdi = fld[ijk0] - fld[ijk0-ii];
                const TF dfdj = fld[ijk0+jj] - fld[ijk0];
                const int ijk = (istart-di) + ((jend-1)+dj)*jj + k*kk;
                fld[ijk] = fld[ijk0] - di*dfdi + dj*dfdj;
            }
            // North-east.
            {
                const int ijk0 = (iend-1) + (jend-1)*jj + k*kk;
                const TF dfdi = fld[ijk0+ii] - fld[ijk0];
                const TF dfdj = fld[ijk0+jj] - fld[ijk0];
                const int ijk = ((iend-1)+di) + ((jend-1)+dj)*jj + k*kk;
                fld[ijk] = fld[ijk0] + di*dfdi + dj*dfdj;
            }
        }
    }


    // Vertical fill of the corner ghost columns below the surface and above the
    // domain top, copying from the first/last interior level.
    template<typename TF> __global__
    void set_corner_ghost_vfill_g(
            TF* const __restrict__ fld,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells, const int kcells)
    {
        const int jj = icells;
        const int kk = ijcells;

        const int igc = istart;
        const int jgc = jstart;

        const int a = blockIdx.x*blockDim.x + threadIdx.x; // 0..igc-1
        const int b = blockIdx.y*blockDim.y + threadIdx.y; // 0..jgc-1
        const int k = blockIdx.z*blockDim.z + threadIdx.z; // 0..kcells-1

        if (a >= igc || b >= jgc || k >= kcells)
            return;
        if (k >= kstart && k < kend)
            return;

        const int ksrc = (k < kstart) ? kstart : (kend-1);

        // Four corners; the i/j ghost ranges mirror the CPU kernel.
        // South-west.
        {
            const int i = istart-1-a;
            const int j = jstart-1-b;
            fld[i + j*jj + k*kk] = fld[i + j*jj + ksrc*kk];
        }
        // South-east.
        {
            const int i = iend+a;
            const int j = jstart-1-b;
            fld[i + j*jj + k*kk] = fld[i + j*jj + ksrc*kk];
        }
        // North-west.
        {
            const int i = istart-1-a;
            const int j = jend+b;
            fld[i + j*jj + k*kk] = fld[i + j*jj + ksrc*kk];
        }
        // North-east.
        {
            const int i = iend+a;
            const int j = jend+b;
            fld[i + j*jj + k*kk] = fld[i + j*jj + ksrc*kk];
        }
    }


    // Lateral sponge for `u` at the west/east boundaries.
    template<typename TF, Lbc_location location> __global__
    void lateral_sponge_u_g(
            TF* const __restrict__ ut,
            const TF* const __restrict__ u,
            const TF* const __restrict__ lbc_u,
            const TF tau_sponge,
            const TF w_diff,
            const int nsponge,
            const int igc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int ijcells)
    {
        const int j = blockIdx.x*blockDim.x + threadIdx.x + jstart;
        const int k = blockIdx.y*blockDim.y + threadIdx.y + kstart;

        if (j >= jend || k >= kend)
            return;

        const int igc_pad = (location==Lbc_location::West) ? igc+1 : igc;
        const int nstart = (location==Lbc_location::West) ? 2 : 1;
        const int jstride_lbc = igc_pad+nsponge;

        const TF w_dt = TF(1) / tau_sponge;

        for (int n=nstart; n<=nsponge; ++n)
        {
            const int ilbc = (location==Lbc_location::West) ? igc+n-1 : nsponge-n;
            const int ijk_lbc = ilbc + j*jstride_lbc + k*jstride_lbc*jcells;

            const int i = (location==Lbc_location::West) ? istart+(n-1) : iend-n;
            const int ijk = i + j*icells + k*ijcells;

            const TF u_diff = diffusion_3x3x3_g(u, lbc_u[ijk_lbc], ijk, icells, ijcells);

            const TF f_sponge = (TF(1)+nsponge-n) / nsponge;
            const TF w1n = w_dt * f_sponge;
            const TF w2n = w_diff * f_sponge;

            ut[ijk] += w1n * (lbc_u[ijk_lbc]-u[ijk]);
            ut[ijk] -= w2n * u_diff;
        }
    }


    // Lateral sponge for `v` at the south/north boundaries.
    template<typename TF, Lbc_location location> __global__
    void lateral_sponge_v_g(
            TF* const __restrict__ vt,
            const TF* const __restrict__ v,
            const TF* const __restrict__ lbc_v,
            const TF tau_sponge,
            const TF w_diff,
            const int nsponge,
            const int jgc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int ijcells)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int k = blockIdx.y*blockDim.y + threadIdx.y + kstart;

        if (i >= iend || k >= kend)
            return;

        const int jgc_pad = (location==Lbc_location::South) ? jgc+1 : jgc;

        const TF w_dt = TF(1) / tau_sponge;

        for (int n=2; n<=nsponge; ++n)
        {
            const int jlbc = (location==Lbc_location::South) ? jgc+n-1 : nsponge-n;
            const int ijk_lbc = i + jlbc*icells + k*icells*(jgc_pad+nsponge);

            const int j = (location==Lbc_location::South) ? jstart+(n-1) : jend-(n-1);
            const int ijk = i + j*icells + k*ijcells;

            const TF v_diff = diffusion_3x3x3_g(v, lbc_v[ijk_lbc], ijk, icells, ijcells);

            const TF f_sponge = (TF(1)+nsponge-n) / nsponge;
            const TF w1n = w_dt * f_sponge;
            const TF w2n = w_diff * f_sponge;

            vt[ijk] += w1n * (lbc_v[ijk_lbc]-v[ijk]);
            vt[ijk] -= w2n * v_diff;
        }
    }


    // Generic scalar/velocity lateral sponge for the west/east boundaries.
    template<typename TF, Lbc_location location, bool sw_recycle> __global__
    void lateral_sponge_s_we_g(
            TF* const __restrict__ at,
            const TF* const __restrict__ a,
            const TF* const __restrict__ lbc,
            const TF tau_sponge,
            const TF w_diff,
            const int nsponge,
            const TF tau_recycle,
            const TF recycle_offset,
            const int npy, const int mpiidy,
            const int igc, const int jgc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int ijcells)
    {
        const int j = blockIdx.x*blockDim.x + threadIdx.x + jstart;
        const int k = blockIdx.y*blockDim.y + threadIdx.y + kstart;

        if (j >= jend || k >= kend)
            return;

        const TF w_dt = TF(1) / tau_sponge;
        const TF r_dt = TF(1) / tau_recycle;

        const int jstride_lbc = igc+nsponge;

        for (int n=1; n<=nsponge; ++n)
        {
            // Offset in y-direction for domain corners.
            const int jstart_loc = mpiidy == 0     ? jstart + (n-1) : jstart;
            const int jend_loc   = mpiidy == npy-1 ? jend   - (n-1) : jend;

            if (j < jstart_loc || j >= jend_loc)
                continue;

            const int ilbc = (location==Lbc_location::West) ? igc+n-1 : nsponge-n;
            const int ijk_lbc = ilbc + j*jstride_lbc + k*jstride_lbc*jcells;

            const int i = (location==Lbc_location::West) ? istart+n-1 : iend-n;
            const int ijk = i + j*icells + k*ijcells;

            const TF a_diff = diffusion_3x3x3_g(a, lbc[ijk_lbc], ijk, icells, ijcells);

            const TF f_sponge = (TF(1)+nsponge-(n+TF(0.5))) / nsponge;
            const TF w1n = w_dt * f_sponge;
            const TF w2n = w_diff * f_sponge;

            at[ijk] += w1n * (lbc[ijk_lbc]-a[ijk]);
            at[ijk] -= w2n * a_diff;

            if (sw_recycle)
            {
                const TF f_recycle = TF(1) - f_sponge;
                const int offset = (location == Lbc_location::West) ? recycle_offset : -recycle_offset;
                const int ijko = (i+offset) + j*icells + k*ijcells;

                TF a_mean = 0;
                for (int jc=-3; jc<4; ++jc)
                    for (int ic=-3; ic<4; ++ic)
                        a_mean += a[ijko + ic + jc*icells];
                a_mean /= TF(49.);

                at[ijk] += f_recycle * r_dt * ((a[ijko] - a_mean) - (a[ijk] - lbc[ijk_lbc]));
            }
        }
    }


    // Generic scalar/velocity lateral sponge for the south/north boundaries.
    template<typename TF, Lbc_location location, bool sw_recycle> __global__
    void lateral_sponge_s_sn_g(
            TF* const __restrict__ at,
            const TF* const __restrict__ a,
            const TF* const __restrict__ lbc,
            const TF tau_sponge,
            const TF w_diff,
            const int nsponge,
            const TF tau_recycle,
            const TF recycle_offset,
            const int npx, const int mpiidx,
            const int igc, const int jgc,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells,
            const int ijcells)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int k = blockIdx.y*blockDim.y + threadIdx.y + kstart;

        if (i >= iend || k >= kend)
            return;

        const TF w_dt = TF(1) / tau_sponge;
        const TF r_dt = TF(1) / tau_recycle;

        const int kstride_lbc = jgc + nsponge;

        for (int n=1; n<=nsponge; ++n)
        {
            const int istart_loc = (mpiidx == 0)     ? istart + n : istart;
            const int iend_loc   = (mpiidx == npx-1) ? iend   - n : iend;

            if (i < istart_loc || i >= iend_loc)
                continue;

            const int jlbc = (location==Lbc_location::South) ? jgc+n-1 : nsponge-n;
            const int ijk_lbc = i + jlbc*icells + k*icells*kstride_lbc;

            const int j = (location==Lbc_location::South) ? jstart+n-1 : jend-n;
            const int ijk = i + j*icells + k*ijcells;

            const TF a_diff = diffusion_3x3x3_g(a, lbc[ijk_lbc], ijk, icells, ijcells);

            const TF f_sponge = (TF(1)+nsponge-(n+TF(0.5))) / nsponge;
            const TF w1n = w_dt * f_sponge;
            const TF w2n = w_diff * f_sponge;

            at[ijk] += w1n*(lbc[ijk_lbc]-a[ijk]);
            at[ijk] -= w2n*a_diff;

            if (sw_recycle)
            {
                const TF f_recycle = TF(1) - f_sponge;
                const int offset = (location == Lbc_location::South) ? recycle_offset : -recycle_offset;
                const int ijko = i + (j+offset)*icells + k*ijcells;

                TF a_mean = 0;
                for (int jc=-3; jc<4; ++jc)
                    for (int ic=-3; ic<4; ++ic)
                        a_mean += a[ijko + ic + jc*icells];
                a_mean /= TF(49.);

                at[ijk] += f_recycle * r_dt * ((a[ijko] - a_mean) - (a[ijk] - lbc[ijk_lbc]));
            }
        }
    }
}


template<typename TF>
void Boundary_lateral<TF>::prepare_device()
{
    if (!sw_openbc)
        return;

    auto& gd = grid.get_grid_data();

    // Allocate device LBC arrays matching the host map sizes, and copy the
    // initial (constant) values. For time-dependent LBCs the host arrays are
    // re-interpolated each step and synced in `forward_device()`.
    auto alloc_and_copy = [&](
            std::map<std::string, cuda_vector<TF>>& dst,
            Lbc_map<TF>& src)
    {
        for (auto& it : src)
        {
            dst.emplace(it.first, cuda_vector<TF>(it.second.size()));
            cuda_safe_call(cudaMemcpy(
                    dst.at(it.first), it.second.data(),
                    it.second.size()*sizeof(TF), cudaMemcpyHostToDevice));
        }
    };

    alloc_and_copy(lbc_w_g, lbc_w);
    alloc_and_copy(lbc_e_g, lbc_e);
    alloc_and_copy(lbc_s_g, lbc_s);
    alloc_and_copy(lbc_n_g, lbc_n);

    if (sw_openbc_uv)
    {
        w_top_2d_g.allocate(gd.ijcells);
        cuda_safe_call(cudaMemcpy(
                w_top_2d_g, w_top_2d.data(),
                gd.ijcells*sizeof(TF), cudaMemcpyHostToDevice));
    }
}


template<typename TF>
void Boundary_lateral<TF>::clear_device()
{
    // `cuda_vector` frees its device memory on destruction; nothing to do.
}


template<typename TF>
void Boundary_lateral<TF>::forward_device()
{
    if (!sw_openbc)
        return;

    auto& gd = grid.get_grid_data();

    auto copy_map = [&](
            std::map<std::string, cuda_vector<TF>>& dst,
            Lbc_map<TF>& src)
    {
        for (auto& it : src)
            cuda_safe_call(cudaMemcpy(
                    dst.at(it.first), it.second.data(),
                    it.second.size()*sizeof(TF), cudaMemcpyHostToDevice));
    };

    copy_map(lbc_w_g, lbc_w);
    copy_map(lbc_e_g, lbc_e);
    copy_map(lbc_s_g, lbc_s);
    copy_map(lbc_n_g, lbc_n);

    if (sw_openbc_uv)
        cuda_safe_call(cudaMemcpy(
                w_top_2d_g, w_top_2d.data(),
                gd.ijcells*sizeof(TF), cudaMemcpyHostToDevice));
}


#ifdef USECUDA
template <typename TF>
void Boundary_lateral<TF>::set_ghost_cells(
        Timeloop<TF>& timeloop)
{
    if (!sw_openbc)
        return;

    auto& gd = grid.get_grid_data();

    // Sync the (host-side interpolated) LBC arrays and `w_top` to the device.
    forward_device();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;

    auto set_gcs_field = [&](const std::string& fld)
    {
        TF* fld_g = fields.ap.at(fld)->fld_g;

        // West.
        {
            const int ngc = (fld == "u") ? gd.igc+1 : gd.igc;
            const int nk = gd.kend - gd.kstart;
            dim3 block(ngc, blockj, 1);
            dim3 grid(1, gd.jcells/blockj + (gd.jcells%blockj > 0), nk);
            set_lbc_gcs_g<TF, Lbc_location::West><<<grid, block>>>(
                    fld_g, lbc_w_g.at(fld),
                    ngc, n_sponge,
                    gd.istart, gd.iend, gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.kcells);
        }
        // East.
        {
            const int ngc = gd.igc;
            const int nk = gd.kend - gd.kstart;
            dim3 block(ngc, blockj, 1);
            dim3 grid(1, gd.jcells/blockj + (gd.jcells%blockj > 0), nk);
            set_lbc_gcs_g<TF, Lbc_location::East><<<grid, block>>>(
                    fld_g, lbc_e_g.at(fld),
                    ngc, n_sponge,
                    gd.istart, gd.iend, gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.kcells);
        }
        // South.
        {
            const int ngc = (fld == "v") ? gd.jgc+1 : gd.jgc;
            const int nk = gd.kend - gd.kstart;
            dim3 block(blocki, ngc, 1);
            dim3 grid(gd.icells/blocki + (gd.icells%blocki > 0), 1, nk);
            set_lbc_gcs_g<TF, Lbc_location::South><<<grid, block>>>(
                    fld_g, lbc_s_g.at(fld),
                    ngc, n_sponge,
                    gd.istart, gd.iend, gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.kcells);
        }
        // North.
        {
            const int ngc = gd.jgc;
            const int nk = gd.kend - gd.kstart;
            dim3 block(blocki, ngc, 1);
            dim3 grid(gd.icells/blocki + (gd.icells%blocki > 0), 1, nk);
            set_lbc_gcs_g<TF, Lbc_location::North><<<grid, block>>>(
                    fld_g, lbc_n_g.at(fld),
                    ngc, n_sponge,
                    gd.istart, gd.iend, gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.jcells, gd.kcells);
        }
    };

    if (sw_openbc_uv)
    {
        set_gcs_field("u");
        set_gcs_field("v");
    }

    if (sw_openbc_w)
        set_gcs_field("w");

    for (auto& fld : slist)
        set_gcs_field(fld);

    cuda_check_error();

    // Set vertical velocity at the domain top.
    if (sw_openbc_uv)
    {
        dim3 block(blocki, blockj, 1);
        dim3 grid(gd.imax/blocki + (gd.imax%blocki > 0),
                  gd.jmax/blockj + (gd.jmax%blockj > 0), 1);
        set_wtop_g<TF><<<grid, block>>>(
                fields.mp.at("w")->fld_g,
                w_top_2d_g,
                gd.istart, gd.iend, gd.jstart, gd.jend,
                gd.kend, gd.icells, gd.ijcells);
        cuda_check_error();
    }

    if (sw_neumann_w)
    {
        TF* w_g = fields.mp.at("w")->fld_g;
        const int nk = (gd.kend+1) - gd.kstart;

        // West / East: thread over (igc, jend-jstart, nk).
        {
            dim3 block(gd.igc, blockj, 1);
            dim3 grid(1, (gd.jend-gd.jstart)/blockj + ((gd.jend-gd.jstart)%blockj > 0), nk);
            set_ghost_cell_w_g<TF, Lbc_location::West><<<grid, block>>>(
                    w_g, gd.istart, gd.iend, gd.igc,
                    gd.jstart, gd.jend, gd.jgc,
                    gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
            set_ghost_cell_w_g<TF, Lbc_location::East><<<grid, block>>>(
                    w_g, gd.istart, gd.iend, gd.igc,
                    gd.jstart, gd.jend, gd.jgc,
                    gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        }
        // South / North: thread over (iend-istart, jgc, nk).
        {
            dim3 block(blocki, gd.jgc, 1);
            dim3 grid((gd.iend-gd.istart)/blocki + ((gd.iend-gd.istart)%blocki > 0), 1, nk);
            set_ghost_cell_w_g<TF, Lbc_location::South><<<grid, block>>>(
                    w_g, gd.istart, gd.iend, gd.igc,
                    gd.jstart, gd.jend, gd.jgc,
                    gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
            set_ghost_cell_w_g<TF, Lbc_location::North><<<grid, block>>>(
                    w_g, gd.istart, gd.iend, gd.igc,
                    gd.jstart, gd.jend, gd.jgc,
                    gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        }
        cuda_check_error();

        // Corner ghost cells: first the in-plane extrapolation, then the
        // vertical fill (which reads the just-set interior-level values).
        {
            dim3 block(gd.igc, gd.jgc, 1);
            dim3 grid_extrap(1, 1, (gd.kend+1)-gd.kstart);
            set_corner_ghost_extrap_g<TF><<<grid_extrap, block>>>(
                    w_g, gd.istart, gd.iend, gd.jstart, gd.jend,
                    gd.kstart, gd.kend+1, gd.icells, gd.ijcells);

            dim3 grid_vfill(1, 1, gd.kcells);
            set_corner_ghost_vfill_g<TF><<<grid_vfill, block>>>(
                    w_g, gd.istart, gd.iend, gd.jstart, gd.jend,
                    gd.kstart, gd.kend+1, gd.icells, gd.ijcells, gd.kcells);
        }
        cuda_check_error();
    }
}


template <typename TF>
void Boundary_lateral<TF>::exec_lateral_sponge(
        Stats<TF>& stats)
{
    if (!sw_openbc or !sw_sponge)
        return;


    auto& gd = grid.get_grid_data();
    auto& md = master.get_MPI_data();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;

    // Grid for kernels threaded over (j, k) [west/east].
    dim3 block_jk(blockj, 8, 1);
    dim3 grid_jk((gd.jend-gd.jstart)/blockj + ((gd.jend-gd.jstart)%blockj > 0),
                 (gd.kend-gd.kstart)/8 + ((gd.kend-gd.kstart)%8 > 0), 1);

    // Grid for kernels threaded over (i, k) [south/north].
    dim3 block_ik(blocki, 8, 1);
    dim3 grid_ik((gd.iend-gd.istart)/blocki + ((gd.iend-gd.istart)%blocki > 0),
                 (gd.kend-gd.kstart)/8 + ((gd.kend-gd.kstart)%8 > 0), 1);

    auto sponge_s_we = [&]<Lbc_location location, bool recycle>(
            std::map<std::string, cuda_vector<TF>>& lbc_map_g,
            const std::string& name)
    {
        const int kstart = (name == "w") ? gd.kstart+1 : gd.kstart;
        lateral_sponge_s_we_g<TF, location, recycle><<<grid_jk, block_jk>>>(
                fields.at.at(name)->fld_g,
                fields.ap.at(name)->fld_g,
                lbc_map_g.at(name),
                tau_sponge, w_diff, n_sponge,
                tau_recycle, recycle_offset,
                md.npy, md.mpicoordy,
                gd.igc, gd.jgc,
                gd.istart, gd.iend, gd.jstart, gd.jend,
                kstart, gd.kend,
                gd.icells, gd.jcells, gd.ijcells);
    };

    auto sponge_s_sn = [&]<Lbc_location location, bool recycle>(
            std::map<std::string, cuda_vector<TF>>& lbc_map_g,
            const std::string& name)
    {
        const int kstart = (name == "w") ? gd.kstart+1 : gd.kstart;
        lateral_sponge_s_sn_g<TF, location, recycle><<<grid_ik, block_ik>>>(
                fields.at.at(name)->fld_g,
                fields.ap.at(name)->fld_g,
                lbc_map_g.at(name),
                tau_sponge, w_diff, n_sponge,
                tau_recycle, recycle_offset,
                md.npx, md.mpicoordx,
                gd.igc, gd.jgc,
                gd.istart, gd.iend, gd.jstart, gd.jend,
                kstart, gd.kend,
                gd.icells, gd.jcells, gd.ijcells);
    };

    if (sw_openbc_uv)
    {
        // NOTE: order matches the CPU version; the `_u`/`_v` kernels ignore the
        // corner offset, so they must run before the generic sponge.
        lateral_sponge_u_g<TF, Lbc_location::West><<<grid_jk, block_jk>>>(
                fields.mt.at("u")->fld_g, fields.mp.at("u")->fld_g, lbc_w_g.at("u"),
                tau_sponge, w_diff, n_sponge, gd.igc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.jcells, gd.ijcells);
        lateral_sponge_u_g<TF, Lbc_location::East><<<grid_jk, block_jk>>>(
                fields.mt.at("u")->fld_g, fields.mp.at("u")->fld_g, lbc_e_g.at("u"),
                tau_sponge, w_diff, n_sponge, gd.igc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.jcells, gd.ijcells);
        sponge_s_sn.template operator()<Lbc_location::South, false>(lbc_s_g, "u");
        sponge_s_sn.template operator()<Lbc_location::North, false>(lbc_n_g, "u");

        lateral_sponge_v_g<TF, Lbc_location::South><<<grid_ik, block_ik>>>(
                fields.mt.at("v")->fld_g, fields.mp.at("v")->fld_g, lbc_s_g.at("v"),
                tau_sponge, w_diff, n_sponge, gd.jgc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.jcells, gd.ijcells);
        lateral_sponge_v_g<TF, Lbc_location::North><<<grid_ik, block_ik>>>(
                fields.mt.at("v")->fld_g, fields.mp.at("v")->fld_g, lbc_n_g.at("v"),
                tau_sponge, w_diff, n_sponge, gd.jgc,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                gd.icells, gd.jcells, gd.ijcells);
        sponge_s_we.template operator()<Lbc_location::West, false>(lbc_w_g, "v");
        sponge_s_we.template operator()<Lbc_location::East, false>(lbc_e_g, "v");

        cuda_check_error();
        cudaDeviceSynchronize();
        stats.calc_tend(*fields.mt.at("u"), tend_name);
        stats.calc_tend(*fields.mt.at("v"), tend_name);
    }

    if (sw_openbc_w)
    {
        sponge_s_we.template operator()<Lbc_location::West, false>(lbc_w_g, "w");
        sponge_s_we.template operator()<Lbc_location::East, false>(lbc_e_g, "w");
        sponge_s_sn.template operator()<Lbc_location::South, false>(lbc_s_g, "w");
        sponge_s_sn.template operator()<Lbc_location::North, false>(lbc_n_g, "w");

        cuda_check_error();
        cudaDeviceSynchronize();
        stats.calc_tend(*fields.mt.at("w"), tend_name);
    }

    for (auto& fld : slist)
    {
        const bool sw_recycle_fld =
                std::find(recycle_list.begin(), recycle_list.end(), fld) != recycle_list.end();

        if (sw_recycle_fld && sw_recycle[Lbc_location::West])
            sponge_s_we.template operator()<Lbc_location::West, true>(lbc_w_g, fld);
        else
            sponge_s_we.template operator()<Lbc_location::West, false>(lbc_w_g, fld);

        if (sw_recycle[Lbc_location::East])
        {
            if (sw_recycle_fld)
                sponge_s_we.template operator()<Lbc_location::East, true>(lbc_e_g, fld);
            else
                sponge_s_we.template operator()<Lbc_location::East, false>(lbc_e_g, fld);
        }
        if (sw_recycle[Lbc_location::South])
        {
            if (sw_recycle_fld)
                sponge_s_sn.template operator()<Lbc_location::South, true>(lbc_s_g, fld);
            else
                sponge_s_sn.template operator()<Lbc_location::South, false>(lbc_s_g, fld);
        }
        if (sw_recycle[Lbc_location::North])
        {
            if (sw_recycle_fld)
                sponge_s_sn.template operator()<Lbc_location::North, true>(lbc_n_g, fld);
            else
                sponge_s_sn.template operator()<Lbc_location::North, false>(lbc_n_g, fld);
        }

        cuda_check_error();
        cudaDeviceSynchronize();
        stats.calc_tend(*fields.at.at(fld), tend_name);
    }
}
#endif


#ifdef FLOAT_SINGLE
template class Boundary_lateral<float>;
#else
template class Boundary_lateral<double>;
#endif
