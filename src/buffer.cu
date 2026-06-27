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
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include "grid.h"
#include "fields.h"
#include "buffer.h"
#include "constants.h"
#include "stats.h"
#include "tools.h"

#include "buffer_kernels.cuh"
#include "cuda_launcher.h"
#include "cuda_tiling.h"

namespace
{
    // Damping towards a (time-interpolated) 3D buffer field. Mirrors the CPU
    // `calc_buffer_3d()`: `abuf` is a 3D field indexed from `kstart_buffer`,
    // and `sigmaz` is the pre-computed damping coefficient per level.
    template<typename TF> __global__
    void calc_buffer_3d_g(
            TF* const __restrict__ at,
            const TF* const __restrict__ a,
            const TF* const __restrict__ abuf,
            const TF* const __restrict__ sigmaz,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart_buffer, const int kend,
            const int jstride, const int kstride)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;

        if (i < iend && j < jend)
        {
            for (int k=kstart_buffer; k<kend; ++k)
            {
                const int ijk  = i + j*jstride + k*kstride;
                const int ijk2 = i + j*jstride + (k-kstart_buffer)*kstride;
                at[ijk] -= sigmaz[k] * (a[ijk]-abuf[ijk2]);
            }
        }
    }
}

template<typename TF>
void Buffer<TF>::prepare_device()
{
    const Grid_data<TF>& gd = grid.get_grid_data();

    if (swbuffer)
    {
        // This buffer sub-feature is not (yet) ported to the GPU. Throw a clear
        // error rather than silently producing wrong results on stale host data.
        if (swupdate_local)
            throw std::runtime_error("Local-mean buffer (\"swupdate_local\") is not (yet) implemented on the GPU...");

        const int nmemsize = gd.kcells*sizeof(TF);

        // Allocate the buffer arrays at GPU.
        if (swbuffer_3d)
        {
            // 3D buffer fields, sized to match the host `bufferprofs` arrays.
            for (auto& fld : buffer3d_list)
            {
                const int ksize = fld == "w" ? gd.kend-bufferkstarth : gd.kend-bufferkstart;
                bufferprofs_g.emplace(fld, cuda_vector<TF>(gd.ijcells*ksize));
                cuda_safe_call(cudaMemcpy(
                        bufferprofs_g.at(fld), bufferprofs.at(fld).data(),
                        gd.ijcells*ksize*sizeof(TF), cudaMemcpyHostToDevice));
            }
        }
        else
        {
            for (auto& it : fields.ap)
            {
                bufferprofs_g.emplace(it.first, cuda_vector<TF>(gd.kcells));
                cuda_safe_call(cudaMemcpy(bufferprofs_g.at(it.first), bufferprofs.at(it.first).data(), nmemsize, cudaMemcpyHostToDevice));
            }
        }

        // Pre-calculate buffer factor.
        sigma_z.allocate(gd.kcells);
        sigma_zh.allocate(gd.kcells);

        auto tmp = fields.get_tmp();
        const TF zsizebufi = 1./(gd.zsize-zstart);

        // Calculate & copy to device.
        for (int k=bufferkstart; k<gd.kend; k++)
            tmp->fld_mean[k] = sigma * pow((gd.z[k]-zstart)*zsizebufi, beta);
        cuda_safe_call(cudaMemcpy(sigma_z, tmp->fld_mean.data(), nmemsize, cudaMemcpyHostToDevice));

        for (int k=bufferkstarth; k<gd.kend; k++)
            tmp->fld_mean[k] = sigma * pow((gd.zh[k]-zstart)*zsizebufi, beta);
        cuda_safe_call(cudaMemcpy(sigma_zh, tmp->fld_mean.data(), nmemsize, cudaMemcpyHostToDevice));
    }
}

template<typename TF>
void Buffer<TF>::clear_device()
{
}

#ifdef USECUDA
template<typename TF>
void Buffer<TF>::exec(Stats<TF>& stats)
{
    if (swbuffer)
    {
        const Grid_data<TF>& gd = grid.get_grid_data();

        if (swbuffer_3d)
        {
            // Sync the (time-interpolated) host buffer fields to the device. This
            // is updated each step in the shared `update_time_dependent()`.
            for (auto& fld : buffer3d_list)
            {
                const int kstart_buffer = fld == "w" ? bufferkstarth : bufferkstart;
                const int ksize = gd.kend - kstart_buffer;
                cuda_safe_call(cudaMemcpy(
                        bufferprofs_g.at(fld), bufferprofs.at(fld).data(),
                        gd.ijcells*ksize*sizeof(TF), cudaMemcpyHostToDevice));
            }

            const int blocki = gd.ithread_block;
            const int blockj = gd.jthread_block;
            const int gridi = gd.imax/blocki + (gd.imax%blocki > 0);
            const int gridj = gd.jmax/blockj + (gd.jmax%blockj > 0);
            dim3 gridGPU (gridi,  gridj,  1);
            dim3 blockGPU(blocki, blockj, 1);

            for (auto& fld : buffer3d_list)
            {
                const int kstart_buffer = fld == "w" ? bufferkstarth : bufferkstart;
                const cuda_vector<TF>& sigma = fld == "w" ? sigma_zh : sigma_z;

                calc_buffer_3d_g<TF><<<gridGPU, blockGPU>>>(
                        fields.at.at(fld)->fld_g,
                        fields.ap.at(fld)->fld_g,
                        bufferprofs_g.at(fld),
                        sigma,
                        gd.istart, gd.iend,
                        gd.jstart, gd.jend,
                        kstart_buffer, gd.kend,
                        gd.icells, gd.ijcells);
            }
            cuda_check_error();

            cudaDeviceSynchronize();
            for (auto& fld : buffer3d_list)
                stats.calc_tend(*fields.at.at(fld), tend_name);

            return;
        }

        Grid_layout grid_layout_full = {
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                bufferkstart, gd.kend,
                gd.istride,
                gd.jstride,
                gd.kstride};

        Grid_layout grid_layout_half = {
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                bufferkstarth, gd.kend,
                gd.istride,
                gd.jstride,
                gd.kstride};

        if (swupdate)
        {
            launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                    grid_layout_full,
                    fields.mt.at("u")->fld_g.view(),
                    fields.mp.at("u")->fld_g,
                    fields.mp.at("u")->fld_mean_g,
                    sigma_z);

            launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                    grid_layout_full,
                    fields.mt.at("v")->fld_g.view(),
                    fields.mp.at("v")->fld_g,
                    fields.mp.at("v")->fld_mean_g,
                    sigma_z);

            launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                    grid_layout_half,
                    fields.mt.at("w")->fld_g.view(),
                    fields.mp.at("w")->fld_g,
                    fields.mp.at("w")->fld_mean_g,
                    sigma_zh);

            for (auto& it : fields.sp)
                launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                        grid_layout_full,
                        fields.st.at(it.first)->fld_g.view(),
                        fields.sp.at(it.first)->fld_g,
                        fields.sp.at(it.first)->fld_mean_g,
                        sigma_z);
        }
        else
        {
            launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                    grid_layout_full,
                    fields.mt.at("u")->fld_g.view(),
                    fields.mp.at("u")->fld_g,
                    bufferprofs_g.at("u"),
                    sigma_z);

            launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                    grid_layout_full,
                    fields.mt.at("v")->fld_g.view(),
                    fields.mp.at("v")->fld_g,
                    bufferprofs_g.at("v"),
                    sigma_z);

            launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                    grid_layout_half,
                    fields.mt.at("w")->fld_g.view(),
                    fields.mp.at("w")->fld_g,
                    bufferprofs_g.at("w"),
                    sigma_zh);

            for (auto& it : fields.sp)
                launch_grid_kernel<Buffer_kernels::buffer_g<TF>>(
                        grid_layout_full,
                        fields.st.at(it.first)->fld_g.view(),
                        fields.sp.at(it.first)->fld_g,
                        bufferprofs_g.at(it.first),
                        sigma_z);
        }

        cudaDeviceSynchronize();
        stats.calc_tend(*fields.mt.at("u"), tend_name);
        stats.calc_tend(*fields.mt.at("v"), tend_name);
        stats.calc_tend(*fields.mt.at("w"), tend_name);
        for (auto it : fields.st)
            stats.calc_tend(*it.second, tend_name);
    }
}
#endif


#ifdef FLOAT_SINGLE
template class Buffer<float>;
#else
template class Buffer<double>;
#endif
