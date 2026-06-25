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
#include <algorithm>
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

template<typename TF>
void Buffer<TF>::prepare_device()
{
    const Grid_data<TF>& gd = grid.get_grid_data();

    if (swbuffer)
    {
        const int nmemsize = gd.kcells*sizeof(TF);

        // Allocate the buffer arrays at GPU.
        if (swbuffer_3d)
        {
            for (auto& fld : buffer3d_list)
            {
                const int ksize = fld == "w" ? gd.kend-bufferkstarth : gd.kend-bufferkstart;
                const int ncells = gd.ijcells * ksize;

                bufferprofs_g.emplace(fld, cuda_vector<TF>(ncells));
                cuda_safe_call(cudaMemcpy(
                        bufferprofs_g.at(fld), bufferprofs.at(fld).data(),
                        ncells*sizeof(TF), cudaMemcpyHostToDevice));
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

        fields.release_tmp(tmp);
    }
}

template<typename TF>
void Buffer<TF>::clear_device()
{
    bufferprofs_g.clear();
    sigma_z.free();
    sigma_zh.free();
}

#ifdef USECUDA
template<typename TF>
void Buffer<TF>::exec(Stats<TF>& stats)
{
    if (swbuffer)
    {
        const Grid_data<TF>& gd = grid.get_grid_data();

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

        if (swbuffer_3d)
        {
            for (auto& fld : buffer3d_list)
            {
                const int kstart = fld == "w" ? bufferkstarth : bufferkstart;
                const Grid_layout grid_layout_buffer = {
                        gd.istart, gd.iend,
                        gd.jstart, gd.jend,
                        kstart, gd.kend,
                        gd.istride,
                        gd.jstride,
                        gd.kstride};

                launch_grid_kernel<Buffer_kernels::buffer_3d_g<TF>>(
                        grid_layout_buffer,
                        fields.at.at(fld)->fld_g.view(),
                        fields.ap.at(fld)->fld_g,
                        bufferprofs_g.at(fld),
                        fld == "w" ? sigma_zh : sigma_z);
            }
        }
        else if (swupdate)
        {
            if (swupdate_local)
            {
                launch_grid_kernel<Buffer_kernels::buffer_local_g<TF>>(
                        grid_layout_full,
                        fields.mt.at("u")->fld_g.view(),
                        fields.mp.at("u")->fld_g,
                        sigma_z);

                launch_grid_kernel<Buffer_kernels::buffer_local_g<TF>>(
                        grid_layout_full,
                        fields.mt.at("v")->fld_g.view(),
                        fields.mp.at("v")->fld_g,
                        sigma_z);

                launch_grid_kernel<Buffer_kernels::buffer_local_g<TF>>(
                        grid_layout_half,
                        fields.mt.at("w")->fld_g.view(),
                        fields.mp.at("w")->fld_g,
                        sigma_zh);

                for (auto& it : fields.sp)
                    launch_grid_kernel<Buffer_kernels::buffer_local_g<TF>>(
                            grid_layout_full,
                            fields.st.at(it.first)->fld_g.view(),
                            fields.sp.at(it.first)->fld_g,
                            sigma_z);
            }
            else
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

template <typename TF>
void Buffer<TF>::update_time_dependent(
        Timeloop<TF>& timeloop)
{
    if (!(swbuffer_3d && swtimedep_buffer_3d))
        return;

    const Grid_data<TF>& gd = grid.get_grid_data();

    unsigned long itime = timeloop.get_itime();
    unsigned long iloadtime = convert_to_itime(loadfreq);
    unsigned long iiotimeprec = timeloop.get_iiotimeprec();

    if (itime > next_itime)
    {
        // Advance time and read new files.
        prev_itime = next_itime;
        next_itime = prev_itime + iloadtime;

        unsigned long next_iotime = next_itime / iiotimeprec;

        auto tmp1 = fields.get_tmp();
        auto tmp2 = fields.get_tmp();

        const TF no_offset = TF(0);
        int nerror = 0;

        // Read 3D buffers from binary files.
        auto load_3d_field = [&](
                TF* const restrict field,
                const std::string& name,
                const int itime,
                const int kstart,
                const int kend)
        {
            char filename[256];
            std::sprintf(filename, "%s_buffer.%07d", name.c_str(), itime);
            master.print_message("Loading \"%s\" ... ", filename);

            if (field3d_io.load_field3d(
                    field,
                    tmp1->fld.data(), tmp2->fld.data(),
                    filename, no_offset,
                    kstart, kend))
            {
                master.print_message("FAILED\n");
                nerror += 1;
            }
            else
                master.print_message("OK\n");
        };

        for (auto& fld : buffer3d_list)
        {
            // Copy old next to new prev buffer.
            buffer_data_prev.at(fld) = buffer_data_next.at(fld);

            // Read new 3D data.
            const int kstart = 0;
            const int ksize = fld == "w" ? gd.kend-bufferkstarth : gd.kend-bufferkstart;

            load_3d_field(
                    buffer_data_next.at(fld).data(),
                    fld, next_iotime, kstart, ksize);
        }

        fields.release_tmp(tmp1);
        fields.release_tmp(tmp2);

        if (nerror > 0)
            throw std::runtime_error("Error loading buffer fields.");
    }

    // Interpolate in time.
    const TF f0 = TF(1) - ((itime - prev_itime) / TF(iloadtime));
    const TF f1 = TF(1) - f0;

    for (auto& fld : buffer3d_list)
    {
        const int ksize = fld == "w" ? gd.kend-bufferkstarth : gd.kend-bufferkstart;
        const int ncells = ksize * gd.ijcells;

        for (int n=0; n<ncells; ++n)
            bufferprofs.at(fld)[n] =
                    f0 * buffer_data_prev.at(fld)[n] +
                    f1 * buffer_data_next.at(fld)[n];

        cuda_safe_call(cudaMemcpy(
                bufferprofs_g.at(fld), bufferprofs.at(fld).data(),
                ncells*sizeof(TF), cudaMemcpyHostToDevice));
    }
}
#endif


#ifdef FLOAT_SINGLE
template class Buffer<float>;
#else
template class Buffer<double>;
#endif
