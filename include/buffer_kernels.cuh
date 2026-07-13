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

#ifndef BUFFER_KERNELS_CUH
#define BUFFER_KERNELS_CUH

#include "finite_difference.h"
#include "cuda_tiling.h"

namespace Buffer_kernels
{
    using namespace Finite_difference::O2;

    template<typename TF>
    struct buffer_g
    {
        DEFINE_GRID_KERNEL("buffer::buffer_g", 0)

        template <typename Level>
        CUDA_DEVICE
        void operator()(
                Grid_layout g, const int i, const int j, const int k, const Level level,
                TF* const __restrict__ at,
                const TF* const __restrict__ a,
                const TF* const __restrict__ abuf,
                const TF* const __restrict__ sigma_z)
        {
            const int ijk = g(i, j, k);
            at[ijk] -= sigma_z[k] * (a[ijk]-abuf[k]);
        }
    };

    template<typename TF>
    struct buffer_3d_g
    {
        DEFINE_GRID_KERNEL("buffer::buffer_3d_g", 0)

        template <typename Level>
        CUDA_DEVICE
        void operator()(
                Grid_layout g, const int i, const int j, const int k, const Level,
                TF* const __restrict__ at,
                const TF* const __restrict__ a,
                const TF* const __restrict__ abuf,
                const TF* const __restrict__ sigma_z,
                const int kstart_buffer,
                const int jstride_buffer,
                const int kstride_buffer)
        {
            const int ijk = g(i, j, k);
            const int ijk_buffer = i + j*jstride_buffer + (k-kstart_buffer)*kstride_buffer;
            at[ijk] -= sigma_z[k] * (a[ijk]-abuf[ijk_buffer]);
        }
    };

    template<typename TF>
    struct buffer_local_g
    {
        DEFINE_GRID_KERNEL("buffer::buffer_local_g", 3)

        template <typename Level>
        CUDA_DEVICE
        void operator()(
                Grid_layout g, const int i, const int j, const int k, const Level,
                TF* const __restrict__ at,
                const TF* const __restrict__ a,
                const TF* const __restrict__ sigma_z)
        {
            const int ijk = g(i, j, k);
            TF local_mean = TF(0);

            for (int jc=-3; jc<=3; ++jc)
                for (int ic=-3; ic<=3; ++ic)
                    local_mean += a[ijk + ic*g.istride + jc*g.jstride];

            local_mean /= TF(49);
            at[ijk] -= sigma_z[k] * (a[ijk]-local_mean);
        }
    };
}
#endif // BUFFER_KERNELS_CUH
