/*
 * MicroHH
 * Copyright (c) 2011-2024 MicroHH contributors
 *
 * GPU implementation of prescribed lateral boundaries and their sponge layer.
 */

#include "boundary_lateral.h"

#include <algorithm>

#include "fields.h"
#include "grid.h"
#include "master.h"
#include "stats.h"
#include "timeloop.h"
#include "tools.h"

namespace
{
    template<typename TF>
    __device__ TF horizontal_diffusion(
            const TF* const fld, const TF target, const int ijk,
            const int jstride)
    {
        const TF center = target-fld[ijk];
        return (target-fld[ijk-1]) + (target-fld[ijk+1])
             + (target-fld[ijk-jstride]) + (target-fld[ijk+jstride])
             - TF(4)*center;
    }

    template<typename TF, Lbc_location location>
    __global__ void set_lbc_ghost_cells_g(
            TF* const fld, const TF* const lbc,
            const int ngc, const int nsponge,
            const int iend, const int jend,
            const int kstart, const int kend,
            const int icells, const int jcells)
    {
        const int a = blockIdx.x*blockDim.x + threadIdx.x;
        const int b = blockIdx.y*blockDim.y + threadIdx.y;
        const int k = blockIdx.z + kstart-1;
        if (k >= kend)
            return;

        const int kstride = icells*jcells;
        if constexpr (location == Lbc_location::West || location == Lbc_location::East)
        {
            const int i = a;
            const int j = b;
            if (i >= ngc || j >= jcells)
                return;

            const int lbc_jstride = ngc+nsponge;
            const int lbc_i = location == Lbc_location::West ? i : i+nsponge;
            const int fld_i = location == Lbc_location::West ? i : i+iend;
            const int source_k = k == kstart-1 ? kstart : k;
            fld[fld_i + j*icells + k*kstride] =
                    lbc[lbc_i + j*lbc_jstride + source_k*lbc_jstride*jcells];
        }
        else
        {
            const int i = a;
            const int j = b;
            if (i >= icells || j >= ngc)
                return;

            const int lbc_j = location == Lbc_location::South ? j : j+nsponge;
            const int fld_j = location == Lbc_location::South ? j : j+jend;
            const int source_k = k == kstart-1 ? kstart : k;
            fld[i + fld_j*icells + k*kstride] =
                    lbc[i + lbc_j*icells + source_k*icells*(ngc+nsponge)];
        }
    }

    template<typename TF>
    __global__ void set_w_top_g(
            TF* const w, const TF* const w_top,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int k, const int icells, const int ijcells)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
        if (i < iend && j < jend)
            w[i+j*icells+k*ijcells] = w_top[i+j*icells];
    }

    template<typename TF, Lbc_location location>
    __global__ void set_w_neumann_g(
            TF* const w,
            const int istart, const int iend, const int igc,
            const int jstart, const int jend, const int jgc,
            const int kstart, const int kend,
            const int icells, const int ijcells)
    {
        const int tangential = blockIdx.x*blockDim.x + threadIdx.x;
        const int ghost = blockIdx.y*blockDim.y + threadIdx.y;
        const int k = blockIdx.z+kstart;
        if (k >= kend)
            return;

        if constexpr (location == Lbc_location::West || location == Lbc_location::East)
        {
            const int j = tangential+jstart;
            if (j >= jend || ghost >= igc)
                return;
            const int source_i = location == Lbc_location::West ? istart : iend-1;
            const int target_i = location == Lbc_location::West ? istart-1-ghost : iend+ghost;
            w[target_i+j*icells+k*ijcells] = w[source_i+j*icells+k*ijcells];
        }
        else
        {
            const int i = tangential+istart;
            if (i >= iend || ghost >= jgc)
                return;
            const int source_j = location == Lbc_location::South ? jstart : jend-1;
            const int target_j = location == Lbc_location::South ? jstart-1-ghost : jend+ghost;
            w[i+target_j*icells+k*ijcells] = w[i+source_j*icells+k*ijcells];
        }
    }

    template<typename TF, bool west, bool south>
    __global__ void set_w_corner_g(
            TF* const w,
            const int istart, const int iend, const int igc,
            const int jstart, const int jend, const int jgc,
            const int kstart, const int kend,
            const int icells, const int ijcells)
    {
        const int di = blockIdx.x*blockDim.x+threadIdx.x+1;
        const int dj = blockIdx.y*blockDim.y+threadIdx.y+1;
        const int k = blockIdx.z+kstart;
        if (di > igc || dj > jgc || k >= kend)
            return;

        const int i0 = west ? istart : iend-1;
        const int j0 = south ? jstart : jend-1;
        const int ijk0 = i0+j0*icells+k*ijcells;
        const TF dfdi = west ? w[ijk0]-w[ijk0-1] : w[ijk0+1]-w[ijk0];
        const TF dfdj = south ? w[ijk0]-w[ijk0-icells] : w[ijk0+icells]-w[ijk0];
        const int i = west ? istart-di : iend-1+di;
        const int j = south ? jstart-dj : jend-1+dj;
        w[i+j*icells+k*ijcells] = w[ijk0]
                +(west ? -TF(di) : TF(di))*dfdi
                +(south ? -TF(dj) : TF(dj))*dfdj;
    }

    template<typename TF>
    __global__ void extend_w_corner_vertical_g(
            TF* const w,
            const int istart, const int iend, const int igc,
            const int jstart, const int jend, const int jgc,
            const int kstart, const int kend, const int kcells,
            const int icells, const int ijcells)
    {
        const int corner_cell = blockIdx.x*blockDim.x+threadIdx.x;
        const int k = blockIdx.y;
        const int corner_size = igc*jgc;
        if (corner_cell >= 4*corner_size || k >= kcells || (k >= kstart && k < kend))
            return;

        const int corner = corner_cell/corner_size;
        const int local = corner_cell%corner_size;
        const int di = local%igc+1;
        const int dj = local/igc+1;
        const bool west = corner == 0 || corner == 2;
        const bool south = corner < 2;
        const int i = west ? istart-di : iend-1+di;
        const int j = south ? jstart-dj : jend-1+dj;
        const int source_k = k < kstart ? kstart : kend-1;
        w[i+j*icells+k*ijcells] = w[i+j*icells+source_k*ijcells];
    }

    template<typename TF, Lbc_location location, bool recycle, bool staggered_normal>
    __global__ void lateral_sponge_g(
            TF* const tendency, const TF* const fld, const TF* const lbc,
            const TF tau_sponge, const TF w_diff,
            const int nsponge, const TF tau_recycle, const int recycle_offset,
            const int istart, const int iend, const int igc,
            const int jstart, const int jend, const int jgc,
            const int kstart, const int kend,
            const int icells, const int jcells, const int ijcells)
    {
        const int tangential = blockIdx.x*blockDim.x + threadIdx.x;
        const int n = blockIdx.y*blockDim.y + threadIdx.y + 1;
        const int k = blockIdx.z+kstart;
        if (n > nsponge || k >= kend)
            return;

        int i, j, ijk_lbc;
        if constexpr (location == Lbc_location::West || location == Lbc_location::East)
        {
            j = tangential+jstart;
            if (j >= jend || (staggered_normal && n == 1 && location == Lbc_location::West))
                return;
            i = location == Lbc_location::West ? istart+n-1 : iend-n;
            const int ngc = igc + ((staggered_normal && location == Lbc_location::West) ? 1 : 0);
            const int ilbc = location == Lbc_location::West ? igc+n-1 : nsponge-n;
            ijk_lbc = ilbc+j*(ngc+nsponge)+k*(ngc+nsponge)*jcells;
        }
        else
        {
            i = tangential+istart;
            if (i >= iend || (staggered_normal && n == 1))
                return;
            j = location == Lbc_location::South
                    ? jstart+n-1
                    : jend-n+(staggered_normal ? 1 : 0);
            const int ngc = jgc + ((staggered_normal && location == Lbc_location::South) ? 1 : 0);
            const int jlbc = location == Lbc_location::South ? jgc+n-1 : nsponge-n;
            ijk_lbc = i+jlbc*icells+k*icells*(ngc+nsponge);
        }

        // Generic fields partition corner triangles between orthogonal edges.
        // Normal staggered momentum follows the CPU path and spans the full
        // tangential range; its orthogonal sponge is added in a later launch.
        if constexpr (!staggered_normal)
        {
            if ((location == Lbc_location::West || location == Lbc_location::East)
                    && (j < jstart+n-1 || j >= jend-(n-1)))
                return;
            if ((location == Lbc_location::South || location == Lbc_location::North)
                    && (i < istart+n || i >= iend-n))
                return;
        }

        const int ijk = i+j*icells+k*ijcells;
        const TF target = lbc[ijk_lbc];
        const TF factor = staggered_normal
                ? TF(1+nsponge-n)/nsponge
                : TF(1+nsponge-(n+TF(.5)))/nsponge;
        tendency[ijk] += factor/tau_sponge*(target-fld[ijk]);
        tendency[ijk] -= factor*w_diff*horizontal_diffusion(fld, target, ijk, icells);

        if constexpr (recycle)
        {
            const int offset = (location == Lbc_location::West || location == Lbc_location::South)
                    ? recycle_offset : -recycle_offset;
            const int source = (location == Lbc_location::West || location == Lbc_location::East)
                    ? ijk+offset : ijk+offset*icells;
            TF mean = TF(0);
            for (int jc=-3; jc<=3; ++jc)
                for (int ic=-3; ic<=3; ++ic)
                    mean += fld[source+ic+jc*icells];
            mean /= TF(49);
            tendency[ijk] += (TF(1)-factor)/tau_recycle
                    *((fld[source]-mean)-(fld[ijk]-target));
        }
    }

    template<typename TF>
    void allocate_map(const Lbc_map<TF>& host, std::map<std::string, cuda_vector<TF>>& device)
    {
        for (const auto& item : host)
            device.emplace(item.first, cuda_vector<TF>(item.second.size()));
    }

    template<typename TF>
    void copy_map(const Lbc_map<TF>& host, std::map<std::string, cuda_vector<TF>>& device)
    {
        for (const auto& item : host)
            cuda_copy(item.second.data(), device.at(item.first).data(), item.second.size());
    }
}

#ifdef USECUDA
template<typename TF>
void Boundary_lateral<TF>::prepare_device()
{
    if (!sw_openbc)
        return;
    allocate_map(lbc_w, lbc_w_g);
    allocate_map(lbc_e, lbc_e_g);
    allocate_map(lbc_s, lbc_s_g);
    allocate_map(lbc_n, lbc_n_g);
    if (sw_openbc_uv)
        w_top_2d_g.allocate(w_top_2d.size());
    forward_device();
}

template<typename TF>
void Boundary_lateral<TF>::forward_device()
{
    if (!sw_openbc || lbc_w_g.empty() && lbc_e_g.empty() && lbc_s_g.empty() && lbc_n_g.empty())
        return;
    copy_map(lbc_w, lbc_w_g);
    copy_map(lbc_e, lbc_e_g);
    copy_map(lbc_s, lbc_s_g);
    copy_map(lbc_n, lbc_n_g);
    if (sw_openbc_uv)
        cuda_copy(w_top_2d.data(), w_top_2d_g.data(), w_top_2d.size());
}

template<typename TF>
void Boundary_lateral<TF>::clear_device()
{
    lbc_w_g.clear();
    lbc_e_g.clear();
    lbc_s_g.clear();
    lbc_n_g.clear();
    w_top_2d_g.allocate(0);
}

template<typename TF>
void Boundary_lateral<TF>::set_ghost_cells(Timeloop<TF>&)
{
    if (!sw_openbc)
        return;
    const auto& gd = grid.get_grid_data();
    const auto& md = master.get_MPI_data();
    const dim3 block(16, 16);

    auto set_edge = [&](const std::string& name, const Lbc_location edge)
    {
        int ngc = (edge == Lbc_location::West || edge == Lbc_location::East) ? gd.igc : gd.jgc;
        if (name == "u" && edge == Lbc_location::West) ++ngc;
        if (name == "v" && edge == Lbc_location::South) ++ngc;
        const bool x_edge = edge == Lbc_location::West || edge == Lbc_location::East;
        const dim3 blocks((x_edge ? ngc : gd.icells)+15 >> 4,
                          (x_edge ? gd.jcells : ngc)+15 >> 4,
                          gd.kend-gd.kstart+1);
        TF* field = fields.ap.at(name)->fld_g;
        if (edge == Lbc_location::West)
            set_lbc_ghost_cells_g<TF, Lbc_location::West><<<blocks, block>>>(field, lbc_w_g.at(name), ngc, n_sponge, gd.iend, gd.jend, gd.kstart, gd.kend, gd.icells, gd.jcells);
        else if (edge == Lbc_location::East)
            set_lbc_ghost_cells_g<TF, Lbc_location::East><<<blocks, block>>>(field, lbc_e_g.at(name), ngc, n_sponge, gd.iend, gd.jend, gd.kstart, gd.kend, gd.icells, gd.jcells);
        else if (edge == Lbc_location::South)
            set_lbc_ghost_cells_g<TF, Lbc_location::South><<<blocks, block>>>(field, lbc_s_g.at(name), ngc, n_sponge, gd.iend, gd.jend, gd.kstart, gd.kend, gd.icells, gd.jcells);
        else
            set_lbc_ghost_cells_g<TF, Lbc_location::North><<<blocks, block>>>(field, lbc_n_g.at(name), ngc, n_sponge, gd.iend, gd.jend, gd.kstart, gd.kend, gd.icells, gd.jcells);
    };

    auto set_all_edges = [&](const std::string& name)
    {
        if (md.mpicoordx == 0) set_edge(name, Lbc_location::West);
        if (md.mpicoordx == md.npx-1) set_edge(name, Lbc_location::East);
        if (md.mpicoordy == 0) set_edge(name, Lbc_location::South);
        if (md.mpicoordy == md.npy-1) set_edge(name, Lbc_location::North);
    };
    if (sw_openbc_uv) { set_all_edges("u"); set_all_edges("v"); }
    if (sw_openbc_w) set_all_edges("w");
    for (const auto& name : slist) set_all_edges(name);

    if (sw_openbc_uv)
    {
        const dim3 top_blocks((gd.imax+15)/16, (gd.jmax+15)/16);
        set_w_top_g<TF><<<top_blocks, block>>>(fields.mp.at("w")->fld_g, w_top_2d_g,
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kend, gd.icells, gd.ijcells);
    }

    if (sw_neumann_w)
    {
        const dim3 bx((gd.jmax+15)/16, (gd.igc+15)/16, gd.kend+1-gd.kstart);
        const dim3 by((gd.imax+15)/16, (gd.jgc+15)/16, gd.kend+1-gd.kstart);
        TF* w = fields.mp.at("w")->fld_g;
        if (md.mpicoordx == 0) set_w_neumann_g<TF, Lbc_location::West><<<bx, block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        if (md.mpicoordx == md.npx-1) set_w_neumann_g<TF, Lbc_location::East><<<bx, block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        if (md.mpicoordy == 0) set_w_neumann_g<TF, Lbc_location::South><<<by, block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        if (md.mpicoordy == md.npy-1) set_w_neumann_g<TF, Lbc_location::North><<<by, block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);

        const dim3 corner_block(8, 8);
        const dim3 corner_blocks((gd.igc+7)/8, (gd.jgc+7)/8, gd.kend+1-gd.kstart);
        if (md.mpicoordx == 0 && md.mpicoordy == 0)
            set_w_corner_g<TF, true, true><<<corner_blocks, corner_block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        if (md.mpicoordx == md.npx-1 && md.mpicoordy == 0)
            set_w_corner_g<TF, false, true><<<corner_blocks, corner_block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        if (md.mpicoordx == 0 && md.mpicoordy == md.npy-1)
            set_w_corner_g<TF, true, false><<<corner_blocks, corner_block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);
        if (md.mpicoordx == md.npx-1 && md.mpicoordy == md.npy-1)
            set_w_corner_g<TF, false, false><<<corner_blocks, corner_block>>>(w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc, gd.kstart, gd.kend+1, gd.icells, gd.ijcells);

        const int corner_cells = 4*gd.igc*gd.jgc;
        extend_w_corner_vertical_g<TF><<<dim3((corner_cells+255)/256, gd.kcells), 256>>>(
                w, gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc,
                gd.kstart, gd.kend+1, gd.kcells, gd.icells, gd.ijcells);
    }
    cuda_check_error();
}

template<typename TF>
void Boundary_lateral<TF>::exec_lateral_sponge(Stats<TF>& stats)
{
    if (!sw_openbc || !sw_sponge)
        return;
    const auto& gd = grid.get_grid_data();
    const auto& md = master.get_MPI_data();
    const dim3 block(16, 8);

    auto launch = [&]<Lbc_location edge, bool recycle, bool staggered>(const std::string& name)
    {
        const bool x_edge = edge == Lbc_location::West || edge == Lbc_location::East;
        const int tangential = x_edge ? gd.jmax : gd.imax;
        const int kstart = name == "w" ? gd.kstart+1 : gd.kstart;
        const dim3 blocks((tangential+15)/16, (n_sponge+7)/8, gd.kend-kstart);
        const auto& map = edge == Lbc_location::West ? lbc_w_g
                        : edge == Lbc_location::East ? lbc_e_g
                        : edge == Lbc_location::South ? lbc_s_g : lbc_n_g;
        lateral_sponge_g<TF, edge, recycle, staggered><<<blocks, block>>>(
                fields.at.at(name)->fld_g, fields.ap.at(name)->fld_g, map.at(name),
                tau_sponge, w_diff, n_sponge, tau_recycle, recycle_offset,
                gd.istart, gd.iend, gd.igc, gd.jstart, gd.jend, gd.jgc,
                kstart, gd.kend, gd.icells, gd.jcells, gd.ijcells);
    };

    auto launch_generic = [&]<Lbc_location edge>(const std::string& name, const bool recycle)
    {
        if (recycle) launch.template operator()<edge, true, false>(name);
        else launch.template operator()<edge, false, false>(name);
    };

    if (sw_openbc_uv)
    {
        if (md.mpicoordx == 0) launch.template operator()<Lbc_location::West, false, true>("u");
        if (md.mpicoordx == md.npx-1) launch.template operator()<Lbc_location::East, false, true>("u");
        if (md.mpicoordy == 0) launch_generic.template operator()<Lbc_location::South>("u", false);
        if (md.mpicoordy == md.npy-1) launch_generic.template operator()<Lbc_location::North>("u", false);
        if (md.mpicoordy == 0) launch.template operator()<Lbc_location::South, false, true>("v");
        if (md.mpicoordy == md.npy-1) launch.template operator()<Lbc_location::North, false, true>("v");
        if (md.mpicoordx == 0) launch_generic.template operator()<Lbc_location::West>("v", false);
        if (md.mpicoordx == md.npx-1) launch_generic.template operator()<Lbc_location::East>("v", false);
        stats.calc_tend(*fields.mt.at("u"), tend_name);
        stats.calc_tend(*fields.mt.at("v"), tend_name);
    }
    if (sw_openbc_w)
    {
        if (md.mpicoordx == 0) launch_generic.template operator()<Lbc_location::West>("w", false);
        if (md.mpicoordx == md.npx-1) launch_generic.template operator()<Lbc_location::East>("w", false);
        if (md.mpicoordy == 0) launch_generic.template operator()<Lbc_location::South>("w", false);
        if (md.mpicoordy == md.npy-1) launch_generic.template operator()<Lbc_location::North>("w", false);
        stats.calc_tend(*fields.mt.at("w"), tend_name);
    }
    for (const auto& name : slist)
    {
        const bool eligible = std::find(recycle_list.begin(), recycle_list.end(), name) != recycle_list.end();
        if (md.mpicoordx == 0) launch_generic.template operator()<Lbc_location::West>(name, eligible && sw_recycle[Lbc_location::West]);
        if (md.mpicoordx == md.npx-1) launch_generic.template operator()<Lbc_location::East>(name, eligible && sw_recycle[Lbc_location::East]);
        if (md.mpicoordy == 0) launch_generic.template operator()<Lbc_location::South>(name, eligible && sw_recycle[Lbc_location::South]);
        if (md.mpicoordy == md.npy-1) launch_generic.template operator()<Lbc_location::North>(name, eligible && sw_recycle[Lbc_location::North]);
        stats.calc_tend(*fields.at.at(name), tend_name);
    }
    cuda_check_error();
}
#endif

#ifdef FLOAT_SINGLE
template class Boundary_lateral<float>;
#else
template class Boundary_lateral<double>;
#endif
