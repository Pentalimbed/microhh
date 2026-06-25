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
#include <iostream>

#include "grid.h"
#include "fields.h"
#include "master.h"
#include "boundary.h"
#include "boundary_surface.h"
#include "defines.h"
#include "constants.h"
#include "thermo.h"
#include "tools.h"
#include "stats.h"
#include "monin_obukhov.h"
#include "fast_math.h"

#include "diff_smag2.h"
#include "diff_kernels.h"
#include "diff_kernels_anisotropic.h"
#include "diff_kernels.cuh"

// Kernel Launcher
#include "cuda_launcher.h"
#include "diff_smag2_kl_kernels.cuh"
#include "diff_kl_kernels.cuh"

namespace
{
    template<typename TF>
    void backward_tendencies_device(Fields<TF>& fields)
    {
        for (auto& it : fields.at)
            fields.backward_field_device_3d(it.second->fld.data(), it.second->fld_g);
    }

    template<typename TF>
    void forward_tendencies_device(Fields<TF>& fields)
    {
        for (auto& it : fields.at)
            fields.forward_field_device_3d(it.second->fld_g, it.second->fld.data());
    }

    template<typename TF>
    void forward_anisotropic_viscosity_device(Fields<TF>& fields)
    {
        fields.forward_field_device_3d(
                fields.sd.at("evisc_h")->fld_g,
                fields.sd.at("evisc_h")->fld.data());
        fields.forward_field_device_3d(
                fields.sd.at("evisc_v")->fld_g,
                fields.sd.at("evisc_v")->fld.data());
    }

    template<typename TF>
    void backward_anisotropic_viscosity_device(Fields<TF>& fields)
    {
        fields.backward_field_device_3d(
                fields.sd.at("evisc_h")->fld.data(),
                fields.sd.at("evisc_h")->fld_g);
        fields.backward_field_device_3d(
                fields.sd.at("evisc_v")->fld.data(),
                fields.sd.at("evisc_v")->fld_g);
    }

    template<typename TF>
    inline TF calc_l_mason_anisotropic(const TF mlen0, const TF z, const TF z0)
    {
        const TF n_mason = TF(2);
        return std::pow(
                TF(1.) / (TF(1.) / std::pow(mlen0, n_mason)
                        + TF(1.) / std::pow(Constants::kappa<TF> * (z + z0), n_mason)),
                TF(1.) / n_mason);
    }

    template<typename TF>
    void calc_evisc_anisotropic(
            TF* const restrict evisc_h,
            TF* const restrict evisc_v,
            const TF* const restrict N2,
            const TF* const restrict bgradbot,
            const TF* const restrict z,
            const TF* const restrict z0m,
            const TF mlen0_h, const TF mlen0_v,
            const TF cs, const TF tPr,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend,
            const int icells, const int ijcells,
            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int jj = icells;
        const int kk = ijcells;

        for (int j=jstart; j<jend; ++j)
        {
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;

                TF RitPrratio = bgradbot[ij] / evisc_h[ijk] / tPr;
                RitPrratio = std::min(RitPrratio, TF(1.-Constants::dsmall));

                const TF mlen_v = calc_l_mason_anisotropic(cs*mlen0_v, z[kstart], z0m[ij]);
                const TF mlen_h = calc_l_mason_anisotropic(cs*mlen0_h, z[kstart], z0m[ij]);
                const TF strain_ritpr_fac = std::sqrt(evisc_h[ijk]) * std::sqrt(TF(1.)-RitPrratio);

                evisc_v[ijk] = Fast_math::pow2(mlen_v) * strain_ritpr_fac;
                evisc_h[ijk] = Fast_math::pow2(mlen_h) * strain_ritpr_fac;
            }
        }

        for (int k=kstart+1; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + k*kk;

                    TF RitPrratio = N2[ijk] / evisc_h[ijk] / tPr;
                    RitPrratio = std::min(RitPrratio, TF(1.-Constants::dsmall));

                    const TF mlen_v = calc_l_mason_anisotropic(cs*mlen0_v, z[k], z0m[ij]);
                    const TF mlen_h = calc_l_mason_anisotropic(cs*mlen0_h, z[k], z0m[ij]);
                    const TF strain_ritpr_fac = std::sqrt(evisc_h[ijk]) * std::sqrt(TF(1.)-RitPrratio);

                    evisc_v[ijk] = Fast_math::pow2(mlen_v) * strain_ritpr_fac;
                    evisc_h[ijk] = Fast_math::pow2(mlen_h) * strain_ritpr_fac;
                }

        boundary_cyclic.exec(evisc_v);
        boundary_cyclic.exec(evisc_h);
    }
}

/* Calculate the mixing length (mlen) offline, and put on GPU */
#ifdef USECUDA
template<typename TF>
void Diff_smag2<TF>::prepare_device(Boundary<TF>& boundary)
{
    auto& gd = grid.get_grid_data();

    std::vector<TF> mlen(gd.kcells);

    if (boundary.get_switch() == "default")
    {
        for (int k=0; k<gd.kcells; ++k)
            mlen[k] = cs * pow(gd.dx*gd.dy*gd.dz[k], 1./3.);
    }
    else
    {
        const TF n_mason = TF(2);
        for (int k=0; k<gd.kcells; ++k)
            mlen[k] = std::pow(cs * std::pow(gd.dx*gd.dy*gd.dz[k], TF(1./3.)), n_mason);
    }

    mlen_g.allocate(gd.kcells);
    cuda_safe_call(cudaMemcpy(mlen_g, mlen.data(), mlen_g.size_in_bytes(), cudaMemcpyHostToDevice));
}

template<typename TF>
void Diff_smag2<TF>::clear_device()
{
    mlen_g.free();
}
#endif

#ifdef USECUDA
template<typename TF>
void Diff_smag2<TF>::exec_viscosity(Stats<TF>&, Thermo<TF>& thermo)
{
    namespace dk = Diff_kernels_g;
    auto& gd = grid.get_grid_data();

    if (sw_anisotropic)
    {
        fields.backward_device();
        boundary.backward_device(thermo);

        if (boundary.get_switch() != "default")
        {
            const std::vector<TF>& dudz = boundary.get_dudz();
            const std::vector<TF>& dvdz = boundary.get_dvdz();

            Diff_kernels::calc_strain2<TF, Surface_model::Enabled>(
                    fields.sd.at("evisc_h")->fld.data(),
                    fields.mp.at("u")->fld.data(),
                    fields.mp.at("v")->fld.data(),
                    fields.mp.at("w")->fld.data(),
                    dudz.data(),
                    dvdz.data(),
                    gd.z.data(),
                    gd.dzi.data(),
                    gd.dzhi.data(),
                    TF(1.)/gd.dx, TF(1.)/gd.dy,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        }
        else
        {
            Diff_kernels::calc_strain2<TF, Surface_model::Disabled>(
                    fields.sd.at("evisc_h")->fld.data(),
                    fields.mp.at("u")->fld.data(),
                    fields.mp.at("v")->fld.data(),
                    fields.mp.at("w")->fld.data(),
                    nullptr,
                    nullptr,
                    gd.z.data(),
                    gd.dzi.data(),
                    gd.dzhi.data(),
                    TF(1.)/gd.dx, TF(1.)/gd.dy,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        }

        auto buoy_tmp = fields.get_tmp();
        thermo.get_thermo_field(*buoy_tmp, "N2", false, false);

        const std::vector<TF>& z0m = boundary.get_z0m();
        const std::vector<TF>& dbdz = boundary.get_dbdz();

        calc_evisc_anisotropic<TF>(
                fields.sd.at("evisc_h")->fld.data(),
                fields.sd.at("evisc_v")->fld.data(),
                buoy_tmp->fld.data(),
                dbdz.data(),
                gd.z.data(),
                z0m.data(),
                mlen0_h, mlen0_v,
                cs, tPr,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells,
                boundary_cyclic);

        fields.release_tmp(buoy_tmp);
        forward_anisotropic_viscosity_device(fields);
        return;
    }

    // Grid layout struct for cuda launcher.
    Grid_layout grid_layout = {
            gd.istart, gd.iend,
            gd.jstart, gd.jend,
            gd.kstart, gd.kend,
            gd.istride,
            gd.jstride,
            gd.kstride};

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;
    const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
    const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);

    dim3 gridGPU (gridi, gridj, gd.kcells);
    dim3 blockGPU(blocki, blockj, 1);

    // Contain the full icells and jcells in this grid.
    const int grid2di  = gd.icells/blocki + (gd.icells%blocki > 0);
    const int grid2dj  = gd.jcells/blockj + (gd.jcells%blockj > 0);

    dim3 grid2dGPU (grid2di, grid2dj);
    dim3 block2dGPU(blocki, blockj);

    // Use surface model.
    if (boundary.get_switch() != "default")
    {
        auto& z0m_g   = boundary.get_z0m_g();

        // Get MO gradients velocity:
        auto& dudz_g  = boundary.get_dudz_g();
        auto& dvdz_g  = boundary.get_dvdz_g();

        // Calculate total strain rate
        launch_grid_kernel<Diff_les_kernels::calc_strain2_g<TF, true>>(
            grid_layout,
            fields.sd.at("evisc")->fld_g.view(),
            fields.mp.at("u")->fld_g,
            fields.mp.at("v")->fld_g,
            fields.mp.at("w")->fld_g,
            dudz_g, dvdz_g,
            gd.dzi_g, gd.dzhi_g,
            gd.dxi, gd.dyi);

        if (thermo.get_switch() == Thermo_type::Disabled)
        {
            // Start with retrieving the stability information
            Diff_smag2_kernels::evisc_neutral_g<TF><<<gridGPU, blockGPU>>>(
                fields.sd.at("evisc")->fld_g,
                z0m_g, gd.z_g, mlen_g,
                gd.istart, gd.jstart, gd.kstart,
                gd.iend,   gd.jend,   gd.kend,
                gd.icells, gd.ijcells);
            cuda_check_error();
        }
        else
        {
            // Assume buoyancy calculation is needed
            auto tmp1 = fields.get_tmp_g();
            thermo.get_thermo_field_g(*tmp1, "N2", false);

            // Get MO gradient buoyancy:
            auto& dbdz_g  = boundary.get_dbdz_g();

            // Calculate eddy viscosity
            TF tPri = 1./tPr;

            launch_grid_kernel<Diff_smag2_kernels::evisc_g<TF, true>>(
                grid_layout,
                fields.sd.at("evisc")->fld_g.view(),
                tmp1->fld_g, dbdz_g,
                mlen_g, z0m_g, gd.z_g,
                tPri);

            fields.release_tmp_g(tmp1);
        }

        boundary_cyclic.exec_g(fields.sd.at("evisc")->fld_g);
    }
    // Do not use surface model.
    else
    {
        // Calculate total strain rate
        launch_grid_kernel<Diff_les_kernels::calc_strain2_g<TF, false>>(
            grid_layout,
            fields.sd.at("evisc")->fld_g.view(),
            fields.mp.at("u")->fld_g,
            fields.mp.at("v")->fld_g,
            fields.mp.at("w")->fld_g,
            nullptr, nullptr,
            gd.dzi_g, gd.dzhi_g,
            gd.dxi, gd.dyi);

        // start with retrieving the stability information
        if (thermo.get_switch() == Thermo_type::Disabled)
        {
            Diff_smag2_kernels::evisc_neutral_vandriest_g<TF><<<gridGPU, blockGPU>>>(
                fields.sd.at("evisc")->fld_g,
                fields.mp.at("u")->fld_g,
                fields.mp.at("v")->fld_g,
                mlen_g, gd.z_g, gd.dzhi_g,
                gd.zsize, fields.visc,
                gd.istart, gd.jstart, gd.kstart,
                gd.iend, gd.jend, gd.kend,
                gd.icells, gd.ijcells);
            cuda_check_error();
        }
        // assume buoyancy calculation is needed
        else
        {
            // store the buoyancyflux in datafluxbot of tmp1
            auto tmp1 = fields.get_tmp_g();
            thermo.get_buoyancy_fluxbot_g(*tmp1);
            // As we only use the fluxbot field of tmp1 we store the N2 in the interior.
            thermo.get_thermo_field_g(*tmp1, "N2", false);

            // Calculate eddy viscosity
            TF tPri = 1./tPr;

            launch_grid_kernel<Diff_smag2_kernels::evisc_g<TF, true>>(
                grid_layout,
                fields.sd.at("evisc")->fld_g.view(),
                tmp1->fld_g, nullptr,
                mlen_g, nullptr, gd.z_g,
                tPri);

            fields.release_tmp_g(tmp1);
        }

        boundary_cyclic.exec_g(fields.sd.at("evisc")->fld_g);

        dk::calc_ghostcells_evisc<TF><<<grid2dGPU, block2dGPU>>>(
                fields.sd.at("evisc")->fld_g,
                gd.icells, gd.jcells,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);
    }

    cuda_check_error();
}
#endif

#ifdef USECUDA
template<typename TF>
void Diff_smag2<TF>::exec(Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    if (sw_anisotropic)
    {
        fields.backward_device();
        backward_tendencies_device(fields);

        Diff_kernels_anisotropic::diff_u<TF>(
                fields.mt.at("u")->fld.data(),
                fields.mp.at("u")->fld.data(),
                fields.mp.at("v")->fld.data(),
                fields.mp.at("w")->fld.data(),
                gd.dzi.data(), gd.dzhi.data(),
                TF(1.)/gd.dx, TF(1.)/gd.dy,
                fields.sd.at("evisc_h")->fld.data(),
                fields.sd.at("evisc_v")->fld.data(),
                fields.mp.at("u")->flux_bot.data(),
                fields.mp.at("u")->flux_top.data(),
                fields.rhoref.data(),
                fields.rhorefh.data(),
                fields.visc,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

        Diff_kernels_anisotropic::diff_v<TF>(
                fields.mt.at("v")->fld.data(),
                fields.mp.at("u")->fld.data(),
                fields.mp.at("v")->fld.data(),
                fields.mp.at("w")->fld.data(),
                gd.dzi.data(), gd.dzhi.data(),
                TF(1.)/gd.dx, TF(1.)/gd.dy,
                fields.sd.at("evisc_h")->fld.data(),
                fields.sd.at("evisc_v")->fld.data(),
                fields.mp.at("v")->flux_bot.data(),
                fields.mp.at("v")->flux_top.data(),
                fields.rhoref.data(),
                fields.rhorefh.data(),
                fields.visc,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

        Diff_kernels_anisotropic::diff_w<TF>(
                fields.mt.at("w")->fld.data(),
                fields.mp.at("u")->fld.data(),
                fields.mp.at("v")->fld.data(),
                fields.mp.at("w")->fld.data(),
                gd.dzi.data(), gd.dzhi.data(),
                TF(1.)/gd.dx, TF(1.)/gd.dy,
                fields.sd.at("evisc_h")->fld.data(),
                fields.sd.at("evisc_v")->fld.data(),
                fields.rhoref.data(),
                fields.rhorefh.data(),
                fields.visc,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

        for (auto it : fields.st)
        {
            Diff_kernels_anisotropic::diff_c<TF>(
                    it.second->fld.data(),
                    fields.sp.at(it.first)->fld.data(),
                    gd.dzi.data(), gd.dzhi.data(),
                    TF(1.)/(gd.dx*gd.dx), TF(1.)/(gd.dy*gd.dy),
                    fields.sd.at("evisc_h")->fld.data(),
                    fields.sd.at("evisc_v")->fld.data(),
                    fields.sp.at(it.first)->flux_bot.data(),
                    fields.sp.at(it.first)->flux_top.data(),
                    fields.rhoref.data(),
                    fields.rhorefh.data(),
                    tPr,
                    fields.sp.at(it.first)->visc,
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, gd.kend,
                    gd.icells, gd.ijcells);
        }

        forward_tendencies_device(fields);

        stats.calc_tend(*fields.mt.at("u"), tend_name);
        stats.calc_tend(*fields.mt.at("v"), tend_name);
        stats.calc_tend(*fields.mt.at("w"), tend_name);
        for (auto it : fields.st)
            stats.calc_tend(*it.second, tend_name);

        return;
    }

    // Grid layout struct for cuda launcher.
    Grid_layout grid_layout = {
            gd.istart, gd.iend,
            gd.jstart, gd.jend,
            gd.kstart, gd.kend,
            gd.istride,
            gd.jstride,
            gd.kstride};

    const TF dxidxi = TF(1)/(gd.dx * gd.dx);
    const TF dyidyi = TF(1)/(gd.dy * gd.dy);
    const TF tPri = TF(1)/tPr;

    // Do not use surface model.
    if (boundary.get_switch() == "default")
    {
        launch_grid_kernel<Diff_les_kernels::diff_uvw_g<TF, false>>(
                grid_layout,
                fields.mt.at("u")->fld_g.view(),
                fields.mt.at("v")->fld_g.view(),
                fields.mt.at("w")->fld_g.view(),
                fields.sd.at("evisc")->fld_g,
                fields.mp.at("u")->fld_g,
                fields.mp.at("v")->fld_g,
                fields.mp.at("w")->fld_g,
                fields.mp.at("u")->flux_bot_g,
                fields.mp.at("u")->flux_top_g,
                fields.mp.at("v")->flux_bot_g,
                fields.mp.at("v")->flux_top_g,
                gd.dzi_g, gd.dzhi_g,
                gd.dxi, gd.dyi,
                fields.rhoref_g, fields.rhorefh_g,
                fields.rhorefi_g, fields.rhorefhi_g,
                fields.visc);

        cuda_check_error();

        for (auto it : fields.st)
        {
            launch_grid_kernel<Diff_les_kernels::diff_c_g<TF, false>>(
                    grid_layout,
                    it.second->fld_g.view(),
                    fields.sp.at(it.first)->fld_g,
                    fields.sd.at("evisc")->fld_g,
                    fields.sp.at(it.first)->flux_bot_g,
                    fields.sp.at(it.first)->flux_top_g,
                    gd.dzi_g, gd.dzhi_g,
                    dxidxi, dyidyi,
                    fields.rhorefi_g, fields.rhorefh_g,
                    tPri, fields.sp.at(it.first)->visc);

            cuda_check_error();
        }
    }
    // Use surface model.
    else
    {
        launch_grid_kernel<Diff_les_kernels::diff_uvw_g<TF, true>>(
                grid_layout,
                fields.mt.at("u")->fld_g.view(),
                fields.mt.at("v")->fld_g.view(),
                fields.mt.at("w")->fld_g.view(),
                fields.sd.at("evisc")->fld_g,
                fields.mp.at("u")->fld_g,
                fields.mp.at("v")->fld_g,
                fields.mp.at("w")->fld_g,
                fields.mp.at("u")->flux_bot_g,
                fields.mp.at("u")->flux_top_g,
                fields.mp.at("v")->flux_bot_g,
                fields.mp.at("v")->flux_top_g,
                gd.dzi_g, gd.dzhi_g,
                gd.dxi, gd.dyi,
                fields.rhoref_g, fields.rhorefh_g,
                fields.rhorefi_g, fields.rhorefhi_g,
                fields.visc);

            cuda_check_error();

        for (auto it : fields.st)
        {
            launch_grid_kernel<Diff_les_kernels::diff_c_g<TF, true>>(
                    grid_layout,
                    it.second->fld_g.view(),
                    fields.sp.at(it.first)->fld_g,
                    fields.sd.at("evisc")->fld_g,
                    fields.sp.at(it.first)->flux_bot_g,
                    fields.sp.at(it.first)->flux_top_g,
                    gd.dzi_g, gd.dzhi_g,
                    dxidxi, dyidyi,
                    fields.rhorefi_g, fields.rhorefh_g,
                    tPri, fields.sp.at(it.first)->visc);

            cuda_check_error();
        }
    }

    cudaDeviceSynchronize();
    stats.calc_tend(*fields.mt.at("u"), tend_name);
    stats.calc_tend(*fields.mt.at("v"), tend_name);
    stats.calc_tend(*fields.mt.at("w"), tend_name);
    for (auto it : fields.st)
        stats.calc_tend(*it.second, tend_name);
}
#endif

#ifdef USECUDA
template<typename TF>
unsigned long Diff_smag2<TF>::get_time_limit(unsigned long idt, double dt)
{
    namespace dk = Diff_kernels_g;
    auto& gd = grid.get_grid_data();

    if (sw_anisotropic)
    {
        backward_anisotropic_viscosity_device(fields);

        double dnmul = Diff_kernels_anisotropic::calc_dnmul<TF>(
                fields.sd.at("evisc_h")->fld.data(),
                fields.sd.at("evisc_v")->fld.data(),
                gd.dzi.data(),
                TF(1.)/(gd.dx*gd.dx),
                TF(1.)/(gd.dy*gd.dy),
                tPr,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

        master.max(&dnmul, 1);
        dnmul = std::max(Constants::dsmall, dnmul);

        return idt * dnmax / (dt * dnmul);
    }

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;
    const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
    const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);

    dim3 gridGPU (gridi, gridj, gd.kmax);
    dim3 blockGPU(blocki, blockj, 1);

    const TF dxidxi = TF(1)/(gd.dx * gd.dx);
    const TF dyidyi = TF(1)/(gd.dy * gd.dy);
    const TF tPrfac_i = TF(1)/std::min(TF(1.), tPr);

    auto tmp1 = fields.get_tmp_g();

    // Calculate dnmul in tmp1 field
    dk::calc_dnmul_g<TF><<<gridGPU, blockGPU>>>(
            tmp1->fld_g,
            fields.sd.at("evisc")->fld_g,
            gd.dzi_g,
            tPrfac_i,
            dxidxi, dyidyi,
            gd.istart, gd.iend,
            gd.jstart, gd.jend,
            gd.kstart, gd.kend,
            gd.icells, gd.ijcells);

    cuda_check_error();

    // Get maximum from tmp1 field
    double dnmul = field3d_operators.calc_max_g(tmp1->fld_g);
    dnmul = std::max(Constants::dsmall, dnmul);

    const unsigned long idtlim = idt * dnmax/(dnmul*dt);

    fields.release_tmp_g(tmp1);

    return idtlim;
}
#endif

#ifdef USECUDA
template<typename TF>
double Diff_smag2<TF>::get_dn(double dt)
{
    namespace dk = Diff_kernels_g;
    auto& gd = grid.get_grid_data();

    if (sw_anisotropic)
    {
        backward_anisotropic_viscosity_device(fields);

        double dnmul = Diff_kernels_anisotropic::calc_dnmul<TF>(
                fields.sd.at("evisc_h")->fld.data(),
                fields.sd.at("evisc_v")->fld.data(),
                gd.dzi.data(),
                TF(1.)/(gd.dx*gd.dx),
                TF(1.)/(gd.dy*gd.dy),
                tPr,
                gd.istart, gd.iend,
                gd.jstart, gd.jend,
                gd.kstart, gd.kend,
                gd.icells, gd.ijcells);

        master.max(&dnmul, 1);
        return dnmul*dt;
    }

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;
    const int gridi  = gd.imax/blocki + (gd.imax%blocki > 0);
    const int gridj  = gd.jmax/blockj + (gd.jmax%blockj > 0);

    dim3 gridGPU (gridi, gridj, gd.kmax);
    dim3 blockGPU(blocki, blockj, 1);

    const TF dxidxi = TF(1)/(gd.dx * gd.dx);
    const TF dyidyi = TF(1)/(gd.dy * gd.dy);
    const TF tPrfac_i = TF(1)/std::min(TF(1.), tPr);

    // Calculate dnmul in tmp1 field
    auto dnmul_tmp = fields.get_tmp_g();

    dk::calc_dnmul_g<TF><<<gridGPU, blockGPU>>>(
            dnmul_tmp->fld_g,
            fields.sd.at("evisc")->fld_g,
            gd.dzi_g,
            tPrfac_i,
            dxidxi, dyidyi,
            gd.istart, gd.iend,
            gd.jstart, gd.jend,
            gd.kstart, gd.kend,
            gd.icells, gd.ijcells);

    cuda_check_error();

    // Get maximum from tmp1 field
    // CvH This is odd, because there might be need for calc_max in CPU version.
    double dnmul = field3d_operators.calc_max_g(dnmul_tmp->fld_g);

    fields.release_tmp_g(dnmul_tmp);

    return dnmul*dt;
}
#endif


#ifdef FLOAT_SINGLE
template class Diff_smag2<float>;
#else
template class Diff_smag2<double>;
#endif
