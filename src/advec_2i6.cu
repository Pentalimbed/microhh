/*
 * MicroHH
 * Copyright (c) 2011-2023 Chiel van Heerwaarden
 * Copyright (c) 2011-2023 Thijs Heus
 * Copyright (c) 2014-2023 Bart van Stratum
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

#include "advec_2i6.h"
#include "advec_2i5_kernels.cuh"
#include "grid.h"
#include "fields.h"
#include "stats.h"
#include "tools.h"
#include "constants.h"
#include "finite_difference.h"
#include "field3d_operators.h"
#include "cuda_launcher.h"


#ifdef USECUDA
template<typename TF>
unsigned long Advec_2i6<TF>::get_time_limit(unsigned long idt, double dt)
{
    double cfl = std::max(cflmin, get_cfl(dt));
    return idt*cflmax/cfl;
}


template<typename TF>
double Advec_2i6<TF>::get_cfl(const double dt)
{
    const auto& gd = grid.get_grid_data();
    auto tmp = fields.get_tmp_g();
    Grid_layout layout = {gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                          gd.istride, gd.jstride, gd.kstride};
    launch_grid_kernel<Advec_2i5_kernels::calc_cfl_g<TF>>(
            layout, tmp->fld_g.view(),
            fields.mp.at("u")->fld_g, fields.mp.at("v")->fld_g, fields.mp.at("w")->fld_g,
            gd.dzi_g, gd.dxi, gd.dyi);
    const TF cfl = field3d_operators.calc_max_g(tmp->fld_g)*dt;
    fields.release_tmp_g(tmp);
    return static_cast<double>(cfl);
}


template<typename TF>
void Advec_2i6<TF>::exec(Stats<TF>& stats)
{
    const auto& gd = grid.get_grid_data();
    Grid_layout layout = {gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                          gd.istride, gd.jstride, gd.kstride};
    const TF centered = TF(0);
    const TF closed_top = TF(0);

    launch_grid_kernel<Advec_2i5_kernels::advec_u_g<TF>>(
            layout, fields.mt.at("u")->fld_g.view(),
            fields.mp.at("u")->fld_g, fields.mp.at("v")->fld_g, fields.mp.at("w")->fld_g,
            fields.rhorefi_g, fields.rhorefh_g, gd.dzi_g, gd.dxi, gd.dyi, centered, closed_top);
    launch_grid_kernel<Advec_2i5_kernels::advec_v_g<TF>>(
            layout, fields.mt.at("v")->fld_g.view(),
            fields.mp.at("u")->fld_g, fields.mp.at("v")->fld_g, fields.mp.at("w")->fld_g,
            fields.rhorefi_g, fields.rhorefh_g, gd.dzi_g, gd.dxi, gd.dyi, centered, closed_top);
    launch_grid_kernel<Advec_2i5_kernels::advec_w_g<TF>>(
            layout, fields.mt.at("w")->fld_g.view(),
            fields.mp.at("u")->fld_g, fields.mp.at("v")->fld_g, fields.mp.at("w")->fld_g,
            fields.rhoref_g, fields.rhorefhi_g, gd.dzhi_g, gd.dxi, gd.dyi, centered);
    for (auto& item : fields.st)
        launch_grid_kernel<Advec_2i5_kernels::advec_s_g<TF>>(
                layout, item.second->fld_g.view(), fields.sp.at(item.first)->fld_g,
                fields.mp.at("u")->fld_g, fields.mp.at("v")->fld_g, fields.mp.at("w")->fld_g,
                fields.rhorefi_g, fields.rhorefh_g, gd.dzi_g, gd.dxi, gd.dyi, centered, closed_top);

    cudaDeviceSynchronize();
    stats.calc_tend(*fields.mt.at("u"), tend_name);
    stats.calc_tend(*fields.mt.at("v"), tend_name);
    stats.calc_tend(*fields.mt.at("w"), tend_name);
    for (auto& item : fields.st)
        stats.calc_tend(*item.second, tend_name);
}
#endif


#ifdef FLOAT_SINGLE
template class Advec_2i6<float>;
#else
template class Advec_2i6<double>;
#endif
