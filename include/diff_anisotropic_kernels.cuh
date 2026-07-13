/*
 * MicroHH
 * Copyright (c) 2011-2024 MicroHH contributors
 *
 * CUDA kernels for the anisotropic Smagorinsky model.
 */

#ifndef DIFF_ANISOTROPIC_KERNELS_CUH
#define DIFF_ANISOTROPIC_KERNELS_CUH

#include "constants.h"
#include "cuda_tiling.h"

namespace Diff_anisotropic_kernels
{
    template<typename TF>
    struct evisc_g
    {
        DEFINE_GRID_KERNEL("diff_smag2::evisc_anisotropic", 0)

        template<typename Level>
        CUDA_DEVICE void operator()(
                Grid_layout g, const int i, const int j, const int k, const Level level,
                TF* const evisc_h, TF* const evisc_v,
                const TF* const N2, const TF* const bgradbot,
                const TF* const z, const TF* const z0m,
                const TF mlen0_h, const TF mlen0_v,
                const TF cs, const TF tPr)
        {
            const int ij = i+j*g.jstride;
            const int ijk = g(i, j, k);
            const TF strain2 = evisc_h[ijk];
            const TF buoyancy_gradient = level.distance_to_start() == 0 ? bgradbot[ij] : N2[ijk];
            const TF ri_pr = fmin(buoyancy_gradient/strain2/tPr, TF(1)-Constants::dsmall);
            const TF wall_length2 = Constants::kappa<TF>*Constants::kappa<TF>
                    *(z[k]+z0m[ij])*(z[k]+z0m[ij]);
            const TF base_h2 = cs*cs*mlen0_h*mlen0_h;
            const TF base_v2 = cs*cs*mlen0_v*mlen0_v;
            const TF length_h2 = base_h2*wall_length2/(base_h2+wall_length2);
            const TF length_v2 = base_v2*wall_length2/(base_v2+wall_length2);
            const TF factor = sqrt(strain2)*sqrt(TF(1)-ri_pr);
            evisc_h[ijk] = length_h2*factor;
            evisc_v[ijk] = length_v2*factor;
        }
    };

    template<typename TF>
    struct diff_uvw_g
    {
        DEFINE_GRID_KERNEL("diff_smag2::diff_uvw_anisotropic", 0)

        template<typename Level>
        CUDA_DEVICE void operator()(
                Grid_layout g, const int i, const int j, const int k, const Level level,
                TF* const ut, TF* const vt, TF* const wt,
                const TF* const eh, const TF* const ev,
                const TF* const u, const TF* const v, const TF* const w,
                const TF* const fluxbotu, const TF* const fluxtopu,
                const TF* const fluxbotv, const TF* const fluxtopv,
                const TF* const dzi, const TF* const dzhi,
                const TF dxi, const TF dyi,
                const TF* const rhoref, const TF* const rhorefh,
                const TF* const rhorefi, const TF* const rhorefhi,
                const TF visc)
        {
            const int ii = g.istride;
            const int jj = g.jstride;
            const int kk = g.kstride;
            const int ij = i+j*jj;
            const int ijk = g(i, j, k);

            const TF eh_eu = eh[ijk]+visc;
            const TF eh_wu = eh[ijk-ii]+visc;
            const TF eh_nu = TF(.25)*(eh[ijk-ii]+eh[ijk]+eh[ijk-ii+jj]+eh[ijk+jj])+visc;
            const TF eh_su = TF(.25)*(eh[ijk-ii-jj]+eh[ijk-jj]+eh[ijk-ii]+eh[ijk])+visc;
            const TF eh_tu = TF(.25)*(eh[ijk-ii]+eh[ijk]+eh[ijk-ii+kk]+eh[ijk+kk])+visc;
            const TF eh_bu = TF(.25)*(eh[ijk-ii-kk]+eh[ijk-kk]+eh[ijk-ii]+eh[ijk])+visc;
            const TF ev_tu = TF(.25)*(ev[ijk-ii]+ev[ijk]+ev[ijk-ii+kk]+ev[ijk+kk])+visc;
            const TF ev_bu = TF(.25)*(ev[ijk-ii-kk]+ev[ijk-kk]+ev[ijk-ii]+ev[ijk])+visc;

            const TF eh_ev = TF(.25)*(eh[ijk-jj]+eh[ijk]+eh[ijk+ii-jj]+eh[ijk+ii])+visc;
            const TF eh_wv = TF(.25)*(eh[ijk-ii-jj]+eh[ijk-ii]+eh[ijk-jj]+eh[ijk])+visc;
            const TF eh_nv = eh[ijk]+visc;
            const TF eh_sv = eh[ijk-jj]+visc;
            const TF eh_tv = TF(.25)*(eh[ijk-jj]+eh[ijk]+eh[ijk+kk-jj]+eh[ijk+kk])+visc;
            const TF eh_bv = TF(.25)*(eh[ijk-kk-jj]+eh[ijk-kk]+eh[ijk-jj]+eh[ijk])+visc;
            const TF ev_tv = TF(.25)*(ev[ijk-jj]+ev[ijk]+ev[ijk+kk-jj]+ev[ijk+kk])+visc;
            const TF ev_bv = TF(.25)*(ev[ijk-kk-jj]+ev[ijk-kk]+ev[ijk-jj]+ev[ijk])+visc;

            TF vertical_u;
            TF vertical_v;
            if (level.distance_to_start() == 0)
            {
                vertical_u = (rhorefh[k+1]*(ev_tu*(u[ijk+kk]-u[ijk])*dzhi[k+1]
                        + eh_tu*(w[ijk+kk]-w[ijk-ii+kk])*dxi)+rhorefh[k]*fluxbotu[ij])*rhorefi[k]*dzi[k];
                vertical_v = (rhorefh[k+1]*(ev_tv*(v[ijk+kk]-v[ijk])*dzhi[k+1]
                        + eh_tv*(w[ijk+kk]-w[ijk-jj+kk])*dyi)+rhorefh[k]*fluxbotv[ij])*rhorefi[k]*dzi[k];
            }
            else if (level.distance_to_end() == 0)
            {
                vertical_u = (-rhorefh[k+1]*fluxtopu[ij]-rhorefh[k]*(ev_bu*(u[ijk]-u[ijk-kk])*dzhi[k]
                        + eh_bu*(w[ijk]-w[ijk-ii])*dxi))*rhorefi[k]*dzi[k];
                vertical_v = (-rhorefh[k+1]*fluxtopv[ij]-rhorefh[k]*(ev_bv*(v[ijk]-v[ijk-kk])*dzhi[k]
                        + eh_bv*(w[ijk]-w[ijk-jj])*dyi))*rhorefi[k]*dzi[k];
            }
            else
            {
                vertical_u = (rhorefh[k+1]*(ev_tu*(u[ijk+kk]-u[ijk])*dzhi[k+1]
                        + eh_tu*(w[ijk+kk]-w[ijk-ii+kk])*dxi)
                        -rhorefh[k]*(ev_bu*(u[ijk]-u[ijk-kk])*dzhi[k]
                        + eh_bu*(w[ijk]-w[ijk-ii])*dxi))*rhorefi[k]*dzi[k];
                vertical_v = (rhorefh[k+1]*(ev_tv*(v[ijk+kk]-v[ijk])*dzhi[k+1]
                        + eh_tv*(w[ijk+kk]-w[ijk-jj+kk])*dyi)
                        -rhorefh[k]*(ev_bv*(v[ijk]-v[ijk-kk])*dzhi[k]
                        + eh_bv*(w[ijk]-w[ijk-jj])*dyi))*rhorefi[k]*dzi[k];
            }

            ut[ijk] += (eh_eu*(u[ijk+ii]-u[ijk])-eh_wu*(u[ijk]-u[ijk-ii]))*TF(2)*dxi*dxi
                    +(eh_nu*((u[ijk+jj]-u[ijk])*dyi+(v[ijk+jj]-v[ijk-ii+jj])*dxi)
                    -eh_su*((u[ijk]-u[ijk-jj])*dyi+(v[ijk]-v[ijk-ii])*dxi))*dyi+vertical_u;
            vt[ijk] += (eh_ev*((v[ijk+ii]-v[ijk])*dxi+(u[ijk+ii]-u[ijk+ii-jj])*dyi)
                    -eh_wv*((v[ijk]-v[ijk-ii])*dxi+(u[ijk]-u[ijk-jj])*dyi))*dxi
                    +(eh_nv*(v[ijk+jj]-v[ijk])-eh_sv*(v[ijk]-v[ijk-jj]))*TF(2)*dyi*dyi+vertical_v;

            if (level.distance_to_start() > 0)
            {
                const TF eh_ew = TF(.25)*(eh[ijk-kk]+eh[ijk]+eh[ijk+ii-kk]+eh[ijk+ii])+visc;
                const TF eh_ww = TF(.25)*(eh[ijk-ii-kk]+eh[ijk-ii]+eh[ijk-kk]+eh[ijk])+visc;
                const TF eh_nw = TF(.25)*(eh[ijk-kk]+eh[ijk]+eh[ijk+jj-kk]+eh[ijk+jj])+visc;
                const TF eh_sw = TF(.25)*(eh[ijk-jj-kk]+eh[ijk-jj]+eh[ijk-kk]+eh[ijk])+visc;
                const TF ev_ew = TF(.25)*(ev[ijk-kk]+ev[ijk]+ev[ijk+ii-kk]+ev[ijk+ii])+visc;
                const TF ev_ww = TF(.25)*(ev[ijk-ii-kk]+ev[ijk-ii]+ev[ijk-kk]+ev[ijk])+visc;
                const TF ev_nw = TF(.25)*(ev[ijk-kk]+ev[ijk]+ev[ijk+jj-kk]+ev[ijk+jj])+visc;
                const TF ev_sw = TF(.25)*(ev[ijk-jj-kk]+ev[ijk-jj]+ev[ijk-kk]+ev[ijk])+visc;
                const TF ev_tw = ev[ijk]+visc;
                const TF ev_bw = ev[ijk-kk]+visc;
                wt[ijk] += ((eh_ew*(w[ijk+ii]-w[ijk])*dxi+ev_ew*(u[ijk+ii]-u[ijk+ii-kk])*dzhi[k])
                        -(eh_ww*(w[ijk]-w[ijk-ii])*dxi+ev_ww*(u[ijk]-u[ijk-kk])*dzhi[k]))*dxi
                        +((eh_nw*(w[ijk+jj]-w[ijk])*dyi+ev_nw*(v[ijk+jj]-v[ijk+jj-kk])*dzhi[k])
                        -(eh_sw*(w[ijk]-w[ijk-jj])*dyi+ev_sw*(v[ijk]-v[ijk-kk])*dzhi[k]))*dyi
                        +(rhoref[k]*ev_tw*(w[ijk+kk]-w[ijk])*dzi[k]
                        -rhoref[k-1]*ev_bw*(w[ijk]-w[ijk-kk])*dzi[k-1])*rhorefhi[k]*TF(2)*dzhi[k];
            }
        }
    };

    template<typename TF>
    struct diff_c_g
    {
        DEFINE_GRID_KERNEL("diff_smag2::diff_c_anisotropic", 0)

        template<typename Level>
        CUDA_DEVICE void operator()(
                Grid_layout g, const int i, const int j, const int k, const Level level,
                TF* const at, const TF* const a,
                const TF* const eh, const TF* const ev,
                const TF* const fluxbot, const TF* const fluxtop,
                const TF* const dzi, const TF* const dzhi,
                const TF dx2i, const TF dy2i,
                const TF* const rhorefi, const TF* const rhorefh,
                const TF tPri, const TF visc)
        {
            const int ii=g.istride, jj=g.jstride, kk=g.kstride;
            const int ij=i+j*jj, ijk=g(i,j,k);
            const TF ee=TF(.5)*(eh[ijk]+eh[ijk+ii])*tPri+visc;
            const TF ew=TF(.5)*(eh[ijk-ii]+eh[ijk])*tPri+visc;
            const TF en=TF(.5)*(eh[ijk]+eh[ijk+jj])*tPri+visc;
            const TF es=TF(.5)*(eh[ijk-jj]+eh[ijk])*tPri+visc;
            TF vertical;
            if (level.distance_to_start()==0)
            {
                const TF et=TF(.5)*(ev[ijk]+ev[ijk+kk])*tPri+visc;
                vertical=(rhorefh[k+1]*et*(a[ijk+kk]-a[ijk])*dzhi[k+1]
                        +rhorefh[k]*fluxbot[ij])*rhorefi[k]*dzi[k];
            }
            else if (level.distance_to_end()==0)
            {
                const TF eb=TF(.5)*(ev[ijk-kk]+ev[ijk])*tPri+visc;
                vertical=(-rhorefh[k+1]*fluxtop[ij]
                        -rhorefh[k]*eb*(a[ijk]-a[ijk-kk])*dzhi[k])*rhorefi[k]*dzi[k];
            }
            else
            {
                const TF et=TF(.5)*(ev[ijk]+ev[ijk+kk])*tPri+visc;
                const TF eb=TF(.5)*(ev[ijk-kk]+ev[ijk])*tPri+visc;
                vertical=(rhorefh[k+1]*et*(a[ijk+kk]-a[ijk])*dzhi[k+1]
                        -rhorefh[k]*eb*(a[ijk]-a[ijk-kk])*dzhi[k])*rhorefi[k]*dzi[k];
            }
            at[ijk]+=(ee*(a[ijk+ii]-a[ijk])-ew*(a[ijk]-a[ijk-ii]))*dx2i
                    +(en*(a[ijk+jj]-a[ijk])-es*(a[ijk]-a[ijk-jj]))*dy2i+vertical;
        }
    };

    template<typename TF>
    __global__ void dnmul_g(
            TF* const out, const TF* const eh, const TF* const ev,
            const TF* const dzi, const TF dx2i, const TF dy2i, const TF tPrfac_i,
            const int istart, const int iend, const int jstart, const int jend,
            const int kstart, const int kend, const int jj, const int kk)
    {
        const int i=blockIdx.x*blockDim.x+threadIdx.x+istart;
        const int j=blockIdx.y*blockDim.y+threadIdx.y+jstart;
        const int k=blockIdx.z+kstart;
        if(i<iend && j<jend && k<kend)
        {
            const int ijk=i+j*jj+k*kk;
            out[ijk]=fabs(eh[ijk]*tPrfac_i*(dx2i+dy2i)+ev[ijk]*tPrfac_i*dzi[k]*dzi[k]);
        }
    }
}

#endif
