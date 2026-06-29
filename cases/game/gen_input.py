from datetime import datetime
from pathlib import Path
import shutil
import xarray as xr
import numpy as np
import ls2d

import microhhpy.io as io
from microhhpy.land import Land_surface_input
from microhhpy.spatial import calc_vertical_grid_2nd
from microhhpy.utils import get_data_file


case_name = 'praa_sands'
input_dir = Path(__file__).resolve().parent
output_dir = input_dir / case_name


def copy_file(src, dst):
    """
    Copy support input files into the generated case folder.
    """
    src = Path(src)
    dst = Path(dst)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    shutil.copy(src, dst)


def copy_radfiles(srcdir, destdir, gpt='128_112'):
    srcdir = Path(srcdir)
    destdir = Path(destdir)

    if gpt == '128_112':
        copy_file(srcdir / 'rrtmgp-gas-lw-g128.nc', destdir / 'coefficients_lw.nc')
        copy_file(srcdir / 'rrtmgp-gas-sw-g112.nc', destdir / 'coefficients_sw.nc')
    elif gpt == '256_224':
        copy_file(srcdir / 'rrtmgp-gas-lw-g256.nc', destdir / 'coefficients_lw.nc')
        copy_file(srcdir / 'rrtmgp-gas-sw-g224.nc', destdir / 'coefficients_sw.nc')
    else:
        raise ValueError("gpt should be in {'128_112', '256_224'}")

    copy_file(srcdir / 'rrtmgp-clouds-lw.nc', destdir / 'cloud_coefficients_lw.nc')
    copy_file(srcdir / 'rrtmgp-clouds-sw.nc', destdir / 'cloud_coefficients_sw.nc')


def copy_aerosolfiles(srcdir, destdir):
    copy_file(Path(srcdir) / 'aerosol_optics.nc',
              Path(destdir) / 'aerosol_optics.nc')


def read_era5_forcing(settings, z):
    """
    Read ERA5/LS2D forcing on the LES vertical grid.
    """
    settings = normalize_ls2d_settings(settings)
    ls2d.download_era5(settings)
    era5 = ls2d.Read_era5(settings)
    era5.calculate_forcings(n_av=settings.get('n_av', 0), method=settings.get('forcing_method', '2nd'))
    return era5.get_les_input(z)


def read_cams_forcing(settings, z):
    """
    Read CAMS gas/aerosol forcing on the LES vertical grid and ERA5 times.
    """
    settings = normalize_ls2d_settings(settings)
    cams_variables = settings.get(
        'cams_variables',
        {
            'eac4_ml': ['aermr01', 'aermr02', 'aermr03', 'aermr04', 'aermr05', 'aermr06',
                        'aermr07', 'aermr08', 'aermr09', 'aermr10', 'aermr11', 'co2', 'ch4'],
            'eac4_sfc': [],
        })
    ls2d.download_cams(settings, cams_variables)
    return ls2d.Read_cams(settings, cams_variables).get_les_input(z, n_av=settings.get('n_av', 0))


def normalize_ls2d_settings(settings):
    settings = dict(settings)
    for key in ('era5_path', 'cams_path'):
        if key in settings:
            settings[key] = str(settings[key])
    return settings


def clamp_array(data, lower, upper):
    if isinstance(data, xr.DataArray):
        return data.clip(min=lower, max=upper)
    return np.clip(data, lower, upper)


def create_case_input(
        start_date,
        end_date,
        ls2d_settings,
        gd,
        use_htessel,
        use_rrtmgp,
        use_rt,
        use_aerosols,
        use_tdep_aerosols,
        use_tdep_gasses,
        use_tdep_background,
        use_homogeneous_z0,
        use_homogeneous_ls,
        gpt_set,
        sw_micro,
        itot, jtot, xsize, ysize,
        TF,
        npx=1, npy=1):

    output_dir.mkdir(parents=True, exist_ok=True)

    if use_rrtmgp:
        copy_radfiles(
            srcdir=input_dir / '../../rte-rrtmgp-cpp/rrtmgp-data/',
            destdir=output_dir,
            gpt=gpt_set)

    if use_htessel:
        copy_file(get_data_file('van_genuchten_parameters.nc'),
                  output_dir / 'van_genuchten_parameters.nc')

    if use_aerosols:
        copy_aerosolfiles(
            srcdir=input_dir / '../../rte-rrtmgp-cpp/data/',
            destdir=output_dir)


    heterogeneous_sfc = not use_homogeneous_z0 or not use_homogeneous_ls

    """
    Use the vertical grid as provided by microhhpy.
    """
    z = np.asarray(gd['z'], dtype=TF)
    zh = np.asarray(gd['zh'], dtype=TF)
    ktot = int(gd['ktot'])
    zsize = float(zh[-1])

    """
    Read / interpolate (LS)2D initial conditions and forcings
    """
    ls2d_settings = dict(ls2d_settings)
    ls2d_settings['start_date'] = start_date
    ls2d_settings['end_date'] = end_date

    era5_input = read_era5_forcing(ls2d_settings, z)

    # Read CAMS for aerosols and gasses other than ozone.
    cams_input = read_cams_forcing(ls2d_settings, z)
    cams_les = cams_input

    # interpolate background CAMS profile to ERA5 levels (CAMS data comes at fewer levels)
    cams_bg = xr.Dataset(
        coords={
            'time': era5_input.time,
            'lay': era5_input.lay,
        })

    def interp_z(array, z_era, z_cams, time):
        out = np.empty(z_era.shape)
        for t in range(time.size):
            out[t,:] = np.interp(z_era[t, :], z_cams[t, :], array[t,:])
        return out

    for var in list(cams_input.keys()):
        if var[-4:] == '_lay':
            if cams_input['lay'].size != era5_input['lay'].size:
                data = interp_z(cams_input[var].data, era5_input['z_lay'].data, cams_input['z_lay'].data, era5_input['time'])
            else:
                data = cams_input[var].data   # in case the interpolation was already done before, just copy the data
            cams_bg[var] = (('time', 'lay'), data)

    if 'co2' not in list(cams_les.keys()):
        print('greenhouse gasses from CAMS not provided, using constant (RFMIP) values for co2 and ch4 instead')
        const_ghg = 1
    else:
        const_ghg = 0

    # Make sure aerosol concentrations are >= 0.
    for v in cams_les:
        if 'aermr' in v:
            cams_les[v] = np.maximum(cams_les[v], 0.)

    if not use_rrtmgp:
        # Read ERA5 radiation, de-accumulate, and interpolate to LS2D times.
        # TODO: add to LS2D download...
        ds_rad = xr.open_dataset(input_dir / 'era_rad_20160815.nc')
        ds_rad = ds_rad/3600.
        ds_rad['time'] = ds_rad['time'] - np.timedelta64(30, 'm')
        ds_rad = ds_rad.interp(time=era5_input.time)

    # Reverse the soil fields. Important NOTE: in MicroHH, the vertical
    # soil index 0 is the lowest level in the soil. In (LS)2D, this
    # is reversed, and soil index 0 is the top soil level....
    # Another NOTE: the soil type in (LS)2D is the ERA5 soil type,
    # which (FORTRAN....) is 1-based, so we need to subtract 1 to
    # get the correct C-indexing.
    theta_soil = era5_input.theta_soil[0,::-1].values
    t_soil = era5_input.t_soil[0,::-1].values
    index_soil = np.ones_like(era5_input.zs) * int(era5_input.type_soil - 1)
    root_frac = era5_input.root_frac_low_veg[::-1].values

    """
    Update .ini file
    """
    ini = io.read_ini(input_dir / 'ini.base')

    ini['master']['npx'] = npx
    ini['master']['npy'] = npy

    ini['grid']['itot'] = itot
    ini['grid']['jtot'] = jtot
    ini['grid']['ktot'] = ktot

    ini['grid']['xsize'] = xsize
    ini['grid']['ysize'] = ysize
    ini['grid']['zsize'] = zsize

    ini['grid']['lat'] = ls2d_settings["central_lat"]
    ini['grid']['lon'] = ls2d_settings["central_lon"]

    ini['buffer']['zstart'] = 0.75 * zsize

    if use_htessel:
        ini['boundary']['swboundary'] = 'surface_lsm'
        ini['boundary']['sbcbot'] = 'dirichlet'
    else:
        ini['boundary']['swboundary'] = 'surface'
        ini['boundary']['sbcbot'] = 'flux'
        ini['boundary']['swtimedep'] = True
        ini['boundary']['timedeplist'] = ['thl_sbot', 'qt_sbot']

    ini['boundary']['swconstantz0'] = use_homogeneous_z0
    ini['land_surface']['swhomogeneous'] = use_homogeneous_ls

    if use_rrtmgp and not use_rt:
        ini['radiation']['swradiation'] = 'rrtmgp'
    elif use_rrtmgp and use_rt:
        ini['radiation']['swradiation'] = 'rrtmgp_rt'
        ini['radiation']['rays_per_pixel'] = 256
        ini['radiation']['kngrid_i'] = 64
        ini['radiation']['kngrid_j'] = 64
        ini['radiation']['kngrid_k'] = 32
    else:
        ini['radiation']['swradiation'] = 'prescribed'
        ini['radiation']['swtimedep_prescribed'] = True

    ini['radiation']['swtimedep_background'] = use_tdep_background
    if use_tdep_gasses:
        if not const_ghg:
            ini['radiation']['timedeplist_gas'] = ['o3', 'co2', 'ch4']
        else:
            ini['radiation']['timedeplist_gas'] = ['o3']

    ini['aerosol']['swaerosol'] = use_aerosols
    ini['aerosol']['swtimedep'] = use_tdep_aerosols

    ini['time']['endtime'] = (end_date - start_date).total_seconds()
    d = start_date
    ini['time']['datetime_utc'] = f'{d.year}-{d.month:02d}-{d.day:02d} {d.hour:02d}:{d.minute:02d}:{d.second:02d}'

    if heterogeneous_sfc:
        ini['stats']['xymasklist'] = ['wet_mask', 'dry_mask']

    if sw_micro == 'nsw6':
        ini['micro']['swmicro'] = sw_micro
        ini['advec']['fluxlimit_list'] = ['qt', 'qr', 'qs', 'qg']
        ini['limiter']['limitlist'] = ['qt', 'qr', 'qs', 'qg']
    elif sw_micro == '2mom_warm':
        ini['micro']['swmicro'] = sw_micro
        ini['advec']['fluxlimit_list'] = ['qt', 'qr', 'nr']
        ini['limiter']['limitlist'] = ['qt', 'qr', 'nr']
    else:
        ini['micro']['swmicro'] = False
        ini['advec']['fluxlimit_list'] = ['qt']
        ini['limiter']['limitlist'] = ['qt']

    ini['column']['coordinates[x]'] = xsize / 2
    ini['column']['coordinates[y]'] = ysize / 2

    io.save_ini(ini, output_dir / f'{case_name}.ini', clobber=True)

    """
    Create MicroHH input NetCDF file.
    """
    init_profiles = {
        'z': z,
        'thl': era5_input.thl[0,:],
        'qt': era5_input.qt[0,:],
        'u': era5_input.u[0,:],
        'v': era5_input.v[0,:],
        'nudgefac': np.ones(ktot)/10800,
    }

    tdep_surface = {
        'time_surface': era5_input.time_sec,
        'p_sbot': era5_input.ps,
    }

    tdep_ls = {
        'time_ls': era5_input.time_sec,
        'u_geo': era5_input.ug,
        'v_geo': era5_input.vg,
        'u_ls': era5_input.dtu_advec,
        'v_ls': era5_input.dtv_advec,
        'thl_ls': era5_input.dtthl_advec,
        'qt_ls': era5_input.dtqt_advec,
        'w_ls': era5_input.wls,
        'u_nudge': era5_input.u,
        'v_nudge': era5_input.v,
        'thl_nudge': era5_input.thl,
        'qt_nudge': era5_input.qt,
    }

    if not use_htessel:
        tdep_surface['thl_sbot'] = era5_input.wth
        tdep_surface['qt_sbot'] = era5_input.wq

    if not use_rrtmgp:
        tdep_surface['sw_flux_dn'] = ds_rad.ssrd
        tdep_surface['sw_flux_up'] = ds_rad.ssrd-ds_rad.ssr
        tdep_surface['lw_flux_dn'] = ds_rad.strd
        tdep_surface['lw_flux_up'] = ds_rad.strd-ds_rad.str

    """
    Radiation variables
    """
    radiation = None
    tdep_radiation = None
    tdep_aerosol = None
    if use_rrtmgp:
        coef_lw = xr.open_dataset(output_dir / 'coefficients_lw.nc')
        press_ref_min = float(np.min(coef_lw.press_ref))
        press_ref_max = float(np.max(coef_lw.press_ref))
        temp_ref_min = float(np.min(coef_lw.temp_ref))
        temp_ref_max = float(np.max(coef_lw.temp_ref))

        # Radiation variables on LES grid.
        xm_air = 28.97; xm_h2o = 18.01528; eps = xm_h2o / xm_air
        qt_mean = era5_input.qt.mean(axis=0)
        h2o = qt_mean / (eps - eps * qt_mean)
        init_profiles['h2o'] = h2o
        init_profiles['o3'] = era5_input.o3[0,:]*1e-6

        if not const_ghg:
            init_profiles['co2'] = cams_les.co2[0,:]*1e-6
            init_profiles['ch4'] = cams_les.ch4[0,:]*1e-6

        # Constant concentrations:
        for profiles in (init_profiles,):
            profiles['n2o'] = 3.2699e-7
            profiles['n2'] = 0.781
            profiles['o2'] = 0.209
            if const_ghg:
                profiles['co2'] = 397.54697e-6
                profiles['ch4'] = 1831.471e-9

        # Radiation variables on radiation grid/levels:
        radiation = {
            'z_lay': era5_input.z_lay.mean(axis=0),
            'z_lev': era5_input.z_lev.mean(axis=0),
            'p_lay': clamp_array(era5_input.p_lay.mean(axis=0), press_ref_min, press_ref_max),
            'p_lev': clamp_array(era5_input.p_lev.mean(axis=0), press_ref_min, press_ref_max),
            't_lay': clamp_array(era5_input.t_lay.mean(axis=0), temp_ref_min, temp_ref_max),
            't_lev': clamp_array(era5_input.t_lev.mean(axis=0), temp_ref_min, temp_ref_max),
            'h2o': era5_input.h2o_lay.mean(axis=0),
            'o3': era5_input.o3_lay.mean(axis=0)*1e-6,
            'n2o': 3.2699e-7,
            'n2': 0.781,
            'o2': 0.209,
        }
        if const_ghg:
            radiation['co2'] = 397.54697e-6
            radiation['ch4'] = 1831.471e-9
        if not const_ghg:
            radiation['co2'] = cams_bg.co2_lay.mean(axis=0)
            radiation['ch4'] = cams_bg.ch4_lay.mean(axis=0)

        if use_tdep_background or use_tdep_aerosols or use_tdep_gasses:
            # NOTE: bit cheap, but ERA and CAMS are at the same time period/interval here.
            tdep_radiation = {'time_rad': era5_input.time_sec}

        if use_tdep_gasses:
            tdep_radiation['o3'] = era5_input.o3*1e-6
            if not const_ghg:
                tdep_radiation['co2'] = cams_les.co2
                tdep_radiation['ch4'] = cams_les.ch4

        # Time dependent background profiles T, h2o, o3, ...
        if use_tdep_background:
            tdep_radiation['z_lay'] = era5_input.z_lay
            tdep_radiation['z_lev'] = era5_input.z_lev
            tdep_radiation['p_lay'] = clamp_array(era5_input.p_lay, press_ref_min, press_ref_max)
            tdep_radiation['p_lev'] = clamp_array(era5_input.p_lev, press_ref_min, press_ref_max)
            tdep_radiation['t_lay'] = clamp_array(era5_input.t_lay, temp_ref_min, temp_ref_max)
            tdep_radiation['t_lev'] = clamp_array(era5_input.t_lev, temp_ref_min, temp_ref_max)
            tdep_radiation['h2o_bg'] = era5_input.h2o_lay
            tdep_radiation['o3_bg'] = era5_input.o3_lay*1e-6
            if not const_ghg:
                tdep_radiation['co2_bg'] = cams_bg.co2_lay
                tdep_radiation['ch4_bg'] = cams_bg.ch4_lay

        # Aerosols for domain and background column
        if use_aerosols:
            for i in range(1, 12):
                name = f'aermr{i:02d}'
                init_profiles[name] = getattr(cams_les, name).mean(axis=0)
                radiation[name] = getattr(cams_bg, f'{name}_lay').mean(axis=0)

            if use_tdep_aerosols:
                tdep_aerosol = {'time_rad': era5_input.time_sec}
                for i in range(1, 12):
                    name = f'aermr{i:02d}'
                    tdep_aerosol[f'{name}_bg'] = getattr(cams_bg, f'{name}_lay')
                    tdep_aerosol[name] = getattr(cams_les, name)

    """
    Land-surface and soil
    """
    soil = None
    if use_htessel:
        soil = {
            'z': era5_input.zs[::-1],
            'theta_soil': theta_soil,
            't_soil': t_soil,
            'index_soil': index_soil,
            'root_frac': root_frac,
        }

    io.save_case_input(
        case_name,
        init_profiles,
        tdep_surface=tdep_surface,
        tdep_ls=tdep_ls,
        tdep_aerosol=tdep_aerosol,
        tdep_radiation=tdep_radiation,
        radiation=radiation,
        soil=soil,
        output_dir=output_dir)

    """
    Create 2D binary input files (if needed)
    """
    def get_patches(blocksize_i, blocksize_j):
        """
        Get mask for the surface patches
        """
        mask = np.zeros((jtot, itot), dtype=bool)
        mask[:] = False

        for j in range(jtot):
            for i in range(itot):
                patch_i = i // blocksize_i % 2 == 0
                patch_j = j // blocksize_j % 2 == 0

                if (patch_i and patch_j) or (not patch_i and not patch_j):
                    mask[j,i] = True

        return mask


    if heterogeneous_sfc:
        """
        Create surface mask for masked statistics.
        """

        mask = get_patches(blocksize_i=8, blocksize_j=8)

        wet = mask.astype(TF)
        dry = 1-wet

        wet.tofile(output_dir / 'wet_mask.0000000')
        dry.tofile(output_dir / 'dry_mask.0000000')


    if not use_homogeneous_z0:
        """
        Create checkerboard pattern for z0m and z0h
        """

        z0m = ini['boundary']['z0m']
        z0h = ini['boundary']['z0h']

        z0m_2d = np.zeros((jtot, itot), dtype=TF)
        z0h_2d = np.zeros((jtot, itot), dtype=TF)

        z0m_2d[ mask] = z0m
        z0m_2d[~mask] = z0m/2.

        z0h_2d[ mask] = z0h
        z0h_2d[~mask] = z0h/2.

        z0m_2d.tofile(output_dir / 'z0m.0000000')
        z0h_2d.tofile(output_dir / 'z0h.0000000')

    if not use_homogeneous_ls:
        """
        Create checkerboard pattern for land-surface fields.
        """

        exclude = ['z0h', 'z0m', 'water_mask', 't_bot_water', 'index_veg']
        lsm_data = Land_surface_input(
            itot,
            jtot,
            ktot=4,
            float_type=TF,
            debug=True,
            exclude_fields=exclude)

        # Patched fields:
        lsm_data.c_veg[ mask] = ini['land_surface']['c_veg']
        lsm_data.c_veg[~mask] = ini['land_surface']['c_veg']/3.

        lsm_data.lai[ mask] = ini['land_surface']['lai']
        lsm_data.lai[~mask] = ini['land_surface']['lai']/2.

        # Non-patched / homogeneous fields:
        lsm_data.gD[:,:] = ini['land_surface']['gD']
        lsm_data.rs_veg_min[:,:] = ini['land_surface']['rs_veg_min']
        lsm_data.rs_soil_min[:,:] = ini['land_surface']['rs_soil_min']
        lsm_data.lambda_stable[:,:] = ini['land_surface']['lambda_stable']
        lsm_data.lambda_unstable[:,:] = ini['land_surface']['lambda_unstable']
        lsm_data.cs_veg[:,:] = ini['land_surface']['cs_veg']

        lsm_data.t_soil[:,:,:] = t_soil[:, np.newaxis, np.newaxis]
        lsm_data.index_soil[:,:,:] = index_soil[:, np.newaxis, np.newaxis]
        lsm_data.root_frac[:,:,:] = root_frac[:, np.newaxis, np.newaxis]

        # Create dry/wet patches.
        vg = xr.open_dataset(get_data_file('van_genuchten_parameters.nc'))

        theta_wp = float(vg.theta_wp[int(index_soil[0])])
        theta_fc = float(vg.theta_fc[int(index_soil[0])])
        theta_cap = theta_fc - theta_wp

        lsm_data.theta_soil[:,  mask] = theta_fc - 0.1 * theta_cap
        lsm_data.theta_soil[:, ~mask] = theta_wp + 0.1 * theta_cap

        # Check if all the variables have been set:
        lsm_data.check()

        # Save binary input MicroHH, and NetCDF file for visual validation/plotting/etc.
        lsm_data.to_binaries(path=output_dir, allow_overwrite=True)
        lsm_data.to_netcdf(output_dir / 'lsm_input.nc', allow_overwrite=True)


if __name__ == '__main__':
    """
    Case switches.
    """
    TF = np.float32              # Switch between double (float64) and single (float32) precision.
    use_htessel = True           # False = prescribed surface H+LE fluxes from ERA5.
    use_rrtmgp = True            # False = prescribed surface radiation from ERA5.
    use_rt = False               # False = 2stream solver for shortwave down, True = raytracer.
    use_homogeneous_z0 = True    # False = checkerboard pattern roughness lengths.
    use_homogeneous_ls = True    # False = checkerboard pattern (some...) land-surface fields.
    use_aerosols = False         # False = no aerosols in RRTMGP.
    use_tdep_aerosols = False    # False = time fixed RRTMGP aerosol in domain and background.
    use_tdep_gasses = False      # False = time fixed ERA5 (o3) and CAMS (co2, ch4) gasses.
    use_tdep_background = False  # False = time fixed RRTMGP T/h2o/o3 background profiles.

    sw_micro = 'nsw6'

    """
    NOTE: `use_tdep_aerosols` and `use_tdep_gasses` specify whether the aerosols and gasses
          used by RRTMGP are updated inside the LES domain. If `use_tdep_background` is true, the
          aerosols, gasses, and the temperature & humidity are also updated on the RRTMGP background levels.
    """

    # Switch between the two default RRTMGP g-point sets.
    gpt_set = '128_112' # or '256_224'

    # Time period.
    start_date = datetime(year=2014, month=11, day=15, hour=10)
    end_date   = datetime(year=2014, month=11, day=15, hour=15)
    # (LS)2D settings, following the pattern in ref.py.
    ls2d_settings = {
        "start_date": start_date,
        "end_date": end_date,
        "central_lon": 5.235,
        "central_lat": 50.159167,
        "area_size": 1,
        "case_name": case_name,
        "era5_path": "/home/flc/Projects/LS2D/data/ERA5/",
        "cams_path": "/home/flc/Projects/LS2D/data/CAMS/",
        "cdsapirc": "/home/flc/.cdsapirc",
        "data_source": "CDS",
        "era5_expver": 1,
        "write_log": False,
        "n_av": 0,
        "forcing_method": "2nd",
    }

    # Vertical grid.
    # zgrid = ls2d.grid.Grid_equidist(kmax=240, dz0=25.0)
    zgrid = ls2d.grid.Grid_linear_stretched(kmax=200, dz0=12, alpha=0.01)
    gd = calc_vertical_grid_2nd(zgrid.z, zgrid.zsize)

    itot = 500
    jtot = 500

    xsize = 6000
    ysize = 6000

    # Create input files.
    create_case_input(
            start_date,
            end_date,
            ls2d_settings,
            gd,
            use_htessel,
            use_rrtmgp,
            use_rt,
            use_aerosols,
            use_tdep_aerosols,
            use_tdep_gasses,
            use_tdep_background,
            use_homogeneous_z0,
            use_homogeneous_ls,
            gpt_set,
            sw_micro,
            itot, jtot, xsize, ysize,
            TF,
            npx=1,
            npy=1)
