#
#  MicroHH
#  Copyright (c) 2011-2024 Chiel van Heerwaarden
#  Copyright (c) 2011-2024 Thijs Heus
#  Copyright (c) 2014-2024 Bart van Stratum
#
#  This file is part of MicroHH
#
#  MicroHH is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  MicroHH is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
#

from datetime import datetime
from pathlib import Path
import argparse
import glob

import netCDF4 as nc4
import numpy as np
import ls2d

from microhhpy.spatial import Domain, calc_vertical_grid_2nd
from microhhpy.real import create_input_from_regular_latlon
from microhhpy.real import regrid_les

import microhhpy.io as io
import microhhpy.thermo as thermo

from microhhpy.logger import logger
from icon_driver import IconDriver

logger.setLevel("DEBUG")


case_dir = Path(__file__).resolve().parent
repo_root = case_dir.parents[1]

"""
User input
"""
parser = argparse.ArgumentParser(description="Nested ICON-EU open-boundary input.")
parser.add_argument("-d", "--domain", type=int, required=True, help="Domain number")
args = parser.parse_args()


"""
Settings
"""
float_type = np.float32

start_date = datetime(year=2026, month=2, day=11, hour=0)
end_date = datetime(year=2026, month=2, day=11, hour=2)

# Latest available ICON forecast cycle that covers the full range is selected automatically.
icon_init_date = None
download_missing = True
ntasks = 8
icon_model_levels = "auto"

# All domains are put in a sub-folder `work_dir/domX`.
work_dir = Path("test")

# ICON-EU settings. The default location is Cabauw, which is inside ICON-EU.
settings = {
    "start_date": start_date,
    "end_date": end_date,
    "icon_init_date": icon_init_date,
    "central_lon": 4.927,
    "central_lat": 51.971,
    "area_size": 1.5,
    "case_name": "cabauw_icon",
    "icon_path": repo_root / "data" / "ICON",
    "download_missing": download_missing,
    "model_levels": icon_model_levels,
    "model_level_margin": 500.0,
}


"""
Vertical grid definition.

NOTE: vertical grid definition in (LS)2D is not identical to MicroHH's grid.
      Use `microhhpy.spatial.calculate_vertical_grid_2nd()` to make sure that the grid
      matches the one from MicroHH, otherwise the initial fields won't be divergence free.
"""
_g = ls2d.grid.Grid_linear_stretched(kmax=128, dz0=20, alpha=0.01)
gd = calc_vertical_grid_2nd(_g.z, _g.zsize)
settings["zsize"] = gd["zsize"]

zstart_buffer = 0.75 * gd["zsize"]


"""
Define projection used for LES coordinates (m) to real world (lat/lon) transforms.
"""
# Outer domain, nested in ICON-EU.
dom0 = Domain(
    xsize=64_000,
    ysize=64_000,
    itot=64,
    jtot=64,
    n_ghost=3,
    n_sponge=5,
    lbc_freq=3600,
    lon=settings["central_lon"],
    lat=settings["central_lat"],
    anchor="center",
    proj_str="+proj=utm +zone=31 +datum=WGS84 +units=m +no_defs +type=crs",
)

# Inner domains(s), nested in parent LES domain.
dom1 = Domain(
    xsize=32_000,
    ysize=32_000,
    itot=64,
    jtot=64,
    n_ghost=3,
    n_sponge=3,
    lbc_freq=60,
    center_in_parent=True,
    parent=dom0,
)

domains = [dom0, dom1]
for i in range(len(domains) - 1):
    domains[i].child = domains[i + 1]

domain = domains[args.domain]
child = domain.child
parent = domain.parent
exp_dir = work_dir / f"dom{args.domain}"
exp_dir.mkdir(parents=True, exist_ok=True)


"""
Read ICON data and calculate LS2D-like profiles.
"""
icon = IconDriver(settings)
icon.read_data()
icon.calculate_forcings(n_av=3, method="2nd")
icon_1d = icon.get_les_input(gd["z"])


"""
Default vertical profile input.
"""
def add_variable(name, dims, nc_group, data):
    fld = nc_group.createVariable(name, float_type, dims)
    fld[:] = data


nc_main = nc4.Dataset(exp_dir / "icon_openbc_input.nc", mode="w", datamodel="NETCDF4", clobber=True)
nc_main.createDimension("z", gd["ktot"])
nc_init = nc_main.createGroup("init")

add_variable("z", ("z",), nc_main, gd["z"])
add_variable("thl", ("z",), nc_init, icon_1d.thl[0, :])
add_variable("qt", ("z",), nc_init, icon_1d.qt[0, :])
add_variable("u", ("z",), nc_init, icon_1d.u[0, :])
add_variable("v", ("z",), nc_init, icon_1d.v[0, :])
add_variable("u_geo", ("z",), nc_init, icon_1d.ug[0, :])
add_variable("v_geo", ("z",), nc_init, icon_1d.vg[0, :])
add_variable("w_ls", ("z",), nc_init, icon_1d.wls[0, :])

nc_main.close()


"""
Calculate boundary conditions thermodynamics.
"""
thl = icon_1d.thl.mean(axis=0).values
qt = icon_1d.qt.mean(axis=0).values

# Neumann top boundary condition.
stop_thl = (thl[-1] - thl[-2]) / (gd["z"][-1] - gd["z"][-2])
stop_qt = (qt[-1] - qt[-2]) / (gd["z"][-1] - gd["z"][-2])

surface_temperature = float(icon_1d.sst.mean())
ps = float(icon_1d.ps.mean())

# Dirichlet lower boundary condition.
sbot_thl = surface_temperature / thermo.exner(ps)
sbot_qt = 0.95 * thermo.qsat(ps, surface_temperature)


"""
Create .ini file.
"""
ini = io.read_ini(case_dir / "icon_openbc.ini.base")

ini["grid"]["itot"] = domain.itot
ini["grid"]["jtot"] = domain.jtot
ini["grid"]["ktot"] = gd["ktot"]

ini["grid"]["xsize"] = domain.xsize
ini["grid"]["ysize"] = domain.ysize
ini["grid"]["zsize"] = gd["zsize"]

ini["buffer"]["zstart"] = zstart_buffer
ini["thermo"]["pbot"] = ps
ini["force"]["fc"] = icon.fc

ini["boundary"]["stop[thl]"] = stop_thl
ini["boundary"]["stop[qt]"] = stop_qt
ini["boundary"]["sbot[thl]"] = sbot_thl
ini["boundary"]["sbot[qt]"] = sbot_qt

ini["time"]["endtime"] = (end_date - start_date).total_seconds()

ini["cross"]["xz"] = domain.ysize / 2
ini["cross"]["yz"] = domain.xsize / 2

# Open-boundary specific settings.
ini["boundary_lateral"]["n_sponge"] = domain.n_sponge
ini["boundary_lateral"]["tau_sponge"] = domain.n_sponge * domain.dx / (10 * 5)
ini["boundary_lateral"]["loadfreq"] = domain.lbc_freq

if args.domain == 0:
    ini["boundary_lateral"]["slist"] = ["thl", "qt"]
    ini["buffer"]["loadfreq"] = 3600
else:
    ini["boundary_lateral"]["slist"] = ["thl", "qt", "qr", "nr"]
    ini["buffer"]["loadfreq"] = 600

# Output LBCs for child domain.
if child is not None:
    ini["subdomain"]["sw_subdomain"] = True
    ini["subdomain"]["xstart"] = child.xstart_in_parent
    ini["subdomain"]["ystart"] = child.ystart_in_parent
    ini["subdomain"]["xend"] = child.xstart_in_parent + child.xsize
    ini["subdomain"]["yend"] = child.ystart_in_parent + child.ysize
    ini["subdomain"]["grid_ratio_ij"] = int(domain.dx / child.dx)
    ini["subdomain"]["grid_ratio_k"] = 1
    ini["subdomain"]["n_ghost"] = child.n_ghost
    ini["subdomain"]["n_sponge"] = child.n_sponge
    ini["subdomain"]["savetime_bcs"] = child.lbc_freq
    ini["subdomain"]["sw_save_wtop"] = True
    ini["subdomain"]["sw_save_buffer"] = True
    ini["subdomain"]["savetime_buffer"] = 600
    ini["subdomain"]["zstart_buffer"] = zstart_buffer

else:
    ini["subdomain"]["sw_subdomain"] = False
    ini["subdomain"]["sw_save_wtop"] = False
    ini["subdomain"]["sw_save_buffer"] = False

if io.check_ini(ini):
    raise RuntimeError("Some ini values are None")

io.save_ini(ini, exp_dir / "icon_openbc.ini")


"""
Calculate and save base state density.
"""
bs = thermo.calc_moist_basestate(
    icon_1d["thl"][0, :].values,
    icon_1d["qt"][0, :].values,
    ps,
    gd["z"],
    gd["zsize"],
    float_type=float_type)

# Only save the density part for the dynamic core.
thermo.save_basestate_density(bs["rho"], bs["rhoh"], exp_dir / "rhoref_overwrite.0000000")


if args.domain == 0:
    """
    Create initial fields and boundary conditions, tri-linearly interpolated from ICON.
    The horizontal velocity fields are corrected to match the horizontal divergence between ICON and LES.
    This is needed to account for interpolation errors and differences in 3D ICON density and the 1D LES base state density.
    """
    fields_icon = {
        "u": icon.u[:, :, :, :],
        "v": icon.v[:, :, :, :],
        "w": icon.wls[:, :, :, :],
        "thl": icon.thl[:, :, :, :],
        "qt": icon.qt[:, :, :, :],
    }

    p_icon = icon.p[:, :, :, :]
    z_icon = icon.z[:, :, :, :]
    time_icon = icon.time_sec

    # Standard dev. of Gaussian filter applied to interpolated fields (m).
    sigma_h = 10_000

    create_input_from_regular_latlon(
        fields_icon,
        icon.lons,
        icon.lats,
        z_icon,
        p_icon,
        time_icon,
        gd["z"],
        gd["zsize"],
        zstart_buffer,
        bs["rho"],
        bs["rhoh"],
        dom0,
        sigma_h,
        perturb_size=4,
        perturb_amplitude={"thl": 0.1, "qt": 0.1e-3},
        clip_at_zero=("qt",),
        name_suffix="overwrite",
        output_dir=exp_dir,
        ntasks=ntasks,
        float_type=float_type)

else:
    """
    Regrid initial fields, and copy boundary conditions from parent domain.
    To prevent file name issues with in- and output, the simulations save
    output with `_out` appended. These need to be renamed.
    """
    parent_exp_dir = work_dir / f"dom{args.domain - 1}"

    fields_3d = {
        "u": 0,
        "v": 0,
        "w": 0,
        "thl": 0,
        "qt": 0,
        "qr": 0,
        "nr": 0,
    }

    fields_2d = {
        "phydro_tod": "*",
    }

    regrid_les(
        fields_3d,
        fields_2d,
        parent.xsize,
        parent.ysize,
        gd["z"],
        gd["zh"],
        parent.itot,
        parent.jtot,
        domain.xsize,
        domain.ysize,
        gd["z"],
        gd["zh"],
        domain.itot,
        domain.jtot,
        domain.xstart_in_parent,
        domain.ystart_in_parent,
        parent_exp_dir,
        exp_dir,
        float_type=float_type,
        name_suffix="overwrite")

    def link_files(src_pattern):
        files = glob.glob(str(src_pattern))

        for f in files:
            src = Path(f).resolve()
            name = src.name.replace("_out", "")
            dst = exp_dir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

    link_files(parent_exp_dir / "lbc_*_out.*")
    link_files(parent_exp_dir / "w_top_out.*")
    link_files(parent_exp_dir / "*_buffer_out.*")
