#
#  MicroHH nested ERA5 case generator.
#
#  This case follows the Skyrim open-boundary workflow, but keeps all
#  user-facing parameters in a TOML file and downloads ERA5 directly with
#  cdsapi-style requests instead of calling ls2d.download_era5().
#

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re
import shutil
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import xarray as xr
import ls2d

from microhhpy.spatial import Domain, calc_vertical_grid_2nd
from microhhpy.real import create_input_from_regular_latlon
from microhhpy.real.input_from_regular_latlon import parse_pressure, setup_interpolations

import microhhpy.constants as cst
import microhhpy.io as io
import microhhpy.thermo as thermo


GRAV = 9.80665
OMEGA = 7.2921e-5

PRESSURE_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100",
    "125", "150", "175", "200", "225", "250", "300", "350", "400",
    "450", "500", "550", "600", "650", "700", "750", "775", "800",
    "825", "850", "875", "900", "925", "950", "975", "1000",
]

PRESSURE_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_humidity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "vorticity",
    "ozone_mass_mixing_ratio",
]

SURFACE_VARIABLES = [
    "surface_pressure",
    "2m_temperature",
    "skin_temperature",
    "sea_surface_temperature",
]

RADIATION_VARIABLES = [
    "surface_solar_radiation_downwards",
    "surface_net_solar_radiation",
    "surface_thermal_radiation_downwards",
    "surface_net_thermal_radiation",
]


@dataclass
class GridData:
    ktot: int
    zsize: float
    z: np.ndarray
    zh: np.ndarray
    dz: np.ndarray
    dzh: np.ndarray
    dzi: np.ndarray
    dzhi: np.ndarray
    dz0: float


@dataclass
class Era5Data:
    lon: np.ndarray
    lat: np.ndarray
    time: np.ndarray
    time_sec: np.ndarray
    z: np.ndarray
    p: np.ndarray
    fields: dict[str, np.ndarray]
    profiles: dict[str, np.ndarray]
    surface: dict[str, float]
    radiation_fluxes: dict[str, np.ndarray] | None = None
    radiation_profiles: dict[str, np.ndarray] | None = None


def parse_datetime(value) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    if isinstance(value, str):
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    raise TypeError(f"Cannot parse datetime from {value!r}")


def load_config(path: Path) -> dict:
    with path.open("rb") as f:
        cfg = tomllib.load(f)

    if "case" not in cfg:
        raise ValueError("TOML must contain a [case] section.")
    if "era5" not in cfg:
        raise ValueError("TOML must contain an [era5] section.")
    if "domains" not in cfg or len(cfg["domains"]) == 0:
        raise ValueError("TOML must contain at least one [[domains]] entry.")

    cfg["_path"] = path.resolve()
    cfg["_base_dir"] = path.resolve().parent
    cfg["case"]["start"] = parse_datetime(cfg["case"]["start"])
    cfg["case"]["end"] = parse_datetime(cfg["case"]["end"])

    return cfg


def float_type_from_config(cfg: dict):
    value = str(cfg["case"].get("float_type", "float64")).lower()
    if value in ("float32", "single", "sp"):
        return np.float32
    if value in ("float64", "double", "dp"):
        return np.float64
    raise ValueError(f"Unsupported float_type {value!r}.")


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def case_work_dir(cfg: dict) -> Path:
    return resolve_path(cfg["case"].get("work_dir", "."), cfg["_base_dir"])


def microhh_name(cfg: dict) -> str:
    return str(cfg["case"].get("microhh_name", "era5_openbc"))


def domain_dir(cfg: dict, domain_index: int) -> Path:
    domains_cfg = cfg["domains"]
    name = domains_cfg[domain_index].get("name", f"dom{domain_index}")
    return case_work_dir(cfg) / str(name)


def run_dir(cfg: dict, domain_index: int, run_name: str | None) -> Path:
    if run_name == "warmup":
        if domain_index != 0:
            raise ValueError("The warmup phase is only available for domain 0.")
        return case_work_dir(cfg) / "warmup"
    return domain_dir(cfg, domain_index)


def make_grid(cfg: dict, float_type) -> GridData:
    grid_cfg = cfg.get("vertical_grid")
    if not isinstance(grid_cfg, dict):
        raise ValueError("TOML must contain a [vertical_grid] table.")

    ktot = int(grid_cfg.get("ktot", 0))
    if ktot <= 0:
        raise ValueError("[vertical_grid].ktot must be positive.")

    grid_type = str(grid_cfg.get("type", "equidistant")).lower()
    if grid_type == "equidistant":
        if "zsize" in grid_cfg:
            zsize = float(grid_cfg["zsize"])
            dz0 = zsize / ktot
        elif "dz0" in grid_cfg:
            dz0 = float(grid_cfg["dz0"])
            zsize = ktot * dz0
        else:
            raise ValueError("Equidistant [vertical_grid] requires zsize or dz0.")
        z = (np.arange(ktot, dtype=float_type) + 0.5) * dz0
    elif grid_type == "linear_stretched":
        if "dz0" not in grid_cfg or "alpha" not in grid_cfg:
            raise ValueError("Linear-stretched [vertical_grid] requires dz0 and alpha.")
        dz0 = float(grid_cfg["dz0"])
        alpha = float(grid_cfg["alpha"])
        if alpha <= -1.0:
            raise ValueError("[vertical_grid].alpha must be greater than -1.")
        ls2d_grid = ls2d.grid.Grid_linear_stretched(kmax=ktot, dz0=dz0, alpha=alpha)
        z = np.asarray(ls2d_grid.z, dtype=float_type)
        zsize = float(ls2d_grid.zsize)
    elif grid_type == "stretched":
        required = ("dz0", "nloc1", "nbuf1", "dz1")
        missing = [key for key in required if key not in grid_cfg]
        if missing:
            raise ValueError(
                f"Stretched [vertical_grid] is missing: {', '.join(missing)}."
            )

        dz0 = float(grid_cfg["dz0"])
        nloc1 = float(grid_cfg["nloc1"])
        nbuf1 = float(grid_cfg["nbuf1"])
        dz1 = float(grid_cfg["dz1"])
        second_keys = ("nloc2", "nbuf2", "dz2")
        configured_second = [key for key in second_keys if key in grid_cfg]
        if configured_second and len(configured_second) != len(second_keys):
            raise ValueError(
                "The second stretched-grid transition requires nloc2, nbuf2, and dz2."
            )

        for name, value in (("nbuf1", nbuf1), ("dz1", dz1)):
            if value <= 0.0:
                raise ValueError(f"[vertical_grid].{name} must be positive.")
        if not 0.0 <= nloc1 <= ktot:
            raise ValueError("[vertical_grid].nloc1 must be between 0 and ktot.")

        kwargs = {}
        if configured_second:
            nloc2 = float(grid_cfg["nloc2"])
            nbuf2 = float(grid_cfg["nbuf2"])
            dz2 = float(grid_cfg["dz2"])
            if not 0.0 <= nloc2 <= ktot:
                raise ValueError("[vertical_grid].nloc2 must be between 0 and ktot.")
            if nbuf2 <= 0.0 or dz2 <= 0.0:
                raise ValueError("[vertical_grid].nbuf2 and dz2 must be positive.")
            kwargs = {"nloc2": nloc2, "nbuf2": nbuf2, "dz2": dz2}

        ls2d_grid = ls2d.grid.Grid_stretched(
            kmax=ktot,
            dz0=dz0,
            nloc1=nloc1,
            nbuf1=nbuf1,
            dz1=dz1,
            **kwargs,
        )
        z = np.asarray(ls2d_grid.z, dtype=float_type)
        zsize = float(ls2d_grid.zsize)
    else:
        raise ValueError(
            f"Unsupported [vertical_grid].type {grid_type!r}; use 'equidistant', "
            "'linear_stretched', or 'stretched'."
        )

    if grid_type != "equidistant":
        if "zsize" in grid_cfg and not np.isclose(
                float(grid_cfg["zsize"]), zsize, rtol=1e-10, atol=1e-8):
            raise ValueError(
                f"Stretched zsize is derived as {zsize}; remove the conflicting "
                f"configured value {grid_cfg['zsize']}."
            )

    if dz0 <= 0.0:
        raise ValueError("[vertical_grid].dz0 must be positive.")
    if zsize <= 0.0:
        raise ValueError("[vertical_grid].zsize must be positive.")

    gd = calc_vertical_grid_2nd(z, zsize, float_type=float_type)
    return GridData(dz0=dz0, **gd)


def validate_shared_vertical_grid(cfg: dict):
    vertical_keys = {"ktot", "kmax", "zsize", "dz", "dz0", "alpha", "vertical_grid"}
    for i, domain_cfg in enumerate(cfg["domains"]):
        configured = sorted(vertical_keys.intersection(domain_cfg))
        if configured:
            raise ValueError(
                f"Remove vertical settings from domains[{i}] ({', '.join(configured)}); "
                "all domains use [vertical_grid]."
            )


def horizontal_ghost_cells(swspatialorder, swadvec) -> int:
    spatial_order = str(swspatialorder).lower()
    advection = str(swadvec).lower()
    ghost_cells = {
        ("2", "0"): 1,
        ("2", "2"): 1,
        ("2", "2i4"): 2,
        ("2", "2i5"): 3,
        ("2", "2i6"): 3,
        ("2", "2i62"): 3,
        ("4", "0"): 3,
        ("4", "4"): 3,
        ("4", "4m"): 3,
    }
    try:
        return ghost_cells[(spatial_order, advection)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported swadvec={advection!r} with swspatialorder={spatial_order!r}."
        ) from exc


def configured_horizontal_ghost_cells(cfg: dict) -> int:
    base_ini = resolve_path(cfg["case"].get("base_ini", "era5_openbc.ini.base"), cfg["_base_dir"])
    ini = io.read_ini(str(base_ini))
    spatial_order = ini["grid"]["swspatialorder"]
    advection = ini.get("advec", {}).get("swadvec", spatial_order)
    return horizontal_ghost_cells(spatial_order, advection)


def build_domains(cfg: dict, grids: list[GridData]) -> list[Domain]:
    validate_shared_vertical_grid(cfg)
    era5_cfg = cfg["era5"]
    run_cfg = run_settings(cfg)
    actual_bcs_frequency = int(run_cfg["bcs_frequency"])
    era5_lbc_frequency = int(float(era5_cfg.get("download_interval_hours", 1)) * 3600.0)
    domains = []
    base_work_dir = case_work_dir(cfg)
    n_ghost = configured_horizontal_ghost_cells(cfg)

    for i, domain_cfg in enumerate(cfg["domains"]):
        default_lbc_frequency = era5_lbc_frequency if i == 0 else actual_bcs_frequency
        if "n_ghost" in domain_cfg:
            raise ValueError(
                f"Remove domains[{i}].n_ghost; it is derived from swadvec in the base INI."
            )
        kwargs = dict(
            xsize=float(domain_cfg["xsize"]),
            ysize=float(domain_cfg["ysize"]),
            itot=int(domain_cfg["itot"]),
            jtot=int(domain_cfg["jtot"]),
            n_ghost=n_ghost,
            n_sponge=int(domain_cfg.get("n_sponge", 3)),
            lbc_freq=default_lbc_frequency,
            buffer_freq=default_lbc_frequency,
            work_dir=str(base_work_dir / str(domain_cfg.get("name", f"dom{i}"))),
        )

        if i == 0:
            kwargs.update(
                lon=float(era5_cfg["central_lon"]),
                lat=float(era5_cfg["central_lat"]),
                anchor=str(domain_cfg.get("anchor", "center")),
                proj_str=str(era5_cfg["proj_str"]),
                start_date=cfg["case"]["start"],
                end_date=cfg["case"]["end"],
            )
        else:
            kwargs["parent"] = domains[i - 1]
            if domain_cfg.get("center_in_parent", True):
                kwargs["center_in_parent"] = True
            else:
                kwargs["xstart_in_parent"] = float(domain_cfg["xstart_in_parent"])
                kwargs["ystart_in_parent"] = float(domain_cfg["ystart_in_parent"])

        domain = Domain(**kwargs)
        domain.grid = grids[i]
        domain.index = i
        domain.name = str(domain_cfg.get("name", f"dom{i}"))
        domains.append(domain)

    for parent, child in zip(domains[:-1], domains[1:]):
        parent.child = child
        child.parent = parent
        child.grid_ratio_ij = integer_ratio(parent.dx, child.dx, "dx", child.name)
        child.grid_ratio_k = 1

    return domains


def integer_ratio(coarse: float, fine: float, name: str, domain_name: str) -> int:
    ratio = coarse / fine
    rounded = int(round(ratio))
    if rounded < 1 or not np.isclose(ratio, rounded, rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"{domain_name}: parent/child {name} ratio must be a positive integer; got {ratio}."
        )
    return rounded


def area_from_config(cfg: dict) -> list[float]:
    era5_cfg = cfg["era5"]
    if "area" in era5_cfg:
        area = [float(x) for x in era5_cfg["area"]]
    else:
        size = float(era5_cfg["area_size"])
        lat = float(era5_cfg["central_lat"])
        lon = float(era5_cfg["central_lon"])
        area = [lat + size, lon - size, lat - size, lon + size]

    if len(area) != 4:
        raise ValueError("ERA5 area must be [north, west, south, east].")
    if area[0] <= area[2]:
        raise ValueError("ERA5 area must be ordered north > south.")
    if area[3] <= area[1]:
        raise ValueError("ERA5 area must be ordered east > west.")
    return area


def hourly_times(start: dt.datetime, end: dt.datetime, interval_hours: int = 1) -> list[dt.datetime]:
    if start.minute or start.second or start.microsecond or end.minute or end.second or end.microsecond:
        raise ValueError("ERA5 download start/end must be on full hours.")
    if end < start:
        raise ValueError("End date must be after start date.")

    out = []
    current = start
    step = dt.timedelta(hours=interval_hours)
    while current <= end:
        out.append(current)
        current += step
    return out


def group_times_by_day(times: list[dt.datetime]) -> dict[dt.date, list[str]]:
    grouped = defaultdict(list)
    for stamp in times:
        grouped[stamp.date()].append(stamp.strftime("%H:%M"))
    return grouped


def era5_data_dir(cfg: dict) -> Path:
    return resolve_path(cfg["era5"].get("data_dir", "era5"), cfg["_base_dir"])


def era5_file(cfg: dict, kind: str, date: dt.date) -> Path:
    data_format = str(cfg["era5"].get("data_format", "grib")).lower()
    suffix = "grib" if data_format in ("grib", "grb") else "nc"
    prefix = str(cfg["era5"].get("file_prefix", microhh_name(cfg)))
    return era5_data_dir(cfg) / f"{prefix}_{kind}_{date:%Y%m%d}.{suffix}"


def expected_era5_files(cfg: dict, kind: str) -> list[Path]:
    start = cfg["case"]["start"]
    end = cfg["case"]["end"]
    if kind == "radiation":
        end = end + dt.timedelta(hours=1)

    dates = sorted({stamp.date() for stamp in hourly_times(start, end)})
    return [era5_file(cfg, kind, date) for date in dates]


def download_one_timestep(cds_client, dataset: str, request: dict, output_filename: Path, idx: str):
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    if output_filename.exists():
        print(f"[{idx}] Skipping existing: {output_filename}")
        return

    print(f"[{idx}] Downloading {output_filename} ...")
    cds_client.retrieve(dataset, request, str(output_filename))
    print(f"[{idx}] Done: {output_filename}")


def download_era5(cfg: dict):
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("cdsapi is required for --download.") from exc

    era5_cfg = cfg["era5"]
    if "cdsapirc" in era5_cfg:
        os.environ["CDSAPI_RC"] = str(resolve_path(era5_cfg["cdsapirc"], cfg["_base_dir"]))

    area = area_from_config(cfg)
    data_format = str(era5_cfg.get("data_format", "grib"))
    download_format = era5_cfg.get("download_format", "unarchived")
    api_format_key = str(era5_cfg.get("api_format_key", "data_format"))

    def base_request(date: dt.date, times: list[str]) -> dict:
        request = {
            "product_type": ["reanalysis"],
            "year": f"{date.year:04d}",
            "month": f"{date.month:02d}",
            "day": f"{date.day:02d}",
            "time": times,
            "area": area,
        }
        if api_format_key == "format":
            request["format"] = data_format
        else:
            request["data_format"] = data_format
            if download_format:
                request["download_format"] = download_format
        return request

    start = cfg["case"]["start"]
    end = cfg["case"]["end"]
    interval = int(era5_cfg.get("download_interval_hours", 1))

    pressure_times = group_times_by_day(hourly_times(start, end, interval))
    radiation_times = group_times_by_day(hourly_times(start, end + dt.timedelta(hours=1), interval))

    client = cdsapi.Client()
    requests = []

    for date, times in pressure_times.items():
        request = base_request(date, times)
        request.update(
            variable=era5_cfg.get("pressure_variables", PRESSURE_VARIABLES),
            pressure_level=[str(v) for v in era5_cfg.get("pressure_levels", PRESSURE_LEVELS)],
        )
        requests.append((
            "reanalysis-era5-pressure-levels",
            request,
            era5_file(cfg, "pressure", date),
            f"pressure:{date:%Y%m%d}",
        ))

        request = base_request(date, times)
        request.update(variable=era5_cfg.get("surface_variables", SURFACE_VARIABLES))
        requests.append((
            "reanalysis-era5-single-levels",
            request,
            era5_file(cfg, "surface", date),
            f"surface:{date:%Y%m%d}",
        ))

    scheme = str(cfg.get("radiation", {}).get("scheme", "none")).lower()
    if scheme == "prescribed":
        for date, times in radiation_times.items():
            request = base_request(date, times)
            request.update(variable=era5_cfg.get("radiation_variables", RADIATION_VARIABLES))
            requests.append((
                "reanalysis-era5-single-levels",
                request,
                era5_file(cfg, "radiation", date),
                f"radiation:{date:%Y%m%d}",
            ))

    for idx, (dataset, request, output_filename, label) in enumerate(requests):
        download_one_timestep(client, dataset, request, output_filename, f"{idx}:{label}")


def open_era5_dataset(files: list[Path]) -> xr.Dataset:
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ERA5 files:\n" + "\n".join(missing))

    first_suffix = files[0].suffix.lower()
    datasets = []
    for path in files:
        if first_suffix in (".grib", ".grb"):
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )
        else:
            ds = xr.open_dataset(path)
        datasets.append(normalize_time_coord(ds))

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = xr.combine_by_coords(datasets, combine_attrs="override")

    return sort_and_deduplicate_time(ds)


def normalize_time_coord(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    if "step" in ds.dims and "valid_time" in ds.coords:
        valid_time = ds["valid_time"].values

        if "time" in ds.dims:
            stacked = ds.stack(_era5_time=("time", "step"))
            valid_time = stacked["valid_time"].values
            drop_names = [name for name in ("time", "step", "valid_time") if name in stacked.coords]
            stacked = stacked.drop_vars(drop_names, errors="ignore")
            stacked = stacked.assign_coords(time=("_era5_time", valid_time))
            ds = stacked.swap_dims({"_era5_time": "time"}).drop_vars("_era5_time")
        else:
            # Accumulated ERA5 fields from cfgrib often have a scalar
            # forecast reference `time`, a `step` dimension, and
            # `valid_time(step)`. Use valid time as the MicroHH time axis.
            ds = ds.drop_vars(["time", "valid_time"], errors="ignore")
            ds = ds.assign_coords(time=("step", valid_time))
            ds = ds.swap_dims({"step": "time"})

    if "time" in ds.coords and "time" not in ds.dims:
        ds = ds.expand_dims(time=np.atleast_1d(ds["time"].values))

    return ds


def sort_and_deduplicate_time(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds

    ds = ds.sortby("time")
    times = ds["time"].values
    if times.size <= 1:
        return ds

    _, indices = np.unique(times, return_index=True)
    return ds.isel(time=np.sort(indices))


def coord_name(ds: xr.Dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find any of coordinates {names}.")


def data_var(ds: xr.Dataset, aliases: tuple[str, ...], required: bool = True):
    for name in aliases:
        if name in ds.data_vars:
            return ds[name]
    if required:
        raise KeyError(f"Could not find any of variables {aliases}.")
    return None


def prepare_lat_lon(ds: xr.Dataset) -> tuple[xr.Dataset, str, str]:
    lat_name = coord_name(ds, ("latitude", "lat"))
    lon_name = coord_name(ds, ("longitude", "lon"))

    lon = ds[lon_name].values
    if np.any(lon > 180.0):
        ds = ds.assign_coords({lon_name: ((lon + 180.0) % 360.0) - 180.0})

    ds = ds.sortby(lat_name).sortby(lon_name)
    return ds, lat_name, lon_name


def prepare_pressure_levels(ds: xr.Dataset) -> tuple[xr.Dataset, str, np.ndarray]:
    level_name = coord_name(ds, ("isobaricInhPa", "level", "pressure_level", "plev"))
    pressure = ds[level_name].values.astype(np.float64)
    if np.nanmax(pressure) < 2000.0:
        pressure = pressure * 100.0

    order = np.argsort(pressure)[::-1]
    ds = ds.isel({level_name: order})
    pressure = pressure[order]
    return ds, level_name, pressure


def select_time(ds: xr.Dataset, start: dt.datetime, end: dt.datetime) -> xr.Dataset:
    if "time" not in ds.coords:
        return ds
    return ds.sel(time=slice(np.datetime64(start), np.datetime64(end)))


def to_4d(da: xr.DataArray, time_name: str, level_name: str, lat_name: str, lon_name: str) -> np.ndarray:
    return da.transpose(time_name, level_name, lat_name, lon_name).values


def to_3d(da: xr.DataArray, time_name: str, lat_name: str, lon_name: str) -> np.ndarray:
    return da.transpose(time_name, lat_name, lon_name).values


def nanmean_silent(values: np.ndarray, axis):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    totals = np.where(finite, values, 0.0).sum(axis=axis)
    result = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def center_average(field: np.ndarray, lon: np.ndarray, lat: np.ndarray, cfg: dict) -> np.ndarray:
    n_av = int(cfg["era5"].get("n_av", 3))
    if lon.size == 0 or lat.size == 0:
        raise ValueError("Cannot average ERA5 fields over an empty lon/lat grid.")

    i = int(np.abs(lon - float(cfg["era5"]["central_lon"])).argmin())
    j = int(np.abs(lat - float(cfg["era5"]["central_lat"])).argmin())

    j0 = max(0, j - n_av)
    j1 = min(lat.size, j + n_av + 1)
    i0 = max(0, i - n_av)
    i1 = min(lon.size, i + n_av + 1)

    if field.ndim == 4:
        window = field[:, :, j0:j1, i0:i1]
        if window.shape[2] == 0 or window.shape[3] == 0:
            raise ValueError("Cannot average ERA5 4D field over an empty lon/lat slice.")
        return nanmean_silent(window, axis=(2, 3))
    if field.ndim == 3:
        window = field[:, j0:j1, i0:i1]
        if window.shape[1] == 0 or window.shape[2] == 0:
            raise ValueError("Cannot average ERA5 3D field over an empty lon/lat slice.")
        return nanmean_silent(window, axis=(1, 2))
    raise ValueError(f"Unsupported field rank {field.ndim}.")


def finite_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    return float(values[finite].mean())


def optional_era5_4d(
        ds: xr.Dataset,
        aliases: tuple[str, ...],
        template: np.ndarray,
        level_name: str,
        lat_name: str,
        lon_name: str) -> np.ndarray:
    da = data_var(ds, aliases, required=False)
    if da is None:
        return np.zeros_like(template)
    return to_4d(da, "time", level_name, lat_name, lon_name)


def read_era5(cfg: dict, include_radiation: bool) -> Era5Data:
    start = cfg["case"]["start"]
    end = cfg["case"]["end"]

    ds_p = open_era5_dataset(expected_era5_files(cfg, "pressure"))
    ds_s = open_era5_dataset(expected_era5_files(cfg, "surface"))

    ds_p = select_time(ds_p, start, end)
    ds_s = select_time(ds_s, start, end)

    ds_p, lat_name, lon_name = prepare_lat_lon(ds_p)
    ds_s, surf_lat_name, surf_lon_name = prepare_lat_lon(ds_s)
    ds_p, level_name, pressure = prepare_pressure_levels(ds_p)

    lon = ds_p[lon_name].values.astype(np.float64)
    lat = ds_p[lat_name].values.astype(np.float64)
    time = ds_p["time"].values
    time_sec = ((time - time[0]) / np.timedelta64(1, "s")).astype(np.float64)

    z_geo = to_4d(data_var(ds_p, ("z", "geopotential")), "time", level_name, lat_name, lon_name)
    z_geo = z_geo / GRAV

    temperature = to_4d(data_var(ds_p, ("t", "temperature")), "time", level_name, lat_name, lon_name)
    qv = to_4d(data_var(ds_p, ("q", "specific_humidity")), "time", level_name, lat_name, lon_name)
    qc = optional_era5_4d(ds_p, ("clwc", "specific_cloud_liquid_water_content"), qv, level_name, lat_name, lon_name)
    qi = optional_era5_4d(ds_p, ("ciwc", "specific_cloud_ice_water_content"), qv, level_name, lat_name, lon_name)
    qr = optional_era5_4d(ds_p, ("rwc", "specific_rain_water_content"), qv, level_name, lat_name, lon_name)
    qs = optional_era5_4d(ds_p, ("swc", "specific_snow_water_content"), qv, level_name, lat_name, lon_name)
    u = to_4d(data_var(ds_p, ("u", "u_component_of_wind")), "time", level_name, lat_name, lon_name)
    v = to_4d(data_var(ds_p, ("v", "v_component_of_wind")), "time", level_name, lat_name, lon_name)

    omega_da = data_var(ds_p, ("w", "vertical_velocity"), required=False)
    omega = np.zeros_like(u) if omega_da is None else to_4d(omega_da, "time", level_name, lat_name, lon_name)

    p = np.broadcast_to(pressure[None, :, None, None], z_geo.shape).copy()
    exn = (p / cst.p0) ** (cst.Rd / cst.cp)
    th = temperature / exn
    ql = qc + qi + qr + qs
    thl = th - cst.Lv * ql / (cst.cp * exn)
    qt = qv + ql

    tv = temperature * (1.0 + (cst.Rv / cst.Rd - 1.0) * qv - ql - qi - qr - qs)
    rho = p / (cst.Rd * tv)
    wls = -omega / (rho * GRAV)

    profiles = {
        "thl": center_average(thl, lon, lat, cfg),
        "qt": center_average(qt, lon, lat, cfg),
        "qv": center_average(qv, lon, lat, cfg),
        "u": center_average(u, lon, lat, cfg),
        "v": center_average(v, lon, lat, cfg),
        "ug": center_average(u, lon, lat, cfg),
        "vg": center_average(v, lon, lat, cfg),
        "w_ls": center_average(wls, lon, lat, cfg),
        "t": center_average(temperature, lon, lat, cfg),
        "p": np.broadcast_to(pressure[None, :], (time.size, pressure.size)),
        "z": center_average(z_geo, lon, lat, cfg),
    }

    o3_da = data_var(ds_p, ("o3", "ozone_mass_mixing_ratio"), required=False)
    if o3_da is not None:
        o3_mass = to_4d(o3_da, "time", level_name, lat_name, lon_name)
        profiles["o3"] = center_average(28.9644 / 47.9982 * o3_mass, lon, lat, cfg)
    else:
        o3_default = float(cfg.get("radiation", {}).get("o3_vmr", 4.0e-8))
        profiles["o3"] = np.full_like(profiles["qt"], o3_default)

    surface = {}
    sp_da = data_var(ds_s, ("sp", "surface_pressure"), required=False)
    t2m_da = data_var(ds_s, ("t2m", "2t", "2m_temperature"), required=False)
    skt_da = data_var(ds_s, ("skt", "skin_temperature"), required=False)
    sst_da = data_var(ds_s, ("sst", "sea_surface_temperature"), required=False)

    if sp_da is not None:
        sp = to_3d(sp_da, "time", surf_lat_name, surf_lon_name)
        ps = finite_mean(center_average(sp, ds_s[surf_lon_name].values, ds_s[surf_lat_name].values, cfg))
        surface["ps"] = ps if ps is not None else float(np.nanmean(profiles["p"][:, 0]))
    else:
        surface["ps"] = float(np.nanmean(profiles["p"][:, 0]))

    sfc_temperature = None
    if t2m_da is not None:
        t2m = to_3d(t2m_da, "time", surf_lat_name, surf_lon_name)
        sfc_temperature = finite_mean(center_average(t2m, ds_s[surf_lon_name].values, ds_s[surf_lat_name].values, cfg))
    if sfc_temperature is None:
        if skt_da is not None:
            skt = to_3d(skt_da, "time", surf_lat_name, surf_lon_name)
            sfc_temperature = finite_mean(center_average(skt, ds_s[surf_lon_name].values, ds_s[surf_lat_name].values, cfg))
    if sfc_temperature is None and sst_da is not None:
        sst = to_3d(sst_da, "time", surf_lat_name, surf_lon_name)
        sfc_temperature = finite_mean(center_average(sst, ds_s[surf_lon_name].values, ds_s[surf_lat_name].values, cfg))
    if sfc_temperature is None:
        sfc_temperature = finite_mean(profiles["t"][:, 0])
    if sfc_temperature is None:
        raise ValueError("Could not determine a finite surface temperature from 2m temperature, skin temperature, SST, or lowest-level air temperature.")
    surface["sfc_temperature"] = sfc_temperature

    radiation_fluxes = read_radiation_fluxes(cfg, time) if include_radiation else None
    radiation_profiles = make_radiation_profiles(cfg, profiles) if needs_rrtmgp(cfg) else None

    return Era5Data(
        lon=lon,
        lat=lat,
        time=time,
        time_sec=time_sec,
        z=z_geo,
        p=p,
        fields={"u": u, "v": v, "w": wls, "thl": thl, "qt": qt},
        profiles=profiles,
        surface=surface,
        radiation_fluxes=radiation_fluxes,
        radiation_profiles=radiation_profiles,
    )


def read_radiation_fluxes(cfg: dict, target_time: np.ndarray) -> dict[str, np.ndarray]:
    ds = open_era5_dataset(expected_era5_files(cfg, "radiation"))
    ds, lat_name, lon_name = prepare_lat_lon(ds)

    # ERA5 surface radiation fields are accumulated over the preceding hour.
    # Shift to interval centers, as done in the Cabauw case.
    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].values - np.timedelta64(30, "m"))

    seconds = float(cfg["era5"].get("radiation_accumulation_seconds", 3600.0))
    ds = ds.interp(time=target_time)

    lon = ds[lon_name].values
    lat = ds[lat_name].values

    def avg(aliases: tuple[str, ...]) -> np.ndarray:
        da = data_var(ds, aliases)
        values = to_3d(da, "time", lat_name, lon_name) / seconds
        return center_average(values, lon, lat, cfg)

    ssrd = avg(("ssrd", "surface_solar_radiation_downwards"))
    ssr = avg(("ssr", "surface_net_solar_radiation"))
    strd = avg(("strd", "surface_thermal_radiation_downwards"))
    net_lw = avg(("str", "surface_net_thermal_radiation"))

    time_sec = ((target_time - target_time[0]) / np.timedelta64(1, "s")).astype(np.float64)
    return {
        "time_surface": time_sec,
        "sw_flux_dn": ssrd,
        "sw_flux_up": ssrd - ssr,
        "lw_flux_dn": strd,
        "lw_flux_up": strd - net_lw,
    }


def needs_rrtmgp(cfg: dict) -> bool:
    return str(cfg.get("radiation", {}).get("scheme", "none")).lower() in ("rrtmgp", "rrtmgp_rt")


def make_radiation_profiles(cfg: dict, profiles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rad_cfg = cfg.get("radiation", {})
    qt_mean = profiles["qt"].mean(axis=0)
    eps = 18.01528 / 28.97
    h2o = qt_mean / (eps - eps * qt_mean)

    z_lay = profiles["z"].mean(axis=0)
    p_lay = profiles["p"].mean(axis=0)
    t_lay = profiles["t"].mean(axis=0)
    o3 = profiles["o3"].mean(axis=0)

    z_lev = center_to_edge(z_lay, lower=0.0)
    p_lev = np.maximum(center_to_edge(p_lay), 1.0)
    t_lev = center_to_edge(t_lay)

    return {
        "h2o": h2o,
        "o3": o3,
        "co2": np.full_like(h2o, float(rad_cfg.get("co2_vmr", 420.0e-6))),
        "ch4": np.full_like(h2o, float(rad_cfg.get("ch4_vmr", 1.9e-6))),
        "z_lay": z_lay,
        "z_lev": z_lev,
        "p_lay": p_lay,
        "p_lev": p_lev,
        "t_lay": t_lay,
        "t_lev": t_lev,
    }


def center_to_edge(values: np.ndarray, lower: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    if lower is None:
        edges[0] = values[0] + 0.5 * (values[0] - values[1])
    else:
        edges[0] = lower
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def add_nc_var(name: str, dims, nc_group, data, float_type):
    if name in nc_group.variables:
        return
    if dims is None:
        var = nc_group.createVariable(name, float_type)
    else:
        var = nc_group.createVariable(name, float_type, dims)
    var[:] = data


def add_nc_dim(name: str, size: int, nc_group):
    if name not in nc_group.dimensions:
        nc_group.createDimension(name, size)


def write_case_input(cfg: dict, exp_dir: Path, grid: GridData, era5: Era5Data, float_type, run_cfg: dict | None = None):
    output = exp_dir / f"{microhh_name(cfg)}_input.nc"
    nc_main = nc4.Dataset(output, mode="w", datamodel="NETCDF4", clobber=True)
    add_nc_dim("z", grid.ktot, nc_main)
    nc_init = nc_main.createGroup("init")

    add_nc_var("z", ("z",), nc_main, grid.z, float_type)

    for name in ("thl", "qt", "u", "v", "ug", "vg", "w_ls"):
        add_nc_var(name, ("z",), nc_init, interpolate_profile(era5.profiles[name][0], era5.profiles["z"][0], grid.z), float_type)

    if era5.radiation_fluxes is not None:
        nc_tdep = nc_main.createGroup("timedep")
        time_surface, timedep_indices = local_timedep_axis(
            era5.radiation_fluxes["time_surface"],
            run_cfg,
        )
        nt = time_surface.size
        add_nc_dim("time_surface", nt, nc_tdep)
        add_nc_var("time_surface", ("time_surface",), nc_tdep, time_surface, float_type)
        for name in ("sw_flux_dn", "sw_flux_up", "lw_flux_dn", "lw_flux_up"):
            add_nc_var(name, ("time_surface",), nc_tdep, era5.radiation_fluxes[name][timedep_indices], float_type)

    if era5.radiation_profiles is not None:
        write_rrtmgp_input(cfg, nc_main, nc_init, era5.radiation_profiles, grid, era5.profiles, float_type)

    nc_main.close()


def local_timedep_axis(time_values: np.ndarray, run_cfg: dict | None) -> tuple[np.ndarray, np.ndarray]:
    time_values = np.asarray(time_values, dtype=np.float64)
    if run_cfg is None:
        return time_values, np.arange(time_values.size)

    local_time = time_values - float(run_cfg.get("_time_origin_offset", 0.0))
    endtime = float(run_cfg.get("endtime", local_time[-1]))

    valid = np.flatnonzero(local_time >= 0.0)
    if valid.size == 0:
        raise ValueError("Timedep input does not cover the local run start.")

    selected = valid[local_time[valid] <= endtime]
    next_after_end = valid[local_time[valid] > endtime]
    if next_after_end.size:
        selected = np.append(selected, next_after_end[0])

    if selected.size < 2:
        raise ValueError("Timedep input must contain at least two local timestamps.")

    return local_time[selected], selected


def interpolate_profile(values: np.ndarray, z_in: np.ndarray, z_out: np.ndarray) -> np.ndarray:
    return np.interp(z_out, z_in, values, left=values[0], right=values[-1])


def write_rrtmgp_input(cfg: dict, nc_main, nc_init, rad: dict[str, np.ndarray], grid: GridData, profiles: dict[str, np.ndarray], float_type):
    add_nc_dim("lay", rad["z_lay"].size, nc_main)
    add_nc_dim("lev", rad["z_lev"].size, nc_main)
    nc_rad = nc_main.createGroup("radiation")
    add_nc_dim("lay", rad["z_lay"].size, nc_rad)
    add_nc_dim("lev", rad["z_lev"].size, nc_rad)

    for name in ("h2o", "o3", "co2", "ch4"):
        add_nc_var(name, ("lay",), nc_rad, rad[name], float_type)

    for name in ("z_lay", "p_lay", "t_lay"):
        add_nc_var(name, ("lay",), nc_rad, rad[name], float_type)
    for name in ("z_lev", "p_lev", "t_lev"):
        add_nc_var(name, ("lev",), nc_rad, rad[name], float_type)

    h2o_les = interpolate_profile(rad["h2o"], rad["z_lay"], grid.z)
    o3_les = interpolate_profile(rad["o3"], rad["z_lay"], grid.z)
    add_nc_var("h2o", ("z",), nc_init, h2o_les, float_type)
    add_nc_var("o3", ("z",), nc_init, o3_les, float_type)
    add_nc_var("co2", ("z",), nc_init, np.full(grid.ktot, float(cfg.get("radiation", {}).get("co2_vmr", 420.0e-6))), float_type)
    add_nc_var("ch4", ("z",), nc_init, np.full(grid.ktot, float(cfg.get("radiation", {}).get("ch4_vmr", 1.9e-6))), float_type)

    for group in (nc_init, nc_rad):
        add_nc_var("n2o", None, group, 3.2699e-7, float_type)
        add_nc_var("n2", None, group, 0.781, float_type)
        add_nc_var("o2", None, group, 0.209, float_type)


def configure_ini(cfg: dict, exp_dir: Path, domain: Domain, grid: GridData, era5: Era5Data, float_type, run_cfg: dict | None = None):
    base_ini = resolve_path(cfg["case"].get("base_ini", "era5_openbc.ini.base"), cfg["_base_dir"])
    ini = io.read_ini(str(base_ini))
    dom_cfg = cfg["domains"][domain.index]

    ini["master"]["npx"] = int(dom_cfg.get("npx", cfg["case"].get("npx", 1)))
    ini["master"]["npy"] = int(dom_cfg.get("npy", cfg["case"].get("npy", 1)))

    ini["grid"]["itot"] = domain.itot
    ini["grid"]["jtot"] = domain.jtot
    ini["grid"]["ktot"] = grid.ktot
    ini["grid"]["xsize"] = domain.xsize
    ini["grid"]["ysize"] = domain.ysize
    ini["grid"]["zsize"] = grid.zsize
    configure_micro_ini(cfg, ini)

    ps = era5.surface["ps"]
    sfc_temperature = era5.surface["sfc_temperature"]

    thl = interpolate_profile(era5.profiles["thl"].mean(axis=0), era5.profiles["z"].mean(axis=0), grid.z)
    qt = interpolate_profile(era5.profiles["qt"].mean(axis=0), era5.profiles["z"].mean(axis=0), grid.z)

    stop_thl = (thl[-1] - thl[-2]) / (grid.z[-1] - grid.z[-2])
    stop_qt = (qt[-1] - qt[-2]) / (grid.z[-1] - grid.z[-2])
    sbot_thl = sfc_temperature / thermo.exner(ps)
    sbot_qt = 0.95 * thermo.qsat(ps, sfc_temperature)

    zstart_buffer = domain_buffer_zstart(cfg, domain.index, grid)

    ini["buffer"]["zstart"] = zstart_buffer
    ini["thermo"]["pbot"] = ps
    ini["force"]["fc"] = 2.0 * OMEGA * np.sin(np.deg2rad(float(cfg["era5"]["central_lat"])))

    ini["boundary"]["stop[thl]"] = stop_thl
    ini["boundary"]["stop[qt]"] = stop_qt
    ini["boundary"]["sbot[thl]"] = sbot_thl
    ini["boundary"]["sbot[qt]"] = sbot_qt

    ini["time"]["endtime"] = (cfg["case"]["end"] - cfg["case"]["start"]).total_seconds()
    d = cfg["case"]["start"]
    ini["time"]["datetime_utc"] = format_datetime_utc(d)

    ini["cross"]["xz"] = domain.ysize / 2.0
    ini["cross"]["yz"] = domain.xsize / 2.0

    ini["boundary_lateral"]["n_sponge"] = domain.n_sponge
    ini["boundary_lateral"]["tau_sponge"] = float(dom_cfg.get("tau_sponge", 60.0))
    ini["boundary_lateral"]["loadfreq"] = domain.lbc_freq
    ini["boundary_lateral"]["slist"] = lbc_scalar_list(cfg, domain.index)

    ini["buffer"]["loadfreq"] = int(domain.buffer_freq or domain.lbc_freq)
    configure_radiation_ini(cfg, ini)
    configure_subdomain_ini(cfg, ini, domain)
    apply_run_overrides(ini, run_cfg)

    if io.check_ini(ini):
        raise RuntimeError("Some ini values are None.")

    io.save_ini(ini, str(exp_dir / f"{microhh_name(cfg)}.ini"))


def domain_buffer_zstart(cfg: dict, domain_index: int, grid: GridData) -> float:
    domain_cfg = cfg["domains"][domain_index]
    return float(domain_cfg.get("zstart_buffer", 0.75 * grid.zsize))


def run_by_name(cfg: dict, run_name: str | None, domain_index: int = 0) -> dict | None:
    if run_name is None:
        return None

    run_cfgs = run_settings(cfg)
    case_start = cfg["case"]["start"]

    if run_name == "warmup":
        frequency = float(run_cfgs["warmup_frequency"])
        return {
            "starttime": 0.0,
            "endtime": (cfg["case"]["end"] - case_start).total_seconds(),
            "savetime": frequency,
            "datetime_utc": case_start,
            "_time_origin_offset": 0.0,
            "dump": {"enabled": True, "sampletime": frequency},
            "subdomain": {"enabled": False},
        }

    if run_name == "run" and domain_index == 0:
        start, end = nesting_interval(cfg)
        return {
            "starttime": seconds_since_case_start(cfg, start),
            "endtime": seconds_since_case_start(cfg, end),
            "savetime": float(run_cfgs["dump_frequency"]),
            "datetime_utc": case_start,
            "_time_origin_offset": 0.0,
            "dump": {"enabled": True, "sampletime": float(run_cfgs["dump_frequency"])},
            "subdomain": {
                "enabled": True,
                "savetime_bcs": int(run_cfgs["bcs_frequency"]),
                "savetime_buffer": int(run_cfgs["bcs_frequency"]),
            },
        }

    if run_name == "run" and domain_index > 0:
        start, end = nesting_interval(cfg)
        duration = (end - start).total_seconds()
        bcs_frequency = int(run_cfgs["bcs_frequency"])
        return {
            "starttime": 0.0,
            "endtime": duration,
            "savetime": float(run_cfgs["dump_frequency"]),
            "datetime_utc": start,
            "_time_origin_offset": seconds_since_case_start(cfg, start),
            "dump": {"enabled": True, "sampletime": float(run_cfgs["dump_frequency"])},
            "boundary_lateral": {"loadfreq": bcs_frequency},
            "buffer": {"loadfreq": bcs_frequency},
            "subdomain": {
                "savetime_bcs": bcs_frequency,
                "savetime_buffer": bcs_frequency,
            },
        }

    raise ValueError(f"Unknown run phase {run_name!r}.")


def run_settings(cfg: dict) -> dict:
    run_cfg = cfg.get("run")
    if not isinstance(run_cfg, dict):
        raise ValueError("TOML must contain a [run] table.")

    required = ("warmup_frequency", "start", "end", "bcs_frequency", "dump_frequency")
    missing = [key for key in required if key not in run_cfg]
    if missing:
        raise ValueError(f"[run] is missing required keys: {', '.join(missing)}.")

    for key in ("warmup_frequency", "bcs_frequency", "dump_frequency"):
        value = float(run_cfg[key])
        rounded = round(value)
        if rounded <= 0 or not np.isclose(value, rounded):
            raise ValueError(f"[run].{key} must be a positive integer number of seconds.")

    return run_cfg


def nesting_interval(cfg: dict) -> tuple[dt.datetime, dt.datetime]:
    run_cfg = run_settings(cfg)
    start = parse_datetime(run_cfg["start"])
    end = parse_datetime(run_cfg["end"])
    if start < cfg["case"]["start"] or end > cfg["case"]["end"] or end <= start:
        raise ValueError("[run] start/end must define a positive interval within case.start/case.end.")
    return start, end


def seconds_since_case_start(cfg: dict, stamp: dt.datetime) -> float:
    return (stamp - cfg["case"]["start"]).total_seconds()


def format_datetime_utc(stamp: dt.datetime) -> str:
    return f"{stamp.year:04d}-{stamp.month:02d}-{stamp.day:02d} {stamp.hour:02d}:{stamp.minute:02d}:{stamp.second:02d}"


def apply_run_overrides(ini: dict, run_cfg: dict | None):
    if run_cfg is None:
        return

    if "starttime" in run_cfg:
        ini["time"]["starttime"] = float(run_cfg["starttime"])
    if "endtime" in run_cfg:
        ini["time"]["endtime"] = float(run_cfg["endtime"])
    if "savetime" in run_cfg:
        ini["time"]["savetime"] = float(run_cfg["savetime"])
    if "dt" in run_cfg:
        ini["time"]["dt"] = float(run_cfg["dt"])
    if "dtmax" in run_cfg:
        ini["time"]["dtmax"] = float(run_cfg["dtmax"])
    if "datetime_utc" in run_cfg:
        ini["time"]["datetime_utc"] = format_datetime_utc(parse_datetime(run_cfg["datetime_utc"]))

    if "dump" in run_cfg:
        dump_cfg = run_cfg["dump"]
        if "enabled" in dump_cfg:
            ini["dump"]["swdump"] = bool(dump_cfg["enabled"])
        if "sampletime" in dump_cfg:
            ini["dump"]["sampletime"] = float(dump_cfg["sampletime"])

    if "boundary_lateral" in run_cfg:
        lbc_cfg = run_cfg["boundary_lateral"]
        if "loadfreq" in lbc_cfg:
            ini["boundary_lateral"]["loadfreq"] = int(lbc_cfg["loadfreq"])

    if "buffer" in run_cfg:
        buffer_cfg = run_cfg["buffer"]
        if "loadfreq" in buffer_cfg:
            ini["buffer"]["loadfreq"] = int(buffer_cfg["loadfreq"])

    if "subdomain" in run_cfg:
        sub_cfg = run_cfg["subdomain"]
        if "enabled" in sub_cfg:
            enabled = bool(sub_cfg["enabled"])
            ini["subdomain"]["sw_subdomain"] = enabled
            if not enabled:
                ini["subdomain"]["sw_save_wtop"] = False
                ini["subdomain"]["sw_save_buffer"] = False
        if "save_wtop" in sub_cfg:
            ini["subdomain"]["sw_save_wtop"] = bool(sub_cfg["save_wtop"])
        if "save_buffer" in sub_cfg:
            ini["subdomain"]["sw_save_buffer"] = bool(sub_cfg["save_buffer"])
        if "savetime_bcs" in sub_cfg:
            ini["subdomain"]["savetime_bcs"] = int(sub_cfg["savetime_bcs"])
        if "savetime_buffer" in sub_cfg:
            ini["subdomain"]["savetime_buffer"] = int(sub_cfg["savetime_buffer"])


def lbc_scalar_list(cfg: dict, domain_index: int) -> list[str]:
    dom_cfg = cfg["domains"][domain_index]
    if "lbc_scalars" in dom_cfg:
        return [str(v) for v in dom_cfg["lbc_scalars"]]

    if domain_index == 0:
        return ["thl", "qt"]

    micro = str(cfg["case"].get("swmicro", "nsw6")).lower()
    if micro == "2mom_warm":
        return ["thl", "qt", "qr", "nr"]
    if micro == "nsw6":
        return ["thl", "qt", "qr", "qs", "qg"]
    return ["thl", "qt"]


def configure_micro_ini(cfg: dict, ini: dict):
    micro = str(cfg["case"].get("swmicro", ini["micro"].get("swmicro", "nsw6"))).lower()
    if micro in ("none", "false", "off", "0"):
        ini["micro"]["swmicro"] = False
        ini["limiter"]["limitlist"] = ["qt"]
        ini["advec"]["fluxlimit_list"] = ["qt"]
    elif micro == "2mom_warm":
        ini["micro"]["swmicro"] = "2mom_warm"
        ini["limiter"]["limitlist"] = ["qt", "qr", "nr"]
        ini["advec"]["fluxlimit_list"] = ["qt", "qr", "nr"]
    elif micro == "nsw6":
        ini["micro"]["swmicro"] = "nsw6"
        ini["limiter"]["limitlist"] = ["qt", "qr", "qs", "qg"]
        ini["advec"]["fluxlimit_list"] = ["qt", "qr", "nr"]
    else:
        raise ValueError(f"Unsupported case.swmicro {micro!r}.")


def configure_radiation_ini(cfg: dict, ini: dict):
    rad_cfg = cfg.get("radiation", {})
    era5_cfg = cfg.get("era5", {})
    scheme = str(rad_cfg.get("scheme", "none")).lower()

    ini["radiation"]["swtimedep_background"] = bool(rad_cfg.get("swtimedep_background", False))
    ini["radiation"]["swtimedep_prescribed"] = False
    ini["grid"]["lat"] = float(era5_cfg.get("central_lat", 256))
    ini["grid"]["lon"] = float(era5_cfg.get("central_lon", 256))

    if scheme in ("none", "false", "off"):
        ini["radiation"]["swradiation"] = False
    elif scheme == "prescribed":
        ini["radiation"]["swradiation"] = "prescribed"
        ini["radiation"]["swtimedep_prescribed"] = True
    elif scheme in ("rrtmgp", "rrtmgp_rt"):
        ini["radiation"]["swradiation"] = scheme
        if scheme == "rrtmgp_rt":
            ini["radiation"]["rays_per_pixel"] = int(rad_cfg.get("rays_per_pixel", 256))
            ini["radiation"]["kngrid_i"] = int(rad_cfg.get("kngrid_i", 64))
            ini["radiation"]["kngrid_j"] = int(rad_cfg.get("kngrid_j", 64))
            ini["radiation"]["kngrid_k"] = int(rad_cfg.get("kngrid_k", 32))
    else:
        raise ValueError(f"Unsupported radiation scheme {scheme!r}.")

    ini["aerosol"]["swaerosol"] = False
    ini["aerosol"]["swtimedep"] = False


def configure_subdomain_ini(cfg: dict, ini: dict, domain: Domain):
    child = domain.child
    if child is None:
        ini["subdomain"]["sw_subdomain"] = False
        ini["subdomain"]["sw_save_wtop"] = False
        ini["subdomain"]["sw_save_buffer"] = False
        return

    ini["subdomain"]["sw_subdomain"] = True
    ini["subdomain"]["xstart"] = child.xstart_in_parent
    ini["subdomain"]["ystart"] = child.ystart_in_parent
    ini["subdomain"]["xend"] = child.xstart_in_parent + child.xsize
    ini["subdomain"]["yend"] = child.ystart_in_parent + child.ysize
    ini["subdomain"]["grid_ratio_ij"] = child.grid_ratio_ij
    ini["subdomain"]["grid_ratio_k"] = 1
    ini["subdomain"]["n_ghost"] = child.n_ghost
    ini["subdomain"]["n_sponge"] = child.n_sponge
    ini["subdomain"]["savetime_bcs"] = child.lbc_freq
    ini["subdomain"]["sw_save_wtop"] = True
    ini["subdomain"]["sw_save_buffer"] = True
    ini["subdomain"]["savetime_buffer"] = child.buffer_freq or child.lbc_freq
    ini["subdomain"]["zstart_buffer"] = domain_buffer_zstart(cfg, child.index, child.grid)


def save_basestate(cfg: dict, exp_dir: Path, grid: GridData, era5: Era5Data, float_type):
    thl = interpolate_profile(era5.profiles["thl"][0], era5.profiles["z"][0], grid.z)
    qt = interpolate_profile(era5.profiles["qt"][0], era5.profiles["z"][0], grid.z)
    bs = thermo.calc_moist_basestate(thl, qt, era5.surface["ps"], grid.z, grid.zsize, float_type=float_type)
    thermo.save_basestate_density(bs["rho"], bs["rhoh"], str(exp_dir / "rhoref_overwrite.0000000"))
    return bs


def create_outer_domain_input(cfg: dict, exp_dir: Path, domain: Domain, grid: GridData, era5: Era5Data, bs: dict, float_type):
    case_cfg = cfg["case"]
    zstart_buffer = float(cfg["domains"][domain.index].get("zstart_buffer", 0.75 * grid.zsize))
    perturb = case_cfg.get("perturb_amplitude", {"thl": 0.1, "qt": 0.1e-3})

    create_input_from_regular_latlon(
        era5.fields,
        era5.lon,
        era5.lat,
        era5.z,
        era5.p,
        era5.time_sec,
        grid.z,
        grid.zsize,
        zstart_buffer,
        bs["rho"],
        bs["rhoh"],
        domain,
        float(case_cfg.get("sigma_h", 10_000.0)),
        perturb_size=int(case_cfg.get("perturb_size", 4)),
        perturb_amplitude={str(k): float(v) for k, v in perturb.items()},
        perturb_max_height=float(case_cfg.get("perturb_max_height", 0.0)),
        clip_at_zero=tuple(case_cfg.get("clip_at_zero", ("qt",))),
        name_suffix="overwrite",
        output_dir=str(exp_dir),
        ntasks=int(case_cfg.get("ntasks", 8)),
        float_type=float_type,
    )

    crop_horizontal_buffer_files(exp_dir, domain, grid, zstart_buffer, float_type)


def crop_horizontal_buffer_files(exp_dir: Path, domain: Domain, grid: GridData, zstart_buffer: float, float_type):
    """
    MicroHH reads 3D buffer files on scalar-shaped horizontal planes. The
    microhhpy regular-latlon writer stores u/v buffers on staggered planes, so
    crop the extra east/north row before MicroHH reads the files.
    """
    kstart_buffer = int(np.where(grid.z >= zstart_buffer)[0][0])
    ksize = grid.ktot - kstart_buffer
    expected = ksize * domain.jtot * domain.itot

    for name in ("u", "v"):
        for path in sorted(exp_dir.glob(f"{name}_buffer.*")):
            if re.search(r"\.\d{7}$", path.name) is None:
                continue

            data = np.fromfile(path, dtype=float_type)
            if data.size == expected:
                continue

            original_size = data.size
            if name == "u" and data.size % (ksize * domain.jtot) == 0:
                nx = data.size // (ksize * domain.jtot)
                if nx < domain.itot:
                    raise ValueError(f"{path} has too few x points for a scalar buffer.")
                data = data.reshape(ksize, domain.jtot, nx)[:, :, :domain.itot]
            elif name == "v" and data.size % (ksize * domain.itot) == 0:
                ny = data.size // (ksize * domain.itot)
                if ny < domain.jtot:
                    raise ValueError(f"{path} has too few y points for a scalar buffer.")
                data = data.reshape(ksize, ny, domain.itot)[:, :domain.jtot, :]
            else:
                raise ValueError(f"Unexpected {path.name} size {data.size}; expected {expected}.")

            print(f"Cropping {path.name} from {original_size} to {expected} values")
            data.astype(float_type, copy=False).tofile(path)


def regrid_child_from_parent(cfg: dict, exp_dir: Path, domain: Domain, grid: GridData, era5: Era5Data, float_type):
    parent = domain.parent
    parent_grid = parent.grid
    parent_exp_dir = domain_dir(cfg, domain.index - 1)
    parent_starttime = parent_output_starttime(cfg, domain)
    child_starttime = child_starttime_for_parent(cfg, domain)
    child_endtime = child_endtime_for_parent(cfg, domain)
    time_offset = child_starttime - parent_starttime
    parent_endtime = parent_starttime + (child_endtime - child_starttime)

    fields_3d = {name: parent_starttime for name in initial_3d_fields(cfg)}
    fields_2d = {}

    regrid_les_initial_state(
        fields_3d,
        fields_2d,
        parent.xsize,
        parent.ysize,
        parent_grid.z,
        parent_grid.zh,
        parent.itot,
        parent.jtot,
        domain.xsize,
        domain.ysize,
        grid.z,
        grid.zh,
        domain.itot,
        domain.jtot,
        domain.xstart_in_parent,
        domain.ystart_in_parent,
        str(parent_exp_dir),
        str(exp_dir),
        time_offset=time_offset,
        float_type=float_type,
        name_suffix="overwrite",
    )

    create_child_phydro_tod(cfg, exp_dir, domain, grid, era5, child_starttime, child_endtime, float_type)

    link_parent_outputs(
        parent_exp_dir,
        exp_dir,
        time_offset,
        child_starttime=child_starttime,
        child_endtime=child_endtime,
    )


def regrid_les_initial_state(
        fields_3d: dict[str, int | list[int]],
        fields_2d: dict[str, int | list[int]],
        xsize_in: float,
        ysize_in: float,
        z_in: np.ndarray,
        zh_in: np.ndarray,
        itot_in: int,
        jtot_in: int,
        xsize_out: float,
        ysize_out: float,
        z_out: np.ndarray,
        zh_out: np.ndarray,
        itot_out: int,
        jtot_out: int,
        xstart_out: float,
        ystart_out: float,
        path_in: str,
        path_out: str,
        time_offset: int = 0,
        float_type=np.float64,
        name_suffix: str = ""):
    name_suffix = f"_{name_suffix}" if name_suffix else ""
    path_in = Path(path_in)
    path_out = Path(path_out)

    if not np.array_equal(z_in, z_out) or not np.array_equal(zh_in, zh_out):
        raise ValueError("Parent and child must use the same vertical grid.")

    dx_in = xsize_in / itot_in
    dy_in = ysize_in / jtot_in
    x_in = np.arange(dx_in / 2.0, xsize_in, dx_in)
    xh_in = np.arange(0.0, xsize_in, dx_in)
    y_in = np.arange(dy_in / 2.0, ysize_in, dy_in)
    yh_in = np.arange(0.0, ysize_in, dy_in)

    dx_out = xsize_out / itot_out
    dy_out = ysize_out / jtot_out
    x_out = np.arange(dx_out / 2.0, xsize_out, dx_out) + xstart_out
    xh_out = np.arange(0.0, xsize_out, dx_out) + xstart_out
    y_out = np.arange(dy_out / 2.0, ysize_out, dy_out) + ystart_out
    yh_out = np.arange(0.0, ysize_out, dy_out) + ystart_out

    def dims(field: str):
        dim_x_in = xh_in if field == "u" else x_in
        dim_x_out = xh_out if field == "u" else x_out
        dim_y_in = yh_in if field == "v" else y_in
        dim_y_out = yh_out if field == "v" else y_out
        dim_z = zh_in[:-1] if field == "w" else z_in
        return dim_x_in, dim_x_out, dim_y_in, dim_y_out, dim_z

    def parse_times(times):
        if isinstance(times, int):
            return [times]
        return list(times)

    for field, times in fields_3d.items():
        dim_x_in, dim_x_out, dim_y_in, dim_y_out, dim_z = dims(field)
        for time in parse_times(times):
            data = np.fromfile(path_in / f"{field}.{time:07d}", dtype=float_type)
            data = data.reshape((dim_z.size, jtot_in, itot_in))
            da_in = xr.DataArray(
                data,
                coords={"z": dim_z, "y": dim_y_in, "x": dim_x_in},
                dims=["z", "y", "x"],
            )
            da_out = da_in.interp(
                x=dim_x_out,
                y=dim_y_out,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            da_out.values.astype(float_type).tofile(path_out / f"{field}{name_suffix}.{time + time_offset:07d}")

    for field, times in fields_2d.items():
        dim_x_in, dim_x_out, dim_y_in, dim_y_out, _ = dims(field)
        for time in parse_times(times):
            data = np.fromfile(path_in / f"{field}.{time:07d}", dtype=float_type)
            data = data.reshape((jtot_in, itot_in))
            da_in = xr.DataArray(
                data,
                coords={"y": dim_y_in, "x": dim_x_in},
                dims=["y", "x"],
            )
            da_out = da_in.interp(
                x=dim_x_out,
                y=dim_y_out,
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            da_out.values.astype(float_type).tofile(path_out / f"{field}{name_suffix}.{time + time_offset:07d}")


def phydro_tod_local_times(cfg: dict, starttime: int, endtime: int) -> list[int]:
    freq = int(float(cfg["era5"].get("download_interval_hours", 1)) * 3600.0)
    if freq <= 0:
        raise ValueError("era5.download_interval_hours must produce a positive phydro_tod frequency.")

    duration = max(0, endtime - starttime)
    last_offset = max(freq, ((duration + freq - 1) // freq) * freq)
    return list(range(starttime, starttime + last_offset + 1, freq))


def create_child_phydro_tod(
        cfg: dict,
        exp_dir: Path,
        domain: Domain,
        grid: GridData,
        era5: Era5Data,
        child_starttime: int,
        child_endtime: int,
        float_type):
    local_times = phydro_tod_local_times(cfg, child_starttime, child_endtime)
    start, _ = nesting_interval(cfg)
    origin_offset = integer_seconds(seconds_since_case_start(cfg, start), "[run].start offset")
    physical_times = [origin_offset + local_time for local_time in local_times]

    ip_s = setup_interpolations(era5.lon, era5.lat, domain.proj_pad, float_type=float_type)[2]
    for local_time, physical_time in zip(local_times, physical_times):
        parse_pressure(
            interpolate_time_field(era5.p, era5.time_sec, physical_time),
            interpolate_time_field(era5.z, era5.time_sec, physical_time),
            grid.zsize,
            ip_s,
            domain,
            local_time,
            str(exp_dir),
            float_type,
        )


def interpolate_time_field(field: np.ndarray, time_sec: np.ndarray, target_time: float) -> np.ndarray:
    time_sec = np.asarray(time_sec, dtype=np.float64)
    if target_time < time_sec[0] or target_time > time_sec[-1]:
        raise ValueError(
            f"Requested time {target_time} s is outside available ERA5 range "
            f"{time_sec[0]}..{time_sec[-1]} s."
        )

    upper = int(np.searchsorted(time_sec, target_time, side="left"))
    if upper < time_sec.size and np.isclose(time_sec[upper], target_time):
        return field[upper]
    if upper == 0 or upper == time_sec.size:
        raise ValueError(f"Could not bracket requested time {target_time} s in ERA5 data.")

    lower = upper - 1
    weight = (target_time - time_sec[lower]) / (time_sec[upper] - time_sec[lower])
    return (1.0 - weight) * field[lower] + weight * field[upper]


def integer_seconds(value: float, label: str) -> int:
    rounded = round(value)
    if not np.isclose(value, rounded):
        raise ValueError(f"{label} must be an integer number of seconds.")
    return int(rounded)


def parent_output_starttime(cfg: dict, domain: Domain) -> int:
    if domain.parent is None:
        return 0
    if domain.parent.index == 0:
        start, _ = nesting_interval(cfg)
        return integer_seconds(seconds_since_case_start(cfg, start), "[run].start offset")
    return 0


def child_starttime_for_parent(cfg: dict, domain: Domain) -> int:
    return 0


def child_endtime_for_parent(cfg: dict, domain: Domain) -> int:
    start, end = nesting_interval(cfg)
    return integer_seconds((end - start).total_seconds(), "[run] interval duration")


def initial_3d_fields(cfg: dict) -> list[str]:
    micro = str(cfg["case"].get("swmicro", "nsw6")).lower()
    if micro == "2mom_warm":
        return ["u", "v", "w", "thl", "qt", "qr", "nr"]
    if micro == "nsw6":
        return ["u", "v", "w", "thl", "qt", "qr", "qs", "qg"]
    return ["u", "v", "w", "thl", "qt"]


def link_file(source: Path, destination: Path):
    if not source.is_file():
        raise FileNotFoundError(f"Required warmup file is missing: {source}")
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def link_outer_run_inputs(cfg: dict, exp_dir: Path):
    warmup_dir = run_dir(cfg, 0, "warmup")
    case_name = microhh_name(cfg)
    start, end = nesting_interval(cfg)
    starttime = integer_seconds(seconds_since_case_start(cfg, start), "[run].start offset")
    endtime = integer_seconds(seconds_since_case_start(cfg, end), "[run].end offset")

    static_files = (
        f"{case_name}_input.nc",
        "grid.0000000",
        "fftwplan.0000000",
        "rhoref.0000000",
    )
    for name in static_files:
        link_file(warmup_dir / name, exp_dir / name)

    restart_fields = set(initial_3d_fields(cfg))
    restart_fields.update({
        "time", "thermo_basestate", "thl_bot", "qt_bot",
        "dudz_mo", "dvdz_mo", "dbdz_mo", "obuk",
    })
    for field in sorted(restart_fields):
        name = f"{field}.{starttime:07d}"
        link_file(warmup_dir / name, exp_dir / name)

    forcing_files: dict[str, dict[int, Path]] = defaultdict(dict)
    forcing_pattern = re.compile(
        r"^(?P<prefix>lbc_.+|w_top|.+_buffer|phydro_tod)\.(?P<time>\d{7})$"
    )
    for source in warmup_dir.iterdir():
        match = forcing_pattern.match(source.name)
        if match is None or match.group("prefix").endswith("_out"):
            continue
        forcing_files[match.group("prefix")][int(match.group("time"))] = source

    if not forcing_files:
        raise FileNotFoundError(f"No forcing files found in warmup directory {warmup_dir}.")

    for prefix, sources in forcing_files.items():
        before = [time for time in sources if time <= starttime]
        after = [time for time in sources if time >= endtime]
        if not before or not after:
            raise FileNotFoundError(
                f"Warmup forcing {prefix!r} does not bracket {starttime}..{endtime} s."
            )
        selected = {max(before), min(after)}
        selected.update(time for time in sources if starttime < time < endtime)
        for time in sorted(selected):
            source = sources[time]
            link_file(source, exp_dir / source.name)


def link_parent_outputs(
        parent_exp_dir: Path,
        exp_dir: Path,
        time_offset: int = 0,
        child_starttime: int | None = None,
        child_endtime: int | None = None):
    for pattern in ("lbc_*_out.*", "w_top_out.*", "*_buffer_out.*"):
        for src_name in glob.glob(str(parent_exp_dir / pattern)):
            src = Path(src_name).resolve()
            match = re.match(r"^(?P<prefix>.+)_out\.(?P<time>\d+)$", src.name)
            if match is None:
                continue

            child_time = int(match.group("time")) + time_offset
            if child_time < 0:
                continue
            if child_starttime is not None and child_time < child_starttime:
                continue
            if child_endtime is not None and child_time > child_endtime:
                continue

            dst = exp_dir / f"{match.group('prefix')}.{child_time:07d}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)


def copy_rrtmgp_coefficients(cfg: dict, exp_dir: Path):
    rad_cfg = cfg.get("radiation", {})
    if not needs_rrtmgp(cfg) or not bool(rad_cfg.get("copy_coefficients", False)):
        return

    microhh_root = resolve_path(rad_cfg.get("microhh_root", "../.."), cfg["_base_dir"])
    data_dir = microhh_root / "rte-rrtmgp-cpp" / "rrtmgp-data"
    gpt_set = str(rad_cfg.get("gpt_set", "128_112"))
    link = bool(rad_cfg.get("link_coefficients", True))

    if gpt_set == "256_224":
        pairs = [
            ("rrtmgp-gas-lw-g256.nc", "coefficients_lw.nc"),
            ("rrtmgp-gas-sw-g224.nc", "coefficients_sw.nc"),
        ]
    elif gpt_set == "128_112":
        pairs = [
            ("rrtmgp-gas-lw-g128.nc", "coefficients_lw.nc"),
            ("rrtmgp-gas-sw-g112.nc", "coefficients_sw.nc"),
        ]
    else:
        raise ValueError("radiation.gpt_set must be '128_112' or '256_224'.")

    pairs.extend([
        ("rrtmgp-clouds-lw.nc", "cloud_coefficients_lw.nc"),
        ("rrtmgp-clouds-sw.nc", "cloud_coefficients_sw.nc"),
    ])

    for src_name, dst_name in pairs:
        src = data_dir / src_name
        dst = exp_dir / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy(src, dst)


def prepare_exp_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_domain(cfg: dict, domain_index: int, download: bool = False, run_name: str | None = None):
    float_type = float_type_from_config(cfg)
    grid = make_grid(cfg, float_type)
    print(f"Domain height: {grid.zsize:.6f} m ({grid.ktot} vertical levels)")

    if download or bool(cfg["era5"].get("download", False)):
        download_era5(cfg)

    grids = [grid] * len(cfg["domains"])
    domains = build_domains(cfg, grids)

    if domain_index < 0 or domain_index >= len(domains):
        raise IndexError(f"Domain index {domain_index} out of range 0..{len(domains)-1}.")

    scheme = str(cfg.get("radiation", {}).get("scheme", "none")).lower()
    include_prescribed_radiation = scheme == "prescribed"
    era5 = read_era5(cfg, include_prescribed_radiation)

    domain = domains[domain_index]
    grid = grids[domain_index]
    exp_dir = run_dir(cfg, domain_index, run_name)
    prepare_exp_dir(exp_dir)
    run_cfg = run_by_name(cfg, run_name, domain_index=domain_index)

    configure_ini(cfg, exp_dir, domain, grid, era5, float_type, run_cfg=run_cfg)

    if run_name == "run" and domain_index == 0:
        link_outer_run_inputs(cfg, exp_dir)
        copy_rrtmgp_coefficients(cfg, exp_dir)
        return

    write_case_input(cfg, exp_dir, grid, era5, float_type, run_cfg=run_cfg)
    bs = save_basestate(cfg, exp_dir, grid, era5, float_type)
    copy_rrtmgp_coefficients(cfg, exp_dir)

    if domain_index == 0:
        create_outer_domain_input(cfg, exp_dir, domain, grid, era5, bs, float_type)
    else:
        regrid_child_from_parent(cfg, exp_dir, domain, grid, era5, float_type)


def list_domains(cfg: dict):
    print(len(cfg["domains"]))


def list_runs(cfg: dict):
    run_settings(cfg)
    print("warmup")
    print("run")


def main():
    parser = argparse.ArgumentParser(description="Nested ERA5 MicroHH input generator.")
    parser.add_argument("-c", "--config", default="nested.toml", help="TOML configuration file.")
    parser.add_argument("-d", "--domain", type=int, default=0, help="Domain index to generate.")
    parser.add_argument("--run-name", help="Apply a derived run phase: warmup or run.")
    parser.add_argument("--download", action="store_true", help="Download ERA5 data before generating input.")
    parser.add_argument("--download-only", action="store_true", help="Only download ERA5 data.")
    parser.add_argument("--list-domains", action="store_true", help="Print number of configured domains and exit.")
    parser.add_argument("--list-runs", action="store_true", help="Print configured run phase names and exit.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    if args.list_domains:
        list_domains(cfg)
        return

    if args.list_runs:
        list_runs(cfg)
        return

    if args.download_only:
        download_era5(cfg)
        return

    generate_domain(cfg, args.domain, download=args.download, run_name=args.run_name)


if __name__ == "__main__":
    main()
