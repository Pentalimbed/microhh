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

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import bz2
import os
import time

import numpy as np
import requests
import xarray as xr

import microhhpy.constants as cst
from microhhpy.logger import logger


ARCHIVE_BASE_URL = "https://data.source.coop/dynamical/dwd-icon-grib"
MODEL = "icon-eu"
GRID = "regular-lat-lon"
AREA = "europe"

REQUIRED_MODEL_FIELDS = ("T", "QV", "U", "V", "P")
W_FIELD = "W"
OPTIONAL_MODEL_FIELDS = ("QC", "QI")
OPTIONAL_SINGLE_LEVEL_FIELDS = ("PS", "T_G", "T_2M")
HHL_FIELD = "HHL"
DEFAULT_MODEL_LEVELS = "auto"
DEFAULT_FULL_MODEL_LEVELS = tuple(range(1, 61))
DEFAULT_MODEL_LEVEL_MARGIN = 500.0
DEFAULT_MAX_HHL_LEVEL_SEARCH = 100

LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")


@dataclass(frozen=True)
class GribField:
    values: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    attrs: dict


def _hourly_times(start: datetime, end: datetime) -> list[datetime]:
    if end < start:
        raise ValueError("end_date must not be earlier than start_date")

    out = []
    date = start
    while date <= end:
        out.append(date)
        date += timedelta(hours=1)
    return out


def _lead_hours(init_date: datetime, valid_date: datetime) -> int:
    seconds = (valid_date - init_date).total_seconds()
    lead = seconds / 3600

    if seconds < 0 or not lead.is_integer():
        raise ValueError("ICON valid times must be whole forecast hours after icon_init_date")

    return int(lead)


def _find_name(names, candidates):
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


class IconDriver:
    """
    Read ICON-EU GRIB output from the Source Cooperative archive and expose the
    arrays expected by microhhpy.real.create_input_from_regular_latlon().
    """

    def __init__(self, settings):
        self.settings = settings

        self.start = settings["start_date"]
        self.end = settings["end_date"]
        self.central_lon = settings["central_lon"]
        self.central_lat = settings["central_lat"]
        self.area_size = settings.get("area_size", 1.5)

        self.case_name = settings.get("case_name", "icon_openbc")
        self.data_path = Path(settings.get("icon_path", Path("data") / "ICON"))
        self.archive_base_url = settings.get("archive_base_url", ARCHIVE_BASE_URL).rstrip("/")
        self.download_missing = settings.get("download_missing", True)
        self.requested_model_levels = settings.get("model_levels", DEFAULT_MODEL_LEVELS)
        self.all_model_levels = tuple(settings.get("all_model_levels", DEFAULT_FULL_MODEL_LEVELS))
        self.all_half_levels = settings.get("all_half_levels")
        if self.all_half_levels is not None:
            self.all_half_levels = tuple(self.all_half_levels)
        self.max_hhl_level_search = settings.get("max_hhl_level_search", DEFAULT_MAX_HHL_LEVEL_SEARCH)
        self.model_level_margin = settings.get("model_level_margin", DEFAULT_MODEL_LEVEL_MARGIN)
        self.vertical_coverage_height = settings.get("vertical_coverage_height", settings.get("zsize"))
        self.hhl_lead = settings.get("hhl_lead")

        if self.requested_model_levels in (None, "auto"):
            self.model_levels = None
        else:
            self.model_levels = tuple(self.requested_model_levels)

        self.cycle_hours = tuple(settings.get("cycle_hours", (0, 6, 12, 18)))
        self.max_cycle_search_days = settings.get("max_cycle_search_days", 10)
        self.coverage_check_fields = tuple(settings.get("coverage_check_fields", ("T",)))
        if "coverage_check_levels" in settings:
            self.coverage_check_levels = tuple(settings["coverage_check_levels"])
        elif self.model_levels is None:
            self.coverage_check_levels = (min(self.all_model_levels), max(self.all_model_levels))
        else:
            self.coverage_check_levels = (self.model_levels[0], self.model_levels[-1])
        self.download_retries = settings.get("download_retries", 5)
        self.download_timeout = settings.get("download_timeout", (30, 180))
        self._availability_cache = {}
        self._hhl_cache = {}
        self.lats = None
        self.lons = None

        self.valid_dates = _hourly_times(self.start, self.end)

        self.init_date = settings.get("icon_init_date")
        if self.init_date is None:
            self.init_date = self._select_latest_cycle()

        self.leads = [_lead_hours(self.init_date, valid_date) for valid_date in self.valid_dates]

        if self.model_levels is None:
            self.model_levels = self._select_model_levels_from_hhl()
        else:
            self._check_contiguous_levels(self.model_levels, "model_levels")
        self.hhl_levels = self._hhl_levels_for_model_levels(self.model_levels)
        self.w_levels = None

        self.datetime = self.valid_dates
        self.time_sec = np.array(
            [(valid_date - self.valid_dates[0]).total_seconds() for valid_date in self.valid_dates],
            dtype=np.float64)

    def _select_latest_cycle(self):
        """
        Find the newest ICON forecast cycle with all required files for the
        requested valid-time range.
        """
        for init_date in self._candidate_cycles():
            if self._cycle_covers_range(init_date):
                logger.info(f"Using ICON forecast cycle {init_date:%Y-%m-%dT%H}.")
                return init_date

        date_range = f"{self.start:%Y-%m-%dT%H} to {self.end:%Y-%m-%dT%H}"
        raise FileNotFoundError(
            "Could not find an ICON forecast cycle that covers "
            f"{date_range}. Searched {self.max_cycle_search_days} days back "
            f"using cycle hours {self.cycle_hours}.")

    def _candidate_cycles(self):
        cycle_hours = sorted(set(self.cycle_hours), reverse=True)
        start_day = self.start.replace(hour=0, minute=0, second=0, microsecond=0)

        for day_offset in range(self.max_cycle_search_days + 1):
            day = start_day - timedelta(days=day_offset)
            for hour in cycle_hours:
                init_date = day.replace(hour=hour)
                if init_date <= self.start:
                    yield init_date

    def _cycle_covers_range(self, init_date):
        try:
            leads = [_lead_hours(init_date, valid_date) for valid_date in self.valid_dates]
        except ValueError:
            return False

        for lead in leads:
            for field in self.coverage_check_fields:
                for level in self.coverage_check_levels:
                    if not self._field_available(field, "model-level", lead, init_date, level=level):
                        return False

        return True

    def _field_available(self, field, level_type, lead, init_date, level=None):
        cache_key = (field, level_type, lead, init_date, level)
        if cache_key in self._availability_cache:
            return self._availability_cache[cache_key]

        bz2_path = self._local_bz2_path(field, level_type, lead, init_date=init_date, level=level)
        grib_path = bz2_path.with_suffix("")

        available = grib_path.exists() or bz2_path.exists()
        if not available and self.download_missing:
            url = self._field_url(field, level_type, lead, init_date=init_date, level=level)
            available = self._remote_file_available(url)

        self._availability_cache[cache_key] = available
        return available

    @staticmethod
    def _remote_file_available(url):
        try:
            response = requests.head(url, allow_redirects=True, timeout=(5, 20))
            if response.status_code == 200:
                return True
            if response.status_code == 404:
                return False
        except requests.RequestException:
            pass

        try:
            headers = {"Range": "bytes=0-0"}
            with requests.get(url, headers=headers, stream=True, timeout=(5, 20)) as response:
                return response.status_code in (200, 206)
        except requests.RequestException:
            return False

    def read_data(self):
        """
        Download missing files, read ICON fields, and calculate derived fields.
        """
        model_fields = {field: [] for field in REQUIRED_MODEL_FIELDS}
        w_fields = []
        optional_model_fields = {field: [] for field in OPTIONAL_MODEL_FIELDS}
        single_level_fields = {field: [] for field in OPTIONAL_SINGLE_LEVEL_FIELDS}

        attrs = {}

        hhl = self._read_hhl_levels(self.hhl_levels, required=True)
        self._store_grid(hhl)
        hhl_values = hhl.values.astype(np.float64)

        for lead in self.leads:
            for field in REQUIRED_MODEL_FIELDS:
                grib = self._read_model_field(field, lead, required=True)
                model_fields[field].append(grib.values)
                attrs[field] = grib.attrs
                self._store_grid(grib)

            grib = self._read_w_field(lead)
            w_fields.append(grib.values)
            attrs[W_FIELD] = grib.attrs
            self._store_grid(grib)

            for field in OPTIONAL_MODEL_FIELDS:
                grib = self._read_model_field(field, lead, required=False)
                if grib is not None:
                    optional_model_fields[field].append(grib.values)
                    attrs[field] = grib.attrs

            for field in OPTIONAL_SINGLE_LEVEL_FIELDS:
                grib = self._read_field(field, "single-level", lead, required=False)
                if grib is not None:
                    single_level_fields[field].append(grib.values)
                    attrs[field] = grib.attrs

        for field in list(optional_model_fields):
            if len(optional_model_fields[field]) != len(self.valid_dates):
                optional_model_fields[field] = None

        for field in list(single_level_fields):
            if len(single_level_fields[field]) != len(self.valid_dates):
                single_level_fields[field] = None

        self.T = np.stack(model_fields["T"]).astype(np.float64)
        self.qv = np.stack(model_fields["QV"]).astype(np.float64)
        self.u = np.stack(model_fields["U"]).astype(np.float64)
        self.v = np.stack(model_fields["V"]).astype(np.float64)
        self.w = np.stack(w_fields).astype(np.float64)
        self.p = self._to_pa(np.stack(model_fields["P"]).astype(np.float64), attrs.get("P", {}))

        self._orient_model_levels()

        self.qc = self._optional_stack(optional_model_fields.get("QC"), like=self.qv)
        self.qi = self._optional_stack(optional_model_fields.get("QI"), like=self.qv)
        self.hhl = np.repeat(hhl_values[np.newaxis, :, :, :], len(self.valid_dates), axis=0)

        self._align_w_levels(attrs.get(W_FIELD, {}))
        self.z = self._calculate_heights()

        self.ps = self._surface_pressure(single_level_fields.get("PS"))
        self.surface_temperature = self._surface_temperature(single_level_fields)

        self.ql = self.qc
        self.qt = self.qv + self.ql + self.qi
        self.exn = (self.p / cst.p0) ** (cst.Rd / cst.cp)
        self.thl = (
            self.T / self.exn
            - cst.Lv / (cst.cp * self.exn) * self.ql
            - cst.Ls / (cst.cp * self.exn) * self.qi)

        self.Tv = self.T * (1.0 + (cst.Rv / cst.Rd - 1.0) * self.qv - self.ql - self.qi)
        self.rho = self.p / (cst.Rd * self.Tv)
        self.wls = self._vertical_velocity_ms(attrs.get(W_FIELD, {}))

        self.fc = 2.0 * cst.e_rot * np.sin(np.deg2rad(self.central_lat))

    def _select_model_levels_from_hhl(self):
        if self.vertical_coverage_height is None:
            raise ValueError(
                "Automatic ICON model-level selection requires settings['zsize'] "
                "or settings['vertical_coverage_height'].")

        if self.all_half_levels is None:
            self.all_half_levels = self._discover_hhl_levels()

        hhl = self._read_hhl_levels(self.all_half_levels, required=True)
        hhl_values = hhl.values.astype(np.float64)

        if hhl_values.shape[0] != len(self.all_half_levels):
            raise ValueError(
                f"Expected {len(self.all_half_levels)} HHL half levels, got {hhl_values.shape[0]}.")

        half_levels_top_to_bottom = tuple(self.all_half_levels)
        hhl_top_to_bottom = hhl_values
        if np.nanmean(hhl_top_to_bottom[0, :, :]) < np.nanmean(hhl_top_to_bottom[-1, :, :]):
            half_levels_top_to_bottom = tuple(reversed(half_levels_top_to_bottom))
            hhl_top_to_bottom = hhl_top_to_bottom[::-1, :, :]

        half_levels_bottom_to_top = tuple(reversed(half_levels_top_to_bottom))
        hhl_bottom_to_top = hhl_top_to_bottom[::-1, :, :]
        hhl_agl = hhl_bottom_to_top - hhl_bottom_to_top[0:1, :, :]

        full_level_heights = 0.5 * (hhl_agl[:-1, :, :] + hhl_agl[1:, :, :])
        full_levels_bottom_to_top = tuple(
            min(lower, upper)
            for lower, upper in zip(half_levels_bottom_to_top[:-1], half_levels_bottom_to_top[1:]))

        required_height = float(self.vertical_coverage_height) + float(self.model_level_margin)
        selected_bottom_to_top = []
        selected_top_height = None
        for level, heights in zip(full_levels_bottom_to_top, full_level_heights):
            selected_bottom_to_top.append(level)
            min_height = float(np.nanmin(heights))
            if min_height >= required_height:
                selected_top_height = min_height
                break

        if selected_top_height is None:
            selected_top_height = float(np.nanmin(full_level_heights[-1, :, :]))
            logger.warning(
                "ICON HHL levels do not reach the requested vertical coverage "
                f"height {required_height:.1f} m AGL; using all available model levels.")

        model_levels = tuple(sorted(selected_bottom_to_top))
        self._check_contiguous_levels(model_levels, "selected model_levels")
        logger.info(
            "Selected ICON model levels "
            f"{model_levels[0]}..{model_levels[-1]} from HHL to cover "
            f"{required_height:.1f} m AGL; selected top is at least "
            f"{selected_top_height:.1f} m AGL over the crop.")
        return model_levels

    def _discover_hhl_levels(self):
        levels = []
        for level in range(1, self.max_hhl_level_search + 1):
            if self._field_available(HHL_FIELD, "model-level", 0, self.init_date, level=level):
                levels.append(level)
            elif levels:
                break

        if len(levels) < 2:
            raise FileNotFoundError(
                "Could not discover ICON HHL half levels in the archive. "
                f"Searched levels 1..{self.max_hhl_level_search}.")

        logger.info(f"Discovered ICON HHL half levels {levels[0]}..{levels[-1]}.")
        return tuple(levels)

    def _candidate_hhl_leads(self):
        leads = []
        if self.hhl_lead is not None:
            leads.append(self.hhl_lead)
        leads.extend([self.leads[0], 0])

        seen = set()
        for lead in leads:
            if lead < 0 or lead in seen:
                continue
            seen.add(lead)
            yield lead

    def _read_hhl_levels(self, levels, required):
        levels = tuple(levels)
        cache_key = levels
        if cache_key in self._hhl_cache:
            return self._hhl_cache[cache_key]

        errors = []
        for lead in self._candidate_hhl_leads():
            try:
                hhl = self._read_model_field_at_levels(HHL_FIELD, lead, levels, required=True)
                self.hhl_lead = lead
                self._hhl_cache[cache_key] = hhl
                return hhl
            except FileNotFoundError as exc:
                errors.append(str(exc))

        if required:
            msg = "\n".join(errors)
            raise FileNotFoundError(f"Could not read ICON HHL half levels {levels}: {msg}")
        return None

    def calculate_forcings(self, n_av=3, method="2nd"):
        """
        Calculate LS2D-like horizontally averaged profiles used by MicroHH.

        ICON already provides 3D boundaries for this case. For the 1D geostrophic
        profile, use the local ICON wind profile as a neutral default.
        """
        del method

        i = int(np.abs(self.lons - self.central_lon).argmin())
        j = int(np.abs(self.lats - self.central_lat).argmin())

        istart = max(0, i - n_av)
        iend = min(self.lons.size, i + n_av + 1)
        jstart = max(0, j - n_av)
        jend = min(self.lats.size, j + n_av + 1)

        center4d = np.s_[:, :, jstart:jend, istart:iend]
        center3d = np.s_[:, jstart:jend, istart:iend]

        for name in ("z", "p", "T", "thl", "qt", "u", "v", "wls"):
            setattr(self, f"{name}_mean", getattr(self, name)[center4d].mean(axis=(2, 3)))

        self.ug_mean = self.u_mean.copy()
        self.vg_mean = self.v_mean.copy()
        self.ps_mean = self.ps[center3d].mean(axis=(1, 2))
        self.sst_mean = self.surface_temperature[center3d].mean(axis=(1, 2))

        dlon = self.lons[iend - 1] - self.lons[istart]
        dlat = self.lats[jend - 1] - self.lats[jstart]
        self.area = f"{dlon:.2f} deg x {dlat:.2f} deg"

    def get_les_input(self, z):
        """
        Interpolate driver profiles to the LES vertical grid.
        """
        if not hasattr(self, "thl_mean"):
            self.calculate_forcings()

        def interp_z(array):
            out = np.empty((self.time_sec.size, z.size), dtype=np.float64)
            for t in range(self.time_sec.size):
                out[t, :] = np.interp(z, self.z_mean[t, :], array[t, :])
            return out

        def add_ds_var(ds, name, data, dims, long_name, units):
            ds[name] = (dims, data)
            ds[name].attrs["long_name"] = long_name
            ds[name].attrs["units"] = units

        ds = xr.Dataset(
            coords={
                "time": np.array(self.datetime, dtype="datetime64[ns]"),
                "z": z,
            }
        )

        add_ds_var(ds, "time_sec", self.time_sec, ("time",), "seconds since start of experiment", "s")
        add_ds_var(ds, "thl", interp_z(self.thl_mean), ("time", "z"), "liquid water potential temperature", "K")
        add_ds_var(ds, "qt", interp_z(self.qt_mean), ("time", "z"), "total specific humidity", "kg kg-1")
        add_ds_var(ds, "u", interp_z(self.u_mean), ("time", "z"), "zonal wind component", "m s-1")
        add_ds_var(ds, "v", interp_z(self.v_mean), ("time", "z"), "meridional wind component", "m s-1")
        add_ds_var(ds, "wls", interp_z(self.wls_mean), ("time", "z"), "vertical wind component", "m s-1")
        add_ds_var(ds, "p", interp_z(self.p_mean), ("time", "z"), "air pressure", "Pa")
        add_ds_var(ds, "ug", interp_z(self.ug_mean), ("time", "z"), "geostrophic zonal wind", "m s-1")
        add_ds_var(ds, "vg", interp_z(self.vg_mean), ("time", "z"), "geostrophic meridional wind", "m s-1")
        add_ds_var(ds, "ps", self.ps_mean, ("time",), "surface pressure", "Pa")
        add_ds_var(ds, "sst", self.sst_mean, ("time",), "surface temperature", "K")

        ds.attrs["fc"] = self.fc
        ds.attrs["central_lon"] = self.central_lon
        ds.attrs["central_lat"] = self.central_lat
        ds.attrs["area"] = self.area
        ds.attrs["source"] = "DWD ICON-EU GRIB via Source Cooperative"
        return ds

    def _field_url(self, field, level_type, lead, init_date=None, level=None):
        init_date = init_date or self.init_date
        init_dir = init_date.strftime("%Y-%m-%dT%H")
        init_stamp = init_date.strftime("%Y%m%d%H")
        if field == HHL_FIELD:
            if level is None:
                raise ValueError("ICON HHL files require a half-level number")
            prefix = f"{MODEL}_{AREA}_{GRID}_time-invariant"
            filename = f"{prefix}_{init_stamp}_{level}_{field}.grib2.bz2"
            return f"{self.archive_base_url}/{MODEL}/{GRID}/{init_dir}/{field.lower()}/{filename}"

        prefix = f"{MODEL}_{AREA}_{GRID}_{level_type}"
        if level_type == "model-level":
            if level is None:
                raise ValueError("model-level ICON files require a level number")
            filename = f"{prefix}_{init_stamp}_{lead:03d}_{level}_{field}.grib2.bz2"
        else:
            filename = f"{prefix}_{init_stamp}_{lead:03d}_{field}.grib2.bz2"
        return f"{self.archive_base_url}/{MODEL}/{GRID}/{init_dir}/{field.lower()}/{filename}"

    def _local_bz2_path(self, field, level_type, lead, init_date=None, level=None):
        init_date = init_date or self.init_date
        init_dir = init_date.strftime("%Y-%m-%dT%H")
        init_stamp = init_date.strftime("%Y%m%d%H")
        if field == HHL_FIELD:
            if level is None:
                raise ValueError("ICON HHL files require a half-level number")
            prefix = f"{MODEL}_{AREA}_{GRID}_time-invariant"
            filename = f"{prefix}_{init_stamp}_{level}_{field}.grib2.bz2"
            return self.data_path / MODEL / GRID / init_dir / field.lower() / filename

        prefix = f"{MODEL}_{AREA}_{GRID}_{level_type}"
        if level_type == "model-level":
            if level is None:
                raise ValueError("model-level ICON files require a level number")
            filename = f"{prefix}_{init_stamp}_{lead:03d}_{level}_{field}.grib2.bz2"
        else:
            filename = f"{prefix}_{init_stamp}_{lead:03d}_{field}.grib2.bz2"
        return self.data_path / MODEL / GRID / init_dir / field.lower() / filename

    def _ensure_grib(self, field, level_type, lead, required, level=None):
        bz2_path = self._local_bz2_path(field, level_type, lead, level=level)
        grib_path = bz2_path.with_suffix("")

        if grib_path.exists():
            return grib_path

        if not bz2_path.exists():
            if not self.download_missing:
                if required:
                    raise FileNotFoundError(f"Missing local ICON file: {bz2_path}")
                return None

            url = self._field_url(field, level_type, lead, level=level)
            bz2_path.parent.mkdir(parents=True, exist_ok=True)

            tmp_path = bz2_path.with_name(f"{bz2_path.name}.tmp")
            try:
                if not self._download_file(url, tmp_path, required):
                    return None
                os.replace(tmp_path, bz2_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        tmp_grib_path = grib_path.with_name(f"{grib_path.name}.tmp")
        try:
            with bz2.open(bz2_path, "rb") as src, open(tmp_grib_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            os.replace(tmp_grib_path, grib_path)
        finally:
            if tmp_grib_path.exists():
                tmp_grib_path.unlink()

        return grib_path

    def _download_file(self, url, tmp_path, required):
        for attempt in range(self.download_retries):
            try:
                with requests.get(url, stream=True, timeout=self.download_timeout) as response:
                    if response.status_code == 404:
                        if required:
                            raise FileNotFoundError(f"Required ICON file not found in archive: {url}")
                        return False

                    response.raise_for_status()
                    with open(tmp_path, "wb") as tmp_file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                tmp_file.write(chunk)
                return True

            except requests.RequestException as exc:
                if attempt + 1 == self.download_retries:
                    if required:
                        raise
                    logger.warning(f"Optional ICON download failed; skipping {url}: {exc}")
                    return False

                wait = 2 ** attempt
                logger.warning(
                    f"Download failed ({exc}); retrying in {wait}s "
                    f"({attempt + 1}/{self.download_retries}): {url}")
                time.sleep(wait)

    def _read_model_field(self, field, lead, required):
        return self._read_model_field_at_levels(field, lead, self.model_levels, required)

    def _read_w_field(self, lead):
        """
        ICON geometric vertical velocity is stored on generalVertical levels.
        Prefer the half-level range bounding the selected full model layers, then
        center adjacent half-level values onto the full-level grid used by MicroHH.
        """
        if self.w_levels is not None:
            return self._read_model_field_at_levels(W_FIELD, lead, self.w_levels, required=True)

        grib = self._read_model_field_at_levels(W_FIELD, lead, self.hhl_levels, required=False)
        if grib is not None:
            self.w_levels = self.hhl_levels
            return grib

        logger.warning(
            "Could not read ICON W on the selected HHL half-level range; "
            "falling back to selected model levels.")
        self.w_levels = self.model_levels
        return self._read_model_field_at_levels(W_FIELD, lead, self.w_levels, required=True)

    def _read_model_field_at_levels(self, field, lead, levels, required):
        fields = []

        for level in levels:
            grib = self._read_field(field, "model-level", lead, required, level=level)
            if grib is None:
                if required:
                    raise FileNotFoundError(f"Missing ICON {field} model level {level} for lead {lead:03d}")
                return None
            fields.append(grib)

        for grib in fields[1:]:
            self._store_grid(grib)

        return GribField(
            values=np.stack([grib.values for grib in fields]),
            lats=fields[0].lats,
            lons=fields[0].lons,
            attrs=fields[0].attrs)

    @staticmethod
    def _check_contiguous_levels(levels, name):
        levels = tuple(levels)
        if not levels:
            raise ValueError(f"{name} must contain at least one level")

        ordered = tuple(sorted(levels))
        expected = tuple(range(ordered[0], ordered[-1] + 1))
        if ordered != expected:
            raise ValueError(f"{name} must be contiguous, got {levels}")

    @classmethod
    def _hhl_levels_for_model_levels(cls, model_levels):
        cls._check_contiguous_levels(model_levels, "model_levels")
        ordered = tuple(sorted(model_levels))
        return tuple(range(ordered[0], ordered[-1] + 2))

    def _read_field(self, field, level_type, lead, required, level=None):
        grib_path = self._ensure_grib(field, level_type, lead, required, level=level)
        if grib_path is None:
            return None

        try:
            import cfgrib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Reading ICON GRIB files requires the cfgrib package.") from exc

        backend_kwargs = {"indexpath": f"{grib_path}.idx"}
        with xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=backend_kwargs) as ds:
            var_name = self._data_var_name(ds, field)
            da = ds[var_name].squeeze(drop=True)
            da = self._subset_latlon(da)

            lat_name = _find_name(da.coords, LAT_NAMES)
            lon_name = _find_name(da.coords, LON_NAMES)
            if lat_name is None or lon_name is None:
                raise ValueError(f"Could not identify lat/lon coordinates in {grib_path}")

            extra_dims = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
            if extra_dims:
                if len(extra_dims) != 1:
                    raise ValueError(f"Unexpected GRIB dimensions for {field}: {da.dims}")
                da = da.transpose(extra_dims[0], lat_name, lon_name)
            else:
                da = da.transpose(lat_name, lon_name)

            values = np.asarray(da.load().values, dtype=np.float64)
            lats = np.asarray(da[lat_name].values, dtype=np.float64)
            lons = np.asarray(da[lon_name].values, dtype=np.float64)
            attrs = dict(da.attrs)

        return GribField(values=values, lats=lats, lons=lons, attrs=attrs)

    def _subset_latlon(self, da):
        lat_name = _find_name(da.coords, LAT_NAMES)
        lon_name = _find_name(da.coords, LON_NAMES)

        if lat_name is None or lon_name is None:
            return da

        lon = da[lon_name]
        if float(lon.max()) > 180.0 and self.central_lon < 0.0:
            da = da.assign_coords({lon_name: (((lon + 180.0) % 360.0) - 180.0)})

        da = da.sortby(lon_name)
        da = da.sortby(lat_name)

        lon_min = self.central_lon - self.area_size
        lon_max = self.central_lon + self.area_size
        lat_min = self.central_lat - self.area_size
        lat_max = self.central_lat + self.area_size

        da = da.sel({lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)})

        if da.sizes[lat_name] < 2 or da.sizes[lon_name] < 2:
            raise ValueError(
                "ICON crop is empty or too small. Check central_lon/central_lat/area_size "
                "against the ICON-EU domain.")

        return da

    def _data_var_name(self, ds, field):
        candidates = [
            field.lower(),
            field.replace("_", "").lower(),
            field.lower().replace("_", ""),
        ]
        for candidate in candidates:
            if candidate in ds.data_vars:
                return candidate

        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            return data_vars[0]

        raise ValueError(f"Could not select variable {field}. Available variables: {data_vars}")

    def _store_grid(self, grib):
        if self.lats is None:
            self.lats = grib.lats
            self.lons = grib.lons
            return

        if not np.allclose(self.lats, grib.lats) or not np.allclose(self.lons, grib.lons):
            raise ValueError("ICON files do not share the same horizontal grid")

    def _optional_stack(self, values, like=None):
        if values is None:
            if like is None:
                return None
            return np.zeros_like(like)

        stacked = np.stack(values).astype(np.float64)
        if stacked.ndim == 4 and stacked.shape[1] == self.p.shape[1]:
            if self._reverse_model_levels:
                stacked = stacked[:, ::-1, :, :]
        return stacked

    def _orient_model_levels(self):
        self._reverse_model_levels = bool(np.nanmean(self.p[:, 0, :, :]) < np.nanmean(self.p[:, -1, :, :]))
        if not self._reverse_model_levels:
            return

        self.T = self.T[:, ::-1, :, :]
        self.qv = self.qv[:, ::-1, :, :]
        self.u = self.u[:, ::-1, :, :]
        self.v = self.v[:, ::-1, :, :]
        self.w = self.w[:, ::-1, :, :]
        self.p = self.p[:, ::-1, :, :]

    def _align_w_levels(self, attrs):
        if self.w.shape[1] == self.p.shape[1] + 1:
            self.w = 0.5 * (self.w[:, :-1, :, :] + self.w[:, 1:, :, :])
        elif self.w.shape[1] != self.p.shape[1]:
            raise ValueError(f"Unexpected ICON W level count: {self.w.shape[1]} for P count {self.p.shape[1]}")
        elif attrs.get("GRIB_typeOfLevel") == "generalVertical":
            logger.warning(
                "ICON W is on generalVertical levels, but only one W value per full level was read. "
                "Centering adjacent W levels and copying the top available W level.")
            w_centered = self.w.copy()
            w_centered[:, :-1, :, :] = 0.5 * (self.w[:, :-1, :, :] + self.w[:, 1:, :, :])
            self.w = w_centered

    def _calculate_heights(self):
        if self.hhl is not None:
            hhl = self.hhl
            if np.nanmean(hhl[:, 0, :, :]) > np.nanmean(hhl[:, -1, :, :]):
                hhl = hhl[:, ::-1, :, :]

            hsurf = hhl[:, 0:1, :, :]
            hhl_agl = hhl - hsurf

            if hhl_agl.shape[1] == self.p.shape[1] + 1:
                return 0.5 * (hhl_agl[:, :-1, :, :] + hhl_agl[:, 1:, :, :])
            if hhl_agl.shape[1] == self.p.shape[1]:
                return hhl_agl

        return self._hypsometric_heights()

    def _hypsometric_heights(self):
        z = np.zeros_like(self.p)
        tv = self.T * (1.0 + (cst.Rv / cst.Rd - 1.0) * self.qv - self.qc - self.qi)

        for k in range(1, self.p.shape[1]):
            tv_mean = 0.5 * (tv[:, k - 1, :, :] + tv[:, k, :, :])
            dz = cst.Rd * tv_mean / cst.grav * np.log(self.p[:, k - 1, :, :] / self.p[:, k, :, :])
            z[:, k, :, :] = z[:, k - 1, :, :] + dz

        return z

    def _surface_pressure(self, ps_values):
        if ps_values is None:
            return self.p[:, 0, :, :].copy()
        return self._to_pa(np.stack(ps_values).astype(np.float64), {})

    def _surface_temperature(self, single_level_fields):
        for field in ("T_2M", "T_G"):
            values = single_level_fields.get(field)
            if values is not None:
                return np.stack(values).astype(np.float64)
        return self.T[:, 0, :, :].copy()

    def _vertical_velocity_ms(self, attrs):
        units = attrs.get("GRIB_units") or attrs.get("units") or ""
        if "Pa" in units:
            return -self.w / (self.rho * cst.grav)
        return self.w.copy()

    @staticmethod
    def _to_pa(values, attrs):
        units = attrs.get("GRIB_units") or attrs.get("units") or ""
        if units.lower() in ("hpa", "hectopascal", "hectopascals"):
            return values * 100.0
        if np.nanmean(values) < 2_000.0:
            return values * 100.0
        return values
