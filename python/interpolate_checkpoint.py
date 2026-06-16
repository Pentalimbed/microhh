#!/usr/bin/env python3
"""Interpolate MicroHH checkpoint fields onto a target grid."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import microhh_tools as mht


STATIC_FILES = (
    "*.ini",
    "*_input.nc",
    "grid.0000000",
    "fftwplan.0000000",
    "thermo_basestate.0000000",
    "rhoref.0000000",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate MicroHH checkpoint fields onto another case grid."
    )
    parser.add_argument("source", type=Path, help="Source case directory")
    parser.add_argument("target", type=Path, help="Target case directory")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument(
        "--source-time",
        type=float,
        required=True,
        help="Source checkpoint time in seconds.",
    )
    parser.add_argument(
        "--source-ini",
        type=Path,
        default=None,
        help="Explicit source .ini file.",
    )
    parser.add_argument(
        "--target-ini",
        type=Path,
        default=None,
        help="Explicit target .ini file.",
    )
    parser.add_argument(
        "--method",
        choices=("linear", "nearest"),
        default="linear",
        help="Interpolation method.",
    )
    return parser.parse_args()


def find_ini(case_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    ini_files = sorted(case_dir.glob("*.ini"))
    if len(ini_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .ini file in {case_dir}, found {len(ini_files)}"
        )
    return ini_files[0]


def case_order(nl: mht.Read_namelist) -> int:
    try:
        return int(nl["grid"]["swspatialorder"])
    except KeyError:
        return 2


def time_suffix(time_seconds: float, iotimeprec: int) -> int:
    return int(round(time_seconds / (10**int(iotimeprec))))


def native_float_dtype(byte_size: int) -> np.dtype:
    if byte_size == 4:
        return np.dtype("f4").newbyteorder("<" if sys.byteorder == "little" else ">")
    if byte_size == 8:
        return np.dtype("f8").newbyteorder("<" if sys.byteorder == "little" else ">")
    raise RuntimeError(f"Unsupported float size: {byte_size}")


def infer_field_dtype(path: Path, npoints: int) -> np.dtype:
    filesize = path.stat().st_size
    if filesize % npoints != 0:
        raise RuntimeError(f"{path} size does not match expected field shape")
    return native_float_dtype(filesize // npoints)


def load_grid(case_dir: Path, ini_path: Path):
    nl = mht.Read_namelist(str(ini_path))
    gd = nl["grid"]
    grid = mht.Read_grid(
        int(gd["itot"]),
        int(gd["jtot"]),
        int(gd["ktot"]),
        order=case_order(nl),
        filename=str(case_dir / "grid.0000000"),
    )
    return nl, grid


def source_suffix(nl: mht.Read_namelist, source_time: float) -> int:
    iotimeprec = int(nl["time"].get("iotimeprec", 0))
    return time_suffix(source_time, iotimeprec)


def field_axes(name: str, grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = grid.dim
    if name == "u":
        return dims["z"], dims["y"], dims["xh"]
    if name == "v":
        return dims["z"], dims["yh"], dims["x"]
    if name == "w":
        return dims["zh"][:-1], dims["y"], dims["x"]
    return dims["z"], dims["y"], dims["x"]


def target_axes(name: str, grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = grid.dim
    if name == "u":
        return dims["z"], dims["y"], dims["xh"]
    if name == "v":
        return dims["z"], dims["yh"], dims["x"]
    if name == "w":
        return dims["zh"][:-1], dims["y"], dims["x"]
    return dims["z"], dims["y"], dims["x"]


def field_shape(name: str, grid) -> tuple[int, int, int]:
    dims = grid.dim
    if name == "u":
        return len(dims["z"]), len(dims["y"]), len(dims["xh"])
    if name == "v":
        return len(dims["z"]), len(dims["yh"]), len(dims["x"])
    if name == "w":
        return len(dims["zh"][:-1]), len(dims["y"]), len(dims["x"])
    return len(dims["z"]), len(dims["y"]), len(dims["x"])


def field_axes_2d(grid) -> tuple[np.ndarray, np.ndarray]:
    return grid.dim["y"], grid.dim["x"]


def field_axes_1d(grid) -> tuple[np.ndarray]:
    return (grid.dim["z"],)


def interpolate_field(
    data: np.ndarray,
    source_coords: tuple[np.ndarray, ...],
    target_coords: tuple[np.ndarray, ...],
    method: str,
    dtype: np.dtype,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        source_coords,
        data,
        method=method,
        bounds_error=False,
        fill_value=None,
    )

    shape = tuple(len(axis) for axis in target_coords)
    out = np.empty(shape, dtype=dtype)

    if len(target_coords) == 1:
        pts = np.asarray(target_coords[0], dtype=dtype).reshape(-1, 1)
        out[:] = np.asarray(interpolator(pts), dtype=dtype).reshape(shape)
    elif len(target_coords) == 2:
        yy, xx = np.meshgrid(target_coords[0], target_coords[1], indexing="ij")
        pts = np.column_stack((yy.ravel(), xx.ravel()))
        out[:] = np.asarray(interpolator(pts), dtype=dtype).reshape(shape)
    elif len(target_coords) == 3:
        zz, yy, xx = np.meshgrid(
            target_coords[0], target_coords[1], target_coords[2], indexing="ij"
        )
        pts = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))
        out[:] = np.asarray(interpolator(pts), dtype=dtype).reshape(shape)
    else:
        raise RuntimeError("Only 1D, 2D, and 3D interpolation are supported")

    return out


def copy_matching_files(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in sorted(src_dir.glob(pattern)):
            if path.is_file():
                shutil.copy2(path, dst_dir / path.name)


def main() -> None:
    args = parse_args()

    source_ini = find_ini(args.source, args.source_ini)
    target_ini = find_ini(args.target, args.target_ini)

    source_nl, source_grid = load_grid(args.source, source_ini)
    target_nl, target_grid = load_grid(args.target, target_ini)

    src_suffix = source_suffix(source_nl, args.source_time)
    dst_suffix = time_suffix(args.source_time, int(target_nl["time"].get("iotimeprec", 0)))
    src_tag = f"{src_suffix:07d}"
    dst_tag = f"{dst_suffix:07d}"

    source_files = sorted(args.source.glob(f"*.{src_tag}"))
    if not source_files:
        raise RuntimeError(f"No source checkpoint files found for *.{src_tag} in {args.source}")

    source_field_size = len(source_grid.dim["x"]) * len(source_grid.dim["y"]) * len(source_grid.dim["z"])
    source_dtype = None
    for path in source_files:
        if path.stem in {"grid", "time", "fftwplan", "thermo_basestate", "rhoref"}:
            continue
        try:
            source_dtype = infer_field_dtype(path, source_field_size)
            break
        except RuntimeError:
            continue

    if source_dtype is None:
        raise RuntimeError(f"Could not infer source field dtype from files in {args.source}")

    target_dtype = infer_field_dtype(
        args.target / "grid.0000000",
        2 * len(target_grid.dim["x"]) + 2 * len(target_grid.dim["y"]) + 2 * len(target_grid.dim["z"]),
    )

    args.output.mkdir(parents=True, exist_ok=True)

    if args.output.resolve() != args.target.resolve():
        copy_matching_files(args.target, args.output, STATIC_FILES)
        copy_matching_files(args.target, args.output, (f"*.{dst_tag}",))
    src_time = args.source / f"time.{src_tag}"
    if not src_time.exists():
        raise RuntimeError(f"Missing source time file: {src_time}")
    shutil.copy2(src_time, args.output / f"time.{dst_tag}")

    out_ini = args.output / target_ini.name
    out_nl = mht.Read_namelist(str(out_ini))
    out_nl.set_value("time", "starttime", args.source_time)
    out_nl.save(str(out_ini), allow_overwrite=True)

    written = []
    for path in source_files:
        if path.stem in {"grid", "time", "fftwplan", "thermo_basestate", "rhoref"}:
            continue
        name = path.name.rsplit(".", 1)[0]
        if "." in name:
            continue
        size = path.stat().st_size // source_dtype.itemsize

        if size == source_field_size:
            data = np.fromfile(path, dtype=source_dtype).reshape(field_shape(name, source_grid))
            src_coords = field_axes(name, source_grid)
            dst_coords = target_axes(name, target_grid)
            interpolated = interpolate_field(
                data,
                src_coords,
                dst_coords,
                method=args.method,
                dtype=target_dtype,
            )
        elif size == len(source_grid.dim["x"]) * len(source_grid.dim["y"]):
            data = np.fromfile(path, dtype=source_dtype).reshape(
                len(source_grid.dim["y"]), len(source_grid.dim["x"])
            )
            interpolated = interpolate_field(
                data,
                field_axes_2d(source_grid),
                field_axes_2d(target_grid),
                method=args.method,
                dtype=target_dtype,
            )
        elif size == len(source_grid.dim["z"]):
            data = np.fromfile(path, dtype=source_dtype).reshape(len(source_grid.dim["z"]))
            interpolated = interpolate_field(
                data,
                field_axes_1d(source_grid),
                field_axes_1d(target_grid),
                method=args.method,
                dtype=target_dtype,
            )
        else:
            shutil.copy2(path, args.output / path.name)
            continue

        out_path = args.output / f"{name}.{dst_tag}"
        interpolated.tofile(out_path)
        written.append(out_path.name)

    print(f"Wrote {len(written)} fields to {args.output}")
    for name in written:
        print(f" - {name}")


if __name__ == "__main__":
    main()
