# Nested ERA5 Case

This case is a TOML-driven variant of `cases/era5_openbc` for chained nested
domains. ERA5 pressure-level, surface, and prescribed-radiation files are
downloaded directly with CDS API requests based on the ERF downloader pattern,
not through `ls2d.download_era5`.

Typical use:

```bash
uv run nested_input.py --config nested.toml --download-only
./run_all.sh nested.toml
```

`nested.toml` keeps only the fixed run knobs under `[run]`: the warmup
save/dump frequency, the actual nesting interval as datetime strings, the
subdomain BC output frequency, and the actual-run dump frequency. The warmup
phase always covers the full `case.start` to `case.end` interval. `run 0` uses
the nesting interval on the outer-domain restart clock and writes BCs for
domain 1; `run N` for `N > 0` uses the same interval on a local zero-based
clock.

Child domains all use the same `[run]` interval. When `dom0` writes BCs from
`run 0`, the generator links those parent-clock files onto the child's local
clock. Child domains that have their own child keep subdomain output enabled by
default; leaf domains keep it disabled.
Dump variable lists are inherited from `era5_openbc.ini.base`.

Individual steps can be run explicitly:

```bash
./run_domain.sh warmup nested.toml
./run_domain.sh run 0 nested.toml
./run_domain.sh run 1 nested.toml
```

The domain chain is the `[[domains]]` array in `nested.toml`. Add another
`[[domains]]` entry to add another child domain. Child horizontal spacing and
equidistant vertical spacing must be integer refinements of the parent grid.
Child `zsize` may be lower than the parent `zsize`, but it may not exceed it.
For shorter children, the child computes its top vertical velocity from lateral
mass balance instead of reading parent-saved `w_top`, because MicroHH's
subdomain writer saves `w_top` at the parent model top.
