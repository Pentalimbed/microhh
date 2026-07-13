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
phase covers the full `case.start` to `case.end` preview in `warmup/`. `run 0`
uses the selected warmup checkpoint in a separate `dom0/` directory and writes
BCs for domain 1; `run N` for `N > 0` uses the same interval on a local
zero-based clock.

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
`[[domains]]` entry to add another child domain. Child horizontal spacing must
be an integer refinement of the parent grid. All domains share the single
`[vertical_grid]` definition, which supports `equidistant`, LS2D's
`linear_stretched`, and LS2D's one- or two-transition `stretched` grid. The
generator prints the resulting domain-top height before preparing each run.
