# AMD Oracle Fixtures

These files pin the output of an external, independently-maintained
SuiteSparse AMD implementation against the inputs `feral-amd` tests
consume. They are the **external source of truth** referenced by
tests T4 and T12 in `dev/plans/ordering-amd-upgrade.md`.

## What is here

Each `*.txt` file records the result of running
[`amd` crate v0.2.2](https://crates.io/crates/amd) — a Rust port of
Timothy Davis's SuiteSparse AMD (BSD-3-Clause) — against one fixed
input, captured during Commit 2 of the `feral-amd` development plan:

- `n`, `nz`, `nz_a_plus_at`, `n_dense`, `ncmpa`
- `lnz` — nonzeros in L (excluding diagonal). This is the
  primary quality metric tests compare against.
- `ndiv`, `nms_ldl`, `nms_lu` — flop counters.
- `d_max` — maximum column count in L.
- `perm` — the computed permutation.

Each file is self-describing; the header comments name the fixture
and quote its provenance (generator spec or file SHA-256).

## Fixtures

| Fixture         | n   | Source                                             |
|-----------------|-----|----------------------------------------------------|
| `arrow_5`       | 5   | programmatic: hub at 0 + diagonal                  |
| `arrow_200`     | 200 | programmatic: hub at 0 + diagonal                  |
| `band_20_3`     | 20  | programmatic: banded, bandwidth 3                  |
| `diag_4`        | 4   | programmatic: diagonal (bandwidth 0)               |
| `tridiag_10`    | 10  | programmatic: tridiagonal                          |
| `grid_7x7`      | 49  | programmatic: 2D 5-point stencil                   |
| `amd_demo_24`   | 24  | programmatic (6×4 grid) — SYNTHETIC SUBSTITUTE (§) |
| `gh_258`        | 52  | file: faer-rs regression matrix (SHA-256 below)    |

### Provenance of `gh_258`

Input file: `../ripopt/ref/faer-rs/faer/test_data/sparse_cholesky/gh_258.txt`
SHA-256: `9f70a3cfb1b068984cf76b8b11da1a786a39c8701a1cc48a909fd25aca282c40`

### §  `amd_demo_24` is a synthetic substitute

The AMD algorithm's canonical worked example (Davis 2006, §7.2) is
shipped as `AMD/Demo/can_24.mtx` with SuiteSparse and requires a
network fetch we did not perform here. `amd_demo_24.txt` is
generated from a 6×4 2D grid as a same-sized stand-in. Replace in a
follow-up commit once `can_24.mtx` is downloaded.

Similarly, `HB/can_24` and `HB/bcsstk01` (referenced in the plan's
§T4) are deferred. They are expected to land in a follow-up commit
that also enables a CI job to fetch them.

## Reproducing

The harness that produced these files is preserved under `harness/`
as `.txt` files (extensionless so Cargo never picks them up).

```
mkdir -p /tmp/reproduce && cd /tmp/reproduce
cp .../tests/data/amd_oracle/harness/main.rs.txt  src/main.rs
cp .../tests/data/amd_oracle/harness/Cargo.toml.txt Cargo.toml
cargo run --release -- /tmp/out
diff -ru /tmp/out <feral-amd>/crates/feral-amd/tests/data/amd_oracle/
```

Harness file SHAs (for audit):

- `harness/main.rs.txt`: `8edbe44d85b94e5ee885570f61fd56f6f2bcbe912141d5f55b75aa397fa50eac`
- `harness/Cargo.toml.txt`: `de827a5d892099c7bf3d4fe56d9cefd54463a7b0964e442a4de50c4981ee3624`

## Clean-room isolation

The `amd` crate appears **only** in the harness (external, separate
Cargo project). It is not in `crates/feral-amd/Cargo.toml` and not
in the feral workspace's dependency graph. A CI grep enforces this
invariant.
