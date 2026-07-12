# The Rust implementation of falkordb

[![Rust](https://github.com/FalkorDB/falkordb-rs-next-gen/actions/workflows/rust-push.yml/badge.svg)](https://github.com/FalkorDB/falkordb-rs-next-gen/actions/workflows/rust-push.yml)
[![codecov](https://codecov.io/gh/FalkorDB/falkordb-rs-next-gen/branch/main/graph/badge.svg)](https://codecov.io/gh/FalkorDB/falkordb-rs-next-gen)
[![license](https://img.shields.io/badge/license-Server_Side_Public_License-green)](https://github.com/FalkorDB/falkordb-rs-next-gen/blob/main/LICENSE)

## Developer Guide

### Quick Start with Dev Container (Recommended)

The easiest way to get started is using the development container, which includes all dependencies pre-installed:

1. Install [Docker](https://docs.docker.com/get-docker/) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
3. Open this project in VS Code
4. Click "Reopen in Container" when prompted (or press F1 and select "Dev Containers: Reopen in Container")
5. Wait for the container to build (first time takes ~10-15 minutes)
6. Start developing! All dependencies are ready to use.

See [.devcontainer/README.md](.devcontainer/README.md) for more details.

### Manual Setup

If you prefer to set up the environment manually:

#### Build

```
cargo build
```

#### Dependencies:

GraphBLAS, LAGraph, and RediSearch must be built and installed before building this project.

##### Toolchain prerequisites

| Host | Compiler | OpenMP runtime |
| --- | --- | --- |
| macOS | `brew install llvm` (provides `clang` with OpenMP support) | `brew install libomp` |
| Linux | `clang-22` (e.g. from [apt.llvm.org](https://apt.llvm.org/)) | `apt install libomp-22-dev` |

Local builds use whatever OpenMP package is on the system — `build/libomp.sh`
is **not** required for local development. It is only invoked by the Docker
toolchain image (`build/Dockerfile`) to produce `/opt/libomp/lib/libomp.a`,
which lets the published `libfalkordb.{so,dylib}` embed libomp statically
and have no `libomp.so.5` / `libgomp.so.1` / `libomp.dylib` runtime
dependency. A locally-built artifact will dynamically link the system
libomp instead — fine for dev, but CI/Docker is the source of truth for
the self-contained image.

If you do want a self-contained local artifact (e.g. to mirror the Docker
build), run `build/libomp.sh` with a writable `PREFIX` and point
`graph/build.rs` at it via `LIBOMP_PREFIX`:

```bash
CC=$(brew --prefix llvm)/bin/clang PREFIX=$HOME/libomp ./build/libomp.sh
LIBOMP_PREFIX=$HOME/libomp cargo build
```

The script auto-detects the libomp source release from `${CC:-clang}
--version`, so it stays ABI-matched to your compiler with no manual
version arg. In Docker the same auto-detection runs against
`clang-${CLANG_MAJOR}`, eliminating the prior drift risk between the
apt-installed clang and a hand-pinned `LLVMORG_VERSION`.

##### Building GraphBLAS + LAGraph

[GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS.git) and
[LAGraph](https://github.com/GraphBLAS/LAGraph.git) are built by a single
script: GraphBLAS is installed system-wide, LAGraph is emitted under
`./lagraph_lib`.

On macOS, point the script at homebrew clang first:

```bash
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
./graphblas.sh
```

On Linux:

```bash
CC=clang-22 CXX=clang++-22 ./graphblas.sh
```

##### Building RediSearch

```bash
./redisearch.sh
```

- pytest - create virtualenv and install tests/requirements.txt

The virtual environment should be activated before running tests.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r tests/requirements.txt
```

### Testing

- run unit tests with `cargo test -p graph`

- run e2e and function tests with `pytest tests/test_e2e.py tests/test_functions.py -vv`

- run MVCC and concurrency tests with `pytest tests/test_mvcc.py tests/test_concurrency.py -vv`

- run flow tests with `./flow.sh`

- run tck tests with `pytest tests/tck/test_tck.py -s`

There is an option to run only part of the TCK tests and stop on the first fail

```bash
TCK_INCLUDE=tests/tck/features/expressions/list pytest tests/tck/test_tck.py -s
```

To run all passing TCK tests use:

```bash
TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s
```

### A/B benchmarking vs the production (C) image

[`.github/workflows/benchmark.yml`](.github/workflows/benchmark.yml) runs the
[FalkorDB/benchmark](https://github.com/FalkorDB/benchmark) A/B/C workload
against live containers — variant A (production C image
`falkordb/falkordb-server:edge`), B (published Rust `edge-rs`), and, on a PR,
C (that PR's image) — on **ephemeral self-hosted GCE VMs** (one per dataset
size, torn down after each run), mirroring FalkorDB/FalkorDB. Requires the
`GH_SA_TOKEN` / `GCP_SA_KEY` / `GCP_PROJECT_ID` secrets, the `GCP_ZONES`
variable, and the `gh-runner` GCP network.

- **On a PR** (from a branch in this repo — a fork PR's read-only token can't
  publish, so those are skipped): add a per-size label — `benchmark-small`,
  `benchmark-medium`, or `benchmark-large` — or comment `/benchmark [size]`
  (default `small`; comment authors need OWNER/MEMBER/COLLABORATOR
  association). Each run does **one size** (a full multi-size run is too
  expensive to fire on a single label) and is **read-only**; results
  **accumulate** across sizes into one published view and one PR comment.
  Variant C is that PR's `rc-pr-<N>` image; the workflow waits for that RC
  image to be published to the registry (built early by `rust-pr.yml`, before
  its slower test jobs finish), so it never runs against a missing build.
  Published to
  `https://falkordb.github.io/falkordb-rs-next-gen/benchmark/branch/pr-<N>/`
  (keyed by PR number, so same-named branches on different PRs never collide).
  A run happens only when a label is **added** (not on every push); to re-run a
  size, remove and re-add its label. Closing the PR removes its published view.
- **Manually**: run the "A/B benchmark" workflow from the Actions tab and
  choose the variant B (and optionally A) image tag, the dataset size, and the
  **write ratio** (default `0.0` — a manual run is the only way to benchmark
  writes; PR/slash runs are always read-only).
- **Profiling**: add a `profile-small` / `profile-medium` / `profile-large`
  label (or run "Profile (flame graph)" from the Actions tab) to publish an
  interactive CPU flame graph of the engine under the workload to
  `.../benchmark/branch/pr-<N>/profile/<size>/flamegraph.svg`.
- **Automatically**: every time `edge-rs` is promoted (merge to main), the
  canonical trend at
  [`/benchmark/`](https://falkordb.github.io/falkordb-rs-next-gen/benchmark/)
  is updated — this is the one continuously-growing history; PR/manual runs
  publish to their own capped-history branch view instead. A **per-query
  trend** (each query shape over time — C engine vs Rust, with server-exec and
  p50/p95/p99) is published alongside at
  [`/benchmark/trend/`](https://falkordb.github.io/falkordb-rs-next-gen/benchmark/trend/).
