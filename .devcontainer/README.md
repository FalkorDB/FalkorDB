# FalkorDB Development Container

This directory contains the configuration for the FalkorDB development container, which provides a fully configured environment for building, testing, and developing FalkorDB-rs.

## Features

The development container includes:

- **Ubuntu 24.04** as the base image
- **Redis server** installed via apt
- **Rust toolchain** with all necessary components
- **LLVM 21** with clang, clang++, llvm-cov, and llvm-profdata for building and code coverage
- **GraphBLAS** (v10.3.1) compiled and installed using `graphblas.sh`
- **RediSearch** with vector similarity support, built using `redisearch.sh`
- **Python 3 virtual environment** at `/data/venv` with all test dependencies

## Usage

### Using with VS Code

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the project in VS Code
3. Click the green button in the bottom-left corner or press `F1` and select "Dev Containers: Reopen in Container"
4. Wait for the container to build (first time will take 10-15 minutes)
5. Once ready, you can use all the Claude skills and build commands

### Manual Docker Usage

If you're not using VS Code, you can manually build and run the container:

```bash
# Build the dev container image (from project root)
docker build -t falkordb-dev -f .devcontainer/Dockerfile .

# Run the container with the workspace mounted
docker run --rm -it -v $(pwd):/workspace -w /workspace falkordb-dev bash

# Inside the container, activate the Python venv
source /data/venv/bin/activate

# Build the project
cargo build

# Run tests
cargo test -p graph
```

## Available Claude Skills

When working inside the devcontainer with Claude Code, you have access to the following skills:

- `/build` - Build and lint the project
- `/test` - Run various test suites (unit, e2e, flow, TCK, etc.)
- `/lint` - Run advanced Clippy lints (pedantic, nursery, cargo)
- `/coverage` - Run comprehensive code coverage analysis

All skills automatically detect whether they're running inside or outside the devcontainer and adjust accordingly.

## Environment Variables

The container sets the following environment variables:

- `CXX=clang++-22` - Required for building with GraphBLAS FFI
- `PATH` includes Rust cargo bin directory

## Cargo Cache

The devcontainer configuration mounts a local `.cargo-cache` directory to `/root/.cargo/registry` to speed up subsequent builds by caching downloaded dependencies. This directory is excluded from Git via `.gitignore`.

## Customization

You can customize the devcontainer by editing:

- `devcontainer.json` - VS Code settings, extensions, and post-create commands
- `Dockerfile` - Base image and installed dependencies

## Troubleshooting

### Container Build Fails

If the container build fails, it might be due to network issues when downloading dependencies. Try building again.

### Python Tests Fail

Make sure to activate the virtual environment before running Python tests:
```bash
source /data/venv/bin/activate
```

### Clippy Fails with FFI Errors

Ensure that `CXX` is set to `clang++-22`:
```bash
export CXX=clang++-22
```

This is automatically set in the container, but may need to be set manually if you encounter issues.
