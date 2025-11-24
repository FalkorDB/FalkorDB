# FalkorDB Copilot Instructions

## Project Overview

FalkorDB is a high-performance graph database designed as a Redis module, optimized for GraphRAG & GenAI applications. It's the first queryable Property Graph database to leverage sparse matrices for representing adjacency matrices and linear algebra for querying. The database supports the OpenCypher query language with proprietary extensions.

**Key Technologies:**
- **Primary Language:** C (Redis module)
- **Secondary Language:** Rust (performance-critical components in deps/FalkorDB-core-rs)
- **Build System:** Make + CMake
- **Target Runtime:** Redis 7.4+ as a loadable module
- **Query Language:** OpenCypher with extensions
- **Test Framework:** Python-based using RLTest, unit tests in C

**Repository Size:** Large (~930MB with dependencies and build artifacts)
**Dependencies:** 12 major submodules including GraphBLAS, LAGraph, RediSearch, libcurl, etc.

## Critical Build Requirements

### Prerequisites (ALWAYS install these first)
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake m4 automake peg libtool libtool-bin autoconf python3 python3-pip

# macOS (requires Homebrew)
brew install cmake m4 automake peg libtool autoconf
# Note: macOS requires GCC for OpenMP support: brew install gcc g++
```

### Submodule Initialization (REQUIRED)
**ALWAYS run this before any build:**
```bash
git submodule update --init --recursive
```
This step is critical - the build will fail without properly initialized submodules.

### Build Commands (Tested Order)

1. **Dependencies Build (takes 10-15 minutes):**
   ```bash
   make deps
   ```
   - Builds all external dependencies including GraphBLAS, LAGraph, RediSearch
   - Creates artifacts in `bin/linux-x64-release/` (or platform variant)
   - **Expected time:** 10-15 minutes on modern hardware

2. **Main Build:**
   ```bash
   make
   ```
   - Produces: `bin/linux-x64-release/src/falkordb.so`
   - **Expected time:** 2-3 minutes

3. **Alternative build with specific options:**
   ```bash
   make DEBUG=1        # Debug build
   make STATIC_OMP=1   # Static OpenMP linking
   make COV=1          # Coverage analysis build
   ```

### Build Troubleshooting

**Common Issues and Solutions:**

1. **Missing tools error:** Install all prerequisites above
2. **Submodule errors:** Run `git submodule update --init --recursive`
3. **OpenMP errors on macOS:** Install GCC via Homebrew
4. **Memory issues:** Large build may require 4GB+ RAM
5. **Permission errors:** Check file permissions on deps/ directory

## Testing Procedures

### Test Dependencies
```bash
cd tests
pip install -r requirements.txt
```
**Key Python packages:** FalkorDB client, behave, docker, RLTest

### Test Categories

1. **Unit Tests (C/Rust):**
   ```bash
   make unit-tests
   ```
   - Tests C components and Rust library
   - No Redis server needed
   - **Runtime:** 1-2 minutes

2. **Flow Tests (Python + Redis):**
   ```bash
   make flow-tests
   ```
   - Requires Redis server
   - Tests full database functionality
   - **Runtime:** 5-10 minutes

3. **TCK Tests (OpenCypher compliance):**
   ```bash
   make tck-tests
   ```
   - Tests OpenCypher compatibility
   - **Runtime:** 5-10 minutes

4. **All Tests:**
   ```bash
   make test
   ```
   - Runs unit-tests, flow-tests, tck-tests, upgrade-tests
   - **Runtime:** 15-20 minutes

### Test Configuration Options
```bash
TEST=specific_test    # Run specific test
PARALLEL=4           # Set parallelism level  
V=1                  # Verbose output
REDIS_SERVER=/path   # Specify Redis location
```

## Project Structure

### Root Directory Files
- `Makefile` - Main build system entry point
- `CMakeLists.txt` - CMake configuration
- `Cargo.toml` - Rust workspace configuration
- `README.md` - Project documentation
- `CONTRIBUTING.md` - Contribution guidelines

### Key Directories

**Source Code:**
- `src/` - Main C source code (45+ subdirectories)
  - `src/module.c` - Redis module entry point
  - `src/algorithms/` - Graph algorithms
  - `src/execution_plan/` - Query execution
  - `src/arithmetic/` - Mathematical operations
  - `src/graph/` - Core graph data structures

**Dependencies:**
- `deps/` - Git submodules (12 external dependencies)
  - `deps/GraphBLAS/` - Sparse matrix operations
  - `deps/LAGraph/` - Graph algorithms library
  - `deps/RediSearch/` - Search functionality
  - `deps/FalkorDB-core-rs/` - Rust performance components
  - `deps/libcypher-parser/` - Cypher query parsing

**Testing:**
- `tests/` - Test suites
  - `tests/flow/` - Integration tests (Python)
  - `tests/unit/` - Unit tests (C)
  - `tests/tck/` - OpenCypher compatibility tests
  - `tests/benchmarks/` - Performance tests

**Build System:**
- `build/` - Build configurations for dependencies
- `bin/` - Build output directory (created during build)
- `sbin/` - Utility scripts

### Configuration Files
- `.github/workflows/build.yml` - CI/CD pipeline
- `.codecov.yml` - Code coverage configuration
- `.gitmodules` - Submodule definitions
- `.spellcheck.yml` - Spell checking configuration

## CI/CD Pipeline

The project uses GitHub Actions with a complex multi-platform build:

**Build Matrix:**
- Platforms: linux/amd64, linux/arm64, RHEL
- Variants: Ubuntu, Alpine, debug builds
- **Total build time:** 30-45 minutes for full matrix

**Build Stages:**
1. Dependency caching and building
2. Compiler image creation 
3. Multi-architecture builds
4. Testing (unit, flow, TCK, fuzz)
5. Security scanning (Trivy)

## Redis Integration

### Loading the Module
```bash
# Method 1: redis.conf
loadmodule /path/to/falkordb.so

# Method 2: Command line
redis-server --loadmodule /path/to/falkordb.so

# Method 3: Runtime (not recommended for production)
MODULE LOAD /path/to/falkordb.so
```

### Basic Usage
```bash
redis-cli
127.0.0.1:6379> GRAPH.QUERY social "CREATE (:person {name: 'Alice'})"
```

## Development Guidelines

### Code Changes
- **Primary development:** C files in `src/`
- **Performance components:** Rust in `deps/FalkorDB-core-rs/`
- **Always build and test:** Changes affect module loading

### Testing Requirements
- Unit tests for C components
- Flow tests for integration
- **Always test module loading** in Redis after changes

### Common Validation Steps
1. `make deps && make` - Full clean build
2. `make test` - Complete test suite
3. Manual Redis module loading test
4. Performance regression check if modifying algorithms

## Docker Usage

**Quick Start:**
```bash
# Standard container
docker run -p 6379:6379 -it --rm falkordb/falkordb:edge

# With browser interface
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:edge
```

## Performance Notes

- **Memory usage:** 4GB+ RAM recommended for building
- **Build time:** 15-20 minutes for full build from scratch
- **Test time:** 15-20 minutes for full test suite
- **Dependencies are cached** in CI to reduce build times
- **GraphBLAS compilation** is the longest single step (~5-8 minutes)

## Trust These Instructions

These instructions are comprehensive and tested. Only explore beyond them if:
1. Specific error messages indicate missing information
2. New build configurations not covered here
3. Platform-specific issues not documented

Always start with the basic requirements and build process before attempting customizations.

## Quick Search Patterns

For efficient code navigation and debugging:

**Find function definitions:**
```bash
grep -r "function_name" src/
grep -r "typedef.*function_name" src/
```

**Find test cases:**
```bash
find tests/ -name "*.py" -exec grep -l "test_name" {} \;
```

**Find configuration options:**
```bash
grep -r "CONFIG\|OPTION" src/ | head -10
```

**Find OpenCypher keywords:**
```bash
grep -r "MATCH\|CREATE\|RETURN" src/
```

**Debug build issues:**
```bash
make clean && make V=1  # Verbose build
make VERBOSE=1          # CMake verbose mode
```

## Module Loading Verification

After building, verify the module loads correctly:
```bash
# Start Redis with module
redis-server --loadmodule bin/linux-x64-release/src/falkordb.so

# In another terminal, test basic functionality
redis-cli GRAPH.QUERY test "CREATE (:Node {prop: 'value'})"
```