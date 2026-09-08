# Vendored zstd

**Version 1.5.7**, matching what the Rust engine bundles (`zstd-sys 2.0.16+zstd.1.5.7`).

Matching is a convenience, not a requirement, and it is worth being precise
about what it does and does not buy. zstd frames are standardised, so
interoperability never depended on the version. Nor does it depend on the API:
C compresses with one-shot `ZSTD_compress` while Rust goes through
`zstd::stream::encode_all`, and those can frame the same bytes differently —
measured on a 4,000 byte repetitive payload they happen to agree exactly, 24
bytes each with the content size recorded, but that is a property of feeding
the whole input in one call, not a guarantee.

So **do not rely on a compressed payload being byte-identical across the two
engines.** The format does not: the checksum is taken over the plaintext rather
than the frame precisely so it stays reproducible when two encoders frame the
same input differently. And the decoder here never reads the frame's declared
content size — the ceiling comes from the header's `uncompressed_length`, passed
as zstd's destination capacity — so a peer on a different version or a
different API is decodable either way. Matching 1.5.7 means only that a
byte-level diff is *likely* to line up when someone is debugging one.

| | |
| --- | --- |
| upstream | `https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz` |
| sha256 | `eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3` |
| regenerate | `./regenerate.sh /path/to/zstd-1.5.7` |

## Why a second copy of zstd, when GraphBLAS already links one

GraphBLAS vendors zstd 1.5.5 at `deps/GraphBLAS/zstd/zstd_subset` and exports
`GB_ZSTD_compress`, `GB_ZSTD_decompress`, `GB_ZSTD_compressBound` and
`GB_ZSTD_getFrameContentSize` as defined global symbols. Calling those instead
of vendoring is therefore *available*. It is still wrong, and the reason is not
the symbol collision described in `zstd_symbols.h` — that one is a build
problem, and a build problem is always one flag or one rename away from going
quiet.

The reason is the allocator. `deps/GraphBLAS/Source/zstd_wrapper/GB_zstd.c:18-21`
defines zstd's allocation hook as

```c
void *ZSTD_malloc (size_t s)
{
    return (GB_Global_malloc_function (s, data_arena)) ;
}
```

So every working buffer of every effects compression would be allocated through
GraphBLAS's allocator and into its data arena. That puts replication traffic
inside GraphBLAS's memory accounting, permanently, and it survives any fix to
the symbol problem. It is a correctness and observability defect rather than an
inconvenience.

Two secondary reasons, for completeness: `GBZSTD(x)` expands to `GB_` or `GM_`
depending on whether `GBMATLAB` is defined (`GB_zstd.h:23-25`), so the symbol
names are build-conditional; and their copy is a *subset* pinned to a version
we do not control, on the path that carries our wire format.

## What differs from a stock amalgamation

`regenerate.sh` starts from upstream's own `build/single_file_libs/zstd-in.c`
and removes two things we never call. Both are supported upstream
configurations, not patches.

| removed | why |
| --- | --- |
| `ZSTD_MULTITHREAD` | a worker thread pool inside a Redis module, for a synchronous level-1 compress on the write path. zstd only spawns workers when `nbWorkers > 0`, so this is dead code rather than a behaviour change, and single-threaded is upstream's default. |
| `dictBuilder/*` | we never train dictionaries; `divsufsort.c` alone is large. |

Measured, compiled `-O2`:

| | object | global symbols |
| --- | --- | --- |
| stock amalgamation | 595,608 B | 371 |
| ours | 493,800 B | 333 |

Two edits are then applied to the generated file, and only two — both are
described at the point they happen in `regenerate.sh`, and both are verified by
the check in its last step:

1. `#include "zstd_symbols.h"` at the top, which renames every global to `FDB_*`.
2. the rename is re-applied after any `#undef` of a renamed symbol. This exists
   because `zstd_common.c` does `#undef ZSTD_isError` before defining the real
   exported function, which removes our macro along with zstd's own — see
   `zstd_symbols.h` for why that is the one symbol where it matters.

## Known build warnings

Three, all from the vendored file, all benign:

```
zstd.c:3656  warning: 'FSE_isError' macro redefined
zstd.c:15542 warning: 'ZSTD_isError' macro redefined
zstd.c:15544 warning: 'HUF_isError' macro redefined
```

`zstd_internal.h` macros these three to `ERR_isError` "for inlining", which
redefines the rename macro for call sites below that point. The call sites are
correct either way — `ERR_isError` is itself renamed — and the real function
definitions are unaffected. The build sets no `-Werror`, so these are noise
rather than a break. If they ever need silencing, the narrow fix is one line:

```cmake
set_source_files_properties(src/util/zstd/zstd.c PROPERTIES COMPILE_OPTIONS -Wno-macro-redefined)
```

That is deliberately not done here, because a warning that is understood and
documented is worth more than a suppressed one, and GCC does not honour the
same flag.

## Build integration

None needed. `CMakeLists.txt:256` is `file(GLOB_RECURSE SOURCES "src/*.c")`, so
`zstd.c` compiles automatically — the same way `src/util/roaring.c`, an 816 KB
CRoaring amalgamation, already does. Note the consequence: **no `.c` file may
be added under `src/` that is not meant to be compiled on its own.** That is
why the amalgamation template lives inside `regenerate.sh` rather than as a
checked-in `.c`.
