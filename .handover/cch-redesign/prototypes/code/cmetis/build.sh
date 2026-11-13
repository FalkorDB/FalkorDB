#!/bin/bash
# Build the C METIS driver `nd` (nd.c + METIS + GKlib, statically, one compiler call).
# RECONSTRUCTED: the command that built the original binary was not saved. This one was checked on
# 2026-11-13: on PA it writes a permutation byte-identical to the original binary's (see README).
#   METIS=/path/to/KarypisLab/METIS GKLIB=/path/to/KarypisLab/GKlib ./build.sh   -> ./nd
set -euo pipefail
METIS=${METIS:?set METIS to a KarypisLab/METIS checkout at 272d4a91c5f66c92327493339a476c553ebf1f5d}
GKLIB=${GKLIB:?set GKLIB to a KarypisLab/GKlib checkout at 3b7d61b9f885063c89901f3901fb4426f9cfb58f}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/nd}
# idx_t must be 64-bit: nd.c reads int64 n, nnz, xadj and adjncy straight into idx_t arrays.
"${CC:-cc}" -O3 -DNDEBUG -DNDEBUG2 -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32 -D_FILE_OFFSET_BITS=64 \
  -I"$GKLIB/include" -I"$METIS/include" -I"$METIS/libmetis" \
  -o "$OUT" "$HERE/nd.c" "$METIS"/libmetis/*.c "$GKLIB"/src/*.c -lm
echo "built $OUT"
