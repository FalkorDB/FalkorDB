# cmetis: the C METIS `METIS_NodeND` reference driver

**Used by:**
- `../../m1-optimize.md`: the C METIS column, and the verifier's three CA runs;
- `../../m1-matching-quality.md`: the C METIS arcs and triangles;
- `../m1-profile` and `../m1-audit`: their `cmetis.log` files.

**Came from:** `cmetis-driver/nd.c` (the driver), `cmetis-src/` (METIS and GKlib sources) and `m1audit/to_nd_bin.py` (the converter).

## Files

| File | What it is |
|---|---|
| `nd.c` | The driver, unchanged. It reads `n, nnz, xadj[n+1], adjncy[nnz]` as int64 little-endian, runs `METIS_NodeND` with `METIS_SetDefaultOptions` (as the C engine's `CCH_EliminationOrder` does), prints the time and peak RSS, and writes `perm[n]` as int64. |
| `build.sh` | **Reconstructed** build command: one compiler call over `nd.c`, all of `libmetis/*.c` and all of GKlib's `src/*.c`, with `-O3 -DNDEBUG -DNDEBUG2 -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32`. The original command was not saved. |
| `to_nd_bin.py` | SNAP text → driver input. It applies the GraphBLAS benches' loader rules: symmetric, self-loops dropped, duplicates removed, isolated nodes removed, ids renumbered in order. |

## Sources

- **METIS:** `https://github.com/KarypisLab/METIS` at `272d4a91c5f66c92327493339a476c553ebf1f5d`.
- **GKlib:** `https://github.com/KarypisLab/GKlib` at `3b7d61b9f885063c89901f3901fb4426f9cfb58f`.

These are the commits the C engine pins as submodules on `master` (`deps/KarypisLab/METIS`, `deps/KarypisLab/GKlib`). On 2026-11-13, the scratch copies' `libmetis/`, `include/` and GKlib's `src/` and `include/` were byte-identical to such a submodule checkout.

The original binary's debug strings show it was compiled from `../cmetis-src/METIS-272d4a9…/libmetis/*.c` and `../cmetis-src/GKlib-3b7d61b…/src/*.c`.

## Rebuild and run

```bash
git clone https://github.com/KarypisLab/METIS && git -C METIS checkout 272d4a91c5f66c92327493339a476c553ebf1f5d
git clone https://github.com/KarypisLab/GKlib && git -C GKlib checkout 3b7d61b9f885063c89901f3901fb4426f9cfb58f
METIS=$PWD/METIS GKLIB=$PWD/GKlib ./build.sh                 # writes ./nd; CC overrides the compiler

gunzip -k roadNet-PA.txt.gz
python3 to_nd_bin.py roadNet-PA.txt PA.bin                   # prints "PA.bin n 1088092 nnz 3083796"
/usr/bin/time -l ./nd PA.bin perm_PA.bin                     # as the sessions ran it
```

`m1bench export` (`../m1-matching`) and m1-profile's `M1_DUMP_ND` write the same file. All three were compared byte for byte on 2026-11-13:
- `export` against `to_nd_bin.py` on PA;
- `M1_DUMP_ND` against `to_nd_bin.py` on CA, PA and TX.

**Checked on 2026-11-13:** `build.sh` with Apple clang 17.0.0 built `nd` in about 3 s. On PA, that binary, the original binary, and the permutation saved in the scratch caches all gave the same bytes (sha256 below). It took 2.6 s against 2.5–2.7 s for the original.

## Checksums

The driver inputs from `to_nd_bin.py`, and the permutations the original binary wrote (the m1-optimize session's `cache/cmetis_perm_<G>.bin`, identical to m1audit's `perm_<G>.bin`):

```
8ac8f55d769024e5d86c1f48c46d4cca160bd2df0120d996baaa7a6178da93a4  CA.bin   (n 1965206, nnz 5533214)
17abff02855315f30840712196d9cba2af6756df5540075d49a668b544a0cc93  PA.bin   (n 1088092, nnz 3083796)
1b85d97248f5e18acfb68736dc0610f402b0ecb1b374e1a4abf9a7cc52c35d0c  TX.bin   (n 1379917, nnz 3843320)
cc87e63a16ef061cd786a91e276583e5f7737ef338006d0ed26677f4a04a2ac8  perm_CA.bin
4c18d556d5c08d509073556b2e6ff7304f97189d6b7256a2ce2f6b55a2c41d9e  perm_PA.bin
928b353cee88b31f74a1177eb9bfbed6863cde2cddfa4c3e84eeea35bc5d3b12  perm_TX.bin
```

## Expected output

`METIS_NodeND status 1, <s>s, maxrss before … MB after … MB`. Status 1 is `METIS_OK`. On the M5:
- CA: 5.05–5.08 s, 32.5 G instructions (from `/usr/bin/time -l`), peak RSS about 551 MB;
- PA: about 2.7 s.

Logs: `../m1-optimize/results/raw-logs.tar.xz`, `session/cmetis.log`.

## Left out

- **The original binary** `cmetis-driver/nd`.
- **The METIS and GKlib source trees and tarballs:** 18 MB, available at the commits above.
- **`cmetis-driver/ca.bin` and `perm.bin`:** an earlier input from 2026-11-10, with n = 1,971,281, so it still held the 6,075 isolated nodes. None of these prototypes used it.
- **The converted `.bin` inputs and the permutations:** these are data; regenerate them and compare with the checksums above.
