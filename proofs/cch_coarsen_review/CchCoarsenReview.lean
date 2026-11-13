import CchCoarsenReview.Sum
import CchCoarsenReview.Spec
import CchCoarsenReview.Rows
import CchCoarsenReview.Emit
import CchCoarsenReview.Hcon
import CchCoarsenReview.Cpack
import CchCoarsenReview.CpackGB
import CchCoarsenReview.Hsplit
import CchCoarsenReview.Hkey
import CchCoarsenReview.Matching
import CchCoarsenReview.Overflow
import CchCoarsenReview.Gath
import CchCoarsenReview.Hb2
import CchCoarsenReview.Book
import CchCoarsenReview.Hb2Shape

/-!
# CCH coarsening review: the optimized host passes compute the GraphBLAS formulas they replaced

**What is reviewed.** The ordering prototype's optimization pass, `m1-optimize`
(`.handover/cch-redesign/prototypes/m1-optimize.md`, read with its corrections note). Its code is not in
the repo: it is the patch `.handover/cch-redesign/prototypes/code/m1-optimize/m1opt.patch`, applied to
base `7df8e588727f47619d9567d111c7fc26a2208caf`, which adds `graph/src/graph/graphblas/m1_coarsen_bench.rs`
and `m1_reuse_bench.rs`. Every cite below is a line of those files with the patch applied
(`m1_coarsen_bench.rs:<line> (m1opt.patch @7df8e5887)`; `:<line>` alone means the file the section names).
The design it serves is `.handover/cch-redesign/design/03-design.md` revision 3.1, §3 (coarsening by
matching rounds, contraction `C·A·Cᵀ`, the depth split, reuse of the parent's levels).

Most switches in `M1O` replace GraphBLAS calls with one Rust pass over GraphBLAS's unloaded CSR arrays
(`hcon`, `cpack`, `pk`, `hsplit`, `hkey`, `hb2`, `gath`, `book`); `usel` and `zmatch` are GraphBLAS
rewrites of the matching rounds; `hpar` and `npar` split host passes across threads. The bench checked
each switch's output equal to the switch-off path only by hashes on three road graphs. This project
proves each of `hcon`, `cpack`, `pk`, `hsplit`, `hkey`, `hb2`, `gath`, `usel`, `zmatch` and `hpar`
(its chunk assembly) equal to what it replaced, for every input that meets the stated preconditions.
`book` is covered only at the level of the numbering's decisions (gap G1), and `npar` is not modelled.

The `cpack` comparison is against the `M1_CT=1` form of the switch-off path (`Cᵀ` built directly, the
`ct()` branches), which every reported run used and the reuse bench asserts
(`assert!(ct(), …)`, m1_reuse_bench.rs:160).

Because the code is a patch, coverage.sh cannot match rows against this checkout: `COVERAGE.tsv` is
header-only (as `cch_redesign_review`'s), and the Rust ↔ Lean table is here.

## The model

* **GraphBLAS objects** (`Spec.lean`): a matrix is the function from a coordinate to its stored value
  (`Mat`), as in `graphblas_wrappers/DeltaLmxm.lean`. Every call the switches replaced is a *definition*
  with the meaning the C API gives it: `mxm` over a `PLUS` monoid with inner dimension `k` (the sum over
  the stored intersection, no entry when it is empty), `tr` (`GrB_DESC_T0/T1`), `offdiag`, `selLE`,
  `ewMult`, `ewAdd`, `applyS` (structural mask), `buildS` (build without duplicates). The `ANY_PAIR`
  pattern is imported from graphblas_wrappers: `GBWDelta.mxmAny` and `GBWDelta.cnt_pos_iff`. No axiom.
* **CSR** (`Rows.lean`, `Emit.lean`): a CSR is its rows, `Rows α := Nat → List (Nat × α)`, row `i` being
  the slice `aj[ap[i]..ap[i+1]]` that every pass reads (`Csr::new`, m1_coarsen_bench.rs:363-424) with
  the values beside it; `den` is the matrix it denotes. `Emit.lean` proves the flat arrays every pass
  writes (`emit_csr_t`, :1277-1349) hold exactly these rows, with one chunk or with `hpar`'s chunks.
* **Values** are `Int`; `Overflow.lean` shows the bench's values stay inside `i32`/`u32`, where `Int`
  and the machine types agree (the release profile has no overflow checks, so this matters).
* **`sort_unstable_by_key`** is any function meeting Rust std's contract (`SortSpec`: a permutation,
  keys ascending, equal keys in any order); `sortSpec_mergeSort` shows the contract is satisfiable.
* **`fmix32`/`unfmix32`** are modelled exactly on `BitVec 32`.

## Rust ↔ Lean

| switch | host pass (m1opt.patch @7df8e5887) | replaced GraphBLAS path | Lean |
|---|---|---|---|
| `hcon` | `host_contract`, m1_coarsen_bench.rs:1392-1437 | `CA`, `CAC'`, `offdiag`: :1226-1247 (non-`cpack`: :1084-1104) | `hconTmp`, `hconRows`, `grp` (Hcon, Rows) |
| `cpack` | `contract_packed` `c_host`, m1_coarsen_bench.rs:1174-1223 | `c_leader`, `c_number_host`, `c_extract`, `c_build`, `Cw`: :1031-1112 | `cpStep`, `cpack`, `leadGB`, `GBContract`, `ctTuplesGB` (Cpack, CpackGB) |
| `pk` | `pack_ct`, m1_reuse_bench.rs:808-837; weights :1382-1383 | `GxB_Matrix_build_Scalar(ct_i, ct_j)` :946-949; `vec_i32` :1385-1386 | `packCt`, `ctTuples` (CpackGB) |
| `hsplit` | `split2`, m1_reuse_bench.rs:1865-1888 | `split_buildM`, `split_mxm`: :1889-1909 | `splitRows`, `splitM`, `splitGB` (Hsplit) |
| `hkey` | `host_keys`, m1_coarsen_bench.rs:1354-1388 | cap mask :792-814, `apply_IndexOp(key)` :816-828 | `hostKeyRows`, `capMask`, `keysGB`, `keyOf`, `fmix32` (Hkey) |
| `hb2` | `b2_splice`, m1_reuse_bench.rs:843-929 | `Mode::B2`: `m_build` … `union`, :976-1032 | `contrTmp`, `prow`, `tcolRow`, `b2Rows`, `zB2`, `xB2`, `RGB`, `b2GB` (Hb2) |
| `gath` | `build_capstop`, m1_reuse_bench.rs:1327-1361 | `GrB_Matrix_extract` + shift + build: :1362-1381 | `rk`, `gathRow`, `gathRows`, `gathGB` (Gath) |
| `book` | `number_fast`, m1_reuse_bench.rs:385-528; `number_patch` :736-797; call :1280-1284 | renumber with the smaller active set :1286 | `touched`, `membersOf`, `clean`, `isPart`, `drop` (Book) |
| `usel` | `select(m == false)`, m1_coarsen_bench.rs:915-921 | `apply(identity, y)⟨¬m, RC⟩` :922-929 | `usel_same_pattern` (Matching) |
| `zmatch` | `Z` rounds :873-890, `z_final` :938-959 | `pf` rounds :891-908, `mate⟨m⟩ = y` :909-914 | `stepZ`, `iterZ`, `finalZ` vs `stepB`, `iterB` (Matching) |
| (rounds) | `mxv`, `unkey`, mutual test, :851-937 | — | `PropSpec`, `MxvSpec`, `pf`, `mutB` (Matching) |
| (2-hop) | :963-1021 | — | `TwoHopGB`, `pairsOf`, `capPairs`, `twohop` (Matching) |
| `hpar` | `emit_csr_t` chunks, m1_coarsen_bench.rs:1303-1348 | one chunk :1292-1302 | `emitOffsets`, `chunkOffsets` (Emit) |
| `npar` | `number_par`, m1_reuse_bench.rs:533-732; `pack_ct`'s threaded branch :813-821 | `number_fast` (serial) | **not modelled** (only its H2 dependency, by reading the code) |

## Theorems (no sorry, no axiom)

* **hcon** — `hcon_eq`: for every coarse row `r` and column `c`, `host_contract`'s row stores at `c`
  exactly `select(OFFDIAG, PLUS_TIMES(PLUS_TIMES(C, A), Cᵀ))(r, c)`, for any order the unstable sort
  leaves equal keys in (:1392-1437). `hcon_row_shape`: rows strictly ascending (sorted, no duplicates),
  no diagonal. `hcon_cols`: every column is some `ca(j)`. `hcon_symm`: `A_c` is symmetric in pattern
  when `A` is (with `cCpack_mem_iff`). `hcon_dead_column_differs` (the precondition is needed; finding
  H1).
* **cpack** — `cpack_inv`: the pass's state in closed form (`nc` = number of leaders; `C`'s rows = the
  leaders' member lists; `Cᵀ` row `i` = `[rank(min(i, mate i))]` for live `i`; `w_c` = the leaders'
  `wsum`), :1189-1214. `cpack_rows_iff` (**the numbering is a bijection from the matched groups onto
  `0..nc`**: row `c < nc` is non-empty and holds exactly the live vertices with id `c`),
  `cpack_ca_lt`, `cpack_ct_rows` (**`Cᵀ` has exactly one entry per live row, none per dead row**),
  `cpack_wch` and `cpack_wc_eq_mxv` (**`w_c = Cᵀᵀ·w`**), `cpack_conserves` (**total weight
  conserved**). Against the switch-off path: `li_eq_leaders`, `cid_eq_ca`, `cpack_ct_eq_gb` (**`cpack`'s
  `Cᵀ` is the one GraphBLAS built**), `cpack_c_is_ct_transposed` (`c_rows = Cᵀᵀ`, so `c_rows·A·ctp` is
  the switch-off `Cᵀᵀ·A·Cᵀ`), `cCpack_nodup_lt` (`cpack`'s `C` meets `hcon_eq`'s premises),
  `cCpack_mem_iff` (`C`'s rows are the classes of `ca`), `cCpack_sorted` (`C`'s rows `[ℓ]` or `[ℓ, m]`,
  `ℓ < m`).
* **pk** — `pk_eq_build`: `pack_ct`'s rows equal `GxB_Matrix_build_Scalar(ct_i, ct_j, 1)` on every row
  (at most one entry, exactly one when `cmap[x] ≠ NONE`); `pk_weights`: `pack_full_i32(gw)` equals
  `vec_i32(0..off, gw)`.
* **hsplit** — `hsplit_eq`: the filter pass equals `A_0⟨struct⟩ = M ANY_PAIR Mᵀ` (1 exactly where `A_0`
  stores and both ends carry the same non-separator label). `hsplit_shape`: rows sorted, separator rows
  empty, no separator column. `hsplit_cols_live`: every column is live under R's `ones_vec` weights
  (discharges `hcon_eq`'s `colLive` at R's first level). `hsplit_symm`: symmetric in, symmetric out.
* **hkey** — `hkey_eq`: for **any** key function, the host pass equals `apply(key, A)` under the cap
  mask's structure, or unmasked when not capped. `capMask_spec`: the mask keeps `A`'s entries with
  `w(i) + w(j) ≤ cap`. `hkey_uncapped_vacuous`: when `2·maxw ≤ cap` the test drops nothing.
  `unfmix_fmix` (`unfmix32 ∘ fmix32 = id` on all of `u32`), `unkey_key` (the key decodes to its column),
  `key_col_injective` (keys are unique in a row, so `MAX` is deterministic), `key_high`.
  `hkey_row_shape`: rows strictly ascending (the keys are packed with `jumbled = false`, :306).
* **Matching** — `prop_of_mxv` (the `MAX_FIRST` `mxv` contract plus `unkey_key` give `PropSpec`),
  `iterB_inv` (after any number of rounds `mate` is an involution, every matched pair is a `K` edge,
  proposers are unmatched), `usel_same_pattern`, `zmatch_eq` (**`zmatch` = the baseline**: same proposer
  sets every round, so the same `y`, counts and exit; `z_final` gives the baseline `mate`), `twohop_inv`
  (the 2-hop pass keeps the involution and the cap; its pairs share an anchor), `keys_K` (`hkey`'s
  pattern is symmetric, has no diagonal, lies in `A` and within the cap when capped), `matching_final`
  (end to end: involution, every pair within the cap, a round pair is an edge of `A`, a 2-hop pair has
  a common neighbour), `matePre_of` and `mate_pre_final` (**`MatePre` discharged**: `keys_K` +
  `matching_final` + `matePre_of` on an `A` whose entries join live vertices below `n`).
* **hb2** — `hb2_eq`: `b2_splice` equals `eWiseAdd(FIRST, x, eWiseAdd(FIRST, R, Rᵀ))` at every coordinate,
  where `x` is the parent level kept between clean ids of one cell and `R = select(OFFDIAG, csᵀ·p.a·Cᵀ)`
  is the straddler rows. Built from `contr_eq` (the guarded contraction, needing no premise on
  columns), `Rm_low` (no `R` row below `np`), `Rm_symm` (`R` symmetric in pattern on part rows),
  `tcolRow_lookup`, `xrow_eq`. Shape (Hb2Shape.lean): `hb2_row_shape` (rows strictly ascending: clean
  entries below `np`, `tcol` parts at or above `np` and ascending, part rows merged; the level is packed
  with `jumbled = false`), `hb2_symm` (symmetric in pattern when the parent's level is), `hb2_no_diag`,
  `hb2_vals` and `prow_bounds` (each value is a parent value, or a part-row sum in `[0, total(p.a)]`).
* **gath** — `gath_eq`: one block's appended rows equal `extract(A_s, ids, ids)` shifted by `off`;
  `gath_row_strict`: rows sorted; `gath_vals`: every gathered value is a value of its block's level, and
  a gathered row is no longer than its source row. Blocks occupy disjoint row and column ranges, so the
  gather is the blocks side by side.
* **book** — `members_restrict`, `clean_restrict`, `part_restrict`, `hole_restrict`: dropping the
  stopped cells changes the numbering's decisions exactly as `number_patch` assumes (finding H3 for the
  premise; the ordering argument is a reading, R7). `book_needs_pk`, `book_needs_hb2`: illustrations
  of finding H2, true by construction, not models of `build_level`.
* **hpar** — `emit_row`: the flat arrays hold the rows; `chunks_eq_serial`, `chunks_cols`: the chunked
  assembly writes the serial arrays.
* **Overflow** — `bench_sizes_fit`, `grp_total`, `grp_val_le`, `hcon_row_bounds`, `hcon_total_le` (a
  level's value total never exceeds the previous level's), `cpack_wch_bounds` (`0 < w_c ≤` the conserved
  total), `grp_none_length_le`, `ids_bound` (B2 id spaces at depth `d` at most `(d+1)·n`, given the cited recurrence and `nparts ≤ n`,
  gap G2).

## Preconditions and assumed contracts

* **C API contracts used as definitions** (Spec.lean, GraphBLAS C API 2.1 and the SuiteSparse 10 user
  guide): `GrB_mxm` over `PLUS_TIMES/PLUS_FIRST/PLUS_SECOND/ANY_PAIR`; descriptors `T0`, `T1`, `S`,
  `RS`, `RC`, `ST1`; `GrB_select` with `OFFDIAG`, `VALUELE`, `VALUEEQ`; `eWiseMult`/`eWiseAdd` with
  `FIRST`, `SECOND`, `PLUS`, `EQ`, `MIN`; `GrB_apply` with an index-unary op under a structural mask;
  build without duplicates (every build here has none); `GrB_Matrix_diag`; `GrB_Matrix_extract`.
* **Contracts stated as structures** (each cites the call): `GBContract` (`extractTuples` lists a
  non-jumbled vector's indices ascending; `build` stores `cid(li[k]) = k`), `MxvSpec` (`MAX_FIRST` `mxv`
  under a structural mask: stored iff some product is, the value one of them), `TwoHopGB` (the 2-hop
  groups: rows of `g` built without duplicates and sorted by `GxB_Matrix_sort`, members unmatched and
  adjacent to their anchor). `SortSpec` is Rust std's `sort_unstable_by_key`.
* **`hcon`** (`HconPre`): `A` the unloaded CSR (rows strictly ascending — the `GrB_wait` in `Csr::new`),
  columns below `n`, `C` rows duplicate-free and below `n`, **every column of `A` live**. Discharged:
  `cCpack_nodup_lt` (the `C` part); `hsplit_cols_live` (live columns at R's first level of a depth);
  `cpack_wch_bounds` (every coarse vertex has positive weight, so every column is live above level 0).
  The depth-0 graph has no dead vertex. Capstop's gathered first level: see below.
* **`cpack`** (`MatePre`): `mate` an involution on the live vertices, the mate of a live vertex live and
  below `n`. Discharged by `mate_pre_final` (`keys_K` + `matching_final` + `matePre_of`), given `A`'s
  entries join live vertices below `n` (`KeyPre`) and the round and 2-hop contracts.
* **Matching** (`KSpec`): `K` symmetric with no diagonal. Discharged by `keys_K` from `A` symmetric with
  no diagonal: the loader makes `A_0` so (`load`, m1_coarsen_bench.rs:1468-1506), `hsplit_symm` keeps
  it, `hcon_symm` and `hb2_symm` keep the pattern symmetric level to level, and `hcon_row_shape` and
  `hb2_no_diag` give no diagonal. Symmetry of the *values* is not proved; nothing here needs it (G3).
* **`hkey`** (`KeyPre`): `A`'s entries join live vertices (the cap mask is built from `D = diag(w)`, so
  GraphBLAS drops an entry at a dead end; the host's dense `wh` is 0 there and would keep it).
* **`hb2`** (`B2Pre`): the parent's level and `p.a` strictly ascending (`parStrict`, `pStrict`:
  discharged by `hcon_row_shape` and `hb2_row_shape` for the levels this bench builds, and
  `hsplit_shape` for the split graph); `p.a` symmetric in pattern (`pSymm`: `hsplit_symm` at level 1 of a
  depth, then `hb2_symm` level to level, from the parent's levels' symmetry: `hcon_symm` at depth 0,
  `hb2_symm` below); part `s`'s members are exactly the previous-level vertices
  `cmap` sends to `s` (`np ≤ s < n`); `cmap` maps into `0..n`, `n ≤ NONE`; clean cells `< ncells`. The
  last three are what `number`/`number_fast` build (:284-350, :431-494); they are premises here, not
  proved from the numbering's loop (gap G2).
* **Capstop's gathered first level** (`hierarchy_keep(&g, Some((wg, gw)))`, m1_reuse_bench.rs:1396, with
  a *full* `wg`, :1382-1387). Its premises are named, not discharged by a theorem: every `gw[k] > 0`
  (each gathered vertex is a live vertex of a stopped cell, weight ≥ 1; checked at run time by
  `assert_eq!(te, nl, "cpack: live count")`, m1_coarsen_bench.rs:1216, since GraphBLAS's `w` is full);
  given that, `HconPre.colLive` and `KeyPre.liveEnds` hold trivially (every vertex is live). `KSpec`
  there needs `g` symmetric with no diagonal: by `gath_eq`, `g` is its blocks side by side, each block a
  principal submatrix (`ids × ids`) of a capstop B2 level (`hb2_symm`, `hb2_no_diag`) or of the depth's
  split graph (`hsplit_symm`), so both carry over. (Whether a block is closed, the comment at :1328-1329,
  affects neither `gath_eq` nor symmetry: the switch-off `extract` drops the same edges.)
* **`book`** (`BookPre`): every counted member has positive weight; a parent id's members weigh at most
  the parent's weight. Both hold in the bench (weights are sums of unit fine weights over subsets of
  the parent's fine vertices); not derived here.

## Readings chosen

* **R1.** Values are `Int`; equality there plus the bounds of `Overflow.lean` is equality in `i32`.
* **R2.** The two nested merge loops (`while q < tmp.len() { … while … tmp[q].0 == cc … }`) are one scan
  with the open group as state (`grp`).
* **R3.** The mutual-match rounds are modelled for any number of rounds; the loop's exit (`np == 0` or
  `newly == 0`) reads only `y` and the next proposer set, which `zmatch_eq` shows equal on both paths.
* **R4.** The proposals `y` are abstracted to a function of the proposer pattern (`Prop'`), meeting the
  `mxv` contract; the exact `MAX` choice does not enter any claim here.
* **R5.** The 2-hop pass's GraphBLAS half (sort, even/odd select, range extract, `eWiseMult`,
  `extractTuples`) is abstracted to `TwoHopGB`; the host half (cap filter, both directions, the masked
  assign) is modelled.
* **R6.** `tcol`'s entries for row `i` are read as the parts with an entry at `i`, ascending
  (`tcolRow`): `tcol` is sorted by the unique key `(jj, s)` (:890) and consumed in row order by a
  pointer from `partition_point` (:899, :913-917). Not proved from the pointer walk.
* **R7.** `book`'s renumbering: the surviving parts keep their order because both numberings take the
  `(q, cell)` runs of the same sorted list; proved here for the set of parts (`part_restrict`), not for
  the list order.
* **R8.** Rows beyond a matrix's dimension are empty (`rowsBeyond`); GraphBLAS has no such rows.
* **R9.** The "live" test of the host passes is `wh[i] > 0`; the GraphBLAS paths use `w`'s pattern. The
  model links them by `wvec`/`wv` (an entry exactly where `wh > 0`), which `dense_i32` and `ones_vec`
  produce. For a full `w` (capstop's gathered graph, every coarse level) the link is the run-time check
  `assert_eq!(te, nl, "cpack: live count")` (m1_coarsen_bench.rs:1216): the number of `wh > 0` vertices
  must equal `w.nvals()`.

## Findings

* **H1 (`hcon` trusts its input; known, now proved).** `host_contract` maps every column through `ca`,
  so a column that is not live becomes `NONE = u32::MAX` and is emitted, where `C·A·Cᵀ` drops it
  (`hcon_dead_column_differs`: `A(0,1) = 1` with vertex 1 dead gives a stored `(0, 4294967295)` on the
  host and nothing in GraphBLAS). This is the verifier's `split` probe (m1-optimize.md "Verification").
  Unreachable with a valid split (`hsplit_cols_live`). `b2_splice` guards the same case (`jj != NONE`,
  :868), and `contr_eq` shows the guarded form needs no premise. Module fix: add the guard to `hcon`, or a
  `debug_assert!(cc != NONE)`.
* **H2 (`book` and `npar` require `pk` and `hb2`; confirmed on the prototype for `book`).**
  `number_fast` and `number_par` leave `ct_i/ct_j` and `cl_q/cl_c` empty (m1_reuse_bench.rs:513-516,
  :717-720), fill `ps_i/ps_j` only for stats, and `number_patch` clears them (:796). The switch-off paths
  of `pk` (:946-949) and `hb2` (:979-1010) read them, so with `book` or `npar` and either switch off, the
  B levels come out wrong (illustrated by `book_needs_pk`, `book_needs_hb2`). Rebuilt from `m1opt.patch`
  and run on 2026-11-13 (binary md5 7b2155e335d72a6feac296e35eb0e231) on PA, depths 0–3, 1 thread,
  against `results/hashes/pa8_base.hash`: `M1O=book` and `M1O=hb2,book` panic "CHECK FAILED: projection
  lost vertices" (m1_reuse_bench.rs:1140, exit 101); `M1O=pk,book` fails "CHECK FAILED: B1 != B2" (:2344,
  exit 101) with 10 of 13 hashes differing; the controls `M1O=hb2,pk,book` and the final configuration
  match 13/13. Steps, logs and `run.sh` are in `.handover/cch-redesign/repro/h2-switch-dependency/`. (A
  first run with the original m1opt v4 binary gave the same results; that binary and its logs were lost
  to a reboot.) The `npar` half rests on reading the code (`number_par` builds the same `Num` with these
  arrays empty, :717-720) and is **unconfirmed**: no `npar` combination was run. No reported number is
  affected: every step that had `book` (step 6) or `npar` (step 14) also had `hb2` (step 1) and `pk`
  (step 5). The switches are not independent, which matters if the module keeps any switch-off path.
  **Other switch dependencies** (by reading the code, harmless in the measured steps): `hcon` takes
  effect only with `cpack` on (it is read only in `contract_packed`, m1_coarsen_bench.rs:1224, reached
  under `cpack`, :1028-1029); `hkey` at levels ≥ 1 needs `cpack`'s `wc_host` (`hkey = opt("hkey") &&
  w_host.is_some()`, :791; `wh = out.wc_host`, :1607), and without it falls back to the GraphBLAS path;
  `npar`'s threading needs `hpar` (`chunks_for`, :483).
* **H3 (`book` needs positive weights; Lean-only, unreachable in the bench).** `book_zero_weight_breaks`:
  a parent id with a weight-0 member in a stopped cell straddles before the stop, so the patch keeps its
  part in the surviving cell, while renumbering finds it clean. Bench weights are ≥ 1, so this cannot
  happen there; it is an invariant the module must keep.
* **No other difference.** Every other pass this project models (`hcon`, `cpack`, `pk`, `hsplit`,
  `hkey`, `hb2`, `gath`, `usel`, `zmatch`, `hpar`'s assembly) equals its formula under the stated
  preconditions, all of which hold for the bench's inputs; G2's numbering facts, the gathered level's
  `gw > 0` and `BookPre` are argued there, not proved. `book` is covered only at the level of the
  numbering's decisions (G1) and `npar` not at all.
* **Overflow: none in the bench, for the levels proved; B2 levels partly.**
  - *`hcon` levels (R and capstop's fresh hierarchies, depth 0).* Depth 0 is iso 1, and a level's value
    total is at most the previous level's (`hcon_total_le`), so every value and partial sum is at most
    `nnz(A_0)` ≤ 5,533,214. Offsets: an output row has at most as many entries as its members' rows
    (`hcon_row_bounds`, `grp_none_length_le`), so offsets ≤ `nnz(A_0)` < 2³².
  - *Weights.* Conserved (`cpack_conserves`), each at most `n` ≤ 1,965,206, pair sums at most `2n`.
  - *B2 levels.* Each value is a value of the parent's level or a part-row sum, at most the previous
    level's total (`hb2_vals`, `prow_bounds`; the `i32` partial sums `v += tmp[q].1`, :879, are bounded
    the same way, nonnegative terms). That bounds a B2 level by its parent and its previous level, not by
    `nnz(A_0)`: an absolute bound would need B2 = B1 (B1 is a contraction, so the argument of
    `hcon_total_le` would apply to it), which the bench checks at run time (`compare`, 65 of 65 levels on PA in the H2 controls) but
    this project does not prove. B2's offsets are not bounded here.
  - *Capstop's gathered graph.* Its values are values of its blocks' levels and its rows are no longer
    than theirs (`gath_vals`), so it inherits their bounds.
  - *Ids.* `nc ≤ n` for `hcon` levels. A B2 level has `np + nparts` ids (:305-306, :848-850) where `np`
    is the parent level's whole id space, holes included, and the parent at depth `d` is depth `d−1`'s
    B2 or capstop hierarchy (:2304-2306); with `nparts ≤ n` per level (a premise, G2), depth `d` has at
    most `(d+1)·n` ids (`ids_bound`). For the deepest depth `D = 14`: `15n` ≤ 29,478,090 on CA < `NONE`.
  - `bench_sizes_fit` checks CA, PA and TX. The bounds for other graphs: `nnz(A_0) < 2³¹ − 1` (values,
    as `i32`), `2n < 2³¹ − 1` (pair weights) and `(D+1)·n < u32::MAX` (ids).

## Gaps (not reached)

* **G1.** `book` is a model of the numbering's decisions, not a line-by-line model of `number_fast`,
  `number_par` and `number_patch` (the `(q, cell, x)` sort, the part runs, `ws` as `i32`, `ccount`, `live`),
  and the order of the surviving parts is a reading (R7).
* **G2.** `B2Pre`'s facts about the numbering (part members, `cmap`'s range, clean cells) and
  `ids_bound`'s `nparts ≤ n` are premises, not proved from `number`.
* **G3.** Only patterns are proved symmetric (`hcon_symm`, `hb2_symm`, `hsplit_symm`), which is what the
  matching and `hb2` use; value symmetry is not. The gathered graph's symmetry is argued from `gath_eq`
  (principal submatrices of symmetric levels), not stated as its own theorem.
* **G4.** The 2-hop pass's GraphBLAS half and the `MAX` choice are contracts (R4, R5); the rounds' exit
  test is not modelled as a loop (R3).
* **G5.** `tcol`'s pointer walk (R6), `hpar`'s `row_splits` (taken to be ascending from 0 to `nrows`),
  and `npar` (`number_par` and its atomics, the threaded `pack_ct`) are not modelled.
* **G6.** The checks (`level_checks`, `check_matching`, `compare`, hashes) and the uncoarsening
  stand-ins (`project`, `bnd`) are not modelled; the quality question (the stop rule, m1-optimize.md's
  corrections) is out of scope.
* **G7.** B2 levels' values are bounded only relative to their parent and previous levels, and their
  offsets not at all (see "Overflow").

-/
