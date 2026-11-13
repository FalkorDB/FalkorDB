import CchSupergraphReview.Game
import CchSupergraphReview.Order
import CchSupergraphReview.Masked
import CchSupergraphReview.Skeleton
import CchSupergraphReview.LogLink
import CchSupergraphReview.Invert
import CchSupergraphReview.Csr
import CchSupergraphReview.Transpose
import CchSupergraphReview.Topo
import CchSupergraphReview.Findings

/-!
# CCH redesign review: the supergraph is the elimination game of exactly the live graph at V_sg

**What is reviewed.** `.handover/cch-redesign/design/03-design.md`, revision 3.1, read in the worktree on branch
`cch/graphblas-redesign-handoff` (based on `origin/main` d6fcd5aac): §3 "Input skeleton", steps 0 and 8; §4 steps
1-5 and its Contract; §5 "The log", "Phases", "Bound and Overflow", "The full-seed pass"; §2 "Arc identity", I1′,
I7. Today's game is proved line by line from `origin/main` d6fcd5aac: `graph/src/index/falkordb/cch/order.rs`
(`chordal_supergraph`, `transpose`, `invert`, `Topology::from_order`, `Topology::build`, `arc_slot` and the
accessors) and `cch/mod.rs` (`id_index`, `index_id`). The tensor's live-value rule is `tensor.rs:52-70, 444, 465`
@main, imported from versioned_matrix. Revision 3.1's deletion masking depends on the result.

**Ownership of the `order.rs` rows.** The 22 PROVEN rows in `COVERAGE.tsv` are for main's functions, but this is
a review project, which the proofs skill folds away once the design's PRs land. Until then this project owns those
rows. When the design's PRs land, the `order.rs` model (Order, Invert, Csr, Transpose, Topo, Game) and its rows
move to an owning project: native_index, or a new `cch_order`. The design-only parts (Masked, Skeleton, LogLink)
are retired or re-stated on the code that lands. `cch_recompute_review` imports this project; its bridge uses
`fromOrder_spec`, `topo_chordal`, `gameRow_sorted`, `up_lt`, `up_supp` and `arc_ids`, whose names stay stable.

## The result, in one paragraph

Per type and slab, 03's masked read `pattern(m)⟨¬struct(dm)⟩ ∪ pattern(dp)` is exactly the pairs with a live edge
at the slab's version (theorem 1). If every pair whose liveness changed after its row's read is in the log (`LogComplete`), then
`B′ = (B ∖ D) ∪ A` is exactly the live pattern at V_sg, off the diagonal, and every live node has a rank
(theorem 2). `LogComplete` follows from cch_redesign_review's `no_lost_write` plus four premises: `Records`
(writers log what they change), `hres` (the build's structure is credited no later than its birth), `hrv` (every
slab is read between the birth and V_sg) and `EncSym` (a logged pair is unordered). An Overflow's rescan needs
none of them. `chordal_supergraph` is exactly the
elimination game for its order; the result is chordal in the CCH sense, and every skeleton edge is an arc
(theorem 3). Arc ids are row-major, `arc_slot` returns them, and `Down` carries the same ids (theorem 4). The
column counts are the game's row lengths: `|up(r)|` is the number of distinct higher skeleton neighbours of `r`'s
elimination subtree (theorem 5). Combined, `supergraph_is_live_game`: **the game over step 3's input is the game
over the live pattern at V_sg.**

## 03 section / Rust line ↔ Lean

| 03 / Rust | Lean |
| --- | --- |
| §3 "Input skeleton": `pattern(m)⟨¬struct(dm)⟩ ∪ pattern(dp)` per type and slab | `maskedRead`, `slabRead` (Masked.lean) |
| §3: union across types; `S = select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ))` (`input.rs:112-113` @05) | `P`, `skel` (Skeleton.lean) |
| §3 step 0, §4 step 1: who gets a rank | `Ranked`, `bprime_rank`'s `halloc` |
| §4 step 2: `A`, `D`, `B′ = (B ∖ D) ∪ A`, both orientations | `Aset`, `Dset`, `Bp` |
| §4 step 2 "Why the drop is valid" | `LogComplete` (premise), `Records`, `logComplete_of_writes`, `logComplete_of_reach` (LogLink.lean) |
| §5 "Bound and Overflow", at Supergraph: rescan | `rescan_logComplete`, `rescan_exact`; both branches in `supergraph_input_exact` |
| §5 "The full-seed pass": `m⟨¬struct(dm) ∧ ¬struct(dp)⟩ ∪ dp` | `seedM`, `seedVal` |
| §4 step 3: `U0 = triu(B′, 1)`, the child-absorption game | `rankU0`; `Adj`, `up`, `gameRow`, `par` (Game.lean); `absorb_iff` is the recurrence |
| §3 step 8, §4 step 2: etree and column counts | `par` (Liu: first entry of the row), `Desc`, `row_subtree`, `counts_spec` |
| §2 "Arc identity", §4 step 4: ids row-major, `Down = GrB_transpose` | `slotOf`, `arc_ids`; `TransposeContract`, `i7_of_contract`; `transposeG`, `down_ids` |
| `tensor.rs:444` `eff_get`, `:465` `get`, `:50-70` invariants | `VMTensor.effV`, `edges`, `Inv` (versioned_matrix) |
| `order.rs:51-73` `Rank::new`, `index`, `Debug::fmt` | `rankNew`, `rankIndex`, `rankDebug` (Topo.lean) |
| `order.rs:129-164` `Topology`, `RankRows`, `slots`, `row` | `Topo`, `rowSlots`, `slice` |
| `order.rs:180-182` `Topology::build`; `nd.rs:48-58` checks | `build`, `ndChecks` |
| `order.rs:195-212` `Topology::from_order` | `fromOrder` |
| `order.rs:214-268` accessors | `Topo.nodeCount`, `ranks`, `arcCount`, `nodeOf`, `rankOf`, `up`, `upRange` |
| `order.rs:287-302` `arc_slot` | `arcSlot` (`binary_search` by its std contract, `BsIdx`) |
| `order.rs:317-319` `node_index`; `cch/mod.rs:58-68` `id_index`, `index_id` | `idIndex` |
| `order.rs:324-329` `row_slots` | `rowSlots` |
| `order.rs:333-360` `invert` | `invertLoop`, `invert` (Invert.lean) |
| `order.rs:381-437` `chordal_supergraph` | `St`, `higherOf`, `absorbKids`, `stepRank`, `runRanks`, `chordalSupergraph` (Order.lean) |
| `order.rs:441-467` `transpose` | `countLoop`, `prefixLoop`, `fillRow`, `fillLoop`, `transposeG`, `transpose` (Transpose.lean) |

## Theorems

**1. The masked read is live** (03 §3).
* `maskedRead_eq_eff`: the masked read is the pattern of `eff_get`, with **no premise**.
* `maskedRead_live`: under `VMTensor.Inv`, a pair is read iff `Tensor::get` is non-empty. `slabRead_live` is the
  per-type, per-slab form, and `P_live` the union across types.
* `maskedRead_live_of_multi_me`: the only invariant used is promotion completeness (`Inv.multi_me`).
  `masked_needs_promotion_completeness` shows it is needed. `dm ⊆ m` and `dp ∩ dm = ∅` are not needed.
* `seedVal_eq_eff`, `seed_pattern_eq_masked` (03 §5 "The full-seed pass"): the seed union is exactly the live
  value, with no premise, and has the skeleton read's pattern.

**2. `B′` is the live pattern at V_sg** (03 §4 steps 1-2).
* `bprime_live`: given `LogComplete`, `B′(u, w) = (u ≠ w ∧ live at V_sg)`. `bprime_offdiag` and
  `triu_bprime_live` need no self-loop premise.
* `bprime_symm`, `skel_symm`, `skel_offdiag`: symmetric, no diagonal.
* `live_node_ranked`: every node with a live edge at V_sg has a rank (03 §4 Contract, I1′ at V_sg).
  `dead_node_no_entry`: a ranked node with no live edge keeps its rank and gets no entry.
  `bprime_rank`: in rank space, `B′` is the permuted live pattern.
* `rescan_exact`: an Overflow's rescan gives the live pattern with no log.
* `logComplete_of_reach`: `LogComplete` holds at any published version of any reachable state of
  cch_redesign_review's model, from `CchReview.no_lost_write`, `Records`, `hres` and `hrv`.
* `supergraph_input_exact`: end to end, step 3's input `U0` is the live pattern at `V_sg`, with or without an
  Overflow. `supergraph_is_live_game`: the game over it equals the game over the live pattern.

**3. The game is exact** (03 §4 step 3; `order.rs:381-437`).
* `chordalSupergraph_spec`, `chordalSupergraph_row`: `chordal_supergraph`'s rows are exactly `gameRow`, the
  elimination game's upward rows (`Adj`: eliminate ranks in order, join every two higher neighbours). Every index
  read is in range by the bounds the proof uses, and the `while let` never runs out (`chain_kids`).
* `absorb_iff`: child absorption is exact (Gilbert–Ng–Peyton). `clique`: chordal in the CCH sense, if `y < z` are
  both in `up(x)` then `z ∈ up(y)`. `skeleton_in_up`: every skeleton edge is an arc. `up_final`: `up(r)` is
  `r`'s row of the filled graph.
* `fromOrder_spec`, `topo_chordal`, `fromOrder_accessors`, `build_game`: the same for the `Topology` that
  `from_order` and `build` return, with `down` the transpose (`transpose_spec`). `Topology::down` is `cfg(test)`
  on main and production code at @08c (`order.rs:268-275` there); `fromOrder_spec` covers it either way.
  `invert_ok_spec`, `invert_ok_of_perm`: `invert` accepts exactly permutations and returns the inverse.
* **No panic.** The models index with `getD`, `List.set` and `List.modify`, which do nothing out of range where
  Rust panics. `chordal_in_range` (with `step_in_range`, `inv_prefix`, `run_split`) proves that at the start of
  every iteration `k` every index `chordal_supergraph` uses is below its array's length: the skeleton row, every
  `node_to_rank[neighbor]` (so `rankAdj`'s default is never taken), `first_child`, each visited child's
  `row_slots` and slice and `next_sibling`, and the parent's slot. `transpose_in_range` (with
  `fillRow_write_in_range`, `fi_prefix`, `transpose_offsets`) does the same for `transpose`: every count, every
  row read, and every write in the state where it happens. `invert_ok_spec` does it for `invert`.

**4. Arc ids** (03 §2 "Arc identity", §4 step 4, I7).
* `arc_ids`: entry `i` of row `lo` is at position `Ap[lo] + i`, its row-major id; `arc_slot` returns it; every id
  below the arc count is one arc's, and only one's.
* `arcSlot_spec`, `arcSlot_isSome`: `arc_slot(e, o)` is `Some` iff the two ranks are an arc, either order.
* `down_ids`: a transpose that writes each arc's id (`transposeG`, the same counting sort as `transpose`) gives
  `Down(h, l)` the id of `Up(l, h)`. `slotOf_is_up_id`: that id is where `Up` holds `h`.
* `i7_of_contract`: if `Down` is `GrB_transpose` of `Up` (03's choice; its C contract, `TransposeContract`) with
  the row-major ids `Ax[k] = k` (`upEntry`, over `t.upNb` and `up_range`), then `Down(h, l)` holds `x` exactly when
  `h ∈ up(l)` and `x = arc_slot(l, h)`.

**5. Column counts** (03 §3 step 8, §4 step 2).
* `row_subtree`, `counts_spec`: `up(r)` is the ascending list of the `w > r` that are skeleton neighbours of `r`
  or of an etree descendant of `r`, where the parent is the row's first entry (`par`, `par_spec`). Its length is
  the column count of `r`, and the counts sum to `arc_count`.

**Counterexamples kept as theorems** (Findings.lean): `asymmetric_misses_arc`,
`masked_needs_promotion_completeness`, `late_slab_breaks` (against `logComplete_of_writes`: `Records`, `EncSym`,
`hcov` and `lo ≤ rv u` hold, only `rv u ≤ V_sg` fails, and `LogComplete` fails).

**The contracts and premises are satisfiable**: `bsIdx_satisfiable` (`BsIdx` by `idxOf`), an `example` for
`TransposeContract`, `records_const` and `late_slab_breaks.1` (`Records`), and for `hres`: `hres_init`, `hres_s7E`
(the build's two births set `res` to the birth version), `batchCopy_keeps_res` (ReadSkeleton and Order batches
keep it) and `appendE_keeps_res` (a writer's segment on a Building copy keeps it).

## Premises (each theorem names its own)

* **Tensor invariants** `GraphInv` (`VMTensor.Inv` at every pair): proved preserved by versioned_matrix
  (`addBatch_correct`, `removeBatch_correct`, `removeFast_correct`, `foldP_correct`). Only `Inv.multi_me` is used.
* **`LogComplete`** (theorem 2): discharged by `logComplete_of_reach` from `no_lost_write` and the premises
  `Records`, `hres`, `hrv` (with `EncSym` for `SymL`):
  * `CchReview.no_lost_write` (proved in cch_redesign_review);
  * **`Records`**: writers log every pair whose liveness their commit changes (03 §5 "The log"). Not proved
    anywhere. Its Rust is on the stack (`maint.rs:238-253, 297-314` @08c), where `set_structural` and `PAIR_CAP`
    still drop pairs (`maint.rs:121-138` @08c); 03 says the design removes the drop;
  * `hres`: the Building entry's `res` is at most the build's birth until Supergraph consumes. That is
    cch_redesign_review's reading R1. The steps that make and keep such an entry preserve it (`hres_init`,
    `hres_s7E`, `batchCopy_keeps_res`, `appendE_keeps_res`); a full invariant over its `Reach` is not proved here
    (gap);
  * `hrv`: every slab was read in `[birth, V_sg]` (reading S2).
* **`EncSym`**: a logged pair is the id of `(min, max)` (03 §5). **`NoSelfLoop`** (only for `bprime_live`'s
  diagonal): writers drop self-loops (`maint.rs:131` @08c).
* **`Mirrored`** (theorem 3): `from_order`'s input is mirrored, which is `CchInput.skeleton`'s contract
  (`cch/mod.rs:168`). Checked at `nd.rs:54`, which discharges it for `Topology::build` (`ndChecks_symm`, with
  native_index's `isSymmetricSimple_iff`); 03's skeleton is mirrored by construction (`skel_symm`, `bprime_symm`).
* **`BsIdx`**: `slice::binary_search` returns the index of the value in a strictly sorted slice (Rust std's
  contract). **`TransposeContract`**: `GrB_transpose` gives `C(j,i) = A(i,j)` with values (GraphBLAS C API).
* `halloc` in `bprime_rank`: the allocation gives a rank to exactly the nodes 03 §3 step 0 and §4 step 1 name.

## Readings chosen (where 03 is ambiguous)

* **S1** (§3). "Live" means `Tensor::get` is non-empty for some indexed type; the skeleton uses either direction.
* **S2** (§5 "Phases", ReadSkeleton "Snapshot: any"). Every slab is read at a version at or after the build's
  birth and at or before V_sg. ReadSkeleton runs before the Supergraph batch forks, on one builder thread.
  `late_slab_breaks` shows the upper bound is needed: with every other premise of `logComplete_of_writes` holding,
  a slab read after V_sg makes `LogComplete` fail. Suggest: "any snapshot at or before the Supergraph fork".
* **S3** (§4 step 1). "A rank-less endpoint with a live indexed edge at V_sg" is an endpoint of a resolved pair
  with a live edge at V_sg to another node, in any type and either direction. A self-loop does not count, as the
  skeleton drops the diagonal.
* **S4** (§4 step 2). The resolved pairs are every pair in the fork's log nodes. All of them have `seq ≤ V_sg`
  (`CchReview.nothing_from_the_future`). `A` and `D` are decided by liveness at V_sg in any type and either
  direction.
* **S5** (§4 step 2, "Why the drop is valid"). "Every pair written after its slab was read is in the log" is two
  premises: `Records` and `no_lost_write`.
* **S6** (§5 Overflow at Supergraph). The rescan reads every row at V_sg and resolves nothing from the log.
* **S7** (§4 step 3). The game over `U0 = triu(B′, 1)` is the game on the undirected `B′`: `Adj` takes the
  symmetric closure, so `U0`'s rows are the higher neighbours, as in `order.rs:401-408`.
* **S8** (§4 step 4). "Assign ids row-major" is `Ax[k] = k`, so arc `(lo, up(lo)[i])` has id `Ap[lo] + i`.
* **S9** (§3 step 8, §4 step 2). The column counts are those of the Cholesky factor of `B′` in rank order,
  `|up(r)|`. 03 names no algorithm (Gilbert–Ng–Peyton, [R3 §1]); `row_subtree` is the characterization the
  algorithm computes.
* **S10** (§5). `m⟨¬struct(dm) ∧ ¬struct(dp)⟩ ∪ dp` at a pair is `dp` if present, else `m` unless `dm` masks it.

## Findings

No theorem of 03 fails under these readings. Two notes on wording, one on a premise and one on today's code:

* **N1, wording (§5 "The full-seed pass").** 03 justifies "the union is exactly the effective value" by the
  tensor's invariants. It needs none (`seedVal_eq_eff`); the `¬struct(dp)` mask makes the union disjoint. The
  skeleton read likewise needs none to be the pattern of `eff_get` (`maskedRead_eq_eff`), and only promotion
  completeness to mean "has a live edge". `dm ⊆ m` is not used.
* **N2, a premise to state (§5 "Phases").** Theorem 2 needs every slab read at or before V_sg. In
  `late_slab_breaks`, writers record, the log holds every write up to V_sg and every slab is read after the
  birth, yet one slab read after V_sg leaves `LogComplete` false and `B′` with a pair that is not live.
  ReadSkeleton's "Snapshot: any" holds only because phases run in order. Say so.
* **N3, the load-bearing premise (§4 step 2).** The drop's justification rests on `Records`: every writer logs
  every pair whose liveness it changes, including the node-delete cascade. Without it, a created edge that is not
  logged can lack an arc (I4 fails), and a deleted one keeps its arc. No proof covers `Records`; the stack code drops
  pairs today (`maint.rs:121-138` @08c), which 03 §5 already says the design removes.
* **N4, today's code, Lean-only, not run live (`order.rs:195-212`).** `Topology::from_order` checks
  `is_well_formed`, not mirroring. On a well-formed skeleton whose entry is listed only in the higher end's row,
  `chordal_supergraph` returns no arc (`asymmetric_misses_arc`). It is not a bug: `Topology::build` is its only
  non-test caller, and `nd.rs:54` refuses such a skeleton first. A `debug_assert!(skeleton.is_symmetric_simple())`
  in `from_order`, or a line in its doc, would state the precondition.

## Gaps

* **Rank allocation** is modelled only as "who gets a rank" (`Ranked`) and an injective `π`. Not modelled: the
  pool and top split by degree, `reserve`, `free_ranks`, P, and the extended dimension `n_r'`.
* **Not modelled, since the theorems hold for every order:** the extract `B = S(π,π)`, the postorder relabel,
  Liu's etree algorithm (C `cch.c:312-375`), the Gilbert–Ng–Peyton count algorithm, and their recompute when
  `B′ ≠ B`. Theorem 5 is the characterization of the counts, not their algorithm.
* **Step 3 over `U0` rows** is design-level: there is no code for it. `absorb_iff` is its recurrence. Today's
  `chordal_supergraph` is what is proved, over a mirrored `Csr`.
* **The full-seed pass**: only its per-pair value and pattern are modelled. Not modelled: MIN across types, `me`'s
  multi-edge minimum, the permutation `W(π,π)` and the by-id scatter.
* **Lineages after a fresh build** are not modelled: base plus delta, S2's appended ids (no longer row-major),
  and `ChunkedArr`.
* **Not proved in any project:** `Records`, and `hres` as an invariant of cch_redesign_review's `Reach` (its
  reading R1). Proving `hres` needs a new inductive invariant over that project's whole `Step` relation (writer
  rebase rows, CatchUp relinks, hand-offs and restarts), not just the four step lemmas here.
* **Taken as contracts:** `binary_search` (`BsIdx`) and `GrB_transpose` (`TransposeContract`).
* `u64`/`usize` are `Nat` except in `idIndex` and `rankNew`, where the 64-bit bound is explicit.

## COVERAGE

`COVERAGE.tsv` holds 22 live rows, all PROVEN: 20 `order.rs` functions, `id_index` and `index_id`. They moved
here from native_index/COVERAGE.tsv on 2026-11-13, where they had been NOT COVERED, and native_index no longer
lists them. `order.rs` is 20 of 20 PROVEN, with 0 rows unmatched. See "Ownership" above for where they go next.

`bash proofs/lean_ci.sh cch_supergraph_review` passes, with no `sorry`, `admit`, `native_decide` or `axiom`.
The theorems use only `propext`, `Classical.choice` and `Quot.sound`.
-/
