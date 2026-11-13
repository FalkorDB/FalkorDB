import CchRecomputeReview.Basic
import CchRecomputeReview.Relax
import CchRecomputeReview.Merge
import CchRecomputeReview.Scoped
import CchRecomputeReview.Step
import CchRecomputeReview.Main
import CchRecomputeReview.Weights
import CchRecomputeReview.MinChar
import CchRecomputeReview.Front
import CchRecomputeReview.Findings
import CchRecomputeReview.Bridge
import CchRecomputeReview.Seed

/-!
# CCH recompute review: the scoped recompute leaves exactly the full customization's metric

**What is reviewed.** Goal 3 of `.handover/cch-redesign/design/03-design.md`, revision 3.1: §6b in full ("What an
arc reads", Propagation, Lemma A, Lemma B, Algorithm, Interleaving, Order) and §2 "Arc identity" with I3′. Read in
the worktree on `cch/graphblas-redesign-handoff` (based on `origin/main` d6fcd5aac).

Rust read:
* the full customization on `origin/main` d6fcd5aac: `customize.rs` (`seed` `:204-240`, `relax` `:264-312`, the
  "Ties" rules `:66-75`) and `order.rs` (`Topology` `:129-139`, `RankRows` `:141-148`, `up` `:255`, `up_range`
  `:263`, `down` `:272` (test-only on main), `arc_slot` `:287-302`); the same `relax` body is `customize.rs:491-539`
  @08c;
* the scoped recompute on the stack: `recustomize_seeded`, `partial.rs:98-258` at `origin/cch/08c-build-catch-up`
  31ca1b802 ("@08c", the lines 03 cites). The file is byte-identical from `cch/08-maint` to the stack tip
  `cch/10-rdb-v20` 484e36d4e. The earlier `recustomize_scoped` (`cch/11-partial` cb2a97519, `:86-232`) differs in
  writing unconditionally (`:208-211`).

## The model

Ranks are `Nat`; `Topo` holds the upward rows `up`, the downward rows `down` and the rank count `n`. An arc is
`(lo, hi)` with `hi ∈ up lo`. The four metric arrays are one function `Metric W := Nat → Nat → Entry W` indexed by
the arc, not by its slot or id (reading RD1). Weights are an abstract `W` with `Ops` = `add`, `lt`, `beq` (the
`==` behind the code's `!=`) and `top`; each theorem names the laws it needs.

* `upd` is one apex step. It is `relax`'s body and the merge body of `recustomize_seeded` term for term and
  operand for operand: `dn(x,y) + up(x,z)` and `dn(x,z) + up(x,y)`, each kept under a strict `<`.
* `recomp` is 03 §6b "What an arc reads", as a computation. `IsFull` says every arc holds its `recomp`.
* `relax` is the code's three loops with the cursor walk; `none` is the panic.
* `mergeL` is the two-pointer merge. `popStep` is one pop of the heap loop, with the code's `!=`-guarded writes
  and propagation. `run` is the `while let` loop, with fuel.

## 03 section / Rust line ↔ Lean

| 03 §, or Rust (`@main` d6fcd5aac, `@08c` 31ca1b802) | Lean |
|---|---|
| §6b "What an arc reads"; `customize.rs:46-62` | `upd`, `recomp`, `IsFull`, `Topo.commonDown` (Basic) |
| `customize.rs:94-99` `Metric`; §2 Arc identity, I3′ | `Entry`, `Metric` (indexed by arc, RD1) |
| `customize.rs:204-240` `seed`: allocation `:208-214`, scatter `:222-237` | `seedAlloc`, `seedLoop`, `seedArrays`, `toMetric`, `seedsOf` (Seed); `seedE`, `seedM` |
| `customize.rs:187-196` `customize` (`seed` then `relax`) | `customize_build`, `arcSlot_lt`, `seedLoop_shape`, `toMetric_seed` (Seed) |
| `customize.rs:264-312` `relax` (`:268` x loop, `:272-276` y loop, `:284-309` z loop, `:288-294` cursor) | `relaxX`, `relaxY`, `relaxZ`, `cursorStep`, `position`, `relax` (Relax) |
| `customize.rs:261-263` "a `z` missing from `up(y)`" panic | `relax_none_of_not_chordal`, `relax_some` |
| `order.rs:141-148` rows ascending; `:119-121` `downward` = transpose | `Topo.Valid` (P-rows), `Topo.DownOK` (P-transpose) |
| `order.rs:195-212` `from_order`, `:180-182` `build` (`cch_supergraph_review`'s model) | `ofSG`, `bridge`, `up_past`, `down_past`, `relax_fromOrder`, `relax_build` (Bridge) |
| `customize.rs:270, 275, 285` `x_slots_start + i`; `:279, 294` `y_slots_start + cursor`; `order.rs:287-302` `arc_slot` | `rd1_slots`, `cursorStep_get` (Bridge; RD1) |
| `partial.rs:106-126` @08c resolve + `sort_unstable`/`dedup` | `resolve`, `ordered`, `dedupL`, `frontOf` (Front) |
| `partial.rs:163-164` heap and `queued` from the front | `init` |
| `partial.rs:168` `heap.pop()`; §6b Algorithm 2 key `(lo, hi, id)` | `PickSpec`, `pickMin` (Weights), `run` |
| `partial.rs:178-187` reseed (`known`, `seed(u,v)`, `seed_of`) | `seedE S` (P-seed) |
| `partial.rs:188-219` merge of `down(lo)`, `down(hi)`; §6b Algorithm 3 | `mergeL`, `merge_eq_recomp` (Merge) |
| `partial.rs:221-236` `!=` change test and writes | `changedB`, `writeNew`, `popStep` (Scoped) |
| `partial.rs:241-254` propagation; §6b Propagation; Algorithm 4 | `pushAll`, `pairOf`, `pairOf_arc` |
| §6b "Order" (pops by increasing lower rank finalize inputs, I8) | `SInv.mono`, `below_threshold` |
| §6b Lemma A | `lemmaA`, `lemmaA_exact`, `new_arc_no_old_apex`, `upd_top_noop` (Main) |
| §6 "Node delete" correctness ("resolves to `+∞` by induction on rank") | `deleted_node_arcs_top` (MinChar) |
| `validate`, `customize.rs:136-161` (no NaN, `≥ -0.0`) | `Agree`, `TopLaws`, P-f64; `SZ` |

## Theorems

**1. Full customization** (`customize.rs:264-312` @main). Premises: P-rows; P-chordal for the no-panic half. For
today's topology both are discharged by the bridge below.
* `relax_isFull`: if `relax` returns `R`, every arc of `R` holds its `recomp`, the fold over its apexes in
  increasing `x` from its seed.
* `relax_some`: under P-chordal it returns.
* `relax_none_of_not_chordal`: it panics on every non-chordal topology, so P-chordal is exactly the panic.
* Closed form (MinChar), for a strict total order (`StrictTotal`; `NInf.order` discharges it):
  * `isFull_up_min`, `isFull_dn_min`: the weight is `min(seed(a), min over x ∈ down(lo) ∩ down(hi) of
    dn(x,lo) + up(x,hi))`, and the mirror;
  * `isFull_up_mid`: the middle is `none` iff no candidate beats the seed; otherwise it is the first apex in
    increasing `x` whose candidate attains that minimum, the minimum being strictly below the seed.
* **The bridge to today's code** (Bridge, importing `cch_supergraph_review`):
  * `bridge`: for every `fromOrder sk r2n = .ok t` with `Mirrored sk`,
    `(ofSG t).Valid ∧ (ofSG t).Chordal ∧ (ofSG t).DownOK`, where `ofSG t = ⟨t.nodeCount, t.up, t.down⟩`.
    It uses `fromOrder_spec`, `gameRow_sorted`, `up_lt`, `up_supp`, `topo_chordal`, and `up_past`/`down_past`
    (rows at or above `n` are `[]`).
  * `rd1_slots` discharges RD1 with `arc_ids`. Every slot `relax` indexes (`x_slots_start + i`, and
    `y_slots_start + cursor` by `cursorStep_get`) is below `arc_count` and holds the arc it means; `arc_slot`
    returns it; slots are injective on arcs and onto `0 .. arc_count`. So no index is out of range, and an
    array indexed by slot is the model's metric indexed by arc.
  * `relax_fromOrder`, `relax_build`: on the topology `from_order` or `build` returns, `relax` never panics and
    returns the full customization. `relax_build` discharges `Mirrored` through `nd.rs`'s symmetric-simple
    check, given `binary_search`'s std contract (`NI.Cch.BsSpec`).
* **`seed` meets `relax`'s input contract** (Seed). `relax_isFull` starts from `seedM` (middles `None`), and
  "no index out of range" needs arrays of `arc_count` entries. `seed` provides both:
  * `seedAlloc` (`customize.rs:208-214`) allocates four arrays of `arc_count` entries, `+∞` and `None`;
  * `seedLoop` (`:222-237`) writes only the weight arrays, at `arc_slot`'s slot;
  * `arcSlot_lt`: that slot is below `arc_count`, by `rd1_slots` and `arc_ids`;
  * `seedLoop_shape`: the four lengths are kept and the middles untouched, for any weights;
  * `toMetric_seed`: read by arc, the arrays are `seedM` of the seeds they hold;
  * `customize_build` composes these with `relax_fromOrder`. On the topology `Topology::build` returns,
    `customize`'s `seed` then `relax` (`:187-196`) runs on `arc_count`-entry arrays and returns the full
    customization.
* No law on `add` is used, so IEEE rounding plays no part: each candidate is one addition, of the same two stored
  values, in the same operand order, in `relax` (`:297, :304`) and in `recustomize_seeded` (`:206, :212` @08c).

**2. Scoped recompute = full customization** (§6b; `partial.rs:98-258` @08c).
* `scoped_agrees` (**Theorem G**, Step): the loop starts from any metric that is consistent off the front:
  every arc outside the front already holds, under `R`, what it reads. Then the loop empties its heap (fuel
  `> arc_count`; `run_spec`: each arc pops at most once) and leaves every arc agreeing with any full
  customization: weights under `R`, middles exactly. Premises: P-rows, P-chordal, P-transpose, `PickSpec`
  (any key ordered by `lo` first), and `Agree o R`.
* `scoped_after_seed_change` (Main): the start is a full customization for `S`, and the front holds every arc
  whose seed changed. The result agrees with the full customization for `S′`.
* `scoped_after_seed_change_exact`: if `beq` is sound for equality, the result equals `relax` for `S′`, arc for
  arc, in weights and middles.
* `scoped_after_seed_change_bitwise`: with the change test made bitwise (`to_bits()` in Rust), the result is
  bit-identical to `relax` for **any** `add` and `lt`.
* End to end from the `changed` pairs (Front). If `changed` names every arc whose seed changed, in either order
  (`partial.rs:79-81`):
  * `recustomize_seeded_agrees`: the result agrees with `relax`, weights under `R` and middles exactly. This is
    the form that holds for the code as written over `f64` (`R = ==`, by P-f64).
  * `recustomize_seeded_exact`: the result equals `relax` bit for bit, but only with a change test sound for
    equality. That holds for exact arithmetic (`recustomize_seeded_exact_NInf`, no premise left) and for a
    `to_bits()` test. It does not hold for the code's `f64 !=` (R-F1).

**3. Lemma A** (§6b). `lemmaA`, `lemmaA_exact`. The setting:
* `H ⊆ H′` (rows; ranks may be added);
* on `H`'s arcs the start metric is `H`'s full customization; the new arcs `N = H′ ∖ H` are `+∞` both ways;
* the front holds `N` and every old arc whose seed changed.

Then the loop on `H′` ends at `H′`'s full customization. The proof is 03's argument:
* an apex that is new to an old arc reaches it through a new, `+∞` arc, so it never wins (`upd_top_noop`,
  needing `TopLaws`);
* so an old arc off the front is consistent, and Theorem G applies.

Only `H′` needs P-rows, P-chordal and P-transpose. `new_arc_no_old_apex` checks 03's first bullet (I5, `H`
chordal), which the proof does not need.

**4. Weights** (Weights, `Basic.Agree`).
* Exact arithmetic is `NInf` (`ℕ ∪ {+∞}`): `NInf.agree` (with `R = Eq`), `NInf.top`, `NInf.order`.
* `f64` on the values `validate` admits (not NaN, `≥ -0.0`, `+∞` allowed) satisfies `Agree` for
  `R = (· == ·)` and `TopLaws`: premise **P-f64**, an IEEE fact this core-only project cannot check. The signed-zero
  model `SZ` checks it (`SZ.agree`, `SZ.top`).
* So for `f64` the code's result is `==`-equal to the full customization, middles identical.
* It is not bit-identical: `==` is not sound for equality at `±0.0` (`SZ.beq_not_sound`), which is R-F1.

## Premises (each theorem names the ones it takes)

* **P-rows** (`Topo.Valid`). Rows are strictly ascending, above their rank, below `n` (`order.rs:141-148`).
  * Today's code: discharged by `cch_supergraph_review` through `bridge`, for every topology `from_order`
    returns on a mirrored skeleton.
  * After S2: discharged by `cch_topology_review` (`CchTopologyReview/RecomputeBridge.lean`). `toTopo_valid`
    turns its invariant into `Topo.Valid`. `lineage_valid_downOK` proves it at every state of a lineage that
    starts from a fresh build. `lineage_scoped_agrees` instantiates `scoped_agrees` there.
* **P-chordal** (`Topo.Chordal`).
  * Today's code: discharged by `bridge` through `topo_chordal`.
  * After S2 (the Davis–Hager fixpoint, 03 §6 table, S2 row): open. `lineage_scoped_agrees` carries it as
    the premise `hC`.
* **P-transpose** (`Topo.DownOK`). `downward` is the transpose of `upward`, rows ascending (`order.rs:119-121`,
  `transpose` `:441`); in the redesign, `Down = Upᵀ` with the same ids (I7).
  * Today's code: discharged by `bridge` through `fromOrder_spec`'s `t.down h = lowersK (gameRow …) n h`.
  * After S2: discharged by `cch_topology_review`. `toTopo_downOK` gives `Topo.DownOK` from its invariant, and
    `lineage_valid_downOK` and `lineage_scoped_agrees` give it at every lineage state.
* **P-seed**. Within one call, `SeedFn` (and the `known` cache, `partial.rs:149-161`) is a function of the pair.
  It gives the bits the full path seeds (`partial.rs:87-91`, Preconditions). Not discharged (see the note under
  Findings).
* **P-front**: `changed` covers every arc whose seed changed (`partial.rs:79-81`; 03 §6b, kept by rank allocation
  §6 and the full-seed pass §5). It is the input premise `hchanged`/`hfr`. Lemma A adds `N ⊆` front (03 §6b
  table, S2 row).
* **P-heap** (`PickSpec`). `BinaryHeap<Reverse<(Rank, usize)>>::pop` returns the least `(lo, slot)` (std
  contract). 03's `(lo, hi, id)` satisfies it too (`pickMin_spec`).
* **P-f64**: above. It is needed only for the agreement form over `f64`. The exact and bitwise forms do not use it.

## Readings chosen (where 03 is ambiguous)

* **RD1** (§2 Arc identity, I3′). The metric is indexed by the arc `(lo, hi)`.
  * Today's code: discharged by `rd1_slots` (`arc_ids`). The slot `up_range(lo).start + i` is row-major,
    injective on arcs, in range, and onto `0 .. arc_count`.
  * The redesign: the id (03) is injective on arcs within one lineage ("No reuse"). But after S2 the positional
    `x_slots_start + i` is **not** the id: `cch_topology_review`'s `B2_positional_id_breaks_after_s2`, and 03 §2
    "No positional writes". This arc-indexed model cannot see that class of bug: a code that reads the metric
    at a position instead of the id is not what it models.
* **RD2** (§6b Propagation, "changes bitwise"). It is modelled as the code's `!beq`, with `beq` abstract.
  "Bitwise" is `beq := decide (=)` (`bitwise`). Read this way, 03's wording is right, and the code at @08c is not
  bitwise (R-F1).
* **RD3** (§6b Algorithm 2). Any heap key that orders by `lo` first. Within one `lo` the pop order does not matter,
  because arcs with the same lower end never read each other.
* **RD4** (§6b, "fronts are node pairs, resolved to ids after all topology changes of the unit"). The front is a
  list of arcs of the final topology, deduplicated.
* **RD5** (§6b Lemma A, "valid topologies"). `H′` satisfies P-rows, P-chordal and P-transpose. `H` matters only
  through its full customization. The preset is `(+∞, +∞)` in weights; the middles of new arcs are free (03
  says `none, none`; they are recomputed because `N` is in the front).
* **RD6** (§6b, "min(seed(a), min over x …)"). It is the strict-`<` fold in increasing `x`, which is what both
  codes run. The `min` form is proved for a strict total order (`isFull_up_min`).
* **RD7** (§6b, "Its middle is the first x to win a strict `<` in increasing x"). Read as "the first `x` whose
  candidate attains the minimum, the minimum being below the seed" (`isFull_up_mid`). Read literally it is
  false: R-W1.
* **RD8** (§6b "seed(a)"). The pair's current cheapest weight per direction, `+∞` if none (`seed_of`).
* **RD9** (Theorem 2). "Fully customized for `S`" means any metric with `IsFull` for `S`, for example `relax`'s
  output. Uniqueness is not needed and is not proved.

## Findings

* **R-F1 (confirmed live, stack code @08c through the tip `cch/10-rdb-v20`; minor). The scoped pass is
  `==`-equal to the full customization, not bit-identical.**
  * The fault: `recustomize_seeded` tests "changed" with `f64 !=` (`partial.rs:221-222`), and writes each weight
    only where `!=` says it differs (`:225-230`).
  * A seed that changes only the sign of a zero is therefore neither written nor propagated, while `customize`
    stores the new sign.
  * Lean: `R_F1_full`, `R_F1_scoped`, `R_F1_not_bit_identical` on the repro's 3-rank graph. `R_F1_agrees` shows
    the result is still `==`-equal there, middles included, as `scoped_after_seed_change` says in general.
  * Live: a Rust test added to `partial.rs` in a scratch worktree at 31ca1b802, run with `cargo test -p graph
    --features index-falkordb --lib -- lean_r_f1`. It printed `arc 0-1 down: scoped 0x0 full 0x8000000000000000;
    arc 1-2 up: scoped 0x0 full 0x8000000000000000`, and the existing `snapshot ==` comparison passed. The repo's
    own tests compare with `PartialEq`, which is why they cannot see it.
  * Repro, kept in the repo: `.handover/cch-redesign/repro/r-f1-signed-zero/`, with `test.patch` (fails at
    31ca1b802), `fix.patch` (the `to_bits()` fix; the test and all 7 `partial.rs` tests pass), and the run logs.
    Re-run there on 2026-11-13 after a reboot wiped the first scratch copy.
  * It contradicts the claims at `partial.rs:19-21, :83` and 03 §1.8 / §6b ("bit-identical"). Weights compare
    equal, so the distances do too; only the sign of a zero weight differs.
  * At `cch/11-partial` the front arc itself was written (unconditional writes `:208-211`), but propagation was
    still skipped.
  * Fix: compare `to_bits()` in the change test and in the the two conditional weight writes. Then
    `scoped_after_seed_change_bitwise` applies, with no IEEE premise; `R_F1_bitwise_fixes_it` shows it on the
    repro.
* **R-W1 (design-level wording, no code to reproduce).** §6b says the middle is "the first x to win a strict `<`".
  Both codes record the *last* strict winner, which is the first apex attaining the minimum.
  `R_W1_middle_is_not_the_first_winner`: apex 0 wins first (5 < ∞), apex 1 wins last (3 < 5), and the middle
  is 1. `customize.rs:66-71` words it correctly ("among detours that tie, the lowest apex"); 03 should copy it.
* **Note (P-seed, unconfirmed, not proved).** The two paths take a pair's minimum with different operators.
  `PairWeights::cheapest_edge` keeps the first edge under `weight < best` (`input/pair_weights.rs:29` @08c).
  `CchInput::build` reduces with GraphBLAS's `MIN` monoid (`input/weight_ops.rs:347` @08c). With parallel
  edges of weights `+0.0` and `-0.0` they may pick different zeros: the same class as R-F1, on the seed side. Not
  reproduced.
* The rest of §6b holds as written, in the model:
  * propagation is complete;
  * popping by lower rank finalizes inputs first;
  * the `queued` set never blocks a needed recompute (an arc pops once, after all its inputs are final);
  * `arc_slot(hi, w).expect` never fires under P-chordal (`pairOf_arc`);
  * the `n/8` valve's full path equals the scoped one (Theorem 1);
  * Lemma A is true.

## Gaps

* **Lemma B (S5, later)** is not modelled. It deletes arcs, so `H ⊄ H′` and Lemma A's argument does not apply;
  the contract change (a "replaced row" marker, deletions in `added_since_fold`) has no text precise enough to
  model. It is design-level and later.
* **`f64` closed form.** `isFull_up_min` needs a strict total order. `f64 <` is one only up to `±0.0`, so for
  `f64` the `min` form holds up to `==`; that is not proved here.
* **Not modelled.**
  * The valve (`partial.rs:131-133`: `NeedsFull`, after which the caller runs `customize`, so Theorem 1 covers it).
  * The no-arc pairs (`:134-145`; 03's premise "every edge has an arc").
  * `InvalidWeight`, and the `arcs` counter of `Scoped { arcs }`.
  * `validate`. `seed`'s allocation and scatter shape are modelled (Seed), but no theorem covers its
    `NeedsRebuild` return or its self-loop skip, so `seed`'s own row stays NOT COVERED.
  * `WeightArr`'s copy-on-write (`customize.rs:182-189` @08c), and the design's base + delta row merge (03 §6b
    Algorithm 3/4: P-rows and P-transpose stand for it).
  * The work-unit budget and abort (Algorithm 5): an aborted run publishes nothing, so there is nothing to prove.
  * Interleaving with a batched build (covered by `cch_redesign_review`).
* **Node space.** `rank_of`/`node_of` (I1, I1′) are taken as a bijection: seeds and fronts are in rank space.

## COVERAGE rows

* `graph/src/index/falkordb/cch/customize.rs	relax	264	PROVEN` (in this project's `COVERAGE.tsv`; moved from
  `native_index`, where the row is deleted). The reason cites:
  * `relax_isFull`, `relax_some` and `relax_none_of_not_chordal`;
  * the closed form `isFull_up_min`/`isFull_dn_min`/`isFull_up_mid`;
  * `bridge` (P-rows, P-chordal, P-transpose for every `from_order` topology, via `cch_supergraph_review`'s
    `fromOrder_spec` and `topo_chordal`);
  * `rd1_slots` (RD1 via `arc_ids`: every slot in range, injective);
  * `relax_fromOrder`/`relax_build` (no panic; `Mirrored` from `nd.rs`'s check, `binary_search`'s std contract);
  * `customize_build` (with `arcSlot_lt`, `seedLoop_shape`, `toMetric_seed`): `seed` hands `relax` four arrays of
    `arc_count` entries with `None` middles, as `relax_isFull` assumes. So the input contract is discharged, not
    assumed.
* `Topology::up`, `up_range`, `arc_slot`, `ranks`, `chordal_supergraph`, `transpose` and the other `order.rs`
  rows are PROVEN in `cch_supergraph_review`, which this project imports. `seed`'s row stays NOT COVERED in `native_index`:
  Seed.lean models its allocation and scatter shape, but not its error returns. `customize` and `validate` keep
  their rows too.

**For when `partial.rs` lands** (stack code, no row on main): the row is
`graph/src/index/falkordb/cch/partial.rs`, `Cch::recustomize_seeded`, line 98, CONDITIONAL. Its reason starts
"Scoped path only" and cites:
* `CchRecompute.popStep`/`run`;
* `recustomize_seeded_agrees`: the form for the code's `f64 !=`, `==`-agreement with middles exact;
* `recustomize_seeded_exact`: bitwise, but only with `to_bits()`;
* `lemmaA`;
* the premises P-rows, P-chordal, P-transpose (for the stack's `from_order` topology the same `bridge` applies),
  P-seed and P-f64;
* R-F1: bit identity is false under `!=`.

What the row's model leaves out (not modelled, and nothing is claimed about what these branches write):
* `NeedsRebuild` for `u >= n` (`partial.rs:115-116`);
* the `n/8` valve (`:131-133`, `NeedsFull`);
* the no-arc pairs (`:137-145`);
* the `InvalidWeight` checks (`:150-161`);
* the `arcs` counter.

The other lines of the Scoped path:
* The `expect`s at `:202-203` hold: the merge visits exactly `commonDown` (`down_inter`, `mem_commonDown`), so
  `lo` and `hi` are in `up(x)`.
* The `expect` at `:247-249` holds by `pairOf_arc` (P-chordal).
* The `debug_assert` at `:184` is dropped.
* The private helpers `seed_of` and `bad_entry` are not modelled.

## Gates

* `bash proofs/lean_ci.sh cch_recompute_review cch_topology_review`: `cch_recompute_review   ok`,
  `cch_topology_review    ok`, `2 projects, 0 failing`. There is no `sorry`, `admit`, `native_decide` or `axiom`.
* `#print axioms` on the main theorems (`bridge`, `rd1_slots`, `relax_build`, `customize_build`,
  `recustomize_seeded_agrees`, `lemmaA_exact`, `scoped_after_seed_change_bitwise`) shows only `propext`,
  `Classical.choice` and `Quot.sound`.
* `bash proofs/coverage.sh`: `0 rows unmatched, 0 bad buckets`.
-/
