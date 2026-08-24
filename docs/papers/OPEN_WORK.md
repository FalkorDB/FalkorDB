# Relationship tensor: open work, and how to close each item

Companion to [`tensor.tex`](tensor.tex). The paper's Future work section says
*what* is open; this says *how*, at the level of detail someone can pick up.

Each item carries the same fields: what is wrong or missing, the design, how to
know it worked, and what could go wrong. Effort is rough and assumes familiarity
with `graph/src/graph/graphblas/tensor.rs`.

Measurement convention throughout: instruction counts (`bench/`'s
`proc_pid_rusage` backend, or `perf stat` on Linux), not wall clock — they
reproduce across machines and do not drift with load. Wall clock is worth
recording and not worth claiming. Hold capacity fixed so identifier reclamation
does not confound write paths, and report allocated bytes alongside whenever the
claim is about the allocator.

---

## 1. The counter — done, by deletion

**Status:** closed by #2439 (stacked on #2431). Recorded here because the route
taken was not the one this document first recommended, and the reason is useful.

The plan was to seal `me` and `multi_count` into one type so no call site could
forget to move both. What landed instead deletes the field and derives the
quantity at each `edge_count()`, which is stronger: there is no cache to keep
honest, so there is no obligation to encapsulate.

Two properties of the artifact made that viable, neither of which was obvious
until checked:

- `has_multi_edge()` never used the counter — it already asks `me.nvals() != 0`.
- `Decode` never read the counter from the blob; it derived it from the tensor
  section's own counts. **The on-disk format is unaffected**, which is what made
  this a refactor rather than a format migration.

The derivation is ordered by cost: a graph with no multi-edge pair answers from
`nvals`; an assembled `me` answers from its vector count, which is metadata; only
an `me` carrying live deltas needs an `O(multi)` walk. The walk was the reason to
expect trouble, and measurement disposed of it — commit folds `me`, so no query
shape reaches that case. Suite instructions came out at 0.9961x and 1.0018x over
two independent before/after pairs.

~~**Still worth doing:** checked arithmetic in the cardinality identity.~~ —
**done in #2565**, which also stopped the identity guessing at the failure
direction: each step names what it hit (`|dm| exceeds |m| + |dp|`) and which
invariant bounds it (`dm ⊆ m is broken`). Deriving the count had removed the way
a *mutation path* can break the identity but not the ways a corrupt blob or a
memory error can, and unchecked those reached `Vec::with_capacity` in
`algo_procedures` — so a storage corruption surfaced as an allocation abort. With
3c landed, that was the last unguarded path from a bad blob to a bad allocation.

**What not to bother with:** the run-time check this document previously
described (recover the count at commit, compare, repair). It is redundant now.
Its two subtleties are recorded only because they would resurface for anyone
tempted to reintroduce a cache: recovery is `O(multi)` rather than metadata
whenever `me` has live deltas, and `GxB_rowIterator_kount` is only an *upper*
bound on non-empty rows, so it can screen but never adjudicate — an over-counting
screen would "repair" a correct counter into a wrong one.

## 2. The block-indexed compound key — done, and two predictions were wrong

**Status:** closed by #2579 (fixing #2578). Recorded here because the design that
shipped differs from the one this document specified in two ways that matter, and
because the ceiling turned out to be a live defect rather than a documented
limit.

**What it actually was.** Not merely a cap. `compound_key` packed `(src << 32) |
dst`, which needs 60 bits at `src = 2^28` and is then out of range for `me`.
`Matrix::set` checked its GraphBLAS status under `debug_assert!` only, so release
builds **dropped the write silently** and left the pair tagged `MULTI_EDGE` over
an empty `me` row — promotion completeness broken, `get` returning nothing,
`edge_count` disagreeing with both the inserts and the reads. The guard that
existed checked `u32`, four bits above the bound that mattered, so it never
fired. Reproduced at two edges per pair: `src=2^27` reads `[10, 11]`; `src=2^28`
and `src=2^29` read `[]` with `edge_count=1`.

**Where the shipped design differs from the one specified above.**

1. **The block is a pair, not a scalar.** This document said "a sparse map from
   block id to a matrix", with the block derived from `src` alone. That leaves
   `dst` at 32 bits and moves the ceiling onto the destination. What shipped
   splits *both* endpoints at `BLOCK_SHIFT = 30`: the block is
   `(src >> 30, dst >> 30)` and the row is `((src & M) << 30) | (dst & M)`. Two
   30-bit halves fill the 60-bit index budget exactly, so the function is
   **total** — it cannot panic and cannot go out of range.
2. **A list, not a map.** "A map lookup and nothing else" was the wrong cost.
   Every real graph occupies exactly one block, so the blocks live in a
   `SmallVec` sized for one with block `(0,0)` at index 0: a compare, no hashing,
   nothing on the heap.

**Where it was right.** Migration-free, exactly as predicted — the blob stores
`(src, dst, ids)`, not row keys, so it is block-agnostic and re-keys on decode.
No version bump, contradicting the risk this document listed. And the mechanised
injectivity statement did have to be restated for pairs, and that restatement was
the improvement claimed: see item 2a.

**Both acceptance measurements were taken.** Per-operation cost on a graph
confined to block 0 is within noise for reads (inline 726.9 → 721.9 instructions,
sentinel 2,922.8 → 2,928.8) and costs the write path a little (promote 3,174.2 →
3,229.8, +1.8%; third-edge control 1,417.3 → 1,478.7, +4.3%) — the block
selection, paid once per identifier written. `block_scaling_bench` covers the
second: maintenance cost is linear in live block count, and block (0,0) alone is
unchanged.

**An unplanned second result.** The same change declared `me` **narrow** — 2^31
columns, so GraphBLAS stores column indices in 32 bits rather than 64. Columns
are edge ids and the column-index array holds one entry per stored id, so this is
the largest array in the structure: 6.44 B/id against 11.69 at full width, the
declaration alone accounting for 45%. `Tensor::widen_me_for_id` widens back if
edge ids ever reach 2^31, so the narrow default costs only the check.

That moved every auxiliary-space figure in the paper by exactly −4.00 B/id and
**inverted** one of its comparisons: the marginal engine-level cost per
identifier was 9.09 B/id against the C engine's 8.04, and is now 5.24. The
paper's §evalspace is updated with the before/after and the attribution.

---

## 2a. Restate the mechanised key lemmas for the block key — done

**Status:** closed alongside #2579, in the same PR as this document.

`Bounded p` (`p.1 < 2^32 ∧ p.2 < 2^32`) existed only to make the old key
injective and in range. It appeared in three `InvCore` fields and every theorem
that touched `me` — 40 sites across 9 files. The new key needs none of it, so
`Bounded` is **deleted** rather than weakened, and what remains is strictly
stronger:

| before | after |
| --- | --- |
| `key_inj : Bounded p → Bounded q → key p = key q → p = q` | `key_inj : key p = key q → p = q` |
| `key_lt : Bounded p → key p < 2^64` | `row_lt : ∀ p, row p < 2^60` |
| `keyHi`/`keyLo` under `Bounded` | `inv_key : ∀ p, keyInverse (blockOf p) (row p) = p` |
| — | `row_le_grbIndexMax : ∀ p, row p ≤ GrBIndexMax` (the property whose absence *was* #2578) |
| `InvCore.bounded` | *(field removed)* |
| `WellFormed.bounded` | *(field removed — a decoder obligation that no longer exists)* |

Two modelling changes came with it, both worth knowing before touching these
files. `me : Finset (Addr × Nat)` where `Addr = (Nat × Nat) × Nat`, because a row
key alone no longer identifies a pair. And `iter_edges`' forward half is
`fwdIterAll` — every row, no filter — because `u64::MAX` in the Rust means "no
bound", and modelling it as the numeral `2^64 - 1` only typechecked while
`Bounded` was quietly supplying that node ids are u64-representable. That one is
a modelling *bug* the refactor exposed rather than introduced.

`lake build` clean, 2024 jobs, no `sorry`, axioms unchanged.

## 3. Close the modelling gaps — done

**Status:** closed. All four sub-items below are proved; the only place the code
is still ahead of the model is the serialised **byte stream** (framing, lengths,
endianness), noted under 3c. Kept in full because each records something the
attempt turned up that the next person would otherwise re-derive.

### 3a. Batch-plan equivalence for deletion — done

**Status:** closed, in `proofs/tensor/Tensor/RemovePlan.lean`.

The read phase's `PairPlan` is modelled (`initPlan`, `stepPlan`, `planFold`) along
with the write phase (`applyPlan`), and:

- `tequiv_applyPlan_removeFold` — replaying one pair's plan and applying it once
  agrees with `foldl removeOne` over the same ids, *including* the
  demote-then-empty interleaving that was the reason to doubt it.
- `reported_iff` — the batch reports a pair exactly when it emptied one that was
  there.
- `applyPlan_comm` — plans for distinct pairs commute, so the write phase's
  hash-map order is irrelevant.
- `inv_applyPlan`, `edgesAt_applyPlan` — the batched path preserves every
  invariant and denotes the same multigraph, inherited through `TEquiv` rather
  than re-proved.

**The finding worth carrying, if this is ever redone.** The two paths are **not**
equal as terms, and attempting term equality is a dead end. Where a pair demotes
and is then emptied in the same batch, the sequential fold writes the survivor
into `dp` and removes it again; the batched path never writes it. `Layer` carries
a total `val` alongside its pattern, so the two layers differ in `val` at a
coordinate *outside* the pattern — unreadable, since every read goes through
`Layer.get`. Hence `TEquiv`, an observational equality, with `TEquiv.inv`
transferring the invariant bundle and `TEquiv.removeOne_congr` pushing the
induction through.

The commutation half came out cheaply once each component's update was named as a
function of the original tensor (`dpOp`, `dmOp`, `mtOp`): the decision a shape
takes reads `m`, which neither shape writes, so both orders take the same
decisions and the nine shape pairs are one-liners over three `Layer` commutation
facts plus `me_sdiff_comm` (where `key_inj` enters).

### 3b. Iteration as the merge that computes it — done

**Status:** closed, in `proofs/tensor/Tensor/Merge.lean`.

`Iter.lean` characterised the iterators by their *result*. This models the merge
that produces it — three ascending cursors, a `dm` lookahead that drops the `m`
entry it matches, and a `dp` entry that wins at a shared position — and proves:

- `mem_merge3` — the merge emits exactly `(m ∖ dm) ∪ dp`, with `dp` winning ties.
- `sortedBy_merge3` — it emits in strictly ascending `(src, dst)` order. Strict,
  so this also says no position is emitted twice. This is what downstream
  operators assume and what nothing checked before.
- `merge3_effGet` — the bridge back to the tensor: instantiate the cursors with
  the three forward layers and the output *is* the effective view.

**Finding: the merge needs Invariant purity, and the model shows why.** At a tie
the algorithm emits the `dp` entry and advances `m` **without advancing `dm`**.
That is sound only because a shadowed position carries no tombstone. Violate
purity and `dm`'s head stays pinned at the shadowed position while `m` moves past
it, so the next masked entry is compared against a stale head, fails the match,
and is emitted — a deleted edge appearing in a scan. `hpure` is that invariant and
the tie branch is where it is used. The paper motivates purity as keeping `dm`'s
meaning crisp for the fold and removal paths; the iterator depends on it too, and
more sharply.

**Note on the shape of the recursion, for anyone extending this.** Writing the
merge the natural way — "if `dp` is behind, emit it and advance `dp`" — recurses
on the pending list while holding the base fixed, so it needs a combined
termination measure, compiles by well-founded recursion, and its equation lemmas
then will not rewrite. Flushing the pending cursor up front (`takeLt`) makes the
recursion structural on the base list and every equation usable. It is also the
better model: *flush every pending entry below the current base position, then
decide about the base position* is what the loop does.

### 3b-bis. The model's counter — done

**Status:** closed. The model carried `multiCount` as a field with
`multi_count_eq : t.multiCount = t.multiPairs.card` in `Inv`, which is what the
Rust held before #2439 deleted it.

`multiCount` is now a *definition* — `t.multiPairs.card` — so `multi_count_eq` is
`rfl` and there is no clause to preserve. Three consequences, all in the
favourable direction:

- Every `multi_count_eq := ?_` obligation disappeared from the operation proofs,
  along with the six `*_multiCount` field-tracking lemmas that fed them.
- The count's transitions are now stated once each rather than tracked
  everywhere: `multiCount_addEdge_promote` (+1), `multiCount_addEdge_multi` (+0),
  `multiCount_addEdge_first` (+0) in `Add.lean`. These say something about which
  pairs read as `MULTI`, where the old lemmas said only that a field was bumped.
- `retro_promote_agrees`'s count conjunct is now *derived* from its `effGet`
  conjunct instead of proved separately, which is the honest dependency: agreeing
  on the effective view forces agreeing on a quantity computed from it.

The `trans_*` state theorems lost their count conjunct, deliberately: the count is
no longer a layer, so it does not belong in a statement about layers. Note the
one thing this exposed — those conjuncts were previously provable with no
`≠ MULTI` hypothesis because bumping a field cannot fail; as claims about the
derived count, several would have needed one. That is a mild argument that the
old formulation was weaker than it looked.

### 3c. Rejecting a malformed blob — done

**Status:** closed, in `proofs/tensor/Tensor/CodecCheck.lean`.

`Codec.lean` proved the round trip, which says nothing about a blob `encode` did
not write — and `decode` is the one entry point that manufactures a tensor out of
bytes. Hand it a blob whose forward matrix tags a pair as multi-edge while the
tensor section carries no ids for it and the modelled `decode`, being total,
produces a tensor violating promotion-completeness, from which the iterator would
index an empty row.

`WellFormed` is the predicate a decoder must check, and `wellFormed_iff_invCore`
says it is **exactly** right: the check accepts a blob if and only if the tensor
it decodes to satisfies the invariants. Soundness is the half that matters
(`invCore_decodeChecked`); completeness matters too (`decodeChecked_encode`),
because a check that rejected valid blobs would be a compatibility bug rather
than a safety one.

Each clause ranges over the blob's own tables, so each is a finite scan a decoder
can run: coordinates fit `u32` and the declared dimensions; every tagged cell has
≥ 2 ids in the tensor section; every tensor-section row belongs to a present,
tagged, bounded pair; stored and inline ids are GraphBLAS indices. `keyed` earns
its keep twice — with `key_inj` it also yields Invariant `row_empty`, since a row
keyed to a bounded tagged pair cannot also be some other bounded pair's row.

**What this does not cover.** The model is of the *structured* blob (`Encoded`),
not of the byte stream: framing, lengths and endianness are still outside. That
is the weaker half of what this item originally asked for, and deliberately so —
the half that carries the safety argument is the validation, and it is done.

## 4. Demotion policy: hysteresis — measured, and the upside is small

**Status:** answered without implementing it. `graph/.../oscillation_bench.rs`.

Eager demotion means a workload oscillating across the one-to-two boundary pays a
promotion *and* a demotion per cycle. The question was whether deferring demotion
to end-of-transaction is worth the space. The measurement that decides it is what
one oscillation costs today, split into the part hysteresis could remove and the
part it could not — 50,000 pairs, three repetitions, spread under 0.1%:

| phase | instr/pair |
| --- | --- |
| promote (+1 edge, crosses the boundary) | 2,967 |
| demote (−1 edge, crosses back) | 6,518 |
| **full oscillation** | **9,486** |
| control: +1/−1 on 3-edge pairs, no crossing | 7,106 |
| **transition-attributable** | **2,380** |

**So hysteresis can save at most 2,380 instructions per oscillation — 25% of the
cycle — and costs 46.2 B per oscillating pair held to commit.** Three quarters of
an oscillation is the delete machinery, which hysteresis does not touch: even the
non-crossing control delete costs 5,894.

**Recommendation: don't build it yet.** A 25% ceiling on a workload that has to
oscillate hard to matter, against a permanent space cost and a new piece of
per-transaction state, is a poor trade. Revisit if a real workload shows up that
oscillates; the bench is committed, so re-deriving the number is one command.

## 5. The two-state point-read cost (#2430) — fixed

**Status:** diagnosed and fixed (PR #2571). `bench/studies/issue_2430/`,
`graph/.../issue_2430_bench.rs`, and the regression bench
`graph/.../me_delta_bench.rs`.

**Cause: `me`'s delta-plus is non-empty.** When the multi-edge id matrix carries
pending entries, every multi-edge row read consults the delta layer instead of
reading the committed base alone — a flat **~1,200 instructions per read**,
independent of how much the delta holds. One pending id costs the same as eight
thousand.

That accounts for all three puzzles: the states are two because a delta is empty
or it isn't; size selects between them because size decides where the fold policy
last fired relative to the final write; and the selection is non-monotonic
because *that* is not a function of size.

Evidence, in order: the step is entirely inside `ExpandInto` and vanishes when
the filler edges use a different relationship type (so it is the tensor being
read); the same logical tensor built incrementally vs in one batch differs by
~1,300 in exactly the rows where `me.dp` is non-zero; and — causally — adding one
pending id to `me`, on a pair the probes never touch, costs ~1,200 per read at
every size. Ruled out: storage format, hyper-hash, index widths, forward deltas.

**A correction, and a lesson about the evidence.** An earlier round of this
concluded the opposite — "not edge storage" — on two grounds that both turned out
to be artifacts of how it was measured. The first fixture gave every pair its own
row, pinning row length at 1. The second held row length right but built the
tensor in one batch, which is precisely the condition that leaves `me.dp` empty
and the read in the *low* state. A synthetic fixture that never reproduces the
bug will always exonerate whatever it is pointed at. What broke the deadlock was
building the tensor the way the engine does — incrementally, one fold per batch —
rather than building what it holds.

**Why the issue's own third hypothesis looked refuted.** It records "a lingering
delta ... an unrelated write, which would flush it, does not move the number".
The fold policy is size-based, so a delta of one entry never meets the threshold
and a write is not obliged to fold `me`. The hypothesis was right; the experiment
could not see it.

**The fix (PR #2571).** The layer-empty short-circuit in `Iter::from_layers`
asks whether a delta is empty *in total*; the fix asks whether it holds anything
in the row being read. `RowFilter` is a 4 KB bitmap of the rows a delta may hold,
allocated only once the delta holds something and dropped when it folds. Its only
safety property is that it never says "no" when the answer is yes, so
`Delta::layer_mut` — the choke point every entry-adding path goes through —
invalidates it by default; a path added later without touching this stays
correct.

A skipped layer is **detached, not dropped**, because `Iter::seek` re-points an
iterator at a different range and a dropped layer would swallow entries in every
later one. `matrix::Iter::detached` keeps the handle and attaches on first seek.

| pairs | 1k | 11k | 41k | 88k | 121k | step |
|---|---|---|---|---|---|---|
| before | 8,723 | 8,787 | 8,786 | 9,929 | 9,963 | **+1,143** |
| after | 8,738 | 8,774 | 8,796 | 8,912 | 8,915 | **+116** |

90% of the step, and the two states are one. Writes pay +0.64%; memory is
unchanged. **Still open:** the more ambitious version, a fold policy that prices
what a resident delta costs *readers*. The square-root rule of §folding is
derived entirely from write amortisation and has no term for the read side, which
this bug was a direct consequence of.

## 6. What the evaluation still does not settle

Not future work so much as honest scope. Listed here so nobody has to re-derive
what was and was not measured.

- **Design (B) itself.** It exists nowhere, so every (B) number in the paper is a
  bound or a model over measured components. Implementing it — Boolean adjacency
  plus an always-materialised overflow matrix — is the only way to replace them,
  and it is a research prototype rather than a change to ship.
- ~~**(A)'s boundary harness is unpublished**~~ — **done**: it ships with the
  paper at `bench/studies/edge_storage/`, with its two out-of-tree inputs
  (a `master` checkout, a C build tree) overridable rather than hardcoded. Both
  halves of the boundary measurement are now reproducible from this branch.
- ~~**Fan-out beyond `k = 2` at the data-structure boundary**~~ — **done, both
  sides, and it inverted a claim.** The engine-level decomposition fitted 448
  instructions per additional id for (C) against 854 for (A), and concluded (C)
  walks a row more cheaply than (A) walks a container. At the boundary the slopes
  are **124 for (C) and 33 for (A)** — both engine-level figures high, by 3.6x and
  26x, so their *ratio* came out backwards. There is no crossover: (C)'s read is
  3.8–3.9x (A)'s at every `k ≥ 2` measured. This is the third instance of
  engine-level differencing overstating a storage cost and the first where the
  error flips an ordering, which is why the paper now states it as a limit of the
  method rather than a caveat about two numbers.
- ~~**Transposed iteration**~~ — **done, both sides.** (C) 786 instr/edge against
  (A) 562 at `k` = 1 (1.40x), and 3.7–4.0x for every `k ≥ 2`. That is the price of
  `mt` carrying structure only, and it makes incoming-edge-dominated traversal the
  shape this design serves worst.
- ~~**Cold-cache and random access order**~~ — **done, both sides, and the
  instruction metric errs in *both* directions.** Access order was the easy half
  and was already covered: scattered reads move instruction counts by 1.9% at
  `k` = 1 and 0.3% above, and wall clock by up to 89%. But 200,000 pairs fits in
  cache whichever order it is read in, so that measured order, not residency.
  Residency is now measured properly — a working-set sweep from 10^4 to 8x10^6
  pairs (0.2 MB to 558 MB), scrambled probe order, both engines, in
  `tensor_cost_cold_cache` and the C harness's `bench_sweep`.

  The instruction columns are the control and are flat: (A) within 0.04% across
  the whole range, (C) within 0.9% (`k`=1) and 3.6% (`k`=2). So the sweep changed
  residency and not work. The times are the result:

  | pairs | `k`=1 (A) | (C) | (C)/(A) | `k`=2 (A) | (C) | (C)/(A) |
  |---|---|---|---|---|---|---|
  | 10^4    |  21.9 |  34.3 | 1.57 |  40.4 | 188.8 | 4.67 |
  | 10^5    |  23.0 |  29.9 | 1.30 |  83.8 | 219.8 | 2.62 |
  | 5x10^5  |  24.5 |  30.7 | 1.25 | 246.3 | 274.4 | 1.11 |
  | 2x10^6  |  81.2 |  98.9 | 1.22 | 360.6 | 566.6 | 1.57 |
  | 4x10^6  | 103.1 | 166.7 | 1.62 | 368.8 | 651.9 | 1.77 |
  | 8x10^6  | 117.7 | 202.5 | 1.72 | 391.4 | 736.9 | 1.88 |

  At `k`=2 instructions say (C) costs 3.8–4.0x (A) at every size; time says 4.67x
  in cache and 1.88x out of it, because **(A) leaves the cache first**. That is
  not a guess: item 5's space constants (295.7 B per multi-edge pair for (A),
  69.2 for (C)) put (A) past 100 MB at ~3.4x10^5 pairs and (C) at ~1.5x10^6, and
  the measured cliffs bracket both predictions. **(C)'s space advantage is also a
  latency advantage**, and no instruction count can see it.

  At `k`=1 it runs the other way — instructions say 1.15x flat, time says 1.30x
  warm and 1.72x cold — and this one has no footprint explanation, since both
  designs measure 23.1 B per all-inline pair. Unexplained, 85 ns at the largest
  size, and stated as such in the paper.

  So the metric overstates (C)'s multi-edge deficit by ~2x out of cache and
  understates its single-edge deficit by ~1.5x. The two do not cancel; they act
  on exactly the two regimes the design trades between. **Still open:** cycles on
  a quiet host, and whether the magnitudes hold on another cache hierarchy (the
  `k`=2 *direction* is anchored by the space constants and should; the `k`=1
  direction rests on the measurement alone).
- ~~**The C-side entry points**~~ — **done, both sides**, in `bench_degrees`,
  `bench_remove_flat`, `bench_clear_elements`, `bench_set_edges` and Rust's
  `tensor_cost_entry_points`. The bulk paths are close: flat bulk delete 1,121 (A)
  / 1,279 (C) = 1.14x, batch insert 1,487 / 1,737 = 1.17x.

  Degree is not, and it produced a **new finding and a fixable defect**:

  | | (A) | (C) | (C)/(A) |
  |---|---|---|---|
  | row degree, `k`=1 | 1,011 | 1,983 | 1.96 |
  | row degree, `k`=2 | 1,160 | 4,886 | 4.21 |
  | col degree, `k`=1 | 1,432 | 2,573 | 1.80 |
  | col degree, `k`=2 | 1,581 | 5,475 | 3.46 |

  `Tensor_RowDegree` scans the row's *cells* and adds `GrB_Vector_nvals(V)` for a
  tagged one — constant time per pair, independent of `k` (hence 1,011 -> 1,160
  for twice the edges). (C) has **no degree entry point at all**:
  `Graph::get_node_outdegree` counts a full one-row iteration, materialising every
  id only to discard it, so its cost is proportional to *edges* and the ratio
  grows with `k`.

  **That explanation was wrong, and taking it apart is what found the real one**
  (PR #2572). Decomposing the k=1 figure: of 1,990 instructions the forward row
  scan underneath is 1,908, so *materialising the ids costs 81* — and the same
  scan through a **re-seeked** iterator costs 300. The dominant term, 1,608 of
  1,990, is constructing a `GxB_Iterator` per call. A degree that only stopped
  collecting ids would have moved 81 instructions.

  Worth recording as a process failure, not just a result: the first explanation
  was plausible, matched the shape of the numbers (a ratio growing with `k`), and
  survived a whole revision of the paper — because nothing had asked it to account
  for the *total*. The decomposition took ten minutes and inverted the conclusion.

  Two fixes followed, and the smaller-sounding one is the valuable one:

  - **Pool `GxB_Iterator` handles** (thread-local free list of 32). Sound because
    GraphBLAS documents re-attaching a handle to another matrix. This is general —
    it is on every row scan in the engine. Multi-edge point read 2,995 → 2,200;
    forward single-row scan 1,908 → 1,125; #2430's engine fixture 8,912 → 8,309.
  - **`Tensor::row_degree` / `col_degree`**, dropping the per-pair `Vec` and
    sharing one `me` cursor across a row.

  | | (C) | before | after | was | now |
  |---|---|---|---|---|---|
  | row degree, `k`=1 | 1,011 | 1,990 | 1,203 | 1.97x | 1.19x |
  | row degree, `k`=2 | 1,160 | 4,810 | 2,793 | 4.15x | 2.41x |
  | col degree, `k`=1 | 1,432 | 2,582 | 1,814 | 1.80x | 1.27x |
  | col degree, `k`=2 | 1,581 | 5,402 | 3,409 | 3.42x | 2.16x |

  **Still open:** the remaining ~830 instructions of attach are
  `GxB_rowIterator_attach` itself, which pooling cannot remove — only *not
  attaching* can. `Tensor::get` still builds a fresh three-layer merge per call
  and `ExpandInto` calls it per row, so a reusable cursor for the hot traversal
  callers is the next step and a larger API change. That is now the single
  biggest known gap to (A) on the read path.
- **Wall clock and cycles.** Some runs shared the machine, so no claim in the
  paper is a latency claim — with one deliberate exception, the residency sweep
  above, where a latency result is the whole point and no instruction count could
  substitute. It carries its own weaker method (minimum of three rather than
  median, ratios rather than magnitudes) and warrants nothing else. Re-running a
  quiet host, with cycles alongside, would let that change.
- **Linux and CI.** Everything is macOS/arm64, with both sides built against
  GraphBLAS 10.3.1. The engines' relative standing on the CI architecture is
  unmeasured.
- **GraphBLAS versions are not aligned across the two sides.** Trunk moved to
  10.5.0 (#2523). (C)'s boundary fixture has been re-run against it — space and
  iteration reproduce to three digits, point reads and promotion are dearer
  (inline 566 → 621, sentinel 2,502 → 2,938, promote 2,735 → 2,965). The C side's
  shipped archive is built against **10.4.0**, not the 10.3.1 the paper's Table 8
  cites, and its rows reproduce that table to within 1%, so the minor version is
  not what moves those numbers. Aligning both sides on one version would need a
  full C rebuild against 10.5.0 and is the remaining gap for any cross-version
  claim. **Also still open:** the whole engine-level multiplicity sweep on 10.5.0.
- **The live-bytes column's parse — re-measured, and it holds.** The space
  figures were collected through a `MEMORY MALLOC-STATS` parser whose whitespace
  split fused adjacent fixed-width fields at high allocation rates (#2492, since
  fixed). Both engines' space columns have now been re-collected under the
  corrected parse. (A)'s `GRAPH.MEMORY` column reproduces exactly (8/12/29/17/8
  MB), its live-byte hump reproduces in shape 1.5–2.2 MB low, and the ratio
  column reproduces its shape (min at mu = 2, crossing 1 at mu = 16). The parse
  defect did not manufacture the finding.

  Two things did *not* reproduce, and the paper now says so: (C)'s live bytes are
  **not** monotonic in mu — they fall after mu = 2 rather than rising, which is
  the favourable direction and therefore worth flagging — and (C) is the less
  stable engine across runs, moving 8% at mu = 2 where (A) reproduced
  bit-for-bit.

  **Still open here:** the *instruction* columns. The original sweep driver was
  not kept, and the reconstruction's build path is different enough that its C-side
  instruction counts are ~11x the reported ones, so it can only speak to space.
  Anyone re-running the insert column needs to rebuild the driver so that the
  build path matches, not just the final graph.
- **Whether multiplicity distributions dominated by 1 describe real
  deployments.** A question about deployments, not data structures, and the one
  assumption the whole design rests on.

---

## 7. Generalising the pattern — done

**Status:** closed, as `docs/papers/pattern.tex` (7 pages, standalone).

*Inline-first with sentinel promotion* stated with the edges taken out: any
sparse store whose cells are wider than their value domain and whose
multiplicity is dominated by 1. `tensor.tex` is its worked instantiation and
now points at it from the future-work section.

The mechanism was never the hard part — it is a paragraph, and arguably
folklore. What the note carries that the mechanism does not:

- **Two preconditions**, stated so they can be *checked* rather than assumed.
  Sentinel headroom is a property of the domain, not of current data: "no id is
  ever 2^64-1 in practice" is not headroom, "ids come from a counter bounded by
  the index limit" is. The distinction matters because a sentinel collision is
  the one failure with no graceful degradation — it returns a wrong answer
  rather than performing badly.
- **Six obligations**, and which ones actually bite. Promotion completeness is
  the whole value proposition (it is what makes a cell self-describing, hence
  why no reader probes the overflow speculatively — exactly the cost design (B)
  pays forever). Cancel-to-clean is invisible until something *counts*, which is
  why it was this design's hardest bug: reads are unaffected, so tests that read
  pass.
- **One obligation you should decline.** Count agreement began as a maintained
  counter and is now derived, which deleted the invariant and every per-mutation
  discipline attached to it. Generalised: *in a design whose invariants make a
  quantity structurally determined, caching it converts a theorem into an
  obligation.* Stated anyway, because an adopter who does not know it exists
  will cache the count.
- **The cost, not just the benefit.** Ten states per cell against a Boolean
  store's four, three of them existing only because cells carry values, plus the
  canonicalisation every mutation path must respect. A design that keeps the
  primary store Boolean needs none of it — and pays a second lookup on every
  read forever. That trade is the decision, stated in one sentence.
- **A second instantiation, sketched and then argued against.** Node labels fit
  both preconditions and would still probably lose on space, because the
  baseline (a bool matrix, no value array) has no container to displace. Kept
  precisely because a pattern note exhibiting only successes is not a guide.
- **The measurement traps, generalised.** All three caught us in print: a ratio
  of two whole-system differences coming out backwards; a plausible mechanism
  surviving a whole revision because nothing asked it to account for the total
  (81 instructions of 1,990); and instruction counts erring in *both* directions
  once residency changes. With the constructive corollary — the pattern's space
  advantage is also a latency advantage, and the space constants predict where.
- **When not to use it**, as five explicit disqualifiers.

**Still open, and deliberately so:** the note's `pattern.tex` claims transfer
from one instantiation. A second *built* instantiation would be the thing that
turns it from a generalisation into a pattern; the labels sketch says what would
have to be measured first.
