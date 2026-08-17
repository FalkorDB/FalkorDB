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

**Still worth doing:** checked arithmetic in the cardinality identity. Deriving
the count removes the way a *mutation path* can break the identity, not the ways
a corrupt decoded blob or a memory error can, and the subtractions reach a `Vec`
pre-allocation in `algo_procedures`. Ten lines, no behaviour change.

**What not to bother with:** the run-time check this document previously
described (recover the count at commit, compare, repair). It is redundant now.
Its two subtleties are recorded only because they would resurface for anyone
tempted to reintroduce a cache: recovery is `O(multi)` rather than metadata
whenever `me` has live deltas, and `GxB_rowIterator_kount` is only an *upper*
bound on non-empty rows, so it can screen but never adjudicate — an over-counting
screen would "repair" a correct counter into a wrong one.

## 2. Land the block-indexed compound key

**Status:** open. Design settled in the paper's limits section; nothing
implemented.

**Problem.** `compound_key(src, dst)` packs two node ids into one `u64` row
index of `me`, which caps node ids at `2^32`. That is a real ceiling, not a
theoretical one, for a graph that churns ids.

**Design.** Make the key `(block, row)`: `me` becomes a sparse map from block id
to a matrix, and `compound_key` returns both halves. When `src` and `dst` both
fit in `2^32` the block id is `0` and the map holds exactly one matrix, so a
graph that never exceeds today's limit pays a map lookup and nothing else.

**Acceptance and the two measurements the paper asks for.**

1. On a graph confined to block `0`, the per-operation cost against today's
   single matrix — point read, promote, demote, full scan. Target: within noise.
2. On a graph whose multi-edge pairs are spread thinly across many blocks, the
   crossover where per-transaction work becomes proportional to the number of
   live blocks rather than to one. Report where it starts to matter, since that
   is the case the design trades away.

**Risks.** Every place that folds, waits or resizes `me` becomes a loop over live
blocks — including the commit path, which is where the per-block cost turns into
per-transaction cost. Serialisation format changes, so `Encode`/`Decode` needs a
version bump and a round-trip test against blobs written by the current code.

**Effort.** Several days. Touches serialisation, so it wants its own PR.

---

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

## 5. The two-state point-read cost (#2430) — localised, outside edge storage

**Status:** cause not named, but the search space is cut decisively.
`graph/.../issue_2430_bench.rs` and `bench/studies/issue_2430/`.

Measured at both grains, on the issue's fixture — node count held at 1,000 while
pairs grow, so a bigger graph means **longer adjacency rows**, not more of them:

| grain | 1,000 pairs | 121k–160k pairs | shape |
| --- | --- | --- | --- |
| whole query | 8,725 | 9,943 | **step** of ~1,150 between 41k and 88k |
| the read alone | 2,771 | 2,879 | **drift** of ~108, roughly logarithmic |

**Edge storage is not the cause, on two independent grounds.** Magnitude: the
tensor contributes at most a tenth of the effect. Shape: the tensor drifts
smoothly — which is what a binary search inside a lengthening row should do — and
a drift cannot produce a step. Storage format is constant throughout (`m` sparse,
`dp`/`dm`/`me` hypersparse at every size), which also refutes the leading
remaining hypothesis, a GraphBLAS format switch.

**A correction worth keeping.** The first version of this measurement gave every
pair its own row, pinning row length at 1 at every size, and so reported the read
as perfectly *flat* and cleared edge storage outright. That was an artifact of
the fixture. Row length is precisely the variable a point read is sensitive to;
the corrected fixture does show the tensor responding to it, just far too little
and in the wrong shape to be the effect. The conclusion survived, the evidence
for it did not.

**What is still open:** which pipeline stage steps, and why the selection is not
monotonic in graph size (the issue reports low/high/low/high/high; this fixture
reproduces a single crossing, so the non-monotonicity is fixture-sensitive). The
tensor can be excluded from that search.

## 6. What the evaluation still does not settle

Not future work so much as honest scope. Listed here so nobody has to re-derive
what was and was not measured.

- **Design (B) itself.** It exists nowhere, so every (B) number in the paper is a
  bound or a model over measured components. Implementing it — Boolean adjacency
  plus an always-materialised overflow matrix — is the only way to replace them,
  and it is a research prototype rather than a change to ship.
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
- ~~**Cold-cache and random access order**~~ — **partly done**. Scattered reads
  (multiplicative stride over the same pairs) move instruction counts by 1.9% at
  `k` = 1 and 0.3% above, and wall clock by up to 89%, rising with `k`. So the
  instruction metric is robust to access order — worth knowing — and blind to the
  cache effect that actually separates designs in time. **Still open:** genuine
  cold-cache (this is warm-but-scattered), and cycles on a quiet host.
- **The C-side entry points** `Tensor_ClearElements`, `Tensor_RemoveElements_Flat`,
  `Tensor_SetEdges`, row and column degree remain unmeasured. The harness now
  builds and runs from a `master` worktree plus the shipped archive, so adding
  them is a C edit and a rebuild, not a setup problem.
- **Wall clock and cycles.** Some runs shared the machine, so no claim in the
  paper is a latency claim. Re-running a quiet host would let that change.
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

## 7. Generalising the pattern

**Status:** open, conceptual, and the most transferable part of the design.

Inline-first with sentinel promotion needs nothing specific to edge identifiers.
Any sparse-matrix-backed store whose cell values are drawn from a domain strictly
smaller than the machine word, and whose multiplicity distribution is dominated
by 1, can use it: reserve a point outside the domain, keep the common case
inline, overflow to one shared hypersparse matrix keyed by a packed coordinate.

Writing it up as a pattern means stating the obligations, not just the mechanism:
the invariant set is what a user of the pattern inherits, and the paper's
experience is that the promotion-completeness invariant is the one that makes a
cell value self-describing and the cancel-to-clean invariant is the one that is
easy to get wrong. A short paper or a design note, not code.
