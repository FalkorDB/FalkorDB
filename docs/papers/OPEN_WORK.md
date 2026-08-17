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

## 3. Close the modelling gaps

**Status:** open. Three gaps, in the order they are worth closing.

### 3a. Batch-plan equivalence for deletion — *new, and the most urgent*

The Lean development models deletion as per-edge state transitions: `removeOne`
with its three reachable shapes (`removeOne_still_multi`,
`removeOne_demote_cancel`, `removeOne_demote_shadow`), composed over the batch.
The implementation no longer works that way. To make demotion linear in the
batch, `Tensor::remove_all` now reads each pair's `me` row once, replays that
pair's transitions into a plan, and applies the plan in a separate write phase.

The decisions are intended to be identical, so the invariants are believed
intact — but *"batching by pair and replaying is equivalent to the sequential
per-edge fold"* is a new obligation the model does not state. It is currently
supported by three regression tests, not by a proof. This is the one place where
the code moved and the model did not follow, so close it first.

**How.** State the plan as a function from a pair and its edge multiset to a
transition, prove it agrees with `foldl removeOne` restricted to that pair, and
prove that plans for distinct pairs commute (they touch disjoint `me` rows and
disjoint forward cells, which is what makes the two-phase split sound).

### 3b. Iteration as the merge that computes it

Iteration is mechanised by its *result* — the effective set restricted to a row
range — rather than as the three-way lookahead merge that produces it. The merge
is where the delicate index reasoning lives, so this is the more valuable of the
two remaining gaps.

**How.** Model the three cursors and their lookahead as a state machine and prove
it yields the effective set in ascending `(row, col)` order. The ordering half is
the part worth having: it is what downstream operators assume and what nothing
currently checks.

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

### 3c. The `Encode`/`Decode` blob format

Outside the model on both layers. Round-trip is proved "by computation" for the
in-memory structure only.

**How.** Model the byte format abstractly — a list of layer descriptors plus
payloads — and prove decode-after-encode is the identity, and that decode either
rejects a malformed blob or returns a tensor satisfying the invariants. The
second half is the one that matters, since a corrupt blob is one of the two ways
a proved invariant can still break in production.

---

## 4. Demotion policy: is hysteresis worth it?

**Status:** open, and now measurable. It was blocked: while a batch's demotions
cost more than linearly in their number, any hysteresis measurement would have
measured that defect instead of the policy. With the defect fixed and the
residual transition a constant, the comparison is worth running.

**Problem.** Demotion is eager — the instant a pair drops to one edge its id
returns inline and its `me` row empties. A workload oscillating across the
one-to-two boundary pays a promotion and a demotion per oscillation.

**Design.** Defer demotion to end of transaction: mark the pair, keep its ids in
`me`, and settle at commit. A pair that re-promotes within the same transaction
then pays nothing. Cost is space — an `me` row outliving its need — and one more
piece of per-transaction state. Note that item 1 removed the last such cache
rather than encapsulating it, so the bar for adding one back is that the
quantity genuinely cannot be derived from `me`; a deferred-demotion set cannot
be, which is what would justify it.

**Acceptance.** A Cypher script driving pairs across the boundary repeatedly, at
several oscillation rates, measuring instructions per cycle and resident bytes
for eager against deferred. Also measure the *non*-oscillating workloads, since
deferral must not cost anything there. The honest outcome may be "eager wins";
the paper says it is not obvious which does, and that remains true.

---

## 5. Explain the two-state point-read cost (issue #2430)

**Status:** open, cause unknown.

**Problem.** A bound multi-edge point read lands in one of two cost states —
about 4.35k or about 5.5k instructions per pair — selected non-monotonically by
graph size, crossing at least twice between 1,000 and 121,000 populated pairs.
The low state is at parity with the C engine; the high state is 1.27–1.29x it.
Both of the largest sizes measured sit in the high state, so this is where an
ordinary graph lands. Three hypotheses are refuted in the issue: scaling with
`|me|`, a lingering delta the read path latches but never flushes, and simply
walking more ids.

**Next candidates, in order.** A storage-format or capacity threshold inside
GraphBLAS is the obvious one — matrix dimensions step with node capacity, and
sparse/hypersparse selection depends on the ratio of populated vectors to
dimension, so print the sparsity status and the hyper-switch for every matrix on
the read path at both a low-state and a high-state size and diff them.

Both tools this needs now exist, which they did not when this was written:
`Matrix::sparsity_status()` (added by #2523) returns
`hypersparse`/`sparse`/`bitmap`/`full` directly, and `tensor_cost_bench.rs` is in
the tree, so a point read can be reproduced without a query pipeline around it —
which is also the cheapest way to tell whether the extra cost is inside the
tensor at all. Do that second step first if the sparsity diff comes back empty:
the paper's boundary numbers say a sentinel read is a flat cost, so a two-state
*engine-level* read with a one-state boundary read would locate the defect
outside edge storage and change who owns the issue.

**Acceptance.** Either a cause and a fix that puts the high-state sizes at the
low-state cost, or a documented explanation of why the two states are inherent.
"Reproducible, cause unknown" is where it stands; naming the cause is progress
even without a fix.

---

## 6. What the evaluation still does not settle

Not future work so much as honest scope. Listed here so nobody has to re-derive
what was and was not measured.

- **Design (B) itself.** It exists nowhere, so every (B) number in the paper is a
  bound or a model over measured components. Implementing it — Boolean adjacency
  plus an always-materialised overflow matrix — is the only way to replace them,
  and it is a research prototype rather than a change to ship.
- ~~**Fan-out beyond `k = 2` at the data-structure boundary**~~ — **done** for
  (C), `k` to 16. Two results: the entry charge is flat in `k` to four figures
  (2,684 instructions at `k` = 2 and at `k` = 16), and the per-identifier slope is
  **124** instructions against the 448 the engine-level decomposition attributed
  to it — a third instance of engine-level differencing overstating edge storage.
  Iteration peaks at `k` = 2 (250/edge) and returns to the all-inline 163 by
  `k` = 16. **Still open:** the same sweep on (A), which is what any *ratio* past
  `k` = 2 would need.
- ~~**Cold-cache and random access order**~~ — **partly done**. Scattered reads
  (multiplicative stride over the same pairs) move instruction counts by 1.9% at
  `k` = 1 and 0.3% above, and wall clock by up to 89%, rising with `k`. So the
  instruction metric is robust to access order — worth knowing — and blind to the
  cache effect that actually separates designs in time. **Still open:** genuine
  cold-cache (this is warm-but-scattered), and cycles on a quiet host.
- ~~**Transposed iteration**~~ — **done** for (C): 786 instructions/edge against
  forward iteration's 163 at `k` = 1, a factor of 4.8 falling to 2.1 at `k` = 16.
  That is the price of `mt` carrying structure only, and it makes
  incoming-edge-dominated traversal the shape this design serves worst. **Still
  open:** on the C side, `Tensor_ClearElements`,
  `Tensor_RemoveElements_Flat`, `Tensor_SetEdges`, row and column degree.
- **Wall clock and cycles.** Some runs shared the machine, so no claim in the
  paper is a latency claim. Re-running a quiet host would let that change.
- **Linux and CI.** Everything is macOS/arm64, with both sides built against
  GraphBLAS 10.3.1. The engines' relative standing on the CI architecture is
  unmeasured.
- **GraphBLAS 10.5.0, on the C side and at engine level.** Trunk moved from
  10.3.1 to 10.5.0 after these measurements (#2523), which regenerates every
  PreJIT kernel and changes the iso-build rule the paper's space model leans on.
  (C)'s half of the boundary fixture *has* been re-run against it: space and
  iteration reproduce to three digits (25.0 and 36.8 B/edge, 24.3 B/id, 163 and
  250 instr/edge), while point reads and promotion are dearer (inline 566 → 621,
  sentinel 2,502 → 2,938, promote 2,735 → 2,965), so the entry charge rises from
  1,936 to 2,317. What remains unmeasured is the C side at 10.5.0 and the whole
  engine-level multiplicity sweep, which is what any *ratio* would need.
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
