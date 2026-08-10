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

## 1. Seal the counter with the structure it caches

**Status:** open. A prototype of the *weaker* repair (a run-time check) exists and
is described below as the road not taken.

**Problem.** `multi_count` is a cache of a structural property of `me` — by
promotion completeness, `ε(p) = MULTI` exactly when row `κ(p)` of `me` is
non-empty. Seven call sites add or remove a row of `me`; three maintain the
counter. Nothing makes the fourth-to-seventh wrong today, but nothing stops an
eighth from forgetting. The failure is quiet and remote: `Tensor::edge_count`
subtracts the counter, so drift surfaces as a wrong edge count in some later
query, and an underflow surfaces as a value near `2^64` — which reaches
`Vec::with_capacity` in `algo_procedures.rs`, i.e. an allocation abort rather
than a wrong answer.

**Design.** Move `me` and `multi_count` into one type — call it `EdgeOverflow` —
whose only mutators are the transitions, so the two cannot move independently:

```rust
struct EdgeOverflow {
    me: VersionedMatrix<bool>,
    multi: u64,
}

impl EdgeOverflow {
    /// A pair gains its second edge: both ids move in, `multi` goes up.
    fn promote(&mut self, key: u64, first: u64, second: u64);
    /// A multi pair gains a further edge: `multi` unchanged.
    fn add(&mut self, key: u64, id: u64);
    /// A multi pair loses an id. Returns the survivor when this demotes it,
    /// in which case `multi` goes down.
    fn remove(&mut self, key: u64, id: u64) -> Option<u64>;
    /// Read-only projections the rest of the tensor needs.
    fn ids(&self, key: u64) -> impl Iterator<Item = u64>;
    fn multi(&self) -> u64;
}
```

The counter stops being independent state and becomes a private field only four
functions can touch. This is the move the delta layers already made: `Delta<T>`
bundles a layer with its approximate count and its fold latch precisely so no
caller has to remember to update them together.

**Why it is available now.** Every mutation of `me`'s rows is inside
`tensor.rs`, and the one external reader — `algo_procedures.rs`, which builds a
row-reduction over `me.m()`, `me.dp()`, `me.dm()` — is strictly read-only. So
the encapsulation needs no changes outside the module beyond keeping that
read-only projection available.

**Acceptance.** `multi_count` no longer appears outside the new type;
`cargo test -p graph` and the flow suite unchanged; the multiplicity sweep
unchanged within noise (this is a refactor, not an optimisation).

**Risks.** The promote path is entangled with the batch map in
`set_all_from_slices`, which retroactively promotes a pair whose second edge
arrives later in the same batch — the new type has to express that without
leaking its internals back out. Keep checked arithmetic in the cardinality
identity regardless: encapsulation defends against a forgetful call site, not
against a corrupt decoded blob or a memory error.

**Effort.** A day, mostly in `set_all_from_slices` and `remove_all`.

**The road not taken.** A commit-time check that recovers the count from `me`
and repairs the counter. It works, and it measured free at the commit level, but
it only makes the drift loud instead of impossible, and it carries two
subtleties that the encapsulation removes entirely: the recovery is
`O(multi)` rather than a metadata read whenever `me` carries live deltas
(5.2 ms at 200,000 pairs, so the check has to skip that case), and
`GxB_rowIterator_kount` is only an *upper* bound on non-empty rows, so it can
serve as a screen but never as the authority — an over-counting screen would
"repair" a correct counter into a wrong one. Prefer the type.

---

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
piece of per-transaction state, which is exactly the kind of state item 1 exists
to keep honest, so build it inside `EdgeOverflow`.

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
dimension, so print `GxB_SPARSITY_STATUS` and the hyper-switch for every matrix
on the read path at both a low-state and a high-state size and diff them. Failing
that, differential instruction counting at the boundary: the Rust `Tensor`
measurement tests in `tensor_cost_bench.rs` can reproduce a point read without a
query pipeline around it, which is the cheapest way to tell whether the extra
cost is even inside the tensor.

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
- **Fan-out beyond `k = 2` at the data-structure boundary** for reads and
  iteration. Memory is covered to `k = 16`; reads are not.
- **Cold-cache and random access order.** Every point read measured walks pairs
  sequentially and warm. Real traversals do neither.
- **Transposed iteration**, and on the C side `Tensor_ClearElements`,
  `Tensor_RemoveElements_Flat`, `Tensor_SetEdges`, row and column degree.
- **Wall clock and cycles.** Some runs shared the machine, so no claim in the
  paper is a latency claim. Re-running a quiet host would let that change.
- **Linux and CI.** Everything is macOS/arm64, with the C side pinned to
  GraphBLAS 10.3.1. The engines' relative standing on the CI architecture is
  unmeasured.
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
