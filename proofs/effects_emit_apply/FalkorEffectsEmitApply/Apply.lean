import FalkorEffectsEmitApply.Basic

/-!
# The apply side: which records check liveness, and what a missing check costs

Models the node-facing arms of `apply_record` (`graph/src/effects/v3/apply.rs:164-557`)
over an abstract replica: a node id space with a recycle bin, per-node labels
and properties, and edges as endpoint pairs. The question is the one a hostile
or divergent `GRAPH.EFFECT` asks: does applying a record the checks accept keep
the graph *well formed* — nothing hangs off a dead id, and every edge's
endpoints are live?
-/

namespace FalkorEA

structure RG where
  bound  : Nat
  free   : Nat → Bool
  labels : Nat → List Nat
  props  : Nat → List (Nat × Val)
  edges  : List (Nat × Nat)

def RG.live (σ : RG) (i : Nat) : Prop := i < σ.bound ∧ σ.free i = false

instance (σ : RG) (i : Nat) : Decidable (σ.live i) := by unfold RG.live; infer_instance

/-- Well formed: the bin is below the boundary, a dead id carries nothing, and
    every edge's endpoints are live. -/
structure RG.WF (σ : RG) : Prop where
  bin   : ∀ i, σ.free i = true → i < σ.bound
  dead  : ∀ i, ¬ σ.live i → σ.labels i = [] ∧ σ.props i = []
  ends  : ∀ e ∈ σ.edges, σ.live e.1 ∧ σ.live e.2

/-! ## The arms as Rust writes them -/

/-- `Graph::create_nodes` for one id (`apply.rs:214-240`, since #3446 the label write follows `check_attr_shape`; checked by
    `IdSpace::create`): a recycled id leaves the bin, a fresh id moves
    the boundary. It sets labels and does **not** reset properties — a recycled
    row keeps whatever the store holds for it (observed: `hostile_update_node_on_dead_id`). -/
def createOne (σ : RG) (id : Nat) (lbls : List Nat) : RG :=
  { σ with
      bound  := if σ.free id then σ.bound else max σ.bound (id + 1)
      free   := fun i => if i = id then false else σ.free i
      labels := fun i => if i = id then lbls else σ.labels i }

/-! ### #3446: labels written a chunk at a time

Since #3446 (`543357f1e`) the `CREATE_NODE` arm no longer collects the whole `IdList` into a
`Vec<u64>` first (a few bytes of range-compressed ids name billions of nodes). `check_attr_shape`
runs on the id *count*; with attributes the ids are materialised once (bounded by the payload: the
shape check made `rows.len() = count * width`); with labels but no attributes the labels are
written `NODE_LABEL_CHUNK = 2^16` ids per call (`apply.rs:268-288`), and with neither the ids are
never expanded. `chunkLoop` is that `loop { chunk = remaining.take(N); if empty { break }; write }`;
`chunkLoop_eq_write` says the chunking changes nothing: for any chunk size `n > 0` the result is
the one-shot write of the whole list (the shape `createOne` and `wf_createOne` reason about). -/

/-- The one-shot label write of `set_node_labels_product(&ids, &labels, ..)`: every listed id gets
exactly `lbls` (`createOne`'s `labels` field, over a list of ids). -/
def writeLabels (σ : RG) (ids : List Nat) (lbls : List Nat) : RG :=
  { σ with labels := fun i => if ids.contains i then lbls else σ.labels i }

/-- `apply.rs:277-287`: `chunk.clear(); chunk.extend(remaining.by_ref().take(N)); if chunk.is_empty()
{ break } write(chunk)`; `fuel` stands for "the loop terminates" (each round drains `min N |remaining|`
ids). -/
def chunkLoop (n : Nat) (lbls : List Nat) : Nat → List Nat → RG → RG
  | 0, _, σ => σ
  | fuel + 1, rest, σ =>
    if rest.take n = [] then σ
    else chunkLoop n lbls fuel (rest.drop n) (writeLabels σ (rest.take n) lbls)

theorem writeLabels_append (σ : RG) (a b lbls : List Nat) :
    writeLabels (writeLabels σ a lbls) b lbls = writeLabels σ (a ++ b) lbls := by
  simp only [writeLabels, List.contains_append, Bool.or_eq_true]
  congr 1
  funext i
  by_cases hb : i ∈ b <;> by_cases ha : i ∈ a <;> simp [hb, ha]

/-- Writing a list in chunks of `n > 0` is writing it at once, given enough fuel (`|ids| + 1`
rounds: every round drains at least one id). Holds for every `σ`, `ids` and label list. -/
theorem chunkLoop_eq_write (n : Nat) (hn : 0 < n) (lbls : List Nat) :
    ∀ (fuel : Nat) (ids : List Nat) (σ : RG), ids.length < fuel →
      chunkLoop n lbls fuel ids σ = writeLabels σ ids lbls
  | 0, ids, σ, h => by omega
  | fuel + 1, ids, σ, h => by
    simp only [chunkLoop]
    by_cases he : ids.take n = []
    · have hnil : ids = [] := by
        rcases List.take_eq_nil_iff.mp he with h' | h'
        · omega
        · exact h'
      subst hnil
      simp [writeLabels]
    · simp only [he, if_false]
      have hpos : 0 < ids.length := by
        cases ids with
        | nil => simp at he
        | cons a t => simp
      have hlen : (ids.drop n).length < fuel := by
        simp only [List.length_drop]; omega
      rw [chunkLoop_eq_write n hn lbls fuel (ids.drop n) _ hlen, writeLabels_append,
        List.take_append_drop]

/-- The loop is fed `ids.length + 1` rounds at most; the real loop has no fuel and stops on the first
empty chunk, which `chunkLoop_eq_write` shows it reaches. A record with no labels writes nothing, and
with `lbls = []` the one-shot write is `createOne`'s own (`labels := lbls`). -/
theorem chunkLoop_labels (n : Nat) (hn : 0 < n) (lbls ids : List Nat) (σ : RG) :
    (chunkLoop n lbls (ids.length + 1) ids σ).labels = fun i => if ids.contains i then lbls else σ.labels i := by
  rw [chunkLoop_eq_write n hn lbls _ ids σ (by omega)]; rfl

/-- `UPDATE_NODE`'s store write (`apply.rs:346-358`). Before #3022 the arm ran
    `check_attr_shape` and `checked_label_ids` only — no liveness check — then
    `set_nodes_attributes_rows_of_labels` writes the store for every id. -/
def updateNode (σ : RG) (ids : List Nat) (row : List (Nat × Val)) : RG :=
  { σ with props := fun i => if ids.contains i then row ++ σ.props i else σ.props i }

/-- `SET_LABELS`'s store write (`apply.rs:380-385`); before #3022 guarded by `checked_label_ids` only. -/
def setLabels (σ : RG) (ids : List Nat) (lbls : List Nat) : RG :=
  { σ with labels := fun i => if ids.contains i then lbls ++ σ.labels i else σ.labels i }

/-- `DELETE_NODE` (`apply.rs:406-427`): `delete_nodes` refuses a recycled or
    never-created id (`okDelete`) and clears the node's labels and properties.
    It does not cascade to incident edges — on a real payload the master's
    `DELETE_EDGE` records came first. Before #3022 nothing checked that there are
    none; now the arm refuses with `NodeHasRelationships` (`Liveness.lean`). -/
def deleteNode (σ : RG) (ids : List Nat) : RG :=
  { σ with
      free   := fun i => if ids.contains i then true else σ.free i
      labels := fun i => if ids.contains i then [] else σ.labels i
      props  := fun i => if ids.contains i then [] else σ.props i }

/-- The checks `delete_nodes` makes: every id live. -/
def okDelete (σ : RG) (ids : List Nat) : Prop := ∀ i ∈ ids, σ.live i

/-! ## What the checks that are there buy -/

theorem contains_iff (ids : List Nat) (i : Nat) : ids.contains i = true ↔ i ∈ ids := by
  simp

/-- Creating a recycled or next-fresh id keeps the graph well formed. -/
theorem wf_createOne (σ : RG) (h : σ.WF) (id : Nat) (lbls : List Nat)
    (hid : σ.free id = true ∨ id = σ.bound) : (createOne σ id lbls).WF := by
  have hlive : ∀ i, σ.live i → (createOne σ id lbls).live i := by
    intro i ⟨hb, hf⟩
    refine ⟨?_, ?_⟩
    · simp only [createOne]; split <;> omega
    · simp only [createOne]; split <;> simp_all
  refine ⟨?_, ?_, ?_⟩
  · intro i hi
    simp only [createOne] at hi ⊢
    by_cases e : i = id
    · simp [e] at hi
    · simp only [e, ↓reduceIte] at hi
      have := h.bin i hi
      split <;> omega
  · intro i hi
    by_cases e : i = id
    · subst e
      exfalso; apply hi
      refine ⟨?_, by simp [createOne]⟩
      simp only [createOne]
      rcases hid with hf | hb
      · simp [hf]; exact h.bin i hf
      · split
        · rename_i hf; have := h.bin i hf; omega
        · omega
    · have hd : ¬ σ.live i := fun hl => hi (hlive i hl)
      have := h.dead i hd
      simp only [createOne, e, ↓reduceIte]; exact this
  · intro e he
    have := h.ends e he
    exact ⟨hlive _ this.1, hlive _ this.2⟩

/-- Deleting live nodes that no edge touches keeps the graph well formed. -/
theorem wf_deleteNode (σ : RG) (h : σ.WF) (ids : List Nat)
    (hok : okDelete σ ids) (hno : ∀ e ∈ σ.edges, e.1 ∉ ids ∧ e.2 ∉ ids) :
    (deleteNode σ ids).WF := by
  refine ⟨?_, ?_, ?_⟩
  · intro i hi
    simp only [deleteNode] at hi ⊢
    by_cases e : ids.contains i = true
    · exact (hok i ((contains_iff ids i).mp e)).1
    · simp only [e] at hi; exact h.bin i (by simpa using hi)
  · intro i hi
    by_cases e : ids.contains i = true
    · simp [deleteNode, (contains_iff ids i).mp e]
    · have hd : ¬ σ.live i := by
        intro hl; apply hi
        exact ⟨hl.1, by simp only [deleteNode]; simp only [e]; simpa using hl.2⟩
      simp only [deleteNode]
      simp only [e]
      simpa using h.dead i hd
  · intro e he
    obtain ⟨h1, h2⟩ := h.ends e he
    obtain ⟨n1, n2⟩ := hno e he
    have c1 : ids.contains e.1 = false := by simpa using n1
    have c2 : ids.contains e.2 = false := by simpa using n2
    exact ⟨⟨h1.1, by simp [deleteNode, n1, h1.2]⟩, ⟨h2.1, by simp [deleteNode, n2, h2.2]⟩⟩

/-- **The fix, proved**: an `UPDATE_NODE` whose ids are all live (the check the
    `CREATE_NODE`/`DELETE_NODE` arms already route through `IdSpace`) keeps the
    graph well formed. -/
theorem wf_updateNode_live (σ : RG) (h : σ.WF) (ids : List Nat) (row : List (Nat × Val))
    (hl : ∀ i ∈ ids, σ.live i) : (updateNode σ ids row).WF := by
  refine ⟨h.bin, ?_, h.ends⟩
  intro i hi
  have hd := h.dead i hi
  have hn : i ∉ ids := fun m => hi (hl i m)
  simp [updateNode, hn, hd]

theorem wf_setLabels_live (σ : RG) (h : σ.WF) (ids lbls : List Nat)
    (hl : ∀ i ∈ ids, σ.live i) : (setLabels σ ids lbls).WF := by
  refine ⟨h.bin, ?_, h.ends⟩
  intro i hi
  have hd := h.dead i hi
  have hn : i ∉ ids := fun m => hi (hl i m)
  simp [setLabels, hn, hd]

/-! ## Historical counterexamples: the arms without a liveness check

**Fixed by #3022 (`18fc277b9`).** Before it, `UPDATE_NODE`, `SET_LABELS` and
`DELETE_NODE` reached the store writes above with no liveness check. They now
`require_live` first (and `DELETE_NODE` refuses a node with relationships):
`Liveness.lean` models the checked arms, refuses each input below
(`updates_on_dead_refused`, `delete_with_edges_refused`) and proves the whole
buffer keeps the replica well formed (`apply_wf`).

    The seeded replica of the Rust repros (`seeded()` in
    `graph/tests/lean_effects_emit_apply.rs`): nodes 0, 1, 2 live, node 3 in the
    bin, edge `0 -> 1`. -/

def σ0 : RG :=
  { bound := 4, free := fun i => i = 3, labels := fun _ => [], props := fun _ => [],
    edges := [(0, 1)] }

theorem σ0_wf : σ0.WF := by
  refine ⟨?_, ?_, ?_⟩
  · intro i hi; simp [σ0] at hi; subst hi; decide
  · intro i _; exact ⟨rfl, rfl⟩
  · intro e he; simp [σ0] at he; subst he; decide

end FalkorEA
