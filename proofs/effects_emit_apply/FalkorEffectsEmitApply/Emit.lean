import FalkorEffectsEmitApply.Basic

/-!
# The emitter: `gather_rows`, the record stream, and the gate in front of it

Models `graph/src/effects/v3/emit.rs` (`for_each_record` and its digests) and
the gate that decides whether it runs at all (`runtime/ops/commit.rs:108`,
`Pending::effects_count` at `runtime/pending.rs:1688`).
-/

namespace FalkorEA

/-! ## Attribute lookup and `gather_rows` (`emit.rs:923-951`) -/

/-- The value an entity holds for attribute `a`, or `Null` (a pad). -/
def lookupAttr : List (Nat × Val) → Nat → Val
  | [],           _ => .null
  | (k, v) :: as, a => if k = a then v else lookupAttr as a

/-- `gather_rows`' inner loop, as written: `j` walks `pairs` forward while
    `pairs[j].0 < attr_id`; a hit pushes the value and advances `j`, a miss
    pushes `Null` and leaves `j` where it is. `ps` is `pairs[j..]`. -/
def mergeRow : List Nat → List (Nat × Val) → List Val
  | [],      _  => []
  | a :: as, ps =>
    match ps.dropWhile (fun p => decide (p.1 < a)) with
    | (k, v) :: rest => if k = a then v :: mergeRow as rest
                        else .null :: mergeRow as ((k, v) :: rest)
    | []             => .null :: mergeRow as []

/-- `gather_rows` for one id: absent from the map → a row of pads. -/
def gatherOne (attrIds : List Nat) (pairs : Option (List (Nat × Val))) : List Val :=
  match pairs with
  | none    => attrIds.map (fun _ => Val.null)
  | some ps => mergeRow attrIds ps

def gatherRows (attrIds : List Nat) (look : Nat → Option (List (Nat × Val)))
    (ids : List Nat) : List Val :=
  ids.flatMap (fun i => gatherOne attrIds (look i))

theorem lookupAttr_dropWhile (a b : Nat) (hab : a ≤ b) :
    ∀ ps : List (Nat × Val),
      lookupAttr (ps.dropWhile (fun p => decide (p.1 < a))) b = lookupAttr ps b := by
  intro ps
  induction ps with
  | nil => rfl
  | cons p rest ih =>
    obtain ⟨k, v⟩ := p
    by_cases h : k < a
    · have hne : ¬ k = b := by omega
      simp [List.dropWhile, h, lookupAttr, hne, ih]
    · simp [List.dropWhile, h]

theorem dropWhile_head (a : Nat) : ∀ (ps : List (Nat × Val)) (k : Nat) (v : Val) (rest : List (Nat × Val)),
    ps.dropWhile (fun p => decide (p.1 < a)) = (k, v) :: rest → ¬ k < a := by
  intro ps
  induction ps with
  | nil => intro k v rest h; simp at h
  | cons p tl ih =>
    obtain ⟨k', v'⟩ := p
    intro k v rest h
    by_cases hk : k' < a
    · simp only [List.dropWhile, hk, decide_true] at h; exact ih k v rest h
    · simp only [List.dropWhile, hk, decide_false, List.cons.injEq, Prod.mk.injEq] at h
      rw [← h.1.1]; exact hk

theorem lookupAttr_absent (a : Nat) : ∀ l : List (Nat × Val),
    (∀ q ∈ l, a < q.1) → lookupAttr l a = .null := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons q tl ih =>
    obtain ⟨k, v⟩ := q
    intro h
    have hk : ¬ k = a := by have := h (k, v) (List.mem_cons_self ..); simp at this; omega
    simp only [lookupAttr, hk, ↓reduceIte]
    exact ih (fun q hq => h q (List.mem_cons_of_mem _ hq))

theorem pairwise_dropWhile {α} (R : α → α → Prop) (f : α → Bool) :
    ∀ l : List α, l.Pairwise R → (l.dropWhile f).Pairwise R := by
  intro l
  induction l with
  | nil => intro _; simp
  | cons x xs ih =>
    intro h
    simp only [List.dropWhile]
    cases f x
    · exact h
    · exact ih (List.pairwise_cons.mp h).2

/-- **`gather_rows` is the lookup it replaces.** For an entity whose staged
    pairs are strictly ascending by attribute id (what `Pending`'s sorted vec
    keeps) and a strictly ascending shape (built from those very vectors), the
    linear merge produces exactly `attr_ids.map(lookup)` — the quadratic
    per-cell search that `emit.rs:939-942` says it replaced. No cell is taken
    from the wrong attribute and none is skipped. -/
theorem mergeRow_eq_lookup :
    ∀ (as : List Nat) (ps : List (Nat × Val)),
      as.Pairwise (· < ·) → (ps.map Prod.fst).Pairwise (· < ·) →
      mergeRow as ps = as.map (lookupAttr ps) := by
  intro as
  induction as with
  | nil => intro ps _ _; rfl
  | cons a as ih =>
    intro ps has hps
    have hlt : ∀ b ∈ as, a < b := (List.pairwise_cons.mp has).1
    have has' := (List.pairwise_cons.mp has).2
    have hd : ((ps.dropWhile (fun p => decide (p.1 < a))).map Prod.fst).Pairwise (· < ·) := by
      rw [List.pairwise_map] at hps ⊢
      exact pairwise_dropWhile _ _ ps hps
    have key : ∀ b, a ≤ b →
        lookupAttr (ps.dropWhile (fun p => decide (p.1 < a))) b = lookupAttr ps b :=
      fun b hb => lookupAttr_dropWhile a b hb ps
    simp only [mergeRow, List.map_cons]
    have hhead := dropWhile_head a ps
    revert hd key hhead
    generalize ps.dropWhile (fun p => decide (p.1 < a)) = d
    intro hd key hhead
    cases d with
    | nil =>
      simp only [List.cons.injEq]
      refine ⟨by rw [← key a (Nat.le_refl a)]; try rfl, ?_⟩
      rw [ih [] has' (by simp)]
      refine List.map_congr_left (fun b hb => ?_)
      rw [← key b (Nat.le_of_lt (hlt b hb))]; try rfl
    | cons p rest =>
      obtain ⟨k, v⟩ := p
      have hrest : (rest.map Prod.fst).Pairwise (· < ·) :=
        (List.pairwise_cons.mp (by simpa using hd)).2
      by_cases hk : k = a
      · subst hk
        dsimp only
        rw [if_pos rfl]
        simp only [List.cons.injEq]
        refine ⟨by rw [← key k (Nat.le_refl k)]; simp [lookupAttr], ?_⟩
        rw [ih rest has' hrest]
        refine List.map_congr_left (fun b hb => ?_)
        have hb' := hlt b hb
        rw [← key b (Nat.le_of_lt hb')]
        have : ¬ k = b := by omega
        simp [lookupAttr, this]
      · dsimp only
        rw [if_neg hk]
        simp only [List.cons.injEq]
        have hka : a < k := by have := hhead k v rest rfl; omega
        have hrk : ∀ q ∈ rest, a < q.1 := by
          intro q hq
          have hd' : List.Pairwise (· < ·) (k :: rest.map Prod.fst) := by simpa using hd
          have : k < q.1 := (List.pairwise_cons.mp hd').1 q.1 (List.mem_map_of_mem hq)
          omega
        refine ⟨by rw [← key a (Nat.le_refl a)]; simp [lookupAttr, hk, lookupAttr_absent a rest hrk], ?_⟩
        rw [ih ((k, v) :: rest) has' hd]
        refine List.map_congr_left (fun b hb => ?_)
        rw [← key b (Nat.le_of_lt (hlt b hb))]

/-- A width-`n` shape yields `n` cells per id, so `rows.len() == ids.len() * n`
    — the shape check the decoder and `check_attr_shape` (`apply.rs:663`)
    apply to it. -/
theorem mergeRow_length : ∀ (as : List Nat) (ps : List (Nat × Val)),
    (mergeRow as ps).length = as.length := by
  intro as
  induction as with
  | nil => intro _; rfl
  | cons a as ih =>
    intro ps
    simp only [mergeRow]
    split
    · split <;> simp [ih]
    · simp [ih]

theorem gatherRows_length (attrIds : List Nat) (look : Nat → Option (List (Nat × Val))) :
    ∀ ids : List Nat, (gatherRows attrIds look ids).length = ids.length * attrIds.length := by
  intro ids
  induction ids with
  | nil => simp [gatherRows]
  | cons i is ih =>
    have h1 : (gatherOne attrIds (look i)).length = attrIds.length := by
      unfold gatherOne; cases look i <;> simp [mergeRow_length]
    simp only [gatherRows, List.flatMap_cons, List.length_append] at ih ⊢
    rw [h1, ih]; simp only [List.length_cons, Nat.add_mul, Nat.one_mul]; omega

/-- A counterexample kept as a regression: an **unsorted** staged vector makes
    the merge skip a cell the lookup would have found. The ascending invariant
    of `Pending`'s attribute vectors is load-bearing, not an optimisation. -/
example : mergeRow [1, 2] [(2, .int 20), (1, .int 10)] ≠ [1, 2].map (lookupAttr [(2, .int 20), (1, .int 10)]) := by
  decide

/-! ## The record stream (`for_each_record`, `emit.rs:325-374`)

    Records are abstracted to their opcode and the ids they name — enough to
    decide *whether* the stream is empty and what the replica's schema
    dictionaries end up as, which is where the gate bug lives. -/

inductive Rec where
  | addLabel (id : Nat) (name : Name)
  | addRelType (id : Nat) (name : Name)
  | addAttribute (id : Nat) (name : Name)
  | createNode (ids : List Nat) (labels : List Nat)
  | createEdge (ids : List Nat) (rel : Nat)
  | updateNode (ids : List Nat)
  | updateEdge (ids : List Nat)
  | setLabels (ids : List Nat) (labels : List Nat)
  | removeLabels (ids : List Nat) (labels : List Nat)
  | deleteEdge (ids : List Nat)
  | deleteNode (ids : List Nat) (labels : List Nat)
deriving Repr, DecidableEq

/-- The graph's schema dictionaries at commit time, and the baseline `Pending`
    took before the query ran (`SchemaBaseline`). -/
structure Schema where
  labels : List Name
  types  : List Name
  attrs  : List Name
deriving Repr, DecidableEq

/-- `Pending`, cut to the fields `effects_count` and `for_each_record` read. -/
structure Pending where
  createdNodes      : List Nat
  createdRelTypes   : List (Name × List Nat)   -- `created_rels_by_type`
  deletedNodes      : List Nat
  deletedRels       : List Nat
  newNodesAttrs     : List Nat                  -- keys of `new_nodes_attrs`
  existingNodesAttrs : List Nat
  newRelsAttrs      : List Nat
  existingRelsAttrs : List Nat
  setLabels         : List (Nat × List Nat)
  removeLabels      : List (Nat × List Nat)
  cancelledNodes    : List Nat
  cancelledRels     : List (Name × Nat)         -- (type name, id)
  baseline          : Schema
deriving Repr

/-- `Pending::effects_count` (`runtime/pending.rs:1688-1707`), term for term.
    Note what is **not** summed: `cancelled_nodes`, `cancelled_relationships`,
    and the schema registered since `baseline`. -/
def effectsCount (p : Pending) : Nat :=
  p.createdNodes.length + p.createdRelTypes.length + p.deletedNodes.length
  + p.deletedRels.length + p.newNodesAttrs.length + p.existingNodesAttrs.length
  + p.newRelsAttrs.length + p.existingRelsAttrs.length
  + (p.setLabels.map (fun q => q.2.length)).sum
  + (p.removeLabels.map (fun q => q.2.length)).sum

def suffixRecs (mk : Nat → Name → Rec) (base : Nat) : List Name → List Rec
  | []      => []
  | n :: ns => mk base n :: suffixRecs mk (base + 1) ns

/-- `emit_schema_additions` (`emit.rs:61-109`): every dictionary entry past the
    baseline, with its offset as id. -/
def schemaAdditions (g : Schema) (b : Schema) : List Rec :=
  suffixRecs .addLabel b.labels.length (g.labels.drop b.labels.length)
  ++ suffixRecs .addRelType b.types.length (g.types.drop b.types.length)
  ++ suffixRecs .addAttribute b.attrs.length (g.attrs.drop b.attrs.length)

def idxOf (l : List Name) (n : Name) : Option Nat :=
  match l.findIdx? (· == n) with
  | some i => some i
  | none   => none

/-- `digest_cancelled` (`emit.rs:397-497`), ids only. A cancelled edge of a type
    the graph never registered is skipped (`emit.rs:450-456`). -/
def digestCancelled (p : Pending) (g : Schema) : List Rec :=
  if p.cancelledNodes.isEmpty && p.cancelledRels.isEmpty then [] else
  let edges := p.cancelledRels.filterMap
    (fun (t, i) => (idxOf g.types t).map (fun r => (r, i)))
  (if p.cancelledNodes.isEmpty then [] else [Rec.createNode p.cancelledNodes []])
  ++ edges.map (fun (r, i) => Rec.createEdge [i] r)
  ++ edges.map (fun (_, i) => Rec.deleteEdge [i])
  ++ (if p.cancelledNodes.isEmpty then [] else [Rec.deleteNode p.cancelledNodes []])

/-- `for_each_record`, at the granularity of "which digests say something".
    Each digest is empty exactly when its source is (`digest_labels` also drops
    empty label vectors, and a created node's labels ride in its CREATE_NODE). -/
def forEachRecord (p : Pending) (g : Schema) : List Rec :=
  schemaAdditions g p.baseline
  ++ (if p.createdNodes.isEmpty then [] else [Rec.createNode p.createdNodes []])
  ++ (p.createdRelTypes.filter (fun q => !q.2.isEmpty)).filterMap
       (fun (t, ids) => (idxOf g.types t).map (fun r => Rec.createEdge ids r))
  ++ (if (p.existingNodesAttrs.filter (fun i => !p.deletedNodes.contains i)).isEmpty then []
      else [Rec.updateNode (p.existingNodesAttrs.filter (fun i => !p.deletedNodes.contains i))])
  ++ (if (p.existingRelsAttrs.filter (fun i => !p.deletedRels.contains i)).isEmpty then []
      else [Rec.updateEdge (p.existingRelsAttrs.filter (fun i => !p.deletedRels.contains i))])
  ++ ((p.setLabels.filter (fun q => !p.createdNodes.contains q.1 && !q.2.isEmpty)).map
       (fun q => Rec.setLabels [q.1] q.2))
  ++ ((p.removeLabels.filter (fun q => !q.2.isEmpty)).map (fun q => Rec.removeLabels [q.1] q.2))
  ++ (if p.deletedRels.isEmpty then [] else [Rec.deleteEdge p.deletedRels])
  ++ (if p.deletedNodes.isEmpty then [] else [Rec.deleteNode p.deletedNodes []])
  ++ digestCancelled p g

/-- `CommitOp` (`runtime/ops/commit.rs:108`): `if estimated > 0 { buf.build(..) }`. -/
def commitShips (p : Pending) (g : Schema) : List Rec :=
  if effectsCount p > 0 then forEachRecord p g else []

/-! ## What the gate drops -/

theorem sum_zero_nil : ∀ l : List (Nat × List Nat),
    (l.map (fun q => q.2.length)).sum = 0 → ∀ q ∈ l, q.2 = [] := by
  intro l
  induction l with
  | nil => intro _ q hq; simp at hq
  | cons x xs ih =>
    intro h q hq
    simp only [List.map_cons, List.sum_cons] at h
    rcases List.mem_cons.mp hq with rfl | hq'
    · exact List.eq_nil_of_length_eq_zero (by omega)
    · exact ih (by omega) q hq'

theorem filter_nil_of_sub {α} (p : α → Bool) (l : List α) (h : l = []) : l.filter p = [] := by
  subst h; rfl

/-- **The gate is complete for everything it counts, and blind to exactly two
    things.** When `effects_count` is zero, the stream the emitter *would* have
    written consists of the schema additions and the cancelled pairs, nothing
    else. So `commitShips` drops a non-empty payload precisely when the query
    registered a name or cancelled an entity and did nothing `effects_count`
    sees. -/
theorem gate_zero_stream (p : Pending) (g : Schema) (h : effectsCount p = 0) :
    forEachRecord p g = schemaAdditions g p.baseline ++ digestCancelled p g := by
  unfold effectsCount at h
  have h1 : p.createdNodes = [] := List.eq_nil_of_length_eq_zero (by omega)
  have h2 : p.createdRelTypes = [] := List.eq_nil_of_length_eq_zero (by omega)
  have h3 : p.deletedNodes = [] := List.eq_nil_of_length_eq_zero (by omega)
  have h4 : p.deletedRels = [] := List.eq_nil_of_length_eq_zero (by omega)
  have h6 : p.existingNodesAttrs = [] := List.eq_nil_of_length_eq_zero (by omega)
  have h8 : p.existingRelsAttrs = [] := List.eq_nil_of_length_eq_zero (by omega)
  have h9 := sum_zero_nil p.setLabels (by omega)
  have h10 := sum_zero_nil p.removeLabels (by omega)
  have hs : p.setLabels.filter (fun q => !p.createdNodes.contains q.1 && !q.2.isEmpty) = [] := by
    rw [List.filter_eq_nil_iff]; intro q hq; simp [h9 q hq]
  have hr : p.removeLabels.filter (fun q => !q.2.isEmpty) = [] := by
    rw [List.filter_eq_nil_iff]; intro q hq; simp [h10 q hq]
  simp [forEachRecord, h1, h2, h3, h4, h6, h8, hr]
  exact fun a b hm => h9 (a, b) hm

/-- Soundness of the gate, stated the way a fix would need it: the commit ships
    what the emitter would write **iff** the count is positive or the would-be
    stream is empty. -/
theorem commitShips_eq_iff (p : Pending) (g : Schema) :
    commitShips p g = forEachRecord p g ↔ effectsCount p > 0 ∨ forEachRecord p g = [] := by
  unfold commitShips
  by_cases h : effectsCount p > 0
  · simp [h]
  · simp only [h, ↓reduceIte, false_or]
    exact ⟨fun e => e.symm, fun e => e.symm⟩

/-! ### Counterexamples (both reproduced against the real code)

    `graph/tests/lean_effects_emit_apply.rs`:
    `fully_cancelled_write_ships_nothing_and_next_payload_is_refused` and
    `schema_only_write_ships_nothing_and_next_payload_is_refused`. -/

/-- `CREATE (a:A {p: 1}) DELETE a` on an empty graph: label `A` and attribute
    `p` are registered, node 0 is reserved and cancelled; `effects_count = 0`. -/
def pCancelled : Pending :=
  { createdNodes := [], createdRelTypes := [], deletedNodes := [], deletedRels := [],
    newNodesAttrs := [], existingNodesAttrs := [], newRelsAttrs := [], existingRelsAttrs := [],
    setLabels := [], removeLabels := [], cancelledNodes := [0], cancelledRels := [],
    baseline := ⟨[], [], []⟩ }

def gCancelled : Schema := ⟨["A"], [], ["p"]⟩

theorem gate_drops_cancelled :
    effectsCount pCancelled = 0 ∧
    forEachRecord pCancelled gCancelled =
      [.addLabel 0 "A", .addAttribute 0 "p", .createNode [0] [], .deleteNode [0] []] ∧
    commitShips pCancelled gCancelled = [] := by
  refine ⟨rfl, ?_, rfl⟩
  decide

/-- `OPTIONAL MATCH (n:Nope) SET n:L6`: `L6` registered, nothing else. -/
def pSchemaOnly : Pending := { pCancelled with cancelledNodes := [] }
def gSchemaOnly : Schema := ⟨["L6"], [], []⟩

theorem gate_drops_schema_only :
    effectsCount pSchemaOnly = 0 ∧
    forEachRecord pSchemaOnly gSchemaOnly = [.addLabel 0 "L6"] ∧
    commitShips pSchemaOnly gSchemaOnly = [] := by
  refine ⟨rfl, ?_, rfl⟩
  decide

/-! ### Why a dropped payload is a divergence and not a no-op

    The replica checks every `ADD_SCHEMA` id against the id its own dictionary
    assigns (`apply_add_schema`, `apply.rs:496`; `verify_id`, `apply.rs:516`).
    One dropped registration shifts every later id by one. -/

/-- `get_label_id_mut` then `verify_id`: get-or-create, and the id must match. -/
def applyAddName (dict : List Name) (id : Nat) (n : Name) : Option (List Name) :=
  match idxOf dict n with
  | some i => if i = id then some dict else none
  | none   => if dict.length = id then some (dict ++ [n]) else none

/-- The next query, `CREATE (:K6)`, ships `ADD_SCHEMA 1 "K6"` (the master's
    dictionary is `[L6, K6]`); the replica never saw `L6` and assigns `0`. -/
theorem next_payload_refused :
    applyAddName [] 1 "K6" = none ∧ applyAddName ["L6"] 1 "K6" = some ["L6", "K6"] := by
  decide

/-- A cancelled edge of an unregistered type, whose attribute name is new
    (`CREATE (a)-[:T1 {p4: 1}]->(b) DELETE a, b`): the master's attribute
    dictionary grows by `p4`, nothing ships, and the next payload that names no
    attribute is *accepted* — a silent divergence
    (`cancelled_edge_attribute_is_a_silent_dictionary_divergence`). Here even an
    un-gated emitter would say nothing about `T1`'s edges, since the type has no
    id (`emit.rs:450-456`), but it would still announce `p4`. -/
theorem cancelled_edge_attr_dropped :
    let p : Pending := { pCancelled with cancelledNodes := [0, 1], cancelledRels := [("T1", 0)] }
    let g : Schema := ⟨[], [], ["p4"]⟩
    effectsCount p = 0 ∧ commitShips p g = [] ∧
    forEachRecord p g = [.addAttribute 0 "p4", .createNode [0, 1] [], .deleteNode [0, 1] []] := by
  refine ⟨rfl, rfl, ?_⟩
  decide

end FalkorEA
