/-
# MERGE — `graph/src/runtime/ops/merge.rs`

Shared definitions for the MERGE model: the openCypher per-row reference (`refMerge`), the
pattern cache keyed by `(operator idx, hash)` (`createFallback`, `Clause.op`, #3050),
`phaseB` (`drain_pending`) and the cache invariants `J`/`G`. The current `MergeOp::next` is
`MergeEager.lean`.

The model is one node pattern `MERGE (n:L {k: key(row)})`: a node is `(id, k)`, the pattern
hash is the key (the FxHash collision bug is known, issue 2913, and assumed away),
`ON CREATE`/`ON MATCH SET n.k = e(row)` are optional key updates.
-/

namespace Merge

variable {α : Type}

structure Node where
  id : Nat
  k : Nat
  deriving DecidableEq, Repr

abbrev Graph := List Node

structure Clause (α : Type) where
  /-- the variable id of `n` (binder-assigned; distinct clauses ⇒ distinct ids,
  except UNION branches, which restart numbering) -/
  var : Nat
  /-- the plan index of this MERGE operator (`MergeOp::idx`); the cache key since #3050 -/
  op : Nat
  key : α → Nat
  onCreate : Option (α → Nat)
  onMatch : Option (α → Nat)

def matchesK (g : Graph) (k : Nat) : List Node := g.filter (fun n => n.k == k)

def setKey (g : Graph) (id v : Nat) : Graph :=
  g.map (fun n => if n.id == id then { n with k := v } else n)

def applySet (s : Option (α → Nat)) (r : α) (g : Graph) (id : Nat) : Graph :=
  match s with
  | none => g
  | some f => setKey g id (f r)

/-! ## Reference (openCypher, per row) -/

structure RState where
  g : Graph
  next : Nat
  out : List (α × Option Nat)

def refRow (cl : Clause α) (st : RState (α := α)) (r : α) : RState (α := α) :=
  match matchesK st.g (cl.key r) with
  | [] =>
    let g := st.g ++ [⟨st.next, cl.key r⟩]
    ⟨applySet cl.onCreate r g st.next, st.next + 1, st.out ++ [(r, some st.next)]⟩
  | ms =>
    ⟨ms.foldl (fun g m => applySet cl.onMatch r g m.id) st.g, st.next,
      st.out ++ ms.map (fun m => (r, some m.id))⟩

def refMerge (cl : Clause α) (st : RState (α := α)) (rows : List α) : RState (α := α) :=
  rows.foldl (refRow cl) st

/-! ## Rust -/

/-- `merge_pattern_cache`: key `(operator idx, pattern hash)` ↦ cached `(var id, entity id)` binding. -/
abbrev Cache := List ((Nat × Nat) × (Nat × Nat))

/-- Since #3050 the key is `(operator idx, pattern hash)` (`merge.rs`). -/
def lookup (c : Cache) (h : Nat × Nat) : Option (Nat × Nat) := (c.find? (fun e => e.1 == h)).map Prod.snd

structure SState where
  g : Graph
  next : Nat
  cache : Cache
  out : List (α × Option Nat)

/-- Binding of the clause's own variable after inserting the cached pairs by
*their* variable ids (`vars.insert_by_id(*id, …)`, `merge.rs`). -/
def bindCached (cl : Clause α) (cv : Nat × Nat) : Option Nat :=
  if cv.1 == cl.var then some cv.2 else none

/-- `do_create_fallback` (`merge.rs`). -/
def createFallback (cl : Clause α) (st : SState (α := α)) (r : α) :
    SState (α := α) × (α × Option Nat) :=
  match lookup st.cache (cl.op, cl.key r) with
  | some cv =>                                                      -- 122-135
    ({ st with g := applySet cl.onMatch r st.g cv.2 }, (r, bindCached cl cv))
  | none =>                                                         -- 136-177
    let g := st.g ++ [⟨st.next, cl.key r⟩]
    ({ g := applySet cl.onCreate r g st.next, next := st.next + 1,
       cache := ((cl.op, cl.key r), (cl.var, st.next)) :: st.cache, out := st.out },
     (r, some st.next))

/-- Phase B (`drain_pending`, `merge.rs`; since #3067 called per row, see `MergeEager.rowMain`): the queued matchesK, row by row. -/
def phaseB (cl : Clause α) : SState (α := α) → List (α × List Node) → SState (α := α)
  | st, [] => st
  | st, (r, ms) :: rest =>
    phaseB cl { st with g := ms.foldl (fun g m => applySet cl.onMatch r g m.id) st.g,
                        out := st.out ++ ms.map (fun m => (r, some m.id)) } rest

/-! ## Without SET clauses -/

/-- No ON CREATE / ON MATCH SET on the merged property. -/
def NoSet (cl : Clause α) : Prop := cl.onCreate = none ∧ cl.onMatch = none

theorem applySet_none (r : α) (g : Graph) (id : Nat) : applySet none r g id = g := rfl

theorem foldl_applySet_none (r : α) (ms : List Node) (g : Graph) :
    ms.foldl (fun g m => applySet (none : Option (α → Nat)) r g m.id) g = g := by
  induction ms generalizing g with
  | nil => rfl
  | cons m ms ih => simp only [List.foldl_cons, applySet_none]; exact ih g

theorem foldl_const_self {β γ : Type} (g : β) (ms : List γ) : ms.foldl (fun g _ => g) g = g := by
  induction ms with
  | nil => rfl
  | cons _ _ ih => simpa using ih

theorem matches_append (g h : Graph) (k : Nat) : matchesK (g ++ h) k = matchesK g k ++ matchesK h k := by
  simp [matchesK]

theorem matches_single (id k k' : Nat) : matchesK [⟨id, k'⟩] k = if k' = k then [⟨id, k'⟩] else [] := by
  by_cases h : k' = k <;> simp [matchesK, h]

/-- Cache entries point at nodes of the graph with that key, and carry this
clause's variable. -/
def CacheOK (cl : Clause α) (g : Graph) (c : Cache) : Prop :=
  ∀ h v id, lookup c (cl.op, h) = some (v, id) → v = cl.var ∧ ⟨id, h⟩ ∈ g

/-- Within-batch invariant.  `g0` is the batch-start graph; `created` the nodes
created so far in this batch (keys absent from `g0`, at most one per key), and
the cache knows exactly those for keys absent from `g0`. -/
structure J (cl : Clause α) (g0 : Graph) (g : Graph) (c : Cache) : Prop where
  ext : ∃ created, g = g0 ++ created ∧ ∀ n ∈ created, matchesK g0 n.k = []
  cache_ok : CacheOK cl g c
  fresh : ∀ k, matchesK g0 k = [] →
    (lookup c (cl.op, k) = none → matchesK g k = []) ∧
    (∀ v id, lookup c (cl.op, k) = some (v, id) → matchesK g k = [⟨id, k⟩])

theorem lookup_cons_self (c : Cache) (h : Nat × Nat) (v id : Nat) :
    lookup ((h, (v, id)) :: c) h = some (v, id) := by
  simp [lookup]

theorem lookup_cons_other (c : Cache) (h h' : Nat × Nat) (e : Nat) (e' : Nat) (hne : h' ≠ h) :
    lookup ((h, (e, e')) :: c) h' = lookup c h' := by
  have : (h == h') = false := by simp; exact fun e => hne e.symm
  simp [lookup, this]

theorem mem_matches {g : Graph} {k : Nat} {n : Node} : n ∈ matchesK g k ↔ n ∈ g ∧ n.k = k := by
  simp [matchesK]

theorem phaseB_noSet (cl : Clause α) (hs : NoSet cl) :
    ∀ (q : List (α × List Node)) (st : SState (α := α)),
      phaseB cl st q = { st with out := st.out ++ q.flatMap (fun q => q.2.map (fun m => (q.1, some m.id))) }
  | [], st => by simp [phaseB]
  | (r, ms) :: rest, st => by
    simp only [phaseB, hs.2, foldl_applySet_none]
    rw [phaseB_noSet cl hs rest]
    simp [List.append_assoc]

/-- Global invariant between batches. -/
def G (cl : Clause α) (st : SState (α := α)) : Prop :=
  CacheOK cl st.g st.cache ∧ ∀ n ∈ st.g, n.id < st.next

theorem J_init (cl : Clause α) (st : SState (α := α)) (h : G cl st) : J cl st.g st.g st.cache := by
  refine ⟨⟨[], by simp, by simp⟩, h.1, ?_⟩
  intro k hk
  refine ⟨fun _ => hk, fun v id hl => ?_⟩
  have := (h.1 k v id hl).2
  have : (⟨id, k⟩ : Node) ∈ matchesK st.g k := mem_matches.mpr ⟨this, rfl⟩
  rw [hk] at this; cases this

/-- `UNWIND … AS x MERGE (n:L {v: CASE x WHEN 1 THEN 1 ELSE 2 END}) ON CREATE SET n.v = 2`. -/
def clCase : Clause Nat := ⟨0, 0, fun x => if x = 1 then 1 else 2, some (fun _ => 2), none⟩

def st0 : SState (α := Nat) := ⟨[], 0, [], []⟩

/-! ### The all-bound shortcut -/

/-- A relationship of the MERGE pattern, as `MergeOp::next` sees it: named (`!alias.is_unnamed()` since #3303; before it a
`_anon_*` name), already bound in the input row, exposed by a path variable. -/
structure RelShape where
  named : Bool
  bound : Bool
  inPath : Bool

/-- The shortcut condition on main (#3055): every relationship is bound, or anonymous and
not exposed by a path variable. -/
def allBoundMain (nodesBound : Bool) (rels : List RelShape) : Bool :=
  nodesBound && rels.all (fun r => r.bound || (!r.named && !r.inPath))

/-- `MATCH (a),(b) MERGE p = (a)-[:R]->(b)`: both nodes bound, one anonymous relationship
in the path. -/
def pathRels : List RelShape := [⟨false, false, true⟩]

end Merge
