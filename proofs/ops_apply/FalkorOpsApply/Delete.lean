/-
DELETE (graph/src/runtime/ops/delete.rs).

| here | there |
| --- | --- |
| `attrsToMap`               | `node_attrs_to_map` (delete.rs:38), `rel_attrs_to_map` (:49) |
| `snapshotRequired`         | `snapshot_required` (delete.rs:69) |
| `DeleteSt.new`             | `DeleteOp::new` (delete.rs:93) |
| `deleteNext`               | `DeleteOp::next` (delete.rs:113) |
| `deleteBatch`, `scan`      | `Runtime::delete_batch` (delete.rs:127) |
| `delNodesBulk`             | `delete_nodes_bulk` (delete.rs:191) |
| `delRelsBulk`              | `delete_relationships_bulk` (delete.rs:297) |
| `delEntity`                | `Runtime::delete_entity` (delete.rs:393) |

State = the id sets the classification reads and writes: pending `created_nodes` (`pc`),
pending `deleted_nodes` (`pd`), the graph's deleted/recycled node set (`gd`, which
`delete_pending_node` extends through `Graph::cancel_node_id` since #2846), and the same
three for relationships. This file is the model with every id-space cancel accepted;
DeleteCancel.lean has the fallible versions and proves them equal to these on success. Snapshot records
(`deleted_nodes`/`deleted_relationships` maps) only serve later reads and are not modelled.
-/
namespace Delete

abbrev S := Nat → Bool
def setS (s : S) (k : Nat) : S := fun j => j = k || s j
def clrS (s : S) (k : Nat) : S := fun j => j ≠ k && s j

structure St where
  pc : S
  pd : S
  gd : S
  rc : S
  rd : S
  grd : S

/-- A node is gone once pending-deleted, or recycled and not pending-created. -/
def goneN (st : St) (id : Nat) : Bool := st.pd id || (st.gd id && !st.pc id)
def goneR (st : St) (id : Nat) : Bool := st.rd id || (st.grd id && !st.rc id)

inductive DV where
  | node (n : Nat)
  | rel (n : Nat)
  | path (vs : List DV)
  | null
  | other

def mismatch : String := "Delete type mismatch, expecting either Node or Relationship."

/-- delete.rs:398-448 (node), :449-462 (relationship), :463-467 (path), :468 (null), :469 (error).
A pending-created node is un-created and its id recycled (`delete_pending_node`, which
calls `cancel_node_id`); a committed live node is pending-deleted. -/
def delNode (st : St) (id : Nat) : St :=
  if st.pd id then st
  else if st.pc id then { st with pc := clrS st.pc id, gd := setS st.gd id }
  else if !st.gd id then { st with pd := setS st.pd id }
  else st

def delRel (st : St) (id : Nat) : St :=
  if st.rd id then st
  else if st.rc id || !st.grd id then { st with rd := setS st.rd id }
  else st

mutual
def delEntity : DV → St → Except String St
  | .node id, st => .ok (delNode st id)
  | .rel id, st => .ok (delRel st id)
  | .path vs, st => delEntityL vs st
  | .null, st => .ok st
  | .other, _ => .error mismatch
def delEntityL : List DV → St → Except String St
  | [], st => .ok st
  | v :: vs, st => do let st' ← delEntity v st; delEntityL vs st'
end

theorem delNode_gone (st : St) (id : Nat) : goneN (delNode st id) id = true := by
  unfold delNode goneN
  by_cases h1 : st.pd id
  · simp [h1]
  · by_cases h2 : st.pc id
    · simp [h1, h2, setS, clrS]
    · by_cases h3 : st.gd id
      · simp [h1, h2, h3]
      · simp [h1, h2, h3, setS]

/-- Deleting one node never touches another node's status. -/
theorem delNode_other (st : St) (id j : Nat) (h : j ≠ id) : goneN (delNode st id) j = goneN st j := by
  unfold delNode goneN
  split
  · rfl
  · split
    · simp [setS, clrS, h]
    · split
      · simp [setS, h]
      · rfl

theorem delNode_mono (st : St) (id j : Nat) (h : goneN st j = true) : goneN (delNode st id) j = true := by
  by_cases hj : j = id
  · subst hj; exact delNode_gone st j
  · rw [delNode_other st id j hj]; exact h

theorem delRel_gone (st : St) (id : Nat) : goneR (delRel st id) id = true := by
  unfold delRel goneR
  by_cases h1 : st.rd id
  · simp [h1]
  · by_cases h2 : st.rc id || !st.grd id
    · simp [h1, h2, setS]
    · simp only [h1, Bool.false_eq_true, ite_false, h2, Bool.false_or]
      simp only [Bool.or_eq_true, Bool.not_eq_true', not_or] at h2
      simp [h2.1, h2.2]

/-- `delete_entity`: a node/relationship ends up gone, `null` is a no-op, a path deletes each
element, anything else is the C engine's type-mismatch error. -/
theorem delEntity_spec (st : St) :
    (∀ id, (delEntity (.node id) st).map (goneN · id) = .ok true) ∧
    (∀ id, (delEntity (.rel id) st).map (goneR · id) = .ok true) ∧
    delEntity .null st = .ok st ∧ delEntity .other st = .error mismatch := by
  refine ⟨fun id => ?_, fun id => ?_, rfl, rfl⟩
  · simp [delEntity, Except.map, delNode_gone]
  · simp [delEntity, Except.map, delRel_gone]

/-! ## `delete_nodes_bulk` (delete.rs:191) -/

/-- The classification loop (delete.rs:197-206): skip pending-deleted, delete pending-created
now, collect live committed ones. -/
def bulkScan : List Nat → St × List Nat → St × List Nat
  | [], acc => acc
  | id :: ids, (st, com) =>
    if st.pd id then bulkScan ids (st, com)
    else if st.pc id then bulkScan ids (delNode st id, com)
    else if !st.gd id then bulkScan ids (st, com ++ [id])
    else bulkScan ids (st, com)

/-- delete.rs:229-252: every collected id is pending-deleted. -/
def delNodesBulk (st : St) (ids : List Nat) : St :=
  let (st1, com) := bulkScan ids (st, [])
  { st1 with pd := fun j => com.contains j || st1.pd j }

theorem bulkScan_spec (ids : List Nat) (st : St) (com : List Nat) :
    let r := bulkScan ids (st, com)
    (∀ j, goneN st j = true → goneN r.1 j = true) ∧
    (∀ j ∈ com, j ∈ r.2) ∧
    (∀ j ∈ ids, goneN r.1 j = true ∨ j ∈ r.2) ∧
    (∀ j, r.1.pd j = st.pd j) ∧ (∀ j, r.1.pc j = true → st.pc j = true) := by
  induction ids generalizing st com with
  | nil => simp [bulkScan]
  | cons id ids ih =>
    simp only [bulkScan]
    by_cases h1 : st.pd id
    · simp only [h1, ite_true]
      obtain ⟨a, b, c, d, e⟩ := ih st com
      refine ⟨a, b, ?_, d, e⟩
      intro j hj
      rcases List.mem_cons.mp hj with rfl | hj'
      · exact Or.inl (a j (by simp [goneN, h1]))
      · exact c j hj'
    · by_cases h2 : st.pc id
      · simp only [h1, h2, Bool.false_eq_true, ite_false, ite_true]
        obtain ⟨a, b, c, d, e⟩ := ih (delNode st id) com
        refine ⟨fun j hj => a j (delNode_mono st id j hj), b, ?_, ?_, ?_⟩
        · intro j hj
          rcases List.mem_cons.mp hj with rfl | hj'
          · exact Or.inl (a j (delNode_gone st j))
          · exact c j hj'
        · intro j; rw [d j]; simp [delNode, h1, h2]
        · intro j hj; have := e j hj
          simp only [delNode, h1, h2, Bool.false_eq_true, ite_false, ite_true, clrS] at this
          simp at this; exact this.2
      · by_cases h3 : st.gd id
        · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_true]
          obtain ⟨a, b, c, d, e⟩ := ih st com
          refine ⟨a, b, ?_, d, e⟩
          intro j hj
          rcases List.mem_cons.mp hj with rfl | hj'
          · exact Or.inl (a j (by simp [goneN, h3, h2]))
          · exact c j hj'
        · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_false, ite_true]
          obtain ⟨a, b, c, d, e⟩ := ih st (com ++ [id])
          refine ⟨a, fun j hj => b j (List.mem_append_left _ hj), ?_, d, e⟩
          intro j hj
          rcases List.mem_cons.mp hj with rfl | hj'
          · exact Or.inr (b j (by simp))
          · exact c j hj'

/-- **`delete_nodes_bulk`**: every requested node is gone afterwards (duplicates included —
a pending-created node recycled by its first occurrence is skipped by the second), and
nothing already gone comes back. -/
theorem delNodesBulk_spec (st : St) (ids : List Nat) :
    (∀ j ∈ ids, goneN (delNodesBulk st ids) j = true) ∧
    (∀ j, goneN st j = true → goneN (delNodesBulk st ids) j = true) := by
  obtain ⟨a, _, c, _, _⟩ := bulkScan_spec ids st []
  unfold delNodesBulk
  constructor
  · intro j hj
    rcases c j hj with h | h
    · unfold goneN at h ⊢; simp only [Bool.or_eq_true] at h ⊢
      rcases h with h | h
      · exact Or.inl (by simp [h])
      · exact Or.inr h
    · unfold goneN; simp only [Bool.or_eq_true, List.contains_iff_mem]; left; left; exact h
  · intro j hj
    have h := a j hj
    unfold goneN at h ⊢; simp only [Bool.or_eq_true] at h ⊢
    rcases h with h | h
    · exact Or.inl (by simp [h])
    · exact Or.inr h

/-! ## `delete_relationships_bulk` (delete.rs:297) -/

/-- delete.rs:309-323: deduplicate with `seen`, skip pending-deleted, split pending-created
from live committed. -/
def relScan (st : St) : List Nat → List Nat → List Nat × List Nat → List Nat × List Nat
  | [], _, acc => acc
  | id :: ids, seen, (pcr, com) =>
    if seen.contains id then relScan st ids seen (pcr, com)
    else if st.rd id then relScan st ids (seen ++ [id]) (pcr, com)
    else if st.rc id then relScan st ids (seen ++ [id]) (pcr ++ [id], com)
    else if !st.grd id then relScan st ids (seen ++ [id]) (pcr, com ++ [id])
    else relScan st ids (seen ++ [id]) (pcr, com)

def delRelsBulk (st : St) (rels : List Nat) : St :=
  let (pcr, com) := relScan st rels [] ([], [])
  let st1 := pcr.foldl delRel st
  { st1 with rd := fun j => com.contains j || st1.rd j }

def Cls (st : St) (pcr com : List Nat) (j : Nat) : Prop :=
  st.rd j = true ∨ j ∈ pcr ∨ j ∈ com ∨ (st.grd j = true ∧ st.rc j = false)

theorem relScan_cls (st : St) (ids seen pcr com : List Nat)
    (hs : ∀ j ∈ seen, Cls st pcr com j) :
    let r := relScan st ids seen (pcr, com)
    ∀ j, (j ∈ ids ∨ j ∈ seen) → Cls st r.1 r.2 j := by
  induction ids generalizing seen pcr com with
  | nil => intro r j hj; simp at hj; exact hs j hj
  | cons id ids ih =>
    intro r j hj
    have mono : ∀ (p c : List Nat) (x : List Nat) (y : List Nat), (∀ k, k ∈ p → k ∈ x) →
        (∀ k, k ∈ c → k ∈ y) → ∀ k, Cls st p c k → Cls st x y k := by
      intro p c x y hp hc k hk
      rcases hk with h | h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl (hp k h))
      · exact Or.inr (Or.inr (Or.inl (hc k h)))
      · exact Or.inr (Or.inr (Or.inr h))
    have hj' : j ∈ ids ∨ j ∈ seen ++ [id] := by
      rcases hj with h | h
      · rcases List.mem_cons.mp h with rfl | h
        · exact Or.inr (by simp)
        · exact Or.inl h
      · exact Or.inr (List.mem_append_left _ h)
    simp only [r, relScan]
    by_cases h0 : seen.contains id
    · simp only [h0, ite_true]
      apply ih seen pcr com hs
      rcases hj' with h | h
      · exact Or.inl h
      · rcases List.mem_append.mp h with h | h
        · exact Or.inr h
        · simp at h; subst h; exact Or.inr (List.contains_iff_mem.mp h0)
    · simp only [h0, Bool.false_eq_true, ite_false]
      by_cases h1 : st.rd id
      · simp only [h1, ite_true]
        apply ih (seen ++ [id]) pcr com _ j hj'
        intro k hk; rcases List.mem_append.mp hk with hk | hk
        · exact hs k hk
        · simp at hk; subst hk; exact Or.inl h1
      · by_cases h2 : st.rc id
        · simp only [h1, h2, Bool.false_eq_true, ite_false, ite_true]
          apply ih (seen ++ [id]) (pcr ++ [id]) com _ j hj'
          intro k hk; rcases List.mem_append.mp hk with hk | hk
          · exact mono pcr com _ com (fun x hx => List.mem_append_left _ hx) (fun _ h => h) k (hs k hk)
          · simp at hk; subst hk; exact Or.inr (Or.inl (by simp))
        · by_cases h3 : st.grd id
          · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_true]
            apply ih (seen ++ [id]) pcr com _ j hj'
            intro k hk; rcases List.mem_append.mp hk with hk | hk
            · exact hs k hk
            · simp at hk; subst hk; exact Or.inr (Or.inr (Or.inr ⟨h3, by simpa using h2⟩))
          · simp only [h1, h2, h3, Bool.false_eq_true, ite_false, Bool.not_false, ite_true]
            apply ih (seen ++ [id]) pcr (com ++ [id]) _ j hj'
            intro k hk; rcases List.mem_append.mp hk with hk | hk
            · exact mono pcr com pcr _ (fun _ h => h) (fun x hx => List.mem_append_left _ hx) k (hs k hk)
            · simp at hk; subst hk; exact Or.inr (Or.inr (Or.inl (by simp)))

theorem delRel_keep (st : St) (id j : Nat) :
    (st.rd j = true → (delRel st id).rd j = true) ∧ (delRel st id).grd = st.grd ∧ (delRel st id).rc = st.rc := by
  unfold delRel
  refine ⟨fun h => ?_, ?_, ?_⟩
  · split
    · exact h
    · split
      · simp [setS, h]
      · exact h
  all_goals (split; rfl; split <;> rfl)

theorem foldl_delRel (ps : List Nat) (st : St) :
    (∀ j, st.rd j = true → (ps.foldl delRel st).rd j = true) ∧
    (ps.foldl delRel st).grd = st.grd ∧ (ps.foldl delRel st).rc = st.rc ∧
    (∀ j ∈ ps, goneR (ps.foldl delRel st) j = true) := by
  induction ps generalizing st with
  | nil => simp
  | cons p ps ih =>
    obtain ⟨a, b, c, d⟩ := ih (delRel st p)
    obtain ⟨_, k2, k3⟩ := delRel_keep st p 0
    simp only [List.foldl_cons]
    refine ⟨fun j hj => a j ((delRel_keep st p j).1 hj), by rw [b, k2], by rw [c, k3], fun j hj => ?_⟩
    rcases List.mem_cons.mp hj with rfl | hj'
    · have hg := delRel_gone st j
      unfold goneR at hg ⊢
      simp only [Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true'] at hg ⊢
      rcases hg with h | h
      · exact Or.inl (a j h)
      · rw [b, c]; exact Or.inr h
    · exact d j hj'

/-- **`delete_relationships_bulk`**: every requested relationship is gone afterwards
(duplicates are scheduled once via `seen`). -/
theorem delRelsBulk_spec (st : St) (rels : List Nat) :
    ∀ j ∈ rels, goneR (delRelsBulk st rels) j = true := by
  intro j hj
  have hc := relScan_cls st rels [] [] [] (by simp) j (Or.inl hj)
  unfold delRelsBulk
  generalize relScan st rels [] ([], []) = r at hc
  obtain ⟨pcr, com⟩ := r
  obtain ⟨a, b, c, d⟩ := foldl_delRel pcr st
  simp only
  unfold goneR
  rcases hc with h | h | h | h
  · simp [a j h]
  · have := d j h; unfold goneR at this
    simp only [Bool.or_eq_true] at this ⊢
    rcases this with h' | h'
    · exact Or.inl (by simp [h'])
    · exact Or.inr h'
  · simp only [Bool.or_eq_true, List.contains_iff_mem]; left; left; exact h
  · simp [b, c, h.1, h.2]

/-! ## `delete_batch` (delete.rs:127) -/

/-- The variable-tree scan (delete.rs:144-157), rows outer, variables inner. -/
def scan (st : St) : List DV → Except String (St × List Nat × List Nat)
  | [] => .ok (st, [], [])
  | v :: vs => match v with
    | .node id => (scan st vs).map fun r => (r.1, id :: r.2.1, r.2.2)
    | .rel id => (scan st vs).map fun r => (r.1, r.2.1, id :: r.2.2)
    | w => do let st' ← delEntity w st; scan st' vs

/-- `delete_batch`: `vals` = `value_at(var, row)` (`Null` when absent) for each active row ×
variable tree; `exprs` = per active row × expression tree, the evaluated value (or error). -/
def deleteBatch (st : St) (vals : List DV) (exprs : List (Except String DV)) : Except String St := do
  let (st1, ns, rs) ← scan st vals
  let st2 := if ns.isEmpty then st1 else delNodesBulk st1 ns
  let st3 := if rs.isEmpty then st2 else delRelsBulk st2 rs
  exprs.foldlM (fun s e => do let v ← e; delEntity v s) st3

theorem scan_nodes (st : St) (vs : List DV) (r : St × List Nat × List Nat) (h : scan st vs = .ok r) :
    (∀ id, DV.node id ∈ vs → id ∈ r.2.1) ∧ (∀ id, DV.rel id ∈ vs → id ∈ r.2.2) := by
  induction vs generalizing st r with
  | nil => simp
  | cons v vs ih =>
    cases v with
    | node id =>
      simp only [scan] at h
      cases hs : scan st vs with
      | error e => simp [hs, Except.map] at h
      | ok r' =>
        simp only [hs, Except.map, Except.ok.injEq] at h; subst h
        obtain ⟨a, b⟩ := ih st r' hs
        exact ⟨fun i hi => by simp at hi; rcases hi with rfl | hi; simp; exact List.mem_cons_of_mem _ (a i hi),
          fun i hi => by simp at hi; exact b i hi⟩
    | rel id =>
      simp only [scan] at h
      cases hs : scan st vs with
      | error e => simp [hs, Except.map] at h
      | ok r' =>
        simp only [hs, Except.map, Except.ok.injEq] at h; subst h
        obtain ⟨a, b⟩ := ih st r' hs
        exact ⟨fun i hi => by simp at hi; exact a i hi,
          fun i hi => by simp at hi; rcases hi with rfl | hi; simp; exact List.mem_cons_of_mem _ (b i hi)⟩
    | path ps | null | other =>
      simp only [scan, bind, Except.bind] at h
      split at h
      · cases h
      · rename_i st' _
        obtain ⟨a, b⟩ := ih st' r h
        exact ⟨fun i hi => a i (by simpa using hi), fun i hi => b i (by simpa using hi)⟩

/-- A non-entity, non-null, non-path value under DELETE raises the C engine's error. -/
theorem deleteBatch_mismatch (st : St) (vs : List DV) (ex : List (Except String DV)) :
    deleteBatch st (.other :: vs) ex = .error mismatch := by
  simp [deleteBatch, scan, delEntity, bind, Except.bind]

/-- **DELETE**: every node and relationship bound by a deleted variable in any active row is
gone afterwards (nodes before relationships, each requested id handled, duplicates harmless). -/
theorem deleteBatch_vars (st : St) (vs : List DV) (st' : St) (h : deleteBatch st vs [] = .ok st') :
    (∀ id, DV.node id ∈ vs → goneN st' id = true) ∧ (∀ id, DV.rel id ∈ vs → goneR st' id = true) := by
  unfold deleteBatch at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i r hr
    obtain ⟨st1, ns, rs⟩ := r
    simp only [List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    obtain ⟨hn, hrl⟩ := scan_nodes st vs _ hr
    have keepRelsN : ∀ (s : St) (rs : List Nat) j, goneN s j = true → goneN (delRelsBulk s rs) j = true := by
      intro s rs j hj
      unfold delRelsBulk
      generalize relScan s rs [] ([], []) = q
      obtain ⟨pcr, com⟩ := q
      have : ∀ (ps : List Nat) (t : St), goneN (ps.foldl delRel t) j = goneN t j := by
        intro ps; induction ps with
        | nil => intro t; rfl
        | cons p ps ihp =>
          intro t; simp only [List.foldl_cons]; rw [ihp]
          unfold delRel goneN; split
          · rfl
          · split <;> rfl
      simpa [goneN] using (this pcr s).trans hj
    constructor
    · intro id hid
      have hm := hn id hid
      have hns : ns.isEmpty = false := by cases ns <;> simp_all
      simp only [hns, Bool.false_eq_true, ite_false]
      split
      · exact (delNodesBulk_spec st1 ns).1 id hm
      · exact keepRelsN _ _ _ ((delNodesBulk_spec st1 ns).1 id hm)
    · intro id hid
      have hm := hrl id hid
      have hrs : rs.isEmpty = false := by cases rs <;> simp_all
      simp only [hrs, Bool.false_eq_true, ite_false]
      exact delRelsBulk_spec _ rs id hm

/-! ## `snapshot_required`, `*_attrs_to_map`, `new`, `next` -/

/-- delete.rs:69-81: walk the ancestors (nearest first); any non-`Commit` ancestor ⇒ true. -/
def snapshotRequired (isCommit : List Bool) : Bool :=
  match isCommit with
  | [] => false
  | c :: cs => if !c then true else snapshotRequired cs

theorem snapshotRequired_spec (cs : List Bool) : snapshotRequired cs = cs.any (!·) := by
  induction cs with
  | nil => rfl
  | cons c cs ih => cases c <;> simp [snapshotRequired, ih]

/-- delete.rs:38 / :49: keep the ids that have a name, renamed, in order. -/
def attrsToMap {V : Type} (name : Nat → Option String) (attrs : List (Nat × V)) : List (String × V) :=
  attrs.filterMap fun p => (name p.1).map (·, p.2)

theorem attrsToMap_spec {V : Type} (name : Nat → Option String) (attrs : List (Nat × V)) (k : String) (v : V) :
    (k, v) ∈ attrsToMap name attrs ↔ ∃ id, (id, v) ∈ attrs ∧ name id = some k := by
  unfold attrsToMap
  simp only [List.mem_filterMap, Option.map_eq_some_iff, Prod.mk.injEq]
  constructor
  · rintro ⟨⟨id, v'⟩, hm, k', hk, rfl, rfl⟩; exact ⟨id, hm, hk⟩
  · rintro ⟨id, hm, hk⟩; exact ⟨(id, v), hm, k, hk, rfl, rfl⟩

structure DeleteSt (C T : Type) where
  child : C
  trees : T
  snapshot : Bool

def DeleteSt.new {C T : Type} (child : C) (trees : T) (ancestorsCommit : List Bool) : DeleteSt C T :=
  { child, trees, snapshot := snapshotRequired ancestorsCommit }

theorem deleteNew_spec {C T : Type} (c : C) (t : T) (anc : List Bool) :
    (DeleteSt.new c t anc).snapshot = anc.any (!·) ∧ (DeleteSt.new c t anc).child = c :=
  ⟨snapshotRequired_spec anc, rfl⟩

/-- delete.rs:113-124: one child batch, `delete_batch`, pass the batch (or error) on. -/
def deleteNext {B : Type} (del : B → Except String Unit) : Option (Except String B) → Option (Except String B)
  | none => none
  | some (.error e) => some (.error e)
  | some (.ok b) => some ((del b).map fun _ => b)

theorem deleteNext_spec {B : Type} (del : B → Except String Unit) (x : Option (Except String B)) :
    deleteNext del x = x.map (fun eb => eb.bind fun b => (del b).map fun _ => b) := by
  cases x with
  | none => rfl
  | some eb => cases eb <;> rfl

end Delete
