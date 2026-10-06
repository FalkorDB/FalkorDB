import PendingCommit.PendingDocs
import PendingCommit.Labels
/-
# `Pending::commit` and constraint enforcement (pending.rs:1099-1496)

* `constraintHas_eq` — `constraint_node_has_label` (:1331) answers from the
  staged label sets, then "created ⇒ no committed label", then the matrix; when
  created nodes carry no committed labels (fresh ids; reclaimed ids carry
  tombstones) that is exactly `Labels.overlay`, the query's own view.
* `scan_ok` — the `seen` map scan of a UNIQUE check (:1396-1416, :1466-1486):
  passes iff no two distinct entities with a full composite key share it.
* `checkNode_spec`, `checkEdge_spec` — `check_node_constraint` (:1355) /
  `check_edge_constraint` (:1425): MANDATORY passes iff every affected member
  has every property; UNIQUE iff, when some affected member participates, the
  label's (type's) scan passes. Membership of an edge created in this batch
  comes from `created_rel_types`.
* `affected_spec`, `enforce_spec` — `enforce_constraints` (:1258): affected =
  (created ∪ staged-attr ∪ staged-label nodes) \ deleted, edges likewise; the
  query fails iff some *operational* constraint's check fails.
* `commit_spec` — `Pending::commit` (:1099), graph operations uninterpreted
  (`GOps`): the statistics it adds are `|created_nodes|`,
  `|created_rel_types|`, `Σ|remove_labels|`, the attribute writers' counts,
  `|deleted_nodes|` and the implicit + actually-deleted explicit edges;
  `deleted_relationships` becomes the implicit ∪ actually-deleted edges, the
  explicit set is handed to both deleters, and the batch's index documents are
  absorbed into the deferred set.
-/
namespace PendingCommit.PC
open PA PS PD

/-! ## Labels for constraints -/

def constraintHas (p : P) (C : Nat → Nat → Bool) (n l : Nat) : Bool :=
  if ((fget p.remL n).getD []).contains l then false
  else if ((fget p.setL n).getD []).contains l then true
  else if n ∈ p.created then false
  else C n l

/-- **`constraintHas_eq`**. -/
theorem constraintHas_eq (p : P) (C : Nat → Nat → Bool) (hC : ∀ n ∈ p.created, ∀ l, C n l = false) (n l : Nat) :
    constraintHas p C n l = Labels.overlay (C n) ⟨(fget p.setL n).getD [], (fget p.remL n).getD []⟩ l := by
  unfold constraintHas Labels.overlay
  simp only
  split
  · rfl
  · split
    · rfl
    · split
      · rename_i h; rw [hC n h l]
      · rfl

/-! ## The UNIQUE scan -/

def lookupK : List (List Nat × Nat) → List Nat → Option Nat
  | [], _ => none
  | (k, v) :: s, k' => if k = k' then some v else lookupK s k'

def scan (key : Nat → List Nat) : List (List Nat × Nat) → List Nat → Bool
  | _, [] => true
  | seen, x :: xs =>
    if key x = [] then scan key seen xs
    else match lookupK seen (key x) with
      | some ex => if ex ≠ x then false else scan key ((key x, x) :: seen) xs
      | none => scan key ((key x, x) :: seen) xs

theorem lookupK_some_mem : ∀ (seen : List (List Nat × Nat)) k v, lookupK seen k = some v → (k, v) ∈ seen
  | [], _, _, h => by simp [lookupK] at h
  | (a, b) :: s, k, v, h => by
    simp only [lookupK] at h; split at h
    · cases h; rename_i h'; subst h'; simp
    · exact List.mem_cons_of_mem _ (lookupK_some_mem s k v h)

theorem scan_gen (key : Nat → List Nat) : ∀ (l : List Nat) (seen : List (List Nat × Nat)), l.Nodup →
    (∀ p ∈ seen, p.2 ∉ l) →
    (scan key seen l = true ↔ ((l.filter (key · ≠ [])).map key).Nodup ∧
      ∀ x ∈ l, key x ≠ [] → lookupK seen (key x) = none)
  | [], _, _, _ => by simp [scan]
  | x :: xs, seen, hnd, hs => by
    simp only [List.nodup_cons] at hnd
    have hs' : ∀ p ∈ seen, p.2 ∉ xs := fun p hp hm => hs p hp (List.mem_cons_of_mem _ hm)
    simp only [scan]
    by_cases hk : key x = []
    · rw [if_pos hk, scan_gen key xs seen hnd.2 hs']
      simp [hk]
    · rw [if_neg hk]
      have hs2 : ∀ p ∈ (key x, x) :: seen, p.2 ∉ xs := by
        intro p hp; simp at hp; rcases hp with rfl | hp
        · exact hnd.1
        · exact hs' p hp
      cases hl : lookupK seen (key x) with
      | some ex =>
        have hex : ex ≠ x := fun e => hs (key x, ex) (lookupK_some_mem _ _ _ hl) (by simp [e])
        simp only [hex, ne_eq, not_false_eq_true, ite_true, Bool.false_eq_true, false_iff, not_and]
        intro _ h; exact absurd (h x (by simp) hk) (by simp [hl])
      | none =>
        rw [scan_gen key xs _ hnd.2 hs2]
        simp only [List.filter_cons, ne_eq, hk, not_false_eq_true, decide_true, ite_true, List.map_cons,
          List.nodup_cons, List.mem_cons, forall_eq_or_imp, hl, true_and]
        have e : ∀ y, lookupK ((key x, x) :: seen) (key y) = if key x = key y then some x else lookupK seen (key y) :=
          fun y => rfl
        constructor
        · rintro ⟨hn, hall⟩
          refine ⟨⟨fun hm => ?_, hn⟩, fun _ => trivial, fun a ha hka => ?_⟩
          · obtain ⟨y, hy, hky⟩ := List.mem_map.1 hm
            have hy' := List.mem_filter.1 hy
            have := hall y hy'.1 (by simpa using hy'.2)
            rw [e, if_pos hky.symm] at this; cases this
          · have := hall a ha hka; rw [e] at this
            split at this
            · cases this
            · exact this
        · rintro ⟨⟨hnm, hn⟩, -, hall⟩
          refine ⟨hn, fun y hy hky => ?_⟩
          rw [e, if_neg ?_]
          · exact hall y hy hky
          · intro heq; exact hnm (List.mem_map.2 ⟨y, List.mem_filter.2 ⟨hy, by simpa using hky⟩, heq.symm⟩)

/-- **`scan_ok`**. -/
theorem scan_ok (key : Nat → List Nat) (l : List Nat) (h : l.Nodup) :
    scan key [] l = true ↔ ((l.filter (key · ≠ [])).map key).Nodup := by
  rw [scan_gen key l [] h (by simp)]; simp [lookupK]

/-! ## Checks -/

inductive CT where
  | mandatory
  | unique
  deriving DecidableEq

/-- One constraint's check over the affected entities. `member e`: label/type
membership; `present e prop`: non-null property; `key e`: composite key
(`[]` when a property is null/absent); `all`: the label matrix (relationship
tensor) listing used by the UNIQUE scan. -/
def check (ct : CT) (props : List Nat) (affected : List Nat) (member : Nat → Bool) (present : Nat → Nat → Bool)
    (key : Nat → List Nat) (all : List Nat) : Bool :=
  affected.all fun e => !member e ||
    match ct with
    | .mandatory => props.all (present e)
    | .unique => key e = [] || scan key [] all

/-- `check_node_constraint` (:1355): no such label ⇒ nothing to check. -/
def checkNode (labelExists : Bool) (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (key : Nat → List Nat) (all : List Nat) : Bool :=
  !labelExists || check ct props affected member present key all

/-- `check_edge_constraint` (:1425): type from `created_rel_types` for edges
created in this batch, else the graph. -/
def edgeMember (p : P) (gHasType : Nat → Bool) (ty : Nat) (e : Nat) : Bool :=
  match fget p.relTypes e with
  | some t => t == ty
  | none => gHasType e

/-- **`check_spec`** (for both `checkNode` and `check_edge_constraint`). -/
theorem check_spec (ct : CT) (props affected : List Nat) (member : Nat → Bool) (present : Nat → Nat → Bool)
    (key : Nat → List Nat) (all : List Nat) (hnd : all.Nodup) :
    check ct props affected member present key all = true ↔
      ∀ e ∈ affected, member e = true →
        (ct = .mandatory → ∀ q ∈ props, present e q = true) ∧
        (ct = .unique → key e ≠ [] → ((all.filter (key · ≠ [])).map key).Nodup) := by
  unfold check
  simp only [List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true']
  constructor
  · intro h e he hm
    have := h e he
    simp only [hm, Bool.true_eq_false, false_or] at this
    cases ct with
    | mandatory =>
      refine ⟨fun _ => by simpa [List.all_eq_true] using this, fun h => absurd h (by decide)⟩
    | unique =>
      refine ⟨fun h => absurd h (by decide), fun _ hk => ?_⟩
      simp only [hk, decide_false, Bool.false_or] at this
      exact (scan_ok key all hnd).1 this
  · intro h e he
    cases hm : member e
    · exact .inl rfl
    · right
      obtain ⟨h1, h2⟩ := h e he hm
      cases ct with
      | mandatory => simpa [List.all_eq_true] using h1 rfl
      | unique =>
        by_cases hk : key e = []
        · simp [hk]
        · simp only [hk, decide_false, Bool.false_or]; exact (scan_ok key all hnd).2 (h2 rfl hk)

theorem checkNode_spec (labelExists : Bool) (ct : CT) (props affected : List Nat) (member : Nat → Bool)
    (present : Nat → Nat → Bool) (key : Nat → List Nat) (all : List Nat) :
    checkNode labelExists ct props affected member present key all =
      (!labelExists || check ct props affected member present key all) := rfl

theorem edgeMember_spec (p : P) (gHasType : Nat → Bool) (ty e : Nat) :
    edgeMember p gHasType ty e = match getRelType p e with | some t => t == ty | none => gHasType e := rfl

/-! ## `enforce_constraints` -/

def affectedNodes (p : P) : List Nat :=
  (p.created ++ p.newN.map (·.1) ++ p.existN.map (·.1) ++ p.setL.map (·.1)).filter (· ∉ p.deletedN)

def affectedEdges (p : P) : List Nat :=
  (p.relsByType.flatMap (fun g => g.2.map (·.1)) ++ p.newR.map (·.1) ++ p.existR.map (·.1)).filter (· ∉ p.deletedR)

theorem affected_spec (p : P) (n : Nat) :
    (n ∈ affectedNodes p ↔ (n ∈ p.created ∨ (fget p.newN n).isSome ∨ (fget p.existN n).isSome ∨
      (fget p.setL n).isSome) ∧ n ∉ p.deletedN) := by
  have key : ∀ {V : Type} (m : FMap V), n ∈ m.map (·.1) ↔ (fget m n).isSome := by
    intro V m
    induction m with
    | nil => simp [fget]
    | cons q m ih =>
      obtain ⟨c, d⟩ := q
      simp only [List.map_cons, List.mem_cons, fget]
      by_cases hc : c = n
      · simp [hc]
      · simp [hc, Ne.symm hc, ih]
  simp only [affectedNodes, List.mem_filter, List.mem_append, decide_eq_true_eq, key]
  constructor
  · rintro ⟨(((h | h) | h) | h), hd⟩ <;> simp_all
  · rintro ⟨(h | h | h | h), hd⟩ <;> simp_all

structure Constraint where
  operational : Bool
  isNode : Bool
  ok : Bool   -- the value of `checkNode` / `check_edge_constraint` for it

/-- `enforce_constraints` (:1258): no constraints ⇒ `Ok`; otherwise every
operational constraint's check, in order, `?` on the first failure. -/
def enforce (cs : List Constraint) : Bool := cs.all fun c => !c.operational || c.ok

theorem enforce_spec (cs : List Constraint) :
    enforce cs = true ↔ ∀ c ∈ cs, c.operational = true → c.ok = true := by
  simp only [enforce, List.all_eq_true, Bool.or_eq_true, Bool.not_eq_true']
  constructor
  · intro h c hc ho; rcases h c hc with h | h
    · rw [ho] at h; cases h
    · exact h
  · intro h c hc
    cases ho : c.operational
    · exact .inl rfl
    · exact .inr (h c hc ho)

/-! ## `Pending::commit` -/

structure Stats where
  nodesCreated : Nat
  relsCreated : Nat
  labelsRemoved : Nat
  propsSet : Nat
  propsRemoved : Nat
  nodesDeleted : Nat
  relsDeleted : Nat

/-- The graph-side operations, uninterpreted (`none` = `Err`). -/
structure GOps (G : Type) where
  createNodes : G → List Nat → Option G
  createRels : G → FMap (List (Nat × Nat × Nat)) → Option G
  setLabels : G → List Nat → List Nat → G
  removeLabels : G → List Nat → List Nat → G
  importNodeAttrs : G → FMap (List Entry) → G × Nat
  setNodeAttrs : G → FMap (List Entry) → Option (G × Nat × Nat)
  importRelAttrs : G → FMap (List Entry) → G × Nat
  setRelAttrs : G → FMap (List Entry) → Option (G × Nat × Nat)
  deleteNodes : G → List Nat → Option G
  implicitEdges : G → List Nat → List Nat → Option (G × List Nat)
  deleteRels : G → List Nat → Option (G × List Nat)
  constraintsOk : G → P → Bool

structure CS (G : Type) where
  g : G
  s : Stats
  imp : List Nat
  act : List Nat

variable {G : Type}

def stCreate (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  if p.created = [] then some c
  else (o.createNodes c.g p.created).map fun g => { c with g := g, s := { c.s with nodesCreated := c.s.nodesCreated + p.created.length } }

def stRels (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  if p.relTypes = [] then some c
  else (o.createRels c.g p.relsByType).map fun g => { c with g := g, s := { c.s with relsCreated := c.s.relsCreated + p.relTypes.length } }

def stLabels (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  let g3 := if p.setL = [] then c.g else o.setLabels c.g (flatten p.setL).1 (flatten p.setL).2
  if p.remL = [] then some { c with g := g3 }
  else some { c with g := o.removeLabels g3 (flatten p.remL).1 (flatten p.remL).2,
                     s := { c.s with labelsRemoved := c.s.labelsRemoved + (flatten p.remL).1.length } }

def stNewN (o : GOps G) (p : P) (c : CS G) : CS G :=
  if p.newN = [] then c else
    let r := o.importNodeAttrs c.g p.newN; { c with g := r.1, s := { c.s with propsSet := c.s.propsSet + r.2 } }
def stExistN (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  if p.existN = [] then some c else (o.setNodeAttrs c.g p.existN).map fun r =>
    { c with g := r.1, s := { c.s with propsSet := c.s.propsSet + r.2.2, propsRemoved := c.s.propsRemoved + r.2.1 } }
def stNewR (o : GOps G) (p : P) (c : CS G) : CS G :=
  if p.newR = [] then c else
    let r := o.importRelAttrs c.g p.newR; { c with g := r.1, s := { c.s with propsSet := c.s.propsSet + r.2 } }
def stExistR (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  if p.existR = [] then some c else (o.setRelAttrs c.g p.existR).map fun r =>
    { c with g := r.1, s := { c.s with propsSet := c.s.propsSet + r.2.2, propsRemoved := c.s.propsRemoved + r.2.1 } }
def stAttrs (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  (stExistN o p (stNewN o p c)).bind fun c6 => stExistR o p (stNewR o p c6)

def stDelNodes (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  if p.deletedN = [] then some c else (o.deleteNodes c.g p.deletedN).map fun g =>
    { c with g := g, s := { c.s with nodesDeleted := c.s.nodesDeleted + p.deletedN.length } }
/-- `explicit_rels = take(deleted_relationships)` happens before the cascade,
which is handed the explicit set for dedup. -/
def stImplicit (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  if p.deletedN = [] then some c else (o.implicitEdges c.g p.deletedN p.deletedR).map fun r =>
    { c with g := r.1, s := { c.s with relsDeleted := c.s.relsDeleted + r.2.length }, imp := r.2 }
def stExplicit (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  if p.deletedR = [] then some c else (o.deleteRels c.g p.deletedR).map fun r =>
    { c with g := r.1, s := { c.s with relsDeleted := c.s.relsDeleted + r.2.length }, act := r.2 }
def stDelete (o : GOps G) (p : P) (c : CS G) : Option (CS G) :=
  (stDelNodes o p c).bind fun c9 => (stImplicit o p c9).bind (stExplicit o p)

/-- `Pending::commit` (:1099), then `enforce_constraints` (:1224); the
index-document absorb (:1232) is `PD.absorb_spec`. -/
def finish (o : GOps G) (p : P) (c : CS G) : Option (P × G × Stats) :=
  let p' := { p with deletedR := c.imp.foldl sins [] ++ c.act }
  if o.constraintsOk c.g p' then some (p', c.g, c.s) else none

def commit (o : GOps G) (p : P) (g0 : G) (s0 : Stats) : Option (P × G × Stats) :=
  (stCreate o p ⟨g0, s0, [], []⟩).bind fun c1 => (stRels o p c1).bind fun c2 => (stLabels o p c2).bind fun c3 =>
    (stAttrs o p c3).bind fun c4 => (stDelete o p c4).bind (finish o p)

theorem stCreate_s (o : GOps G) (p : P) (c c' : CS G) (h : stCreate o p c = some c') :
    c'.s = { c.s with nodesCreated := c.s.nodesCreated + p.created.length } ∧ c'.imp = c.imp ∧ c'.act = c.act := by
  unfold stCreate at h; split at h
  · rename_i he; cases h; simp [he]
  · obtain ⟨g, -, rfl⟩ := Option.map_eq_some_iff.1 h; exact ⟨rfl, rfl, rfl⟩

theorem stRels_s (o : GOps G) (p : P) (c c' : CS G) (h : stRels o p c = some c') :
    c'.s = { c.s with relsCreated := c.s.relsCreated + p.relTypes.length } ∧ c'.imp = c.imp ∧ c'.act = c.act := by
  unfold stRels at h; split at h
  · rename_i he; cases h; simp [he]
  · obtain ⟨g, -, rfl⟩ := Option.map_eq_some_iff.1 h; exact ⟨rfl, rfl, rfl⟩

theorem stLabels_s (o : GOps G) (p : P) (c c' : CS G) (h : stLabels o p c = some c') :
    c'.s = { c.s with labelsRemoved := c.s.labelsRemoved + (p.remL.map (·.2.length)).sum } ∧
      c'.imp = c.imp ∧ c'.act = c.act := by
  have hf := (flatten_spec p.remL).1
  unfold stLabels at h; dsimp only at h; split at h
  · rename_i he; cases h; simp [he]
  · cases h; exact ⟨by rw [hf], rfl, rfl⟩

/-- Fields a step does not touch. -/
def Keeps (c c' : CS G) : Prop :=
  c'.s.nodesCreated = c.s.nodesCreated ∧ c'.s.relsCreated = c.s.relsCreated ∧
  c'.s.labelsRemoved = c.s.labelsRemoved ∧ c'.s.nodesDeleted = c.s.nodesDeleted ∧
  c'.s.relsDeleted = c.s.relsDeleted ∧ c'.imp = c.imp ∧ c'.act = c.act

theorem keeps_trans {c1 c2 c3 : CS G} (a : Keeps c1 c2) (b : Keeps c2 c3) : Keeps c1 c3 := by
  obtain ⟨a1, a2, a3, a4, a5, a6, a7⟩ := a; obtain ⟨b1, b2, b3, b4, b5, b6, b7⟩ := b
  exact ⟨b1.trans a1, b2.trans a2, b3.trans a3, b4.trans a4, b5.trans a5, b6.trans a6, b7.trans a7⟩

theorem stAttrs_s (o : GOps G) (p : P) (c c' : CS G) (h : stAttrs o p c = some c') : Keeps c c' := by
  have kn : ∀ c, Keeps c (stNewN o p c) := fun c => by unfold stNewN; split <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  have kr : ∀ c, Keeps c (stNewR o p c) := fun c => by unfold stNewR; split <;> exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  have ke : ∀ c c', stExistN o p c = some c' → Keeps c c' := fun c c' h => by
    unfold stExistN at h; split at h
    · cases h; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
    · obtain ⟨r, -, rfl⟩ := Option.map_eq_some_iff.1 h; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  have kx : ∀ c c', stExistR o p c = some c' → Keeps c c' := fun c c' h => by
    unfold stExistR at h; split at h
    · cases h; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
    · obtain ⟨r, -, rfl⟩ := Option.map_eq_some_iff.1 h; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  unfold stAttrs at h
  obtain ⟨c6, h6, h7⟩ := Option.bind_eq_some_iff.1 h
  exact keeps_trans (keeps_trans (kn c) (ke _ _ h6)) (keeps_trans (kr c6) (kx _ _ h7))

theorem stDelete_s (o : GOps G) (p : P) (c c' : CS G) (h : stDelete o p c = some c') (hi : c.imp = []) (ha : c.act = []) :
    c'.s.nodesCreated = c.s.nodesCreated ∧ c'.s.relsCreated = c.s.relsCreated ∧
    c'.s.labelsRemoved = c.s.labelsRemoved ∧ c'.s.nodesDeleted = c.s.nodesDeleted + p.deletedN.length ∧
    c'.s.relsDeleted = c.s.relsDeleted + c'.imp.length + c'.act.length := by
  unfold stDelete at h
  obtain ⟨c9, h9, h'⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c10, h10, h11⟩ := Option.bind_eq_some_iff.1 h'
  have e9 : c9.s.nodesCreated = c.s.nodesCreated ∧ c9.s.relsCreated = c.s.relsCreated ∧
      c9.s.labelsRemoved = c.s.labelsRemoved ∧ c9.s.nodesDeleted = c.s.nodesDeleted + p.deletedN.length ∧
      c9.s.relsDeleted = c.s.relsDeleted ∧ c9.imp = [] ∧ c9.act = [] := by
    unfold stDelNodes at h9; split at h9
    · rename_i he; cases h9; simp [he, hi, ha]
    · obtain ⟨g, -, rfl⟩ := Option.map_eq_some_iff.1 h9; exact ⟨rfl, rfl, rfl, rfl, rfl, hi, ha⟩
  have e10 : c10.s.nodesCreated = c.s.nodesCreated ∧ c10.s.relsCreated = c.s.relsCreated ∧
      c10.s.labelsRemoved = c.s.labelsRemoved ∧ c10.s.nodesDeleted = c.s.nodesDeleted + p.deletedN.length ∧
      c10.s.relsDeleted = c.s.relsDeleted + c10.imp.length ∧ c10.act = [] := by
    obtain ⟨a1, a2, a3, a4, a5, a6, a7⟩ := e9
    unfold stImplicit at h10; split at h10
    · cases h10; exact ⟨a1, a2, a3, a4, by simp [a5, a6], a7⟩
    · obtain ⟨r, -, rfl⟩ := Option.map_eq_some_iff.1 h10
      exact ⟨a1, a2, a3, a4, by simp [a5], a7⟩
  obtain ⟨b1, b2, b3, b4, b5, b6⟩ := e10
  unfold stExplicit at h11; split at h11
  · cases h11; exact ⟨b1, b2, b3, b4, by simp [b5, b6]⟩
  · obtain ⟨r, -, rfl⟩ := Option.map_eq_some_iff.1 h11
    exact ⟨b1, b2, b3, b4, by simp [b5] <;> omega⟩

/-- **`commit_spec`**. -/
theorem commit_spec (o : GOps G) (p : P) (g0 : G) (s0 : Stats) (r : P × G × Stats)
    (h : commit o p g0 s0 = some r) :
    r.2.2.nodesCreated = s0.nodesCreated + p.created.length ∧
    r.2.2.relsCreated = s0.relsCreated + p.relTypes.length ∧
    r.2.2.labelsRemoved = s0.labelsRemoved + (p.remL.map (·.2.length)).sum ∧
    r.2.2.nodesDeleted = s0.nodesDeleted + p.deletedN.length ∧
    (∃ imp act : List Nat, r.1.deletedR = imp.foldl sins [] ++ act ∧
      r.2.2.relsDeleted = s0.relsDeleted + imp.length + act.length) ∧
    r.1.deletedN = p.deletedN ∧ r.1.created = p.created ∧ o.constraintsOk r.2.1 r.1 = true := by
  unfold commit at h
  obtain ⟨c1, h1, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c2, h2, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c3, h3, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c4, h4, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨c5, h5, h6⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨e1, i1, a1⟩ := stCreate_s o p _ c1 h1
  obtain ⟨e2, i2, a2⟩ := stRels_s o p _ c2 h2
  obtain ⟨e3, i3, a3⟩ := stLabels_s o p _ c3 h3
  obtain ⟨f1, f2, f3, f4, f5, i4, a4⟩ := stAttrs_s o p _ c4 h4
  have hi : c4.imp = [] := by rw [i4, i3, i2, i1]
  have ha : c4.act = [] := by rw [a4, a3, a2, a1]
  obtain ⟨d1, d2, d3, d4, d5⟩ := stDelete_s o p _ c5 h5 hi ha
  unfold finish at h6
  dsimp only at h6
  split at h6
  · rename_i hok; cases h6
    simp only [e1, e2, e3] at f1 f2 f3 f4 f5
    refine ⟨by simp [d1, f1, e3, e2, e1], by simp [d2, f2, e3, e2], by simp [d3, f3, e3], by simp [d4, f4, e3, e2, e1],
      ⟨c5.imp, c5.act, rfl, by simp [d5, f5, e3, e2, e1]⟩, rfl, rfl, hok⟩
  · cases h6

end PendingCommit.PC
