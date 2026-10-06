import PlannerBuild.Expr
/-
# `plan_match`: per-component operator choice (mod.rs:1710-2298)

| here | there |
| --- | --- |
| `QN`, `QR`, `Comp` | `QueryNode`, `QueryRelationship`, a connected component (ast.rs:470-640) |
| `MSt` | `Planner.visited` / `verified_labels` |
| `markLabels`, `unverified` | `mark_labels_verified` mod.rs:1665-1683, `unverified_labels` mod.rs:1687-1699 |
| `relKey`, `sortRels` | the `sort_by_key` of mod.rs:1752-1784 (stable) |
| `emitRel` | the `emit_rel` closure mod.rs:1867-1874 |
| `scanOf`, `withF` | `AllNodeScan`/`NodeByLabelScan` choice; optional inline-attribute `Filter` |
| `firstRel`, `chainRel` | first relationship mod.rs:1890-2048, the others mod.rs:2055-2236 |
| `nodeOnly` | node-only component mod.rs:1786-1863 |
| `boundLabelFilters` | mod.rs:1733-1747 |
| `planComp`, `planMatch` | the component loop and the join / WHERE / bound filters mod.rs:1728-2298 |
| `buildPatternSubPlan` | `build_pattern_sub_plan` mod.rs:916-927 |

Expressions and the WHERE (`plan_filter`, FilterPlan.lean) are parameters;
operators are recorded with the relationship they traverse.
-/
namespace PlannerBuild.E

structure QN where
  alias : V
  labels : List String := []
  attrs : List (String × Ex) := []

inductive Kind | fixed | varLen | shortest
  deriving DecidableEq

structure QR where
  alias : V
  named : Bool
  frm : QN
  to : QN
  kind : Kind := .fixed
  attrs : List (String × Ex) := []

structure Comp where
  nodes : List QN
  rels : List QR
  paths : List (V × List V) := []

structure MSt where
  visited : List V
  verified : List (V × List String)

def MSt.bound (st : MSt) (v : V) : Bool := st.visited.contains v

/-! ## Label bookkeeping -/

def lookupV (m : List (V × List String)) (v : V) : List String :=
  match m.find? (fun p => p.1 == v) with
  | some p => p.2
  | none => []

/-- `mark_labels_verified`: no entry for an empty label list; else union in. -/
def markLabels (st : MSt) (v : V) (ls : List String) : MSt :=
  if ls.isEmpty then st
  else
    let old := lookupV st.verified v
    { st with verified := (v, old ++ ls.filter (fun l => !old.contains l)) :: st.verified }

def unverified (st : MSt) (n : QN) : List String :=
  n.labels.filter fun l => !(lookupV st.verified n.alias).contains l

theorem lookupV_cons_self (m : List (V × List String)) (v : V) (ls : List String) :
    lookupV ((v, ls) :: m) v = ls := by simp [lookupV]

theorem lookupV_cons_other (m : List (V × List String)) (v w : V) (ls : List String) (h : w ≠ v) :
    lookupV ((v, ls) :: m) w = lookupV m w := by
  simp only [lookupV, List.find?_cons]
  have : (v == w) = false := by rw [beq_eq_false_iff_ne]; exact fun e => h e.symm
  simp [this]

/-- After marking, the marked labels are verified… -/
theorem unverified_after_mark (st : MSt) (n : QN) (h : n.labels ≠ []) :
    unverified (markLabels st n.alias n.labels) n = [] := by
  unfold markLabels unverified
  have : n.labels.isEmpty = false := by cases hl : n.labels <;> simp_all
  simp only [this, Bool.false_eq_true, ↓reduceIte, lookupV_cons_self]
  rw [List.filter_eq_nil_iff]
  intro l hl
  by_cases hc : (lookupV st.verified n.alias).contains l <;> simp [hc, hl]

/-- …and what was verified stays verified. -/
theorem markLabels_mono (st : MSt) (v w : V) (ls : List String) (l : String)
    (h : (lookupV st.verified w).contains l) : (lookupV (markLabels st v ls).verified w).contains l := by
  unfold markLabels
  split
  · exact h
  · by_cases hw : w = v
    · subst hw; simp only [lookupV_cons_self]; simp_all
    · simp only [lookupV_cons_other _ _ _ _ hw]; exact h

theorem unverified_spec (st : MSt) (n : QN) (l : String) :
    l ∈ unverified st n ↔ l ∈ n.labels ∧ ¬ (lookupV st.verified n.alias).contains l := by
  simp [unverified]

/-! ## Operators -/

inductive MP
  | arg
  | allScan (n : QN)
  | labelScan (n : QN)
  | filter (e : Ex) (c : MP)
  | pathB (ps : List (V × List V)) (c : MP)
  | cvlt (r : QR) (expandInto : Bool) (pathVar : Option V) (cs : List MP)
  | shortest (r : QR) (cs : List MP)
  | expandInto (r : QR) (emit : Bool) (cs : List MP)
  | condTr (r : QR) (emit : Bool) (cs : List MP)
  | cp (cs : List MP)

def scanOf (n : QN) : MP := if n.labels.isEmpty then .allScan n else .labelScan n

def withF : Option Ex → MP → MP
  | some e, c => .filter e c
  | none, c => c

def attrF (n : QN) : Option Ex := inlineAttrsToFilter n.alias n.attrs
def relF (r : QR) : Option Ex := inlineAttrsToFilter r.alias r.attrs

/-- `emit_relationship`: a named edge, or one a named path lists (by id only, mod.rs:1872). -/
def emitRel (c : Comp) (r : QR) : Bool :=
  r.named || c.paths.any fun p => p.2.any fun v => v.id == r.alias.id

/-- Strip the inline-attribute filters: the operator actually chosen. -/
def MP.core : MP → MP
  | .filter _ c => c.core
  | p => p

theorem core_withF (f : Option Ex) (c : MP) : (withF f c).core = c.core := by
  cases f <;> rfl

/-! ## Relationship order -/

def tier (r : QR) : Nat :=
  match r.kind with
  | .fixed => 0
  | .varLen => 1
  | .shortest => 2

/-- `(base, has_bound, unfiltered)`; `has_bound` is 0 when an endpoint is
bound; `unfiltered` only distinguishes hops leaving a bound node. -/
def relKey (st : MSt) (filterVars : List Nat) (r : QR) : Nat × Nat × Nat :=
  let hb := if st.bound r.frm.alias || st.bound r.to.alias then 0 else 1
  let unf := if hb = 0 then
      (if [r.frm, r.to].any (fun n => !n.attrs.isEmpty || filterVars.contains n.alias.id) then 0 else 1)
    else 0
  (tier r, hb, unf)

def keyLe (a b : Nat × Nat × Nat) : Bool :=
  a.1 < b.1 || (a.1 == b.1 && (a.2.1 < b.2.1 || (a.2.1 == b.2.1 && a.2.2 ≤ b.2.2)))

def sortRels (st : MSt) (fv : List Nat) (rs : List QR) : List QR :=
  rs.mergeSort fun a b => keyLe (relKey st fv a) (relKey st fv b)

theorem keyLe_trans (a b c : Nat × Nat × Nat) (h1 : keyLe a b = true) (h2 : keyLe b c = true) :
    keyLe a c = true := by
  obtain ⟨a1, a2, a3⟩ := a; obtain ⟨b1, b2, b3⟩ := b; obtain ⟨c1, c2, c3⟩ := c
  simp only [keyLe, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at *
  omega

theorem keyLe_total (a b : Nat × Nat × Nat) : (keyLe a b || keyLe b a) = true := by
  obtain ⟨a1, a2, a3⟩ := a; obtain ⟨b1, b2, b3⟩ := b
  simp only [keyLe, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  omega

/-- The order is a permutation, sorted by key: fixed-length hops first, then
variable-length, then shortest paths; within a tier, hops touching a bound
node first (mod.rs:1752-1784). -/
theorem sortRels_spec (st : MSt) (fv : List Nat) (rs : List QR) :
    (sortRels st fv rs).Perm rs ∧
    (sortRels st fv rs).Pairwise (fun a b => keyLe (relKey st fv a) (relKey st fv b) = true) := by
  refine ⟨List.mergeSort_perm _ _, ?_⟩
  exact List.pairwise_mergeSort
    (fun a b c h1 h2 => keyLe_trans (relKey st fv a) (relKey st fv b) (relKey st fv c) h1 h2)
    (fun a b => keyLe_total (relKey st fv a) (relKey st fv b)) rs

theorem sortRels_tiers (st : MSt) (fv : List Nat) (rs : List QR) :
    (sortRels st fv rs).Pairwise (fun a b => tier a ≤ tier b) := by
  apply (sortRels_spec st fv rs).2.imp
  intro a b h
  simp only [keyLe, relKey, Bool.or_eq_true, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  omega

/-! ## The first relationship of a component -/

def visit (st : MSt) (r : QR) : MSt :=
  let st := { st with visited := st.visited ++ [r.frm.alias, r.to.alias, r.alias] }
  markLabels (markLabels st r.frm.alias r.frm.labels) r.to.alias r.to.labels

def firstRel (st : MSt) (c : Comp) (r : QR) : MP :=
  let res := match r.kind with
    | .shortest => .shortest r []
    | .varLen =>
      if st.bound r.frm.alias then
        .cvlt r (r.frm.alias.id != r.to.alias.id && st.bound r.to.alias) none []
      else .cvlt r false none [withF (attrF r.frm) (scanOf r.frm)]
    | .fixed =>
      if r.frm.alias.id = r.to.alias.id then
        if st.bound r.frm.alias then withF (attrF r.frm) (.expandInto r (emitRel c r) [])
        else .expandInto r (emitRel c r) [withF (attrF r.frm) (scanOf r.frm)]
      else if st.bound r.frm.alias && st.bound r.to.alias then
        withF (attrF r.to) (withF (attrF r.frm) (.expandInto r (emitRel c r) []))
      else withF (relF r) (.condTr r (emitRel c r) [])
  let res := if st.bound r.to.alias then res else withF (attrF r.to) res
  if st.bound r.frm.alias then res else withF (attrF r.frm) res

def chainRel (st : MSt) (c : Comp) (res : MP) (r : QR) : MP :=
  let res := match r.kind with
    | .shortest => .shortest r [res]
    | .varLen =>
      let cv := MP.cvlt r (r.frm.alias.id != r.to.alias.id && st.bound r.frm.alias && st.bound r.to.alias)
        none [res]
      if st.bound r.frm.alias then cv else withF (attrF r.frm) cv
    | .fixed =>
      if r.frm.alias.id = r.to.alias.id then
        if st.bound r.frm.alias then withF (attrF r.frm) (.expandInto r (emitRel c r) [res])
        else .expandInto r (emitRel c r) [withF (attrF r.frm) (scanOf r.frm), res]
      else if st.bound r.frm.alias && st.bound r.to.alias then
        withF (attrF r.to) (withF (attrF r.frm) (.expandInto r (emitRel c r) [res]))
      else withF (relF r) (.condTr r (emitRel c r) [res])
  if st.bound r.to.alias then res else withF (attrF r.to) res

/-- The operator chosen for the first hop (attribute filters stripped). -/
theorem firstRel_core (st : MSt) (c : Comp) (r : QR) :
    (firstRel st c r).core =
      match r.kind with
      | .shortest => .shortest r []
      | .varLen =>
        if st.bound r.frm.alias then
          .cvlt r (r.frm.alias.id != r.to.alias.id && st.bound r.to.alias) none []
        else .cvlt r false none [withF (attrF r.frm) (scanOf r.frm)]
      | .fixed =>
        if r.frm.alias.id = r.to.alias.id then .expandInto r (emitRel c r)
          (if st.bound r.frm.alias then [] else [withF (attrF r.frm) (scanOf r.frm)])
        else if st.bound r.frm.alias && st.bound r.to.alias then .expandInto r (emitRel c r) []
        else .condTr r (emitRel c r) [] := by
  unfold firstRel
  cases hk : r.kind <;> by_cases h1 : st.bound r.frm.alias <;> by_cases h2 : st.bound r.to.alias <;>
    by_cases h3 : r.frm.alias.id = r.to.alias.id <;> simp [h1, h2, h3, core_withF, MP.core]

end PlannerBuild.E
