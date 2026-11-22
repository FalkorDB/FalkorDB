import PlannerBuild.PlanMatch2
/-
# `verified_labels` across REMOVE and WITH (#3450, fixed by #3457)

Property `VerifiedSound st gl`: every label `verified_labels` records for a variable is on
that variable's node in `gl` — the invariant `unverified_labels` silently assumes when it
lets an `ExpandInto`/`CondTraverse` skip the label check. Planning a clause must preserve it
(`RemoveKeepsVerifiedSound`). Rust source of truth: `graph/src/planner/mod.rs` (PR #3457):
the `QueryIR::Remove` arm of `Planner::plan` takes the labels of each `n:L…` item
(`removed_node_labels`), records them in `removed_labels` and drops them from the verified
set of *every* variable (the node may be bound under several names); the `QueryIR::Union` arm
does the same with each branch planner's `removed_labels`.

Repro of the fixed bug, after `CREATE (:A)-[:R]->(:B)`:
`MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH (a:A)-[:R]->(b) RETURN labels(a), labels(b)`:
before #3457 1 row `[] [B]`, now 0 rows as in C (`remove_unverifies_label`).
-/
namespace PlannerBuild.E

/-- Node labels of the row's variables, as the graph the next operator reads has them. -/
abbrev NL := V → List String

def VerifiedSound (st : MSt) (gl : NL) : Prop :=
  ∀ v l, l ∈ lookupV st.verified v → l ∈ gl v

/-- REMOVE `v:ls` on the graph. -/
def removeG (gl : NL) (v : V) (ls : List String) : NL :=
  fun w => if w = v then (gl w).filter (fun l => !ls.contains l) else gl w

/-- mod.rs `QueryIR::Remove`: the removed labels leave every variable's verified set (the
`OrderSet::remove` per label, on `verified_labels.values_mut()`). -/
def unverify (st : MSt) (ls : List String) : MSt :=
  { st with verified := st.verified.map fun p => (p.1, p.2.filter (fun l => !ls.contains l)) }

def rsRemove (st : MSt) (_v : V) (ls : List String) : MSt := unverify st ls

theorem lookupV_unverify (st : MSt) (ls : List String) (w : V) :
    lookupV (unverify st ls).verified w = (lookupV st.verified w).filter (fun l => !ls.contains l) := by
  simp only [unverify, lookupV]
  induction st.verified with
  | nil => rfl
  | cons p t ih =>
    simp only [List.map_cons, List.find?_cons]
    by_cases hp : p.1 = w
    · simp [hp]
    · have : (p.1 == w) = false := by simpa using hp
      simp only [this]
      exact ih

/-- **Property `RemoveKeepsVerifiedSound`**: planning `REMOVE v:ls` keeps `VerifiedSound`
for the graph the REMOVE leaves behind. -/
def RemoveKeepsVerifiedSound (plan : MSt → V → List String → MSt) : Prop :=
  ∀ st gl v ls, VerifiedSound st gl → VerifiedSound (plan st v ls) (removeG gl v ls)

/-- **#3457 fixes #3450 (planner half)**: the REMOVE arm keeps `VerifiedSound`. -/
theorem rsRemove_sound : RemoveKeepsVerifiedSound rsRemove := by
  intro st gl v ls h w l hl
  rw [rsRemove, lookupV_unverify, List.mem_filter] at hl
  unfold removeG
  by_cases hw : w = v
  · subst hw
    rw [ite_eq_left rfl, List.mem_filter]
    exact ⟨h _ _ hl.1, hl.2⟩
  · simp only [ite_eq_right hw]
    exact h _ _ hl.1

/-- Unverifying more labels (a UNION branch's removals, or another variable's) never breaks
soundness, for any graph. -/
theorem unverify_sound (st : MSt) (gl : NL) (ls : List String) (h : VerifiedSound st gl) :
    VerifiedSound (unverify st ls) gl := by
  intro w l hl
  rw [lookupV_unverify, List.mem_filter] at hl
  exact h _ _ hl.1

/-- The UNION arm: after the branches, every label a branch removed is unverified. -/
def unionUnverify (st : MSt) (branchRemoved : List (List String)) : MSt :=
  branchRemoved.foldl unverify st

theorem unionUnverify_sound (st : MSt) (gl : NL) (bs : List (List String))
    (h : VerifiedSound st gl) : VerifiedSound (unionUnverify st bs) gl := by
  induction bs generalizing st with
  | nil => exact h
  | cons b r ih => exact ih _ (unverify_sound _ _ _ h)

theorem unionUnverify_drops (st : MSt) (bs : List (List String)) (b : List String) (hb : b ∈ bs)
    (l : String) (hl : l ∈ b) (w : V) : l ∉ lookupV (unionUnverify st bs).verified w := by
  induction bs generalizing st with
  | nil => cases hb
  | cons c r ih =>
    rcases List.mem_cons.1 hb with e | e
    · subst e
      simp only [unionUnverify, List.foldl_cons]
      have : l ∉ lookupV (unverify st b).verified w := by
        rw [lookupV_unverify, List.mem_filter]; simp [hl]
      revert this
      generalize unverify st b = s
      intro hs
      clear ih hb
      induction r generalizing s with
      | nil => exact hs
      | cons d t ih2 =>
        simp only [List.foldl_cons]
        apply ih2
        rw [lookupV_unverify, List.mem_filter]; exact fun h => hs h.1
    · exact ih _ e

/-- `plan_project`'s `carried_verified` for variables projected as-is (`(old, new)`). -/
def project (st : MSt) (ren : List (V × V)) : MSt :=
  { visited := ren.map (·.2), verified := ren.map (fun p => (p.2, lookupV st.verified p.1)) }

/-- The projected row: each new variable is the old one's node. -/
def projectG (gl : NL) (ren : List (V × V)) : NL :=
  fun n => match ren.find? (fun p => p.2 == n) with
    | some p => gl p.1
    | none => []

theorem lookupV_map_ren (m : List (V × List String)) (ren : List (V × V)) (n : V) :
    lookupV (ren.map (fun p => (p.2, lookupV m p.1))) n =
      match ren.find? (fun p => p.2 == n) with
      | some p => lookupV m p.1
      | none => [] := by
  induction ren with
  | nil => rfl
  | cons p t ih =>
    by_cases hp : p.2 = n
    · subst hp; simp [lookupV]
    · simp only [List.map_cons, List.find?_cons]
      have : (p.2 == n) = false := by simpa using hp
      rw [this, lookupV_cons_other _ _ _ _ (Ne.symm hp)]
      exact ih

/-- WITH preserves soundness: `carried_verified` is sound for the projected row. -/
theorem project_sound (st : MSt) (gl : NL) (ren : List (V × V)) (h : VerifiedSound st gl) :
    VerifiedSound (project st ren) (projectG gl ren) := by
  intro n l hl
  simp only [project, lookupV_map_ren] at hl
  unfold projectG
  split at hl
  · exact h _ _ hl
  · simp at hl

/-! ## The counterexample -/

def aRm : V := ⟨0, 0⟩
def bRm : V := ⟨1, 0⟩
def a1 : V := ⟨0, 1⟩
def b1 : V := ⟨1, 1⟩

/-- After `MATCH (a:A), (b:B)`: both scans verified their label; the graph agrees. -/
def stMatched : MSt := markLabels (markLabels { visited := [aRm, bRm], verified := [] } aRm ["A"]) bRm ["B"]
def glMatched : NL := fun v => if v = aRm then ["A"] else if v = bRm then ["B"] else []

theorem matched_sound : VerifiedSound stMatched glMatched := by
  intro v l hl
  by_cases ha : v = aRm
  · subst ha
    rw [show lookupV stMatched.verified aRm = ["A"] from rfl, List.mem_singleton] at hl
    subst hl; decide
  · by_cases hb : v = bRm
    · subst hb
      rw [show lookupV stMatched.verified bRm = ["B"] from rfl, List.mem_singleton] at hl
      subst hl; decide
    · simp only [stMatched, markLabels, List.isEmpty_cons, Bool.false_eq_true, ↓reduceIte,
        lookupV_cons_other _ _ _ _ ha, lookupV_cons_other _ _ _ _ hb] at hl
      simp [lookupV] at hl

/-- `WITH a, b`. -/
def renAB : List (V × V) := [(aRm, a1), (bRm, b1)]

/-- The second MATCH's relationship `(a:A)-[:R]->(b)`, both endpoints bound. -/
def qAB : QR := { alias := ⟨2, 1⟩, named := false, frm := { alias := a1, labels := ["A"], ptr := 0 }, to := { alias := b1, ptr := 0 }, ptr := 0 }

/-- The repro of #3450: `MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH (a:A)-[:R]->(b)`.
After the REMOVE, `A` is no longer verified for `a`, so the bound endpoint of the second MATCH
gets its `hasLabels` filter (before #3457 none was planned: 1 row, C 0). -/
theorem remove_unverifies_label :
    (boundLabelFilters (project (rsRemove stMatched aRm ["A"]) renAB) [qAB]).2 =
      [hasLabelsFilter a1 ["A"]] ∧
    projectG (removeG glMatched aRm ["A"]) renAB a1 = [] ∧
    VerifiedSound (project (rsRemove stMatched aRm ["A"]) renAB)
      (projectG (removeG glMatched aRm ["A"]) renAB) :=
  ⟨rfl, by decide, project_sound _ _ _ (rsRemove_sound _ _ _ _ matched_sound)⟩

/-- `WITH n AS a, n AS c, b REMOVE c:A WITH a, b MATCH (a:A)-[:R]->(b)`: removing through
the other name unverifies `A` for `a` too. -/
theorem remove_other_name_unverifies :
    lookupV (rsRemove stMatched bRm ["A"]).verified aRm = [] := rfl

end PlannerBuild.E
