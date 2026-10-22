import PlannerBuild.PlanMatch2
/-
# `verified_labels` across REMOVE and WITH (#3450) — gap closed 2026-10-22

`MSt.verified` was only ever unioned (`markLabels`) and `remove_correct` treats the REMOVE
operator's graph effect as opaque (`specWrite`), so nobody stated that a verified label is
still on the node when a later operator reads the graph. Rust source of truth: origin/main
b84473a7f, `graph/src/planner/mod.rs`.

| here | there |
| --- | --- |
| `rsRemove`    | the `QueryIR::Remove` arm mod.rs:3212-3218: plans `IR::Remove`, `verified_labels` untouched |
| `specRemove`  | the fix: drop the removed labels from the variable's verified set |
| `project`     | `plan_project` mod.rs:2479-2492: the verified set of a variable projected as-is is carried to the new variable (`carried_verified`), everything else dropped |
| `boundLabelFilters` (PlanMatch2) | mod.rs:1840-1856: a bound endpoint gets a `hasLabels` filter only for `unverified_labels` (:1793-1805) |
| `NL`, `removeG`, `projectG` | the node labels (per variable of the current row) in the graph state the next operator reads; REMOVE's effect; WITH renaming |

Property `VerifiedSound st gl`: every label `verified_labels` records for a variable is on
that variable's node in `gl` — the invariant `unverified_labels` silently assumes when it
lets an `ExpandInto`/`CondTraverse` skip the label check. Planning a clause must preserve it
(`RemoveKeepsVerifiedSound`).

Live (main b84473a7f vs C), after `CREATE (:A)-[:R]->(:B)`:
`MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH (a:A)-[:R]->(b) RETURN labels(a), labels(b)`:
Rust 1 row `[] [B]`, C 0 rows.
-/
namespace PlannerBuild.E

/-- Node labels of the row's variables, as the graph the next operator reads has them. -/
abbrev NL := V → List String

def VerifiedSound (st : MSt) (gl : NL) : Prop :=
  ∀ v l, l ∈ lookupV st.verified v → l ∈ gl v

/-- REMOVE `v:ls` on the graph. -/
def removeG (gl : NL) (v : V) (ls : List String) : NL :=
  fun w => if w = v then (gl w).filter (fun l => !ls.contains l) else gl w

/-- mod.rs:3212-3218: the planner state is unchanged. -/
def rsRemove (st : MSt) (_v : V) (_ls : List String) : MSt := st

/-- The fix: `v`'s verified set loses the removed labels. -/
def specRemove (st : MSt) (v : V) (ls : List String) : MSt :=
  { st with verified := (v, (lookupV st.verified v).filter (fun l => !ls.contains l)) :: st.verified }

/-- **Property `RemoveKeepsVerifiedSound`**: planning `REMOVE v:ls` keeps `VerifiedSound`
for the graph the REMOVE leaves behind. -/
def RemoveKeepsVerifiedSound (plan : MSt → V → List String → MSt) : Prop :=
  ∀ st gl v ls, VerifiedSound st gl → VerifiedSound (plan st v ls) (removeG gl v ls)

theorem specRemove_sound : RemoveKeepsVerifiedSound specRemove := by
  intro st gl v ls h w l hl
  unfold removeG
  by_cases hw : w = v
  · subst hw
    simp only [specRemove, lookupV_cons_self, List.mem_filter] at hl
    rw [if_pos rfl, List.mem_filter]
    exact ⟨h _ _ hl.1, hl.2⟩
  · rw [specRemove, lookupV_cons_other _ _ _ _ hw] at hl
    simp only [if_neg hw]
    exact h _ _ hl

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
def qAB : QR := { alias := ⟨2, 1⟩, named := false, frm := { alias := a1, labels := ["A"] }, to := { alias := b1 } }

/-- **BUG #3450 (planner half)**: `MATCH (a:A), (b:B) REMOVE a:A WITH a, b MATCH
(a:A)-[:R]->(b)` — after the REMOVE the planner still has `A` verified for `a`, carries it
through `WITH`, and plans the bound endpoint with no `hasLabels` filter, although `a` has no
labels in the graph the `ExpandInto` reads (Rust 1 row, C 0). With `specRemove` the filter is
planned. `rsRemove` violates `RemoveKeepsVerifiedSound`. -/
theorem bug3450_remove_keeps_verified_label :
    (boundLabelFilters (project (rsRemove stMatched aRm ["A"]) renAB) [qAB]).2 = [] ∧
    projectG (removeG glMatched aRm ["A"]) renAB a1 = [] ∧
    (boundLabelFilters (project (specRemove stMatched aRm ["A"]) renAB) [qAB]).2 =
      [hasLabelsFilter a1 ["A"]] ∧
    ¬ RemoveKeepsVerifiedSound rsRemove := by
  refine ⟨rfl, by decide, rfl, fun h => ?_⟩
  have := h stMatched glMatched aRm ["A"] matched_sound aRm "A" (by decide)
  revert this; decide

end PlannerBuild.E
