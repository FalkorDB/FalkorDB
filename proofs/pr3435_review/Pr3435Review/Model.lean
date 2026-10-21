import Pr3435Review.AlgoBfs
/-! # The FFI hypotheses are satisfiable (no vacuous theorems)

`gbModel n` is a reference implementation of the four GraphBLAS calls on
size-`n` vectors; `gbModel_spec` proves it meets `GBSpec n`. `lagModel` (our own
push/pull over `gbModel`, always pushing) meets `LagSpec n` for `n < 2^31`. Small
`decide` checks run the loop with an always-push, an always-pull and an
alternating heuristic and get the same levels.
-/
namespace Pr3435.Model
open Pr3435.Bfs Pr3435.Algo

def gbModel (n : Nat) : GB where
  vxm q v A := some (fun j => decide (j < n) && (v j).isNone && (List.range n).any (fun i => q i && A i j))
  mxv q v AT := some (fun j => decide (j < n) && (v j).isNone && (List.range n).any (fun i => AT j i && q i))
  assign v q k := some (fun j => if q j then some k else v j)
  nvals q := count n q

theorem gbModel_spec (n : Nat) : GBSpec n (gbModel n) where
  vxm q v A q' h j := by
    simp only [gbModel, Option.some.injEq] at h; subst h
    simp [List.any_eq_true, Option.isNone_iff_eq_none, and_assoc]
  mxv q v AT q' h j := by
    simp only [gbModel, Option.some.injEq] at h; subst h
    simp [List.any_eq_true, Option.isNone_iff_eq_none, and_assoc]
  assign v q k v' h j := by simp only [gbModel, Option.some.injEq] at h; subst h; rfl
  nvals q := rfl

def pushAlways : Heur Unit := ⟨fun _ _ _ _ => true, fun _ _ => ()⟩
def pullAlways : Heur Unit := ⟨fun _ _ _ _ => false, fun _ _ => ()⟩
def alternate : Heur Unit := ⟨fun _ _ _ p => !p, fun _ _ => ()⟩

/-- LAGraph's spec is met by the push/pull loop itself (so `LagSpec` is consistent). -/
def lagModel (n : Nat) : Pat → Nat → Int → Bool → Option LVec := fun A src ml _ =>
  match bfsLevelsPushPull (gbModel n) pushAlways () A (fun j i => A i j) n src ml with
  | .ok w => some w
  | .error _ => none

theorem lagModel_spec (n : Nat) (hn : n < 2147483648) : LagSpec n (lagModel n) where
  level A src ml me w hb hs h := by
    unfold lagModel at h
    generalize hr : bfsLevelsPushPull (gbModel n) pushAlways () A (fun j i => A i j) n src ml = r at h
    cases r with
    | ok w' =>
      cases h
      exact (pushPull_correct (gbModel_spec n) pushAlways () (fun _ _ => rfl) hb hs hn ml).1 _ hr
    | error _ => cases h

/-! ## Checks on a small graph: 0→1, 1→2, 0→3, 3→2, 4→0 (node 4 unreachable), n = 5 -/

def edges5 : List (Nat × Nat) := [(0, 1), (1, 2), (0, 3), (3, 2), (4, 0)]
def A5 : Pat := patOf edges5
def AT5 : Pat := fun j i => A5 i j

def levelsOf (hr : Heur Unit) (ml : Int) : List (Option Int) :=
  match bfsLevelsPushPull (gbModel 5) hr () A5 AT5 5 0 ml with
  | .ok w => (List.range 5).map w
  | .error _ => []

example : levelsOf pushAlways (-1) = [some 0, some 1, some 2, some 1, none] := by decide
example : levelsOf pullAlways (-1) = [some 0, some 1, some 2, some 1, none] := by decide
example : levelsOf alternate (-1) = [some 0, some 1, some 2, some 1, none] := by decide
example : levelsOf pullAlways 1 = [some 0, some 1, none, some 1, none] := by decide
example : levelsOf alternate 0 = [some 0, none, none, none, none] := by decide

end Pr3435.Model
