import FalkorAggOps.SortCmp
import FalkorAggOps.AggState
import FalkorValueOrder.Main
/-!
# Discharging the `compare_value`/`Hash` hypotheses on main (4a4f2a72f)

`SortCmp.lean` (`Lawful cv`) and `AggState.lean` (`hc`) take the value comparison as a
parameter. Until #2912/#2917 those parameters were `compare_value(..).0` / `Value::eq`,
for which the hypotheses are false, so the sort.rs `cmp_key`/`cmp`/`cmp` rows and the
aggregate.rs `GroupKey::hash` row were CONDITIONAL. On main:

* sort.rs sorts by `Value::sort_cmp` (#2912 b033aecc6): `cv = scMain`, a total preorder on
  every well-formed value (proofs/value_order `Main.scMain_total`, imported via the Lake path dependency `../value_order`).
* `GroupKey::eq` is `sort_cmp == Equal` and `Hash for Value` canonicalises NaN and f32
  components (#2917 7fdf06a0e): `hc` holds (proofs/value_order `Main.main_hash_consistent`).

Values are well-formed (`wf`: Ints are `i64`, Float `num`s representable, one value per
map key) — every `Value` the engine builds is.
-/
namespace FalkorAggOps.Discharge
open FalkorAggOps.SortCmp FalkorAggOps.AggPlan ValueOrder ValueOrder.SC ValueOrder.Main

/-- A well-formed `Value`. -/
abbrev WV := {v : V // wf v = true}

/-- `Value::sort_cmp` on well-formed values. -/
def cvMain (a b : WV) : Ordering := scMain a.1 b.1

/-- **`Lawful sort_cmp`** — the hypothesis of `cmpKey_lawful`, `heapCmp_swap`,
`heapCmp_refl`, discharged. -/
theorem lawful_main : Lawful cvMain :=
  ⟨fun a b => (scMain_total a.1 b.1 b.1 a.2 b.2 b.2).2.1,
   fun a => (scMain_total a.1 a.1 a.1 a.2 a.2 a.2).1⟩

theorem cvMain_trans (a b d : WV) : cvMain a b ≠ .gt → cvMain b d ≠ .gt → cvMain a d ≠ .gt :=
  (scMain_total a.1 b.1 d.1 a.2 b.2 d.2).2.2

/-- `OrderedKey::cmp_key` (sort.rs:76) on main: antisymmetric and reflexive in either
direction (`cmpKey_lawful` instantiated). -/
theorem cmpKey_lawful_main (d : Bool) : Lawful (fun x y => SortCmp.cmpKey cvMain ⟨x, d⟩ ⟨y, d⟩) :=
  cmpKey_lawful cvMain lawful_main d

/-- ... and transitive. -/
theorem cmpKey_trans_main (d : Bool) (a b c : WV) :
    SortCmp.cmpKey cvMain ⟨a, d⟩ ⟨b, d⟩ ≠ .gt → SortCmp.cmpKey cvMain ⟨b, d⟩ ⟨c, d⟩ ≠ .gt →
    SortCmp.cmpKey cvMain ⟨a, d⟩ ⟨c, d⟩ ≠ .gt := by
  cases d
  · exact cvMain_trans a b c
  · simp only [SortCmp.cmpKey, ite_true]
    rw [← lawful_main.swap a b, ← lawful_main.swap b c, ← lawful_main.swap a c]
    intro h1 h2
    exact cvMain_trans c b a h2 h1

/-- `Ord for HeapEntry` (sort.rs:173) on main: antisymmetric (shared directions) and
reflexive; with `heapCmp_seq_strict`, a strict total order on distinct arrivals. -/
theorem heapCmp_main (nul : WV) (x y : HEntry WV) (hd : x.keys.map (·.desc) = y.keys.map (·.desc)) :
    heapCmp cvMain nul y x = (heapCmp cvMain nul x y).swap ∧ heapCmp cvMain nul x x = .eq :=
  ⟨heapCmp_swap cvMain lawful_main nul x y hd, heapCmp_refl cvMain lawful_main nul x⟩

/-- `GroupKey::eq`'s element equality on main: `a.sort_cmp(b) == Equal`. -/
def veqMain (x y : WV) : Bool := scMain x.1 y.1 == .eq

/-- `Hash for Value` as the word stream fed to the hasher. -/
def hvMain (s : List UInt64) (x : WV) : List UInt64 := s ++ hwords x.1

/-- **`hc` discharged on main**: `Value`'s hash agrees with `GroupKey::eq`'s equality. -/
theorem hc_main : ∀ s x y, veqMain x y = true → hvMain s x = hvMain s y := by
  intro s x y h
  simp only [veqMain, beq_iff_eq] at h
  simp only [hvMain, main_hash_consistent x.1 y.1 x.2 y.2 h]

/-- **The `HashMap<GroupKey, _>` contract on main** (aggregate.rs:94/:110): equal keys hash
alike. -/
theorem gk_hash_consistent_main (st : List UInt64) (a b : List WV) (h : gkEq veqMain a b = true) :
    gkHash hvMain st a = gkHash hvMain st b :=
  gkHash_consistent veqMain hvMain hc_main st a b h

/-- `GroupKey::eq` (aggregate.rs:94) written directly: same length, and every zipped pair
`sort_cmp`-equal. -/
def gkEqR (a b : List V) : Bool :=
  a.length == b.length && (a.zip b).all (fun p => scMain p.1 p.2 == .eq)

/-- The generic `gkEq veqMain` is main's `GroupKey::eq` (from the retired pr2917_review). -/
theorem gkEq_spec : ∀ a b : List WV, gkEq veqMain a b = gkEqR (a.map (·.1)) (b.map (·.1))
  | [], [] => rfl
  | [], _ :: _ => by simp [gkEq, gkEqR]
  | _ :: _, [] => by simp [gkEq, gkEqR]
  | x :: xs, y :: ys => by
    have ih := gkEq_spec xs ys
    simp only [gkEqR, List.map_cons, List.length_cons, List.zip_cons_cons, List.all_cons] at ih ⊢
    simp only [gkEq, veqMain, ih, gkEqR]
    cases (scMain x.1 y.1 == .eq) <;> simp [Nat.add_right_cancel_iff]

/-- `GroupKey::eq` on main is an equivalence (needed by a `HashMap` key): it inherits
reflexivity, symmetry and transitivity from `sort_cmp`. -/
theorem veqMain_equiv (x y z : WV) :
    veqMain x x = true ∧ (veqMain x y = true → veqMain y x = true) ∧
    (veqMain x y = true → veqMain y z = true → veqMain x z = true) := by
  have T := scMain_total x.1 y.1 z.1 x.2 y.2 z.2
  have T' := scMain_total y.1 z.1 x.1 y.2 z.2 x.2
  have T'' := scMain_total z.1 y.1 x.1 z.2 y.2 x.2
  simp only [veqMain, beq_iff_eq]
  refine ⟨T.1, fun h => by rw [T.2.1, h]; rfl, fun h1 h2 => ?_⟩
  have le1 : scMain x.1 z.1 ≠ .gt := T.2.2 (by rw [h1]; simp) (by rw [h2]; simp)
  have hzy : scMain z.1 y.1 = .eq := by rw [T'.2.1, h2]; rfl
  have hyx : scMain y.1 x.1 = .eq := by rw [T.2.1, h1]; rfl
  have le2 : scMain z.1 x.1 ≠ .gt := T''.2.2 (by rw [hzy]; simp) (by rw [hyx]; simp)
  have sw := (scMain_total x.1 z.1 z.1 x.2 z.2 z.2).2.1
  rw [sw] at le2
  revert le1 le2
  cases scMain x.1 z.1 <;> simp [Ordering.swap]

#print axioms lawful_main
#print axioms cmpKey_trans_main
#print axioms heapCmp_main
#print axioms gk_hash_consistent_main
#print axioms veqMain_equiv

end FalkorAggOps.Discharge
