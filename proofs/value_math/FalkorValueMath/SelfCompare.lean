import FalkorValueMath.MathFns
/-!
# `compare_value`, `is_never_equal`, `may_fail_self_match`, `compare_vecf32`

A structural model of `CompareValue for Value` (value.rs:1224-1272) with its helpers
`compare_list` (:1399), `compare_map` (:1472), `compare_floats` (:1801) and
`compare_vecf32` (:1817), used here for two contracts the code states in comments:

1. `may_fail_self_match` (:1554) is a *conservative pre-filter*: when it says `false`,
   `compare_value(v, v)` is `(Equal, None)` — so `is_never_equal` (:1531) is exactly
   "the self-comparison is inconclusive".
2. The 16-lane block scan of `compare_vecf32` is the plain first-difference scan.

The ORDER BY / hash properties of the same comparator are in `proofs/value_order`.
Maps are held with keys already sorted and unique (the `sort()` of `compare_map`;
`OrderMap` keys are unique) — same assumption as proofs/value_order.
-/

namespace ValueMath

inductive DN where
  | disjoint | comparedNull | nan | none
  deriving DecidableEq

/-- What the comparator needs from `f64`/`f32` (`partial_cmp`, `!=`, `>`, `is_nan`). -/
structure FloatCmp (F : Type) where
  pc : F → F → Option Ordering
  ne : F → F → Bool
  gt : F → F → Bool
  isNan : F → Bool
  ofInt : Int → F
  pc_self : ∀ x, isNan x = false → pc x x = some .eq
  ne_self : ∀ x, isNan x = false → ne x x = false

variable {F : Type} (fc : FloatCmp F)

def orderF : V F → Nat
  | .null => 2 ^ 15 | .bool _ => 2 ^ 12 | .int _ => 2 ^ 13 | .float _ => 2 ^ 14
  | .str _ => 2 ^ 11 | .list _ => 2 ^ 3 | .map _ => 2 ^ 0 | .node _ => 2 ^ 1
  | .rel _ => 2 ^ 2 | .path _ => 2 ^ 4 | .point .. => 2 ^ 5 | .datetime _ => 2 ^ 6
  | .date _ => 2 ^ 7 | .time _ => 2 ^ 8 | .duration _ => 2 ^ 10 | .vec _ => 2 ^ 18

/-- `compare_floats` (value.rs:1801). -/
def cmpFloats (a b : F) : Ordering × DN :=
  match fc.pc a b with
  | some o => (o, .none)
  | none => (.lt, .nan)

/-- `first_difference` (value.rs:1821). -/
def firstDiff : List F → List F → Ordering
  | x :: xs, y :: ys => if fc.ne x y then (if fc.gt x y then .gt else .lt) else firstDiff xs ys
  | _, _ => .eq

/-- `compare_vecf32`'s block loop (value.rs:1841-1848) for equal lengths. -/
def chunked (a b : List F) : Ordering :=
  if h : 16 ≤ a.length then
    if ((a.take 16).zip (b.take 16)).any (fun p => fc.ne p.1 p.2) then firstDiff fc (a.take 16) (b.take 16)
    else chunked (a.drop 16) (b.drop 16)
  else firstDiff fc a b
  termination_by a.length
  decreasing_by simp; omega

def compareVec (a b : List F) : Ordering :=
  match compare a.length b.length with
  | .eq => chunked fc a b
  | o => o

theorem firstDiff_append (x y x' y' : List F) (hl : x.length = x'.length) :
    firstDiff fc (x ++ y) (x' ++ y') =
      if (x.zip x').any (fun p => fc.ne p.1 p.2) then firstDiff fc x x' else firstDiff fc y y' := by
  induction x generalizing x' with
  | nil => cases x' <;> simp_all [firstDiff]
  | cons a as ih =>
    cases x' with
    | nil => simp at hl
    | cons b bs =>
      simp only [List.cons_append, List.zip_cons_cons, List.any_cons, firstDiff]
      split
      · rename_i h; simp [h]
      · rename_i h
        simp only [Bool.not_eq_true] at h
        simp only [h, Bool.false_or]
        exact ih bs (by simpa using hl)

/-- The vectorised block scan equals the element-by-element first-difference scan. -/
theorem chunked_eq_firstDiff (a b : List F) (hl : a.length = b.length) :
    chunked fc a b = firstDiff fc a b := by
  suffices H : ∀ n (a b : List F), a.length ≤ n → a.length = b.length → chunked fc a b = firstDiff fc a b from
    H a.length a b (Nat.le_refl _) hl
  intro n
  induction n with
  | zero =>
    intro a b ha hl
    rw [chunked, dif_neg (by omega)]
  | succ n ih0 =>
    intro a b ha hl
    have ih : ∀ m, m < a.length → ∀ (a' b' : List F), a'.length = b'.length → a'.length = m →
        chunked fc a' b' = firstDiff fc a' b' := fun m hm a' b' hl' he => ih0 a' b' (by omega) hl'
    rw [chunked]
    split
    · rename_i h16
      have e1 := List.take_append_drop 16 a
      have e2 := List.take_append_drop 16 b
      have hl' : (a.take 16).length = (b.take 16).length := by simp; omega
      conv => rhs; rw [← e1, ← e2]
      rw [firstDiff_append fc _ _ _ _ hl']
      split
      · rfl
      · exact ih _ (by simp; omega) _ _ (by simp; omega) rfl
    · rfl

mutual
/-- `compare_value` (value.rs:1224), arms in source order. -/
def cmpV : V F → V F → Ordering × DN
  | .bool a, .bool b => (compare a b, .none)
  | .float a, .float b => cmpFloats fc a b
  | .str a, .str b => (compare a b, .none)
  | .list a, .list b => cmpList a b
  | .path a, .path b => cmpList a b
  | .map a, .map b => cmpMap a b
  | .vec a, .vec b => (compareVec fc a b, .none)
  | .node a, .node b => (compare a b, .none)
  | .rel a, .rel b => (compare a b, .none)
  | .point la1 lo1, .point la2 lo2 =>
    match fc.pc lo1 lo2 with
    | some .eq => match fc.pc la1 la2 with
      | some o => (o, .none)
      | none => (.lt, .nan)
    | some o => (o, .none)
    | none => (.lt, .nan)
  | .int a, .int b | .datetime a, .datetime b | .date a, .date b | .time a, .time b
  | .duration a, .duration b => (compare a b, .none)
  | .int i, .float f => cmpFloats fc (fc.ofInt i) f
  | .float f, .int i => cmpFloats fc f (fc.ofInt i)
  | .null, b => (compare (orderF (V.null : V F)) (orderF b), .comparedNull)
  | a, .null => (compare (orderF a) (orderF (V.null : V F)), .comparedNull)
  | a, b => (compare (orderF a) (orderF b), .disjoint)
  termination_by a _ => (sizeOf a, 2)

/-- `compare_list` (value.rs:1399-1470). -/
def cmpList (a b : List (V F)) : Ordering × DN :=
  if a.length = 0 ∧ b.length = 0 then (.eq, .none) else
  let minLen := min a.length b.length
  let (fne, nullC, neC, incC) := listLoop2 a b (.eq, 0, 0, 0)
  if neC = minLen ∧ nullC < neC ∧ fne ≠ .eq then (fne, .none)
  else if nullC > 0 ∧ a.length = b.length then (fne, .comparedNull)
  else if incC > 0 ∧ fne = .eq ∧ a.length = b.length then (.eq, .disjoint)
  else if fne ≠ .eq then (fne, .none)
  else (compare a.length b.length, .none)
  termination_by (sizeOf a, 1)

/-- The zip loop: (first_not_equal, null_counter, not_equal_counter, inconclusive_counter). -/
def listLoop2 : List (V F) → List (V F) → Ordering × Nat × Nat × Nat → Ordering × Nat × Nat × Nat
  | x :: xs, y :: ys, (fne, nc, ne, ic) =>
    let (r, d) := cmpV x y
    let st := if d ≠ .none then
        (if fne = .eq then r else fne, if d = .comparedNull then nc + 1 else nc, ne + 1,
         if d = .comparedNull then ic else ic + 1)
      else if r ≠ .eq then (if fne = .eq then r else fne, nc, ne + 1, ic)
      else (fne, nc, ne, ic)
    listLoop2 xs ys st
  | _, _, st => st
  termination_by a _ _ => (sizeOf a, 0)

/-- `compare_map` (value.rs:1472) on key-sorted entry lists. -/
def cmpMap (a b : List (String × V F)) : Ordering × DN :=
  if a.length ≠ b.length then (compare a.length b.length, .none) else
  match keysCmp a b with
  | some o => (o, .none)
  | none => mapLoop a b
  termination_by (sizeOf a, 1)

def keysCmp : List (String × V F) → List (String × V F) → Option Ordering
  | (k, _) :: as, (k', _) :: bs => if k ≠ k' then some (compare k k') else keysCmp as bs
  | _, _ => none
  termination_by a _ => (sizeOf a, 0)

/-- The values loop: first disjoint/null ends with `Equal`, first unequal decides. -/
def mapLoop : List (String × V F) → List (String × V F) → Ordering × DN
  | (_, x) :: as, (_, y) :: bs =>
    let (r, d) := cmpV x y
    if d = .comparedNull ∨ d = .disjoint then (.eq, d)
    else if r ≠ .eq then (r, d)
    else mapLoop as bs
  | _, _ => (.eq, .none)
  termination_by a _ => (sizeOf a, 0)
end

mutual
/-- `may_fail_self_match` (value.rs:1554). -/
def mayFail : V F → Bool
  | .bool _ | .int _ | .str _ | .node _ | .rel _ | .datetime _ | .date _ | .time _
  | .duration _ => false
  | .float f => fc.isNan f
  | .vec v => v.any fc.isNan
  | .list items => mayFailList items
  | .path items => mayFailList items
  | .map entries => mayFailMap entries
  | _ => true
def mayFailList : List (V F) → Bool
  | [] => false
  | x :: xs => mayFail x || mayFailList xs
def mayFailMap : List (String × V F) → Bool
  | [] => false
  | (_, x) :: xs => mayFail x || mayFailMap xs
end

/-- `is_never_equal` (value.rs:1531). -/
def isNeverEqual (v : V F) : Bool := mayFail fc v && !(decide (cmpV fc v v = (.eq, .none)))

theorem listLoop2_self (xs : List (V F)) (h : ∀ x ∈ xs, cmpV fc x x = (.eq, .none)) :
    listLoop2 fc xs xs (.eq, 0, 0, 0) = (.eq, 0, 0, 0) := by
  induction xs with
  | nil => simp [listLoop2]
  | cons x xs ih =>
    rw [listLoop2]
    simp [h x (by simp), ih (fun y hy => h y (by simp [hy]))]

theorem keysCmp_self (xs : List (String × V F)) : keysCmp fc xs xs = none := by
  induction xs with
  | nil => simp [keysCmp]
  | cons p xs ih => obtain ⟨k, v⟩ := p; rw [keysCmp]; simp [ih]

theorem mapLoop_self (xs : List (String × V F)) (h : ∀ p ∈ xs, cmpV fc p.2 p.2 = (.eq, .none)) :
    mapLoop fc xs xs = (.eq, .none) := by
  induction xs with
  | nil => simp [mapLoop]
  | cons p xs ih =>
    obtain ⟨k, v⟩ := p
    rw [mapLoop]
    simp [h (k, v) (by simp), ih (fun q hq => h q (by simp [hq]))]

theorem firstDiff_self (v : List F) (h : v.any fc.isNan = false) : firstDiff fc v v = .eq := by
  induction v with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.any_cons, Bool.or_eq_false_iff] at h
    simp [firstDiff, fc.ne_self x h.1, ih h.2]

theorem compare_self_eq {α} [Ord α] [Std.ReflCmp (compare : α → α → Ordering)] (a : α) :
    compare a a = .eq := Std.ReflCmp.compare_self

/-- **Soundness of the pre-filter.** If `may_fail_self_match(v)` is false, the
self-comparison is conclusive-equal, hence `is_never_equal(v) = false` is correct. -/
theorem mayFail_sound : ∀ (n : Nat) (v : V F), sizeOf v < n → mayFail fc v = false →
    cmpV fc v v = (.eq, .none) := by
  intro n
  induction n with
  | zero => intro v h; omega
  | succ n ih =>
    intro v hs hm
    have hlist : ∀ items : List (V F), sizeOf items < n → mayFailList fc items = false →
        ∀ x ∈ items, cmpV fc x x = (.eq, .none) := by
      intro items hsz hmf x hx
      have hx' : sizeOf x < sizeOf items := List.sizeOf_lt_of_mem hx
      have : mayFail fc x = false := by
        clear hsz hx'
        induction items with
        | nil => simp at hx
        | cons y ys ihy =>
          simp only [mayFailList, Bool.or_eq_false_iff] at hmf
          rcases List.mem_cons.1 hx with rfl | h
          · exact hmf.1
          · exact ihy hmf.2 h
      exact ih x (by omega) this
    cases v with
    | bool b => simp [cmpV]
    | int i => simp [cmpV]
    | str s => simp [cmpV]
    | node i => simp [cmpV]
    | rel i => simp [cmpV]
    | datetime t => simp [cmpV]
    | date t => simp [cmpV]
    | time t => simp [cmpV]
    | duration t => simp [cmpV]
    | float f =>
      simp only [mayFail] at hm
      simp [cmpV, cmpFloats, fc.pc_self f hm]
    | vec v =>
      simp only [mayFail] at hm
      simp only [cmpV, compareVec, compare_self_eq]
      rw [chunked_eq_firstDiff fc v v rfl, firstDiff_self fc v hm]
    | list items =>
      simp only [mayFail] at hm
      have hall := hlist items (by simp at hs; omega) hm
      rw [cmpV, cmpList, listLoop2_self fc items hall]
      by_cases h0 : items.length = 0 <;> simp [h0, compare_self_eq]
    | path items =>
      simp only [mayFail] at hm
      have hall := hlist items (by simp at hs; omega) hm
      rw [cmpV, cmpList, listLoop2_self fc items hall]
      by_cases h0 : items.length = 0 <;> simp [h0, compare_self_eq]
    | map entries =>
      simp only [mayFail] at hm
      have hall : ∀ p ∈ entries, cmpV fc p.2 p.2 = (.eq, .none) := by
        intro p hp
        have hp' : sizeOf p < sizeOf entries := List.sizeOf_lt_of_mem hp
        have hp2 : sizeOf p.2 < sizeOf p := by obtain ⟨k, v⟩ := p; simp; omega
        have : mayFail fc p.2 = false := by
          clear hp' hp2 hs
          induction entries with
          | nil => simp at hp
          | cons q qs ihq =>
            obtain ⟨k, w⟩ := q
            simp only [mayFailMap, Bool.or_eq_false_iff] at hm
            rcases List.mem_cons.1 hp with rfl | h
            · exact hm.1
            · exact ihq hm.2 h
        exact ih p.2 (by simp at hs; omega) this
      rw [cmpV, cmpMap]
      simp [keysCmp_self fc, mapLoop_self fc entries hall]
    | null => simp [mayFail] at hm
    | point la lo => simp [mayFail] at hm

theorem isNeverEqual_iff (v : V F) :
    isNeverEqual fc v = true ↔ cmpV fc v v ≠ (.eq, .none) := by
  unfold isNeverEqual
  cases h : mayFail fc v
  · simp [mayFail_sound fc (sizeOf v + 1) v (by omega) h]
  · simp

end ValueMath
