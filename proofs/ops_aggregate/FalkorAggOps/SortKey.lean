/-
`classify_sort_key` (sort.rs:203) over `classify_numeric(_, false, FloatLane::Pure)`
(batch.rs:191). Floats are an opaque type `F` (no arithmetic is done on them).

Proven: the classified key column reads back, row for row, exactly the evaluated key
with nulls re-inserted (`classifySortKey_get`), so typed-lane comparisons see the same
values as `Value::compare_value` on the `Values` lane.
-/
namespace FalkorAggOps.SortKey

variable {F : Type}

inductive SV (F : Type) where
  | null
  | int (i : Int)
  | float (f : F)
  | other (n : Nat)

inductive Col (F : Type) where
  | ints (v : List Int)
  | floats (v : List F)
  | values (v : List (SV F))

/-- `Column::get` (batch.rs) as a value list. -/
def Col.toValues : Col F → List (SV F)
  | .ints v => v.map .int
  | .floats v => v.map .float
  | .values v => v

def asInt : SV F → Option Int | .int i => some i | _ => none
def asFloat : SV F → Option F | .float f => some f | _ => none

/-- batch.rs:191-231 with `allow_null = false`, `FloatLane::Pure`: the `all(..)` loops that
push while testing are `mapM`. -/
def classifyPure (vs : List (SV F)) : Col F :=
  match vs.mapM asInt with
  | some is => .ints is
  | none =>
    match vs.mapM asFloat with
    | some fs => .floats fs
    | none => .values vs

theorem mapM_asInt (vs : List (SV F)) (is : List Int) (h : vs.mapM asInt = some is) :
    is.map SV.int = vs := by
  induction vs generalizing is with
  | nil => simp at h; subst h; rfl
  | cons v vs ih =>
    cases v <;> simp only [List.mapM_cons, asInt] at h <;> try (simp at h)
    rename_i x
    cases ht : vs.mapM asInt with
    | none => simp [ht] at h
    | some t =>
      simp [ht] at h; subst h
      simp [ih t ht]

theorem mapM_asFloat (vs : List (SV F)) (fs : List F) (h : vs.mapM asFloat = some fs) :
    fs.map SV.float = vs := by
  induction vs generalizing fs with
  | nil => simp at h; subst h; rfl
  | cons v vs ih =>
    cases v <;> simp only [List.mapM_cons, asFloat] at h <;> try (simp at h)
    rename_i x
    cases ht : vs.mapM asFloat with
    | none => simp [ht] at h
    | some t =>
      simp [ht] at h; subst h
      simp [ih t ht]

/-- `classify_numeric(_, false, Pure)` is lossless. -/
theorem classifyPure_lossless (vs : List (SV F)) : (classifyPure vs).toValues = vs := by
  unfold classifyPure
  split
  · rename_i is h; exact mapM_asInt vs is h
  · split
    · rename_i fs h; exact mapM_asFloat vs fs h
    · rfl

/-- sort.rs:203-231. `nulls` = the evaluation's `NullBitmap` as a per-row flag list. -/
def classifySortKey (col : Col F) (nulls : List Bool) : Col F :=
  if nulls.any id then
    .values ((List.range col.toValues.length).map fun i =>
      if nulls.getD i false then .null else col.toValues.getD i .null)
  else
    match col with
    | .ints v => .ints v
    | .floats v => .floats v
    | other => classifyPure other.toValues

/-- Every row of the classified key is the evaluated value, or `Null` where the bitmap says
null; without nulls the column's values are unchanged. -/
theorem classifySortKey_get (col : Col F) (nulls : List Bool) :
    (classifySortKey col nulls).toValues =
      (List.range col.toValues.length).map fun i =>
        if nulls.getD i false then .null else col.toValues.getD i .null := by
  unfold classifySortKey
  split
  · rfl
  · rename_i hn
    have hall : ∀ i, nulls.getD i false = false := by
      intro i
      cases hi : nulls.getD i false
      · rfl
      · exfalso; apply hn
        simp only [List.getD_eq_getElem?_getD] at hi
        cases hg : nulls[i]? with
        | none => simp [hg] at hi
        | some b =>
          simp [hg] at hi; subst hi
          exact List.any_eq_true.mpr ⟨true, List.mem_of_getElem? hg, rfl⟩
    simp only [hall]
    have hid : ∀ l : List (SV F), (List.range l.length).map (fun i => l.getD i .null) = l := by
      intro l
      apply List.ext_getElem <;> simp
      intro i h1 _; simp [List.getElem?_eq_getElem h1]
    have : (match col with
        | .ints v => Col.ints v
        | .floats v => .floats v
        | other => classifyPure other.toValues).toValues = col.toValues := by
      cases col <;> simp only [Col.toValues]
      exact classifyPure_lossless _
    rw [this]; simp only [Bool.false_eq_true, ite_false]; exact (hid _).symm

end FalkorAggOps.SortKey
