import FalkorNativeIndex.Query
set_option linter.unusedSimpArgs false

/-!
# Predicates and union members for `describe_predicate` (`graph/src/index/falkordb/mod.rs`)

Under `index-falkordb` a predicate the native index declines (`query_numeric` = `None`) raises an
error, never an empty result (`nodeScanOp_error_iff`, `Scan.lean`). The message names the entity kind,
the label and the predicate's *shape* (`describe_predicate`, modelled as `Tag.describeK` with
`unsupportedByNativeIndex` in `Tag.lean`). This file holds what that model reads: `Pred`,
`union_members` and the one-attribute test.

`Pred` is the full `IndexQuery` (the other projects' `IndexQuery` folds `And`/`Point`/`ArrayContains`
into `other`, which is all the numeric column needs); `Pred.toQuery` is that fold.
-/

namespace NI

/-- `IndexQuery<Value>` (`index/mod.rs`) with every variant. -/
inductive Pred where
  | equal (key : String) (value : Value)
  | range (key : String) (min max : Option Value) (incMin incMax : Bool)
  | or (children : List Pred)
  | and (children : List Pred)
  | inList (key : String) (list : Option (List Value))
  | point (key : String)
  | arrayContains (key : String) (value : Value)

mutual
/-- The column's view of a predicate (`IndexQuery`). -/
def Pred.toQuery : Pred → IndexQuery
  | .equal k v => .equal k v
  | .range k mn mx a b => .range k mn mx a b
  | .or cs => .or (Pred.toQueries cs)
  | .and cs => .and (Pred.toQueries cs)
  | .inList k l => .inList k l
  | .point k => .point k
  | .arrayContains k v => .arrayContains k v
def Pred.toQueries : List Pred → List IndexQuery
  | [] => []
  | c :: cs => c.toQuery :: Pred.toQueries cs
end

mutual
/-- `union_members` (`mod.rs`): the attribute of every member of a union, with whether the
    native union serves that member kind (an `Equal` or an `IN` list, directly in the union). Nested
    combinators count by their attributes but are not served. -/
def unionMembers : Bool → List Pred → List (String × Bool)
  | _, [] => []
  | top, c :: cs =>
    (match c with
      | .equal k _ => [(k, top)]
      | .inList k _ => [(k, top)]
      | .or nested => unionMembers false nested
      | .and nested => unionMembers false nested
      | .range k .. => [(k, false)]
      | .point k => [(k, false)]
      | .arrayContains k _ => [(k, false)]) ++ unionMembers top cs
end

/-- The attribute every member names, if there is one (`one_key`, `mod.rs`). -/
def oneKey (ms : List (String × Bool)) : Option String :=
  (ms.head?.map Prod.fst).filter (fun first => ms.all (fun m => m.1 == first))

theorem oneKey_eq_some (ms : List (String × Bool)) (k : String) :
    oneKey ms = some k ↔ ms.head?.map Prod.fst = some k ∧ ∀ m ∈ ms, m.1 = k := by
  unfold oneKey
  cases hh : ms.head?.map Prod.fst with
  | none => simp
  | some k0 =>
    by_cases hall : ms.all (fun m => m.1 == k0) = true
    · have hall' : ∀ m ∈ ms, m.1 = k0 := fun m hm => by simpa using List.all_eq_true.1 hall m hm
      simp only [Option.filter, hall, ite_true, Option.some.injEq]
      constructor
      · rintro rfl; exact ⟨rfl, hall'⟩
      · rintro ⟨h, _⟩; exact h
    · have : ¬ ∀ m ∈ ms, m.1 = k0 := by
        intro h; apply hall; rw [List.all_eq_true]; intro m hm; simpa using h m hm
      simp only [Option.filter, hall, Bool.false_eq_true, ite_false, reduceCtorEq, false_iff]
      intro ⟨h1, h2⟩
      have hk : k0 = k := Option.some.inj h1
      subst hk
      exact this h2

end NI
