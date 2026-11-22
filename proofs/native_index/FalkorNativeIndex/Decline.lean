import FalkorNativeIndex.Query
set_option linter.unusedSimpArgs false

/-!
# `unsupported_by_native_index` and `describe_predicate` (`graph/src/index/falkordb/mod.rs`)

Under `index-falkordb` a predicate the native index declines (`query_numeric` = `None`) raises an
error, never an empty result (`nodeScanOp_error_iff`, `Scan.lean`). The message names the entity kind,
the label and the predicate's *shape* (`describe_predicate`), so a CI failure says what is missing. The
Rust returns text; the model returns the reason as a value (`Desc`), which is what the tests in
`mod.rs` check.

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
  /-- `Point { key, point, radius }`; `cover` as in `IndexQuery.point`. -/
  | point (key : String) (centre radius : Value) (cover : List (Nat × Nat))
  | arrayContains (key : String) (value : Value)

mutual
/-- The column's view of a predicate (`IndexQuery`). -/
def Pred.toQuery : Pred → IndexQuery
  | .equal k v => .equal k v
  | .range k mn mx a b => .range k mn mx a b
  | .or cs => .or (Pred.toQueries cs)
  | .and cs => .and (Pred.toQueries cs)
  | .inList k l => .inList k l
  | .point k c r cv => .point k c r cv
  | .arrayContains k v => .arrayContains k v
def Pred.toQueries : List Pred → List IndexQuery
  | [] => []
  | c :: cs => c.toQuery :: Pred.toQueries cs
end

/-- A predicate's shape, as `describe_predicate` words it. -/
inductive Desc where
  | equality (key : String) (numeric : Bool)
  | range (key : String) (numeric : Bool)
  | orSpanning (n : Nat)
  | orNotAllNumeric (n : Nat) (key : String)
  | orNonMember (n : Nat) (key : String)
  | and (children : List Desc)
  | inList (key : String) (numeric : Bool)
  | geo (key : String)
  | arrayContains (key : String) (numeric : Bool)
deriving Repr

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
      | .point k .. => [(k, false)]
      | .arrayContains k _ => [(k, false)]) ++ unionMembers top cs
end

/-- The attribute every member names, if there is one (`one_key`, `mod.rs`). -/
def oneKey (ms : List (String × Bool)) : Option String :=
  (ms.head?.map Prod.fst).filter (fun first => ms.all (fun m => m.1 == first))

mutual
/-- `describe_predicate` (`mod.rs`). -/
def describe : Pred → Desc
  | .equal k v => .equality k (encodeNumeric v).isSome
  | .range k mn mx _ _ =>
    .range k (mn.all (fun v => (encodeNumeric v).isSome) && mx.all (fun v => (encodeNumeric v).isSome))
  | .or cs =>
    let members := unionMembers true cs
    let n := cs.length
    match oneKey members with
    | none => .orSpanning n
    | some key => if members.all (·.2) then .orNotAllNumeric n key else .orNonMember n key
  | .and cs => .and (describeAll cs)
  | .inList k l =>
    .inList k (match l with
      | some items => items.all (fun v => (encodeNumeric v).isSome)
      | none => false)
  | .point k .. => .geo k
  | .arrayContains k v => .arrayContains k (encodeNumeric v).isSome
def describeAll : List Pred → List Desc
  | [] => []
  | c :: cs => describe c :: describeAll cs
end

/-- `unsupported_by_native_index` (`mod.rs`): what the error names. -/
structure Unsupported where
  entity : String
  label : String
  shape : Desc

def unsupportedByNativeIndex (entity label : String) (q : Pred) : Unsupported := ⟨entity, label, describe q⟩

/-- The error carries the entity kind, the label/type and the predicate's shape. -/
theorem unsupported_spec (entity label : String) (q : Pred) :
    (unsupportedByNativeIndex entity label q).entity = entity ∧
    (unsupportedByNativeIndex entity label q).label = label ∧
    (unsupportedByNativeIndex entity label q).shape = describe q := ⟨rfl, rfl, rfl⟩

/-- Encodability is tested, not assumed: a declined `Equal` is "non-numeric" only if its value does
    not encode (it can be a servable leaf inside a declined combination). -/
theorem describe_equal (k : String) (v : Value) : describe (.equal k v) = .equality k (encodeNumeric v).isSome := rfl

theorem describe_inList (k : String) (items : List Value) :
    describe (.inList k (some items)) = .inList k (items.all (fun v => (encodeNumeric v).isSome)) := rfl

theorem describe_leaves (k : String) (v : Value) (c r : Value) (cv : List (Nat × Nat)) :
    describe (.point k c r cv) = .geo k ∧ describe (.arrayContains k v) = .arrayContains k (encodeNumeric v).isSome :=
  ⟨rfl, rfl⟩

/-- A conjunction names its children (not just their count). -/
theorem describe_and (cs : List Pred) : describe (.and cs) = .and (describeAll cs) := rfl

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

/-- A union is described by one of three reasons, decided by its flattened members: more than one
    attribute; one attribute with a member that is not an equality / `IN` list; one attribute, all
    served shapes (so a value is what is not numeric). -/
theorem describe_or (cs : List Pred) :
    (describe (.or cs) = .orSpanning cs.length ↔ oneKey (unionMembers true cs) = none) ∧
    (∀ key, describe (.or cs) = .orNonMember cs.length key ↔
      oneKey (unionMembers true cs) = some key ∧ ¬ (unionMembers true cs).all (·.2) = true) ∧
    (∀ key, describe (.or cs) = .orNotAllNumeric cs.length key ↔
      oneKey (unionMembers true cs) = some key ∧ (unionMembers true cs).all (·.2) = true) := by
  simp only [describe]
  generalize unionMembers true cs = ms
  cases hk : oneKey ms with
  | none => simp
  | some k =>
    by_cases hs : ms.all (·.2) = true
    · simp [hs]
    · simp [hs]

/-! ### Reasons and the single-column union

`Selection::disjunction` is the single-column union. Since #3737 a union spanning attributes is served by
`combine_columns` (`queryColumn`, one column per attribute), so "spanning more than one attribute" is
reported only when a column is missing or a share is unservable: the wording is a hint, not the claim
that the union is unservable. What remains true of the one-column union: -/

/-- The one inexact reason (cosmetic, not reachable from the planner today: the scan op drops what it
    cannot index before a list gets here): `a = 1 OR a IN []` is reported "not all-numeric", but every
    value encodes; the real reason is the empty `IN` list, which declines the union. -/
theorem describe_empty_in_misreported (v : Value) :
    describe (.or [.equal "a" v, .inList "a" (some [])]) = .orNotAllNumeric 2 "a" ∧
    Selection.disjunction [.equal "a" v, .inList "a" (some [])] = none := by
  refine ⟨by simp [describe, unionMembers, oneKey, Option.filter], ?_⟩
  unfold Selection.disjunction
  split
  · rfl
  · simp only [Option.map_eq_none_iff]
    cases hh : pushEqualKeys [v] [] <;> simp [disjGo, disjMemberValues, hh]

end NI
