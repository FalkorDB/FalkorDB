import FalkorNativeIndex.Query
set_option linter.unusedSimpArgs false

/-!
# `unsupported_by_native_index` and `describe_predicate` (`graph/src/index/falkordb/mod.rs:43-165`)

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
  | point (key : String)
  | arrayContains (key : String)

mutual
/-- The numeric column's view of a predicate: `And`, `Point`, `ArrayContains` are `other`. -/
def Pred.toQuery : Pred → IndexQuery
  | .equal k v => .equal k v
  | .range k mn mx a b => .range k mn mx a b
  | .or cs => .or (Pred.toQueries cs)
  | .and _ => .other
  | .inList k l => .inList k l
  | .point _ => .other
  | .arrayContains _ => .other
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
  | arrayContains (key : String)
deriving Repr

mutual
/-- `union_members` (`mod.rs:145-163`): the attribute of every member of a union, with whether the
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
      | .arrayContains k => [(k, false)]) ++ unionMembers top cs
end

/-- The attribute every member names, if there is one (`one_key`, `mod.rs:98-101`). -/
def oneKey (ms : List (String × Bool)) : Option String :=
  (ms.head?.map Prod.fst).filter (fun first => ms.all (fun m => m.1 == first))

mutual
/-- `describe_predicate` (`mod.rs:59-143`). -/
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
  | .point k => .geo k
  | .arrayContains k => .arrayContains k
def describeAll : List Pred → List Desc
  | [] => []
  | c :: cs => describe c :: describeAll cs
end

/-- `unsupported_by_native_index` (`mod.rs:43-57`): what the error names. -/
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

theorem describe_leaves (k : String) :
    describe (.point k) = .geo k ∧ describe (.arrayContains k) = .arrayContains k := ⟨rfl, rfl⟩

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
      simp only [Option.filter, hall, if_true, Option.some.injEq]
      constructor
      · rintro rfl; exact ⟨rfl, hall'⟩
      · rintro ⟨h, _⟩; exact h
    · have : ¬ ∀ m ∈ ms, m.1 = k0 := by
        intro h; apply hall; rw [List.all_eq_true]; intro m hm; simpa using h m hm
      simp only [Option.filter, hall, Bool.false_eq_true, if_false, reduceCtorEq, false_iff]
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

/-! ### The reasons are sound (a reason is only given for a union the column really declines) -/

/-- If every child of the union is a member the column serves on attribute `a`, the flattened members
    are all `(a, true)`. -/
theorem unionMembers_sameAttr (a : String) : ∀ (cs : List Pred), SameAttr a (Pred.toQueries cs) →
    ∀ top, unionMembers top cs = cs.map (fun _ => (a, top)) := by
  intro cs
  induction cs with
  | nil => intro _ top; simp [unionMembers]
  | cons c cs ih =>
    intro h top
    have hc := h c.toQuery (by simp [Pred.toQueries])
    obtain ⟨vs, hvs⟩ := hc
    have hrest : SameAttr a (Pred.toQueries cs) := fun q hq => h q (by simp [Pred.toQueries, hq])
    simp only [List.map_cons]
    cases c with
    | equal k v =>
      simp only [Pred.toQuery, unionMember, Option.some.injEq, Prod.mk.injEq] at hvs
      obtain ⟨rfl, _⟩ := hvs
      simp [unionMembers, ih hrest top]
    | inList k l =>
      cases l with
      | none => simp [Pred.toQuery, unionMember] at hvs
      | some items =>
        simp only [Pred.toQuery, unionMember] at hvs
        by_cases he : items.isEmpty = true
        · simp [he] at hvs
        · simp only [he, Bool.false_eq_true, if_false, Option.some.injEq, Prod.mk.injEq] at hvs
          obtain ⟨rfl, _⟩ := hvs
          simp [unionMembers, ih hrest top]
    | range k mn mx a' b' => simp [Pred.toQuery, unionMember] at hvs
    | or _ => simp [Pred.toQuery, unionMember] at hvs
    | and _ => simp [Pred.toQuery, unionMember] at hvs
    | point _ => simp [Pred.toQuery, unionMember] at hvs
    | arrayContains _ => simp [Pred.toQuery, unionMember] at hvs

/-- A union the column serves is non-empty, on one attribute, and every member is served: so
    `describe_predicate` would call it "not all-numeric" (never "spanning"/"non-member"). -/
theorem unionKeys_served (cs : List Pred) (ks : KeySet) (h : NumericIndex.unionKeys (Pred.toQueries cs) = some ks) :
    cs ≠ [] ∧ ∃ a, unionMembers true cs = cs.map (fun _ => (a, true)) := by
  unfold NumericIndex.unionKeys at h
  have hne : cs ≠ [] := by
    rintro rfl; simp [Pred.toQueries] at h
  refine ⟨hne, ?_⟩
  by_cases hc : (Pred.toQueries cs).isEmpty = true
  · exfalso; apply hne; cases cs with
    | nil => rfl
    | cons c cs => simp [Pred.toQueries] at hc
  · simp only [hc, Bool.false_eq_true, if_false] at h
    obtain ⟨out, hg⟩ : ∃ out, unionGo none [] (Pred.toQueries cs) = some out := by
      cases hg : unionGo none [] (Pred.toQueries cs) with
      | none => simp [hg] at h
      | some out => exact ⟨out, rfl⟩
    obtain ⟨a, _, hs, _⟩ := unionGo_spec _ none [] out hg
    exact ⟨a, unionMembers_sameAttr a cs hs true⟩

/-- **A "spanning more than one attribute" or "member that is not an equality or IN list" report is
    always a real decline.** -/
theorem describe_or_reason_sound (cs : List Pred) :
    (describe (.or cs) = .orSpanning cs.length →
      NumericIndex.unionKeys (Pred.toQueries cs) = none) ∧
    (∀ key, describe (.or cs) = .orNonMember cs.length key →
      NumericIndex.unionKeys (Pred.toQueries cs) = none) := by
  have key : NumericIndex.unionKeys (Pred.toQueries cs) ≠ none →
      oneKey (unionMembers true cs) ≠ none ∧ (unionMembers true cs).all (·.2) = true := by
    intro hne
    obtain ⟨ks, hks⟩ := Option.ne_none_iff_exists'.1 hne
    obtain ⟨hcs, a, hm⟩ := unionKeys_served cs ks hks
    rw [hm]
    cases cs with
    | nil => exact absurd rfl hcs
    | cons c cs =>
      refine ⟨?_, by simp⟩
      simp only [List.map_cons, oneKey]
      simp [Option.filter]
  refine ⟨fun h => ?_, fun k h => ?_⟩
  · by_cases hn : NumericIndex.unionKeys (Pred.toQueries cs) = none
    · exact hn
    · exact absurd ((describe_or cs).1.1 h) (key hn).1
  · by_cases hn : NumericIndex.unionKeys (Pred.toQueries cs) = none
    · exact hn
    · exact absurd (key hn).2 (((describe_or cs).2.1 k).1 h).2

/-- The one inexact reason (cosmetic, not reachable from the planner today: the scan op drops what it
    cannot index before a list gets here): `a = 1 OR a IN []` is reported "not all-numeric", but every
    value encodes; the real reason is the empty `IN` list. -/
theorem describe_empty_in_misreported (v : Value) :
    describe (.or [.equal "a" v, .inList "a" (some [])]) = .orNotAllNumeric 2 "a" ∧
    NumericIndex.unionKeys (Pred.toQueries [.equal "a" v, .inList "a" (some [])]) = none := by
  refine ⟨by simp [describe, unionMembers, oneKey, Option.filter], ?_⟩
  simp only [Pred.toQueries, Pred.toQuery, NumericIndex.unionKeys, List.isEmpty_cons, Bool.false_eq_true, if_false]
  cases hh : encodeAll (nonNull [v]) <;> simp [unionGo, unionMember, hh]

end NI
