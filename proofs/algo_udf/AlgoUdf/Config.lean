import AlgoUdf.Value
/-! # Argument parsing of the algo.* procedures

Line-by-line models of the config helpers in
`graph/src/runtime/functions/algo_procedures.rs`:

| here | there |
| --- | --- |
| `optString`          | `opt_string` :278 |
| `validateConfigMap`  | `validate_config_map` :290 |
| `strList`            | `extract_node_labels` :303, `extract_rel_types` :321, `relTypes` in `parse_common_path_config` :1934 |
| `parseConfig`        | `parse_config` :345 |
| `samplingSize`       | betweenness `samplingSize` :926-936 |
| `maxIterations`      | labelPropagation `maxIterations` :1221-1231 |
| `maxLen`             | `parse_common_path_config` maxLen :1962-1970 |
| `pathCount`          | `parse_common_path_config` pathCount :1993-2002 |
-/
namespace AlgoUdf.Config
open AlgoUdf

def optString (v : Val) : Except String (Option String) :=
  match v with
  | .null => .ok none
  | .str s => .ok (some s)
  | _ => .error "Type mismatch: expected String or Null"

def validateConfigMap (m : List (String × Val)) (allowed : List String) : Except String Unit :=
  match m.find? (fun kv => !(allowed.contains kv.1)) with
  | some kv => .error s!"Unknown parameter: {kv.1}"
  | none => .ok ()

/-- All-strings check shared by `extract_node_labels` / `extract_rel_types`. -/
def allStrings : List Val → Option (List String)
  | [] => some []
  | .str s :: t => (allStrings t).map (s :: ·)
  | _ :: _ => none

def strList (m : List (String × Val)) (key : String) : Except String (List String) :=
  match lookup m key with
  | none | some .null => .ok []
  | some (.list xs) => match allStrings xs with
    | some l => .ok l
    | none => .error s!"{key} must be an array of strings"
  | some _ => .error s!"{key} must be an array of strings"

def parseConfig (args : List Val) : Except String (List (String × Val)) :=
  match args with
  | [] => .ok []
  | .null :: _ => .ok []
  | .map m :: _ => .ok m
  | _ :: _ => .error "Invalid argument type: expected a map or null"

/-- betweenness samplingSize: positive check on the i64, then `as i32 as usize`
(i32 → usize sign-extends, so a negative i32 becomes a huge usize). -/
def samplingSize (m : List (String × Val)) : Except String Nat :=
  match lookup m "samplingSize" with
  | none | some .null => .ok 16
  | some (.int n) =>
      if n ≤ 0 then .error "samplingSize must be a positive integer"
      else let t := asI32 n
           .ok (if t < 0 then (t + 18446744073709551616).toNat else t.toNat)
  | some _ => .error "samplingSize must be a positive integer"

def maxIterations (m : List (String × Val)) : Except String Int :=
  match lookup m "maxIterations" with
  | none | some .null => .ok 10
  | some (.int n) => if n ≤ 0 then .error "maxIterations must be a positive integer" else .ok (asI32 n)
  | some _ => .error "maxIterations must be a positive integer"

def u32Max : Int := 4294967295

def maxLen (m : List (String × Val)) : Except String Int :=
  match lookup m "maxLen" with
  | none | some .null => .ok u32Max
  | some (.int n) => if n < 0 then .error "maxLen must be non-negative integer" else .ok (asU32 n)
  | some _ => .error "maxLen must be integer"

def pathCount (m : List (String × Val)) : Except String Int :=
  match lookup m "pathCount" with
  | none | some .null => .ok 1
  | some (.int n) => if n < 0 then .error "pathCount must be a non-negative integer" else .ok n
  | some _ => .error "pathCount must be integer"

/-! ## Theorems -/

theorem allStrings_spec (xs : List Val) (l : List String) :
    allStrings xs = some l → xs = l.map Val.str := by
  induction xs generalizing l with
  | nil => intro h; simp [allStrings] at h; subst h; rfl
  | cons x t ih =>
    intro h
    cases x with
    | str s =>
      simp only [allStrings] at h
      cases ht : allStrings t with
      | none => rw [ht] at h; cases h
      | some l' =>
        rw [ht] at h; simp at h; subst h
        simp [ih l' ht]
    | _ => simp [allStrings] at h

/-- `strList` accepts exactly absent/null/list-of-strings and returns those strings. -/
theorem strList_ok (m : List (String × Val)) (k : String) (l : List String) :
    strList m k = .ok l →
      (l = [] ∧ (lookup m k = none ∨ lookup m k = some .null)) ∨
      lookup m k = some (.list (l.map Val.str)) := by
  unfold strList
  split
  · intro h; cases h; exact Or.inl ⟨rfl, by simp_all⟩
  · intro h; cases h; exact Or.inl ⟨rfl, by simp_all⟩
  · rename_i xs hx
    split
    · rename_i l' hl; intro h; cases h
      exact Or.inr (by rw [hx, allStrings_spec xs _ hl])
    · intro h; cases h
  · intro h; cases h

theorem validate_ok (m : List (String × Val)) (allowed : List String) :
    validateConfigMap m allowed = .ok () → ∀ kv ∈ m, kv.1 ∈ allowed := by
  unfold validateConfigMap
  split
  · intro h; cases h
  · rename_i hn; intro _ kv hkv
    have := List.find?_eq_none.mp hn kv hkv
    simpa using this

/-- Wrap bug (shared with C, checked live): `maxIterations: 4294967296` passes the
positivity check and truncates to 0 iterations. -/
theorem maxIterations_wraps_to_zero :
    maxIterations [("maxIterations", .int 4294967296)] = .ok 0 := rfl

/-- `samplingSize: 4294967296` becomes 0 samples; `2147483648` becomes a huge
usize (so "sample everything"). Shared with C (same scores on live servers). -/
theorem samplingSize_wraps :
    samplingSize [("samplingSize", .int 4294967296)] = .ok 0 ∧
    samplingSize [("samplingSize", .int 2147483648)] = .ok 18446744071562067968 :=
  ⟨rfl, rfl⟩

/-- `maxLen: 4294967296` is truncated to 0 ("no path"); C's `uint32_t max_hops`
truncates the same way. -/
theorem maxLen_wraps : maxLen [("maxLen", .int 4294967296)] = .ok 0 := rfl

theorem maxLen_small (n : Int) (h0 : 0 ≤ n) (h1 : n < 4294967296) :
    maxLen [("maxLen", .int n)] = .ok n := by
  have hn : ¬ n < 0 := by omega
  simp [maxLen, lookup, asU32, hn, Int.emod_eq_of_lt h0 h1]

end AlgoUdf.Config
