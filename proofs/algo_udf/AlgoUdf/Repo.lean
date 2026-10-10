/-! # UDF library registry (`graph/src/udf/repository.rs`)

| here | there |
| --- | --- |
| `Lib`                 | `UdfLibrary` :42 (name, qualified `function_names`) |
| `qual`                | `format!("{name}.{fn_name}")` :102-105 |
| `key`                 | registry key `name.to_lowercase()` in `register_udf`/`unregister_udf` (`runtime/functions/mod.rs:1121,1129`) and `persistent_funcs` (`udf/js_context.rs:262`) |
| `load`                | `UdfRepo::load` :94 + the caller's `register_udf` loop (`src/commands/udf.rs`) |
| `delete`              | `UdfRepo::delete` :140 + the caller's `unregister_udf` loop |
| `Consistent`          | "LIST shows f ⇔ calling f runs f's body" |

`to_lowercase` is modelled as an arbitrary function `low` (ASCII lowercasing in
the counterexamples).
-/
namespace AlgoUdf.Repo

structure Lib where
  name : String
  fns : List String
  deriving DecidableEq, Repr

def qual (lib : String) (f : String) : String := lib ++ "." ++ f

/-- Registry entry: lowercase key ↦ (library, function) whose body runs. -/
abbrev Reg := List (String × (String × String))

def keys (low : String → String) (l : Lib) : List String := l.fns.map (fun f => low (qual l.name f))

/-- `load` (no replace, name not present): only library *names* are compared, then
every qualified key is (re-)inserted, overwriting any colliding entry. -/
def load (low : String → String) (libs : List Lib) (reg : Reg) (l : Lib) :
    Except String (List Lib × Reg) :=
  if libs.any (·.name == l.name) then .error "library already registered"
  else .ok (libs ++ [l],
    l.fns.foldl (fun r f => (low (qual l.name f), (l.name, f)) :: r.filter (·.1 != low (qual l.name f))) reg)

def delete (low : String → String) (libs : List Lib) (reg : Reg) (n : String) : List Lib × Reg :=
  let gone := (libs.filter (·.name == n)).flatMap (keys low)
  (libs.filter (·.name != n), reg.filter (fun e => !(gone.contains e.1)))

/-- Every listed function is callable and runs its own body. -/
def Consistent (low : String → String) (libs : List Lib) (reg : Reg) : Prop :=
  ∀ l ∈ libs, ∀ f ∈ l.fns, (reg.find? (·.1 == low (qual l.name f))).map (·.2) = some (l.name, f)

/-- The missing check: distinct (library, function) pairs have distinct keys. -/
def KeysUnique (low : String → String) (libs : List Lib) : Prop :=
  ∀ l₁ ∈ libs, ∀ l₂ ∈ libs, ∀ f₁ ∈ l₁.fns, ∀ f₂ ∈ l₂.fns,
    low (qual l₁.name f₁) = low (qual l₂.name f₂) → l₁.name = l₂.name ∧ f₁ = f₂

theorem find_filter (reg : Reg) (k : String) (p : String → Bool) (hk : p k = false) :
    (reg.filter (fun e => !p e.1)).find? (·.1 == k) = reg.find? (·.1 == k) := by
  induction reg with
  | nil => rfl
  | cons e t ih =>
    by_cases h : p e.1 = true
    · have hne : ¬ ((e.1 == k) = true) := by
        intro he; simp at he; rw [he] at h; rw [h] at hk; cases hk
      rw [List.filter_cons_of_neg (by simp [h]), ih]
      simp only [List.find?_cons]
      simp only [Bool.eq_false_iff.mpr hne]
    · rw [List.filter_cons_of_pos (by simp [h])]
      simp only [List.find?_cons]
      rw [ih]

/-- Under `KeysUnique`, deleting a library keeps every other library callable. -/
theorem delete_preserves (low : String → String) (libs : List Lib) (reg : Reg) (n : String)
    (hu : KeysUnique low libs) (hc : Consistent low libs reg) :
    Consistent low (delete low libs reg n).1 (delete low libs reg n).2 := by
  intro l hl f hf
  simp only [delete, List.mem_filter, bne_iff_ne, ne_eq] at hl
  obtain ⟨hl, hn⟩ := hl
  have hnot : ((libs.filter (·.name == n)).flatMap (keys low)).contains (low (qual l.name f)) = false := by
    apply Bool.eq_false_iff.mpr
    intro hc'
    simp only [List.contains_iff_mem, List.mem_flatMap, List.mem_filter, keys, List.mem_map] at hc'
    obtain ⟨l', ⟨hl', hn'⟩, f', hf', he⟩ := hc' 
    have := (hu l' hl' l hl f' hf' f hf he).1
    simp at hn'; exact hn (this ▸ hn')
  show Option.map _ (List.find? _ (reg.filter _)) = _
  rw [find_filter reg _ _ hnot]
  exact hc l hl f hf

/-- ASCII lowercasing, enough for the counterexamples. -/
def asciiLow (s : String) : String := String.ofList (s.toList.map Char.toLower)

/-- BUG (confirmed, `bug_udf_qualified_name_collision_accepted`): `Coll` then
`coll`, each registering `f`: the second load is accepted, the keys collide. -/
theorem collision_accepted :
    let l1 : Lib := ⟨"Coll", ["f"]⟩
    let l2 : Lib := ⟨"coll", ["f"]⟩
    (load asciiLow [l1] [("coll.f", ("Coll", "f"))] l2).toOption.isSome ∧
    ¬ KeysUnique asciiLow [l1, l2] := by
  refine ⟨by decide, fun h => ?_⟩
  have := h ⟨"Coll", ["f"]⟩ (by simp) ⟨"coll", ["f"]⟩ (by simp) "f" (by simp) "f" (by simp) (by decide)
  exact absurd this.1 (by decide)

/-- `a` registering `b.c` and `a.b` registering `c` both own key `a.b.c`. -/
theorem dotted_collision : qual "a" "b.c" = qual "a.b" "c" := by decide

/-- ... and deleting `coll` unregisters `Coll.f` while LIST still shows it. -/
theorem delete_breaks_other :
    let libs := [(⟨"Coll", ["f"]⟩ : Lib), ⟨"coll", ["f"]⟩]
    let reg : Reg := [("coll.f", ("coll", "f"))]
    (delete asciiLow libs reg "coll").1 = [⟨"Coll", ["f"]⟩] ∧
    (delete asciiLow libs reg "coll").2 = [] := by decide

/-- BUG (confirmed, `bug_udf_function_names_case_folded`): `f` and `F` in one
library share a key, so both run `F`'s body. -/
theorem case_folded : asciiLow (qual "cased" "f") = asciiLow (qual "cased" "F") := by decide

end AlgoUdf.Repo
