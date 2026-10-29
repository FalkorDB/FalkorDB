/-! # UDF library registry (`graph/src/udf/repository.rs`)

Re-targeted to origin/main 1c0e6c8de (#3066, fixes #3065): registry keys are the
case-sensitive qualified names, and `load` rejects a qualified name another library
already owns, compared ignoring ASCII case, as C does.

| here | there |
| --- | --- |
| `Lib`                 | `UdfLibrary` :42 (name, qualified `function_names`) |
| `qual`                | `format!("{name}.{fn_name}")` :102-105 |
| `load`                | `UdfRepo::load` :94-153 (replace branch :134-139, collision check :118-132) + the caller's `register_udf` loop (`src/commands/udf.rs`); registry key is the qualified name as given (`register_udf` `runtime/functions/mod.rs:1131-1138`, `persistent_funcs` `udf/js_context.rs:261`) |
| `collides`            | the `eq_ignore_ascii_case` scan over the other libraries' `function_names` :118-127 |
| `delete`              | `UdfRepo::delete` :156 + the caller's `unregister_udf` loop (`unregister_udf` :1142) |
| `Consistent`          | "LIST shows f ⇔ calling f runs f's body" |
| `preLoad`, `pre3066_*` | the code before #3066 (keys lowercased, no collision check): historical |

`to_lowercase` / `to_ascii_lowercase` is modelled as `asciiLow` (Lean's `Char.toLower` is ASCII-only).
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

/-- HISTORICAL (before #3066): `load` (no replace, name not present): only library
*names* are compared, then every lowercased qualified key is (re-)inserted,
overwriting any colliding entry. -/
def preLoad (low : String → String) (libs : List Lib) (reg : Reg) (l : Lib) :
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

/-- ASCII lowercasing (`to_lowercase` / `eq_ignore_ascii_case` on ASCII names). -/
def asciiLow (s : String) : String := String.ofList (s.toList.map Char.toLower)

/-- `str::eq_ignore_ascii_case`. -/
def foldEq (a b : String) : Bool := asciiLow a == asciiLow b

theorem foldEq_comm (a b : String) : foldEq a b = foldEq b a := by
  unfold foldEq
  generalize asciiLow a = x
  generalize asciiLow b = y
  rw [Bool.eq_iff_iff]
  simp only [beq_iff_eq]; exact ⟨Eq.symm, Eq.symm⟩

theorem foldEq_refl (a : String) : foldEq a a = true := by
  unfold foldEq; exact beq_self_eq_true _

/-! ## Historical: the code before #3066 (fixed by #3066) -/

/-- Pre-#3066 BUG: `Coll` then `coll`, each registering `f`: the second load was
accepted, the lowercased keys collide. -/
theorem pre3066_collision_accepted :
    let l1 : Lib := ⟨"Coll", ["f"]⟩
    let l2 : Lib := ⟨"coll", ["f"]⟩
    (preLoad asciiLow [l1] [("coll.f", ("Coll", "f"))] l2).toOption.isSome ∧
    ¬ KeysUnique asciiLow [l1, l2] := by
  refine ⟨by decide, fun h => ?_⟩
  have := h ⟨"Coll", ["f"]⟩ (by simp) ⟨"coll", ["f"]⟩ (by simp) "f" (by simp) "f" (by simp) (by decide)
  exact absurd this.1 (by decide)

/-- ... and deleting `coll` unregistered `Coll.f` while LIST still showed it. -/
theorem pre3066_delete_breaks_other :
    let libs := [(⟨"Coll", ["f"]⟩ : Lib), ⟨"coll", ["f"]⟩]
    let reg : Reg := [("coll.f", ("coll", "f"))]
    (delete asciiLow libs reg "coll").1 = [⟨"Coll", ["f"]⟩] ∧
    (delete asciiLow libs reg "coll").2 = [] := by decide

/-- Pre-#3066 BUG: `f` and `F` in one library shared a lowercased key, so both ran
`F`'s body. -/
theorem pre3066_case_folded : asciiLow (qual "cased" "f") = asciiLow (qual "cased" "F") := by decide

/-! ## After #3066: exact keys, collision check -/

/-- `a` registering `b.c` and `a.b` registering `c` build the same qualified name
`a.b.c`: the reason the check compares qualified names, not (library, function) pairs. -/
theorem dotted_collision : qual "a" "b.c" = qual "a.b" "c" := by decide

/-- The :118-127 scan: some qualified name of `l` equals, ignoring ASCII case, one
owned by a library with another name (a replaced library's own names are exempt). -/
def collides (libs : List Lib) (l : Lib) : Bool :=
  l.fns.any fun f => (libs.filter (·.name != l.name)).any fun o =>
    o.fns.any fun g => foldEq (qual o.name g) (qual l.name f)

/-- `UdfRepo::load` (`replace` = the flag): a name clash without `replace` and a
collision both error; otherwise the old library (if any) is removed first (:134-139),
then the new one pushed (:148) and every qualified name registered as given. -/
def load (libs : List Lib) (reg : Reg) (l : Lib) (replace : Bool) :
    Except String (List Lib × Reg) :=
  if !replace && libs.any (·.name == l.name) then .error "library already registered"
  else if collides libs l then .error "function already registered"
  else
    let d := delete id libs reg l.name
    .ok (d.1 ++ [l],
      l.fns.foldl (fun r f => (qual l.name f, (l.name, f)) :: r.filter (·.1 != qual l.name f)) d.2)

/-- Pre-#3066 counterexamples now error. -/
theorem collision_rejected :
    (load [⟨"Coll", ["f"]⟩] [("Coll.f", ("Coll", "f"))] ⟨"coll", ["f"]⟩ false).toOption.isNone := by
  decide

theorem collision_rejected_replace :
    (load [⟨"Coll", ["f"]⟩] [("Coll.f", ("Coll", "f"))] ⟨"coll", ["f"]⟩ true).toOption.isNone := by
  decide

theorem dotted_collision_rejected :
    (load [⟨"a", ["b.c"]⟩] [("a.b.c", ("a", "b.c"))] ⟨"a.b", ["c"]⟩ false).toOption.isNone := by
  decide

/-- Replacing a library with itself is not a collision. -/
theorem replace_self_ok :
    (load [⟨"Coll", ["f"]⟩] [("Coll.f", ("Coll", "f"))] ⟨"Coll", ["f"]⟩ true).toOption.isSome := by
  decide

/-- Calls stay case-sensitive: `f` and `F` in one library are two registry keys (not
a collision: the scan skips the loading library's own names) and each runs its own body. -/
theorem case_sensitive : qual "cased" "f" ≠ qual "cased" "F" := by decide

theorem same_lib_cases_coexist :
    (load [] [] ⟨"cased", ["f", "F"]⟩ false).toOption.map (·.2.map (·.1)) =
      some ["cased.F", "cased.f"] := by decide

/-- No two libraries own names that agree ignoring case. -/
def CrossFoldUnique (libs : List Lib) : Prop :=
  ∀ l₁ ∈ libs, ∀ l₂ ∈ libs, l₁.name ≠ l₂.name → ∀ f₁ ∈ l₁.fns, ∀ f₂ ∈ l₂.fns,
    foldEq (qual l₁.name f₁) (qual l₂.name f₂) = false

theorem collides_false {libs : List Lib} {l : Lib} (h : collides libs l = false) :
    ∀ o ∈ libs, o.name ≠ l.name → ∀ g ∈ o.fns, ∀ f ∈ l.fns,
      foldEq (qual o.name g) (qual l.name f) = false := by
  intro o ho hn g hg f hf
  simp only [collides, Bool.eq_false_iff, ne_eq, List.any_eq_true, not_exists, not_and] at h
  have := h f hf o (by simpa [List.mem_filter, hn] using ho) g hg
  simpa using this

/-- `load` keeps the cross-library no-collision invariant: a successful load of a new
or replacing library cannot introduce a pair of libraries that fold-collide. -/
theorem load_preserves_crossFold (libs : List Lib) (reg : Reg) (l : Lib) (rep : Bool)
    (libs' : List Lib) (reg' : Reg) (hu : CrossFoldUnique libs)
    (h : load libs reg l rep = .ok (libs', reg')) : CrossFoldUnique libs' := by
  unfold load at h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hc
  have hc' : collides libs l = false := by simpa using hc
  simp only [delete, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, _⟩ := h
  intro l₁ h₁ l₂ h₂ hn f₁ hf₁ f₂ hf₂
  simp only [List.mem_append, List.mem_filter, bne_iff_ne, ne_eq, List.mem_singleton] at h₁ h₂
  rcases h₁ with ⟨h₁, hn₁⟩ | rfl <;> rcases h₂ with ⟨h₂, hn₂⟩ | rfl
  · exact hu l₁ h₁ l₂ h₂ hn f₁ hf₁ f₂ hf₂
  · exact collides_false hc' l₁ h₁ hn₁ f₁ hf₁ f₂ hf₂
  · rw [foldEq_comm]; exact collides_false hc' l₂ h₂ hn₂ f₂ hf₂ f₁ hf₁
  · exact absurd rfl hn

/-- With no fold-collision between libraries, the exact qualified names are a faithful
key: distinct (library, function) pairs get distinct registry keys (`KeysUnique id`). -/
theorem keysUnique_of_crossFold (libs : List Lib) (h : CrossFoldUnique libs) :
    KeysUnique id libs := by
  intro l₁ h₁ l₂ h₂ f₁ hf₁ f₂ hf₂ he
  by_cases hn : l₁.name = l₂.name
  · refine ⟨hn, ?_⟩
    simp only [qual, id] at he
    rw [hn, String.append_assoc, String.append_assoc] at he
    exact (String.append_right_inj _ |>.mp ((String.append_right_inj _ |>.mp he)))
  · have := h l₁ h₁ l₂ h₂ hn f₁ hf₁ f₂ hf₂
    simp only [id] at he
    rw [he, foldEq_refl] at this
    cases this

/-- Under the invariant `load` maintains, deleting a library keeps every other library
callable (the pre-#3066 `delete_breaks_other` can no longer arise). -/
theorem delete_preserves_after3066 (libs : List Lib) (reg : Reg) (n : String)
    (hu : CrossFoldUnique libs) (hc : Consistent id libs reg) :
    Consistent id (delete id libs reg n).1 (delete id libs reg n).2 :=
  delete_preserves id libs reg n (keysUnique_of_crossFold libs hu) hc

end AlgoUdf.Repo
