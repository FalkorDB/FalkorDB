/-
# `src/graph_core.rs`: C name rules and the graph registry

The parts of `graph_core.rs` that the transition systems of `Mvcc`/`Locks` treat as one atomic
step or leave out: the C name rules and the graph registry (`Exec` has the rest). Each is a pure
function here; a theorem states what the Rust returns. (`redis_layer` proves the same fns against
its own models; these are this project's, so the rows here do not lean on another project.)

| Lean | Rust (origin/main 256ab4d3f) |
| --- | --- |
| `upToNul`                      | `up_to_nul` `graph_core.rs` |
| `cGraphKey`, `cGraphName`      | `c_graph_key` (bytes up to the first NUL), `c_graph_name` (`up_to_nul` of the lossy string) |
| `regPut`, `registerGraph`      | `register_graph` (`GRAPH_REGISTRY.lock().insert`, the displaced entry is dropped on another thread) |
| `isRegistered`                 | `graph_is_registered` (a scan for the same `Arc`) |

`String::from_utf8_lossy` is a std function: the two properties of it that `c_graph_name` needs
are a hypothesis structure (`Lossy`), not an axiom.
-/
namespace SC.Glue

/-! ## C name rules -/

/-- `up_to_nul` (`s.find('\0').map_or(s, |end| &s[..end])`), over code points. -/
def upToNul : List Nat → List Nat
  | [] => []
  | x :: r => if x = 0 then [] else x :: upToNul r

theorem upToNul_nonul : ∀ (s : List Nat), ∀ x ∈ upToNul s, x ≠ 0
  | [] => by simp [upToNul]
  | x :: r => by
    intro y hy
    by_cases h : x = 0
    · simp [upToNul, h] at hy
    · simp only [upToNul, h, ite_false, List.mem_cons] at hy
      rcases hy with rfl | hy
      · exact h
      · exact upToNul_nonul r y hy

/-- The result is a prefix of the input. -/
theorem upToNul_append : ∀ (s : List Nat), ∃ r, s = upToNul s ++ r
  | [] => ⟨[], rfl⟩
  | x :: r => by
    by_cases h : x = 0
    · exact ⟨x :: r, by simp [upToNul, h]⟩
    · obtain ⟨t, ht⟩ := upToNul_append r
      exact ⟨t, by simp only [upToNul, h, ite_false, List.cons_append]; rw [← ht]⟩

theorem upToNul_of_nonul : ∀ (s : List Nat), (∀ x ∈ s, x ≠ 0) → upToNul s = s
  | [], _ => rfl
  | x :: r, h => by
    have hx : x ≠ 0 := h x (by simp)
    simp only [upToNul, hx, ite_false]
    rw [upToNul_of_nonul r (fun y hy => h y (by simp [hy]))]

/-- Idempotent: a name that went through `up_to_nul` once is unchanged by another pass. -/
theorem upToNul_idem (s : List Nat) : upToNul (upToNul s) = upToNul s :=
  upToNul_of_nonul _ (upToNul_nonul s)

/-- A NUL ends the value: `a\0b` is `a` (C reads it as a C string). -/
theorem upToNul_cut : ∀ (a b : List Nat), (∀ x ∈ a, x ≠ 0) → upToNul (a ++ 0 :: b) = a
  | [], b, _ => by simp [upToNul]
  | x :: r, b, h => by
    have hx : x ≠ 0 := h x (by simp)
    simp only [List.cons_append, upToNul, hx, ite_false]
    rw [upToNul_cut r b (fun y hy => h y (by simp [hy]))]

theorem upToNul_split : ∀ (s : List Nat), (∀ x ∈ s, x ≠ 0) ∨ ∃ b, s = upToNul s ++ 0 :: b
  | [] => Or.inl (by simp)
  | x :: r => by
    by_cases h : x = 0
    · exact Or.inr ⟨r, by simp [upToNul, h]⟩
    · rcases upToNul_split r with hr | ⟨b, hb⟩
      · left; intro y hy
        simp only [List.mem_cons] at hy
        rcases hy with rfl | hy
        · exact h
        · exact hr y hy
      · right; refine ⟨b, ?_⟩
        simp only [upToNul, h, ite_false, List.cons_append]
        rw [← hb]

/-- `c_graph_key`: the key's bytes up to the first NUL (it works on the bytes, so a
key that is not UTF-8 is kept as it is). -/
def cGraphKey (key : List Nat) : List Nat := upToNul key

theorem cGraphKey_nonul (key : List Nat) : ∀ x ∈ cGraphKey key, x ≠ 0 := upToNul_nonul key

/-- Identical to the key unless it holds a NUL. -/
theorem cGraphKey_id (key : List Nat) (h : ∀ x ∈ key, x ≠ 0) : cGraphKey key = key :=
  upToNul_of_nonul key h

/-- `String::from_utf8_lossy`, as far as `c_graph_name` uses it: a NUL-free byte string decodes to
NUL-free text, and a NUL byte decodes to itself, splitting the text there (it is a whole
character, so nothing before it depends on what follows). -/
structure Lossy (lossy : List Nat → List Nat) : Prop where
  nonul : ∀ a, (∀ x ∈ a, x ≠ 0) → ∀ x ∈ lossy a, x ≠ 0
  split : ∀ a b, (∀ x ∈ a, x ≠ 0) → lossy (a ++ 0 :: b) = lossy a ++ 0 :: lossy b

/-- `c_graph_name`: `up_to_nul(&key.to_string_lossy())`. -/
def cGraphName (lossy : List Nat → List Nat) (key : List Nat) : List Nat := upToNul (lossy key)

/-- **The name is the key `c_graph_key` stores a new graph under** (read back as text): both
are the key cut at its first NUL. -/
theorem cGraphName_eq_lossy_key {lossy : List Nat → List Nat} (hl : Lossy lossy) (key : List Nat) :
    cGraphName lossy key = lossy (cGraphKey key) := by
  rcases upToNul_split key with h | ⟨b, hb⟩
  · simp only [cGraphName, cGraphKey, upToNul_of_nonul key h]
    exact upToNul_of_nonul _ (hl.nonul key h)
  · have hn := upToNul_nonul key
    have e : lossy key = lossy (upToNul key) ++ 0 :: lossy b := by
      conv => lhs; rw [hb]
      exact hl.split _ _ hn
    simp only [cGraphName, cGraphKey]
    rw [e, upToNul_cut _ _ (hl.nonul _ hn)]

/-- No NUL ever reaches a name. -/
theorem cGraphName_nonul {lossy : List Nat → List Nat} (key : List Nat) :
    ∀ x ∈ cGraphName lossy key, x ≠ 0 := upToNul_nonul _

/-! ## The registry (`GRAPH_REGISTRY: Mutex<HashMap<String, Arc<RwLock<ThreadedGraph>>>>`) -/

/-- Names and `Arc` identities, one entry per name. -/
abbrev Reg := List (Nat × Nat)

def regGet : Reg → Nat → Option Nat
  | [], _ => none
  | (k, v) :: r, n => if k = n then some v else regGet r n

def regDel : Reg → Nat → Reg
  | [], _ => []
  | (k, v) :: r, n => if k = n then regDel r n else (k, v) :: regDel r n

/-- `HashMap::insert`: the new entry, and the entry it displaced. -/
def regPut (r : Reg) (n a : Nat) : Reg × Option Nat := ((n, a) :: regDel r n, regGet r n)

theorem regGet_regDel_same : ∀ (r : Reg) (n : Nat), regGet (regDel r n) n = none
  | [], n => rfl
  | (k, v) :: r, n => by
    by_cases h : k = n
    · simp only [regDel, h, ite_true]; exact regGet_regDel_same r n
    · simp only [regDel, h, ite_false, regGet]; simpa [h] using regGet_regDel_same r n

theorem regGet_regDel_other : ∀ (r : Reg) (n m : Nat), m ≠ n → regGet (regDel r n) m = regGet r m
  | [], n, m, _ => rfl
  | (k, v) :: r, n, m, hm => by
    have ih := regGet_regDel_other r n m hm
    by_cases h : k = n
    · subst h
      have hkm : k ≠ m := Ne.symm hm
      simp only [regDel, ite_true, regGet, hkm, ite_false]; exact ih
    · simp only [regDel, h, ite_false, regGet]
      by_cases hp : k = m
      · simp [hp]
      · simp [hp, ih]

/-- `register_graph`: the registry afterwards, and the displaced graph (dropped off the main
thread; a name can already be registered under lazy free, which is not a leak). -/
def registerGraph (r : Reg) (n a : Nat) : Reg × Option Nat := regPut r n a

/-- The new graph is registered under its name. -/
theorem registerGraph_get (r : Reg) (n a : Nat) : regGet (registerGraph r n a).1 n = some a := by
  simp [registerGraph, regPut, regGet]

/-- Every other name is untouched. -/
theorem registerGraph_other (r : Reg) (n a m : Nat) (h : m ≠ n) :
    regGet (registerGraph r n a).1 m = regGet r m := by
  have h' : n ≠ m := Ne.symm h
  simp only [registerGraph, regPut, regGet, h', ite_false]
  exact regGet_regDel_other r n m h

/-- What it displaces is whatever the name held (`None` for a fresh name). -/
theorem registerGraph_displaced (r : Reg) (n a : Nat) : (registerGraph r n a).2 = regGet r n := rfl

/-- `graph_is_registered`: the `Arc` is registered under some name. -/
def isRegistered (r : Reg) (a : Nat) : Bool := r.any (·.2 == a)

theorem registerGraph_registered (r : Reg) (n a : Nat) : isRegistered (registerGraph r n a).1 a = true := by
  simp [registerGraph, regPut, isRegistered]

end SC.Glue
