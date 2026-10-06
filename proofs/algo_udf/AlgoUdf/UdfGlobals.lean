import AlgoUdf.Repo
/-! # The `falkor` / `graph` globals (`graph/src/udf/js_globals.rs`)

The logic of these functions lives in the JavaScript they `eval` into the
context; that JavaScript is modelled here. Installing properties on the global
object is QuickJS (AXIOMATISED: `globals.set(k, v)` makes `k` resolve to `v`).

| here | there |
| --- | --- |
| `vRegister`              | `globalThis.__falkor_register` of `setup_validate_globals` :62-72 |
| `validateGlobals`        | `setup_validate_globals` :45-95 (installs `falkor.register`, a no-op `falkor.log`) |
| `collectNames`           | `collect_validate_names` :98-108 (`names_arr.get::<String>`) |
| `rRegister`              | `__falkor_register` of `setup_runtime_globals` :121-127 |
| `runtimeGlobals`         | `setup_runtime_globals` :111-190 (`falkor.{register,log}`, `graph.{traverse,getNodeById,iterateNodes,iterateEdges}`) |
| `collectFuncs`           | `collect_runtime_funcs` :193-220 |
| `logString`              | `js_value_to_log_string` :222-248 |

Results: validation rejects non-functions and exact duplicate names and
returns names in registration order (`validate_names_nodup`); at runtime a
library `L` (non-empty name) registering the same sequence stores exactly the
keys `L.name` — the qualified names `UdfRepo::load` computes
(`runtime_keys_eq_qualified`). An empty library name breaks that agreement
(`empty_lib_name_mismatch`).
-/
namespace AlgoUdf.UdfGlobals

inductive JVal | null | undef | bool (b : Bool) | int (i : Int) | float (f : Int) | str (s : String)
  | fn (tag : Nat) | other (json : Option String)
  deriving DecidableEq, Repr

def regErrFn : String := "Failed to register UDF library: second argument must be a function"
def regErrDup (n : String) : String := "Failed to register UDF library: function '" ++ n ++ "' already registered"

/-- Validation-mode `falkor.register(name, func)` over the names array. -/
def vRegister (names : List String) (name : String) (func : JVal) : Except String (List String) :=
  match func with
  | .fn _ => if name ∈ names then .error (regErrDup name) else .ok (names ++ [name])
  | _ => .error regErrFn

/-- A script's sequence of `falkor.register` calls in validation mode. -/
def vRun : List String → List (String × JVal) → Except String (List String)
  | names, [] => .ok names
  | names, (n, f) :: t =>
    match vRegister names n f with
    | .error e => .error e
    | .ok ns => vRun ns t

/-- Globals installed by `setup_validate_globals`. -/
def validateGlobals : List String := ["falkor.register", "falkor.log"]

/-- `collect_validate_names`: every element must convert to a String. -/
def collectNames (arr : List JVal) : Except String (List String) :=
  arr.foldr (fun v acc => match v, acc with
    | .str s, .ok l => .ok (s :: l)
    | .str _, .error e => .error e
    | _, _ => .error "Error converting from js value into type 'String'") (.ok [])

theorem vRun_nodup : ∀ (calls : List (String × JVal)) (names out : List String), names.Nodup →
    vRun names calls = .ok out → out.Nodup := by
  intro calls
  induction calls with
  | nil => intro names out h e; cases e; exact h
  | cons p t ih =>
    obtain ⟨n, f⟩ := p
    intro names out h e
    simp only [vRun] at e
    split at e
    · cases e
    · rename_i ns hns
      apply ih ns out _ e
      unfold vRegister at hns
      split at hns
      · split at hns
        · cases hns
        · rename_i hn; cases hns
          rw [List.nodup_append]; exact ⟨h, by simp, by intro a ha b hb; simp at hb; subst hb; intro e; subst e; exact hn ha⟩
      · cases hns

/-- **Validation returns the registered names, without duplicates, in order.** -/
theorem validate_names_nodup (calls : List (String × JVal)) (out : List String)
    (h : vRun [] calls = .ok out) : out.Nodup := vRun_nodup calls [] out (by simp) h

theorem vRegister_rejects (names : List String) (n : String) :
    vRegister names n (.int 1) = .error regErrFn ∧
    (n ∈ names → vRegister names n (.fn 0) = .error (regErrDup n)) := by
  refine ⟨rfl, fun h => ?_⟩; simp [vRegister, h]

/-- Runtime-mode `falkor.register(name, func)`: store under the qualified key
(`current_lib + '.' + name`, or `name` when `current_lib` is empty). -/
def rRegister (lib : String) (reg : List (String × Nat)) (name : String) (func : JVal) :
    Except String (List (String × Nat)) :=
  match func with
  | .fn t =>
    let q := if lib = "" then name else lib ++ "." ++ name
    .ok (reg.filter (·.1 != q) ++ [(q, t)])
  | _ => .error regErrFn

def rRun (lib : String) : List (String × Nat) → List (String × JVal) → Except String (List (String × Nat))
  | reg, [] => .ok reg
  | reg, (n, f) :: t =>
    match rRegister lib reg n f with
    | .error e => .error e
    | .ok r => rRun lib r t

/-- Globals installed by `setup_runtime_globals`. -/
def runtimeGlobals : List String :=
  ["falkor.register", "falkor.log", "graph.traverse", "graph.getNodeById", "graph.iterateNodes",
   "graph.iterateEdges"]

theorem rRun_keys (lib : String) (hl : lib ≠ "") :
    ∀ (calls : List (String × JVal)) (reg out : List (String × Nat)),
      rRun lib reg calls = .ok out → ∀ k, k ∈ out.map (·.1) →
        k ∈ reg.map (·.1) ∨ ∃ n ∈ calls.map (·.1), k = Repo.qual lib n := by
  intro calls
  induction calls with
  | nil => intro reg out h k hk; cases h; exact Or.inl hk
  | cons p t ih =>
    obtain ⟨n, f⟩ := p
    intro reg out h k hk
    simp only [rRun] at h
    split at h
    · cases h
    · rename_i r hr
      rcases ih r out h k hk with h1 | ⟨n', hn', he⟩
      · unfold rRegister at hr
        split at hr
        · cases hr
          simp only [hl, if_false, List.map_append, List.mem_append, List.mem_map, List.mem_filter,
            List.map_cons, List.map_nil, List.mem_singleton] at h1
          rcases h1 with ⟨q, ⟨hq, _⟩, rfl⟩ | h1
          · exact Or.inl (List.mem_map_of_mem hq)
          · right; exact ⟨n, by simp, by rw [h1]; rfl⟩
        · cases hr
      · right; exact ⟨n', by simp [hn'], he⟩

/-- **Runtime keys are the qualified names** for a library with a non-empty
name: every key the JS registry ends up with is `lib.name` for a registered
`name` (plus pre-existing keys). -/
theorem runtime_keys_eq_qualified (lib : String) (hl : lib ≠ "") (calls : List (String × JVal))
    (out : List (String × Nat)) (h : rRun lib [] calls = .ok out) :
    ∀ k ∈ out.map (·.1), ∃ n ∈ calls.map (·.1), k = Repo.qual lib n := by
  intro k hk
  rcases rRun_keys lib hl calls [] out h k hk with h1 | h1
  · simp at h1
  · exact h1

/-- With an empty library name the runtime key is `f` but `UdfRepo::load`
records `.f` (`format!("{name}.{fn_name}")`), so `rebuild_context` never finds it. -/
theorem empty_lib_name_mismatch :
    rRun "" [] [("f", .fn 0)] = .ok [("f", 0)] ∧ Repo.qual "" "f" = ".f" := ⟨rfl, rfl⟩

/-- `collect_runtime_funcs`: every value must be a function. -/
def collectFuncs (obj : List (String × JVal)) : Except String (List (String × Nat)) :=
  obj.foldr (fun kv acc => match kv.2, acc with
    | .fn t, .ok l => .ok ((kv.1, t) :: l)
    | .fn _, .error e => .error e
    | _, _ => .error ("Expected a function for '" ++ kv.1 ++ "', got non-function value")) (.ok [])

theorem collectFuncs_ok (obj : List (String × JVal)) (out : List (String × Nat))
    (h : collectFuncs obj = .ok out) : out.map (·.1) = obj.map (·.1) := by
  induction obj generalizing out with
  | nil => simp [collectFuncs] at h; subst h; rfl
  | cons kv t ih =>
    simp only [collectFuncs, List.foldr_cons] at h
    revert h
    split
    · intro h; cases h; simp; exact ih _ (by assumption)
    · intro h; cases h
    · intro h; cases h

theorem collectNames_ok (names : List String) : collectNames (names.map .str) = .ok names := by
  induction names with
  | nil => rfl
  | cons n t ih => simp only [collectNames, List.map_cons, List.foldr_cons] at ih ⊢; rw [ih]

/-- `js_value_to_log_string` (`Display` of bool/i32/f64 abstracted to `toString`). -/
def logString : JVal → String
  | .null => "null"
  | .undef => "undefined"
  | .bool b => toString b
  | .int i => toString i
  | .float f => toString f
  | .str s => s
  | .fn _ => "[object]"
  | .other (some j) => j
  | .other none => "[object]"

theorem logString_spec : logString .null = "null" ∧ logString .undef = "undefined" ∧
    logString (.bool true) = "true" ∧ logString (.str "x") = "x" ∧ logString (.other none) = "[object]" := by
  decide

end AlgoUdf.UdfGlobals
