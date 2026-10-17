import AlgoUdf.Repo
/-! # `UdfRepo` bookkeeping (`graph/src/udf/repository.rs`) and the singleton (`udf/mod.rs`)

| here | there |
| --- | --- |
| `RLib`, `RSt`        | `UdfLibrary` :42-48, `UdfRepo { inner, version }` :56-64 (the `RwLock` serialises writers; each op is atomic) |
| `rnew`               | `UdfRepo::new` :74 / `Default::default` :67 |
| `rversion`, `bump`   | `version` :83, `bump_version` :89 |
| `flush`              | `flush` :157-167 |
| `strip`, `list`      | `list` :170-202 (`strip_prefix(&format!("{}.", lib.name)).unwrap_or(qn)`) |
| `getAll`, `serialize`| `get_all_libraries` :205, `serialize` :211 |
| `deserialize`        | `deserialize` :226-264 (validate all into `staged`, then swap) |
| `Once`, `initRepo`, `getRepo` | `UDF_REPO: OnceLock`, `init_udf_repo` (mod.rs:59), `get_udf_repo` (mod.rs:63; `none` = the `expect` panic) |

Strings are `List Char`; `validate` is `js_context::validate_script` (QuickJS,
AXIOMATISED as a function of the code).
-/
namespace AlgoUdf.UdfRepoOps

abbrev Str := List Char

structure RLib where
  name : Str
  code : Str
  fns : List Str
  deriving DecidableEq, Repr

structure RSt where
  libs : List RLib
  version : Nat
  deriving DecidableEq, Repr

def qual (n f : Str) : Str := n ++ ['.'] ++ f

def rnew : RSt := ⟨[], 0⟩
def rversion (s : RSt) : Nat := s.version
def bump (s : RSt) : RSt := { s with version := s.version + 1 }

theorem rnew_spec : rnew.libs = [] ∧ rversion rnew = 0 := ⟨rfl, rfl⟩
theorem bump_spec (s : RSt) : rversion (bump s) = rversion s + 1 ∧ (bump s).libs = s.libs := ⟨rfl, rfl⟩

/-- `flush`: every registered name is returned for unregistering; all
libraries go; the version moves. -/
def flush (s : RSt) : List Str × RSt := (s.libs.flatMap (·.fns), ⟨[], s.version + 1⟩)

theorem flush_spec (s : RSt) :
    (flush s).1 = s.libs.flatMap (·.fns) ∧ (flush s).2.libs = [] ∧ (flush s).2.version = s.version + 1 :=
  ⟨rfl, rfl, rfl⟩

def strip (p s : Str) : Str := if p.isPrefixOf s then s.drop p.length else s

def list (s : RSt) (filter : Option Str) (withCode : Bool) : List (Str × List Str × Option Str) :=
  (s.libs.filter (fun l => filter.all (· == l.name))).map fun l =>
    (l.name, l.fns.map (strip (l.name ++ ['.'])), if withCode then some l.code else none)

theorem isPrefixOf_append (p s : Str) : p.isPrefixOf (p ++ s) = true := by
  induction p with
  | nil => rfl
  | cons c t ih => simp [List.isPrefixOf, ih]

/-- `list` shows the raw names a library registered. -/
theorem strip_qual (n f : Str) : strip (n ++ ['.']) (qual n f) = f := by
  unfold strip qual
  rw [isPrefixOf_append]
  simp

theorem list_shows_raw (s : RSt) (l : RLib) (hl : l ∈ s.libs) (raw : List Str)
    (hq : l.fns = raw.map (qual l.name)) :
    (l.name, raw, none) ∈ list s none false := by
  unfold list
  simp only [Option.all_none, List.mem_map]
  refine ⟨l, by simpa using hl, ?_⟩
  rw [hq, List.map_map]
  simp [Function.comp_def, strip_qual]

theorem list_filter (s : RSt) (n : Str) : ∀ e ∈ list s (some n) true, e.1 = n := by
  intro e he
  simp only [list, List.mem_map, List.mem_filter, Option.all_some, beq_iff_eq] at he
  obtain ⟨l, ⟨_, h⟩, rfl⟩ := he
  exact h.symm

def getAll (s : RSt) : List RLib := s.libs
def serialize (s : RSt) : List (Str × Str) := s.libs.map (fun l => (l.name, l.code))

theorem getAll_spec (s : RSt) : getAll s = s.libs := rfl

/-- Stage every library or fail at the first validation error. -/
def stage (validate : Str → Except Str (List Str)) : List (Str × Str) → Except Str (List RLib)
  | [] => .ok []
  | (n, c) :: t =>
    match validate c with
    | .error e => .error ("UDF library '".toList ++ n ++ "' failed validation during RDB load: ".toList ++ e)
    | .ok raw =>
      match stage validate t with
      | .error e => .error e
      | .ok ls => .ok (⟨n, c, raw.map (qual n)⟩ :: ls)

/-- `deserialize`: `(result, unregistered names, new state)`. -/
def deserialize (validate : Str → Except Str (List Str)) (s : RSt) (input : List (Str × Str)) :
    Except Str (List RLib) × List Str × RSt :=
  match stage validate input with
  | .error e => (.error e, [], s)
  | .ok staged => (.ok staged, s.libs.flatMap (·.fns), ⟨staged, s.version + 1⟩)

/-- **All or nothing**: a validation failure leaves the live repository (and the
function registry) untouched. -/
theorem deserialize_atomic (validate : Str → Except Str (List Str)) (s : RSt) (input : List (Str × Str))
    (e : Str) (h : stage validate input = .error e) :
    deserialize validate s input = (.error e, [], s) := by
  simp [deserialize, h]

/-- A library was stored by `load`: its function names are the qualified
validation result of its code. -/
def Loaded (validate : Str → Except Str (List Str)) (l : RLib) : Prop :=
  ∃ raw, validate l.code = .ok raw ∧ l.fns = raw.map (qual l.name)

/-- **Round trip**: reloading what `serialize` wrote restores the same
libraries (with a deterministic `validate_script`). -/
theorem serialize_roundtrip (validate : Str → Except Str (List Str)) (s : RSt)
    (h : ∀ l ∈ s.libs, Loaded validate l) :
    (deserialize validate ⟨[], 0⟩ (serialize s)).2.2.libs = s.libs := by
  have key : ∀ ls : List RLib, (∀ l ∈ ls, Loaded validate l) →
      stage validate (ls.map (fun l => (l.name, l.code))) = .ok ls := by
    intro ls
    induction ls with
    | nil => intro _; rfl
    | cons l t ih =>
      intro hh
      obtain ⟨raw, hv, hf⟩ := hh l (by simp)
      simp only [List.map_cons, stage, hv]
      rw [ih (fun l' hl' => hh l' (List.mem_cons_of_mem _ hl'))]
      rw [← hf]
  have : stage validate (serialize s) = .ok s.libs := key s.libs h
  simp [deserialize, this]

/-! ## The singleton -/

abbrev Once := Option RSt

def initRepo (o : Once) : Once := match o with | none => some rnew | some s => some s
def getRepo (o : Once) : Option RSt := o

theorem initRepo_idem (o : Once) : initRepo (initRepo o) = initRepo o := by cases o <;> rfl
theorem getRepo_after_init (o : Once) : (getRepo (initRepo o)).isSome := by cases o <;> rfl
theorem getRepo_uninit : getRepo none = none := rfl

end AlgoUdf.UdfRepoOps
