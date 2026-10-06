import AlgoUdf.Repo
/-! # Thread-local QuickJS context management (`graph/src/udf/js_context.rs`)

QuickJS itself (runtime creation, `eval`, function calls) is AXIOMATISED as the
oracles passed in (`evalLib`, `call`, …); what is modelled is the Rust control
flow around it.

| here | there |
| --- | --- |
| `Caught`, `errMsg`        | `CaughtError`, `caught_error_message` :70-102 |
| `normNotDefined`          | the `"x is not defined"` → `"'x' is not defined"` rewrite :81-85 |
| `effTimeout`              | `compute_effective_js_timeout_ms` :130-137 |
| `validateTimeout`         | the `effective_ms` of `validate_script` :164-168 |
| `validateScript`          | `validate_script` :154-184 (setup → eval → collect) |
| `Cache`, `ensureCurrent`  | `JS_STATE`, `ensure_context_current` :187-202 |
| `rebuild`                 | `rebuild_context` :204-277 |
| `bridge`                  | `call_udf_bridge` :281-355 (error mapping :328-338; graph slot cleared :351) |
-/
namespace AlgoUdf.UdfContext

/-! ## Error messages -/

inductive Caught
  | error (display : String)                         -- `CaughtError::Error`
  | exception (message : Option String) (name : Option String)
  | value (asString : Option String) (debug : String)

def notDefSuffix : List Char := " is not defined".toList

/-- `msg.strip_suffix(" is not defined")` followed by the quote check. -/
def normNotDefined (msg : List Char) : List Char :=
  if notDefSuffix.isSuffixOf msg then
    let stripped := msg.take (msg.length - notDefSuffix.length)
    if stripped.head? = some '\'' then msg else ['\''] ++ stripped ++ ['\''] ++ notDefSuffix
  else msg

def errMsg (e : Caught) (includeName : Bool) : List Char :=
  match e with
  | .error d => d.toList
  | .exception m n =>
    let msg := normNotDefined ((m.getD "").toList)
    if includeName then
      match n with
      | some nm => if nm ≠ "" ∧ nm ≠ "Error" then nm.toList ++ ": ".toList ++ msg else msg
      | none => msg
    else msg
  | .value s d => (s.getD d).toList

/-- The rewrite matches C QuickJS's ReferenceError text. -/
theorem norm_example : normNotDefined "foo is not defined".toList = "'foo' is not defined".toList := by
  decide

/-- Already-quoted messages and unrelated messages are left alone. -/
theorem norm_quoted : normNotDefined "'foo' is not defined".toList = "'foo' is not defined".toList := by
  decide
theorem norm_other (m : List Char) (h : notDefSuffix.isSuffixOf m = false) : normNotDefined m = m := by
  simp [normNotDefined, h]

/-- The rewrite is idempotent. -/
theorem norm_idem_example :
    normNotDefined (normNotDefined "x is not defined".toList) = normNotDefined "x is not defined".toList := by
  decide

theorem errMsg_name (m nm : String) (h1 : nm ≠ "") (h2 : nm ≠ "Error") :
    errMsg (.exception (some m) (some nm)) true = nm.toList ++ ": ".toList ++ normNotDefined m.toList := by
  simp [errMsg, h1, h2]

theorem errMsg_plain_error (m : String) :
    errMsg (.exception (some m) (some "Error")) true = normNotDefined m.toList ∧
    errMsg (.exception (some m) (some "TypeError")) false = normNotDefined m.toList := by
  simp [errMsg]

/-! ## Timeouts -/

def absCap : Int := 30000
def validateCap : Int := 10000

def effTimeout (t : Int) : Int := if t > 0 then min t absCap else absCap
def validateTimeout (t : Int) : Int := if t > 0 then min t validateCap else validateCap

/-- Every UDF call and every traversal deadline is in `(0, 30 s]`; `0`
("unlimited") and negative settings fall back to the 30 s cap. -/
theorem effTimeout_bounds (t : Int) : 0 < effTimeout t ∧ effTimeout t ≤ absCap := by
  unfold effTimeout absCap; split <;> omega

theorem effTimeout_spec (t : Int) :
    (t ≤ 0 → effTimeout t = absCap) ∧ (0 < t → t ≤ absCap → effTimeout t = t) := by
  unfold effTimeout absCap; constructor <;> intro h <;> (try intro h') <;> split <;> omega

theorem validateTimeout_bounds (t : Int) : 0 < validateTimeout t ∧ validateTimeout t ≤ validateCap := by
  unfold validateTimeout validateCap; split <;> omega

/-! ## validate_script -/

/-- `validate_script`: setup the validation globals, eval the code (errors are
caught and formatted with the error name), then collect the registered names. -/
def validateScript (setup : Except String Unit) (eval : Except Caught Unit)
    (collect : Except String (List String)) : Except String (List String) :=
  match setup with
  | .error e => .error e
  | .ok () =>
    match eval with
    | .error c => .error (String.ofList (errMsg c true))
    | .ok () => collect

theorem validateScript_spec (eval : Except Caught Unit) (collect : Except String (List String)) :
    (∀ c, eval = .error c → validateScript (.ok ()) eval collect = .error (String.ofList (errMsg c true))) ∧
    (eval = .ok () → validateScript (.ok ()) eval collect = collect) := by
  constructor
  · intro c h; subst h; rfl
  · intro h; subst h; rfl

/-! ## Cached context -/

structure Lib where
  name : String
  fns : List String       -- qualified names, `UdfLibrary.function_names`

structure Cache where
  version : Nat
  funcs : List String     -- the keys of `persistent_funcs`

/-- `rebuild_context`: the old state is dropped first (so a failed rebuild
leaves `None`); every library is evaluated in order, the first failure aborts
with its formatted message; then each qualified name found in the JS registry
is cached under its lowercase. -/
def rebuild (low : String → String) (libs : List Lib) (evalLib : Lib → Except Caught Unit)
    (raw : List String) (v : Nat) : Except String Cache :=
  match libs.find? (fun l => (evalLib l).toBool == false) with
  | some l =>
    match evalLib l with
    | .error c => .error ("Failed to load UDF library '" ++ l.name ++ "': " ++ String.ofList (errMsg c true))
    | .ok () => .error "unreachable"
  | none => .ok ⟨v, (libs.flatMap (·.fns)).filter (· ∈ raw) |>.map low⟩

/-- `ensure_context_current`: rebuild iff no context or a stale version. The
state after a failed rebuild is `None` (the old one was taken first). -/
def ensureCurrent (low : String → String) (st : Option Cache) (repoVersion : Nat) (libs : List Lib)
    (evalLib : Lib → Except Caught Unit) (raw : List String) : Except String Unit × Option Cache :=
  if st.all (·.version == repoVersion) && st.isSome then (.ok (), st)
  else
    match rebuild low libs evalLib raw repoVersion with
    | .ok c => (.ok (), some c)
    | .error e => (.error e, none)

/-- After a successful `ensure_context_current` the cache carries the version
that was read from the repository. -/
theorem ensureCurrent_version (low : String → String) (st : Option Cache) (v : Nat) (libs : List Lib)
    (ev : Lib → Except Caught Unit) (raw : List String) :
    (ensureCurrent low st v libs ev raw).1 = .ok () → ∃ c, (ensureCurrent low st v libs ev raw).2 = some c ∧ c.version = v := by
  unfold ensureCurrent
  split
  · rename_i h
    intro _
    cases st with
    | none => simp at h
    | some c => simp at h; exact ⟨c, rfl, h⟩
  · split
    · rename_i c hc; intro _
      refine ⟨c, rfl, ?_⟩
      unfold rebuild at hc
      split at hc
      · split at hc <;> cases hc
      · cases hc; rfl
    · intro h; cases h

/-- Cached keys are exactly the lowercased qualified names the scripts really
registered. -/
theorem rebuild_keys (low : String → String) (libs : List Lib) (ev : Lib → Except Caught Unit)
    (raw : List String) (v : Nat) (c : Cache) (h : rebuild low libs ev raw v = .ok c) :
    ∀ k, k ∈ c.funcs ↔ ∃ l ∈ libs, ∃ q ∈ l.fns, q ∈ raw ∧ k = low q := by
  unfold rebuild at h
  split at h
  · split at h <;> cases h
  · cases h
    intro k
    simp only [List.mem_map, List.mem_filter, List.mem_flatMap, decide_eq_true_eq]
    constructor
    · rintro ⟨q, ⟨⟨l, hl, hq⟩, hr⟩, rfl⟩; exact ⟨l, hl, q, hq, hr, rfl⟩
    · rintro ⟨l, hl, q, hq, hr, rfl⟩; exact ⟨q, ⟨⟨l, hl, hq⟩, hr⟩, rfl⟩

/-- The first failing library aborts the rebuild with its formatted error. -/
theorem rebuild_fail (low : String → String) (l : Lib) (rest : List Lib) (ev : Lib → Except Caught Unit)
    (raw : List String) (v : Nat) (c : Caught) (h : ev l = .error c) :
    rebuild low (l :: rest) ev raw v =
      .error ("Failed to load UDF library '" ++ l.name ++ "': " ++ String.ofList (errMsg c true)) := by
  simp [rebuild, List.find?_cons, h, Except.toBool]

/-! ## call_udf_bridge -/

def containsSub (s sub : List Char) : Bool := (List.range (s.length + 1)).any (fun i => sub.isPrefixOf (s.drop i))

/-- The error mapping :328-338. -/
def mapCallErr (msg : List Char) : String :=
  if containsSub msg "interrupted".toList then "UDF Exception: Query timed out"
  else if containsSub msg "out of memory".toList || containsSub msg "InternalError: stack overflow".toList then
    "out of memory"
  else "UDF Exception: " ++ String.ofList msg

/-- `call_udf_bridge`: lookup by lowercase name, convert args, call, convert
back; the graph slot is set before the call and cleared after it whatever the
outcome. Returns the result and the final graph slot. -/
def bridge {γ ρ : Type} (low : String → String) (c : Cache) (name : String) (slot : Option γ) (g : γ)
    (args : Except String Unit) (call : Except Caught ρ) : Except String ρ × Option γ :=
  if low name ∉ c.funcs then (.error ("UDF function '" ++ name ++ "' not found in JS context"), slot)
  else
    let _slot := some g
    let r : Except String ρ :=
      match args with
      | .error e => .error e
      | .ok () =>
        match call with
        | .error ce => .error (mapCallErr (errMsg ce false))
        | .ok v => .ok v
    (r, none)

/-- Once the function is found, the current-graph slot is always cleared. -/
theorem bridge_clears {γ ρ : Type} (low : String → String) (c : Cache) (name : String) (slot : Option γ)
    (g : γ) (args : Except String Unit) (call : Except Caught ρ) (h : low name ∈ c.funcs) :
    (bridge low c name slot g args call).2 = none := by
  simp [bridge, h]

theorem bridge_timeout {γ ρ : Type} (low : String → String) (c : Cache) (name : String) (slot : Option γ)
    (g : γ) (h : low name ∈ c.funcs) :
    (bridge (ρ := ρ) low c name slot g (.ok ()) (.error (.exception (some "interrupted") none))).1 =
      .error "UDF Exception: Query timed out" := by
  simp [bridge, h, errMsg, mapCallErr]
  decide

theorem bridge_not_found {γ ρ : Type} (low : String → String) (c : Cache) (name : String) (slot : Option γ)
    (g : γ) (args : Except String Unit) (call : Except Caught ρ) (h : low name ∉ c.funcs) :
    bridge low c name slot g args call = (.error ("UDF function '" ++ name ++ "' not found in JS context"), slot) := by
  simp [bridge, h]

end AlgoUdf.UdfContext
