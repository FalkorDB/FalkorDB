import RedisLayer.Args
/-!
# `GRAPH.CONFIG SET` (`src/commands/config_cmd.rs`)

* `validate` — `validate_config_set` (`config_cmd.rs:114-202`), the integer/boolean
  configs that change behaviour (TIMEOUT family, RESULTSET_SIZE, MAX_QUEUED_QUERIES,
  VKEY_MAX_ENTITY_COUNT, MAX_INFO_QUERIES, JS sizes, CMD_INFO), as of #3021
  (30b4fb7dc). `validatePre3021` keeps the earlier arms as a historical model.
* `dispatch` — the arity / sub-command / name-folding head of `graph_config`
  (`config_cmd.rs:308-343`, #3021).
* `cross` — `validate_timeout_cross_constraints` (`config_cmd.rs:210-242`).
* `apply1` — `apply_config_set` (`config_cmd.rs:251-291`).
* `rustSet` — the `SET` arm of `graph_config` (`config_cmd.rs:336-379`): validate all,
  cross-check once against the *final* values, then apply all.
* `cSet` — C's `_Config_set` (`cmd_config.c`): dry-run each pair against the *current*
  configuration, then apply each in order. C's per-field validators are only modelled
  for the timeout family (`Config_timeout_default_set` / `_max_set`, which compare
  against the stored other value).
-/

namespace RedisLayer

inductive Name where
  | timeout | timeoutDefault | timeoutMax | resultsetSize | maxQueued | vkeyMax
  | maxInfo | jsHeap | cmdInfo | cacheSize
  deriving DecidableEq, Repr

structure Cfg where
  timeout : Int := 0
  timeoutDefault : Int := 0
  timeoutMax : Int := 0
  resultsetSize : Int := -1
  maxQueued : Nat := 2 ^ 32 - 1
  vkeyMax : Int := 100000
  maxInfo : Int := 1000
  jsHeap : Int := 268435456
  cmdInfo : Bool := true
  deriving DecidableEq, Repr

inductive CfgErr where
  | failed | readOnly | deprecated | defaultAboveMax | maxBelowDefault
  | jsTooSmall   -- "`{name}` must be at least 1MB (1048576)" (#3021, `config_cmd.rs:185-188`)
  deriving DecidableEq, Repr

/-- `JS_MIN_SIZE` (`config_cmd.rs:54`). -/
def jsMinSize : Int := 1048576

/-- `validate_config_set` (`config_cmd.rs:114-202`) since #3021: the parsed value, or
the error. -/
def validate (n : Name) (v : List Nat) : Except CfgErr Int :=
  match n with
  | .timeout | .timeoutDefault | .timeoutMax =>     -- `:120-134`
    match rustParseI64 v with
    | some x => if x < 0 then .error .failed else .ok x
    | none => .error .failed
  | .resultsetSize =>                               -- `:148-154`
    match rustParseI64 v with
    | some x => .ok (if x < 0 then -1 else x)
    | none => .error .failed
  | .maxQueued =>                                   -- `:155-163`
    match rustParseI64 v with
    | some x => if x ≤ 0 then .error .failed else .ok x
    | none => .error .failed
  | .vkeyMax =>                                     -- `:164-172`: negative now refused
    match rustParseI64 v with
    | some x => if x < 0 then .error .failed else .ok x
    | none => .error .failed
  | .maxInfo =>                                     -- `:173-183`
    match rustParseI64 v with
    | some x => if x < 0 then .error .failed else .ok (min x 1000)
    | none => .error .failed
  | .jsHeap =>                                      -- `:185-188`: at least 1MB, one message
    match rustParseI64 v with
    | some x => if x ≥ jsMinSize then .ok x else .error .jsTooSmall
    | none => .error .jsTooSmall
  | .cmdInfo =>                                     -- `:138-147`: `eq_ignore_ascii_case` yes / no
    if eqIC v (b "yes") then .ok 1
    else if eqIC v (b "no") then .ok 0
    else .error .failed
  | .cacheSize => .error .readOnly

/-- **Historical**: `validate_config_set` before #3021 (da6f808c3). -/
def validatePre3021 (n : Name) (v : List Nat) : Except CfgErr Int :=
  match n with
  | .timeout | .timeoutDefault | .timeoutMax =>
    match rustParseI64 v with
    | some x => if x < 0 then .error .failed else .ok x
    | none => .error .failed
  | .resultsetSize =>
    match rustParseI64 v with
    | some x => .ok (if x < 0 then -1 else x)
    | none => .error .failed
  | .maxQueued =>
    match rustParseI64 v with
    | some x => if x ≤ 0 then .error .failed else .ok x
    | none => .error .failed
  | .vkeyMax =>                                     -- `config_cmd.rs:175-180`: no range check
    match rustParseI64 v with
    | some x => .ok x
    | none => .error .failed
  | .maxInfo =>
    match rustParseI64 v with
    | some x => if x < 0 then .error .failed else .ok (min x 1000)
    | none => .error .failed
  | .jsHeap =>
    match rustParseI64 v with
    | some x => if x < 0 then .error .failed else .ok x
    | none => .error .failed
  | .cmdInfo =>
    -- `value.to_lowercase()` against yes/1/true, no/0/false
    let l := v.map lowerA
    if l = b "yes" ∨ l = b "1" ∨ l = b "true" then .ok 1
    else if l = b "no" ∨ l = b "0" ∨ l = b "false" then .ok 0
    else .error .failed
  | .cacheSize => .error .readOnly

def apply1 (c : Cfg) : Name × Int → Cfg
  | (.timeout, x) => { c with timeout := x }
  | (.timeoutDefault, x) => { c with timeoutDefault := x }
  | (.timeoutMax, x) => { c with timeoutMax := x }
  | (.resultsetSize, x) => { c with resultsetSize := x }
  | (.maxQueued, x) => { c with maxQueued := x.toNat }
  | (.vkeyMax, x) => { c with vkeyMax := x }
  | (.maxInfo, x) => { c with maxInfo := x }
  | (.jsHeap, x) => { c with jsHeap := x }
  | (.cmdInfo, x) => { c with cmdInfo := x ≠ 0 }
  | (.cacheSize, _) => c

/-- `validate_timeout_cross_constraints`: against the values *after* the whole batch. -/
def cross (c : Cfg) (vs : List (Name × Int)) : Except CfgErr Unit :=
  let c' := vs.foldl apply1 c
  let settingTimeout := vs.any (·.1 == .timeout)
  if settingTimeout ∧ (c'.timeoutDefault > 0 ∨ c'.timeoutMax > 0) then .error .deprecated
  else if c'.timeoutDefault > 0 ∧ c'.timeoutMax > 0 ∧ c'.timeoutDefault > c'.timeoutMax then
    (if vs.any (·.1 == .timeoutDefault) then .error .defaultAboveMax else .error .maxBelowDefault)
  else .ok ()

def validateAll : List (Name × List Nat) → Except CfgErr (List (Name × Int))
  | [] => .ok []
  | (n, v) :: rest =>
    match validate n v, validateAll rest with
    | .error e, _ => .error e
    | .ok _, .error e => .error e
    | .ok x, .ok xs => .ok ((n, x) :: xs)

def rustSet (c : Cfg) (ps : List (Name × List Nat)) : Except CfgErr Cfg :=
  match validateAll ps with
  | .error e => .error e
  | .ok vs =>
    match cross c vs with
    | .error e => .error e
    | .ok () => .ok (vs.foldl apply1 c)

/-- The timeout invariant: a default never exceeds a configured maximum. -/
def TInv (c : Cfg) : Prop := c.timeoutDefault > 0 → c.timeoutMax > 0 → c.timeoutDefault ≤ c.timeoutMax

theorem validateAll_ok_map (ps : List (Name × List Nat)) (vs : List (Name × Int))
    (h : validateAll ps = .ok vs) : vs.map (·.1) = ps.map (·.1) := by
  induction ps generalizing vs with
  | nil => simp [validateAll] at h; subst h; rfl
  | cons p rest ih =>
    obtain ⟨n, v⟩ := p
    simp only [validateAll] at h
    split at h
    · simp at h
    · simp at h
    · rename_i hr
      simp at h; subst h
      simp [ih _ hr]

/-- **Every successful `GRAPH.CONFIG SET` leaves the timeout invariant true**, whatever
the batch and whatever state it started from (the cross check reads the final values). -/
theorem rustSet_inv (c c' : Cfg) (ps : List (Name × List Nat)) (h : rustSet c ps = .ok c') :
    TInv c' := by
  unfold rustSet at h
  split at h
  · simp at h
  · rename_i vs _
    split at h
    · simp at h
    · rename_i hc
      simp at h; subst h
      unfold cross at hc
      intro h1 h2
      simp only at hc
      split at hc
      · simp at hc
      · split at hc
        · split at hc <;> simp at hc
        · rename_i hn; omega


/-- The deprecated `TIMEOUT` can only be set while neither new knob is in force. -/
theorem rustSet_timeout_deprecated (c c' : Cfg) (ps : List (Name × List Nat))
    (h : rustSet c ps = .ok c') (ht : ps.any (·.1 == .timeout)) :
    c'.timeoutDefault ≤ 0 ∧ c'.timeoutMax ≤ 0 := by
  unfold rustSet at h
  split at h
  · simp at h
  · rename_i vs hv
    have hm := validateAll_ok_map ps vs hv
    have ht' : vs.any (·.1 == .timeout) := by
      rw [List.any_eq_true] at ht ⊢
      obtain ⟨x, hx, hx2⟩ := ht
      have : x.1 ∈ vs.map (·.1) := by rw [hm]; exact List.mem_map_of_mem hx
      obtain ⟨y, hy, hy2⟩ := List.mem_map.mp this
      exact ⟨y, hy, by rw [hy2]; exact hx2⟩
    split at h
    · simp at h
    · rename_i hc
      simp at h; subst h
      unfold cross at hc
      simp only at hc
      split at hc
      · simp at hc
      · rename_i hn
        simp only [ht', true_and, not_or] at hn
        omega

/-! ## C's sequential `_Config_set`

C dry-runs each pair against the configuration *as it is before the batch*, then applies
them one by one. For `TIMEOUT_DEFAULT`/`TIMEOUT_MAX` each dry-run compares against the
*stored* other value, so a batch that raises the default and lowers the maximum at once
passes both dry-runs. -/

def cDry (c : Cfg) : Name × Int → Bool
  | (.timeoutDefault, x) => !(c.timeoutMax > 0 ∧ x > c.timeoutMax)
  | (.timeoutMax, x) => !(x > 0 ∧ c.timeoutDefault > 0 ∧ x < c.timeoutDefault)
  | _ => true

def cSet (c : Cfg) (vs : List (Name × Int)) : Option Cfg :=
  if vs.all (cDry c) then some (vs.foldl apply1 c) else none

/-- **C breaks the invariant** (confirmed live: `SET TIMEOUT_DEFAULT 10 TIMEOUT_MAX 5`
answers OK on C and leaves `TIMEOUT_DEFAULT = 10 > TIMEOUT_MAX = 5`); Rust refuses. -/
theorem c_breaks_inv :
    let c : Cfg := {}
    let ps := [(Name.timeoutDefault, b "10"), (Name.timeoutMax, b "5")]
    (∃ c', cSet c [(.timeoutDefault, 10), (.timeoutMax, 5)] = some c' ∧ ¬ TInv c') ∧
    rustSet c ps = .error .defaultAboveMax := by
  refine ⟨⟨_, rfl, ?_⟩, by decide⟩
  unfold TInv; decide

/-! ## Per-field validators: the pre-#3021 divergences (historical) and the fix

Before #3021 these were divergences from C, all confirmed live; #3021 (30b4fb7dc,
issue #3020) fixed each one. The `pre3021_` theorems are the original
counterexamples, about `validatePre3021`; the theorems after them are about the
current `validate`. -/

/-- Historical: `VKEY_MAX_ENTITY_COUNT` accepted a negative value (C: "Failed to set
config value"); it was then read as `vkey_max as u64` (`redis_type.rs`), ≈ 2^64. -/
theorem pre3021_vkey_negative_accepted : validatePre3021 .vkeyMax (b "-5") = .ok (-5) := by decide

/-- Historical: `CMD_INFO` accepted `1`, `TRUE`, `true` — C accepts only `yes`/`no`. -/
theorem pre3021_cmdinfo_extra_spellings :
    validatePre3021 .cmdInfo (b "1") = .ok 1 ∧ validatePre3021 .cmdInfo (b "TRUE") = .ok 1 := by decide

/-- Historical: `JS_HEAP_SIZE 5` was accepted (C: "JS_HEAP_SIZE must be at least 1MB (1048576)"). -/
theorem pre3021_jsheap_tiny_accepted : validatePre3021 .jsHeap (b "5") = .ok 5 := by decide

/-- `MAX_INFO_QUERIES` is clamped, not rejected, as in C. -/
theorem maxinfo_clamped : validate .maxInfo (b "5000") = .ok 1000 := by decide

/-- Fixed: the three counterexamples are now refused, as C refuses them. -/
theorem fixed3021_counterexamples :
    validate .vkeyMax (b "-5") = .error .failed ∧
    validate .cmdInfo (b "1") = .error .failed ∧ validate .cmdInfo (b "TRUE") = .error .failed ∧
    validate .jsHeap (b "5") = .error .jsTooSmall ∧
    validate .cmdInfo (b "YeS") = .ok 1 ∧ validate .cmdInfo (b "no") = .ok 0 ∧
    validate .jsHeap (b "1048576") = .ok 1048576 := by decide

/-- Every accepted `VKEY_MAX_ENTITY_COUNT` is non-negative, so its `as u64` read is exact. -/
theorem vkey_ok_nonneg (v : List Nat) (x : Int) (h : validate .vkeyMax v = .ok x) : 0 ≤ x := by
  simp only [validate] at h
  split at h
  · split at h
    · cases h
    · cases h; omega
  · cases h

/-- Every accepted JS heap/stack size is at least 1MB (C's bound). -/
theorem jsheap_ok_ge (v : List Nat) (x : Int) (h : validate .jsHeap v = .ok x) : jsMinSize ≤ x := by
  simp only [validate] at h
  split at h
  · split at h
    · cases h; omega
    · cases h
  · cases h

/-- `CMD_INFO` accepts exactly the ASCII-case-insensitive `yes` / `no`, as `1` / `0`. -/
theorem cmdinfo_ok_iff (v : List Nat) (x : Int) :
    validate .cmdInfo v = .ok x ↔ (eqIC v (b "yes") ∧ x = 1) ∨ (eqIC v (b "no") = true ∧ eqIC v (b "yes") = false ∧ x = 0) := by
  simp only [validate]
  by_cases h1 : eqIC v (b "yes") = true
  · simp only [h1, if_true, Except.ok.injEq, true_and]
    constructor
    · intro h; exact Or.inl h.symm
    · rintro (h | ⟨_, h, _⟩)
      · exact h.symm
      · simp at h
  · have h1' : eqIC v (b "yes") = false := by simpa using h1
    simp only [h1', Bool.false_eq_true, if_false, false_and, false_or, true_and]
    by_cases h2 : eqIC v (b "no") = true
    · simp only [h2, if_true, Except.ok.injEq, true_and]
      exact ⟨fun h => h.symm, fun h => h.symm⟩
    · have h2' : eqIC v (b "no") = false := by simpa using h2
      simp only [h2', Bool.false_eq_true, if_false, false_and, iff_false]
      intro h; cases h

/-! ## `graph_config` head (`config_cmd.rs:308-343`, #3021)

`argc` counts `GRAPH.CONFIG` itself. Sub-command and names fold with
`to_ascii_uppercase` (`upA`); before #3021 they used Unicode `to_uppercase`, which
mapped a dotless `ı` (U+0131) onto `I`, so `tımeout` named `TIMEOUT`. -/

/-- `u8::to_ascii_uppercase`, per code point. -/
def upperA (c : Nat) : Nat := if 97 ≤ c ∧ c ≤ 122 then c - 32 else c
def upA (s : List Nat) : List Nat := s.map upperA

inductive Dispatch where
  | wrongArity | get | set | unknownSub
  deriving DecidableEq, Repr

def dispatch (argc : Nat) (sub : List Nat) : Dispatch :=
  if argc < 3 then .wrongArity
  else if upA sub = b "GET" then (if argc ≠ 3 then .wrongArity else .get)
  else if upA sub = b "SET" then (if argc < 4 ∨ argc % 2 = 1 then .wrongArity else .set)
  else .unknownSub

/-- Fixed (#3021): `GET x extra` is a wrong-arity error, as in C (was: accepted). -/
theorem get_extra_wrong_arity : dispatch 4 (b "get") = .wrongArity := by decide

/-- `SET` reaches the pair loop only with complete name/value pairs, so the loop's
"Missing value" and "Missing configuration parameter name" errors are unreachable. -/
theorem set_pairs_complete (argc : Nat) (sub : List Nat) (h : dispatch argc sub = .set) :
    4 ≤ argc ∧ (argc - 2) % 2 = 0 := by
  unfold dispatch at h
  split at h
  · cases h
  · split at h
    · split at h <;> cases h
    · split at h
      · split at h
        · cases h
        · rename_i hn; omega
      · cases h

/-- Fixed (#3021): ASCII folding leaves a dotless `ı` alone, so `tımeout` is not `TIMEOUT`. -/
theorem dotless_i_not_folded : upA (b "tımeout") ≠ b "TIMEOUT" ∧ upA (b "timeout") = b "TIMEOUT" := by
  decide

/-- `ASYNC_DELETE`'s default (`src/config.rs:110`) is now C's `1` (was `0`). -/
def asyncDeleteDefault : Int := 1
theorem asyncDelete_default_matches_C : asyncDeleteDefault = 1 := rfl

end RedisLayer
