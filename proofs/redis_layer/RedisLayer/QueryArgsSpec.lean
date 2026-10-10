import RedisLayer.QueryArgsAgree

/-!
# What `parse_query_flags` accepts, exactly

* `parse_ok_iff` — the parser succeeds **iff** the vector has 3..8 arguments and its
  tail is in `Grammar`: a sequence of tokens, each either a non-keyword argument
  (`--compact`, `--track-memory`, or anything unknown: skipped), `TIMEOUT t` with `t` a
  canonical (`string2ll`) integer in `0..=TIMEOUT_MAX`, or `version v` with `v` a canonical
  integer in `0..=u32::MAX`. Keywords match ASCII case-insensitively up to the first NUL.
* `parse_documented` — the documented rendering
  `GRAPH.<CMD> key query [--compact] [TIMEOUT ms] [version v] [--track-memory]`
  is accepted with the documented meaning (last timeout/version wins).
* The #3009 regressions: each pre-`557f18868` divergence from C, now agreement.
-/

namespace RedisLayer

inductive Grammar (tmax : Int) : List Bytes → Prop
  | nil : Grammar tmax []
  | other {a : Bytes} {rest : List Bytes} :
      eqIC (upToNul a) (b "timeout") = false → eqIC (upToNul a) (b "version") = false →
      Grammar tmax rest → Grammar tmax (a :: rest)
  | timeout {a v : Bytes} {rest : List Bytes} (t : Int) :
      eqIC (upToNul a) (b "timeout") = true → string2ll v = some t → 0 ≤ t →
      ¬ (tmax > 0 ∧ t > tmax) → Grammar tmax rest → Grammar tmax (a :: v :: rest)
  | version {a v : Bytes} {rest : List Bytes} (n : Int) :
      eqIC (upToNul a) (b "version") = true → string2ll v = some n → 0 ≤ n → n ≤ (uintMax : Int) →
      Grammar tmax rest → Grammar tmax (a :: v :: rest)

theorem grammar_scan_ok (tmax : Int) {l : List Bytes} (g : Grammar tmax l) :
    ∀ f, ∃ r, scanFlags tmax l f = .ok r := by
  induction g with
  | nil => exact fun f => ⟨f, rfl⟩
  | @other a rest hT hV _ ih =>
    intro f
    unfold scanFlags
    by_cases h1 : eqIC (upToNul a) (b "--compact") = true
    · simp only [h1, ↓reduceIte]; exact ih _
    by_cases h2 : eqIC (upToNul a) (b "--track-memory") = true
    · simp only [h1, h2, ↓reduceIte, Bool.false_eq_true]; exact ih _
    simp only [h1, h2, hT, hV, Bool.false_eq_true, ↓reduceIte]; exact ih f
  | @timeout a v rest t hT hp h0 hm _ ih =>
    intro f
    have h1 := eqIC_excl hT k4
    have h2 := eqIC_excl hT k5
    unfold scanFlags
    simp only [h1, h2, hT, hp, hm, show ¬ t < 0 by omega, Bool.false_eq_true, ↓reduceIte]
    exact ih _
  | @version a v rest n hV hp h0 hu _ ih =>
    intro f
    have h1 := eqIC_excl hV k6
    have h2 := eqIC_excl hV k7
    have h3 := eqIC_excl hV k10
    unfold scanFlags
    simp only [h1, h2, h3, hV, hp, h0, hu, and_self, Bool.false_eq_true, ↓reduceIte]
    exact ih _

theorem scan_ok_grammar (tmax : Int) : ∀ (l : List Bytes) (f r : QFlags),
    scanFlags tmax l f = .ok r → Grammar tmax l
  | [], _, _, _ => .nil
  | a :: rest, f, r, h => by
    unfold scanFlags at h
    by_cases h1 : eqIC (upToNul a) (b "--compact") = true
    · simp only [h1, ↓reduceIte] at h
      exact .other (eqIC_excl h1 k8) (eqIC_excl h1 k9) (scan_ok_grammar tmax rest _ _ h)
    by_cases h2 : eqIC (upToNul a) (b "--track-memory") = true
    · simp only [h1, h2, ↓reduceIte, Bool.false_eq_true] at h
      exact .other (eqIC_excl h2 k2) (eqIC_excl h2 k3) (scan_ok_grammar tmax rest _ _ h)
    by_cases h3 : eqIC (upToNul a) (b "timeout") = true
    · simp only [h1, h2, h3, ↓reduceIte, Bool.false_eq_true] at h
      cases rest with
      | nil => simp at h
      | cons v rest' =>
        simp only at h
        cases hp : string2ll v with
        | none => simp [hp] at h
        | some t =>
          simp only [hp] at h
          by_cases hm : tmax > 0 ∧ t > tmax
          · simp [hm] at h
          · by_cases hn : t < 0
            · simp [hm, hn] at h
            · simp only [hm, hn, ↓reduceIte] at h
              exact .timeout t h3 hp (by omega) hm (scan_ok_grammar tmax rest' _ _ h)
    by_cases h4 : eqIC (upToNul a) (b "version") = true
    · simp only [h1, h2, h3, h4, ↓reduceIte, Bool.false_eq_true] at h
      cases rest with
      | nil => simp at h
      | cons v rest' =>
        simp only at h
        cases hp : string2ll v with
        | none => simp [hp] at h
        | some n =>
          simp only [hp] at h
          by_cases hn : 0 ≤ n ∧ n ≤ (uintMax : Int)
          · simp only [hn, and_self, ↓reduceIte] at h
            exact .version n h4 hp hn.1 hn.2 (scan_ok_grammar tmax rest' _ _ h)
          · simp [hn] at h
    · simp only [h1, h2, h3, h4, ↓reduceIte, Bool.false_eq_true] at h
      exact .other (by simpa using h3) (by simpa using h4) (scan_ok_grammar tmax rest _ _ h)

/-- **The parser accepts exactly the grammar** (and is total: `parse_total`). -/
theorem parse_ok_iff (tmax : Int) (args : List Bytes) :
    (∃ r, parseQueryFlags tmax args = .ok r) ↔
      3 ≤ args.length ∧ args.length ≤ 8 ∧ Grammar tmax (args.drop 3) := by
  unfold parseQueryFlags maxArgs
  constructor
  · rintro ⟨r, h⟩
    split at h
    · simp at h
    · exact ⟨by omega, by omega, scan_ok_grammar tmax _ _ _ h⟩
  · rintro ⟨h1, h2, g⟩
    rw [if_neg (by omega)]
    exact grammar_scan_ok tmax g {}

/-! ## The documented rendering and its meaning -/

inductive Flag where
  | compact
  | track
  | timeout (ms : Nat)
  | version (v : Nat)
  deriving DecidableEq, Repr

def Flag.render : Flag → List Bytes
  | .compact => [b "--compact"]
  | .track => [b "--track-memory"]
  | .timeout n => [b "TIMEOUT", dec n]
  | .version n => [b "version", dec n]

def renderAll (fs : List Flag) : List Bytes := fs.flatMap Flag.render

def Flag.ok (tmax : Int) : Flag → Prop
  | .timeout n => n ≤ i64Max ∧ (tmax > 0 → (n : Int) ≤ tmax)
  | .version n => n ≤ uintMax
  | _ => True

def Flag.apply (f : QFlags) : Flag → QFlags
  | .compact => { f with compact := true }
  | .track => { f with track := true }
  | .timeout n => { f with timeout := some n }
  | .version n => { f with version := some n }

def specFlags (fs : List Flag) (f : QFlags) : QFlags := fs.foldl Flag.apply f

private theorem d1 : eqIC (upToNul (b "TIMEOUT")) (b "--compact") = false := by decide
private theorem d2 : eqIC (upToNul (b "TIMEOUT")) (b "--track-memory") = false := by decide
private theorem d3 : eqIC (upToNul (b "TIMEOUT")) (b "timeout") = true := by decide
private theorem d4 : eqIC (upToNul (b "version")) (b "--compact") = false := by decide
private theorem d5 : eqIC (upToNul (b "version")) (b "--track-memory") = false := by decide
private theorem d6 : eqIC (upToNul (b "version")) (b "timeout") = false := by decide
private theorem d7 : eqIC (upToNul (b "version")) (b "version") = true := by decide
private theorem d8 : eqIC (upToNul (b "--compact")) (b "--compact") = true := by decide
private theorem d9 : eqIC (upToNul (b "--track-memory")) (b "--compact") = false := by decide
private theorem d10 : eqIC (upToNul (b "--track-memory")) (b "--track-memory") = true := by decide

theorem scan_step (tmax : Int) (x : Flag) (hx : x.ok tmax) (rest : List Bytes) (f : QFlags) :
    scanFlags tmax (x.render ++ rest) f = scanFlags tmax rest (x.apply f) := by
  cases x with
  | compact => rw [Flag.render, List.singleton_append, scanFlags.eq_def]; simp [d8, Flag.apply]
  | track => rw [Flag.render, List.singleton_append, scanFlags.eq_def]; simp [d9, d10, Flag.apply]
  | timeout n =>
    obtain ⟨h1, h2⟩ := hx
    have hm : ¬ (tmax > 0 ∧ (n : Int) > tmax) := fun ⟨a, c⟩ => by have := h2 a; omega
    rw [Flag.render, List.cons_append, List.singleton_append, scanFlags.eq_def]
    simp [d1, d2, d3, string2ll_dec n h1, hm, Flag.apply, show ¬ ((n : Int) < 0) by omega]
  | version n =>
    have hn : n ≤ i64Max := by unfold Flag.ok uintMax at hx; unfold i64Max; omega
    have hu : (n : Int) ≤ (uintMax : Int) := by unfold Flag.ok at hx; omega
    rw [Flag.render, List.cons_append, List.singleton_append, scanFlags.eq_def]
    simp [d4, d5, d6, d7, string2ll_dec n hn, hu, Flag.apply]

theorem scan_documented (tmax : Int) (fs : List Flag) (hok : ∀ x ∈ fs, x.ok tmax) (f : QFlags) :
    scanFlags tmax (renderAll fs) f = .ok (specFlags fs f) := by
  induction fs generalizing f with
  | nil => rfl
  | cons x xs ih =>
    simp only [renderAll, List.flatMap_cons] at ih ⊢
    rw [scan_step tmax x (hok x (List.mem_cons_self ..))]
    exact ih (fun y hy => hok y (List.mem_cons_of_mem _ hy)) _

/-- **Every documented command line is accepted with its documented meaning**, for any
command name, key and query bytes, as long as it fits C's 8-argument cap. -/
theorem parse_documented (tmax : Int) (cmd key q : Bytes) (fs : List Flag)
    (hok : ∀ x ∈ fs, x.ok tmax) (hlen : (renderAll fs).length ≤ 5) :
    parseQueryFlags tmax (cmd :: key :: q :: renderAll fs) = .ok (specFlags fs {}) := by
  unfold parseQueryFlags maxArgs
  rw [if_neg (by simp; omega)]
  exact scan_documented tmax fs hok {}

/-! ## #3009: the pre-`557f18868` divergences, now agreement

Each was a confirmed live divergence (Rust release vs C `falkordb.so`, `repro_live.py`
A1-A6) against the per-command loops that `557f18868` deleted. Each is now a theorem
that Rust's shared parser and C give corresponding answers. -/

def pre : List Bytes := [b "GRAPH.QUERY", b "g", b "RETURN 1"]
def c0 : TCfg := { timeoutMax := 0, timeoutDefault := 0, legacy := 0 }

/-- Historical (A1): a non-UTF-8 argument used to end Rust's `while let Ok(arg) =
args.next_str()` loop, dropping a later `--compact`. Now bytes are compared as bytes. -/
theorem nonutf8_keeps_compact :
    parseQueryFlags 0 (pre ++ [[255], b "--compact"]) = .ok { compact := true } ∧
    (cReadFlags c0 (pre ++ [[255], b "--compact"])).map (·.compact) = .ok true := by
  decide

/-- Both engines read a flag up to its first NUL (`up_to_nul` / `strcasecmp`). -/
theorem nul_truncated_flag :
    parseQueryFlags 0 (pre ++ [b "--compact" ++ [0, 120]]) = .ok { compact := true } ∧
    (cReadFlags c0 (pre ++ [b "--compact" ++ [0, 120]])).map (·.compact) = .ok true := by
  decide

/-- Historical (A2/A3): `-5`, `+10`, `010` were accepted by Rust's `str::parse::<i64>`;
now `string2ll` + the sign check reject them, as C does. -/
theorem timeout_noncanonical_rejected :
    parseQueryFlags 0 (pre ++ [b "TIMEOUT", b "-5"]) = .error .timeoutParse ∧
    cReadFlags c0 (pre ++ [b "TIMEOUT", b "-5"]) = .error .badTimeout ∧
    parseQueryFlags 0 (pre ++ [b "TIMEOUT", b "+10"]) = .error .timeoutParse ∧
    cReadFlags c0 (pre ++ [b "TIMEOUT", b "+10"]) = .error .badTimeout ∧
    parseQueryFlags 0 (pre ++ [b "TIMEOUT", b "010"]) = .error .timeoutParse ∧
    cReadFlags c0 (pre ++ [b "TIMEOUT", b "010"]) = .error .badTimeout := by
  decide

/-- Historical (A4): RO_QUERY and PROFILE silently ignored a garbage or missing timeout
(and a garbage one cleared an earlier good one). Now every query command rejects it. -/
theorem garbage_timeout_rejected :
    roQueryFront 0 (pre ++ [b "TIMEOUT", b "abc"]) = .error .timeoutParse ∧
    profileFront 0 (pre ++ [b "TIMEOUT", b "abc"]) = .error .timeoutParse ∧
    roQueryFront 0 (pre ++ [b "TIMEOUT"]) = .error .timeoutParse ∧
    roQueryFront 0 (pre ++ [b "TIMEOUT", b "5", b "TIMEOUT", b "x"]) = .error .timeoutParse ∧
    cReadFlags c0 (pre ++ [b "TIMEOUT", b "abc"]) = .error .badTimeout ∧
    cReadFlags c0 (pre ++ [b "TIMEOUT"]) = .error .badTimeout := by
  decide

/-- Historical (A5): `version 4294967296` (> `UINT_MAX`) was accepted by Rust. -/
theorem version_above_uint_rejected :
    parseQueryFlags 0 (pre ++ [b "version", b "4294967296"]) = .error .versionParse ∧
    cReadFlags c0 (pre ++ [b "version", b "4294967296"]) = .error .badVersion ∧
    parseQueryFlags 0 (pre ++ [b "version", b "4294967295"]) = .ok { version := some 4294967295 } := by
  decide

/-- Historical (A6): Rust had no arity cap; both now refuse more than 8 arguments. -/
theorem arity_capped :
    parseQueryFlags 0 (pre ++ (List.range 6).map fun _ => b "x") = .error .wrongArity ∧
    cReadFlags c0 (pre ++ (List.range 6).map fun _ => b "x") = .error .wrongArity ∧
    parseQueryFlags 0 (pre ++ (List.range 5).map fun _ => b "x") = .ok {} := by
  decide

/-- A timeout above a configured `TIMEOUT_MAX` is rejected at parse time by both. -/
theorem timeout_above_max_rejected :
    parseQueryFlags 100 (pre ++ [b "TIMEOUT", b "101"]) = .error .timeoutMax ∧
    cReadFlags { c0 with timeoutMax := 100 } (pre ++ [b "TIMEOUT", b "101"]) = .error .exceedsMax := by
  decide

/-- **Divergence (confirmed live, still open — `graph_core.rs` unchanged).** With
`TIMEOUT_MAX = 100000`, a *write* sent with `TIMEOUT 1`: C arms a 1 ms timeout
(`timeout_rw`), Rust parses `1` but `compute_effective_timeout` arms 100000 ms. -/
theorem write_timeout_divergence :
    let c : TCfg := { timeoutMax := 100000, timeoutDefault := 0, legacy := 0 }
    (parseQueryFlags c.timeoutMax (pre ++ [b "TIMEOUT", b "1"])).map (·.timeout) = .ok (some 1) ∧
    rustTimeout c (some 1) true = .ok (some 100000) ∧
    (cReadFlags c (pre ++ [b "TIMEOUT", b "1"])).map (fun f => (f.timeout, f.timeoutRw))
      = .ok (1, true) := by
  decide

end RedisLayer
