import RedisLayer.QueryArgs

/-!
# `parse_query_flags` agrees with C's `_read_flags`

`RustCAgree` (proved as `rust_c_agree` from #3010 until #3757): for every argument vector,
under the configuration invariant that `GRAPH.CONFIG SET` maintains (`CfgOk`), Rust's
`parseQueryFlags` and C's `cReadFlags` either both fail with corresponding errors
(`QErr.toC`) or both succeed with corresponding flags (`Rel`: same compact flag, same
version, same timeout, where C's `TIMEOUT 0` → default substitution is the one
`compute_effective_timeout` performs for Rust).

**#3757 (`SCHEMA_HASH h`, Rust-only).** Rust now refuses a `SCHEMA_HASH` with a missing
or non-integer value (`Failed to parse schema hash value`); C skips both tokens as
unknown flags. `rust_c_agree_refuted` proves `¬ RustCAgree` on
`GRAPH.QUERY g "RETURN 1" SCHEMA_HASH abc`; `rust_c_same_accept_refuted` likewise.
Since `SCHEMA_HASH requires --compact` (#3757, `a908d8a23`) Rust also refuses a well-formed
`SCHEMA_HASH h` sent without `--compact` (`schemaVerbose`), which C skips too:
`rust_c_agree_except_schemaParse_refuted` proves that the first #3757 form of the headline
(only `schemaParse` excepted) no longer holds, on `GRAPH.QUERY g "RETURN 1" SCHEMA_HASH 5`.
`rust_c_agree_except_schema` proves that these two Rust-only `SCHEMA_HASH` errors are the
only divergence: on every vector Rust does not refuse with `schemaParse` or
`schemaVerbose`, the two agree exactly (a valid `SCHEMA_HASH h` is read by Rust and
skipped, value included, by C).
-/

namespace RedisLayer

/-- C's error for a Rust error; `schemaParse` and `schemaVerbose` (#3757) have none: C
never refuses there. -/
def QErr.toC : QErr → Option CErr
  | .wrongArity => some .wrongArity
  | .timeoutParse => some .badTimeout
  | .timeoutMax => some .exceedsMax
  | .versionParse => some .badVersion
  | .schemaParse => none
  | .schemaVerbose => none

/-- What a Rust flag record means in C terms. -/
def Rel (c : TCfg) (r : QFlags) (g : CFlags) : Prop :=
  r.compact = g.compact ∧ g.timeoutRw = (cInit c).timeoutRw ∧
  g.hash = (match r.version with | none => -1 | some v => (v : Int)) ∧
  g.timeout = (match r.timeout with
    | none => (cInit c).timeout
    | some t => if t = 0 ∧ g.timeoutRw then dflt c else t)

/-- The `GRAPH.CONFIG` invariant (`Config.rustSet_inv`): non-negative values and
`TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` when a max is set. -/
structure CfgOk (c : TCfg) : Prop where
  max_nn : 0 ≤ c.timeoutMax
  def_nn : 0 ≤ c.timeoutDefault
  inv : c.timeoutMax ≠ 0 → c.timeoutDefault ≤ c.timeoutMax

def Agree (c : TCfg) : Except QErr QFlags → Except CErr CFlags → Prop
  | .error e, .error e' => e.toC = some e'
  | .ok r, .ok g => Rel c r g
  | _, _ => False

theorem eqIC_excl {a x y : List Nat} (h : eqIC a x = true) (hxy : eqIC x y = false) :
    eqIC a y = false := by
  simp only [eqIC, beq_iff_eq] at h hxy ⊢
  rw [h]; exact hxy

theorem k1 : eqIC (b "--track-memory") (b "--compact") = false := by decide
theorem k2 : eqIC (b "--track-memory") (b "timeout") = false := by decide
theorem k3 : eqIC (b "--track-memory") (b "version") = false := by decide
theorem k4 : eqIC (b "timeout") (b "--compact") = false := by decide
theorem k5 : eqIC (b "timeout") (b "--track-memory") = false := by decide
theorem k6 : eqIC (b "version") (b "--compact") = false := by decide
theorem k7 : eqIC (b "version") (b "--track-memory") = false := by decide
theorem k8 : eqIC (b "--compact") (b "timeout") = false := by decide
theorem k9 : eqIC (b "--compact") (b "version") = false := by decide
theorem k10 : eqIC (b "version") (b "timeout") = false := by decide
theorem k11 : eqIC (b "--compact") (b "schema_hash") = false := by decide
theorem k12 : eqIC (b "--track-memory") (b "schema_hash") = false := by decide
theorem k13 : eqIC (b "timeout") (b "schema_hash") = false := by decide
theorem k14 : eqIC (b "version") (b "schema_hash") = false := by decide
theorem k15 : eqIC (b "schema_hash") (b "--compact") = false := by decide
theorem k16 : eqIC (b "schema_hash") (b "timeout") = false := by decide
theorem k17 : eqIC (b "schema_hash") (b "version") = false := by decide

theorem psh_neg_nil : parseSchemaHash [45] = none := by decide

/-- An accepted schema-hash value starts with a digit, or with `-` and a digit. -/
theorem parseSchemaHash_head (v : Bytes) (h : Nat) (hv : parseSchemaHash v = some h) :
    ∃ d ds, v = d :: ds ∧ ((48 ≤ d ∧ d ≤ 57) ∨
      (d = 45 ∧ ∃ e es, ds = e :: es ∧ 49 ≤ e ∧ e ≤ 57)) := by
  cases v with
  | nil => simp [parseSchemaHash, string2ll] at hv
  | cons d ds =>
    refine ⟨d, ds, rfl, ?_⟩
    by_cases hd : 48 ≤ d ∧ d ≤ 57
    · exact .inl hd
    right
    have hd' : ¬ (49 ≤ d ∧ d ≤ 57) := fun ⟨a, c⟩ => hd ⟨by omega, c⟩
    by_cases h45 : d = 45
    · subst h45
      cases ds with
      | nil => rw [psh_neg_nil] at hv; cases hv
      | cons e es =>
        by_cases he : 49 ≤ e ∧ e ≤ 57
        · exact ⟨rfl, e, es, rfl, he⟩
        · exfalso
          unfold parseSchemaHash string2ll at hv
          simp only [he, ↓reduceIte] at hv
          repeat' (first | (split at hv) | simp_all)
    · exfalso
      unfold parseSchemaHash string2ll at hv
      simp only [h45, hd', ↓reduceIte] at hv
      repeat' (first | (split at hv) | simp_all)

theorem eqIC_cons_ne {d x0 : Nat} {ds xs : Bytes} (h : lowerA d ≠ lowerA x0) :
    eqIC (d :: ds) (x0 :: xs) = false := by
  simp [eqIC, h]

theorem eqIC_cons_tail {d : Nat} {ds xs : Bytes} (h : eqIC ds xs = false) :
    eqIC (d :: ds) (d :: xs) = false := by
  simp only [eqIC, List.map_cons] at h ⊢
  simp [h]

theorem upToNul_cons (d : Nat) (ds : Bytes) (hd : d ≠ 0) :
    ∃ t, upToNul (d :: ds) = d :: t ∧ (∀ e es, ds = e :: es → e ≠ 0 → ∃ t', t = e :: t') := by
  unfold upToNul
  simp only [List.findIdx?_cons, beq_iff_eq, hd, ↓reduceIte]
  cases hf : List.findIdx? (· == 0) ds with
  | none =>
    refine ⟨ds, by simp, fun e es he _ => ⟨es, he⟩⟩
  | some k =>
    refine ⟨ds.take k, by simp, fun e es he he0 => ?_⟩
    subst he
    cases k with
    | zero => simp [List.findIdx?_cons, he0] at hf
    | succ k => exact ⟨es.take k, by simp⟩

theorem kw_compact : b "--compact" = 45 :: 45 :: b "compact" := by decide
theorem kw_timeout : b "timeout" = 116 :: b "imeout" := by decide
theorem kw_version : b "version" = 118 :: b "ersion" := by decide

theorem lowerA_lt (x : Nat) (h : x < 65) : lowerA x = x := by unfold lowerA; split <;> omega

/-- A value `parseSchemaHash` accepts is no keyword C recognises, so C skips it. -/
theorem schemaVal_not_kw (v : Bytes) (h : Nat) (hv : parseSchemaHash v = some h) :
    eqIC (upToNul v) (b "--compact") = false ∧ eqIC (upToNul v) (b "timeout") = false ∧
    eqIC (upToNul v) (b "version") = false := by
  obtain ⟨d, ds, rfl, hd⟩ := parseSchemaHash_head v h hv
  rcases hd with ⟨h1, h2⟩ | ⟨rfl, e, es, rfl, h1, h2⟩
  · obtain ⟨t, ht, -⟩ := upToNul_cons d ds (by omega)
    have hl : lowerA d = d := lowerA_lt d (by omega)
    rw [ht, kw_compact, kw_timeout, kw_version]
    refine ⟨eqIC_cons_ne ?_, eqIC_cons_ne ?_, eqIC_cons_ne ?_⟩ <;> rw [hl] <;>
      unfold lowerA <;> split <;> omega
  · obtain ⟨t, ht, hn⟩ := upToNul_cons 45 (e :: es) (by omega)
    obtain ⟨t', rfl⟩ := hn e es rfl (by omega)
    have hl : lowerA e = e := lowerA_lt e (by omega)
    rw [ht, kw_compact, kw_timeout, kw_version]
    refine ⟨eqIC_cons_tail (eqIC_cons_ne ?_), eqIC_cons_ne (by decide), eqIC_cons_ne (by decide)⟩
    rw [hl]; unfold lowerA; split <;> omega

theorem dflt_le (c : TCfg) (hc : CfgOk c) (h : c.timeoutMax ≠ 0) : dflt c ≤ c.timeoutMax := by
  unfold dflt; split
  · omega
  · exact hc.inv h

theorem dflt_nn (c : TCfg) (hc : CfgOk c) : 0 ≤ dflt c := by
  unfold dflt; split
  · exact hc.max_nn
  · exact hc.def_nn

theorem scan_agree (c : TCfg) (hc : CfgOk c) : ∀ (l : List Bytes) (r : QFlags) (g : CFlags),
    Rel c r g → (c.timeoutMax ≠ 0 → g.timeout ≤ c.timeoutMax) →
    scanFlags c.timeoutMax l r ≠ .error .schemaParse →
    Agree c (scanFlags c.timeoutMax l r) (cScan c l g)
  | [], r, g, hr, _, _ => by unfold scanFlags cScan; simp only [Agree]; exact hr
  | a :: rest, r, g, hr, hi, hne => by
    obtain ⟨hcmp, hrw, hh, ht⟩ := hr
    by_cases h1 : eqIC (upToNul a) (b "--compact") = true
    · unfold scanFlags at hne ⊢; unfold cScan; simp only [h1, ↓reduceIte] at hne ⊢
      exact scan_agree c hc rest _ _ ⟨by simp, hrw, hh, ht⟩ hi hne
    by_cases h2 : eqIC (upToNul a) (b "--track-memory") = true
    · have h3 := eqIC_excl h2 k2
      have h4 := eqIC_excl h2 k3
      unfold scanFlags at hne ⊢; unfold cScan
      simp only [h1, h2, h3, h4, ↓reduceIte, Bool.false_eq_true] at hne ⊢
      exact scan_agree c hc rest _ _ ⟨hcmp, hrw, hh, ht⟩ hi hne
    by_cases h3 : eqIC (upToNul a) (b "timeout") = true
    · unfold scanFlags at hne ⊢; unfold cScan
      simp only [h1, h2, h3, ↓reduceIte, Bool.false_eq_true] at hne ⊢
      cases rest with
      | nil => rfl
      | cons v rest' =>
        simp only at hne ⊢
        cases hp : string2ll v with
        | none =>
          simp only [Option.getD_none, Option.isNone_none, true_or, ↓reduceIte]
          have : ¬ (c.timeoutMax ≠ 0 ∧ g.timeout > c.timeoutMax) := fun ⟨a1, a2⟩ => by
            have := hi a1; omega
          simp only [this, ↓reduceIte]; rfl
        | some t =>
          simp only [hp] at hne
          simp only [Option.getD_some, Option.isNone_some]
          have hm := hc.max_nn
          by_cases hx : c.timeoutMax ≠ 0 ∧ t > c.timeoutMax
          · have hx' : c.timeoutMax > 0 ∧ t > c.timeoutMax := ⟨by omega, hx.2⟩
            simp [hx.1, hx.2, show c.timeoutMax > 0 by omega, Agree, QErr.toC]
          · have hx' : ¬ (c.timeoutMax > 0 ∧ t > c.timeoutMax) := fun ⟨a1, a2⟩ => hx ⟨by omega, a2⟩
            simp only [hx, hx', ↓reduceIte] at hne ⊢
            by_cases hn : t < 0
            · have : ¬ (t = 0 ∧ g.timeoutRw = true) := fun ⟨e, _⟩ => by omega
              simp only [hn, this, ↓reduceIte]; rfl
            · simp only [hn, ↓reduceIte] at hne ⊢
              have hd := dflt_nn c hc
              have hnn : ¬ ((if t = 0 ∧ g.timeoutRw = true then dflt c else t) < 0) := by
                split <;> omega
              simp only [hnn]
              refine scan_agree c hc rest' _ _ ⟨hcmp, hrw, hh, ?_⟩ ?_ hne
              · simp
              · intro hz
                dsimp only
                split
                · exact dflt_le c hc hz
                · omega
    by_cases h4 : eqIC (upToNul a) (b "version") = true
    · unfold scanFlags at hne ⊢; unfold cScan
      simp only [h1, h2, h3, h4, ↓reduceIte, Bool.false_eq_true] at hne ⊢
      cases rest with
      | nil => rfl
      | cons v rest' =>
        simp only at hne ⊢
        cases hp : string2ll v with
        | none => rfl
        | some n =>
          simp only [hp] at hne
          simp only
          by_cases hn : 0 ≤ n ∧ n ≤ (uintMax : Int)
          · have : ¬ (n < 0 ∨ n > (uintMax : Int)) := by omega
            simp only [hn, this, and_self, ↓reduceIte] at hne ⊢
            refine scan_agree c hc rest' _ _ ⟨hcmp, hrw, ?_, ht⟩ hi hne
            simp; omega
          · have : (n < 0 ∨ n > (uintMax : Int)) := by omega
            simp only [hn, this, ↓reduceIte]; rfl
    by_cases h5 : eqIC (upToNul a) (b "schema_hash") = true
    · unfold scanFlags at hne ⊢; unfold cScan
      simp only [h1, h2, h3, h4, h5, ↓reduceIte, Bool.false_eq_true] at hne ⊢
      cases rest with
      | nil => exact absurd rfl hne
      | cons v rest' =>
        simp only at hne ⊢
        cases hp : parseSchemaHash v with
        | none => simp only [hp] at hne; exact absurd rfl hne
        | some hv =>
          simp only [hp] at hne ⊢
          obtain ⟨n1, n2, n3⟩ := schemaVal_not_kw v hv hp
          unfold cScan
          simp only [n1, n2, n3, Bool.false_eq_true, ↓reduceIte]
          exact scan_agree c hc rest' _ _ ⟨hcmp, hrw, hh, ht⟩ hi hne
    · unfold scanFlags at hne ⊢; unfold cScan
      simp only [h1, h2, h3, h4, h5, Bool.false_eq_true, ↓reduceIte] at hne ⊢
      exact scan_agree c hc rest _ _ ⟨hcmp, hrw, hh, ht⟩ hi hne

/-- The pre-#3757 headline, as a proposition: Rust = C on every argument vector. -/
def RustCAgree : Prop :=
  ∀ (c : TCfg), CfgOk c → ∀ (args : List Bytes),
    Agree c (parseQueryFlags c.timeoutMax args) (cReadFlags c args)

/-- **Rust = C on every argument vector Rust does not refuse with a Rust-only
`SCHEMA_HASH` error** (`557f18868`, #3010; `SCHEMA_HASH h` #3757: a malformed value,
`schemaParse`, or no `--compact`, `schemaVerbose`). -/
theorem rust_c_agree_except_schema (c : TCfg) (hc : CfgOk c) (args : List Bytes)
    (hs : parseQueryFlags c.timeoutMax args ≠ .error .schemaParse)
    (hv : parseQueryFlags c.timeoutMax args ≠ .error .schemaVerbose) :
    Agree c (parseQueryFlags c.timeoutMax args) (cReadFlags c args) := by
  unfold parseQueryFlags at hs hv ⊢; unfold cReadFlags maxArgs at *
  split
  · rfl
  · rename_i hl
    simp only [hl, ↓reduceIte] at hs hv
    have hi : Rel c {} (cInit c) := by
      refine ⟨?_, rfl, ?_, ?_⟩ <;> unfold cInit <;> split <;> rfl
    have hm : c.timeoutMax ≠ 0 → (cInit c).timeout ≤ c.timeoutMax := by
      intro hz
      unfold cInit
      rw [ite_eq_left (Or.inl hz)]
      exact dflt_le c hc hz
    cases hsc : scanFlags c.timeoutMax (args.drop 3) {} with
    | error e =>
      have := scan_agree c hc _ {} (cInit c) hi hm (by rw [hsc]; rintro ⟨⟩; simp [Except.bind, hsc] at hs)
      rw [hsc] at this
      simpa [Except.bind] using this
    | ok r =>
      have := scan_agree c hc _ {} (cInit c) hi hm (by rw [hsc]; simp)
      rw [hsc] at this
      rw [hsc] at hv
      simp only [Except.bind, checkFormat] at hv ⊢
      split
      · rename_i hf; simp [hf] at hv
      · exact this

/-- Every vector Rust accepts, C accepts with the same flags. -/
theorem rust_ok_c_ok (c : TCfg) (hc : CfgOk c) (args : List Bytes) (r : QFlags)
    (h : parseQueryFlags c.timeoutMax args = .ok r) :
    ∃ g, cReadFlags c args = .ok g ∧ Rel c r g := by
  have := rust_c_agree_except_schema c hc args (by rw [h]; simp) (by rw [h]; simp)
  rw [h] at this
  revert this
  cases cReadFlags c args <;> simp [Agree]

/-- The pre-#3757 corollary, as a proposition: both accept exactly the same vectors. -/
def RustCSameAccept : Prop :=
  ∀ (c : TCfg), CfgOk c → ∀ (args : List Bytes),
    ((∃ r, parseQueryFlags c.timeoutMax args = .ok r) ↔ (∃ g, cReadFlags c args = .ok g))

def schemaGarbage : List Bytes := [b "GRAPH.QUERY", b "g", b "RETURN 1", b "SCHEMA_HASH", b "abc"]
def cfg0 : TCfg := { timeoutMax := 0, timeoutDefault := 0, legacy := 0 }

theorem cfg0_ok : CfgOk cfg0 := ⟨by decide, by decide, by decide⟩

theorem schemaGarbage_rust : parseQueryFlags 0 schemaGarbage = .error .schemaParse := by decide
theorem schemaGarbage_c :
    cReadFlags cfg0 schemaGarbage = .ok { compact := false, timeout := 0, timeoutRw := false, hash := -1 } := by
  decide

/-- **COUNTEREXAMPLE (#3757)**: `RustCAgree`, proved as `rust_c_agree` before #3757, no
longer holds. `GRAPH.QUERY g "RETURN 1" SCHEMA_HASH abc`: Rust replies `Failed to parse
schema hash value`; C skips both unknown tokens and runs the query. -/
theorem rust_c_agree_refuted : ¬ RustCAgree := by
  intro h
  have := h cfg0 cfg0_ok schemaGarbage
  simp only [show cfg0.timeoutMax = 0 from rfl, schemaGarbage_rust, schemaGarbage_c, Agree] at this

/-- **COUNTEREXAMPLE (#3757)**: the same vector is accepted by C and refused by Rust. -/
theorem rust_c_same_accept_refuted : ¬ RustCSameAccept := by
  intro h
  have := (h cfg0 cfg0_ok schemaGarbage).2 ⟨_, schemaGarbage_c⟩
  rw [show cfg0.timeoutMax = 0 from rfl, schemaGarbage_rust] at this
  obtain ⟨r, hr⟩ := this
  cases hr

/-- The first #3757 form of the headline (`55d2271e9`), as a proposition: Rust = C on
every vector Rust does not refuse with `schemaParse`. -/
def RustCAgreeExceptSchemaParse : Prop :=
  ∀ (c : TCfg), CfgOk c → ∀ (args : List Bytes),
    parseQueryFlags c.timeoutMax args ≠ .error .schemaParse →
      Agree c (parseQueryFlags c.timeoutMax args) (cReadFlags c args)

def schemaVerboseArgs : List Bytes := [b "GRAPH.QUERY", b "g", b "RETURN 1", b "SCHEMA_HASH", b "5"]

theorem schemaVerbose_rust : parseQueryFlags 0 schemaVerboseArgs = .error .schemaVerbose := by decide
theorem schemaVerbose_c :
    cReadFlags cfg0 schemaVerboseArgs = .ok { compact := false, timeout := 0, timeoutRw := false, hash := -1 } := by
  decide

/-- **#3757 (`a908d8a23`, by design)**: `SCHEMA_HASH requires --compact` is a second
Rust-only refusal. `GRAPH.QUERY g "RETURN 1" SCHEMA_HASH 5`: Rust replies `SCHEMA_HASH
requires --compact`; C skips both tokens and runs the query. So excepting `schemaParse`
alone no longer suffices; `rust_c_agree_except_schema` excepts both. -/
theorem rust_c_agree_except_schemaParse_refuted : ¬ RustCAgreeExceptSchemaParse := by
  intro h
  have := h cfg0 cfg0_ok schemaVerboseArgs (by
    rw [show cfg0.timeoutMax = 0 from rfl, schemaVerbose_rust]; simp)
  simp only [show cfg0.timeoutMax = 0 from rfl, schemaVerbose_rust, schemaVerbose_c, Agree,
    QErr.toC] at this

end RedisLayer
