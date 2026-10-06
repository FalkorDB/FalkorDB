/-!
# Integer parsers and `compute_effective_timeout`

Shared primitives for the query-command argument models (`QueryArgs.lean`) and
`GRAPH.CONFIG` (`Config.lean`):

* `string2ll`     — Redis `util.c:string2ll`, which `RedisModule_StringToLongLong`
  (`RedisString::parse_integer`) calls: what both engines use for `TIMEOUT`/`version`
  since #3010 (`557f18868`).
* `rustParseI64`  — `core::str::parse::<i64>` (`from_str_radix`, radix 10), used by
  `GRAPH.CONFIG SET` (`Config.lean`).
* `rustTimeout`   — `compute_effective_timeout` (`src/graph_core.rs:896-931`).

Bytes are `Nat`s; `eq_ignore_ascii_case` is byte-wise (`eqIC`).
-/

namespace RedisLayer

/-- ASCII bytes of a string literal (all literals below are ASCII). -/
def b (s : String) : List Nat := s.toList.map Char.toNat

def lowerA (c : Nat) : Nat := if 65 ≤ c ∧ c ≤ 90 then c + 32 else c

/-- `str::eq_ignore_ascii_case` — byte-wise ASCII case folding. -/
def eqIC (x y : List Nat) : Bool := x.map lowerA == y.map lowerA

def isDig (c : Nat) : Bool := 48 ≤ c && c ≤ 57

/-- Decimal value of a digit string. -/
def digVal (ds : List Nat) : Nat := ds.foldl (fun a d => a * 10 + (d - 48)) 0

/-- A non-empty all-digit string's value. -/
def digitsVal (ds : List Nat) : Option Nat :=
  if ds ≠ [] ∧ ds.all isDig then some (digVal ds) else none

def i64Max : Nat := 2 ^ 63 - 1
def u64Max : Nat := 2 ^ 64 - 1
def uintMax : Nat := 2 ^ 32 - 1

/-- `"...".parse::<i64>()`: optional `+`/`-`, then ≥ 1 digits, value in range. -/
def rustParseI64 : List Nat → Option Int
  | [] => none
  | d :: ds =>
    if d = 43 then ((digitsVal ds).filter (· ≤ i64Max)).map Int.ofNat
    else if d = 45 then ((digitsVal ds).filter (· ≤ i64Max + 1)).map (fun n => -(Int.ofNat n))
    else ((digitsVal (d :: ds)).filter (· ≤ i64Max)).map Int.ofNat

/-- Redis `string2ll`: `0`, or an optional `-` followed by `[1-9][0-9]*`, nothing else
(no `+`, no leading zeros, no spaces), at most 20 bytes, within `i64`. -/
def string2ll (s : List Nat) : Option Int :=
  if s.length = 0 ∨ s.length ≥ 21 then none
  else if s = [48] then some 0
  else match s with
    | [] => none
    | d :: ds =>
      if d = 45 then
        match ds with
        | [] => none
        | e :: es =>
          if 49 ≤ e ∧ e ≤ 57 then
            ((digitsVal (e :: es)).filter (· ≤ i64Max + 1)).map (fun n => -(Int.ofNat n))
          else none
      else if 49 ≤ d ∧ d ≤ 57 then ((digitsVal (d :: ds)).filter (· ≤ i64Max)).map Int.ofNat
      else none

/-- `Except` equality is decidable when both sides are (used by `decide` below). -/
instance instDecEqExcept {ε α : Type} [DecidableEq ε] [DecidableEq α] :
    DecidableEq (Except ε α) := fun x y =>
  match x, y with
  | .ok a, .ok c => if h : a = c then isTrue (h ▸ rfl) else isFalse (by intro e; cases e; exact h rfl)
  | .error a, .error c =>
    if h : a = c then isTrue (h ▸ rfl) else isFalse (by intro e; cases e; exact h rfl)
  | .ok _, .error _ => isFalse (by intro e; cases e)
  | .error _, .ok _ => isFalse (by intro e; cases e)

/-! ## `compute_effective_timeout` (`graph_core.rs:896-931`) -/

structure TCfg where
  timeoutMax : Int
  timeoutDefault : Int
  legacy : Int
  deriving DecidableEq, Repr

inductive TErr where
  | exceedsMax
  deriving DecidableEq, Repr

/-- The global fallback (`graph_core.rs:917-930`). -/
def fallback (c : TCfg) (isWrite : Bool) : Except TErr (Option Nat) :=
  if c.timeoutDefault > 0 then .ok (some c.timeoutDefault.toNat)
  else if c.timeoutMax > 0 then .ok (some c.timeoutMax.toNat)
  else if c.legacy > 0 ∧ isWrite = false then .ok (some c.legacy.toNat)
  else .ok none

def rustTimeout (c : TCfg) (pq : Option Int) (isWrite : Bool) : Except TErr (Option Nat) :=
  match pq with
  | some p =>
    if c.timeoutMax > 0 ∧ p > c.timeoutMax then .error .exceedsMax
    else if isWrite = false ∧ p > 0 then .ok (some p.toNat)
    else fallback c isWrite
  | none => fallback c isWrite

/-! ## Decimal rendering, and the integer parsers on it -/

/-- Canonical decimal rendering (what every client sends). -/
def dec (n : Nat) : List Nat :=
  if n < 10 then [48 + n] else dec (n / 10) ++ [48 + n % 10]
termination_by n
decreasing_by omega

theorem dec_ne_nil (n : Nat) : dec n ≠ [] := by
  unfold dec; split <;> simp

theorem dec_all_dig (n : Nat) : (dec n).all isDig = true := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    unfold dec
    split
    · simp [isDig]; omega
    · simp only [List.all_append, Bool.and_eq_true]
      refine ⟨ih _ (by omega), ?_⟩
      simp [isDig]; omega

theorem digVal_append (xs ys : List Nat) :
    digVal (xs ++ ys) = ys.foldl (fun a d => a * 10 + (d - 48)) (digVal xs) := by
  simp [digVal, List.foldl_append]

theorem digVal_dec (n : Nat) : digVal (dec n) = n := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    unfold dec
    split
    · simp [digVal]
    · rw [digVal_append, ih _ (by omega)]
      simp; omega

theorem digitsVal_dec (n : Nat) : digitsVal (dec n) = some n := by
  simp [digitsVal, dec_ne_nil, dec_all_dig, digVal_dec]

theorem dec_head (n : Nat) : ∃ d ds, dec n = d :: ds ∧ 48 ≤ d ∧ d ≤ 57 ∧ (n ≠ 0 → 49 ≤ d) := by
  induction n using Nat.strongRecOn with
  | _ n ih =>
    unfold dec
    split
    · exact ⟨48 + n, [], rfl, by omega, by omega, by omega⟩
    · obtain ⟨d, ds, h, h1, h2, h3⟩ := ih (n / 10) (by omega)
      exact ⟨d, ds ++ [48 + n % 10], by rw [h]; rfl, h1, h2, fun _ => h3 (by omega)⟩

theorem dec_length_le (n k : Nat) (h : n < 10 ^ k) (hk : 1 ≤ k) : (dec n).length ≤ k := by
  induction k generalizing n with
  | zero => omega
  | succ k ih =>
    unfold dec
    split
    · simp
    · have : n / 10 < 10 ^ k := by
        rw [Nat.pow_succ] at h; omega
      have hk' : 1 ≤ k := by
        rcases k with _ | k
        · simp at this; omega
        · omega
      have := ih (n / 10) this hk'
      simp; omega

theorem dec_length_i64 (n : Nat) (h : n ≤ i64Max + 1) : (dec n).length ≤ 19 :=
  dec_length_le n 19 (by unfold i64Max at h; omega) (by omega)
theorem rustParseI64_dec (n : Nat) (h : n ≤ i64Max) :
    rustParseI64 (dec n) = some (Int.ofNat n) := by
  obtain ⟨d, ds, hd, h1, h2, _⟩ := dec_head n
  have hv := digitsVal_dec n
  rw [hd] at hv ⊢
  simp [rustParseI64, show d ≠ 43 by omega, show d ≠ 45 by omega, hv]
  exact ⟨n, ⟨rfl, h⟩, rfl⟩

/-- `string2ll` reads every canonical rendering of an `i64`-range value back. -/
theorem string2ll_dec (n : Nat) (h : n ≤ i64Max) :
    string2ll (dec n) = some (Int.ofNat n) := by
  have hl := dec_length_i64 n (by omega)
  obtain ⟨d, ds, hd, h1, h2, h3⟩ := dec_head n
  have hv := digitsVal_dec n
  by_cases hz : n = 0
  · subst hz
    rw [show dec 0 = [48] by rw [dec]; rfl]; rfl
  · have hd1 := h3 hz
    rw [hd] at hv hl ⊢
    simp only [List.length_cons] at hl
    unfold string2ll
    rw [if_neg (by simp; omega), if_neg (by intro heq; simp at heq; omega)]
    simp [show d ≠ 45 by omega, show 49 ≤ d by omega, h2, hv]
    exact ⟨n, ⟨rfl, h⟩, rfl⟩

/-! ## `compute_effective_timeout` -/

/-- Writes never see the per-query timeout: the answer does not depend on it. -/
theorem rustTimeout_write_ignores_pq (c : TCfg) (p : Int)
    (h : ¬ (c.timeoutMax > 0 ∧ p > c.timeoutMax)) :
    rustTimeout c (some p) true = rustTimeout c none true := by
  simp [rustTimeout, h]

/-- A per-query timeout above `TIMEOUT_MAX` is always rejected, for reads and writes. -/
theorem rustTimeout_exceeds (c : TCfg) (p : Int) (w : Bool)
    (h : c.timeoutMax > 0 ∧ p > c.timeoutMax) :
    rustTimeout c (some p) w = .error .exceedsMax := by
  simp [rustTimeout, h]

theorem fallback_spec (c : TCfg) (w : Bool) (t : Nat) (h : fallback c w = .ok (some t)) :
    0 < t ∧ (c.timeoutDefault > 0 → (t : Int) = c.timeoutDefault) ∧
      (c.timeoutDefault ≤ 0 → c.timeoutMax > 0 → (t : Int) = c.timeoutMax) := by
  unfold fallback at h
  by_cases h1 : c.timeoutDefault > 0
  · rw [if_pos h1] at h; simp at h; omega
  · rw [if_neg h1] at h
    by_cases h2 : c.timeoutMax > 0
    · rw [if_pos h2] at h; simp at h; omega
    · rw [if_neg h2] at h
      by_cases h3 : c.legacy > 0 ∧ w = false
      · rw [if_pos h3] at h; simp at h; omega
      · rw [if_neg h3] at h; simp at h

/-- Under the configuration invariant `GRAPH.CONFIG SET` maintains
(`TIMEOUT_DEFAULT ≤ TIMEOUT_MAX` when both are set, see `Config.lean`), no query ever
runs longer than `TIMEOUT_MAX`. -/
theorem rustTimeout_le_max (c : TCfg) (pq : Option Int) (w : Bool) (t : Nat)
    (hmax : c.timeoutMax > 0) (hinv : c.timeoutDefault ≤ c.timeoutMax)
    (h : rustTimeout c pq w = .ok (some t)) : (t : Int) ≤ c.timeoutMax := by
  cases pq with
  | none =>
    have := fallback_spec c w t h
    by_cases hd : c.timeoutDefault > 0 <;> omega
  | some p =>
    simp only [rustTimeout] at h
    by_cases h1 : c.timeoutMax > 0 ∧ p > c.timeoutMax
    · rw [if_pos h1] at h; simp at h
    · rw [if_neg h1] at h
      by_cases h2 : w = false ∧ p > 0
      · rw [if_pos h2] at h; simp at h; omega
      · rw [if_neg h2] at h
        have := fallback_spec c w t h
        by_cases hd : c.timeoutDefault > 0 <;> omega

/-- A produced timeout is always positive (0 means "no timeout" and is never produced). -/
theorem rustTimeout_pos (c : TCfg) (pq : Option Int) (w : Bool) (t : Nat)
    (h : rustTimeout c pq w = .ok (some t)) : 0 < t := by
  cases pq with
  | none => exact (fallback_spec c w t h).1
  | some p =>
    simp only [rustTimeout] at h
    by_cases h1 : c.timeoutMax > 0 ∧ p > c.timeoutMax
    · rw [if_pos h1] at h; simp at h
    · rw [if_neg h1] at h
      by_cases h2 : w = false ∧ p > 0
      · rw [if_pos h2] at h; simp at h; omega
      · rw [if_neg h2] at h; exact (fallback_spec c w t h).1

/-- Negative per-query timeout on a read: `compute_effective_timeout` alone falls back to
the global value. Since #3010 (`557f18868`) no command can hand it a negative value:
`parse_query_flags` rejects it first (`QueryArgs.parse_timeout_nonneg`). -/
theorem neg_timeout_falls_back :
    rustTimeout { timeoutMax := 0, timeoutDefault := 0, legacy := 7 } (some (-5)) false
      = .ok (some 7) := by decide

end RedisLayer
