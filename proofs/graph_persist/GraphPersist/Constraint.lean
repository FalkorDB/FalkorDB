import FalkorCore.Float
import FalkorCore.Laws
/-
# UNIQUE / MANDATORY constraint enforcement, compared with C

Models `Graph::build_composite_key` (graph.rs), the whole-graph validators
`validate_unique_constraint` / `validate_mandatory_constraint` (graph.rs), and
compares them with C FalkorDB's `EnforceUniqueEntity`, which asks the supporting
range index for an exact match.

Here ↔ there:

| Lean | Rust |
| --- | --- |
| `KVal`        | the subset of `Value` a property can hold (`kother`: list, point, temporal, vecf32) |
| `F`           | `f64` (proofs/core `FalkorCore.F`: NaN, `-0.0`, integral, fractional, ±inf) |
| `ToBits`      | `f64::to_bits`, with its one law the key needs: injective on non-NaN values |
| `comp`        | the per-property `match get(prop)` in `build_composite_key`; `none` = `return Vec::new()` |
| `enc`         | the bytes pushed for one component: `'s'`, `len as u64` LE, the bytes; or `'n'`, `(num + 0.0).to_bits()` LE |
| `buildKey`    | `build_composite_key` |
| `validate`    | `validate_unique_constraint`'s loop over the label (type) with its `seen` set |
| `cEqv`        | C's duplicate test per property: numbers (Bool/Int/Float) compared as `f64` with IEEE `==` (`fpcmp`), strings byte for byte, anything else never equal |

Bytes are `Nat`s. `i as f64` is `roundF` (round to nearest even), so integers beyond
2^53 collide with their `f64` neighbours, as in C.

Results:
* `buildKey_eq_nil_iff` — the key is empty iff some component is NULL/absent, NaN or a
  value kind C does not check (or there are no properties).
* `buildKey_eq_iff` — two non-empty keys are equal iff their components are (after `+ 0.0`).
* `rust_C_unique_agree` — two entities collide under Rust's key iff C calls them
  duplicates (#2774 fixed: `1`, `1.0` and `true` collide, `0.0` and `-0.0` collide,
  lists/points/temporals/vectors and NaN never do).
* `validate_spec`, `constraint_creation_agrees` — creating a UNIQUE constraint over
  existing data: Rust brings it up `Operational` iff C does.
* `mandatory_agrees` — MANDATORY agrees with C.
-/

set_option linter.unusedSimpArgs false

namespace GraphPersist
open FalkorCore

/-- The values a constrained property can hold. `knull` is NULL or an absent property;
`kstr` carries the string's UTF-8 bytes; `kother` is a list, point, date, time, datetime,
duration or vecf32. -/
inductive KVal where
  | knull
  | kint (i : Int)
  | kfloat (f : F)
  | kbool (b : Bool)
  | kstr (bytes : List Nat)
  | kother

/-- `f64::to_bits`, abstract: a 64-bit pattern, distinct for distinct non-NaN values
(`-0.0` and `+0.0` included). CAVEAT: `F.num` and `F.frac` range over unbounded `Int`/`Nat`
while `lt` allows only `2 ^ 64` patterns, so no `ToBits` exists for this `F`; every theorem
taking `(tb : ToBits)` is vacuous until the value domain is restricted to representable `f64`
values (the COVERAGE rows that cite them are CONDITIONAL on it). -/
structure ToBits where
  bits : F → Nat
  lt : ∀ x, bits x < 2 ^ 64
  inj : ∀ x y, x.nkey.isSome → y.nkey.isSome → bits x = bits y → x = y

/-- `x + 0.0`: folds `-0.0` into `+0.0`, leaves every other non-NaN value alone. -/
def addZero : F → F
  | .negZero => .num 0
  | x => x

/-- One key component: a string, or a number as `f64`. -/
inductive Comp where
  | str (bytes : List Nat)
  | num (x : F)

/-- `build_composite_key`'s `match get(prop)`: `String` ↦ its bytes; `Bool` ↦
`f64::from(u8::from(b))`; `Int` ↦ `i as f64`; a non-NaN `Float` ↦ itself; anything else
(absent, NULL, NaN, list, point, temporal, vecf32) ↦ `return Vec::new()`. -/
def comp : KVal → Option Comp
  | .kstr s => some (.str s)
  | .kbool b => some (.num (.num (if b then 1 else 0)))
  | .kint i => some (.num (.num (roundF i)))
  | .kfloat f => if f.nkey.isSome then some (.num f) else none
  | .knull => none
  | .kother => none

/-- `n.to_le_bytes()` for a `u64`. -/
def le8 (n : Nat) : List Nat :=
  [n % 256, n / 256 % 256, n / 65536 % 256, n / 16777216 % 256, n / 4294967296 % 256,
   n / 1099511627776 % 256, n / 281474976710656 % 256, n / 72057594037927936 % 256]

/-- The bytes one component appends: `b's'`, `(s.len() as u64).to_le_bytes()`,
`s.as_bytes()`; or `b'n'`, `(num + 0.0).to_bits().to_le_bytes()`. -/
def enc (tb : ToBits) : Comp → List Nat
  | .str s => 115 :: (le8 s.length ++ s)
  | .num x => 110 :: le8 (tb.bits (addZero x))

/-- `Graph::build_composite_key`: the concatenation of every component, or `[]` as soon as
one property is not enforced. -/
def buildKey (tb : ToBits) (vals : List KVal) : List Nat :=
  match vals.mapM comp with
  | some cs => cs.flatMap (enc tb)
  | none => []

/-- A string is shorter than 2^64 bytes (a `usize` length). -/
def Short : KVal → Prop
  | .kstr s => s.length < 2 ^ 64
  | _ => True

/-! ## The key is unambiguous -/

theorem le8_length (n : Nat) : (le8 n).length = 8 := rfl

theorem le8_inj (a b : Nat) (ha : a < 2 ^ 64) (hb : b < 2 ^ 64) (h : le8 a = le8 b) : a = b := by
  simp only [le8, List.cons.injEq, and_true] at h
  omega

theorem append_inj_len {α : Type} {a b r r' : List α} (hl : a.length = b.length)
    (h : a ++ r = b ++ r') : a = b ∧ r = r' := List.append_inj h hl

/-- What `enc` writes for a component: equal bytes mean equal canonical components. -/
def Comp.canon : Comp → Comp
  | .str s => .str s
  | .num x => .num (addZero x)

/-- A component `comp` can produce: a non-NaN number or a string shorter than 2^64. -/
def Comp.Valid : Comp → Prop
  | .str s => s.length < 2 ^ 64
  | .num x => x.nkey.isSome

theorem addZero_valid (x : F) (h : x.nkey.isSome) : (addZero x).nkey.isSome := by
  cases x <;> simp_all [addZero, F.nkey]

theorem enc_prefix (tb : ToBits) (c c' : Comp) (hc : c.Valid) (hc' : c'.Valid)
    (r r' : List Nat) (h : enc tb c ++ r = enc tb c' ++ r') : c.canon = c'.canon ∧ r = r' := by
  cases c with
  | str s =>
    cases c' with
    | str s' =>
      simp only [enc, List.cons_append, List.cons.injEq, true_and, List.append_assoc] at h
      obtain ⟨h1, h2⟩ := append_inj_len (by simp [le8_length]) h
      have hl := le8_inj _ _ hc hc' h1
      obtain ⟨h3, h4⟩ := append_inj_len hl h2
      exact ⟨by simp [Comp.canon, h3], h4⟩
    | num x => simp [enc] at h
  | num x =>
    cases c' with
    | str s => simp [enc] at h
    | num y =>
      simp only [enc, List.cons_append, List.cons.injEq, true_and] at h
      obtain ⟨h1, h2⟩ := append_inj_len (by simp [le8_length]) h
      have hb := le8_inj _ _ (tb.lt _) (tb.lt _) h1
      have := tb.inj _ _ (addZero_valid x hc) (addZero_valid y hc') hb
      exact ⟨by simp [Comp.canon, this], h2⟩

theorem enc_ne_nil (tb : ToBits) (c : Comp) : enc tb c ≠ [] := by
  cases c <;> simp [enc]

theorem flatMap_enc_inj (tb : ToBits) : ∀ (cs cs' : List Comp), (∀ c ∈ cs, c.Valid) →
    (∀ c ∈ cs', c.Valid) → cs.flatMap (enc tb) = cs'.flatMap (enc tb) →
    cs.map Comp.canon = cs'.map Comp.canon
  | [], [], _, _, _ => rfl
  | [], c :: cs', _, _, h => by
    simp only [List.flatMap_nil, List.flatMap_cons] at h
    exact absurd (List.append_eq_nil_iff.1 h.symm).1 (enc_ne_nil tb c)
  | c :: cs, [], _, _, h => by
    simp only [List.flatMap_nil, List.flatMap_cons] at h
    exact absurd (List.append_eq_nil_iff.1 h).1 (enc_ne_nil tb c)
  | c :: cs, c' :: cs', hv, hv', h => by
    simp only [List.flatMap_cons] at h
    obtain ⟨h1, h2⟩ := enc_prefix tb c c' (hv c (by simp)) (hv' c' (by simp)) _ _ h
    simp only [List.map_cons, h1,
      flatMap_enc_inj tb cs cs' (fun x hx => hv x (by simp [hx])) (fun x hx => hv' x (by simp [hx])) h2]

theorem comp_valid (v : KVal) (c : Comp) (hs : Short v) (h : comp v = some c) : c.Valid := by
  cases v with
  | kstr s => simp only [comp, Option.some.injEq] at h; subst h; exact hs
  | kbool b => simp only [comp, Option.some.injEq] at h; subst h; cases b <;> simp [Comp.Valid, F.nkey]
  | kint i => simp only [comp, Option.some.injEq] at h; subst h; simp [Comp.Valid, F.nkey]
  | kfloat f =>
    by_cases hf : f.nkey.isSome
    · simp only [comp, hf, ite_true, Option.some.injEq] at h; subst h; exact hf
    · simp [comp, hf] at h
  | knull => cases h
  | kother => cases h

theorem mapM_comp_valid : ∀ (vs : List KVal) (cs : List Comp), (∀ v ∈ vs, Short v) →
    vs.mapM comp = some cs → ∀ c ∈ cs, c.Valid
  | [], cs, _, h => by simp [List.mapM_nil] at h; subst h; simp
  | v :: vs, cs, hs, h => by
    rw [List.mapM_cons] at h
    cases hv : comp v with
    | none => rw [hv] at h; cases h
    | some c =>
      cases hvs : vs.mapM comp with
      | none => rw [hv, hvs] at h; cases h
      | some cs' =>
        simp [hv, hvs] at h
        subst h
        intro x hx
        simp only [List.mem_cons] at hx
        rcases hx with rfl | hx
        · exact comp_valid v x (hs v (by simp)) hv
        · exact mapM_comp_valid vs cs' (fun y hy => hs y (by simp [hy])) hvs x hx

theorem mapM_some_len : ∀ (vs : List KVal) (cs : List Comp), vs.mapM comp = some cs →
    cs.length = vs.length
  | [], cs, h => by simp [List.mapM_nil] at h; subst h; rfl
  | v :: vs, cs, h => by
    rw [List.mapM_cons] at h
    cases hv : comp v <;> cases hvs : vs.mapM comp <;> simp_all
    obtain ⟨-, rfl⟩ := h
    simp [mapM_some_len vs _ hvs]

theorem mapM_comp_none_iff : ∀ (vs : List KVal), vs.mapM comp = none ↔ ∃ v ∈ vs, comp v = none
  | [] => by simp [List.mapM_nil]
  | v :: vs => by
    rw [List.mapM_cons]
    have ih := mapM_comp_none_iff vs
    cases hv : comp v with
    | none => simp [hv]
    | some c =>
      cases hvs : vs.mapM comp with
      | none => rw [hvs] at ih; simp_all
      | some cs => rw [hvs] at ih; simp_all

theorem flatMap_enc_nil (tb : ToBits) (cs : List Comp) : cs.flatMap (enc tb) = [] ↔ cs = [] := by
  cases cs with
  | nil => simp
  | cons c cs => simp [List.flatMap_cons, enc_ne_nil tb c]

/-- **`buildKey_eq_nil_iff`**: the key is empty iff there are no properties or some
component is not enforced (absent, NULL, NaN, list, point, temporal, vecf32). -/
theorem buildKey_eq_nil_iff (tb : ToBits) (vals : List KVal) :
    buildKey tb vals = [] ↔ vals = [] ∨ ∃ v ∈ vals, comp v = none := by
  unfold buildKey
  cases h : vals.mapM comp with
  | none => simp [(mapM_comp_none_iff vals).1 h]
  | some cs =>
    have hn : ¬ ∃ v ∈ vals, comp v = none := fun e => by
      rw [← mapM_comp_none_iff] at e; rw [e] at h; cases h
    simp only [flatMap_enc_nil, hn, or_false]
    have := mapM_some_len vals cs h
    constructor
    · rintro rfl; cases vals <;> simp_all
    · rintro rfl; cases cs <;> simp_all

theorem addZero_idem (x : F) : addZero (addZero x) = addZero x := by
  cases x <;> rfl

theorem enc_canon (tb : ToBits) (c : Comp) : enc tb c.canon = enc tb c := by
  cases c <;> simp [Comp.canon, enc, addZero_idem]

/-- **`buildKey_eq_iff`**: two non-empty keys are equal iff their canonical component lists
are (the key is the list `vals.mapM comp`, `-0.0` read as `0.0`, written unambiguously). -/
theorem buildKey_eq_iff (tb : ToBits) (a b : List KVal) (cs cs' : List Comp)
    (ha : ∀ v ∈ a, Short v) (hb : ∀ v ∈ b, Short v)
    (hca : a.mapM comp = some cs) (hcb : b.mapM comp = some cs') :
    buildKey tb a = buildKey tb b ↔ cs.map Comp.canon = cs'.map Comp.canon := by
  unfold buildKey
  rw [hca, hcb]; dsimp only
  refine ⟨flatMap_enc_inj tb cs cs' (mapM_comp_valid a cs ha hca) (mapM_comp_valid b cs' hb hcb),
    fun h => ?_⟩
  have e : ∀ l : List Comp, l.flatMap (enc tb) = (l.map Comp.canon).flatMap (enc tb) := by
    intro l; induction l with
    | nil => rfl
    | cons c l ih => simp only [List.flatMap_cons, List.map_cons, enc_canon, ih]
  rw [e cs, e cs', h]

/-! ## C's duplicate test -/

/-- The number C's range index holds for a value (`Bool` as 0/1, `Int` as `f64`). -/
def cNum : KVal → Option F
  | .kint i => some (.num (roundF i))
  | .kfloat f => some f
  | .kbool b => some (.num (if b then 1 else 0))
  | _ => none

/-- C's per-property duplicate test: strings byte for byte, numbers by IEEE `==` (NaN
equals nothing, `-0.0 = 0.0`), every other kind never (C does not check it). -/
def cEqv (a b : KVal) : Bool :=
  match a, b with
  | .kstr x, .kstr y => x == y
  | _, _ =>
    match cNum a, cNum b with
    | some x, some y => fpcmp x y == some .eq
    | _, _ => false

/-- C calls two entities duplicates iff every constrained property is a duplicate. -/
def uniqueCollideC (a b : List KVal) : Bool :=
  a.length == b.length && (a.zip b).all fun p => cEqv p.1 p.2

/-- Rust: both keys non-empty and equal (`seen.insert(key)` fails, `keys.contains`). -/
def uniqueCollideRust (tb : ToBits) (a b : List KVal) : Bool :=
  !(buildKey tb a).isEmpty && buildKey tb a == buildKey tb b

theorem nkey_eq_iff (x y : F) (hx : x.nkey.isSome) (hy : y.nkey.isSome) :
    x.nkey = y.nkey ↔ addZero x = addZero y := by
  cases x <;> cases y <;> simp [F.nkey, addZero] at hx hy ⊢ <;> omega

/-- IEEE `==` on two non-NaN values is equality after folding `-0.0` into `+0.0`. -/
theorem fpcmp_eq_iff (x y : F) (hx : x.nkey.isSome) (hy : y.nkey.isSome) :
    fpcmp x y = some .eq ↔ addZero x = addZero y := by
  rw [← nkey_eq_iff x y hx hy]
  unfold fpcmp
  cases hkx : x.nkey with
  | none => simp [hkx] at hx
  | some kx =>
    cases hky : y.nkey with
    | none => simp [hky] at hy
    | some ky => simp [Laws.cmp3_eq_iff]

theorem fpcmp_nan (x y : F) (h : x.nkey = none) : fpcmp x y = none ∧ fpcmp y x = none := by
  unfold fpcmp
  rw [h]
  cases y.nkey <;> simp

theorem comp_eq (a : KVal) (c : Comp) : comp a = some c ↔
    (∃ s, a = .kstr s ∧ c = .str s) ∨ (∃ x, cNum a = some x ∧ x.nkey.isSome ∧ c = .num x) := by
  constructor
  · intro h
    cases a with
    | kfloat f =>
      by_cases hf : f.nkey.isSome
      · simp only [comp, hf, ite_true, Option.some.injEq] at h
        exact .inr ⟨f, rfl, hf, h.symm⟩
      · simp [comp, hf] at h
    | kstr s => simp only [comp, Option.some.injEq] at h; exact .inl ⟨s, rfl, h.symm⟩
    | kbool b =>
      simp only [comp, Option.some.injEq] at h
      exact .inr ⟨_, rfl, by cases b <;> rfl, h.symm⟩
    | kint i =>
      simp only [comp, Option.some.injEq] at h
      exact .inr ⟨_, rfl, rfl, h.symm⟩
    | knull => cases h
    | kother => cases h
  · rintro (⟨s, rfl, rfl⟩ | ⟨x, hx, hv, rfl⟩)
    · rfl
    · cases a with
      | kfloat f =>
        simp only [cNum, Option.some.injEq] at hx; subst hx
        simp [comp, hv]
      | kbool b => simp only [cNum, Option.some.injEq] at hx; subst hx; rfl
      | kint i => simp only [cNum, Option.some.injEq] at hx; subst hx; rfl
      | kstr s => cases hx
      | knull => cases hx
      | kother => cases hx

theorem cEqv_num (a b : KVal) (x y : F) (ha : cNum a = some x) (hb : cNum b = some y) :
    cEqv a b = (fpcmp x y == some .eq) := by
  cases a <;> cases b <;> simp_all [cNum, cEqv]

theorem comp_cEqv (a b : KVal) (ca cb : Comp) (ha : comp a = some ca) (hb : comp b = some cb) :
    cEqv a b = true ↔ ca.canon = cb.canon := by
  rcases (comp_eq a ca).1 ha with ⟨s, rfl, rfl⟩ | ⟨x, hx, hxv, rfl⟩ <;>
    rcases (comp_eq b cb).1 hb with ⟨t, rfl, rfl⟩ | ⟨y, hy, hyv, rfl⟩
  · simp [cEqv, Comp.canon]
  · cases b <;> simp_all [cNum, cEqv, Comp.canon]
  · cases a <;> simp_all [cNum, cEqv, Comp.canon]
  · rw [cEqv_num a b x y hx hy]
    simp [Comp.canon, fpcmp_eq_iff x y hxv hyv]

theorem comp_none_cEqv (a b : KVal) (h : comp a = none ∨ comp b = none) : cEqv a b = false := by
  have side : ∀ a b : KVal, comp a = none →
      cEqv a b = false ∧ cEqv b a = false := by
    intro a b ha
    cases a with
    | kfloat f =>
      by_cases hf : f.nkey.isSome
      · simp [comp, hf] at ha
      · have hn : f.nkey = none := by cases h' : f.nkey <;> simp_all
        cases b <;> simp [cEqv, cNum, (fpcmp_nan f _ hn).1, (fpcmp_nan f _ hn).2]
    | kstr s => simp [comp] at ha
    | kbool b => simp [comp] at ha
    | kint i => simp [comp] at ha
    | knull => cases b <;> simp [cEqv, cNum]
    | kother => cases b <;> simp [cEqv, cNum]
  rcases h with h | h
  · exact (side a b h).1
  · exact (side b a h).2

/-- C's per-component verdict, read through `comp`. -/
theorem collideC_of_some : ∀ (a b : List KVal) (ca cb : List Comp), a.mapM comp = some ca →
    b.mapM comp = some cb →
    (uniqueCollideC a b = true ↔ ca.map Comp.canon = cb.map Comp.canon)
  | [], [], ca, cb, h1, h2 => by simp_all [List.mapM_nil, uniqueCollideC]
  | [], _ :: _, ca, cb, h1, h2 => by
    simp [List.mapM_nil] at h1; subst h1
    have := mapM_some_len _ _ h2
    cases cb <;> simp_all [uniqueCollideC]
  | _ :: _, [], ca, cb, h1, h2 => by
    simp [List.mapM_nil] at h2; subst h2
    have := mapM_some_len _ _ h1
    cases ca <;> simp_all [uniqueCollideC]
  | x :: xs, y :: ys, ca, cb, h1, h2 => by
    rw [List.mapM_cons] at h1 h2
    cases hx : comp x <;> cases hxs : xs.mapM comp <;> simp_all
    cases hy : comp y <;> cases hys : ys.mapM comp <;> simp_all
    obtain ⟨-, rfl⟩ := h1
    obtain ⟨-, rfl⟩ := h2
    rename_i cx cxs cy cys
    have ih := collideC_of_some xs ys cxs cys hxs hys
    have hc := comp_cEqv x y cx cy hx hy
    simp only [uniqueCollideC, List.length_cons, List.zip_cons_cons, List.all_cons,
      List.map_cons, List.cons.injEq, beq_iff_eq, Nat.add_right_cancel_iff, Bool.and_eq_true] at ih hc ⊢
    rw [← hc, ← ih]
    constructor
    · rintro ⟨h1, h2, h3⟩; exact ⟨h2, h1, h3⟩
    · rintro ⟨h2, h1, h3⟩; exact ⟨h1, h2, h3⟩

theorem collideC_of_none : ∀ (a b : List KVal), (a.mapM comp = none ∨ b.mapM comp = none) →
    uniqueCollideC a b = false
  | [], [], h => by simp [List.mapM_nil] at h
  | [], _ :: _, _ => by simp [uniqueCollideC]
  | _ :: _, [], _ => by simp [uniqueCollideC]
  | x :: xs, y :: ys, h => by
    rw [List.mapM_cons, List.mapM_cons] at h
    simp only [uniqueCollideC, List.length_cons, List.zip_cons_cons, List.all_cons,
      Bool.and_eq_false_iff, beq_eq_false_iff_ne, ne_eq, Nat.add_right_cancel_iff]
    cases hx : comp x <;> cases hy : comp y
    · exact .inr (.inl (comp_none_cEqv x y (.inl hx)))
    · exact .inr (.inl (comp_none_cEqv x y (.inl hx)))
    · exact .inr (.inl (comp_none_cEqv x y (.inr hy)))
    · rw [hx, hy] at h
      have h' : xs.mapM comp = none ∨ ys.mapM comp = none := by
        cases hxs : xs.mapM comp <;> cases hys : ys.mapM comp <;> simp_all
      have := collideC_of_none xs ys h'
      simp only [uniqueCollideC, Bool.and_eq_false_iff, beq_eq_false_iff_ne, ne_eq] at this
      rcases this with h | h
      · exact .inl h
      · exact .inr (.inr h)

/-- **`rust_C_unique_agree`** (#2774 fixed): Rust's key makes two entities collide
exactly when C calls them duplicates, for every pair of property-value lists (a
constraint has at least one property; strings shorter than 2^64 bytes). -/
theorem rust_C_unique_agree (tb : ToBits) (a b : List KVal) (hne : a ≠ []) (ha : ∀ v ∈ a, Short v)
    (hb : ∀ v ∈ b, Short v) : uniqueCollideRust tb a b = uniqueCollideC a b := by
  unfold uniqueCollideRust
  cases h1 : a.mapM comp with
  | none =>
    have : buildKey tb a = [] := by simp [buildKey, h1]
    simp [this, collideC_of_none a b (.inl h1)]
  | some ca =>
    cases h2 : b.mapM comp with
    | none =>
      have hb' : buildKey tb b = [] := by simp [buildKey, h2]
      rw [collideC_of_none a b (.inr h2), hb']
      cases buildKey tb a <;> simp
    | some cb =>
      have e := collideC_of_some a b ca cb h1 h2
      have va := mapM_comp_valid a ca ha h1
      have vb := mapM_comp_valid b cb hb h2
      simp only [buildKey, h1, h2]
      cases hc : uniqueCollideC a b
      · simp only [Bool.and_eq_false_iff, Bool.not_eq_false', List.isEmpty_iff, beq_eq_false_iff_ne, ne_eq]
        by_cases hk : ca.flatMap (enc tb) = cb.flatMap (enc tb)
        · have := flatMap_enc_inj tb ca cb va vb hk
          rw [← e] at this; rw [this] at hc; cases hc
        · exact .inr hk
      · have hm := e.1 hc
        have hk : ca.flatMap (enc tb) = cb.flatMap (enc tb) := by
          clear e hc h1 h2
          induction ca generalizing cb with
          | nil => cases cb <;> simp_all
          | cons c cs ih =>
            cases cb with
            | nil => simp at hm
            | cons d ds =>
              simp only [List.map_cons, List.cons.injEq] at hm
              simp only [List.flatMap_cons]
              have hcd : enc tb c = enc tb d := by
                cases c <;> cases d <;> simp_all [Comp.canon, enc]
              rw [hcd, ih ds (fun x hx => va x (by simp [hx])) (fun x hx => vb x (by simp [hx])) hm.2]
        have hca : ca ≠ [] := by
          intro h; subst h
          have := mapM_some_len a [] h1
          cases a <;> simp_all
        obtain ⟨c, cs, rfl⟩ := List.exists_cons_of_ne_nil hca
        rw [← hk]
        simp [List.flatMap_cons, enc_ne_nil tb c]

/-- Concrete cases (the old counterexamples, now agreeing with C). -/
theorem int_float_bool_collide (tb : ToBits) :
    uniqueCollideRust tb [.kint 1] [.kfloat (.num 1)] = true ∧
    uniqueCollideRust tb [.kint 1] [.kbool true] = true ∧
    uniqueCollideRust tb [.kfloat (.num 0)] [.kfloat .negZero] = true ∧
    uniqueCollideRust tb [.kother] [.kother] = false := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    rw [rust_C_unique_agree tb _ _ (by simp) (by simp [Short]) (by simp [Short])] <;> decide

/-! ## Constraint *creation over existing data* (`validate_unique_constraint`) -/

/-- Do any two distinct participating entities collide (engine given by `coll`)? -/
def hasDup {α : Type} (coll : α → α → Bool) : List α → Bool
  | []      => false
  | x :: xs => (xs.any (fun y => coll x y)) || hasDup coll xs

/-- `validate_unique_constraint`'s loop: skip an entity whose key is empty, fail on a key
already in `seen`, else insert it. `true` = the constraint comes up `Operational`. -/
def validate (tb : ToBits) : List (List KVal) → List (List Nat) → Bool
  | [], _ => true
  | e :: es, seen =>
    let k := buildKey tb e
    if k.isEmpty then validate tb es seen
    else if k ∈ seen then false
    else validate tb es (k :: seen)

def rustConstraintOK (tb : ToBits) (vals : List (List KVal)) : Bool := validate tb vals []
def cConstraintOK (vals : List (List KVal)) : Bool := !hasDup uniqueCollideC vals

/-- **`validate_spec`**: the loop accepts iff no entity's non-empty key is in `seen` and
no two entities collide under the key. -/
theorem validate_spec (tb : ToBits) : ∀ (es : List (List KVal)) (seen : List (List Nat)),
    validate tb es seen = true ↔
      (∀ e ∈ es, buildKey tb e ≠ [] → buildKey tb e ∉ seen) ∧
      hasDup (uniqueCollideRust tb) es = false
  | [], seen => by simp [validate, hasDup]
  | e :: es, seen => by
    have ih := validate_spec tb es
    simp only [validate, hasDup, Bool.or_eq_false_iff, List.forall_mem_cons]
    by_cases he : (buildKey tb e).isEmpty = true
    · have he' : buildKey tb e = [] := List.isEmpty_iff.1 he
      have hany : es.any (fun y => uniqueCollideRust tb e y) = false := by
        rw [List.any_eq_false]; intro x _; simp [uniqueCollideRust, he]
      rw [ite_eq_left he, ih seen, hany]
      simp [he']
    · have hne : buildKey tb e ≠ [] := fun h => he (List.isEmpty_iff.2 h)
      rw [ite_eq_right he]
      by_cases hs : buildKey tb e ∈ seen
      · rw [ite_eq_left hs]; simp [hs, hne]
      · rw [ite_eq_right hs, ih]
        have hr : ∀ x, uniqueCollideRust tb e x = true ↔ buildKey tb e = buildKey tb x := by
          intro x; simp [uniqueCollideRust, he]
        constructor
        · rintro ⟨h1, h2⟩
          refine ⟨⟨fun _ => hs, fun x hx hxe hm => h1 x hx hxe (List.mem_cons_of_mem _ hm)⟩, ?_, h2⟩
          rw [List.any_eq_false]; intro x hx hc
          have hk := (hr x).1 hc
          exact h1 x hx (hk ▸ hne) (by simp [hk])
        · rintro ⟨⟨-, h1⟩, h2, h3⟩
          refine ⟨fun x hx hxe hm => ?_, h3⟩
          simp only [List.mem_cons] at hm
          rcases hm with hm | hm
          · exact (List.any_eq_false.1 h2) x hx ((hr x).2 hm.symm)
          · exact h1 x hx hxe hm

theorem any_congr' {α : Type} (f g : α → Bool) : ∀ (l : List α), (∀ y ∈ l, f y = g y) →
    l.any f = l.any g
  | [], _ => rfl
  | y :: ys, h => by
    simp only [List.any_cons, h y (by simp), any_congr' f g ys (fun z hz => h z (by simp [hz]))]

theorem hasDup_congr {α : Type} (c c' : α → α → Bool) : ∀ (l : List α),
    (∀ x ∈ l, ∀ y ∈ l, c x y = c' x y) → hasDup c l = hasDup c' l
  | [], _ => rfl
  | x :: xs, h => by
    simp only [hasDup]
    rw [hasDup_congr c c' xs (fun a ha b hb => h a (by simp [ha]) b (by simp [hb]))]
    rw [any_congr' _ _ xs (fun y hy => h x (by simp) y (by simp [hy]))]

/-- **`constraint_creation_agrees`** (fixed with #2774): over any stored data, Rust brings a
fresh UNIQUE constraint up `Operational` iff C does (C: not `FAILED`). -/
theorem constraint_creation_agrees (tb : ToBits) (vals : List (List KVal))
    (hp : ∀ e ∈ vals, e ≠ []) (hs : ∀ e ∈ vals, ∀ v ∈ e, Short v) :
    rustConstraintOK tb vals = cConstraintOK vals := by
  unfold rustConstraintOK cConstraintOK
  rw [← hasDup_congr (uniqueCollideRust tb) _ vals
    (fun x hx y hy => rust_C_unique_agree tb x y (hp x hx) (hs x hx) (hs y hy))]
  have hv := validate_spec tb vals []
  cases h : hasDup (uniqueCollideRust tb) vals
  · rw [hv.2 ⟨by simp, h⟩]; rfl
  · cases h' : validate tb vals []
    · rfl
    · have := (hv.1 h').2; rw [h] at this; cases this

/-! ## MANDATORY enforcement agrees with C (a positive result)

`check_node_constraint`'s Mandatory arm requires every constrained property be
present and non-NULL on each affected entity of the label — this matched C on
every probe (create-missing, `SET x = null`, `REMOVE x`, `SET n = {}`,
create-then-delete, `SET n:L` adding the label, `MERGE ... ON CREATE`). Modelled
as: a value satisfies the mandatory check iff it is not NULL. -/
def isNull : KVal → Bool | .knull => true | _ => false
def mandatorySatisfied (v : KVal) : Bool := !isNull v
def mandatorySatisfiedC : KVal → Bool | .knull => false | _ => true

theorem mandatory_agrees (v : KVal) : mandatorySatisfied v = mandatorySatisfiedC v := by
  cases v <;> simp [mandatorySatisfied, mandatorySatisfiedC, isNull]

end GraphPersist
