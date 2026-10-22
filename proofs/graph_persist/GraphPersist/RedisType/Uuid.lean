/-! # `uuid_v4` / `uuid_fields` (redis_type.rs:790-830) and virtual-key names (:463-468)

**Since #3286 (`bf11a2774`)** `uuid_v4` keeps the clock and the counter in separate
fields: `uuid_fields(t, seq)` prints `t` in the first three fields and
`b = seq * 6364136223846793005 + 1442695040888963407` (wrapping) in the last two.

* `counter_field_bijective` — the last field, `b mod 2^48`, determines `seq mod 2^48`
  (the multiplier is odd): the bijection the Rust `const _` block checks at compile time
  on sample points, proven here for every `seq`.
* `uuid_counter_distinct` — **two calls whose counters differ mod 2^48 never share a
  name, whatever the clock reads**; `uuid_same_clock_distinct`,
  `uuid_regression_pair_distinct` (`(t, seq) = (4, 2)` vs `(5, 3)`, the old collision).
* `vkeyName_inj`, `vkeyName_ne` — virtual-key names of one graph differ iff their uuids do,
  and none equals the graph's own name; `vkeyName_uuid_counter` — two virtual-key names
  (of any graphs) that are equal came from the same counter mod 2^48, and
  `names_vname_inj` discharges `VKeys.Names.vname_inj` for any naming that hands each
  `(graph, slot)` its own counter value; `sameFields_iff`, `const_check_holds` — `same_fields`
  and the `const _` block's assertions, for every clock reading.

History (before #3286): `a = (nanos as u64) ^ seq` fed all five fields.
`pre3286_uuid_injective` (the name determines `a`), `pre3286_uuid_collision_iff` (two calls
collide iff `t₁ ⊕ s₁ = t₂ ⊕ s₂`), `pre3286_uuid_collision_example` (counters `2, 3` at
clocks `4, 5` ns collide), `pre3286_uuid_same_clock_distinct`.
-/
namespace GraphPersist.RedisType

def U64 : Nat := 2 ^ 64
def MUL : Nat := 6364136223846793005
def INC : Nat := 1442695040888963407

/-- `wrapping_mul(MUL).wrapping_add(INC)`. -/
def mixB (a : Nat) : Nat := (a * MUL + INC) % U64

/-- `{:0w$x}`: `w` hex digits, most significant first. -/
def hex : Nat → Nat → List Nat
  | 0, _ => []
  | w + 1, n => hex w (n / 16) ++ [n % 16]

theorem hex_length (w n : Nat) : (hex w n).length = w := by
  induction w generalizing n <;> simp [hex, *]

theorem hex_inj : ∀ (w n m : Nat), hex w n = hex w m → n % 16 ^ w = m % 16 ^ w
  | 0, n, m, _ => by simp [Nat.mod_one]
  | w + 1, n, m, h => by
    simp only [hex] at h
    have hl : (hex w (n / 16)).length = (hex w (m / 16)).length := by simp [hex_length]
    obtain ⟨h1, h2⟩ := List.append_inj h hl
    simp at h2
    have ih := hex_inj w _ _ h1
    rw [Nat.pow_succ', Nat.mod_mul, Nat.mod_mul, h2, ih]

/-- Before #3286, the five fields of `a = t ^ seq`: `(a >> 32) as u32`,
`(a >> 16) as u16`, `a & 0xFFF`, `0x8000 | (b & 0x3FFF)`, `b & 0xFFFF_FFFF_FFFF`
with `b = mixB a`. -/
def pre3286_fields (a : Nat) : Nat × Nat × Nat × Nat × Nat :=
  (a / 2 ^ 32 % 2 ^ 32, a / 2 ^ 16 % 2 ^ 16, a % 4096, 32768 + mixB a % 16384, mixB a % 2 ^ 48)

/-- `format!("{:8x}-{:4x}-4{:3x}-{:4x}-{:14x}", ..)`, hex digits as numbers and the
literal characters as distinct markers (`16` for `-`, `4` for the version digit). -/
def pre3286_uuidStr (a : Nat) : List Nat :=
  let f := pre3286_fields a
  hex 8 f.1 ++ [16] ++ hex 4 f.2.1 ++ [16, 4] ++ hex 3 f.2.2.1 ++ [16] ++ hex 4 f.2.2.2.1 ++ [16] ++ hex 12 f.2.2.2.2

/-- `u64` inverse of `MUL` modulo `2^48` (it is odd). -/
def MINV : Nat := 263363948849317

theorem minv_spec : MUL * MINV % 2 ^ 48 = 1 := by decide

/-- The low 48 bits of `b` give back the low 48 bits of `a`. -/
theorem low48 (a : Nat) : ((mixB a % 2 ^ 48 + 2 ^ 48 - INC % 2 ^ 48) * MINV) % 2 ^ 48 = a % 2 ^ 48 := by
  have h48 : (2 : Nat) ^ 64 = 2 ^ 48 * 2 ^ 16 := by decide
  have hb : mixB a % 2 ^ 48 = (a * MUL + INC) % 2 ^ 48 := by
    simp only [mixB, U64, h48, Nat.mod_mul_right_mod]
  rw [hb]
  -- (x % m + m - c % m) % m = (x - c) mod m with x = a*MUL + c
  have e1 : ((a * MUL + INC) % 2 ^ 48 + 2 ^ 48 - INC % 2 ^ 48) % 2 ^ 48 = (a * MUL) % 2 ^ 48 := by
    have := Nat.add_mod (a * MUL) INC (2 ^ 48)
    have h1 := Nat.mod_lt (a * MUL) (by decide : 0 < 2 ^ 48)
    have h2 := Nat.mod_lt INC (by decide : 0 < 2 ^ 48)
    rw [this]
    generalize (a * MUL) % 2 ^ 48 = x at *
    generalize INC % 2 ^ 48 = y at *
    by_cases hxy : x + y < 2 ^ 48
    · rw [Nat.mod_eq_of_lt hxy]
      rw [show x + y + 2 ^ 48 - y = x + 2 ^ 48 by omega, Nat.add_mod_right, Nat.mod_eq_of_lt h1]
    · rw [show (x + y) % 2 ^ 48 = x + y - 2 ^ 48 by
          rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt (by omega)]]
      rw [show x + y - 2 ^ 48 + 2 ^ 48 - y = x by omega, Nat.mod_eq_of_lt h1]
  rw [Nat.mul_mod, e1, ← Nat.mul_mod, Nat.mul_assoc, Nat.mul_mod, minv_spec, Nat.mul_one, Nat.mod_mod]

/-- History: **the pre-#3286 name determines `a`.** -/
theorem pre3286_uuid_injective (a a' : Nat) (ha : a < 2 ^ 64) (ha' : a' < 2 ^ 64)
    (h : pre3286_uuidStr a = pre3286_uuidStr a') : a = a' := by
  simp only [pre3286_uuidStr, List.append_assoc, List.cons_append, List.nil_append] at h
  obtain ⟨h1, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  have f1 := hex_inj 8 _ _ h1
  have f5 := hex_inj 12 _ _ h
  simp only [pre3286_fields] at f1 f5
  rw [show (16 : Nat) ^ 8 = 2 ^ 32 by decide, Nat.mod_mod, Nat.mod_mod] at f1
  rw [show (16 : Nat) ^ 12 = 2 ^ 48 by decide, Nat.mod_mod, Nat.mod_mod] at f5
  have hlow : a % 2 ^ 48 = a' % 2 ^ 48 := by rw [← low48 a, ← low48 a', f5]
  have d1 := Nat.div_add_mod a (2 ^ 32)
  have d2 := Nat.div_add_mod a' (2 ^ 32)
  have q1 : a / 2 ^ 32 < 2 ^ 32 := by omega
  have q2 : a' / 2 ^ 32 < 2 ^ 32 := by omega
  rw [Nat.mod_eq_of_lt q1, Nat.mod_eq_of_lt q2] at f1
  have m1 : a % 2 ^ 32 = a % 2 ^ 48 % 2 ^ 32 := by
    rw [show (2 : Nat) ^ 48 = 2 ^ 32 * 2 ^ 16 by decide, Nat.mod_mul_right_mod]
  have m2 : a' % 2 ^ 32 = a' % 2 ^ 48 % 2 ^ 32 := by
    rw [show (2 : Nat) ^ 48 = 2 ^ 32 * 2 ^ 16 by decide, Nat.mod_mul_right_mod]
  have hlo : a % 2 ^ 32 = a' % 2 ^ 32 := by rw [m1, m2, hlow]
  omega

/-- One pre-#3286 `uuid_v4` call: clock `t` (nanoseconds, truncated to `u64`) and counter `s`. -/
def pre3286_uuid (t s : Nat) : List Nat := pre3286_uuidStr ((t % U64) ^^^ (s % U64))

theorem xor_lt (x y : Nat) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64) : x ^^^ y < 2 ^ 64 :=
  Nat.xor_lt_two_pow hx hy

/-- History: **collision iff the clocks and counters cancel.** -/
theorem pre3286_uuid_collision_iff (t s t' s' : Nat) :
    pre3286_uuid t s = pre3286_uuid t' s' ↔ ((t % U64) ^^^ (s % U64)) = ((t' % U64) ^^^ (s' % U64)) := by
  have hm : ∀ x, x % U64 < 2 ^ 64 := fun x => Nat.mod_lt _ (by decide)
  constructor
  · intro h; exact pre3286_uuid_injective _ _ (xor_lt _ _ (hm _) (hm _)) (xor_lt _ _ (hm _) (hm _)) h
  · intro h; simp [pre3286_uuid, h]

/-- History: two pre-#3286 calls on the same clock reading never collided. -/
theorem pre3286_uuid_same_clock_distinct (t s s' : Nat) (hs : s < U64) (hs' : s' < U64) (hne : s ≠ s') :
    pre3286_uuid t s ≠ pre3286_uuid t s' := by
  intro h
  rw [pre3286_uuid_collision_iff] at h
  apply hne
  have e : (t % U64) ^^^ ((t % U64) ^^^ (s % U64)) = (t % U64) ^^^ ((t % U64) ^^^ (s' % U64)) := by rw [h]
  rw [← Nat.xor_assoc, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor, Nat.zero_xor,
    Nat.mod_eq_of_lt hs, Nat.mod_eq_of_lt hs'] at e
  exact e

/-- **History (#3286): consecutive counters `2, 3` read at clocks `4 ns` and `5 ns`
printed the same uuid**, and the second virtual key replaced the first one's slice. -/
theorem pre3286_uuid_collision_example : pre3286_uuid 4 2 = pre3286_uuid 5 3 := by
  rw [pre3286_uuid_collision_iff]; decide

/-! ## On main: `uuid_fields` (#3286) -/

/-- `uuid_fields(t, seq)` (redis_type.rs:812-830): `[a >> 32, (a >> 16) & 0xFFFF,
a & 0xFFF, 0x8000 | (b & 0x3FFF), b & 0xFFFF_FFFF_FFFF]` with `a = t`,
`b = seq.wrapping_mul(MUL).wrapping_add(INC)`. `0x8000 | x` is `0x8000 + x` for
`x < 0x4000`. -/
def uuidFields (t s : Nat) : Nat × Nat × Nat × Nat × Nat :=
  let a := t % U64
  let b := mixB (s % U64)
  (a / 2 ^ 32, a / 2 ^ 16 % 2 ^ 16, a % 4096, 32768 + b % 16384, b % 2 ^ 48)

/-- `format!("{f1:08x}-{f2:04x}-4{f3:03x}-{f4:04x}-{f5:012x}")` (`uuid_v4`, :799); every
field fits its width, so no field prints wider. -/
def uuidStrOf (f : Nat × Nat × Nat × Nat × Nat) : List Nat :=
  hex 8 f.1 ++ [16] ++ hex 4 f.2.1 ++ [16, 4] ++ hex 3 f.2.2.1 ++ [16] ++ hex 4 f.2.2.2.1 ++ [16] ++ hex 12 f.2.2.2.2

/-- One `uuid_v4` call on main: clock `t` (nanoseconds, `as u64`) and counter `s`. -/
def uuid (t s : Nat) : List Nat := uuidStrOf (uuidFields t s)

theorem uuidFields_fit (t s : Nat) :
    let f := uuidFields t s
    f.1 < 16 ^ 8 ∧ f.2.1 < 16 ^ 4 ∧ f.2.2.1 < 16 ^ 3 ∧ f.2.2.2.1 < 16 ^ 4 ∧ f.2.2.2.2 < 16 ^ 12 := by
  have ht := Nat.mod_lt t (show 0 < U64 by decide)
  simp only [uuidFields, U64] at *
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · have : (t % 2 ^ 64) / 2 ^ 32 < 2 ^ 32 := by omega
    simpa using this
  · exact Nat.mod_lt _ (by decide)
  · exact Nat.lt_of_lt_of_le (Nat.mod_lt _ (by decide)) (by decide)
  · have := Nat.mod_lt (mixB (s % 2 ^ 64)) (show 0 < 16384 by decide); omega
  · exact Nat.lt_of_lt_of_le (Nat.mod_lt _ (by decide)) (by decide)

theorem uuid_length (t s : Nat) : (uuid t s).length = 36 := by
  simp [uuid, uuidStrOf, hex_length]

/-- The last field depends on the counter only through `seq mod 2^48`. -/
theorem mixB_mod48 (a : Nat) : mixB a % 2 ^ 48 = ((a % 2 ^ 48) * MUL + INC) % 2 ^ 48 := by
  have h48 : (2 : Nat) ^ 64 = 2 ^ 48 * 2 ^ 16 := by decide
  simp only [mixB, U64, h48, Nat.mod_mul_right_mod]
  rw [Nat.add_mod, Nat.add_mod ((a % 2 ^ 48) * MUL), Nat.mul_mod]
  rw [Nat.mul_mod (a % 2 ^ 48) MUL, Nat.mod_mod]

/-- **The counter field is a bijection of `seq` modulo 2^48** — what the Rust
`const _` block (`redis_type.rs:839-889`) checks on sample points, for every `seq`. -/
theorem counter_field_bijective (s s' : Nat) :
    (uuidFields 0 s).2.2.2.2 = (uuidFields 0 s').2.2.2.2 ↔ s % 2 ^ 48 = s' % 2 ^ 48 := by
  simp only [uuidFields]
  have hU : ∀ x, x % U64 % 2 ^ 48 = x % 2 ^ 48 := fun x => by
    rw [show U64 = 2 ^ 48 * 2 ^ 16 by decide, Nat.mod_mul_right_mod]
  constructor
  · intro h
    rw [← hU s, ← hU s', ← low48 (s % U64), ← low48 (s' % U64), h]
  · intro h
    rw [mixB_mod48, mixB_mod48, hU, hU, h]

/-- The printed name determines the last field. -/
theorem uuid_last_field {t s t' s' : Nat} (h : uuid t s = uuid t' s') :
    (uuidFields t s).2.2.2.2 = (uuidFields t' s').2.2.2.2 := by
  simp only [uuid, uuidStrOf, List.append_assoc, List.cons_append, List.nil_append] at h
  obtain ⟨_, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  obtain ⟨_, h⟩ := List.append_inj h (by simp [hex_length])
  simp only [List.cons.injEq, true_and] at h
  have f5 := hex_inj 12 _ _ h
  have a := (uuidFields_fit t s).2.2.2.2
  have b := (uuidFields_fit t' s').2.2.2.2
  rwa [Nat.mod_eq_of_lt a, Nat.mod_eq_of_lt b] at f5

/-- **#3286: names differ for every two calls whose counters differ mod 2^48,
whatever the clock does** — in one process, any 2^48 consecutive `fetch_add`s. -/
theorem uuid_counter_distinct (t t' s s' : Nat) (hne : s % 2 ^ 48 ≠ s' % 2 ^ 48) :
    uuid t s ≠ uuid t' s' := by
  intro h
  apply hne
  have h5 := uuid_last_field h
  simp only [uuidFields] at h5
  exact (counter_field_bijective s s').1 (by simpa [uuidFields] using h5)

/-- Two calls on the same clock reading never collide (still true). -/
theorem uuid_same_clock_distinct (t s s' : Nat) (hs : s < 2 ^ 48) (hs' : s' < 2 ^ 48) (hne : s ≠ s') :
    uuid t s ≠ uuid t s' :=
  uuid_counter_distinct t t s s' (by rwa [Nat.mod_eq_of_lt hs, Nat.mod_eq_of_lt hs'])

/-- `same_fields` (:839-850, inside the compile-time `const _` check): `same &= a[i] == b[i]`
for `i` in `0..5`, as a fold over the arrays. -/
def sameFields (a b : List Nat) : Bool :=
  (List.range 5).foldl (fun same i => same && (a.getD i 0 == b.getD i 0)) true

theorem sameFields_iff (a b : List Nat) (ha : a.length = 5) (hb : b.length = 5) :
    sameFields a b = true ↔ a = b := by
  match a, b, ha, hb with
  | [a0, a1, a2, a3, a4], [b0, b1, b2, b3, b4], _, _ =>
    simp [sameFields, List.range, List.range.loop, and_assoc]

/-- The array `uuid_fields` returns, as `same_fields` reads it. -/
def fieldsL (t s : Nat) : List Nat :=
  let f := uuidFields t s
  [f.1, f.2.1, f.2.2.1, f.2.2.2.1, f.2.2.2.2]

/-- The `const _` block's assertions hold for **every** clock reading, not just
its samples: `!same_fields(uuid_fields(t, s), uuid_fields(t', s'))` whenever the
counters differ mod 2^48. -/
theorem const_check_holds (t t' s s' : Nat) (hne : s % 2 ^ 48 ≠ s' % 2 ^ 48) :
    sameFields (fieldsL t s) (fieldsL t' s') = false := by
  have h := (sameFields_iff (fieldsL t s) (fieldsL t' s') (by simp [fieldsL]) (by simp [fieldsL]))
  cases hb : sameFields (fieldsL t s) (fieldsL t' s') with
  | false => rfl
  | true =>
    exfalso
    have e := h.1 hb
    simp only [fieldsL, List.cons.injEq] at e
    exact hne ((counter_field_bijective s s').1 (by simpa [uuidFields] using e.2.2.2.2.1))

/-- The reported regression pair now differs (`!same_fields(uuid_fields(4, 2), uuid_fields(5, 3))`). -/
theorem uuid_regression_pair_distinct : uuid 4 2 ≠ uuid 5 3 :=
  uuid_counter_distinct 4 5 2 3 (by decide)

/-! ## Virtual-key names -/

/-- `if graph_name.contains('{') { "{g}_{u}" } else { "{{g}}{g}_{u}" }`, as character codes
(`123` = `{`, `125` = `}`, `95` = `_`). -/
def vkeyName (g : List Nat) (u : List Nat) : List Nat :=
  if 123 ∈ g then g ++ [95] ++ u else [123] ++ g ++ [125] ++ g ++ [95] ++ u

theorem vkeyName_inj (g u u' : List Nat) (hl : u.length = u'.length) (h : vkeyName g u = vkeyName g u') : u = u' := by
  unfold vkeyName at h
  split at h
  · exact (List.append_inj h (by simp)).2
  · exact (List.append_inj h (by simp)).2

theorem vkeyName_ne (g u : List Nat) (hu : u ≠ []) : vkeyName g u ≠ g := by
  intro h
  have := congrArg List.length h
  unfold vkeyName at this
  have : 0 < u.length := List.length_pos_iff.2 hu
  split at * <;> simp at * <;> omega

theorem vkeyName_suffix (g u : List Nat) : ∃ p, vkeyName g u = p ++ u := by
  unfold vkeyName; split
  · exact ⟨g ++ [95], by simp⟩
  · exact ⟨[123] ++ g ++ [125] ++ g ++ [95], by simp⟩

/-- **Two equal virtual-key names, of any graphs, came from the same counter mod
2^48**: the uuid is the name's fixed-length tail. (Before #3286 this failed:
`pre3286_uuid_collision_example`.) -/
theorem vkeyName_uuid_counter (g g' : List Nat) (t t' s s' : Nat)
    (h : vkeyName g (uuid t s) = vkeyName g' (uuid t' s')) : s % 2 ^ 48 = s' % 2 ^ 48 := by
  obtain ⟨p, hp⟩ := vkeyName_suffix g (uuid t s)
  obtain ⟨p', hp'⟩ := vkeyName_suffix g' (uuid t' s')
  rw [hp, hp'] at h
  have hu := (List.append_inj' h (by rw [uuid_length, uuid_length])).2
  by_cases hne : s % 2 ^ 48 = s' % 2 ^ 48
  · exact hne
  · exact absurd hu (uuid_counter_distinct t t' s s' hne)

/-- **Discharges `VKeys.Names.vname_inj`** for a naming that gives the `i`-th virtual
key of graph `g` the uuid of call `ctr g i` (at any clock reading `clk g i`), provided
the calls are distinct mod 2^48 — `COUNTER.fetch_add` hands each call its own value,
so this holds for any 2^48 consecutive names of one process. -/
theorem names_vname_inj (clk ctr : List Nat → Nat → Nat)
    (hctr : ∀ g i g' j, ctr g i % 2 ^ 48 = ctr g' j % 2 ^ 48 → g = g' ∧ i = j) :
    ∀ g i g' j, vkeyName g (uuid (clk g i) (ctr g i)) = vkeyName g' (uuid (clk g' j) (ctr g' j)) →
      g = g' ∧ i = j :=
  fun g i g' j h => hctr g i g' j (vkeyName_uuid_counter _ _ _ _ _ _ h)

end GraphPersist.RedisType
