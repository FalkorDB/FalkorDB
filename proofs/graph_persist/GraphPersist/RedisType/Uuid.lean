/-! # `uuid_v4` (redis_type.rs:783-802) and virtual-key names (:463-468)

`uuid_v4` mixes the wall clock with a process-wide counter: `a = (nanos as u64) ^ seq`,
`b = a * 6364136223846793005 + 1442695040888963407` (wrapping), and prints five
fixed-width hex fields of `a` and `b`.

* `uuid_injective` — the printed string determines `a`: distinct `a` give distinct names.
* `uuid_same_clock_distinct` — two calls reading the same clock value never collide.
* `uuid_collision_iff` / `uuid_collision_example` — two calls collide **iff**
  `t₁ ⊕ s₁ = t₂ ⊕ s₂`; consecutive counters `2, 3` at clock `4, 5` ns collide. Uniqueness of
  a save's virtual-key names therefore rests on the clock, not on the counter.
* `vkeyName_inj`, `vkeyName_ne` — virtual-key names of one graph differ iff their uuids do,
  and none equals the graph's own name.
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

/-- The five fields: `(a >> 32) as u32`, `(a >> 16) as u16`, `a & 0xFFF`,
`0x8000 | (b & 0x3FFF)`, `b & 0xFFFF_FFFF_FFFF`. -/
def fields (a : Nat) : Nat × Nat × Nat × Nat × Nat :=
  (a / 2 ^ 32 % 2 ^ 32, a / 2 ^ 16 % 2 ^ 16, a % 4096, 32768 + mixB a % 16384, mixB a % 2 ^ 48)

/-- `format!("{:8x}-{:4x}-4{:3x}-{:4x}-{:14x}", ..)`, hex digits as numbers and the
literal characters as distinct markers (`16` for `-`, `4` for the version digit). -/
def uuidStr (a : Nat) : List Nat :=
  let f := fields a
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

/-- **The name determines `a`.** -/
theorem uuid_injective (a a' : Nat) (ha : a < 2 ^ 64) (ha' : a' < 2 ^ 64) (h : uuidStr a = uuidStr a') : a = a' := by
  simp only [uuidStr, List.append_assoc, List.cons_append, List.nil_append] at h
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
  simp only [fields] at f1 f5
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

/-- One `uuid_v4` call: clock `t` (nanoseconds, truncated to `u64`) and counter `s`. -/
def uuid (t s : Nat) : List Nat := uuidStr ((t % U64) ^^^ (s % U64))

theorem xor_lt (x y : Nat) (hx : x < 2 ^ 64) (hy : y < 2 ^ 64) : x ^^^ y < 2 ^ 64 :=
  Nat.xor_lt_two_pow hx hy

/-- **Collision iff the clocks and counters cancel.** -/
theorem uuid_collision_iff (t s t' s' : Nat) :
    uuid t s = uuid t' s' ↔ ((t % U64) ^^^ (s % U64)) = ((t' % U64) ^^^ (s' % U64)) := by
  have hm : ∀ x, x % U64 < 2 ^ 64 := fun x => Nat.mod_lt _ (by decide)
  constructor
  · intro h; exact uuid_injective _ _ (xor_lt _ _ (hm _) (hm _)) (xor_lt _ _ (hm _) (hm _)) h
  · intro h; simp [uuid, h]

/-- Two calls on the same clock reading never collide: the counter differs. -/
theorem uuid_same_clock_distinct (t s s' : Nat) (hs : s < U64) (hs' : s' < U64) (hne : s ≠ s') :
    uuid t s ≠ uuid t s' := by
  intro h
  rw [uuid_collision_iff] at h
  apply hne
  have e : (t % U64) ^^^ ((t % U64) ^^^ (s % U64)) = (t % U64) ^^^ ((t % U64) ^^^ (s' % U64)) := by rw [h]
  rw [← Nat.xor_assoc, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor, Nat.zero_xor,
    Nat.mod_eq_of_lt hs, Nat.mod_eq_of_lt hs'] at e
  exact e

/-- Consecutive counters `2, 3` read at clocks `4 ns` and `5 ns` print the same uuid. -/
theorem uuid_collision_example : uuid 4 2 = uuid 5 3 := by
  rw [uuid_collision_iff]; decide

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

end GraphPersist.RedisType
