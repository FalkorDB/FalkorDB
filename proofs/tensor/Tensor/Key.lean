/-
# `compound_key`

`tensor.rs`:

```rust
pub fn compound_key(src: u64, dst: u64) -> u64 {
    assert!(u32::try_from(src).is_ok() && u32::try_from(dst).is_ok(), ...);
    (src << 32) | dst
}
```

and the inverse used by `iter_edges`:

```rust
.map(|(key, edge_id)| (key >> 32, key & 0xFFFF_FFFF, edge_id))
```

Proved here:

* the bitwise form equals the arithmetic model `key p = src * 2^32 + dst`
  (`keyBits_eq_key`), so the rest of the development can use arithmetic;
* the key of a `Bounded` pair fits in a `u64` — no truncation (`key_lt`);
* `key` is **injective** on `Bounded` pairs (`key_inj`), which is what makes
  "one `me` row per pair" meaningful;
* `>> 32` / `& 0xFFFF_FFFF` invert it (`keyHi`, `keyLo`), which `iter_edges`
  and `Iter` rely on to recover `(src, dst)`.
-/
import Tensor.Model

namespace FalkorDB
namespace Tensor

/-- `compound_key` exactly as Rust writes it: a shift and an or. -/
def keyBits (p : Pair) : Nat := (p.1 <<< 32) ||| p.2

/-- The bitwise definition agrees with the arithmetic model whenever `dst` fits
in a `u32` — which `compound_key`'s assertion guarantees. -/
theorem keyBits_eq_key {p : Pair} (h : p.2 < 2 ^ 32) : keyBits p = key p := by
  rw [keyBits, ← Nat.shiftLeft_add_eq_or_of_lt h, Nat.shiftLeft_eq, key]

/-- A `Bounded` pair's key fits in a `u64`: the packing never truncates. -/
theorem key_lt {p : Pair} (h : Bounded p) : key p < 2 ^ 64 := by
  obtain ⟨h1, h2⟩ := h
  simp only [key]
  omega

/-- `key >> 32` recovers `src`. -/
theorem keyHi {p : Pair} (h : Bounded p) : key p >>> 32 = p.1 := by
  have h2 := h.2
  simp only [key, Nat.shiftRight_eq_div_pow]
  omega

/-- `key & 0xFFFF_FFFF` recovers `dst`. -/
theorem keyLo {p : Pair} (h : Bounded p) : key p % 2 ^ 32 = p.2 := by
  have h2 := h.2
  simp only [key]
  omega

/-- `compound_key` is injective on `Bounded` pairs: two different node pairs can
never share an `me` row. -/
theorem key_inj {p q : Pair} (hp : Bounded p) (hq : Bounded q) (h : key p = key q) : p = q := by
  have h1 : p.1 = q.1 := by rw [← keyHi hp, ← keyHi hq, h]
  have h2 : p.2 = q.2 := by rw [← keyLo hp, ← keyLo hq, h]
  exact Prod.ext h1 h2

theorem key_ne {p q : Pair} (hp : Bounded p) (hq : Bounded q) (h : p ≠ q) : key p ≠ key q :=
  fun he => h (key_inj hp hq he)

end Tensor
end FalkorDB
