/-
# `compound_key`

`tensor.rs`, since #2579:

```rust
pub const BLOCK_SHIFT: u32 = 30;
const BLOCK_MASK: u64 = (1u64 << BLOCK_SHIFT) - 1;

pub fn compound_key(src: u64, dst: u64) -> (MeBlock, u64) {
    (
        (src >> BLOCK_SHIFT, dst >> BLOCK_SHIFT),
        ((src & BLOCK_MASK) << BLOCK_SHIFT) | (dst & BLOCK_MASK),
    )
}

pub fn compound_key_inverse(block: MeBlock, row: u64) -> (u64, u64) {
    (
        (block.0 << BLOCK_SHIFT) | (row >> BLOCK_SHIFT),
        (block.1 << BLOCK_SHIFT) | (row & BLOCK_MASK),
    )
}
```

Each endpoint is cut in two: the low `BLOCK_SHIFT` bits pack into a row key, the
high bits become one coordinate of a *block*. Equivalently, the `src x dst`
identifier space is tiled into `2^30 x 2^30` tiles; the block names the tile and
the row names the cell inside it.

**What changed, and why it matters here.** The predecessor was
`(src << 32) | dst`, and its correctness was *conditional*: injective only for
`dst < 2^32`, in range for `me` only for `src < 2^28`. Both conditions appeared
in this development as the hypothesis `Bounded`, threaded through the invariants
and every theorem that touched `me`. The new key is **total** — it is built from
masked halves, so it cannot alias and cannot leave range whatever the input —
and the theorems below are therefore stated with no hypothesis at all. That is
the improvement the paper's limits section predicted: `Bounded` is not weakened
here, it is *gone*, and every result that used to assume it now holds outright.

Proved here:

* the bitwise form equals the arithmetic model (`rowBits_eq_row`), so the rest of
  the development can use arithmetic — and unlike its predecessor this needs no
  side condition;
* a row key is always in range for `me`'s declared row count
  (`row_lt`, `row_le_grbIndexMax`), so no write can be dropped;
* the address `key = (block, row)` is **injective**, unconditionally
  (`key_inj`), which is what makes "one `me` row per pair" meaningful;
* `compound_key_inverse` inverts it, unconditionally (`inv_key`, `blockRow_hi`,
  `blockRow_lo`), which `iter_edges` and `Iter` rely on to recover `(src, dst)`;
* within one block the row alone is injective (`row_inj_of_block`), which is the
  form the per-block reasoning in `Iter` and `Remove` wants.
-/
import Tensor.Model

namespace FalkorDB
namespace Tensor

/-- `BLOCK_MASK`. `blockShift`, `blockOf`, `row` and the address `key` itself are
in `Model.lean`, since the tensor's own definition needs them. -/
def blockMask : Nat := 2 ^ blockShift - 1

/-- `compound_key` exactly as Rust writes it: masks, a shift and an or. -/
def rowBits (p : Pair) : Nat :=
  ((p.1 &&& blockMask) <<< blockShift) ||| (p.2 &&& blockMask)

/-- `x &&& (2^n - 1) = x % 2^n`: masking is the low-bit projection. -/
private theorem and_mask (x n : Nat) : x &&& (2 ^ n - 1) = x % 2 ^ n := by
  simpa using Nat.and_two_pow_sub_one_eq_mod x n

/-- `2 ^ blockShift` is positive. -/
private theorem B_pos : 0 < 2 ^ blockShift := Nat.two_pow_pos _

/-- The low half is genuinely below `2 ^ blockShift`. -/
theorem row_hi_lt (p : Pair) : p.1 % 2 ^ blockShift < 2 ^ blockShift := Nat.mod_lt _ B_pos

/-- Likewise for the destination. -/
theorem row_lo_lt (p : Pair) : p.2 % 2 ^ blockShift < 2 ^ blockShift := Nat.mod_lt _ B_pos

/-- Packing two sub-`B` halves stays below `B * B`. The generic step behind
`row_lt`: stated over a variable `B` because `omega` does not reason about
`2 ^ blockShift` as an exponential. -/
private theorem pack_lt {a b B : Nat} (ha : a < B) (hb : b < B) : a * B + b < B * B := by
  have hsucc : (a + 1) * B = a * B + B := Nat.succ_mul a B
  have hle : (a + 1) * B ≤ B * B := Nat.mul_le_mul_right B (by omega)
  omega

/-- Dividing a packed key by `B` recovers the high half. -/
private theorem pack_div {a b B : Nat} (hB : 0 < B) (hb : b < B) : (a * B + b) / B = a := by
  rw [Nat.mul_comm, Nat.mul_add_div hB, Nat.div_eq_of_lt hb, Nat.add_zero]

/-- Taking a packed key mod `B` recovers the low half. -/
private theorem pack_mod {a b B : Nat} (hb : b < B) : (a * B + b) % B = b := by
  rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hb]

/-- The bitwise definition agrees with the arithmetic model — **unconditionally**.
Its predecessor `keyBits_eq_key` needed `dst < 2 ^ 32`; masking supplies that
here, so there is no hypothesis to discharge. -/
theorem rowBits_eq_row (p : Pair) : rowBits p = row p := by
  unfold rowBits row blockMask
  rw [and_mask, and_mask, ← Nat.shiftLeft_add_eq_or_of_lt (row_lo_lt p), Nat.shiftLeft_eq]

/-- A row key is below `2 ^ 60`: the two halves are `blockShift` bits each and
`2 * blockShift = 60`. Unconditional, which is exactly the property whose absence
made the old key drop writes above `src = 2 ^ 28`. -/
theorem row_lt (p : Pair) : row p < 2 ^ 60 := by
  have hsq : 2 ^ blockShift * 2 ^ blockShift = 2 ^ 60 := by
    rw [← Nat.pow_add]
    rfl
  rw [← hsq]
  simp only [row]
  exact pack_lt (row_hi_lt p) (row_lo_lt p)

/-- Restated against `me`'s declared row count. `ME_DIM = GrB_INDEX_MAX + 1`, and
a matrix of `n` rows indexes `0 .. n-1`, so this is the writability condition —
the Rust counterpart is `the_top_row_key_of_a_block_is_writable`. -/
theorem row_le_grbIndexMax (p : Pair) : row p ≤ GrBIndexMax := by
  have h := row_lt p
  simp only [GrBIndexMax]
  omega

/-- `row >>> blockShift` recovers the source's low half. -/
theorem row_hi (p : Pair) : row p / 2 ^ blockShift = p.1 % 2 ^ blockShift :=
  pack_div B_pos (row_lo_lt p)

/-- `row &&& blockMask` recovers the destination's low half. -/
theorem row_lo (p : Pair) : row p % 2 ^ blockShift = p.2 % 2 ^ blockShift :=
  pack_mod (row_lo_lt p)

/-- `compound_key_inverse`, as Rust writes it. -/
def keyInverse (b : Nat × Nat) (r : Nat) : Pair :=
  (b.1 * 2 ^ blockShift + r / 2 ^ blockShift, b.2 * 2 ^ blockShift + r % 2 ^ blockShift)

/-- The source is reassembled from its block coordinate and the row's high half. -/
theorem blockRow_hi (p : Pair) :
    (blockOf p).1 * 2 ^ blockShift + row p / 2 ^ blockShift = p.1 := by
  rw [row_hi]
  have h := Nat.div_add_mod p.1 (2 ^ blockShift)
  rw [Nat.mul_comm] at h
  simpa only [blockOf] using h

/-- The destination is reassembled from its block coordinate and the row's low
half. -/
theorem blockRow_lo (p : Pair) :
    (blockOf p).2 * 2 ^ blockShift + row p % 2 ^ blockShift = p.2 := by
  rw [row_lo]
  have h := Nat.div_add_mod p.2 (2 ^ blockShift)
  rw [Nat.mul_comm] at h
  simpa only [blockOf] using h

/-- `compound_key_inverse` inverts `compound_key`, for **every** pair. -/
theorem inv_key (p : Pair) : keyInverse (blockOf p) (row p) = p := by
  have h1 := blockRow_hi p
  have h2 := blockRow_lo p
  simp only [keyInverse]
  exact Prod.ext h1 h2

/-- `compound_key` is injective, unconditionally: two different node pairs can
never share a `(block, row)` address, so never share an `me` row. -/
theorem key_inj {p q : Pair} (h : key p = key q) : p = q := by
  have hb : blockOf p = blockOf q := (Prod.ext_iff.mp h).1
  have hr : row p = row q := (Prod.ext_iff.mp h).2
  rw [← inv_key p, ← inv_key q, hb, hr]

theorem key_ne {p q : Pair} (h : p ≠ q) : key p ≠ key q :=
  fun he => h (key_inj he)

/-- Within one block the row key alone is injective. This is the form the
per-block reasoning wants: a single `me` matrix holds one block, so inside it a
row identifies a pair. -/
theorem row_inj_of_block {p q : Pair} (hb : blockOf p = blockOf q) (hr : row p = row q) :
    p = q :=
  key_inj (by simp only [key, hb, hr])

/-- Distinct pairs in the *same* block have distinct rows — the contrapositive,
which is what a disjointness argument inside one matrix uses. -/
theorem row_ne_of_block {p q : Pair} (hb : blockOf p = blockOf q) (h : p ≠ q) :
    row p ≠ row q :=
  fun hr => h (row_inj_of_block hb hr)

/-- Every pair within `2 ^ blockShift` per axis lands in `ME_BLOCK_0` — the block
`Tensor` always holds, and the only one a graph under 2^30 nodes per axis ever
uses. -/
theorem blockOf_eq_zero {p : Pair} (h1 : p.1 < 2 ^ blockShift) (h2 : p.2 < 2 ^ blockShift) :
    blockOf p = (0, 0) := by
  simp only [blockOf, Nat.div_eq_of_lt h1, Nat.div_eq_of_lt h2]

/-- In block `0` the row key is the old-style packing of the two identifiers,
now at `blockShift` bits per side rather than 32. Ties the general statement back
to the single-matrix case the engine actually runs in. -/
theorem row_of_block_zero {p : Pair} (h1 : p.1 < 2 ^ blockShift) (h2 : p.2 < 2 ^ blockShift) :
    row p = p.1 * 2 ^ blockShift + p.2 := by
  simp only [row, Nat.mod_eq_of_lt h1, Nat.mod_eq_of_lt h2]

end Tensor
end FalkorDB
