/-
# Byte-level helpers of the COW B+-tree leaf pages
(`graph/src/index/falkordb/data_structures/cow_btree/{mod.rs,leaf/*.rs}`, origin/main 3fec7d7c9)

Bytes are `Nat`s (`< 256` where it matters); `le w x` is `x.to_le_bytes()[..w]`
(for `x < 256^8`), `rd b off w` reads `w` little-endian bytes at `off`.

| Lean | Rust |
| --- | --- |
| `le`, `unle`, `rd` | `to_le_bytes()[..w]`, `from_le_bytes` |
| `readU64` | `read_u64` `cow_btree/mod.rs:64` |
| `readWidth` | `read_width` `cow_btree/mod.rs:73` |
| `readU16` | `read_u16` `leaf/compact.rs:18` |
| `widthFor` | `narrow_int::width_for` via `pow2_bytes_for` `leaf/mod.rs:42` |
| `slice` | `&bytes[lo..hi]` |
-/
namespace IndexLayer.Leaf

def le : Nat → Nat → List Nat
  | 0, _ => []
  | w + 1, x => x % 256 :: le w (x / 256)

def unle : List Nat → Nat
  | [] => 0
  | b :: bs => b + 256 * unle bs

theorem le_length : ∀ w x, (le w x).length = w
  | 0, _ => rfl
  | w + 1, x => by simp [le, le_length w]

theorem unle_le : ∀ w x, unle (le w x) = x % 256 ^ w
  | 0, x => by simp [le, unle, Nat.mod_one]
  | w + 1, x => by
    simp only [le, unle, unle_le w]
    rw [Nat.pow_succ, Nat.mul_comm (256 ^ w) 256, Nat.mod_mul]

theorem unle_le_of_lt (w x : Nat) (h : x < 256 ^ w) : unle (le w x) = x := by
  rw [unle_le, Nat.mod_eq_of_lt h]

def rd (b : List Nat) (off w : Nat) : Nat := unle ((b.drop off).take w)

/-- `read_u64` (`cow_btree/mod.rs:64`). -/
def readU64 (b : List Nat) (off : Nat) : Nat := rd b off 8
/-- `read_u16` (`leaf/compact.rs:18`). -/
def readU16 (b : List Nat) (off : Nat) : Nat := rd b off 2
/-- `read_width` (`cow_btree/mod.rs:73`): widths 1/2/4, anything else reads 8. -/
def readWidth (b : List Nat) (off w : Nat) : Nat :=
  match w with
  | 1 => rd b off 1
  | 2 => rd b off 2
  | 4 => rd b off 4
  | _ => rd b off 8

def W (w : Nat) : Prop := w = 1 ∨ w = 2 ∨ w = 4 ∨ w = 8

theorem readWidth_eq (b : List Nat) (off w : Nat) (hw : W w) : readWidth b off w = rd b off w := by
  rcases hw with rfl | rfl | rfl | rfl <;> rfl

/-- `narrow_int::width_for`. -/
def widthFor (x : Nat) : Nat :=
  if x ≤ 0xFF then 1 else if x ≤ 0xFFFF then 2 else if x ≤ 0xFFFFFFFF then 4 else 8

theorem widthFor_W (x : Nat) : W (widthFor x) := by
  unfold widthFor W; split <;> (try split) <;> (try split) <;> simp

/-- The width is big enough for any `u64`. -/
theorem lt_widthFor (x : Nat) (hx : x < 2 ^ 64) : x < 256 ^ widthFor x := by
  unfold widthFor; split
  · omega
  · split
    · omega
    · split
      · show x < 256 ^ 4; omega
      · show x < 256 ^ 8; simpa using hx

theorem lt_of_widthFor_le (x w : Nat) (hx : x < 2 ^ 64) (hw : W w) (h : widthFor x ≤ w) : x < 256 ^ w := by
  have h1 := lt_widthFor x hx
  have : 256 ^ widthFor x ≤ 256 ^ w := Nat.pow_le_pow_right (by omega) h
  omega

/-! ## Fixed-width chunks -/

theorem drop_app (l1 l2 : List Nat) (i : Nat) : (l1 ++ l2).drop (l1.length + i) = l2.drop i := by
  induction l1 with
  | nil => simp
  | cons a l ih => simp only [List.cons_append, List.length_cons]; rw [show l.length + 1 + i = (l.length + i) + 1 by omega]; simp [ih]

theorem take_app (l1 l2 : List Nat) (i : Nat) : (l1 ++ l2).take (l1.length + i) = l1 ++ l2.take i := by
  induction l1 with
  | nil => simp
  | cons a l ih => simp only [List.cons_append, List.length_cons]; rw [show l.length + 1 + i = (l.length + i) + 1 by omega]; simp [ih]

theorem succ_mul' (i w : Nat) : (i + 1) * w = w + i * w := by rw [Nat.succ_mul, Nat.add_comm]

theorem drop_flatMap {α : Type} (f : α → List Nat) (w : Nat) (hf : ∀ a, (f a).length = w) :
    ∀ (xs : List α) (i : Nat), (xs.flatMap f).drop (i * w) = (xs.drop i).flatMap f
  | [], i => by simp
  | x :: xs, 0 => by simp
  | x :: xs, i + 1 => by
    rw [List.flatMap_cons, succ_mul', ← hf x, drop_app, hf x, drop_flatMap f w hf xs i]
    simp

theorem take_flatMap {α : Type} (f : α → List Nat) (w : Nat) (hf : ∀ a, (f a).length = w) :
    ∀ (xs : List α) (i : Nat), (xs.flatMap f).take (i * w) = (xs.take i).flatMap f
  | [], i => by simp
  | x :: xs, 0 => by simp
  | x :: xs, i + 1 => by
    rw [List.flatMap_cons, succ_mul', ← hf x, take_app, hf x, take_flatMap f w hf xs i]
    simp

theorem length_flatMap_const {α : Type} (f : α → List Nat) (w : Nat) (hf : ∀ a, (f a).length = w)
    (xs : List α) : (xs.flatMap f).length = xs.length * w := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih, hf, Nat.succ_mul]; omega

/-- Reading chunk `i` of a run of `w`-byte chunks behind a prefix. -/
theorem rd_chunk {α : Type} (f : α → List Nat) (w : Nat) (hf : ∀ a, (f a).length = w)
    (pre post : List Nat) (xs : List α) (i : Nat) (hi : i < xs.length) (off : Nat)
    (ho : off = pre.length + i * w) :
    rd (pre ++ xs.flatMap f ++ post) off w = unle (f xs[i]) := by
  subst ho
  unfold rd
  rw [List.append_assoc, drop_app, List.drop_append_of_le_length, drop_flatMap f w hf,
    List.drop_eq_getElem_cons hi, List.flatMap_cons, List.append_assoc,
    List.take_left' (hf _)]
  rw [length_flatMap_const f w hf]
  exact Nat.mul_le_mul_right _ (Nat.le_of_lt hi)

def slice (b : List Nat) (lo hi : Nat) : List Nat := (b.drop lo).take (hi - lo)

/-- `&bytes[pre.len() + i*w .. pre.len() + j*w]` of a chunk run is the chunks `i..j`. -/
theorem slice_chunks {α : Type} (f : α → List Nat) (w : Nat) (hf : ∀ a, (f a).length = w)
    (pre post : List Nat) (xs : List α) (i j : Nat) (hij : i ≤ j) (hj : j ≤ xs.length) :
    slice (pre ++ xs.flatMap f ++ post) (pre.length + i * w) (pre.length + j * w) =
      ((xs.drop i).take (j - i)).flatMap f := by
  unfold slice
  rw [List.append_assoc, drop_app, List.drop_append_of_le_length, drop_flatMap f w hf]
  · rw [show pre.length + j * w - (pre.length + i * w) = (j - i) * w by
        rw [Nat.add_sub_add_left, ← Nat.sub_mul]]
    rw [List.take_append_of_le_length, take_flatMap f w hf]
    rw [length_flatMap_const f w hf]; simp
    exact Nat.mul_le_mul_right _ (by omega)
  · rw [length_flatMap_const f w hf]; exact Nat.mul_le_mul_right _ (by omega)

/-- Reading `k` bytes at offset `o` inside chunk `i`. -/
theorem rd_in_chunk {α : Type} (f : α → List Nat) (w : Nat) (hf : ∀ a, (f a).length = w)
    (pre post : List Nat) (xs : List α) (i : Nat) (hi : i < xs.length) (o k : Nat) (hok : o + k ≤ w)
    (off : Nat) (ho : off = pre.length + i * w + o) :
    rd (pre ++ xs.flatMap f ++ post) off k = unle (((f xs[i]).drop o).take k) := by
  subst ho
  unfold rd
  rw [List.append_assoc, Nat.add_assoc, drop_app, ← List.drop_drop, List.drop_append_of_le_length,
    drop_flatMap f w hf, List.drop_eq_getElem_cons hi, List.flatMap_cons, List.append_assoc,
    List.drop_append_of_le_length (by rw [hf]; omega), List.take_append_of_le_length]
  · simp [hf]; omega
  · rw [length_flatMap_const f w hf]; exact Nat.mul_le_mul_right _ (Nat.le_of_lt hi)

theorem rd_app_left (A B : List Nat) (off w : Nat) (h : off + w ≤ A.length) : rd (A ++ B) off w = rd A off w := by
  unfold rd
  rw [List.drop_append_of_le_length (by omega), List.take_append_of_le_length (by simp; omega)]

theorem rd_app_right (A B : List Nat) (off w : Nat) : rd (A ++ B) (A.length + off) w = rd B off w := by
  unfold rd; rw [drop_app]

theorem rd_app_right' (A B : List Nat) (off w : Nat) (h : A.length ≤ off) :
    rd (A ++ B) off w = rd B (off - A.length) w := by
  have := rd_app_right A B (off - A.length) w; rwa [Nat.add_sub_cancel' h] at this

theorem rd_le (w x : Nat) (h : x < 256 ^ w) : rd (le w x) 0 w = x := by
  unfold rd; rw [List.drop_zero, List.take_of_length_le (by rw [le_length]; exact Nat.le_refl _)]; exact unle_le_of_lt w x h

theorem le_take : ∀ (w k x : Nat), (le (w + k) x).take w = le w x
  | 0, k, x => by simp [le]
  | w + 1, k, x => by
    rw [show w + 1 + k = (w + k) + 1 by omega]; simp only [le, List.take_succ_cons]; rw [le_take w k (x / 256)]

/-- `x.to_le_bytes()[..w]` for `w ≤ 8`. -/
theorem le8_take (w x : Nat) (hw : w ≤ 8) : (le 8 x).take w = le w x := by
  have := le_take w (8 - w) x; rwa [show w + (8 - w) = 8 by omega] at this

def byteAt (b : List Nat) (i : Nat) : Nat := b.getD i 0

end IndexLayer.Leaf
