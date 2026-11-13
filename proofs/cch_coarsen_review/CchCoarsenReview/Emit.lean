import CchCoarsenReview.Rows
/-!
# `emit_csr`: the flat arrays every host pass writes, and the `hpar` chunks

**Rust** (m1opt.patch @7df8e5887): `emit_csr_t`, m1_coarsen_bench.rs:1277-1349. With one chunk
(:1292-1302) the pass `f` pushes `cols.len()` into `cnt` before each row and appends the row's columns
and values; a last `cnt.push(cols.len())` closes the offsets, and the three `Vec`s become the CSR
(`pack_csr_t`). With `k` chunks (`hpar`, :1303-1348; boundaries from `row_splits`, :486-498, ascending from 0 to `nrows`) chunk `c` runs `f` on rows `splits[c]..splits[c+1]`
into its own buffers; then `p[r0 + q] = offs[c] + cnt_c[q]` and the columns are copied at `offs[c]`,
where `offs` are the chunk sizes' prefix sums.

The row-level models (`Rows`) say what each row holds. `emit_row` proves the flat arrays hold exactly
those rows (row `i` is the slice between consecutive offsets), and `chunks_eq_serial` that the chunked
assembly writes the same offsets as one chunk. So the denotation of every host pass's output is the
row lists the other files reason about. (`Csr::new` reads a row the same way, as the slice
`aj[ap[i]..ap[i+1]]`; that is the definition of `Rows` there.)
-/

namespace CchCoarsen

/-- The offsets one chunk writes: `cnt` before each row, starting at `acc`, then the closing push. -/
def emitOffsets {β : Type} : List (List β) → Nat → List Nat
  | [], acc => [acc]
  | r :: rs, acc => acc :: emitOffsets rs (acc + r.length)

/-- Row `i` of a flat CSR: the entries between offsets `i` and `i + 1`. -/
def sliceRow {β : Type} (p : List Nat) (e : List β) (i : Nat) : List β :=
  (e.drop (p.getD i 0)).take (p.getD (i + 1) 0 - p.getD i 0)

theorem emitOffsets_head {β : Type} (rows : List (List β)) (acc : Nat) : (emitOffsets rows acc).getD 0 0 = acc := by
  cases rows <;> rfl

theorem emit_row_gen {β : Type} (rows : List (List β)) (acc : Nat) (pre : List β) (hpre : pre.length = acc)
    (i : Nat) (hi : i < rows.length) :
    sliceRow (emitOffsets rows acc) (pre ++ rows.flatten) i = rows[i] := by
  induction rows generalizing acc pre i with
  | nil => simp at hi
  | cons r rs ih =>
    cases i with
    | zero =>
      simp only [sliceRow, emitOffsets, List.getD_cons_zero, List.getD_cons_succ, emitOffsets_head,
        List.flatten_cons, List.getElem_cons_zero]
      rw [← hpre, List.drop_left' rfl]
      simp
    | succ k =>
      simp only [sliceRow, emitOffsets, List.getD_cons_succ, List.flatten_cons, List.getElem_cons_succ]
      have := ih (acc + r.length) (pre ++ r) (by simp [hpre]) k (by simpa using hi)
      simpa [sliceRow, List.append_assoc] using this

/-- **The flat arrays hold the rows**: row `i` of `(emitOffsets rows 0, rows.flatten)` is `rows[i]`. -/
theorem emit_row {β : Type} (rows : List (List β)) (i : Nat) (hi : i < rows.length) :
    sliceRow (emitOffsets rows 0) rows.flatten i = rows[i] := by
  simpa using emit_row_gen rows 0 [] rfl i hi

theorem emitOffsets_shift {β : Type} (rows : List (List β)) (acc : Nat) :
    emitOffsets rows acc = (emitOffsets rows 0).map (· + acc) := by
  induction rows generalizing acc with
  | nil => simp [emitOffsets]
  | cons r rs ih =>
    simp only [emitOffsets, List.map_cons, Nat.zero_add]
    rw [ih (acc + r.length), ih r.length, List.map_map]
    congr 1
    apply List.map_congr_left; intro x _; simp only [Function.comp]; omega

theorem emitOffsets_append {β : Type} (rs₁ rs₂ : List (List β)) (acc : Nat) :
    emitOffsets (rs₁ ++ rs₂) acc = (emitOffsets rs₁ acc).dropLast ++ emitOffsets rs₂ (acc + rs₁.flatten.length) := by
  induction rs₁ generalizing acc with
  | nil => simp [emitOffsets]
  | cons r rs ih =>
    simp only [List.cons_append, emitOffsets, List.flatten_cons, List.length_append]
    rw [ih (acc + r.length)]
    have : (acc :: emitOffsets rs (acc + r.length)).dropLast = acc :: (emitOffsets rs (acc + r.length)).dropLast := by
      cases h : emitOffsets rs (acc + r.length) with
      | nil => cases rs <;> simp [emitOffsets] at h
      | cons _ _ => simp
    rw [this, Nat.add_assoc]; rfl

/-- The chunked assembly (:1321-1347): each chunk's local offsets (all but its closing push) shifted by
`offs[c]`, then the total (:1347). -/
def chunkOffsets {β : Type} : List (List (List β)) → Nat → List Nat
  | [], acc => [acc]
  | ch :: chs, acc => ((emitOffsets ch 0).dropLast.map (· + acc)) ++ chunkOffsets chs (acc + ch.flatten.length)

/-- **`hpar` writes what one chunk writes**: splitting the rows into chunks and assembling gives the
serial offsets, and the columns are the chunks' columns concatenated in order, which is the serial
`flatten`. -/
theorem chunks_eq_serial {β : Type} (chs : List (List (List β))) (acc : Nat) :
    chunkOffsets chs acc = emitOffsets chs.flatten acc := by
  induction chs generalizing acc with
  | nil => rfl
  | cons ch chs ih =>
    simp only [chunkOffsets, List.flatten_cons]
    rw [emitOffsets_append, ih, emitOffsets_shift ch acc]
    congr 1
    rw [List.map_dropLast]

theorem chunks_cols {β : Type} (chs : List (List (List β))) :
    (chs.map List.flatten).flatten = chs.flatten.flatten := by
  induction chs with
  | nil => rfl
  | cons ch chs ih => simp [ih]

end CchCoarsen
