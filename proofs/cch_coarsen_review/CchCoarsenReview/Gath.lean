import CchCoarsenReview.Rows
/-!
# `gath`: capstop's gather of the stopped blocks by appending CSR rows

**Rust** (m1opt.patch @7df8e5887): `build_capstop`, m1_reuse_bench.rs:1327-1355. For each block
`(s, ids)` (the stopped cells' live vertices at their last reused level, `ids` ascending, :1305-1325)
it numbers `rank[x] = k` for the `k`-th id (:1337-1340) and appends, for each `x` in `ids`, the row
`[(off + rank[j], A_s(x, j)) | j in A_s's row x, rank[j] ≠ NONE]` (:1342-1351); then `off += |ids|`.

**Replaced** (:1362-1381): `e = GrB_Matrix_extract(A_s, ids, ids)` (C API: `e(a, b) = A_s(ids[a], ids[b])`),
its tuples shifted by `off`, and one `GrB_Matrix_build_INT32` of all blocks (no duplicates: blocks are
disjoint ranges).

Blocks do not interact: block `b`'s host rows and GraphBLAS tuples both live in rows and columns
`[off_b, off_b + |ids_b|)`. So the gather is proved one block at a time (`gath_eq`).
-/

namespace CchCoarsen

/-- `rank[j]` (:1337-1340): the position of `j` in `ids`, `NONE` (here `none`) elsewhere. -/
def rk (ids : List Nat) (j : Nat) : Option Nat := if j ∈ ids then some (ids.idxOf j) else none

theorem rk_eq_some {ids : List Nat} (hn : ids.Nodup) {j r : Nat} : rk ids j = some r ↔ ids[r]? = some j := by
  unfold rk
  constructor
  · intro h
    by_cases hj : j ∈ ids
    · simp only [hj, ite_true, Option.some.injEq] at h
      subst h
      have hlt := List.idxOf_lt_length_of_mem hj
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    · simp [hj] at h
  · intro h
    have hj : j ∈ ids := List.mem_of_getElem? h
    obtain ⟨hr, hget⟩ := List.getElem?_eq_some_iff.1 h
    have := List.Nodup.idxOf_getElem hn r hr
    rw [hget] at this
    simp [hj, this]

/-- One gathered row: `ids`' member `x`, columns renumbered into the block at `off`. -/
def gathRow (A : Rows Int) (ids : List Nat) (off x : Nat) : List (Nat × Int) :=
  (A x).filterMap (fun e => (rk ids e.1).map (fun r => (off + r, e.2)))

/-- The block's rows in the gathered CSR: row `off + a` is `gathRow` of `ids[a]`. -/
def gathRows (A : Rows Int) (ids : List Nat) (off : Nat) : Rows Int := fun i =>
  if off ≤ i then (match ids[i - off]? with | some x => gathRow A ids off x | none => []) else []

/-- The switch-off result on this block: `G(off + a, off + b) = A_s(ids[a], ids[b])`. -/
def gathGB (A : Mat Int) (ids : List Nat) (off : Nat) : Mat Int := fun i c =>
  if off ≤ i ∧ off ≤ c then
    (match ids[i - off]?, ids[c - off]? with
     | some x, some y => A x y
     | _, _ => none)
  else none

theorem lookup_gathRow {ids : List Nat} (hn : ids.Nodup) {row : List (Nat × Int)} (hk : (row.map (·.1)).Nodup)
    (off c : Nat) :
    (row.filterMap (fun e => (rk ids e.1).map (fun r => (off + r, e.2)))).lookup c =
      if off ≤ c then (match ids[c - off]? with | some y => row.lookup y | none => none) else none := by
  induction row with
  | nil =>
    by_cases h : off ≤ c
    · simp only [h, ite_true]; cases ids[c - off]? <;> rfl
    · simp [h]
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hk
    have hna : row.lookup a = none := lookup_eq_none_of (fun e he hea => hk.1 ⟨e, he, hea⟩)
    rw [List.filterMap_cons]
    cases hr : rk ids a with
    | none =>
      simp only [Option.map_none]
      rw [ih hk.2]
      have ha : a ∉ ids := by intro h; simp [rk, h] at hr
      by_cases h : off ≤ c
      · simp only [h, ite_true]
        cases hy : ids[c - off]? with
        | none => rfl
        | some y =>
          have : y ≠ a := fun e => ha (e ▸ List.mem_of_getElem? hy)
          have hb : (y == a) = false := by simp [this]
          simp [List.lookup_cons, hb]
      · simp [h]
    | some r =>
      have hra := (rk_eq_some hn).1 hr
      simp only [Option.map_some, List.lookup_cons]
      by_cases hc : c = off + r
      · subst hc
        have : (off + r == off + r) = true := by simp
        rw [this]
        simp [hra]
      · have hb : (c == off + r) = false := by simp [hc]
        rw [hb]; simp only
        rw [ih hk.2]
        by_cases h : off ≤ c
        · simp only [h, ite_true]
          cases hy : ids[c - off]? with
          | none => rfl
          | some y =>
            have : y ≠ a := by
              intro e; subst e
              have := (List.Nodup.getElem?_inj (by
                obtain ⟨h1, _⟩ := List.getElem?_eq_some_iff.1 hra; exact h1) hn).1 (hra.trans hy.symm)
              omega
            have hb2 : (y == a) = false := by simp [this]
            simp [hb2]
        · simp [h]

/-- **`gath` = extract, shift and build**, on every coordinate of the block's rows. -/
theorem gath_eq {A : Rows Int} (hA : ∀ i, StrictRow (A i)) {ids : List Nat} (hn : ids.Nodup) (off : Nat)
    (i c : Nat) : den (gathRows A ids off) i c = gathGB (den A) ids off i c := by
  unfold den gathRows gathGB
  by_cases hi : off ≤ i
  · simp only [hi, ite_true, true_and]
    cases hx : ids[i - off]? with
    | none =>
      simp only [List.lookup_nil]
      by_cases hc : off ≤ c <;> simp [hc]
    | some x =>
      simp only [gathRow]
      rw [lookup_gathRow hn (hA x).keys_nodup off c]
      by_cases hc : off ≤ c
      · simp only [hc, ite_true]; cases ids[c - off]? <;> rfl
      · simp [hc]
  · simp [hi]

/-- The gathered rows are sorted with no duplicates, when `ids` is ascending (it is: a filter of
`0..v.n`, :1316). -/
theorem gath_row_strict {A : Rows Int} (hA : ∀ i, StrictRow (A i)) {ids : List Nat}
    (hs : ids.Pairwise (· < ·)) (off x : Nat) : StrictRow (gathRow A ids off x) := by
  have hn : ids.Nodup := hs.imp (fun h => Nat.ne_of_lt h)
  unfold gathRow StrictRow
  rw [List.pairwise_filterMap]
  refine (hA x).imp (fun {e e'} hlt => ?_)
  intro b hb b' hb'
  simp only [Option.map_eq_some_iff] at hb hb'
  obtain ⟨r, hr, rfl⟩ := hb
  obtain ⟨r', hr', rfl⟩ := hb'
  have h1 := (rk_eq_some hn).1 hr
  have h2 := (rk_eq_some hn).1 hr'
  show off + r < off + r'
  obtain ⟨hr1, hg1⟩ := List.getElem?_eq_some_iff.1 h1
  obtain ⟨hr2, hg2⟩ := List.getElem?_eq_some_iff.1 h2
  by_cases hrr : r < r'
  · omega
  · exfalso
    rcases Nat.lt_or_ge r' r with h | h
    · have := (List.pairwise_iff_getElem.1 hs) r' r hr2 hr1 h
      rw [hg1, hg2] at this; omega
    · have : r = r' := by omega
      subst this; rw [hg1] at hg2; rw [hg2] at hlt; omega

end CchCoarsen
