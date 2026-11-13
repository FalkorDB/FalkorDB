import CchCoarsenReview.Spec
/-!
# A CSR at row granularity, and the host passes' merge of equal keys

`Csr::new` (m1_coarsen_bench.rs:363-424 (m1opt.patch @7df8e5887)) unloads a matrix's arrays after a
`GrB_wait`, so (GraphBLAS user guide, "sparse by row, not jumbled") row `i` is
`Aj[Ap[i]..Ap[i+1]]`, columns strictly ascending, with `Ax` (or the iso value) beside them. Every
host pass reads a row as `for kk in ap[i]..ap[i+1] { (aj[kk], ar.val(kk)) }`, which is the list
`R i` below. `Emit.lean` links the flat `(Ap, Aj, Ax)` arrays to these row lists.

`den R` is the matrix a CSR denotes: `A(i,j)` is the value beside column `j` in row `i`.
-/

namespace CchCoarsen

/-- A CSR, row by row: row `i` is the list of `(column, value)` in the order stored. -/
abbrev Rows (α : Type) := Nat → List (Nat × α)

/-- The matrix a CSR denotes. -/
def den {α : Type} (R : Rows α) : Mat α := fun i j => (R i).lookup j

/-- Columns strictly ascending (sorted, no duplicates): the unload after `GrB_wait`. -/
def StrictRow {α : Type} (row : List (Nat × α)) : Prop := row.Pairwise (fun a b => a.1 < b.1)

/-- Keys ascending, duplicates allowed: what `sort_unstable_by_key(|e| e.0)` leaves. -/
def KeySorted {α : Type} (row : List (Nat × α)) : Prop := row.Pairwise (fun a b => a.1 ≤ b.1)

theorem StrictRow.keys_nodup {α : Type} {row : List (Nat × α)} (h : StrictRow row) :
    (row.map (·.1)).Nodup := by
  rw [List.nodup_iff_pairwise_ne, List.pairwise_map]
  exact h.imp (fun h => Nat.ne_of_lt h)

theorem lookup_isSome_iff {α : Type} (row : List (Nat × α)) (j : Nat) :
    (row.lookup j).isSome ↔ ∃ v, (j, v) ∈ row := by
  induction row with
  | nil => simp
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    rw [List.lookup_cons]
    by_cases h : j = a
    · subst h; simp
    · have : (j == a) = false := by simp [h]
      rw [this]; simp only [ih, List.mem_cons, Prod.mk.injEq]
      constructor
      · rintro ⟨v, hv⟩; exact ⟨v, Or.inr hv⟩
      · rintro ⟨v, hv | hv⟩
        · exact absurd hv.1 h
        · exact ⟨v, hv⟩

theorem lookup_eq_none_of {α : Type} {row : List (Nat × α)} {j : Nat} (h : ∀ e ∈ row, e.1 ≠ j) :
    row.lookup j = none := by
  cases hl : row.lookup j with
  | none => rfl
  | some v =>
    obtain ⟨w, hw⟩ := (lookup_isSome_iff row j).1 (by simp [hl])
    exact absurd rfl (h _ hw)

theorem lookup_of_mem {α : Type} {row : List (Nat × α)} (hn : (row.map (·.1)).Nodup) {j : Nat} {v : α}
    (hm : (j, v) ∈ row) : row.lookup j = some v := by
  induction row with
  | nil => simp at hm
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hn
    rw [List.lookup_cons]
    rcases List.mem_cons.1 hm with h | h
    · cases h; simp
    · have hja : j ≠ a := fun hja => hn.1 ⟨(j, v), h, by simp [hja]⟩
      have : (j == a) = false := by simp [hja]
      rw [this]; exact ih hn.2 h

/-- The value of a stored row as a function of the column, summed against `g` over the inner
dimension, is the sum over the row's entries: `Σ_{t<n} A(i,t) g(t) = Σ_{(t,v) ∈ row i} v g(t)`. -/
theorem lsum_range_lookup {row : List (Nat × Int)} (hn : (row.map (·.1)).Nodup) {n : Nat}
    (hb : ∀ e ∈ row, e.1 < n) (g : Nat → Int) :
    lsum (List.range n) (fun t => (row.lookup t).getD 0 * g t) = lsum row (fun e => e.2 * g e.1) := by
  induction row with
  | nil => simp [lsum_zero]
  | cons e row ih =>
    obtain ⟨a, b⟩ := e
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hn
    have hna : row.lookup a = none :=
      lookup_eq_none_of (fun e he hea => hn.1 ⟨e, he, hea⟩)
    have hsplit : ∀ t, (((a, b) :: row).lookup t).getD 0 * g t =
        (if t = a then b * g t else 0) + (row.lookup t).getD 0 * g t := by
      intro t
      rw [List.lookup_cons]
      by_cases h : t = a
      · subst h; simp [hna]
      · have : (t == a) = false := by simp [h]
        rw [this]; simp [h]
    rw [lsum_congr (fun t _ => hsplit t), lsum_add,
      lsum_single List.nodup_range (List.mem_range.2 (hb (a, b) (by simp))),
      ih hn.2 (fun e he => hb e (by simp [he]))]
    simp

/-! ## `grp`: the two `while` loops that merge equal keys

`host_contract` (m1_coarsen_bench.rs:1418-1431) and `b2_splice` (m1_reuse_bench.rs:874-887):
```
let mut q = 0;
while q < tmp.len() {
    let (cc, mut sm) = tmp[q]; q += 1;
    while q < tmp.len() && tmp[q].0 == cc { sm += tmp[q].1; q += 1; }
    cols.push(cc); vals.push(sm);
}
```
As one left-to-right scan, the open group `(cc, sm)` is the state: an equal key adds to it, another
key closes it and opens the next. -/

def grp : Option (Nat × Int) → List (Nat × Int) → List (Nat × Int)
  | none, [] => []
  | some g, [] => [g]
  | none, e :: l => grp (some e) l
  | some (c, s), e :: l => if e.1 = c then grp (some (c, s + e.2)) l else (c, s) :: grp (some e) l

/-- The sum of the values with key `k`. -/
def ksum (l : List (Nat × Int)) (k : Nat) : Int := lsum l (fun e => if e.1 = k then e.2 else 0)

theorem ksum_cons (a : Nat) (b : Int) (l : List (Nat × Int)) (k : Nat) :
    ksum ((a, b) :: l) k = (if a = k then b else 0) + ksum l k := rfl

theorem keys_cons (a : Nat) (b : Int) (l : List (Nat × Int)) (k : Nat) :
    k ∈ ((a, b) :: l).map (·.1) ↔ k = a ∨ k ∈ l.map (·.1) := by simp

theorem grp_some_spec (l : List (Nat × Int)) (hs : KeySorted l) (c : Nat) (s : Int)
    (hc : ∀ e ∈ l, c ≤ e.1) :
    (∀ x ∈ grp (some (c, s)) l, c ≤ x.1) ∧ StrictRow (grp (some (c, s)) l) ∧
    ∀ k, (grp (some (c, s)) l).lookup k =
      if k = c then some (s + ksum l c) else if k ∈ l.map (·.1) then some (ksum l k) else none := by
  induction l generalizing c s with
  | nil =>
    refine ⟨by simp [grp], by simp [grp, StrictRow], fun k => ?_⟩
    by_cases h : k = c
    · subst h; simp [grp, ksum]
    · simp [grp, h]
  | cons e l ih =>
    obtain ⟨a, b⟩ := e
    simp only [KeySorted, List.pairwise_cons] at hs
    have hle : ∀ x ∈ l, a ≤ x.1 := fun x hx => hs.1 x hx
    have hca : c ≤ a := hc (a, b) (by simp)
    by_cases hac : a = c
    · subst hac
      obtain ⟨h1, h2, h3⟩ := ih hs.2 a (s + b) hle
      have hg : grp (some (a, s)) ((a, b) :: l) = grp (some (a, s + b)) l := by simp [grp]
      rw [hg]
      refine ⟨h1, h2, fun k => ?_⟩
      rw [h3 k, ksum_cons]
      by_cases hk : k = a
      · subst hk; simp [Int.add_assoc]
      · have hm : k ∈ ((k, b) :: l).map (·.1) ∨ True := Or.inr trivial
        clear hm
        simp only [hk, ite_false, keys_cons, false_or, ksum_cons]
        have : a ≠ k := Ne.symm hk
        simp [this]
    · have hlt : c < a := Nat.lt_of_le_of_ne hca (Ne.symm hac)
      obtain ⟨h1, h2, h3⟩ := ih hs.2 a b hle
      have hg : grp (some (c, s)) ((a, b) :: l) = (c, s) :: grp (some (a, b)) l := by
        simp [grp, hac]
      rw [hg]
      have hcl : ksum l c = 0 :=
        lsum_zero (fun x hx => by
          have := hle x hx
          have : x.1 ≠ c := by omega
          simp [this])
      refine ⟨?_, ?_, fun k => ?_⟩
      · intro x hx
        rcases List.mem_cons.1 hx with h | h
        · subst h; exact Nat.le_refl _
        · exact Nat.le_trans (Nat.le_of_lt hlt) (h1 x h)
      · simp only [StrictRow, List.pairwise_cons]
        exact ⟨fun x hx => Nat.lt_of_lt_of_le hlt (h1 x hx), h2⟩
      · rw [List.lookup_cons]
        by_cases hk : k = c
        · subst hk; simp [ksum_cons, hac, hcl]
        · have hkb : (k == c) = false := by simp [hk]
          rw [hkb]; simp only [hk, ite_false]
          rw [h3 k]
          simp only [keys_cons, ksum_cons]
          by_cases hka : k = a
          · subst hka; simp
          · have : a ≠ k := Ne.symm hka
            by_cases hm : k ∈ l.map (·.1)
            · simp [hm, this, hka]
            · simp [hm, hka]

/-- **The merge of a key-sorted list**: one entry per key, keys strictly ascending, each holding the
sum of that key's values. -/
theorem grp_spec (l : List (Nat × Int)) (hs : KeySorted l) :
    StrictRow (grp none l) ∧
    ∀ k, (grp none l).lookup k = if k ∈ l.map (·.1) then some (ksum l k) else none := by
  cases l with
  | nil => exact ⟨by simp [grp, StrictRow], fun k => by simp [grp]⟩
  | cons e l =>
    obtain ⟨a, b⟩ := e
    have hs' := hs
    simp only [KeySorted, List.pairwise_cons] at hs'
    obtain ⟨_, h2, h3⟩ := grp_some_spec l hs'.2 a b (fun x hx => hs'.1 x hx)
    refine ⟨by simpa [grp] using h2, fun k => ?_⟩
    simp only [grp]
    rw [h3 k]
    by_cases hk : k = a
    · subst hk; simp [ksum]
    · have : a ≠ k := Ne.symm hk
      by_cases hm : k ∈ l.map (·.1)
      · simp [hm, ksum, this, hk]
      · simp [hm, hk]

theorem grp_keys (l : List (Nat × Int)) (hs : KeySorted l) (x : Nat × Int) (hx : x ∈ grp none l) :
    x.1 ∈ l.map (·.1) := by
  obtain ⟨hst, hl⟩ := grp_spec l hs
  have hlk := lookup_of_mem hst.keys_nodup hx
  rw [hl x.1] at hlk
  by_cases h : x.1 ∈ l.map (·.1)
  · exact h
  · simp [h] at hlk

/-- Key sums and key membership do not see the order: `sort_unstable_by_key` may put equal keys in
any order. -/
theorem ksum_perm {l₁ l₂ : List (Nat × Int)} (h : l₁.Perm l₂) (k : Nat) : ksum l₁ k = ksum l₂ k :=
  lsum_perm h _

theorem keys_perm {l₁ l₂ : List (Nat × Int)} (h : l₁.Perm l₂) (k : Nat) :
    k ∈ l₁.map (·.1) ↔ k ∈ l₂.map (·.1) := (h.map _).mem_iff

/-- The contract of `slice::sort_unstable_by_key(|e| e.0)` (Rust std): a permutation of the input,
keys ascending, equal keys in any order. -/
def SortSpec (srt : List (Nat × Int) → List (Nat × Int)) : Prop :=
  ∀ l, (srt l).Perm l ∧ KeySorted (srt l)

/-- The contract is satisfiable (so the theorems below are not vacuous): core's merge sort meets it. -/
theorem sortSpec_mergeSort : SortSpec (fun l => l.mergeSort (fun a b => decide (a.1 ≤ b.1))) := by
  intro l
  refine ⟨List.mergeSort_perm _ _, ?_⟩
  have := List.pairwise_mergeSort (le := fun (a b : Nat × Int) => decide (a.1 ≤ b.1))
    (fun a b c h1 h2 => by simp at h1 h2 ⊢; omega)
    (fun a b => by simp; omega) l
  simpa [KeySorted] using this

/-- `sort_unstable_by_key` then the merge, as a function of the unsorted list: the sorted order does
not matter. -/
theorem sortGrp_spec {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt) (l : List (Nat × Int)) :
    StrictRow (grp none (srt l)) ∧
    ∀ k, (grp none (srt l)).lookup k = if k ∈ l.map (·.1) then some (ksum l k) else none := by
  obtain ⟨hp, hs⟩ := hS l
  obtain ⟨h1, h2⟩ := grp_spec _ hs
  refine ⟨h1, fun k => ?_⟩
  rw [h2 k, ksum_perm hp k]
  by_cases h : k ∈ l.map (·.1)
  · have h' := (keys_perm hp k).2 h
    simp [h, h']
  · have h' : k ∉ (srt l).map (·.1) := fun h' => h ((keys_perm hp k).1 h')
    simp [h, h']

theorem sortGrp_keys {srt : List (Nat × Int) → List (Nat × Int)} (hS : SortSpec srt) (l : List (Nat × Int))
    (x : Nat × Int) (hx : x ∈ grp none (srt l)) : x.1 ∈ l.map (·.1) :=
  (keys_perm (hS l).1 x.1).1 (grp_keys _ (hS l).2 x hx)

theorem grp_length_le (o : Option (Nat × Int)) (l : List (Nat × Int)) :
    (grp o l).length ≤ l.length + (if o.isSome then 1 else 0) := by
  induction l generalizing o with
  | nil => cases o <;> simp [grp]
  | cons e l ih =>
    cases o with
    | none => have := ih (some e); simp [grp] at this ⊢; omega
    | some g =>
      obtain ⟨c, s⟩ := g
      simp only [grp]
      by_cases h : e.1 = c
      · simp only [h, ite_true]; have := ih (some (c, s + e.2)); simp at this ⊢; omega
      · simp only [h, ite_false, List.length_cons]; have := ih (some e); simp at this ⊢; omega

/-- The merge never emits more entries than it reads (`Overflow.lean`: offsets stay under `nnz(A)`). -/
theorem grp_none_length_le (l : List (Nat × Int)) : (grp none l).length ≤ l.length := by
  have := grp_length_le none l; simpa using this

end CchCoarsen
