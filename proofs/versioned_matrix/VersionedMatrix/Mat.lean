/-
# A GraphBLAS matrix handle, as the bookkeeping layer sees it

`Mat α` is a `Matrix<T>` (`matrix.rs`): stored entries (key ↦ value, at most
one per key), dimensions, and whether it has pending GraphBLAS work
(`is_synced`). The operations below are the GraphBLAS C calls the Rust wraps,
each defined by its GraphBLAS-spec meaning (these are the AXIOMATISED FFI
rows of `graphblas_wrappers`; they are definitions here, not axioms):

| here | C call | Rust |
| --- | --- | --- |
| `setE`      | `GrB_Matrix_setElement` (marks pending) | `Matrix::set` |
| `removeE`   | `GrB_Matrix_removeElement` (pending)    | `Matrix::remove` |
| `wait`      | `GrB_Matrix_wait`                        | `Matrix::wait` |
| `resizeM`   | `GrB_Matrix_resize` (drops out-of-range, waits) | `Matrix::resize` |
| `transposeM`| `GrB_transpose`                          | `Matrix::transpose` |
| `removeAllM`| `C<!M,replace> = C`                      | `Matrix::remove_all` |
| `maskMult`  | `C<M> = M .* B` (`ANY_PAIR`, no accum, no replace) | `element_wise_multiply` |
| `assignProd`| `GrB_Matrix_assign` of `true` over `rows×cols` | `assign_product_true` |
| `nvals`     | `GrB_Matrix_nvals` (number of stored keys) | `Matrix::nvals` |
-/
namespace VMMat

abbrev Pair := Nat × Nat

structure Mat (α : Type) where
  ents : List (Pair × α)
  nrows : Nat
  ncols : Nat
  synced : Bool

variable {α : Type}

def keys (m : Mat α) : List Pair := m.ents.map (·.1)
def nvals (m : Mat α) : Nat := m.ents.length
def rowsOf (m : Mat α) : List Nat := m.ents.map (·.1.1)
def KeyNodup (m : Mat α) : Prop := (keys m).Nodup

def empty (r c : Nat) : Mat α := ⟨[], r, c, true⟩
def setE (m : Mat α) (p : Pair) (v : α) : Mat α :=
  { m with ents := (p, v) :: m.ents.filter (fun e => e.1 != p), synced := false }
def removeE (m : Mat α) (p : Pair) : Mat α :=
  { m with ents := m.ents.filter (fun e => e.1 != p), synced := false }
def wait (m : Mat α) : Mat α := { m with synced := true }
def resizeM (m : Mat α) (r c : Nat) : Mat α :=
  ⟨m.ents.filter (fun e => decide (e.1.1 < r) && decide (e.1.2 < c)), r, c, true⟩
def transposeM (m : Mat α) : Mat α :=
  ⟨m.ents.map (fun e => ((e.1.2, e.1.1), e.2)), m.ncols, m.nrows, true⟩
def removeAllM (m : Mat α) (mask : List Pair) : Mat α :=
  { m with ents := m.ents.filter (fun e => !(decide (e.1 ∈ mask))), synced := false }
/-- `C<M> = M .* B`: inside the mask the result is `M ∩ B`; outside it is kept. -/
def maskMult (c : Mat Unit) (mask : List Pair) (b : List Pair) : Mat Unit :=
  { c with ents := c.ents.filter (fun e => !(decide (e.1 ∈ mask))) ++
      ((mask.filter (fun p => decide (p ∈ b))).map (fun p => (p, ()))), synced := false }
def product (rows cols : List Nat) : List Pair :=
  rows.flatMap (fun i => cols.map (fun j => (i, j)))
def assignProd (m : Mat Unit) (rows cols : List Nat) : Mat Unit :=
  { m with ents := m.ents.filter (fun e => !(decide (e.1 ∈ product rows cols))) ++
      (product rows cols).map (fun p => (p, ())), synced := false }

theorem mem_rowsOf {m : Mat α} {r : Nat} : r ∈ rowsOf m ↔ ∃ e ∈ m.ents, e.1.1 = r := by
  simp [rowsOf]

theorem rowsOf_filter_sub {m : Mat α} {f : Pair × α → Bool} {r : Nat}
    (h : r ∈ (m.ents.filter f).map (·.1.1)) : r ∈ rowsOf m := by
  obtain ⟨e, he, rfl⟩ := List.mem_map.1 h
  exact List.mem_map.2 ⟨e, (List.mem_filter.1 he).1, rfl⟩

theorem nvals_setE_absent {m : Mat α} (_hn : KeyNodup m) {p : Pair} (hp : p ∉ keys m) (v : α) :
    nvals (setE m p v) = nvals m + 1 := by
  simp only [nvals, setE, List.length_cons]
  rw [List.filter_eq_self.2]
  intro e he; simp only [bne_iff_ne, ne_eq]; intro h
  exact hp (h ▸ List.mem_map_of_mem he)

theorem length_filter_ne_of_nodup : ∀ {l : List (Pair × α)} {p : Pair}, (l.map (·.1)).Nodup →
    p ∈ l.map (·.1) → (l.filter (fun e => e.1 != p)).length + 1 = l.length
  | [], _, _, h => by simp at h
  | e :: es, p, hn, hm => by
    simp only [List.map_cons, List.nodup_cons] at hn
    simp only [List.map_cons, List.mem_cons] at hm
    by_cases he : e.1 = p
    · subst he
      have : es.filter (fun x => x.1 != e.1) = es := List.filter_eq_self.2 (by
        intro x hx; simp; intro h; exact hn.1 (h ▸ List.mem_map_of_mem hx))
      simp [this]
    · rcases hm with h | h
      · exact absurd h.symm he
      · have := length_filter_ne_of_nodup hn.2 h
        simp [he]; omega

theorem nvals_removeE_present {m : Mat α} (hn : KeyNodup m) {p : Pair} (hp : p ∈ keys m) :
    nvals (removeE m p) + 1 = nvals m := length_filter_ne_of_nodup hn hp

theorem keys_setE {m : Mat α} {p q : Pair} {v : α} : q ∈ keys (setE m p v) ↔ q = p ∨ q ∈ keys m := by
  simp only [keys, setE, List.map_cons, List.mem_cons, List.mem_map, List.mem_filter]
  constructor
  · rintro (h | ⟨e, ⟨he, _⟩, rfl⟩); exact Or.inl h; exact Or.inr ⟨e, he, rfl⟩
  · rintro (h | ⟨e, he, rfl⟩)
    · exact Or.inl h
    · by_cases h : e.1 = p
      · exact Or.inl h
      · exact Or.inr ⟨e, ⟨he, by simpa using h⟩, rfl⟩

theorem keyNodup_setE {m : Mat α} (h : KeyNodup m) (p : Pair) (v : α) : KeyNodup (setE m p v) := by
  unfold KeyNodup keys setE at *
  simp only [List.map_cons, List.nodup_cons]
  refine ⟨?_, ?_⟩
  · simp
  · exact List.Nodup.sublist (List.Sublist.map _ List.filter_sublist) h

end VMMat
