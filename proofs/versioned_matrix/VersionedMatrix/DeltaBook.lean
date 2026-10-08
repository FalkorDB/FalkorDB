import VersionedMatrix.Mat
import VersionedMatrix.RowFilter
import VersionedMatrix.Policy
/-
# `Delta<T>`: one delta layer plus its bookkeeping (`versioned_matrix.rs:297-641`)

| here | there |
| --- | --- |
| `Delta`        | `struct Delta<T>` (:297): layer, `count`, `tx_nvals`, `fold`, `rows` |
| `ofLayer`      | `Delta::new` (:347) |
| `relayer`/`clone` | `relayer` (:375) / `impl Clone` (:331) |
| `deref`        | `impl Deref` (:339) |
| `transposed`   | `transposed` (:365) |
| `newVersion`   | `new_version` (:392) |
| `count`/`resync` | `count` (:407) / `resync` (:413) |
| `latch`/`foldDecision`/`folding`/`takeFold` | :421 / :434 / :446 / :452 |
| `clear`/`replace`/`resize` | :459 / :479 / :488 |
| `layerMut`/`layerMutRow` | :503 / :514 (the mutation is passed as `f`) |
| `mayHoldRows`  | `may_hold_rows` (:524) |
| `erase`        | `erase` (:535) |
| `insertB`/`insertProduct`/`tombstoneMasked` | `impl Delta<bool>` :554 / :574 / :589 |
| `removeAll`    | `remove_all` (:603) |
| `insertV`      | `impl Delta<u64>::insert` (:621) |

`Atomic*` fields are plain values: every access is `Relaxed` under the
single-writer discipline (`Cow.lean`), so only their values matter.
`count` is a `u64`; `+= 1` never reaches `2^64` (it is bounded by the
number of operations), `saturating_sub` is `Nat` subtraction.

Main results: `RowsOk` (the row filter over-approximates the stored rows)
is preserved by **every** method (`*_rowsOk`), so `mayHoldRows_sound`
holds in every reachable state; `count` is exact after `resync`/`clear`/
`ofLayer`, and stays exact through `insert`/`erase` of absent/present keys;
latching is monotone; a fresh version has `tx_added = 0`, so no policy can
newly decide to fold it (`fresh_version_decision`).
-/
namespace VMDelta
open VMMat VMRowFilter VMPolicy

structure Delta (α : Type) where
  layer : Mat α
  count : Nat
  tx : Nat
  fold : Bool
  rows : RF

variable {α : Type}

/-- The safety invariant of the row filter. -/
def RowsOk (d : Delta α) : Prop := Sound d.rows (rowsOf d.layer)
/-- `count` is exact. -/
def Exact (d : Delta α) : Prop := d.count = nvals d.layer

/-- `Delta::new` (:347); `into_hyper` changes the format only. -/
def ofLayer (m : Mat α) : Delta α :=
  ⟨m, nvals m, 0, false, if nvals m = 0 then .empty else .unknown⟩

def relayer (d : Delta α) (l : Mat α) : Delta α := { d with layer := l }
def clone (d : Delta α) : Delta α := relayer d d.layer
def deref (d : Delta α) : Mat α := d.layer
def transposed (d : Delta α) : Delta α := { relayer d (transposeM d.layer) with rows := .unknown }
def newVersion (d : Delta α) (f : Bool) : Delta α := { d with tx := d.count, fold := f }
def resync (d : Delta α) : Delta α :=
  { d with layer := wait d.layer, count := nvals (wait d.layer) }
def latch (d : Delta α) (b : Bool) : Delta α := if b then { d with fold := true } else d
def foldDecision (d : Delta α) (policy : Nat → Nat → Nat → Bool) (base : Nat) : Bool :=
  d.fold || policy d.count (d.count - d.tx) base
def folding (d : Delta α) : Bool := d.fold
def takeFold (d : Delta α) : Bool × Delta α :=
  (d.fold && decide (nvals d.layer > 0), { d with fold := false })
def clear (d : Delta α) (r c : Nat) : Delta α := ⟨empty r c, 0, 0, false, .empty⟩
def replace (d : Delta α) (l : Mat α) : Delta α := { d with layer := l }
def layerMut (d : Delta α) (f : Mat α → Mat α) : Delta α := { d with rows := .unknown, layer := f d.layer }
def layerMutRow (d : Delta α) (i : Nat) (f : Mat α → Mat α) : Delta α :=
  { d with rows := add d.rows (BitVec.ofNat 64 i), layer := f d.layer }
/-- `resize` (:488): `layer_mut` invalidates, then the saved filter is restored. -/
def resize (d : Delta α) (r c : Nat) : Delta α :=
  { layerMut d (fun m => resizeM m r c) with rows := d.rows }
def mayHoldRows (d : Delta α) (lo hi : Nat) : Bool := mayHold d.rows lo hi
def erase (d : Delta α) (i j : Nat) : Delta α :=
  { layerMutRow d i (fun m => removeE m (i, j)) with count := d.count - 1 }
def insertB (d : Delta Unit) (i j : Nat) : Delta Unit :=
  { layerMutRow d i (fun m => setE m (i, j) ()) with count := d.count + 1 }
def insertProduct (d : Delta Unit) (rs cs : List Nat) : Delta Unit :=
  { d with rows := rs.foldl (fun F i => add F (BitVec.ofNat 64 i)) d.rows,
           layer := assignProd d.layer rs cs, count := d.count + rs.length * cs.length }
def tombstoneMasked (d : Delta Unit) (mask base : List Pair) : Delta Unit :=
  resync (layerMut d (fun m => maskMult m mask base))
def removeAll (d : Delta α) (mask : List Pair) : Delta α :=
  resync (layerMut d (fun m => removeAllM m mask))
def insertV (d : Delta α) (i j : Nat) (v : α) : Delta α :=
  { layerMut d (fun m => setE m (i, j) v) with count := d.count + 1 }

/-! ## The row filter is sound in every state -/

theorem ofLayer_rowsOk (m : Mat α) : RowsOk (ofLayer m) := by
  unfold RowsOk ofLayer; split
  · next h =>
    simp [nvals] at h; simp [rowsOf, h]; exact sound_empty
  · exact sound_unknown _

theorem rowsOf_transpose_irrelevant (d : Delta α) : RowsOk (transposed d) := sound_unknown _
theorem relayer_rowsOk {d : Delta α} (h : RowsOk d) {l : Mat α} (hl : ∀ r ∈ rowsOf l, r ∈ rowsOf d.layer) :
    RowsOk (relayer d l) := sound_sub h hl
theorem clone_eq (d : Delta α) : clone d = d := rfl
theorem clone_rowsOk {d : Delta α} (h : RowsOk d) : RowsOk (clone d) := h
theorem newVersion_rowsOk {d : Delta α} (h : RowsOk d) (f : Bool) : RowsOk (newVersion d f) := h
theorem resync_rowsOk {d : Delta α} (h : RowsOk d) : RowsOk (resync d) := h
theorem latch_rowsOk {d : Delta α} (h : RowsOk d) (b : Bool) : RowsOk (latch d b) := by
  unfold latch; split <;> exact h
theorem takeFold_rowsOk {d : Delta α} (h : RowsOk d) : RowsOk (takeFold d).2 := h
theorem clear_rowsOk (d : Delta α) (r c : Nat) : RowsOk (clear d r c) := by
  simp [RowsOk, clear, empty, rowsOf]; exact sound_empty
theorem replace_rowsOk {d : Delta α} (h : RowsOk d) {l : Mat α}
    (hl : ∀ r ∈ rowsOf l, r ∈ rowsOf d.layer) : RowsOk (replace d l) := sound_sub h hl
theorem layerMut_rowsOk (d : Delta α) (f : Mat α → Mat α) : RowsOk (layerMut d f) := sound_unknown _
theorem resize_rowsOk {d : Delta α} (h : RowsOk d) (r c : Nat) : RowsOk (resize d r c) :=
  sound_sub h (fun _ hx => rowsOf_filter_sub hx)
theorem erase_rowsOk {d : Delta α} (h : RowsOk d) (i j : Nat) : RowsOk (erase d i j) := by
  have h2 := sound_add h i
  exact sound_sub h2 (fun r hr => List.mem_cons_of_mem _ (rowsOf_filter_sub hr))
theorem insertB_rowsOk {d : Delta Unit} (h : RowsOk d) (i j : Nat) : RowsOk (insertB d i j) := by
  have h2 := sound_add h i
  refine sound_sub h2 ?_
  intro r hr
  simp only [insertB, layerMutRow, setE, rowsOf, List.map_cons, List.mem_cons] at hr
  rcases hr with rfl | hr
  · exact List.mem_cons_self
  · exact List.mem_cons_of_mem _ (rowsOf_filter_sub hr)
theorem foldl_add_sound : ∀ (rs : List Nat) {F : RF} {rows : List Nat}, Sound F rows →
    Sound (rs.foldl (fun F i => add F (BitVec.ofNat 64 i)) F) (rs ++ rows)
  | [], _, _, h => h
  | r :: rs, F, rows, h => by
    have := foldl_add_sound rs (sound_add h r)
    refine sound_sub this ?_
    intro x hx
    simp only [List.mem_append, List.mem_cons] at hx ⊢
    rcases hx with (hx | hx) | hx <;> simp [hx]
theorem insertProduct_rowsOk {d : Delta Unit} (h : RowsOk d) (rs cs : List Nat) :
    RowsOk (insertProduct d rs cs) := by
  refine sound_sub (foldl_add_sound rs h) ?_
  intro r hr
  simp only [insertProduct, assignProd, rowsOf, List.map_append, List.mem_append] at hr
  rcases hr with hr | hr
  · exact List.mem_append_right _ (rowsOf_filter_sub hr)
  · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hr
    simp only [product, List.mem_flatMap, List.mem_map] at hq
    rcases hq with ⟨q', ⟨a, ha, b, _, he⟩, hq'⟩
    subst hq'; subst he; exact List.mem_append_left _ ha
theorem tombstoneMasked_rowsOk (d : Delta Unit) (mask base : List Pair) :
    RowsOk (tombstoneMasked d mask base) := sound_unknown _
theorem removeAll_rowsOk (d : Delta α) (mask : List Pair) : RowsOk (removeAll d mask) := sound_unknown _
theorem insertV_rowsOk (d : Delta α) (i j : Nat) (v : α) : RowsOk (insertV d i j v) := sound_unknown _

/-- What every reader relies on: a stored row in range makes the layer attach. -/
theorem mayHoldRows_sound {d : Delta α} (h : RowsOk d) {p : Pair} {v : α} (hp : (p, v) ∈ d.layer.ents)
    {lo hi : Nat} (h1 : lo ≤ p.1) (h2 : p.1 ≤ hi) : mayHoldRows d lo hi = true :=
  mayHold_sound (h p.1 (List.mem_map.2 ⟨(p, v), hp, rfl⟩)) h1 h2

/-! ## The approximate counter -/

theorem ofLayer_exact (m : Mat α) : Exact (ofLayer m) := rfl
theorem resync_exact (d : Delta α) : Exact (resync d) := rfl
theorem clear_exact (d : Delta α) (r c : Nat) : Exact (clear d r c) := rfl
theorem resync_count (d : Delta α) : (resync d).count = nvals d.layer := rfl
theorem insertB_exact {d : Delta Unit} (h : Exact d) (hn : KeyNodup d.layer) {i j : Nat}
    (ha : (i, j) ∉ keys d.layer) : Exact (insertB d i j) := by
  simp only [Exact, insertB, layerMutRow] at *
  rw [nvals_setE_absent hn ha, h]
theorem erase_exact {d : Delta α} (h : Exact d) (hn : KeyNodup d.layer) {i j : Nat}
    (hp : (i, j) ∈ keys d.layer) : Exact (erase d i j) := by
  have := nvals_removeE_present hn hp
  simp only [Exact, erase, layerMutRow] at *; omega
/-- Erasing an absent key under-counts by one (saturating): the drift `resync` bounds. -/
theorem erase_absent_drift (d : Delta α) (i j : Nat) : (erase d i j).count = d.count - 1 := rfl
theorem insertProduct_count (d : Delta Unit) (rs cs : List Nat) :
    (insertProduct d rs cs).count = d.count + rs.length * cs.length := rfl
theorem tombstoneMasked_exact (d : Delta Unit) (mask base : List Pair) :
    Exact (tombstoneMasked d mask base) := rfl
theorem removeAll_exact (d : Delta α) (mask : List Pair) : Exact (removeAll d mask) := rfl
theorem resize_count (d : Delta α) (r c : Nat) : (resize d r c).count = d.count := rfl
theorem replace_bookkeeping (d : Delta α) (l : Mat α) :
    (replace d l).count = d.count ∧ (replace d l).tx = d.tx ∧ (replace d l).fold = d.fold := ⟨rfl, rfl, rfl⟩

/-! ## Layer contents -/

theorem deref_eq (d : Delta α) : deref d = d.layer := rfl
theorem insertB_layer (d : Delta Unit) (i j : Nat) : (insertB d i j).layer = setE d.layer (i, j) () := rfl
theorem erase_layer (d : Delta α) (i j : Nat) : (erase d i j).layer = removeE d.layer (i, j) := rfl
theorem insertV_layer (d : Delta α) (i j : Nat) (v : α) : (insertV d i j v).layer = setE d.layer (i, j) v := rfl
theorem insertProduct_keys (d : Delta Unit) (rs cs : List Nat) (p : Pair) :
    p ∈ keys (insertProduct d rs cs).layer ↔ p ∈ product rs cs ∨ p ∈ keys d.layer := by
  simp only [insertProduct, assignProd, keys, List.map_append, List.mem_append, List.map_map]
  constructor
  · rintro (h | h)
    · obtain ⟨e, he, rfl⟩ := List.mem_map.1 h; exact Or.inr (List.mem_map_of_mem (List.mem_filter.1 he).1)
    · obtain ⟨q, hq, rfl⟩ := List.mem_map.1 h; exact Or.inl hq
  · rintro (h | h)
    · exact Or.inr (List.mem_map.2 ⟨p, h, rfl⟩)
    · by_cases hp : p ∈ product rs cs
      · exact Or.inr (List.mem_map.2 ⟨p, hp, rfl⟩)
      · obtain ⟨e, he, rfl⟩ := List.mem_map.1 h
        exact Or.inl (List.mem_map_of_mem (List.mem_filter.2 ⟨he, by simpa using hp⟩))
theorem tombstoneMasked_keys (d : Delta Unit) (mask base : List Pair) (p : Pair) :
    p ∈ keys (tombstoneMasked d mask base).layer ↔
      (p ∈ keys d.layer ∧ p ∉ mask) ∨ (p ∈ mask ∧ p ∈ base) := by
  simp only [tombstoneMasked, resync, layerMut, wait, maskMult, keys, List.map_append,
    List.mem_append, List.map_map]
  constructor
  · rintro (h | h)
    · obtain ⟨e, he, rfl⟩ := List.mem_map.1 h
      have := List.mem_filter.1 he; exact Or.inl ⟨List.mem_map_of_mem this.1, by simpa using this.2⟩
    · simp at h; exact Or.inr h
  · rintro (⟨h1, h2⟩ | h)
    · obtain ⟨e, he, rfl⟩ := List.mem_map.1 h1
      exact Or.inl (List.mem_map_of_mem (List.mem_filter.2 ⟨he, by simpa using h2⟩))
    · exact Or.inr (by simpa using h)
theorem removeAll_keys (d : Delta α) (mask : List Pair) (p : Pair) :
    p ∈ keys (removeAll d mask).layer ↔ p ∈ keys d.layer ∧ p ∉ mask := by
  simp only [removeAll, resync, layerMut, wait, removeAllM, keys]
  constructor
  · intro h; obtain ⟨e, he, rfl⟩ := List.mem_map.1 h
    have := List.mem_filter.1 he; exact ⟨List.mem_map_of_mem this.1, by simpa using this.2⟩
  · rintro ⟨h1, h2⟩; obtain ⟨e, he, rfl⟩ := List.mem_map.1 h1
    exact List.mem_map_of_mem (List.mem_filter.2 ⟨he, by simpa using h2⟩)
theorem transposed_keys (d : Delta α) (i j : Nat) :
    (i, j) ∈ keys (transposed d).layer ↔ (j, i) ∈ keys d.layer := by
  simp only [transposed, relayer, transposeM, keys, List.map_map, List.mem_map, Function.comp_def]
  constructor
  · rintro ⟨e, he, h⟩; simp only [Prod.mk.injEq] at h; obtain ⟨h1, h2⟩ := h
    exact ⟨e, he, Prod.ext h2 h1⟩
  · rintro ⟨e, he, h⟩; exact ⟨e, he, by rw [h]⟩
theorem transposed_bookkeeping (d : Delta α) :
    (transposed d).count = d.count ∧ (transposed d).tx = d.tx ∧ (transposed d).fold = d.fold :=
  ⟨rfl, rfl, rfl⟩

/-! ## Fold latch -/

theorem latch_fold (d : Delta α) (b : Bool) : (latch d b).fold = (d.fold || b) := by
  unfold latch; cases b <;> simp
theorem latch_monotone (d : Delta α) (b : Bool) (h : d.fold = true) : (latch d b).fold = true := by
  rw [latch_fold, h]; rfl
theorem latch_layer (d : Delta α) (b : Bool) : (latch d b).layer = d.layer := by
  unfold latch; split <;> rfl
theorem foldDecision_latched (d : Delta α) (p : Nat → Nat → Nat → Bool) (b : Nat) (h : d.fold = true) :
    foldDecision d p b = true := by simp [foldDecision, h]
theorem folding_eq (d : Delta α) : folding d = d.fold := rfl
theorem takeFold_spec (d : Delta α) :
    (takeFold d).1 = (d.fold && decide (0 < nvals d.layer)) ∧ (takeFold d).2.fold = false ∧
    (takeFold d).2.layer = d.layer := ⟨rfl, rfl, rfl⟩
theorem newVersion_spec (d : Delta α) (f : Bool) :
    (newVersion d f).tx = d.count ∧ (newVersion d f).count = d.count ∧
    (newVersion d f).fold = f ∧ (newVersion d f).layer = d.layer := ⟨rfl, rfl, rfl, rfl⟩
/-- A version that has added nothing yet cannot newly decide to fold. -/
theorem fresh_version_decision (d : Delta α) (f : Bool) (k base : Nat) :
    foldDecision (newVersion d f) (fun a b c => foldBalance a b c k) base = f := by
  simp [foldDecision, newVersion, foldBalance_readOnly]
theorem clear_spec (d : Delta α) (r c : Nat) :
    (clear d r c).layer.ents = [] ∧ (clear d r c).count = 0 ∧ (clear d r c).tx = 0 ∧
    (clear d r c).fold = false := ⟨rfl, rfl, rfl, rfl⟩
theorem ofLayer_spec (m : Mat α) :
    (ofLayer m).tx = 0 ∧ (ofLayer m).fold = false ∧ (ofLayer m).layer = m := ⟨rfl, rfl, rfl⟩

end VMDelta
