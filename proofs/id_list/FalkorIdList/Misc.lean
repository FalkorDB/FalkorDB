import FalkorIdList.PushT
/-!
# `IdList`'s accessors and trait impls (`id_list.rs:804-1276`)

`IdList` is `IdListPush.St` (`segs`, `len`, `run.start`, `run.desc`).
-/
namespace IdListMisc
open IdListPush IdListWire IdListCost Seg

/-- `IdList::is_empty` (`:1114`). -/
def isEmpty (st : St) : Bool := st.len == 0

/-- `IdList::count` (`:1136`): `u32::try_from(len).expect(..)`; `none` = the panic. -/
def count (st : St) : Option Nat := if st.len < 4294967296 then some st.len else none

/-- `PartialEq for IdList` (`:818`): `len == len && iter().eq(iter())`. -/
def eqList (a b : St) : Bool := a.len == b.len && flat a.segs == flat b.segs

/-- `PartialEq<[u64]>` (`:1242`). -/
def eqSlice (a : St) (o : List Nat) : Bool := a.len == o.length && flat a.segs == o

/-- `PartialEq<[u64; N]>` (`:1251`): `self == other.as_slice()`. -/
def eqArr (a : St) (o : List Nat) : Bool := eqSlice a o

/-- `From<[u64; N]>` (`:1270`) and `From<&[u64]>` (`:1276`): `ids.into_iter().collect()`. -/
def fromIds (R : Roaring) (ids : List Nat) : Option (St × Tally) := pushAllT R St.empty restart ids

/-- `Debug for IdList` (`:804`): `debug_struct("IdList").field("len").field("segments")`.
The Lean `Repr` of `Seg` stands in for the derived `Debug` of `Segment`. -/
def fmt (st : St) : String := s!"IdList \{ len: {st.len}, segments: {repr st.segs} }"

/-- What `to_roaring` (`:1146`) contributes per segment, over the ascending set list. -/
def addSeg (bm : List Nat) : Seg → List Nat
  | .range b l => insRange b (b + (l - 1)) bm
  | .rdesc b l => insRange (b - (l - 1)) b bm
  | .asc s | .dsc s => insAll bm s          -- `out |= bitmap`
  | .rep i _ => ins i bm

/-- `IdList::to_roaring` (`:1146`). -/
def toRoaring (st : St) : List Nat := st.segs.foldl addSeg []

/-! ### Theorems -/

/-- The list's `len` is the number of ids it yields, for every built list. -/
def LenOk (st : St) : Prop := st.len = (flat st.segs).length

theorem fromIds_lenOk (R : Roaring) (ids : List Nat) (hx : ∀ x ∈ ids, x < W) :
    ∃ st t, fromIds R ids = some (st, t) ∧ LenOk st ∧ flat st.segs = ids := by
  obtain ⟨st, t, h1, _, h3, h4⟩ := fromIterT R ids hx
  exact ⟨st, t, h1, by unfold LenOk; rw [h4, h3], h3⟩

theorem isEmpty_iff (st : St) (h : LenOk st) : isEmpty st = true ↔ flat st.segs = [] := by
  unfold isEmpty LenOk at *; rw [h]; simp

theorem count_spec (st : St) :
    (count st = some st.len ↔ st.len < 4294967296) ∧ (count st = none ↔ 4294967296 ≤ st.len) := by
  unfold count; split <;> simp <;> omega

/-- Equality is by the ids, not by segmentation (for lists whose `len` is right). -/
theorem eqList_iff (a b : St) (ha : LenOk a) (hb : LenOk b) :
    eqList a b = true ↔ flat a.segs = flat b.segs := by
  unfold eqList LenOk at *; rw [ha, hb]
  simp only [Bool.and_eq_true, beq_iff_eq]
  constructor
  · exact fun h => h.2
  · intro h; rw [h]; exact ⟨rfl, rfl⟩

theorem eqSlice_iff (a : St) (o : List Nat) (ha : LenOk a) :
    eqSlice a o = true ↔ flat a.segs = o := by
  unfold eqSlice LenOk at *; rw [ha]
  simp only [Bool.and_eq_true, beq_iff_eq]
  constructor
  · exact fun h => h.2
  · intro h; rw [h]; exact ⟨rfl, rfl⟩

theorem eqArr_eq_eqSlice (a : St) (o : List Nat) : eqArr a o = eqSlice a o := rfl

/-- `From` round-trips: `IdList::from(ids) == ids`. -/
theorem from_eq (R : Roaring) (ids : List Nat) (hx : ∀ x ∈ ids, x < W) :
    ∃ st t, fromIds R ids = some (st, t) ∧ eqSlice st ids = true := by
  obtain ⟨st, t, h1, h2, h3⟩ := fromIds_lenOk R ids hx
  exact ⟨st, t, h1, (eqSlice_iff st ids h2).mpr h3⟩

/-- `Debug` prints only `len` and the segments — never the run tally/state. -/
theorem fmt_ignores_run (a b : St) (h1 : a.len = b.len) (h2 : a.segs = b.segs) : fmt a = fmt b := by
  simp [fmt, h1, h2]

theorem mem_addSeg (bm : List Nat) (s : Seg) (hw : WF s) (y : Nat) :
    y ∈ addSeg bm s ↔ y ∈ bm ∨ y ∈ s.iter := by
  cases s with
  | range b l =>
    simp only [addSeg, insRange, mem_insAll]
    have := mem_iter_RL hw trivial y; simp only [Seg.min, Seg.max] at this; rw [this]
  | rdesc b l =>
    simp only [addSeg, insRange, mem_insAll]
    have := mem_iter_RL hw trivial y; simp only [Seg.min, Seg.max] at this; rw [this]
  | asc s => simp [addSeg, mem_insAll, iter]
  | dsc s => simp [addSeg, mem_insAll, iter]
  | rep i c =>
    simp only [addSeg, mem_ins, iter, List.mem_replicate]
    simp [WF] at hw; constructor
    · rintro (h | h); exact Or.inr ⟨by omega, h⟩; exact Or.inl h
    · rintro (h | ⟨_, h⟩); exact Or.inr h; exact Or.inl h

theorem sorted_addSeg (bm : List Nat) (s : Seg) (h : List.Pairwise (· < ·) bm) :
    List.Pairwise (· < ·) (addSeg bm s) := by
  cases s <;> simp only [addSeg, insRange] <;> first | exact sorted_insAll h | exact sorted_ins h

/-- **`to_roaring` is the set of the list's ids**, as a sorted set (one container
operation per segment loses nothing and invents nothing). -/
theorem toRoaring_spec (st : St) (hw : ∀ s ∈ st.segs, WF s) :
    List.Pairwise (· < ·) (toRoaring st) ∧ ∀ y, y ∈ toRoaring st ↔ y ∈ flat st.segs := by
  unfold toRoaring
  suffices ∀ (segs : List Seg) (bm : List Nat), (∀ s ∈ segs, WF s) → List.Pairwise (· < ·) bm →
      List.Pairwise (· < ·) (segs.foldl addSeg bm) ∧
      ∀ y, y ∈ segs.foldl addSeg bm ↔ y ∈ bm ∨ y ∈ flat segs by
    obtain ⟨a, b⟩ := this st.segs [] hw List.Pairwise.nil
    exact ⟨a, fun y => by rw [b]; simp⟩
  intro segs
  induction segs with
  | nil => intro bm _ h; simpa [flat] using h
  | cons s t ih =>
    intro bm hw' h
    simp only [List.foldl_cons]
    obtain ⟨a, b⟩ := ih (addSeg bm s) (fun x hx => hw' x (by simp [hx])) (sorted_addSeg bm s h)
    refine ⟨a, fun y => ?_⟩
    rw [b, mem_addSeg bm s (hw' s (by simp))]
    simp [flat, or_assoc]

end IdListMisc
