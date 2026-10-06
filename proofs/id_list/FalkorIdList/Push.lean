/-
# `IdList::push` — the builder keeps every id, in push order

Model of the segment builder in `graph/src/effects/v3/id_list.rs`.

| here | there |
| --- | --- |
| `Seg`                 | `enum Segment` (id_list.rs:450) — the bitmap variants hold the *set* as its ascending list |
| `Seg.min` / `Seg.max` | `Segment::min` (:570) / `Segment::max` (:557) |
| `Seg.iter`            | `Segment::iter` (:763) |
| `flat`                | `IdList::iter` (:1126) |
| `ins` / `insRange`    | `RoaringTreemap::insert` / `insert_range` — set union on a sorted list |
| `St`                  | `IdList { segments, len, run: Run { start, desc, .. } }` (:783, :182) — the run's cost tally is abstracted to a free `Bool` (see below) |
| `claim`               | `IdList::claim_direction` (:857) |
| `claimStep`           | the first `match self.segments.last()` in `push` (:896-908) |
| `extend?`             | the `match self.segments.last_mut()` hot paths (:913-952) |
| `collapse`            | `IdList::maybe_collapse_run` (:1029), including its `assert!` (:1051) as `none` |
| `push`                | `IdList::push` (:886) |
| `fromSegments`        | `IdList::from_segments` (:1176) |

**The collapse decision is a free Boolean.** `Run::prefers_bitmap` (:337) is
roaring-size arithmetic over the run's history. It decides *whether* to build a
bitmap, never *what* goes into it, so the theorems quantify over every decision
sequence (`pushAll : List Bool → …`): whatever the arithmetic says — including a
future roaring release that drifts from it — the ids come back in order. What
the proof does not cover is that the arithmetic equals `serialized_size()`
(that is `predicted_matches_roaring`'s job).

Ids are `Nat` with the `u64` bound `< 2^64` carried as a hypothesis; every
`checked_add` / guarded subtraction in `push` is then exactly the `Nat`
condition written here (`base.checked_add(1) == Some(id)` ↔ `base + 1 = id`
because `id < 2^64`).
-/

namespace IdListPush

def W : Nat := 18446744073709551616  -- 2^64

/-- `enum Segment`. `asc`/`dsc` hold the roaring *set* as its ascending list
(the order `RoaringTreemap::iter` yields); `len`/`min`/`max` caches are derived. -/
inductive Seg where
  | range (base len : Nat)
  | rep (id count : Nat)
  | rdesc (base len : Nat)
  | asc (bm : List Nat)
  | dsc (bm : List Nat)
  deriving Repr, DecidableEq

namespace Seg

/-- `Segment::min` (:570). Bitmap: `bitmap.min().unwrap_or(0)`. -/
def min : Seg → Nat
  | range b _ => b
  | rep i _ => i
  | rdesc b l => b - (l - 1)
  | asc bm | dsc bm => bm.head?.getD 0

/-- `Segment::max` (:557). -/
def max : Seg → Nat
  | range b l => b + (l - 1)
  | rep i _ => i
  | rdesc b _ => b
  | asc bm | dsc bm => bm.getLast?.getD 0

/-- `Segment::len` (:545). -/
def len : Seg → Nat
  | range _ l | rdesc _ l | rep _ l => l
  | asc bm | dsc bm => bm.length

/-- `Segment::iter` (:763). -/
def iter : Seg → List Nat
  | range b l => List.range' b l
  | rdesc b l => (List.range' (b - (l - 1)) l).reverse
  | rep i c => List.replicate c i
  | asc bm => bm
  | dsc bm => bm.reverse

/-- Range-like: what a run may hold (the `assert!` at :1051). -/
def RL : Seg → Prop
  | range _ _ | rdesc _ _ => True
  | _ => False

instance : DecidablePred RL := fun s => by cases s <;> simp [RL] <;> infer_instance

/-- Well-formedness: what makes `iter`, `min`, `max` panic-free in Rust
(`base + (len-1)` / `base - (len-1)` in range, `len ≠ 0`). -/
def WF : Seg → Prop
  | range b l => 1 ≤ l ∧ b + l ≤ W
  | rdesc b l => 1 ≤ l ∧ l ≤ b + 1 ∧ b < W
  | rep i c => 1 ≤ c ∧ i < W
  | asc bm | dsc bm => bm ≠ [] ∧ bm.Pairwise (· < ·) ∧ ∀ x ∈ bm, x < W

end Seg

open Seg

/-- `IdList::iter` (:1126). -/
def flat (segs : List Seg) : List Nat := segs.flatMap Seg.iter

/-- `RoaringTreemap::insert` on the ascending list of its set. -/
def ins (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: t => if x < y then x :: y :: t else if x = y then y :: t else y :: ins x t

def insAll (bm : List Nat) (xs : List Nat) : List Nat := xs.foldl (fun b x => ins x b) bm

/-- `RoaringTreemap::insert_range(lo..=hi)`. -/
def insRange (lo hi : Nat) (bm : List Nat) : List Nat := insAll bm (List.range' lo (hi + 1 - lo))

/-- `IdList` minus the cost tally: `segments`, `len`, `run.start`, `run.desc`. -/
structure St where
  segs : List Seg
  len : Nat
  start : Nat
  desc : Option Bool
  deriving Repr

def St.empty : St := ⟨[], 0, 0, none⟩

/-- `IdList::claim_direction` (:857). `Run::restart(i)` sets `start := i, desc := None`. -/
def claim (st : St) (d : Bool) : St :=
  match st.desc with
  | none => { st with desc := some d }
  | some d' => if d' = d then st else { st with start := st.segs.length - 1, desc := some d }

/-- The pre-extension `match self.segments.last()` in `push` (:896-908). -/
def claimStep (st : St) (id : Nat) : St :=
  match st.segs.getLast? with
  | some (.range b 1) => if b + 1 = id then claim st false else st
  | some (.rdesc b 1) => if id + 1 = b then claim st true else st
  | _ => st

/-- The hot-path `match self.segments.last_mut()` (:913-952). -/
def extend? (s : Seg) (id : Nat) : Option Seg :=
  match s with
  | .range b l => if b + l = id then some (.range b (l + 1)) else none
  | .rdesc b l => if l ≤ b ∧ id = b - l then some (.rdesc b (l + 1)) else none
  | .rep r c => if r = id then some (.rep r (c + 1)) else none
  | .asc bm => if bm.getLast?.getD 0 < id then some (.asc (ins id bm)) else none
  | .dsc bm => if id < bm.head?.getD 0 then some (.dsc (ins id bm)) else none

/-- `IdList::maybe_collapse_run` (:1029). `c` is `Run::prefers_bitmap()`; `none`
is the `assert!` at :1051 firing. -/
def collapse (c : Bool) (st : St) : Option St :=
  if c then
    let run := st.segs.drop st.start
    if run.all (fun s => decide (RL s)) then
      let bm := run.foldl (fun bm s => insRange s.min s.max bm) []
      let seg := if st.desc.getD false then Seg.dsc bm else Seg.asc bm
      some { st with segs := st.segs.take st.start ++ [seg], start := st.start + 1, desc := none }
    else none
  else some st

/-- `IdList::push` (:886). `c` is what `prefers_bitmap` would answer if asked. -/
def push (c : Bool) (st0 : St) (id : Nat) : Option St :=
  let st := claimStep { st0 with len := st0.len + 1 } id
  match st.segs.getLast? with
  | none =>
    -- `continues_run` is false with no last segment: open one, restart.
    some { st with segs := [.range id 1], start := 0, desc := none }
  | some last =>
    match extend? last id with
    | some s' => some { st with segs := st.segs.dropLast ++ [s'] }
    | none =>
      match last with
      | .range b 1 =>
        if id + 1 = b then
          some (claim { st with segs := st.segs.dropLast ++ [.rdesc b 2] } true)
        else if b = id then
          let segs := st.segs.dropLast ++ [.rep b 2]
          some { st with segs := segs, start := segs.length, desc := none }
        else pushTail c st last id
      | _ => pushTail c st last id
where
  /-- From `let bounds = …` (:1076) to the end of `push`. -/
  pushTail (c : Bool) (st : St) (last : Seg) (id : Nat) : Option St :=
    let continues : Bool := match st.desc with
      | some false => decide (last.max < id)
      | some true => decide (id < last.min)
      | none => decide (last.max < id ∨ id < last.min)
    if continues then
      let desc := match st.desc with | some d => some d | none => some (decide (id < last.min))
      collapse c { st with segs := st.segs ++ [.range id 1], desc := desc }
    else
      some { st with segs := st.segs ++ [.range id 1], start := st.segs.length, desc := none }

/-- `FromIterator for IdList`: push each id, with the `i`-th collapse decision. -/
def pushAll : List Bool → St → List Nat → Option St
  | _, st, [] => some st
  | cs, st, x :: xs => do
    let st' ← push (cs.headD false) st x
    pushAll cs.tail st' xs

/-! ## Invariant -/

def Adj (R : Seg → Seg → Prop) : List Seg → Prop
  | a :: b :: t => R a b ∧ Adj R (b :: t)
  | _ => True

def AscOk : Seg → Prop
  | .range _ _ => True
  | .rdesc _ l => l = 1
  | _ => False

def DescOk : Seg → Prop
  | .range _ l => l = 1
  | .rdesc _ _ => True
  | _ => False

def ascR (a b : Seg) : Prop := a.max < b.min
def descR (a b : Seg) : Prop := b.max < a.min

/-- What the run (`segments[start..]`) looks like for each `desc`. -/
def RunOk : Option Bool → List Seg → Prop
  | none, run => run = [] ∨ ∃ b, run = [.range b 1]
  | some false, run => run ≠ [] ∧ (∀ s ∈ run, AscOk s) ∧ Adj ascR run
  | some true, run => run ≠ [] ∧ (∀ s ∈ run, DescOk s) ∧ Adj descR run

structure Inv (st : St) : Prop where
  start_le : st.start ≤ st.segs.length
  wf : ∀ s ∈ st.segs, WF s
  rl : ∀ s ∈ st.segs.drop st.start, RL s
  run : RunOk st.desc (st.segs.drop st.start)
  lastIn : ∀ s, st.segs.getLast? = some s → RL s → st.start < st.segs.length

/-! ## Sorted-list lemmas -/


theorem mem_ins {x y : Nat} {l : List Nat} : y ∈ ins x l ↔ y = x ∨ y ∈ l := by
  induction l with
  | nil => simp [ins]
  | cons z t ih =>
    unfold ins
    by_cases h1 : x < z
    · simp [h1]
    · by_cases h2 : x = z
      · subst h2; simp
      · simp [h1, h2, ih]; exact or_left_comm

theorem sorted_ins {x : Nat} {l : List Nat} (h : List.Pairwise (· < ·) l) : List.Pairwise (· < ·) (ins x l) := by
  induction l with
  | nil => simp [ins]
  | cons z t ih =>
    rw [List.pairwise_cons] at h
    unfold ins
    by_cases h1 : x < z
    · rw [if_pos h1]
      refine List.pairwise_cons.2 ⟨?_, List.pairwise_cons.2 h⟩
      intro a ha
      simp at ha
      rcases ha with rfl | ha
      · exact h1
      · exact Nat.lt_trans h1 (h.1 a ha)
    · by_cases h2 : x = z
      · rw [if_neg h1, if_pos h2]; exact List.pairwise_cons.2 h
      · rw [if_neg h1, if_neg h2]
        refine List.pairwise_cons.2 ⟨?_, ih h.2⟩
        intro a ha
        rw [mem_ins] at ha
        rcases ha with rfl | ha
        · omega
        · exact h.1 a ha

theorem mem_insAll {xs bm : List Nat} {y : Nat} : y ∈ insAll bm xs ↔ y ∈ bm ∨ y ∈ xs := by
  induction xs generalizing bm with
  | nil => simp [insAll]
  | cons x t ih =>
    simp only [insAll, List.foldl_cons] at *
    rw [ih, mem_ins]; rw [or_comm (a := y = x), or_assoc]; simp

theorem sorted_insAll {xs bm : List Nat} (h : List.Pairwise (· < ·) bm) : List.Pairwise (· < ·) (insAll bm xs) := by
  induction xs generalizing bm with
  | nil => simpa [insAll]
  | cons x t ih =>
    simp only [insAll, List.foldl_cons] at *
    exact ih (sorted_ins h)

/-- Two strictly sorted lists with the same members are equal. -/
theorem sorted_ext : ∀ {a b : List Nat}, List.Pairwise (· < ·) a → List.Pairwise (· < ·) b → (∀ y, y ∈ a ↔ y ∈ b) → a = b
  | [], [], _, _, _ => rfl
  | [], y :: _, _, _, h => by have := (h y).2 (by simp); simp at this
  | x :: _, [], _, _, h => by have := (h x).1 (by simp); simp at this
  | x :: a, y :: b, ha, hb, h => by
    rw [List.pairwise_cons] at ha hb
    have hx := (h x).1 (by simp)
    have hy := (h y).2 (by simp)
    simp at hx hy
    have hxy : x = y := by
      rcases hx with hx | hx
      · exact hx
      · rcases hy with hy | hy
        · exact hy.symm
        · have := hb.1 x hx; have := ha.1 y hy; omega
    subst hxy
    congr 1
    apply sorted_ext ha.2 hb.2
    intro z
    constructor
    · intro hz
      have := (h z).1 (by simp [hz])
      simp at this
      rcases this with rfl | h'
      · have := ha.1 _ hz; omega
      · exact h'
    · intro hz
      have := (h z).2 (by simp [hz])
      simp at this
      rcases this with rfl | h'
      · have := hb.1 _ hz; omega
      · exact h'

theorem le_getLast {l : List Nat} (h : List.Pairwise (· < ·) l) {x : Nat} (hx : x ∈ l) : x ≤ l.getLast?.getD 0 := by
  induction l generalizing x with
  | nil => simp at hx
  | cons z t ih =>
    rw [List.pairwise_cons] at h
    cases t with
    | nil => simp at hx; simp [hx]
    | cons w u =>
      have hl : (z :: w :: u).getLast? = (w :: u).getLast? := by simp [List.getLast?_cons_cons]
      rw [hl]
      simp at hx
      rcases hx with rfl | hx
      · have hw := h.1 w (by simp)
        have := ih h.2 (x := w) (by simp); omega
      · exact ih h.2 (by simpa using hx)

theorem head_le {l : List Nat} (h : List.Pairwise (· < ·) l) {x : Nat} (hx : x ∈ l) : l.head?.getD 0 ≤ x := by
  cases l with
  | nil => simp at hx
  | cons z t =>
    rw [List.pairwise_cons] at h
    simp at hx ⊢
    rcases hx with rfl | hx
    · omega
    · exact Nat.le_of_lt (h.1 x hx)

theorem ins_snoc {x : Nat} {l : List Nat} (h : ∀ y ∈ l, y < x) : ins x l = l ++ [x] := by
  induction l with
  | nil => simp [ins]
  | cons z t ih =>
    have hz := h z (by simp)
    unfold ins
    simp only [show ¬ x < z by omega, show ¬ x = z by omega, if_false]
    simp [ih (fun y hy => h y (by simp [hy]))]

theorem ins_cons {x : Nat} {l : List Nat} (h : ∀ y ∈ l, x < y) : ins x l = x :: l := by
  cases l with
  | nil => simp [ins]
  | cons z t => unfold ins; simp [h z (by simp)]

/-! ## Segment lemmas -/

theorem iter_bounds {s : Seg} (hw : WF s) (hr : RL s) {x : Nat} (hx : x ∈ s.iter) :
    s.min ≤ x ∧ x ≤ s.max := by
  cases s with
  | range b l =>
    simp [iter, List.mem_range'] at hx; obtain ⟨i, hi, rfl⟩ := hx
    simp [Seg.min, Seg.max]; omega
  | rdesc b l =>
    simp [iter, List.mem_range'] at hx; obtain ⟨i, hi, rfl⟩ := hx
    simp [WF] at hw; simp [Seg.min, Seg.max]; omega
  | _ => simp [RL] at hr

theorem mem_iter_RL {s : Seg} (hw : WF s) (hr : RL s) (y : Nat) :
    y ∈ s.iter ↔ y ∈ List.range' s.min (s.max + 1 - s.min) := by
  cases s with
  | range b l =>
    simp [WF] at hw
    simp [iter, Seg.min, Seg.max, List.mem_range']
    constructor <;> rintro ⟨i, hi, h⟩ <;> exact ⟨i, by omega, by omega⟩
  | rdesc b l =>
    simp [WF] at hw
    simp [iter, Seg.min, Seg.max, List.mem_range']
    constructor <;> rintro ⟨i, hi, h⟩ <;> exact ⟨i, by omega, by omega⟩
  | _ => simp [RL] at hr

theorem min_mem {s : Seg} (hw : WF s) (hr : RL s) : s.min ∈ s.iter := by
  cases s with
  | range b l => simp [WF] at hw; simp [iter, Seg.min, List.mem_range']; first | omega | exact ⟨0, by omega, by omega⟩
  | rdesc b l => simp [WF] at hw; simp [iter, Seg.min, List.mem_range']; first | omega | exact ⟨0, by omega, by omega⟩
  | _ => simp [RL] at hr

theorem iter_ne_nil {s : Seg} (hw : WF s) : s.iter ≠ [] := by
  cases s <;> simp [WF, iter] at hw ⊢ <;> first | omega | exact hw.1

/-- The collapsed bitmap holds exactly the run's ids, as a sorted set. -/
theorem collapse_bm (run : List Seg) (bm0 : List Nat) (hs : List.Pairwise (· < ·) bm0)
    (hw : ∀ s ∈ run, WF s) (hr : ∀ s ∈ run, RL s) :
    List.Pairwise (· < ·) (run.foldl (fun bm s => insRange s.min s.max bm) bm0) ∧
    ∀ y, y ∈ run.foldl (fun bm s => insRange s.min s.max bm) bm0 ↔ y ∈ bm0 ∨ y ∈ flat run := by
  induction run generalizing bm0 with
  | nil => simpa [flat] using hs
  | cons s t ih =>
    simp only [List.foldl_cons]
    have hs' : List.Pairwise (· < ·) (insRange s.min s.max bm0) := sorted_insAll hs
    obtain ⟨h1, h2⟩ := ih _ hs' (fun x hx => hw x (by simp [hx])) (fun x hx => hr x (by simp [hx]))
    refine ⟨h1, fun y => ?_⟩
    rw [h2, insRange, mem_insAll, ← mem_iter_RL (hw s (by simp)) (hr s (by simp))]
    simp only [flat, List.flatMap_cons, List.mem_append, or_assoc]

theorem adj_cons {R : Seg → Seg → Prop} {s : Seg} {t : List Seg} :
    Adj R (s :: t) ↔ (∀ u, t.head? = some u → R s u) ∧ Adj R t := by
  cases t <;> simp [Adj]

theorem adj_snoc {R : Seg → Seg → Prop} : ∀ {l : List Seg} {x : Seg},
    Adj R (l ++ [x]) ↔ Adj R l ∧ ∀ u, l.getLast? = some u → R u x
  | [], _ => by simp [Adj]
  | [a], _ => by simp [Adj]
  | a :: b :: t, x => by
    have ih := @adj_snoc R (b :: t) x
    simp only [List.cons_append] at ih ⊢
    simp only [Adj]
    rw [ih]
    simp [List.getLast?_cons_cons]
    exact and_assoc.symm

/-- An ascending run flattens to a strictly ascending list. -/
theorem asc_sorted : ∀ (run : List Seg), (∀ s ∈ run, WF s) → (∀ s ∈ run, RL s) →
    (∀ s ∈ run, AscOk s) → Adj ascR run →
    List.Pairwise (· < ·) (flat run) ∧ ∀ s, run.head? = some s → ∀ x ∈ flat run, s.min ≤ x
  | [], _, _, _, _ => by simp [flat]
  | s :: t, hw, hr, ha, hadj => by
    rw [adj_cons] at hadj
    obtain ⟨ih1, ih2⟩ := asc_sorted t (fun x hx => hw x (by simp [hx]))
      (fun x hx => hr x (by simp [hx])) (fun x hx => ha x (by simp [hx])) hadj.2
    have hws := hw s (by simp); have hrs := hr s (by simp)
    have hs_sorted : List.Pairwise (· < ·) s.iter := by
      have := ha s (by simp)
      cases s with
      | range b l => exact List.pairwise_lt_range' 1
      | rdesc b l => simp [AscOk] at this; subst this; simp [iter]
      | _ => simp [RL] at hrs
    refine ⟨?_, ?_⟩
    · simp only [flat, List.flatMap_cons]
      refine List.pairwise_append.2 ⟨hs_sorted, ih1, fun a ha' b hb => ?_⟩
      have := (iter_bounds hws hrs ha').2
      cases t with
      | nil => simp [flat] at hb
      | cons u t' =>
        have h3 := ih2 u rfl b hb
        have h4 : ascR s u := hadj.1 u rfl
        simp [ascR] at h4; omega
    · intro s' hs' x hx
      simp at hs'; subst hs'
      simp only [flat, List.flatMap_cons, List.mem_append] at hx
      rcases hx with hx | hx
      · exact (iter_bounds hws hrs hx).1
      · cases t with
        | nil => simp [flat] at hx
        | cons u t' =>
          have h3 := ih2 u rfl x hx
          have h4 : ascR s u := hadj.1 u rfl
          have h5 : s.min ≤ s.max := by
            cases s with
            | range b l => simp [Seg.min, Seg.max]
            | rdesc b l => simp [Seg.min, Seg.max]
            | _ => simp [RL] at hrs
          simp [ascR] at h4; omega

/-- A descending run flattens to a strictly descending list. -/
theorem desc_sorted : ∀ (run : List Seg), (∀ s ∈ run, WF s) → (∀ s ∈ run, RL s) →
    (∀ s ∈ run, DescOk s) → Adj descR run →
    (flat run).Pairwise (· > ·) ∧ ∀ s, run.head? = some s → ∀ x ∈ flat run, x ≤ s.max
  | [], _, _, _, _ => by simp [flat]
  | s :: t, hw, hr, ha, hadj => by
    rw [adj_cons] at hadj
    obtain ⟨ih1, ih2⟩ := desc_sorted t (fun x hx => hw x (by simp [hx]))
      (fun x hx => hr x (by simp [hx])) (fun x hx => ha x (by simp [hx])) hadj.2
    have hws := hw s (by simp); have hrs := hr s (by simp)
    have hs_sorted : s.iter.Pairwise (· > ·) := by
      have := ha s (by simp)
      cases s with
      | range b l => simp [DescOk] at this; subst this; simp [iter]
      | rdesc b l =>
        simp only [iter]; rw [List.pairwise_reverse]; exact List.pairwise_lt_range' 1
      | _ => simp [RL] at hrs
    refine ⟨?_, ?_⟩
    · simp only [flat, List.flatMap_cons]
      refine List.pairwise_append.2 ⟨hs_sorted, ih1, fun a ha' b hb => ?_⟩
      have := (iter_bounds hws hrs ha').1
      cases t with
      | nil => simp [flat] at hb
      | cons u t' =>
        have h3 := ih2 u rfl b hb
        have h4 : descR s u := hadj.1 u rfl
        simp [descR] at h4; omega
    · intro s' hs' x hx
      simp at hs'; subst hs'
      simp only [flat, List.flatMap_cons, List.mem_append] at hx
      rcases hx with hx | hx
      · exact (iter_bounds hws hrs hx).2
      · cases t with
        | nil => simp [flat] at hx
        | cons u t' =>
          have h3 := ih2 u rfl x hx
          have h4 : descR s u := hadj.1 u rfl
          have h5 : s.min ≤ s.max := by
            cases s with
            | range b l => simp [Seg.min, Seg.max]
            | rdesc b l => simp [Seg.min, Seg.max]
            | _ => simp [RL] at hrs
          simp [descR] at h4; omega

/-! ## Preservation -/

theorem flat_snoc (l : List Seg) (x : Seg) : flat (l ++ [x]) = flat l ++ x.iter := by
  simp [flat, List.flatMap_append]

theorem iter_lt_W {s : Seg} (hw : WF s) {x : Nat} (hx : x ∈ s.iter) : x < W := by
  cases s with
  | range b l => simp [iter, List.mem_range'] at hx; simp [WF] at hw; omega
  | rdesc b l => simp [iter, List.mem_range'] at hx; simp [WF] at hw; omega
  | rep i c => simp [iter] at hx; simp [WF] at hw; omega
  | asc bm => simp [iter] at hx; simp [WF] at hw; exact hw.2.2 x hx
  | dsc bm => simp [iter] at hx; simp [WF] at hw; exact hw.2.2 x hx

theorem getLast?_drop {l : List Seg} {n : Nat} (h : n < l.length) : (l.drop n).getLast? = l.getLast? := by
  induction l generalizing n with
  | nil => simp at h
  | cons a t ih =>
    cases n with
    | zero => simp
    | succ n =>
      simp at h
      simp only [List.drop_succ_cons]
      rw [ih (by omega)]
      cases t with
      | nil => simp at h
      | cons b u => simp [List.getLast?_cons_cons]

theorem run_nil_of_not_RL {st : St} (hi : Inv st) {s : Seg} (hl : st.segs.getLast? = some s)
    (hn : ¬ RL s) : st.segs.drop st.start = [] := by
  apply Classical.byContradiction; intro hne
  have hlt : st.start < st.segs.length := by
    rw [List.drop_eq_nil_iff] at hne; omega
  have := getLast?_drop hlt
  rw [hl] at this
  exact hn (hi.rl s (List.mem_of_getLast? this))

theorem runOk_some_ne {d : Bool} {run : List Seg} (h : RunOk (some d) run) : run ≠ [] := by
  cases d <;> exact h.1

theorem runOk_none_snoc {p : List Seg} {x : Seg} (h : RunOk none (p ++ [x])) :
    p = [] ∧ ∃ b, x = .range b 1 := by
  rcases h with h | ⟨b, h⟩
  · simp at h
  · cases p with
    | nil => simp at h; exact ⟨rfl, b, h⟩
    | cons a t => simp at h

theorem replace_last_asc {p : List Seg} {x0 x : Seg} (h : RunOk (some false) (p ++ [x0]))
    (hx : AscOk x) (hm : x.min = x0.min) : RunOk (some false) (p ++ [x]) := by
  obtain ⟨_, h2, h3⟩ := h
  refine ⟨by simp, fun s hs => ?_, ?_⟩
  · simp at hs; rcases hs with hs | rfl
    · exact h2 s (by simp [hs])
    · exact hx
  · rw [adj_snoc] at h3 ⊢
    exact ⟨h3.1, fun u hu => by have := h3.2 u hu; simp [ascR] at this ⊢; omega⟩

theorem replace_last_desc {p : List Seg} {x0 x : Seg} (h : RunOk (some true) (p ++ [x0]))
    (hx : DescOk x) (hm : x.max = x0.max) : RunOk (some true) (p ++ [x]) := by
  obtain ⟨_, h2, h3⟩ := h
  refine ⟨by simp, fun s hs => ?_, ?_⟩
  · simp at hs; rcases hs with hs | rfl
    · exact h2 s (by simp [hs])
    · exact hx
  · rw [adj_snoc] at h3 ⊢
    exact ⟨h3.1, fun u hu => by have := h3.2 u hu; simp [descR] at this ⊢; omega⟩

/-- What one push must re-establish. -/
def Post (st0 st : St) (id : Nat) : Prop :=
  Inv st ∧ flat st.segs = flat st0.segs ++ [id] ∧ st.len = st0.len + 1

theorem collapse_ok (c : Bool) (st : St) (hi : Inv st) (d : Bool) (hd : st.desc = some d) :
    ∃ st', collapse c st = some st' ∧ Inv st' ∧ flat st'.segs = flat st.segs ∧ st'.len = st.len := by
  cases c with
  | false => exact ⟨st, rfl, hi, rfl, rfl⟩
  | true =>
    have hrun := hi.run; rw [hd] at hrun
    have hne := runOk_some_ne hrun
    have hwr : ∀ s ∈ st.segs.drop st.start, WF s := fun s hs => hi.wf s (List.mem_of_mem_drop hs)
    have hall : (st.segs.drop st.start).all (fun s => decide (RL s)) = true := by
      simp only [List.all_eq_true, decide_eq_true_eq]; exact hi.rl
    obtain ⟨hbs, hbm⟩ := collapse_bm (st.segs.drop st.start) [] (by simp) hwr hi.rl
    have hsplit : flat st.segs = flat (st.segs.take st.start) ++ flat (st.segs.drop st.start) := by
      simp only [flat]; rw [← List.flatMap_append, List.take_append_drop]
    have htl : (st.segs.take st.start).length = st.start := by
      rw [List.length_take]; exact Nat.min_eq_left hi.start_le
    have hflat_ne : flat (st.segs.drop st.start) ≠ [] := by
      obtain ⟨s, hs⟩ := List.exists_mem_of_ne_nil _ hne
      obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ (iter_ne_nil (hwr s hs))
      intro h; have : y ∈ flat (st.segs.drop st.start) := List.mem_flatMap.2 ⟨s, hs, hy⟩
      rw [h] at this; simp at this
    have hlt : ∀ y ∈ flat (st.segs.drop st.start), y < W := by
      intro y hy
      obtain ⟨s, hs, hy⟩ := List.mem_flatMap.1 hy
      exact iter_lt_W (hwr s hs) hy
    -- the bitmap is exactly the run, in the run's direction
    have key : ∀ bm, List.Pairwise (· < ·) bm →
        (∀ y, y ∈ bm ↔ y ∈ ([] : List Nat) ∨ y ∈ flat (st.segs.drop st.start)) →
        (if d then (Seg.dsc bm).iter else (Seg.asc bm).iter) = flat (st.segs.drop st.start) ∧
        WF (if d then Seg.dsc bm else Seg.asc bm) := by
      intro bm hs hm
      have hbm_ne : bm ≠ [] := by
        intro h; subst h
        obtain ⟨y, hy⟩ := List.exists_mem_of_ne_nil _ hflat_ne
        have := (hm y).2 (Or.inr hy); simp at this
      have hlt' : ∀ x ∈ bm, x < W := fun x hx => hlt x (by have := (hm x).1 hx; simpa using this)
      cases d with
      | false =>
        obtain ⟨_, h2, h3⟩ := hrun
        obtain ⟨hsorted, _⟩ := asc_sorted _ hwr hi.rl h2 h3
        refine ⟨?_, hbm_ne, hs, hlt'⟩
        simp only [Bool.false_eq_true, if_false, iter]
        exact sorted_ext hs hsorted (fun y => by rw [hm]; simp)
      | true =>
        obtain ⟨_, h2, h3⟩ := hrun
        obtain ⟨hsorted, _⟩ := desc_sorted _ hwr hi.rl h2 h3
        refine ⟨?_, hbm_ne, hs, hlt'⟩
        simp only [if_true, iter]
        have hrev : List.Pairwise (· < ·) (flat (st.segs.drop st.start)).reverse := by
          rw [List.pairwise_reverse]; exact hsorted
        have := sorted_ext hs hrev (fun y => by rw [hm]; simp)
        rw [this, List.reverse_reverse]
    obtain ⟨hk1, hk2⟩ := key _ hbs hbm
    let bm := (st.segs.drop st.start).foldl (fun bm s => insRange s.min s.max bm) []
    have hc : collapse true st = some { st with
        segs := st.segs.take st.start ++ [if st.desc.getD false then Seg.dsc bm else Seg.asc bm],
        start := st.start + 1, desc := none } := by
      simp only [collapse, if_true, hall]; rfl
    refine ⟨_, hc, ?_, ?_, ?_⟩
    · simp only [hd, Option.getD_some]
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · simp only [List.length_append, htl, List.length_singleton]; omega
      · intro s hs
        simp only [List.mem_append, List.mem_singleton] at hs
        rcases hs with hs | rfl
        · exact hi.wf s (List.mem_of_mem_take hs)
        · cases d <;> simpa using hk2
      · intro s hs
        dsimp only at hs
        rw [List.drop_eq_nil_iff.2 (by simp only [List.length_append, htl, List.length_singleton]; omega)] at hs
        simp at hs
      · show RunOk none _
        left
        dsimp only
        exact List.drop_eq_nil_iff.2 (by simp only [List.length_append, htl, List.length_singleton]; omega)
      · intro s hs hr
        simp at hs; subst hs
        cases d <;> simp [RL] at hr
    · simp only [hd, Option.getD_some]
      rw [flat_snoc, hsplit]
      congr 1
      cases d <;> simpa using hk1
    · rfl

theorem runOk_single (d : Option Bool) (hd : d ≠ none) (x : Nat) : RunOk d [.range x 1] := by
  match d with
  | none => exact absurd rfl hd
  | some false => exact ⟨by simp, by simp [AscOk], by simp [Adj]⟩
  | some true => exact ⟨by simp, by simp [DescOk], by simp [Adj]⟩

/-- Appending `Range{id,1}` to the run (the `continues_run` branch, before the collapse). -/
theorem inv_append_cont (st : St) (hi : Inv st) (last : Seg) (hl : st.segs.getLast? = some last)
    (id : Nat) (hid : id < W) (d' : Option Bool) (hd' : d' ≠ none)
    (hrun : RunOk d' (st.segs.drop st.start ++ [.range id 1])) :
    Inv { st with segs := st.segs ++ [.range id 1], desc := d' } := by
  have hsl := hi.start_le
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp; omega
  · intro s hs; simp at hs; rcases hs with hs | rfl
    · exact hi.wf s hs
    · simp [WF]; exact hid
  · intro s hs; dsimp only at hs
    rw [List.drop_append_of_le_length hsl] at hs
    simp at hs; rcases hs with hs | rfl
    · exact hi.rl s hs
    · simp [RL]
  · dsimp only; rw [List.drop_append_of_le_length hsl]; exact hrun
  · intro _ _ _; simp; omega

theorem pushTail_ok (c : Bool) (st : St) (hi : Inv st) (last : Seg)
    (hl : st.segs.getLast? = some last) (id : Nat) (hid : id < W) :
    ∃ st', push.pushTail c st last id = some st' ∧ Inv st' ∧
      flat st'.segs = flat st.segs ++ [id] ∧ st'.len = st.len := by
  have hsl := hi.start_le
  -- the run's last segment, when the run is nonempty, is `last`
  have hrl : st.segs.drop st.start ≠ [] → (st.segs.drop st.start).getLast? = some last := by
    intro hne; rw [getLast?_drop]; exact hl
    rw [Ne, List.drop_eq_nil_iff] at hne; omega
  have fresh : Inv { st with segs := st.segs ++ [.range id 1], start := st.segs.length, desc := none } := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · simp
    · intro s hs; simp at hs; rcases hs with hs | rfl
      · exact hi.wf s hs
      · simp [WF]; exact hid
    · intro s hs; dsimp only at hs
      rw [List.drop_append_of_le_length (Nat.le_refl _), List.drop_length] at hs
      simp at hs; subst hs; simp [RL]
    · dsimp only; rw [List.drop_append_of_le_length (Nat.le_refl _), List.drop_length]
      exact Or.inr ⟨id, rfl⟩
    · intro _ _ _; simp
  -- the continuing branch, given the new run is well-formed
  have cont : ∀ d', d' ≠ none → RunOk d' (st.segs.drop st.start ++ [.range id 1]) →
      ∃ st', collapse c { st with segs := st.segs ++ [.range id 1], desc := d' } = some st' ∧
        Inv st' ∧ flat st'.segs = flat st.segs ++ [id] ∧ st'.len = st.len := by
    intro d' hd' hrun
    have hinv := inv_append_cont st hi last hl id hid d' hd' hrun
    match d', hd' with
    | some d, _ =>
      obtain ⟨st', h1, h2, h3, h4⟩ := collapse_ok c _ hinv d rfl
      exact ⟨st', h1, h2, by rw [h3, flat_snoc]; rfl, h4⟩
  have hr1 : (Seg.range id 1).iter = [id] := by simp [iter]
  cases hd : st.desc with
  | none =>
    have hrun := hi.run; rw [hd] at hrun
    by_cases hc : last.max < id ∨ id < last.min
    · simp only [push.pushTail, hd, hc, decide_true, if_true]
      apply cont _ (by simp)
      rcases hrun with hrun | ⟨b, hrun⟩
      · rw [hrun]; exact runOk_single _ (by simp) _
      · have := hrl (by rw [hrun]; simp); rw [hrun] at this; simp at this; subst this
        rw [hrun]
        simp [Seg.min, Seg.max] at hc
        by_cases hlt : id < b
        · have e : decide (id < (Seg.range b 1).min) = true := by simp [Seg.min, hlt]
          rw [e]
          exact ⟨by simp, by simp [DescOk], by simp [Adj, descR, Seg.min, Seg.max]; exact hlt⟩
        · have e : decide (id < (Seg.range b 1).min) = false := by simp [Seg.min, hlt]
          rw [e]
          exact ⟨by simp, by simp [AscOk], by simp [Adj, ascR, Seg.min, Seg.max]; omega⟩
    · simp only [push.pushTail, hd, hc, decide_false]
      exact ⟨_, rfl, fresh, by simp [flat_snoc, hr1], rfl⟩
  | some d =>
    have hrun := hi.run; rw [hd] at hrun
    have hne := runOk_some_ne hrun
    have hlast := hrl hne
    cases d with
    | false =>
      by_cases hc : last.max < id
      · simp only [push.pushTail, hd, hc, decide_true, if_true]
        apply cont _ (by simp)
        obtain ⟨_, h2, h3⟩ := hrun
        refine ⟨by simp, fun s hs => ?_, ?_⟩
        · simp at hs; rcases hs with hs | rfl
          · exact h2 s hs
          · simp [AscOk]
        · rw [adj_snoc]; refine ⟨h3, fun u hu => ?_⟩
          rw [hlast] at hu; cases hu; simp [ascR, Seg.min]; exact hc
      · simp only [push.pushTail, hd, hc, decide_false]
        exact ⟨_, rfl, fresh, by simp [flat_snoc, hr1], rfl⟩
    | true =>
      by_cases hc : id < last.min
      · simp only [push.pushTail, hd, hc, decide_true, if_true]
        apply cont _ (by simp)
        obtain ⟨_, h2, h3⟩ := hrun
        refine ⟨by simp, fun s hs => ?_, ?_⟩
        · simp at hs; rcases hs with hs | rfl
          · exact h2 s hs
          · simp [DescOk]
        · rw [adj_snoc]; refine ⟨h3, fun u hu => ?_⟩
          rw [hlast] at hu; cases hu; simp [descR, Seg.max]; exact hc
      · simp only [push.pushTail, hd, hc, decide_false]
        exact ⟨_, rfl, fresh, by simp [flat_snoc, hr1], rfl⟩

theorem inv_of {init : List Seg} {x : Seg} {len start : Nat} {desc : Option Bool}
    (hs : start ≤ init.length) (hw : ∀ s ∈ init, WF s) (hx : WF x) (hrx : RL x)
    (hr : ∀ s ∈ init.drop start, RL s) (hrun : RunOk desc (init.drop start ++ [x])) :
    Inv ⟨init ++ [x], len, start, desc⟩ := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp; omega
  · intro s h; simp at h; rcases h with h | rfl
    · exact hw s h
    · exact hx
  · intro s h; dsimp only at h; rw [List.drop_append_of_le_length hs] at h
    simp at h; rcases h with h | rfl
    · exact hr s h
    · exact hrx
  · dsimp only; rw [List.drop_append_of_le_length hs]; exact hrun
  · intro _ _ _; simp; omega

/-- A segment outside the run (repeat / bitmap) grew in place. -/
theorem inv_of_outside {init : List Seg} {x x' : Seg} {len start : Nat} {desc : Option Bool}
    (hi : Inv ⟨init ++ [x], len, start, desc⟩) (hnx : ¬ RL x) (hnx' : ¬ RL x') (hx' : WF x') :
    Inv ⟨init ++ [x'], len, start, desc⟩ := by
  have hnil := run_nil_of_not_RL hi (s := x) (by simp) hnx
  dsimp only at hnil
  have hlen : init.length + 1 ≤ start := by
    rw [List.drop_eq_nil_iff] at hnil; simpa using hnil
  have hsl := hi.start_le; simp at hsl
  have hnil' : (init ++ [x']).drop start = [] := List.drop_eq_nil_iff.2 (by simp; omega)
  have hrun := hi.run; dsimp only at hrun; rw [hnil] at hrun
  refine ⟨by simp; omega, ?_, ?_, ?_, ?_⟩
  · intro s h; simp at h; rcases h with h | rfl
    · exact hi.wf s (by simp [h])
    · exact hx'
  · intro s h; dsimp only at h; rw [hnil'] at h; simp at h
  · dsimp only; rw [hnil']; exact hrun
  · intro s h hr; simp at h; subst h; exact absurd hr hnx'

theorem claimStep_segs (st : St) (id : Nat) : (claimStep st id).segs = st.segs ∧ (claimStep st id).len = st.len := by
  unfold claimStep claim
  split
  · split
    · split
      · simp
      · split <;> simp
    · simp
  · split
    · split
      · simp
      · split <;> simp
    · simp
  · simp

theorem claimStep_noop (st : St) (id : Nat) (last : Seg) (hl : st.segs.getLast? = some last)
    (hn : extend? last id = none) : claimStep st id = st := by
  unfold claimStep
  rw [hl]
  cases last with
  | range b l =>
    by_cases h1 : l = 1
    · subst h1; simp [extend?] at hn; simp; intro h; omega
    · split
      · rename_i h; simp at h; try omega
      · rename_i h; simp at h; try omega
      · rfl
  | rdesc b l =>
    by_cases h1 : l = 1
    · subst h1; simp [extend?] at hn; simp; intro h; have := hn (by omega); omega
    · split
      · rename_i h; simp at h; try omega
      · rename_i h; simp at h; try omega
      · rfl
  | _ => rfl

theorem claim_segs (st : St) (d : Bool) : (claim st d).segs = st.segs := by
  unfold claim; split <;> (try split) <;> rfl

theorem push_ok (c : Bool) (st0 : St) (hi : Inv st0) (id : Nat) (hid : id < W) :
    ∃ st', push c st0 id = some st' ∧ Post st0 st' id := by
  obtain ⟨segs, len, start, desc⟩ := st0
  rcases List.eq_nil_or_concat segs with h | ⟨init, last, h⟩
  · subst h
    have hp : push c ⟨[], len, start, desc⟩ id = some ⟨[.range id 1], len + 1, 0, none⟩ := by
      simp [push, claimStep]
    refine ⟨_, hp, ?_, by simp [flat, iter], rfl⟩
    have := @inv_of [] (.range id 1) (len + 1) 0 none (by simp) (by simp) (by simp [WF]; exact hid)
      (by simp [RL]) (by simp) (Or.inr ⟨id, rfl⟩)
    simpa using this
  rw [List.concat_eq_append] at h
  subst h
  let st1 : St := ⟨init ++ [last], len + 1, start, desc⟩
  have hi1 : Inv st1 := ⟨hi.1, hi.2, hi.3, hi.4, hi.5⟩
  have hsl := hi.start_le; simp at hsl
  have hw := hi.wf
  have hwl : WF last := hw last (by simp)
  have hwi : ∀ s ∈ init, WF s := fun s hs => hw s (by simp [hs])
  -- when `last` is range-like it is in the run
  have inrun : RL last → start ≤ init.length := by
    intro hr; have := hi.lastIn last (by simp) hr; simp at this; omega
  have hrun_of : RL last → RunOk desc (init.drop start ++ [last]) := by
    intro hr; have := hi.run; dsimp only at this
    rwa [List.drop_append_of_le_length (inrun hr)] at this
  have hrl_of : RL last → ∀ s ∈ init.drop start, RL s := by
    intro hr s hs; apply hi.rl; dsimp only
    rw [List.drop_append_of_le_length (inrun hr)]; simp [hs]
  cases hext : extend? last id with
  | some s' =>
    have hp : push c ⟨init ++ [last], len, start, desc⟩ id =
        some { claimStep st1 id with segs := init ++ [s'] } := by
      simp only [push]
      rw [(claimStep_segs _ _).1]
      simp only [st1, List.getLast?_concat, hext, List.dropLast_concat]
    refine ⟨_, hp, ?_, ?_, ?_⟩
    rotate_left
    · -- ids
      simp only [flat_snoc]
      rw [List.append_assoc]; congr 1
      cases last with
      | range b l =>
        simp [extend?] at hext; obtain ⟨h1, rfl⟩ := hext
        simp only [iter]; rw [← List.range'_append (m := l) (n := 1)]; simp; try omega
      | rdesc b l =>
        simp [extend?] at hext; obtain ⟨⟨h1, h2⟩, rfl⟩ := hext
        simp [WF] at hwl
        simp only [iter]
        rw [show b - (l + 1 - 1) = b - l by omega, List.range'_succ,
          show b - l + 1 = b - (l - 1) by omega]
        simp [h2]
      | rep r k =>
        simp [extend?] at hext; obtain ⟨rfl, rfl⟩ := hext
        simp [iter, List.replicate_succ']
      | asc bm =>
        simp [extend?] at hext; obtain ⟨h1, rfl⟩ := hext
        simp [WF] at hwl
        simp only [iter]
        exact ins_snoc (fun y hy => Nat.lt_of_le_of_lt (le_getLast hwl.2.1 hy) h1)
      | dsc bm =>
        simp [extend?] at hext; obtain ⟨h1, rfl⟩ := hext
        simp [WF] at hwl
        simp only [iter]
        rw [ins_cons (fun y hy => Nat.lt_of_lt_of_le h1 (head_le hwl.2.1 hy))]
        simp
    · -- len
      exact (claimStep_segs st1 id).2
    · -- invariant
      cases last with
      | rep r k =>
        simp [extend?] at hext; obtain ⟨rfl, rfl⟩ := hext
        have : claimStep st1 r = st1 := by simp [claimStep, st1]
        rw [this]
        exact inv_of_outside (x := .rep r k) hi1 (by simp [RL]) (by simp [RL])
          (by simp [WF] at hwl ⊢; exact hwl.2)
      | asc bm =>
        simp [extend?] at hext; obtain ⟨h1, rfl⟩ := hext
        have : claimStep st1 id = st1 := by simp [claimStep, st1]
        rw [this]
        simp [WF] at hwl
        refine inv_of_outside (x := .asc bm) hi1 (by simp [RL]) (by simp [RL]) ?_
        refine ⟨?_, sorted_ins hwl.2.1, fun x hx => ?_⟩
        · intro h; have := (mem_ins (x := id) (l := bm) (y := id)).2 (Or.inl rfl); rw [h] at this; simp at this
        · rw [mem_ins] at hx; rcases hx with rfl | hx
          · exact hid
          · exact hwl.2.2 x hx
      | dsc bm =>
        simp [extend?] at hext; obtain ⟨h1, rfl⟩ := hext
        have : claimStep st1 id = st1 := by simp [claimStep, st1]
        rw [this]
        simp [WF] at hwl
        refine inv_of_outside (x := .dsc bm) hi1 (by simp [RL]) (by simp [RL]) ?_
        refine ⟨?_, sorted_ins hwl.2.1, fun x hx => ?_⟩
        · intro h; have := (mem_ins (x := id) (l := bm) (y := id)).2 (Or.inl rfl); rw [h] at this; simp at this
        · rw [mem_ins] at hx; rcases hx with rfl | hx
          · exact hid
          · exact hwl.2.2 x hx
      | range b l =>
        simp [extend?] at hext; obtain ⟨h1, rfl⟩ := hext
        have hrun := hrun_of (by simp [RL])
        have hrl := hrl_of (by simp [RL])
        have hin := inrun (by simp [RL])
        simp [WF] at hwl
        have hwn : WF (.range b (l + 1)) := by simp [WF]; omega
        by_cases hl1 : l = 1
        · subst hl1
          have hcs : claimStep st1 id = claim st1 false := by
            simp [claimStep, st1, h1]
          rw [hcs]
          cases desc with
          | none =>
            obtain ⟨hp0, _⟩ := runOk_none_snoc hrun
            have : claim st1 false = { st1 with desc := some false } := rfl
            rw [this]
            refine inv_of hin hwi hwn (by simp [RL]) hrl ?_
            rw [hp0]; exact ⟨by simp, by simp [AscOk], by simp [Adj]⟩
          | some d =>
            cases d with
            | false =>
              have : claim st1 false = st1 := by simp [claim, st1]
              rw [this]
              exact inv_of hin hwi hwn (by simp [RL]) hrl
                (replace_last_asc hrun (by simp [AscOk]) (by simp [Seg.min]))
            | true =>
              have : claim st1 false = { st1 with start := init.length, desc := some false } := by
                simp [claim, st1]
              rw [this]
              refine inv_of (Nat.le_refl _) hwi hwn (by simp [RL]) (by simp) ?_
              simp [RunOk, AscOk, Adj]
        · have hcs : claimStep st1 id = st1 := by
            simp only [claimStep, st1, List.getLast?_concat]
            split
            · rename_i h; simp at h; omega
            · rename_i h; simp at h
            · rfl
          rw [hcs]
          refine inv_of hin hwi hwn (by simp [RL]) hrl ?_
          cases desc with
          | none => obtain ⟨_, b', hb'⟩ := runOk_none_snoc hrun; simp at hb'; omega
          | some d =>
            cases d with
            | false => exact replace_last_asc hrun (by simp [AscOk]) (by simp [Seg.min])
            | true =>
              obtain ⟨_, h2, _⟩ := hrun
              have := h2 (.range b l) (by simp); simp [DescOk] at this; omega
      | rdesc b l =>
        simp [extend?] at hext; obtain ⟨⟨h1, h2⟩, rfl⟩ := hext
        have hrun := hrun_of (by simp [RL])
        have hrl := hrl_of (by simp [RL])
        have hin := inrun (by simp [RL])
        simp [WF] at hwl
        have hwn : WF (.rdesc b (l + 1)) := by simp [WF]; omega
        by_cases hl1 : l = 1
        · subst hl1
          have hcs : claimStep st1 id = claim st1 true := by
            simp [claimStep, st1]; omega
          rw [hcs]
          cases desc with
          | none =>
            obtain ⟨_, b', hb'⟩ := runOk_none_snoc hrun; simp at hb'
          | some d =>
            cases d with
            | true =>
              have : claim st1 true = st1 := by simp [claim, st1]
              rw [this]
              exact inv_of hin hwi hwn (by simp [RL]) hrl
                (replace_last_desc hrun (by simp [DescOk]) (by simp [Seg.max]))
            | false =>
              have : claim st1 true = { st1 with start := init.length, desc := some true } := by
                simp [claim, st1]
              rw [this]
              refine inv_of (Nat.le_refl _) hwi hwn (by simp [RL]) (by simp) ?_
              simp [RunOk, DescOk, Adj]
        · have hcs : claimStep st1 id = st1 := by
            simp only [claimStep, st1, List.getLast?_concat]
            split
            · rename_i h; simp at h
            · rename_i h; simp at h; omega
            · rfl
          rw [hcs]
          refine inv_of hin hwi hwn (by simp [RL]) hrl ?_
          cases desc with
          | none => obtain ⟨_, b', hb'⟩ := runOk_none_snoc hrun; simp at hb'
          | some d =>
            cases d with
            | true => exact replace_last_desc hrun (by simp [DescOk]) (by simp [Seg.max])
            | false =>
              obtain ⟨_, h2, _⟩ := hrun
              have := h2 (.rdesc b l) (by simp); simp [AscOk] at this; omega
  | none =>
    have hcs := claimStep_noop st1 id last (by simp [st1]) hext
    obtain ⟨st', t1, t2, t3, t4⟩ := pushTail_ok c st1 hi1 last (by simp [st1]) id hid
    have tail : ∃ st', push.pushTail c st1 last id = some st' ∧
        Post ⟨init ++ [last], len, start, desc⟩ st' id := ⟨st', t1, t2, t3, t4⟩
    simp only [push]
    rw [hcs]
    simp only [st1, List.getLast?_concat, hext, List.dropLast_concat]
    cases last with
    | range b l =>
      by_cases hl1 : l = 1
      · subst hl1
        simp only
        simp [WF] at hwl
        have hrun := hrun_of (by simp [RL])
        have hrl := hrl_of (by simp [RL])
        have hin := inrun (by simp [RL])
        by_cases hd1 : id + 1 = b
        · simp only [hd1, if_true]
          refine ⟨_, rfl, ?_, ?_, by unfold claim; split <;> (try split) <;> rfl⟩
          · have hwn : WF (.rdesc b 2) := by simp [WF]; omega
            cases desc with
            | none =>
              obtain ⟨hp0, _⟩ := runOk_none_snoc hrun
              refine inv_of hin hwi hwn (by simp [RL]) hrl ?_
              rw [hp0]; exact ⟨by simp, by simp [DescOk], by simp [Adj]⟩
            | some d =>
              cases d with
              | true =>
                exact inv_of hin hwi hwn (by simp [RL]) hrl
                  (replace_last_desc hrun (by simp [DescOk]) (by simp [Seg.max]))
              | false =>
                show Inv ⟨init ++ [.rdesc b 2], len + 1, (init ++ [Seg.rdesc b 2]).length - 1, some true⟩
                refine inv_of (by simp) hwi hwn (by simp [RL]) (by simp) ?_
                simp [RunOk, DescOk, Adj]
          · rw [claim_segs]
            show flat (init ++ [Seg.rdesc b 2]) = flat (init ++ [Seg.range b 1]) ++ [id]
            simp only [flat_snoc, iter, List.append_assoc]
            congr 1
            rw [show b - (2 - 1) = id by omega]
            simp [List.range'_succ]; omega
        · by_cases hd2 : b = id
          · subst hd2
            simp only [hd1, if_false, if_true]
            refine ⟨_, rfl, ?_, ?_, rfl⟩
            · refine ⟨by simp, ?_, ?_, ?_, ?_⟩
              · intro s hs; simp at hs; rcases hs with hs | rfl
                · exact hwi s hs
                · simp [WF]; omega
              · intro s hs; dsimp only at hs; rw [List.drop_length] at hs; simp at hs
              · dsimp only; rw [List.drop_length]; exact Or.inl rfl
              · intro s hs hr; simp at hs; subst hs; simp [RL] at hr
            · simp only [flat_snoc, iter, List.append_assoc]; simp
          · simp only [hd1, hd2, if_false]
            exact tail
      · split
        · rename_i h; simp at h; omega
        · exact tail
    | rdesc b l => exact tail
    | rep r k => exact tail
    | asc bm => exact tail
    | dsc bm => exact tail

/-! ## The builder, end to end -/

theorem inv_empty : Inv St.empty :=
  ⟨by simp [St.empty], by simp [St.empty], by simp [St.empty], Or.inl (by simp [St.empty]),
   by simp [St.empty]⟩

/-- **Every push sequence is kept, in order**, whatever `prefers_bitmap` decides
at each step, and the `assert!` in `maybe_collapse_run` never fires. -/
theorem pushAll_ok : ∀ (cs : List Bool) (st : St) (xs : List Nat), Inv st → (∀ x ∈ xs, x < W) →
    ∃ st', pushAll cs st xs = some st' ∧ Inv st' ∧ flat st'.segs = flat st.segs ++ xs ∧
      st'.len = st.len + xs.length
  | _, st, [], hi, _ => ⟨st, rfl, hi, by simp, rfl⟩
  | cs, st, x :: xs, hi, hx => by
    obtain ⟨st1, h1, hi1, hf1, hl1⟩ := push_ok (cs.headD false) st hi x (hx x (by simp))
    obtain ⟨st2, h2, hi2, hf2, hl2⟩ :=
      pushAll_ok cs.tail st1 xs hi1 (fun y hy => hx y (by simp [hy]))
    refine ⟨st2, ?_, hi2, ?_, ?_⟩
    · simp only [pushAll]; rw [h1]; exact h2
    · rw [hf2, hf1]; simp
    · rw [hl2, hl1]; simp; omega

theorem iter_length {s : Seg} (hw : WF s) : s.iter.length = s.len := by
  cases s <;> simp [iter, Seg.len]

/-- `IdList::from_iter` (:1259): the ids come back as pushed. -/
theorem fromIter_iter (cs : List Bool) (xs : List Nat) (hx : ∀ x ∈ xs, x < W) :
    ∃ st, pushAll cs St.empty xs = some st ∧ Inv st ∧ flat st.segs = xs ∧ st.len = xs.length := by
  obtain ⟨st, h1, h2, h3, h4⟩ := pushAll_ok cs St.empty xs inv_empty hx
  exact ⟨st, h1, h2, by simpa [St.empty, flat] using h3, by simpa [St.empty] using h4⟩

/-- Every segment carries at least one id, so a built list never has more
segments than ids (what `read_ids`' `n_segments > count` check relies on). -/
theorem segs_le_len {segs : List Seg} (hw : ∀ s ∈ segs, WF s) : segs.length ≤ (flat segs).length := by
  induction segs with
  | nil => simp
  | cons s t ih =>
    simp only [flat, List.flatMap_cons, List.length_append, List.length_cons] at *
    have := iter_ne_nil (hw s (by simp))
    have : 0 < s.iter.length := List.length_pos_iff.2 this
    have := ih (fun x hx => hw x (by simp [hx])); omega

/-! ## Counterexample: pushing onto a *decoded* list

`from_segments` (:1176) starts the run AT the last decoded segment with
`desc = None` and a fresh tally, whatever that segment is. `Inv` does not hold
there: the run may hold a `Repeat` or a bitmap, or a descending range under an
undecided direction. `push` is `pub`, so a decoded list can be extended. -/

/-- `IdList::from_segments` (:1176). -/
def fromSegments (segs : List Seg) : St := ⟨segs, (flat segs).length, segs.length - 1, none⟩

/-- A decoded `[10, 9, 8]` (one `RangeDescending`), then `20, 22` with the
bitmap winning at the second: the collapse reads the descending range back
ascending. Reproduced in Rust: `push_after_decode_reorders_a_descending_tail`. -/
theorem decoded_then_pushed_reorders :
    ((pushAll [false, true] (fromSegments [.rdesc 10 3]) [20, 22]).map (fun st => flat st.segs))
      = some [8, 9, 10, 20, 22] := by decide

/-- A decoded `[7, 7]` (one `Repeat`), then `20, 22` with the bitmap winning:
the run starts at the `Repeat` and the `assert!` fires (`none`). Reproduced in
Rust: `push_after_decode_of_a_repeat_hits_the_collapse_assert`. -/
theorem decoded_then_pushed_asserts :
    pushAll [false, true] (fromSegments [.rep 7 2]) [20, 22] = none := by decide

/-- The same sequences pushed from empty are fine (the property that fails is
only the decoded-list precondition). -/
example : ((pushAll [false, false, false, false, true] St.empty [10, 9, 8, 20, 22]).map
    (fun st => flat st.segs)) = some [10, 9, 8, 20, 22] := by decide

end IdListPush
