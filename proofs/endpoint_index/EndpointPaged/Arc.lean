import EndpointPaged.Writes
/-! # `Arc` pages: copy-on-write isolation -/
namespace Falkor.EndpointIndex.Paged
open Falkor.EndpointIndex

/-! ## `Arc` pages: copy-on-write isolation

The store of `Arc` allocations: contents and strong count per allocation id. A
*version* (a `Paged`, or the four tiers of an `EndpointIndex`) is the list of the ids it
points at. `HInv H vs`: each strong count is the number of pointers to it across the live
versions `vs`, and every pointer names an allocated id. -/

structure Heap (β : Type) where
  mem : Nat → β
  rc : Nat → Nat
  next : Nat

def upd {γ : Type} (f : Nat → γ) (a : Nat) (x : γ) : Nat → γ := fun b => if b = a then x else f b

def cnt (a : Nat) (vs : List (List Nat)) : Nat := (vs.map (List.count a)).sum

def HInv {β : Type} (H : Heap β) (vs : List (List Nat)) : Prop :=
  (∀ a, H.rc a = cnt a vs) ∧ ∀ v ∈ vs, ∀ a ∈ v, a < H.next

def contents {β : Type} (H : Heap β) (v : List Nat) : List β := v.map H.mem

variable {β : Type}

/-- `page_mut(p)` followed by the write `f` (:293-305): `Arc::get_mut` succeeds iff the
strong count is 1 (no `Weak` exists); otherwise the page is copied into a fresh `Arc`
(`Arc::from(to_vec())`), the old one loses this pointer, and the copy is written. -/
def pageMut (H : Heap β) (w : List Nat) (p : Nat) (f : β → β) : Heap β × List Nat :=
  let a := w.getD p 0
  if H.rc a = 1 then ({ H with mem := upd H.mem a (f (H.mem a)) }, w)
  else
    let b := H.next
    ({ mem := upd H.mem b (f (H.mem a)), rc := upd (upd H.rc a (H.rc a - 1)) b 1, next := b + 1 },
     w.set p b)

/-- `Paged::clone` (:156): the page *pointers* are cloned, each count goes up. -/
def cloneV (H : Heap β) (w : List Nat) : Heap β := { H with rc := fun a => H.rc a + w.count a }

/-- Dropping a version: each of its pointers releases one count. -/
def dropV (H : Heap β) (w : List Nat) : Heap β := { H with rc := fun a => H.rc a - w.count a }

/-- A fresh `Arc::from(..)` appended (`ensure_pages`' `empty_page`, `from_slots`' pages). -/
def pushFresh (H : Heap β) (w : List Nat) (c : β) : Heap β × List Nat :=
  ({ mem := upd H.mem H.next c, rc := upd H.rc H.next 1, next := H.next + 1 }, w ++ [H.next])

/-- `*page = Arc::from(next)` (`regrow`, :218): a fresh allocation replaces pointer `p`. -/
def replace (H : Heap β) (w : List Nat) (p : Nat) (c : β) : Heap β × List Nat :=
  let a := w.getD p 0
  ({ mem := upd H.mem H.next c, rc := upd (upd H.rc a (H.rc a - 1)) H.next 1, next := H.next + 1 },
   w.set p H.next)

theorem cnt_cons (a : Nat) (v : List Nat) (vs : List (List Nat)) : cnt a (v :: vs) = v.count a + cnt a vs := by
  simp [cnt]

theorem cnt_zero {a : Nat} : ∀ {vs : List (List Nat)}, (∀ v ∈ vs, a ∉ v) → cnt a vs = 0
  | [], _ => rfl
  | v :: vs, h => by
    rw [cnt_cons, List.count_eq_zero.2 (h v (by simp)), cnt_zero (fun u hu => h u (by simp [hu]))]

theorem not_mem_of_cnt_zero {a : Nat} {vs : List (List Nat)} (h : cnt a vs = 0) : ∀ v ∈ vs, a ∉ v := by
  intro v hv ha
  induction vs with
  | nil => simp at hv
  | cons u us ih =>
    rw [cnt_cons] at h
    rcases List.mem_cons.1 hv with rfl | hv
    · have := List.count_pos_iff.2 ha; omega
    · exact ih (by omega) hv

theorem contents_frame (H H' : Heap β) (v : List Nat) (h : ∀ a ∈ v, H'.mem a = H.mem a) :
    contents H' v = contents H v := List.map_congr_left h

theorem unique_index {w : List Nat} {p j : Nat} (hp : p < w.length) (hj : j < w.length)
    (h1 : w.count w[p] = 1) (hjp : w[j] = w[p]) : j = p := by
  by_cases hne : j = p
  · exact hne
  exfalso
  have hc := List.count_set (a := w[p] + 1) (b := w[p]) hp
  simp at hc
  rw [h1] at hc
  have : w[p] ∈ w.set p (w[p] + 1) := by
    rw [List.mem_iff_getElem]
    exact ⟨j, by simpa using hj, by rw [List.getElem_set, if_neg (Ne.symm hne)]; exact hjp⟩
  have := List.count_pos_iff.2 this
  omega

theorem fresh_not_mem (H : Heap β) (vs : List (List Nat)) (hlt : ∀ v ∈ vs, ∀ a ∈ v, a < H.next) :
    ∀ v ∈ vs, ∀ a ∈ v, a ≠ H.next := fun v hv a ha => Nat.ne_of_lt (hlt v hv a ha)

/-- The copy branch shared by `page_mut` (shared page) and `regrow` (`*page = Arc::from`):
pointer `p` moves to a fresh allocation holding `x`. -/
theorem copy_spec (H : Heap β) (w : List Nat) (os : List (List Nat)) (p : Nat) (x : β)
    (hI : HInv H (w :: os)) (hp : p < w.length) :
    HInv ⟨upd H.mem H.next x, upd (upd H.rc w[p] (H.rc w[p] - 1)) H.next 1, H.next + 1⟩
      (w.set p H.next :: os) ∧
    (∀ o ∈ os, contents ⟨upd H.mem H.next x, upd (upd H.rc w[p] (H.rc w[p] - 1)) H.next 1, H.next + 1⟩ o
      = contents H o) ∧
    contents ⟨upd H.mem H.next x, upd (upd H.rc w[p] (H.rc w[p] - 1)) H.next 1, H.next + 1⟩
      (w.set p H.next) = (contents H w).set p x := by
  obtain ⟨hrc, hlt⟩ := hI
  have hmem : w[p] ∈ w := List.getElem_mem hp
  have hcw : 0 < w.count w[p] := List.count_pos_iff.2 hmem
  have hb := fresh_not_mem H _ hlt
  have hab : w[p] ≠ H.next := hb w (by simp) _ hmem
  have hnc : cnt H.next os = 0 := cnt_zero (fun o ho hm => hb o (by simp [ho]) _ hm rfl)
  have hnw : w.count H.next = 0 := List.count_eq_zero.2 (fun hm => hb w (by simp) _ hm rfl)
  refine ⟨⟨?_, ?_⟩, ?_, ?_⟩
  · intro c
    show upd (upd H.rc w[p] (H.rc w[p] - 1)) H.next 1 c = _
    rw [cnt_cons, List.count_set hp]
    simp only [upd]
    by_cases hc : c = H.next
    · subst hc
      have e1 : (w[p] == H.next) = false := by simp [hab]
      simp [e1, hnw, hnc]
    · rw [if_neg hc]
      by_cases hca : c = w[p]
      · subst hca
        have e2 : (H.next == w[p]) = false := by simp [Ne.symm hab]
        have := hrc w[p]
        rw [cnt_cons] at this
        simp [e2]
        omega
      · have e1 : (w[p] == c) = false := by simp [Ne.symm hca]
        have e2 : (H.next == c) = false := by simp [Ne.symm hc]
        rw [if_neg hca, hrc c, cnt_cons]
        simp [e1, e2]
  · intro v hv a ha
    show a < H.next + 1
    rcases List.mem_cons.1 hv with rfl | hv
    · rcases List.mem_or_eq_of_mem_set ha with ha | rfl
      · have := hlt _ (by simp) a ha; omega
      · omega
    · have := hlt v (by simp [hv]) a ha; omega
  · intro o ho
    apply contents_frame
    intro a ha
    simp [upd, hb o (by simp [ho]) a ha]
  · apply List.ext_getElem
    · simp [contents]
    · intro j h1' h2'
      simp only [contents, List.getElem_map, List.getElem_set, upd]
      by_cases hj : p = j
      · subst hj; simp
      · rw [if_neg hj, if_neg hj]
        have hjl : j < w.length := by simpa [contents] using h1'
        simp [hb w (by simp) _ (List.getElem_mem hjl)]

/-- **Copy-on-write isolation (`page_mut`).** A write through one version changes that
version at exactly page `p` — to `f` of what it held — changes no other live version
(the snapshot a reader holds), and keeps every count exact. -/
theorem pageMut_spec (H : Heap β) (w : List Nat) (os : List (List Nat)) (p : Nat) (f : β → β)
    (hI : HInv H (w :: os)) (hp : p < w.length) :
    HInv (pageMut H w p f).1 ((pageMut H w p f).2 :: os) ∧
    (∀ o ∈ os, contents (pageMut H w p f).1 o = contents H o) ∧
    contents (pageMut H w p f).1 (pageMut H w p f).2 = (contents H w).set p (f (H.mem w[p])) := by
  have hgd : w.getD p 0 = w[p] := by simp [List.getD_eq_getElem?_getD, hp]
  unfold pageMut
  simp only [hgd]
  split
  · -- unique: write in place
    rename_i h1
    obtain ⟨hrc, hlt⟩ := hI
    have hmem : w[p] ∈ w := List.getElem_mem hp
    have hcw : 0 < w.count w[p] := List.count_pos_iff.2 hmem
    have hrca := hrc w[p]
    rw [cnt_cons] at hrca
    have hw1 : w.count w[p] = 1 := by omega
    have hos : cnt w[p] os = 0 := by omega
    have hno := not_mem_of_cnt_zero hos
    refine ⟨⟨hrc, hlt⟩, ?_, ?_⟩
    · intro o ho
      apply contents_frame
      intro a ha
      have : a ≠ w[p] := fun e => hno o ho (e ▸ ha)
      simp [upd, this]
    · apply List.ext_getElem
      · simp [contents]
      · intro j h1' h2'
        simp only [contents, List.getElem_map, List.getElem_set, upd]
        by_cases hj : p = j
        · subst hj; simp
        · rw [if_neg hj]
          have hjl : j < w.length := by simpa [contents] using h1'
          have : w[j] ≠ w[p] := fun e => hj (unique_index hp hjl hw1 e).symm
          simp [this]
  · exact copy_spec H w os p _ hI hp

/-- `Paged::clone`: the clone reads the same slots and the counts stay exact. -/
theorem cloneV_spec (H : Heap β) (w : List Nat) (os : List (List Nat)) (hI : HInv H (w :: os)) :
    HInv (cloneV H w) (w :: w :: os) ∧ contents (cloneV H w) w = contents H w := by
  obtain ⟨hrc, hlt⟩ := hI
  refine ⟨⟨fun a => ?_, fun v hv a ha => ?_⟩, rfl⟩
  · simp [cloneV, hrc a, cnt_cons]; omega
  · rcases List.mem_cons.1 hv with rfl | hv
    · exact hlt v (by simp) a ha
    · exact hlt v hv a ha

/-- Dropping a version releases exactly its pointers. -/
theorem dropV_spec (H : Heap β) (w : List Nat) (os : List (List Nat)) (hI : HInv H (w :: os)) :
    HInv (dropV H w) os := by
  obtain ⟨hrc, hlt⟩ := hI
  refine ⟨fun a => ?_, fun v hv a ha => hlt v (by simp [hv]) a ha⟩
  simp [dropV, hrc a, cnt_cons]

/-- A fresh page appended: the writer gains it, nobody else sees anything. -/
theorem pushFresh_spec (H : Heap β) (w : List Nat) (os : List (List Nat)) (c : β) (hI : HInv H (w :: os)) :
    HInv (pushFresh H w c).1 ((pushFresh H w c).2 :: os) ∧
    (∀ o ∈ os, contents (pushFresh H w c).1 o = contents H o) ∧
    contents (pushFresh H w c).1 (pushFresh H w c).2 = contents H w ++ [c] := by
  obtain ⟨hrc, hlt⟩ := hI
  have hb := fresh_not_mem H _ hlt
  have hnc : cnt H.next os = 0 := cnt_zero (fun o ho hm => hb o (by simp [ho]) _ hm rfl)
  have hnw : w.count H.next = 0 := List.count_eq_zero.2 (fun hm => hb w (by simp) _ hm rfl)
  refine ⟨⟨fun a => ?_, fun v hv a ha => ?_⟩, fun o ho => ?_, ?_⟩
  · show upd H.rc H.next 1 a = cnt a ((w ++ [H.next]) :: os)
    rw [cnt_cons, List.count_append]
    simp only [upd]
    by_cases h : a = H.next
    · subst h; simp [hnw, hnc]
    · rw [if_neg h, hrc a, cnt_cons]
      have : ([H.next].count a) = 0 := by simp [Ne.symm h]
      omega
  · show a < H.next + 1
    rcases List.mem_cons.1 hv with rfl | hv
    · rcases List.mem_append.1 ha with ha | ha
      · have := hlt _ (by simp) a ha; omega
      · simp at ha; omega
    · have := hlt v (by simp [hv]) a ha; omega
  · apply contents_frame; intro a ha; simp [pushFresh, upd, hb o (by simp [ho]) a ha]
  · show (w ++ [H.next]).map (upd H.mem H.next c) = w.map H.mem ++ [c]
    rw [List.map_append]
    simp only [List.map_cons, List.map_nil, upd, ite_true]
    congr 1
    apply List.map_congr_left; intro a ha; exact if_neg (hb w (by simp) a ha)

/-- `regrow`'s `*page = Arc::from(next)`: pointer `p` now names fresh contents `c`; the
page it replaced loses one count and stays intact for any snapshot still holding it. -/
theorem replace_spec (H : Heap β) (w : List Nat) (os : List (List Nat)) (p : Nat) (c : β)
    (hI : HInv H (w :: os)) (hp : p < w.length) :
    HInv (replace H w p c).1 ((replace H w p c).2 :: os) ∧
    (∀ o ∈ os, contents (replace H w p c).1 o = contents H o) ∧
    contents (replace H w p c).1 (replace H w p c).2 = (contents H w).set p c := by
  have hgd : w.getD p 0 = w[p] := by simp [List.getD_eq_getElem?_getD, hp]
  unfold replace
  simp only [hgd]
  exact copy_spec H w os p c hI hp

end Falkor.EndpointIndex.Paged
