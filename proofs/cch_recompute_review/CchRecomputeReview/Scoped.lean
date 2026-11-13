import CchRecomputeReview.Merge

/-!
# The scoped recompute: model and its one-step invariant

`recustomize_seeded`'s pop loop (`partial.rs:163-255` @08c, identical at the stack tip `cch/10-rdb-v20`),
read with 03 §6b's changes: the heap carries arcs and pops one with the least lower rank (`(lo, slot)` in the
code, `(lo, hi, id)` in 03 §6b "Algorithm" 2: any key that orders by `lo` first satisfies `PickSpec`); the apex
list is the merge of `down(lo)` and `down(hi)` (`mergeL`); propagation enqueues the top arc of every triangle
whose apex is `lo` (`pushAll`, `partial.rs:241-254`; 03 §6b "Propagation"); `queued` keeps an arc from being
pushed twice (`:163, :250`).

The change test and the writes are the code's: `best != metric[slot]` (`:221-236`), that is `!beq`. With
`R = Eq` they are bitwise only when `beq` is (finding R-F1).
-/

namespace CchRecompute

variable {W : Type}

/-- 03 §6b "Propagation": the top arc `(min(hi,w), max(hi,w))`; the code's `if hi < w { (hi, w) } else { (w, hi) }`
(`partial.rs:246`). -/
def pairOf (hi w : Nat) : Nat × Nat := if hi < w then (hi, w) else (w, hi)

structure St (W : Type) where
  M : Metric W
  heap : List (Nat × Nat)
  queued : Nat → Nat → Bool

def qset (q : Nat → Nat → Bool) (a b : Nat) : Nat → Nat → Bool :=
  fun x y => if x = a ∧ y = b then true else q x y

/-- `partial.rs:241-254`: for `w ∈ up(lo) ∖ {hi}`, push `pairOf hi w` unless already queued. -/
def pushAll (hi : Nat) : List Nat → List (Nat × Nat) → (Nat → Nat → Bool) →
    List (Nat × Nat) × (Nat → Nat → Bool)
  | [], h, q => (h, q)
  | w :: ws, h, q =>
    if w = hi then pushAll hi ws h q
    else
      let d := pairOf hi w
      if q d.1 d.2 then pushAll hi ws h q
      else pushAll hi ws (h ++ [d]) (qset q d.1 d.2)

/-- `partial.rs:178-219`: reseed (`known` or `seed(u, v)`, both `S` here) and relax over the merge. -/
def recompAt (o : Ops W) (T : Topo) (S : Seeds W) (M : Metric W) (lo hi : Nat) : Entry W :=
  mergeL o M lo hi ((T.down lo).length + (T.down hi).length) (T.down lo) (T.down hi) (seedE S lo hi)

/-- `partial.rs:221-222`: `best_up != upward_weight[slot] || best_dn != downward_weight[slot]`. -/
def changedB (o : Ops W) (e old : Entry W) : Bool := !(o.beq e.up old.up) || !(o.beq e.dn old.dn)

/-- `partial.rs:225-236`: each of the four values is written only where `!=` says it differs. -/
def writeNew (o : Ops W) (e old : Entry W) : Entry W :=
  { up := if !(o.beq e.up old.up) then e.up else old.up
    dn := if !(o.beq e.dn old.dn) then e.dn else old.dn
    mup := if e.mup ≠ old.mup then e.mup else old.mup
    mdn := if e.mdn ≠ old.mdn then e.mdn else old.mdn }

/-- One pop (`partial.rs:168-255`): `a` has been popped (erased from the heap); recompute it, write, and if a
weight changed, propagate (`:241-254`). -/
def popStep (o : Ops W) (T : Topo) (S : Seeds W) (st : St W) (a : Nat × Nat) : St W :=
  let e := recompAt o T S st.M a.1 a.2
  let old := st.M a.1 a.2
  if changedB o e old then
    ⟨st.M.set a.1 a.2 (writeNew o e old), (pushAll a.2 (T.up a.1) (st.heap.erase a) st.queued).1,
      (pushAll a.2 (T.up a.1) (st.heap.erase a) st.queued).2⟩
  else ⟨st.M.set a.1 a.2 (writeNew o e old), st.heap.erase a, st.queued⟩

/-- `while let Some(..) = heap.pop()` (`partial.rs:168`), with fuel. -/
def run (o : Ops W) (T : Topo) (S : Seeds W) (pick : List (Nat × Nat) → Option (Nat × Nat)) :
    Nat → St W → St W
  | 0, st => st
  | k + 1, st =>
    match pick st.heap with
    | none => st
    | some a => run o T S pick k (popStep o T S st a)

/-- `partial.rs:163-164`: the heap and `queued` both start as the front. -/
def init (M0 : Metric W) (fr : List (Nat × Nat)) : St W :=
  ⟨M0, fr, fun a b => decide ((a, b) ∈ fr)⟩

/-- What a heap's `pop` must give: a member with the least lower rank. -/
structure PickSpec (pick : List (Nat × Nat) → Option (Nat × Nat)) : Prop where
  none_nil : ∀ h, pick h = none → h = []
  mem : ∀ h a, pick h = some a → a ∈ h
  least : ∀ h a, pick h = some a → ∀ b ∈ h, a.1 ≤ b.1

/-! ## `pushAll` -/

theorem pushAll_spec (hi : Nat) : ∀ (ws : List Nat) (h : List (Nat × Nat)) (q : Nat → Nat → Bool),
    ∃ new, (pushAll hi ws h q).1 = h ++ new ∧ new.Nodup ∧
      (∀ p ∈ new, q p.1 p.2 = false ∧ ∃ w ∈ ws, w ≠ hi ∧ p = pairOf hi w) ∧
      (∀ a b, (pushAll hi ws h q).2 a b = true ↔ q a b = true ∨ (a, b) ∈ new) ∧
      (∀ w ∈ ws, w ≠ hi → (pushAll hi ws h q).2 (pairOf hi w).1 (pairOf hi w).2 = true)
  | [], h, q => ⟨[], by simp [pushAll], by simp, by simp, by simp [pushAll], by simp⟩
  | w :: ws, h, q => by
    by_cases hw : w = hi
    · obtain ⟨new, e1, e2, e3, e4, e5⟩ := pushAll_spec hi ws h q
      refine ⟨new, ?_, e2, ?_, ?_, ?_⟩
      · simp only [pushAll, hw, ↓reduceIte]; exact e1
      · intro p hp
        obtain ⟨hq, w', hw', hne, hpe⟩ := e3 p hp
        exact ⟨hq, w', List.mem_cons_of_mem _ hw', hne, hpe⟩
      · simp only [pushAll, hw, ↓reduceIte]; exact e4
      · intro w' hw' hne
        simp only [pushAll, hw, ↓reduceIte]
        rcases List.mem_cons.mp hw' with rfl | hw'
        · exact absurd hw hne
        · exact e5 w' hw' hne
    · by_cases hq : q (pairOf hi w).1 (pairOf hi w).2 = true
      · obtain ⟨new, e1, e2, e3, e4, e5⟩ := pushAll_spec hi ws h q
        have hP : pushAll hi (w :: ws) h q = pushAll hi ws h q := by
          simp only [pushAll, hw, hq, ↓reduceIte]
        refine ⟨new, by rw [hP]; exact e1, e2, ?_, by rw [hP]; exact e4, ?_⟩
        · intro p hp
          obtain ⟨hq', w', hw', hne, hpe⟩ := e3 p hp
          exact ⟨hq', w', List.mem_cons_of_mem _ hw', hne, hpe⟩
        · intro w' hw' hne
          rw [hP]
          rcases List.mem_cons.mp hw' with rfl | hw'
          · exact (e4 _ _).mpr (Or.inl hq)
          · exact e5 w' hw' hne
      · obtain ⟨new, e1, e2, e3, e4, e5⟩ :=
          pushAll_spec hi ws (h ++ [pairOf hi w]) (qset q (pairOf hi w).1 (pairOf hi w).2)
        have hq' : q (pairOf hi w).1 (pairOf hi w).2 = false := by simpa using hq
        have hP : pushAll hi (w :: ws) h q =
            pushAll hi ws (h ++ [pairOf hi w]) (qset q (pairOf hi w).1 (pairOf hi w).2) := by
          simp only [pushAll, hw, hq', ↓reduceIte, Bool.false_eq_true]
        have hdq : qset q (pairOf hi w).1 (pairOf hi w).2 (pairOf hi w).1 (pairOf hi w).2 = true := by
          simp [qset]
        refine ⟨pairOf hi w :: new, ?_, ?_, ?_, ?_, ?_⟩
        · rw [hP, e1]; simp
        · refine List.nodup_cons.mpr ⟨fun hm => ?_, e2⟩
          have := (e3 _ hm).1
          rw [hdq] at this; exact Bool.noConfusion this
        · intro p hp
          rcases List.mem_cons.mp hp with rfl | hp
          · exact ⟨hq', w, List.mem_cons_self, hw, rfl⟩
          · obtain ⟨hq2, w', hw', hne, hpe⟩ := e3 p hp
            refine ⟨?_, w', List.mem_cons_of_mem _ hw', hne, hpe⟩
            simp only [qset] at hq2
            split at hq2
            · exact Bool.noConfusion hq2
            · exact hq2
        · intro a b
          rw [hP, e4]
          simp only [qset, List.mem_cons]
          constructor
          · rintro (h1 | h1)
            · split at h1
              · rename_i hab; right; left; obtain ⟨rfl, rfl⟩ := hab; rfl
              · left; exact h1
            · right; right; exact h1
          · rintro (h1 | h1 | h1)
            · left; split
              · rfl
              · exact h1
            · left
              have hab : a = (pairOf hi w).1 ∧ b = (pairOf hi w).2 := by rw [← h1]; exact ⟨rfl, rfl⟩
              simp [hab]
            · right; exact h1
        · intro w' hw' hne
          rw [hP]
          rcases List.mem_cons.mp hw' with rfl | hw'
          · exact (e4 _ _).mpr (Or.inl hdq)
          · exact e5 w' hw' hne

/-! ## The invariant -/

/-- Popped: queued, and no longer on the heap. -/
def popped (st : St W) (a b : Nat) : Prop := st.queued a b = true ∧ (a, b) ∉ st.heap

/-- The invariant of the pop loop, for target `F` and starting metric `M0`. -/
structure SInv (T : Topo) (R : W → W → Prop) (M0 F : Metric W) (fr : List (Nat × Nat)) (st : St W) : Prop where
  heap_arc : ∀ p ∈ st.heap, p.2 ∈ T.up p.1
  heap_queued : ∀ p ∈ st.heap, st.queued p.1 p.2 = true
  heap_nodup : st.heap.Nodup
  front_queued : ∀ p ∈ fr, st.queued p.1 p.2 = true
  queued_arc : ∀ a b, st.queued a b = true → b ∈ T.up a
  /-- Only popped arcs were written. -/
  untouched : ∀ a b, ¬popped st a b → st.M a b = M0 a b
  /-- A popped arc holds its target. -/
  done : ∀ a b, b ∈ T.up a → popped st a b → ER R (st.M a b) (F a b)
  /-- A popped arc whose target differs from its start has queued every arc that reads it. -/
  notify : ∀ x y, y ∈ T.up x → popped st x y → ¬WR R (M0 x y) (F x y) →
    ∀ w ∈ T.up x, w ≠ y → st.queued (pairOf y w).1 (pairOf y w).2 = true
  /-- Pops never go back down in lower rank. -/
  mono : ∀ a b, popped st a b → ∀ p ∈ st.heap, a ≤ p.1

theorem pairOf_lt {y z : Nat} (h : y < z) : pairOf y z = (y, z) := by simp [pairOf, h]

theorem pairOf_gt {y z : Nat} (h : y < z) : pairOf z y = (y, z) := by
  simp [pairOf, Nat.not_lt.mpr (Nat.le_of_lt h)]

/-- **Below the least lower rank on the heap, everything is final.** An arc there that was never popped kept
its start value, which by `Cons` and the induction already equals the target. -/
theorem below_threshold {o : Ops W} {T : Topo} (hV : T.Valid) {R : W → W → Prop} (hA : Agree o R)
    {S : Seeds W} {F M0 : Metric W} (hF : IsFull o T S F) {fr : List (Nat × Nat)}
    (hCons : ∀ a b, b ∈ T.up a → (a, b) ∉ fr → ER R (M0 a b) (recomp o T S M0 a b))
    {st : St W} (hI : SInv T R M0 F fr st) {t : Nat} (ht : ∀ p ∈ st.heap, t ≤ p.1) :
    ∀ k, ∀ y < k, y < t → ∀ z, z ∈ T.up y →
      ER R (st.M y z) (F y z) ∧ (¬popped st y z → WR R (M0 y z) (F y z)) := by
  intro k
  induction k with
  | zero => intro y hy; omega
  | succ k ih =>
    intro y hyk hyt z hyz
    by_cases hp : popped st y z
    · exact ⟨hI.done y z hyz hp, fun h => absurd hp h⟩
    · have hnh : (y, z) ∉ st.heap := fun hm => by have := ht _ hm; simp at this; omega
      have hnq : st.queued y z = false := by
        cases h : st.queued y z
        · rfl
        · exact absurd ⟨h, hnh⟩ hp
      have hnf : (y, z) ∉ fr := fun hm => by
        have := hI.front_queued _ hm; simp at this; rw [hnq] at this; exact Bool.noConfusion this
      have hyz' := hV.above y z hyz
      -- every input of (y, z) starts at its target
      have hin : ∀ x ∈ T.commonDown y z, WR R (M0 x y) (F x y) ∧ WR R (M0 x z) (F x z) := by
        intro x hx
        obtain ⟨hxy, hy, hz⟩ := mem_commonDown.mp hx
        have hxk : x < k := by omega
        have hxt : x < t := by omega
        refine ⟨?_, ?_⟩
        · by_cases hpx : popped st x y
          · apply Classical.byContradiction
            intro hn
            have := hI.notify x y hy hpx hn z hz (Nat.ne_of_gt hyz')
            rw [pairOf_lt hyz'] at this; rw [hnq] at this; exact Bool.noConfusion this
          · exact (ih x hxk hxt y hy).2 hpx
        · by_cases hpx : popped st x z
          · apply Classical.byContradiction
            intro hn
            have := hI.notify x z hz hpx hn y hy (Nat.ne_of_lt hyz')
            rw [pairOf_gt hyz'] at this; rw [hnq] at this; exact Bool.noConfusion this
          · exact (ih x hxk hxt z hz).2 hpx
      have hc := recomp_congr hA T S (M := M0) (M' := F) hin
      rw [← hF y z hyz] at hc
      have hER : ER R (M0 y z) (F y z) := ER.trans hA (hCons y z hyz hnf) hc
      rw [hI.untouched y z hp]
      exact ⟨hER, fun _ => hER.1⟩

end CchRecompute
