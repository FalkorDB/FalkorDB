import CchRecomputeReview.Scoped

/-!
# The pop loop keeps its invariant, ends, and ends at the full customization

* `popStep_inv`: one pop of an arc with the least lower rank keeps `SInv`, and pops exactly that arc.
* `run_spec`: with more fuel than arcs the loop empties its heap (each arc pops at most once).
* `scoped_agrees` (**Theorem G**): if the start metric is consistent off the front (`Cons`), the loop ends with
  every arc agreeing with any full customization `F`, weights under `R` and middles exactly.
-/

namespace CchRecompute

variable {W : Type}

/-- What a pop does to the heap and `queued`, whichever branch it takes. -/
theorem popStep_shape (o : Ops W) (T : Topo) (S : Seeds W) (st : St W) (a : Nat × Nat) :
    ∃ new : List (Nat × Nat),
      (popStep o T S st a).heap = st.heap.erase a ++ new ∧ new.Nodup ∧
      (∀ p ∈ new, st.queued p.1 p.2 = false ∧ ∃ w ∈ T.up a.1, w ≠ a.2 ∧ p = pairOf a.2 w) ∧
      (∀ x y, (popStep o T S st a).queued x y = true ↔ st.queued x y = true ∨ (x, y) ∈ new) ∧
      (popStep o T S st a).M =
        st.M.set a.1 a.2 (writeNew o (recompAt o T S st.M a.1 a.2) (st.M a.1 a.2)) ∧
      (changedB o (recompAt o T S st.M a.1 a.2) (st.M a.1 a.2) = true →
        ∀ w ∈ T.up a.1, w ≠ a.2 → (popStep o T S st a).queued (pairOf a.2 w).1 (pairOf a.2 w).2 = true) := by
  by_cases hc : changedB o (recompAt o T S st.M a.1 a.2) (st.M a.1 a.2) = true
  · obtain ⟨new, e1, e2, e3, e4, e5⟩ := pushAll_spec a.2 (T.up a.1) (st.heap.erase a) st.queued
    have hP : popStep o T S st a =
        ⟨st.M.set a.1 a.2 (writeNew o (recompAt o T S st.M a.1 a.2) (st.M a.1 a.2)),
          (pushAll a.2 (T.up a.1) (st.heap.erase a) st.queued).1,
          (pushAll a.2 (T.up a.1) (st.heap.erase a) st.queued).2⟩ := by
      simp only [popStep, hc, ↓reduceIte]
    rw [hP]
    exact ⟨new, e1, e2, e3, e4, rfl, fun _ => e5⟩
  · have hc' : changedB o (recompAt o T S st.M a.1 a.2) (st.M a.1 a.2) = false := by simpa using hc
    have hP : popStep o T S st a =
        ⟨st.M.set a.1 a.2 (writeNew o (recompAt o T S st.M a.1 a.2) (st.M a.1 a.2)), st.heap.erase a,
          st.queued⟩ := by
      simp only [popStep, hc', Bool.false_eq_true, ↓reduceIte]
    rw [hP]
    exact ⟨[], by simp, List.nodup_nil, by simp, by simp, rfl, fun h => absurd h hc⟩

/-- `if mid != old { set(mid) }` always leaves `mid`. -/
theorem writeNew_mids (o : Ops W) (e old : Entry W) :
    (writeNew o e old).mup = e.mup ∧ (writeNew o e old).mdn = e.mdn := by
  simp only [writeNew]
  constructor
  · split
    · rfl
    · rename_i h; exact (Classical.not_not.mp h).symm
  · split
    · rfl
    · rename_i h; exact (Classical.not_not.mp h).symm

/-- A value kept because `==` said "unchanged" agrees with the new one. -/
theorem writeNew_agrees {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {e old f : Entry W}
    (he : ER R e f) : ER R (writeNew o e old) f := by
  obtain ⟨m1, m2⟩ := writeNew_mids o e old
  refine ⟨⟨?_, ?_⟩, m1.trans he.2.1, m2.trans he.2.2⟩
  · simp only [writeNew]
    cases hb : o.beq e.up old.up
    · simpa using he.1.1
    · simpa using hA.trans (hA.symm (hA.beq_sound hb)) he.1.1
  · simp only [writeNew]
    cases hb : o.beq e.dn old.dn
    · simpa using he.1.2
    · simpa using hA.trans (hA.symm (hA.beq_sound hb)) he.1.2

/-- If `==` says neither weight changed, the old weights agree with the new ones. -/
theorem unchanged_agrees {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {e old : Entry W}
    (h : changedB o e old = false) : WR R old e := by
  simp only [changedB, Bool.or_eq_false_iff, Bool.not_eq_eq_eq_not, Bool.not_false] at h
  exact ⟨hA.symm (hA.beq_sound h.1), hA.symm (hA.beq_sound h.2)⟩

theorem pairOf_arc {T : Topo} (hC : T.Chordal) {lo hi w : Nat} (hh : hi ∈ T.up lo) (hw : w ∈ T.up lo)
    (hne : w ≠ hi) : (pairOf hi w).2 ∈ T.up (pairOf hi w).1 := by
  unfold pairOf
  split
  · rename_i h; exact hC lo hi w hh hw h
  · rename_i h; exact hC lo w hi hw hh (by omega)

theorem pairOf_fst_gt {T : Topo} (hV : T.Valid) {lo hi w : Nat} (hh : hi ∈ T.up lo) (hw : w ∈ T.up lo) :
    lo < (pairOf hi w).1 := by
  have := hV.above lo hi hh; have := hV.above lo w hw
  unfold pairOf; split <;> simp <;> omega

/-- **One pop keeps the invariant.** -/
theorem popStep_inv {o : Ops W} {T : Topo} (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK)
    {R : W → W → Prop} (hA : Agree o R) {S : Seeds W} {F M0 : Metric W} (hF : IsFull o T S F)
    {fr : List (Nat × Nat)} (hCons : ∀ a b, b ∈ T.up a → (a, b) ∉ fr → ER R (M0 a b) (recomp o T S M0 a b))
    {st : St W} (hI : SInv T R M0 F fr st) {a : Nat × Nat} (ha : a ∈ st.heap)
    (hleast : ∀ b ∈ st.heap, a.1 ≤ b.1) :
    SInv T R M0 F fr (popStep o T S st a) ∧
      (∀ x y, popped (popStep o T S st a) x y ↔ popped st x y ∨ (x, y) = a) := by
  obtain ⟨lo, hi⟩ := a
  obtain ⟨new, eh, enodup, enew, eq, eM, epush⟩ := popStep_shape o T S st (lo, hi)
  simp only at eh enew eq eM epush
  have harc : hi ∈ T.up lo := hI.heap_arc _ ha
  have hqa : st.queued lo hi = true := hI.heap_queued _ ha
  have hnotpop : ¬popped st lo hi := fun h => h.2 ha
  have hold : st.M lo hi = M0 lo hi := hI.untouched lo hi hnotpop
  -- the recomputed entry agrees with the target: its inputs sit below the threshold `lo`
  have hthr := below_threshold hV hA hF hCons hI (t := lo) (fun p hp => hleast p hp)
  have hE : ER R (recompAt o T S st.M lo hi) (F lo hi) := by
    rw [recompAt, merge_eq_recomp o hV hD S st.M harc, hF lo hi harc]
    apply recomp_congr hA
    intro x hx
    obtain ⟨hxlo, h1, h2⟩ := mem_commonDown.mp hx
    exact ⟨(hthr (x + 1) x (Nat.lt_succ_self _) hxlo lo h1).1.1,
      (hthr (x + 1) x (Nat.lt_succ_self _) hxlo hi h2).1.1⟩
  have hnew_not_q : ∀ p ∈ new, st.queued p.1 p.2 = false := fun p hp => (enew p hp).1
  have ha_notin_new : (lo, hi) ∉ new := fun hm => by
    have := hnew_not_q _ hm; simp at this; rw [hqa] at this; exact Bool.noConfusion this
  have ha_notin_erase : (lo, hi) ∉ st.heap.erase (lo, hi) := fun hm =>
    ((List.Nodup.mem_erase_iff hI.heap_nodup).mp hm).1 rfl
  -- popped status: exactly one more arc
  have hpop : ∀ x y, popped (popStep o T S st (lo, hi)) x y ↔ popped st x y ∨ (x, y) = (lo, hi) := by
    intro x y
    simp only [popped, eh, eq, List.mem_append]
    constructor
    · rintro ⟨hq, hn⟩
      rcases hq with hq | hq
      · by_cases hxy : (x, y) = (lo, hi)
        · exact Or.inr hxy
        · left
          refine ⟨hq, fun hm => hn (Or.inl ((List.mem_erase_of_ne hxy).mpr hm))⟩
      · exact absurd (Or.inr hq) hn
    · rintro (⟨hq, hn⟩ | hxy)
      · refine ⟨Or.inl hq, ?_⟩
        rintro (hm | hm)
        · exact hn (List.mem_of_mem_erase hm)
        · have := hnew_not_q _ hm; simp at this; rw [hq] at this; exact Bool.noConfusion this
      · cases hxy
        exact ⟨Or.inl hqa, fun hm => hm.elim ha_notin_erase ha_notin_new⟩
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, hpop⟩
  · -- heap_arc
    intro p hp
    rw [eh] at hp
    rcases List.mem_append.mp hp with hp | hp
    · exact hI.heap_arc p (List.mem_of_mem_erase hp)
    · obtain ⟨_, w, hw, hne, rfl⟩ := enew p hp
      exact pairOf_arc hC harc hw hne
  · -- heap_queued
    intro p hp
    rw [eh] at hp
    rw [eq]
    rcases List.mem_append.mp hp with hp | hp
    · exact Or.inl (hI.heap_queued p (List.mem_of_mem_erase hp))
    · exact Or.inr hp
  · -- heap_nodup
    rw [eh]
    refine List.nodup_append.mpr ⟨hI.heap_nodup.erase _, enodup, ?_⟩
    intro p hp p' hp' hpp
    subst hpp
    have := hI.heap_queued p (List.mem_of_mem_erase hp)
    rw [hnew_not_q p hp'] at this; exact Bool.noConfusion this
  · -- front_queued
    intro p hp; rw [eq]; exact Or.inl (hI.front_queued p hp)
  · -- queued_arc
    intro x y hq
    rcases (eq x y).mp hq with hq | hq
    · exact hI.queued_arc x y hq
    · obtain ⟨_, w, hw, hne, hpe⟩ := enew _ hq
      have := pairOf_arc hC harc hw hne
      rw [← hpe] at this; exact this
  · -- untouched
    intro x y hn
    have hn' : ¬popped st x y ∧ (x, y) ≠ (lo, hi) := by
      refine ⟨fun h => hn ((hpop x y).mpr (Or.inl h)), fun h => hn ((hpop x y).mpr (Or.inr h))⟩
    rw [eM, Metric.set_ne _ _ (fun ⟨h1, h2⟩ => hn'.2 (by rw [h1, h2]))]
    exact hI.untouched x y hn'.1
  · -- done
    intro x y hxy hp
    rcases (hpop x y).mp hp with hp | hp
    · have hne : (x, y) ≠ (lo, hi) := fun h => by cases h; exact hnotpop hp
      rw [eM, Metric.set_ne _ _ (fun ⟨h1, h2⟩ => hne (by rw [h1, h2]))]
      exact hI.done x y hxy hp
    · cases hp
      rw [eM, Metric.set_same]
      exact writeNew_agrees hA hE
  · -- notify
    intro x y hxy hp hn w hw hne
    rcases (hpop x y).mp hp with hp | hp
    · rw [eq]; exact Or.inl (hI.notify x y hxy hp hn w hw hne)
    · cases hp
      apply epush _ w hw hne
      apply Classical.byContradiction
      intro hc
      have hc' : changedB o (recompAt o T S st.M lo hi) (st.M lo hi) = false := by simpa using hc
      apply hn
      rw [← hold]
      exact WR.trans hA (unchanged_agrees hA hc') hE.1
  · -- mono
    intro x y hp p hp'
    rw [eh] at hp'
    have hx : x ≤ lo := by
      rcases (hpop x y).mp hp with hp | hp
      · exact hI.mono x y hp _ ha
      · cases hp; exact Nat.le_refl _
    rcases List.mem_append.mp hp' with hp' | hp'
    · rcases (hpop x y).mp hp with hp | hp
      · exact hI.mono x y hp p (List.mem_of_mem_erase hp')
      · cases hp; exact hleast p (List.mem_of_mem_erase hp')
    · obtain ⟨_, w, hw, _, rfl⟩ := enew p hp'
      have := pairOf_fst_gt hV harc hw
      omega

/-! ## Termination: each arc pops at most once -/

/-- Every arc, row by row: `arcList T` has `arc_count` entries. -/
def arcList (T : Topo) : List (Nat × Nat) := (List.range T.n).flatMap (fun lo => (T.up lo).map (lo, ·))

def arcCount (T : Topo) : Nat := (arcList T).length

theorem mem_arcList {T : Topo} (hV : T.Valid) {lo hi : Nat} (h : hi ∈ T.up lo) : (lo, hi) ∈ arcList T := by
  have hlo : lo < T.n := Nat.lt_trans (hV.above lo hi h) (hV.bound lo hi h)
  simp only [arcList, List.mem_flatMap, List.mem_range, List.mem_map]
  exact ⟨lo, hlo, hi, h, rfl⟩

def poppedB (st : St W) (a b : Nat) : Bool := st.queued a b && !decide ((a, b) ∈ st.heap)

theorem poppedB_iff (st : St W) (a b : Nat) : poppedB st a b = true ↔ popped st a b := by
  simp [poppedB, popped]

/-- Arcs not yet popped. -/
def unpopped (T : Topo) (st : St W) : Nat := (arcList T).countP (fun p => !poppedB st p.1 p.2)

theorem countP_lt {α : Type} (p q : α → Bool) :
    ∀ (l : List α), (∀ x ∈ l, q x = true → p x = true) → (∃ x ∈ l, p x = true ∧ q x = false) →
      l.countP q < l.countP p
  | [], _, ⟨_, hx, _⟩ => by simp at hx
  | y :: t, hle, ⟨x, hx, hpx, hqx⟩ => by
    have mono : ∀ (l : List α), (∀ x ∈ l, q x = true → p x = true) → l.countP q ≤ l.countP p := by
      intro l hl
      induction l with
      | nil => simp
      | cons z t ih =>
        simp only [List.countP_cons]
        have := ih (fun x hx => hl x (List.mem_cons_of_mem _ hx))
        have hz := hl z List.mem_cons_self
        cases hq : q z <;> cases hp : p z <;> simp_all <;> omega
    simp only [List.countP_cons]
    rcases List.mem_cons.mp hx with rfl | hx
    · have := mono t (fun z hz => hle z (List.mem_cons_of_mem _ hz))
      simp [hpx, hqx]; omega
    · have := countP_lt p q t (fun z hz => hle z (List.mem_cons_of_mem _ hz)) ⟨x, hx, hpx, hqx⟩
      have hy := hle y List.mem_cons_self
      cases hq : q y <;> cases hp : p y <;> simp_all <;> omega

theorem unpopped_lt {o : Ops W} {T : Topo} (hV : T.Valid) {S : Seeds W} {st : St W} {a : Nat × Nat}
    (harc : a.2 ∈ T.up a.1) (ha : ¬popped st a.1 a.2)
    (hpop : ∀ x y, popped (popStep o T S st a) x y ↔ popped st x y ∨ (x, y) = a) :
    unpopped T (popStep o T S st a) < unpopped T st := by
  apply countP_lt
  · intro p _ h
    simp only [Bool.not_eq_eq_eq_not, Bool.not_true] at h ⊢
    cases h' : poppedB st p.1 p.2
    · rfl
    · have := (poppedB_iff _ _ _).mp h'
      have := (poppedB_iff _ _ _).mpr ((hpop p.1 p.2).mpr (Or.inl this))
      rw [this] at h; exact Bool.noConfusion h
  · refine ⟨a, mem_arcList hV harc, ?_, ?_⟩
    · cases h : poppedB st a.1 a.2
      · rfl
      · exact absurd ((poppedB_iff _ _ _).mp h) ha
    · have := (poppedB_iff _ _ _).mpr ((hpop a.1 a.2).mpr (Or.inr rfl))
      simp [this]

/-- **The loop ends.** With more fuel than arcs not yet popped, the heap ends empty, and the invariant holds. -/
theorem run_spec {o : Ops W} {T : Topo} (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK)
    {R : W → W → Prop} (hA : Agree o R) {S : Seeds W} {F M0 : Metric W} (hF : IsFull o T S F)
    {fr : List (Nat × Nat)} (hCons : ∀ a b, b ∈ T.up a → (a, b) ∉ fr → ER R (M0 a b) (recomp o T S M0 a b))
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) :
    ∀ (fuel : Nat) (st : St W), SInv T R M0 F fr st → unpopped T st < fuel →
      (run o T S pick fuel st).heap = [] ∧ SInv T R M0 F fr (run o T S pick fuel st)
  | 0, _, _, h => absurd h (Nat.not_lt_zero _)
  | k + 1, st, hI, hf => by
    simp only [run]
    cases hp : pick st.heap with
    | none => exact ⟨hpick.none_nil _ hp, hI⟩
    | some a =>
      have ha := hpick.mem _ _ hp
      obtain ⟨hI', hpop⟩ := popStep_inv hV hC hD hA hF hCons hI ha (hpick.least _ _ hp)
      have hlt := unpopped_lt (o := o) (S := S) hV (hI.heap_arc a ha) (fun h => h.2 ha) hpop
      exact run_spec hV hC hD hA hF hCons hpick k _ hI' (by omega)

/-- The front's start state satisfies the invariant: nothing is popped yet. -/
theorem init_inv {T : Topo} {R : W → W → Prop} {M0 F : Metric W} {fr : List (Nat × Nat)}
    (hfr_arc : ∀ p ∈ fr, p.2 ∈ T.up p.1) (hfr_nodup : fr.Nodup) : SInv T R M0 F fr (init M0 fr) := by
  have hnp : ∀ a b, ¬popped (init M0 fr) a b := fun a b ⟨hq, hn⟩ => hn (by simpa [init] using hq)
  refine ⟨hfr_arc, fun p hp => by simpa [init] using hp, hfr_nodup, fun p hp => by simpa [init] using hp,
    fun a b hq => hfr_arc (a, b) (by simpa [init] using hq), fun _ _ _ => rfl,
    fun a b _ h => absurd h (hnp a b), fun x y _ h => absurd h (hnp x y), fun a b h => absurd h (hnp a b)⟩

theorem unpopped_le (T : Topo) (st : St W) : unpopped T st ≤ arcCount T := List.countP_le_length

/-- **Theorem G.** Start from any metric `M0` that is consistent off the front: every arc not in the front
already holds, under `R`, what it reads from `M0` (`Cons`). Then the pop loop, run to an empty heap (any fuel
above `arc_count`), leaves every arc agreeing with any full customization `F` for the seeds `S`: weights
under `R`, middles exactly. Premises: P-rows (`hV`), P-chordal (`hC`), P-transpose (`hD`), a heap that pops a
least lower rank (`hpick`), and the agreement laws (`hA`). -/
theorem scoped_agrees {o : Ops W} {T : Topo} (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK)
    {R : W → W → Prop} (hA : Agree o R) {S : Seeds W} {F M0 : Metric W} (hF : IsFull o T S F)
    {fr : List (Nat × Nat)} (hfr_arc : ∀ p ∈ fr, p.2 ∈ T.up p.1) (hfr_nodup : fr.Nodup)
    (hCons : ∀ a b, b ∈ T.up a → (a, b) ∉ fr → ER R (M0 a b) (recomp o T S M0 a b))
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount T < fuel) :
    (run o T S pick fuel (init M0 fr)).heap = [] ∧
      ∀ lo hi, hi ∈ T.up lo → ER R ((run o T S pick fuel (init M0 fr)).M lo hi) (F lo hi) := by
  obtain ⟨hnil, hI⟩ := run_spec hV hC hD hA hF hCons hpick fuel (init M0 fr) (init_inv hfr_arc hfr_nodup)
    (Nat.lt_of_le_of_lt (unpopped_le T _) hfuel)
  refine ⟨hnil, fun lo hi harc => ?_⟩
  have ht : ∀ p ∈ (run o T S pick fuel (init M0 fr)).heap, lo + 1 ≤ p.1 := by rw [hnil]; simp
  exact (below_threshold hV hA hF hCons hI ht (lo + 1) lo (Nat.lt_succ_self _) (Nat.lt_succ_self _) hi harc).1

end CchRecompute
