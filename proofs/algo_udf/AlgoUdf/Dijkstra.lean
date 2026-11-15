import AlgoUdf.DijkstraInv
/-! # `dijkstra_single_path`: correctness and termination (see `AlgoUdf.DijkstraModel`) -/
namespace AlgoUdf.Dijkstra
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

theorem inv_step {g : G} {src tgt : Nat} {st st' : St} (I : Inv g src tgt noEx st)
    (T : TgtInv tgt st) (h : Step g tgt st st') : Inv g src tgt noEx st' ∧ TgtInv tgt st' := by
  cases h with
  | @skip w v hd hm hmin hs =>
    refine ⟨⟨?_, I.lbl, ?_, ?_, I.setLbl, I.ordLt, ?_, I.edge, I.fr, I.opt⟩, T⟩
    · intro h; have := (I.h0 h).1; have h2 := (I.h0 h).2 v
      rw [this] at hm; simp at hm; rw [hm.2, h] at hs; cases hs
    · intro w' x hx; exact I.heapE w' x (List.mem_of_mem_erase hx)
    · intro x l hl hx
      refine mem_erase_of_ne (I.latest x l hl hx) ?_
      intro he; injection he with _ he; subst he; rw [hs] at hx; cases hx
    · intro u w' x hu hx; exact I.mono u w' x hu (List.mem_of_mem_erase hx)
  | @reach w v hd hm hmin hs hvt =>
    have I1 := (inv_settle I hd hm hmin hs).1
    refine ⟨⟨I1.h0, I1.lbl, I1.heapE, I1.latest, I1.setLbl, I1.ordLt, I1.mono, ?_, ?_, I1.opt⟩,
      ⟨fun _ => rfl, fun _ => by subst hvt; simp [settle]⟩⟩
    · intro h; cases h
    · intro h; cases h
  | @relax w v hd hm hmin hs hvt =>
    obtain ⟨I1, hDv⟩ := inv_settle I hd hm hmin hs
    have hv : (settle st v (st.heap.erase (w, v))).settled v = true := by simp [settle]
    have hle : ∀ u, (settle st v (st.heap.erase (w, v))).settled u = true →
        D src (settle st v (st.heap.erase (w, v))).labels u ≤ w := by
      intro u hu
      simp only [settle] at hu ⊢
      by_cases huv : u = v
      · subst huv; omega
      · rw [upd_ne _ _ _ _ huv] at hu; exact I.mono u w v hu hm
    have I1' : Inv g src tgt (fun x r' => x = v ∧ r' ∉ ([] : List Rel))
        (settle st v (st.heap.erase (w, v))) := by
      refine ⟨I1.h0, I1.lbl, I1.heapE, I1.latest, I1.setLbl, I1.ordLt, I1.mono, I1.edge, ?_, I1.opt⟩
      intro hd' x y r c hx he hne; exact I1.fr hd' x y r c hx he (fun h => hne ⟨h, by simp⟩)
    have F := inv_fold (g.rels v) [] (settle st v (st.heap.erase (w, v))) rfl I1' hd hv
      (by exact hDv) hle
    obtain ⟨I2, hd2⟩ := F
    refine ⟨⟨I2.h0, I2.lbl, I2.heapE, I2.latest, I2.setLbl, I2.ordLt, I2.mono, I2.edge, ?_, I2.opt⟩, ?_⟩
    · intro hd' x y r c hx he _
      exact I2.fr hd' x y r c hx he (fun h => h.2 (h.1 ▸ he.1))
    · unfold TgtInv; rw [hd2]
      constructor
      · intro ht
        exfalso
        have hs2 := (fold_settled (g := g) (g.rels v) v w _) ▸ ht
        simp only [settle] at hs2
        rw [upd_ne _ _ _ _ (Ne.symm hvt)] at hs2
        have := T.mp hs2; rw [this] at hd; cases hd
      · intro h; cases h

theorem inv_star {g : G} {src tgt : Nat} {st : St} (h : Star g tgt (init src) st) :
    Inv g src tgt noEx st ∧ TgtInv tgt st := by
  generalize hi : init src = s0 at h
  induction h with
  | refl => subst hi; exact ⟨inv_init g src tgt, by simp [TgtInv, init]⟩
  | tail _ hs ih => exact inv_step ih.1 ih.2 hs

/-- The parent walk from any settled node succeeds and spells out a walk of
weight `D`. -/
theorem walkBack_ok {g : G} {src tgt : Nat} {st : St} (I : Inv g src tgt noEx st) :
    ∀ n cur acc, st.settled cur = true → st.ord cur < n →
      ∃ es, walkBack st.labels src n cur acc = some (es ++ acc) ∧
        Walk g src es cur (D src st.labels cur) := by
  intro n
  induction n with
  | zero => intro _ _ _ h; omega
  | succ n ih =>
    intro cur acc hc ho
    unfold walkBack
    by_cases hcs : cur = src
    · subst hcs; simp only [if_pos]; exact ⟨[], by simp, by simp [D]; exact .nil _⟩
    · rw [if_neg hcs]
      cases hl : st.labels cur with
      | none => exact absurd hl (I.setLbl cur hc hcs)
      | some l =>
        simp only
        obtain ⟨_, hp, hord, c, he, hw⟩ := I.lbl cur l hl
        have := hord hc
        obtain ⟨es, h1, h2⟩ := ih l.parent (l.edge :: acc) hp (by omega)
        refine ⟨es ++ [l.edge], by rw [h1]; simp, ?_⟩
        rw [D_lbl hcs hl, hw]; exact h2.snoc he

/-- **Correctness of `dijkstra_single_path`.** -/
theorem dijkstra_correct (g : G) (src tgt : Nat) (st : St)
    (hst : Star g tgt (init src) st) (hT : Terminal st) (hne : src ≠ tgt) :
    match finish g src tgt st with
    | none => False
    | some none => ∀ es W, ¬ Walk g src es tgt W
    | some (some (es, W, C)) =>
        Walk g src es tgt W ∧ (∀ es' W', Walk g src es' tgt W' → W ≤ W') ∧
        C = (es.map (fun r => g.cost r.2.2)).sum := by
  obtain ⟨I, T⟩ := inv_star hst
  unfold finish
  cases hd : st.done with
  | true =>
    simp only [if_true]
    have hts : st.settled tgt = true := T.mpr hd
    obtain ⟨es, hwb, hw⟩ := walkBack_ok I (st.clock + 1) tgt [] hts (by have := I.ordLt _ hts; omega)
    rw [hwb]
    simp only [List.append_nil]
    cases hl : st.labels tgt with
    | none => exact I.setLbl tgt hts (Ne.symm hne) hl
    | some l =>
      simp only
      have hD : D src st.labels tgt = l.w := D_lbl (Ne.symm hne) hl
      refine ⟨hD ▸ hw, fun es' W' hw' => hD ▸ I.opt tgt es' W' hts hw', by simp⟩
  | false =>
    simp only [Bool.false_eq_true, if_false]
    have hh : st.heap = [] := by rcases hT with h | h; rw [hd] at h; cases h; exact h
    intro es W hw
    have hs : st.settled src = true := by
      cases h : st.settled src with
      | true => rfl
      | false => have := (I.h0 h).1; rw [hh] at this; cases this
    have ht : st.settled tgt = false := by
      cases h : st.settled tgt with
      | false => rfl
      | true => have := T.mp h; rw [hd] at this; cases this
    obtain ⟨z, l, hz, hl, _⟩ := frontier_lemma I hd (fun _ _ => Or.inl id) hw hs ht
    have := I.latest z l hl hz
    rw [hh] at this; cases this

/-! ## Termination -/

theorem exists_min (h : List (Nat × Nat)) (hne : h ≠ []) : ∃ p ∈ h, IsMin h p.1 := by
  induction h with
  | nil => exact absurd rfl hne
  | cons x t ih =>
    by_cases ht : t = []
    · subst ht; exact ⟨x, by simp, by intro y hy; simp at hy; subst hy; exact Nat.le_refl _⟩
    · obtain ⟨p, hp, hmin⟩ := ih ht
      by_cases hle : x.1 ≤ p.1
      · refine ⟨x, by simp, ?_⟩
        intro y hy; simp at hy; rcases hy with rfl | hy
        · exact Nat.le_refl _
        · exact Nat.le_trans hle (hmin y hy)
      · refine ⟨p, List.mem_cons_of_mem _ hp, ?_⟩
        intro y hy; simp at hy; rcases hy with rfl | hy
        · omega
        · exact hmin y hy

/-- A finite node universe closed under the traversal. -/
def Closed (g : G) (U : List Nat) : Prop :=
  ∀ u ∈ U, ∀ r ∈ g.rels u, ∀ v, farEndpoint u r.1 r.2.1 g.dir = some v → v ∈ U

theorem relaxOne_heap {g : G} {U : List Nat} (hU : Closed g U) {v w : Nat} (hv : v ∈ U)
    (st : St) (r : Rel) (hr : r ∈ g.rels v) :
    ∀ p ∈ (relaxOne g v w st r).heap, p ∈ st.heap ∨ p.2 ∈ U := by
  unfold relaxOne
  split
  · intro p hp; exact Or.inl hp
  · rename_i far hf
    split
    · intro p hp; exact Or.inl hp
    · split
      · intro p hp; exact Or.inl hp
      · split
        · intro p hp
          simp only [List.mem_cons] at hp
          rcases hp with rfl | hp
          · exact Or.inr (hU v hv r hr far hf)
          · exact Or.inl hp
        · intro p hp; exact Or.inl hp

theorem fold_heap {g : G} {U : List Nat} (hU : Closed g U) {v w : Nat} (hv : v ∈ U) :
    ∀ (rs : List Rel) (st : St), (∀ r ∈ rs, r ∈ g.rels v) →
      ∀ p ∈ (rs.foldl (relaxOne g v w) st).heap, p ∈ st.heap ∨ p.2 ∈ U := by
  intro rs
  induction rs with
  | nil => intro st _ p hp; exact Or.inl hp
  | cons r t ih =>
    intro st hr p hp
    simp only [List.foldl_cons] at hp
    rcases ih _ (fun r' h => hr r' (List.mem_cons_of_mem _ h)) p hp with h | h
    · exact relaxOne_heap hU hv st r (hr r (by simp)) p h
    · exact Or.inr h

def unsettledIn (U : List Nat) (st : St) : Nat := (U.filter (fun u => st.settled u = false)).length

theorem filter_len_le (l : List Nat) (p q : Nat → Bool) (h : ∀ x, p x = true → q x = true) :
    (l.filter p).length ≤ (l.filter q).length := by
  induction l with
  | nil => simp
  | cons x t ih =>
    simp only [List.filter_cons]
    cases hp : p x <;> cases hq : q x <;> simp <;> try omega
    exact absurd (h x hp) (by rw [hq]; simp)

theorem filter_len_lt (l : List Nat) (p q : Nat → Bool) (h : ∀ x, p x = true → q x = true)
    (x : Nat) (hx : x ∈ l) (hq : q x = true) (hp : p x = false) :
    (l.filter p).length < (l.filter q).length := by
  induction l with
  | nil => cases hx
  | cons y t ih =>
    simp only [List.filter_cons]
    by_cases hyx : y = x
    · subst hyx; rw [hp, hq]; simp; exact Nat.lt_succ_of_le (filter_len_le t p q h)
    · have hx' : x ∈ t := by simp at hx; rcases hx with h | h; exact absurd h.symm hyx; exact h
      have := ih hx'
      cases hp' : p y <;> cases hq' : q y <;> simp <;> try omega
      exact absurd (h y hp') (by rw [hq']; simp)

theorem unsettled_settle_lt (U : List Nat) (st : St) (v : Nat) (h : List (Nat × Nat))
    (hv : v ∈ U) (hs : st.settled v = false) :
    unsettledIn U (settle st v h) < unsettledIn U st := by
  unfold unsettledIn settle
  apply filter_len_lt U _ _ _ v hv (by simp [hs]) (by simp)
  intro x hx
  by_cases hxv : x = v
  · subst hxv; simp at hx
  · simpa [upd_ne _ _ _ _ hxv] using hx

theorem fold_unsettled (g : G) (U : List Nat) (v w : Nat) (rs : List Rel) (st : St) :
    unsettledIn U (rs.foldl (relaxOne g v w) st) = unsettledIn U st := by
  unfold unsettledIn; rw [fold_settled]

theorem fold_done (g : G) (v w : Nat) :
    ∀ (rs : List Rel) (st : St), (rs.foldl (relaxOne g v w) st).done = st.done := by
  intro rs; induction rs with
  | nil => intro _; rfl
  | cons r t ih =>
    intro st; simp only [List.foldl_cons]; rw [ih]
    unfold relaxOne; split
    · rfl
    · split
      · rfl
      · split
        · rfl
        · split <;> rfl

/-- **Termination**: on a finite universe closed under the traversal, the loop
always reaches a terminal state. -/
theorem dijkstra_terminates (g : G) (src tgt : Nat) (U : List Nat) (hU : Closed g U)
    (hsrc : src ∈ U) : ∃ st, Star g tgt (init src) st ∧ Terminal st := by
  suffices H : ∀ k m st, unsettledIn U st = k → st.heap.length = m →
      Star g tgt (init src) st → (∀ p ∈ st.heap, p.2 ∈ U) →
      ∃ st', Star g tgt (init src) st' ∧ Terminal st' from
    H _ _ (init src) rfl rfl (.refl _) (by intro p hp; simp [init] at hp; rw [hp]; exact hsrc)
  intro k
  induction k using Nat.strongRecOn with
  | _ k ihk =>
  intro m
  induction m using Nat.strongRecOn with
  | _ m ihm =>
  intro st hk hm hstar hheap
  by_cases hT : Terminal st
  · exact ⟨st, hstar, hT⟩
  have hd : st.done = false := by
    cases h : st.done; rfl; exact absurd (Or.inl h) hT
  have hne : st.heap ≠ [] := fun h => hT (Or.inr h)
  obtain ⟨⟨w, v⟩, hmem, hmin⟩ := exists_min st.heap hne
  have hvU : v ∈ U := hheap _ hmem
  cases hs : st.settled v with
  | true =>
    have step := Step.skip (g := g) (tgt := tgt) hd hmem hmin hs
    refine ihm (m - 1) ?_ { st with heap := st.heap.erase (w, v) } hk ?_ (hstar.tail step) ?_
    · have := List.length_pos_of_mem hmem; omega
    · simp [List.length_erase_of_mem hmem, hm]
    · intro p hp; exact hheap p (List.mem_of_mem_erase hp)
  | false =>
    by_cases hvt : v = tgt
    · exact ⟨_, hstar.tail (Step.reach hd hmem hmin hs hvt), Or.inl rfl⟩
    · have step := Step.relax (g := g) hd hmem hmin hs hvt
      refine ihk _ ?_ _ _ rfl rfl (hstar.tail step) ?_
      · rw [← hk, fold_unsettled]; exact unsettled_settle_lt U st v _ hvU hs
      · intro p hp
        rcases fold_heap hU hvU (g.rels v) _ (fun _ h => h) p hp with h | h
        · exact hheap p (List.mem_of_mem_erase h)
        · exact h

end AlgoUdf.Dijkstra
