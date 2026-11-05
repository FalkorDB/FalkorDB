import FalkorIndexLayer.LeafDispatch
/-
# `CompactIndexedLeaf::block_copy_merge` (`compact_indexed.rs`, origin/main 8743953a8)

The list-level algorithm (`bcmL`): for each batch entry, gallop to the first
leaf entry not below it, emit the leaf entries skipped over, then the batch
entry unless it repeats the previous batch entry or equals that leaf entry;
finally the leaf tail. `bcmL_eq_merge`: it emits exactly `merge_sorted`.
-/
namespace IndexLayer.Leaf

def pltB (a b : P) : Bool := lexLe a b && !(a == b)

theorem pltB_iff (a b : P) : pltB a b = true ↔ lt lexLe a b := by
  simp [pltB, lt]

def Lg (L : List P) (i : Nat) : P := L.getD i (0, 0)

def bcmL (L : List P) : Nat → List P → Option P → List (P × Option Nat)
  | pos, [], _ => (List.range' pos (L.length - pos)).map (fun i => (Lg L i, some i))
  | pos, x :: bs, prev =>
    let p := gallop (fun i => pltB (Lg L i) x) pos L.length
    let dup := prev == some x || (decide (p < L.length) && Lg L p == x)
    (List.range' pos (p - pos)).map (fun i => (Lg L i, some i)) ++ (if dup then [] else [(x, none)]) ++
      bcmL L p bs (some x)

/-- Two strictly increasing lists with the same members are equal. -/
theorem sorted_unique {α : Type} [DecidableEq α] (le : α → α → Bool) (O : LinOrd le) :
    ∀ (l1 l2 : List α), l1.Pairwise (lt le) → l2.Pairwise (lt le) → (∀ x, x ∈ l1 ↔ x ∈ l2) → l1 = l2
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => absurd ((h b).mpr (by simp)) (by simp)
  | a :: _, [], _, _, h => absurd ((h a).mp (by simp)) (by simp)
  | a :: as, b :: bs, h1, h2, h => by
    have p1 := List.pairwise_cons.mp h1
    have p2 := List.pairwise_cons.mp h2
    have hab : a = b := by
      rcases List.mem_cons.mp ((h a).mp (by simp)) with e | ha
      · exact e
      · rcases List.mem_cons.mp ((h b).mpr (by simp)) with e | hb
        · exact e.symm
        · have x1 := (p2.1 a ha).1
          have x2 := (p1.1 b hb).1
          exact O.antisymm _ _ x2 x1
    subst hab
    rw [sorted_unique le O as bs p1.2 p2.2 (fun x => ?_)]
    constructor
    · intro hx
      rcases List.mem_cons.mp ((h x).mp (List.mem_cons_of_mem _ hx)) with e | e
      · subst e; exact absurd rfl (p1.1 x hx).2
      · exact e
    · intro hx
      rcases List.mem_cons.mp ((h x).mpr (List.mem_cons_of_mem _ hx)) with e | e
      · subst e; exact absurd rfl (p2.1 x hx).2
      · exact e

theorem lt_trans_le (O : LinOrd lexLe) (a b c : P) (h1 : lt lexLe a b) (h2 : lexLe b c = true) : lt lexLe a c := by
  refine ⟨O.trans _ _ _ h1.1 h2, fun e => ?_⟩
  subst e
  exact h1.2 (O.antisymm _ _ h1.1 h2)

theorem not_lt_le (O : LinOrd lexLe) (a b : P) (h : ¬ lt lexLe a b) : lexLe b a = true := by
  rcases O.total a b with h1 | h1
  · by_cases e : a = b
    · subst e; exact O.refl _
    · exact absurd ⟨h1, e⟩ h
  · exact h1

/-- Facts about the strictly sorted leaf list `L`. -/
theorem Lg_eq (L : List P) (i : Nat) (hi : i < L.length) : Lg L i = L[i] := getD_of_lt _ _ _ hi

theorem mem_drop_iff (L : List P) (pos : Nat) (x : P) :
    x ∈ L.drop pos ↔ ∃ i, pos ≤ i ∧ ∃ hi : i < L.length, L[i] = x := by
  rw [List.mem_iff_getElem]
  constructor
  · rintro ⟨j, hj, rfl⟩; exact ⟨pos + j, by omega, by simp at hj; omega, by simp⟩
  · rintro ⟨i, h1, h2, rfl⟩; exact ⟨i - pos, by simp; omega, by simp; congr 1; omega⟩

theorem range_mem (L : List P) (a c : Nat) (x : P) (hc : c ≤ L.length) :
    x ∈ ((List.range' a (c - a)).map (fun i => (Lg L i, some i))).map (·.1) ↔
      ∃ i, a ≤ i ∧ i < c ∧ ∃ hi : i < L.length, L[i] = x := by
  simp only [List.map_map, List.mem_map, List.mem_range', Function.comp]
  constructor
  · rintro ⟨i, ⟨j, hj, rfl⟩, rfl⟩
    have h3 : a + j < L.length := by omega
    exact ⟨a + j, by omega, by omega, h3, by simp only [Nat.one_mul]; exact (Lg_eq L _ h3).symm⟩
  · rintro ⟨i, h1, h2, hi, rfl⟩
    exact ⟨i, ⟨i - a, by omega, by simp only [Nat.one_mul]; omega⟩, Lg_eq L i hi⟩

theorem range_eq_slice (L : List P) (a k : Nat) (h : a + k ≤ L.length) :
    ((List.range' a k).map (fun i => (Lg L i, some i))).map (·.1) = (L.drop a).take k := by
  apply List.ext_getElem (by simp; omega)
  intro i h1 h2
  simp only [List.map_map, List.getElem_map, List.getElem_range', Function.comp, Nat.one_mul,
    List.getElem_take, List.getElem_drop]
  exact Lg_eq L _ (by simp at h1; omega)

theorem le_lt_trans (O : LinOrd lexLe) (a b c : P) (h1 : lexLe a b = true) (h2 : lt lexLe b c) : lt lexLe a c := by
  refine ⟨O.trans _ _ _ h1 h2.1, fun e => ?_⟩
  subst e
  exact h2.2 (O.antisymm _ _ h2.1 h1)

theorem sorted_le (L : List P) (hL : L.Pairwise (lt lexLe)) (i j : Nat) (hj : j < L.length) (hij : i ≤ j) :
    lexLe L[i] L[j] = true := by
  rcases Nat.lt_or_ge i j with h | h
  · exact (List.pairwise_iff_getElem.mp hL i j (by omega) hj h).1
  · have : i = j := by omega
    subst this; exact lexLe_lin.refl _

/-- **PROVEN** (the block-copy merge is a merge): on a strictly sorted leaf and a
non-decreasing batch, the emitted value sequence is strictly increasing and
contains exactly the leaf tail and the batch (minus a repeat of `prev`). -/
theorem bcmL_spec (L : List P) (hL : L.Pairwise (lt lexLe)) :
    ∀ (batch : List P) (pos : Nat) (prev : Option P),
    batch.Pairwise (fun a b => lexLe a b = true) → pos ≤ L.length →
    (∀ y, prev = some y → (∀ z ∈ batch, lexLe y z = true) ∧ ∀ i (hi : i < L.length), pos ≤ i → lexLe y L[i] = true) →
    let out := (bcmL L pos batch prev).map (·.1)
    out.Pairwise (lt lexLe) ∧ ∀ x, x ∈ out ↔ x ∈ L.drop pos ∨ (x ∈ batch ∧ prev ≠ some x)
  | [], pos, prev, _, hp, _ => by
    simp only [bcmL]
    refine ⟨?_, fun x => ?_⟩
    · rw [range_eq_slice L pos _ (by omega)]
      exact hL.sublist ((List.take_sublist _ _).trans (List.drop_sublist _ _))
    · rw [range_mem L pos L.length x (Nat.le_refl _), mem_drop_iff]
      simp only [List.not_mem_nil, false_and, or_false]
      constructor
      · rintro ⟨i, h1, -, h3, h4⟩; exact ⟨i, h1, h3, h4⟩
      · rintro ⟨i, h1, h3, h4⟩; exact ⟨i, h1, h3, h3, h4⟩
  | x :: bs, pos, prev, hb, hp, hprev => by
    have O := lexLe_lin
    have hb' := List.pairwise_cons.mp hb
    have hm : Mono (fun i => pltB (Lg L i) x) pos L.length := by
      intro i j _ hij hj hpj
      rw [pltB_iff] at hpj ⊢
      rw [Lg_eq L j hj] at hpj; rw [Lg_eq L i (by omega)]
      exact le_lt_trans O _ _ _ (sorted_le L hL i j hj hij) hpj
    obtain ⟨g1, g2, g3, g4⟩ := gallop_spec _ pos L.length hp hm
    simp only [bcmL]
    generalize hgp : gallop (fun i => pltB (Lg L i) x) pos L.length = p at g1 g2 g3 g4
    have below : ∀ i (hi : i < L.length), pos ≤ i → i < p → lt lexLe L[i] x := by
      intro i hi h1 h2; have := g3 i h1 h2; rw [pltB_iff, Lg_eq L i hi] at this; exact this
    have above : ∀ i (hi : i < L.length), p ≤ i → lexLe x L[i] = true := by
      intro i hi h1
      have := g4 i h1 hi
      apply not_lt_le O
      intro hl; rw [← pltB_iff, ← Lg_eq L i hi] at hl; rw [hl] at this; cases this
    obtain ⟨r1, r2⟩ := bcmL_spec L hL bs p (some x) hb'.2 g2
      (fun y hy => by cases hy; exact ⟨hb'.1, fun i hi h => above i hi h⟩)
    simp only [List.map_append] at r1 r2 ⊢
    rw [range_eq_slice L pos (p - pos) (by omega)]
    -- what the rest can contain
    have restGe : ∀ c, c ∈ (bcmL L p bs (some x)).map (·.1) → lexLe x c = true := by
      intro c hc
      rcases (r2 c).mp hc with h | ⟨h, -⟩
      · obtain ⟨i, h1, h2, rfl⟩ := (mem_drop_iff L p c).mp h; exact above i h2 h1
      · exact hb'.1 c h
    have segLt : ∀ a, a ∈ (L.drop pos).take (p - pos) → lt lexLe a x := by
      intro a ha
      obtain ⟨j, hj, rfl⟩ := List.getElem_of_mem ha
      simp only [List.getElem_take, List.getElem_drop]
      simp at hj
      exact below _ (by omega) (by omega) (by omega)
    have segIn : ∀ y, y ∈ (L.drop pos).take (p - pos) ↔ ∃ i, pos ≤ i ∧ i < p ∧ ∃ hi : i < L.length, L[i] = y := by
      intro y
      rw [← range_eq_slice L pos (p - pos) (by omega), range_mem L pos p y g2]
    constructor
    · -- sortedness
      refine List.pairwise_append.mpr ⟨List.pairwise_append.mpr ⟨?_, ?_, ?_⟩, r1, ?_⟩
      · exact hL.sublist ((List.take_sublist _ _).trans (List.drop_sublist _ _))
      · split <;> simp
      · intro a ha c hc
        split at hc
        · simp at hc
        · simp at hc; subst hc; exact segLt a ha
      · intro a ha c hc
        rcases List.mem_append.mp ha with ha | ha
        · exact lt_trans_le O _ _ _ (segLt a ha) (restGe c hc)
        · split at ha
          · simp at ha
          · next hnd =>
            simp at ha; subst ha
            refine ⟨restGe c hc, fun e => ?_⟩
            subst e
            rcases (r2 a).mp hc with h | ⟨-, h⟩
            · obtain ⟨i, h1, h2, h3⟩ := (mem_drop_iff L p a).mp h
              have hip : i = p := by
                rcases Nat.lt_or_ge p i with hlt | hge
                · have := List.pairwise_iff_getElem.mp hL p i (by omega) h2 hlt
                  rw [h3] at this
                  exact absurd (O.antisymm _ _ this.1 (above p (by omega) (Nat.le_refl _))) this.2
                · omega
              subst hip
              apply hnd
              simp [Lg_eq L i h2, h3, h2]
            · exact h rfl
    · intro y
      rw [List.mem_append, List.mem_append, segIn, r2, mem_drop_iff, mem_drop_iff]
      constructor
      · rintro ((⟨i, h1, -, h3, h4⟩ | hmid) | hrest)
        · exact Or.inl ⟨i, h1, h3, h4⟩
        · split at hmid
          · simp at hmid
          · next hnd =>
            simp at hmid; subst hmid
            right; refine ⟨by simp, fun e => hnd (by simp [e])⟩
        · rcases hrest with ⟨i, h1, h3, h4⟩ | ⟨h1, h2⟩
          · exact Or.inl ⟨i, by omega, h3, h4⟩
          · right; refine ⟨by simp [h1], fun e => ?_⟩
            obtain ⟨hz, -⟩ := hprev y e
            have hyx : lexLe y x = true := hz x (by simp)
            have hxy : lexLe x y = true := hb'.1 y h1
            exact h2 (by rw [O.antisymm _ _ hxy hyx])
      · rintro (⟨i, h1, h3, h4⟩ | ⟨hy, hpv⟩)
        · rcases Nat.lt_or_ge i p with h | h
          · exact Or.inl (Or.inl ⟨i, h1, h, h3, h4⟩)
          · exact Or.inr (Or.inl ⟨i, h, h3, h4⟩)
        · rcases List.mem_cons.mp hy with rfl | hy
          · -- the batch head
            by_cases hd : (prev == some y || (decide (p < L.length) && Lg L p == y)) = true
            · simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, decide_eq_true_eq] at hd
              rcases hd with hd | ⟨hpn, hd⟩
              · exact absurd hd hpv
              · exact Or.inr (Or.inl ⟨p, Nat.le_refl _, hpn, by rw [← Lg_eq L p hpn]; exact hd⟩)
            · exact Or.inl (Or.inr (by simp [hd]))
          · by_cases e : y = x
            · subst e
              by_cases hd : (prev == some y || (decide (p < L.length) && Lg L p == y)) = true
              · simp only [Bool.or_eq_true, beq_iff_eq, Bool.and_eq_true, decide_eq_true_eq] at hd
                rcases hd with hd | ⟨hpn, hd⟩
                · exact absurd hd hpv
                · exact Or.inr (Or.inl ⟨p, Nat.le_refl _, hpn, by rw [← Lg_eq L p hpn]; exact hd⟩)
              · exact Or.inl (Or.inr (by simp [hd]))
            · exact Or.inr (Or.inr ⟨hy, fun h' => e (by cases h'; rfl)⟩)

/-- **PROVEN**: from the start of the leaf, the block-copy merge emits exactly
`merge_sorted(leaf, batch)`. -/
theorem bcmL_eq_merge (L batch : List P) (hL : L.Pairwise (lt lexLe))
    (hb : batch.Pairwise (fun a b => lexLe a b = true)) :
    (bcmL L 0 batch none).map (·.1) = mergeD lexLe L batch none := by
  obtain ⟨s1, s2⟩ := bcmL_spec L hL batch 0 none hb (Nat.zero_le _) (fun y h => by cases h)
  have hL' : L.Pairwise (fun a b => lexLe a b = true) := hL.imp (fun h => h.1)
  obtain ⟨m1, -, m3, m4⟩ := mergeD_spec lexLe lexLe_lin L batch none hL' hb (fun _ _ => rfl)
  apply sorted_unique lexLe lexLe_lin _ _ s1 m4
  intro x
  rw [s2, List.drop_zero]
  constructor
  · rintro (h | ⟨h, -⟩)
    · exact (m3 x (Or.inl h)).elim id (fun h' => by cases h')
    · exact (m3 x (Or.inr h)).elim id (fun h' => by cases h')
  · intro h
    rcases m1 x h with h' | h'
    · exact Or.inl h'
    · exact Or.inr ⟨h', by simp⟩

/-! ## Gallop/partition-point congruence: only indices below `count` are read -/

theorem pp_congr (p q : Nat → Bool) : ∀ (f lo hi : Nat), (∀ i, lo ≤ i → i < hi → p i = q i) →
    pp p f lo hi = pp q f lo hi
  | 0, _, _, _ => rfl
  | f + 1, lo, hi, h => by
    simp only [pp]
    split
    · next hlt =>
      rw [h (lo + (hi - lo) / 2) (by omega) (by omega)]
      split
      · exact pp_congr p q f _ hi (fun i h1 h2 => h i (by omega) h2)
      · exact pp_congr p q f lo _ (fun i h1 h2 => h i h1 (by omega))
    · rfl

theorem gallopStep_congr (p q : Nat → Bool) (start count : Nat) (h : ∀ i, i < count → p i = q i) :
    ∀ f step, gallopStep p start count f step = gallopStep q start count f step
  | 0, _ => rfl
  | f + 1, step => by
    simp only [gallopStep]
    by_cases hc : start + step < count
    · rw [h _ hc]; split <;> (try rfl); exact gallopStep_congr p q start count h f _
    · simp [hc]

theorem gallop_congr (p q : Nat → Bool) (start count : Nat) (h : ∀ i, i < count → p i = q i) :
    gallop p start count = gallop q start count := by
  unfold gallop
  simp only [gallopStep_congr p q start count h]
  unfold partitionPoint
  exact pp_congr p q _ _ _ (fun i _ h2 => h i (by omega))

end IndexLayer.Leaf
