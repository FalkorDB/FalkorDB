import CchTopologyReview.Alloc

/-!
# S2 (03 §6, §2 "Arc identity"): the closure's new arcs, and the append that keeps every invariant

* `S2Ok t N`: what S2's append needs of its new arcs `N`: sorted by `(lo, hi)`, `lo < hi`, none an
  arc yet, both ends held.
* `dhRun`: Davis–Hager's closure as 03 §6 writes it ("add v to up(u) in a host overlay … pop rows by rank
  … push U(j)∖{p} into up(p)"), for **any** order and number of pops. `dh_ends`, `dh_lt`: every arc it
  holds joins two ends of arcs it started from, lower to higher. So `s2New_ok`: the new arcs of any
  closure from the version's arcs plus inserted pairs of held ranks (one, a unit's, or a whole
  adjacency) satisfy `S2Ok`. That discharges `held`,
  which is P's "only ranked rows gain arcs". Whether the closure reaches a valid topology (I2, I4–I6,
  DH01 Thm 4.2) is not this project's (03 §6 cites [R2 §2]).
* `s2_inv`: the append keeps `Inv`.
-/

namespace CchTopo

variable {α : Type}

/-- `(lo, hi)` order. -/
def PLt (p q : Nat × Nat) : Prop := p.1 < q.1 ∨ (p.1 = q.1 ∧ p.2 < q.2)

/-- What S2's append needs of the new arcs `N` of one pass. -/
structure S2Ok (t : Cch α) (N : List (Nat × Nat)) : Prop where
  sorted : N.Pairwise PLt
  lt : ∀ p ∈ N, p.1 < p.2
  new : ∀ p ∈ N, ∀ i, ¬ upE t ⟨p.1, p.2, i⟩
  held : ∀ p ∈ N, (t.nodeOf p.1).isSome ∧ (t.nodeOf p.2).isSome

theorem mem_upEnts : ∀ (c : Nat) (N : List (Nat × Nat)) (e : Ent), e ∈ upEnts c N →
    (e.row, e.col) ∈ N ∧ c ≤ e.id ∧ e.id < c + N.length
  | _, [], e, h => by simp [upEnts] at h
  | c, p :: N, e, h => by
    simp only [upEnts, List.mem_cons] at h
    rcases h with h | h
    · subst h; simp
    · have := mem_upEnts (c + 1) N e h
      exact ⟨List.mem_cons_of_mem _ this.1, by omega, by simp; omega⟩

theorem upEnts_covers : ∀ (c : Nat) (N : List (Nat × Nat)) (p : Nat × Nat), p ∈ N → ∃ i, (⟨p.1, p.2, i⟩ : Ent) ∈ upEnts c N
  | _, [], p, h => by simp at h
  | c, q :: N, p, h => by
    rcases List.mem_cons.1 h with h | h
    · subst h; exact ⟨c, List.mem_cons_self ..⟩
    · obtain ⟨i, hi⟩ := upEnts_covers (c + 1) N p h
      exact ⟨i, List.mem_cons_of_mem _ hi⟩

theorem upEnts_id_asc : ∀ (c : Nat) (N : List (Nat × Nat)), (upEnts c N).Pairwise (fun a b => a.id < b.id)
  | _, [] => by simp [upEnts]
  | c, p :: N => by
    simp only [upEnts, List.pairwise_cons]
    exact ⟨fun b hb => by have := mem_upEnts (c + 1) N b hb; simp; omega, upEnts_id_asc (c + 1) N⟩

theorem upEnts_keys : ∀ (c : Nat) (N : List (Nat × Nat)), N.Pairwise PLt → KSorted (upEnts c N)
  | _, [], _ => by simp [upEnts, KSorted]
  | c, p :: N, h => by
    rw [List.pairwise_cons] at h
    unfold KSorted
    simp only [upEnts, List.pairwise_cons]
    refine ⟨fun b hb => ?_, upEnts_keys (c + 1) N h.2⟩
    have := h.1 _ (mem_upEnts (c + 1) N b hb).1
    unfold PLt at this; unfold keyLt; simpa using this

theorem pairwise_lt_inj {β} {f : β → Nat} : ∀ {l : List β}, l.Pairwise (fun a b => f a < f b) →
    ∀ {a b : β}, a ∈ l → b ∈ l → f a = f b → a = b
  | [], _, _, _, ha, _, _ => by simp at ha
  | x :: l, h, a, b, ha, hb, he => by
    rw [List.pairwise_cons] at h
    rcases List.mem_cons.1 ha with e | e <;> rcases List.mem_cons.1 hb with e' | e'
    · rw [e, e']
    · subst e; have := h.1 b e'; omega
    · subst e'; have := h.1 a e; omega
    · exact pairwise_lt_inj h.2 e e' he

theorem pairwise_not_same_of_ksorted {l : List Ent} (h : KSorted l) : l.Pairwise (fun a b => ¬ sameKey a b) :=
  h.imp (fun hab hs => keyLt_irrefl_of_same hs hab)

theorem s2_ranks (inf : α) (t : Cch α) (N : List (Nat × Nat)) : (s2 inf t N).ranks = t.ranks := rfl

theorem upE_s2 (inf : α) (t : Cch α) (N : List (Nat × Nat)) (_h : AInv t.arcs) (hN : S2Ok t N) (e : Ent) :
    upE (s2 inf t N) e ↔ upE t e ∨ e ∈ upEnts t.cap N := by
  have hf : ∀ x ∈ upEnts t.cap N, ∀ y ∈ t.upD, ¬ sameKey x y := by
    intro x hx y hy hs
    have hx' := (mem_upEnts _ _ x hx).1
    apply hN.new _ hx' y.id
    right
    have : y = ⟨x.row, x.col, y.id⟩ := by cases y; simp_all [sameKey]
    exact this ▸ hy
  have hm := mem_insAll_fresh (upEnts t.cap N) t.upD (pairwise_not_same_of_ksorted (upEnts_keys _ _ hN.sorted)) hf e
  unfold upE
  show inB t.upB e ∨ e ∈ insAll t.upD (upEnts t.cap N) ↔ _
  rw [hm]
  constructor
  · rintro (h | h | h)
    · exact Or.inl (Or.inl h)
    · exact Or.inl (Or.inr h)
    · exact Or.inr h
  · rintro ((h | h) | h)
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr h)

/-- **S2's two batches are fresh and key-distinct**: the `Up` batch (already sorted by key) against
    `Up.delta`, and the `Down` batch (not sorted, T1) against `Down.delta`. -/
theorem s2_batches (t : Cch α) (N : List (Nat × Nat)) (ha : AInv t.arcs) (hN : S2Ok t N) :
    KSorted (upEnts t.cap N) ∧ (∀ x ∈ upEnts t.cap N, ∀ y ∈ t.upD, ¬ sameKey x y) ∧
    ((upEnts t.cap N).map tr).Pairwise (fun a b => ¬ sameKey a b) ∧
    (∀ x ∈ (upEnts t.cap N).map tr, ∀ y ∈ t.downD, ¬ sameKey x y) := by
  have hUk := upEnts_keys t.cap N hN.sorted
  have hf : ∀ x ∈ upEnts t.cap N, ∀ y ∈ t.upD, ¬ sameKey x y := by
    intro x hx y hy hs
    apply hN.new _ (mem_upEnts _ _ x hx).1 y.id
    right
    have : y = ⟨x.row, x.col, y.id⟩ := by cases y; simp_all [sameKey]
    exact this ▸ hy
  refine ⟨hUk, hf, ?_, ?_⟩
  · rw [List.pairwise_map]
    exact (pairwise_not_same_of_ksorted hUk).imp (fun h hs => h ⟨hs.2, hs.1⟩)
  · intro x hx y hy hs
    obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hx
    have : tr y ∈ t.upD := (ha.i7D (tr y)).2 (by rw [tr_tr]; exact hy)
    exact hf u hu (tr y) this ⟨hs.2, hs.1⟩

/-- **S2's append keeps the invariant.** -/
theorem s2_inv (inf : α) (t : Cch α) (N : List (Nat × Nat)) (hI : Inv t) (hN : S2Ok t N) : Inv (s2 inf t N) := by
  obtain ⟨hr, ha, hh⟩ := hI
  let U := upEnts t.cap N
  have hUk : U.Pairwise (fun a b => ¬ sameKey a b) := pairwise_not_same_of_ksorted (upEnts_keys _ _ hN.sorted)
  have hf : ∀ x ∈ U, ∀ y ∈ t.upD, ¬ sameKey x y := by
    intro x hx y hy hs
    apply hN.new _ (mem_upEnts _ _ x hx).1 y.id
    right
    have : y = ⟨x.row, x.col, y.id⟩ := by cases y; simp_all [sameKey]
    exact this ▸ hy
  have hmU : ∀ e, e ∈ insAll t.upD U ↔ e ∈ t.upD ∨ e ∈ U := mem_insAll_fresh U t.upD hUk hf
  have hUkT : (U.map tr).Pairwise (fun a b => ¬ sameKey a b) := by
    rw [List.pairwise_map]; exact hUk.imp (fun h hs => h ⟨hs.2, hs.1⟩)
  have hfT : ∀ x ∈ U.map tr, ∀ y ∈ t.downD, ¬ sameKey x y := by
    intro x hx y hy hs
    obtain ⟨u, hu, rfl⟩ := List.mem_map.1 hx
    have : tr y ∈ t.upD := (ha.i7D (tr y)).2 (by rw [tr_tr]; exact hy)
    exact hf u hu (tr y) this ⟨hs.2, hs.1⟩
  have hmD : ∀ e, e ∈ insAll t.downD (U.map tr) ↔ e ∈ t.downD ∨ e ∈ U.map tr := mem_insAll_fresh _ _ hUkT hfT
  have hup : ∀ e, (s2 inf t N).arcs.upE e ↔ t.arcs.upE e ∨ e ∈ U := fun e => upE_s2 inf t N ha hN e
  have hUid : ∀ e ∈ U, (e.row, e.col) ∈ N ∧ t.cap ≤ e.id ∧ e.id < t.cap + N.length := fun e he => mem_upEnts _ _ e he
  have hUnodup : U.Nodup := (upEnts_id_asc t.cap N).imp (fun h e => by subst e; omega)
  refine ⟨hr, ?_, ?_⟩
  · refine
      { up_lt := ?_, upB_strict := ha.upB_strict, downB_strict := ha.downB_strict,
        upD_sorted := ksorted_insAll _ _ ha.upD_sorted, downD_sorted := ksorted_insAll _ _ ha.downD_sorted,
        up_disj := ?_, down_disj := ?_, i7B := ha.i7B, i7D := ?_, id_lt := ?_, id_inj := ?_,
        met_len := ?_, added_eq := ?_, added_nodup := ?_, added_seq := ?_ }
    · intro e he
      rcases (hup e).1 he with h | h
      · exact ha.up_lt e h
      · exact hN.lt _ (hUid e h).1
    · intro e he i hb
      rcases (hmU e).1 he with h | h
      · exact ha.up_disj e h i hb
      · exact hN.new _ (hUid e h).1 i (Or.inl hb)
    · intro e he i hb
      rcases (hmD e).1 he with h | h
      · exact ha.down_disj e h i hb
      · obtain ⟨u, hu, rfl⟩ := List.mem_map.1 h
        have : inB t.upB ⟨u.row, u.col, i⟩ := (ha.i7B ⟨u.row, u.col, i⟩).2 hb
        exact hN.new _ (hUid u hu).1 i (Or.inl this)
    · intro e
      show e ∈ insAll t.upD U ↔ tr e ∈ insAll t.downD (U.map tr)
      have hi7 : e ∈ t.upD ↔ tr e ∈ t.downD := ha.i7D e
      rw [hmU, hmD, hi7]
      constructor
      · rintro (h | h)
        · exact Or.inl h
        · exact Or.inr (List.mem_map_of_mem h)
      · rintro (h | h)
        · exact Or.inl h
        · obtain ⟨u, hu, hue⟩ := List.mem_map.1 h
          have : u = e := by rw [← tr_tr u, hue, tr_tr]
          exact Or.inr (this ▸ hu)
    · intro e he
      show e.id < t.cap + N.length
      rcases (hup e).1 he with h | h
      · have : e.id < t.cap := ha.id_lt e h; omega
      · exact (hUid e h).2.2
    · intro e e' he he' hid
      rcases (hup e).1 he with h | h <;> rcases (hup e').1 he' with h' | h'
      · exact ha.id_inj e e' h h' hid
      · have : e.id < t.cap := ha.id_lt e h; have := (hUid e' h').2.1; omega
      · have : e'.id < t.cap := ha.id_lt e' h'; have := (hUid e h).2.1; omega
      · exact pairwise_lt_inj (upEnts_id_asc t.cap N) h h' hid
    · show (t.met ++ List.replicate N.length inf).length = t.cap + N.length
      have hml : t.met.length = t.cap := ha.met_len
      rw [List.length_append, List.length_replicate, hml]
    · intro e
      show e ∈ insAll t.upD U ↔ e ∈ ents (t.added ++ [(t.ver, U)])
      have hae : e ∈ t.upD ↔ e ∈ ents t.added := ha.added_eq e
      rw [hmU, ents_append, List.mem_append, hae]
      simp [ents]
    · show (ents (t.added ++ [(t.ver, U)])).Nodup
      rw [ents_append]
      have hs : ents [(t.ver, U)] = U := by simp [ents]
      rw [hs, List.nodup_append]
      refine ⟨ha.added_nodup, hUnodup, fun x hx y hy hxy => ?_⟩
      subst hxy
      have : x ∈ t.upD := (ha.added_eq x).2 hx
      exact hf x hy x this ⟨rfl, rfl⟩
    · intro p hp
      show p.1 ≤ t.ver
      rcases List.mem_append.1 hp with h | h
      · exact ha.added_seq p h
      · simp at h; rw [h]; exact Nat.le_refl _
  · intro e he
    rcases (hup e).1 he with h | h
    · exact hh e h
    · exact hN.held _ (hUid e h).1

/-! ## Davis–Hager's closure, for any pop order -/

/-- `up(j)` in the pattern S2 reads (base, delta and overlay merged). -/
def dhUp (P : List (Nat × Nat)) (j : Nat) : List Nat := (P.filter (fun q => decide (q.1 = j))).map (·.2)

/-- The least element. -/
def minL : List Nat → Option Nat
  | [] => none
  | a :: l => match minL l with
    | none => some a
    | some m => some (min a m)

theorem minL_spec : ∀ (l : List Nat) (m : Nat), minL l = some m → m ∈ l ∧ ∀ x ∈ l, m ≤ x
  | [], m, h => by simp [minL] at h
  | a :: l, m, h => by
    simp only [minL] at h
    cases e : minL l with
    | none =>
      rw [e] at h; cases h
      have : l = [] := by cases l with
        | nil => rfl
        | cons b l' => simp [minL] at e; split at e <;> simp at e
      subst this; simp
    | some m' =>
      rw [e] at h; cases h
      have ih := minL_spec l m' e
      refine ⟨?_, fun x hx => ?_⟩
      · by_cases hle : a ≤ m'
        · rw [Nat.min_eq_left hle]; exact List.mem_cons_self ..
        · rw [Nat.min_eq_right (by omega)]; exact List.mem_cons_of_mem _ ih.1
      · rcases List.mem_cons.1 hx with hx | hx
        · subst hx; exact Nat.min_le_left _ _
        · exact Nat.le_trans (Nat.min_le_right _ _) (ih.2 x hx)

/-- One pop of row `j`: `p = min U(j)`, and `U(j) ∖ {p}` is pushed into `up(p)` (03 §6 S2 row). -/
def dhPush (P : List (Nat × Nat)) (j : Nat) : List (Nat × Nat) :=
  match minL (dhUp P j) with
  | none => P
  | some p => P ++ ((dhUp P j).filter (fun w => decide (w ≠ p))).map (fun w => (p, w))

/-- Any sequence of pops. -/
def dhRun (P : List (Nat × Nat)) (js : List Nat) : List (Nat × Nat) := js.foldl dhPush P

/-- `x` is an end of an arc of `P`. -/
def Ends (P : List (Nat × Nat)) (x : Nat) : Prop := ∃ q ∈ P, q.1 = x ∨ q.2 = x

theorem mem_dhUp (P : List (Nat × Nat)) (j w : Nat) : w ∈ dhUp P j ↔ (j, w) ∈ P := by
  unfold dhUp
  simp only [List.mem_map, List.mem_filter, decide_eq_true_eq]
  constructor
  · rintro ⟨q, ⟨hq, hj⟩, hw⟩; subst hj hw; exact hq
  · intro h; exact ⟨(j, w), ⟨h, rfl⟩, rfl⟩

theorem mem_dhPush (P : List (Nat × Nat)) (j : Nat) (q : Nat × Nat) (hq : q ∈ dhPush P j) :
    q ∈ P ∨ ∃ p, minL (dhUp P j) = some p ∧ q.1 = p ∧ q.2 ∈ dhUp P j ∧ q.2 ≠ p := by
  unfold dhPush at hq
  split at hq
  · exact Or.inl hq
  · rename_i p hp
    rcases List.mem_append.1 hq with h | h
    · exact Or.inl h
    · obtain ⟨w, hw, rfl⟩ := List.mem_map.1 h
      simp only [List.mem_filter, decide_eq_true_eq] at hw
      exact Or.inr ⟨p, hp, rfl, hw.1, hw.2⟩

/-- A pop adds arcs only between ends of arcs already in the pattern, lower to higher. -/
theorem dhPush_ends (P : List (Nat × Nat)) (j : Nat) (x : Nat) (h : Ends (dhPush P j) x) : Ends P x := by
  obtain ⟨q, hq, hx⟩ := h
  rcases mem_dhPush P j q hq with h | ⟨p, hp, h1, h2, _⟩
  · exact ⟨q, h, hx⟩
  · rcases hx with hx | hx
    · have := (minL_spec _ p hp).1
      exact ⟨(j, p), (mem_dhUp P j p).1 this, Or.inr (h1 ▸ hx)⟩
    · exact ⟨(j, q.2), (mem_dhUp P j q.2).1 h2, Or.inr hx⟩

theorem dhPush_lt (P : List (Nat × Nat)) (j : Nat) (h : ∀ q ∈ P, q.1 < q.2) : ∀ q ∈ dhPush P j, q.1 < q.2 := by
  intro q hq
  rcases mem_dhPush P j q hq with h' | ⟨p, hp, h1, h2, h3⟩
  · exact h q h'
  · have := (minL_spec _ p hp).2 q.2 h2; omega

theorem dh_ends : ∀ (js : List Nat) (P : List (Nat × Nat)) (x : Nat), Ends (dhRun P js) x → Ends P x
  | [], _, _, h => h
  | j :: js, P, x, h => dhPush_ends P j x (dh_ends js (dhPush P j) x h)

theorem dh_lt : ∀ (js : List Nat) (P : List (Nat × Nat)), (∀ q ∈ P, q.1 < q.2) → ∀ q ∈ dhRun P js, q.1 < q.2
  | [], _, h => h
  | j :: js, P, h => dh_lt js (dhPush P j) (dhPush_lt P j h)

/-- Whether `(lo, hi)` is an arc, read through `up_ids(lo)`. -/
def hasArc (t : Cch α) (lo hi : Nat) : Bool := (upIds t lo).any (fun q => decide (q.1 = hi))

theorem hasArc_iff (t : Cch α) (lo hi : Nat) : hasArc t lo hi = true ↔ ∃ i, upE t ⟨lo, hi, i⟩ := by
  unfold hasArc
  simp only [List.any_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨q, hq, rfl⟩; exact ⟨q.2, (mem_upIds t lo q.1 q.2).1 hq⟩
  · rintro ⟨i, h⟩; exact ⟨(hi, i), (mem_upIds t lo hi i).2 h, rfl⟩

/-- The pass's new arcs: the closure's arcs that are not arcs yet, sorted by `(lo, hi)` and deduplicated
    ("Then sort the new arcs by (lo,hi)", 03 §6). -/
def s2New (t : Cch α) (C : List (Nat × Nat)) : List (Nat × Nat) :=
  (insAll [] ((C.filter (fun q => !hasArc t q.1 q.2)).map (fun q => (⟨q.1, q.2, 0⟩ : Ent)))).map (fun e => (e.row, e.col))

theorem mem_s2New (t : Cch α) (C : List (Nat × Nat)) (q : Nat × Nat) :
    q ∈ s2New t C ↔ q ∈ C ∧ hasArc t q.1 q.2 = false := by
  unfold s2New
  constructor
  · intro h
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 h
    rcases mem_insAll_sub _ [] e he with h | h
    · simp at h
    · obtain ⟨⟨a, b⟩, hq', rfl⟩ := List.mem_map.1 h
      show (a, b) ∈ C ∧ hasArc t a b = false
      rw [List.mem_filter] at hq'
      simpa using hq'
  · rintro ⟨hc, ha⟩
    have hm : q ∈ C.filter (fun q => !hasArc t q.1 q.2) := List.mem_filter.2 ⟨hc, by simp [ha]⟩
    have hm2 : (⟨q.1, q.2, 0⟩ : Ent) ∈ (C.filter (fun q => !hasArc t q.1 q.2)).map (fun q => (⟨q.1, q.2, 0⟩ : Ent)) :=
      List.mem_map_of_mem hm
    obtain ⟨z, hz, hs⟩ := insAll_covers _ [] ⟨q.1, q.2, 0⟩ (Or.inr hm2)
    refine List.mem_map.2 ⟨z, hz, ?_⟩
    rw [hs.1, hs.2]

/-- **The new arcs of any Davis–Hager closure satisfy `S2Ok`**, when the closure starts from arcs of the
    version and any set of inserted pairs, each `lo < hi` between held ranks: the one pair of a single
    insert, every inserted pair of a commit unit (03 §6 "resolve every pair of the unit"), or a whole
    adjacency (an Overflow rescan, 03 §5). `alloc_ranks_all` gives every endpoint a rank first. -/
theorem s2New_ok (t : Cch α) (hI : Inv t) (P0 : List (Nat × Nat))
    (hP0 : ∀ q ∈ P0, (∃ i, upE t ⟨q.1, q.2, i⟩) ∨ (q.1 < q.2 ∧ (t.nodeOf q.1).isSome ∧ (t.nodeOf q.2).isSome))
    (js : List Nat) : S2Ok t (s2New t (dhRun P0 js)) := by
  obtain ⟨_, ha, hh⟩ := hI
  have hlt0 : ∀ q ∈ P0, q.1 < q.2 := by
    intro q hq
    rcases hP0 q hq with ⟨i, hi⟩ | h
    · exact ha.up_lt _ hi
    · exact h.1
  have hends : ∀ x, Ends P0 x → (t.nodeOf x).isSome := by
    rintro x ⟨q, hq, hx⟩
    rcases hP0 q hq with ⟨i, hi⟩ | h
    · have := hh _ hi; rcases hx with hx | hx
      · exact hx ▸ this.1
      · exact hx ▸ this.2
    · rcases hx with hx | hx
      · exact hx ▸ h.2.1
      · exact hx ▸ h.2.2
  refine ⟨?_, ?_, ?_, ?_⟩
  · unfold s2New
    rw [List.pairwise_map]
    exact (ksorted_insAll _ [] (by simp [KSorted])).imp (fun h => by unfold keyLt at h; unfold PLt; exact h)
  · intro q hq; exact dh_lt js P0 hlt0 q ((mem_s2New t _ q).1 hq).1
  · intro q hq i hi
    have := ((mem_s2New t _ q).1 hq).2
    rw [← Bool.not_eq_true, hasArc_iff] at this
    exact this ⟨i, hi⟩
  · intro q hq
    have hc := ((mem_s2New t _ q).1 hq).1
    exact ⟨hends _ (dh_ends js P0 _ ⟨q, hc, Or.inl rfl⟩), hends _ (dh_ends js P0 _ ⟨q, hc, Or.inr rfl⟩)⟩

end CchTopo
