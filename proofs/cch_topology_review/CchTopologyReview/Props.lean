import CchTopologyReview.Lineage

/-!
# What `Inv` gives every reader: I3′, the id rules, I7, I1′ and P (03 §2 "Contract changes")
-/

namespace CchTopo

variable {α : Type}

/-- **I3′**: `up_ids(r)` is strictly ascending by `hi` (so by `(hi, id)`), given the base rows ascending and
    the delta disjoint from the base. -/
theorem upIds_strict {t : Cch α} (h : Inv t) (r : Nat) : StrictCol (upIds t r) := by
  obtain ⟨_, ha, _⟩ := h
  refine mergeRow_strict _ _ (ha.upB_strict r) (dRow_strict t.upD ha.upD_sorted r) ?_
  rintro ⟨c, j⟩ hx ⟨c', i⟩ hy hc
  simp only at hc
  subst hc
  have hy' : (⟨r, c, i⟩ : Ent) ∈ t.upD := (mem_dRow t.upD r c i).1 hy
  exact ha.up_disj _ hy' j hx

/-- I3′ for `down_ids(r)`. -/
theorem downIds_strict {t : Cch α} (h : Inv t) (r : Nat) : StrictCol (downIds t r) := by
  obtain ⟨_, ha, _⟩ := h
  refine mergeRow_strict _ _ (ha.downB_strict r) (dRow_strict t.downD ha.downD_sorted r) ?_
  rintro ⟨c, j⟩ hx ⟨c', i⟩ hy hc
  simp only at hc
  subst hc
  have hy' : (⟨r, c, i⟩ : Ent) ∈ t.downD := (mem_dRow t.downD r c i).1 hy
  exact ha.down_disj _ hy' j hx

/-- I3′'s premise, kept by every operation (`Inv` is): no key is in both the base and the delta. -/
theorem delta_disjoint {t : Cch α} (h : Inv t) :
    (∀ e ∈ t.upD, ∀ i, ¬ inB t.upB ⟨e.row, e.col, i⟩) ∧ (∀ e ∈ t.downD, ∀ i, ¬ inB t.downB ⟨e.row, e.col, i⟩) :=
  ⟨h.2.1.up_disj, h.2.1.down_disj⟩

/-- **Arc ids are unique across base and delta.** -/
theorem ids_unique {t : Cch α} (h : Inv t) (e e' : Ent) (he : upE t e) (he' : upE t e') (hid : e.id = e'.id) :
    e = e' := h.2.1.id_inj e e' he he' hid

/-- **Every arc id is below `arc_capacity`**, and the metric has exactly `arc_capacity` entries (03 §2
    "Arrays are sized by `arc_capacity`"). -/
theorem ids_lt_cap {t : Cch α} (h : Inv t) (e : Ent) (he : upE t e) : e.id < t.cap ∧ t.met.length = t.cap :=
  ⟨h.2.1.id_lt e he, h.2.1.met_len⟩

/-- **I7**: `Up(lo, hi) = id` exactly when `Down(hi, lo) = id`, base and delta merged. -/
theorem i7 {t : Cch α} (h : Inv t) (e : Ent) : upE t e ↔ downE t (tr e) := by
  rw [downE_arcs, h.2.1.downE_iff, tr_tr]; rfl

/-- I7 as rows: `(hi, id) ∈ up_ids(lo)` exactly when `(lo, id) ∈ down_ids(hi)`. -/
theorem i7_rows {t : Cch α} (h : Inv t) (lo hi i : Nat) : (hi, i) ∈ upIds t lo ↔ (lo, i) ∈ downIds t hi := by
  rw [mem_upIds, mem_downIds, i7 h]; rfl

/-- One id names one unordered pair across `Up` and `Down`. -/
theorem id_names_one_pair {t : Cch α} (h : Inv t) (e e' : Ent) (he : upE t e) (he' : downE t e') (hid : e.id = e'.id) :
    e' = tr e := by
  have : upE t (tr e') := by rw [i7 h, tr_tr]; exact he'
  have := ids_unique h e (tr e') he this hid
  rw [this, tr_tr]

/-- An arc's key determines its id. -/
theorem key_determines_id {t : Cch α} (h : Inv t) (x y : Ent) (hx : upE t x) (hy : upE t y) (hs : x.row = y.row)
    (hc : x.col = y.col) : x = y := h.2.1.up_key_unique x y hx hy ⟨hs, hc⟩

/-- **I1′**: `rank_of` is injective, `node_of` is injective, and each is the other's inverse. -/
theorem rank_of_injective {t : Cch α} (h : Inv t) (x y r : Nat) (hx : t.rankOf x = some r) (hy : t.rankOf y = some r) :
    x = y := by
  have h1 := (h.1.rank_node x r).1 hx
  have h2 := (h.1.rank_node y r).1 hy
  exact Option.some.inj (h1.symm.trans h2)

theorem node_of_injective {t : Cch α} (h : Inv t) (r s x : Nat) (hr : t.nodeOf r = some x) (hs : t.nodeOf s = some x) :
    r = s := by
  have h1 := (h.1.rank_node x r).2 hr
  have h2 := (h.1.rank_node x s).2 hs
  exact Option.some.inj (h1.symm.trans h2)

theorem node_of_inverse {t : Cch α} (h : Inv t) (x r : Nat) : t.rankOf x = some r ↔ t.nodeOf r = some x :=
  h.1.rank_node x r

/-- **P**: inside the rank space, `r ∈ free_ranks ⇔ node_of[r] = None`. -/
theorem pool_iff {t : Cch α} (h : Inv t) (r : Nat) (hr : r < t.nr) : r ∈ t.free ↔ t.nodeOf r = none := by
  rw [show r ∈ t.free ↔ r < t.nr ∧ t.nodeOf r = none from h.1.free_iff r]
  exact ⟨fun h => h.2, fun h => ⟨hr, h⟩⟩

/-- Every rank past the rank space is holderless and not free. -/
theorem past_space {t : Cch α} (h : Inv t) (r : Nat) (hr : t.nr ≤ r) : t.nodeOf r = none ∧ r ∉ t.free := by
  constructor
  · cases e : t.nodeOf r with
    | none => rfl
    | some x => have : r < t.nr := h.1.node_lt r x e; omega
  · intro hf; have : r < t.nr ∧ t.nodeOf r = none := (h.1.free_iff r).1 hf; omega

/-- **P**: a holderless rank has `up(r) = down(r) = ∅`, base and delta merged. -/
theorem pool_rows_empty {t : Cch α} (h : Inv t) (r : Nat) (hr : t.nodeOf r = none) : upIds t r = [] ∧ downIds t r = [] := by
  constructor
  · cases e : upIds t r with
    | nil => rfl
    | cons p _ =>
      have : (p.1, p.2) ∈ upIds t r := by rw [e]; exact List.mem_cons_self ..
      have : (t.nodeOf r).isSome := (h.2.2 _ ((mem_upIds t r p.1 p.2).1 this)).1
      simp [hr] at this
  · cases e : downIds t r with
    | nil => rfl
    | cons p _ =>
      have : (p.1, p.2) ∈ downIds t r := by rw [e]; exact List.mem_cons_self ..
      have hd := (mem_downIds t r p.1 p.2).1 this
      rw [downE_arcs, h.2.1.downE_iff] at hd
      have : (t.nodeOf r).isSome := (h.2.2 _ hd).2
      simp [hr] at this

theorem free_rows_empty {t : Cch α} (h : Inv t) (r : Nat) (hr : r ∈ t.free) : upIds t r = [] ∧ downIds t r = [] :=
  pool_rows_empty h r ((h.1.free_iff r).1 hr).2

/-- **`cch_recompute_review`'s premise P-rows, for the redesign's merged rows**: every `up_ids(r)` is strictly
    ascending, above `r`, and inside the rank space. -/
theorem p_rows {t : Cch α} (h : Inv t) (r : Nat) :
    StrictCol (upIds t r) ∧ ∀ p ∈ upIds t r, r < p.1 ∧ p.1 < t.nr := by
  refine ⟨upIds_strict h r, fun p hp => ?_⟩
  have he : upE t ⟨r, p.1, p.2⟩ := (mem_upIds t r p.1 p.2).1 hp
  refine ⟨h.2.1.up_lt _ he, ?_⟩
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 (h.2.2 _ he).2
  exact h.1.node_lt _ x hx

/-- **`cch_recompute_review`'s premise P-transpose, for the redesign**: `down_ids` rows are strictly ascending
    and `Down` is `Upᵀ` with the same ids. -/
theorem p_transpose {t : Cch α} (h : Inv t) :
    (∀ r, StrictCol (downIds t r)) ∧ ∀ lo hi i, (hi, i) ∈ upIds t lo ↔ (lo, i) ∈ downIds t hi :=
  ⟨downIds_strict h, i7_rows h⟩

/-- S2 numbers its arcs in the pass's `(lo, hi)` order from `arc_capacity` up: `N[i]` gets `cap + i`. -/
theorem upEnts_getElem? : ∀ (c : Nat) (N : List (Nat × Nat)) (i : Nat),
    (upEnts c N)[i]? = N[i]?.map (fun p => (⟨p.1, p.2, c + i⟩ : Ent))
  | _, [], i => by simp [upEnts]
  | c, p :: N, 0 => by simp [upEnts]
  | c, p :: N, i + 1 => by
    simp only [upEnts, List.getElem?_cons_succ]
    rw [upEnts_getElem? (c + 1) N i]
    congr 2; funext q; congr 1; omega

/-- **S2 presets the metric of every new arc** to `inf` (`(+∞, +∞, none, none)`), and leaves every old
    entry alone. -/
theorem s2_metric (inf : α) {t : Cch α} (h : Inv t) (N : List (Nat × Nat)) :
    (s2 inf t N).met = t.met ++ List.replicate N.length inf ∧
    ∀ e ∈ upEnts t.cap N, (s2 inf t N).met[e.id]? = some inf := by
  refine ⟨rfl, fun e he => ?_⟩
  have hm : t.met.length = t.cap := h.2.1.met_len
  obtain ⟨_, h1, h2⟩ := mem_upEnts t.cap N e he
  show (t.met ++ List.replicate N.length inf)[e.id]? = some inf
  rw [List.getElem?_append_right (by omega), List.getElem?_replicate]
  rw [ite_eq_left (by omega)]

end CchTopo
