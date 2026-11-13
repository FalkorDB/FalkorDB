import CchTopologyReview.Lifetime

/-!
# A fresh build (03 §4) starts every lineage with the invariant, and ids are `Ap[lo] + i`

* `fresh`: the hierarchy Supergraph publishes: `Up.base` with ids row-major (§4 step 4, §2 "Fresh build":
  `id = Ap[lo] + i`), `Down.base` its GraphBLAS transpose with the same ids (`TransposeSpec`), empty
  deltas, `arc_capacity` = the arc count, the metric that long, `free_ranks` = the holderless ranks (§4
  step 5), `rank_of` the inverse of `node_of`.
* `fresh_ids`: entry `i` of row `r` has id `ap pat r + i`, where `ap pat r` is `Ap[r]`.
* `fresh_inv`: the result satisfies `Inv`, given `FreshOk` (premises on Order's and Supergraph's output,
  named below) and `TransposeSpec` (GraphBLAS). With `lin_all` (Lifetime.lean), every state of a lineage
  that starts from a fresh build satisfies `Inv`.
-/

namespace CchTopo

variable {α : Type}

/-- Ids from `off`, one per entry: entry `i` gets `off + i`. -/
def idsFrom : Nat → List Nat → List RE
  | _, [] => []
  | off, h :: hs => (h, off) :: idsFrom (off + 1) hs

/-- The base's rows, ids row-major from `off` (§4 step 4: "Assign ids row-major"). -/
def freshRows : Nat → List (List Nat) → Rows
  | _, [] => []
  | off, row :: rest => idsFrom off row :: freshRows (off + row.length) rest

/-- `Ap[r]`: the arcs in rows before `r`. -/
def ap (pat : List (List Nat)) (r : Nat) : Nat := ((pat.take r).map List.length).sum

/-- The arc count. -/
def total (pat : List (List Nat)) : Nat := (pat.map List.length).sum

/-- The first rank holding `x`. -/
def firstIdx (x : Nat) : List (Option Nat) → Nat → Option Nat
  | [], _ => none
  | h :: hs, i => if h = some x then some i else firstIdx x hs (i + 1)

/-- Inputs from Order and Supergraph (§3 step 0, §4 steps 1–5). -/
structure FreshIn where
  ver : Nat
  reserve : Nat
  /-- `node_of` over the rank space: `none` for a pool rank (§2, §3 step 0). -/
  holders : List (Option Nat)
  /-- `Up`'s rows from the elimination game (§4 step 3): row `r` lists the higher ranks. -/
  pat : List (List Nat)

/-- **Premises on Order's and Supergraph's output** (other components; 03 §4 "Contract"): `π` is injective;
    each row is ascending; every arc joins two held ranks, lower to higher ("P holds by construction: a
    holderless rank has no row in B′"). -/
structure FreshOk (fi : FreshIn) : Prop where
  inj : ∀ (r s x : Nat), fi.holders[r]? = some (some x) → fi.holders[s]? = some (some x) → r = s
  asc : ∀ r, (fi.pat.getD r []).Pairwise (· < ·)
  arcs : ∀ r h, h ∈ fi.pat.getD r [] → r < h ∧ (fi.holders[r]?.join).isSome ∧ (fi.holders[h]?.join).isSome

/-- **GraphBLAS contract** (C, a hypothesis structure): `Down.base = GrB_transpose` of a read-only load of
    `Up.base`, unloaded (§4 step 4): the same ids at the mirrored positions, rows strictly ascending. -/
structure TransposeSpec (up down : Rows) : Prop where
  mem : ∀ e, inB up e ↔ inB down (tr e)
  strict : ∀ r, StrictCol (bRow down r)

/-- The fresh hierarchy. -/
def fresh (inf : α) (fi : FreshIn) (down : Rows) : Cch α :=
  { ver := fi.ver, reserve := fi.reserve, nr := fi.holders.length,
    nodeOf := fun r => (fi.holders[r]?).join,
    rankOf := fun x => firstIdx x fi.holders 0,
    free := (List.range fi.holders.length).filter (fun r => decide (fi.holders[r]? = some none)),
    upB := freshRows 0 fi.pat, downB := down, upD := [], downD := [], cap := total fi.pat, added := [],
    met := List.replicate (total fi.pat) inf,
    stats := ⟨0, 0, total fi.pat, 0, 0⟩ }

theorem ap_succ (pat : List (List Nat)) (r : Nat) : ap pat (r + 1) = ap pat r + (pat.getD r []).length := by
  unfold ap
  induction pat generalizing r with
  | nil => simp
  | cons row rest ih =>
    cases r with
    | zero => simp
    | succ r => simp only [List.take_succ_cons, List.map_cons, List.sum_cons, List.getD_cons_succ]; rw [ih r]; omega

theorem ap_mono (pat : List (List Nat)) {r s : Nat} (h : r ≤ s) : ap pat r ≤ ap pat s := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => rw [ap_succ]; omega

theorem ap_le_total (pat : List (List Nat)) (r : Nat) : ap pat r ≤ total pat := by
  unfold ap total
  induction pat generalizing r with
  | nil => simp
  | cons row rest ih =>
    cases r with
    | zero => simp
    | succ r => simp only [List.take_succ_cons, List.map_cons, List.sum_cons]; have := ih r; omega

theorem mem_idsFrom : ∀ (off : Nat) (row : List Nat) (c i : Nat),
    (c, i) ∈ idsFrom off row ↔ ∃ j, row[j]? = some c ∧ i = off + j
  | _, [], c, i => by simp [idsFrom]
  | off, h :: hs, c, i => by
    simp only [idsFrom, List.mem_cons, Prod.mk.injEq]
    rw [mem_idsFrom (off + 1) hs c i]
    constructor
    · rintro (⟨rfl, rfl⟩ | ⟨j, hj, rfl⟩)
      · exact ⟨0, by simp, by omega⟩
      · exact ⟨j + 1, by simpa using hj, by omega⟩
    · rintro ⟨j, hj, rfl⟩
      cases j with
      | zero => simp at hj; exact Or.inl ⟨hj.symm, by omega⟩
      | succ j => exact Or.inr ⟨j, by simpa using hj, by omega⟩

theorem idsFrom_strict : ∀ (off : Nat) (row : List Nat), row.Pairwise (· < ·) → StrictCol (idsFrom off row)
  | _, [], _ => by simp [idsFrom, StrictCol]
  | off, h :: hs, hp => by
    rw [List.pairwise_cons] at hp
    rw [idsFrom, strictCol_cons]
    refine ⟨fun b hb => ?_, idsFrom_strict (off + 1) hs hp.2⟩
    obtain ⟨j, hj, _⟩ := (mem_idsFrom (off + 1) hs b.1 b.2).1 hb
    exact hp.1 _ (List.mem_of_getElem? hj)

/-- **Row `r` of the fresh base is row `r` of the game with ids `Ap[r] + i`.** -/
theorem bRow_freshRows : ∀ (pat : List (List Nat)) (off r : Nat),
    bRow (freshRows off pat) r = idsFrom (off + ap pat r) (pat.getD r [])
  | [], off, r => by simp [freshRows, bRow, idsFrom]
  | row :: rest, off, r => by
    cases r with
    | zero => simp [freshRows, bRow, ap]
    | succ r =>
      have := bRow_freshRows rest (off + row.length) r
      simp only [bRow, freshRows, List.getD_cons_succ] at this ⊢
      rw [this]
      congr 1
      unfold ap; simp only [List.take_succ_cons, List.map_cons, List.sum_cons]; omega

/-- **03 §2 "Fresh build": `id = Ap[lo] + i`.** -/
theorem fresh_ids (pat : List (List Nat)) (lo hi id : Nat) :
    inB (freshRows 0 pat) ⟨lo, hi, id⟩ ↔ ∃ i, (pat.getD lo [])[i]? = some hi ∧ id = ap pat lo + i := by
  unfold inB
  rw [bRow_freshRows, Nat.zero_add, mem_idsFrom]

theorem firstIdx_spec (x : Nat) : ∀ (hs : List (Option Nat)) (i r : Nat),
    (∀ (a b : Nat), hs[a]? = some (some x) → hs[b]? = some (some x) → a = b) →
    (firstIdx x hs i = some r ↔ i ≤ r ∧ hs[r - i]? = some (some x))
  | [], i, r, _ => by simp [firstIdx]
  | h :: hs, i, r, hinj => by
    unfold firstIdx
    split
    · rename_i hh
      constructor
      · intro e; cases e; simp [hh]
      · rintro ⟨hle, hr⟩
        have : r - i = 0 := hinj _ _ hr (by simp [hh])
        simp only [Option.some.injEq]; omega
    · rename_i hh
      have hinj' : ∀ (a b : Nat), hs[a]? = some (some x) → hs[b]? = some (some x) → a = b :=
        fun a b ha hb => by have := hinj (a + 1) (b + 1) (by simpa using ha) (by simpa using hb); omega
      rw [firstIdx_spec x hs (i + 1) r hinj']
      constructor
      · rintro ⟨hle, hr⟩
        refine ⟨by omega, ?_⟩
        have : r - i = (r - (i + 1)) + 1 := by omega
        rw [this]; simpa using hr
      · rintro ⟨hle, hr⟩
        rcases Nat.lt_or_ge i r with hlt | hge
        · refine ⟨hlt, ?_⟩
          have : r - i = (r - (i + 1)) + 1 := by omega
          rw [this] at hr; simpa using hr
        · have : r - i = 0 := by omega
          rw [this] at hr; simp at hr; exact absurd hr hh

/-- **A fresh build satisfies the invariant.** -/
theorem fresh_inv (inf : α) (fi : FreshIn) (down : Rows) (hok : FreshOk fi) (ht : TransposeSpec (freshRows 0 fi.pat) down) :
    Inv (fresh inf fi down) := by
  have hrow : ∀ e, inB (freshRows 0 fi.pat) e → ∃ i, (fi.pat.getD e.row [])[i]? = some e.col ∧ e.id = ap fi.pat e.row + i :=
    fun e he => (fresh_ids fi.pat e.row e.col e.id).1 he
  have hupE : ∀ e, upE (fresh inf fi down) e ↔ inB (freshRows 0 fi.pat) e := fun e => by
    show inB (freshRows 0 fi.pat) e ∨ e ∈ ([] : List Ent) ↔ _; simp
  refine ⟨?_, ?_, ?_⟩
  · refine ⟨?_, ?_, ?_, ?_⟩
    · intro r x hx
      have : fi.holders[r]? = some (some x) := Option.join_eq_some_iff.1 hx
      exact (List.getElem?_eq_some_iff.1 this).1
    · intro x r
      show firstIdx x fi.holders 0 = some r ↔ (fi.holders[r]?).join = some x
      rw [firstIdx_spec x fi.holders 0 r (fun a b => hok.inj a b x), Option.join_eq_some_iff]; simp
    · intro r
      show r ∈ (List.range fi.holders.length).filter _ ↔ r < fi.holders.length ∧ (fi.holders[r]?).join = none
      rw [List.mem_filter, List.mem_range]
      constructor
      · rintro ⟨hr, hn⟩
        refine ⟨hr, ?_⟩
        simp only [decide_eq_true_eq] at hn; rw [hn]; rfl
      · rintro ⟨hr, hn⟩
        refine ⟨hr, ?_⟩
        simp only [decide_eq_true_eq]
        rw [List.getElem?_eq_getElem hr] at hn ⊢
        cases e : fi.holders[r] with
        | none => rfl
        | some y => rw [e] at hn; simp at hn
    · exact List.pairwise_lt_range.sublist List.filter_sublist
  · refine
      { up_lt := ?_, upB_strict := ?_, downB_strict := ht.strict, upD_sorted := by simp [KSorted, fresh],
        downD_sorted := by simp [KSorted, fresh], up_disj := by simp [fresh], down_disj := by simp [fresh],
        i7B := ht.mem, i7D := by simp [fresh], id_lt := ?_, id_inj := ?_, met_len := by simp [Cch.arcs, fresh],
        added_eq := by simp [ents, fresh], added_nodup := by simp [ents, fresh], added_seq := by simp [fresh] }
    · intro e he
      obtain ⟨i, hi, _⟩ := hrow e ((hupE e).1 he)
      exact (hok.arcs e.row e.col (List.mem_of_getElem? hi)).1
    · intro r
      show StrictCol (bRow (freshRows 0 fi.pat) r)
      rw [bRow_freshRows]; exact idsFrom_strict _ _ (hok.asc r)
    · intro e he
      obtain ⟨i, hi, hid⟩ := hrow e ((hupE e).1 he)
      have hlt : i < (fi.pat.getD e.row []).length := (List.getElem?_eq_some_iff.1 hi).1
      have := ap_le_total fi.pat (e.row + 1)
      rw [ap_succ] at this
      show e.id < total fi.pat
      omega
    · intro e e' he he' hid
      obtain ⟨i, hi, hid1⟩ := hrow e ((hupE e).1 he)
      obtain ⟨i', hi', hid2⟩ := hrow e' ((hupE e').1 he')
      have hlt : i < (fi.pat.getD e.row []).length := (List.getElem?_eq_some_iff.1 hi).1
      have hlt' : i' < (fi.pat.getD e'.row []).length := (List.getElem?_eq_some_iff.1 hi').1
      have hr : e.row = e'.row := by
        rcases Nat.lt_trichotomy e.row e'.row with h | h | h
        · have := ap_mono fi.pat (Nat.succ_le_of_lt h); rw [ap_succ] at this; omega
        · exact h
        · have := ap_mono fi.pat (Nat.succ_le_of_lt h); rw [ap_succ] at this; omega
      rw [hr] at hid1 hi
      have hii : i = i' := by omega
      rw [hii, hi'] at hi
      cases e; cases e'; simp_all
  · intro e he
    obtain ⟨i, hi, _⟩ := hrow e ((hupE e).1 he)
    exact (hok.arcs e.row e.col (List.mem_of_getElem? hi)).2

/-- **Every state of a lineage that starts from a fresh build satisfies the invariant**, so every theorem of
    Props.lean, Fold.lean and Lifetime.lean applies to it. -/
theorem lineage_inv (inf : α) (fi : FreshIn) (down : Rows) (hok : FreshOk fi)
    (ht : TransposeSpec (freshRows 0 fi.pat) down) {t : Cch α} (hl : Lin inf (fresh inf fi down) t) : Inv t :=
  (lin_all inf hl (fresh_inv inf fi down hok ht)).inv

end CchTopo
