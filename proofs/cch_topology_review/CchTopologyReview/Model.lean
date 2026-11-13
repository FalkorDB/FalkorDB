import CchTopologyReview.Basic

/-!
# The design's topology state (03 §2) and its operations (03 §4, §6)

Every structure of 03 §2's table that the topology invariants talk about, as a value. How the values
are shared between versions (the `Arc`s, the chunks, the B-tree pages) is in `Chunked.lean` and
`Delta.lean`; here a version is a value, which those two files justify.

| 03 §2 | Lean |
|---|---|
| `Up.base`, `Down.base` (`RankMatrix`: `Ap`, `Aj`, `Ax`, value = arc id) | `Cch.upB`, `Cch.downB : Rows` (row `r` = its `(col, id)` entries; `freshRows`, `ap` in Fresh.lean relate them to `Ap`) |
| `Up.delta`, `Down.delta` (`CowBTree`, key `(row ≪ 32) \| col`, doc = id) | `Cch.upD`, `Cch.downD : List Ent` (the tree's content; `Delta.lean` is the refinement) |
| `rank_of`, `node_of` (`ChunkedArr<Option<…>>`) | `Cch.rankOf`, `Cch.nodeOf : Nat → Option Nat` (`Chunked.lean` is the storage) |
| `free_ranks` (`Arc<RoaringBitmap>`) | `Cch.free : List Nat`, ascending |
| Metric, length `arc_capacity` | `Cch.met : List α` (format out of scope, 03 §2) |
| `arc_capacity` | `Cch.cap` |
| `LineageStats` | `Cch.stats : Stats` |
| `added_since_fold`: `(seq, Arc<[(lo, hi, id)]>)` | `Cch.added : List (Nat × List Ent)` |
| the data version a copy publishes as (`seq`, 03 §5 "The log") | `Cch.ver` |

Ranks are 0-based indices (`Rank::index`, `order.rs:60` @main); 03's "the next top rank `n_r' + 1`" in
1-based words is index `nr` here, and "the lowest `reserve` ranks" are `0 … reserve-1`.
-/

namespace CchTopo

/-- A topology entry: `(row, col)` is the key, `id` the doc. In `Up` row = lo, col = hi; in `Down`
    row = hi, col = lo (03 §2 "Arc identity"). -/
structure Ent where
  row : Nat
  col : Nat
  id : Nat
deriving DecidableEq, Repr

/-- The same arc seen from the other side: `Up(lo, hi) = id` ↔ `Down(hi, lo) = id` (I7). -/
def tr (e : Ent) : Ent := ⟨e.col, e.row, e.id⟩

@[simp] theorem tr_tr (e : Ent) : tr (tr e) = e := by cases e; rfl
@[simp] theorem tr_row (e : Ent) : (tr e).row = e.col := rfl
@[simp] theorem tr_col (e : Ent) : (tr e).col = e.row := rfl
@[simp] theorem tr_id (e : Ent) : (tr e).id = e.id := rfl

/-- The delta's key order, `(row ≪ 32) | col` (03 §2). -/
def keyLt (a b : Ent) : Prop := a.row < b.row ∨ (a.row = b.row ∧ a.col < b.col)
instance (a b : Ent) : Decidable (keyLt a b) := inferInstanceAs (Decidable (_ ∨ _))

def sameKey (a b : Ent) : Prop := a.row = b.row ∧ a.col = b.col
instance (a b : Ent) : Decidable (sameKey a b) := inferInstanceAs (Decidable (_ ∧ _))

/-- A delta's content: strictly ascending by key, so one id per `(row, col)`. -/
def KSorted (l : List Ent) : Prop := l.Pairwise keyLt

/-- Insert by key; an entry whose key is already present is not added. -/
def sinsK (x : Ent) : List Ent → List Ent
  | [] => [x]
  | y :: ys => if keyLt x y then x :: y :: ys else if sameKey x y then y :: ys else y :: sinsK x ys

/-- Insert a batch, one entry at a time. -/
def insAll (d b : List Ent) : List Ent := b.foldl (fun acc e => sinsK e acc) d

/-- A base, as its rows (`fresh_ids` in Fresh.lean ties a fresh base's ids to `Ap`). -/
abbrev Rows := List (List RE)

/-- Base row `r`; a rank past the base's dimension has an empty base row (03 §2: "a rank above the
    base's dimension can live in the delta"). -/
def bRow (b : Rows) (r : Nat) : List RE := b.getD r []

/-- `e` is in the base. -/
def inB (b : Rows) (e : Ent) : Prop := (e.col, e.id) ∈ bRow b e.row

/-- Delta row `r`, in the delta's order (a `range` read of keys `r ≪ 32 … r ≪ 32 | u32::MAX`). -/
def dRow (d : List Ent) (r : Nat) : List RE := (d.filter (fun e => decide (e.row = r))).map (fun e => (e.col, e.id))

/-- `LineageStats` (03 §2): deletions, dead ids, arcs at build, top ranks used, arcs added by S2 at
    pool-ranked rows. -/
structure Stats where
  deletions : Nat
  deadIds : Nat
  arcsAtBuild : Nat
  topUsed : Nat
  poolFill : Nat
deriving DecidableEq, Repr

/-- One version's CCH hierarchy (03 §6b MVCC: `Cch{Arc<TopologyVersion>, Metric}`). -/
structure Cch (α : Type) where
  ver : Nat
  reserve : Nat
  nr : Nat
  nodeOf : Nat → Option Nat
  rankOf : Nat → Option Nat
  free : List Nat
  upB : Rows
  downB : Rows
  upD : List Ent
  downD : List Ent
  cap : Nat
  added : List (Nat × List Ent)
  met : List α
  stats : Stats

variable {α : Type}

/-- An `Up` arc, base or delta. -/
def upE (t : Cch α) (e : Ent) : Prop := inB t.upB e ∨ e ∈ t.upD
/-- A `Down` arc, base or delta. -/
def downE (t : Cch α) (e : Ent) : Prop := inB t.downB e ∨ e ∈ t.downD

/-- **`up_ids(r)`** (03 §2 I3′): base row `r` merged with the delta's row `r`. -/
def upIds (t : Cch α) (r : Nat) : List RE := mergeRow (bRow t.upB r) (dRow t.upD r)
/-- `down_ids(r)`: the same over `Down`. -/
def downIds (t : Cch α) (r : Nat) : List RE := mergeRow (bRow t.downB r) (dRow t.downD r)

/-- Point update of a partial map (`ChunkedArr::set`). -/
def upd (f : Nat → Option Nat) (a : Nat) (b : Option Nat) : Nat → Option Nat := fun z => if z = a then b else f z

/-! ## Operations -/

/-- A writer's private copy: `published.version + 1` (03 §5 "The log", `mvcc_graph.rs:301` @08c). Every
    node it logs, and every S2 pass it runs, carries this as `seq`. -/
def fork (t : Cch α) : Cch α := { t with ver := t.ver + 1 }

/-- A new top rank `n_r' + 1` (index `nr`) for `x`; it lives only in the deltas (03 §6 step 3). -/
def allocTop (t : Cch α) (x : Nat) : Cch α :=
  { t with nodeOf := upd t.nodeOf t.nr (some x), rankOf := upd t.rankOf x (some t.nr), nr := t.nr + 1,
           stats := { t.stats with topUsed := t.stats.topUsed + 1 } }

/-- **Rank allocation for one node** (03 §6 "Rank allocation", steps 2–3): a rank-less `x` of degree
    `d` takes the lowest free rank if `d ≤ 1` and one is free, otherwise the next top rank. A node that
    already has a rank is left alone ("for each rank-less node x in E"). -/
def allocOne (t : Cch α) (x d : Nat) : Cch α :=
  match t.rankOf x with
  | some _ => t
  | none =>
    match t.free with
    | r :: rest =>
      if d ≤ 1 then { t with nodeOf := upd t.nodeOf r (some x), rankOf := upd t.rankOf x (some r), free := rest }
      else allocTop t x
    | [] => allocTop t x

/-- **The rank allocation phase** over `E`, the endpoints with a live indexed edge and their degrees, in
    id order (03 §6 steps 1–3). It runs before any classification (step 4). -/
def alloc (t : Cch α) (E : List (Nat × Nat)) : Cch α := E.foldl (fun t p => allocOne t p.1 p.2) t

/-- S2's new `Up` entries: arc `N[i]` gets id `cap + i` (03 §2 "S2": "appends ids `arc_capacity,
    arc_capacity+1, …`"). -/
def upEnts : Nat → List (Nat × Nat) → List Ent
  | _, [] => []
  | c, p :: ps => ⟨p.1, p.2, c⟩ :: upEnts (c + 1) ps

/-- **S2's append** (03 §2 "Arc identity" S2 and §6's S2 row), for the new arcs `N` of one pass, sorted
    by `(lo, hi)`: ids from `arc_capacity` up, into both deltas and `added_since_fold` (with this copy's
    seq), the metric preset to `inf` (`(+∞, +∞, none, none)`), and the pool-row counter. -/
def s2 (inf : α) (t : Cch α) (N : List (Nat × Nat)) : Cch α :=
  let ups := upEnts t.cap N
  { t with upD := insAll t.upD ups, downD := insAll t.downD (ups.map tr), cap := t.cap + N.length,
           added := t.added ++ [(t.ver, ups)], met := t.met ++ List.replicate N.length inf,
           stats := { t.stats with poolFill := t.stats.poolFill + (N.filter (fun p => decide (p.1 < t.reserve))).length } }

/-- **Node delete** (03 §6 table): its edges cascade as logged pairs (resolved as S1 metric writes,
    `mset`); its rank and arcs stay, and `deletions += deg`. -/
def ndel (t : Cch α) (deg : Nat) : Cch α := { t with stats := { t.stats with deletions := t.stats.deletions + deg } }

/-- **Node created, or id reused** (03 §6 table): "Nothing until its first indexed edge. A reused id keeps
    its old rank and its arcs". -/
def reuse (t : Cch α) : Cch α := t

/-- A metric write by id (S1, a recompute, the full-seed scatter: 03 §2 "No positional writes"). -/
def mset (t : Cch α) (i : Nat) (x : α) : Cch α := { t with met := t.met.set i x }

/-- The arcs a fold keeps in the new deltas: `added_since_fold` with `seq > F` (03 §6 "Fold job"). -/
def keepAfter (F : Nat) (a : List (Nat × List Ent)) : List (Nat × List Ent) := a.filter (fun p => decide (F < p.1))

/-- Their entries. -/
def ents (a : List (Nat × List Ent)) : List Ent := a.flatMap (·.2)

/-- **The fold's publish** (03 §6 "Fold job"): "the published Cch with only the four structures (Up.base,
    Down.base, Up.delta, Down.delta) replaced". `newUp`, `newDown` are the bases the job built at its
    fork `F` (GraphBLAS `eWiseAdd` then unload, `AddSpec` in Fold.lean); the new deltas are the arcs of
    the published `added_since_fold` with `seq > F`, "inserted into empty `cow_btree`s on both sides";
    `added_since_fold` keeps only those arcs. Everything else is the published copy's. -/
def fold (F : Nat) (newUp newDown : Rows) (t : Cch α) : Cch α :=
  let keep := keepAfter F t.added
  { t with upB := newUp, downB := newDown, upD := insAll [] (ents keep),
           downD := insAll [] ((ents keep).map tr), added := keep }

/-! ## List lemmas for the delta -/

theorem keyLt_trans {a b c : Ent} (h1 : keyLt a b) (h2 : keyLt b c) : keyLt a c := by
  unfold keyLt at *; omega

theorem keyLt_irrefl_of_same {a b : Ent} (h : sameKey a b) : ¬ keyLt a b := by
  unfold keyLt sameKey at *; omega

theorem mem_sinsK_sub (x : Ent) : ∀ (l : List Ent) (e : Ent), e ∈ sinsK x l → e ∈ l ∨ e = x
  | [], e, h => by simp [sinsK] at h; exact Or.inr h
  | y :: ys, e, h => by
    unfold sinsK at h
    split at h
    · rcases List.mem_cons.1 h with h | h
      · exact Or.inr h
      · exact Or.inl h
    · split at h
      · exact Or.inl h
      · rcases List.mem_cons.1 h with h | h
        · exact Or.inl (h ▸ List.mem_cons_self ..)
        · rcases mem_sinsK_sub x ys e h with h | h
          · exact Or.inl (List.mem_cons_of_mem _ h)
          · exact Or.inr h

theorem mem_sinsK_of_mem (x : Ent) : ∀ (l : List Ent) (e : Ent), e ∈ l → e ∈ sinsK x l
  | [], e, h => by simp at h
  | y :: ys, e, h => by
    unfold sinsK
    split
    · exact List.mem_cons_of_mem _ h
    · split
      · exact h
      · rcases List.mem_cons.1 h with h | h
        · exact h ▸ List.mem_cons_self ..
        · exact List.mem_cons_of_mem _ (mem_sinsK_of_mem x ys e h)

/-- `x` is in the result, unless an entry with its key was already there. -/
theorem sinsK_covers (x : Ent) : ∀ (l : List Ent), ∃ z ∈ sinsK x l, sameKey z x
  | [] => ⟨x, by simp [sinsK], rfl, rfl⟩
  | y :: ys => by
    unfold sinsK
    split
    · exact ⟨x, List.mem_cons_self .., rfl, rfl⟩
    · split
      · rename_i h; exact ⟨y, List.mem_cons_self .., h.1.symm, h.2.symm⟩
      · obtain ⟨z, hz, hs⟩ := sinsK_covers x ys
        exact ⟨z, List.mem_cons_of_mem _ hz, hs⟩

theorem mem_sinsK_fresh (x : Ent) (l : List Ent) (hx : ∀ y ∈ l, ¬ sameKey x y) (e : Ent) :
    e ∈ sinsK x l ↔ e ∈ l ∨ e = x := by
  refine ⟨mem_sinsK_sub x l e, fun h => h.elim (mem_sinsK_of_mem x l e) (fun h => ?_)⟩
  subst h
  obtain ⟨z, hz, hs⟩ := sinsK_covers e l
  rcases mem_sinsK_sub e l z hz with h | h
  · exact absurd ⟨hs.1.symm, hs.2.symm⟩ (hx z h)
  · exact h ▸ hz

theorem ksorted_sinsK (x : Ent) : ∀ (l : List Ent), KSorted l → KSorted (sinsK x l)
  | [], _ => by simp [sinsK, KSorted]
  | y :: ys, h => by
    unfold KSorted at h
    rw [List.pairwise_cons] at h
    unfold sinsK
    split
    · rename_i hxy
      unfold KSorted
      rw [List.pairwise_cons]
      refine ⟨fun z hz => ?_, List.pairwise_cons.2 h⟩
      rcases List.mem_cons.1 hz with e | e
      · exact e ▸ hxy
      · exact keyLt_trans hxy (h.1 z e)
    · split
      · exact List.pairwise_cons.2 h
      · rename_i h1 h2
        have hyx : keyLt y x := by unfold keyLt sameKey at *; omega
        unfold KSorted
        rw [List.pairwise_cons]
        refine ⟨fun z hz => ?_, ksorted_sinsK x ys h.2⟩
        rcases mem_sinsK_sub x ys z hz with e | e
        · exact h.1 z e
        · exact e ▸ hyx

theorem ksorted_insAll : ∀ (b d : List Ent), KSorted d → KSorted (insAll d b)
  | [], _, h => h
  | x :: b, d, h => ksorted_insAll b (sinsK x d) (ksorted_sinsK x d h)

theorem mem_insAll_sub : ∀ (b d : List Ent) (e : Ent), e ∈ insAll d b → e ∈ d ∨ e ∈ b
  | [], d, e, h => Or.inl h
  | x :: b, d, e, h => by
    rcases mem_insAll_sub b (sinsK x d) e h with h | h
    · rcases mem_sinsK_sub x d e h with h | h
      · exact Or.inl h
      · exact Or.inr (h ▸ List.mem_cons_self ..)
    · exact Or.inr (List.mem_cons_of_mem _ h)

theorem mem_insAll_of_mem : ∀ (b d : List Ent) (e : Ent), e ∈ d → e ∈ insAll d b
  | [], _, _, h => h
  | x :: b, d, e, h => mem_insAll_of_mem b (sinsK x d) e (mem_sinsK_of_mem x d e h)

/-- Every key of the batch ends up in the result (perhaps with the entry that already held it). -/
theorem insAll_covers : ∀ (b d : List Ent) (e : Ent), e ∈ d ∨ e ∈ b → ∃ z ∈ insAll d b, sameKey z e
  | [], d, e, h => by
    rcases h with h | h
    · exact ⟨e, h, rfl, rfl⟩
    · simp at h
  | x :: b, d, e, h => by
    rcases h with h | h
    · exact insAll_covers b (sinsK x d) e (Or.inl (mem_sinsK_of_mem x d e h))
    · rcases List.mem_cons.1 h with h | h
      · subst h
        obtain ⟨z, hz, hs⟩ := sinsK_covers e d
        obtain ⟨w, hw, hs'⟩ := insAll_covers b (sinsK e d) z (Or.inl hz)
        exact ⟨w, hw, hs'.1.trans hs.1, hs'.2.trans hs.2⟩
      · exact insAll_covers b (sinsK x d) e (Or.inr h)

/-- With fresh, pairwise key-distinct batch entries, the insert holds exactly old ∪ batch. -/
theorem mem_insAll_fresh : ∀ (b d : List Ent), b.Pairwise (fun a c => ¬ sameKey a c) →
    (∀ x ∈ b, ∀ y ∈ d, ¬ sameKey x y) → ∀ e, e ∈ insAll d b ↔ e ∈ d ∨ e ∈ b
  | [], d, _, _, e => by simp [insAll]
  | x :: b, d, hb, hf, e => by
    rw [List.pairwise_cons] at hb
    have hx : ∀ y ∈ d, ¬ sameKey x y := hf x (List.mem_cons_self ..)
    have ih := mem_insAll_fresh b (sinsK x d) hb.2 (fun z hz y hy => by
      rcases mem_sinsK_sub x d y hy with h | h
      · exact hf z (List.mem_cons_of_mem _ hz) y h
      · subst h; intro hs; exact hb.1 z hz ⟨hs.1.symm, hs.2.symm⟩) e
    show e ∈ insAll (sinsK x d) b ↔ _
    rw [ih, mem_sinsK_fresh x d hx, List.mem_cons]
    constructor
    · rintro ((h | h) | h)
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr h)
    · rintro (h | h | h)
      · exact Or.inl (Or.inl h)
      · exact Or.inl (Or.inr h)
      · exact Or.inr h

/-- A key-sorted list holds one entry per key. -/
theorem ksorted_key_unique : ∀ {l : List Ent}, KSorted l → ∀ a ∈ l, ∀ b ∈ l, sameKey a b → a = b
  | [], _, a, ha, _, _, _ => by simp at ha
  | x :: l, h, a, ha, b, hb, hs => by
    unfold KSorted at h
    rw [List.pairwise_cons] at h
    rcases List.mem_cons.1 ha with e | e <;> rcases List.mem_cons.1 hb with e' | e'
    · rw [e, e']
    · subst e; exact absurd (h.1 b e') (keyLt_irrefl_of_same hs)
    · subst e'; exact absurd (h.1 a e) (keyLt_irrefl_of_same ⟨hs.1.symm, hs.2.symm⟩)
    · exact ksorted_key_unique h.2 a e b e' hs

theorem keyLt_asymm {a b : Ent} (h : keyLt a b) : ¬ keyLt b a := by unfold keyLt at *; omega

/-- A key-sorted list is determined by its entries. -/
theorem ksorted_ext : ∀ {l₁ l₂ : List Ent}, KSorted l₁ → KSorted l₂ → (∀ e, e ∈ l₁ ↔ e ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => by have := (h b).2 (List.mem_cons_self ..); simp at this
  | a :: _, [], _, _, h => by have := (h a).1 (List.mem_cons_self ..); simp at this
  | a :: as, b :: bs, h1, h2, h => by
    unfold KSorted at h1 h2
    rw [List.pairwise_cons] at h1 h2
    have hab : a = b := by
      rcases List.mem_cons.1 ((h a).1 (List.mem_cons_self ..)) with e | e
      · exact e
      · rcases List.mem_cons.1 ((h b).2 (List.mem_cons_self ..)) with e' | e'
        · exact e'.symm
        · exact absurd (h2.1 a e) (keyLt_asymm (h1.1 b e'))
    subst hab
    congr 1
    refine ksorted_ext h1.2 h2.2 (fun p => ⟨fun hp => ?_, fun hp => ?_⟩)
    · rcases List.mem_cons.1 ((h p).1 (List.mem_cons_of_mem _ hp)) with e | e
      · subst e; exact absurd (h1.1 _ hp) (fun k => by unfold keyLt at k; omega)
      · exact e
    · rcases List.mem_cons.1 ((h p).2 (List.mem_cons_of_mem _ hp)) with e | e
      · subst e; exact absurd (h2.1 _ hp) (fun k => by unfold keyLt at k; omega)
      · exact e

/-- **The order of a batch does not matter**: inserting a fresh, key-distinct batch, or the same batch sorted
    by key first (`insAll [] b`), gives the same delta. -/
theorem insAll_sorted_batch (d b : List Ent) (hd : KSorted d) (hb : b.Pairwise (fun a c => ¬ sameKey a c))
    (hf : ∀ x ∈ b, ∀ y ∈ d, ¬ sameKey x y) : insAll d (insAll [] b) = insAll d b := by
  have hm0 : ∀ e, e ∈ insAll [] b ↔ e ∈ b := fun e => by
    rw [mem_insAll_fresh b [] hb (by simp)]; simp
  have hs0 : KSorted (insAll [] b) := ksorted_insAll b [] (by simp [KSorted])
  have hb0 : (insAll [] b).Pairwise (fun a c => ¬ sameKey a c) := hs0.imp (fun h hs => keyLt_irrefl_of_same hs h)
  have hf0 : ∀ x ∈ insAll [] b, ∀ y ∈ d, ¬ sameKey x y := fun x hx => hf x ((hm0 x).1 hx)
  refine ksorted_ext (ksorted_insAll _ d hd) (ksorted_insAll b d hd) (fun e => ?_)
  rw [mem_insAll_fresh _ d hb0 hf0, mem_insAll_fresh b d hb hf, hm0]

theorem mem_dRow (d : List Ent) (r c i : Nat) : (c, i) ∈ dRow d r ↔ (⟨r, c, i⟩ : Ent) ∈ d := by
  unfold dRow
  simp only [List.mem_map, List.mem_filter, decide_eq_true_eq, Prod.mk.injEq]
  constructor
  · rintro ⟨e, ⟨he, hr⟩, hc, hi⟩
    have : e = ⟨r, c, i⟩ := by cases e; simp_all
    exact this ▸ he
  · intro h; exact ⟨_, ⟨h, rfl⟩, rfl, rfl⟩

/-- A delta row read is strictly ascending by column. -/
theorem dRow_strict (d : List Ent) (hd : KSorted d) (r : Nat) : StrictCol (dRow d r) := by
  unfold dRow StrictCol
  rw [List.pairwise_map]
  have hf : (d.filter (fun e => decide (e.row = r))).Pairwise keyLt := hd.sublist List.filter_sublist
  refine hf.imp_of_mem (fun {a b} ha hb hab => ?_)
  simp only [List.mem_filter, decide_eq_true_eq] at ha hb
  unfold keyLt at hab; omega

theorem mem_upIds (t : Cch α) (r c i : Nat) : (c, i) ∈ upIds t r ↔ upE t ⟨r, c, i⟩ := by
  unfold upIds upE inB
  rw [mem_mergeRow, mem_dRow]

theorem mem_downIds (t : Cch α) (r c i : Nat) : (c, i) ∈ downIds t r ↔ downE t ⟨r, c, i⟩ := by
  unfold downIds downE inB
  rw [mem_mergeRow, mem_dRow]

end CchTopo
