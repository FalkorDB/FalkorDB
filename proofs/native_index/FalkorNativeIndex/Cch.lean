import FalkorNativeIndex.Rebase
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# The CCH path-index contract skeleton (#3319, 1ff68d538)

`graph/src/index/falkordb/cch/` declares the CCH shortest-path index: its types, signatures and the
`VersionedIndex` kind, with every body `unimplemented!()` until #3208-#3217. Nothing calls them (a graph with no
CCH index behaves as before), so what is checkable today is small:

| Lean | Rust |
| --- | --- |
| `Csr`, `Csr.row?` | `Csr` (`cch/mod.rs:57`), `Csr::row` (`:71`) |
| `Csr.isWellFormed` | `Csr::is_well_formed` (`:82`) |
| `Csr.isSymmetricSimple` | `Csr::is_symmetric_simple` (`:97`) |
| `CchIndexDef.new` | `CchIndexDef::new` (`:139`) |
| `Res`, `unimpl` | an `unimplemented!()` / `unreachable!()` body: always panics |
| `cchEntry` | `CchIndex`'s `VersionedIndex::entry` / `entry_mut` (`:314`, `:321`): a `BTreeMap` lookup |

* `row?_spec`, `row?_isSome_of_wf`: `row` is the slice `i[p[u]..p[u+1]]`, and it panics iff an offset is out of range;
  a well-formed `Csr` never panics for `u < n`.
* `isWellFormed_iff`: exactly the three shape rules of the type's documentation (offsets `n + 1`, from 0, never
  decreasing, ending at `i.len()`; every id below `n`).
* `isSymmetricSimple_iff`: well formed, rows strictly increasing, no self-loop, every entry mirrored.
  `binary_search` is the standard library's: its only property used is `Ok` iff present in a strictly sorted
  slice, and `isSymmetricSimple_any_bs` shows the result does not depend on what it does for an unsorted row
  (the row's own check has already failed then).
* `CchIndexDef.new_spec`: sorted, duplicate-free, the same members, so equal sets are equal indexes.
* every `unimplemented!()` / `unreachable!()` body is a function that always panics (`*_panics`), which is what
  their docs say. The delegating ones (`Graph::drop_cch_index`, `cch_indexes`, `cch_index`) panic through the
  `FalkorDbIndex` body they call.
-/
namespace NI
namespace Cch

/-! ## Bodies that always panic -/

/-- A result that may panic. -/
inductive Res (α : Type) where
  | ok (a : α)
  | panic
deriving DecidableEq

/-- `unimplemented!()` / `unreachable!()`. -/
def unimpl {α : Type} : Res α := .panic

/-- `Res.bind`: a callee that panics makes the caller panic. -/
def Res.andThen {α β : Type} (r : Res α) (f : α → Res β) : Res β :=
  match r with | .ok a => f a | .panic => .panic

theorem andThen_panic {α β : Type} (f : α → Res β) : (Res.panic : Res α).andThen f = .panic := rfl

/-- The CCH definition and hierarchy are opaque here: their bodies are `unimplemented!()`. -/
structure Def where
  relTypes : List Nat
  weightAttr : Nat

/-- The `Cch`/`Def`/graph types are opaque type parameters in this file (the bodies never return), so only
    each result's *shape* is pinned to the Rust signature: `customize` returns `Result<(), CchError>`, hence
    `Res Unit`; the `Err` arm is unreachable while the body panics.
    `Cch::build` (`:193`, `unimplemented!`, #3208). -/
def cchBuild {Input Cch : Type} (_ : Input) : Res Cch := unimpl
/-- `Cch::customize` (`:204`, #3209). -/
def cchCustomize {Cch W : Type} (_ : Cch) (_ : W) : Res Unit := unimpl
/-- `Cch::recustomize_scoped` (`:236`, #3217). -/
def cchRecustomizeScoped {Cch S R : Type} (_ : Cch) (_ : S) : Res R := unimpl
/-- `Cch::query` (`:249`, #3210). -/
def cchQuery {Cch Q R : Type} (_ : Cch) (_ : Q) : Res R := unimpl
/-- `Cch::node_count` (`:260`, #3209). -/
def cchNodeCount {Cch : Type} (_ : Cch) : Res Nat := unimpl
/-- `FalkorDbIndex::cch_index` (`falkordb_index.rs:967`, #3212). -/
def indexCchIndex {X Cch : Type} (_ : X) (_ : Def) : Res (Option Cch) := unimpl
/-- `FalkorDbIndex::cch_defs` (`:975`, #3212): the body panics before the `std::iter::empty()` that pins the type. -/
def indexCchDefs {X : Type} (_ : X) : Res (List Def) := unimpl
/-- `FalkorDbIndex::drop_cch` (`:987`, #3212). -/
def indexDropCch {X : Type} (_ : X) (_ : Def) : Res Unit := unimpl
/-- `Graph::create_cch_index` (`graph.rs:5225`, #3212). -/
def graphCreateCchIndex {G : Type} (_ : G) (_ : Def) : Res Unit := unimpl
/-- `Graph::rebuild_cch_indexes` (`:5274`, #3212). -/
def graphRebuildCchIndexes {G : Type} (_ : G) : Res Unit := unimpl
/-- `Graph::drop_cch_index` (`:5237`): `self.falkordb_index.drop_cch(def)`. -/
def graphDropCchIndex {G X : Type} (g : G) (index : G → X) (d : Def) : Res Unit := indexDropCch (index g) d
/-- `Graph::cch_indexes` (`:5248`): `self.falkordb_index.cch_defs()`. -/
def graphCchIndexes {G X : Type} (g : G) (index : G → X) : Res (List Def) := indexCchDefs (index g)
/-- `Graph::cch_index` (`:5258`): `self.falkordb_index.cch_index(def)`. -/
def graphCchIndex {G X Cch : Type} (g : G) (index : G → X) (d : Def) : Res (Option Cch) := indexCchIndex (index g) d

/-- The `VersionedIndex for CchIndex` bodies that are `unreachable!` (`:330-413`): `writer_record`,
    `writer_rebase`, `indexer_catch_up`, `indexer_insert`, `batch_len`, `indexer_read`, `indexer_add`,
    `memory_usage` (#3568, #3569). No `IndexId` variant dispatches to them
    until the background build; a rebase is only onto a `Building` column and every CCH entry is `Ready`. -/
def cchVersionedUnreachable {α : Type} : Res α := unimpl

theorem cchBuild_panics {Input Cch : Type} (i : Input) : (cchBuild i : Res Cch) = .panic := rfl
theorem cchCustomize_panics {Cch W : Type} (c : Cch) (w : W) : cchCustomize c w = .panic := rfl
theorem cchRecustomizeScoped_panics {Cch S R : Type} (c : Cch) (s : S) :
    (cchRecustomizeScoped c s : Res R) = .panic := rfl
theorem cchQuery_panics {Cch Q R : Type} (c : Cch) (q : Q) : (cchQuery c q : Res R) = .panic := rfl
theorem cchNodeCount_panics {Cch : Type} (c : Cch) : cchNodeCount c = .panic := rfl
theorem indexCchIndex_panics {X Cch : Type} (x : X) (d : Def) : (indexCchIndex x d : Res (Option Cch)) = .panic := rfl
theorem indexCchDefs_panics {X : Type} (x : X) : indexCchDefs x = .panic := rfl
theorem indexDropCch_panics {X : Type} (x : X) (d : Def) : indexDropCch x d = .panic := rfl
theorem graphCreateCchIndex_panics {G : Type} (g : G) (d : Def) : graphCreateCchIndex g d = .panic := rfl
theorem graphRebuildCchIndexes_panics {G : Type} (g : G) : graphRebuildCchIndexes g = .panic := rfl
theorem graphDropCchIndex_panics {G X : Type} (g : G) (index : G → X) (d : Def) :
    graphDropCchIndex g index d = .panic := rfl
theorem graphCchIndexes_panics {G X : Type} (g : G) (index : G → X) : graphCchIndexes g index = .panic := rfl
theorem graphCchIndex_panics {G X Cch : Type} (g : G) (index : G → X) (d : Def) :
    (graphCchIndex g index d : Res (Option Cch)) = .panic := rfl
theorem cchVersionedUnreachable_panics {α : Type} : (cchVersionedUnreachable : Res α) = .panic := rfl

/-- `VersionedIndex::entry` / `entry_mut` for `CchIndex` (`cch/mod.rs:314-327`): `cch_indexes.get(def)` and
    `get_mut(def)`, found by the whole definition. -/
def cchEntry {Ent : Type} (m : Def → Option Ent) (d : Def) : Option Ent := m d

theorem cchEntry_spec {Ent : Type} (m : Def → Option Ent) (d : Def) : cchEntry m d = m d := rfl

/-! ## `CchIndexDef::new` (`cch/mod.rs:139-150`) -/

/-- `rel_types.sort_unstable(); rel_types.dedup()`. -/
def defNew (relTypes : List Nat) (weightAttr : Nat) : Def := ⟨sortDedup relTypes, weightAttr⟩

/-- **Equal sets are equal indexes**: the relationship types come out sorted, duplicate-free, with the same
    members, so two lists with the same members give the same definition. -/
theorem defNew_spec (rts : List Nat) (a : Nat) :
    Sorted (defNew rts a).relTypes ∧ (∀ x, x ∈ (defNew rts a).relTypes ↔ x ∈ rts) ∧ (defNew rts a).weightAttr = a :=
  ⟨(sortDedup_spec rts).1, (sortDedup_spec rts).2, rfl⟩

theorem defNew_eq_of_same_members (rts rts' : List Nat) (a : Nat) (h : ∀ x, x ∈ rts ↔ x ∈ rts') :
    defNew rts a = defNew rts' a := by
  have h1 := sortDedup_spec rts
  have h2 := sortDedup_spec rts'
  have : sortDedup rts = sortDedup rts' :=
    sorted_ext h1.1 h2.1 (fun e => by rw [h1.2, h2.2]; exact h e)
  simp [defNew, this]

/-! ## `Csr` (`cch/mod.rs:57-107`) -/

/-- `Csr { n, p, i }`. -/
structure Csr where
  n : Nat
  p : List Nat
  i : List Nat

/-- `slice::windows(2)`. -/
def windows2 : List Nat → List (Nat × Nat)
  | a :: b :: r => (a, b) :: windows2 (b :: r)
  | _ => []

theorem mem_windows2 : ∀ (l : List Nat) (a b : Nat),
    (a, b) ∈ windows2 l ↔ ∃ k, l[k]? = some a ∧ l[k+1]? = some b
  | [], a, b => by simp [windows2]
  | [x], a, b => by simp [windows2]
  | x :: y :: r, a, b => by
    rw [windows2, List.mem_cons, mem_windows2 (y :: r) a b]
    constructor
    · rintro (h | ⟨k, hk⟩)
      · simp only [Prod.mk.injEq] at h
        exact ⟨0, by simp [h.1], by simp [h.2]⟩
      · exact ⟨k + 1, by simpa using hk.1, by simpa using hk.2⟩
    · rintro ⟨k, h1, h2⟩
      cases k with
      | zero => left; simp at h1 h2; simp [h1, h2]
      | succ k => right; exact ⟨k, by simpa using h1, by simpa using h2⟩

/-- Never decreasing. -/
def Mono (l : List Nat) : Prop := ∀ (k a b : Nat), l[k]? = some a → l[k+1]? = some b → a ≤ b

theorem windowsLe_iff (l : List Nat) : (windows2 l).all (fun w => decide (w.1 ≤ w.2)) = true ↔ Mono l := by
  simp only [List.all_eq_true, decide_eq_true_eq, Mono]
  constructor
  · intro h k a b ha hb
    exact h (a, b) ((mem_windows2 l a b).2 ⟨k, ha, hb⟩)
  · rintro h ⟨a, b⟩ hw
    obtain ⟨k, hk1, hk2⟩ := (mem_windows2 l a b).1 hw
    exact h k a b hk1 hk2

theorem mono_le (l : List Nat) (h : Mono l) : ∀ (j k a b : Nat), k ≤ j → l[k]? = some a → l[j]? = some b → a ≤ b := by
  intro j
  induction j with
  | zero => intro k a b hk ha hb; have : k = 0 := by omega
            subst this; rw [ha] at hb; cases hb; exact Nat.le_refl _
  | succ j ih =>
    intro k a b hk ha hb
    rcases Nat.lt_or_ge k (j + 1) with hlt | hge
    · have hj : j < l.length := by
        have := (List.getElem?_eq_some_iff.1 hb).1; omega
      obtain ⟨c, hc⟩ : ∃ c, l[j]? = some c := ⟨l[j], List.getElem?_eq_getElem hj⟩
      exact Nat.le_trans (ih k a c (by omega) ha hc) (h j c b hc hb)
    · have : k = j + 1 := by omega
      subst this; rw [ha] at hb; cases hb; exact Nat.le_refl _

/-- `slice::binary_search`'s result as a membership test: the standard library's contract on a strictly
    sorted slice. -/
def BsSpec (bs : List Nat → Nat → Bool) : Prop := ∀ l x, l.Pairwise (· < ·) → bs l x = decide (x ∈ l)

/-- `Csr::row` (`:71-77`): `i[p[u]..p[u + 1]]`; `none` is the panic (`u` or `u + 1` past `p`, `p[u] > p[u+1]`,
    or `p[u+1] > i.len()`). -/
def Csr.row? (c : Csr) (u : Nat) : Option (List Nat) :=
  match c.p[u]?, c.p[u + 1]? with
  | some a, some b => if a ≤ b ∧ b ≤ c.i.length then some ((c.i.drop a).take (b - a)) else none
  | _, _ => none

theorem row?_spec (c : Csr) (u a b : Nat) (ha : c.p[u]? = some a) (hb : c.p[u + 1]? = some b) :
    c.row? u = if a ≤ b ∧ b ≤ c.i.length then some ((c.i.drop a).take (b - a)) else none := by
  simp [Csr.row?, ha, hb]

/-- `Csr::is_well_formed` (`:82-91`): the `checked_sub`, `p[0]`, `p[n]`, `windows(2)` and `iter().all` conjuncts. -/
def Csr.isWellFormed (c : Csr) : Bool :=
  decide (c.p.length - 1 = c.n ∧ 0 < c.p.length) && (c.p[0]? == some 0) && (c.p[c.n]? == some c.i.length) &&
  (windows2 c.p).all (fun w => decide (w.1 ≤ w.2)) && c.i.all (fun v => decide (v < c.n))

/-- **Exactly the shape rules of the type**: `n + 1` offsets, from 0, never decreasing, the last one `i.len()`,
    every id below `n`. -/
theorem isWellFormed_iff (c : Csr) : c.isWellFormed = true ↔
    c.p.length = c.n + 1 ∧ c.p[0]? = some 0 ∧ c.p[c.n]? = some c.i.length ∧ Mono c.p ∧ ∀ v ∈ c.i, v < c.n := by
  have hw := windowsLe_iff c.p
  simp only [Csr.isWellFormed, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, hw]
  simp only [List.all_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨⟨⟨⟨⟨h1, h1'⟩, h2⟩, h3⟩, h4⟩, h5⟩
    exact ⟨by omega, h2, h3, h4, h5⟩
  · rintro ⟨h1, h2, h3, h4, h5⟩
    exact ⟨⟨⟨⟨⟨by omega, by omega⟩, h2⟩, h3⟩, h4⟩, h5⟩

/-- **A well-formed `Csr` never panics in `row` for `u < n`.** -/
theorem row?_isSome_of_wf (c : Csr) (hwf : c.isWellFormed = true) (u : Nat) (hu : u < c.n) :
    ∃ r, c.row? u = some r := by
  obtain ⟨hl, h0, hn, hm, hv⟩ := (isWellFormed_iff c).1 hwf
  obtain ⟨a, ha⟩ : ∃ a, c.p[u]? = some a := ⟨c.p[u]'(by omega), List.getElem?_eq_getElem (by omega)⟩
  obtain ⟨b, hb⟩ : ∃ b, c.p[u + 1]? = some b := ⟨c.p[u + 1]'(by omega), List.getElem?_eq_getElem (by omega)⟩
  have hab : a ≤ b := hm u a b ha hb
  have hbn : b ≤ c.i.length := mono_le c.p hm c.n (u + 1) b c.i.length (by omega) hb hn
  exact ⟨_, by rw [row?_spec c u a b ha hb, if_pos ⟨hab, hbn⟩]⟩

/-- `row` read after the shape check: the panic case does not arise for `u < n` (`row?_isSome_of_wf`). -/
def Csr.rowD (c : Csr) (u : Nat) : List Nat := (c.row? u).getD []

theorem rowD_subset (c : Csr) (u x : Nat) (h : x ∈ c.rowD u) : x ∈ c.i := by
  unfold Csr.rowD at h
  cases hr : c.row? u with
  | none => simp [hr] at h
  | some r =>
    simp only [hr, Option.getD_some] at h
    unfold Csr.row? at hr
    split at hr
    · split at hr
      · cases hr
        exact List.mem_of_mem_drop (List.mem_of_mem_take h)
      · cases hr
    · cases hr

/-- A row's strict increase, as `windows(2)`: `w[0] < w[1]`. -/
def strictWin (r : List Nat) : Bool := (windows2 r).all (fun w => decide (w.1 < w.2))

theorem strictWin_iff : ∀ r : List Nat, strictWin r = true ↔ r.Pairwise (· < ·)
  | [] => by simp [strictWin, windows2]
  | [a] => by simp [strictWin, windows2]
  | a :: b :: t => by
    have ih := strictWin_iff (b :: t)
    have hs : strictWin (a :: b :: t) = true ↔ a < b ∧ strictWin (b :: t) = true := by
      simp [strictWin, windows2, List.all_cons]
    refine hs.trans ?_
    constructor
    · rintro ⟨hab, hp⟩
      have hp' := ih.1 hp
      refine List.pairwise_cons.2 ⟨?_, hp'⟩
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hab
      · exact Nat.lt_trans hab ((List.pairwise_cons.1 hp').1 x hx)
    · intro hp
      obtain ⟨h1, hp'⟩ := List.pairwise_cons.1 hp
      exact ⟨h1 b (by simp), ih.2 hp'⟩

/-- `Csr::is_symmetric_simple` (`:97-106`), over `bs`, the standard library's `binary_search(..).is_ok()`. -/
def Csr.isSymmetricSimple (bs : List Nat → Nat → Bool) (c : Csr) : Bool :=
  c.isWellFormed && (List.range c.n).all (fun u =>
    strictWin (c.rowD u) && (c.rowD u).all (fun v => v != u && bs (c.rowD v) u))

/-- **Exactly a simple undirected graph's adjacency**: well formed, every row strictly increasing, no
    self-loop, every entry mirrored (`v` in row `u` implies `u` in row `v`). -/
theorem isSymmetricSimple_iff (bs : List Nat → Nat → Bool) (hbs : BsSpec bs) (c : Csr) :
    c.isSymmetricSimple bs = true ↔ c.isWellFormed = true ∧ ∀ u, u < c.n →
      (c.rowD u).Pairwise (· < ·) ∧ ∀ v ∈ c.rowD u, v ≠ u ∧ u ∈ c.rowD v := by
  unfold Csr.isSymmetricSimple
  rw [Bool.and_eq_true, List.all_eq_true]
  constructor
  · rintro ⟨hwf, h⟩
    obtain ⟨_, _, _, _, hid⟩ := (isWellFormed_iff c).1 hwf
    refine ⟨hwf, fun u hu => ?_⟩
    have hu' := h u (List.mem_range.2 hu)
    rw [Bool.and_eq_true, strictWin_iff, List.all_eq_true] at hu'
    obtain ⟨hs, hall⟩ := hu'
    refine ⟨hs, fun v hv => ?_⟩
    have hv' := hall v hv
    rw [Bool.and_eq_true, bne_iff_ne] at hv'
    refine ⟨hv'.1, ?_⟩
    have hvn : v < c.n := hid v (rowD_subset c u v hv)
    have hsv : (c.rowD v).Pairwise (· < ·) := by
      have := h v (List.mem_range.2 hvn)
      rw [Bool.and_eq_true, strictWin_iff] at this
      exact this.1
    have := hbs (c.rowD v) u hsv
    rw [hv'.2] at this
    simpa using this.symm
  · rintro ⟨hwf, h⟩
    obtain ⟨_, _, _, _, hid⟩ := (isWellFormed_iff c).1 hwf
    refine ⟨hwf, fun u hu => ?_⟩
    obtain ⟨hs, hall⟩ := h u (List.mem_range.1 hu)
    rw [Bool.and_eq_true, strictWin_iff, List.all_eq_true]
    refine ⟨hs, fun v hv => ?_⟩
    rw [Bool.and_eq_true, bne_iff_ne]
    refine ⟨(hall v hv).1, ?_⟩
    have hvn : v < c.n := hid v (rowD_subset c u v hv)
    rw [hbs _ _ (h v hvn).1]
    simpa using (hall v hv).2

/-- **The answer does not depend on `binary_search` for an unsorted row**: any two implementations agreeing with
    the contract on strictly sorted slices (the only thing std promises) give the same result. -/
theorem isSymmetricSimple_any_bs (bs bs' : List Nat → Nat → Bool) (h : BsSpec bs) (h' : BsSpec bs') (c : Csr) :
    c.isSymmetricSimple bs = c.isSymmetricSimple bs' := by
  have e1 := isSymmetricSimple_iff bs h c
  have e2 := isSymmetricSimple_iff bs' h' c
  by_cases hb : c.isSymmetricSimple bs' = true
  · rw [e1.2 (e2.1 hb), hb]
  · have h1 : c.isSymmetricSimple bs ≠ true := fun h1 => hb (e2.2 (e1.1 h1))
    simp only [ne_eq, Bool.not_eq_true] at hb h1
    rw [hb, h1]

end Cch
end NI
