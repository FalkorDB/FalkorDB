import OpsTraverse.ExpandIntoPending
import OpsTraverse.UnionGlue
/-
# CondTraverse over uncommitted deletes (#3576)

`CondTraverse.processPairs` / `expandRow` and `UnionGlue.expandUnionRow` model the
operator over one committed graph. Within a commit segment `CondTraverseOp` also
reads `Pending` (`EIP.Pending`): a `DETACH DELETE`d node stays in the matrices,
with its edges, until commit.

| here | there |
| --- | --- |
| `pesP`, `pairRowsP`, `processPairsP` | `CondTraverseOp::process_pairs` (`cond_traverse.rs`): a pair with a pending-deleted endpoint (`is_node_deleted(src) \|\| is_node_deleted(dst)`, raw matrix coordinates) is skipped; collapse and emit skip `is_relationship_deleted(id)` and sibling-used edges |
| `expandRowPP`, `condTraverseP` | `CondTraverseOp::expand_row`: the forward/reverse blocks over one `Pending` borrow (the single-type stream is `pairs g`) |
| `expandUnionRowP` | `CondTraverseOp::expand_union_row`: the same blocks over the union stream, `pending` passed through |
| `batchedP` | `CondTraverseOp::expand_batch` (single hop): `has_deleted_relationships` ⇒ fall back to the per-row path (`none`); a pending-deleted source is not seeded into `F`, a pending-deleted destination of `F·A` is skipped. Fused chains never meet pending deletes: `fuse_anonymous_traverse` does not fuse a plan that contains a `Delete` |

With nothing pending every model here is literally its committed-graph model
(`processPairsP_empty`, `condTraverseP_empty`, `expandUnionRowP_empty`,
`batchedP_empty`), so every earlier CondTraverse theorem is the no-pending
instance. Property `LiveCondTraverse` (only edges of the effective graph): proved
(`condTraverseP_live`, `expandUnionRowP_live`); emit mode is exactly the live edge
set (`pend_emit_directed_iff`, `pend_emit_bidir_iff`), and the batched path agrees
with the per-row collapse (`batchedP_eq_collapse`).
-/

namespace OpsTraverse.CTP

open OpsTraverse OpsTraverse.EIP OpsTraverse.UnionGlue
open OpsTraverse.ASP (eq_of_mem_of_id)

/-- The edges `process_pairs` may bind on a matrix pair: not pending-deleted, not used by a sibling. -/
def pesP (g : Graph) (p : Pending) (r : RowIn) (q : Nat × Nat) : List Nat :=
  (edgesBetween g q.1 q.2).filter fun e => !(p.delRels.contains e) && !(r.used.contains e)

/-- One matrix pair of `process_pairs` with `Pending`. -/
def pairRowsP (g : Graph) (p : Pending) (isRev : Bool) (r : RowIn) (emit : Bool)
    (q : Nat × Nat) : List (Nat × Nat × Nat) :=
  if !(p.delNodes.contains q.1) && !(p.delNodes.contains q.2) && pcond r isRev q then
    (if emit then pesP g p r q else (pesP g p r q).take 1).map fun e => (pf isRev q, pt isRev q, e)
  else []

def processPairsP (g : Graph) (p : Pending) (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn)
    (emit : Bool) : List (Nat × Nat × Nat) :=
  ps.flatMap (pairRowsP g p isRev r emit)

/-- `expand_row`'s two blocks over a pair stream, with `Pending`. -/
def expandRowPP (g : Graph) (p : Pending) (ps : List (Nat × Nat)) (bidir emit : Bool)
    (r : RowIn) : List (Nat × Nat × Nat) :=
  processPairsP g p ps false r emit ++
    (if bidir then processPairsP g p (ps.filter fun q => q.1 != q.2) true r emit else [])

/-- `CondTraverseOp::expand_row` (single type) with `Pending`. Rows are `(from, to, edge)`. -/
def condTraverseP (g : Graph) (p : Pending) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  expandRowPP g p (pairs g) bidir emit r

/-- `CondTraverseOp::expand_union_row` with `Pending`. -/
noncomputable def expandUnionRowP (S : Graph → List (Nat × Nat)) (gs : List Graph)
    (sk : Nat × Nat → Bool) (p : Pending) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  expandRowPP gs.flatten p (unionStream S gs sk) bidir emit r

/-! ## Reduction to the committed model: deleted edges as used, deleted-endpoint pairs dropped -/

/-- The row with the pending-deleted edges added to the sibling-used ones. -/
def withDel (p : Pending) (r : RowIn) : RowIn := { r with used := r.used ++ p.delRels }

def liveQ (p : Pending) (q : Nat × Nat) : Bool :=
  !(p.delNodes.contains q.1) && !(p.delNodes.contains q.2)

theorem pesP_eq (g : Graph) (p : Pending) (r : RowIn) (q : Nat × Nat) :
    pesP g p r q = pes g (withDel p r) q := by
  unfold pesP pes withDel
  apply List.filter_congr
  intro e _
  simp only [List.contains_append]
  cases p.delRels.contains e <;> cases r.used.contains e <;> rfl

theorem pcond_withDel (p : Pending) (r : RowIn) (isRev : Bool) (q : Nat × Nat) :
    pcond (withDel p r) isRev q = pcond r isRev q := rfl

theorem processPairsP_eq (g : Graph) (p : Pending) (isRev : Bool) (r : RowIn) (emit : Bool) :
    ∀ ps : List (Nat × Nat),
      processPairsP g p ps isRev r emit = processPairs g (ps.filter (liveQ p)) isRev (withDel p r) emit
  | [] => rfl
  | q :: ps => by
    have ih := processPairsP_eq g p isRev r emit ps
    simp only [processPairsP, processPairs, List.flatMap_cons, List.filter_cons] at ih ⊢
    by_cases hl : liveQ p q = true
    · simp only [hl, ite_true, List.flatMap_cons, ih]
      congr 1
      unfold pairRowsP pairRows
      simp only [liveQ] at hl
      simp only [hl, Bool.true_and, pcond_withDel, pesP_eq]
      rfl
    · simp only [hl, Bool.false_eq_true, ite_false, ih]
      have : pairRowsP g p isRev r emit q = [] := by
        unfold pairRowsP
        split
        · rename_i hc
          exfalso
          apply hl
          simp only [liveQ, Bool.and_eq_true] at hc ⊢
          exact hc.1
        · rfl
      rw [this, List.nil_append]

/-! ## Nothing pending: the committed-graph models -/

theorem filter_tt {α : Type} : ∀ l : List α, l.filter (fun _ => true) = l
  | [] => rfl
  | a :: l => by simp [filter_tt l]

theorem processPairsP_empty (g : Graph) (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn)
    (emit : Bool) : processPairsP g ⟨[], []⟩ ps isRev r emit = processPairs g ps isRev r emit := by
  rw [processPairsP_eq]
  have h1 : (ps.filter (liveQ ⟨[], []⟩)) = ps := by
    simp [liveQ]
  have h2 : withDel ⟨[], []⟩ r = r := by simp [withDel]
  rw [h1, h2]

theorem condTraverseP_empty (g : Graph) (b em : Bool) (r : RowIn) :
    condTraverseP g ⟨[], []⟩ b em r = expandRow g b em r := by
  simp [condTraverseP, expandRowPP, expandRow, processPairsP_empty]

theorem expandUnionRowP_empty (S : Graph → List (Nat × Nat)) (gs : List Graph)
    (sk : Nat × Nat → Bool) (b em : Bool) (r : RowIn) :
    expandUnionRowP S gs sk ⟨[], []⟩ b em r = expandUnionRow S gs sk b em r := by
  simp [expandUnionRowP, expandUnionRow, expandRowPP, expandRowP, processPairsP_empty]

/-- Emit-mode membership with `Pending` (cf. `mem_processPairs`). -/
theorem mem_processPairsP {g : Graph} {p : Pending} {ps : List (Nat × Nat)} {isRev : Bool}
    {r : RowIn} {f t e : Nat} (Q : Nat → Nat → Prop)
    (hps : ∀ s d, (s, d) ∈ ps ↔ (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ Q s d) :
    (f, t, e) ∈ processPairsP g p ps isRev r true ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ e ∉ p.delRels ∧ x.src ∉ p.delNodes ∧
        x.dst ∉ p.delNodes ∧ okBound r.fromId f ∧ okBound r.toId t ∧ Q x.src x.dst ∧
        (if isRev then x.src = t ∧ x.dst = f else x.src = f ∧ x.dst = t) := by
  rw [processPairsP_eq]
  have hps' : ∀ s d, (s, d) ∈ ps.filter (liveQ p) ↔
      (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ (s ∉ p.delNodes ∧ d ∉ p.delNodes ∧ Q s d) := by
    intro s d
    simp only [List.mem_filter, hps, liveQ, Bool.and_eq_true, Bool.not_eq_true',
      List.contains_eq_mem, decide_eq_false_iff_not]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3, h4⟩; exact ⟨h1, h3, h4, h2⟩
    · rintro ⟨h1, h3, h4, h2⟩; exact ⟨⟨h1, h2⟩, h3, h4⟩
  rw [mem_processPairs _ hps']
  simp only [withDel, List.mem_append, not_or]
  constructor
  · rintro ⟨x, hx, h1, ⟨hu, hd⟩, hf, ht, ⟨hs, hdn, hq⟩, hdir⟩
    exact ⟨x, hx, h1, hu, hd, hs, hdn, hf, ht, hq, hdir⟩
  · rintro ⟨x, hx, h1, hu, hd, hs, hdn, hf, ht, hq, hdir⟩
    exact ⟨x, hx, h1, ⟨hu, hd⟩, hf, ht, ⟨hs, hdn, hq⟩, hdir⟩

/-- Collapse yields a row for `(f,t)` iff emit mode does (one representative per pair). -/
theorem collapse_exists_iff {g : Graph} {ps : List (Nat × Nat)} {isRev : Bool} {r : RowIn}
    {f t : Nat} :
    (∃ e, (f, t, e) ∈ processPairs g ps isRev r false) ↔
      ∃ e, (f, t, e) ∈ processPairs g ps isRev r true := by
  constructor
  · rintro ⟨e, he⟩; exact ⟨e, collapse_sub_emit he⟩
  · rintro ⟨e, he⟩
    simp only [processPairs, List.mem_flatMap, mem_pairRows, ite_true, Bool.false_eq_true,
      ite_false] at he ⊢
    obtain ⟨q, hq, hc, e', he', hx⟩ := he
    revert he'
    cases hes : pes g r q with
    | nil => intro h; simp at h
    | cons e0 rest =>
      intro _
      simp only [Prod.mk.injEq] at hx
      exact ⟨e0, q, hq, hc, e0, by rw [hes]; simp, by rw [hx.1, hx.2.1]⟩

/-! ## Emit mode is the live single-hop match set -/

/-- **Directed, emit mode, with `Pending`**: `(f,t,e)` iff `e` is a live edge `f→t` (not
deleted, neither endpoint deleted) that the row's bindings admit and no sibling uses. -/
theorem pend_emit_directed_iff {g : Graph} {p : Pending} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ condTraverseP g p false true r ↔
      ∃ x ∈ g, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧ e ∉ p.delRels ∧
        f ∉ p.delNodes ∧ t ∉ p.delNodes ∧ okBound r.fromId f ∧ okBound r.toId t := by
  simp only [condTraverseP, expandRowPP, Bool.false_eq_true, ite_false, List.append_nil]
  rw [mem_processPairsP (fun _ _ => True) (pairs_spec g)]
  simp only [Bool.false_eq_true, ite_false]
  constructor
  · rintro ⟨x, hx, h1, hu, hd, hs, hdn, hf, ht, _, rfl, rfl⟩
    exact ⟨x, hx, h1, rfl, rfl, hu, hd, hs, hdn, hf, ht⟩
  · rintro ⟨x, hx, h1, rfl, rfl, hu, hd, hs, hdn, hf, ht⟩
    exact ⟨x, hx, h1, hu, hd, hs, hdn, hf, ht, trivial, rfl, rfl⟩

/-- **Undirected, emit mode, with `Pending`**: `emit_bidir_iff` restricted to live edges. -/
theorem pend_emit_bidir_iff {g : Graph} {p : Pending} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ condTraverseP g p true true r ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ e ∉ p.delRels ∧ x.src ∉ p.delNodes ∧
        x.dst ∉ p.delNodes ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  simp only [condTraverseP, expandRowPP, ite_true, List.mem_append]
  rw [mem_processPairsP (fun _ _ => True) (pairs_spec g),
    mem_processPairsP (fun s d => s ≠ d) (pairs_filter_spec g)]
  simp only [Bool.false_eq_true, ite_false, ite_true]
  constructor
  · rintro (⟨x, hx, h1, h2, h3, h4, h5, h6, h7, _, h8⟩ |
      ⟨x, hx, h1, h2, h3, h4, h5, h6, h7, hne, h8⟩)
    · exact ⟨x, hx, h1, h2, h3, h4, h5, h6, h7, Or.inl h8⟩
    · exact ⟨x, hx, h1, h2, h3, h4, h5, h6, h7, Or.inr ⟨hne, h8⟩⟩
  · rintro ⟨x, hx, h1, h2, h3, h4, h5, h6, h7, (h8 | ⟨hne, h8⟩)⟩
    · exact Or.inl ⟨x, hx, h1, h2, h3, h4, h5, h6, h7, trivial, h8⟩
    · exact Or.inr ⟨x, hx, h1, h2, h3, h4, h5, h6, h7, hne, h8⟩

/-! ## Property `LiveCondTraverse` -/

/-- Property: CondTraverse only matches edges of the effective graph. -/
def LiveCondTraverse
    (op : Graph → Pending → Bool → Bool → RowIn → List (Nat × Nat × Nat)) : Prop :=
  ∀ g p b em r, g.WF → ∀ x ∈ op g p b em r, Live g p x.2.2

/-- Every row of `process_pairs` (either mode, any pair stream) carries a non-deleted edge of
`g` whose endpoints are not deleted. -/
theorem mem_processPairsP_sound {g : Graph} {p : Pending} {ps : List (Nat × Nat)}
    {isRev em : Bool} {r : RowIn} {y : Nat × Nat × Nat}
    (hy : y ∈ processPairsP g p ps isRev r em) :
    ∃ x ∈ g, x.id = y.2.2 ∧ y.2.2 ∉ p.delRels ∧ x.src ∉ p.delNodes ∧ x.dst ∉ p.delNodes := by
  simp only [processPairsP, List.mem_flatMap] at hy
  obtain ⟨q, _, hq⟩ := hy
  unfold pairRowsP at hq
  split at hq
  · rename_i hc
    simp only [Bool.and_eq_true, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not] at hc
    obtain ⟨e, he, rfl⟩ := List.mem_map.1 hq
    have he' : e ∈ pesP g p r q := by
      cases em
      · exact List.mem_of_mem_take he
      · exact he
    simp only [pesP, List.mem_filter, Bool.and_eq_true, Bool.not_eq_true',
      List.contains_eq_mem, decide_eq_false_iff_not] at he'
    obtain ⟨hb, hdr, _⟩ := he'
    obtain ⟨x, hx, hid, h1, h2⟩ := mem_edgesBetween.1 hb
    exact ⟨x, hx, hid, hdr, h1 ▸ hc.1.1, h2 ▸ hc.1.2⟩
  · simp at hq

theorem live_of_sound {g : Graph} {p : Pending} (hwf : g.WF) {e : Nat}
    (h : ∃ x ∈ g, x.id = e ∧ e ∉ p.delRels ∧ x.src ∉ p.delNodes ∧ x.dst ∉ p.delNodes) :
    Live g p e := by
  obtain ⟨x, hx, hid, hdr, hs, hd⟩ := h
  refine ⟨hdr, fun y hy hyid => ?_⟩
  have : y = x := eq_of_mem_of_id hwf hy hx (hyid.trans hid.symm)
  subst this
  exact ⟨hs, hd⟩

theorem expandRowPP_live {g : Graph} {p : Pending} {ps : List (Nat × Nat)} {b em : Bool}
    {r : RowIn} (hwf : g.WF) : ∀ y ∈ expandRowPP g p ps b em r, Live g p y.2.2 := by
  intro y hy
  simp only [expandRowPP, List.mem_append] at hy
  rcases hy with hy | hy
  · exact live_of_sound hwf (mem_processPairsP_sound hy)
  · split at hy
    · exact live_of_sound hwf (mem_processPairsP_sound hy)
    · simp at hy

/-- **Main satisfies the property** (#3576): `expand_row` never matches a pending-deleted
edge, nor an edge of a pending-deleted node (the doomed edges of `DETACH DELETE`). -/
theorem condTraverseP_live : LiveCondTraverse condTraverseP :=
  fun _ _ _ _ _ hwf => expandRowPP_live hwf

/-- The multi-type path (`expand_union_row`) satisfies it too, on the union of its types' edges. -/
theorem expandUnionRowP_live (S : Graph → List (Nat × Nat)) (gs : List Graph)
    (sk : Nat × Nat → Bool) (p : Pending) (b em : Bool) (r : RowIn) (hwf : Graph.WF gs.flatten) :
    ∀ y ∈ expandUnionRowP S gs sk p b em r, Live gs.flatten p y.2.2 :=
  expandRowPP_live hwf

/-! ## `expand_batch` (single hop) -/

/-- `expand_batch` on one hop with `Pending`: `none` = fall back to the per-row path
(`has_deleted_relationships`); otherwise `F` seeded with the non-deleted sources, then the
deleted destinations of `F·A` dropped. Row `i` of `F` holds `(i, src_i)`. -/
def batchedP (mxm : Mat → Mat → Mat) (g : Graph) (p : Pending) (F : Mat) : Option Mat :=
  if p.delRels.isEmpty then
    some ((chain mxm (F.filter fun q => !(p.delNodes.contains q.2)) [pairs g]).filter
      fun q => !(p.delNodes.contains q.2))
  else none

theorem batchedP_empty (mxm : Mat → Mat → Mat) (g : Graph) (F : Mat) :
    batchedP mxm g ⟨[], []⟩ F = some (chain mxm F [pairs g]) := by
  simp [batchedP, filter_tt]

theorem batchedP_fallback (mxm : Mat → Mat → Mat) (g : Graph) (p : Pending) (F : Mat)
    (h : p.delRels ≠ []) : batchedP mxm g p F = none := by
  simp [batchedP, List.isEmpty_iff, h]

/-- **Batched path = per-row collapse, with `Pending`**: whenever `expand_batch` does not fall
back, row `i` of its result holds `d` exactly when the per-row collapsed expansion (`condTraverseP`)
of `src_i` yields a row for `(src_i, d)`. -/
theorem batchedP_eq_collapse {mxm : Mat → Mat → Mat} (h : MxmSpec mxm) (g : Graph) (p : Pending)
    (F M : Mat) (hM : batchedP mxm g p F = some M) (i d : Nat) :
    (i, d) ∈ M ↔
      ∃ s, (i, s) ∈ F ∧ ∃ e, (s, d, e) ∈ condTraverseP g p false false ⟨some s, none, []⟩ := by
  unfold batchedP at hM
  split at hM
  · rename_i hr
    have hr' : p.delRels = [] := List.isEmpty_iff.1 hr
    simp only [Option.some.injEq] at hM
    subst hM
    have key : ∀ s, (∃ e, (s, d, e) ∈ condTraverseP g p false false ⟨some s, none, []⟩) ↔
        s ∉ p.delNodes ∧ d ∉ p.delNodes ∧ ∃ x ∈ g, x.src = s ∧ x.dst = d := by
      intro s
      have : ∀ e, (s, d, e) ∈ condTraverseP g p false false ⟨some s, none, []⟩ ↔
          (s, d, e) ∈ processPairs g ((pairs g).filter (liveQ p)) false
            (withDel p ⟨some s, none, []⟩) false := by
        intro e
        simp only [condTraverseP, expandRowPP, Bool.false_eq_true, ite_false, List.append_nil,
          processPairsP_eq]
      simp only [this]
      rw [collapse_exists_iff]
      have hps : ∀ a b, (a, b) ∈ (pairs g).filter (liveQ p) ↔
          (∃ x ∈ g, x.src = a ∧ x.dst = b) ∧ (a ∉ p.delNodes ∧ b ∉ p.delNodes) := by
        intro a b
        simp [List.mem_filter, mem_pairs, liveQ]
      simp only [mem_processPairs _ hps]
      simp only [withDel, hr', List.append_nil, Bool.false_eq_true, ite_false, okBound,
        Option.all_some, Option.all_none, beq_iff_eq, List.not_mem_nil, not_false_eq_true]
      constructor
      · rintro ⟨e, x, hx, _, _, hf, _, ⟨hs, hd⟩, h1, h2⟩
        subst h1 h2
        exact ⟨hs, hd, x, hx, rfl, rfl⟩
      · rintro ⟨hs, hd, x, hx, h1, h2⟩
        refine ⟨x.id, x, hx, rfl, ?_⟩
        subst h1 h2
        simp_all
    simp only [List.mem_filter, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not, mem_chain h, Reach, mem_pairs, key]
    constructor
    · rintro ⟨⟨s, ⟨hF, hs⟩, k, ⟨x, hx, h1, h2⟩, rfl⟩, hd⟩
      exact ⟨s, hF, hs, hd, x, hx, h1, h2⟩
    · rintro ⟨s, hF, hs, hd, x, hx, h1, h2⟩
      exact ⟨⟨s, ⟨hF, hs⟩, d, ⟨x, hx, h1, h2⟩, rfl⟩, hd⟩
  · simp at hM

/-! ## `DETACH DELETE a MERGE (a)-[r:R]->(x:B)` (#3576) -/

/-- `a = 0` bound, `x` unbound. -/
def rowA : RowIn := ⟨some 0, none, []⟩

/-- `MATCH (a:A) DETACH DELETE a MERGE (a)-[r:R]->(x:B)`: the doomed edge `0 -> 1` is not
matched, so MERGE takes its create branch (which refuses the deleted endpoint). -/
theorem main_3576_empty : condTraverseP gAB1 pDetach false true rowA = [] := by decide

/-- A pending-deleted target (`x` side) is not matched either. -/
theorem main_3576_empty_dst : condTraverseP gAB1 ⟨[1], []⟩ false true rowA = [] := by decide

/-- An explicitly `DELETE`d edge is not matched (emit and collapse mode). -/
theorem main_3576_deleted_edge :
    condTraverseP gAB1 ⟨[], [0]⟩ false true rowA = [] ∧
    condTraverseP gAB1 ⟨[], [0]⟩ false false rowA = [] := by decide

/-- With nothing pending the edge is matched, as before. -/
theorem main_3576_live_edge : condTraverseP gAB1 ⟨[], []⟩ false true rowA = [(0, 1, 0)] := by
  decide

end OpsTraverse.CTP
