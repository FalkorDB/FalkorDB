import OpsTraverse.Basic
/-
# CondTraverse / ExpandInto: single-hop expansion, per-row and batched F·A

| here | there |
| --- | --- |
| `RowIn`                  | the endpoint bindings `from_id` / `to_id` read at `cond_traverse.rs` and the sibling-edge columns read by `edge_already_used` (`runtime/ops/mod.rs`) |
| `processPairs`           | `CondTraverseOp::process_pairs` (`cond_traverse.rs`): bound-endpoint filter; no inline-attr checks since #2390 (the planner lowers them to a Filter) |
| `expandRow`              | `CondTraverseOp::expand_row` (`cond_traverse.rs`): forward scan, then the reverse scan with `s != d` (the multi-type twin `expand_union_row` , `UnionGlue`) for bidirectional patterns |
| `revBlock`, `dropRev`    | the reverse block of `expand_row` / `expand_union_row`; in collapse mode `CondTraverseOp::drop_reverse_duplicates` (#3107) |
| `expandInto`             | `ExpandIntoOp::expand_row` (`expand_into.rs`): pairs `[(src,dst),(dst,src)]`, the second only when bidirectional and `src != dst`; collapse stops at the first pair with an edge (#3107) |
| `Mat`, `MxmSpec`         | **GraphBLAS boundary**: `Matrix::delta_lmxm` = `GrB_mxm` over the structural `ANY_PAIR` semiring (`cond_traverse.rs`, `graphblas/matrix.rs`; proved in `UnionGlue.deltaInto_eff`, `GBWDelta.deltaLmxm_correct`) |
| `mxmRef`                 | a reference implementation proving `MxmSpec` is satisfiable |
| `chain`                  | `expand_batch`'s `F = F·A₀·A₁·…` (`cond_traverse.rs`) |

The GraphBLAS primitive is not given an `axiom`: every theorem about the
batched path takes an arbitrary `mxm` together with a proof of `MxmSpec mxm`,
the property the GraphBLAS C API specifies for `GrB_mxm` with the
`GxB_ANY_PAIR_BOOL` semiring (C(i,j) is present iff ∃k. A(i,k) ∧ B(k,j), and a
GraphBLAS matrix holds each (i,j) at most once).
-/

namespace OpsTraverse

/-- The part of an input row the single-hop operators read. -/
structure RowIn where
  fromId : Option Nat
  toId   : Option Nat
  /-- Edge ids bound to *sibling* relationship aliases on this row. -/
  used   : List Nat
deriving DecidableEq, Repr

def okBound (b : Option Nat) (n : Nat) : Bool := b.all (· == n)

/-- Pattern endpoints of matrix pair `p` (swapped on the reverse scan). -/
def pf (isRev : Bool) (p : Nat × Nat) : Nat := if isRev then p.2 else p.1
def pt (isRev : Bool) (p : Nat × Nat) : Nat := if isRev then p.1 else p.2
/-- Bound-endpoint check (`cond_traverse.rs`). -/
def pcond (r : RowIn) (isRev : Bool) (p : Nat × Nat) : Bool :=
  okBound r.fromId (pf isRev p) && okBound r.toId (pt isRev p)
/-- Unused edges on the pair (`edge_already_used`, `ops/mod.rs`). -/
def pes (g : Graph) (r : RowIn) (p : Nat × Nat) : List Nat :=
  (edgesBetween g p.1 p.2).filter fun e => !(r.used.contains e)

/-- One matrix pair of `process_pairs`: every unused edge when
`emit_relationship`, else only the first. -/
def pairRows (g : Graph) (isRev : Bool) (r : RowIn) (emit : Bool) (p : Nat × Nat) :
    List (Nat × Nat × Nat) :=
  if pcond r isRev p then
    (if emit then pes g r p else (pes g r p).take 1).map fun e => (pf isRev p, pt isRev p, e)
  else []

def processPairs (g : Graph) (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn)
    (emit : Bool) : List (Nat × Nat × Nat) :=
  ps.flatMap (pairRows g isRev r emit)

theorem mem_pes {g : Graph} {r : RowIn} {p : Nat × Nat} {e : Nat} :
    e ∈ pes g r p ↔ (∃ x ∈ g, x.id = e ∧ x.src = p.1 ∧ x.dst = p.2) ∧ e ∉ r.used := by
  simp [pes, mem_edgesBetween]

theorem mem_pairRows {g : Graph} {isRev emit : Bool} {r : RowIn} {p : Nat × Nat}
    {x : Nat × Nat × Nat} :
    x ∈ pairRows g isRev r emit p ↔
      pcond r isRev p = true ∧
        ∃ e ∈ (if emit then pes g r p else (pes g r p).take 1), x = (pf isRev p, pt isRev p, e) := by
  unfold pairRows
  by_cases hc : pcond r isRev p = true
  · simp only [hc, ite_true, List.mem_map, true_and]
    constructor
    · rintro ⟨e, he, rfl⟩; exact ⟨e, he, rfl⟩
    · rintro ⟨e, he, rfl⟩; exact ⟨e, he, rfl⟩
  · simp [hc]

/-- The `(from, to)` key of a result (`drop_reverse_duplicates`' `key`). -/
def key (x : Nat × Nat × Nat) : Nat × Nat := (x.1, x.2.1)

/-- One iteration of `drop_reverse_duplicates`' loop over the reverse results, on the
decomposition `rev = kept ++ dropped ++ x :: rest` (`w = kept.length`, `r = w + dropped.length`):
a kept `x` is swapped to position `w` (a no-op when nothing was dropped yet, else the first
dropped element moves to position `r`), and `w` grows. -/
def dropStep {α : Type} (seen : α → Bool) : List α × List α → α → List α × List α
  | (kept, dropped), x =>
    if seen x then (kept, dropped ++ [x])
    else match dropped with
      | [] => (kept ++ [x], [])
      | d :: ds => (kept ++ [x], ds ++ [d])

/-- `CondTraverseOp::drop_reverse_duplicates` (`cond_traverse.rs`): `out.truncate(fwd_end + w)`
keeps the first `w` reverse slots. `seen` is the binary search of the reverse result's key in
the forward results (sorted in place, or a sorted copy): `binary_search` on a sorted slice is
`Ok` iff the key is present (Rust `slice::binary_search_by_key` contract). -/
def dropRev (fwd rev : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
  (rev.foldl (dropStep fun x => (fwd.map key).contains (key x)) ([], [])).1

theorem foldl_dropStep {α : Type} (seen : α → Bool) :
    ∀ (l kept dropped : List α),
      (l.foldl (dropStep seen) (kept, dropped)).1 = kept ++ l.filter (fun x => !seen x)
  | [], kept, _ => by simp
  | x :: l, kept, dropped => by
    simp only [List.foldl_cons, dropStep]
    by_cases h : seen x = true
    · simp only [h, ite_true, List.filter_cons, Bool.not_true, Bool.false_eq_true, ite_false]
      exact foldl_dropStep seen l kept _
    · simp only [h, Bool.false_eq_true, ite_false, List.filter_cons, Bool.not_false, ite_true]
      cases dropped with
      | nil => simp only; rw [foldl_dropStep seen l]; simp
      | cons d ds => simp only; rw [foldl_dropStep seen l]; simp

/-- **`drop_reverse_duplicates` = the reverse results whose `(from, to)` the forward scan did
not produce, in scan order.** -/
theorem dropRev_eq (fwd rev : List (Nat × Nat × Nat)) :
    dropRev fwd rev = rev.filter fun x => !(fwd.map key).contains (key x) := by
  simp [dropRev, foldl_dropStep]

theorem mem_dropRev {fwd rev : List (Nat × Nat × Nat)} {x : Nat × Nat × Nat} :
    x ∈ dropRev fwd rev ↔ x ∈ rev ∧ key x ∉ fwd.map key := by
  rw [dropRev_eq]; simp [List.mem_filter]

/-- `expand_row`'s reverse block for an undirected pattern: the reverse scan over non-loop
pairs; in collapse mode (#3107) the guarded `drop_reverse_duplicates` call. -/
def revBlock (g : Graph) (ps : List (Nat × Nat)) (r : RowIn) (emit : Bool)
    (fwd : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
  let rev := processPairs g (ps.filter fun p => p.1 != p.2) true r emit
  if !emit && !fwd.isEmpty && !rev.isEmpty then dropRev fwd rev else rev

/-- The guard only skips no-op calls: collapse mode always drops the forward keys. -/
theorem revBlock_eq (g : Graph) (ps : List (Nat × Nat)) (r : RowIn) (emit : Bool)
    (fwd : List (Nat × Nat × Nat)) :
    revBlock g ps r emit fwd =
      if emit then processPairs g (ps.filter fun p => p.1 != p.2) true r emit
      else dropRev fwd (processPairs g (ps.filter fun p => p.1 != p.2) true r emit) := by
  unfold revBlock
  cases emit
  · simp only [Bool.not_false, Bool.true_and, Bool.false_eq_true, ite_false]
    split
    · rfl
    · rename_i h
      rw [dropRev_eq]
      cases hf : fwd with
      | nil => simp; exact (List.filter_eq_self.2 (by simp)).symm
      | cons a l =>
        cases hr : processPairs g (ps.filter fun p => p.1 != p.2) true r false with
        | nil => simp
        | cons b l' => simp [hf, hr] at h
  · simp

/-- `expand_row`'s two blocks over a pair stream `ps` (the single-type iterator, or the
union iterators of `expand_union_row`): forward, then (undirected) `revBlock`. -/
def expandRowP (g : Graph) (ps : List (Nat × Nat)) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  processPairs g ps false r emit ++
    (if bidir then revBlock g ps r emit (processPairs g ps false r emit) else [])

/-- `expand_row` (untransposed): forward scan, plus the reverse scan over
non-loop pairs for an undirected pattern. -/
def expandRow (g : Graph) (bidir emit : Bool) (r : RowIn) : List (Nat × Nat × Nat) :=
  expandRowP g (pairs g) bidir emit r

/-- `ExpandIntoOp::expand_row`: both endpoints bound. Collapse mode takes one representative
per pair and stops at the first pair that has one (`break 'pairs`, #3107). -/
def expandInto (g : Graph) (bidir emit : Bool) (src dst : Nat) (used : List Nat) :
    List Nat :=
  let ps := if bidir && src != dst then [(src, dst), (dst, src)] else [(src, dst)]
  let rows := ps.flatMap fun p =>
    let es := (edgesBetween g p.1 p.2).filter fun e => !(used.contains e)
    if emit then es else es.take 1
  if emit then rows else rows.take 1

/-! ## Emit mode is exactly the openCypher single-hop match set -/

theorem mem_processPairs {g : Graph} {ps : List (Nat × Nat)} {isRev : Bool} {r : RowIn}
    {f t e : Nat} (Q : Nat → Nat → Prop)
    (hps : ∀ s d, (s, d) ∈ ps ↔ (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ Q s d) :
    (f, t, e) ∈ processPairs g ps isRev r true ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        Q x.src x.dst ∧
        (if isRev then x.src = t ∧ x.dst = f else x.src = f ∧ x.dst = t) := by
  simp only [processPairs, List.mem_flatMap, mem_pairRows, ite_true]
  constructor
  · rintro ⟨⟨s, d⟩, hp, hc, e', he', hx⟩
    obtain ⟨⟨x, hx', hid, hs, hd⟩, hu⟩ := mem_pes.1 he'
    have hq := ((hps s d).1 hp).2
    simp only [Prod.mk.injEq] at hx
    obtain ⟨hf, ht, rfl⟩ := hx
    simp only at hs hd
    subst hs hd
    simp only [pcond, Bool.and_eq_true] at hc
    refine ⟨x, hx', hid, hu, hf ▸ hc.1, ht ▸ hc.2, hq, ?_⟩
    cases isRev <;> simp_all [pf, pt]
  · rintro ⟨x, hx, rfl, hu, hf, ht, hq, hdir⟩
    refine ⟨(x.src, x.dst), (hps _ _).2 ⟨⟨x, hx, rfl, rfl⟩, hq⟩, ?_⟩
    have hft : pf isRev (x.src, x.dst) = f ∧ pt isRev (x.src, x.dst) = t := by
      cases isRev <;> simp_all [pf, pt]
    refine ⟨?_, x.id, mem_pes.2 ⟨⟨x, hx, rfl, rfl, rfl⟩, hu⟩, ?_⟩
    · simp [pcond, hft.1, hft.2, hf, ht]
    · rw [hft.1, hft.2]

theorem pairs_spec (g : Graph) (s d : Nat) :
    (s, d) ∈ pairs g ↔ (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ True := by
  simp [mem_pairs]

theorem pairs_filter_spec (g : Graph) (s d : Nat) :
    (s, d) ∈ (pairs g).filter (fun p => p.1 != p.2) ↔
      (∃ x ∈ g, x.src = s ∧ x.dst = d) ∧ s ≠ d := by
  simp [List.mem_filter, mem_pairs]

/-- **Directed, emit mode**: `(f,t,e)` is produced iff `e` is an edge `f→t`
that satisfies the row's bindings and is not already used by a sibling. -/
theorem emit_directed_iff {g : Graph} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRow g false true r ↔
      ∃ x ∈ g, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t := by
  simp only [expandRow, expandRowP, revBlock_eq, Bool.false_eq_true, ite_false, List.append_nil]
  rw [mem_processPairs (fun _ _ => True) (pairs_spec g)]
  simp only [Bool.false_eq_true, ite_false]
  constructor
  · rintro ⟨x, hx, h1, h2, h3, h4, _, h5, h6⟩; exact ⟨x, hx, h1, h5, h6, h2, h3, h4⟩
  · rintro ⟨x, hx, h1, h5, h6, h2, h3, h4⟩; exact ⟨x, hx, h1, h2, h3, h4, trivial, h5, h6⟩

/-- **Undirected, emit mode**: every edge is matched in its stored orientation,
and additionally reversed unless it is a self-loop — so a self-loop is matched
exactly once (openCypher; C instead reports a self-loop twice under an
undirected variable-length pattern). -/
theorem emit_bidir_iff {g : Graph} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRow g true true r ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  simp only [expandRow, expandRowP, revBlock_eq, ite_true, List.mem_append]
  rw [mem_processPairs (fun _ _ => True) (pairs_spec g),
    mem_processPairs (fun s d => s ≠ d) (pairs_filter_spec g)]
  simp only [Bool.false_eq_true, ite_false, ite_true]
  constructor
  · rintro (⟨x, hx, h1, h2, h3, h4, _, h5⟩ | ⟨x, hx, h1, h2, h3, h4, hne, h5⟩)
    · exact ⟨x, hx, h1, h2, h3, h4, Or.inl h5⟩
    · exact ⟨x, hx, h1, h2, h3, h4, Or.inr ⟨hne, h5⟩⟩
  · rintro ⟨x, hx, h1, h2, h3, h4, (h5 | ⟨hne, h5⟩)⟩
    · exact Or.inl ⟨x, hx, h1, h2, h3, h4, trivial, h5⟩
    · exact Or.inr ⟨x, hx, h1, h2, h3, h4, hne, h5⟩

/-! ## Collapse mode (`emit_relationship = false`) -/

/-- Collapse only ever emits rows emit mode would also emit. -/
theorem collapse_sub_emit {g : Graph} {ps : List (Nat × Nat)} {isRev : Bool} {r : RowIn}
    {x : Nat × Nat × Nat} :
    x ∈ processPairs g ps isRev r false → x ∈ processPairs g ps isRev r true := by
  simp only [processPairs, List.mem_flatMap, mem_pairRows, Bool.false_eq_true, ite_false,
    ite_true]
  rintro ⟨p, hp, hc, e, he, hx⟩
  exact ⟨p, hp, hc, e, List.mem_of_mem_take he, hx⟩

/-- A collapsed row exists for a bound pair iff some unused edge carries it. -/
theorem collapse_directed_iff {g : Graph} {r : RowIn} {f t : Nat} :
    (∃ e, (f, t, e) ∈ expandRow g false false r) ↔
      ∃ x ∈ g, x.src = f ∧ x.dst = t ∧ x.id ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t := by
  constructor
  · rintro ⟨e, he⟩
    have := (emit_directed_iff (g := g)).1 (by
      simp only [expandRow, expandRowP, revBlock_eq, Bool.false_eq_true, ite_false, List.append_nil] at he ⊢
      exact collapse_sub_emit he)
    obtain ⟨x, hx, rfl, h1, h2, h3, h4, h5⟩ := this
    exact ⟨x, hx, h1, h2, h3, h4, h5⟩
  · rintro ⟨x, hx, rfl, rfl, hu, hf, ht⟩
    simp only [expandRow, expandRowP, Bool.false_eq_true, ite_false, List.append_nil, processPairs,
      List.mem_flatMap, mem_pairRows]
    have hne : x.id ∈ pes g r (x.src, x.dst) := mem_pes.2 ⟨⟨x, hx, rfl, rfl, rfl⟩, hu⟩
    have hc : pcond r false (x.src, x.dst) = true := by simp [pcond, pf, pt, hf, ht]
    revert hne
    cases hes : pes g r (x.src, x.dst) with
    | nil => intro hne; simp at hne
    | cons e0 rest =>
      intro _
      refine ⟨e0, (x.src, x.dst), mem_pairs.2 ⟨x, hx, rfl, rfl⟩, hc, e0, ?_, rfl⟩
      simp [hes]

/-- Directed collapse emits each matrix pair at most once: its `(from,to)`
keys form a sublist of the (duplicate-free) pair list. This is the C
FalkorDB "one row per (src,dst)" semantics for anonymous edges. -/
theorem pairRows_keys (g : Graph) (r : RowIn) (p : Nat × Nat) :
    ((pairRows g false r false p).map fun x => (x.1, x.2.1)) = [] ∨
    ((pairRows g false r false p).map fun x => (x.1, x.2.1)) = [p] := by
  unfold pairRows
  by_cases hc : pcond r false p = true
  · simp only [hc, ite_true, Bool.false_eq_true, ite_false, List.map_map]
    cases pes g r p <;> simp [pf, pt]
  · simp [hc]

theorem collapse_keys_sublist (g : Graph) (r : RowIn) :
    ∀ ps : List (Nat × Nat),
      ((processPairs g ps false r false).map fun x => (x.1, x.2.1)).Sublist ps
  | [] => by simp [processPairs]
  | p :: ps => by
    have ih := collapse_keys_sublist g r ps
    simp only [processPairs, List.flatMap_cons, List.map_append] at ih ⊢
    rcases pairRows_keys g r p with h | h
    · rw [h]; exact ih.cons p
    · rw [h]; exact ih.cons_cons p

theorem collapse_directed_keys_nodup (g : Graph) (r : RowIn) :
    ((expandRow g false false r).map fun x => (x.1, x.2.1)).Nodup := by
  simp only [expandRow, expandRowP, revBlock_eq, Bool.false_eq_true, ite_false, List.append_nil]
  exact (nodup_pairs g).sublist (collapse_keys_sublist g r _)

/-! ### Undirected collapse: one row per pattern pair (#3107)

C collapses `(a)-[]-(b)` to one row per `(a, b)` (`A + Aᵀ` is boolean). The reverse
scan's results whose `(from, to)` the forward scan already produced are dropped
(`drop_reverse_duplicates`), so a pair stored in both directions is emitted once. -/

/-- A collapsed row exists for `(f, t)` iff an emitted row does (any pair stream). -/
theorem collapse_exists_iff {g : Graph} {ps : List (Nat × Nat)} {isRev : Bool} {r : RowIn}
    {f t : Nat} :
    (∃ e, (f, t, e) ∈ processPairs g ps isRev r false) ↔
      ∃ e, (f, t, e) ∈ processPairs g ps isRev r true := by
  constructor
  · rintro ⟨e, he⟩; exact ⟨e, collapse_sub_emit he⟩
  · rintro ⟨e, he⟩
    simp only [processPairs, List.mem_flatMap, mem_pairRows, ite_true, Bool.false_eq_true,
      ite_false] at he ⊢
    obtain ⟨p, hp, hc, e', he', hx⟩ := he
    revert he'
    cases hes : pes g r p with
    | nil => intro h; simp at h
    | cons e0 rest =>
      intro _
      injection hx with hf hx2
      injection hx2 with ht
      exact ⟨e0, p, hp, hc, e0, by simp [hes], by rw [hf, ht]⟩

/-- Collapsed keys of one scan are a sublist of its pair stream mapped to pattern endpoints. -/
theorem collapse_keys_sublist' (g : Graph) (isRev : Bool) (r : RowIn) :
    ∀ ps : List (Nat × Nat),
      ((processPairs g ps isRev r false).map key).Sublist (ps.map fun p => (pf isRev p, pt isRev p))
  | [] => by simp [processPairs]
  | p :: ps => by
    have ih := collapse_keys_sublist' g isRev r ps
    simp only [processPairs, List.flatMap_cons, List.map_append, List.map_cons] at ih ⊢
    have h : (pairRows g isRev r false p).map key = [] ∨
        (pairRows g isRev r false p).map key = [(pf isRev p, pt isRev p)] := by
      unfold pairRows
      by_cases hc : pcond r isRev p = true
      · simp only [hc, ite_true, Bool.false_eq_true, ite_false, List.map_map]
        cases pes g r p <;> simp [key]
      · simp [hc]
    rcases h with h | h
    · rw [h]; exact ih.cons _
    · rw [h]; exact ih.cons_cons _

theorem map_pfpt_nodup (isRev : Bool) {ps : List (Nat × Nat)} (h : ps.Nodup) :
    (ps.map fun p => (pf isRev p, pt isRev p)).Nodup := by
  refine List.Pairwise.map _ (fun a b (hne : a ≠ b) heq => hne ?_) h
  cases isRev <;> simp only [pf, pt, Bool.false_eq_true, ite_false, ite_true, Prod.mk.injEq] at heq <;>
    exact Prod.ext (by omega) (by omega)

/-- **Undirected collapse, keys unique**: over any duplicate-free pair stream, each
`(from, to)` is emitted at most once. -/
theorem collapseP_bidir_keys_nodup (g : Graph) {ps : List (Nat × Nat)} (hps : ps.Nodup)
    (r : RowIn) : ((expandRowP g ps true false r).map key).Nodup := by
  simp only [expandRowP, ite_true, revBlock_eq, Bool.false_eq_true, ite_false, List.map_append]
  have h1 := (map_pfpt_nodup false hps).sublist (collapse_keys_sublist' g false r ps)
  have h2 := (map_pfpt_nodup true (hps.sublist (List.filter_sublist))).sublist
    (collapse_keys_sublist' g true r (ps.filter fun p => p.1 != p.2))
  have h3 : ((dropRev (processPairs g ps false r false)
      (processPairs g (ps.filter fun p => p.1 != p.2) true r false)).map key).Nodup := by
    rw [dropRev_eq]; exact h2.sublist (List.filter_sublist.map _)
  refine List.nodup_append.2 ⟨h1, h3, ?_⟩
  intro a ha b hb hab
  subst hab
  obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hb
  exact (mem_dropRev.1 hy).2 ha

/-- **Undirected collapse, complete and sound**: a row for `(f, t)` exists iff emit mode has
one (`emit_bidir_iff`: an unused edge `f→t`, or a non-loop `t→f`). -/
theorem collapseP_bidir_exists_iff {g : Graph} {ps : List (Nat × Nat)} {r : RowIn} {f t : Nat} :
    (∃ e, (f, t, e) ∈ expandRowP g ps true false r) ↔ ∃ e, (f, t, e) ∈ expandRowP g ps true true r := by
  simp only [expandRowP, ite_true, revBlock_eq, Bool.false_eq_true, ite_false, List.mem_append]
  constructor
  · rintro ⟨e, he | he⟩
    · exact ⟨e, Or.inl (collapse_sub_emit he)⟩
    · exact ⟨e, Or.inr (collapse_sub_emit (mem_dropRev.1 he).1)⟩
  · rintro ⟨e, he | he⟩
    · obtain ⟨e', h'⟩ := collapse_exists_iff.2 ⟨e, he⟩
      exact ⟨e', Or.inl h'⟩
    · obtain ⟨e', h'⟩ := collapse_exists_iff.2 ⟨e, he⟩
      by_cases hk : (f, t) ∈ (processPairs g ps false r false).map key
      · obtain ⟨y, hy, hyk⟩ := List.mem_map.1 hk
        refine ⟨y.2.2, Or.inl ?_⟩
        simp only [key, Prod.mk.injEq] at hyk
        rw [← hyk.1, ← hyk.2]; exact hy
      · exact ⟨e', Or.inr (mem_dropRev.2 ⟨h', hk⟩)⟩

theorem collapse_bidir_keys_nodup (g : Graph) (r : RowIn) :
    ((expandRow g true false r).map key).Nodup :=
  collapseP_bidir_keys_nodup g (nodup_pairs g) r

theorem collapse_bidir_iff {g : Graph} {r : RowIn} {f t : Nat} :
    (∃ e, (f, t, e) ∈ expandRow g true false r) ↔
      ∃ x ∈ g, x.id ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  unfold expandRow
  rw [collapseP_bidir_exists_iff]
  constructor
  · rintro ⟨e, he⟩
    obtain ⟨x, hx, rfl, h⟩ := emit_bidir_iff.1 he
    exact ⟨x, hx, h⟩
  · rintro ⟨x, hx, h⟩
    exact ⟨x.id, emit_bidir_iff.2 ⟨x, hx, rfl, h⟩⟩

/-- `a→b` twice and `b→a` once: before #3107 `(1,2)` came out twice; now once. -/
def gAB : Graph := [⟨0, 1, 2⟩, ⟨1, 2, 1⟩, ⟨2, 1, 2⟩]

theorem bidir_collapse_once_pair :
    ((expandRow gAB true false ⟨none, none, []⟩).map key).count (1, 2) = 1 := by
  decide

theorem expandInto_bidir_collapse_one_row :
    (expandInto gAB true false 1 2 []).length = 1 := by decide

/-- **ExpandInto collapse: at most one row per pattern pair** (C's semantics). -/
theorem expandInto_collapse_length_le (g : Graph) (b : Bool) (s d : Nat) (used : List Nat) :
    (expandInto g b false s d used).length ≤ 1 := by
  simp only [expandInto, Bool.false_eq_true, ite_false]
  exact List.length_take_le _ _

/-- ExpandInto collapse yields a row iff emit mode does, and that row is one emit mode yields. -/
theorem expandInto_collapse_iff (g : Graph) (b : Bool) (s d : Nat) (used : List Nat) :
    (expandInto g b false s d used ≠ [] ↔ expandInto g b true s d used ≠ []) ∧
      ∀ e ∈ expandInto g b false s d used, e ∈ expandInto g b true s d used := by
  simp only [expandInto, Bool.false_eq_true, ite_false, ite_true]
  constructor
  · rw [ne_eq, ne_eq, List.take_eq_nil_iff]
    simp only [Nat.add_one_ne_zero, false_or, List.flatMap_eq_nil_iff]
    constructor
    · intro h h'; exact h (fun p hp => List.take_eq_nil_iff.2 (Or.inr (h' p hp)))
    · intro h h'; exact h (fun p hp => (List.take_eq_nil_iff.1 (h' p hp)).resolve_left (by omega))
  · intro e he
    have he := List.mem_of_mem_take he
    simp only [List.mem_flatMap] at he ⊢
    obtain ⟨p, hp, he⟩ := he
    exact ⟨p, hp, List.mem_of_mem_take he⟩

/-- ExpandInto in emit mode returns every unused edge between the endpoints, in
either orientation when undirected, a self-loop once. -/
theorem mem_expandInto_emit {g : Graph} {b : Bool} {s d e : Nat} {used : List Nat} :
    e ∈ expandInto g b true s d used ↔
      e ∉ used ∧ ∃ x ∈ g, x.id = e ∧
        ((x.src = s ∧ x.dst = d) ∨ (b = true ∧ s ≠ d ∧ x.src = d ∧ x.dst = s)) := by
  unfold expandInto
  by_cases h : (b && s != d) = true
  · simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h
    simp [h, mem_edgesBetween]
    constructor
    · rintro (⟨⟨x, hx, h1, h2⟩, hu⟩ | ⟨⟨x, hx, h1, h2⟩, hu⟩)
      · exact ⟨hu, x, hx, h1, Or.inl h2⟩
      · exact ⟨hu, x, hx, h1, Or.inr h2⟩
    · rintro ⟨hu, x, hx, h1, (h2 | h2)⟩
      · exact Or.inl ⟨⟨x, hx, h1, h2⟩, hu⟩
      · exact Or.inr ⟨⟨x, hx, h1, h2⟩, hu⟩
  · simp [h, mem_edgesBetween]
    simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, not_and, Decidable.not_not] at h
    constructor
    · rintro ⟨⟨x, hx, h1, h2⟩, hu⟩; exact ⟨hu, x, hx, h1, Or.inl h2⟩
    · rintro ⟨hu, x, hx, h1, (h2 | ⟨hb, hne, h2⟩)⟩
      · exact ⟨⟨x, hx, h1, h2⟩, hu⟩
      · exact absurd (h hb) hne

/-! ## Two hops, sibling uniqueness, and the collapse of an unreferenced edge

`(a)-[r]->(x)<-[s]-(c)`: the second hop reads `r` for uniqueness
(`edge_already_used`). `reduce_expand_into` (`planner/optimizer/reduce_expand_into.rs`)
collapses `r` and `s` when no ancestor *expression* names them, ignoring
that the ancestor traverse's `sibling_edges` does. -/

def twoHop (g : Graph) (nodes : List Nat) (emit1 emit2 : Bool) :
    List (Nat × Nat × Nat × Nat × Nat) :=
  nodes.flatMap fun a =>
    (expandRow g false emit1 ⟨some a, none, []⟩).flatMap fun h1 =>
      (processPairs g (pairs g) true ⟨some h1.2.1, none, [h1.2.2]⟩ emit2).map
        fun h2 => (a, h1.2.1, h1.2.2, h2.2.1, h2.2.2)

/-- Emit mode on both hops is exactly the openCypher match set of
`(a)-[r]->(x)<-[s]-(c)` with `r ≠ s`. -/
theorem twoHop_emit_iff {g : Graph} {nodes : List Nat} {a x r c s : Nat} :
    (a, x, r, c, s) ∈ twoHop g nodes true true ↔
      a ∈ nodes ∧ (∃ e ∈ g, e.id = r ∧ e.src = a ∧ e.dst = x) ∧
        (∃ e ∈ g, e.id = s ∧ e.src = c ∧ e.dst = x) ∧ s ≠ r := by
  simp only [twoHop, List.mem_flatMap, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨a', ha, ⟨f1, x1, r1⟩, h1, ⟨f2, c2, s2⟩, h2, rfl, rfl, rfl, rfl, rfl⟩
    obtain ⟨e1, he1, hid1, hs1, hd1, -, hb1, -⟩ := emit_directed_iff.1 h1
    obtain ⟨e2, he2, hid2, hu2, hb2, -, -, hdir⟩ :=
      (mem_processPairs (fun _ _ => True) (pairs_spec g)).1 h2
    simp only [okBound, Option.all_some, beq_iff_eq] at hb1 hb2
    simp only [ite_true] at hdir
    refine ⟨ha, ⟨e1, he1, hid1, hs1.trans hb1.symm, hd1⟩, ⟨e2, he2, hid2, hdir.1, hdir.2.trans hb2.symm⟩, ?_⟩
    simpa using hu2
  · rintro ⟨ha, ⟨e1, he1, rfl, rfl, rfl⟩, ⟨e2, he2, rfl, rfl, hd2⟩, hne⟩
    refine ⟨e1.src, ha, (e1.src, e1.dst, e1.id), ?_, (e1.dst, e2.src, e2.id), ?_, rfl, rfl, rfl, rfl, rfl⟩
    · exact emit_directed_iff.2 ⟨e1, he1, rfl, rfl, rfl, by simp, by simp [okBound], by simp [okBound]⟩
    · refine (mem_processPairs (fun _ _ => True) (pairs_spec g)).2
        ⟨e2, he2, rfl, by simpa using hne, by simp [okBound], by simp [okBound], trivial, ?_⟩
      simp [hd2]

/-- `e1, e2 : 1→2`, `e3 : 2→3` (the repro graph). -/
def gPar : Graph := [⟨0, 1, 2⟩, ⟨1, 1, 2⟩, ⟨2, 2, 3⟩]

/-- **Bug** (`lean_ops_traverse::bug_unreferenced_edge_collapse_breaks_sibling_uniqueness`):
the collapsed plan returns 1 row, the emitting plan (= openCypher = C) 2. -/
theorem twoHop_collapse_changes_count :
    (twoHop gPar [1, 2, 3] true true).length = 2 ∧
    (twoHop gPar [1, 2, 3] false false).length = 1 := by decide

/-! ## GraphBLAS boundary and the batched F·A path -/

/-- A boolean sparse matrix as its list of present coordinates. -/
abbrev Mat := List (Nat × Nat)

/-- **Axiomatised FFI primitive** (as a hypothesis, not an `axiom`): the
behaviour the GraphBLAS C API specifies for `GrB_mxm(C, NULL, NULL,
GxB_ANY_PAIR_BOOL, F, A, NULL)` — used by `Matrix::delta_lmxm`
(`cond_traverse.rs`). C(i,j) is present iff ∃k. F(i,k) ∧ A(k,j), and a
matrix stores each coordinate at most once. -/
structure MxmSpec (mxm : Mat → Mat → Mat) : Prop where
  mem   : ∀ F A i j, (i, j) ∈ mxm F A ↔ ∃ k, (i, k) ∈ F ∧ (k, j) ∈ A
  nodup : ∀ F A, (mxm F A).Nodup

def mxmRef (F A : Mat) : Mat :=
  dedup (F.flatMap fun p => (A.filter fun q => q.1 == p.2).map fun q => (p.1, q.2))

theorem mxmRef_spec : MxmSpec mxmRef where
  mem F A i j := by
    simp only [mxmRef, mem_dedup, List.mem_flatMap, List.mem_map, List.mem_filter, beq_iff_eq,
      Prod.mk.injEq]
    constructor
    · rintro ⟨⟨i', k⟩, hp, ⟨k', j'⟩, ⟨hq, hk⟩, rfl, rfl⟩
      simp only at hk; subst hk; exact ⟨k', hp, hq⟩
    · rintro ⟨k, hp, hq⟩; exact ⟨(i, k), hp, (k, j), ⟨hq, rfl⟩, rfl, rfl⟩
  nodup F A := nodup_dedup _

/-- Reachability through a sequence of pattern matrices. -/
def Reach : List Mat → Nat → Nat → Prop
  | [], a, b => a = b
  | A :: As, a, b => ∃ k, (a, k) ∈ A ∧ Reach As k b

/-- `F ← F·A₀`, then every fused chain hop (`cond_traverse.rs`). -/
def chain (mxm : Mat → Mat → Mat) (F : Mat) : List Mat → Mat
  | [] => F
  | A :: As => chain mxm (mxm F A) As

theorem mem_chain {mxm : Mat → Mat → Mat} (h : MxmSpec mxm) :
    ∀ (As : List Mat) (F : Mat) (i d : Nat),
      (i, d) ∈ chain mxm F As ↔ ∃ s, (i, s) ∈ F ∧ Reach As s d
  | [], F, i, d => by simp [chain, Reach]
  | A :: As, F, i, d => by
    rw [chain, mem_chain h As]
    simp only [Reach, h.mem]
    constructor
    · rintro ⟨s', ⟨k, hk, hA⟩, hr⟩; exact ⟨k, hk, s', hA, hr⟩
    · rintro ⟨k, hk, s', hA, hr⟩; exact ⟨s', ⟨k, hk, hA⟩, hr⟩

theorem chain_nodup {mxm : Mat → Mat → Mat} (h : MxmSpec mxm) :
    ∀ (As : List Mat) (F : Mat), As ≠ [] → (chain mxm F As).Nodup
  | [], _, hne => absurd rfl hne
  | [A], F, _ => by simp only [chain]; exact h.nodup F A
  | A :: B :: As, F, _ => chain_nodup h (B :: As) (mxm F A) (by simp)

/-- **Batched path = per-row collapse.** For any `mxm` meeting the GraphBLAS
spec, row `i` of `F·A` (F seeded with `(i, src_i)`) holds `d` exactly when the
per-row collapsed expansion of `src_i` yields a row for `(src_i, d)`; and the
batched output has no duplicate `(i,d)`. -/
theorem batched_eq_collapse {mxm : Mat → Mat → Mat} (h : MxmSpec mxm) (g : Graph)
    (F : Mat) (i d : Nat) :
    (i, d) ∈ chain mxm F [pairs g] ↔
      ∃ s, (i, s) ∈ F ∧ ∃ e, (s, d, e) ∈ expandRow g false false ⟨some s, none, []⟩ := by
  rw [mem_chain h]
  simp only [Reach, mem_pairs]
  constructor
  · rintro ⟨s, hF, k, ⟨x, hx, h1, h2⟩, rfl⟩
    exact ⟨s, hF, collapse_directed_iff.2 ⟨x, hx, h1, h2, by simp, by simp [okBound], by simp [okBound]⟩⟩
  · rintro ⟨s, hF, he⟩
    obtain ⟨x, hx, h1, h2, -, -, -⟩ := collapse_directed_iff.1 he
    exact ⟨s, hF, d, ⟨x, hx, h1, h2⟩, rfl⟩

/-- Fused chain of `n` anonymous hops: `(i,d)` iff a walk of exactly the chain
length exists — C's algebraic-expression semantics (one row per endpoint pair,
parallel edges and repeated edges not distinguished). -/
theorem fused_chain_semantics {mxm : Mat → Mat → Mat} (h : MxmSpec mxm)
    (As : List Mat) (F : Mat) (i d : Nat) :
    (i, d) ∈ chain mxm F As ↔ ∃ s, (i, s) ∈ F ∧ Reach As s d := mem_chain h As F i d

end OpsTraverse
