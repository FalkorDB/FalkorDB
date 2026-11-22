import OpsTraverse.Basic
/-
# CondTraverse / ExpandInto: single-hop expansion, per-row and batched F·A

| here | there |
| --- | --- |
| `RowIn`                  | the endpoint bindings `from_id` / `to_id` read at `cond_traverse.rs` and the sibling-edge columns read by `edge_already_used` (`runtime/ops/mod.rs`) |
| `processPairs`           | `CondTraverseOp::process_pairs` (`cond_traverse.rs`): bound-endpoint filter; no inline-attr checks since #2390 (the planner lowers them to a Filter) |
| `expandRow`              | `CondTraverseOp::expand_row` (`cond_traverse.rs`): forward scan, then the reverse scan with `s != d` (the multi-type twin `expand_union_row` , `UnionGlue`) for bidirectional patterns |
| `expandInto`             | `ExpandIntoOp::expand_row` (`expand_into.rs`): pairs `[(src,dst),(dst,src)]`, the second only when bidirectional and `src != dst` |
| `Mat`, `MxmSpec`         | **GraphBLAS boundary**: `Matrix::delta_lmxm` = `GrB_mxm` over the structural `ANY_PAIR` semiring (`graphblas/matrix.rs`; proved in `GBWDelta.deltaLmxm_correct`); since #3639 the traversal counts with `count_lmxm` (`PLUS_PAIR`), whose support is this product |
| `mxmRef`                 | a reference implementation proving `MxmSpec` is satisfiable |
| `chain`                  | `expand_batch`'s `F = F·A₀·A₁·…` (`cond_traverse.rs`); `batched_eq_collapse` gives the support of the path-count frontier |
| `multiEdge`, `expandRowN`, `expandIntoN` | #3639: `expand_row` collapses only when `!(emit \|\| multi_edge \|\| edge_type_indices.len() > 1)` (`build_state`'s `has_multi_edge`), `ExpandIntoOp::expand_row` when `!(emit \|\| multi_edge)`: then collapse = emit (`expandRowN_eq_emit`, `expandIntoN_eq_emit`) |
| `processPairsIdFree`     | `process_pairs` with `id_free` (no id read or compared): `counts.get(pair).unwrap_or(1)` rows of edge 0 |
| `nuRetain`               | `CondTraverseOp::next`'s node-unique retain over `node_comparable_siblings` / `sibling_edge_pairs` |
| `hopCount`               | `TraversalMatrix::count_lmxm_into` / `Matrix::count_lmxm` (`PLUS_PAIR`) plus `add_counted_product` of `parallel_surplus` |
| `walks2`, `sharedLoops`  | `expand_batch_rows`' `chain_overlap` subtraction of `shared_self_loops` |
| `twoHopN`                | two hops with the #3639 collapse rule (W2-traverse-1 fixed: `twoHopN_eq_emit`) |

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

/-- `expand_row` (untransposed): forward scan, plus the reverse scan over
non-loop pairs for an undirected pattern. -/
def expandRow (g : Graph) (bidir emit : Bool) (r : RowIn) : List (Nat × Nat × Nat) :=
  processPairs g (pairs g) false r emit ++
    (if bidir then processPairs g ((pairs g).filter fun p => p.1 != p.2) true r emit else [])

/-- `ExpandIntoOp::expand_row`: both endpoints bound. -/
def expandInto (g : Graph) (bidir emit : Bool) (src dst : Nat) (used : List Nat) :
    List Nat :=
  let ps := if bidir && src != dst then [(src, dst), (dst, src)] else [(src, dst)]
  ps.flatMap fun p =>
    let es := (edgesBetween g p.1 p.2).filter fun e => !(used.contains e)
    if emit then es else es.take 1

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
  simp only [expandRow, Bool.false_eq_true, ite_false, List.append_nil]
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
  simp only [expandRow, ite_true, List.mem_append]
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
      simp only [expandRow, Bool.false_eq_true, ite_false, List.append_nil] at he ⊢
      exact collapse_sub_emit he)
    obtain ⟨x, hx, rfl, h1, h2, h3, h4, h5⟩ := this
    exact ⟨x, hx, h1, h2, h3, h4, h5⟩
  · rintro ⟨x, hx, rfl, rfl, hu, hf, ht⟩
    simp only [expandRow, Bool.false_eq_true, ite_false, List.append_nil, processPairs,
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
  simp only [expandRow, Bool.false_eq_true, ite_false, List.append_nil]
  exact (nodup_pairs g).sublist (collapse_keys_sublist g r _)

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

/-! ## Collapse only where it is exact (#3639)

`expand_row` and `ExpandIntoOp::expand_row` force emit mode when the relation holds a pair
with parallel edges (`CtState::multi_edge`, `Tensor::has_multi_edge`) or spans several types
(`edge_type_indices.len() > 1`, so that `Graph` is the union of several tensors). Collapse is
then reached only on a relation where every pair holds at most one edge, where it is emit mode:
an anonymous or unread edge counts edges, as openCypher does, not node pairs. -/

/-- `CtState::multi_edge`: some pair holds more than one edge. -/
def multiEdge (g : Graph) : Bool :=
  (pairs g).any fun p => decide (1 < (edgesBetween g p.1 p.2).length)

theorem edgesBetween_nil_of_not_pair {g : Graph} {s d : Nat} (hp : (s, d) ∉ pairs g) :
    edgesBetween g s d = [] := by
  apply List.eq_nil_iff_forall_not_mem.2
  intro e he
  obtain ⟨x, hx, -, h1, h2⟩ := mem_edgesBetween.1 he
  exact hp (mem_pairs.2 ⟨x, hx, h1, h2⟩)

theorem edgesBetween_length_le_one {g : Graph} (h : multiEdge g = false) (s d : Nat) :
    (edgesBetween g s d).length ≤ 1 := by
  by_cases hp : (s, d) ∈ pairs g
  · have := List.any_eq_false.1 h (s, d) hp
    simp only [decide_eq_true_eq] at this
    omega
  · simp [edgesBetween_nil_of_not_pair hp]

theorem pairRows_collapse_eq_emit {g : Graph} (h : multiEdge g = false) (isRev : Bool)
    (r : RowIn) (p : Nat × Nat) : pairRows g isRev r false p = pairRows g isRev r true p := by
  have hl : (pes g r p).length ≤ 1 :=
    Nat.le_trans (List.length_filter_le _ _) (edgesBetween_length_le_one h p.1 p.2)
  simp [pairRows, List.take_of_length_le hl]

theorem processPairs_collapse_eq_emit {g : Graph} (h : multiEdge g = false)
    (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn) :
    processPairs g ps isRev r false = processPairs g ps isRev r true := by
  simp only [processPairs]
  congr 1
  funext p
  exact pairRows_collapse_eq_emit h isRev r p

theorem expandRow_collapse_eq_emit {g : Graph} (h : multiEdge g = false) (bidir : Bool)
    (r : RowIn) : expandRow g bidir false r = expandRow g bidir true r := by
  simp only [expandRow, processPairs_collapse_eq_emit h]

/-- `expand_row` (#3639): `emit_relationship || multi_edge || edge_type_indices.len() > 1`. -/
def expandRowN (g : Graph) (bidir emit multiType : Bool) (r : RowIn) : List (Nat × Nat × Nat) :=
  expandRow g bidir (emit || multiEdge g || multiType) r

/-- **A single hop is emit mode whatever the edge's use**: an anonymous or unread edge
returns one row per matching edge, so `emit_directed_iff` and `emit_bidir_iff` describe every
hop. -/
theorem expandRowN_eq_emit (g : Graph) (bidir emit mt : Bool) (r : RowIn) :
    expandRowN g bidir emit mt r = expandRow g bidir true r := by
  unfold expandRowN
  cases hm : multiEdge g
  · cases emit <;> cases mt <;> simp [expandRow_collapse_eq_emit hm]
  · simp

theorem expandRowN_directed_iff {g : Graph} {emit mt : Bool} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRowN g false emit mt r ↔
      ∃ x ∈ g, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t := by
  rw [expandRowN_eq_emit]; exact emit_directed_iff

theorem expandRowN_bidir_iff {g : Graph} {emit mt : Bool} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRowN g true emit mt r ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  rw [expandRowN_eq_emit]; exact emit_bidir_iff

theorem expandInto_collapse_eq_emit {g : Graph} (h : multiEdge g = false) (b : Bool)
    (s d : Nat) (used : List Nat) : expandInto g b false s d used = expandInto g b true s d used := by
  simp only [expandInto, Bool.false_eq_true, ite_false, ite_true]
  congr 1
  funext p
  exact List.take_of_length_le
    (Nat.le_trans (List.length_filter_le _ _) (edgesBetween_length_le_one h p.1 p.2))

/-- `ExpandIntoOp::expand_row` (#3639): `emit_relationship || multi_edge`, where its `multi_edge`
includes `indices.len() > 1`. -/
def expandIntoN (g : Graph) (bidir emit multiType : Bool) (src dst : Nat) (used : List Nat) :
    List Nat :=
  expandInto g bidir (emit || multiEdge g || multiType) src dst used

theorem expandIntoN_eq_emit (g : Graph) (b emit mt : Bool) (s d : Nat) (used : List Nat) :
    expandIntoN g b emit mt s d used = expandInto g b true s d used := by
  unfold expandIntoN
  cases hm : multiEdge g
  · cases emit <;> cases mt <;> simp [expandInto_collapse_eq_emit hm]
  · simp

theorem mem_expandIntoN {g : Graph} {b emit mt : Bool} {s d e : Nat} {used : List Nat} :
    e ∈ expandIntoN g b emit mt s d used ↔
      e ∉ used ∧ ∃ x ∈ g, x.id = e ∧
        ((x.src = s ∧ x.dst = d) ∨ (b = true ∧ s ≠ d ∧ x.src = d ∧ x.dst = s)) := by
  rw [expandIntoN_eq_emit]; exact mem_expandInto_emit

/-- `a→b` twice and `b→a` once: the undirected anonymous `(a)-[]-(b)` returns one row per
edge, 3 for `(1,2)`; before #3639 the per-direction collapse returned 2 (W2-traverse-3, C 1). -/
def gAB : Graph := [⟨0, 1, 2⟩, ⟨1, 2, 1⟩, ⟨2, 1, 2⟩]

theorem bidir_anonymous_per_edge :
    ((expandRowN gAB true false false ⟨none, none, []⟩).map fun x => (x.1, x.2.1)).count (1, 2) = 3 := by
  decide

theorem expandIntoN_bidir_per_edge :
    (expandIntoN gAB true false false 1 2 []).length = 3 := by decide

/-! ### `id_free` rows and the node-compared siblings

With `node_unique` set (`node_comparable_siblings`: one type, every sibling a plain unread
directed traverse) nothing reads or compares an edge id: `process_pairs` pushes, per pair, one
placeholder row (edge id 0) per edge, and `next` drops, per pair, as many rows as siblings
bound to that pair of nodes. -/

/-- `process_pairs` with `id_free`: `counts.get(pair).unwrap_or(1)` rows, the pair's edge
count (`counts` holds the pairs with parallel edges; collapse is reached only when there are
none). -/
def pairRowsIdFree (g : Graph) (isRev : Bool) (r : RowIn) (p : Nat × Nat) :
    List (Nat × Nat × Nat) :=
  if pcond r isRev p then List.replicate (edgesBetween g p.1 p.2).length (pf isRev p, pt isRev p, 0)
  else []

def processPairsIdFree (g : Graph) (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  ps.flatMap (pairRowsIdFree g isRev r)

/-- The `id_free` rows are the emitted rows with the edge id blanked (the sibling list is empty
in this mode: `sibling_edges = &[]`). -/
theorem processPairsIdFree_eq (g : Graph) (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn)
    (hr : r.used = []) :
    processPairsIdFree g ps isRev r =
      (processPairs g ps isRev r true).map fun x => (x.1, x.2.1, 0) := by
  simp only [processPairsIdFree, processPairs, List.map_flatMap]
  congr 1
  funext p
  simp only [pairRowsIdFree, pairRows, ite_true]
  split
  · simp [pes, hr, Function.comp_def, List.map_const', List.filter_eq_self.2]
  · simp

/-- `next`'s `retain` for `node_unique`: a row whose node pair `key` is shared by `shared k`
siblings is dropped until that many rows of the pair have been taken. -/
def nuRetain {R K : Type} [DecidableEq K] (key : R → K) (shared : K → Nat) :
    (K → Nat) → List R → List R
  | _, [] => []
  | taken, x :: xs =>
    if shared (key x) = 0 then x :: nuRetain key shared taken xs
    else
      let taken' := fun j => if j = key x then taken (key x) + 1 else taken j
      if taken (key x) + 1 > shared (key x) then x :: nuRetain key shared taken' xs
      else nuRetain key shared taken' xs

theorem nuRetain_count {R K : Type} [DecidableEq K] (key : R → K) (shared : K → Nat) (k : K)
    (taken : K → Nat) (xs : List R) :
    (nuRetain key shared taken xs).countP (fun x => key x = k) =
      xs.countP (fun x => key x = k) - (shared k - taken k) := by
  induction xs generalizing taken with
  | nil => simp [nuRetain]
  | cons x xs ih =>
    unfold nuRetain
    by_cases h0 : shared (key x) = 0
    · simp only [h0, ite_true, List.countP_cons, ih taken]
      by_cases hk : key x = k
      · subst hk; simp [h0]
      · simp [hk]
    · simp only [h0, ite_false]
      have e := ih (fun j => if j = key x then taken (key x) + 1 else taken j)
      by_cases hk : key x = k
      · subst hk
        simp only [ite_true] at e
        by_cases ht : taken (key x) + 1 > shared (key x)
        · simp only [ht, ite_true, List.countP_cons, e]; simp; omega
        · simp only [ht, ite_false, List.countP_cons, e]; simp; omega
      · have hk' : (k = key x) = False := propext ⟨fun h => hk h.symm, False.elim⟩
        simp only [hk', ite_false] at e
        split <;> simp [e, hk]

/-- **Node-compared uniqueness counts edges**: starting from nothing taken, a pair keeps its
edge count less the siblings bound to it (`taken` starts empty per input row). -/
theorem nuRetain_idFree_count (g : Graph) (ps : List (Nat × Nat)) (r : RowIn)
    (siblings : List (Nat × Nat)) (f t : Nat) :
    (nuRetain (fun x : Nat × Nat × Nat => (x.1, x.2.1)) siblings.count (fun _ => 0)
        (processPairsIdFree g ps false r)).countP (fun x => (x.1, x.2.1) = (f, t)) =
      (processPairsIdFree g ps false r).countP (fun x => (x.1, x.2.1) = (f, t)) -
        siblings.count (f, t) := by
  rw [nuRetain_count]; simp

/-! ### Path counts on the batched path

`expand_batch` multiplies a `u64` frontier: `count_lmxm` over `PLUS_PAIR` counts each pair once,
and `add_counted_product` adds `f · surplus`, the edges each pair holds beyond the first. -/

/-- One hop from the frontier `F` (a count per node of `nodes`) to `d`: the pair product plus
the surplus product. -/
def hopCount (g : Graph) (nodes : List Nat) (F : Nat → Nat) (d : Nat) : Nat :=
  (nodes.map fun k => F k * (if (k, d) ∈ pairs g then 1 else 0)).sum +
    (nodes.map fun k => F k * ((edgesBetween g k d).length - 1)).sum

theorem pair_plus_surplus (g : Graph) (k d : Nat) :
    (if (k, d) ∈ pairs g then 1 else 0) + ((edgesBetween g k d).length - 1) =
      (edgesBetween g k d).length := by
  by_cases hp : (k, d) ∈ pairs g
  · obtain ⟨x, hx, h1, h2⟩ := mem_pairs.1 hp
    have : x.id ∈ edgesBetween g k d := mem_edgesBetween.2 ⟨x, hx, rfl, h1, h2⟩
    have := List.length_pos_of_mem this
    simp only [hp, ite_true]; omega
  · simp [hp, edgesBetween_nil_of_not_pair hp]

/-- **The count is per edge**: the pair product plus the surplus is the number of edges from the
frontier to `d`, each weighted by the paths reaching its source. -/
theorem hopCount_eq (g : Graph) (nodes : List Nat) (F : Nat → Nat) (d : Nat) :
    hopCount g nodes F d = (nodes.map fun k => F k * (edgesBetween g k d).length).sum := by
  have hp : ∀ k, F k * (if (k, d) ∈ pairs g then 1 else 0) +
      F k * ((edgesBetween g k d).length - 1) = F k * (edgesBetween g k d).length := fun k => by
    rw [← Nat.mul_add, pair_plus_surplus]
  unfold hopCount
  induction nodes with
  | nil => simp
  | cons k ks ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [← hp k, ← ih]
    omega

/-- The two-hop walks `e1 : s → v` (in `g1`), `e2 : v → d` (in `g2`): what `f = f·A₀·A₁`
counts for a fused two-hop chain. -/
def walks2 (g1 g2 : Graph) (s d : Nat) : List (Edge × Edge) :=
  (g1.filter fun e => e.src = s).flatMap fun e1 =>
    (g2.filter fun e2 => e2.src = e1.dst ∧ e2.dst = d).map fun e2 => (e1, e2)

/-- The self-loops at `v` both hops can bind (`shared_self_loops`). -/
def sharedLoops (g1 g2 : Graph) (v : Nat) : List Edge :=
  g1.filter fun e => e.src = v ∧ e.dst = v ∧ e ∈ g2

theorem countP_eq_nodup {α : Type} [DecidableEq α] (a : α) (q : Prop) [Decidable q] :
    ∀ (l : List α), l.Nodup → l.countP (fun x => decide (x = a ∧ q)) = if a ∈ l ∧ q then 1 else 0
  | [], _ => by simp
  | b :: l, h => by
    rw [List.nodup_cons] at h
    rw [List.countP_cons, countP_eq_nodup a q l h.2]
    by_cases hb : b = a
    · subst hb; by_cases hq : q <;> simp [hq, h.1]
    · by_cases hm : a ∈ l
      · simp [hb, hm]
      · have : ¬ a = b := fun e => hb e.symm
        simp [hb, hm, this]

theorem walks2_eq_len (g1 g2 : Graph) (h2 : g2.Nodup) (s d : Nat) :
    (walks2 g1 g2 s d).countP (fun w => decide (w.1 = w.2)) =
      if s = d then (sharedLoops g1 g2 s).length else 0 := by
  induction g1 with
  | nil => simp [walks2, sharedLoops]
  | cons a t ih =>
    have e : walks2 (a :: t) g2 s d =
        (if a.src = s then (g2.filter fun e2 => e2.src = a.dst ∧ e2.dst = d).map (fun e2 => (a, e2))
          else []) ++ walks2 t g2 s d := by
      unfold walks2; rw [List.filter_cons]
      by_cases ha : a.src = s <;> simp [ha]
    rw [e, List.countP_append, ih]
    have inner : ((g2.filter fun e2 => e2.src = a.dst ∧ e2.dst = d).map (fun e2 => (a, e2))).countP
        (fun w => decide (w.1 = w.2)) = if a ∈ g2 ∧ (a.src = a.dst ∧ a.dst = d) then 1 else 0 := by
      rw [List.countP_map, List.countP_filter, ← countP_eq_nodup a _ g2 h2]
      congr 1; funext x
      by_cases hx : x = a
      · subst hx; simp
      · have : ¬ a = x := fun h => hx h.symm
        simp [hx, this]
    by_cases ha : a.src = s
    · subst ha
      simp only [ite_true, inner]
      by_cases hsd : a.src = d
      · subst hsd
        simp only [sharedLoops, List.filter_cons, ite_true]
        by_cases h1 : a.dst = a.src <;> by_cases hm : a ∈ g2 <;> simp [h1, hm, eq_comm] <;> omega
      · have : ¬ (a ∈ g2 ∧ a.src = a.dst ∧ a.dst = d) := by
          rintro ⟨-, h1, h3⟩; exact hsd (by rw [h1, h3])
        simp [this, hsd]
    · simp only [ha, ite_false, List.countP_nil, Nat.zero_add]
      by_cases hsd : s = d
      · subst hsd; simp [sharedLoops, ha]
      · simp [hsd]

/-- **The self-loop subtraction is exact** (`shared_self_loops`, `chain_overlap`): the two-hop
walks that bind two distinct edges are the walks less, when the chain returns to its start, the
self-loops at the start both hops match. A repeated edge is a self-loop used twice; any other
repeat needs three hops or an undirected one, which `can_fuse` refuses. -/
theorem walks2_distinct_count (g1 g2 : Graph) (h2 : g2.Nodup) (s d : Nat) :
    (walks2 g1 g2 s d).countP (fun w => decide (w.1 ≠ w.2)) =
      (walks2 g1 g2 s d).length - (if s = d then (sharedLoops g1 g2 s).length else 0) := by
  simp only [ne_eq]
  rw [← walks2_eq_len g1 g2 h2 s d,
    List.length_eq_countP_add_countP (fun w : Edge × Edge => decide (w.1 = w.2))]
  simp only [decide_eq_true_eq]
  omega

/-! ## Two hops, sibling uniqueness, and the collapse of an unreferenced edge

`(a)-[r]->(x)<-[s]-(c)`: the second hop reads `r` for uniqueness
(`edge_already_used`). `reduce_expand_into` (`planner/optimizer/reduce_expand_into.rs`)
collapses `r` and `s` when no ancestor *expression* names them, ignoring
that the ancestor traverse's `sibling_edges` does. Since #3639 the operator collapses only
a relation without parallel edges, where the representative edge is the only one, so the
collapse is exact (`twoHopN_eq_emit`). -/

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

/-- The two hops since #3639: each collapses only where it is exact
(`emit || multi_edge`). -/
def twoHopN (g : Graph) (nodes : List Nat) (emit1 emit2 : Bool) :
    List (Nat × Nat × Nat × Nat × Nat) :=
  twoHop g nodes (emit1 || multiEdge g) (emit2 || multiEdge g)

/-- **Fixed (#3639, was W2-traverse-1)**: collapsing an unreferenced edge no longer changes
the rows, whatever the planner decided — the plan is the emitting one, `twoHop_emit_iff`. -/
theorem twoHopN_eq_emit (g : Graph) (nodes : List Nat) (e1 e2 : Bool) :
    twoHopN g nodes e1 e2 = twoHop g nodes true true := by
  unfold twoHopN
  cases hm : multiEdge g
  · cases e1 <;> cases e2 <;>
      simp [twoHop, expandRow_collapse_eq_emit hm, processPairs_collapse_eq_emit hm]
  · simp

/-- The W2-traverse-1 repro graph: both hops collapsed still give C's 2 rows (it gave 1). -/
theorem twoHopN_gPar_two_rows : (twoHopN gPar [1, 2, 3] false false).length = 2 := by
  decide

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
batched output has no duplicate `(i,d)`. Since #3639 the batched path counts
(`hopCount_eq`): this is the support of its count matrix, the count being the
per-edge row count. -/
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
