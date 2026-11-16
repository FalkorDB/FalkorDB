import OpsTraverse.UnionGlue
/-
# #3639: parallel-edge cardinality no longer depends on naming

| here | there |
| --- | --- |
| `multiEdge`              | `Tensor::has_multi_edge` over the traversed types (`CtState.multi_edge`, `cond_traverse.rs:702`; `expand_into.rs:187-202`); a multi-type pattern is the union graph, whose pairs can hold several edges, and both sites force emit for it (`edge_type_indices.len() > 1`) |
| `expandRowEff`           | `CondTraverseOp::expand_row` (`cond_traverse.rs:1401`): `emit_relationship \|\| multi_edge \|\| types > 1` (:1452) |
| `expandIntoEff`          | `ExpandIntoOp::expand_row` (`expand_into.rs:126`): `emit_relationship \|\| multi_edge` |
| `pairRowsIdFree`, `processPairsIdFree` | `process_pairs` (`cond_traverse.rs:1735`) with `id_free = Some(counts)`: one `RelationshipId(0)` row per pair, or `counts.get(pair).unwrap_or(1)` of them in emit mode |
| `CMat`, `cstep`, `cchain` | `expand_batch_rows` (`cond_traverse.rs:925`): `f` is `u64`, each hop `count_lmxm_into` (`PLUS_PAIR` over the pattern) plus `add_counted_product` (`PLUS_TIMES` over the parallel-edge surplus) — together `F·A` over edge counts |
| `hop2`, `selfLoops`      | the two-hop fused chain's `shared_self_loops` correction (`cond_traverse.rs:824`, applied at :1243-1251) |

The GraphBLAS counting product is a boundary like `MxmSpec`: the batched theorems are
stated over `cstep`, the count `F·A` the C API specifies for `PLUS_PAIR` + the
surplus `PLUS_TIMES` product (`graphblas_wrappers` rows `count_lmxm*`, `add_counted_product`).
-/

namespace OpsTraverse

/-! ## Collapse is taken only where it is exact -/

/-- Some pair of the relation holds more than one edge. -/
def multiEdge (g : Graph) : Bool := g.any fun x => decide (1 < (edgesBetween g x.src x.dst).length)

theorem edgesBetween_le_one {g : Graph} (h : multiEdge g = false) (s d : Nat) :
    (edgesBetween g s d).length ≤ 1 := by
  rcases hl : edgesBetween g s d with _ | ⟨e, rest⟩
  · simp
  · have he : e ∈ edgesBetween g s d := by simp [hl]
    obtain ⟨x, hx, -, rfl, rfl⟩ := mem_edgesBetween.1 he
    have := List.any_eq_false.1 h x hx
    simp only [decide_eq_true_eq, Nat.not_lt] at this
    rw [hl] at this; exact this

theorem take_one_filter {g : Graph} (h : multiEdge g = false) (s d : Nat) (f : Nat → Bool) :
    ((edgesBetween g s d).filter f).take 1 = (edgesBetween g s d).filter f :=
  List.take_of_length_le (Nat.le_trans (List.length_filter_le _ _) (edgesBetween_le_one h s d))

theorem pairRows_eff (g : Graph) (isRev : Bool) (r : RowIn) (emit : Bool) (p : Nat × Nat) :
    pairRows g isRev r (emit || multiEdge g) p = pairRows g isRev r true p := by
  unfold pairRows pes
  cases h : multiEdge g
  · cases emit <;> simp [take_one_filter h]
  · simp

/-- `expand_row` since #3639: collapse only when the relation has no parallel edge. -/
def expandRowEff (g : Graph) (bidir emit : Bool) (r : RowIn) : List (Nat × Nat × Nat) :=
  expandRow g bidir (emit || multiEdge g) r

/-- **The per-row result no longer depends on `emit_relationship`**: whatever the planner
decided, every row is the emit-mode (openCypher) row set. -/
theorem expandRowEff_eq_emit (g : Graph) (bidir emit : Bool) (r : RowIn) :
    expandRowEff g bidir emit r = expandRow g bidir true r := by
  have hf : ∀ isRev, pairRows g isRev r (emit || multiEdge g) = pairRows g isRev r true :=
    fun isRev => funext (pairRows_eff g isRev r emit)
  simp only [expandRowEff, expandRow, processPairs, hf]

theorem expandRowEff_directed_iff {g : Graph} {emit : Bool} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRowEff g false emit r ↔
      ∃ x ∈ g, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t := by
  rw [expandRowEff_eq_emit]; exact emit_directed_iff

theorem expandRowEff_bidir_iff {g : Graph} {emit : Bool} {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandRowEff g true emit r ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  rw [expandRowEff_eq_emit]; exact emit_bidir_iff

/-- `bidir_collapse_duplicates_pair` is fixed: on `gAB` (`1→2` twice, `2→1` once)
`(a)-[]-(b)` gives three `(1,2)` rows, one per edge, as with a named edge. -/
theorem expandRowEff_gAB :
    ((expandRowEff gAB true false ⟨none, none, []⟩).map fun x => (x.1, x.2.1)).count (1, 2) = 3 ∧
    ((expandRow gAB true true ⟨none, none, []⟩).map fun x => (x.1, x.2.1)).count (1, 2) = 3 := by
  decide

theorem expandRowP_eff (g : Graph) (ps : List (Nat × Nat)) (bidir emit : Bool) (r : RowIn) :
    UnionGlue.expandRowP g ps bidir (emit || multiEdge g) r = UnionGlue.expandRowP g ps bidir true r := by
  have hf : ∀ isRev, pairRows g isRev r (emit || multiEdge g) = pairRows g isRev r true :=
    fun isRev => funext (pairRows_eff g isRev r emit)
  simp only [UnionGlue.expandRowP, processPairs, hf]

/-- `expand_union_row` (`cond_traverse.rs:1618`) runs with the same forced flag
(`emit || multi_edge || types > 1`, the latter folded into `emit` here): the emit rows of
the union graph. -/
theorem expandUnionRowEff_eq_emit (S : Graph → List (Nat × Nat)) (gs : List Graph)
    (p : Nat × Nat → Bool) (bidir emit : Bool) (r : RowIn) :
    UnionGlue.expandUnionRow S gs p bidir (emit || multiEdge gs.flatten) r =
      UnionGlue.expandUnionRow S gs p bidir true r :=
  expandRowP_eff _ _ _ _ _

/-- `ExpandIntoOp::expand_row` since #3639. -/
def expandIntoEff (g : Graph) (bidir emit : Bool) (src dst : Nat) (used : List Nat) : List Nat :=
  expandInto g bidir (emit || multiEdge g) src dst used

theorem expandIntoEff_eq_emit (g : Graph) (bidir emit : Bool) (src dst : Nat) (used : List Nat) :
    expandIntoEff g bidir emit src dst used = expandInto g bidir true src dst used := by
  unfold expandIntoEff expandInto
  cases h : multiEdge g
  · cases emit <;> simp [take_one_filter h]
  · simp

theorem mem_expandIntoEff {g : Graph} {b emit : Bool} {s d e : Nat} {used : List Nat} :
    e ∈ expandIntoEff g b emit s d used ↔
      e ∉ used ∧ ∃ x ∈ g, x.id = e ∧
        ((x.src = s ∧ x.dst = d) ∨ (b = true ∧ s ≠ d ∧ x.src = d ∧ x.dst = s)) := by
  rw [expandIntoEff_eq_emit]; exact mem_expandInto_emit

/-- `expandInto_bidir_collapse_two_rows` is fixed: three edges join 1 and 2, three rows. -/
theorem expandIntoEff_gAB : (expandIntoEff gAB true false 1 2 []).length = 3 := by decide

/-! ## `id_free`: rows without an edge id (#3639)

When no operator reads or compares this edge's id (`need_edge_id = false`, i.e. the
siblings are compared by node), `process_pairs` pushes `RelationshipId(0)`: one row per
pair, or `counts.get(pair).unwrap_or(1)` rows in emit mode, where `counts` holds the
edge count of every pair with parallel edges (`iter_multi_edges`). -/

def pairRowsIdFree (isRev : Bool) (r : RowIn) (emit : Bool) (cnt : Nat × Nat → Nat)
    (p : Nat × Nat) : List (Nat × Nat × Nat) :=
  if pcond r isRev p then List.replicate (if emit then cnt p else 1) (pf isRev p, pt isRev p, 0)
  else []

def processPairsIdFree (ps : List (Nat × Nat)) (isRev : Bool) (r : RowIn) (emit : Bool)
    (cnt : Nat × Nat → Nat) : List (Nat × Nat × Nat) :=
  ps.flatMap (pairRowsIdFree isRev r emit cnt)

theorem edgesBetween_pos {g : Graph} {p : Nat × Nat} (hp : p ∈ pairs g) :
    1 ≤ (edgesBetween g p.1 p.2).length := by
  obtain ⟨x, hx, h1, h2⟩ := mem_pairs.1 (show (p.1, p.2) ∈ pairs g from hp)
  have : x.id ∈ edgesBetween g p.1 p.2 := mem_edgesBetween.2 ⟨x, hx, rfl, h1, h2⟩
  exact List.length_pos_of_mem this

/-- **Id-free rows are the emit rows with the id blanked**: the same `(from, to)` with the
same multiplicity, for every pattern pair (`cnt` = the pair's edge count, which
`counts.get(..).unwrap_or(1)` is on a pair of the pattern). -/
theorem processPairsIdFree_eq (g : Graph) (isRev emit : Bool) (r : RowIn) (cnt : Nat × Nat → Nat) :
    ∀ ps : List (Nat × Nat), (∀ p ∈ ps, p ∈ pairs g) →
      (∀ p ∈ ps, cnt p = (edgesBetween g p.1 p.2).length) →
      processPairsIdFree ps isRev r (emit || multiEdge g) cnt =
        (processPairs g ps isRev { r with used := [] } true).map fun x => (x.1, x.2.1, 0)
  | [], _, _ => rfl
  | p :: ps, hps, hc => by
    have ih := processPairsIdFree_eq g isRev emit r cnt ps (fun q hq => hps q (by simp [hq]))
      (fun q hq => hc q (by simp [hq]))
    simp only [processPairsIdFree, processPairs] at ih
    simp only [processPairsIdFree, processPairs, List.flatMap_cons, List.map_append, ih]
    congr 1
    have hp := hps p (by simp)
    have hcp := hc p (by simp)
    unfold pairRowsIdFree pairRows pes
    have hpc : pcond { r with used := [] } isRev p = pcond r isRev p := rfl
    rw [hpc]
    split
    · simp only [ite_true, List.contains_nil, Bool.not_false, List.map_map,
        List.filter_eq_self.mpr (fun _ _ => rfl)]
      rw [show ((fun x : Nat × Nat × Nat => (x.1, x.2.1, 0)) ∘ fun e => (pf isRev p, pt isRev p, e))
          = fun _ => (pf isRev p, pt isRev p, 0) from rfl, List.map_const']
      congr 1
      cases emit <;> cases h : multiEdge g <;> simp [hcp]
      have := edgesBetween_le_one h p.1 p.2
      have := edgesBetween_pos hp
      omega
    · rfl

/-! ## Siblings compared by node (#3639 `node_unique`)

A sibling of the same single type bound to the pair `(s,d)` holds one of its edges, so
the edges left for this hop are the pair's edges less those siblings: `count - shared`
(`expand_batch_rows` :1217-1240, the per-row `retain` in `next` :1946-1969). The siblings
are distinct edges (they passed the same check), which `hu` states. -/

theorem pes_length_add_shared {g : Graph} {r : RowIn} {p : Nat × Nat}
    (hnd : (edgesBetween g p.1 p.2).Nodup) (hu : r.used.Nodup) :
    (pes g r p).length + (r.used.filter fun e => e ∈ edgesBetween g p.1 p.2).length =
      (edgesBetween g p.1 p.2).length := by
  have hperm : ((edgesBetween g p.1 p.2).filter fun e => r.used.contains e).Perm
      (r.used.filter fun e => e ∈ edgesBetween g p.1 p.2) := by
    rw [List.perm_ext_iff_of_nodup (hnd.sublist List.filter_sublist)
      (hu.sublist List.filter_sublist)]
    intro a; simp [and_comm]
  rw [← hperm.length_eq, pes]
  have := List.length_eq_countP_add_countP (fun e => r.used.contains e) (l := edgesBetween g p.1 p.2)
  simp only [List.countP_eq_length_filter] at this
  simp only [Bool.not_eq_true, Bool.decide_eq_false] at this
  omega

/-! ## The batched path counts paths (#3639)

`f[i, v]` is the number of paths from row `i`'s source to `v`; each is one output row.
`cstep g F` is `F·A` over edge counts — what `count_lmxm_into` (one per stored pair) plus
`add_counted_product` (the surplus of every pair with parallel edges) compute. -/

abbrev CMat := Nat → Nat → Nat

def cstep (g : Graph) (F : CMat) : CMat :=
  fun i d => ((g.filter fun x => x.dst == d).map fun x => F i x.src).sum

def cchain (F : CMat) : List Graph → CMat
  | [] => F
  | g :: gs => cchain (cstep g F) gs

/-- `f.build(rows, srcs, ones)`: row `i` seeded at node `s`. -/
def seed (i s : Nat) : CMat := fun i' s' => if i' = i ∧ s' = s then 1 else 0

theorem sum_pos_iff {α} (l : List α) (f : α → Nat) :
    0 < (l.map f).sum ↔ ∃ x ∈ l, 0 < f x := by
  induction l with
  | nil => simp
  | cons x xs ih => simp only [List.map_cons, List.sum_cons, List.mem_cons]; constructor
                    · intro h
                      by_cases hx : 0 < f x
                      · exact ⟨x, Or.inl rfl, hx⟩
                      · obtain ⟨y, hy, h'⟩ := ih.1 (by omega); exact ⟨y, Or.inr hy, h'⟩
                    · rintro ⟨y, rfl | hy, h'⟩
                      · omega
                      · have := ih.2 ⟨y, hy, h'⟩; omega

theorem cstep_pos {g : Graph} {F : CMat} {i d : Nat} :
    0 < cstep g F i d ↔ ∃ x ∈ g, x.dst = d ∧ 0 < F i x.src := by
  simp [cstep, sum_pos_iff, List.mem_filter]
  constructor
  · rintro ⟨x, ⟨hx, h1⟩, h2⟩; exact ⟨x, hx, h1, h2⟩
  · rintro ⟨x, hx, h1, h2⟩; exact ⟨x, ⟨hx, h1⟩, h2⟩

/-- **Support = reachability**: an entry of the counted chain is non-zero exactly when the
pattern chain reaches it — the property `fused_chain_semantics` proved of the boolean
`ANY_PAIR` product, so the rows emitted (with multiplicity) are over the same pairs. -/
theorem cchain_pos : ∀ (gs : List Graph) (F : CMat) (i d : Nat),
    0 < cchain F gs i d ↔ ∃ s, 0 < F i s ∧ Reach (gs.map pairs) s d
  | [], F, i, d => by simp [cchain, Reach]
  | g :: gs, F, i, d => by
    rw [cchain, cchain_pos gs]
    simp only [List.map_cons, Reach, cstep_pos, mem_pairs]
    constructor
    · rintro ⟨k, ⟨x, hx, rfl, hF⟩, hr⟩; exact ⟨x.src, hF, x.dst, ⟨x, hx, rfl, rfl⟩, hr⟩
    · rintro ⟨s, hF, k, ⟨x, hx, rfl, rfl⟩, hr⟩; exact ⟨x.dst, ⟨x, hx, rfl, hF⟩, hr⟩

/-- The counted chain has the same support as the boolean `chain` it replaces. -/
theorem cchain_support_eq_chain {mxm : Mat → Mat → Mat} (h : MxmSpec mxm) (gs : List Graph)
    (F : CMat) (Fb : Mat) (hF : ∀ i s, (i, s) ∈ Fb ↔ 0 < F i s) (i d : Nat) :
    0 < cchain F gs i d ↔ (i, d) ∈ chain mxm Fb (gs.map pairs) := by
  rw [cchain_pos, mem_chain h]
  simp only [hF]

theorem length_insertAsc (a : Nat) : ∀ l : List Nat, (insertAsc a l).length = l.length + 1
  | [] => rfl
  | b :: l => by unfold insertAsc; split <;> simp [length_insertAsc a l]

theorem length_isort : ∀ l : List Nat, (isort l).length = l.length
  | [] => rfl
  | a :: l => by simp [isort, length_insertAsc, length_isort l]

theorem sum_ite_eq_length {α} (l : List α) (P : α → Bool) :
    (l.map fun x => if P x then 1 else 0).sum = (l.filter P).length := by
  induction l with
  | nil => rfl
  | cons x xs ih => cases h : P x <;> simp [h, ih] <;> omega

/-- One counted hop from a seeded row is the number of edges `s → d`. -/
theorem cstep_seed (g : Graph) (i s d : Nat) :
    cstep g (seed i s) i d = (edgesBetween g s d).length := by
  simp only [cstep, seed, edgesBetween, length_isort, List.length_map, true_and]
  rw [show (fun x : Edge => if x.src = s then 1 else 0)
      = (fun x : Edge => if (x.src == s) = true then 1 else 0) by funext x; simp,
    sum_ite_eq_length, List.filter_filter]

theorem pairRows_seed_filter (g : Graph) (s d : Nat) (p : Nat × Nat) :
    (pairRows g false ⟨some s, none, []⟩ true p).filter (fun x => x.2.1 == d) =
      if p = (s, d) then (edgesBetween g s d).map fun e => (s, d, e) else [] := by
  obtain ⟨a, b⟩ := p
  unfold pairRows pes
  have hf : (edgesBetween g a b).filter (fun e => !(([] : List Nat).contains e)) = edgesBetween g a b :=
    List.filter_eq_self.mpr (by simp)
  simp only [hf]
  by_cases ha : a = s
  · subst ha
    by_cases hb : b = d
    · subst hb; simp [pcond, okBound, pf, pt, List.filter_map, Function.comp_def]
      rw [List.filter_eq_self.mpr (fun _ _ => rfl)]
    · simp [pcond, okBound, pf, pt, hb, List.filter_map, Function.comp_def]
  · have : (s == a) = false := by simp [Ne.symm ha]
    simp [pcond, okBound, pf, pt, ha, this]

theorem flatMap_single_length {β} (q : Nat × Nat) (L : List β) :
    ∀ ps : List (Nat × Nat), ps.Nodup →
      (ps.flatMap fun p => if p = q then L else []).length = if q ∈ ps then L.length else 0
  | [], _ => by simp
  | p :: ps, hnd => by
    have ih := flatMap_single_length q L ps (List.nodup_cons.1 hnd).2
    simp only [List.flatMap_cons, List.length_append, ih, List.mem_cons]
    by_cases hp : p = q
    · subst hp; simp [(List.nodup_cons.1 hnd).1]
    · simp [hp, Ne.symm hp]

/-- **Batched multiplicity = per-row emit rows**: the count the batched path emits for row
`i` (seeded at `s`) and destination `d` is the number of rows the per-row emit expansion
of `s` produces for `d` — one per edge, as when the edge is named. -/
theorem cstep_seed_eq_emit_rows (g : Graph) (i s d : Nat) :
    cstep g (seed i s) i d =
      ((expandRow g false true ⟨some s, none, []⟩).filter fun x => x.2.1 == d).length := by
  rw [cstep_seed]
  simp only [expandRow, Bool.false_eq_true, ite_false, List.append_nil, processPairs,
    List.filter_flatMap, pairRows_seed_filter]
  rw [flatMap_single_length _ _ _ (nodup_pairs g)]
  split
  · simp
  · rename_i hn
    rcases hl : edgesBetween g s d with _ | ⟨e, _⟩
    · rfl
    · have : e ∈ edgesBetween g s d := by simp [hl]
      obtain ⟨x, hx, -, rfl, rfl⟩ := mem_edgesBetween.1 this
      exact absurd (mem_pairs.2 ⟨x, hx, rfl, rfl⟩) hn

/-! ### Two fused hops share an edge only at a self-loop

`()-[e1]->()-[e2]->()` fused into one operator (`can_fuse` lets the pair of siblings
through, #3639): the product counts every walk; relationship uniqueness forbids `e1 = e2`,
which needs `e1 = e2 = (v, v)` with `v` both the start and the end. `expand_batch_rows`
subtracts those self-loops when `src == dest` (`shared_self_loops`). -/

def hop2 (g : Graph) (a c : Nat) (distinct : Bool) : Nat :=
  ((g.filter fun e1 => e1.src == a).map fun e1 =>
    (g.filter fun e2 => e1.dst == e2.src && e2.dst == c && (!distinct || e2.id != e1.id)).length).sum

def selfLoops (g : Graph) (a : Nat) : Nat := (edgesBetween g a a).length

theorem filter_ne_id_len (P : Edge → Bool) (E : List Edge) (hnd : (E.map Edge.id).Nodup)
    (e : Edge) (he : e ∈ E) :
    (E.filter fun x => P x && x.id != e.id).length + (if P e then 1 else 0) = (E.filter P).length := by
  induction E with
  | nil => simp at he
  | cons x xs ih =>
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
    obtain ⟨hx, hnd⟩ := hnd
    rcases List.mem_cons.mp he with rfl | hmem
    · have : xs.filter (fun y => P y && y.id != e.id) = xs.filter P := by
        apply List.filter_congr
        intro y hy
        have : y.id ≠ e.id := fun h => hx ⟨y, hy, h⟩
        simp [this]
      simp only [List.filter_cons, bne_self_eq_false, Bool.and_false, this]
      cases P e <;> simp <;> omega
    · have hne : x.id ≠ e.id := fun h => hx ⟨e, hmem, h.symm⟩
      have := ih hnd hmem
      have hb : (x.id != e.id) = true := by simp [hne]
      simp only [List.filter_cons, hb, Bool.and_true]
      cases P x <;> simp <;> omega

theorem sum_map_add' {α} (l : List α) (f g : α → Nat) :
    (l.map fun x => f x + g x).sum = (l.map f).sum + (l.map g).sum := by
  induction l with
  | nil => rfl
  | cons x xs ih => simp only [List.map_cons, List.sum_cons, ih]; omega

/-- **Self-loop correction**: the walk count less the self-loops at `a` when the walk
returns to `a` is the number of edge-distinct two-hop paths. -/
theorem hop2_distinct_add_selfLoops (g : Graph) (hwf : g.WF) (a c : Nat) :
    hop2 g a c true + (if a = c then selfLoops g a else 0) = hop2 g a c false := by
  unfold hop2 selfLoops
  have step : ∀ e1 ∈ g.filter (fun e1 => e1.src == a),
      (g.filter fun e2 => e1.dst == e2.src && e2.dst == c && (!true || e2.id != e1.id)).length
        + (if (e1.dst == e1.src && e1.dst == c) then 1 else 0)
      = (g.filter fun e2 => e1.dst == e2.src && e2.dst == c && (!false || e2.id != e1.id)).length := by
    intro e1 h1
    have := filter_ne_id_len (fun e2 => e1.dst == e2.src && e2.dst == c) g hwf e1
      (List.mem_filter.mp h1).1
    simpa using this
  rw [← List.map_congr_left (fun e1 h1 => step e1 h1), sum_map_add', sum_ite_eq_length,
    List.filter_filter]
  congr 1
  split
  · subst c
    simp only [edgesBetween, length_isort, List.length_map]
    congr 1
    apply List.filter_congr
    intro e _
    cases h1 : (e.src == a) <;> cases h2 : (e.dst == a) <;> simp_all <;> omega
  · rw [eq_comm, List.length_eq_zero_iff, List.filter_eq_nil_iff]
    intro e _
    simp only [Bool.and_eq_true, beq_iff_eq, not_and]
    omega

/-- A self-loop `5 → 5`: `MATCH (a)-->()-->(a)` has no edge-distinct match; the product
counts the walk `r; r` once, which the correction removes. -/
theorem hop2_selfLoop_example :
    hop2 [⟨0, 5, 5⟩] 5 5 true = 0 ∧ hop2 [⟨0, 5, 5⟩] 5 5 false = 1 ∧ selfLoops [⟨0, 5, 5⟩] 5 = 1 := by
  decide

end OpsTraverse
