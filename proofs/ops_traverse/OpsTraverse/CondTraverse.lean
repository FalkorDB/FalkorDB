import OpsTraverse.Basic
/-
# CondTraverse / ExpandInto: single-hop expansion, per-row and batched F·A

| here | there |
| --- | --- |
| `RowIn`                  | the endpoint bindings `from_id` / `to_id` read at `cond_traverse.rs:915-928` and the sibling-edge columns read by `edge_already_used` (`runtime/ops/mod.rs:155`) |
| `processPairs`           | `CondTraverseOp::process_pairs` (`cond_traverse.rs:1235-1316`): bound-endpoint filter; no inline-attr checks since #2390 (the planner lowers them to a Filter) |
| `expandRow`              | `CondTraverseOp::expand_row` (`cond_traverse.rs:900-1118`): forward scan, then the reverse scan with `s != d` (`:1050`, `:1062`; the multi-type twin `expand_union_row` :1121, `UnionGlue`) for bidirectional patterns |
| `dropRev`, `revKept`     | `CondTraverseOp::drop_reverse_duplicates` (`cond_traverse.rs:1138-1165`), called in collapse mode after the reverse scan (`:1093-1099`, `:1284-1288`; #3111) |
| `expandInto`             | `ExpandIntoOp::expand_row` (`expand_into.rs:125-250`): pairs `[(src,dst),(dst,src)]`, the second only when bidirectional and `src != dst`; collapse mode stops at the first pair with an edge (`:201-209`, #3111) |
| `Mat`, `MxmSpec`         | **GraphBLAS boundary**: `Matrix::delta_lmxm` = `GrB_mxm` over the structural `ANY_PAIR` semiring (`cond_traverse.rs:86-112`, `graphblas/matrix.rs:1520`; proved in `UnionGlue.deltaInto_eff`, `GBWDelta.deltaLmxm_correct`) |
| `mxmRef`                 | a reference implementation proving `MxmSpec` is satisfiable |
| `chain`                  | `expand_batch`'s `F = F·A₀·A₁·…` (`cond_traverse.rs:742-747`) |

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
/-- Bound-endpoint check (`cond_traverse.rs:1268-1273`). -/
def pcond (r : RowIn) (isRev : Bool) (p : Nat × Nat) : Bool :=
  okBound r.fromId (pf isRev p) && okBound r.toId (pt isRev p)
/-- Unused edges on the pair (`edge_already_used`, `ops/mod.rs:155`). -/
def pes (g : Graph) (r : RowIn) (p : Nat × Nat) : List Nat :=
  (edgesBetween g p.1 p.2).filter fun e => !(r.used.contains e)

/-- One matrix pair of `process_pairs`: every unused edge when
`emit_relationship` (`:1302-1314`), else only the first (`:1282-1299`). -/
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
  · simp only [hc, if_true, List.mem_map, true_and]
    constructor
    · rintro ⟨e, he, rfl⟩; exact ⟨e, he, rfl⟩
    · rintro ⟨e, he, rfl⟩; exact ⟨e, he, rfl⟩
  · simp [hc]

/-- The `(from, to)` key of a result row. -/
def rowKey (x : Nat × Nat × Nat) : Nat × Nat := (x.1, x.2.1)

/-- `CondTraverseOp::drop_reverse_duplicates` (`cond_traverse.rs:1138-1165`): keep, in scan
order, the reverse results whose `(from, to)` is not among the forward results (the sorted
slice or its sorted copy is only a membership test). -/
def dropRev (fwd rev : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
  rev.filter fun x => !((fwd.map rowKey).contains (rowKey x))

/-- The reverse block's results as `expand_row` keeps them: all of them in emit mode, else
those the forward scan did not already produce (`cond_traverse.rs:1093-1099`, #3111; the
`fwd_end > start` guard only skips a no-op). -/
def revKept (emit : Bool) (fwd rev : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
  if emit then rev else dropRev fwd rev

/-- `expand_row` (untransposed): forward scan, plus the reverse scan over
non-loop pairs for an undirected pattern; in collapse mode a reverse result the
forward scan already gave is dropped (#3111). -/
def expandRow (g : Graph) (bidir emit : Bool) (r : RowIn) : List (Nat × Nat × Nat) :=
  processPairs g (pairs g) false r emit ++
    (if bidir then revKept emit (processPairs g (pairs g) false r emit)
      (processPairs g ((pairs g).filter fun p => p.1 != p.2) true r emit) else [])

/-- `expand_row` before #3111: no reverse de-duplication (historical; `pre3111_*`). -/
def expandRowPre3111 (g : Graph) (bidir emit : Bool) (r : RowIn) : List (Nat × Nat × Nat) :=
  processPairs g (pairs g) false r emit ++
    (if bidir then processPairs g ((pairs g).filter fun p => p.1 != p.2) true r emit else [])

/-- `ExpandIntoOp::expand_row`: both endpoints bound. Collapse mode stops at the first
pair that yields an edge (`expand_into.rs:201-209`, #3111): one representative per
pattern pair, i.e. the first candidate over both directions. -/
def expandInto (g : Graph) (bidir emit : Bool) (src dst : Nat) (used : List Nat) :
    List Nat :=
  let ps := if bidir && src != dst then [(src, dst), (dst, src)] else [(src, dst)]
  let all := ps.flatMap fun p => (edgesBetween g p.1 p.2).filter fun e => !(used.contains e)
  if emit then all else all.take 1

/-- `ExpandIntoOp::expand_row` before #3111: one representative per direction (historical). -/
def expandIntoPre3111 (g : Graph) (bidir emit : Bool) (src dst : Nat) (used : List Nat) :
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
  simp only [processPairs, List.mem_flatMap, mem_pairRows, if_true]
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
  simp only [expandRow, Bool.false_eq_true, if_false, List.append_nil]
  rw [mem_processPairs (fun _ _ => True) (pairs_spec g)]
  simp only [Bool.false_eq_true, if_false]
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
  simp only [expandRow, revKept, if_true, List.mem_append]
  rw [mem_processPairs (fun _ _ => True) (pairs_spec g),
    mem_processPairs (fun s d => s ≠ d) (pairs_filter_spec g)]
  simp only [Bool.false_eq_true, if_false, if_true]
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
  simp only [processPairs, List.mem_flatMap, mem_pairRows, Bool.false_eq_true, if_false,
    if_true]
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
      simp only [expandRow, Bool.false_eq_true, if_false, List.append_nil] at he ⊢
      exact collapse_sub_emit he)
    obtain ⟨x, hx, rfl, h1, h2, h3, h4, h5⟩ := this
    exact ⟨x, hx, h1, h2, h3, h4, h5⟩
  · rintro ⟨x, hx, rfl, rfl, hu, hf, ht⟩
    simp only [expandRow, Bool.false_eq_true, if_false, List.append_nil, processPairs,
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
  · simp only [hc, if_true, Bool.false_eq_true, if_false, List.map_map]
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
  simp only [expandRow, Bool.false_eq_true, if_false, List.append_nil]
  exact (nodup_pairs g).sublist (collapse_keys_sublist g r _)

/-! ### Undirected collapse: one row per pattern pair (W2-traverse-3, fixed by #3111)

`a→b` twice and `b→a` once. C collapses `(a)-[]-(b)` to one row per `(a,b)`.
Before #3111 the forward and the reverse scan each contributed a representative,
so Rust emitted `(1,2)` twice (`pre3111_*`, on the historical models). Main drops
the reverse duplicate (`cond_traverse.rs:1093-1099`) and ExpandInto stops at the
first direction with an edge (`expand_into.rs:206-209`). -/
def gAB : Graph := [⟨0, 1, 2⟩, ⟨1, 2, 1⟩, ⟨2, 1, 2⟩]

/-- Historical (pre-#3111, `cond_traverse.rs` before cc8569b90): `(1,2)` twice. -/
theorem pre3111_bidir_collapse_duplicates_pair :
    ((expandRowPre3111 gAB true false ⟨none, none, []⟩).map fun x => (x.1, x.2.1)).count (1, 2) = 2 := by
  decide

/-- Historical (pre-#3111): ExpandInto gave two rows for one pattern pair. -/
theorem pre3111_expandInto_bidir_collapse_two_rows :
    (expandIntoPre3111 gAB true false 1 2 []).length = 2 := by decide

/-- Main: the same input now yields `(1,2)` once. -/
theorem bidir_collapse_one_row_per_pair :
    ((expandRow gAB true false ⟨none, none, []⟩).map fun x => (x.1, x.2.1)).count (1, 2) = 1 := by
  decide

theorem expandInto_bidir_collapse_one_row :
    (expandInto gAB true false 1 2 []).length = 1 := by decide

/-- Collapse-mode ExpandInto gives at most one row per input row. -/
theorem expandInto_collapse_length_le_one (g : Graph) (b : Bool) (s d : Nat) (u : List Nat) :
    (expandInto g b false s d u).length ≤ 1 := by
  simp only [expandInto, Bool.false_eq_true, if_false]
  exact List.length_take_le _ _

/-- Collapse-mode ExpandInto finds a row iff emit mode finds one (it is emit mode's first). -/
theorem expandInto_collapse_eq_take (g : Graph) (b : Bool) (s d : Nat) (u : List Nat) :
    expandInto g b false s d u = (expandInto g b true s d u).take 1 := by
  simp [expandInto]

/-- The keys of `processPairs` in collapse mode: a sublist of the pairs, oriented. -/
theorem pairRows_keys_gen (g : Graph) (isRev : Bool) (r : RowIn) (p : Nat × Nat) :
    ((pairRows g isRev r false p).map rowKey) = [] ∨
    ((pairRows g isRev r false p).map rowKey) = [(pf isRev p, pt isRev p)] := by
  unfold pairRows
  by_cases hc : pcond r isRev p = true
  · simp only [hc, if_true, Bool.false_eq_true, if_false, List.map_map]
    cases pes g r p <;> simp [rowKey]
  · simp [hc]

theorem collapse_keys_sublist_gen (g : Graph) (isRev : Bool) (r : RowIn) :
    ∀ ps : List (Nat × Nat),
      ((processPairs g ps isRev r false).map rowKey).Sublist
        (ps.map fun p => (pf isRev p, pt isRev p))
  | [] => by simp [processPairs]
  | p :: ps => by
    have ih := collapse_keys_sublist_gen g isRev r ps
    simp only [processPairs, List.flatMap_cons, List.map_append, List.map_cons] at ih ⊢
    rcases pairRows_keys_gen g isRev r p with h | h
    · rw [h]; exact ih.cons _
    · rw [h]; exact ih.cons_cons _

theorem nodup_map_swap : ∀ (l : List (Nat × Nat)), l.Nodup →
    (l.map fun p => (pf true p, pt true p)).Nodup
  | [], _ => by simp
  | p :: l, h => by
    rw [List.nodup_cons] at h
    simp only [List.map_cons, List.nodup_cons]
    refine ⟨?_, nodup_map_swap l h.2⟩
    intro hm
    obtain ⟨q, hq, he⟩ := List.mem_map.1 hm
    simp only [pf, pt, if_true, Prod.mk.injEq] at he
    have : q = p := Prod.ext he.2 he.1
    exact h.1 (this ▸ hq)

theorem nodup_dropRev {fwd rev : List (Nat × Nat × Nat)}
    (hf : (fwd.map rowKey).Nodup) (hr : (rev.map rowKey).Nodup) :
    ((fwd ++ dropRev fwd rev).map rowKey).Nodup := by
  rw [List.map_append, List.nodup_append]
  refine ⟨hf, ?_, ?_⟩
  · exact hr.sublist ((List.filter_sublist).map rowKey)
  · intro a ha b hb hab
    subst hab
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hb
    simp only [dropRev, List.mem_filter, Bool.not_eq_true', List.contains_eq_mem,
      decide_eq_false_iff_not] at hx
    exact hx.2 ha

/-- **Undirected collapse (#3111): one row per pattern pair.** The `(from, to)` keys
`expand_row` emits for an undirected anonymous hop are duplicate-free. -/
theorem collapse_bidir_keys_nodup (g : Graph) (r : RowIn) :
    ((expandRow g true false r).map rowKey).Nodup := by
  simp only [expandRow, revKept, if_true, Bool.false_eq_true, if_false]
  apply nodup_dropRev
  · have hm : ((pairs g).map fun p => (pf false p, pt false p)) = pairs g := by simp [pf, pt]
    exact List.Nodup.sublist (collapse_keys_sublist_gen g false r (pairs g))
      (by rw [hm]; exact nodup_pairs g)
  · refine List.Nodup.sublist (collapse_keys_sublist_gen g true r _) ?_
    exact nodup_map_swap _ ((nodup_pairs g).sublist List.filter_sublist)

theorem mem_dropRev_key {fwd rev : List (Nat × Nat × Nat)} {k : Nat × Nat} :
    k ∈ (fwd ++ dropRev fwd rev).map rowKey ↔ k ∈ fwd.map rowKey ∨ k ∈ rev.map rowKey := by
  simp only [List.map_append, List.mem_append]
  constructor
  · rintro (h | h)
    · exact Or.inl h
    · exact Or.inr ((List.filter_sublist.map rowKey).subset h)
  · rintro (h | h)
    · exact Or.inl h
    · by_cases hk : k ∈ fwd.map rowKey
      · exact Or.inl hk
      · right
        obtain ⟨x, hx, rfl⟩ := List.mem_map.1 h
        exact List.mem_map.2 ⟨x, List.mem_filter.2 ⟨hx, by simpa using hk⟩, rfl⟩

/-- **Undirected collapse loses no pair (#3111)**: a pattern pair has a row iff some
unused edge joins it in either orientation (a self-loop through the forward scan),
exactly the pairs emit mode matches. -/
theorem collapse_bidir_iff {g : Graph} {r : RowIn} {f t : Nat} :
    (f, t) ∈ (expandRow g true false r).map rowKey ↔
      ∃ x ∈ g, x.id ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  have hk : (f, t) ∈ (expandRow g true false r).map rowKey ↔
      (f, t) ∈ (expandRowPre3111 g true false r).map rowKey := by
    simp only [expandRow, revKept, expandRowPre3111, if_true, Bool.false_eq_true, if_false]
    rw [mem_dropRev_key, List.map_append, List.mem_append]
  rw [hk]
  have hsub : ∀ x, x ∈ expandRowPre3111 g true false r → x ∈ expandRowPre3111 g true true r := by
    intro x hx
    simp only [expandRowPre3111, if_true, List.mem_append] at hx ⊢
    rcases hx with hx | hx
    · exact Or.inl (collapse_sub_emit hx)
    · exact Or.inr (collapse_sub_emit hx)
  have hemit : ∀ e, (f, t, e) ∈ expandRowPre3111 g true true r ↔
      ∃ x ∈ g, x.id = e ∧ e ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
    intro e
    have := emit_bidir_iff (g := g) (r := r) (f := f) (t := t) (e := e)
    simpa [expandRow, revKept, expandRowPre3111] using this
  constructor
  · intro h
    obtain ⟨⟨f', t', e⟩, hx, hkey⟩ := List.mem_map.1 h
    simp only [rowKey, Prod.mk.injEq] at hkey
    obtain ⟨rfl, rfl⟩ := hkey
    obtain ⟨x, hx', rfl, hu, h1, h2, h3⟩ := (hemit e).1 (hsub _ hx)
    exact ⟨x, hx', hu, h1, h2, h3⟩
  · rintro ⟨x, hx, hu, hf, ht, hdir⟩
    have hne : ∀ (isRev : Bool) (ps : List (Nat × Nat)) (p : Nat × Nat), p ∈ ps →
        pcond r isRev p = true → x.id ∈ pes g r p →
        (pf isRev p, pt isRev p) ∈ (processPairs g ps isRev r false).map rowKey := by
      intro isRev ps p hp hc hm
      simp only [List.mem_map, processPairs, List.mem_flatMap, mem_pairRows,
        Bool.false_eq_true, if_false]
      revert hm
      cases hes : pes g r p with
      | nil => intro hm; simp at hm
      | cons e0 rest =>
        intro _
        exact ⟨(pf isRev p, pt isRev p, e0), ⟨p, hp, hc, e0, by simp [hes], rfl⟩, rfl⟩
    have hmem : x.id ∈ pes g r (x.src, x.dst) := mem_pes.2 ⟨⟨x, hx, rfl, rfl, rfl⟩, hu⟩
    simp only [expandRowPre3111, if_true, List.map_append, List.mem_append]
    rcases hdir with ⟨rfl, rfl⟩ | ⟨hsd, rfl, rfl⟩
    · left
      exact hne false _ (x.src, x.dst) (mem_pairs.2 ⟨x, hx, rfl, rfl⟩)
        (by simp [pcond, pf, pt, hf, ht]) hmem
    · right
      exact hne true _ (x.src, x.dst)
        (List.mem_filter.2 ⟨mem_pairs.2 ⟨x, hx, rfl, rfl⟩, by simpa using hsd⟩)
        (by simp [pcond, pf, pt, hf, ht]) hmem

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
(`edge_already_used`). `reduce_expand_into` (`planner/optimizer/reduce_expand_into.rs:30-55`)
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
    simp only [if_true] at hdir
    refine ⟨ha, ⟨e1, he1, hid1, hs1.trans hb1.symm, hd1⟩, ⟨e2, he2, hid2, hdir.1, hdir.2.trans hb2.symm⟩, ?_⟩
    simpa using hu2
  · rintro ⟨ha, ⟨e1, he1, rfl, rfl, rfl⟩, ⟨e2, he2, rfl, rfl, hd2⟩, hne⟩
    refine ⟨e1.src, ha, (e1.src, e1.dst, e1.id), ?_, (e1.dst, e2.src, e2.id), ?_, rfl, rfl, rfl, rfl, rfl⟩
    · exact emit_directed_iff.2 ⟨e1, he1, rfl, rfl, rfl, by simp, by simp [okBound], by simp [okBound]⟩
    · refine (mem_processPairs (fun _ _ => True) (pairs_spec g)).2
        ⟨e2, he2, rfl, by simpa using hne, by simp [okBound, hd2], by simp [okBound], trivial, ?_⟩
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
(`cond_traverse.rs:86-112`). C(i,j) is present iff ∃k. F(i,k) ∧ A(k,j), and a
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

/-- `F ← F·A₀`, then every fused chain hop (`cond_traverse.rs:742-747`). -/
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
    exact ⟨s, hF, collapse_directed_iff.2 ⟨x, hx, h1, h2, by simp, by simp [okBound, h1], by simp [okBound]⟩⟩
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
