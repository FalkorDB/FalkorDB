import OpsTraverse.Drive
/-
Index and vector scans (graph/src/runtime/ops/node_by_index_scan.rs, node_by_vector_scan.rs).

| here | there |
| --- | --- |
| `IQ`, `evalIQ`        | `IndexQuery`, `NodeByIndexScanOp::evaluate_index_query` (node_by_index_scan.rs:98) |
| `indexable`, `canUse` | `can_utilize_index` (:250) and its inner `is_indexable` (:253) |
| `extraLabels`, `IxSt.new` | `NodeByIndexScanOp::new` (:56) |
| `scanRow`             | the per-row closure of `NodeByIndexScanOp::next` (:293) |
| `vecArgs`             | `eval_vector_args` (node_by_vector_scan.rs:120) |
| `VecSt.new`           | `NodeByVectorScanOp::new` (:47) |
| `vecRow`              | the per-row closure of `NodeByVectorScanOp::next` (:83) |

FFI boundary (RediSearch / the vector index): `get_indexed_nodes(index, q)` and
`vector_query_nodes` are parameters; their contract is the hypothesis `IndexSound`
(the index answers exactly the label's nodes satisfying the evaluated query).
-/
namespace OpsTraverse.IndexScan

/-- Value kinds that matter to the index. -/
inductive V where
  | int (i : Int) | float | str (s : String) | bool (b : Bool) | point | null | list | other
  deriving DecidableEq

inductive IQ (E : Type) where
  | eq (key : String) (v : E)
  | range (key : String) (lo hi : Option E) (incLo incHi : Bool)
  | pt (key : String) (p r : E)
  | and (qs : List (IQ E))
  | or (qs : List (IQ E))
  | arr (key : String) (v : E)
  | inList (key : String) (l : E)
  /-- `IndexQuery::InList { list: Value::List(items) }`, the evaluated form (#2383): only the
  scalar items of the list are kept, in order, instead of being expanded to `Or(Equal, ..)`. Only
  `evalIQ` produces it. -/
  | inListV (key : String) (items : List V)

def isScalar : V → Bool
  | .int _ | .float | .str _ | .bool _ => true
  | _ => false

/-- node_by_index_scan.rs:98-246. `ev` = `ExprEval::eval` on the row; `asList` = the
`Value::List` match of the `InList` arm. -/
def evalIQ {E : Type} (ev : E → Except String V) (asList : E → Except String (Option (List V))) :
    IQ E → Except String (IQ V)
  | .eq k v => do pure (.eq k (← ev v))
  | .range k lo hi a b => do
    let lo' ← match lo with | some e => (some <$> ev e) | none => pure none
    let hi' ← match hi with | some e => (some <$> ev e) | none => pure none
    pure (.range k lo' hi' a b)
  | .pt k p r => do let p' ← ev p; let r' ← ev r; pure (.pt k p' r')
  | .and qs => do pure (.and (← evalL ev asList qs))
  | .or qs => do pure (.or (← evalL ev asList qs))
  | .arr k v => do pure (.arr k (← ev v))
  | .inList k l => do
    match ← asList l with
    | some items => pure (.inListV k (items.filter isScalar))
    | none => throw "IN operator requires a list"
  | .inListV k items => pure (.inListV k items)
where
  evalL ev asList : List (IQ E) → Except String (List (IQ V))
  | [] => pure []
  | q :: qs => do let a ← evalIQ ev asList q; let b ← evalL ev asList qs; pure (a :: b)

/-- `Index::int_loses_f64_precision` (index/mod.rs:1359): `i.unsigned_abs() & 0x7FF0_0000_0000_0000 != 0`
(conservative: any |i| ≥ 2^52 except 2^63 falls back). -/
def losesPrecision (i : Int) : Bool := (i.natAbs &&& 0x7FF0000000000000) != 0

/-- node_by_index_scan.rs:253-263 -/
def indexable : V → Bool
  | .int i => !losesPrecision i
  | .float | .str _ | .bool _ | .point | .null => true
  | _ => false

/-- node_by_index_scan.rs:250-283 -/
def canUse : IQ V → Bool
  | .eq _ v => indexable v
  | .range _ lo hi _ _ => (lo.all indexable) && (hi.all indexable)
  | .and qs => !qs.isEmpty && canUseL qs
  | .or qs => !qs.isEmpty && canUseL qs
  | .arr _ v => isScalar v
  -- #2383: `matches!(list, Value::List(items) if !items.is_empty() && items.iter().all(is_indexable))`
  | .inListV _ items => !items.isEmpty && items.all indexable
  -- an `InList` whose payload is not a list never reaches the index
  | .inList _ _ => false
  | _ => true
where
  canUseL : List (IQ V) → Bool
  | [] => true
  | q :: qs => canUse q && canUseL qs

theorem canUseL_iff (qs : List (IQ V)) : canUse.canUseL qs = qs.all canUse := by
  induction qs with
  | nil => rfl
  | cons q qs ih => simp [canUse.canUseL, ih]

/-- The index is consulted only for values it stores losslessly; a 2^53+1 integer, a list,
a map, a node… falls back to a label scan (the planner keeps the `Filter` above). -/
theorem canUse_spec (k : String) (v : V) (qs : List (IQ V)) :
    canUse (.eq k v) = indexable v ∧
    canUse (.eq k (.int (2 ^ 53 + 1))) = false ∧ canUse (.eq k .list) = false ∧
    canUse (.or qs) = (!qs.isEmpty && qs.all canUse) ∧ canUse (.or []) = false := by
  refine ⟨rfl, by simp only [canUse, indexable]; decide, rfl, by simp [canUse, canUseL_iff], rfl⟩

/-- `IN list` keeps only the scalar items, in order, as one `InList` (since #2383; before it was an
`Or` of equalities over the same members, which `canUse_inListV_eq_or` shows decides alike). Known W2-index-3, still open on main:
a non-scalar item — date, array — is dropped before the index is asked. -/
theorem evalIQ_inList {E : Type} (ev : E → Except String V) (asList : E → Except String (Option (List V)))
    (k : String) (l : E) (items : List V) (h : asList l = .ok (some items)) :
    evalIQ ev asList (.inList k l) = .ok (.inListV k (items.filter isScalar)) := by
  simp [evalIQ, h, bind, Except.bind, pure, Except.pure]

/-- **#2383 keeps the fallback rule.** A list reaches the index iff its members, as an `Or` of
`Equal`s, would have (the Rust test `in_list_reaches_the_index_on_the_same_terms_as_an_or_of_its_members`):
non-empty and every member indexable. Hence `IN [NULL]` (filtered to empty) falls back to a label scan. -/
theorem canUse_inListV_eq_or (k : String) (items : List V) :
    canUse (.inListV k items) = canUse (.or (items.map (.eq k))) := by
  simp only [canUse, canUse.canUseL, canUseL_iff, List.isEmpty_map, List.all_map]
  rfl

theorem canUse_inListV_empty (k : String) : canUse (.inListV k []) = false := rfl

theorem canUse_inList_nonlist (k : String) (v : V) : canUse (.inList k v) = false := rfl

theorem evalIQ_inList_err {E : Type} (ev : E → Except String V) (asList : E → Except String (Option (List V)))
    (k : String) (l : E) (h : asList l = .ok none) :
    evalIQ ev asList (.inList k l) = .error "IN operator requires a list" := by
  simp [evalIQ, h, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]

/-- Every leaf expression is evaluated in place; the query shape is kept. -/
theorem evalIQ_eq {E : Type} (ev : E → Except String V) (asList : E → Except String (Option (List V)))
    (k : String) (e : E) (v : V) (h : ev e = .ok v) :
    evalIQ ev asList (.eq k e) = .ok (.eq k v) ∧ evalIQ ev asList (.arr k e) = .ok (.arr k v) := by
  simp [evalIQ, h, bind, Except.bind, pure, Except.pure]

/-! ## `new` and the per-row node set -/

/-- node_by_index_scan.rs:65-69: labels after the first are post-filtered. -/
def extraLabels {L : Type} (labels : List L) : Option (List L) :=
  if labels.length > 1 then some labels.tail else none

structure IxSt (L : Type) where
  alias : Nat
  extra : Option (List L)

def IxSt.new {L : Type} (alias : Nat) (labels : List L) : IxSt L := ⟨alias, extraLabels labels⟩

theorem ixNew_spec {L : Type} (alias : Nat) (l : L) (ls : List L) :
    (IxSt.new alias [l]).extra = none ∧ (IxSt.new alias (l :: ls)).extra = (if ls = [] then none else some ls) := by
  refine ⟨rfl, ?_⟩
  cases ls <;> simp [IxSt.new, extraLabels]

/-- node_by_index_scan.rs:297-323: index answer (or label scan), then the extra-label filter. -/
def scanRow {N L : Type} (idx : IQ V → List N) (byLabel : List N) (hasLabel : N → L → Bool)
    (extra : Option (List L)) (q : IQ V) : List N :=
  let base := if canUse q then idx q else byLabel
  match extra with
  | some ls => base.filter fun n => ls.all (hasLabel n)
  | none => base

/-- Index contract (FFI): the index answers the first label's nodes satisfying `q`. -/
def IndexSound {N : Type} (idx : IQ V → List N) (byLabel : List N) (sat : IQ V → N → Bool) : Prop :=
  ∀ q n, canUse q → (n ∈ idx q ↔ n ∈ byLabel ∧ sat q n = true)

/-- **Index scan is a sound pre-filter**: every node of the first label that satisfies the
query and carries the extra labels is emitted, and only nodes of the first label are
(the `Filter` the planner keeps above then makes the result exact). -/
theorem scanRow_spec {N L : Type} (idx : IQ V → List N) (byLabel : List N) (hasLabel : N → L → Bool)
    (sat : IQ V → N → Bool) (hs : IndexSound idx byLabel sat) (extra : Option (List L)) (q : IQ V) (n : N) :
    (n ∈ byLabel → sat q n = true → (extra.all fun ls => ls.all (hasLabel n)) →
      n ∈ scanRow idx byLabel hasLabel extra q) ∧
    (n ∈ scanRow idx byLabel hasLabel extra q → n ∈ byLabel) := by
  have base_in : n ∈ byLabel → sat q n = true → n ∈ (if canUse q then idx q else byLabel) := by
    intro h1 h2; split
    · rename_i hc; exact (hs q n hc).mpr ⟨h1, h2⟩
    · exact h1
  have base_sub : n ∈ (if canUse q then idx q else byLabel) → n ∈ byLabel := by
    split
    · rename_i hc; exact fun h => ((hs q n hc).mp h).1
    · exact id
  unfold scanRow
  constructor
  · intro h1 h2 h3
    cases extra with
    | none => exact base_in h1 h2
    | some ls => simp only [Option.all_some] at h3; exact List.mem_filter.mpr ⟨base_in h1 h2, h3⟩
  · intro h
    cases extra with
    | none => exact base_sub h
    | some ls => exact base_sub (List.mem_filter.mp h).1

/-! ## Vector scan -/

inductive VA where
  | str (s : String) | int (i : Int) | vec (n : Nat) | other

/-- node_by_vector_scan.rs:120-155: label, attribute (strings), `k > 0`, a `vecf32`;
anything else is "Invalid arguments for procedure '…'"; evaluation errors pass through. -/
def vecArgs (proc : String) (l a k v : Except String VA) : Except String (String × String × Nat × Nat) := do
  let invalid := s!"Invalid arguments for procedure '{proc}'"
  let ls ← match ← l with | .str s => pure s | _ => throw invalid
  let as ← match ← a with | .str s => pure s | _ => throw invalid
  let kv ← match ← k with | .int n => if n > 0 then pure n.toNat else throw invalid | _ => throw invalid
  let vv ← match ← v with | .vec n => pure n | _ => throw invalid
  pure (ls, as, kv, vv)

theorem vecArgs_ok (proc ls as : String) (n : Int) (vv : Nat) (hn : n > 0) :
    vecArgs proc (.ok (.str ls)) (.ok (.str as)) (.ok (.int n)) (.ok (.vec vv)) = .ok (ls, as, n.toNat, vv) := by
  simp [vecArgs, hn, bind, Except.bind, pure, Except.pure]

theorem vecArgs_bad_k (proc ls as : String) (n : Int) (v : Except String VA) (hn : ¬ n > 0) :
    vecArgs proc (.ok (.str ls)) (.ok (.str as)) (.ok (.int n)) v =
      .error s!"Invalid arguments for procedure '{proc}'" := by
  simp [vecArgs, hn, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure]

/-- node_by_vector_scan.rs:58-65: bind the node id, plus the score column when named. -/
structure VecSt where
  nodeCol : Nat
  scoreCol : Option Nat
  cap : Option Nat

def VecSt.new (node : Nat) (score : Option Nat) : VecSt := ⟨node, score, none⟩

theorem vecNew_spec (n : Nat) (s : Option Nat) :
    (VecSt.new n s).nodeCol = n ∧ (VecSt.new n s).scoreCol = s ∧ (VecSt.new n s).cap = none :=
  ⟨rfl, rfl, rfl⟩

/-- One row of `NodeByVectorScanOp::next`: arguments, then the index's `(node, score)`
answer (FFI) emitted as is. -/
def vecRow {N : Type} (q : String → String → Nat → Nat → Except String (List (N × Float)))
    (args : Except String (String × String × Nat × Nat)) : Except String (List (N × Float)) := do
  let (l, a, k, v) ← args
  q l a v k

theorem vecRow_spec {N : Type} (q : String → String → Nat → Nat → Except String (List (N × Float)))
    (l a : String) (k v : Nat) :
    vecRow q (.ok (l, a, k, v)) = q l a v k ∧ ∀ e, vecRow q (.error e) = .error e :=
  ⟨rfl, fun _ => rfl⟩

/-- Both scans drive the shared loop: output rows = per-child-batch expansions, in order
(`Drive.drive_flatten`). -/
theorem scans_stream {C R : Type} (pack : C → List (List R)) (cs : List C) :
    (Drive.drive pack [] cs).flatten = (cs.flatMap pack).flatten := by
  rw [Drive.drive_flatten]; simp

/-! ## Edge index scan (edge_by_index_scan.rs) -/

/-- edge_by_index_scan.rs:595-640: as `canUse`, except `ArrayContains` also refuses integers that
lose precision as `f64`. -/
def canUseE : IQ V → Bool
  | .arr _ v => match v with
    | .int i => !losesPrecision i
    | .float | .str _ | .bool _ => true
    | _ => false
  | .and qs => !qs.isEmpty && canUseEL qs
  | .or qs => !qs.isEmpty && canUseEL qs
  | .eq k v => canUse (.eq k v)
  | .range k lo hi a b => canUse (.range k lo hi a b)
  | .pt _ _ _ => true
  | .inListV _ items => !items.isEmpty && items.all indexable   -- #2383, as the node op
  | .inList _ _ => false
where
  canUseEL : List (IQ V) → Bool
  | [] => true
  | q :: qs => canUseE q && canUseEL qs

/-- The edge op's `IN` rule (#2383) is the node op's, and decides as the `Or` of `Equal`s it replaced. -/
theorem canUseE_inListV_eq_or (k : String) (items : List V) :
    canUseE (.inListV k items) = canUseE (.or (items.map (.eq k))) := by
  have h : ∀ qs : List V, canUseE.canUseEL (qs.map (.eq k)) = qs.all (fun v => indexable v) := by
    intro qs; induction qs with
    | nil => rfl
    | cons v vs ih => simp [canUseE.canUseEL, canUseE, canUse, ih]
  simp only [canUseE, h, List.isEmpty_map]

theorem canUseE_spec (k : String) (v : V) (lo hi : Option V) (a b : Bool) :
    canUseE (.eq k v) = canUse (.eq k v) ∧ canUseE (.range k lo hi a b) = canUse (.range k lo hi a b) ∧
    canUseE (.arr k (.int (2 ^ 60))) = false ∧ canUse (.arr k (.int (2 ^ 60))) = true := by
  refine ⟨by simp [canUseE], by simp [canUseE], by simp only [canUseE]; decide, rfl⟩

/-- edge_by_index_scan.rs:399-444: `to` is not bound separately for a self-loop alias. -/
structure EdgeCfg where
  fromA : Nat
  toA : Option Nat
  edge : Nat
  transposed : Bool
  cap : Option Nat

def edgeIxNew (fromA toA edge : Nat) (transposed : Bool) : EdgeCfg :=
  ⟨fromA, if toA ≠ fromA then some toA else none, edge, transposed, none⟩

theorem edgeIxNew_spec (f t e : Nat) (tr : Bool) :
    ((edgeIxNew f t e tr).toA = none ↔ t = f) ∧ (edgeIxNew f t e tr).cap = none := by
  refine ⟨by by_cases h : t = f <;> simp [edgeIxNew, h], rfl⟩

/-- edge_by_index_scan.rs:713-723 (#3081, `ff3d24ba7`): an undirected pattern
(`rp.bidirectional`) sees each edge the index yields once, `src -> dst`, in both
orientations — a self-loop once — as `CondTraverse` does. -/
def orient (bidir : Bool) (base : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
  if bidir then base.flatMap fun (s, d, e) => (s, d, e) :: (if s ≠ d then [(d, s, e)] else [])
  else base

/-- edge_by_index_scan.rs:65-70 `pattern_endpoints`: the pattern's `(from, to)` of a hit
`(src, dst, edge)`; a transposed scan matches `from` on the edge's destination. -/
def patternEnds (transposed : Bool) (h : Nat × Nat × Nat) : Nat × Nat :=
  if transposed then (h.2.1, h.1) else (h.1, h.2.1)

/-- edge_by_index_scan.rs:76-82 `EndpointFilter`: the endpoints the row binds, whether the
pattern is a self-loop (one alias on both ends), and `transposed`. -/
structure EndpointFilter where
  bFrom : Option Nat
  bTo : Option Nat
  sameAlias : Bool
  transposed : Bool

/-- edge_by_index_scan.rs:87-89 `EndpointFilter::keeps_all`. -/
def EndpointFilter.keepsAll (f : EndpointFilter) : Bool :=
  f.bFrom.isNone && f.bTo.isNone && !f.sameAlias

/-- edge_by_index_scan.rs:91-99 `EndpointFilter::keeps`: the one per-hit test of `next`, used by
the cached and the streaming path alike. -/
def EndpointFilter.keeps (f : EndpointFilter) (h : Nat × Nat × Nat) : Bool :=
  let e := patternEnds f.transposed h
  (f.bFrom.all (· == e.1)) && (f.bTo.all (· == e.2)) && (!f.sameAlias || e.1 == e.2)

/-- `keeps_all` is sound: skipping the filter when it holds drops nothing. -/
theorem EndpointFilter.keepsAll_keeps (f : EndpointFilter) (h : f.keepsAll = true)
    (x : Nat × Nat × Nat) : f.keeps x = true := by
  obtain ⟨bf, bt, sa, tr⟩ := f
  simp only [keepsAll, Bool.and_eq_true, Option.isNone_iff_eq_none, Bool.not_eq_true'] at h
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
  simp [keeps]

/-- edge_by_index_scan.rs:705-732 (`indexed_edges` :199 / `scan_type` :165): the index answer (or
the cached full edge list), oriented (`orient`), then `EndpointFilter` unless it keeps all. -/
def edgeRow (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat)) (q : IQ V)
    (bFrom bTo : Option Nat) (sameAlias transposed bidir : Bool) : List (Nat × Nat × Nat) :=
  let base := orient bidir (if canUseE q then idx q else all)
  let f : EndpointFilter := ⟨bFrom, bTo, sameAlias, transposed⟩
  if f.keepsAll then base else base.filter f.keeps

/-- Every emitted edge agrees with the bound endpoints (in pattern orientation) and is a loop
when both endpoints share an alias. -/
theorem edgeRow_sound (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat)) (q : IQ V)
    (bf bt : Option Nat) (sa tr bd : Bool) (s d e : Nat)
    (h : (s, d, e) ∈ edgeRow idx all q bf bt sa tr bd) :
    (∀ x, bf = some x → x = (if tr then d else s)) ∧ (∀ x, bt = some x → x = (if tr then s else d)) ∧
    (sa = true → (if tr then d else s) = (if tr then s else d)) := by
  unfold edgeRow at h
  generalize orient bd (if canUseE q = true then idx q else all) = base at h
  dsimp only at h
  split at h
  · rename_i hk
    simp only [EndpointFilter.keepsAll, Bool.and_eq_true, Option.isNone_iff_eq_none,
      Bool.not_eq_true'] at hk
    obtain ⟨⟨h1, h2⟩, h3⟩ := hk
    refine ⟨fun x hx => by simp [h1] at hx, fun x hx => by simp [h2] at hx, fun hs => by simp [h3] at hs⟩
  · rw [List.mem_filter] at h
    obtain ⟨_, hp⟩ := h
    cases tr <;> simp only [EndpointFilter.keeps, patternEnds, Bool.and_eq_true, Bool.or_eq_true,
        Bool.not_eq_true', beq_iff_eq, Bool.false_eq_true, ite_false, ite_true] at hp ⊢ <;>
      obtain ⟨⟨h1, h2⟩, h3⟩ := hp <;>
      refine ⟨fun x hx => ?_, fun x hx => ?_, fun hs => ?_⟩ <;> simp_all

/-- **Undirected scans see both orientations** (#3081): an oriented hit is an index hit read
either way round. -/
theorem orient_mem (base : List (Nat × Nat × Nat)) (x y e : Nat) :
    (x, y, e) ∈ orient true base ↔ (x, y, e) ∈ base ∨ (y, x, e) ∈ base := by
  simp only [orient, ite_true, List.mem_flatMap]
  constructor
  · rintro ⟨⟨s, d, e'⟩, hm, hx⟩
    simp only [List.mem_cons] at hx
    rcases hx with h | h
    · simp at h; obtain ⟨rfl, rfl, rfl⟩ := h; exact Or.inl hm
    · split at h
      · simp at h; obtain ⟨rfl, rfl, rfl⟩ := h; exact Or.inr hm
      · simp at h
  · rintro (h | h)
    · exact ⟨_, h, by simp⟩
    · refine ⟨_, h, ?_⟩
      by_cases hxy : y = x
      · subst hxy; simp
      · simp [hxy]

/-- ... and each index hit yields two rows, a self-loop one: the row count is the hit count
plus the non-loop hit count. A directed scan is unchanged. -/
theorem orient_length (base : List (Nat × Nat × Nat)) :
    (orient true base).length = base.length + (base.filter fun (s, d, _) => s ≠ d).length ∧
    orient false base = base := by
  refine ⟨?_, rfl⟩
  induction base with
  | nil => rfl
  | cons hd tl ih =>
    obtain ⟨s, d, e⟩ := hd
    simp only [orient, ite_true, List.flatMap_cons, List.length_append, List.length_cons] at ih ⊢
    rw [ih]
    by_cases h : s = d
    · simp [h]; omega
    · simp [h, List.filter_cons]; omega

/-! ### The hit cache (#3376): edge_by_index_scan.rs:249-362, 646-744 -/

/-- edge_by_index_scan.rs:257-278 `collect_hits`, from an accumulator `acc`: append each hit
oriented as `orient` does, refusing (`none`) before an append that would pass `max`. -/
def collectHits (bidir : Bool) (max : Nat) :
    List (Nat × Nat × Nat) → List (Nat × Nat × Nat) → Option (List (Nat × Nat × Nat))
  | acc, [] => some acc
  | acc, (s, d, e) :: t =>
    if acc.length = max then none
    else if bidir && s ≠ d then
      if acc.length + 1 = max then none else collectHits bidir max (acc ++ [(s, d, e), (d, s, e)]) t
    else collectHits bidir max (acc ++ [(s, d, e)]) t

theorem orient_cons (b : Bool) (s d e : Nat) (t : List (Nat × Nat × Nat)) :
    orient b ((s, d, e) :: t) = (s, d, e) :: ((if b && s ≠ d then [(d, s, e)] else []) ++ orient b t) := by
  cases b <;> by_cases h : s = d <;> simp [orient, h]

theorem collectHits_acc (bidir : Bool) (max : Nat) (hs acc : List (Nat × Nat × Nat))
    (hacc : acc.length ≤ max) :
    collectHits bidir max acc hs =
      if acc.length + (orient bidir hs).length ≤ max then some (acc ++ orient bidir hs) else none := by
  induction hs generalizing acc with
  | nil => simp [collectHits, orient, hacc]
  | cons hd t ih =>
    obtain ⟨s, d, e⟩ := hd
    rw [orient_cons]
    have hcol : collectHits bidir max acc ((s, d, e) :: t) =
        if acc.length = max then none
        else if (bidir && s ≠ d) = true then
          (if acc.length + 1 = max then none
           else collectHits bidir max (acc ++ [(s, d, e), (d, s, e)]) t)
        else collectHits bidir max (acc ++ [(s, d, e)]) t := rfl
    rw [hcol]
    by_cases hb : (bidir && s ≠ d) = true
    · rw [if_pos hb, if_pos hb]
      have hl : (acc ++ (s, d, e) :: ([(d, s, e)] ++ orient bidir t)).length =
          acc.length + 2 + (orient bidir t).length := by simp <;> omega
      have ha : acc ++ (s, d, e) :: ([(d, s, e)] ++ orient bidir t) =
          acc ++ [(s, d, e), (d, s, e)] ++ orient bidir t := by simp
      rw [List.length_append] at hl
      rw [hl, ha]
      by_cases h0 : acc.length = max
      · rw [if_pos h0, if_neg (by omega)]
      · rw [if_neg h0]
        by_cases h1 : acc.length + 1 = max
        · rw [if_pos h1, if_neg (by omega)]
        · rw [if_neg h1, ih _ (by simp; omega)]
          simp only [List.length_append, List.length_cons, List.length_nil]
          all_goals
            by_cases hm : acc.length + 2 + (orient bidir t).length ≤ max
            · rw [if_pos (by omega), if_pos hm]
            · rw [if_neg (by omega), if_neg hm]
    · rw [if_neg hb, if_neg hb]
      have hl : (acc ++ (s, d, e) :: ([] ++ orient bidir t)).length =
          acc.length + 1 + (orient bidir t).length := by simp <;> omega
      have ha : acc ++ (s, d, e) :: ([] ++ orient bidir t) = acc ++ [(s, d, e)] ++ orient bidir t := by
        simp
      rw [List.length_append] at hl
      rw [hl, ha]
      by_cases h0 : acc.length = max
      · rw [if_pos h0, if_neg (by omega)]
      · rw [if_neg h0, ih _ (by simp; omega)]
        simp only [List.length_append, List.length_cons, List.length_nil]
        all_goals
          by_cases hm : acc.length + 1 + (orient bidir t).length ≤ max
          · rw [if_pos (by omega), if_pos hm]
          · rw [if_neg (by omega), if_neg hm]

/-- `collect_hits` keeps exactly the oriented hits when they fit in `max`, else refuses: the
cache never holds more than `MAX_CACHED_HITS`, and what it holds is `orient`'s list. -/
theorem collectHits_spec (bidir : Bool) (max : Nat) (hs : List (Nat × Nat × Nat)) :
    collectHits bidir max [] hs =
      if (orient bidir hs).length ≤ max then some (orient bidir hs) else none := by
  rw [collectHits_acc bidir max hs [] (Nat.zero_le _)]; simp

/-- edge_by_index_scan.rs:280-297 `HitCacheState`. `built` holds the oriented hits; the
by-endpoint maps are `groupByEndpoint` of them, built on demand. -/
inductive CacheState where
  | off
  | cold
  | warm
  | built (all : List (Nat × Nat × Nat))
  | overBudget

/-- edge_by_index_scan.rs:433-441 (`EdgeByIndexScanOp::new`): the cache is off for a write
query or a row-dependent index query, decided once instead of on every row. -/
def initState (write rowInv : Bool) : CacheState :=
  if write || !rowInv then .off else .cold

/-- edge_by_index_scan.rs:54-60 `shared_hits`: index `0..len` into the shared slice, by value. -/
def sharedHits (hits : List (Nat × Nat × Nat)) : List (Nat × Nat × Nat) :=
  (List.range hits.length).map fun i => hits.getD i (0, 0, 0)

/-- `shared_hits` walks the slice exactly, in order. -/
theorem sharedHits_eq (hits : List (Nat × Nat × Nat)) : sharedHits hits = hits := by
  apply List.ext_getElem
  · simp [sharedHits]
  · intro i _ h2
    simp [sharedHits, List.getElem?_eq_getElem h2]

/-- The key `group_by_endpoint` files a hit under: its pattern `from` (`byFrom`) or `to`. -/
def endpointKey (transposed byFrom : Bool) (h : Nat × Nat × Nat) : Nat :=
  if byFrom then (patternEnds transposed h).1 else (patternEnds transposed h).2

/-- edge_by_index_scan.rs:115-134 `group_by_endpoint`, with the `HashMap` as a function: each hit
is pushed onto its key's group in list order, and a node with no group reads `[]` (the
`unwrap_or_default` of `lookup`). -/
def groupByEndpoint (hits : List (Nat × Nat × Nat)) (transposed byFrom : Bool) :
    Nat → List (Nat × Nat × Nat) :=
  hits.foldl (fun g h n => if n = endpointKey transposed byFrom h then g n ++ [h] else g n)
    (fun _ => [])

theorem groupByEndpoint_acc (hits : List (Nat × Nat × Nat)) (tr bf : Bool)
    (g : Nat → List (Nat × Nat × Nat)) (n : Nat) :
    hits.foldl (fun g h n => if n = endpointKey tr bf h then g n ++ [h] else g n) g n =
      g n ++ hits.filter (fun h => endpointKey tr bf h == n) := by
  induction hits generalizing g with
  | nil => simp
  | cons h t ih =>
    simp only [List.foldl_cons, ih, List.filter_cons]
    by_cases hn : n = endpointKey tr bf h
    · simp [hn]
    · have : (endpointKey tr bf h == n) = false := by
        simp only [beq_eq_false_iff_ne, ne_eq]; exact fun e => hn e.symm
      simp [hn, this]

/-- A group is the hits whose key is that node, in list order. -/
theorem groupByEndpoint_spec (hits : List (Nat × Nat × Nat)) (tr bf : Bool) (n : Nat) :
    groupByEndpoint hits tr bf n = hits.filter (fun h => endpointKey tr bf h == n) := by
  simp [groupByEndpoint, groupByEndpoint_acc]

/-- edge_by_index_scan.rs:323-327, the first `match` of `HitCache::lookup`: the bound node and
whether it is the pattern's `from`; `none` when the row binds neither endpoint. -/
def boundKey : Option Nat → Option Nat → Option (Nat × Bool)
  | some id, _ => some (id, true)
  | none, some id => some (id, false)
  | none, none => none

/-- edge_by_index_scan.rs:332-345, the `Warm` step of `HitCache::lookup`: keep the hits
(`collect_hits`), or go over budget; any other state is left alone. -/
def build (idx : IQ V → List (Nat × Nat × Nat)) (bidir : Bool) (max : Nat) (q : IQ V) :
    CacheState → CacheState
  | .warm => match collectHits bidir max [] (idx q) with
    | some all => .built all
    | none => .overBudget
  | s => s

/-- edge_by_index_scan.rs:314-361 `HitCache::lookup`: the next state, and the bucket of the bound
node (`none` = stream). An unbound row leaves the state alone; `cold` warms and streams; `warm`
builds; then a `built` cache answers and `off` / `overBudget` stream. -/
def lookup (idx : IQ V → List (Nat × Nat × Nat)) (bidir : Bool) (max : Nat) (q : IQ V)
    (bFrom bTo : Option Nat) (transposed : Bool) (s : CacheState) :
    CacheState × Option (List (Nat × Nat × Nat)) :=
  match boundKey bFrom bTo with
  | none => (s, none)
  | some (node, byFrom) =>
    match s with
    | .cold => (.warm, none)
    | s =>
      match build idx bidir max q s with
      | .built all => (.built all, some (sharedHits (groupByEndpoint all transposed byFrom node)))
      | s' => (s', none)

/-- edge_by_index_scan.rs:658-732, one input row: a usable query asks the hit cache, and a
bucket it returns is filtered by the same `EndpointFilter` as the streaming path (`edgeRow`). An
empty bucket skips the row (:696-700), which emits the same nothing as filtering it. -/
def edgeNextRow (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat)) (q : IQ V)
    (bFrom bTo : Option Nat) (sameAlias transposed bidir : Bool) (max : Nat) (s : CacheState) :
    CacheState × List (Nat × Nat × Nat) :=
  if canUseE q then
    match lookup idx bidir max q bFrom bTo transposed s with
    | (s', some b) => (s', b.filter (EndpointFilter.keeps ⟨bFrom, bTo, sameAlias, transposed⟩))
    | (s', none) => (s', edgeRow idx all q bFrom bTo sameAlias transposed bidir)
  else (s, edgeRow idx all q bFrom bTo sameAlias transposed bidir)

/-- What a built cache holds: the hits of `q`, oriented as the pattern matches them. -/
def CacheInv (idx : IQ V → List (Nat × Nat × Nat)) (bidir : Bool) (q : IQ V) : CacheState → Prop
  | .built all => all = orient bidir (idx q)
  | _ => True

/-- `build` keeps exactly `q`'s oriented hits, by `collectHits_spec`. -/
theorem build_inv (idx : IQ V → List (Nat × Nat × Nat)) (bd : Bool) (max : Nat) (q : IQ V)
    (s : CacheState) (h : CacheInv idx bd q s) : CacheInv idx bd q (build idx bd max q s) := by
  cases s with
  | warm =>
    simp only [build, collectHits_spec]
    split <;> rename_i h' <;> split at h' <;> simp_all [CacheInv]
  | _ => exact h

theorem lookup_bound (idx : IQ V → List (Nat × Nat × Nat)) (bd : Bool) (max : Nat) (q : IQ V)
    (bf bt : Option Nat) (tr : Bool) (node : Nat) (byFrom : Bool)
    (hk : boundKey bf bt = some (node, byFrom)) (s : CacheState) (hs : s ≠ .cold) :
    lookup idx bd max q bf bt tr s =
      match build idx bd max q s with
      | .built all => (.built all, some (sharedHits (groupByEndpoint all tr byFrom node)))
      | s' => (s', none) := by
  unfold lookup
  rw [hk]
  cases s <;> first | exact absurd rfl hs | rfl

/-- The bound node's group, filtered by the row's test, is the whole list filtered by it: the
group key is one conjunct of the test. -/
theorem bucket_filter (all : List (Nat × Nat × Nat)) (bf bt : Option Nat) (sa tr : Bool)
    (node : Nat) (byFrom : Bool) (hk : boundKey bf bt = some (node, byFrom)) :
    (groupByEndpoint all tr byFrom node).filter (EndpointFilter.keeps ⟨bf, bt, sa, tr⟩) =
      all.filter (EndpointFilter.keeps ⟨bf, bt, sa, tr⟩) := by
  rw [groupByEndpoint_spec, List.filter_filter]
  apply List.filter_congr
  intro ⟨s, d, e⟩ _
  match bf, bt with
  | none, none => simp [boundKey] at hk
  | some id, _ =>
    simp only [boundKey, Option.some.injEq, Prod.mk.injEq] at hk
    obtain ⟨hid, hbf⟩ := hk
    subst hbf; subst hid
    cases tr <;> by_cases h1 : s = id <;> by_cases h2 : d = id <;>
      simp [EndpointFilter.keeps, endpointKey, patternEnds, h1, h2] <;> intros <;> simp_all
  | none, some id =>
    simp only [boundKey, Option.some.injEq, Prod.mk.injEq] at hk
    obtain ⟨hid, hbf⟩ := hk
    subst hbf; subst hid
    cases tr <;> by_cases h1 : s = id <;> by_cases h2 : d = id <;>
      simp [EndpointFilter.keeps, endpointKey, patternEnds, h1, h2] <;> intros <;> simp_all

/-- A usable query's streamed row is its oriented index answer through `EndpointFilter`. -/
theorem edgeRow_usable (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat))
    (q : IQ V) (bf bt : Option Nat) (sa tr bd : Bool) (hq : canUseE q = true) :
    edgeRow idx all q bf bt sa tr bd =
      (orient bd (idx q)).filter (EndpointFilter.keeps ⟨bf, bt, sa, tr⟩) := by
  unfold edgeRow
  simp only [hq, ite_true]
  split
  · rename_i h
    exact (List.filter_eq_self.mpr fun x _ => EndpointFilter.keepsAll_keeps _ h x).symm
  · rfl

/-- **The hit cache changes no row** (#3376): while the cache holds `q`'s hits (or none), a row
emits exactly what streaming `q` emits, in the same order, and the cache still holds `q`'s hits
afterwards. -/
theorem edgeNextRow_eq_edgeRow (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat))
    (q : IQ V) (bf bt : Option Nat) (sa tr bd : Bool) (max : Nat) (s : CacheState)
    (hinv : CacheInv idx bd q s) :
    (edgeNextRow idx all q bf bt sa tr bd max s).2 = edgeRow idx all q bf bt sa tr bd ∧
      CacheInv idx bd q (edgeNextRow idx all q bf bt sa tr bd max s).1 := by
  unfold edgeNextRow
  by_cases hq : canUseE q = true
  · simp only [hq, ite_true]
    cases hk : boundKey bf bt with
    | none =>
      have : lookup idx bd max q bf bt tr s = (s, none) := by unfold lookup; rw [hk]
      rw [this]; exact ⟨rfl, hinv⟩
    | some k =>
      obtain ⟨node, byFrom⟩ := k
      by_cases hc : s = .cold
      · subst hc
        have : lookup idx bd max q bf bt tr .cold = (.warm, none) := by unfold lookup; rw [hk]
        rw [this]; exact ⟨rfl, trivial⟩
      · rw [lookup_bound idx bd max q bf bt tr node byFrom hk s hc]
        have hb := build_inv idx bd max q s hinv
        revert hb
        generalize build idx bd max q s = s'
        intro hb
        cases s' with
        | built a =>
          simp only [CacheInv] at hb
          subst hb
          refine ⟨?_, rfl⟩
          simp only [sharedHits_eq]
          rw [bucket_filter _ _ _ _ _ _ _ hk, edgeRow_usable _ _ _ _ _ _ _ _ hq]
        | _ => exact ⟨rfl, hb⟩
  · simp only [hq]
    exact ⟨rfl, hinv⟩

/-- The operator over its input rows, each with its evaluated index query and bound endpoints,
threading the cache state from row to row. -/
def edgeRun (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat))
    (sa tr bd : Bool) (max : Nat) :
    CacheState → List (IQ V × Option Nat × Option Nat) → List (List (Nat × Nat × Nat))
  | _, [] => []
  | s, (q, bf, bt) :: rows =>
    (edgeNextRow idx all q bf bt sa tr bd max s).2 ::
      edgeRun idx all sa tr bd max (edgeNextRow idx all q bf bt sa tr bd max s).1 rows

theorem edgeNextRow_off (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat))
    (q : IQ V) (bf bt : Option Nat) (sa tr bd : Bool) (max : Nat) :
    edgeNextRow idx all q bf bt sa tr bd max .off = (.off, edgeRow idx all q bf bt sa tr bd) := by
  unfold edgeNextRow lookup
  by_cases hq : canUseE q = true
  · simp only [hq, ite_true]
    cases boundKey bf bt <;> rfl
  · simp [hq]

/-- An off cache (a write query, or a row-dependent index query) stays off, and every row
streams its own query. -/
theorem edgeRun_off (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat))
    (sa tr bd : Bool) (max : Nat) (rows : List (IQ V × Option Nat × Option Nat)) :
    edgeRun idx all sa tr bd max .off rows =
      rows.map fun (q, bf, bt) => edgeRow idx all q bf bt sa tr bd := by
  induction rows with
  | nil => rfl
  | cons r rows ih =>
    obtain ⟨q, bf, bt⟩ := r
    simp only [edgeRun, edgeNextRow_off, ih, List.map_cons]

/-- Rows that all ask the same query `q`, from a state holding `q`'s hits (or none): the run
emits what streaming emits, row by row. -/
theorem edgeRun_same_query (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat))
    (sa tr bd : Bool) (max : Nat) (q : IQ V) (rows : List (IQ V × Option Nat × Option Nat))
    (hq : ∀ r ∈ rows, r.1 = q) (s : CacheState) (hinv : CacheInv idx bd q s) :
    edgeRun idx all sa tr bd max s rows =
      rows.map fun (q, bf, bt) => edgeRow idx all q bf bt sa tr bd := by
  induction rows generalizing s with
  | nil => rfl
  | cons r rows ih =>
    obtain ⟨q', bf, bt⟩ := r
    have hq' : q' = q := hq _ (List.mem_cons_self ..)
    subst hq'
    obtain ⟨h1, h2⟩ := edgeNextRow_eq_edgeRow idx all q' bf bt sa tr bd max s hinv
    simp only [edgeRun, List.map_cons, h1]
    rw [ih (fun r hr => hq r (List.mem_cons_of_mem _ hr)) _ h2]

/-- **The operator emits the streamed rows** (#3376): from the state `new` sets, every input row
emits exactly what streaming its own index query emits, in order. The premise `hrow` is the claim
of `query_is_row_invariant` (unmodelled, see COVERAGE.tsv): a row-invariant index query evaluates
to the same `q` on every row. It is needed only when the cache can turn on (a read query); off
needs none. -/
theorem edgeScan_eq_streamed (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat))
    (sa tr bd : Bool) (max : Nat) (write rowInv : Bool) (q : IQ V)
    (rows : List (IQ V × Option Nat × Option Nat))
    (hrow : (!write && rowInv) = true → ∀ r ∈ rows, r.1 = q) :
    edgeRun idx all sa tr bd max (initState write rowInv) rows =
      rows.map fun (q, bf, bt) => edgeRow idx all q bf bt sa tr bd := by
  unfold initState
  cases hw : (write || !rowInv)
  · have hr : (!write && rowInv) = true := by cases write <;> cases rowInv <;> simp_all
    exact edgeRun_same_query idx all sa tr bd max q rows (hrow hr) .cold trivial
  · exact edgeRun_off ..

/-! ## Fulltext scans and the edge vector scan -/

/-- node_by_fulltext_scan.rs:73-105 / edge_by_fulltext_scan.rs: label and query must be
strings; the index answer (FFI) is emitted with its scores. -/
def ftArgs (what : String) (label query : Option String) : Except String (String × String) :=
  match label with
  | none => .error s!"fulltext query expects a string {what}"
  | some l => match query with
    | none => .error "fulltext query expects a string query"
    | some q => .ok (l, q)

theorem ftArgs_spec (w l q : String) :
    ftArgs w (some l) (some q) = .ok (l, q) ∧ ftArgs w none (some q) = .error s!"fulltext query expects a string {w}" ∧
    ftArgs w (some l) none = .error "fulltext query expects a string query" := ⟨rfl, rfl, rfl⟩

/-- `NodeByFulltextScanOp::new` (:42) / `EdgeByFulltextScanOp::new` (:47): scored column with the
operator's record cap; `EdgeByVectorScanOp::new` (:40): scored column, no cap. -/
def ftNew (ent : Nat) (score : Option Nat) (cap : Option Nat) : VecSt := ⟨ent, score, cap⟩

theorem ftNew_spec (e : Nat) (s c : Option Nat) :
    (ftNew e s c).nodeCol = e ∧ (ftNew e s c).scoreCol = s ∧ (ftNew e s c).cap = c ∧
    VecSt.new e s = ftNew e s none := ⟨rfl, rfl, rfl, rfl⟩

/-- edge_by_vector_scan.rs:75-105: the relationship KNN answer `(src, dst, edge, score)` is
projected to `(edge, score)`. -/
def edgeVecRow (q : String → String → Nat → Nat → Except String (List (Nat × Nat × Nat × Float)))
    (args : Except String (String × String × Nat × Nat)) : Except String (List (Nat × Float)) := do
  let (l, a, k, v) ← args
  let rs ← q l a v k
  pure (rs.map fun (_, _, e, s) => (e, s))

theorem edgeVecRow_spec (q : String → String → Nat → Nat → Except String (List (Nat × Nat × Nat × Float)))
    (l a : String) (k v : Nat) (rs : List (Nat × Nat × Nat × Float)) (h : q l a v k = .ok rs) :
    edgeVecRow q (.ok (l, a, k, v)) = .ok (rs.map fun (_, _, e, s) => (e, s)) := by
  simp [edgeVecRow, h, bind, Except.bind, pure, Except.pure]

end OpsTraverse.IndexScan
