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
    | some items => pure (.or ((items.filter isScalar).map (.eq k)))
    | none => throw "IN operator requires a list"
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

/-- `IN list` keeps only the scalar items, in order, as an `Or` of equalities (known
W2-index-3: a non-scalar item — date, array — is dropped). -/
theorem evalIQ_inList {E : Type} (ev : E → Except String V) (asList : E → Except String (Option (List V)))
    (k : String) (l : E) (items : List V) (h : asList l = .ok (some items)) :
    evalIQ ev asList (.inList k l) = .ok (.or ((items.filter isScalar).map (.eq k))) := by
  simp [evalIQ, h, bind, Except.bind, pure, Except.pure]

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

/-- edge_by_index_scan.rs:265-301: as `canUse`, except `ArrayContains` also refuses integers that
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
  | .inList _ _ => true
where
  canUseEL : List (IQ V) → Bool
  | [] => true
  | q :: qs => canUseE q && canUseEL qs

theorem canUseE_spec (k : String) (v : V) (lo hi : Option V) (a b : Bool) :
    canUseE (.eq k v) = canUse (.eq k v) ∧ canUseE (.range k lo hi a b) = canUse (.range k lo hi a b) ∧
    canUseE (.arr k (.int (2 ^ 60))) = false ∧ canUse (.arr k (.int (2 ^ 60))) = true := by
  refine ⟨by simp [canUseE], by simp [canUseE], by simp only [canUseE]; decide, rfl⟩

/-- edge_by_index_scan.rs:78-112: `to` is not bound separately for a self-loop alias. -/
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

/-- edge_by_index_scan.rs:316-366: the index answer (or the cached full edge list), then the
bound-endpoint and same-alias filter (orientation by `transposed`). -/
def edgeRow (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat)) (q : IQ V)
    (bFrom bTo : Option Nat) (sameAlias transposed : Bool) : List (Nat × Nat × Nat) :=
  let base := if canUseE q then idx q else all
  if bFrom.isSome || bTo.isSome || sameAlias then
    base.filter fun (s, d, _) =>
      let (f, t) := if transposed then (d, s) else (s, d)
      (bFrom.all (· == f)) && (bTo.all (· == t)) && (!sameAlias || f == t)
  else base

/-- Every emitted edge agrees with the bound endpoints (in pattern orientation) and is a loop
when both endpoints share an alias. -/
theorem edgeRow_sound (idx : IQ V → List (Nat × Nat × Nat)) (all : List (Nat × Nat × Nat)) (q : IQ V)
    (bf bt : Option Nat) (sa tr : Bool) (s d e : Nat) (h : (s, d, e) ∈ edgeRow idx all q bf bt sa tr) :
    (∀ x, bf = some x → x = (if tr then d else s)) ∧ (∀ x, bt = some x → x = (if tr then s else d)) ∧
    (sa = true → (if tr then d else s) = (if tr then s else d)) := by
  unfold edgeRow at h
  generalize (if canUseE q = true then idx q else all) = base at h
  by_cases hc : (bf.isSome || bt.isSome || sa) = true
  · rw [if_pos hc, List.mem_filter] at h
    obtain ⟨_, hp⟩ := h
    cases tr <;> simp only [Bool.and_eq_true, Bool.or_eq_true, Bool.not_eq_true', beq_iff_eq] at hp <;>
      simp only [Bool.false_eq_true, ite_false, ite_true] <;>
      obtain ⟨⟨h1, h2⟩, h3⟩ := hp <;>
      refine ⟨fun x hx => ?_, fun x hx => ?_, fun hs => ?_⟩ <;> simp_all
  · rw [if_neg hc] at h
    simp only [Bool.or_eq_true, not_or, Bool.not_eq_true, Option.isSome_eq_false_iff, Option.isNone_iff_eq_none] at hc
    obtain ⟨⟨h1, h2⟩, h3⟩ := hc
    refine ⟨fun x hx => by simp [h1] at hx, fun x hx => by simp [h2] at hx, fun hs => by simp [h3] at hs⟩

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
