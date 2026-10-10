import OptimizerScan.EdgeInline
/-
# Expression walkers, the fixed-point driver, and the edge clean-up passes of
# `utilize_index.rs` (origin/main 8743953a8, after #2390 d2c42e032)

| here | there |
| --- | --- |
| `D`, `Ex`, `bfs` | `ExprIR<Variable>` trees and `indices::<Bfs>()` |
| `extractAttr` | `extract_attribute_from_subtree` 357 |
| `hasPropOf`, `hasAnyProp`, `nestedListE` | `subtree_has_property_of` 547, `subtree_has_any_property` 569, `list_has_nested_list` 582 |
| `nonIdxD` | `is_non_indexable_subexpr` 855 |
| `IQx`, `pre2390_refsVar` | `IndexQuery<QueryExpr<Variable>>`; HISTORICAL `index_query_references_var` (1123 at a9377c636), replaced by `references.rs::index_query_references_variable` (proven in `proofs/optimizer_rewrites`, `References.iqRefs_iff`, by `(id, scope)`) |
| `rowsWithScan`, `prune_sound` | `prune_all_node_scan_child` 1055 |
| `hasLabelsFilter`, `isHasLabelsFor`, `addToLabels` | `build_has_labels_filter` 71, `is_has_labels_for` 1142, `add_to_labels_filter` 1084 |
| `untilStable` | `rewrite_until_stable` 881 |
| `governingFilter`, `matchScanWithFilter` | `governing_filter` 920, `match_scan_with_filter` 942 |
| `utilizeP`, `pendingSel` | `apply_filter_pushdown` 973 (`keep_filter = over_pending || needs_post_filter`) |
| `tryIndexRewrite`, `utilizeIndex` | `try_index_rewrite` 1021, `utilize_index` 1124 |
-/
namespace OptimizerScan.Cleanup
open OptimizerScan.Index

/-! ## Expression trees and BFS -/

inductive D
  | var (id : Nat)
  | param
  | cint (v : Int)
  | cstr (s : Nat)
  | cother
  | list
  | map
  | func (name : String)
  | prop (k : Nat)
  | other
  deriving DecidableEq, Repr

inductive Ex where
  | node (d : D) (cs : List Ex)

def Ex.d : Ex → D | .node d _ => d
def Ex.cs : Ex → List Ex | .node _ cs => cs

mutual
def Ex.size : Ex → Nat
  | .node _ cs => 1 + Ex.sizeL cs
def Ex.sizeL : List Ex → Nat
  | [] => 0
  | e :: es => e.size + Ex.sizeL es
end

theorem sizeL_append (a b : List Ex) : Ex.sizeL (a ++ b) = Ex.sizeL a + Ex.sizeL b := by
  induction a with
  | nil => simp [Ex.sizeL]
  | cons e es ih => simp [Ex.sizeL, ih]; omega

theorem sizeL_flatMap (l : List Ex) : Ex.sizeL (l.flatMap Ex.cs) + l.length = Ex.sizeL l := by
  induction l with
  | nil => simp [Ex.sizeL]
  | cons e es ih =>
    cases e with
    | node d cs => simp [Ex.sizeL, Ex.size, Ex.cs, sizeL_append, List.flatMap_cons] at *; omega

/-- Level-order traversal (`indices::<Bfs>()`): the current level, then the
next. -/
def bfsF : Nat → List Ex → List D
  | 0, _ => []
  | n + 1, l => if l = [] then [] else l.map Ex.d ++ bfsF n (l.flatMap Ex.cs)
def bfs (l : List Ex) : List D := bfsF (Ex.sizeL l) l

theorem size_pos (e : Ex) : 0 < e.size := by cases e; simp only [Ex.size]; omega
theorem sizeL_ge_length (l : List Ex) : l.length ≤ Ex.sizeL l := by
  induction l with
  | nil => simp [Ex.sizeL]
  | cons e es ih => have := size_pos e; simp [Ex.sizeL]; omega

-- Every node of a forest.
mutual
def Ex.nodes : Ex → List D
  | .node d cs => d :: Ex.nodesL cs
def Ex.nodesL : List Ex → List D
  | [] => []
  | e :: es => e.nodes ++ Ex.nodesL es
end

theorem nodesL_append (a b : List Ex) : Ex.nodesL (a ++ b) = Ex.nodesL a ++ Ex.nodesL b := by
  induction a with
  | nil => simp [Ex.nodesL]
  | cons e es ih => simp [Ex.nodesL, ih]

theorem mem_nodesL_split (l : List Ex) (x : D) :
    x ∈ Ex.nodesL l ↔ x ∈ l.map Ex.d ∨ x ∈ Ex.nodesL (l.flatMap Ex.cs) := by
  induction l with
  | nil => simp [Ex.nodesL]
  | cons e es ih =>
    cases e with
    | node d cs =>
      simp only [Ex.nodesL, Ex.nodes, List.map_cons, Ex.d, List.flatMap_cons, Ex.cs, nodesL_append,
        List.mem_append, List.mem_cons, ih]
      constructor
      · rintro ((h | h) | h | h)
        · exact Or.inl (Or.inl h)
        · exact Or.inr (Or.inl h)
        · exact Or.inl (Or.inr h)
        · exact Or.inr (Or.inr h)
      · rintro ((h | h) | h | h)
        · exact Or.inl (Or.inl h)
        · exact Or.inr (Or.inl h)
        · exact Or.inl (Or.inr h)
        · exact Or.inr (Or.inr h)

/-- **PROVEN**: BFS visits exactly the nodes of the forest. -/
theorem mem_bfsF (x : D) : ∀ (n : Nat) (l : List Ex), Ex.sizeL l ≤ n → (x ∈ bfsF n l ↔ x ∈ Ex.nodesL l)
  | 0, l, h => by
    have : l = [] := List.length_eq_zero_iff.mp (by have := sizeL_ge_length l; omega)
    subst this; simp [bfsF, Ex.nodesL]
  | n + 1, l, h => by
    simp only [bfsF]
    split
    · next he => subst he; simp [Ex.nodesL]
    · next he =>
      have hlt : Ex.sizeL (l.flatMap Ex.cs) ≤ n := by
        have := sizeL_flatMap l
        have : l.length > 0 := List.length_pos_iff.mpr he
        omega
      rw [List.mem_append, mem_bfsF x n _ hlt]; exact (mem_nodesL_split l x).symm

/-- **PROVEN**: BFS visits exactly the nodes of the forest. -/
theorem mem_bfs (l : List Ex) (x : D) : x ∈ bfs l ↔ x ∈ Ex.nodesL l :=
  mem_bfsF x _ l (Nat.le_refl _)

/-! ## The walkers -/

def propOf : D → Option Nat | .prop k => some k | _ => none

/-- `extract_attribute_from_subtree`: the first `Property` in BFS order. -/
def extractAttr (t : Ex) : Option Nat := (bfs [t]).findSome? propOf

theorem extractAttr_some (t : Ex) (k : Nat) (h : extractAttr t = some k) : D.prop k ∈ t.nodes := by
  obtain ⟨x, hx, hp⟩ := List.exists_of_findSome?_eq_some h
  have := (mem_bfs [t] x).mp hx
  simp [Ex.nodesL] at this
  cases x <;> simp [propOf] at hp
  subst hp; exact this

theorem extractAttr_none (t : Ex) : extractAttr t = none ↔ ∀ k, D.prop k ∉ t.nodes := by
  unfold extractAttr
  rw [List.findSome?_eq_none_iff]
  constructor
  · intro h k hk
    have := h (.prop k) ((mem_bfs [t] _).mpr (by simp [Ex.nodesL, hk]))
    simp [propOf] at this
  · intro h x hx
    have := (mem_bfs [t] x).mp hx
    simp [Ex.nodesL] at this
    cases x <;> simp [propOf]
    exact h _ this

/-- A `Property` node whose variable child is `alias`. -/
def propOfAlias (alias : Nat) (t : Ex) : Bool :=
  match t with
  | .node (.prop _) cs => cs.any (fun c => c.d == .var alias)
  | _ => false

mutual
def Ex.subtrees : Ex → List Ex
  | .node d cs => .node d cs :: Ex.subtreesL cs
def Ex.subtreesL : List Ex → List Ex
  | [] => []
  | e :: es => e.subtrees ++ Ex.subtreesL es
end

/-- `subtree_has_property_of` (the BFS `any` is order-independent). -/
def hasPropOf (t : Ex) (alias : Nat) : Bool := t.subtrees.any (propOfAlias alias)
/-- `subtree_has_any_property`. -/
def hasAnyProp (t : Ex) : Bool := (bfs [t]).any (fun d => (propOf d).isSome)
/-- `list_has_nested_list`: the root is a `List` with a direct `List` child. -/
def nestedListE (t : Ex) : Bool := t.d == .list && t.cs.any (fun c => c.d == .list)

theorem hasPropOf_iff (t : Ex) (a : Nat) :
    hasPropOf t a = true ↔ ∃ k cs, Ex.node (.prop k) cs ∈ t.subtrees ∧ ∃ c ∈ cs, c.d = .var a := by
  unfold hasPropOf
  simp only [List.any_eq_true]
  constructor
  · rintro ⟨s, hs, hp⟩
    match s, hp with
    | .node (.prop k) cs, hp =>
      simp only [propOfAlias, List.any_eq_true, beq_iff_eq] at hp
      exact ⟨k, cs, hs, hp⟩
  · rintro ⟨k, cs, hs, c, hc, hd⟩
    exact ⟨_, hs, by simp only [propOfAlias, List.any_eq_true, beq_iff_eq]; exact ⟨c, hc, hd⟩⟩

theorem hasAnyProp_iff (t : Ex) : hasAnyProp t = true ↔ ∃ k, D.prop k ∈ t.nodes := by
  unfold hasAnyProp
  simp only [List.any_eq_true, Option.isSome_iff_exists]
  constructor
  · rintro ⟨x, hx, k, hk⟩
    have := (mem_bfs [t] x).mp hx; simp [Ex.nodesL] at this
    match x, hk, this with
    | .prop k', hk, this => simp [propOf] at hk; subst hk; exact ⟨_, this⟩
  · rintro ⟨k, hk⟩
    exact ⟨.prop k, (mem_bfs [t] _).mpr (by simp [Ex.nodesL, hk]), k, rfl⟩

theorem nestedListE_iff (d : D) (cs : List Ex) :
    nestedListE (.node d cs) = true ↔ d = .list ∧ ∃ c ∈ cs, c.d = .list := by
  simp [nestedListE, Ex.d, Ex.cs]

/-- `is_non_indexable_subexpr` (`utilize_index.rs:855-873`). -/
def nonIdxD (lossyI : Int → Bool) (scan : Option Nat) : D → Bool
  | .var v => scan.all (· != v)
  | .param => true
  | .cint v => lossyI v
  | .list | .map => true
  | .func _ => true
  | _ => false

theorem nonIdxD_spec (lossyI : Int → Bool) (scan : Option Nat) (d : D) :
    nonIdxD lossyI scan d = true ↔
      (∃ v, d = .var v ∧ scan ≠ some v) ∨ d = .param ∨ (∃ v, d = .cint v ∧ lossyI v = true) ∨
      d = .list ∨ d = .map ∨ ∃ n, d = .func n := by
  cases d <;> simp [nonIdxD]
  cases scan <;> simp

/-! ## HISTORICAL `index_query_references_var` (removed by #2390) and `prune_all_node_scan_child`

#2390 replaced the id-only walker below by `references.rs::index_query_references_variable(q, id,
scope)`, proven in `proofs/optimizer_rewrites` (`References.iqRefs_iff`: true iff some operand
mentions the variable `(id, scope)`, at any And/Or depth). `prune_all_node_scan_child` now asks it
with the child scan's `(id, scope_id)` (`utilize_index.rs:1066-1069`). -/

inductive IQx
  | eq (e : Ex)
  | range (lo hi : Option Ex)
  | point (p r : Ex)
  | inList (e : Ex)
  | contains (e : Ex)
  | and (qs : List IQx)
  | or (qs : List IQx)

def exRefs (a : Nat) (e : Ex) : Bool := (bfs [e]).any (· == .var a)

def pre2390_refsVar (a : Nat) : IQx → Bool
  | .eq e | .contains e | .inList e => exRefs a e
  | .range lo hi => lo.any (exRefs a) || hi.any (exRefs a)
  | .point p r => exRefs a p || exRefs a r
  | .and qs | .or qs => qs.attach.any (fun ⟨q, _⟩ => pre2390_refsVar a q)

def IQx.exprs : IQx → List Ex
  | .eq e | .contains e | .inList e => [e]
  | .range lo hi => lo.toList ++ hi.toList
  | .point p r => [p, r]
  | .and qs | .or qs => qs.attach.flatMap (fun ⟨q, _⟩ => q.exprs)

theorem exRefs_iff (a : Nat) (e : Ex) : exRefs a e = true ↔ D.var a ∈ e.nodes := by
  simp [exRefs, mem_bfs, Ex.nodesL]

/-- **PROVEN** (historical walker): `index_query_references_var q a` iff some expression of `q`
mentions variable `a` (by id only). -/
theorem pre2390_refsVar_iff (a : Nat) : ∀ q : IQx, pre2390_refsVar a q = true ↔ ∃ e ∈ q.exprs, D.var a ∈ e.nodes
  | .eq e | .contains e | .inList e => by simp [pre2390_refsVar, IQx.exprs, exRefs_iff]
  | .range lo hi => by
    cases lo <;> cases hi <;> simp [pre2390_refsVar, IQx.exprs, exRefs_iff]
  | .point p r => by simp [pre2390_refsVar, IQx.exprs, exRefs_iff]
  | .and qs | .or qs => by
    simp only [pre2390_refsVar, IQx.exprs, List.any_eq_true, List.mem_flatMap]
    constructor
    · rintro ⟨⟨q, hq⟩, -, h⟩
      obtain ⟨e, he, hv⟩ := (pre2390_refsVar_iff a q).mp h
      exact ⟨e, ⟨⟨q, hq⟩, List.mem_attach _ _, he⟩, hv⟩
    · rintro ⟨e, ⟨⟨q, hq⟩, -, he⟩, hv⟩
      exact ⟨⟨q, hq⟩, List.mem_attach _ _, (pre2390_refsVar_iff a q).mpr ⟨e, he, hv⟩⟩

/-- An edge as the scan sees it, in the direction the pattern binds `from`. -/
structure Ed where
  src : Nat
  id : Nat
  deriving DecidableEq

/-- Rows of `EdgeByIndexScan` over an `AllNodeScan(from)` child: for every node
`n`, the index edges whose `from` endpoint is `n` (the bound-endpoint filter). -/
def rowsWithScan (N : List Nat) (E : List Ed) : List Ed := N.flatMap (fun n => E.filter (·.src == n))

theorem sum_ite_single (s c : Nat) : ∀ N : List Nat, N.Nodup →
    (N.map (fun n => if s = n then c else 0)).sum = if s ∈ N then c else 0
  | [], _ => by simp
  | n :: N, h => by
    have hn := (List.nodup_cons.mp h)
    rw [List.map_cons, List.sum_cons, sum_ite_single s c N hn.2]
    by_cases h1 : s = n
    · subst h1; simp [hn.1]
    · by_cases h2 : s ∈ N <;> simp [h1, h2]

/-- **PROVEN** (`prune_all_node_scan_child` is sound): dropping a leaf
`AllNodeScan` child yields the same edge multiset — every index edge is emitted
exactly once, by the row binding its `from` node — provided the node scan
enumerates each node once and covers every edge endpoint, and (by
`index_query_references_variable`, by `(id, scope)`) the index query does not read the dropped alias. -/
theorem prune_sound (N : List Nat) (E : List Ed) (hN : N.Nodup) (hcov : ∀ e ∈ E, e.src ∈ N) :
    (rowsWithScan N E).Perm E := by
  rw [List.perm_iff_count]
  intro e
  rw [rowsWithScan, List.count_flatMap]
  have hm : (N.map (List.count e ∘ fun n => E.filter (·.src == n))) =
      N.map (fun n => if e.src = n then List.count e E else 0) :=
    List.map_congr_left (fun n _ => by
      simp only [Function.comp]
      by_cases h : e.src = n
      · rw [if_pos h, List.count_filter (by simp [h])]
      · rw [if_neg h]; exact List.count_eq_zero.mpr (by simp [h]))
  rw [hm, sum_ite_single _ _ N hN]
  by_cases he : e ∈ E
  · rw [if_pos (hcov e he)]
  · rw [List.count_eq_zero.mpr he]; simp

/-! ## Endpoint label filter -/

def hasLabelsFilter (v : Nat) (labels : List Nat) : Ex :=
  .node (.func "hasLabels") [.node (.var v) [], .node .list (labels.map (fun l => .node (.cstr l) []))]

def isHasLabelsFor (f : Ex) (v : Nat) : Bool :=
  match f with
  | .node (.func "hasLabels") (.node (.var v') _ :: _) => v' == v
  | _ => false

/-- `add_to_labels_filter` at an `EdgeByIndexScan` whose `to` is `(toAlias, toLabels)`
and whose parent is `parent` (`some f` = a `Filter(f)`): `some f'` = push `Filter(f')`. -/
def addToLabels (toAlias : Nat) (toLabels : List Nat) (parent : Option (Option Ex)) : Option Ex :=
  if toLabels.isEmpty then none
  else if parent.any (fun p => p.any (isHasLabelsFor · toAlias)) then none
  else some (hasLabelsFilter toAlias toLabels)

theorem isHasLabelsFor_build (v : Nat) (ls : List Nat) : isHasLabelsFor (hasLabelsFilter v ls) v = true := by
  simp [isHasLabelsFor, hasLabelsFilter]

/-- **PROVEN**: the pass adds the `hasLabels(to, labels)` filter once, and the
re-run of the driver over the new parent is a no-op (so `rewrite_until_stable`
terminates on this pass). -/
theorem addToLabels_idem (v : Nat) (ls : List Nat) (p : Option (Option Ex)) (f : Ex)
    (h : addToLabels v ls p = some f) : f = hasLabelsFilter v ls ∧ addToLabels v ls (some (some f)) = none := by
  unfold addToLabels at h
  split at h; · cases h
  split at h; · cases h
  cases h
  refine ⟨rfl, ?_⟩
  next h1 _ => simp [addToLabels, h1, isHasLabelsFor_build]

/-- Semantics of `hasLabels(n, ls)`: `n` carries every label in `ls`. With it the
edge scan again enforces the `to` endpoint's labels (the edge index cannot). -/
def hasLabelsSem (labelsOf : Nat → List Nat) (n : Nat) (ls : List Nat) : Bool := ls.all (· ∈ labelsOf n)

/-! ## The fixed-point driver -/

/-- `rewrite_until_stable` with an explicit fuel bound (Rust loops without one):
each round tries every position in BFS order and restarts after the first
success. `step p i = some p'` is a successful rewrite at position `i`. -/
def untilStable {Pl : Type} (positions : Pl → List Nat) (step : Pl → Nat → Option Pl) : Nat → Pl → Pl
  | 0, p => p
  | fuel + 1, p => match (positions p).findSome? (step p) with
    | none => p
    | some p' => untilStable positions step fuel p'

/-- **PROVEN**: every invariant (e.g. "selects the same rows") preserved by a
single rewrite is preserved by the driver. -/
theorem untilStable_preserves {Pl : Type} (positions : Pl → List Nat) (step : Pl → Nat → Option Pl)
    (P : Pl → Prop) (hstep : ∀ p i p', P p → step p i = some p' → P p') :
    ∀ fuel p, P p → P (untilStable positions step fuel p)
  | 0, _, h => h
  | fuel + 1, p, h => by
    simp only [untilStable]
    split
    · exact h
    · next p' hp =>
      obtain ⟨i, -, hi⟩ := List.exists_of_findSome?_eq_some hp
      exact untilStable_preserves positions step P hstep fuel p' (hstep p i p' h hi)

/-- **PROVEN**: if every rewrite strictly decreases a measure `μ`, fuel `μ p + 1`
suffices and the result is stable (no position rewrites). -/
theorem untilStable_stable {Pl : Type} (positions : Pl → List Nat) (step : Pl → Nat → Option Pl)
    (μ : Pl → Nat) (hdec : ∀ p i p', step p i = some p' → μ p' < μ p) :
    ∀ fuel p, μ p < fuel → ∀ i ∈ positions (untilStable positions step fuel p),
      step (untilStable positions step fuel p) i = none
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | fuel + 1, p, h => by
    simp only [untilStable]
    split
    · next hn => intro i hi; exact (List.findSome?_eq_none_iff.mp hn) i hi
    · next p' hp =>
      obtain ⟨i, -, hi⟩ := List.exists_of_findSome?_eq_some hp
      exact untilStable_stable positions step μ hdec fuel p' (by have := hdec p i p' hi; omega)

/-! ## `governing_filter`, `match_scan_with_filter`, `apply_filter_pushdown`, `try_index_rewrite` -/

/-- The scan's ancestors as `governing_filter` sees them, nearest first. -/
inductive Anc
  | filter (f : F)
  | pending (nchildren : Nat)
  | other

/-- `governing_filter` (`utilize_index.rs:920-935`): the `Filter` directly above the scan, or above a
    single-child `IncludePending` above it. Returns the Filter's depth (its `NodeIdx`), the filter, and
    whether an `IncludePending` was skipped. -/
def governingFilter : List Anc → Option (Nat × F × Bool)
  | [] => none                                                       -- no parent (line 924)
  | .pending 1 :: rest => match rest with                            -- lines 926-929
    | .filter f :: _ => some (1, f, true)
    | _ => none
  | .filter f :: _ => some (0, f, false)                             -- lines 930-934
  | _ => none

/-- **PROVEN**: the governing Filter is the parent, or the grandparent across exactly one
    single-child `IncludePending` (and then `over_pending` is set). -/
theorem governingFilter_spec (anc : List Anc) (d : Nat) (f : F) (op : Bool) :
    governingFilter anc = some (d, f, op) ↔
      (∃ rest, anc = .filter f :: rest ∧ d = 0 ∧ op = false) ∨
      (∃ rest, anc = .pending 1 :: .filter f :: rest ∧ d = 1 ∧ op = true) := by
  constructor
  · intro h
    rcases anc with _ | ⟨a, rest⟩
    · simp [governingFilter] at h
    · cases a with
      | filter g =>
        simp only [governingFilter, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h; exact Or.inl ⟨rest, rfl, rfl, rfl⟩
      | pending k =>
        by_cases hk : k = 1
        · subst hk
          rcases rest with _ | ⟨b, rest'⟩
          · simp [governingFilter] at h
          · cases b with
            | filter g =>
              simp only [governingFilter, Option.some.injEq, Prod.mk.injEq] at h
              obtain ⟨rfl, rfl, rfl⟩ := h; exact Or.inr ⟨rest', rfl, rfl, rfl⟩
            | pending _ => simp [governingFilter] at h
            | other => simp [governingFilter] at h
        · unfold governingFilter at h
          split at h <;> simp_all
      | other => simp [governingFilter] at h
  · rintro (⟨rest, rfl, rfl, rfl⟩ | ⟨rest, rfl, rfl, rfl⟩) <;> rfl

/-- `match_scan_with_filter` (`utilize_index.rs:942-950`): the scan source matches and a Filter
    governs it. -/
def matchScanWithFilter {S : Type} (src : Option S) (anc : List Anc) : Option (S × Nat × F × Bool) :=
  match src, governingFilter anc with
  | some s, some (d, f, op) => some (s, d, f, op)
  | _, _ => none

theorem matchScanWithFilter_spec {S : Type} (src : Option S) (anc : List Anc) (s : S) (d : Nat) (f : F) (op : Bool) :
    matchScanWithFilter src anc = some (s, d, f, op) ↔ src = some s ∧ governingFilter anc = some (d, f, op) := by
  unfold matchScanWithFilter
  cases src with
  | none => simp
  | some s' =>
    cases hg : governingFilter anc with
    | none => simp
    | some t =>
      obtain ⟨d', f', op'⟩ := t
      simp only [Option.some.injEq, Prod.mk.injEq]

/-- `apply_filter_pushdown` (`utilize_index.rs:973-1013`) with its `over_pending` flag:
    `keep_filter = over_pending || needs_post_filter`. `utilizeP _ _ _ false` is `utilize`. -/
def utilizeP (idx : Nat → List Nat) (ls : List Nat) (f : F) (overPending : Bool) : Plan :=
  match tryPushdown idx ls f with
  | none => .filter f (.labelScan ls)
  | some (L, q, rem) =>
    let keep := overPending || needsPost f
    let scan := Plan.idxScan ls L q
    if rem.isEmpty then (if keep then .filter f scan else scan)
    else if keep then .filter f scan else .filter (.and rem) scan

theorem utilizeP_false (idx : Nat → List Nat) (ls : List Nat) (f : F) :
    utilizeP idx ls f false = utilize idx ls f := by
  unfold utilizeP utilize; rfl

/-- Over an `IncludePending` the original Filter always stays above the index scan. -/
theorem utilizeP_true (idx : Nat → List Nat) (ls : List Nat) (f : F) (L' : Nat) (q : IQ) (rem : List A)
    (h : tryPushdown idx ls f = some (L', q, rem)) :
    utilizeP idx ls f true = .filter f (.idxScan ls L' q) := by
  simp [utilizeP, h]

/-- The MERGE shape `Filter → IncludePending → X`: `X`'s rows plus the pending (created-in-this-query)
    nodes, all re-checked by the Filter. -/
def pendingSel (idx : Nat → List Nat) (opq : Nat → Node → Bool) (pend : Node → Bool) (f : F) (X : Plan)
    (n : Node) : Bool :=
  (X.sel idx opq n || pend n) && evalF opq n f

theorem idxScan_labels (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L L' : Nat) (q : IQ) (n : Node)
    (h : (Plan.idxScan [L] L' q).sel idx opq n = true) : L ∈ n.labels := by
  simp only [Plan.sel, Bool.and_eq_true] at h
  obtain ⟨h1, h2⟩ := h
  by_cases hL : L' = L
  · subst hL
    split at h1
    · simp only [idxSel, Bool.and_eq_true, decide_eq_true_eq] at h1; exact h1.1
    · simpa [hasAll] using h1
  · have he : [L].erase L' = [L] := by
      rw [List.erase_cons]; simp [show ¬ L = L' from fun e => hL e.symm]
    rw [he] at h2; simpa [hasAll] using h2

/-- On a single-label scan, the index scan under the original filter selects what
    `Filter → NodeByLabelScan` does (from `utilize_sound`, every shape). -/
theorem idxScan_filter_sound (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L : Nat) (f : F)
    (hf : goodF f = true) (L' : Nat) (q : IQ) (rem : List A) (hp : tryPushdown idx [L] f = some (L', q, rem))
    (n : Node) (hn : Faithful n) :
    ((Plan.idxScan [L] L' q).sel idx opq n && evalF opq n f) = (reference [L] f).sel idx opq n := by
  have hu := utilize_sound idx L opq f hf n hn
  rw [utilize_some idx [L] f L' q rem hp] at hu
  have href : (reference [L] f).sel idx opq n = (decide (L ∈ n.labels) && evalF opq n f) := by
    simp [reference, Plan.sel, hasAll_single]
  rw [href] at hu ⊢
  have hlab := idxScan_labels idx opq L L' q n
  cases hfv : evalF opq n f
  · simp
  · cases hsc : (Plan.idxScan [L] L' q).sel idx opq n
    · -- the index scan rejects `n` although `f` holds: the reference rejects it too
      rw [hfv] at hu
      simp only [Bool.false_and, Bool.and_true]
      have hF : ∀ X, Plan.sel idx opq (.filter X (.idxScan [L] L' q)) n =
          ((Plan.idxScan [L] L' q).sel idx opq n && evalF opq n X) := fun X => rfl
      split at hu <;> (try split at hu) <;> (simp only [hF, hsc, Bool.false_and, Bool.and_true] at hu; exact hu)
    · simp [hlab hsc]

/-- **PROVEN** (#2390, MERGE over `IncludePending`): with the Filter kept (`over_pending`), the index
    scan under `Filter → IncludePending` selects exactly what `Filter → IncludePending →
    NodeByLabelScan` does, whatever the pending nodes are. -/
theorem overPending_sound (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L : Nat) (f : F)
    (hf : goodF f = true) (L' : Nat) (q : IQ) (rem : List A) (hp : tryPushdown idx [L] f = some (L', q, rem))
    (pend : Node → Bool) (n : Node) (hn : Faithful n) :
    pendingSel idx opq pend f (.idxScan [L] L' q) n = pendingSel idx opq pend f (.labelScan [L]) n := by
  have h := idxScan_filter_sound idx opq L f hf L' q rem hp n hn
  simp only [reference, Plan.sel] at h
  unfold pendingSel
  cases hfv : evalF opq n f <;> simp_all [Plan.sel]

/-- ...and the flag is necessary: dropping the Filter there would let a pending node that fails it
    through (the doc's `MERGE (p1:person {age: 40}) MERGE (p2:person {age: 41})` matching `p1` for
    `p2`). A pending node `p` failing `f` is selected by the unfiltered plan, not by the reference. -/
theorem overPending_needs_filter (idx : Nat → List Nat) (opq : Nat → Node → Bool) (X : Plan) (f : F)
    (pend : Node → Bool) (p : Node) (hp : pend p = true) (hfail : evalF opq p f = false) :
    (X.sel idx opq p || pend p) = true ∧ pendingSel idx opq pend f (.labelScan []) p = false := by
  simp [pendingSel, hp, hfail]

/-- `try_index_rewrite` (`utilize_index.rs:1021-1047`) for a single-label node scan: only the
    filter push-down path remains (#2390). -/
def tryIndexRewrite (idx : Nat → List Nat) (L : Nat) (gov : Option (Nat × F × Bool)) : Option Plan :=
  match gov with
  | some (_, f, op) => match tryPushdown idx [L] f with
    | some _ => some (utilizeP idx [L] f op)
    | none => none
  | none => none

/-- **PROVEN** (`try_index_rewrite` is sound): on the covered shapes the new subtree selects what
`Filter → NodeByLabelScan` did (`utilize_sound`), and over an `IncludePending` what
`Filter → IncludePending → NodeByLabelScan` did (`overPending_sound`). -/
theorem tryIndexRewrite_sound (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L : Nat) (f : F)
    (hf : goodF f = true) (d : Nat) (p : Plan)
    (h : tryIndexRewrite idx L (some (d, f, false)) = some p)
    (n : Node) (hn : Faithful n) : p.sel idx opq n = (reference [L] f).sel idx opq n := by
  unfold tryIndexRewrite at h
  cases hp : tryPushdown idx [L] f with
  | none => simp [hp] at h
  | some _ => simp [hp] at h; subst h; rw [utilizeP_false]; exact utilize_sound idx L opq f hf n hn

theorem tryIndexRewrite_pending_sound (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L : Nat) (f : F)
    (hf : goodF f = true) (d : Nat) (p : Plan) (h : tryIndexRewrite idx L (some (d, f, true)) = some p)
    (pend : Node → Bool) (n : Node) (hn : Faithful n) :
    ∃ L' q, p = .filter f (.idxScan [L] L' q) ∧
      pendingSel idx opq pend f (.idxScan [L] L' q) n = pendingSel idx opq pend f (.labelScan [L]) n := by
  unfold tryIndexRewrite at h
  cases hp : tryPushdown idx [L] f with
  | none => simp [hp] at h
  | some t =>
    obtain ⟨L', q, rem⟩ := t
    simp [hp] at h; subst h
    exact ⟨L', q, utilizeP_true idx [L] f L' q rem hp, overPending_sound idx opq L f hf L' q rem hp pend n hn⟩

/-- `utilize_index`: four `rewrite_until_stable` passes in sequence. Soundness
composes: if each pass's rewrites preserve the plan's semantics, so does the
whole pass. -/
def utilizeIndex {Pl : Type} (positions : Pl → List Nat) (s1 s2 s3 s4 : Pl → Nat → Option Pl) (fuel : Nat)
    (p : Pl) : Pl :=
  untilStable positions s4 fuel (untilStable positions s3 fuel
    (untilStable positions s2 fuel (untilStable positions s1 fuel p)))

theorem utilizeIndex_sound {Pl : Type} (positions : Pl → List Nat) (s1 s2 s3 s4 : Pl → Nat → Option Pl)
    (sem : Pl → Pl → Prop) (p0 : Pl) (htrans : ∀ a b, sem p0 a → sem a b → sem p0 b)
    (hrefl : sem p0 p0)
    (h1 : ∀ p i p', s1 p i = some p' → sem p p') (h2 : ∀ p i p', s2 p i = some p' → sem p p')
    (h3 : ∀ p i p', s3 p i = some p' → sem p p') (h4 : ∀ p i p', s4 p i = some p' → sem p p')
    (fuel : Nat) : sem p0 (utilizeIndex positions s1 s2 s3 s4 fuel p0) := by
  have pres : ∀ (s : Pl → Nat → Option Pl), (∀ p i p', s p i = some p' → sem p p') →
      ∀ q, sem p0 q → sem p0 (untilStable positions s fuel q) := fun s hs q hq =>
    untilStable_preserves positions s (sem p0) (fun p i p' hp hi => htrans _ _ hp (hs p i p' hi)) fuel q hq
  exact pres s4 h4 _ (pres s3 h3 _ (pres s2 h2 _ (pres s1 h1 _ hrefl)))

end OptimizerScan.Cleanup
