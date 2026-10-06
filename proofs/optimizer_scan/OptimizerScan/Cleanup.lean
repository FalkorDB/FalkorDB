import OptimizerScan.EdgeInline
/-
# Expression walkers, the fixed-point driver, and the edge clean-up passes of
# `utilize_index.rs` (origin/main 3fec7d7c9)

| here | there |
| --- | --- |
| `D`, `Ex`, `bfs` | `ExprIR<Variable>` trees and `indices::<Bfs>()` |
| `extractAttr` | `extract_attribute_from_subtree` 367 |
| `hasPropOf`, `hasAnyProp`, `nestedListE` | `subtree_has_property_of` 557, `subtree_has_any_property` 579, `list_has_nested_list` 592 |
| `nonIdxD` | `is_non_indexable_subexpr` 910 |
| `IQx`, `refsVar` | `IndexQuery<QueryExpr<Variable>>`, `index_query_references_var` 1123 |
| `rowsWithScan`, `prune_sound` | `prune_all_node_scan_child` 1094 |
| `hasLabelsFilter`, `isHasLabelsFor`, `addToLabels` | `build_has_labels_filter` 72, `is_has_labels_for` 1209, `add_to_labels_filter` 1151 |
| `untilStable` | `rewrite_until_stable` 936 |
| `tryIndexRewrite`, `matchScanWithFilter`, `utilizeIndex` | `try_index_rewrite` 1063, `match_scan_with_filter` 962, `utilize_index` 1191 |
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

/-- `is_non_indexable_subexpr` (`utilize_index.rs:910-928`). -/
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

/-! ## `index_query_references_var` and `prune_all_node_scan_child` -/

inductive IQx
  | eq (e : Ex)
  | range (lo hi : Option Ex)
  | point (p r : Ex)
  | inList (e : Ex)
  | contains (e : Ex)
  | and (qs : List IQx)
  | or (qs : List IQx)

def exRefs (a : Nat) (e : Ex) : Bool := (bfs [e]).any (· == .var a)

def refsVar (a : Nat) : IQx → Bool
  | .eq e | .contains e | .inList e => exRefs a e
  | .range lo hi => lo.any (exRefs a) || hi.any (exRefs a)
  | .point p r => exRefs a p || exRefs a r
  | .and qs | .or qs => qs.attach.any (fun ⟨q, _⟩ => refsVar a q)

def IQx.exprs : IQx → List Ex
  | .eq e | .contains e | .inList e => [e]
  | .range lo hi => lo.toList ++ hi.toList
  | .point p r => [p, r]
  | .and qs | .or qs => qs.attach.flatMap (fun ⟨q, _⟩ => q.exprs)

theorem exRefs_iff (a : Nat) (e : Ex) : exRefs a e = true ↔ D.var a ∈ e.nodes := by
  simp [exRefs, mem_bfs, Ex.nodesL]

/-- **PROVEN**: `index_query_references_var q a` iff some expression of `q`
mentions variable `a`. -/
theorem refsVar_iff (a : Nat) : ∀ q : IQx, refsVar a q = true ↔ ∃ e ∈ q.exprs, D.var a ∈ e.nodes
  | .eq e | .contains e | .inList e => by simp [refsVar, IQx.exprs, exRefs_iff]
  | .range lo hi => by
    cases lo <;> cases hi <;> simp [refsVar, IQx.exprs, exRefs_iff]
  | .point p r => by simp [refsVar, IQx.exprs, exRefs_iff]
  | .and qs | .or qs => by
    simp only [refsVar, IQx.exprs, List.any_eq_true, List.mem_flatMap]
    constructor
    · rintro ⟨⟨q, hq⟩, -, h⟩
      obtain ⟨e, he, hv⟩ := (refsVar_iff a q).mp h
      exact ⟨e, ⟨⟨q, hq⟩, List.mem_attach _ _, he⟩, hv⟩
    · rintro ⟨e, ⟨⟨q, hq⟩, -, he⟩, hv⟩
      exact ⟨⟨q, hq⟩, List.mem_attach _ _, (refsVar_iff a q).mpr ⟨e, he, hv⟩⟩

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
`index_query_references_var`) the index query does not read the dropped alias. -/
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

/-! ## `try_index_rewrite`, `match_scan_with_filter`, `utilize_index` -/

/-- `match_scan_with_filter`: the scan source matches and its parent is a `Filter`. -/
def matchScanWithFilter {S : Type} (src : Option S) (parent : Option (Option F)) : Option (S × F) :=
  match src, parent with
  | some s, some (some f) => some (s, f)
  | _, _ => none

theorem matchScanWithFilter_spec {S : Type} (src : Option S) (parent : Option (Option F)) (s : S) (f : F) :
    matchScanWithFilter src parent = some (s, f) ↔ src = some s ∧ parent = some (some f) := by
  unfold matchScanWithFilter; split <;> simp_all

/-- `try_index_rewrite` for a single-label node scan: filter push-down first,
then the inline-attribute path. -/
def tryIndexRewrite (idx : Nat → List Nat) (L : Nat) (parentFilter : Option F) (attrs : List (Nat × T)) :
    Option Plan :=
  match parentFilter with
  | some f => match tryPushdown idx [L] f with
    | some _ => some (utilize idx [L] f)
    | none => inl
  | none => inl
where inl := (inlineIdx (fun L' k => k ∈ idx L') [L] attrs).map (fun ⟨L', k, v⟩ => applyInline [L] L' k v)

/-- **PROVEN** (`try_index_rewrite` is sound): whichever path fires, on the
covered shapes the new subtree selects what `Filter → NodeByLabelScan` did
(filter path: `utilize_sound`; inline path: `applyInline_sound`). -/
theorem tryIndexRewrite_sound (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L : Nat) (f : F)
    (hf : goodF f = true) (attrs : List (Nat × T)) (p : Plan)
    (h : tryIndexRewrite idx L (some f) attrs = some p) (hfire : (tryPushdown idx [L] f).isSome)
    (n : Node) (hn : Faithful n) : p.sel idx opq n = (reference [L] f).sel idx opq n := by
  unfold tryIndexRewrite at h
  cases hp : tryPushdown idx [L] f with
  | none => simp [hp] at hfire
  | some _ => simp [hp] at h; subst h; exact utilize_sound idx L opq f hf n hn

theorem tryIndexRewrite_inline_sound (idx : Nat → List Nat) (opq : Nat → Node → Bool) (L : Nat)
    (attrs : List (Nat × T)) (p : Plan) (k : Nat) (c : V) (hc : goodV c = true)
    (hfirst : inlineIdx (fun L' k => k ∈ idx L') [L] attrs = some (L, k, .lit c))
    (h : tryIndexRewrite idx L none attrs = some p) (n : Node) (hn : Faithful n) :
    p.sel idx opq n = (decide (L ∈ n.labels) && Op.eq.holds (n.prop k) c) := by
  simp only [tryIndexRewrite, tryIndexRewrite.inl, hfirst, Option.map_some] at h
  cases h
  have hk := (inlineIdx_spec _ _ _ _ _ _ hfirst).2.2
  exact applyInline_sound idx opq L k c hc (by simpa using hk) n hn

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
