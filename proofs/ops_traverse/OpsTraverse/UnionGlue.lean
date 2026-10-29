import OpsTraverse.UnionStream
import OpsTraverse.CondTraverse
import OpsTraverse.CTGlue
import GraphblasWrappers.DeltaLmxm
/-
# Multi-type traversal (#3492): matrix selection, `delta_lmxm_into`, builders, `expand_union_row`

| here | there |
| --- | --- |
| `relationshipTensors`        | `relationship_tensors` (`cond_traverse.rs:288`) |
| `buildUnrestricted`, `buildTransposed` | `build_unrestricted_iter` (:302), `build_transposed_iter` (:337): untyped or single type |
| `buildUnion`, `buildTransposedUnion`   | `build_union_iter` (:317), `build_transposed_union_iter` (:352) |
| `fwdOkOf`                    | the `fwd_ok` / `rev_ok` of `build_state` (:508-528) |
| `pickMatrix`                 | the `TraversalMatrix` selection of `expand_batch` (:615-650, chain hops :658-680) |
| `deltaInto`, `TMat`          | `TraversalMatrix::delta_lmxm_into` (:86-112) |
| `expandRowP`, `expandUnionRow` | `expand_row`'s forward/reverse blocks over `PairIters<UnionIter>` = `expand_union_row` (:1121) |
| `expandRowDispatch`          | the `state.union.is_some()` branch of `expand_row` (:963-990) |
-/

namespace OpsTraverse.UnionGlue
open OpsTraverse OpsTraverse.UnionStream

/-! ## `relationship_tensors` and the iterator builders -/

/-- The tensors of `types` that exist, in type order; `None` when none do. -/
def relationshipTensors {T Ty : Type} (get : Ty → Option T) (types : List Ty) : Option (List T) :=
  let ts := types.filterMap get
  if ts.isEmpty then none else some ts

theorem relationshipTensors_none {T Ty : Type} (get : Ty → Option T) (types : List Ty) :
    relationshipTensors get types = none ↔ ∀ t ∈ types, get t = none := by
  unfold relationshipTensors
  by_cases h : (types.filterMap get).isEmpty = true
  · simp only [h, if_true, true_iff]
    intro t ht
    cases hg : get t with
    | none => rfl
    | some x =>
      have : x ∈ types.filterMap get := List.mem_filterMap.2 ⟨t, ht, hg⟩
      simp [List.isEmpty_iff.1 h] at this
  · simp only [h, Bool.false_eq_true, if_false]
    have : types.filterMap get ≠ [] := by simpa using h
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ this
    obtain ⟨t, ht, hg⟩ := List.mem_filterMap.1 hx
    constructor
    · intro h; cases h
    · intro h; exfalso; have := h t ht; simp [this] at hg

theorem relationshipTensors_some {T Ty : Type} (get : Ty → Option T) (types : List Ty) (ts : List T)
    (h : relationshipTensors get types = some ts) :
    ts = types.filterMap get ∧ ts ≠ [] ∧ ∀ x ∈ ts, ∃ t ∈ types, get t = some x := by
  unfold relationshipTensors at h
  by_cases hE : (types.filterMap get).isEmpty = true
  · simp [hE] at h
  · simp only [hE, Bool.false_eq_true, if_false, Option.some.injEq] at h
    subst h
    refine ⟨rfl, by simpa using hE, fun x hx => ?_⟩
    obtain ⟨t, ht, hg⟩ := List.mem_filterMap.1 hx
    exact ⟨t, ht, hg⟩

/-- `build_unrestricted_iter` / `build_transposed_iter`: untyped is the adjacency (transposed) iterator,
a single type is that type's iterator (`None` when the type does not exist). The other types of a
longer list are never read: `debug_assert!(types.len() <= 1)` (:307, :342), several types go
through `buildUnion`. -/
def buildUnrestricted {I Ty : Type} (adj : I) (rel : Ty → Option I) : List Ty → Option I
  | [] => some adj
  | t :: _ => rel t

theorem buildUnrestricted_spec {I Ty : Type} (adj : I) (rel : Ty → Option I) (t : Ty) :
    buildUnrestricted adj rel [] = some adj ∧ buildUnrestricted adj rel [t] = rel t := ⟨rfl, rfl⟩

/-- `build_union_iter` / `build_transposed_union_iter`: one iterator per existing type, `None` when
no type exists; `UnionIter::new` leaves every head empty. -/
def buildUnion {T I Ty : Type} (get : Ty → Option T) (iter : T → I) (types : List Ty) :
    Option (List I) :=
  (relationshipTensors get types).map (·.map iter)

theorem buildUnion_spec {T I Ty : Type} (get : Ty → Option T) (iter : T → I) (types : List Ty) :
    (buildUnion get iter types = none ↔ ∀ t ∈ types, get t = none) ∧
    ∀ is, buildUnion get iter types = some is →
      is = (types.filterMap get).map iter ∧ is ≠ [] := by
  refine ⟨?_, fun is h => ?_⟩
  · unfold buildUnion
    rw [← relationshipTensors_none]
    cases relationshipTensors get types <;> simp
  · unfold buildUnion at h
    cases hr : relationshipTensors get types with
    | none => simp [hr] at h
    | some ts =>
      simp only [hr, Option.map_some, Option.some.injEq] at h
      obtain ⟨h1, h2, _⟩ := relationshipTensors_some get types ts hr
      subst h
      refine ⟨by rw [h1], ?_⟩
      cases ts with
      | nil => exact absurd rfl h2
      | cons a t => simp

/-- Whether `build_state` finds its forward iterator: untyped always, a single type iff it
exists, several types iff at least one exists. -/
def fwdOkOf {Ty : Type} (exists_ : Ty → Bool) : List Ty → Bool
  | [] => true
  | [t] => exists_ t
  | t :: u :: rest => (t :: u :: rest).any exists_

theorem fwdOkOf_multi {Ty : Type} (get : Ty → Option Nat) (t u : Ty) (rest : List Ty) :
    fwdOkOf (fun x => (get x).isSome) (t :: u :: rest) = true ↔
      (buildUnion get id (t :: u :: rest)).isSome = true := by
  unfold buildUnion
  have := relationshipTensors_none get (t :: u :: rest)
  cases hr : relationshipTensors get (t :: u :: rest) with
  | none =>
    rw [hr] at this
    simp only [fwdOkOf, List.any_eq_true, Option.map_none, Option.isSome_none, Bool.false_eq_true,
      iff_false, not_exists, not_and]
    intro x hx hs
    have := this.1 rfl x hx
    simp [this] at hs
  | some ts =>
    simp only [Option.map_some, Option.isSome_some, iff_true]
    obtain ⟨_, hne, hall⟩ := relationshipTensors_some get _ ts hr
    obtain ⟨x, hx⟩ := List.exists_mem_of_ne_nil _ hne
    obtain ⟨y, hy, hg⟩ := hall x hx
    simp only [fwdOkOf, List.any_eq_true]
    exact ⟨y, hy, by simp [hg]⟩

/-! ## `TraversalMatrix` selection (`expand_batch`) -/

/-- Which `TraversalMatrix` a (hop's) relationship types select: untyped is the adjacency, a
single type its tensor, several the per-type `Union` of the existing ones; `None` = unknown type(s)
(`no_match`). -/
inductive Sel (A T : Type)
  | bool (a : A)
  | u64 (t : T)
  | union (ts : List T)

def pickMatrix {A T Ty : Type} (adj : A) (get : Ty → Option T) : List Ty → Option (Sel A T)
  | [] => some (.bool adj)
  | [t] => (get t).map .u64
  | t :: u :: rest => (relationshipTensors get (t :: u :: rest)).map .union

theorem pickMatrix_spec {A T Ty : Type} (adj : A) (get : Ty → Option T) (t u : Ty) (rest : List Ty) :
    pickMatrix adj get [] = some (Sel.bool adj) ∧
    pickMatrix adj get [t] = (get t).map Sel.u64 ∧
    (∀ ts, pickMatrix adj get (t :: u :: rest) = some (Sel.union ts) →
      ts = (t :: u :: rest).filterMap get ∧ ts ≠ []) ∧
    (pickMatrix adj get (t :: u :: rest) = none ↔ ∀ x ∈ t :: u :: rest, get x = none) := by
  refine ⟨rfl, rfl, fun ts h => ?_, ?_⟩
  · simp only [pickMatrix] at h
    cases hr : relationshipTensors get (t :: u :: rest) with
    | none => simp [hr] at h
    | some l =>
      simp only [hr, Option.map_some, Option.some.injEq, Sel.union.injEq] at h
      subst h
      obtain ⟨h1, h2, _⟩ := relationshipTensors_some get _ l hr
      exact ⟨h1, h2⟩
  · simp only [pickMatrix]
    rw [← relationshipTensors_none]
    cases relationshipTensors get (t :: u :: rest) <;> simp

/-! ## `delta_lmxm_into` -/

open GBWDelta

/-- A traversal matrix by its versioned layers. -/
inductive TMat
  | bool (v : VM.VM)
  | u64 (v : VM.VM)
  | union (vs : List VM.VM)

/-- `Matrix::delta_lmxm` against one set of layers. -/
noncomputable def lm (n : Nat) (F : Pat) (v : VM.VM) : Pat :=
  deltaLmxm n F (patOf v.m) (patOf v.dp) (patOf v.dm)

theorem foldl_eadd (n : Nat) (F : Pat) (rest : List VM.VM) :
    ∀ (init : Pat) (i j : Nat),
      (rest.foldl (fun f t => eadd f (lm n F t)) init) i j = true ↔
        init i j = true ∨ ∃ t ∈ rest, lm n F t i j = true := by
  induction rest with
  | nil => intro init i j; simp
  | cons a t ih =>
    intro init i j
    rw [List.foldl_cons, ih]
    simp [eadd, or_assoc]

/-- `TraversalMatrix::delta_lmxm_into` (cond_traverse.rs:86-112): one `delta_lmxm`, or for the union
the first type into `f` and each further type, from a copy of the original frontier, OR-ed in. -/
noncomputable def deltaInto (n : Nat) (F : Pat) : TMat → Pat
  | .bool v => lm n F v
  | .u64 v => lm n F v
  | .union [] => F
  | .union (v :: rest) => rest.foldl (fun f t => eadd f (lm n F t)) (lm n F v)

/-- The logical matrix of a traversal matrix: the union of its types' `(m ∖ dm) ∪ dp`. -/
def effT : TMat → (Nat × Nat) → Prop
  | .bool v => VM.eff v
  | .u64 v => VM.eff v
  | .union vs => fun p => ∃ v ∈ vs, VM.eff v p

/-- Layer invariants of every layer set; a union is never empty (`relationship_tensors`). -/
def InvT : TMat → Prop
  | .bool v => VM.Inv v
  | .u64 v => VM.Inv v
  | .union vs => vs ≠ [] ∧ ∀ v ∈ vs, VM.Inv v

/-- **`delta_lmxm_into` is `F * A`** for `A` the (union of the) logical matrix(es): `(i,j)` is
stored iff some frontier entry `(i,k)`, `k < n`, steps to `j`. For several types
`F*A1 + F*A2 = F*(A1 + A2)` on the pattern, so a pair joined by several types is stored once, and a
deletion in one type never masks a path through another (each product sees only its own layers). -/
theorem deltaInto_eff (n : Nat) (F : Pat) (T : TMat) (h : InvT T) (i j : Nat) :
    deltaInto n F T i j = true ↔ ∃ k, k < n ∧ F i k = true ∧ effT T (k, j) := by
  cases T with
  | bool v => exact deltaLmxm_eff n F v h i j
  | u64 v => exact deltaLmxm_eff n F v h i j
  | union vs =>
    obtain ⟨hne, hinv⟩ := h
    cases vs with
    | nil => exact absurd rfl hne
    | cons v rest =>
      unfold deltaInto
      rw [foldl_eadd]
      have hs : ∀ w ∈ v :: rest, lm n F w i j = true ↔ ∃ k, k < n ∧ F i k = true ∧ VM.eff w (k, j) :=
        fun w hw => deltaLmxm_eff n F w (hinv w hw) i j
      constructor
      · rintro (h | ⟨w, hw, h⟩)
        · obtain ⟨k, hk, h1, h2⟩ := (hs v (by simp)).1 h
          exact ⟨k, hk, h1, v, by simp, h2⟩
        · obtain ⟨k, hk, h1, h2⟩ := (hs w (List.mem_cons_of_mem _ hw)).1 h
          exact ⟨k, hk, h1, w, List.mem_cons_of_mem _ hw, h2⟩
      · rintro ⟨k, hk, h1, w, hw, h2⟩
        rcases List.mem_cons.1 hw with rfl | hw
        · exact Or.inl ((hs w (by simp)).2 ⟨k, hk, h1, h2⟩)
        · exact Or.inr ⟨w, hw, (hs w (List.mem_cons_of_mem _ hw)).2 ⟨k, hk, h1, h2⟩⟩

/-- A union of one type is that type's `delta_lmxm`. -/
theorem deltaInto_union_singleton (n : Nat) (F : Pat) (v : VM.VM) :
    deltaInto n F (.union [v]) = deltaInto n F (.u64 v) := rfl

/-! ## `expand_union_row` -/

/-- `expand_row`'s two blocks over an arbitrary pair stream: forward, then (undirected) the
reverse block over non-loop pairs. -/
def expandRowP (g : Graph) (ps : List (Nat × Nat)) (bidir emit : Bool) (r : RowIn) :
    List (Nat × Nat × Nat) :=
  processPairs g ps false r emit ++
    (if bidir then processPairs g (ps.filter fun p => p.1 != p.2) true r emit else [])

/-- The single-type expansion is the pair stream `pairs g`. -/
theorem expandRow_eq_expandRowP (g : Graph) (bidir emit : Bool) (r : RowIn) :
    expandRow g bidir emit r = expandRowP g (pairs g) bidir emit r := rfl

/-- The per-type iterator contract the model identifies with `Basic.pairs` (a relationship
tensor's structural iterator yields its pairs ascending, once each): `layerStream_spec` for the
layers, the tensor/edge-list correspondence for the content. -/
structure StructIter (S : Graph → List (Nat × Nat)) : Prop where
  sorted : ∀ g, Sorted (S g)
  mem : ∀ g s d, (s, d) ∈ S g ↔ ∃ x ∈ g, x.src = s ∧ x.dst = d

/-- The pairs `expand_union_row` reads after `seek` (`p` = the row range): the merged stream. -/
def unionStream (S : Graph → List (Nat × Nat)) (gs : List Graph) (p : Nat × Nat → Bool) :
    List (Nat × Nat) :=
  UnionStream.drain (UnionStream.meas (UnionStream.seek p (UnionStream.new (gs.map S))) + 1)
    (UnionStream.seek p (UnionStream.new (gs.map S)))

/-- `expand_union_row` (forward block over `fwd`, reverse over `rev`): the single-type blocks
over the union stream; the graph the edges are read from is every type's edges. -/
noncomputable def expandUnionRow (S : Graph → List (Nat × Nat)) (gs : List Graph) (p : Nat × Nat → Bool)
    (bidir emit : Bool) (r : RowIn) : List (Nat × Nat × Nat) :=
  expandRowP gs.flatten (unionStream S gs p) bidir emit r

theorem unionStream_spec {S : Graph → List (Nat × Nat)} (hS : StructIter S) (gs : List Graph)
    (p : Nat × Nat → Bool) :
    Sorted (unionStream S gs p) ∧ (unionStream S gs p).Nodup ∧
      ∀ s d, (s, d) ∈ unionStream S gs p ↔
        (∃ x ∈ gs.flatten, x.src = s ∧ x.dst = d) ∧ p (s, d) = true := by
  have hs : ∀ l ∈ gs.map S, Sorted l := by
    intro l hl
    obtain ⟨g, _, rfl⟩ := List.mem_map.1 hl
    exact hS.sorted g
  obtain ⟨h1, h2, h3⟩ := union_stream (gs.map S) hs p
  refine ⟨h1, h2, fun s d => ?_⟩
  rw [unionStream, h3]
  constructor
  · rintro ⟨⟨l, hl, hk⟩, hp⟩
    obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hl
    obtain ⟨x, hx, hx1, hx2⟩ := (hS.mem g s d).1 hk
    exact ⟨⟨x, List.mem_flatten.2 ⟨g, hg, hx⟩, hx1, hx2⟩, hp⟩
  · rintro ⟨⟨x, hx, hx1, hx2⟩, hp⟩
    obtain ⟨g, hg, hxg⟩ := List.mem_flatten.1 hx
    exact ⟨⟨S g, List.mem_map.2 ⟨g, hg, rfl⟩, (hS.mem g s d).2 ⟨x, hxg, hx1, hx2⟩⟩, hp⟩

/-- A seek that keeps every pair the row's bound endpoints admit changes nothing: the
rows are filtered by `pcond` anyway (`process_pairs`, :977-982). -/
theorem processPairs_seek (g : Graph) (isRev emit : Bool) (r : RowIn) (p : Nat × Nat → Bool) :
    ∀ ps : List (Nat × Nat), (∀ q ∈ ps, pcond r isRev q = true → p q = true) →
      processPairs g (ps.filter p) isRev r emit = processPairs g ps isRev r emit
  | [], _ => rfl
  | q :: ps, h => by
    have ih := processPairs_seek g isRev emit r p ps (fun x hx => h x (List.mem_cons_of_mem _ hx))
    by_cases hp : p q = true
    · simp only [List.filter_cons, hp, if_true, processPairs, List.flatMap_cons] at ih ⊢
      rw [ih]
    · have hc : pcond r isRev q = false := by
        cases hh : pcond r isRev q
        · rfl
        · exact absurd (h q (by simp) hh) hp
      have e : pairRows g isRev r emit q = [] := by
        unfold pairRows; simp [hc]
      simp only [List.filter_cons, hp, if_false, processPairs, List.flatMap_cons, e, List.nil_append] at ih ⊢
      exact ih

/-- **Multi-type, directed, emit mode**: `expand_union_row` produces `(f,t,e)` iff `e` is an edge
`f→t` of one of the pattern's types that satisfies the row's bindings and is not used by a sibling:
the same set the union graph's single-matrix expansion produces (`emit_directed_iff`). -/
theorem union_emit_directed_iff {S : Graph → List (Nat × Nat)} (hS : StructIter S) (gs : List Graph)
    {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandUnionRow S gs (fun _ => true) false true r ↔
      ∃ x ∈ gs.flatten, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t := by
  have hs := (unionStream_spec hS gs (fun _ => true)).2.2
  simp only [expandUnionRow, expandRowP, Bool.false_eq_true, if_false, List.append_nil]
  rw [mem_processPairs (fun _ _ => True) (fun s d => by rw [hs]; simp)]
  simp only [Bool.false_eq_true, if_false]
  constructor
  · rintro ⟨x, hx, h1, h2, h3, h4, _, h5, h6⟩; exact ⟨x, hx, h1, h5, h6, h2, h3, h4⟩
  · rintro ⟨x, hx, h1, h5, h6, h2, h3, h4⟩; exact ⟨x, hx, h1, h2, h3, h4, trivial, h5, h6⟩

/-- **Multi-type, undirected, emit mode**: as `emit_bidir_iff` on the union graph (a self-loop once). -/
theorem union_emit_bidir_iff {S : Graph → List (Nat × Nat)} (hS : StructIter S) (gs : List Graph)
    {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ expandUnionRow S gs (fun _ => true) true true r ↔
      ∃ x ∈ gs.flatten, x.id = e ∧ e ∉ r.used ∧ okBound r.fromId f ∧ okBound r.toId t ∧
        ((x.src = f ∧ x.dst = t) ∨ (x.src ≠ x.dst ∧ x.src = t ∧ x.dst = f)) := by
  have hs := (unionStream_spec hS gs (fun _ => true)).2.2
  have hf : ∀ s d, (s, d) ∈ (unionStream S gs (fun _ => true)).filter (fun q => q.1 != q.2) ↔
      (∃ x ∈ gs.flatten, x.src = s ∧ x.dst = d) ∧ s ≠ d := by
    intro s d; simp [List.mem_filter, hs]
  simp only [expandUnionRow, expandRowP, if_true, List.mem_append]
  rw [mem_processPairs (fun _ _ => True) (fun s d => by rw [hs]; simp),
    mem_processPairs (fun s d => s ≠ d) hf]
  simp only [Bool.false_eq_true, if_false, if_true]
  constructor
  · rintro (⟨x, hx, h1, h2, h3, h4, _, h5⟩ | ⟨x, hx, h1, h2, h3, h4, hne, h5⟩)
    · exact ⟨x, hx, h1, h2, h3, h4, Or.inl h5⟩
    · exact ⟨x, hx, h1, h2, h3, h4, Or.inr ⟨hne, h5⟩⟩
  · rintro ⟨x, hx, h1, h2, h3, h4, (h5 | ⟨hne, h5⟩)⟩
    · exact Or.inl ⟨x, hx, h1, h2, h3, h4, trivial, h5⟩
    · exact Or.inr ⟨x, hx, h1, h2, h3, h4, hne, h5⟩

/-- **Multi-type, directed, collapse mode: one row per matrix pair, no duplicates** (the claim of
the `UnionIter` docs: a pair joined by several types is emitted once). -/
theorem union_collapse_directed_keys_nodup {S : Graph → List (Nat × Nat)} (hS : StructIter S)
    (gs : List Graph) (p : Nat × Nat → Bool) (r : RowIn) :
    ((expandUnionRow S gs p false false r).map fun x => (x.1, x.2.1)).Nodup := by
  simp only [expandUnionRow, expandRowP, Bool.false_eq_true, if_false, List.append_nil]
  exact (unionStream_spec hS gs p).2.1.sublist (collapse_keys_sublist gs.flatten r _)

/-- The per-type *transposed* iterator (`Tensor::matrix_t`): dst-major, `(dst, src)` ascending. -/
structure StructIterT (S : Graph → List (Nat × Nat)) : Prop where
  sorted : ∀ g, Sorted (S g)
  mem : ∀ g d s, (d, s) ∈ S g ↔ ∃ x ∈ g, x.src = s ∧ x.dst = d

/-- A destination-bound expansion reads the transposed union (`build_transposed_union_iter`,
`bwd`), merged the same way and swapped back to `(src, dst)`
(`.map(|(dest, src)| (src, dest))`, :1148-1150): the pair set of the forward union, each pair
once. -/
def transposedPairs (S : Graph → List (Nat × Nat)) (gs : List Graph) (p : Nat × Nat → Bool) :
    List (Nat × Nat) :=
  (unionStream S gs p).map Prod.swap

theorem transposedPairs_spec {S : Graph → List (Nat × Nat)} (hS : StructIterT S) (gs : List Graph)
    (p : Nat × Nat → Bool) :
    (transposedPairs S gs p).Nodup ∧ ∀ s d, (s, d) ∈ transposedPairs S gs p ↔
      (∃ x ∈ gs.flatten, x.src = s ∧ x.dst = d) ∧ p (d, s) = true := by
  have hs : ∀ l ∈ gs.map S, Sorted l := by
    intro l hl
    obtain ⟨g, _, rfl⟩ := List.mem_map.1 hl
    exact hS.sorted g
  obtain ⟨h1, h2, h3⟩ := union_stream (gs.map S) hs p
  have hn : (transposedPairs S gs p).Nodup := by
    unfold transposedPairs
    exact List.Pairwise.map Prod.swap (fun a b (hne : a ≠ b) (h : Prod.swap a = Prod.swap b) =>
      hne (Prod.ext (congrArg Prod.snd h) (congrArg Prod.fst h))) h2
  refine ⟨hn, fun s d => ?_⟩
  unfold transposedPairs unionStream
  rw [List.mem_map]
  constructor
  · rintro ⟨⟨a, b⟩, hab, hsw⟩
    simp only [Prod.swap, Prod.mk.injEq] at hsw
    obtain ⟨rfl, rfl⟩ := hsw
    obtain ⟨⟨l, hl, hk⟩, hp⟩ := (h3 (a, b)).1 hab
    obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hl
    obtain ⟨x, hx, hx1, hx2⟩ := (hS.mem g a b).1 hk
    exact ⟨⟨x, List.mem_flatten.2 ⟨g, hg, hx⟩, hx1, hx2⟩, hp⟩
  · rintro ⟨⟨x, hx, hx1, hx2⟩, hp⟩
    obtain ⟨g, hg, hxg⟩ := List.mem_flatten.1 hx
    refine ⟨(d, s), (h3 (d, s)).2 ⟨⟨S g, List.mem_map.2 ⟨g, hg, rfl⟩, (hS.mem g d s).2 ⟨x, hxg, hx1, hx2⟩⟩, hp⟩, rfl⟩

/-- Destination-bound, directed, emit mode over the transposed union: the same edges. -/
theorem union_bwd_emit_directed_iff {S : Graph → List (Nat × Nat)} (hS : StructIterT S)
    (gs : List Graph) {r : RowIn} {f t e : Nat} :
    (f, t, e) ∈ processPairs gs.flatten (transposedPairs S gs (fun _ => true)) false r true ↔
      ∃ x ∈ gs.flatten, x.id = e ∧ x.src = f ∧ x.dst = t ∧ e ∉ r.used ∧
        okBound r.fromId f ∧ okBound r.toId t := by
  have hs := (transposedPairs_spec hS gs (fun _ => true)).2
  rw [mem_processPairs (fun _ _ => True) (fun s d => by rw [hs]; simp)]
  simp only [Bool.false_eq_true, if_false]
  constructor
  · rintro ⟨x, hx, h1, h2, h3, h4, _, h5, h6⟩; exact ⟨x, hx, h1, h5, h6, h2, h3, h4⟩
  · rintro ⟨x, hx, h1, h5, h6, h2, h3, h4⟩; exact ⟨x, hx, h1, h2, h3, h4, trivial, h5, h6⟩

/-- The `state.union.is_some()` branch of `expand_row` (:963-990): a multi-type state expands over
the union iterators, anything else over the single-type ones (whose model is `expandRow`). -/
def expandRowDispatch {α : Type} (st : CTGlue.CtState) (unionPath singlePath : α) : α :=
  if st.multi then unionPath else singlePath

theorem expandRowDispatch_spec {α : Type} (st : CTGlue.CtState) (u s : α) :
    (st.multi = true → expandRowDispatch st u s = u) ∧
    (st.multi = false → expandRowDispatch st u s = s) := by
  unfold expandRowDispatch
  exact ⟨fun h => by simp [h], fun h => by simp [h]⟩

end OpsTraverse.UnionGlue
