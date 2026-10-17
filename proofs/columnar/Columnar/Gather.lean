import Columnar.Emitter
/-
`GatherItem` (graph/src/runtime/ops/batched_result_emitter.rs:40-430): the typed lanes the
emitter packs one item per output row into, and how they become columns — proved ONCE
generically, then instantiated per lane.

Generic spec (`Spec`): a binding `b` determines an ordered list of output columns `cols b`
and, for each column index `i`, the cell an item contributes there (`val b i x`). The spec
emitter keeps one list per column, appends every item's cell to every list, and `finish`
sets column `cols b [i]` to list `i`, in order (`setCol`: later writes win).

An implementation is `Lawful` when an abstraction `abs b : Lanes → List (List Cell)` maps
`new_lanes` to all-empty lists, commutes `push_into` with the spec append, and `finish` to
the spec finish. `gather_spec`: for a lawful lane, `finish (push* xs (new_lanes))` sets each
column `cols b [i]` to exactly `xs.map (val b i)` — row `k` of the output carries item `k`'s
cell. Each of the seven Rust impls is then shown lawful by three one-step lemmas.

| here | there |
| --- | --- |
| `GI`, `Lawful`, `gather_spec` | `trait GatherItem { new_lanes; push_into; finish }` (:40-75) |
| `giNode`, `giRel` | `impl GatherItem for NodeId` (:80-108) / `RelationshipId` (:110-139) |
| `giScoredN`, `giScoredR` | `impl for (NodeId, f64)` (:150-190) / `(RelationshipId, f64)` (:192-230) |
| `giEdge` | `impl for (NodeId, NodeId, RelationshipId)` with `EdgeEndpoints` (:257-305) |
| `giVarLen` | `impl for (NodeId, NodeId, Option<Value>)` with `VarLenEndpoints` (:339-400) |
| `giValue` | `impl for Value` (:404-430) |
| `rowIterList` | `RowIter::{one, spread, many}` (:458-470) and its `next` (:478) |
-/
namespace Columnar.Gather

/-- The cell kinds the lanes produce (`Column::NodeIds/RelIds/Floats/Values`). -/
inductive Cell (F W : Type) where
  | node (n : Nat)
  | rel (n : Nat)
  | float (f : F)
  | value (w : W)
  | null

abbrev Out (C : Type) := Nat → Option (List C)

/-- `Batch::set_column(alias, column)` on the output batch (columns as cell lists). -/
def setCol {C : Type} (o : Out C) (c : Nat) (vs : List C) : Out C := fun j => if j = c then some vs else o j

structure GI (X B L C : Type) where
  newL : B → Nat → L
  push : X → B → L → L
  finish : L → B → Out C → Out C

def specFinish {C : Type} (cols : List Nat) (ls : List (List C)) (o : Out C) : Out C :=
  (cols.zip ls).foldl (fun o p => setCol o p.1 p.2) o

structure LawfulWith {X B L C : Type} (g : GI X B L C) (cols : B → List Nat) (val : B → Nat → X → C)
    (abs : B → L → List (List C)) : Prop where
  new_ok : ∀ b cap, abs b (g.newL b cap) = (cols b).map fun _ => []
  push_ok : ∀ x b l, abs b (g.push x b l) = (abs b l).mapIdx fun i lane => lane ++ [val b i x]
  finish_ok : ∀ l b o, g.finish l b o = specFinish (cols b) (abs b l) o

def Lawful {X B L C : Type} (g : GI X B L C) (cols : B → List Nat) (val : B → Nat → X → C) : Prop :=
  ∃ abs, LawfulWith g cols val abs

theorem mapIdx_mapIdx {α β γ : Type} (l : List α) (f : Nat → α → β) (g : Nat → β → γ) :
    (l.mapIdx f).mapIdx g = l.mapIdx fun i a => g i (f i a) := by
  apply List.ext_getElem <;> simp

theorem map_eq_mapIdx {α β : Type} (l : List α) (c : β) : l.map (fun _ => c) = l.mapIdx fun _ _ => c := by
  apply List.ext_getElem <;> simp

/-- **Generic gather theorem**: pushing `xs` into fresh lanes and finishing sets column
`cols b [i]` (in column order) to `xs.map (val b i)`. -/
theorem gather_spec {X B L C : Type} (g : GI X B L C) (cols : B → List Nat) (val : B → Nat → X → C)
    (hl : Lawful g cols val) (b : B) (cap : Nat) (xs : List X) (o : Out C) :
    g.finish (xs.foldl (fun l x => g.push x b l) (g.newL b cap)) b o =
      specFinish (cols b) ((cols b).mapIdx fun i _ => xs.map (val b i)) o := by
  obtain ⟨abs, h⟩ := hl
  rw [h.finish_ok]
  congr 1
  suffices H : ∀ (pre : List X) (l : L), abs b l = (cols b).mapIdx (fun i _ => pre.map (val b i)) →
      abs b (xs.foldl (fun l x => g.push x b l) l) = (cols b).mapIdx (fun i _ => (pre ++ xs).map (val b i)) by
    have := H [] (g.newL b cap) (by rw [h.new_ok, map_eq_mapIdx]; simp)
    simpa using this
  induction xs with
  | nil => intro pre l hl; simpa using hl
  | cons x xs ih =>
    intro pre l hl
    simp only [List.foldl_cons]
    have := ih (pre ++ [x]) (g.push x b l) (by rw [h.push_ok, hl, mapIdx_mapIdx]; simp)
    simpa using this

/-- Reading back: with distinct column ids, column `cols b [i]`'s row `k` is item `k`'s cell. -/
theorem specFinish_get {C : Type} (cols : List Nat) (ls : List (List C)) (o : Out C) (hn : cols.Nodup)
    (hl : ls.length = cols.length) (i : Nat) (hi : i < cols.length) :
    specFinish cols ls o (cols[i]) = some (ls[i]'(by omega)) := by
  induction cols generalizing ls o i with
  | nil => simp at hi
  | cons c cs ih =>
    cases ls with
    | nil => simp at hl
    | cons v vs =>
      simp only [specFinish, List.zip_cons_cons, List.foldl_cons] at ih ⊢
      have hn' := List.nodup_cons.mp hn
      cases i with
      | zero =>
        simp only [List.getElem_cons_zero]
        -- later columns differ from `c`, so they leave it untouched
        have keep : ∀ (ps : List (Nat × List C)) (o' : Out C), (∀ p ∈ ps, p.1 ≠ c) →
            (ps.foldl (fun o p => setCol o p.1 p.2) o') c = o' c := by
          intro ps; induction ps with
          | nil => intro _ _; rfl
          | cons p ps ihp =>
            intro o' hp; simp only [List.foldl_cons]
            rw [ihp _ (fun q hq => hp q (List.mem_cons_of_mem _ hq))]
            simp only [setCol]
            rw [if_neg (fun he => hp p (List.mem_cons_self ..) he.symm)]
        rw [keep _ _ (fun p hp => fun he => hn'.1 (he ▸ (List.of_mem_zip hp).1))]
        simp [setCol]
      | succ i =>
        simp only [List.getElem_cons_succ]
        exact ih vs _ hn'.2 (by simpa using hl) i (by simpa using hi)

/-! ## The seven lanes -/

section Lanes
variable {F W : Type}

/-- `NodeId` / `RelationshipId` (binding `Option<u32>`): one id lane; finish binds it when an
alias is given (`new_without_alias` binds nothing). -/
def giIds (mk : Nat → Cell F W) : GI Nat (Option Nat) (List Nat) (Cell F W) where
  newL _ _ := []
  push x _ l := l ++ [x]
  finish l b o := match b with
    | some a => setCol o a (l.map mk)
    | none => o

def giNode : GI Nat (Option Nat) (List Nat) (Cell F W) := giIds .node
def giRel : GI Nat (Option Nat) (List Nat) (Cell F W) := giIds .rel

def idsCols (b : Option Nat) : List Nat := b.toList

theorem giIds_lawful (mk : Nat → Cell F W) : Lawful (giIds mk) idsCols (fun _ _ x => mk x) :=
  ⟨fun b l => match b with | some _ => [l.map mk] | none => [], {
    new_ok b _ := by cases b <;> rfl
    push_ok x b l := by cases b <;> simp [giIds]
    finish_ok l b o := by cases b <;> simp [giIds, specFinish, idsCols] }⟩

theorem giNode_lawful : Lawful (giNode : GI Nat _ _ (Cell F W)) idsCols (fun _ _ x => .node x) := giIds_lawful _
theorem giRel_lawful : Lawful (giRel : GI Nat _ _ (Cell F W)) idsCols (fun _ _ x => .rel x) := giIds_lawful _

/-- `(NodeId, f64)` / `(RelationshipId, f64)` with `ScoredColumn { id, score }`: ids always,
scores only when a score alias is bound. -/
structure Scored where
  id : Nat
  score : Option Nat

def giScored (mk : Nat → Cell F W) : GI (Nat × F) Scored (List Nat × List F) (Cell F W) where
  newL _ _ := ([], [])
  push x b l := (l.1 ++ [x.1], if b.score.isSome then l.2 ++ [x.2] else l.2)
  finish l b o :=
    let o1 := setCol o b.id (l.1.map mk)
    match b.score with
    | some s => setCol o1 s (l.2.map .float)
    | none => o1

def giScoredN : GI (Nat × F) Scored (List Nat × List F) (Cell F W) := giScored .node
def giScoredR : GI (Nat × F) Scored (List Nat × List F) (Cell F W) := giScored .rel

def scoredCols (b : Scored) : List Nat := b.id :: b.score.toList
def scoredVal (mk : Nat → Cell F W) (_ : Scored) (i : Nat) (x : Nat × F) : Cell F W :=
  if i = 0 then mk x.1 else .float x.2

theorem giScored_lawful (mk : Nat → Cell F W) : Lawful (giScored mk) scoredCols (scoredVal mk) :=
  ⟨fun b l => l.1.map mk :: match b.score with | some _ => [l.2.map .float] | none => [], {
    new_ok b _ := by cases h : b.score <;> simp [giScored, scoredCols, h]
    push_ok x b l := by cases h : b.score <;> simp [giScored, h, scoredVal]
    finish_ok l b o := by cases h : b.score <;> simp [giScored, specFinish, scoredCols, h] }⟩

theorem giScoredN_lawful : Lawful (giScoredN : GI _ _ _ (Cell F W)) scoredCols (scoredVal .node) := giScored_lawful _
theorem giScoredR_lawful : Lawful (giScoredR : GI _ _ _ (Cell F W)) scoredCols (scoredVal .rel) := giScored_lawful _

/-- `(NodeId, NodeId, RelationshipId)` with `EdgeEndpoints { from, to, edge, transposed }`:
`transposed` swaps the endpoints; `to = None` (self-loop pattern) skips the `to` lane. -/
structure Ends where
  fromA : Nat
  toA : Option Nat
  edge : Nat
  transposed : Bool

structure EdgeLanes where
  froms : List Nat
  tos : List Nat
  edges : List Nat

def giEdge : GI (Nat × Nat × Nat) Ends EdgeLanes (Cell F W) where
  newL _ _ := ⟨[], [], []⟩
  push x b l :=
    let (f, t) := if b.transposed then (x.2.1, x.1) else (x.1, x.2.1)
    ⟨l.froms ++ [f], if b.toA.isSome then l.tos ++ [t] else l.tos, l.edges ++ [x.2.2]⟩
  finish l b o :=
    let o1 := setCol o b.fromA (l.froms.map .node)
    let o2 := match b.toA with | some t => setCol o1 t (l.tos.map .node) | none => o1
    setCol o2 b.edge (l.edges.map .rel)

def edgeCols (b : Ends) : List Nat := b.fromA :: (b.toA.toList ++ [b.edge])
def edgeVal (b : Ends) (i : Nat) (x : Nat × Nat × Nat) : Cell F W :=
  let (f, t) := if b.transposed then (x.2.1, x.1) else (x.1, x.2.1)
  if i = 0 then .node f
  else if b.toA.isSome ∧ i = 1 then .node t
  else .rel x.2.2

theorem giEdge_lawful : Lawful (giEdge : GI _ _ _ (Cell F W)) edgeCols edgeVal :=
  ⟨fun b l => l.froms.map .node :: ((match b.toA with | some _ => [l.tos.map .node] | none => []) ++
    [l.edges.map .rel]), {
    new_ok b _ := by cases h : b.toA <;> simp [giEdge, edgeCols, h]
    push_ok x b l := by
      cases h : b.toA <;> cases ht : b.transposed <;> simp [giEdge, h, ht, edgeVal]
    finish_ok l b o := by cases h : b.toA <;> simp [giEdge, specFinish, edgeCols, h] }⟩

/-- `(NodeId, NodeId, Option<Value>)` with `VarLenEndpoints { from, to, distinct, path, path_copy }`:
a shared endpoint alias binds one column holding the `to` node; the path lane (filled with
`Null` for a missing path) feeds the copy column, then the path column. -/
structure VLEnds where
  fromA : Nat
  toA : Nat
  distinct : Bool
  path : Option Nat
  copy : Option Nat

structure VLLanes (W : Type) where
  froms : List Nat
  tos : List Nat
  paths : List (Option W)

def vlCell (p : Option W) : Cell F W := match p with | some w => .value w | none => .null

def wantsPath (b : VLEnds) : Bool := b.path.isSome || b.copy.isSome

def giVarLen : GI (Nat × Nat × Option W) VLEnds (VLLanes W) (Cell F W) where
  newL _ _ := ⟨[], [], []⟩
  push x b l :=
    let l1 : VLLanes W := if b.distinct then { l with froms := l.froms ++ [x.1], tos := l.tos ++ [x.2.1] }
      else { l with froms := l.froms ++ [x.2.1] }
    if wantsPath b then { l1 with paths := l1.paths ++ [x.2.2] } else l1
  finish l b o :=
    let cell : Option W → Cell F W := vlCell
    let o1 := setCol o b.fromA (l.froms.map .node)
    let o2 := if b.distinct then setCol o1 b.toA (l.tos.map .node) else o1
    let o3 := match b.copy with | some c => setCol o2 c (l.paths.map cell) | none => o2
    match b.path with | some p => setCol o3 p (l.paths.map cell) | none => o3

def vlCols (b : VLEnds) : List Nat :=
  b.fromA :: ((if b.distinct then [b.toA] else []) ++ b.copy.toList ++ b.path.toList)

def vlVal (b : VLEnds) (i : Nat) (x : Nat × Nat × Option W) : Cell F W :=
  if i = 0 then .node (if b.distinct then x.1 else x.2.1)
  else if b.distinct ∧ i = 1 then .node x.2.1
  else vlCell x.2.2

theorem giVarLen_lawful : Lawful (giVarLen : GI _ _ _ (Cell F W)) vlCols vlVal :=
  ⟨fun b l => l.froms.map .node :: ((if b.distinct then [l.tos.map .node] else []) ++
    ((match b.copy with | some _ => [l.paths.map vlCell] | none => []) ++
     (match b.path with | some _ => [l.paths.map vlCell] | none => []))), {
    new_ok b _ := by
      cases hd : b.distinct <;> cases hc : b.copy <;> cases hp : b.path <;> simp [giVarLen, vlCols, hd, hc, hp]
    push_ok x b l := by
      cases hd : b.distinct <;> cases hc : b.copy <;> cases hp : b.path <;>
        simp [giVarLen, wantsPath, hd, hc, hp, vlVal, vlCell]
    finish_ok l b o := by
      cases hd : b.distinct <;> cases hc : b.copy <;> cases hp : b.path <;>
        simp [giVarLen, specFinish, vlCols, hd, hc, hp, vlCell] }⟩

/-- `Value` (binding: the alias): one value lane. -/
def giValue : GI W Nat (List W) (Cell F W) where
  newL _ _ := []
  push x _ l := l ++ [x]
  finish l b o := setCol o b (l.map .value)

theorem giValue_lawful : Lawful (giValue : GI W Nat _ (Cell F W)) (fun b => [b]) (fun _ _ x => .value x) :=
  ⟨fun _ l => [l.map .value], {
    new_ok _ _ := rfl
    push_ok x b l := by simp [giValue]
    finish_ok l b o := by simp [giValue, specFinish] }⟩

end Lanes

/-- Instances: the output column of every lane, row by row (one corollary per impl). -/
theorem gather_node {F W : Type} (a cap : Nat) (xs : List Nat) (o : Out (Cell F W)) :
    let g : GI Nat (Option Nat) (List Nat) (Cell F W) := giNode
    g.finish (xs.foldl (fun l x => g.push x (some a) l) (g.newL (some a) cap)) (some a) o a =
      some (xs.map .node) := by
  intro g
  rw [gather_spec _ _ _ giNode_lawful]; simp [specFinish, idsCols, setCol]

theorem gather_value {F W : Type} (a cap : Nat) (xs : List W) (o : Out (Cell F W)) :
    let g : GI W Nat (List W) (Cell F W) := giValue
    g.finish (xs.foldl (fun l x => g.push x a l) (g.newL a cap)) a o a =
      some (xs.map .value) := by
  intro g
  rw [gather_spec _ _ _ giValue_lawful]; simp [specFinish, setCol]

/-! ## `RowIter` (:458-485) -/

inductive RowIter (I : Type) where
  | one (x : Option I)
  | spread (xs : List I)
  | many (xs : List I)

def rowIterNext {I : Type} : RowIter I → Option I × RowIter I
  | .one x => (x, .one none)
  | .spread [] => (none, .spread [])
  | .spread (x :: xs) => (some x, .spread xs)
  | .many [] => (none, .many [])
  | .many (x :: xs) => (some x, .many xs)

def rowIterList {I : Type} : RowIter I → List I
  | .one x => x.toList
  | .spread xs => xs
  | .many xs => xs

def drainIt {I : Type} : Nat → RowIter I → List I
  | 0, _ => []
  | n + 1, it => match rowIterNext it with
    | (some x, it') => x :: drainIt n it'
    | (none, _) => []

/-- `one(x)` yields exactly `x`; `spread`/`many` yield their items in order. -/
theorem rowIter_spec {I : Type} (x : I) (xs : List I) :
    drainIt 2 (.one (some x)) = [x] ∧
    drainIt (xs.length + 1) (.spread xs) = xs ∧ drainIt (xs.length + 1) (.many xs) = xs := by
  refine ⟨rfl, ?_, ?_⟩
  · induction xs with
    | nil => rfl
    | cons y ys ih => simp [drainIt, rowIterNext, ih]
  · induction xs with
    | nil => rfl
    | cons y ys ih => simp [drainIt, rowIterNext, ih]

/-! ## `BatchedResultEmitter::new` (:770), `new_without_alias` (:780), `batch` (:756) -/

structure Em (B : Type) where
  binding : Option Nat
  batch : Option B
  ceiling : Nat

/-- `with_binding(binding, record_cap)`: no seeded batch, ceiling from the cap. -/
def emWithBinding {B : Type} (binding : Option Nat) (cap : Option Nat) : Em B :=
  ⟨binding, none, Columnar.withBinding cap⟩

def emNew {B : Type} (alias : Nat) (cap : Option Nat) : Em B := emWithBinding (some alias) cap
def emNewWithoutAlias {B : Type} (cap : Option Nat) : Em B := emWithBinding none cap
def emBatch {B : Type} (e : Em B) : Option B := e.batch

theorem emNew_spec {B : Type} (a : Nat) (cap : Option Nat) :
    (emNew a cap : Em B).binding = some a ∧ (emNew a cap : Em B).batch = none ∧
    1 ≤ (emNew a cap : Em B).ceiling ∧ (emNew a cap : Em B).ceiling ≤ Columnar.BATCH_SIZE ∧
    (emNewWithoutAlias cap : Em B).binding = none ∧ emBatch (emNew a cap : Em B) = none :=
  ⟨rfl, rfl, (Columnar.withBinding_le_batch cap).1, (Columnar.withBinding_le_batch cap).2, rfl, rfl⟩

end Columnar.Gather
