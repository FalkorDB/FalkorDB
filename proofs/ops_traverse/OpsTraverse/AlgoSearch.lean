import OpsTraverse.AlgoPaths
import OpsTraverse.RecordFound
import OpsTraverse.RunAlgo
import OpsTraverse.BfsFind
import OpsTraverse.Dijkstra
import OpsTraverse.DijkstraNat
import OpsTraverse.Enum
import OpsTraverse.EnumSpec
/-
# The three path searches of `algo_procedures.rs`, instantiated at the Rust data

`BfsFind.bfsFindBound`, `Dijkstra.dijkstra` and `Enum.enumerate` read the graph through a `Step`; here
the step is `node_successors` (`AlgoPaths.successors` over the node's relationships
`g.get_node_relationships_by_type(node, rel_types, direction)`, `rels node`), so a walk over it is a
walk over edges of the requested types that leave each node in the requested direction.
`record_found_path` is `RecordFound.record` over `Found` triples. `RunAlgo.runAlgo_ok_good` then gives
the end-to-end statement for `run_path_algo`.

Exhausting the fuel of the loops (the Rust has none; a finished run needs none) is reported as an error,
standing for the query timeout / memory cap that bounds the real loops.
-/
namespace OpsTraverse.AlgoSearch
open OpsTraverse.Search OpsTraverse.AlgoPaths OpsTraverse.RecordFound OpsTraverse.RunAlgo
  OpsTraverse.BfsFind OpsTraverse.Dijkstra OpsTraverse.Enum OpsTraverse.EnumSpec

variable {F : Type}

/-- An edge as the searches keep it: `(edge id, edge src, edge dst)`, in its original direction. -/
abbrev Edge := Nat × Nat × Nat

/-- `node_successors` as a `Step`: far endpoint and the edge, in iterator order. -/
def stepOf (dir : Dir) (rels : Nat → List (Nat × Nat × Nat)) : Step Edge :=
  fun u => (successors dir u (rels u)).map (fun r => (r.2.2.2, (r.1, r.2.1, r.2.2.1)))

/-- **PROVEN** (`node_successors` as a `Step`): `v` is reached by edge `(id, s, d)` from `u` exactly when
`(s, d, id)` is one of `u`'s relationships (of the requested types) and `far_endpoint` accepts it in
the requested direction; so every walk edge respects relTypes (through `rels`) and direction. -/
theorem mem_stepOf (dir : Dir) (rels : Nat → List (Nat × Nat × Nat)) (u v id s d : Nat) :
    (v, (id, s, d)) ∈ stepOf dir rels u ↔ (s, d, id) ∈ rels u ∧ farEndpoint u s d dir = some v := by
  unfold stepOf
  rw [List.mem_map]
  constructor
  · rintro ⟨⟨id', s', d', far⟩, hm, he⟩
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl, rfl⟩ := he
    exact (mem_successors dir u (rels u) id' s' d' far).mp hm
  · rintro h
    exact ⟨(id, s, d, v), (mem_successors dir u (rels u) id s d v).mpr h, rfl⟩

/-- Every edge of a walk over `stepOf` is a relationship of the node it leaves, accepted by
`far_endpoint` in the configured direction. -/
theorem walk_edges (dir : Dir) (rels : Nat → List (Nat × Nat × Nat)) :
    ∀ (es : List (Nat × Edge)) (u w : Nat), WalkTo (stepOf dir rels) u es w →
      ∀ i (hi : i < es.length), ∃ prev, (es[i].2.2.1, es[i].2.2.2, es[i].2.1) ∈ rels prev ∧
        farEndpoint prev es[i].2.2.1 es[i].2.2.2 dir = some es[i].1 := by
  intro es
  induction es with
  | nil => intro u w _ i hi; simp at hi
  | cons x r ih =>
    intro u w h i hi
    obtain ⟨hx, hw⟩ := h
    cases i with
    | zero =>
      obtain ⟨far, id, s, d⟩ := x
      have := (mem_stepOf dir rels u far id s d).mp hx
      exact ⟨u, by simpa using this.1, by simpa using this.2⟩
    | succ i => exact ih x.1 w hw i (by simpa using hi)

/-- The `f64` operations the searches use. -/
structure Fl (F : Type) where
  zero : F
  one : F
  add : F → F → F
  fin : F → Bool
  gt : F → F → Bool
  lt : F → F → Bool
  pcmp : F → F → Option Ordering
  ofInt : Int → F
  inf : F
  le : F → F → Bool

def Fl.ops (o : Fl F) : Ops F := ⟨o.zero, o.add, o.fin, o.gt⟩
def Fl.fo (o : Fl F) : FO F := ⟨o.zero, o.add, o.lt, o.fin⟩

/-- Edge attribute reads: `lookup id prop`. -/
structure Attrs (F : Type) where
  lookup : Nat → String → Option (AttrV F)

def wattr (o : Fl F) (cfg : Cfg F) (a : Attrs F) (e : Edge) : F :=
  edgeAttr o.ofInt cfg.weightProp (a.lookup e.1) o.one

/-- Cost used by `bfs_find_bound` and `enumerate_paths` (default `1.0`). -/
def cattrE (o : Fl F) (cfg : Cfg F) (a : Attrs F) (e : Edge) : F :=
  edgeAttr o.ofInt cfg.costProp (a.lookup e.1) o.one

/-- Cost used by `dijkstra_single_path`: default `1.0` when `costProp` is set, `0.0` otherwise. -/
def cattrD (o : Fl F) (cfg : Cfg F) (a : Attrs F) (e : Edge) : F :=
  edgeAttr o.ofInt cfg.costProp (a.lookup e.1) (if cfg.costProp.isSome then o.one else o.zero)

def dirOf (cfg : Cfg F) := cfg.dir

/-- The step every search walks. -/
def stepC (cfg : Cfg F) (rels : Nat → List (Nat × Nat × Nat)) : Step Edge := stepOf cfg.dir rels

/-- `bfs_find_bound`. -/
def bfsBoundR (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (t : Nat) :
    Option (F × F) :=
  bfsFindBound (stepC cfg rels) cfg.source t cfg.maxLen (wattr o cfg a) (cattrE o cfg a) o.add o.zero

/-- `dijkstra_single_path` (heap = `popMax` under `DijkstraItem::cmp`). -/
def dijR (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (t fuel : Nat) :
    Option (Option (List (Nat × Edge) × F × F)) :=
  dijkstra o.fo (popMax (itemCmp o.pcmp)) (stepC cfg rels) cfg.source t (wattr o cfg a) (cattrD o cfg a) fuel

/-- The `enumerate_paths` configuration. -/
def ecOf (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) : EC Edge F :=
  { o := o.ops, succ := stepC cfg rels, source := cfg.source, target := cfg.target, maxLen := cfg.maxLen,
    maxCost := cfg.maxCost, wf := wattr o cfg a, cf := cattrE o cfg a }

/-- A found path as `AlgoPaths.Found` (edges in original direction). -/
def toFound (x : Fnd Edge F) : Found Nat F := ⟨x.1.map (·.2), x.2.1, x.2.2⟩

/-- `cmp_found_path` on found paths. -/
def cmpF (o : Fl F) (a b : Fnd Edge F) : Ordering := cmpFound o.pcmp (toFound a) (toFound b)

/-- `record_found_path` as the searches' recorder. -/
def recR (o : Fl F) (cfg : Cfg F) : List (Fnd Edge F) → Fnd Edge F → F → List (Fnd Edge F) × F :=
  fun r f b => record (cmpF o) (fun x => x.2.1) o.gt o.lt cfg.pathCount.toNat r f b

/-- `enumerate_paths` with `record_found_path`. -/
def enumR (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (b0 : F) (fuel : Nat) :
    Option (List (Fnd Edge F) × F) :=
  enumerate (ecOf o cfg a rels) (recR o cfg) [] b0 fuel

theorem record_members {α : Type} (cmp : α → α → Ordering) (w : α → F) (gtF ltF : F → F → Bool) (pcount : Nat)
    (rs : List α) (found : α) (bound : F) :
    ∀ x ∈ (record cmp w gtF ltF pcount rs found bound).1, x = found ∨ x ∈ rs := by
  intro x hx
  unfold record at hx
  split at hx
  · split at hx
    · simp at hx; exact Or.inl hx
    · exact Or.inr hx
  · split at hx
    · next best =>
      split at hx
      · exact Or.inr hx
      · split at hx
        · simp at hx; exact Or.inl hx
        · rcases List.mem_append.mp hx with h | h
          · exact Or.inr h
          · simp at h; exact Or.inl h
    · simp at hx; exact Or.inl hx
  · split at hx
    · exact Or.inr hx
    · have hp := ins_perm (cmp := cmp) found (if rs.length == pcount then rs.dropLast else rs)
      have := hp.subset hx
      rcases List.mem_cons.mp this with h | h
      · exact Or.inl h
      · right
        split at h
        · exact List.dropLast_subset _ h
        · exact h

/-! ## What each search guarantees, at the Rust data -/

/-- **PROVEN** (`bfs_find_bound`): `bfsFind_spec` at `node_successors`: `None` exactly when no walk of
`1..max_len` edges leads to a target other than the source; otherwise the pair is the left folds of
`weightProp` / `costProp` (default `1.0`) over a shortest walk, simple, within `max_len`. -/
theorem bfsBound_spec (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (t : Nat) :
    (bfsBoundR o cfg a rels t = none ↔
      ¬ (t ≠ cfg.source ∧ ∃ k, 1 ≤ k ∧ k ≤ cfg.maxLen ∧ R (stepC cfg rels) cfg.source k t)) ∧
    (∀ w c, bfsBoundR o cfg a rels t = some (w, c) →
      ∃ es, WalkTo (stepC cfg rels) cfg.source es t ∧ 1 ≤ es.length ∧ es.length ≤ cfg.maxLen ∧
        (∀ es', WalkTo (stepC cfg rels) cfg.source es' t → es.length ≤ es'.length) ∧
        (nodesOf cfg.source es).Nodup ∧ w = fsum o.add o.zero (wattr o cfg a) es ∧
        c = fsum o.add o.zero (cattrE o cfg a) es) :=
  bfsFind_spec (stepC cfg rels) cfg.source t cfg.maxLen (wattr o cfg a) (cattrE o cfg a) o.add o.zero

/-- **PROVEN** (`dijkstra_single_path`, `f64` model): a returned path is a walk over `node_successors`
edges, visits distinct nodes, its cost is the left fold of `costProp` (default `1.0`, or `0.0` without
`costProp`) in path order, and its weight is finite. Optimality over exact weights:
`DijkstraNat.dijkstraNat_spec`. -/
theorem dijR_valid (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (t fuel : Nat)
    (es : List (Nat × Edge)) (w c : F) (h : dijR o cfg a rels t fuel = some (some (es, w, c))) :
    t ≠ cfg.source ∧ WalkTo (stepC cfg rels) cfg.source es t ∧ (nodesOf cfg.source es).Nodup ∧
      c = fsum o.add o.zero (cattrD o cfg a) es ∧ o.fin w = true := by
  have := dijkstra_valid o.fo (popMax (itemCmp o.pcmp)) (popOk_popMax _) (stepC cfg rels) cfg.source t
    (wattr o cfg a) (cattrD o cfg a) fuel es w c h
  exact this

/-- What a reported path is. -/
def Good (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (x : Fnd Edge F) : Prop :=
  (∃ e, WalkTo (stepC cfg rels) cfg.source x.1 e) ∧ (nodesOf cfg.source x.1).Nodup ∧ o.fin x.2.1 = true ∧
    (x.2.2 = fsum o.add o.zero (cattrE o cfg a) x.1 ∨ x.2.2 = fsum o.add o.zero (cattrD o cfg a) x.1)

/-- **PROVEN** (`enumerate_paths`, soundness at the Rust data): every path left in the result set is a
non-empty simple walk over `node_successors` edges from the source, within `max_len`, with finite
weight, weight/cost the left folds over its edges, cost within `maxCost`. -/
theorem enumR_sound (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (b0 : F)
    (fuel : Nat) (r : List (Fnd Edge F)) (b : F) (h : enumR o cfg a rels b0 fuel = some (r, b)) :
    ∀ x ∈ r, (∃ e, WalkTo (stepC cfg rels) cfg.source x.1 e) ∧ x.1 ≠ [] ∧ (nodesOf cfg.source x.1).Nodup ∧
      x.1.length ≤ cfg.maxLen ∧ x.2.1 = fsum o.add o.zero (wattr o cfg a) x.1 ∧
      x.2.2 = fsum o.add o.zero (cattrE o cfg a) x.1 ∧ o.fin x.2.1 = true ∧ o.fin x.2.2 = true ∧
      cfg.maxCost.all (fun m => !o.gt x.2.2 m) = true := by
  have hq := enum_eq_rec (ecOf o cfg a rels) (recR o cfg) [] b0 fuel (r, b) h
  let Q : List (Fnd Edge F) → F → Prop := fun r _ => ∀ x ∈ r, (∃ e, WalkTo (stepC cfg rels) cfg.source x.1 e) ∧
    x.1 ≠ [] ∧ (nodesOf cfg.source x.1).Nodup ∧ x.1.length ≤ cfg.maxLen ∧
    x.2.1 = fsum o.add o.zero (wattr o cfg a) x.1 ∧ x.2.2 = fsum o.add o.zero (cattrE o cfg a) x.1 ∧
    o.fin x.2.1 = true ∧ o.fin x.2.2 = true ∧ cfg.maxCost.all (fun m => !o.gt x.2.2 m) = true
  have hQ := enum_sound (ecOf o cfg a rels) (recR o cfg) Q (fun r f b' hq hv => by
    intro x hx
    rcases record_members (cmpF o) (fun x => x.2.1) o.gt o.lt cfg.pathCount.toNat r f b' x hx with rfl | hm
    · obtain ⟨e, hw, hne, hnd, hl, _, hf1, hf2, hc, hf3, hf4, hm⟩ := hv
      exact ⟨⟨e, hw⟩, hne, hnd, hl, hf1, hf2, hc, hf3, hm⟩
    · exact hq x hm) [] b0 (fun x hx => by simp at hx)
  rw [← hq] at hQ
  exact hQ

/-- The searches in `run_path_algo`'s `Except` interface (fuel exhaustion = the timeout error). -/
def dijE (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (fuel : Nat) (t : Nat) :
    Except String (Option (Fnd Edge F)) :=
  match dijR o cfg a rels t fuel with
  | none => .error "out of fuel"
  | some none => .ok none
  | some (some (es, w, c)) => .ok (some (es, w, c))

def bfsE (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (t : Nat) :
    Except String (Option (F × F)) := .ok (bfsBoundR o cfg a rels t)

def enumE (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat)) (fuel : Nat) (b : F) :
    Except String (List (Fnd Edge F)) :=
  match enumR o cfg a rels b fuel with
  | none => .error "out of fuel"
  | some (r, _) => .ok r

/-- `sort_by(cmp_found_path)` (stable). -/
def sortR (o : Fl F) (l : List (Fnd Edge F)) : List (Fnd Edge F) :=
  l.mergeSort (fun x y => cmpF o x y != .gt)

/-- **PROVEN** (`run_path_algo`, end to end at the Rust data): with the three searches instantiated by
`dijR` / `bfsBoundR` / `enumR` and `sort_by(cmp_found_path)`, every batch it returns is `build_path_batch`
of a list of paths each of which is a walk over `node_successors` edges (relTypes and direction as
configured) from the source with distinct nodes and finite weight, whose cost is the left fold of its
edges' costs; `build` is `build_path_batch` (any function of the list). -/
theorem runAlgoR_good (o : Fl F) (cfg : Cfg F) (a : Attrs F) (rels : Nat → List (Nat × Nat × Nat))
    (fuel : Nat) {R : Type} (build : List (Fnd Edge F) → R) (out : R)
    (h : runAlgo cfg o.inf o.le (dijE o cfg a rels fuel) (bfsE o cfg a rels) (enumE o cfg a rels fuel)
      (sortR o) build = .ok out) :
    ∃ L, out = build L ∧ ∀ x ∈ L, Good o cfg a rels x := by
  refine runAlgo_ok_good cfg o.inf o.le (dijE o cfg a rels fuel) (bfsE o cfg a rels) (enumE o cfg a rels fuel)
    (sortR o) build (Good o cfg a rels) ?_ ?_ ?_ out h
  · intro t f hd
    unfold dijE at hd
    rcases hq : dijR o cfg a rels t fuel with _ | _ | ⟨es, w, c⟩
    · rw [hq] at hd; simp at hd
    · rw [hq] at hd; simp at hd
    · rw [hq] at hd
      simp only [Except.ok.injEq, Option.some.injEq] at hd
      subst hd
      obtain ⟨_, hw, hnd, hc, hf⟩ := dijR_valid o cfg a rels t fuel es w c hq
      exact ⟨⟨t, hw⟩, hnd, hf, Or.inr hc⟩
  · intro b rs he x hx
    unfold enumE at he
    rcases hq : enumR o cfg a rels b fuel with _ | ⟨r, b'⟩
    · rw [hq] at he; simp at he
    · rw [hq] at he
      simp only [Except.ok.injEq] at he
      subst he
      obtain ⟨⟨e, hw⟩, _, hnd, _, _, hc, hf, _, _⟩ := enumR_sound o cfg a rels b fuel r b' hq x hx
      exact ⟨⟨e, hw⟩, hnd, hf, Or.inl hc⟩
  · intro l x hx
    exact (List.mem_mergeSort).mp hx

end OpsTraverse.AlgoSearch
