/-
# `QueryGraph` and friends (`graph/src/parser/ast.rs:470-890`)

The pattern graph a MATCH / CREATE / MERGE builds, generic over the alias
type `V` exactly as the Rust is generic over `TVar` (`Arc<String>` before
binding, `Variable` after). Alias equality is `==` on `V`; for `Variable`
that is `Variable::eq`, which compares ids only (ast.rs:112-118).
-/
import FalkorParserGrammar.C.Core

namespace FalkorParserGrammar.C

/-! ## `Variable` (ast.rs:90-139) -/

/-- `Variable { name, id, scope_id, ty }`; the type is irrelevant here. -/
structure Var where
  name : Option Nat
  id : Nat
  scope : Nat
  deriving Repr

/-- `impl PartialEq for Variable` (ast.rs:112-118): by id only. -/
instance : BEq Var := ⟨fun a b => a.id == b.id⟩

/-- `Variable::eq`: two variables are equal iff their ids are. -/
theorem Var.eq_spec (a b : Var) : (a == b) = true ↔ a.id = b.id := by
  show (a.id == b.id) = true ↔ _; simp

/-- `impl Hash for Variable` (ast.rs:122-129) hashes the id alone, so equal
variables hash alike (the `Hash`/`Eq` contract). The hash is modelled as
the hashed key. -/
def Var.hashKey (a : Var) : Nat := a.id
theorem Var.hash_eq (a b : Var) (h : (a == b) = true) : a.hashKey = b.hashKey :=
  (Var.eq_spec a b).1 h

/-- `Variable::as_str` (ast.rs:133-136). Names are ids; `none` is `"?"`. -/
def Var.asStr (a : Var) : Option Nat := a.name
theorem Var.asStr_spec (a : Var) : a.asStr = a.name := rfl

/-- `impl Display for Variable` (ast.rs:99-109): the name, or `?<id>`. -/
def Var.fmt (a : Var) : String :=
  match a.name with
  | some n => toString n
  | none => "?" ++ toString a.id
theorem Var.fmt_anon (a : Var) (h : a.name = none) : a.fmt = "?" ++ toString a.id := by
  simp [Var.fmt, h]
theorem Var.fmt_named (a : Var) (n : Nat) (h : a.name = some n) : a.fmt = toString n := by
  simp [Var.fmt, h]

/-! ## Nodes, relationships, paths -/

structure QNode (V : Type) where
  alias : V
  labels : List Nat
  attrs : ET

inductive ASP | no | fwd | rev
  deriving DecidableEq, Repr

structure QRel (V : Type) where
  alias : V
  types : List Nat
  attrs : ET
  src : QNode V
  dst : QNode V
  bidir : Bool
  minH : Option Nat
  maxH : Option Nat
  asp : ASP

structure QPath (V : Type) where
  var : V
  vars : List V

/-- `QueryNode::new` (ast.rs:494-505). -/
def QNode.new {V} (alias : V) (labels : List Nat) (attrs : ET) : QNode V := ⟨alias, labels, attrs⟩
theorem QNode.new_spec {V} (a : V) (l : List Nat) (t : ET) :
    (QNode.new a l t).alias = a ∧ (QNode.new a l t).labels = l ∧ (QNode.new a l t).attrs = t :=
  ⟨rfl, rfl, rfl⟩

/-- `QueryRelationship::new` (ast.rs:569-594): `all_shortest_paths = No`. -/
def QRel.new {V} (alias : V) (types : List Nat) (attrs : ET) (src dst : QNode V) (bidir : Bool)
    (minH maxH : Option Nat) : QRel V := ⟨alias, types, attrs, src, dst, bidir, minH, maxH, .no⟩
theorem QRel.new_spec {V} (a : V) ts t (s d : QNode V) b mn mx :
    let r := QRel.new a ts t s d b mn mx
    r.alias = a ∧ r.types = ts ∧ r.src.alias = s.alias ∧ r.dst.alias = d.alias ∧ r.bidir = b ∧
      r.minH = mn ∧ r.maxH = mx ∧ r.asp = .no :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `QueryPath::new` (ast.rs:607-613). -/
def QPath.new {V} (var : V) (vars : List V) : QPath V := ⟨var, vars⟩
theorem QPath.new_spec {V} (v : V) (vs : List V) :
    (QPath.new v vs).var = v ∧ (QPath.new v vs).vars = vs := ⟨rfl, rfl⟩

/-! ## `QueryGraph` (ast.rs:622-790) -/

structure QG (V : Type) where
  nodes : List (QNode V)
  rels : List (QRel V)
  paths : List (QPath V)

/-- `QueryGraph::default` (ast.rs:630-637). -/
def QG.empty {V} : QG V := ⟨[], [], []⟩
theorem QG.default_spec {V} : (QG.empty : QG V).nodes = [] ∧ (QG.empty : QG V).rels = [] ∧
    (QG.empty : QG V).paths = [] := ⟨rfl, rfl, rfl⟩

section ops
variable {V : Type} [BEq V]

/-- `QueryGraph::add_node` (ast.rs:685-694). -/
def QG.addNode (g : QG V) (n : QNode V) : Bool × QG V :=
  if g.nodes.any (fun m => m.alias == n.alias) then (false, g)
  else (true, { g with nodes := g.nodes ++ [n] })

/-- `QueryGraph::add_relationship` (ast.rs:733-747). -/
def QG.addRel (g : QG V) (r : QRel V) : Bool × QG V :=
  if g.rels.any (fun m => m.alias == r.alias) then (false, g)
  else (true, { g with rels := g.rels ++ [r] })

/-- `QueryGraph::add_path` (ast.rs:749-759). -/
def QG.addPath (g : QG V) (p : QPath V) : Bool × QG V :=
  if g.paths.any (fun m => m.var == p.var) then (false, g)
  else (true, { g with paths := g.paths ++ [p] })

theorem QG.addNode_spec (g : QG V) (n : QNode V) :
    (g.nodes.any (fun m => m.alias == n.alias) = true ∧ g.addNode n = (false, g)) ∨
    (g.nodes.any (fun m => m.alias == n.alias) = false ∧
      g.addNode n = (true, { g with nodes := g.nodes ++ [n] })) := by
  unfold QG.addNode; cases h : g.nodes.any (fun m => m.alias == n.alias) <;> simp

theorem QG.addRel_spec (g : QG V) (r : QRel V) :
    (g.rels.any (fun m => m.alias == r.alias) = true ∧ g.addRel r = (false, g)) ∨
    (g.rels.any (fun m => m.alias == r.alias) = false ∧
      g.addRel r = (true, { g with rels := g.rels ++ [r] })) := by
  unfold QG.addRel; cases h : g.rels.any (fun m => m.alias == r.alias) <;> simp

theorem QG.addPath_spec (g : QG V) (p : QPath V) :
    (g.paths.any (fun m => m.var == p.var) = true ∧ g.addPath p = (false, g)) ∨
    (g.paths.any (fun m => m.var == p.var) = false ∧
      g.addPath p = (true, { g with paths := g.paths ++ [p] })) := by
  unfold QG.addPath; cases h : g.paths.any (fun m => m.var == p.var) <;> simp

/-- Aliases are pairwise distinct (w.r.t. `==`). -/
def DistinctBy {α} (key : α → V) : List α → Prop
  | [] => True
  | a :: as => as.all (fun b => !(key b == key a)) = true ∧ DistinctBy key as

theorem DistinctBy.append {α} (key : α → V) [LawfulBEq V] :
    ∀ (l : List α) (x : α), DistinctBy key l → l.any (fun m => key m == key x) = false →
      DistinctBy key (l ++ [x])
  | [], x, _, _ => by simp [DistinctBy]
  | a :: as, x, ⟨h1, h2⟩, h => by
    simp only [List.any_cons, Bool.or_eq_false_iff] at h
    refine ⟨?_, DistinctBy.append key as x h2 h.2⟩
    simp only [List.append_eq, List.all_append, h1, Bool.true_and, List.all_cons, List.all_nil,
      Bool.and_true]
    have := h.1
    cases e : (key x == key a)
    · rfl
    · simp only [beq_iff_eq] at e; rw [e] at this; simp at this

/-- **add_node keeps node aliases distinct** (for a lawful `==`, e.g. names). -/
theorem QG.addNode_distinct [LawfulBEq V] (g : QG V) (n : QNode V)
    (h : DistinctBy QNode.alias g.nodes) : DistinctBy QNode.alias (g.addNode n).2.nodes := by
  rcases QG.addNode_spec g n with ⟨_, e⟩ | ⟨h2, e⟩ <;> rw [e]
  · exact h
  · exact DistinctBy.append _ _ _ h h2

theorem QG.addRel_distinct [LawfulBEq V] (g : QG V) (r : QRel V)
    (h : DistinctBy QRel.alias g.rels) : DistinctBy QRel.alias (g.addRel r).2.rels := by
  rcases QG.addRel_spec g r with ⟨_, e⟩ | ⟨h2, e⟩ <;> rw [e]
  · exact h
  · exact DistinctBy.append _ _ _ h h2

/-- `merge_attr_maps` (ast.rs:667-683): concatenate the entries of two maps;
an empty side returns the other one unchanged. -/
def mergeAttrs (lhs rhs : ET) : ET :=
  if rhs.kids = [] then lhs
  else if lhs.kids = [] then rhs
  else .node lhs.root (lhs.kids ++ rhs.kids)

/-- **merge_attr_maps keeps every entry of both sides, in order** (repeated
keys stay repeated, ast.rs:663-665). -/
theorem mergeAttrs_kids (l r : ET) : (mergeAttrs l r).kids = l.kids ++ r.kids := by
  unfold mergeAttrs
  by_cases hr : r.kids = []
  · rw [if_pos hr, hr, List.append_nil]
  · by_cases hl : l.kids = []
    · rw [if_neg hr, if_pos hl, hl, List.nil_append]
    · rw [if_neg hr, if_neg hl]; rfl

theorem mergeAttrs_map (l r : ET) (hl : l.root = .map) (hr : r.root = .map) :
    (mergeAttrs l r).root = .map := by
  unfold mergeAttrs; split
  · exact hl
  · split
    · exact hr
    · exact hl

/-- `OrderSet::extend`: insert each label not already there. -/
def extendSet (acc : List Nat) : List Nat → List Nat
  | [] => acc
  | l :: ls => extendSet (if l ∈ acc then acc else acc ++ [l]) ls

theorem extendSet_mem (acc ls : List Nat) (x : Nat) :
    x ∈ extendSet acc ls ↔ x ∈ acc ∨ x ∈ ls := by
  induction ls generalizing acc with
  | nil => simp [extendSet]
  | cons l ls ih =>
    simp only [extendSet, ih]
    by_cases hl : l ∈ acc <;> by_cases hx : x = l <;> simp_all

def findIdx {α} (p : α → Bool) : List α → Option Nat
  | [] => none
  | a :: as => if p a then some 0 else (findIdx p as).map (· + 1)

/-- `QueryGraph::merge_node` (ast.rs:705-721). -/
def QG.mergeNode (g : QG V) (n : QNode V) : QG V :=
  match findIdx (fun m => m.alias == n.alias) g.nodes with
  | none => { g with nodes := g.nodes ++ [n] }
  | some pos =>
    match g.nodes[pos]? with
    | none => g    -- unreachable: `position` returns a valid index
    | some ex =>
      let m : QNode V := ⟨ex.alias, extendSet ex.labels n.labels, mergeAttrs ex.attrs n.attrs⟩
      { g with nodes := g.nodes.set pos m }

/-- `QueryGraph::replace_node` (ast.rs:723-731). -/
def QG.replaceNode (g : QG V) (a : V) (n : QNode V) : QG V :=
  match findIdx (fun m => m.alias == a) g.nodes with
  | none => g
  | some pos => { g with nodes := g.nodes.set pos n }

theorem findIdx_lt {α} (p : α → Bool) : ∀ (l : List α) (i : Nat), findIdx p l = some i →
    i < l.length ∧ ∃ h : i < l.length, p (l[i]'h) = true
  | [], _, h => by simp [findIdx] at h
  | a :: as, i, h => by
    unfold findIdx at h
    split at h
    · cases h; exact ⟨by simp, by simp, by simpa⟩
    · cases e : findIdx p as with
      | none => rw [e] at h; simp at h
      | some j =>
        rw [e] at h; simp at h; subst h
        obtain ⟨h1, h2, h3⟩ := findIdx_lt p as j e
        exact ⟨by simp; omega, by simp; omega, by simpa using h3⟩

theorem findIdx_none {α} (p : α → Bool) : ∀ l : List α, findIdx p l = none → l.any p = false
  | [], _ => rfl
  | a :: as, h => by
    unfold findIdx at h; split at h
    · cases h
    · cases e : findIdx p as with
      | none => simp [*, findIdx_none p as e]
      | some _ => rw [e] at h; cases h

/-- **merge_node**: when the alias is present, the node list keeps its length
and alias order and the entry at that alias gets the union of labels and the
concatenation of attribute entries; otherwise the node is appended. -/
theorem QG.mergeNode_spec (g : QG V) (n : QNode V) :
    (g.nodes.any (fun m => m.alias == n.alias) = false ∧ (g.mergeNode n).nodes = g.nodes ++ [n]) ∨
    (∃ pos ex, g.nodes[pos]? = some ex ∧ (ex.alias == n.alias) = true ∧
      (g.mergeNode n).nodes = g.nodes.set pos
        (⟨ex.alias, extendSet ex.labels n.labels, mergeAttrs ex.attrs n.attrs⟩ : QNode V)) := by
  unfold QG.mergeNode
  cases e : findIdx (fun m => m.alias == n.alias) g.nodes with
  | none => exact .inl ⟨findIdx_none _ _ e, rfl⟩
  | some pos =>
    obtain ⟨hl, hlt, hp⟩ := findIdx_lt _ _ _ e
    have : g.nodes[pos]? = some (g.nodes[pos]'hl) := by simp [hl]
    simp only [this]
    exact .inr ⟨pos, _, this, hp, rfl⟩

theorem QG.mergeNode_rels (g : QG V) (n : QNode V) :
    (g.mergeNode n).rels = g.rels ∧ (g.mergeNode n).paths = g.paths := by
  unfold QG.mergeNode; split
  · exact ⟨rfl, rfl⟩
  · split <;> exact ⟨rfl, rfl⟩

theorem QG.replaceNode_spec (g : QG V) (a : V) (n : QNode V) :
    (g.nodes.any (fun m => m.alias == a) = false ∧ g.replaceNode a n = g) ∨
    (∃ pos, pos < g.nodes.length ∧ (g.replaceNode a n).nodes = g.nodes.set pos n) := by
  unfold QG.replaceNode
  cases e : findIdx (fun m => m.alias == a) g.nodes with
  | none => exact .inl ⟨findIdx_none _ _ e, rfl⟩
  | some pos => exact .inr ⟨pos, (findIdx_lt _ _ _ e).1, rfl⟩

/-- `QueryGraph::variables` (ast.rs:761-767). -/
def QG.variables (g : QG V) : List V :=
  g.nodes.map (·.alias) ++ g.rels.map (·.alias) ++ g.paths.map (·.var)
theorem QG.variables_length (g : QG V) :
    g.variables.length = g.nodes.length + g.rels.length + g.paths.length := by
  simp [QG.variables]; omega

end ops

/-! Accessors (ast.rs:769-790): `nodes`, `nodes_mut`, `relationships`,
`relationships_mut`, `paths` return the field itself. -/
theorem QG.accessors {V} (g : QG V) :
    g.nodes = QG.nodes g ∧ g.rels = QG.rels g ∧ g.paths = QG.paths g := ⟨rfl, rfl, rfl⟩

end FalkorParserGrammar.C
