import OpsTraverse.ShortestPaths
/-
# PathBuilder, `Value::Path` branch: inlining a variable-length segment (#3451)

`ShortestPaths.lean` models PathBuilder's `Value::List` branch (allShortestPaths).
This file models the other branch: a variable-length relationship variable is
bound by `CondVarLenTraverseOp` to a `Value::Path` `[Node, Rel, Node, …, Node]`
in *traversal* order, and PathBuilder splices it into the named path.

| here | there (origin/main b84473a7f) |
| --- | --- |
| `PV` | the `Value::Node` / `Value::Relationship` elements of a `Value::Path` |
| `enc` | the VLT binding `Value::Path([Node(s), Rel(e₁), Node(n₁), …])` (`runtime/ops/cond_var_len_traverse.rs`, `path_value`) |
| `lastNode` | `elems.iter().rev().find_map(Node)` (`runtime/ops/path_builder.rs:95-101`) |
| `revWalk` | the reversed loop `path_elems.iter().rev().skip(1)` (`path_builder.rs:115-137`): push the rel, next node = the endpoint other than the last node in `elems` (`:119-130`), intermediate nodes skipped (`:132-134`) |
| `vltMain` | `PathBuilderOp::next`'s `Value::Path` arm (`path_builder.rs:88-139`): `vlt_first_matches` (`:102-105`) picks forward `skip(1)` (`:106-110`) or `revWalk` (`:111-138`) |
| `vltFix` | the suggested fix: the pattern direction (`<-`) recorded at plan time picks the branch |

For an incoming pattern `(m)<-[*]-(n)` the parser swaps the endpoints, so the
VLT walks `n → … → m` and binds `[Node n, …, Node m]`; PathBuilder must emit it
reversed. Main does not know the pattern direction: it *infers* it by comparing
the VLT's first node with the previous node of the path (`m`). When `n = m` (a
cycle) the comparison says "forward" and the segment is emitted in traversal
order.

`MATCH p = (a:N {id: 0})<-[:R*3..3]-(a)` on `0→1→2→0` (w 1, 2, 3): Rust
`[0,1,2,0]`/`[1,2,3]`, expected `[0,2,1,0]`/`[3,2,1]` (C `[0,2,1,0]`/`[1,2,3]`,
inconsistent). Property `PathFollowsPattern`; Lean `bug3451_*`.
-/

namespace OpsTraverse.PathVlt

open OpsTraverse.ASP

inductive PV where
  | node (n : Nat)
  | rel (e : Nat)
deriving DecidableEq, Repr

/-- The VLT binding for a walk from `s` taking `(edge, next node)` steps. -/
def enc (s : Nat) (steps : List (Nat × Nat)) : List PV :=
  .node s :: steps.flatMap fun p => [.rel p.1, .node p.2]

/-- The walk's last node. -/
def endOf (s : Nat) : List (Nat × Nat) → Nat
  | [] => s
  | p :: r => endOf p.2 r

/-- Each step's edge connects the current node and the next one, in either
stored orientation (a VLT step follows `step`, so an incoming pattern's edges
are stored opposite to the path direction). -/
def Linked (g : Graph) : Nat → List (Nat × Nat) → Prop
  | _, [] => True
  | s, (e, n) :: r =>
    (∃ x ∈ g, x.id = e ∧ ((x.src = s ∧ x.dst = n) ∨ (x.src = n ∧ x.dst = s))) ∧ Linked g n r

/-- `elems.iter().rev().find_map(Node)` (`path_builder.rs:95-101`). -/
def lastNode (elems : List PV) : Option Nat :=
  elems.reverse.findSome? fun | .node n => some n | _ => none

/-- `path_builder.rs:115-137`, with `cur` = the last node already in `elems`. -/
def revWalk (g : Graph) : Option Nat → List PV → List PV
  | _, [] => []
  | cur, .node _ :: r => revWalk g cur r
  | cur, .rel e :: r =>
    let next := if cur == some (ends g e).1 then (ends g e).2 else (ends g e).1
    .rel e :: .node next :: revWalk g (some next) r

/-- Main: `PathBuilderOp::next`, `Value::Path` arm (`path_builder.rs:88-139`).
The pattern direction is not an input. -/
def vltMain (g : Graph) (elems pe : List PV) : List PV :=
  if pe.length ≤ 1 then elems
  else
    let firstMatches := match pe.head? with
      | some (.node n) => lastNode elems == some n
      | _ => true
    if firstMatches then elems ++ pe.tail
    else elems ++ revWalk g (lastNode elems) pe.reverse.tail

/-- The fix: the plan records whether the hop was written `<-`. -/
def vltFix (g : Graph) (reversed : Bool) (elems pe : List PV) : List PV :=
  if pe.length ≤ 1 then elems
  else if reversed then elems ++ revWalk g (lastNode elems) pe.reverse.tail
  else elems ++ pe.tail

/-- **Property**: splicing a VLT walk `s → … → t` follows the pattern: written
forward (`prev = s`) it is appended as is; written `<-` (`prev = t`) it is
appended reversed — for every walk, cycles included. `build` gets the pattern
direction; an implementation may ignore it. -/
def PathFollowsPattern (build : Graph → Bool → List PV → List PV → List PV) : Prop :=
  ∀ (g : Graph) (s : Nat) (steps : List (Nat × Nat)) (elems : List PV) (rev : Bool),
    g.WF → Linked g s steps → steps ≠ [] →
    lastNode elems = some (if rev then endOf s steps else s) →
    build g rev elems (enc s steps) =
      elems ++ (if rev then (enc s steps).reverse.tail else (enc s steps).tail)

/-! ## The "other endpoint" walk reconstructs any reversed VLT walk -/

theorem snoc_ind {α : Type} {P : List α → Prop} (h0 : P [])
    (hs : ∀ l a, P l → P (l ++ [a])) : ∀ l, P l := by
  intro l
  have : ∀ r : List α, P r.reverse := by
    intro r
    induction r with
    | nil => exact h0
    | cons a r ih => simpa using hs _ a ih
  simpa using this l.reverse

theorem endOf_snoc (s : Nat) (l : List (Nat × Nat)) (p : Nat × Nat) :
    endOf s (l ++ [p]) = p.2 := by
  induction l generalizing s with
  | nil => rfl
  | cons q l ih => exact ih q.2

theorem linked_snoc {g : Graph} {e n : Nat} :
    ∀ {s : Nat} {l : List (Nat × Nat)}, Linked g s (l ++ [(e, n)]) →
      Linked g s l ∧
        ∃ x ∈ g, x.id = e ∧ ((x.src = endOf s l ∧ x.dst = n) ∨ (x.src = n ∧ x.dst = endOf s l))
  | s, [], h => ⟨trivial, h.1⟩
  | s, (e', n') :: l, h => by
    obtain ⟨h1, h2⟩ := linked_snoc (s := n') (l := l) h.2
    exact ⟨⟨h.1, h1⟩, h2⟩

theorem enc_snoc (s : Nat) (l : List (Nat × Nat)) (e n : Nat) :
    enc s (l ++ [(e, n)]) = enc s l ++ [.rel e, .node n] := by
  simp [enc]

theorem enc_reverse_head (s : Nat) (l : List (Nat × Nat)) :
    ∃ r, (enc s l).reverse = .node (endOf s l) :: r := by
  induction l using snoc_ind with
  | h0 => exact ⟨[], rfl⟩
  | hs l p _ =>
    obtain ⟨e, n⟩ := p
    refine ⟨.rel e :: (enc s l).reverse, ?_⟩
    rw [enc_snoc, endOf_snoc]; simp

/-- The reversed branch is right whenever it is taken on a real walk: starting
from the walk's end, `revWalk` re-derives exactly the reversed encoding. -/
theorem revWalk_reconstructs {g : Graph} (hwf : g.WF) (s : Nat) :
    ∀ (steps : List (Nat × Nat)), Linked g s steps →
      revWalk g (some (endOf s steps)) (enc s steps).reverse.tail = (enc s steps).reverse.tail := by
  intro steps
  induction steps using snoc_ind with
  | h0 => intro; rfl
  | hs l p ih =>
    obtain ⟨e, n⟩ := p
    intro h
    obtain ⟨hl, x, hx, rfl, hdir⟩ := linked_snoc h
    obtain ⟨r, hr⟩ := enc_reverse_head s l
    have hih := ih hl
    rw [hr] at hih
    simp only [List.tail_cons] at hih
    have hend := ends_of_mem hwf hx
    have hnext : (if some n == some (ends g x.id).1 then (ends g x.id).2 else (ends g x.id).1)
        = endOf s l := by
      rw [hend]
      rcases hdir with ⟨h1, h2⟩ | ⟨h1, h2⟩
      · by_cases hm : n = endOf s l
        · simp [h1, h2, hm]
        · have : (some n == some x.src) = false := by
            simp only [beq_eq_false_iff_ne, ne_eq, Option.some.injEq]; rw [h1]; exact hm
          simp only [this, Bool.false_eq_true, ite_false]; exact h1
      · simp [h1, h2]
    rw [enc_snoc, endOf_snoc]
    simp only [List.reverse_append, List.reverse_cons, List.reverse_nil, List.nil_append,
      List.cons_append, List.tail_cons]
    rw [hr]
    simp only [revWalk, hnext, hih]

theorem enc_length (s : Nat) (steps : List (Nat × Nat)) (h : steps ≠ []) :
    ¬ (enc s steps).length ≤ 1 := by
  cases steps with
  | nil => exact absurd rfl h
  | cons p r => simp [enc]

/-- **The fix satisfies the property.** -/
theorem vltFix_follows : PathFollowsPattern vltFix := by
  intro g s steps elems rev hwf hl hne hlast
  unfold vltFix
  simp only [enc_length s steps hne, ite_false]
  cases rev
  · simp
  · simp only [ite_true] at hlast ⊢
    rw [hlast, revWalk_reconstructs hwf s steps hl]

/-- Main is right on every non-cyclic walk: there the endpoint comparison does
recover the pattern direction. -/
theorem vltMain_follows_of_ne {g : Graph} {s : Nat} {steps : List (Nat × Nat)}
    {elems : List PV} {rev : Bool} (hwf : g.WF) (hl : Linked g s steps) (hne : steps ≠ [])
    (hcyc : endOf s steps ≠ s)
    (hlast : lastNode elems = some (if rev then endOf s steps else s)) :
    vltMain g elems (enc s steps) =
      elems ++ (if rev then (enc s steps).reverse.tail else (enc s steps).tail) := by
  unfold vltMain
  simp only [enc_length s steps hne, ite_false]
  cases rev
  · simp only [Bool.false_eq_true, ite_false] at hlast ⊢
    simp [enc, hlast]
  · simp only [ite_true] at hlast ⊢
    have : (lastNode elems == some s) = false := by
      rw [hlast]; simp [hcyc]
    simp only [enc, List.head?_cons, this, Bool.false_eq_true, ite_false]
    rw [hlast]
    exact congrArg (elems ++ ·) (revWalk_reconstructs hwf s steps hl)

/-! ## #3451: the cyclic incoming case -/

/-- The issue's triangle `0→1→2→0`, edge ids = the `w` values 1, 2, 3. -/
def gTri : Graph := [⟨1, 0, 1⟩, ⟨2, 1, 2⟩, ⟨3, 2, 0⟩]

/-- `(a)<-[:R*3..3]-(a)`: the VLT walks `0 →1→ 1 →2→ 2 →3→ 0`. -/
def triWalk : List (Nat × Nat) := [(1, 1), (2, 2), (3, 0)]

/-- Main emits the cycle in traversal order: nodes `[0,1,2,0]`, rels `[1,2,3]`. -/
theorem bug3451_cycle_not_reversed :
    vltMain gTri [.node 0] (enc 0 triWalk) =
      [.node 0, .rel 1, .node 1, .rel 2, .node 2, .rel 3, .node 0] := by decide

/-- …where the incoming pattern requires nodes `[0,2,1,0]`, rels `[3,2,1]`. -/
theorem bug3451_expected :
    [PV.node 0] ++ (enc 0 triWalk).reverse.tail =
      [.node 0, .rel 3, .node 2, .rel 2, .node 1, .rel 1, .node 0] := by decide

/-- The non-cyclic incoming case `(a {id:0})<-[:R*2..2]-(b)` is right on main:
the VLT walks `1 → 2 → 0` and PathBuilder emits `[0,2,1]`/`[3,2]`. -/
theorem bug3451_noncyclic_ok :
    vltMain gTri [.node 0] (enc 1 [(2, 2), (3, 0)]) =
      [.node 0, .rel 3, .node 2, .rel 2, .node 1] := by decide

/-- **Main violates `PathFollowsPattern`.** -/
theorem bug3451_main_violates : ¬ PathFollowsPattern (fun g _ elems pe => vltMain g elems pe) := by
  intro h
  have := h gTri 0 triWalk [.node 0] true (by unfold Graph.WF; decide)
    ⟨⟨⟨1, 0, 1⟩, by decide, rfl, Or.inl ⟨rfl, rfl⟩⟩,
     ⟨⟨2, 1, 2⟩, by decide, rfl, Or.inl ⟨rfl, rfl⟩⟩,
     ⟨⟨3, 2, 0⟩, by decide, rfl, Or.inl ⟨rfl, rfl⟩⟩, trivial⟩
    (by decide) (by decide)
  revert this; decide

end OpsTraverse.PathVlt
