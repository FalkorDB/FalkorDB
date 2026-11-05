import OpsTraverse.Drive
/-
Variable-length traverse glue (graph/src/runtime/ops/cond_var_len_traverse.rs).

| here | there |
| --- | --- |
| `pathValue`, `Alt`   | `VarLenIter::path_value` (:139) |
| `Frame`, `vlNext`, `run` | `Iterator for VarLenIter::next` (:392): pop the frame buffer, stop on a parked error, else begin the next start node or `advance` one DFS frame |
| `rowPlan`            | `CondVarLenTraverseOp::expand_row` (:500): endpoint resolution, reversal, start nodes, destination, hop defaults |
| `VlSt.new`, `findPrev` | `CondVarLenTraverseOp::new` (:441) |
| `vlOpNext`           | `CondVarLenTraverseOp::next` (:611) — `Drive.driveE` with the parked-error check |

A DFS frame is modelled by the finite tree it unfolds into (`Frame`): its emissions and its
child frames (`advance` computes them from the graph; the DFS proper — trails, `min..max`
hops — is `VarLen.dfs`, proved in VarLen.lean).
-/
namespace OpsTraverse.VarLenGlue

/-! ## `path_value` (:139) -/

def pathValue {α : Type} (elems : List α) (reversed : Bool) : List α :=
  if reversed then elems.reverse else elems

/-- `[Node, Rel, Node, …, Node]`. -/
inductive Alt {N R : Type} : List (Sum N R) → Prop
  | one (n : N) : Alt [.inl n]
  | cons (n : N) (r : R) (rest : List (Sum N R)) : Alt rest → Alt (.inl n :: .inr r :: rest)

theorem Alt.snoc {N R : Type} : ∀ {l : List (Sum N R)} (r : R) (n : N), Alt l → Alt (l ++ [.inr r, .inl n])
  | _, r, n, .one m => .cons m r _ (.one n)
  | _, r, n, .cons m r' rest h => by
      simp only [List.cons_append]
      exact .cons m r' _ (Alt.snoc r n h)

theorem Alt.reverse {N R : Type} : ∀ {l : List (Sum N R)}, Alt l → Alt l.reverse
  | _, .one n => .one n
  | _, .cons n r rest h => by
      simp only [List.reverse_cons, List.append_assoc, List.singleton_append]
      exact Alt.snoc r n (Alt.reverse h)

/-- A reversed traversal's element list is flipped, and stays an alternating
node/relationship path, so `nodes(p)`/`relationships(p)` follow the pattern. -/
theorem pathValue_spec {N R : Type} (l : List (Sum N R)) (rev : Bool) (h : Alt l) :
    Alt (pathValue l rev) ∧ (rev = true → pathValue l rev = l.reverse) ∧ (rev = false → pathValue l rev = l) := by
  cases rev <;> simp [pathValue, h, Alt.reverse h]

/-! ## `VarLenIter::next` (:392) -/

inductive Frame (I : Type) where
  | mk (items : List I) (children : List (Frame I))

mutual
def allF {I : Type} : Frame I → List I
  | .mk is cs => is ++ allL cs
def allL {I : Type} : List (Frame I) → List I
  | [] => []
  | f :: fs => allF f ++ allL fs
end

mutual
def sizeF {I : Type} : Frame I → Nat
  | .mk is cs => is.length + 1 + sizeL cs
def sizeL {I : Type} : List (Frame I) → Nat
  | [] => 0
  | f :: fs => sizeF f + sizeL fs
end

structure It (I : Type) where
  buf : List I
  stack : List (Frame I)
  starts : List (Frame I)
  err : Bool

/-- One call of `next`'s loop body: `none` = the iterator returns `None`. `some (some i)` =
yield `i`; `some none` = state changed, loop again. -/
def vlStep {I : Type} (s : It I) : Option (Option I × It I) :=
  match s.buf with
  | i :: is => some (some i, { s with buf := is })
  | [] =>
    if s.err then none
    else match s.stack with
      | .mk is cs :: ws => some (none, { s with buf := is, stack := cs ++ ws })
      | [] => match s.starts with
        | [] => none
        | .mk is cs :: ss => some (none, { s with buf := is, stack := cs, starts := ss })

def run {I : Type} : Nat → It I → List I
  | 0, _ => []
  | n + 1, s => match vlStep s with
    | none => []
    | some (o, s') => o.toList ++ run n s'

def pending {I : Type} (s : It I) : List I := s.buf ++ allL s.stack ++ allL s.starts
def pot {I : Type} (s : It I) : Nat := s.buf.length + sizeL s.stack + sizeL s.starts

theorem allL_append {I : Type} (a b : List (Frame I)) : allL (a ++ b) = allL a ++ allL b := by
  induction a with
  | nil => rfl
  | cons f fs ih => simp [allL, ih]

theorem sizeL_append {I : Type} (a b : List (Frame I)) : sizeL (a ++ b) = sizeL a + sizeL b := by
  induction a with
  | nil => simp [sizeL]
  | cons f fs ih => simp [sizeL, ih]; omega

/-- **Without a parked error, `next` yields exactly the items of every frame of every start
node** (DFS emissions, in stack order), given enough steps. -/
theorem run_complete {I : Type} : ∀ (n : Nat) (s : It I), s.err = false → pot s ≤ n →
    run n s = pending s := by
  intro n
  induction n with
  | zero =>
    intro s _ hp
    have hb : s.buf = [] := List.eq_nil_of_length_eq_zero (by unfold pot at hp; omega)
    cases hs : s.stack with
    | cons f _ => cases f; simp [pot, hs, sizeL, sizeF] at hp
    | nil =>
      cases ht : s.starts with
      | cons f _ => cases f; simp [pot, hs, ht, sizeL, sizeF] at hp
      | nil => simp [run, pending, hb, hs, ht, allL]
  | succ n ih =>
    intro s he hp
    unfold run vlStep
    cases hb : s.buf with
    | cons i is =>
      simp only [Option.toList_some, List.singleton_append]
      have := ih { s with buf := is } he (by simp only [pot, hb, List.length_cons] at hp ⊢; omega)
      rw [this]
      simp [pending, hb]
    | nil =>
      simp only [he, Bool.false_eq_true, ite_false]
      cases hs : s.stack with
      | cons f ws =>
        obtain ⟨is, cs⟩ := f
        simp only [Option.toList_none, List.nil_append]
        have := ih { s with buf := is, stack := cs ++ ws, err := false } rfl
          (by simp only [pot, hb, hs, sizeL, sizeF, sizeL_append, List.length_nil] at hp ⊢; omega)
        rw [this]
        simp [pending, hb, hs, allL, allF, allL_append]
      | nil =>
        cases ht : s.starts with
        | nil => simp [pending, hb, hs, ht, allL]
        | cons f ss =>
          obtain ⟨is, cs⟩ := f
          simp only [Option.toList_none, List.nil_append]
          have := ih { s with buf := is, stack := cs, starts := ss, err := false } rfl
            (by simp only [pot, hb, hs, ht, sizeL, sizeF, List.length_nil] at hp ⊢; omega)
          rw [this]
          simp [pending, hb, hs, ht, allL, allF]

/-- A parked error stops the iterator once the current frame's buffer is drained. -/
theorem run_err {I : Type} (n : Nat) (st sts : List (Frame I)) :
    run (n + 1) ⟨[], st, sts, true⟩ = [] := by simp [run, vlStep]

/-! ## `expand_row` (:500) -/

inductive EV where
  | node (n : Nat)
  | other

structure Plan where
  reversed : Bool
  starts : Option (List Nat)   -- `none` = every node with the `from` labels
  dest : Option Nat
  minH : Nat
  maxH : Nat

def asNode : Option EV → Option Nat
  | some (.node n) => some n
  | _ => none

/-- cond_var_len_traverse.rs:531-570. `fromBound`/`toBound` = `batch.is_bound_at`. -/
def rowPlan (fromV toV : Option EV) (fromBound toBound bidir : Bool) (minH maxH : Option Nat) :
    Option Plan :=
  let fromId := asNode fromV
  let toId := asNode toV
  if fromId.isNone && fromBound then none
  else if toId.isNone && toBound then none
  else
    let reversed := fromId.isNone && toId.isSome && !bidir
    some { reversed
           starts := if reversed then toId.map ([·]) else fromId.map ([·])
           dest := if reversed then fromId else toId
           minH := minH.getD 1
           maxH := maxH.getD 4294967295 }

/-- A bound non-node endpoint skips the row; hops default to `*1..u32::MAX`; the walk is
reversed exactly when only `to` is a node and the edge is directed; the start set is the bound
endpoint (or the `from` label scan) and the destination the other endpoint. -/
theorem rowPlan_spec (fromV toV : Option EV) (fb tb bidir : Bool) (mn mx : Option Nat) :
    (asNode fromV = none → fb = true → rowPlan fromV toV fb tb bidir mn mx = none) ∧
    (∀ p, rowPlan fromV toV fb tb bidir mn mx = some p →
      p.minH = mn.getD 1 ∧ p.maxH = mx.getD 4294967295 ∧
      (p.reversed = true ↔ (asNode fromV = none ∧ (asNode toV).isSome ∧ bidir = false)) ∧
      (p.reversed = true → p.starts = (asNode toV).map ([·]) ∧ p.dest = asNode fromV) ∧
      (p.reversed = false → p.starts = (asNode fromV).map ([·]) ∧ p.dest = asNode toV)) := by
  constructor
  · intro h1 h2; simp [rowPlan, h1, h2]
  · intro p hp
    simp only [rowPlan] at hp
    split at hp
    · cases hp
    · split at hp
      · cases hp
      · simp only [Option.some.injEq] at hp
        subst hp
        refine ⟨rfl, rfl, ?_, fun h => ?_, fun h => ?_⟩
        · simp [Bool.and_eq_true, Option.isNone_iff_eq_none, and_assoc]
        · simp only at h; simp [h]
        · simp only at h; simp [h]

/-! ## `new` (:441) -/

/-- The `@prev(<id>)` marker: first variable (breadth-first) whose name starts with `@prev(`. -/
def findPrev (bfsVars : List (Nat × String)) : Option (Nat × String) :=
  bfsVars.find? fun v => v.2.startsWith "@prev("

structure VlSt where
  fromCol : Nat
  toCol : Nat
  distinct : Bool
  pathCol : Option Nat
  pathCopy : Option Nat
  cap : Option Nat
  prev : Option (Nat × String)
  emitPath : Bool
  err : Option String

def VlSt.new (fromA toA relA : Nat) (emitPath : Bool) (pathVar : Option Nat) (cap : Option Nat)
    (filterVars : Option (List (Nat × String))) : VlSt :=
  { fromCol := fromA, toCol := toA, distinct := fromA != toA,
    pathCol := if emitPath then some relA else none, pathCopy := pathVar, cap,
    prev := filterVars.bind findPrev, emitPath := emitPath || pathVar.isSome, err := none }

/-- A shared endpoint alias binds one column; the path column is bound only when emitted;
a directly bound named path forces path materialisation; no error parked initially. -/
theorem vlNew_spec (f t r : Nat) (ep : Bool) (pv : Option Nat) (cap : Option Nat) (fv : Option (List (Nat × String))) :
    let s := VlSt.new f t r ep pv cap fv
    (s.distinct = true ↔ f ≠ t) ∧ (s.pathCol.isSome ↔ ep = true) ∧
    (s.emitPath = true ↔ ep = true ∨ pv.isSome) ∧ s.err = none ∧ s.cap = cap ∧
    (∀ v, s.prev = some v → v.2.startsWith "@prev(" = true) := by
  refine ⟨by simp [VlSt.new], by cases ep <;> simp [VlSt.new], by simp [VlSt.new], rfl, rfl, ?_⟩
  intro v hv
  simp only [VlSt.new, Option.bind_eq_some_iff] at hv
  obtain ⟨l, _, hl⟩ := hv
  have := List.find?_some hl
  simpa using this

/-! ## `CondVarLenTraverseOp::next` (:611) -/

/-- Output stream: the shared loop with errors (`Drive.driveE`); a parked error is reported
instead of the batch that hit it. Error-free, it is the per-batch expansion stream. -/
theorem vlOpNext_stream {C R : Type} (pack : C → Except String (List (List R))) (f : C → List (List R))
    (hf : ∀ c, pack c = .ok (f c)) (cs : List C) :
    Drive.driveE pack [] (cs.map .ok) = (Drive.drive f [] cs).map .ok ∧
    (Drive.drive f [] cs).flatten = (cs.flatMap f).flatten := by
  refine ⟨Drive.driveE_ok pack f hf [] cs, by rw [Drive.drive_flatten]; simp⟩

end OpsTraverse.VarLenGlue
