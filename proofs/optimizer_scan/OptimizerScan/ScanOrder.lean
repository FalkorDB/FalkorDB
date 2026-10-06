/-
# Scan selection and hop ordering: `select_scan_node`, `reorder_labels`, label-primary reorder

Semantics used: a row is an assignment `σ : Nat → Nat` of node/edge ids to variables, and a
`CondTraverse` hop, a label check and a `Filter` are each a *test* on the finished row
(generate-and-test). Under that reading a chain of hops is the conjunction of its hop
predicates, so what the pass must preserve is: the same hops (up to orientation) and the
same filters, all applied somewhere. Expansion order (which operator *binds* a variable)
only affects cost — it is the modelling gap stated in `OptimizerScan.lean`.

| here | there |
| --- | --- |
| `Hop`, `Hop.holds`         | `IR::CondTraverse { relationship, transposed, .. }` and `CondTraverseOp` reading the relation matrix, or its transpose when `transposed` (`runtime/ops/cond_traverse.rs`) |
| `Hop.swap`                 | `swap_relationship(rel, rel.to, rel.from)` + `transposed = true`, `select_scan_node.rs:294-311, 965-970` |
| `pick`, `order`            | the greedy hop ordering loop, `select_scan_node.rs:914-974` |
| `hasAll`                   | label conjunction checked by `NodeByLabelScan` / `get_nodes(labels)` |
| `reorderLabels`            | `reorder_labels` (stable sort by schema id), `optimizer/reorder_labels.rs:16-47` |
| `withPrimary`              | `IndexSubject::with_primary_label`, `utilize_index.rs:206-222, 296-319` |
| `placeFilters`             | re-attachment of inter-CT filters "as soon as their inputs are bound", `select_scan_node.rs:1067-1117` |
-/
namespace OptimizerScan.ScanOrder

/-- A hop `(src)-[e:rel]->(dst)` evaluated on a row; `transposed` reads the relation backwards. -/
structure Hop where
  src : Nat
  dst : Nat
  edge : Nat
  rel : Nat
  transposed : Bool
  deriving DecidableEq, Repr

/-- The relationship store: `E r a b e` iff edge `e` of type `r` goes from `a` to `b`. -/
abbrev Store := Nat → Nat → Nat → Nat → Bool

/-- `CondTraverseOp`: storage direction `from → to`, or the transposed matrix. -/
def Hop.holds (E : Store) (σ : Nat → Nat) (h : Hop) : Bool :=
  if h.transposed then E h.rel (σ h.dst) (σ h.src) (σ h.edge)
  else E h.rel (σ h.src) (σ h.dst) (σ h.edge)

/-- `swap_relationship` + flipping `transposed` (`select_scan_node.rs:965-970`, `1001-1003`). -/
def Hop.swap (h : Hop) : Hop :=
  { h with src := h.dst, dst := h.src, transposed := !h.transposed }

/-- **PROVEN**: swapping a hop's endpoints and flipping `transposed` tests the same edge. -/
theorem swap_holds (E : Store) (σ : Nat → Nat) (h : Hop) : h.swap.holds E σ = h.holds E σ := by
  cases h with
  | mk s d e r t => cases t <;> simp [Hop.swap, Hop.holds]

theorem swap_swap (h : Hop) : h.swap.swap = h := by
  cases h; simp [Hop.swap]

/-- A chain of hops and filters, run as tests in order. -/
inductive Op
  | hop (h : Hop)
  | filter (p : (Nat → Nat) → Bool)

def Op.holds (E : Store) (σ : Nat → Nat) : Op → Bool
  | .hop h => h.holds E σ
  | .filter p => p σ

def run (E : Store) (ops : List Op) (rows : List (Nat → Nat)) : List (Nat → Nat) :=
  rows.filter (fun σ => ops.all (Op.holds E σ))

/-- **PROVEN** (reordering is sound): running any permutation of the same operators selects
the same rows, in the same order. -/
theorem run_perm (E : Store) (ops ops' : List Op) (hp : ops.Perm ops') (rows : List (Nat → Nat)) :
    run E ops rows = run E ops' rows := by
  unfold run
  congr 1
  funext σ
  exact List.Perm.all_eq hp

/-- **PROVEN**: sequencing = conjunction — running `ops₁` then `ops₂` is running `ops₁ ++ ops₂`,
so a filter placed right after the hop that binds its inputs selects what it would at the end. -/
theorem run_append (E : Store) (a b : List Op) (rows : List (Nat → Nat)) :
    run E b (run E a rows) = run E (a ++ b) rows := by
  simp only [run, List.filter_filter, List.all_append]
  congr 1; funext σ; exact Bool.and_comm _ _

/-! ### The greedy ordering (`select_scan_node.rs:914-974`) -/

/-- Pick the first pending hop with an already-bound endpoint (the Rust loop picks the
best-scoring such hop; which one is chosen does not matter for soundness, so the model
takes the first; the scoring is MODELLED only as "some admissible hop"). -/
def pick (bound : List Nat) : List Hop → Option (Hop × List Hop)
  | [] => none
  | h :: hs =>
    if h.src ∈ bound ∨ h.dst ∈ bound then some (h, hs)
    else match pick bound hs with
      | some (h', rest) => some (h', h :: rest)
      | none => none

/-- Orient the picked hop to start from its bound endpoint (`select_scan_node.rs:965-970`). -/
def orient (bound : List Nat) (h : Hop) : Hop :=
  if h.src ∈ bound then h else h.swap

/-- The ordering loop with fuel = number of hops; `none` is `orderable = false`
(the rewrite is abandoned and the plan left alone). -/
def order : Nat → List Nat → List Hop → Option (List Hop)
  | _, _, [] => some []
  | 0, _, _ :: _ => none
  | fuel + 1, bound, hs@(_ :: _) =>
    match pick bound hs with
    | none => none
    | some (h, rest) =>
      let h' := orient bound h
      match order fuel (h'.dst :: h'.edge :: bound) rest with
      | none => none
      | some tail => some (h' :: tail)

theorem pick_perm (bound : List Nat) :
    ∀ hs h rest, pick bound hs = some (h, rest) → hs.Perm (h :: rest)
  | [], _, _, hp => by simp [pick] at hp
  | x :: xs, h, rest, hp => by
    simp only [pick] at hp
    split at hp
    · simp at hp; obtain ⟨rfl, rfl⟩ := hp; exact List.Perm.refl _
    · split at hp
      · next h' rest' heq =>
        simp at hp; obtain ⟨rfl, rfl⟩ := hp
        have := pick_perm bound xs h' rest' heq
        exact (List.Perm.cons x this).trans (List.Perm.swap h' x rest')
      · simp at hp

/-- Undo orientation: every ordered hop is the original or its swap. -/
def canon (h : Hop) : Hop := if h.transposed then h.swap else h

theorem holds_orient (E : Store) (σ : Nat → Nat) (bound : List Nat) (h : Hop) :
    (orient bound h).holds E σ = h.holds E σ := by
  unfold orient; split
  · rfl
  · exact swap_holds E σ h

/-- **PROVEN** (hop ordering is sound): whatever order and orientation the greedy loop
produces, the ordered chain tests exactly the conjunction of the original hops. -/
theorem order_sound (E : Store) (σ : Nat → Nat) :
    ∀ fuel bound hs out, order fuel bound hs = some out →
      out.all (Hop.holds E σ) = hs.all (Hop.holds E σ)
  | _, _, [], out, h => by
    cases ‹Nat› <;> simp [order] at h <;> subst h <;> rfl
  | 0, _, _ :: _, _, h => by simp [order] at h
  | fuel + 1, bound, x :: xs, out, h => by
    simp only [order] at h
    split at h
    · simp at h
    · next hp rest hpk =>
      split at h
      · simp at h
      · next tail ht =>
        simp at h; subst h
        have ih := order_sound E σ fuel _ rest tail ht
        have hperm := pick_perm bound (x :: xs) hp rest hpk
        rw [List.all_cons, ih, holds_orient, ← List.all_cons]
        exact (List.Perm.all_eq hperm).symm

/-- **PROVEN**: the ordered chain has as many hops as the pattern — nothing dropped or
duplicated (`ordered.len() == hop_count`). -/
theorem order_length : ∀ fuel bound hs out, order fuel bound hs = some out → out.length = hs.length
  | _, _, [], out, h => by
    cases ‹Nat› <;> simp [order] at h <;> subst h <;> rfl
  | 0, _, _ :: _, _, h => by simp [order] at h
  | fuel + 1, bound, x :: xs, out, h => by
    simp only [order] at h
    split at h
    · simp at h
    · next hp rest hpk =>
      split at h
      · simp at h
      · next tail ht =>
        simp at h; subst h
        have := order_length fuel _ rest tail ht
        have hperm := pick_perm bound (x :: xs) hp rest hpk
        have hl := hperm.length_eq
        simp at hl
        simp [this, hl]

/-- A pattern `(a)-[e0]->(b)-[e1]->(c)` ordered from `c` (the chain reversal of the header
diagram of `select_scan_node.rs`): both hops come out transposed. -/
example :
    order 2 [2] [⟨0, 1, 10, 0, false⟩, ⟨1, 2, 11, 0, false⟩] =
      some [⟨2, 1, 11, 0, true⟩, ⟨1, 0, 10, 0, true⟩] := by decide

/-- A hop with no bound endpoint cannot be placed: the rewrite is abandoned. -/
example : order 1 [7] [⟨0, 1, 10, 0, false⟩] = none := by decide

/-! ### Labels -/

def hasAll (ls : List Nat) (labels : List Nat) : Bool := ls.all (· ∈ labels)

/-- **PROVEN**: the label check is order-insensitive, so `reorder_labels`' stable sort by schema
id cannot change which nodes a `NodeByLabelScan` yields. -/
theorem hasAll_perm (ls ls' labels : List Nat) (hp : ls.Perm ls') :
    hasAll ls labels = hasAll ls' labels := List.Perm.all_eq hp

/-- Insertion sort by key, stable (`sort_by_key`, `reorder_labels.rs:32`). -/
def insertBy (key : Nat → Nat) (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if key x < key y then x :: y :: ys else y :: insertBy key x ys

def sortBy (key : Nat → Nat) : List Nat → List Nat
  | [] => []
  | x :: xs => insertBy key x (sortBy key xs)

theorem insertBy_perm (key : Nat → Nat) (x : Nat) : ∀ ys, (insertBy key x ys).Perm (x :: ys)
  | [] => List.Perm.refl _
  | y :: ys => by
    simp only [insertBy]; split
    · exact List.Perm.refl _
    · exact ((insertBy_perm key x ys).cons y).trans (List.Perm.swap x y ys)

theorem sortBy_perm (key : Nat → Nat) : ∀ xs, (sortBy key xs).Perm xs
  | [] => List.Perm.refl _
  | x :: xs => (insertBy_perm key x _).trans ((sortBy_perm key xs).cons x)

/-- `reorder_labels` as a function on one scan's label list. -/
def reorderLabels (key : Nat → Nat) (ls : List Nat) : List Nat := sortBy key ls

/-- **PROVEN** (`reorder_labels` is sound). -/
theorem reorderLabels_sound (key : Nat → Nat) (ls labels : List Nat) :
    hasAll (reorderLabels key ls) labels = hasAll ls labels :=
  hasAll_perm _ _ _ (sortBy_perm key ls)

/-- `with_primary_label`: `label` first, then the others in order, without `label`. -/
def withPrimary (label : Nat) (ls : List Nat) : List Nat := label :: ls.filter (· != label)

/-- **PROVEN**: moving the indexed label first keeps the label set — provided the label is one
of the pattern's labels, which `find` over `all_labels()` guarantees. -/
theorem withPrimary_sound (label : Nat) (ls labels : List Nat) (h : label ∈ ls) :
    hasAll (withPrimary label ls) labels = hasAll ls labels := by
  unfold hasAll withPrimary
  rw [Bool.eq_iff_iff]
  simp only [List.all_cons, Bool.and_eq_true, List.all_eq_true, List.mem_filter, bne_iff_ne,
    decide_eq_true_eq]
  constructor
  · rintro ⟨hl, hr⟩ x hx
    by_cases hxl : x = label
    · subst hxl; exact hl
    · exact hr x ⟨hx, hxl⟩
  · intro hall
    exact ⟨hall label h, fun x hx => hall x hx.1⟩

/-- `NodeByIndexScanOp` checks the primary label through the index and the rest
(`labels.skip(1)`) explicitly: together that is the full label set. -/
theorem primary_split (label : Nat) (ls labels : List Nat) (h : label ∈ ls) :
    (decide (label ∈ labels) && hasAll (ls.filter (· != label)) labels) = hasAll ls labels := by
  have := withPrimary_sound label ls labels h
  simpa [hasAll, withPrimary] using this


/-! ### Binding-aware semantics and filter placement (`select_scan_node.rs:893-1117`)

Generate-and-test hides one thing: a `Filter` evaluated before its variable is bound reads
Null and drops the row. `runScoped` is the sequential semantics with a bound set `B`: a hop
binds its endpoints and edge, a filter passes only if every variable it reads is bound. -/

inductive Step
  | hop (h : Hop)
  | filt (vs : List Nat) (p : (Nat → Nat) → Bool)

def runScoped (E : Store) (σ : Nat → Nat) : List Nat → List Step → Bool
  | _, [] => true
  | B, .hop h :: rest => h.holds E σ && runScoped E σ (h.src :: h.dst :: h.edge :: B) rest
  | B, .filt vs p :: rest => (vs.all (· ∈ B) && p σ) && runScoped E σ B rest

/-- Every filter reads only variables bound before it. -/
def wellScoped : List Nat → List Step → Prop
  | _, [] => True
  | B, .hop h :: rest => wellScoped (h.src :: h.dst :: h.edge :: B) rest
  | B, .filt vs _ :: rest => (∀ v ∈ vs, v ∈ B) ∧ wellScoped B rest

def fullHolds (E : Store) (σ : Nat → Nat) : Step → Bool
  | .hop h => h.holds E σ
  | .filt _ p => p σ

/-- **PROVEN**: on a well-scoped chain the sequential (binding-aware) run agrees with
generate-and-test, so `order_sound`/`run_perm` carry over. -/
theorem runScoped_wellScoped (E : Store) (σ : Nat → Nat) :
    ∀ B steps, wellScoped B steps → runScoped E σ B steps = steps.all (fullHolds E σ)
  | _, [], _ => rfl
  | B, .hop h :: rest, hw => by
    simp only [runScoped, List.all_cons, fullHolds]
    rw [runScoped_wellScoped E σ _ rest hw]
  | B, .filt vs p :: rest, ⟨hv, hw⟩ => by
    simp only [runScoped, List.all_cons, fullHolds]
    have : vs.all (· ∈ B) = true := List.all_eq_true.mpr (fun v h => by simpa using hv v h)
    rw [this, runScoped_wellScoped E σ B rest hw, Bool.true_and]

/-- Filter placement after each hop (`select_scan_node.rs:1084-1111`): emit the hop, add its
destination and edge to `placed`, then emit every pending filter whose variables are all in
`placed`. Returns the steps and the filters left over (non-empty ⇒ rewrite abandoned). -/
def place : List Nat → List Hop → List (List Nat × ((Nat → Nat) → Bool)) →
    List Step × List (List Nat × ((Nat → Nat) → Bool))
  | _, [], fs => ([], fs)
  | placed, h :: hs, fs =>
    let placed' := h.dst :: h.edge :: placed
    let now := fs.filter (fun f => f.1.all (· ∈ placed'))
    let later := fs.filter (fun f => !f.1.all (· ∈ placed'))
    let (rest, left) := place placed' hs later
    (.hop h :: now.map (fun f => .filt f.1 f.2) ++ rest, left)

theorem wellScoped_filts (B : List Nat) (fs : List (List Nat × ((Nat → Nat) → Bool)))
    (hf : ∀ f ∈ fs, ∀ v ∈ f.1, v ∈ B) (rest : List Step) (hr : wellScoped B rest) :
    wellScoped B (fs.map (fun f => Step.filt f.1 f.2) ++ rest) := by
  induction fs with
  | nil => simpa using hr
  | cons f fs ih =>
    simp only [List.map_cons, List.cons_append, wellScoped]
    exact ⟨hf f (by simp), ih (fun g hg => hf g (by simp [hg]))⟩

/-- **PROVEN** (placement is sound *if the initial bound set is honest*): when every
variable in `placed` is really bound by the operators below (`placed ⊆ B`), the emitted
chain is well-scoped. -/
theorem place_wellScoped : ∀ (placed B : List Nat) (hs : List Hop) fs,
    (∀ v ∈ placed, v ∈ B) → wellScoped B (place placed hs fs).1
  | _, _, [], _, _ => trivial
  | placed, B, h :: hs, fs, hpl => by
    simp only [place, wellScoped]
    apply wellScoped_filts
    · intro f hf v hv
      have := (List.mem_filter.mp hf).2
      have hv' := List.all_eq_true.mp this v hv
      simp only [decide_eq_true_eq, List.mem_cons] at hv'
      simp only [List.mem_cons]
      rcases hv' with h1 | h1 | h1
      · exact Or.inr (Or.inl h1)
      · exact Or.inr (Or.inr (Or.inl h1))
      · exact Or.inr (Or.inr (Or.inr (hpl v h1)))
    · apply place_wellScoped
      intro v hv
      simp only [List.mem_cons] at hv ⊢
      rcases hv with h1 | h1 | h1
      · exact Or.inr (Or.inl h1)
      · exact Or.inr (Or.inr (Or.inl h1))
      · exact Or.inr (Or.inr (Or.inr (hpl v h1)))

/-! #### Counterexample: the dishonest initial bound set (CONFIRMED bug)

`select_scan_node.rs:893-900` seeds `initial_bound` with `best_node.alias.id` *and* the
variables of an existing (outer-context) child. When that child is kept, no scan for
`best_node` is built (`subtree = existing_child.clone().unwrap_or_else(..)`, line 1080), so
`best_node` is not bound — yet its inline-attribute Filter is placed as soon as the first hop
runs. Query (C returns 1, Rust 0):
`MATCH (b:A) WITH b LIMIT 1 MATCH (b)<-[]-(d)-[:R]->(c {v:1})-[:R]->(z) RETURN count(*)`.
Variables: b=0, d=1, c=2, z=3, edges 10, 11, 12. -/

def exHops : List Hop := [⟨0, 1, 10, 0, true⟩, ⟨1, 2, 11, 0, false⟩, ⟨2, 3, 12, 0, false⟩]
/-- The one edge set: d→b, d→c, c→z (node ids = variable ids here). -/
def exE : Store := fun _ a b e =>
  (a == 1 && b == 0 && e == 10) || (a == 1 && b == 2 && e == 11) || (a == 2 && b == 3 && e == 12)
def exσ : Nat → Nat := id
/-- `c.v = 1` holds for the only match. -/
def exFilter : List Nat × ((Nat → Nat) → Bool) := ([2], fun _ => true)

/-- Honest seed `[b]` (what the Limit child binds): the filter lands after `d→c`, the run
finds the row. -/
example : runScoped exE exσ [0] (place [0] exHops [exFilter]).1 = true := by decide
/-- Rust's seed `[c, b]`: the filter lands after the first hop, before `c` is bound —
the row is lost. -/
example : runScoped exE exσ [0] (place [2, 0] exHops [exFilter]).1 = false := by decide
/-- …although generate-and-test (all hops and the filter) accepts it. -/
example : ((place [2, 0] exHops [exFilter]).1.all (fullHolds exE exσ)) = true := by decide

end OptimizerScan.ScanOrder
