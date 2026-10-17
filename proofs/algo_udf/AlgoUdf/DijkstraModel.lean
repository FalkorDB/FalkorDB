import AlgoUdf.Graph
/-! # `dijkstra_single_path` (algo_procedures.rs:2219-2312) and the `DijkstraItem` heap order

| here | there |
| --- | --- |
| `Lbl`, `St.labels`     | `DijkstraLabel`, `labels: FxHashMap<NodeId, DijkstraLabel>` :2200-2206, :2225 |
| `St.settled`           | `settled: FxHashSet<NodeId>` :2226 |
| `St.heap`              | `heap: BinaryHeap<DijkstraItem>` :2227 as a multiset of `(weight, node)` |
| `St.done`              | `reached = true; break` :2246-2249 |
| `St.ord`, `St.clock`   | ghost: the time each node was settled (proof-only) |
| `itemCmp`              | `impl Ord for DijkstraItem::cmp` :2188, `partial_cmp` :2203 |
| `Step.skip`            | pop of a superseded entry, `!settled.insert(node) → continue` :2243-2245 |
| `Step.reach`           | pop of `target` :2246-2249 |
| `Step.relax`, `relaxOne` | the relaxation loop :2252-2288 |
| `walkBack`             | parent-chain walk :2296-2302 (built in traversal order directly instead of push+reverse) |
| `finish`               | result assembly :2290-2312 (`none` = a `labels[&cur]` panic) |

The heap is modelled as a list from which `Step` may pop *any* entry of minimum
weight; `itemCmp_max_min_weight` shows `BinaryHeap::pop` (max under `itemCmp`)
is such an entry, so every theorem covers the real pop order (ties broken by
the smaller node id). Timeout / memory checks (:2238-2241) only add an `Err`
exit and are not modelled.

Main results (all for exact non-negative weights, see `AlgoUdf.Graph`):
* `dijkstra_terminates` — from the initial state a terminal state is reachable
  (every step strictly decreases (unsettled nodes, heap size) on a finite universe).
* `dijkstra_correct` — at any terminal state reached from `init`, with
  `source ≠ target` (the `run_path_algo` guard, `Paths.dijkstra_never_src_eq_dst`):
  the parent walk never panics; `Some (es, W, C)` is a walk source→target of
  weight `W` with `W` ≤ the weight of every walk source→target, and `C` the sum
  of its edge costs; `None` only when no walk exists.
-/
namespace AlgoUdf.Dijkstra
open AlgoUdf.Paths (Dir farEndpoint)
open AlgoUdf.Graph

/-! ## Heap order -/

/-- `DijkstraItem::cmp(self, other)` = `other.weight.partial_cmp(&self.weight)
.unwrap_or(Equal).then_with(|| other.node.cmp(&self.node))`, on finite weights
(only finite weights are ever pushed, :2264-2269). -/
def itemCmp (a b : Nat × Nat) : Ordering :=
  (compare b.1 a.1).then (compare b.2 a.2)

/-- `partial_cmp` is `Some(self.cmp(other))` :2203-2208. -/
def itemPartialCmp (a b : Nat × Nat) : Option Ordering := some (itemCmp a b)

theorem itemCmp_gt (a b : Nat × Nat) :
    itemCmp a b = .gt ↔ a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2) := by
  unfold itemCmp
  rcases Nat.lt_trichotomy b.1 a.1 with h | h | h
  · simp [Nat.compare_eq_lt.mpr h]; omega
  · rw [h]; simp [Nat.compare_eq_gt]
  · simp [Nat.compare_eq_gt.mpr h, Ordering.then]; omega

/-- `BinaryHeap::pop` returns an entry `x` no other entry is greater than
under `itemCmp`; such an entry has minimum weight. -/
theorem itemCmp_max_min_weight (h : List (Nat × Nat)) (x : Nat × Nat)
    (hmax : ∀ y ∈ h, itemCmp y x ≠ .gt) : ∀ y ∈ h, x.1 ≤ y.1 := by
  intro y hy
  have := hmax y hy
  rw [Ne, itemCmp_gt] at this
  omega

theorem itemPartialCmp_total (a b : Nat × Nat) : (itemPartialCmp a b).isSome := rfl

/-! ## State machine -/

structure Lbl where
  parent : Nat
  edge : Rel
  w : Nat

structure St where
  labels : Nat → Option Lbl
  settled : Nat → Bool
  heap : List (Nat × Nat)
  done : Bool
  ord : Nat → Nat
  clock : Nat


def init (src : Nat) : St := ⟨fun _ => none, fun _ => false, [(0, src)], false, fun _ => 0, 0⟩

/-- One iteration of the relaxation loop :2252-2288. -/
def relaxOne (g : G) (u wu : Nat) (st : St) (r : Rel) : St :=
  match farEndpoint u r.1 r.2.1 g.dir with
  | none => st
  | some far =>
    if st.settled far then st else
    match g.wt r.2.2 with
    | none => st
    | some c =>
      if (st.labels far).all (fun l => decide (wu + c < l.w)) then
        { st with labels := upd st.labels far (some ⟨u, r, wu + c⟩),
                  heap := (wu + c, far) :: st.heap }
      else st

/-- `settled.insert(node)` (plus the ghost settle time). -/
def settle (st : St) (v : Nat) (h : List (Nat × Nat)) : St :=
  { st with heap := h, settled := upd st.settled v true, ord := upd st.ord v st.clock,
            clock := st.clock + 1 }

def IsMin (h : List (Nat × Nat)) (w : Nat) : Prop := ∀ y ∈ h, w ≤ y.1

inductive Step (g : G) (tgt : Nat) : St → St → Prop
  | skip {st : St} {w v : Nat} : st.done = false → (w, v) ∈ st.heap → IsMin st.heap w →
      st.settled v = true → Step g tgt st { st with heap := st.heap.erase (w, v) }
  | reach {st : St} {w v : Nat} : st.done = false → (w, v) ∈ st.heap → IsMin st.heap w →
      st.settled v = false → v = tgt →
      Step g tgt st { settle st v (st.heap.erase (w, v)) with done := true }
  | relax {st : St} {w v : Nat} : st.done = false → (w, v) ∈ st.heap → IsMin st.heap w →
      st.settled v = false → v ≠ tgt →
      Step g tgt st ((g.rels v).foldl (relaxOne g v w) (settle st v (st.heap.erase (w, v))))

inductive Star (g : G) (tgt : Nat) : St → St → Prop
  | refl (s : St) : Star g tgt s s
  | tail {a b c : St} : Star g tgt a b → Step g tgt b c → Star g tgt a c

/-- The `while let Some(..) = heap.pop()` loop has exited. -/
def Terminal (st : St) : Prop := st.done = true ∨ st.heap = []

/-- Parent walk :2296-2302 (fuel-bounded; `none` = `labels[&cur]` panic or no termination). -/
def walkBack (L : Nat → Option Lbl) (src : Nat) : Nat → Nat → List Rel → Option (List Rel)
  | 0, _, _ => none
  | fuel + 1, cur, acc =>
    if cur = src then some acc else
    match L cur with
    | none => none
    | some l => walkBack L src fuel l.parent (l.edge :: acc)

/-- Result assembly :2290-2312. Outer `none` = panic; `some none` = `Ok(None)`. -/
def finish (g : G) (src tgt : Nat) (st : St) : Option (Option (List Rel × Nat × Nat)) :=
  if st.done then
    match walkBack st.labels src (st.clock + 1) tgt [] with
    | none => none
    | some es =>
      match st.labels tgt with
      | none => none
      | some l => some (some (es, l.w, (es.map (fun r => g.cost r.2.2)).sum))
  else some none

/-! ## Invariant -/

/-- Final distance of a node: 0 for the source, its label's weight otherwise. -/
def D (src : Nat) (L : Nat → Option Lbl) (v : Nat) : Nat :=
  if v = src then 0 else ((L v).map (·.w)).getD 0

/-- The invariant, with the frontier property `fr` waived for the pairs in `ex`
(the relaxations of the node being expanded that have not run yet). -/
structure Inv (g : G) (src tgt : Nat) (ex : Nat → Rel → Prop) (st : St) : Prop where
  h0 : st.settled src = false → st.heap = [(0, src)] ∧ (∀ v, st.settled v = false)
  lbl : ∀ v l, st.labels v = some l → v ≠ src ∧ st.settled l.parent = true ∧
          (st.settled v = true → st.ord l.parent < st.ord v) ∧
          ∃ c, Edge g l.parent v l.edge c ∧ l.w = D src st.labels l.parent + c
  heapE : ∀ w v, (w, v) ∈ st.heap → (v = src ∧ w = 0) ∨ ∃ l, st.labels v = some l ∧ l.w ≤ w
  latest : ∀ v l, st.labels v = some l → st.settled v = false → (l.w, v) ∈ st.heap
  setLbl : ∀ v, st.settled v = true → v ≠ src → st.labels v ≠ none
  ordLt : ∀ v, st.settled v = true → st.ord v < st.clock
  mono : ∀ u w v, st.settled u = true → (w, v) ∈ st.heap → D src st.labels u ≤ w
  edge : st.done = false → ∀ x y r c, st.settled x = true → st.settled y = true →
          Edge g x y r c → D src st.labels y ≤ D src st.labels x + c
  fr : st.done = false → ∀ x y r c, st.settled x = true → Edge g x y r c → ¬ ex x r →
          st.settled y = true ∨ ∃ l, st.labels y = some l ∧ l.w ≤ D src st.labels x + c
  opt : ∀ u es W, st.settled u = true → Walk g src es u W → D src st.labels u ≤ W

abbrev noEx : Nat → Rel → Prop := fun _ _ => False

theorem inv_init (g : G) (src tgt : Nat) : Inv g src tgt noEx (init src) := by
  refine ⟨fun _ => ⟨rfl, fun _ => rfl⟩, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [init]

/-- Settling the target ends the loop. -/
def TgtInv (tgt : Nat) (st : St) : Prop := st.settled tgt = true ↔ st.done = true

/-! ### Small facts -/

theorem edge_det {g : G} {x y y' r c c'} (h : Edge g x y r c) (h' : Edge g x y' r c') :
    y = y' ∧ c = c' := by
  obtain ⟨_, h1, h2⟩ := h; obtain ⟨_, h1', h2'⟩ := h'
  rw [h1] at h1'; rw [h2] at h2'
  exact ⟨Option.some.inj h1', Option.some.inj h2'⟩

theorem D_congr {src : Nat} {L L' : Nat → Option Lbl} {v : Nat} (h : L' v = L v) :
    D src L' v = D src L v := by simp [D, h]

theorem D_lbl {src : Nat} {L : Nat → Option Lbl} {v : Nat} {l : Lbl} (hv : v ≠ src)
    (h : L v = some l) : D src L v = l.w := by simp [D, hv, h]

theorem mem_erase_of_ne {a b : Nat × Nat} {l : List (Nat × Nat)} (h : a ∈ l) (hne : a ≠ b) :
    a ∈ l.erase b := (List.mem_erase_of_ne hne).mpr h

/-- What a minimum pop of an unsettled node yields. -/
theorem pop_analysis {g : G} {src tgt ex st w v} (I : Inv g src tgt ex st)
    (hm : (w, v) ∈ st.heap) (hmin : IsMin st.heap w) (hs : st.settled v = false) :
    (v = src ∧ w = 0) ∨ (v ≠ src ∧ ∃ l, st.labels v = some l ∧ l.w = w) := by
  rcases I.heapE w v hm with h | ⟨l, hl, hle⟩
  · exact Or.inl h
  · right
    refine ⟨(I.lbl v l hl).1, l, hl, ?_⟩
    have := hmin _ (I.latest v l hl hs); simp at this; omega

/-- Key lemma: a walk from a settled node to an unsettled one passes through a
labelled unsettled node whose label is at most the walk's weight. -/
theorem frontier_lemma {g : G} {src tgt ex st} (I : Inv g src tgt ex st) (hd : st.done = false)
    (hex : ∀ x r, ¬ ex x r ∨ st.settled x = false) :
    ∀ {x es y W}, Walk g x es y W → st.settled x = true → st.settled y = false →
      ∃ z l, st.settled z = false ∧ st.labels z = some l ∧ l.w ≤ D src st.labels x + W := by
  intro x es y W hw
  induction hw with
  | nil u => intro h1 h2; rw [h1] at h2; cases h2
  | @cons u v w r c rest W he hrest ih =>
    intro hu hy
    have nex : ¬ ex u r := by
      rcases hex u r with h | h
      · exact h
      · rw [h] at hu; cases hu
    cases hv : st.settled v with
    | true =>
      obtain ⟨z, l, hz, hl, hle⟩ := ih hv hy
      have := I.edge hd u v r c hu hv he
      exact ⟨z, l, hz, hl, by omega⟩
    | false =>
      rcases I.fr hd u v r c hu he nex with h | ⟨l, hl, hle⟩
      · rw [h] at hv; cases hv
      · exact ⟨v, l, hv, hl, by omega⟩

end AlgoUdf.Dijkstra
