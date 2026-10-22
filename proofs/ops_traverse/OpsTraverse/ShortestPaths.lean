import OpsTraverse.VarLen
/-
# allShortestPaths (BFS + predecessor backtrack) and PathBuilder's edge-list branch

| here | there |
| --- | --- |
| `St`, `relax`       | the BFS body of `AllShortestPathsOp::expand_row` (`runtime/ops/all_shortest_paths.rs:141-265`): cycle case `:215-238`, same-distance predecessor `:240-248`, first discovery `:249-263` |
| `bfs`               | the `while let Some(current) = queue.pop_front()` loop (`:146-265`), fuel-bounded; neighbours via `step` (`:161-191`, self-loop once when undirected) |
| `back`              | the DFS backtrack iterator (`:278-304`): push `edge` then continue at `prev`, stop at `src` with a non-empty list |
| `asp`               | the whole row: reverse unless `is_cycle` (`:284-288`), reverse again for `AllShortestPaths::Reversed` (`:289-291`) |
| `ends`, `buildNodes`| `PathBuilderOp`'s `Value::List` branch (`runtime/ops/path_builder.rs:148-172`): next node = the edge endpoint that differs from the previous node, via `Runtime::get_relationship_endpoints` |

Edge-attribute filters (`:193-208`) are a pure pre-filter of the neighbour
list and are not modelled. *Minimality*, *completeness* and *exactly-once* are proved in
`ASPFold`/`ASPInv`/`ASPStep`/`ASPFinal`/`ASPMain`/`ASPFuel` (`asp_minimal`,
`asp_complete'`, `asp_nodup`; non-cycle, `min_hops = 1` as the planner enforces).
-/

namespace OpsTraverse.ASP

open OpsTraverse.VarLen

structure St where
  dist  : Nat → Option Nat
  preds : List (Nat × Nat × Nat)   -- (node, prev, edge)
  queue : List Nat
  sd    : Option Nat

def upd (f : Nat → Option Nat) (k v : Nat) : Nat → Option Nat :=
  fun x => if x = k then some v else f x

/-- One neighbour `(edge, next)` of `cur` at distance `cd`. -/
def relax (src dst minH maxH : Nat) (isCycle : Bool) (cur cd : Nat) (st : St)
    (p : Nat × Nat) : St :=
  let e := p.1
  let next := p.2
  let nd := cd + 1
  if isCycle && next == src then
    if nd < minH then st
    else match st.sd with
      | some sd => if nd == sd then { st with preds := st.preds ++ [(next, cur, e)] } else st
      | none => { st with sd := some nd, preds := st.preds ++ [(next, cur, e)] }
  else match st.dist next with
    | some ed => if nd == ed then { st with preds := st.preds ++ [(next, cur, e)] } else st
    | none =>
      { dist := upd st.dist next nd
        preds := st.preds ++ [(next, cur, e)]
        sd := if next == dst && minH ≤ nd then some nd else st.sd
        queue := if nd < maxH then st.queue ++ [next] else st.queue }

def bfs (g : Graph) (b : Bool) (src dst minH maxH : Nat) (isCycle : Bool) :
    Nat → St → St
  | 0, st => st
  | fuel + 1, st =>
    match st.queue with
    | [] => st
    | cur :: q =>
      let st := { st with queue := q }
      let cd := (st.dist cur).getD 0
      if st.sd.any (fun sd => sd ≤ cd) || maxH ≤ cd then bfs g b src dst minH maxH isCycle fuel st
      else bfs g b src dst minH maxH isCycle fuel
        ((step g b cur).foldl (relax src dst minH maxH isCycle cur cd) st)

def predsOf (st : St) (v : Nat) : List (Nat × Nat) :=
  (st.preds.filter fun x => x.1 == v).map fun x => x.2

def back (st : St) (src : Nat) : Nat → Nat → List Nat → List (List Nat)
  | 0, _, _ => []
  | fuel + 1, node, acc =>
    if node == src && !acc.isEmpty then [acc]
    else (predsOf st node).flatMap fun p => back st src fuel p.1 (acc ++ [p.2])

/-- `distances = {src: 0}`, `queue = [src]` (`:141-142`). -/
def st0 (src : Nat) : St where
  dist := upd (fun _ => none) src 0
  preds := []
  queue := [src]
  sd := none

def asp (g : Graph) (b : Bool) (src dst minH maxH : Nat) (reversed : Bool)
    (fuel : Nat) : List (List Nat) :=
  let isCycle := src == dst
  let st := bfs g b src dst minH maxH isCycle fuel (st0 src)
  if (predsOf st dst).isEmpty then []
  else (back st src fuel dst []).map fun es =>
    let es := if isCycle then es else es.reverse
    if reversed then es.reverse else es

/-! ## Every predecessor entry is a traversable edge -/

def PredOK (g : Graph) (b : Bool) (st : St) : Prop :=
  ∀ v u e, (v, u, e) ∈ st.preds → (e, v) ∈ step g b u

theorem relax_ok {g : Graph} {b : Bool} {src dst minH maxH : Nat} {isCycle : Bool} {cur cd : Nat}
    {st : St} {p : Nat × Nat} (hst : PredOK g b st) (hp : p ∈ step g b cur) :
    PredOK g b (relax src dst minH maxH isCycle cur cd st p) := by
  have add : PredOK g b { st with preds := st.preds ++ [(p.2, cur, p.1)] } := by
    intro v u e h
    simp only [List.mem_append, List.mem_singleton, Prod.mk.injEq] at h
    rcases h with h | ⟨rfl, rfl, rfl⟩
    · exact hst v u e h
    · exact hp
  unfold relax
  simp only
  split
  · split
    · exact hst
    · split
      · split
        · exact add
        · exact hst
      · intro v u e h
        simp only [List.mem_append, List.mem_singleton, Prod.mk.injEq] at h
        rcases h with h | ⟨rfl, rfl, rfl⟩
        · exact hst v u e h
        · exact hp
  · split
    · split
      · exact add
      · exact hst
    · intro v u e h
      simp only [List.mem_append, List.mem_singleton, Prod.mk.injEq] at h
      rcases h with h | ⟨rfl, rfl, rfl⟩
      · exact hst v u e h
      · exact hp

theorem foldl_relax_ok {g : Graph} {b : Bool} {src dst minH maxH : Nat} {isCycle : Bool}
    {cur cd : Nat} : ∀ (ps : List (Nat × Nat)) (st : St), PredOK g b st →
      (∀ p ∈ ps, p ∈ step g b cur) →
      PredOK g b (ps.foldl (relax src dst minH maxH isCycle cur cd) st)
  | [], _, h, _ => h
  | p :: ps, st, h, hps =>
    foldl_relax_ok ps _ (relax_ok h (hps p (List.mem_cons_self ..)))
      (fun q hq => hps q (List.mem_cons_of_mem _ hq))

theorem bfs_ok {g : Graph} {b : Bool} {src dst minH maxH : Nat} {isCycle : Bool} :
    ∀ (fuel : Nat) (st : St), PredOK g b st →
      PredOK g b (bfs g b src dst minH maxH isCycle fuel st)
  | 0, _, h => h
  | fuel + 1, st, h => by
    unfold bfs
    split
    · exact h
    · simp only
      split
      · exact bfs_ok fuel _ h
      · exact bfs_ok fuel _ (foldl_relax_ok _ _ h (fun _ hq => hq))

/-! ## The backtrack yields reversed walks -/

theorem Walk.snoc {g : Graph} {b : Bool} :
    ∀ {s m t e : Nat} {es : List Nat}, Walk g b s es m → (e, t) ∈ step g b m →
      Walk g b s (es ++ [e]) t
  | _, _, _, _, [], .nil _, h => .cons h (.nil _)
  | _, _, _, _, _ :: _, .cons hp hw, h => .cons hp (Walk.snoc hw h)

theorem back_sound {g : Graph} {b : Bool} {st : St} {src : Nat} (hok : PredOK g b st) :
    ∀ (fuel node : Nat) (acc res : List Nat), res ∈ back st src fuel node acc →
      ∃ w, res = acc ++ w ∧ Walk g b src w.reverse node
  | 0, _, _, _, h => by simp [back] at h
  | fuel + 1, node, acc, res, h => by
    unfold back at h
    by_cases hc : (node == src && !acc.isEmpty) = true
    · rw [if_pos hc] at h
      simp only [List.mem_singleton] at h
      subst h
      simp only [Bool.and_eq_true, beq_iff_eq] at hc
      exact ⟨[], by simp, by rw [hc.1]; exact Walk.nil _⟩
    · rw [if_neg hc] at h
      simp only [List.mem_flatMap] at h
      obtain ⟨⟨u, e⟩, hp, hr⟩ := h
      obtain ⟨w, rfl, hw⟩ := back_sound hok fuel u _ _ hr
      have hstep : (e, node) ∈ step g b u := by
        simp only [predsOf, List.mem_map, List.mem_filter, beq_iff_eq] at hp
        obtain ⟨⟨v, u', e'⟩, ⟨hm, hv⟩, heq⟩ := hp
        simp only [Prod.mk.injEq] at heq
        obtain ⟨rfl, rfl⟩ := heq
        simp only at hv
        subst hv
        exact hok _ _ _ hm
      refine ⟨e :: w, by simp, ?_⟩
      simpa using Walk.snoc hw hstep

/-- **Soundness (non-cycle)**: every path allShortestPaths returns for
`src ≠ dst` is a walk from `src` to `dst` (in the pattern orientation). -/
theorem asp_sound {g : Graph} {b : Bool} {src dst minH maxH fuel : Nat} (hne : src ≠ dst)
    {es : List Nat} (h : es ∈ asp g b src dst minH maxH false fuel) :
    Walk g b src es dst := by
  simp only [asp] at h
  have hsd : (src == dst) = false := by simpa using hne
  simp only [hsd] at h
  split at h
  · simp at h
  · simp only [Bool.false_eq_true, if_false, List.mem_map] at h
    obtain ⟨res, hres, rfl⟩ := h
    have hok := bfs_ok (g := g) (b := b) (src := src) (dst := dst) (minH := minH) (maxH := maxH)
      (isCycle := false) fuel (st0 src) (by intro v u e h; simp [st0] at h)
    obtain ⟨w, rfl, hw⟩ := back_sound hok fuel dst [] res hres
    simpa using hw

/-! ## Counterexamples -/

/-- `0 -e0-> 1 -e1-> 2 -e2-> 0` (the repro graph, ids shifted to 0). -/
def gCycle : Graph := [⟨0, 0, 1⟩, ⟨1, 1, 2⟩, ⟨2, 2, 0⟩]

/-- **Bug** (`lean_ops_traverse::bug_all_shortest_paths_directed_cycle_reversed`): on a
directed cycle the edge list comes back in backtrack order. -/
theorem directed_cycle_backtrack_order : asp gCycle false 0 0 1 10 false 10 = [[2, 1, 0]] := by
  decide

/-- …and that list is not a directed walk from the source: its first edge
`e2 : 2→0` does not leave node 0. -/
theorem directed_cycle_not_a_walk : ¬ Walk gCycle false 0 [2, 1, 0] 0 := by
  intro h
  cases h with
  | cons hp _ => simp [mem_step, gCycle] at hp

/-- Shared with C (`shared_with_c_undirected_asp_cycle_repeats_edge`): the
undirected cycle closes on the edge it left by, so the "path" repeats it. -/
theorem undirected_cycle_repeats_edge :
    asp gCycle true 0 0 1 10 false 10 = [[0, 0], [2, 2]] ∧ ¬ [0, 0].Nodup := by
  refine ⟨by decide, by simp⟩

/-- A diamond `0→1→3`, `0→2→3`: both shortest paths, nothing longer. -/
theorem diamond_all_paths :
    asp [⟨0, 0, 1⟩, ⟨1, 1, 3⟩, ⟨2, 0, 2⟩, ⟨3, 2, 3⟩, ⟨4, 0, 3⟩] false 0 3 1 10 false 10 = [[4]] ∧
    asp [⟨0, 0, 1⟩, ⟨1, 1, 3⟩, ⟨2, 0, 2⟩, ⟨3, 2, 3⟩] false 0 3 1 10 false 10 = [[0, 1], [2, 3]] := by
  decide

/-! ## PathBuilder, `Value::List` branch -/

/-- `Runtime::get_relationship_endpoints`. -/
def ends (g : Graph) (e : Nat) : Nat × Nat :=
  match g.find? (fun x => x.id == e) with
  | some x => (x.src, x.dst)
  | none => (0, 0)

/-- Nodes PathBuilder appends after `prev` for an edge list (`path_builder.rs:148-172`). -/
def buildNodes (g : Graph) : Nat → List Nat → List Nat
  | _, [] => []
  | prev, e :: es =>
    let n := if prev == (ends g e).1 then (ends g e).2 else (ends g e).1
    n :: buildNodes g n es

def lastOr (d : Nat) (l : List Nat) : Nat := l.getLastD d

theorem eq_of_mem_of_id {g : Graph} (hwf : g.WF) :
    ∀ {x y : Edge}, x ∈ g → y ∈ g → x.id = y.id → x = y := by
  induction g with
  | nil => intro x y hx; simp at hx
  | cons a g ih =>
    intro x y hx hy hid
    have hwf' : (g.map Edge.id).Nodup := (List.nodup_cons.1 hwf).2
    have ha : a.id ∉ g.map Edge.id := (List.nodup_cons.1 hwf).1
    rcases List.mem_cons.1 hx with hxa | hx' <;> rcases List.mem_cons.1 hy with hya | hy'
    · rw [hxa, hya]
    · subst hxa; exact absurd (List.mem_map.2 ⟨y, hy', hid.symm⟩) ha
    · subst hya; exact absurd (List.mem_map.2 ⟨x, hx', hid⟩) ha
    · exact ih hwf' hx' hy' hid

theorem ends_of_mem {g : Graph} (hwf : g.WF) {x : Edge} (hx : x ∈ g) :
    ends g x.id = (x.src, x.dst) := by
  unfold ends
  cases hf : g.find? (fun y => y.id == x.id) with
  | none =>
    have := List.find?_eq_none.1 hf x hx
    simp at this
  | some y =>
    have hy := List.mem_of_find?_eq_some hf
    have hid : y.id = x.id := by simpa using List.find?_some hf
    have : y = x := eq_of_mem_of_id hwf hy hx hid
    subst this
    rfl

/-- **PathBuilder is faithful on walks**: for any walk (directed or
undirected) the "other endpoint" rule reconstructs the walk's nodes, ending at
its target — provided edge ids are unique. -/
theorem buildNodes_walk {g : Graph} {b : Bool} (hwf : g.WF) :
    ∀ {s es t}, Walk g b s es t → lastOr s (buildNodes g s es) = t := by
  intro s es t h
  induction h with
  | nil => rfl
  | @cons s n t e es hp hw ih =>
    obtain ⟨x, hx, rfl, hdir⟩ := mem_step.1 hp
    have hend := ends_of_mem hwf hx
    have hn : (if s == (ends g x.id).1 then (ends g x.id).2 else (ends g x.id).1) = n := by
      rw [hend]
      rcases hdir with ⟨h1, h2⟩ | ⟨_, h1, _, h2⟩
      · simp [h1, h2]
      · have : (s == x.src) = false := by
          simp only [beq_eq_false_iff_ne]; exact fun h => h1 h.symm
        simp only [this, Bool.false_eq_true, if_false]; exact h2
    simp only [buildNodes, hn, lastOr, List.getLastD_cons]
    exact ih

/-- The cycle case feeds PathBuilder a backward list: nodes `[2,1,0]` after
`0`, i.e. the directed cycle `0→1→2→0` rendered as `0,2,1,0`. -/
theorem pathbuilder_on_cycle : buildNodes gCycle 0 [2, 1, 0] = [2, 1, 0] := by decide

end OpsTraverse.ASP
