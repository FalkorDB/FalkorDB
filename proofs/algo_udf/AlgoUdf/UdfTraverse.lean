import AlgoUdf.Neighbors
/-! # `graph.traverse` (`js_traverse_impl`, udf/js_classes.rs:676-859)

| here | there |
| --- | --- |
| `TCfg`              | parsed config :683-718 (`direction`, `types`, `labels`, `returnType`, `maxDepth`; `labelOk n` = the label filter :766-774 for neighbour `n`) |
| `edgesOf`           | `collect_edges` + the `types` retain :755-764 (`Neighbors.collectEdges`, `applyTypes`) |
| `TS`                | per-source state: `visited`, `seen_edges` (as `(node, rel)` pairs), `next_frontier`, `edges_to_create` |
| `visitEdge`         | loop body :766-800 |
| `level`             | one depth :749-803 |
| `runLevels`         | `for _ in 0..max_depth` :741-835 (deadline checks are an extra `Err` exit, not modelled) |
| `traverse1`         | one source's inner result array |

Results:
* `traverse_nodes_nodup` — node results never repeat and never contain the start.
* `traverse_both_edges_nodup` — `direction:'both'` edge results never repeat a
  relationship (the `seen_edges` de-dup), given relationship ids identify edges.
* CONFIRMED divergences from C (live 18900 C / 18901 Rust and `rtest/tests/w5.rs`):
  `traverse_self_loop_twice` (edges mode returns a self-loop twice: C `[1,0]`,
  Rust `[1,0,1]`) and `traverse_drops_self_neighbour` (nodes mode omits the start
  even when it is its own neighbour: C `[0,1]`, Rust `[1]`, while Rust's own
  `getNeighbors` gives `[0,1]`).
-/
namespace AlgoUdf.UdfTraverse
open AlgoUdf.Neighbors

structure TCfg where
  dir : Dir
  types : List String
  labelOk : Nat → Bool
  retEdges : Bool
  maxDepth : Nat

/-- Incident edges of `n` after the type filter. -/
def edgesOf (es : List E) (c : TCfg) (n : Nat) : List E := applyTypes c.types (collectEdges es n c.dir)

/-- `dedup_edges = return_type == "edges" && direction == "both"` :745. -/
def dedup (c : TCfg) : Bool := c.retEdges && c.dir == .both

structure TS where
  visited : List Nat
  seen : List (Nat × Nat)
  next : List Nat
  out : List E

def far (nid : Nat) (e : E) : Nat := if e.src = nid then e.dst else e.src

def visitEdge (c : TCfg) (nid : Nat) (st : TS) (e : E) : TS :=
  let nb := far nid e
  if !c.labelOk nb then st else
  let st := if c.retEdges then
      if dedup c then
        (if (nid, e.id) ∈ st.seen then st
         else { st with seen := (nb, e.id) :: (nid, e.id) :: st.seen, out := st.out ++ [e] })
      else { st with out := st.out ++ [e] }
    else st
  if nb ∈ st.visited then st else { st with visited := st.visited ++ [nb], next := st.next ++ [nb] }

def level (es : List E) (c : TCfg) (fr : List Nat) (st : TS) : TS :=
  fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c nid) st) { st with next := [], out := [] }

/-- Returns the node results and the edge results (only one is used per mode). -/
def runLevels (es : List E) (c : TCfg) : Nat → List Nat → TS → List Nat × List E → List Nat × List E
  | 0, _, _, acc => acc
  | d + 1, fr, st, acc =>
    let st' := level es c fr st
    let acc' := if c.retEdges then (acc.1, acc.2 ++ st'.out) else (acc.1 ++ st'.next, acc.2)
    if st'.next = [] then acc' else runLevels es c d st'.next st' acc'

def traverse1 (es : List E) (c : TCfg) (start : Nat) : List Nat × List E :=
  runLevels es c c.maxDepth [start] ⟨[start], [], [], []⟩ ([], [])

/-! ## Node results -/

theorem visitEdge_visited (c : TCfg) (nid : Nat) (st : TS) (e : E) :
    ∃ add, (visitEdge c nid st e).visited = st.visited ++ add ∧ (visitEdge c nid st e).next = st.next ++ add ∧
      (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
  unfold visitEdge
  simp only
  split
  · exact ⟨[], by simp, by simp, by simp, by simp⟩
  · have hv : ∀ st' : TS, st'.visited = st.visited → st'.next = st.next →
        ∃ add, (if far nid e ∈ st'.visited then st'
            else { st' with visited := st'.visited ++ [far nid e], next := st'.next ++ [far nid e] }).visited =
            st.visited ++ add ∧
          (if far nid e ∈ st'.visited then st'
            else { st' with visited := st'.visited ++ [far nid e], next := st'.next ++ [far nid e] }).next =
            st.next ++ add ∧ (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
      intro st' h1 h2
      split
      · exact ⟨[], by simp [h1], by simp [h2], by simp, by simp⟩
      · rename_i hn; rw [h1] at hn
        exact ⟨[far nid e], by simp [h1], by simp [h2], by simpa using hn, by simp⟩
    apply hv
    · split
      · split
        · split <;> rfl
        · rfl
      · rfl
    · split
      · split
        · split <;> rfl
        · rfl
      · rfl

theorem fold_visit_visited (c : TCfg) (nid : Nat) :
    ∀ (l : List E) (st : TS), ∃ add, (l.foldl (visitEdge c nid) st).visited = st.visited ++ add ∧
      (l.foldl (visitEdge c nid) st).next = st.next ++ add ∧ (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
  intro l
  induction l with
  | nil => intro st; exact ⟨[], by simp, by simp, by simp, by simp⟩
  | cons e t ih =>
    intro st
    simp only [List.foldl_cons]
    obtain ⟨a1, h1, h2, h3, h4⟩ := visitEdge_visited c nid st e
    obtain ⟨a2, h5, h6, h7, h8⟩ := ih (visitEdge c nid st e)
    refine ⟨a1 ++ a2, by rw [h5, h1]; simp, by rw [h6, h2]; simp, ?_, ?_⟩
    · intro x hx
      simp only [List.mem_append] at hx
      rcases hx with hx | hx
      · exact h3 x hx
      · intro hv; exact h7 x hx (by rw [h1]; exact List.mem_append_left _ hv)
    · rw [List.nodup_append]
      refine ⟨h4, h8, ?_⟩
      intro a ha b hb hab; subst hab
      exact h7 a hb (by rw [h1]; exact List.mem_append_right _ ha)

theorem level_visited (es : List E) (c : TCfg) :
    ∀ (fr : List Nat) (st : TS), ∃ add, (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c nid) st) st).visited =
        st.visited ++ add ∧
      (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c nid) st) st).next = st.next ++ add ∧
      (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
  intro fr
  induction fr with
  | nil => intro st; exact ⟨[], by simp, by simp, by simp, by simp⟩
  | cons nid t ih =>
    intro st
    simp only [List.foldl_cons]
    obtain ⟨a1, h1, h2, h3, h4⟩ := fold_visit_visited c nid (edgesOf es c nid) st
    obtain ⟨a2, h5, h6, h7, h8⟩ := ih ((edgesOf es c nid).foldl (visitEdge c nid) st)
    refine ⟨a1 ++ a2, by rw [h5, h1]; simp, by rw [h6, h2]; simp, ?_, ?_⟩
    · intro x hx
      simp only [List.mem_append] at hx
      rcases hx with hx | hx
      · exact h3 x hx
      · intro hv; exact h7 x hx (by rw [h1]; exact List.mem_append_left _ hv)
    · rw [List.nodup_append]
      refine ⟨h4, h8, ?_⟩
      intro a ha b hb hab; subst hab
      exact h7 a hb (by rw [h1]; exact List.mem_append_right _ ha)

theorem runLevels_nodes (es : List E) (c : TCfg) (hn : c.retEdges = false) :
    ∀ d fr (st : TS) (acc : List Nat × List E) start, st.visited.Nodup →
      st.visited = start :: acc.1 → (start :: (runLevels es c d fr st acc).1).Nodup := by
  intro d
  induction d with
  | zero => intro fr st acc start hnd hv; simp only [runLevels]; rw [← hv]; exact hnd
  | succ d ih =>
    intro fr st acc start hnd hv
    obtain ⟨add, h1, h2, h3, h4⟩ := level_visited es c fr { st with next := [], out := [] }
    have hvis : (level es c fr st).visited = st.visited ++ add := by simpa [level] using h1
    have hnext : (level es c fr st).next = add := by simpa [level] using h2
    have hnd' : (st.visited ++ add).Nodup := by
      rw [List.nodup_append]; exact ⟨hnd, h4, fun a ha b hb hab => by subst hab; exact h3 a hb ha⟩
    simp only [runLevels, hn, Bool.false_eq_true, if_false]
    split
    · rw [hnext, ← List.cons_append, ← hv]; exact hnd'
    · apply ih _ _ _ start (by rw [hvis]; exact hnd') (by rw [hvis, hv, hnext]; simp)

/-- **Node results never repeat and never contain the start node.** -/
theorem traverse_nodes_nodup (es : List E) (c : TCfg) (hn : c.retEdges = false) (start : Nat) :
    (start :: (traverse1 es c start).1).Nodup :=
  runLevels_nodes es c hn c.maxDepth [start] ⟨[start], [], [], []⟩ ([], []) start (by simp) rfl

/-! ## `direction:'both'` edge de-dup -/

/-- Relationship ids identify edges. -/
def IdsUnique (es : List E) : Prop := ∀ e ∈ es, ∀ e' ∈ es, e.id = e'.id → e = e'

theorem mem_edgesOf_both (es : List E) (c : TCfg) (hd : c.dir = .both) (nid : Nat) (e : E)
    (h : e ∈ edgesOf es c nid) : e ∈ es ∧ (e.src = nid ∨ e.dst = nid) := by
  unfold edgesOf applyTypes at h
  have h' : e ∈ collectEdges es nid c.dir := by
    split at h
    · exact h
    · exact (List.mem_filter.mp h).1
  rw [hd] at h'
  simp only [collectEdges, nodeRels, List.mem_append, List.mem_filter, beq_iff_eq] at h'
  rcases h' with ⟨h1, h2⟩ | ⟨h1, h2⟩
  · exact ⟨h1, Or.inl h2⟩
  · exact ⟨h1, Or.inr h2⟩

/-- Every emitted edge is recorded as seen at both endpoints; ids are distinct. -/
def J (es : List E) (emitted : List E) (st : TS) : Prop :=
  (emitted.map (·.id)).Nodup ∧ ∀ e ∈ emitted, e ∈ es ∧ (e.src, e.id) ∈ st.seen ∧ (e.dst, e.id) ∈ st.seen

theorem J_of {es em em' : List E} {st X : TS} (h : J es em st) (hs : X.seen = st.seen) (ho : em' = em) :
    J es em' X := by
  subst ho; unfold J at *; rw [hs]; exact h

theorem visitEdge_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) (nid : Nat) (accE : List E) (st : TS) (e : E) (he : e ∈ edgesOf es c nid)
    (hj : J es (accE ++ st.out) st) : J es (accE ++ (visitEdge c nid st e).out) (visitEdge c nid st e) := by
  obtain ⟨hes, hinc⟩ := mem_edgesOf_both es c hd nid e he
  have hdd : dedup c = true := by simp [dedup, hr, hd]
  -- the visited/next update never touches `seen`/`out`
  have tail : ∀ st' : TS, (if far nid e ∈ st'.visited then st'
      else { st' with visited := st'.visited ++ [far nid e], next := st'.next ++ [far nid e] }).out = st'.out ∧
      (if far nid e ∈ st'.visited then st'
      else { st' with visited := st'.visited ++ [far nid e], next := st'.next ++ [far nid e] }).seen = st'.seen := by
    intro st'; split <;> exact ⟨rfl, rfl⟩
  unfold visitEdge
  simp only [hr, hdd, if_true]
  split
  · exact hj
  · split
    · obtain ⟨t1, t2⟩ := tail st; exact J_of hj t2 (by simp [t1])
    · rename_i hns
      obtain ⟨t1, t2⟩ := tail { st with seen := (far nid e, e.id) :: (nid, e.id) :: st.seen, out := st.out ++ [e] }
      refine J_of (st := { st with seen := (far nid e, e.id) :: (nid, e.id) :: st.seen, out := st.out ++ [e] })
        (em := accE ++ (st.out ++ [e])) ?_ t2 (by simp [t1])
      obtain ⟨hn, hall⟩ := hj
      have hfar : (far nid e = e.src ∧ nid = e.dst) ∨ (far nid e = e.dst ∧ nid = e.src) := by
        unfold far; split
        · right; rename_i h; exact ⟨rfl, h.symm⟩
        · left; rename_i h; rcases hinc with h' | h'
          · exact absurd h' h
          · exact ⟨rfl, h'.symm⟩
      refine ⟨?_, ?_⟩
      · rw [← List.append_assoc, List.map_append, List.nodup_append]
        refine ⟨hn, by simp, ?_⟩
        intro a ha b hb hab
        simp at hb; subst hb
        obtain ⟨e', he', rfl⟩ := List.mem_map.mp ha
        have := hu e' (hall e' he').1 e hes hab; subst this
        rcases hinc with h | h
        · exact hns (h ▸ (hall e' he').2.1)
        · exact hns (h ▸ (hall e' he').2.2)
      · intro e' he'
        rw [← List.append_assoc, List.mem_append] at he'
        rcases he' with he' | he'
        · obtain ⟨h1, h2, h3⟩ := hall e' he'
          exact ⟨h1, List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h2),
            List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h3)⟩
        · simp at he'; subst he'
          rcases hfar with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1] <;> subst h2 <;> exact ⟨hes, by simp, by simp⟩

theorem level_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) (accE : List E) :
    ∀ (fr : List Nat) (st : TS), J es (accE ++ st.out) st →
      J es (accE ++ (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c nid) st) st).out)
        (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c nid) st) st) := by
  intro fr
  induction fr with
  | nil => intro st h; exact h
  | cons nid t ih =>
    intro st h
    simp only [List.foldl_cons]
    apply ih
    have : ∀ (l : List E) (st : TS), (∀ e ∈ l, e ∈ edgesOf es c nid) → J es (accE ++ st.out) st →
        J es (accE ++ (l.foldl (visitEdge c nid) st).out) (l.foldl (visitEdge c nid) st) := by
      intro l
      induction l with
      | nil => intro st _ h; exact h
      | cons e t' ih' =>
        intro st hl h
        simp only [List.foldl_cons]
        exact ih' _ (fun e' h' => hl e' (List.mem_cons_of_mem _ h'))
          (visitEdge_J es c hd hr hu nid accE st e (hl e (by simp)) h)
    exact this _ st (fun _ h => h) h

theorem runLevels_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) :
    ∀ d fr (st : TS) (acc : List Nat × List E), J es acc.2 st →
      ((runLevels es c d fr st acc).2.map (·.id)).Nodup := by
  intro d
  induction d with
  | zero => intro fr st acc h; exact h.1
  | succ d ih =>
    intro fr st acc h
    have hl := level_J es c hd hr hu acc.2 fr { st with next := [], out := [] }
      (J_of h rfl (by simp))
    simp only [runLevels, hr, if_true]
    split
    · exact hl.1
    · exact ih _ _ _ hl

/-- **`direction:'both'` edge results never repeat a relationship.** -/
theorem traverse_both_edges_nodup (es : List E) (c : TCfg) (hd : c.dir = .both)
    (hr : c.retEdges = true) (hu : IdsUnique es) (start : Nat) :
    ((traverse1 es c start).2.map (·.id)).Nodup :=
  runLevels_J es c hd hr hu c.maxDepth [start] _ ([], []) ⟨by simp, by simp⟩

/-! ## Divergences from C -/

/-- e0 a→b, e1 a→a, e2 b→a; `labels`/`types` unset; depth 1. -/
def selfLoopGraph : List E := [⟨0, 0, 1, "R"⟩, ⟨1, 0, 0, "R"⟩, ⟨2, 1, 0, "R"⟩]

def cfg1 (retEdges : Bool) : TCfg := ⟨.outgoing, [], fun _ => true, retEdges, 1⟩

/-- BUG (confirmed): edges mode returns the self-loop e1 twice (C: [1, 0]). -/
theorem traverse_self_loop_twice :
    ((traverse1 selfLoopGraph (cfg1 true) 0).2.map (·.id)) = [0, 1, 1] := by decide

/-- BUG (confirmed): nodes mode omits the start although it is its own
neighbour (C: [0, 1]); `getNeighbors` on the same node gives [1, 0] as a set {0, 1}. -/
theorem traverse_drops_self_neighbour :
    (traverse1 selfLoopGraph (cfg1 false) 0).1 = [1] ∧
    neighborNodes 0 (collectEdges selfLoopGraph 0 .outgoing) [] = [1, 0] := by decide

end AlgoUdf.UdfTraverse
