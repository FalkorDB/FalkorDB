import AlgoUdf.Neighbors
/-! # `graph.traverse` (`js_traverse_impl`, udf/js_classes.rs:692-891)

| here | there |
| --- | --- |
| `TCfg`              | parsed config :697-733 (`direction`, `types`, `labels`, `returnType`, `maxDepth`; `labelOk n` = the label filter :806-814 for neighbour `n`) |
| `collectOnce`       | `collect_edges` :103 = `get_node_relationships_by_type(.., &[], dir)` (graph.rs:2485): outgoing edges, then incoming ones, a self-loop once (`Queries.byType_*`); the per-matrix order is abstracted to list order |
| `edgesOf`           | `collectOnce` + the `types` retain :793-801 (`Neighbors.applyTypes`) |
| `TS`                | per-source state: `visited` (start pre-marked :764-765), `seen_edges` (as `(node, rel)` pairs), `next_frontier`, `edges_to_create`, `start_reported` :766, `self_neighbour` :786 |
| `emit`              | the `returnType == "edges"` block :816-834 (`dedup_edges = edges && both` :772) |
| `mark`              | the `visited.insert` / self-loop branch :836-844 |
| `visitEdge`         | loop body :803-845 |
| `level`             | one depth :777-881 |
| `runLevels`         | `for _ in 0..max_depth` from :777; nodes mode reports `start` first when `self_neighbour` :856-867 (deadline checks are an extra `Err` exit, not modelled) |
| `traverse1`         | one source's inner result array |

`Neighbors.collectEdges` (the `get_node_relationships` model, a self-loop twice) is
not what `collect_edges` calls any more, so the traversal uses `collectOnce`.

Results (PR #3253 @ 2e3468240: the start stays pre-marked visited and is
reported once, at depth 1, only through a self-loop on it):
* `traverse_nodes_nodup` — node results never repeat.
* `traverse_start_needs_self_loop` — the start is reported only if it has a
  self-loop that passes the `types`/`labels` filters;
  `traverse_nodes_nodup_no_self_loop` — without one, the old statement
  (`start :: result` has no duplicates) still holds.
* `traverse_both_edges_nodup` — `direction:'both'` edge results never repeat a
  relationship (the `seen_edges` de-dup), given relationship ids identify edges.
* `traverse_self_loop_once`, `traverse_keeps_self_neighbour` — the former C
  divergences (edges mode `[1,0,1]`, nodes mode `[1]`) give C's sets now;
  `traverse_cycle_skips_start` — walking back to the start over a cycle does
  not report it.
-/
namespace AlgoUdf.UdfTraverse
open AlgoUdf.Neighbors

structure TCfg where
  dir : Dir
  types : List String
  labelOk : Nat → Bool
  retEdges : Bool
  maxDepth : Nat

/-- `collect_edges`: a self-loop once in every direction. -/
def collectOnce (es : List E) (v : Nat) : Dir → List E
  | .outgoing => es.filter (·.src == v)
  | .incoming => es.filter (·.dst == v)
  | .both => es.filter (·.src == v) ++ es.filter (fun e => e.dst == v && e.src != v)

/-- Incident edges of `n` after the type filter. -/
def edgesOf (es : List E) (c : TCfg) (n : Nat) : List E := applyTypes c.types (collectOnce es n c.dir)

/-- `dedup_edges = return_type == "edges" && direction == "both"` :772. -/
def dedup (c : TCfg) : Bool := c.retEdges && c.dir == .both

structure TS where
  visited : List Nat
  seen : List (Nat × Nat)
  next : List Nat
  out : List E
  rep : Bool
  selfN : Bool

def far (nid : Nat) (e : E) : Nat := if e.src = nid then e.dst else e.src

def emit (c : TCfg) (nid nb : Nat) (st : TS) (e : E) : TS :=
  if c.retEdges then
    if dedup c then
      (if (nid, e.id) ∈ st.seen then st
       else { st with seen := (nb, e.id) :: (nid, e.id) :: st.seen, out := st.out ++ [e] })
    else { st with out := st.out ++ [e] }
  else st

def mark (start nid nb : Nat) (st : TS) : TS :=
  if nb ∈ st.visited then
    (if nb = start ∧ nid = start ∧ st.rep = false then { st with rep := true, selfN := true } else st)
  else { st with visited := st.visited ++ [nb], next := st.next ++ [nb] }

def visitEdge (c : TCfg) (start nid : Nat) (st : TS) (e : E) : TS :=
  if !c.labelOk (far nid e) then st else mark start nid (far nid e) (emit c nid (far nid e) st e)

def level (es : List E) (c : TCfg) (start : Nat) (fr : List Nat) (st : TS) : TS :=
  fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c start nid) st)
    { st with next := [], out := [], selfN := false }

/-- Returns the node results and the edge results (only one is used per mode). -/
def runLevels (es : List E) (c : TCfg) (start : Nat) :
    Nat → List Nat → TS → List Nat × List E → List Nat × List E
  | 0, _, _, acc => acc
  | d + 1, fr, st, acc =>
    let st' := level es c start fr st
    let acc' := if c.retEdges then (acc.1, acc.2 ++ st'.out)
      else (acc.1 ++ ((if st'.selfN then [start] else []) ++ st'.next), acc.2)
    if st'.next = [] then acc' else runLevels es c start d st'.next st' acc'

def traverse1 (es : List E) (c : TCfg) (start : Nat) : List Nat × List E :=
  runLevels es c start c.maxDepth [start] ⟨[start], [], [], [], false, false⟩ ([], [])

/-! ## Node results -/

/-- The start has a self-loop that survives the `types` and `labels` filters. -/
def SelfLoop (es : List E) (c : TCfg) (start : Nat) : Prop :=
  ∃ e ∈ edgesOf es c start, far start e = start ∧ c.labelOk start = true

/-- One visit step: `visited`/`next` grow by the same fresh nodes; the flags are
untouched, or `start_reported` flips (once) together with `self_neighbour`. -/
def Step (P : Prop) (st st' : TS) : Prop :=
  ∃ add, st'.visited = st.visited ++ add ∧ st'.next = st.next ++ add ∧
    (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup ∧
    ((st'.rep = st.rep ∧ st'.selfN = st.selfN) ∨
      (st.rep = false ∧ st'.rep = true ∧ st'.selfN = true ∧ P))

theorem Step.refl (P : Prop) (st : TS) : Step P st st :=
  ⟨[], by simp, by simp, by simp, by simp, Or.inl ⟨rfl, rfl⟩⟩

theorem Step.trans {P : Prop} {a b c : TS} (h1 : Step P a b) (h2 : Step P b c) : Step P a c := by
  obtain ⟨a1, v1, n1, f1, d1, g1⟩ := h1
  obtain ⟨a2, v2, n2, f2, d2, g2⟩ := h2
  refine ⟨a1 ++ a2, by rw [v2, v1]; simp, by rw [n2, n1]; simp, ?_, ?_, ?_⟩
  · intro x hx
    simp only [List.mem_append] at hx
    rcases hx with hx | hx
    · exact f1 x hx
    · intro hv; exact f2 x hx (by rw [v1]; exact List.mem_append_left _ hv)
  · rw [List.nodup_append]
    refine ⟨d1, d2, ?_⟩
    intro a ha b hb hab; subst hab
    exact f2 a hb (by rw [v1]; exact List.mem_append_right _ ha)
  · rcases g1 with ⟨r1, s1⟩ | ⟨r1, r1', s1, p1⟩ <;> rcases g2 with ⟨r2, s2⟩ | ⟨r2, r2', s2, p2⟩
    · exact Or.inl ⟨r2.trans r1, s2.trans s1⟩
    · exact Or.inr ⟨r1 ▸ r2, r2', s2, p2⟩
    · exact Or.inr ⟨r1, r2.trans r1', s2.trans s1, p1⟩
    · rw [r1'] at r2; cases r2

theorem fold_step {P : Prop} {α : Type} (f : TS → α → TS) :
    ∀ (l : List α), (∀ a ∈ l, ∀ st, Step P st (f st a)) → ∀ st, Step P st (l.foldl f st) := by
  intro l
  induction l with
  | nil => intro _ st; exact Step.refl P st
  | cons a t ih =>
    intro h st
    simp only [List.foldl_cons]
    exact (h a (by simp) st).trans (ih (fun b hb => h b (List.mem_cons_of_mem _ hb)) _)

theorem emit_eq (c : TCfg) (nid nb : Nat) (st : TS) (e : E) :
    (emit c nid nb st e).visited = st.visited ∧ (emit c nid nb st e).next = st.next ∧
      (emit c nid nb st e).rep = st.rep ∧ (emit c nid nb st e).selfN = st.selfN := by
  unfold emit; split
  · split
    · split <;> exact ⟨rfl, rfl, rfl, rfl⟩
    · exact ⟨rfl, rfl, rfl, rfl⟩
  · exact ⟨rfl, rfl, rfl, rfl⟩

theorem visitEdge_step (es : List E) (c : TCfg) (start nid : Nat) (e : E)
    (he : e ∈ edgesOf es c nid) (st : TS) :
    Step (SelfLoop es c start) st (visitEdge c start nid st e) := by
  unfold visitEdge
  split
  · exact Step.refl _ st
  · rename_i hl
    obtain ⟨ev, en, er, es'⟩ := emit_eq c nid (far nid e) st e
    unfold mark
    rw [ev]
    split
    · split
      · rename_i h
        obtain ⟨h1, h2, h3⟩ := h
        refine ⟨[], by simp, by simp [en], by simp, by simp, Or.inr ⟨by rw [← er]; exact h3, rfl, rfl, ?_⟩⟩
        subst h2
        exact ⟨e, he, h1, by rw [h1] at hl; simpa using hl⟩
      · exact ⟨[], by simp [ev], by simp [en], by simp, by simp, Or.inl ⟨er, es'⟩⟩
    · rename_i hn
      exact ⟨[far nid e], by simp, by simp [en], by simpa using hn, by simp, Or.inl ⟨er, es'⟩⟩

theorem level_step (es : List E) (c : TCfg) (start : Nat) (fr : List Nat) (st : TS) :
    Step (SelfLoop es c start) { st with next := [], out := [], selfN := false } (level es c start fr st) := by
  unfold level
  apply fold_step
  intro nid _ st'
  apply fold_step
  intro e he st''
  exact visitEdge_step es c start nid e he st''

/-- Node-mode invariant between depths. -/
def NInv (es : List E) (c : TCfg) (start : Nat) (st : TS) (out : List Nat) : Prop :=
  st.visited.Nodup ∧ start ∈ st.visited ∧ out.Nodup ∧ (∀ x ∈ out, x ∈ st.visited) ∧
    (start ∈ out → st.rep = true) ∧ (st.rep = true → SelfLoop es c start)

theorem level_NInv (es : List E) (c : TCfg) (start : Nat) (fr : List Nat) (st : TS) (out : List Nat)
    (h : NInv es c start st out) :
    NInv es c start (level es c start fr st)
      (out ++ ((if (level es c start fr st).selfN then [start] else []) ++ (level es c start fr st).next)) := by
  obtain ⟨hnd, hst, hout, hsub, hrep, hsl⟩ := h
  obtain ⟨add, hv, hn, hf, hd, hg⟩ := level_step es c start fr st
  simp only [List.nil_append] at hn
  generalize level es c start fr st = L at hv hn hg ⊢
  simp only at hv hf hg
  subst hn
  have hsa : start ∉ L.next := fun h => hf _ h hst
  refine ⟨?_, by rw [hv]; exact List.mem_append_left _ hst, ?_, ?_, ?_, ?_⟩
  · rw [hv, List.nodup_append]
    exact ⟨hnd, hd, fun a ha b hb hab => by subst hab; exact hf a hb ha⟩
  · rw [List.nodup_append]
    refine ⟨hout, ?_, ?_⟩
    · split
      · exact List.nodup_cons.mpr ⟨hsa, hd⟩
      · simpa using hd
    · intro a ha b hb hab; subst hab
      simp only [List.mem_append] at hb
      rcases hb with hb | hb
      · split at hb
        next hsN =>
          simp at hb; subst hb
          rcases hg with ⟨_, hs⟩ | ⟨hr, _⟩
          · rw [hs] at hsN; cases hsN
          · have := hrep ha; rw [hr] at this; cases this
        next => simp at hb
      · exact hf a hb (hsub a ha)
  · intro x hx
    rw [hv]
    simp only [List.mem_append] at hx
    rcases hx with hx | hx | hx
    · exact List.mem_append_left _ (hsub x hx)
    · split at hx
      · simp at hx; subst hx; exact List.mem_append_left _ hst
      · simp at hx
    · exact List.mem_append_right _ hx
  · intro hx
    simp only [List.mem_append] at hx
    rcases hg with ⟨hr, hs⟩ | ⟨_, hr, _, _⟩
    · rcases hx with hx | hx | hx
      · rw [hr]; exact hrep hx
      · rw [hs] at hx; simp at hx
      · exact absurd hx hsa
    · exact hr
  · intro hr
    rcases hg with ⟨hr', _⟩ | ⟨_, _, _, hp⟩
    · rw [hr'] at hr; exact hsl hr
    · exact hp

theorem runLevels_NInv (es : List E) (c : TCfg) (start : Nat) (hn : c.retEdges = false) :
    ∀ d fr (st : TS) (acc : List Nat × List E), NInv es c start st acc.1 →
      ∃ st', NInv es c start st' (runLevels es c start d fr st acc).1 := by
  intro d
  induction d with
  | zero => intro fr st acc h; exact ⟨st, h⟩
  | succ d ih =>
    intro fr st acc h
    have hl := level_NInv es c start fr st acc.1 h
    simp only [runLevels, hn, Bool.false_eq_true, ite_false]
    split
    · exact ⟨_, hl⟩
    · exact ih _ _ (acc.1 ++ _, acc.2) hl

theorem traverse_NInv (es : List E) (c : TCfg) (hn : c.retEdges = false) (start : Nat) :
    ∃ st', NInv es c start st' (traverse1 es c start).1 :=
  runLevels_NInv es c start hn c.maxDepth [start] _ ([], [])
    ⟨by simp, by simp, by simp, by simp, by simp, by simp⟩

/-- **Node results never repeat** (the start included, reported at most once). -/
theorem traverse_nodes_nodup (es : List E) (c : TCfg) (hn : c.retEdges = false) (start : Nat) :
    (traverse1 es c start).1.Nodup := by
  obtain ⟨_, h⟩ := traverse_NInv es c hn start; exact h.2.2.1

/-- **The start is reported only through a self-loop on it** (that passes the
`types` and `labels` filters); never by walking back to it. -/
theorem traverse_start_needs_self_loop (es : List E) (c : TCfg) (hn : c.retEdges = false) (start : Nat)
    (h : start ∈ (traverse1 es c start).1) : SelfLoop es c start := by
  obtain ⟨_, h'⟩ := traverse_NInv es c hn start
  exact h'.2.2.2.2.2 (h'.2.2.2.2.1 h)

/-- Without a self-loop on the start, the pre-#3253 statement still holds:
node results never repeat and never contain the start. -/
theorem traverse_nodes_nodup_no_self_loop (es : List E) (c : TCfg) (hn : c.retEdges = false)
    (start : Nat) (hs : ¬ SelfLoop es c start) : (start :: (traverse1 es c start).1).Nodup :=
  List.nodup_cons.mpr ⟨fun h => hs (traverse_start_needs_self_loop es c hn start h),
    traverse_nodes_nodup es c hn start⟩

/-! ## `direction:'both'` edge de-dup -/

/-- Relationship ids identify edges. -/
def IdsUnique (es : List E) : Prop := ∀ e ∈ es, ∀ e' ∈ es, e.id = e'.id → e = e'

theorem mem_edgesOf (es : List E) (c : TCfg) (nid : Nat) (e : E)
    (h : e ∈ edgesOf es c nid) : e ∈ es ∧ (e.src = nid ∨ e.dst = nid) := by
  unfold edgesOf applyTypes at h
  have h' : e ∈ collectOnce es nid c.dir := by
    split at h
    · exact h
    · exact (List.mem_filter.mp h).1
  cases hd : c.dir <;> rw [hd] at h' <;>
    simp only [collectOnce, List.mem_append, List.mem_filter, beq_iff_eq, Bool.and_eq_true] at h'
  · exact ⟨h'.1, Or.inl h'.2⟩
  · exact ⟨h'.1, Or.inr h'.2⟩
  · rcases h' with ⟨h1, h2⟩ | ⟨h1, h2, _⟩
    · exact ⟨h1, Or.inl h2⟩
    · exact ⟨h1, Or.inr h2⟩

/-- Every emitted edge is recorded as seen at both endpoints; ids are distinct. -/
def J (es : List E) (emitted : List E) (st : TS) : Prop :=
  (emitted.map (·.id)).Nodup ∧ ∀ e ∈ emitted, e ∈ es ∧ (e.src, e.id) ∈ st.seen ∧ (e.dst, e.id) ∈ st.seen

theorem J_of {es em em' : List E} {st X : TS} (h : J es em st) (hs : X.seen = st.seen) (ho : em' = em) :
    J es em' X := by
  subst ho; unfold J at *; rw [hs]; exact h

theorem mark_eq (start nid nb : Nat) (st : TS) :
    (mark start nid nb st).out = st.out ∧ (mark start nid nb st).seen = st.seen := by
  unfold mark; split
  · split <;> exact ⟨rfl, rfl⟩
  · exact ⟨rfl, rfl⟩

theorem visitEdge_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) (start nid : Nat) (accE : List E) (st : TS) (e : E) (he : e ∈ edgesOf es c nid)
    (hj : J es (accE ++ st.out) st) :
    J es (accE ++ (visitEdge c start nid st e).out) (visitEdge c start nid st e) := by
  obtain ⟨hes, hinc⟩ := mem_edgesOf es c nid e he
  have hdd : dedup c = true := by simp [dedup, hr, hd]
  unfold visitEdge
  split
  · exact hj
  · obtain ⟨t1, t2⟩ := mark_eq start nid (far nid e) (emit c nid (far nid e) st e)
    refine J_of (st := emit c nid (far nid e) st e) ?_ t2 (by rw [t1])
    unfold emit
    simp only [hr, hdd, ite_true]
    split
    · exact hj
    · rename_i hns
      obtain ⟨hn, hall⟩ := hj
      have hfar : (far nid e = e.src ∧ nid = e.dst) ∨ (far nid e = e.dst ∧ nid = e.src) := by
        unfold far; split
        · right; rename_i h; exact ⟨rfl, h.symm⟩
        · left; rename_i h; rcases hinc with h' | h'
          · exact absurd h' h
          · exact ⟨rfl, h'.symm⟩
      refine ⟨?_, ?_⟩
      · simp only
        rw [← List.append_assoc, List.map_append, List.nodup_append]
        refine ⟨hn, by simp, ?_⟩
        intro a ha b hb hab
        simp at hb; subst hb
        obtain ⟨e', he', rfl⟩ := List.mem_map.mp ha
        have := hu e' (hall e' he').1 e hes hab; subst this
        rcases hinc with h | h
        · exact hns (h ▸ (hall e' he').2.1)
        · exact hns (h ▸ (hall e' he').2.2)
      · intro e' he'
        simp only at he'
        rw [← List.append_assoc, List.mem_append] at he'
        rcases he' with he' | he'
        · obtain ⟨h1, h2, h3⟩ := hall e' he'
          exact ⟨h1, List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h2),
            List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h3)⟩
        · simp at he'; subst he'
          rcases hfar with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1] <;> subst h2 <;> exact ⟨hes, by simp, by simp⟩

theorem level_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) (start : Nat) (accE : List E) :
    ∀ (fr : List Nat) (st : TS), J es (accE ++ st.out) st →
      J es (accE ++ (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c start nid) st) st).out)
        (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c start nid) st) st) := by
  intro fr
  induction fr with
  | nil => intro st h; exact h
  | cons nid t ih =>
    intro st h
    simp only [List.foldl_cons]
    apply ih
    have : ∀ (l : List E) (st : TS), (∀ e ∈ l, e ∈ edgesOf es c nid) → J es (accE ++ st.out) st →
        J es (accE ++ (l.foldl (visitEdge c start nid) st).out) (l.foldl (visitEdge c start nid) st) := by
      intro l
      induction l with
      | nil => intro st _ h; exact h
      | cons e t' ih' =>
        intro st hl h
        simp only [List.foldl_cons]
        exact ih' _ (fun e' h' => hl e' (List.mem_cons_of_mem _ h'))
          (visitEdge_J es c hd hr hu start nid accE st e (hl e (by simp)) h)
    exact this _ st (fun _ h => h) h

theorem runLevels_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) (start : Nat) :
    ∀ d fr (st : TS) (acc : List Nat × List E), J es acc.2 st →
      ((runLevels es c start d fr st acc).2.map (·.id)).Nodup := by
  intro d
  induction d with
  | zero => intro fr st acc h; exact h.1
  | succ d ih =>
    intro fr st acc h
    have hl := level_J es c hd hr hu start acc.2 fr { st with next := [], out := [], selfN := false }
      (J_of h rfl (by simp))
    simp only [runLevels, hr, ite_true]
    split
    · exact hl.1
    · exact ih _ _ _ hl

/-- **`direction:'both'` edge results never repeat a relationship.** -/
theorem traverse_both_edges_nodup (es : List E) (c : TCfg) (hd : c.dir = .both)
    (hr : c.retEdges = true) (hu : IdsUnique es) (start : Nat) :
    ((traverse1 es c start).2.map (·.id)).Nodup :=
  runLevels_J es c hd hr hu start c.maxDepth [start] _ ([], []) ⟨by simp, by simp⟩

/-! ## Former divergences from C (fixed on this code) -/

/-- e0 a→b, e1 a→a, e2 b→a; `labels`/`types` unset; depth 1. -/
def selfLoopGraph : List E := [⟨0, 0, 1, "R"⟩, ⟨1, 0, 0, "R"⟩, ⟨2, 1, 0, "R"⟩]

def cfg1 (retEdges : Bool) : TCfg := ⟨.outgoing, [], fun _ => true, retEdges, 1⟩

/-- Was `traverse_self_loop_twice` (`[0, 1, 1]`, through `get_node_relationships`):
edges mode returns the self-loop e1 once (C: [1, 0]), in every direction. -/
theorem traverse_self_loop_once :
    ((traverse1 selfLoopGraph (cfg1 true) 0).2.map (·.id)) = [0, 1] ∧
    ((traverse1 selfLoopGraph { cfg1 true with dir := .both } 0).2.map (·.id)) = [0, 1, 2] := by decide

/-- Fixed by #3253 (was `traverse_drops_self_neighbour`, `[1]`): nodes mode
reports the start when it is its own neighbour (C: [0, 1]), the same set as
`getNeighbors` on that node. -/
theorem traverse_keeps_self_neighbour :
    (traverse1 selfLoopGraph (cfg1 false) 0).1 = [0, 1] ∧
    (traverse1 selfLoopGraph { cfg1 false with dir := .incoming } 0).1 = [0, 1] ∧
    neighborNodes 0 (collectEdges selfLoopGraph 0 .outgoing) [] = [1, 0] := by decide

/-- e0 a→b, e1 b→a; depth 3: walking back to the start over the cycle does not
report it, and no edge repeats. -/
def cycleGraph : List E := [⟨0, 0, 1, "R"⟩, ⟨1, 1, 0, "R"⟩]

theorem traverse_cycle_skips_start :
    ((traverse1 cycleGraph ⟨.outgoing, [], fun _ => true, true, 3⟩ 0).2.map (·.id)) = [0, 1] ∧
    (traverse1 cycleGraph ⟨.outgoing, [], fun _ => true, false, 3⟩ 0).1 = [1] ∧
    (traverse1 cycleGraph ⟨.both, [], fun _ => true, false, 3⟩ 0).1 = [1] := by decide

end AlgoUdf.UdfTraverse
