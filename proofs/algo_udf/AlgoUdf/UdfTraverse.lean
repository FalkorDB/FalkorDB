import AlgoUdf.Neighbors
/-! # `graph.traverse` (`js_traverse_impl`, udf/js_classes.rs)

Results:
* `traverse_nodes_nodup` — node results never repeat, and contain the start
  only when it is its own neighbour (a self-loop that passes the filters).
* `fixed3253_traverse_self_neighbour` — ... and then they do contain it
  (`start_reported`/`self_neighbour`, #3253; was `traverse_drops_self_neighbour`, W5-udf-2).
* `traverse_both_edges_nodup` — `direction:'both'` edge results never repeat a
  relationship (the `seen_edges` de-dup), given relationship ids identify edges.
* `fixed3084_traverse_self_loop_once` (was `traverse_self_loop_twice`, fixed by #3084 5f147f10e: a
  self-loop is reported once by `get_node_relationships_by_type`).
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

/-- `dedup_edges = return_type == "edges" && direction == "both"`. -/
def dedup (c : TCfg) : Bool := c.retEdges && c.dir == .both

/-- Per-source BFS state: `visited`, `seen_edges`, `next_frontier`, `edges_to_create`,
`start_reported` (per source) and `self_neighbour` (per level). -/
structure TS where
  visited : List Nat
  seen : List (Nat × Nat)
  next : List Nat
  out : List E
  reported : Bool
  selfNb : Bool

def far (nid : Nat) (e : E) : Nat := if e.src = nid then e.dst else e.src

/-- The `return_type == "edges"` block. -/
def emit (c : TCfg) (nid : Nat) (st : TS) (e : E) : TS :=
  if c.retEdges then
    if dedup c then
      (if (nid, e.id) ∈ st.seen then st
       else { st with seen := (far nid e, e.id) :: (nid, e.id) :: st.seen, out := st.out ++ [e] })
    else { st with out := st.out ++ [e] }
  else st

/-- `if visited.insert(nb) { push } else if nb == start && nid == start && !start_reported { … }`. -/
def mark (start nid nb : Nat) (st : TS) : TS :=
  if nb ∈ st.visited then
    (if nb = start ∧ nid = start ∧ st.reported = false then { st with reported := true, selfNb := true }
     else st)
  else { st with visited := st.visited ++ [nb], next := st.next ++ [nb] }

def visitEdge (c : TCfg) (start nid : Nat) (st : TS) (e : E) : TS :=
  if !c.labelOk (far nid e) then st else mark start nid (far nid e) (emit c nid st e)

def level (es : List E) (c : TCfg) (start : Nat) (fr : List Nat) (st : TS) : TS :=
  fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c start nid) st)
    { st with next := [], out := [], selfNb := false }

/-- Returns the node results and the edge results (only one is used per mode). Nodes mode
reports the start (if `self_neighbour`) before `next_frontier`. -/
def runLevels (es : List E) (c : TCfg) (start : Nat) :
    Nat → List Nat → TS → List Nat × List E → List Nat × List E
  | 0, _, _, acc => acc
  | d + 1, fr, st, acc =>
    let st' := level es c start fr st
    let acc' := if c.retEdges then (acc.1, acc.2 ++ st'.out)
      else (acc.1 ++ ((if st'.selfNb then [start] else []) ++ st'.next), acc.2)
    if st'.next = [] then acc' else runLevels es c start d st'.next st' acc'

def traverse1 (es : List E) (c : TCfg) (start : Nat) : List Nat × List E :=
  runLevels es c start c.maxDepth [start] ⟨[start], [], [], [], false, false⟩ ([], [])

/-! ## Frame lemmas -/

theorem emit_frame (c : TCfg) (nid : Nat) (st : TS) (e : E) :
    (emit c nid st e).visited = st.visited ∧ (emit c nid st e).next = st.next ∧
    (emit c nid st e).reported = st.reported ∧ (emit c nid st e).selfNb = st.selfNb := by
  unfold emit; split
  · split
    · split <;> exact ⟨rfl, rfl, rfl, rfl⟩
    · exact ⟨rfl, rfl, rfl, rfl⟩
  · exact ⟨rfl, rfl, rfl, rfl⟩

theorem mark_frame (start nid nb : Nat) (st : TS) :
    (mark start nid nb st).seen = st.seen ∧ (mark start nid nb st).out = st.out := by
  unfold mark; split
  · split <;> exact ⟨rfl, rfl⟩
  · exact ⟨rfl, rfl⟩

/-! ## Node results -/

theorem mark_visited (start nid nb : Nat) (st : TS) :
    ∃ add, (mark start nid nb st).visited = st.visited ++ add ∧ (mark start nid nb st).next = st.next ++ add ∧
      (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
  unfold mark; split
  · split <;> exact ⟨[], by simp, by simp, by simp, by simp⟩
  · rename_i hn; exact ⟨[nb], rfl, rfl, by simpa using hn, by simp⟩

theorem visitEdge_visited (c : TCfg) (start nid : Nat) (st : TS) (e : E) :
    ∃ add, (visitEdge c start nid st e).visited = st.visited ++ add ∧
      (visitEdge c start nid st e).next = st.next ++ add ∧
      (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
  unfold visitEdge
  split
  · exact ⟨[], by simp, by simp, by simp, by simp⟩
  · obtain ⟨h1, h2, _, _⟩ := emit_frame c nid st e
    obtain ⟨add, a1, a2, a3, a4⟩ := mark_visited start nid (far nid e) (emit c nid st e)
    rw [h1] at a1 a3; rw [h2] at a2
    exact ⟨add, a1, a2, a3, a4⟩

theorem fold_visit_visited (c : TCfg) (start nid : Nat) :
    ∀ (l : List E) (st : TS), ∃ add, (l.foldl (visitEdge c start nid) st).visited = st.visited ++ add ∧
      (l.foldl (visitEdge c start nid) st).next = st.next ++ add ∧ (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
  intro l
  induction l with
  | nil => intro st; exact ⟨[], by simp, by simp, by simp, by simp⟩
  | cons e t ih =>
    intro st
    simp only [List.foldl_cons]
    obtain ⟨a1, h1, h2, h3, h4⟩ := visitEdge_visited c start nid st e
    obtain ⟨a2, h5, h6, h7, h8⟩ := ih (visitEdge c start nid st e)
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

theorem level_visited (es : List E) (c : TCfg) (start : Nat) :
    ∀ (fr : List Nat) (st : TS), ∃ add,
      (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c start nid) st) st).visited =
        st.visited ++ add ∧
      (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c start nid) st) st).next = st.next ++ add ∧
      (∀ x ∈ add, x ∉ st.visited) ∧ add.Nodup := by
  intro fr
  induction fr with
  | nil => intro st; exact ⟨[], by simp, by simp, by simp, by simp⟩
  | cons nid t ih =>
    intro st
    simp only [List.foldl_cons]
    obtain ⟨a1, h1, h2, h3, h4⟩ := fold_visit_visited c start nid (edgesOf es c nid) st
    obtain ⟨a2, h5, h6, h7, h8⟩ := ih ((edgesOf es c nid).foldl (visitEdge c start nid) st)
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

/-- The start is its own neighbour: a self-loop on it survives the type filter, and the start
passes the label filter. -/
def SelfLoop (es : List E) (c : TCfg) (start : Nat) : Prop :=
  c.labelOk start = true ∧ ∃ e ∈ edgesOf es c start, e.src = start ∧ e.dst = start

/-- Within a level that began with `reported = r0`: `reported = r0 || self_neighbour`, and
`self_neighbour` is only set when `r0` was false, on a real self-loop. -/
def Q (es : List E) (c : TCfg) (start : Nat) (r0 : Bool) (st : TS) : Prop :=
  st.reported = (r0 || st.selfNb) ∧ (st.selfNb = true → r0 = false ∧ SelfLoop es c start)

theorem visitEdge_Q (es : List E) (c : TCfg) (start nid : Nat) (r0 : Bool) (st : TS) (e : E)
    (he : e ∈ edgesOf es c nid) (hq : Q es c start r0 st) : Q es c start r0 (visitEdge c start nid st e) := by
  unfold visitEdge
  split
  · exact hq
  · rename_i hl
    obtain ⟨_, _, f3, f4⟩ := emit_frame c nid st e
    have hq' : Q es c start r0 (emit c nid st e) := by unfold Q; rw [f3, f4]; exact hq
    unfold mark
    split
    · split
      · rename_i hc
        obtain ⟨hnb, hnid, hrep⟩ := hc
        refine ⟨by simp, fun _ => ⟨?_, ?_⟩⟩
        · have := hq'.1; rw [hrep] at this
          cases r0 with
          | false => rfl
          | true => simp at this
        · subst hnid
          refine ⟨by simpa [hnb] using hl, e, he, ?_⟩
          unfold far at hnb; split at hnb
          · rename_i hs; exact ⟨hs, hnb⟩
          · rename_i hs; exact absurd hnb hs
      · exact hq'
    · exact hq'

theorem level_Q (es : List E) (c : TCfg) (start : Nat) (r0 : Bool) :
    ∀ (fr : List Nat) (st : TS), Q es c start r0 st →
      Q es c start r0 (fr.foldl (fun st nid => (edgesOf es c nid).foldl (visitEdge c start nid) st) st) := by
  intro fr
  induction fr with
  | nil => intro st h; exact h
  | cons nid t ih =>
    intro st h
    simp only [List.foldl_cons]
    apply ih
    have : ∀ (l : List E) (st : TS), (∀ e ∈ l, e ∈ edgesOf es c nid) → Q es c start r0 st →
        Q es c start r0 (l.foldl (visitEdge c start nid) st) := by
      intro l
      induction l with
      | nil => intro st _ h; exact h
      | cons e t' ih' =>
        intro st hl h
        simp only [List.foldl_cons]
        exact ih' _ (fun e' h' => hl e' (List.mem_cons_of_mem _ h'))
          (visitEdge_Q es c start nid r0 st e (hl e (by simp)) h)
    exact this _ st (fun _ h => h) h

/-- Node-mode invariant between levels. -/
def NI (es : List E) (c : TCfg) (start : Nat) (acc : List Nat) (st : TS) : Prop :=
  acc.Nodup ∧ (∀ x ∈ acc, x ∈ st.visited) ∧ start ∈ st.visited ∧
    (start ∈ acc → st.reported = true ∧ SelfLoop es c start)

theorem runLevels_nodes (es : List E) (c : TCfg) (start : Nat) (hn : c.retEdges = false) :
    ∀ d fr (st : TS) (acc : List Nat × List E), NI es c start acc.1 st →
      ∃ st', NI es c start (runLevels es c start d fr st acc).1 st' := by
  intro d
  induction d with
  | zero => intro fr st acc h; exact ⟨st, h⟩
  | succ d ih =>
    intro fr st acc h
    obtain ⟨hnd, hsub, hsv, hsr⟩ := h
    let st0 : TS := { st with next := [], out := [], selfNb := false }
    obtain ⟨add, h1, h2, h3, h4⟩ := level_visited es c start fr st0
    have hq := level_Q es c start st.reported fr st0 ⟨by simp [st0], by simp [st0]⟩
    have hvis : (level es c start fr st).visited = st.visited ++ add := h1
    have hnext : (level es c start fr st).next = add := by
      have h2' := h2; simp only [st0, List.nil_append] at h2'; exact h2'
    have hq' : Q es c start st.reported (level es c start fr st) := hq
    have h3' : ∀ x ∈ add, x ∉ st.visited := h3
    have hNI : NI es c start
        (acc.1 ++ ((if (level es c start fr st).selfNb then [start] else []) ++ (level es c start fr st).next))
        (level es c start fr st) := by
      rw [hnext]
      have hsa : start ∉ add := fun hm => h3' start hm hsv
      refine ⟨?_, ?_, ?_, ?_⟩
      · rw [List.nodup_append]
        refine ⟨hnd, ?_, ?_⟩
        · rw [List.nodup_append]
          refine ⟨by split <;> simp, h4, ?_⟩
          intro a ha b hb hab; subst hab
          split at ha
          · simp at ha; subst ha; exact hsa hb
          · simp at ha
        · intro a ha b hb hab; subst hab
          rw [List.mem_append] at hb
          rcases hb with hb | hb
          · split at hb
            · rename_i hs; simp at hb; subst hb
              have := (hq'.2 hs).1
              have := (hsr ha).1
              simp_all
            · simp at hb
          · exact h3' a hb (hsub a ha)
      · intro x hx
        rw [hvis]
        rw [List.mem_append] at hx
        rcases hx with hx | hx
        · exact List.mem_append_left _ (hsub x hx)
        · rw [List.mem_append] at hx
          rcases hx with hx | hx
          · split at hx
            · simp at hx; subst hx; exact List.mem_append_left _ hsv
            · simp at hx
          · exact List.mem_append_right _ hx
      · rw [hvis]; exact List.mem_append_left _ hsv
      · intro hx
        rw [List.mem_append] at hx
        rcases hx with hx | hx
        · obtain ⟨r, s⟩ := hsr hx
          exact ⟨by rw [hq'.1, r]; simp, s⟩
        · rw [List.mem_append] at hx
          rcases hx with hx | hx
          · split at hx
            · rename_i hs
              exact ⟨by rw [hq'.1, hs]; simp, (hq'.2 hs).2⟩
            · simp at hx
          · exact absurd hx hsa
    simp only [runLevels, hn, Bool.false_eq_true, ite_false]
    split
    · exact ⟨_, hNI⟩
    · exact ih _ _ _ hNI

/-- **Node results never repeat, and contain the start node only when it is its own
neighbour** (a self-loop on it that passes the type and label filters). -/
theorem traverse_nodes_nodup (es : List E) (c : TCfg) (hn : c.retEdges = false) (start : Nat) :
    (traverse1 es c start).1.Nodup ∧ (start ∈ (traverse1 es c start).1 → SelfLoop es c start) := by
  have h0 : NI es c start (([], []) : List Nat × List E).1 ⟨[start], [], [], [], false, false⟩ :=
    ⟨by simp, by simp, by simp, by simp⟩
  obtain ⟨_, h⟩ := runLevels_nodes es c start hn c.maxDepth [start] _ ([], []) h0
  exact ⟨h.1, fun hm => (h.2.2.2 hm).2⟩

/-! ### The start is reported when it is its own neighbour (#3253) -/

theorem visitEdge_selfNb_mono (c : TCfg) (start nid : Nat) (st : TS) (e : E) (h : st.selfNb = true) :
    (visitEdge c start nid st e).selfNb = true := by
  unfold visitEdge
  split
  · exact h
  · have : (emit c nid st e).selfNb = true := by rw [(emit_frame c nid st e).2.2.2]; exact h
    unfold mark; split
    · split
      · rfl
      · exact this
    · exact this

theorem fold_selfNb_mono (c : TCfg) (start nid : Nat) :
    ∀ (l : List E) (st : TS), st.selfNb = true → (l.foldl (visitEdge c start nid) st).selfNb = true := by
  intro l
  induction l with
  | nil => intro st h; exact h
  | cons e t ih => intro st h; exact ih _ (visitEdge_selfNb_mono c start nid st e h)

/-- Visiting the self-loop with the start already visited sets `self_neighbour`, given
`reported = r0 || selfNb` (nothing reported in an earlier level). -/
theorem visitEdge_selfLoop (es : List E) (c : TCfg) (start : Nat) (st : TS) (e : E)
    (hs : e.src = start) (hd : e.dst = start) (hl : c.labelOk start = true)
    (hv : start ∈ st.visited) (hq : Q es c start false st) :
    (visitEdge c start start st e).selfNb = true := by
  have hf : far start e = start := by unfold far; simp [hs, hd]
  unfold visitEdge
  rw [hf]
  simp only [hl, Bool.not_true, Bool.false_eq_true, ite_false]
  obtain ⟨f1, _, f3, f4⟩ := emit_frame c start st e
  unfold mark
  rw [f1]; simp only [hv, ite_true]
  split
  · rfl
  · rename_i hc
    have : (emit c start st e).reported = true := by
      cases hr : (emit c start st e).reported
      · exact absurd ⟨trivial, trivial, hr⟩ hc
      · rfl
    rw [f3, hq.1] at this; rw [f4]; simpa using this

theorem fold_selfLoop (es : List E) (c : TCfg) (start : Nat) (e : E)
    (hs : e.src = start) (hd : e.dst = start) (hl : c.labelOk start = true) :
    ∀ (l : List E) (st : TS), e ∈ l → (∀ x ∈ l, x ∈ edgesOf es c start) → start ∈ st.visited →
      Q es c start false st → (l.foldl (visitEdge c start start) st).selfNb = true := by
  intro l
  induction l with
  | nil => intro _ h; simp at h
  | cons x t ih =>
    intro st hm hall hv hq
    simp only [List.foldl_cons]
    have hv' : start ∈ (visitEdge c start start st x).visited := by
      obtain ⟨add, h1, _⟩ := visitEdge_visited c start start st x
      rw [h1]; exact List.mem_append_left _ hv
    have hq' := visitEdge_Q es c start start false st x (hall x (by simp)) hq
    rcases List.mem_cons.mp hm with h | h
    · subst h
      exact fold_selfNb_mono c start start t _ (visitEdge_selfLoop es c start st e hs hd hl hv hq)
    · exact ih _ h (fun y hy => hall y (List.mem_cons_of_mem _ hy)) hv' hq'

/-- **Fixed by #3253 (was `traverse_drops_self_neighbour`, W5-udf-2).** With `maxDepth ≥ 1`, a
start that is its own neighbour is in the node results, as in C. With `traverse_nodes_nodup`
this is exact: the start is reported iff `SelfLoop`, and once. -/
theorem fixed3253_traverse_self_neighbour (es : List E) (c : TCfg) (hn : c.retEdges = false)
    (start : Nat) (hd : 1 ≤ c.maxDepth) (hs : SelfLoop es c start) :
    start ∈ (traverse1 es c start).1 := by
  obtain ⟨hl, e, he, hes, hed⟩ := hs
  obtain ⟨d, hdm⟩ : ∃ d, c.maxDepth = d + 1 := ⟨c.maxDepth - 1, by omega⟩
  have hsn : (level es c start [start] ⟨[start], [], [], [], false, false⟩).selfNb = true := by
    unfold level
    simp only [List.foldl_cons, List.foldl_nil]
    exact fold_selfLoop es c start e hes hed hl _ _ he (fun _ h => h) (by simp) ⟨by simp, by simp⟩
  -- the start's slot in the level-1 output is kept by later levels
  have keep : ∀ d fr (st : TS) (acc : List Nat × List E), start ∈ acc.1 →
      start ∈ (runLevels es c start d fr st acc).1 := by
    intro d
    induction d with
    | zero => intro _ _ _ h; exact h
    | succ d ih =>
      intro fr st acc h
      simp only [runLevels, hn, Bool.false_eq_true, ite_false]
      have : start ∈ (acc.1 ++ ((if (level es c start fr st).selfNb then [start] else []) ++
          (level es c start fr st).next)) := List.mem_append_left _ h
      split
      · exact this
      · exact ih _ _ _ this
  unfold traverse1
  rw [hdm]
  simp only [runLevels, hn, Bool.false_eq_true, ite_false]
  rw [hsn]
  split
  · simp
  · exact keep _ _ _ _ (by simp)

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
  rcases h' with ⟨h1, h2⟩ | ⟨⟨h1, h2⟩, _⟩
  · exact ⟨h1, Or.inl h2⟩
  · exact ⟨h1, Or.inr h2⟩

/-- Every emitted edge is recorded as seen at both endpoints; ids are distinct. -/
def J (es : List E) (emitted : List E) (st : TS) : Prop :=
  (emitted.map (·.id)).Nodup ∧ ∀ e ∈ emitted, e ∈ es ∧ (e.src, e.id) ∈ st.seen ∧ (e.dst, e.id) ∈ st.seen

theorem J_of {es em em' : List E} {st X : TS} (h : J es em st) (hs : X.seen = st.seen) (ho : em' = em) :
    J es em' X := by
  subst ho; unfold J at *; rw [hs]; exact h

/-- The `seen_edges` block keeps `J`. -/
theorem emit_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) (nid : Nat) (accE : List E) (st : TS) (e : E) (he : e ∈ edgesOf es c nid)
    (hj : J es (accE ++ st.out) st) : J es (accE ++ (emit c nid st e).out) (emit c nid st e) := by
  obtain ⟨hes, hinc⟩ := mem_edgesOf_both es c hd nid e he
  have hdd : dedup c = true := by simp [dedup, hr, hd]
  unfold emit
  simp only [hr, hdd, ite_true]
  split
  · exact hj
  · rename_i hns
    show J es (accE ++ (st.out ++ [e])) { st with seen := (far nid e, e.id) :: (nid, e.id) :: st.seen, out := st.out ++ [e] }
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

theorem visitEdge_J (es : List E) (c : TCfg) (hd : c.dir = .both) (hr : c.retEdges = true)
    (hu : IdsUnique es) (start nid : Nat) (accE : List E) (st : TS) (e : E) (he : e ∈ edgesOf es c nid)
    (hj : J es (accE ++ st.out) st) :
    J es (accE ++ (visitEdge c start nid st e).out) (visitEdge c start nid st e) := by
  unfold visitEdge
  split
  · exact hj
  · obtain ⟨t1, t2⟩ := mark_frame start nid (far nid e) (emit c nid st e)
    exact J_of (emit_J es c hd hr hu nid accE st e he hj) t1 (by rw [t2])

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
    have hl := level_J es c hd hr hu start acc.2 fr { st with next := [], out := [], selfNb := false }
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

/-! ## Examples -/

/-- e0 a→b, e1 a→a, e2 b→a; `labels`/`types` unset; depth 1. -/
def selfLoopGraph : List E := [⟨0, 0, 1, "R"⟩, ⟨1, 0, 0, "R"⟩, ⟨2, 1, 0, "R"⟩]

def cfg1 (retEdges : Bool) : TCfg := ⟨.outgoing, [], fun _ => true, retEdges, 1⟩

/-- Edges mode lists the self-loop e1 once (#3084). -/
theorem fixed3084_traverse_self_loop_once :
    ((traverse1 selfLoopGraph (cfg1 true) 0).2.map (·.id)) = [0, 1] := by decide

/-- Nodes mode reports the start first, then its other neighbour: C's `[0, 1]` (#3253); deeper
levels never report it again. -/
theorem fixed3253_traverse_self_neighbour_example :
    (traverse1 selfLoopGraph (cfg1 false) 0).1 = [0, 1] ∧
    (traverse1 selfLoopGraph { cfg1 false with maxDepth := 3 } 0).1 = [0, 1] := by decide

end AlgoUdf.UdfTraverse
