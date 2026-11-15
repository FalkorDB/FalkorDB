/-
# `QueryGraph::filter_visited`, `connected_components`, `dfs` (ast.rs:792-887)

The bound graph (`QueryGraph<_, _, Variable>`). `visited` is one
`HashSet<u32>` shared by node, relationship and path ids, modelled as a
list with `insert` = "append if absent".
-/
import FalkorParserGrammar.C.Graph

namespace FalkorParserGrammar.C

abbrev G := QG Var

def vins (x : Nat) (vis : List Nat) : List Nat := if x ∈ vis then vis else vis ++ [x]

theorem mem_vins {x y : Nat} {vis : List Nat} : y ∈ vins x vis ↔ y ∈ vis ∨ y = x := by
  unfold vins; split <;> simp_all <;> constructor <;> intro h <;> (try rcases h with h | h) <;>
    simp_all

/-- The path step at the end of `dfs` (ast.rs:878-882). -/
def pathStep (ps : List (QPath Var)) (vis : List Nat) (c : G) : List Nat × G :=
  match ps with
  | [] => (vis, c)
  | p :: ps =>
    if p.vars.all (fun v => v.id ∈ vis) && !(p.var.id ∈ vis) then
      pathStep ps (vis ++ [p.var.id]) (c.addPath p).2
    else pathStep ps vis c

/-- One relationship's own bookkeeping in the loop (ast.rs:854-856 / 865-867):
`if visited.insert(relationship.alias.id) { component.add_relationship(..) }`. -/
def relPart (r : QRel Var) (vis : List Nat) (c : G) : List Nat × G :=
  if r.alias.id ∈ vis then (vis, c) else (vis ++ [r.alias.id], (c.addRel r).2)

mutual
/-- `QueryGraph::dfs` (ast.rs:844-883); `none` = model budget exhausted. -/
def dfs (g : G) : Nat → QNode Var → List Nat → G → Option (List Nat × G)
  | 0, _, _, _ => none
  | f + 1, n, vis, c =>
    match dfsRels g f n g.rels (vins n.alias.id vis) (c.addNode n).2 with
    | none => none
    | some p => some (pathStep g.paths p.1 p.2)
termination_by f _ _ _ => (f, 0)
/-- The `for relationship in &self.relationships` loop (ast.rs:852-876). -/
def dfsRels (g : G) : Nat → QNode Var → List (QRel Var) → List Nat → G → Option (List Nat × G)
  | _, _, [], vis, c => some (vis, c)
  | f, n, r :: rs, vis, c =>
    if r.src.alias.id = n.alias.id then
      if r.dst.alias.id ∈ (relPart r vis c).1 then dfsRels g f n rs (relPart r vis c).1 (relPart r vis c).2
      else match dfs g f r.dst (relPart r vis c).1 (relPart r vis c).2 with
        | none => none
        | some p => dfsRels g f n rs p.1 p.2
    else if r.dst.alias.id = n.alias.id then
      if r.src.alias.id ∈ (relPart r vis c).1 then dfsRels g f n rs (relPart r vis c).1 (relPart r vis c).2
      else match dfs g f r.src (relPart r vis c).1 (relPart r vis c).2 with
        | none => none
        | some p => dfsRels g f n rs p.1 p.2
    else dfsRels g f n rs vis c
termination_by f _ rs _ _ => (f, rs.length + 1)
end

/-- `connected_components` (ast.rs:823-842). -/
def ccLoop (g : G) (f : Nat) : List (QNode Var) → List Nat → List G → Option (List G)
  | [], _, acc => some acc
  | n :: ns, vis, acc =>
    if n.alias.id ∈ vis then ccLoop g f ns vis acc
    else match dfs g f n vis QG.empty with
      | none => none
      | some (vis', c) => ccLoop g f ns vis' (acc ++ [c])

def connectedComponents (g : G) (f : Nat) : Option (List G) := ccLoop g f g.nodes [] []

/-! ## Specification -/

def relIds (g : G) : List Nat := g.rels.map (·.alias.id)
def pathIds (g : G) : List Nat := g.paths.map (·.var.id)
/-- Where a component's nodes may come from: the node list or a relationship endpoint. -/
def Cand (g : G) (x : QNode Var) : Prop := x ∈ g.nodes ∨ ∃ r ∈ g.rels, x = r.src ∨ x = r.dst

def ids (l : List (QNode Var)) : List Nat := l.map (·.alias.id)

/-- What a run of `dfs`/`dfsRels`/`pathStep` does to `(visited, component)`:
it appends fresh, distinct, visited, candidate nodes, only ever grows
`visited`, and every newly visited id is a new node, a relationship or a path. -/
def Step (g : G) (vis : List Nat) (c : G) (vis' : List Nat) (c' : G) : Prop :=
  ∃ new : List (QNode Var), c'.nodes = c.nodes ++ new ∧ (ids new).Nodup ∧
    (∀ x ∈ new, x.alias.id ∉ vis ∧ x.alias.id ∈ vis' ∧ Cand g x) ∧
    (∀ v ∈ vis, v ∈ vis') ∧
    (∀ v ∈ vis', v ∈ vis ∨ v ∈ ids new ∨ v ∈ relIds g ∨ v ∈ pathIds g) ∧
    (∀ r ∈ c'.rels, r ∈ c.rels ∨ r ∈ g.rels)

theorem Step.refl (g : G) vis c : Step g vis c vis c :=
  ⟨[], by simp, by simp [ids], by simp, fun _ h => h, fun v h => .inl h, fun r h => .inl h⟩

theorem Step.trans {g : G} {v0 c0 v1 c1 v2 c2} (h1 : Step g v0 c0 v1 c1) (h2 : Step g v1 c1 v2 c2) :
    Step g v0 c0 v2 c2 := by
  obtain ⟨n1, e1, d1, f1, s1, o1, r1⟩ := h1
  obtain ⟨n2, e2, d2, f2, s2, o2, r2⟩ := h2
  refine ⟨n1 ++ n2, by rw [e2, e1, List.append_assoc], ?_, ?_, fun v h => s2 v (s1 v h), ?_, ?_⟩
  · unfold ids at *; rw [List.map_append, List.nodup_append]
    refine ⟨d1, d2, ?_⟩
    intro a ha b hb hab; subst hab
    simp only [List.mem_map] at ha hb
    obtain ⟨x, hx, rfl⟩ := ha; obtain ⟨y, hy, hxy⟩ := hb
    exact (f2 y hy).1 (hxy ▸ (f1 x hx).2.1)
  · intro x hx; rcases List.mem_append.1 hx with hx | hx
    · exact ⟨(f1 x hx).1, s2 _ (f1 x hx).2.1, (f1 x hx).2.2⟩
    · refine ⟨fun h => (f2 x hx).1 (s1 _ h), (f2 x hx).2⟩
  · intro v hv
    rcases o2 v hv with h | h | h | h
    · rcases o1 v h with h | h | h | h
      · exact .inl h
      · exact .inr (.inl (by unfold ids at *; simp only [List.map_append, List.mem_append]; exact .inl h))
      · exact .inr (.inr (.inl h))
      · exact .inr (.inr (.inr h))
    · exact .inr (.inl (by unfold ids at *; simp only [List.map_append, List.mem_append]; exact .inr h))
    · exact .inr (.inr (.inl h))
    · exact .inr (.inr (.inr h))
  · intro r hr; rcases r2 r hr with h | h
    · exact r1 r h
    · exact .inr h

/-- Nodes of the component are all visited. -/
def CInv (vis : List Nat) (c : G) : Prop := ∀ x ∈ c.nodes, x.alias.id ∈ vis

theorem Step.cinv {g : G} {v c v' c'} (h : Step g v c v' c') (hc : CInv v c) : CInv v' c' := by
  obtain ⟨new, e, _, f, s, _, _⟩ := h
  intro x hx; rw [e] at hx
  rcases List.mem_append.1 hx with hx | hx
  · exact s _ (hc x hx)
  · exact (f x hx).2.1

theorem addRel_nodes (c : G) (r : QRel Var) : (c.addRel r).2.nodes = c.nodes := by
  rcases QG.addRel_spec c r with ⟨_, e⟩ | ⟨_, e⟩ <;> rw [e]

theorem addRel_rels (c : G) (r : QRel Var) : ∀ x ∈ (c.addRel r).2.rels, x ∈ c.rels ∨ x = r := by
  intro x hx
  rcases QG.addRel_spec c r with ⟨_, e⟩ | ⟨_, e⟩ <;> rw [e] at hx
  · exact .inl hx
  · simp at hx; exact hx

theorem pathStep_step (g : G) : ∀ (ps : List (QPath Var)), (∀ p ∈ ps, p ∈ g.paths) →
    ∀ vis (c : G), Step g vis c (pathStep ps vis c).1 (pathStep ps vis c).2
  | [], _, vis, c => Step.refl g vis c
  | p :: ps, hps, vis, c => by
    unfold pathStep
    have hps' : ∀ q ∈ ps, q ∈ g.paths := fun q h => hps q (List.mem_cons_of_mem _ h)
    split
    · refine Step.trans ?_ (pathStep_step g ps hps' _ _)
      have hn : (c.addPath p).2.nodes = c.nodes ∧ (c.addPath p).2.rels = c.rels := by
        rcases QG.addPath_spec c p with ⟨_, e⟩ | ⟨_, e⟩ <;> rw [e] <;> exact ⟨rfl, rfl⟩
      refine ⟨[], by simp [hn.1], by simp [ids], by simp, fun v h => by simp [h], ?_, ?_⟩
      · intro v hv; simp only [List.mem_append, List.mem_singleton] at hv
        rcases hv with h | h
        · exact .inl h
        · exact .inr (.inr (.inr (by subst h; unfold pathIds; exact List.mem_map_of_mem (hps p (by simp)))))
      · intro r hr; rw [hn.2] at hr; exact .inl hr
    · exact pathStep_step g ps hps' vis c

/-- The relationship part of one `dfsRels` iteration. -/
theorem relPart_step (g : G) (r : QRel Var) (hr : r ∈ g.rels) (vis : List Nat) (c : G) :
    Step g vis c (relPart r vis c).1 (relPart r vis c).2 := by
  unfold relPart; split
  · exact Step.refl g vis c
  · refine ⟨[], by simp [addRel_nodes], by simp [ids], by simp, fun v h => by simp [h], ?_, ?_⟩
    · intro v hv; simp only [List.mem_append, List.mem_singleton] at hv
      rcases hv with h | h
      · exact .inl h
      · exact .inr (.inr (.inl (by subst h; unfold relIds; exact List.mem_map_of_mem hr)))
    · intro x hx; rcases addRel_rels c r x hx with h | h
      · exact .inl h
      · exact .inr (h ▸ hr)

mutual
/-- **dfs on an unvisited candidate node** emits it first, then only fresh
candidate nodes, all visited and pairwise distinct. -/
theorem dfs_step (g : G) : ∀ (f : Nat) (n : QNode Var) (vis : List Nat) (c : G) vis' c',
    dfs g f n vis c = some (vis', c') → n.alias.id ∉ vis → Cand g n → CInv vis c →
    Step g vis c vis' c' ∧ CInv vis' c' ∧ n.alias.id ∈ vis'
  | 0, _, _, _, _, _, h, _, _, _ => by simp [dfs] at h
  | f + 1, n, vis, c, vis', c', h, hn, hcand, hc => by
    unfold dfs at h
    split at h
    · cases h
    · rename_i p e
      have hp := Option.some.inj h
      -- adding the node
      have hadd : (c.addNode n).2.nodes = c.nodes ++ [n] ∧ (c.addNode n).2.rels = c.rels := by
        rcases QG.addNode_spec c n with ⟨h1, _⟩ | ⟨_, e2⟩
        · exfalso
          obtain ⟨m, hm, hmeq⟩ := List.any_eq_true.1 h1
          exact hn ((Var.eq_spec _ _).1 hmeq ▸ hc m hm)
        · rw [e2]; exact ⟨rfl, rfl⟩
      have s0 : Step g vis c (vins n.alias.id vis) (c.addNode n).2 := by
        refine ⟨[n], hadd.1, by simp [ids], by simp [hn, mem_vins, hcand], fun v h => mem_vins.2 (.inl h), ?_, ?_⟩
        · intro v hv; rcases mem_vins.1 hv with h | h
          · exact .inl h
          · exact .inr (.inl (by simp [ids, h]))
        · intro r hr; rw [hadd.2] at hr; exact .inl hr
      have hc0 := s0.cinv hc
      have ⟨s1, hc1⟩ := dfsRels_step g f n g.rels (fun r h => h) _ _ _ _ e hc0
      have s2 := pathStep_step g g.paths (fun p h => h) p.1 p.2
      rw [hp] at s2
      have st := Step.trans (Step.trans s0 s1) s2
      obtain ⟨_, _, _, _, s12, _, _⟩ := Step.trans s1 s2
      exact ⟨st, st.cinv hc, s12 _ (mem_vins.2 (.inr rfl))⟩
theorem dfsRels_step (g : G) : ∀ (f : Nat) (n : QNode Var) (rs : List (QRel Var)),
    (∀ r ∈ rs, r ∈ g.rels) → ∀ (vis : List Nat) (c : G) vis' c',
    dfsRels g f n rs vis c = some (vis', c') → CInv vis c → Step g vis c vis' c' ∧ CInv vis' c'
  | _, _, [], _, vis, c, vis', c', h, hc => by
    simp [dfsRels] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨Step.refl g vis c, hc⟩
  | f, n, r :: rs, hrs, vis, c, vis', c', h, hc => by
    have hr : r ∈ g.rels := hrs r (by simp)
    have hrs' : ∀ q ∈ rs, q ∈ g.rels := fun q h => hrs q (List.mem_cons_of_mem _ h)
    have sp := relPart_step g r hr vis c
    have hcp := sp.cinv hc
    unfold dfsRels at h
    split at h
    · -- outgoing from n
      split at h
      · have ⟨s, hc'⟩ := dfsRels_step g f n rs hrs' _ _ _ _ h hcp
        exact ⟨sp.trans s, hc'⟩
      · split at h
        · cases h
        · rename_i hnot _ p e
          have ⟨s1, hc1, _⟩ := dfs_step g f r.dst _ _ _ _ e hnot (.inr ⟨r, hr, .inr rfl⟩) hcp
          have ⟨s2, hc2⟩ := dfsRels_step g f n rs hrs' _ _ _ _ h hc1
          exact ⟨(sp.trans s1).trans s2, hc2⟩
    · split at h
      · split at h
        · have ⟨s, hc'⟩ := dfsRels_step g f n rs hrs' _ _ _ _ h hcp
          exact ⟨sp.trans s, hc'⟩
        · split at h
          · cases h
          · rename_i hnot _ p e
            have ⟨s1, hc1, _⟩ := dfs_step g f r.src _ _ _ _ e hnot (.inr ⟨r, hr, .inl rfl⟩) hcp
            have ⟨s2, hc2⟩ := dfsRels_step g f n rs hrs' _ _ _ _ h hc1
            exact ⟨(sp.trans s1).trans s2, hc2⟩
      · exact dfsRels_step g f n rs hrs' _ _ _ _ h hc
end

/-- All nodes of a list of components, in order. -/
def allNodes (cs : List G) : List (QNode Var) := cs.flatMap (·.nodes)

theorem ccLoop_spec (g : G) (f : Nat) : ∀ (ns : List (QNode Var)) (vis : List Nat) (acc : List G) out,
    (∀ n ∈ ns, n ∈ g.nodes) → ccLoop g f ns vis acc = some out →
    (ids (allNodes acc)).Nodup → (∀ x ∈ allNodes acc, x.alias.id ∈ vis ∧ Cand g x) →
    (∀ v ∈ vis, v ∈ ids (allNodes acc) ∨ v ∈ relIds g ∨ v ∈ pathIds g) →
    (ids (allNodes out)).Nodup ∧ (∀ x ∈ allNodes out, Cand g x) ∧
    (∀ n ∈ ns, n.alias.id ∈ vis ∨ n.alias.id ∈ ids (allNodes out) ∨ n.alias.id ∈ relIds g ∨
      n.alias.id ∈ pathIds g) ∧
    (∀ v ∈ vis, v ∈ ids (allNodes out) ∨ v ∈ relIds g ∨ v ∈ pathIds g) ∧
    (∀ x ∈ allNodes acc, x ∈ allNodes out)
  | [], vis, acc, out, _, h, hd, hx, hv => by
    simp [ccLoop] at h; subst h
    exact ⟨hd, fun x h => (hx x h).2, by simp, hv, fun _ h => h⟩
  | n :: ns, vis, acc, out, hns, h, hd, hx, hv => by
    have hns' : ∀ m ∈ ns, m ∈ g.nodes := fun m h => hns m (List.mem_cons_of_mem _ h)
    unfold ccLoop at h
    split at h
    · obtain ⟨a, b, c, d, e⟩ := ccLoop_spec g f ns vis acc out hns' h hd hx hv
      refine ⟨a, b, ?_, d, e⟩
      intro m hm; rcases List.mem_cons.1 hm with rfl | hm
      · exact .inl ‹_›
      · exact c m hm
    · split at h
      · cases h
      · rename_i hnv _ vis' comp e
        have hcinv : CInv vis QG.empty := by simp [CInv, QG.empty]
        obtain ⟨⟨new, en, dn, fn, sn, on, _⟩, _, hnin⟩ :=
          dfs_step g f n vis QG.empty vis' comp e hnv (.inl (hns n (by simp))) hcinv
        simp [QG.empty] at en
        have hall : allNodes (acc ++ [comp]) = allNodes acc ++ new := by
          simp [allNodes, en]
        have hd' : (ids (allNodes (acc ++ [comp]))).Nodup := by
          rw [hall]; unfold ids at *; rw [List.map_append, List.nodup_append]
          refine ⟨hd, dn, ?_⟩
          intro a ha b hb hab; subst hab
          simp only [List.mem_map] at ha hb
          obtain ⟨x, hx', rfl⟩ := ha; obtain ⟨y, hy, hxy⟩ := hb
          exact (fn y hy).1 (hxy ▸ (hx x hx').1)
        have hx' : ∀ x ∈ allNodes (acc ++ [comp]), x.alias.id ∈ vis' ∧ Cand g x := by
          rw [hall]; intro x hm; rcases List.mem_append.1 hm with h1 | h1
          · exact ⟨sn _ (hx x h1).1, (hx x h1).2⟩
          · exact ⟨(fn x h1).2.1, (fn x h1).2.2⟩
        have hv' : ∀ v ∈ vis', v ∈ ids (allNodes (acc ++ [comp])) ∨ v ∈ relIds g ∨ v ∈ pathIds g := by
          rw [hall]; intro v hm
          rcases on v hm with h1 | h1 | h1 | h1
          · rcases hv v h1 with h2 | h2 | h2
            · exact .inl (by unfold ids at *; simp only [List.map_append, List.mem_append]; exact .inl h2)
            · exact .inr (.inl h2)
            · exact .inr (.inr h2)
          · exact .inl (by unfold ids at *; simp only [List.map_append, List.mem_append]; exact .inr h1)
          · exact .inr (.inl h1)
          · exact .inr (.inr h1)
        obtain ⟨a, b, c, d, e2⟩ := ccLoop_spec g f ns vis' (acc ++ [comp]) out hns' h hd' hx' hv'
        refine ⟨a, b, ?_, ?_, ?_⟩
        · intro m hm; rcases List.mem_cons.1 hm with rfl | hm
          · rcases d _ hnin with h1 | h1 | h1
            · exact .inr (.inl h1)
            · exact .inr (.inr (.inl h1))
            · exact .inr (.inr (.inr h1))
          · rcases c m hm with h1 | h1
            · rcases d _ h1 with h2 | h2 | h2
              · exact .inr (.inl h2)
              · exact .inr (.inr (.inl h2))
              · exact .inr (.inr (.inr h2))
            · exact .inr h1
        · intro v hm; rcases hv v hm with h1 | h1 | h1
          · exact .inl (by
              have : v ∈ ids (allNodes (acc ++ [comp])) := by
                rw [hall]; unfold ids at *; simp only [List.map_append, List.mem_append]; exact .inl h1
              unfold ids at *; simp only [List.mem_map] at this ⊢
              obtain ⟨x, hx2, rfl⟩ := this; exact ⟨x, e2 x hx2, rfl⟩)
          · exact .inr (.inl h1)
          · exact .inr (.inr h1)
        · intro x hm; apply e2; unfold allNodes at *; rw [List.flatMap_append]
          exact List.mem_append_left _ hm

/-- **connected_components** (whenever the model's budget suffices):
the components' nodes are pairwise distinct by id (no node is in two
components or twice in one), each is a node of the graph or an endpoint of
one of its relationships, and every node of the graph is in some component
*unless its id is also a relationship or path id* (`visited` is one set for
all three kinds of id, ast.rs:827,849,855,880). -/
theorem connectedComponents_spec (g : G) (f : Nat) (out : List G)
    (h : connectedComponents g f = some out) :
    (ids (allNodes out)).Nodup ∧ (∀ x ∈ allNodes out, Cand g x) ∧
    (∀ n ∈ g.nodes, n.alias.id ∈ ids (allNodes out) ∨ n.alias.id ∈ relIds g ∨
      n.alias.id ∈ pathIds g) := by
  obtain ⟨a, b, c, _, _⟩ := ccLoop_spec g f g.nodes [] [] out (fun _ h => h) h
    (by simp [ids, allNodes]) (by simp [allNodes]) (by simp)
  exact ⟨a, b, fun n hn => by rcases c n hn with h | h <;> simp_all⟩

/-- A two-node graph `(a)-[r]->(b)` is one component (concrete run). -/
def exG : G :=
  let a : QNode Var := ⟨⟨none, 0, 0⟩, [], leaf .map⟩
  let b : QNode Var := ⟨⟨none, 1, 0⟩, [], leaf .map⟩
  ⟨[a, b], [⟨⟨none, 2, 0⟩, [7], leaf .map, a, b, false, none, none, .no⟩], []⟩
#guard ((connectedComponents exG 10).map (·.length)) == some 1

/-- `QueryGraph::filter_visited` (ast.rs:795-821): keep the entities whose
`(id, scope_id)` is not in `visited`, re-adding them with `add_*`. -/
def filterVisited (g : G) (vis : List (Nat × Nat)) : G :=
  let g1 := (g.nodes.filter (fun n => !((n.alias.id, n.alias.scope) ∈ vis))).foldl
    (fun acc n => (acc.addNode n).2) QG.empty
  let g2 := (g.rels.filter (fun r => !((r.alias.id, r.alias.scope) ∈ vis))).foldl
    (fun acc r => (acc.addRel r).2) g1
  (g.paths.filter (fun p => !((p.var.id, p.var.scope) ∈ vis))).foldl
    (fun acc p => (acc.addPath p).2) g2

theorem foldAddNode_sub : ∀ (l : List (QNode Var)) (acc : G),
    ∀ x ∈ (l.foldl (fun acc n => (acc.addNode n).2) acc).nodes, x ∈ acc.nodes ∨ x ∈ l
  | [], acc, x, h => .inl h
  | n :: l, acc, x, h => by
    simp only [List.foldl_cons] at h
    rcases foldAddNode_sub l _ x h with h | h
    · rcases QG.addNode_spec acc n with ⟨_, e⟩ | ⟨_, e⟩ <;> rw [e] at h
      · exact .inl h
      · simp at h; rcases h with h | h
        · exact .inl h
        · exact .inr (by simp [h])
    · exact .inr (List.mem_cons_of_mem _ h)

theorem foldAddRel_nodes : ∀ (l : List (QRel Var)) (acc : G),
    (l.foldl (fun acc r => (acc.addRel r).2) acc).nodes = acc.nodes
  | [], _ => rfl
  | r :: l, acc => by simp only [List.foldl_cons]; rw [foldAddRel_nodes l, addRel_nodes]

theorem foldAddPath_nodes : ∀ (l : List (QPath Var)) (acc : G),
    (l.foldl (fun acc p => (acc.addPath p).2) acc).nodes = acc.nodes
  | [], _ => rfl
  | p :: l, acc => by
    simp only [List.foldl_cons]; rw [foldAddPath_nodes l]
    rcases QG.addPath_spec acc p with ⟨_, e⟩ | ⟨_, e⟩ <;> rw [e]

/-- **filter_visited** keeps only nodes of the graph whose `(id, scope)` is
not visited. -/
theorem filterVisited_nodes (g : G) (vis : List (Nat × Nat)) :
    ∀ x ∈ (filterVisited g vis).nodes, x ∈ g.nodes ∧ (x.alias.id, x.alias.scope) ∉ vis := by
  intro x hx
  simp only [filterVisited, foldAddPath_nodes, foldAddRel_nodes] at hx
  rcases foldAddNode_sub _ _ x hx with h | h
  · simp [QG.empty] at h
  · simp at h; exact ⟨h.1, h.2⟩

end FalkorParserGrammar.C
