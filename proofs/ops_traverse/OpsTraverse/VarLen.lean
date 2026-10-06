import OpsTraverse.Basic
/-
# CondVarLenTraverse: the DFS enumerates exactly the trails of the requested length

| here | there |
| --- | --- |
| `step`             | `get_node_relationships_by_type(current, types, direction)` cached in `adj_cache` (`cond_var_len_traverse.rs:239-242`); `bidir` = `EdgeDirection::Both` |
| `dfs`              | `VarLenIter::advance` (`cond_var_len_traverse.rs:196-386`): skip used edges (`:247`), `will_emit = hop >= min_hops` (`:314`), `will_continue = hop < max_hops` (`:319`), push `(dest, used+[e], hop)` |
| fuel               | `max_hops - depth` (`if hop > max_hops continue`, `:222`) |
| `varLen`           | `VarLenIter::begin_start_node` (`:152-183`, the 0-hop emission when `min_hops == 0`) + the DFS |
| `Walk`, trail      | openCypher variable-length semantics: a walk whose relationships are pairwise distinct |

Modelled as a recursive DFS; the engine uses an explicit stack, which changes
only the *order* of emissions (it pops the last pushed neighbour first), not
the multiset. Destination id / label filters and edge-attribute / WHERE
filters are pure post-filters on each candidate and are not modelled; the
reversed traversal (`reversed`, `:543`) is the same DFS on the transposed
direction.
-/

namespace OpsTraverse.VarLen

inductive Walk (g : Graph) (b : Bool) : Nat → List Nat → Nat → Prop
  | nil (s : Nat) : Walk g b s [] s
  | cons {s n t e : Nat} {es : List Nat} :
      (e, n) ∈ step g b s → Walk g b n es t → Walk g b s (e :: es) t

def dfs (g : Graph) (b : Bool) (minH : Nat) :
    Nat → Nat → List Nat → Nat → List (List Nat × Nat)
  | 0, _, _, _ => []
  | fuel + 1, cur, used, depth =>
    (step g b cur).flatMap fun p =>
      if p.1 ∈ used then []
      else (if minH ≤ depth + 1 then [(used ++ [p.1], p.2)] else []) ++
        dfs g b minH fuel p.2 (used ++ [p.1]) (depth + 1)

def varLen (g : Graph) (b : Bool) (start minH maxH : Nat) : List (List Nat × Nat) :=
  (if minH = 0 then [([], start)] else []) ++ dfs g b minH maxH start [] 0

theorem dfs_sound {g : Graph} {b : Bool} {minH : Nat} :
    ∀ (fuel cur : Nat) (used : List Nat) (depth : Nat) (es : List Nat) (d : Nat),
      (es, d) ∈ dfs g b minH fuel cur used depth →
      ∃ suf, es = used ++ suf ∧ suf ≠ [] ∧ Walk g b cur suf d ∧ (∀ e ∈ suf, e ∉ used) ∧
        suf.Nodup ∧ minH ≤ depth + suf.length ∧ suf.length ≤ fuel
  | 0, _, _, _, _, _, h => by simp [dfs] at h
  | fuel + 1, cur, used, depth, es, d, h => by
    simp only [dfs, List.mem_flatMap] at h
    obtain ⟨⟨e, n⟩, hp, hm⟩ := h
    by_cases hu : e ∈ used
    · simp [hu] at hm
    · simp only [hu, if_false, List.mem_append] at hm
      rcases hm with hm | hm
      · by_cases hmin : minH ≤ depth + 1
        · simp only [hmin, if_true, List.mem_singleton, Prod.mk.injEq] at hm
          obtain ⟨rfl, rfl⟩ := hm
          refine ⟨[e], rfl, by simp, Walk.cons hp (Walk.nil _), ?_, by simp, by simpa using hmin,
            by simp⟩
          simpa using hu
        · simp [hmin] at hm
      · obtain ⟨suf, rfl, _, hw, hfresh, hnd, hmin, hlen⟩ := dfs_sound fuel n _ _ _ _ hm
        refine ⟨e :: suf, by simp, by simp, Walk.cons hp hw, ?_, ?_, ?_, ?_⟩
        · intro x hx
          rcases List.mem_cons.1 hx with rfl | hx
          · exact hu
          · exact fun h => hfresh x hx (List.mem_append_left _ h)
        · refine List.nodup_cons.2 ⟨fun h => hfresh e h (by simp), hnd⟩
        · simp only [List.length_cons]; omega
        · simp only [List.length_cons]; omega

theorem dfs_complete {g : Graph} {b : Bool} {minH : Nat} :
    ∀ (suf : List Nat) (fuel cur : Nat) (used : List Nat) (depth d : Nat),
      Walk g b cur suf d → suf ≠ [] → (∀ e ∈ suf, e ∉ used) → suf.Nodup →
      minH ≤ depth + suf.length → suf.length ≤ fuel →
      (used ++ suf, d) ∈ dfs g b minH fuel cur used depth
  | [], _, _, _, _, _, _, hne, _, _, _, _ => absurd rfl hne
  | e :: es, fuel, cur, used, depth, d, hw, _, hfresh, hnd, hmin, hlen => by
    cases hw with
    | cons hp hw' =>
      rename_i n
      cases fuel with
      | zero => simp at hlen
      | succ f =>
        simp only [dfs, List.mem_flatMap]
        refine ⟨(e, n), hp, ?_⟩
        have hu : e ∉ used := hfresh e (List.mem_cons_self ..)
        simp only [hu, if_false, List.mem_append]
        cases es with
        | nil =>
          cases hw'
          left
          have : minH ≤ depth + 1 := by simpa using hmin
          simp [this]
        | cons e2 es2 =>
          right
          have h := dfs_complete (g := g) (b := b) (minH := minH) (e2 :: es2) f n (used ++ [e]) (depth + 1) d hw' (by simp)
            (by
              intro x hx hmem
              rcases List.mem_append.1 hmem with h | h
              · exact hfresh x (List.mem_cons_of_mem _ hx) h
              · simp only [List.mem_singleton] at h
                subst h
                exact (List.nodup_cons.1 hnd).1 hx)
            (List.nodup_cons.1 hnd).2
            (by simp only [List.length_cons] at hmin ⊢; omega)
            (by simp only [List.length_cons] at hlen ⊢; omega)
          simpa using h

/-- **Soundness**: every emitted `(edges, dest)` is a trail from `start` to
`dest` (a walk with pairwise-distinct relationships) whose length lies in
`[minH, maxH]`. -/
theorem varLen_sound {g : Graph} {b : Bool} {start minH maxH : Nat} {es : List Nat} {d : Nat}
    (h : (es, d) ∈ varLen g b start minH maxH) :
    Walk g b start es d ∧ es.Nodup ∧ minH ≤ es.length ∧ es.length ≤ maxH := by
  simp only [varLen, List.mem_append] at h
  rcases h with h | h
  · by_cases h0 : minH = 0
    · rw [if_pos h0] at h
      simp only [List.mem_singleton, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨Walk.nil _, by simp, by simp [h0], by simp⟩
    · simp [h0] at h
  · obtain ⟨suf, rfl, _, hw, _, hnd, hmin, hlen⟩ := dfs_sound _ _ _ _ _ _ h
    exact ⟨by simpa using hw, hnd, by simpa using hmin, by simpa using hlen⟩

/-- **Completeness**: every such trail is emitted. -/
theorem varLen_complete {g : Graph} {b : Bool} {start minH maxH : Nat} {es : List Nat} {d : Nat}
    (hw : Walk g b start es d) (hnd : es.Nodup) (hmin : minH ≤ es.length)
    (hmax : es.length ≤ maxH) : (es, d) ∈ varLen g b start minH maxH := by
  simp only [varLen, List.mem_append]
  cases es with
  | nil =>
    cases hw
    left
    have : minH = 0 := by simpa using hmin
    simp [this]
  | cons e es =>
    right
    have := dfs_complete (minH := minH) (e :: es) maxH start [] 0 d hw (by simp) (by simp) hnd
      (by simpa using hmin) hmax
    simpa using this

/-- Together: the operator's result set is exactly the openCypher match set. -/
theorem varLen_iff {g : Graph} {b : Bool} {start minH maxH : Nat} {es : List Nat} {d : Nat} :
    (es, d) ∈ varLen g b start minH maxH ↔
      Walk g b start es d ∧ es.Nodup ∧ minH ≤ es.length ∧ es.length ≤ maxH :=
  ⟨varLen_sound, fun ⟨hw, hnd, h1, h2⟩ => varLen_complete hw hnd h1 h2⟩

/-- `*0..0` yields exactly the start node. -/
theorem zero_zero (g : Graph) (b : Bool) (s : Nat) : varLen g b s 0 0 = [([], s)] := by
  simp [varLen, dfs]

/-! ## Concrete checks -/

/-- Undirected self-loop: matched once (openCypher), C reports it twice. -/
theorem undirected_self_loop_once : varLen [⟨0, 3, 3⟩] true 3 1 5 = [([0], 3)] := by decide

/-- The repro graph `e0,e1 : 1→2`, `e2 : 2→3`, undirected `*1..2` from 1: 6 trails
(`lean_ops_traverse::agrees_var_len_trails`; C 6). -/
theorem six_trails :
    (varLen [⟨0, 1, 2⟩, ⟨1, 1, 2⟩, ⟨2, 2, 3⟩] true 1 1 2).length = 6 := by decide

/-- Shared deviation (C does the same): the DFS knows nothing about edges bound
by *sibling* relationships of the same MATCH, so in
`(a)-[r]->(b)-[*1..1]->(c)`-shaped plans that are not rewritten to a
single hop, e.g. `(a)-[*1..1]->(b)-[r]->(c)`, the self-loop `e0` is used by
both. openCypher forbids it. -/
theorem no_sibling_uniqueness : ([0], 3) ∈ varLen [⟨0, 3, 3⟩] false 3 1 1 := by decide

end OpsTraverse.VarLen
