import GraphPersist.Persist.Pending
/-! # Multi-key save → load in any key order = the graph

`multi_load`: save a graph as `build_multi_key_payloads` lays it out (`K ≥ 2` keys), let
Redis hand the keys back in **any** order (`ord`, a permutation of `0..K`); every key reads
back (`key_rt`), the `DECODE_STATE` machine finalizes on the last one (`run_keys`), and the
finalized graph is exactly the single-key restore of the graph (`restoreOf`), while the
meta-key list is exactly the keys not named after the graph.
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

/-! ## Membership in the accumulated id lists -/

theorem mem_nodeIds (x : Nat) : ∀ P : List (PR M Tn), x ∈ nodeIds P ↔ ∃ es, PR.nodes es ∈ P ∧ x ∈ es.map Prod.fst
  | [] => by simp [nodeIds]
  | p :: ps => by
    have ih := mem_nodeIds x ps
    cases p <;> simp only [nodeIds, List.mem_append, ih, List.mem_cons, reduceCtorEq, false_or] <;>
      first
      | (constructor
         · rintro (h | ⟨es, h1, h2⟩)
           · exact ⟨_, Or.inl rfl, h⟩
           · exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1; exact Or.inl h2
           · exact Or.inr ⟨es, h1, h2⟩)
      | (constructor
         · rintro ⟨es, h1, h2⟩; exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1
           · exact ⟨es, h1, h2⟩)

theorem mem_edgeIds (x : Nat) : ∀ P : List (PR M Tn), x ∈ edgeIds P ↔ ∃ es, PR.edges es ∈ P ∧ x ∈ es.map Prod.fst
  | [] => by simp [edgeIds]
  | p :: ps => by
    have ih := mem_edgeIds x ps
    cases p <;> simp only [edgeIds, List.mem_append, ih, List.mem_cons, reduceCtorEq, false_or] <;>
      first
      | (constructor
         · rintro (h | ⟨es, h1, h2⟩)
           · exact ⟨_, Or.inl rfl, h⟩
           · exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1; exact Or.inl h2
           · exact Or.inr ⟨es, h1, h2⟩)
      | (constructor
         · rintro ⟨es, h1, h2⟩; exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1
           · exact ⟨es, h1, h2⟩)

theorem mem_delNIds (x : Nat) : ∀ P : List (PR M Tn), x ∈ delNIds P ↔ ∃ l, PR.delN l ∈ P ∧ x ∈ l
  | [] => by simp [delNIds]
  | p :: ps => by
    have ih := mem_delNIds x ps
    cases p <;> simp only [delNIds, List.mem_append, ih, List.mem_cons, reduceCtorEq, false_or] <;>
      first
      | (constructor
         · rintro (h | ⟨es, h1, h2⟩)
           · exact ⟨_, Or.inl rfl, h⟩
           · exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1; exact Or.inl h2
           · exact Or.inr ⟨es, h1, h2⟩)
      | (constructor
         · rintro ⟨es, h1, h2⟩; exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1
           · exact ⟨es, h1, h2⟩)

theorem mem_delEIds (x : Nat) : ∀ P : List (PR M Tn), x ∈ delEIds P ↔ ∃ l, PR.delE l ∈ P ∧ x ∈ l
  | [] => by simp [delEIds]
  | p :: ps => by
    have ih := mem_delEIds x ps
    cases p <;> simp only [delEIds, List.mem_append, ih, List.mem_cons, reduceCtorEq, false_or] <;>
      first
      | (constructor
         · rintro (h | ⟨es, h1, h2⟩)
           · exact ⟨_, Or.inl rfl, h⟩
           · exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1; exact Or.inl h2
           · exact Or.inr ⟨es, h1, h2⟩)
      | (constructor
         · rintro ⟨es, h1, h2⟩; exact ⟨es, Or.inr h1, h2⟩
         · rintro ⟨es, (h1 | h1), h2⟩
           · cases h1
           · exact ⟨es, h1, h2⟩)

/-! ## What each entry decodes to -/

theorem slice_mem (L : List Nat) (off cnt o : Nat) (h1 : off ≤ o) (h2 : o < off + cnt) (h3 : o < L.length) :
    L[o] ∈ (L.drop off).take cnt := by
  rw [List.mem_iff_getElem]
  refine ⟨o - off, by simp; omega, ?_⟩
  simp only [List.getElem_take, List.getElem_drop]
  congr 1; omega

theorem prOfE_nodes (g : G M Tn Nm Ix Cn) (e : Entry) (es : List (Nat × List (Option Nat × Val)))
    (h : prOfE g e = .nodes es) :
    e.st = .nodes ∧ es = (((live g.nodeCount g.delNodes).drop e.offset).take e.count).map (rawOf g.nodes) := by
  obtain ⟨st, c, o⟩ := e
  cases st <;> simp [prOfE, prOf] at h <;> simp [h]

theorem prOfE_edges (g : G M Tn Nm Ix Cn) (e : Entry) (es : List (Nat × List (Option Nat × Val)))
    (h : prOfE g e = .edges es) :
    e.st = .edges ∧ es = (((live g.edgeCount g.delEdges).drop e.offset).take e.count).map (rawOf g.edges) := by
  obtain ⟨st, c, o⟩ := e
  cases st <;> simp [prOfE, prOf] at h <;> simp [h]

theorem prOfE_delN (g : G M Tn Nm Ix Cn) (e : Entry) (l : List Nat) (h : prOfE g e = .delN l) :
    e.st = .delNodes ∧ l = (g.delNodes.drop e.offset).take e.count := by
  obtain ⟨st, c, o⟩ := e
  cases st <;> simp [prOfE, prOf] at h <;> simp [h]

theorem prOfE_delE (g : G M Tn Nm Ix Cn) (e : Entry) (l : List Nat) (h : prOfE g e = .delE l) :
    e.st = .delEdges ∧ l = (g.delEdges.drop e.offset).take e.count := by
  obtain ⟨st, c, o⟩ := e
  cases st <;> simp [prOfE, prOf] at h <;> simp [h]

theorem prOfE_faithful (g : G M Tn Nm Ix Cn) (e : Entry) : Faithful g (prOfE g e) := by
  obtain ⟨st, c, o⟩ := e
  cases st <;> simp only [prOfE, prOf, Faithful] <;> first | trivial | exact ⟨_, rfl⟩

/-! ## The layout's keys -/

/-- Key `k`'s entries. -/
def keyAt (keys : List (List Entry)) (k : Nat) : List Entry := keys[k]?.getD []

theorem mem_of_keysFrom (g : G M Tn Nm Ix Cn) (vmax : Nat) (e : Entry)
    (he : e ∈ (keysFrom (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)).flatten) :
    ∃ k < (buildMulti g vmax).length, e ∈ keyAt (buildMulti g vmax) k := by
  obtain ⟨l, hl, hel⟩ := List.mem_flatten.1 he
  obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.1 hl
  simp only [buildMulti]
  split
  · rename_i heq; simp [heq] at hk
  · rename_i k0 ks heq
    simp only [heq] at hk hel
    refine ⟨k, by simpa using hk, ?_⟩
    cases k with
    | zero => simp [keyAt]; exact Or.inl hel
    | succ k =>
      simp only [keyAt, List.getElem?_cons_succ]
      simp only [List.length_cons] at hk
      rw [List.getElem?_eq_getElem (by omega)]
      simpa using hel

theorem keyAt_cases (g : G M Tn Nm Ix Cn) (vmax : Nat) (k : Nat) (e : Entry)
    (he : e ∈ keyAt (buildMulti g vmax) k) :
    (e ∈ (keysFrom (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)).flatten) ∨
      (k = 0 ∧ e ∈ matrixEntries g) := by
  simp only [keyAt, buildMulti] at he
  split at he
  · simp at he
  · rename_i k0 ks heq
    rw [heq]
    cases k with
    | zero =>
      simp at he
      rcases he with he | he
      · exact Or.inl (List.mem_flatten.2 ⟨k0, by simp, he⟩)
      · exact Or.inr ⟨rfl, he⟩
    | succ k =>
      simp at he
      left
      cases hks : ks[k]? with
      | none => simp [hks] at he
      | some l => simp [hks] at he; exact List.mem_flatten.2 ⟨l, by simp [List.mem_iff_getElem?]; exact ⟨k + 1, by simpa using hks⟩, he⟩

end GraphPersist.Persist
