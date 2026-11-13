import CchTopologyReview.Fresh

/-!
# Findings and breaks (design-level, no code to reproduce)

* **T2** (§6 "Built-entry jobs", S6's input order). `T2_literal_release_breaks_I1`,
  `T2_literal_release_then_alloc_shares_a_rank`, `T2_release_keeps_rinv`.
* Breaks: each shows that a premise or rule the proofs use is needed, on a concrete state.
  * `B1_merge_needs_disjoint`: without "delta disjoint from base", `up_ids` repeats a column (I3′ fails).
  * `B2_positional_id_breaks_after_s2`: after one S2, the id at position `i` of a row is not `Ap[lo] + i`, so
    every reader 03 lists under I3′ (`customize.rs:497-513` relax's `x_slots_start + i`, `query.rs:113`) must
    read the id, as 03 says.
  * `B3_fold_loses_unforked_write`: an S2 run on the published copy itself (seq = F, no fork) is dropped by the
    fold; `WLin`'s "every step on a copy above F" is what 03 §5 gives ("a writer's private version is
    `published.version + 1`").
  * `B4_fold_ge_duplicates`: a fold that keeps `seq ≥ F` instead of `seq > F` puts an arc in both the new base
    and the new delta, so `up_ids` repeats it. 03's `> F` is exact.
-/

namespace CchTopo

instance (t : Cch α) (e : Ent) : Decidable (upE t e) := by unfold upE inB; infer_instance

/-- A two-rank hierarchy at version 5 with no arcs: ranks 0 and 1 held by nodes 0 and 1. -/
def tiny : Cch Nat :=
  { ver := 5, reserve := 0, nr := 2, nodeOf := fun r => if r < 2 then some r else none,
    rankOf := fun x => if x < 2 then some x else none, free := [], upB := [], downB := [], upD := [], downD := [],
    cap := 0, added := [], met := [], stats := ⟨0, 0, 0, 0, 0⟩ }

/-- **B1.** Base and delta both hold column 2 of a row: the merge yields it twice. -/
theorem B1_merge_needs_disjoint : mergeRow [(2, 0)] [(2, 5)] = [(2, 5), (2, 0)] ∧ ¬ StrictCol [(2, 5), (2, 0)] := by
  refine ⟨by simp [mergeRow], ?_⟩
  simp [StrictCol]

/-- A fresh row `0 → {1, 3}` with ids 0, 1 (`Ap[0] = 0`). -/
def rowState : Cch Nat :=
  { tiny with
    nr := 4, nodeOf := (fun r => if r < 4 then some r else none),
    rankOf := (fun x => if x < 4 then some x else none), upB := [[(1, 0), (3, 1)]], cap := 2, met := [0, 0] }

/-- **B2.** After S2 adds `(0, 2)` with id 2, position 1 of row 0 holds id 2, not `Ap[0] + 1 = 1`. -/
theorem B2_positional_id_breaks_after_s2 :
    upIds (s2 0 (fork rowState) [(0, 2)]) 0 = [(1, 0), (2, 2), (3, 1)] := by
  simp [upIds, s2, fork, rowState, tiny, bRow, dRow, upEnts, insAll, sinsK, mergeRow]

/-- **B3.** An S2 on the published copy itself (seq 5 = the fold's fork) is lost by the fold: the arc is in the
    copy and not in the fold's result. With a fork (seq 6) it is kept. -/
theorem B3_fold_loses_unforked_write :
    upE (s2 0 tiny [(0, 1)]) ⟨0, 1, 0⟩ ∧ ¬ upE (fold 5 [] [] (s2 0 tiny [(0, 1)])) ⟨0, 1, 0⟩ ∧
    upE (fold 5 [] [] (s2 0 (fork tiny) [(0, 1)])) ⟨0, 1, 0⟩ := by
  refine ⟨by decide, by decide, by decide⟩

/-- **B4.** A fold forked at 6 over a state whose arc `(0, 1)` has seq 6: keeping `seq ≥ 6` (`keepAfter 5`)
    puts the arc in the new base and in the new delta. -/
theorem B4_fold_ge_duplicates :
    upIds (fold 5 [[(1, 0)]] [[], [(0, 0)]] (s2 0 (fork tiny) [(0, 1)])) 0 = [(1, 0), (1, 0)] ∧
    ¬ StrictCol [((1 : Nat), (0 : Nat)), (1, 0)] := by
  refine ⟨?_, by simp [StrictCol]⟩
  simp [upIds, fold, s2, fork, tiny, keepAfter, ents, bRow, dRow, upEnts, insAll, sinsK, mergeRow]

/-! ## T2: S6 must clear `rank_of` too -/

/-- Sorted insert into an ascending list of ranks. -/
def insN (x : Nat) : List Nat → List Nat
  | [] => [x]
  | y :: ys => if x < y then x :: y :: ys else if x = y then y :: ys else y :: insN x ys

theorem mem_insN (x : Nat) : ∀ (l : List Nat) (z : Nat), z ∈ insN x l ↔ z ∈ l ∨ z = x
  | [], z => by simp [insN]
  | y :: ys, z => by
    unfold insN
    split
    · simp only [List.mem_cons]
      constructor
      · rintro (h | h | h)
        · exact Or.inr h
        · exact Or.inl (Or.inl h)
        · exact Or.inl (Or.inr h)
      · rintro ((h | h) | h)
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr h)
        · exact Or.inl h
    · split
      · rename_i h; subst h; simp only [List.mem_cons]
        constructor
        · intro h; exact Or.inl h
        · rintro (h | h)
          · exact h
          · exact Or.inl h
      · simp only [List.mem_cons]; rw [mem_insN x ys z]
        constructor
        · rintro (h | h | h)
          · exact Or.inl (Or.inl h)
          · exact Or.inl (Or.inr h)
          · exact Or.inr h
        · rintro ((h | h) | h)
          · exact Or.inl h
          · exact Or.inr (Or.inl h)
          · exact Or.inr (Or.inr h)

theorem sorted_insN (x : Nat) : ∀ (l : List Nat), l.Pairwise (· < ·) → (insN x l).Pairwise (· < ·)
  | [], _ => by simp [insN]
  | y :: ys, h => by
    rw [List.pairwise_cons] at h
    unfold insN
    split
    · rename_i hxy
      refine List.pairwise_cons.2 ⟨fun z hz => ?_, List.pairwise_cons.2 h⟩
      rcases List.mem_cons.1 hz with e | e
      · omega
      · have := h.1 z e; omega
    · split
      · exact List.pairwise_cons.2 h
      · refine List.pairwise_cons.2 ⟨fun z hz => ?_, sorted_insN x ys h.2⟩
        rcases (mem_insN x ys z).1 hz with e | e
        · exact h.1 z e
        · omega

/-- 03 §6, S6's input order, read literally: "Ranks whose holders have no indexed edge at F go back to the
    pool (`node_of = None`)". Only `node_of` and the pool change (the pool kept ascending). -/
def releaseLit (t : Cch α) (r : Nat) : Cch α := { t with nodeOf := upd t.nodeOf r none, free := insN r t.free }

/-- Rank 0 held by node 7, no arcs. -/
def sixState : Cch Nat :=
  { tiny with
    nr := 1, nodeOf := (fun r => if r = 0 then some 7 else none),
    rankOf := (fun x => if x = 7 then some 0 else none), free := [] }

/-- **T2, literal reading.** Node 7 still has rank 0, whose `node_of` is now `None`: I1′ (`rank_of` and
    `node_of` inverse) fails. -/
theorem T2_literal_release_breaks_I1 :
    (releaseLit sixState 0).rankOf 7 = some 0 ∧ (releaseLit sixState 0).nodeOf 0 = none ∧ ¬ RInv (releaseLit sixState 0).ranks := by
  refine ⟨rfl, by simp [releaseLit, upd], fun h => ?_⟩
  have := (h.rank_node 7 0).1 (by simp [releaseLit, sixState, Cch.ranks])
  simp [releaseLit, upd, Cch.ranks] at this

/-- **T2, what follows.** The released rank 0 is free, so the next degree-1 node, 9, takes it; node 7 still
    claims it, and when 7 gains an edge again the allocation skips it (it "has" a rank): two nodes share
    rank 0, and 7's pairs are classified at a rank that belongs to 9. -/
theorem T2_literal_release_then_alloc_shares_a_rank :
    let t := allocOne (releaseLit sixState 0) 9 1
    t.rankOf 9 = some 0 ∧ t.rankOf 7 = some 0 ∧ (allocOne t 7 1).rankOf 7 = some 0 ∧ (allocOne t 7 1).nodeOf 0 = some 9 := by
  simp [allocOne, releaseLit, sixState, tiny, upd, insN]

/-- **The fixed reading**: releasing rank `r` from its holder `x` clears `rank_of[x]` as well and puts `r`
    in `free_ranks`. It keeps I1′ and P's first half. -/
def releaseFix (t : Cch α) (r x : Nat) : Cch α :=
  { t with nodeOf := upd t.nodeOf r none, rankOf := upd t.rankOf x none, free := insN r t.free }

theorem T2_release_keeps_rinv (t : Cch α) (h : RInv t.ranks) (r x : Nat) (hx : t.nodeOf r = some x) :
    RInv (releaseFix t r x).ranks := by
  have rn : ∀ x r, t.rankOf x = some r ↔ t.nodeOf r = some x := h.rank_node
  have nl : ∀ r x, t.nodeOf r = some x → r < t.nr := h.node_lt
  have fi : ∀ r, r ∈ t.free ↔ r < t.nr ∧ t.nodeOf r = none := h.free_iff
  have hrx : t.rankOf x = some r := (rn x r).2 hx
  have hr : r < t.nr := nl r x hx
  refine ⟨?_, ?_, ?_, sorted_insN r t.free h.free_asc⟩
  · intro s y hs
    simp only [releaseFix, Cch.ranks, upd] at hs ⊢
    split at hs
    · cases hs
    · exact nl s y hs
  · intro y s
    simp only [releaseFix, Cch.ranks, upd]
    by_cases hy : y = x <;> by_cases hs : s = r
    · rw [ite_eq_left hy, ite_eq_left hs]; simp
    · rw [ite_eq_left hy, ite_eq_right hs]
      constructor
      · intro e; cases e
      · intro e; subst hy
        have := (rn y s).2 e; rw [hrx] at this; exact absurd (Option.some.inj this).symm hs
    · rw [ite_eq_right hy, ite_eq_left hs]
      constructor
      · intro e; subst hs
        have := (rn y s).1 e; rw [hx] at this; exact absurd (Option.some.inj this).symm hy
      · intro e; cases e
    · rw [ite_eq_right hy, ite_eq_right hs]; exact rn y s
  · intro s
    simp only [releaseFix, Cch.ranks, upd]
    rw [mem_insN, fi s]
    by_cases hs : s = r
    · subst hs; simp [hr]
    · rw [ite_eq_right hs]; constructor
      · rintro (h | h)
        · exact h
        · exact absurd h hs
      · intro h; exact Or.inl h

end CchTopo
