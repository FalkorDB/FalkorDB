import Pr3435Review.PushPull
/-! # Claim 2, continued: the loop's exits, its fuel, and `k: i32`

* `step_visited` — each level that does not end the loop adds `nq ≥ 1` new
  nodes, so `k` never exceeds the number of visited nodes, which never exceeds `n`.
* `step_inv` — one full iteration (step, `nq != 0`, assign, `k += 1`) keeps `Inv`.
* `run_fuel_stable` — the fuel is never observed: Rust's unbounded `loop` is `run`
  with any fuel `≥ n`.
* `overflow_needs_huge_graph` — `k += 1` (:1144) can only wrap/panic when a node is
  at shortest-path distance `2^31 - 1` from the source, so `n ≥ 2^31` nodes and a
  shortest path of `2^31 - 1` edges.
* `exit_conditions` — the loop's two exits are exactly `k > max_level`
  (with `max_level ≥ 0`) and an empty next level.
-/
namespace Pr3435.Bfs

theorem step_visited {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) {A AT : Pat}
    (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    {ml : Int} {K : Nat} {s : St H} (hI : Inv n A src ml K s) (push : Bool) {q' : BVec}
    (hq : (if push then g.vxm s.q s.v A else g.mxv s.q s.v AT) = some q')
    (h0 : g.nvals q' ≠ 0) {v' : LVec} (ha : g.assign s.v q' s.k = some v') :
    K + 1 ≤ count n (fun j => (v' j).isSome) ∧ (∀ j, q' j = true ↔ IsDist A src j K) := by
  have hfr := step_frontier hg hT hb hs hI push hq
  have hv' := hg.assign _ _ _ _ ha
  have hdisj : ∀ j, q' j = true → (s.v j).isSome = false := by
    intro j hj
    have hD := (hfr j).1 hj
    have : s.v j = none := (hI.v_none j).2 (fun d hd hd' =>
      absurd (dist_unique A src hD hd') (by omega))
    simp [this]
  have hcount : count n (fun j => (v' j).isSome) =
      count n (fun j => (s.v j).isSome) + count n q' := by
    rw [← count_or n _ _ hdisj]
    congr 1; funext j; rw [hv' j]; cases q' j <;> simp
  have hpos : 0 < count n q' := by rw [hg.nvals] at h0; omega
  have := hI.visited
  exact ⟨by omega, hfr⟩

/-- One non-final iteration keeps the invariant (needs `n < 2^31` for `k`). -/
theorem step_inv {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) {A AT : Pat}
    (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n < 2147483648) {ml : Int} {K : Nat} {s : St H} (hI : Inv n A src ml K s)
    (hx : ¬ (0 ≤ ml ∧ ml < s.k)) (push : Bool) {q' : BVec}
    (hq : (if push then g.vxm s.q s.v A else g.mxv s.q s.v AT) = some q')
    (h0 : g.nvals q' ≠ 0) {v' : LVec} (ha : g.assign s.v q' s.k = some v') :
    Inv n A src ml (K + 1) ⟨v', q', g.nvals q', s.nq, push, wrap32 (s.k + 1), hr.upd s.h (g.nvals q')⟩ := by
  obtain ⟨hvis, hfr⟩ := step_visited hg hT hb hs hI push hq h0 ha
  have hv' := hg.assign _ _ _ _ ha
  have hk := hI.k_eq
  have hKn : K + 1 ≤ n := Nat.le_trans hvis (count_le _ _)
  refine ⟨?_, by omega, ?_, ?_, hvis, ?_⟩
  · show wrap32 (s.k + 1) = ((K + 1 : Nat) : Int)
    rw [hk, wrap32_id (by omega) (by omega)]; omega
  · intro j l
    show v' j = some l ↔ _
    rw [hv' j]
    cases hqj : q' j with
    | true =>
      have hD := (hfr j).1 hqj
      simp only [ite_true, Option.some.injEq]
      constructor
      · rintro rfl; exact ⟨K, by omega, by rw [hk], hD⟩
      · rintro ⟨d, _, rfl, hd⟩
        rw [hk, dist_unique A src hD hd]
    | false =>
      simp only [Bool.false_eq_true, ite_false]
      rw [hI.v_spec j l]
      constructor
      · rintro ⟨d, hd, rfl, hD⟩; exact ⟨d, by omega, rfl, hD⟩
      · rintro ⟨d, hd, rfl, hD⟩
        refine ⟨d, ?_, rfl, hD⟩
        rcases Nat.lt_or_ge d K with h | h
        · exact h
        · have : d = K := by omega
          subst this
          have := (hfr j).2 hD; rw [hqj] at this; exact absurd this (by simp)
  · intro j; show q' j = true ↔ _; rw [hfr j]; simp
  · intro hm; have : ¬ (ml < s.k) := fun h => hx ⟨hm, h⟩
    omega

/-- **The fuel is never observed**: one more unit of fuel changes nothing, so
every fuel `≥ n` gives the result of Rust's unbounded `loop`. -/
theorem run_fuel_stable {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) {A AT : Pat}
    (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n < 2147483648) (ml : Int) :
    ∀ (f : Nat) (s : St H) (K : Nat), Inv n A src ml K s → n + 1 ≤ f + K →
      run g hr A AT ml (f + 1) s = run g hr A AT ml f s := by
  intro f
  induction f with
  | zero => intro s K hI hf; have := hI.k_le; omega
  | succ f ih =>
    intro s K hI hf
    -- unfold one iteration on both sides
    show run g hr A AT ml (f + 1 + 1) s = run g hr A AT ml (f + 1) s
    simp only [run]
    by_cases hx : 0 ≤ ml ∧ ml < s.k
    · rw [if_pos hx, if_pos hx]
    · rw [if_neg hx, if_neg hx]
      generalize hp : hr.choose s.h s.nq s.lastNq s.push = push
      generalize hq : (if push then g.vxm s.q s.v A else g.mxv s.q s.v AT) = res
      cases res with
      | none => rfl
      | some q' =>
        simp only
        by_cases h0 : g.nvals q' = 0
        · rw [if_pos h0, if_pos h0]
        · rw [if_neg h0, if_neg h0]
          generalize ha : g.assign s.v q' s.k = ra
          cases ra with
          | none => rfl
          | some v' =>
            simp only
            exact ih _ (K + 1) (step_inv hg hr hT hb hs hn hI hx push hq h0 ha) (by omega)

/-- Every fuel `≥ n` gives the same answer as `bfsLevelsPushPull` (fuel `n + 1`). -/
theorem run_fuel_any {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) (h0 : H)
    {A AT : Pat} (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n < 2147483648) (ml : Int) :
    ∀ e, run g hr A AT ml (n + e) (init n src h0) = bfsLevelsPushPull g hr h0 A AT n src ml := by
  intro e
  induction e with
  | zero =>
    exact (run_fuel_stable hg hr hT hb hs hn ml n _ 1 (init_inv hs A ml h0) (by omega)).symm
  | succ e ih =>
    rw [← ih]
    exact run_fuel_stable hg hr hT hb hs hn ml (n + e) _ 1 (init_inv hs A ml h0) (by omega)

/-- **When can `k: i32` overflow?** At the `k += 1` (:1144) of a level `K` that
did not end the loop, there is a node at shortest-path distance exactly `K` and
at least `K + 1` nodes. So `k = i32::MAX` there needs a node at distance
`2^31 - 1`, i.e. `n ≥ 2^31` nodes on one shortest path. No `n < 2^31` premise. -/
theorem overflow_needs_huge_graph {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) {A AT : Pat}
    (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    {ml : Int} {K : Nat} {s : St H} (hI : Inv n A src ml K s) (push : Bool) {q' : BVec}
    (hq : (if push then g.vxm s.q s.v A else g.mxv s.q s.v AT) = some q')
    (h0 : g.nvals q' ≠ 0) {v' : LVec} (ha : g.assign s.v q' s.k = some v') :
    (∃ j, IsDist A src j K) ∧ K + 1 ≤ n := by
  obtain ⟨hvis, hfr⟩ := step_visited hg hT hb hs hI push hq h0 ha
  refine ⟨?_, Nat.le_trans hvis (count_le _ _)⟩
  rw [hg.nvals] at h0
  apply Classical.byContradiction; intro hne
  apply h0; apply (count_eq_zero_iff n q').2
  intro j _
  cases hqj : q' j
  · rfl
  · exact absurd ⟨j, (hfr j).1 hqj⟩ hne

/-- In the state the invariant describes, `k ≤ n` (`Inv.k_le`): the level
counter is bounded by the node count. -/
theorem k_le_n {H : Type} {n : Nat} {A : Pat} {src : Nat} {ml : Int} {K : Nat} {s : St H}
    (h : Inv n A src ml K s) : s.k ≤ n := by rw [h.k_eq]; have := h.k_le; omega

/-- **The two exits.** `run` stops on fuel `f + 1` without recursing exactly
when `max_level ≥ 0 ∧ k > max_level`, or the step fails, or the next level is
empty (`nq == 0`), or the assign fails. -/
theorem exit_conditions {H : Type} (g : GB) (hr : Heur H) (A AT : Pat) (ml : Int) (f : Nat) (s : St H) :
    (0 ≤ ml ∧ ml < s.k → run g hr A AT ml (f + 1) s = .ok s.v) ∧
    (∀ q', ¬ (0 ≤ ml ∧ ml < s.k) →
      (if hr.choose s.h s.nq s.lastNq s.push then g.vxm s.q s.v A else g.mxv s.q s.v AT) = some q' →
      g.nvals q' = 0 → run g hr A AT ml (f + 1) s = .ok s.v) := by
  refine ⟨fun hx => by simp only [run]; rw [if_pos hx], fun q' hx hq h0 => ?_⟩
  simp only [run]; rw [if_neg hx]; simp only [hq, h0, ite_true]

end Pr3435.Bfs
