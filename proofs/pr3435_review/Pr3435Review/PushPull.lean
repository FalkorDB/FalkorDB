import Pr3435Review.Dist
/-! # Claim 2: `bfs_levels_push_pull` computes BFS levels, for every push/pull choice

PR branch a76e66420 (28ab35368 + the review fix), `graph/src/runtime/functions/algo_procedures.rs`:

| here | there |
| --- | --- |
| `GB`, `GBSpec` | the GraphBLAS calls `GrB_vxm` :1141, `GrB_mxv` :1144, `GrB_Vector_nvals` :1150, `GrB_Vector_assign_INT64` :1155 (FFI, hypothesis structure; C API spec below) |
| `Heur`        | the switching heuristic :1116-1138 — `avg_degree`, `edges_unexplored`, `alpha`/`beta`, `frontier_edges`; **arbitrary** here |
| `rustHeur`    | that heuristic over an abstract float type (`Fl`), showing it is one instance |
| `init`        | the loop's start state after :1100-1114 succeeded (`v(src) = 0`, `q(src) = true`); the allocation / source-check prologue is `Alloc.bfsFull` |
| `run`         | the `loop` :1127-1160 (exit `k > max_level` :1128, step :1139-1148, `last_nq`/`nvals` :1149-1150, exit `nq == 0` :1151, `edges_unexplored -=` :1154, assign :1155-1158, `k += 1` :1159) |
| `wrap64`      | `k += 1` on `k: i64` (:1126, :1159) in a release build (wrapping; debug panics); never wraps (`step_inv` needs only `n ≤ 2^60`) |
| `bfsLevelsPushPull` | the loop from `init` (fuel `n + 1`); the whole fn `bfs_levels_push_pull` :1066-1163 is `Alloc.bfsFull` |

`GxB_Vector_Option_set_INT32(.., GxB_SPARSITY_CONTROL, ..)` (:1107, :1140, :1143)
is a storage hint and does not change a vector's contents (SuiteSparse user
guide, "sparsity control"); it is not modelled. A failing step or assign returns
`Err` through `fail` (:1089-1097, frees each non-null vector) (:1146-1148, :1156-1158); `run` returns the same
`Err` string (allocation and cleanup: `Alloc.lean`). The Rust `loop` has no fuel; `run` takes fuel and
`run_fuel_stable` / `run_fuel_any` (`PushPullFuel.lean`) show any fuel `≥ n` is never observed.

## FFI spec assumed (`GBSpec`, C API 2.0 / SuiteSparse:GraphBLAS user guide)

* `GrB_vxm(q, v, NULL, GxB_ANY_PAIR_BOOL, q, A, GrB_DESC_RSC)`: `C<!struct(M), replace> = u'A`;
  with `ANY_PAIR` the result pattern is `{ j < n | v(j) absent ∧ ∃ i, q(i) ∧ A(i,j) }`
  (PAIR ignores values, so element types of `A`/`q` are irrelevant).
* `GrB_mxv(q, v, NULL, GxB_ANY_PAIR_BOOL, AT, q, GrB_DESC_RSC)`: pattern
  `{ j < n | v(j) absent ∧ ∃ i, AT(j,i) ∧ q(i) }`.
* `GrB_Vector_assign_INT64(v, q, NULL, k, GrB_ALL, n, GrB_DESC_S)`: `v<struct(q)> = k`,
  no replace: `v'(j) = k` if `q(j)` is stored, else `v(j)`.
* `GrB_Vector_nvals(&nq, q)`: the number of stored entries.
* Input aliasing (`q` as both output and input) is allowed by the C API.
-/
namespace Pr3435.Bfs

structure GB where
  vxm : BVec → LVec → Pat → Option BVec
  mxv : BVec → LVec → Pat → Option BVec
  assign : LVec → BVec → Int → Option LVec
  nvals : BVec → Nat

/-- The C API behaviour of the four calls on size-`n` vectors (FFI hypothesis). -/
structure GBSpec (n : Nat) (g : GB) : Prop where
  vxm : ∀ q v A q', g.vxm q v A = some q' → ∀ j,
    (q' j = true ↔ j < n ∧ v j = none ∧ ∃ i, i < n ∧ q i = true ∧ A i j = true)
  mxv : ∀ q v AT q', g.mxv q v AT = some q' → ∀ j,
    (q' j = true ↔ j < n ∧ v j = none ∧ ∃ i, i < n ∧ AT j i = true ∧ q i = true)
  assign : ∀ v q k v', g.assign v q k = some v' → ∀ j, v' j = if q j then some k else v j
  nvals : ∀ q, g.nvals q = count n q

/-- **Pull = push.** With `AT` the transpose of `A`, a successful pull step and
a successful push step from the same `(q, v)` produce the same frontier. -/
theorem pull_eq_push {n : Nat} {g : GB} (hg : GBSpec n g) {A AT : Pat}
    (hT : ∀ i j, AT j i = A i j) {q : BVec} {v : LVec} {q1 q2 : BVec}
    (h1 : g.vxm q v A = some q1) (h2 : g.mxv q v AT = some q2) : q1 = q2 := by
  funext j
  have e : q1 j = true ↔ q2 j = true := by
    rw [hg.vxm q v A q1 h1 j, hg.mxv q v AT q2 h2 j]
    simp only [hT]
    constructor <;> rintro ⟨a, b, i, c, d, e⟩ <;> exact ⟨a, b, i, c, e, d⟩
  cases hq1 : q1 j <;> cases hq2 : q2 j <;> simp_all

/-! ## The switching heuristic, kept abstract -/

/-- Any heuristic: `choose h nq last_nq push` is the next direction (`true` =
push), `upd h nq` the bookkeeping after a non-empty level. -/
structure Heur (H : Type) where
  choose : H → Nat → Nat → Bool → Bool
  upd : H → Nat → H

/-- An abstract f64: only the operations the heuristic uses. -/
structure Fl where
  T : Type
  ofNat : Nat → T
  mul : T → T → T
  div : T → T → T
  sub : T → T → T
  lt : T → T → Bool

/-- The Rust heuristic (:1116-1138, :1154) over any float model: `h` is
`edges_unexplored`. -/
def rustHeur (F : Fl) (n nvals : Nat) : Heur F.T :=
  let avg := F.div (F.ofNat nvals) (F.ofNat (max n 1))
  let alpha := F.ofNat 14
  let beta := F.ofNat 24
  { choose := fun eu nq lastNq push =>
      let frontierEdges := F.mul (F.ofNat nq) avg
      if push then
        (if decide (nq > lastNq) && F.lt (F.div eu alpha) frontierEdges then false else true)
      else
        (if decide (nq < lastNq) && F.lt (F.ofNat nq) (F.div (F.ofNat n) beta) then true else false)
    upd := fun eu nq => F.sub eu (F.mul (F.ofNat nq) avg) }

/-! ## The loop -/

/-- Release-build `i64` addition (two's-complement wrap). -/
def wrap64 (x : Int) : Int := (x + 9223372036854775808) % 18446744073709551616 - 9223372036854775808

theorem wrap64_id {x : Int} (h1 : -9223372036854775808 ≤ x) (h2 : x < 9223372036854775808) :
    wrap64 x = x := by
  unfold wrap64; omega

/-- `GrB_INDEX_MAX + 1` = 2^60: the largest vector size `GrB_Vector_new` accepts. -/
abbrev GRB_DIM_MAX : Nat := 1152921504606846976

structure St (H : Type) where
  v : LVec
  q : BVec
  nq : Nat
  lastNq : Nat
  push : Bool
  k : Int
  h : H

section
variable {H : Type} (g : GB) (hr : Heur H) (A AT : Pat) (maxLevel : Int)

/-- The `loop` (:1127-1160). -/
def run : Nat → St H → Except String LVec
  | 0, _ => .error "fuel"
  | f + 1, s =>
    if 0 ≤ maxLevel ∧ maxLevel < s.k then .ok s.v else
    let push := hr.choose s.h s.nq s.lastNq s.push
    match (if push then g.vxm s.q s.v A else g.mxv s.q s.v AT) with
    | none => .error "BFS step failed"
    | some q' =>
      let nq' := g.nvals q'
      if nq' = 0 then .ok s.v else
      match g.assign s.v q' s.k with
      | none => .error "BFS level assign failed"
      | some v' => run f ⟨v', q', nq', s.nq, push, wrap64 (s.k + 1), hr.upd s.h nq'⟩

end

/-- The state after the two `setElement`s. At a76e66420 (:1110-1114) they are
checked, so the loop only starts when `src < n`; the `src < n` conjunct keeps the
pre-a76e664 behaviour (28ab35368 :1098-1099 ignored the codes, leaving both
vectors empty for `src ≥ n`) for `pre_a76e664_pushPull_src_out_of_range`. -/
def init {H : Type} (n src : Nat) (h0 : H) : St H :=
  ⟨fun j => if j = src ∧ src < n then some 0 else none,
   fun j => decide (j = src ∧ src < n), 1, 0, true, 1, h0⟩

/-- The loop of `bfs_levels_push_pull` from `init`; `h0` is the initial
`edges_unexplored`. Fuel `n + 1` (any fuel `≥ n` gives the same, `run_fuel_stable`).
At 28ab35368 this was the whole function after the (unchecked) prologue. -/
def bfsLevelsPushPull {H : Type} (g : GB) (hr : Heur H) (h0 : H) (A AT : Pat)
    (n src : Nat) (maxLevel : Int) : Except String LVec :=
  run g hr A AT maxLevel (n + 1) (init n src h0)

/-! ## The invariant -/

/-- Levels `0 … K-1` are done: `v` holds exactly the nodes at distance `< K`,
with their distance; `q` is level `K-1`; `k = K`. -/
structure Inv {H : Type} (n : Nat) (A : Pat) (src : Nat) (maxLevel : Int) (K : Nat) (s : St H) : Prop where
  k_eq : s.k = K
  k_pos : 1 ≤ K
  v_spec : ∀ j l, s.v j = some l ↔ ∃ d, d < K ∧ l = (d : Int) ∧ IsDist A src j d
  q_spec : ∀ j, s.q j = true ↔ IsDist A src j (K - 1)
  visited : K ≤ count n (fun j => (s.v j).isSome)
  bound : 0 ≤ maxLevel → (K : Int) ≤ maxLevel + 1

theorem Inv.k_le {H : Type} {n : Nat} {A : Pat} {src : Nat} {ml : Int} {K : Nat} {s : St H}
    (h : Inv n A src ml K s) : K ≤ n :=
  Nat.le_trans h.visited (count_le _ _)

theorem Inv.v_none {H : Type} {n : Nat} {A : Pat} {src : Nat} {ml : Int} {K : Nat} {s : St H}
    (h : Inv n A src ml K s) (j : Nat) : s.v j = none ↔ ∀ d, d < K → ¬ IsDist A src j d := by
  constructor
  · intro hv d hd hD
    have := (h.v_spec j d).2 ⟨d, hd, rfl, hD⟩
    rw [hv] at this; exact absurd this (by simp)
  · intro hno
    cases hv : s.v j with
    | none => rfl
    | some l =>
      obtain ⟨d, hd, _, hD⟩ := (h.v_spec j l).1 hv
      exact absurd hD (hno d hd)

/-- The frontier a step produces is exactly level `K`, push or pull. -/
theorem step_frontier {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) {A AT : Pat}
    (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    {ml : Int} {K : Nat} {s : St H} (hI : Inv n A src ml K s) (push : Bool) {q' : BVec}
    (hq : (if push then g.vxm s.q s.v A else g.mxv s.q s.v AT) = some q') :
    ∀ j, q' j = true ↔ IsDist A src j K := by
  have key : ∀ j, (j < n ∧ s.v j = none ∧ ∃ i, i < n ∧ s.q i = true ∧ A i j = true) ↔
      IsDist A src j K := by
    intro j
    rw [dist_step A src j K hI.k_pos]
    constructor
    · rintro ⟨_, hv, i, _, hqi, he⟩
      exact ⟨(hI.v_none j).1 hv, i, (hI.q_spec i).1 hqi, he⟩
    · rintro ⟨hno, i, hi, he⟩
      exact ⟨(hb i j he).2, (hI.v_none j).2 hno, i, (hb i j he).1, (hI.q_spec i).2 hi, he⟩
  intro j
  cases push with
  | true =>
    simp only [ite_true] at hq
    rw [hg.vxm _ _ _ _ hq j]; exact key j
  | false =>
    simp only [Bool.false_eq_true, ite_false] at hq
    rw [hg.mxv _ _ _ _ hq j, ← key j]
    simp only [hT]
    constructor
    · rintro ⟨a, b, i, c, d, e⟩; exact ⟨a, b, i, c, e, d⟩
    · rintro ⟨a, b, i, c, d, e⟩; exact ⟨a, b, i, c, e, d⟩

/-- What the result must be: the level of every node at distance `≤ max_level`
(every reachable node when `max_level < 0`). -/
def Levels (A : Pat) (src : Nat) (maxLevel : Int) (w : LVec) : Prop :=
  ∀ j l, w j = some l ↔ ∃ d : Nat, l = (d : Int) ∧ IsDist A src j d ∧ (maxLevel < 0 ∨ (d : Int) ≤ maxLevel)

/-- The result is `Ok`, or one of the two GraphBLAS-failure errors — never fuel. -/
def NoFuel (r : Except String LVec) : Prop :=
  (∃ w, r = .ok w) ∨ r = .error "BFS step failed" ∨ r = .error "BFS level assign failed"

/-- **Main lemma.** From any state satisfying the invariant at level `K`, with
enough fuel, `run` returns `Ok` only with the exact BFS levels, and never runs
out of fuel — for **every** heuristic, hence every push/pull sequence. -/
theorem run_correct {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) {A AT : Pat}
    (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n ≤ GRB_DIM_MAX) (ml : Int) :
    ∀ (f : Nat) (s : St H) (K : Nat), Inv n A src ml K s → n + 1 ≤ f + K →
      (∀ w, run g hr A AT ml f s = .ok w → Levels A src ml w) ∧ NoFuel (run g hr A AT ml f s) := by
  unfold GRB_DIM_MAX at hn
  intro f
  induction f with
  | zero => intro s K hI hf; have := hI.k_le; omega
  | succ f ih =>
    intro s K hI hf
    simp only [run]
    by_cases hx : 0 ≤ ml ∧ ml < s.k
    · -- `k > max_level`: levels `< K = max_level + 1` are exactly those `≤ max_level`.
      rw [if_pos hx]
      refine ⟨fun w hw => ?_, Or.inl ⟨_, rfl⟩⟩
      cases hw
      intro j l
      rw [hI.v_spec j l]
      have hk := hI.k_eq
      have hbd := hI.bound hx.1
      constructor
      · rintro ⟨d, hd, rfl, hD⟩; exact ⟨d, rfl, hD, Or.inr (by omega)⟩
      · rintro ⟨d, rfl, hD, hc⟩
        exact ⟨d, by rcases hc with hc | hc <;> omega, rfl, hD⟩
    · rw [if_neg hx]
      generalize hp : hr.choose s.h s.nq s.lastNq s.push = push
      generalize hq : (if push then g.vxm s.q s.v A else g.mxv s.q s.v AT) = res
      cases res with
      | none => exact ⟨fun w hw => (by cases hw), Or.inr (Or.inl rfl)⟩
      | some q' =>
        simp only
        have hfr := step_frontier hg hT hb hs hI push hq
        by_cases h0 : g.nvals q' = 0
        · -- `nq == 0`: no node at distance `K`, so none at any distance `≥ K`.
          rw [if_pos h0]
          refine ⟨fun w hw => ?_, Or.inl ⟨_, rfl⟩⟩
          cases hw
          rw [hg.nvals] at h0
          have hz := (count_eq_zero_iff n q').1 h0
          have none_K : ∀ j, ¬ IsDist A src j K := by
            intro j hD
            have := (hfr j).2 hD
            rw [hz j (dist_lt A src hb hs hD)] at this; exact absurd this (by simp)
          intro j l
          rw [hI.v_spec j l]
          constructor
          · rintro ⟨d, hd, rfl, hD⟩
            refine ⟨d, rfl, hD, ?_⟩
            by_cases hm : 0 ≤ ml
            · have := hI.bound hm; exact Or.inr (by omega)
            · exact Or.inl (by omega)
          · rintro ⟨d, rfl, hD, _⟩
            refine ⟨d, ?_, rfl, hD⟩
            apply Classical.byContradiction; intro hge
            obtain ⟨i, hi⟩ := dist_gap A src d j K (by omega) hD
            exact none_K i hi
        · rw [if_neg h0]
          generalize ha : g.assign s.v q' s.k = ra
          cases ra with
          | none => exact ⟨fun w hw => (by cases hw), Or.inr (Or.inr rfl)⟩
          | some v' =>
            simp only
            have hv' := hg.assign _ _ _ _ ha
            have hk := hI.k_eq
            -- `q'` is disjoint from the visited set, so the visited count grows by `nq' ≥ 1`.
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
            have hvis : K + 1 ≤ count n (fun j => (v' j).isSome) := by
              have := hI.visited; omega
            have hKn : K + 1 ≤ n := Nat.le_trans hvis (count_le _ _)
            have hI' : Inv n A src ml (K + 1) ⟨v', q', g.nvals q', s.nq, push, wrap64 (s.k + 1),
                hr.upd s.h (g.nvals q')⟩ := by
              refine ⟨?_, by omega, ?_, ?_, hvis, ?_⟩
              · show wrap64 (s.k + 1) = ((K + 1 : Nat) : Int)
                rw [hk, wrap64_id (by omega) (by omega)]; omega
              · intro j l
                show v' j = some l ↔ _
                rw [hv' j]
                cases hqj : q' j with
                | true =>
                  have hD := (hfr j).1 hqj
                  simp only [if_true, Option.some.injEq]
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
            exact ih _ (K + 1) hI' (by omega)

theorem init_inv {H : Type} {n src : Nat} (hs : src < n) (A : Pat) (ml : Int) (h0 : H) :
    Inv n A src ml 1 (init n src h0) := by
  refine ⟨rfl, Nat.le_refl _, ?_, ?_, ?_, ?_⟩
  · intro j l
    simp only [init]
    constructor
    · intro h
      split at h
      · rename_i hj
        simp only [Option.some.injEq] at h
        exact ⟨0, by omega, by rw [← h]; rfl, (dist_zero_iff A src j).2 hj.1⟩
      · exact absurd h (by simp)
    · rintro ⟨d, hd, rfl, hD⟩
      have : d = 0 := by omega
      subst this
      rw [if_pos ⟨(dist_zero_iff A src j).1 hD, hs⟩]; rfl
  · intro j
    simp only [init, Nat.sub_self, dist_zero_iff, decide_eq_true_eq]
    exact ⟨fun h => h.1, fun h => ⟨h, hs⟩⟩
  · simp only [init]
    have : (fun j => (if j = src ∧ src < n then some (0 : Int) else none).isSome) =
        (fun j => decide (j = src)) := by
      funext j; by_cases h : j = src <;> simp [h, hs]
    rw [this, count_single n src hs]; exact Nat.le_refl 1
  · intro _; omega

/-- **Claim 2.** For any graph (`Bounded` pattern `A` over `n` nodes, `AT` its
transpose), any source `src < n`, any `max_level` (`< 0` = unbounded), and
**any heuristic** (so any sequence of push/pull choices), if every GraphBLAS
call succeeds the result is `Ok w` with `w` exactly the BFS levels
(`Levels`); otherwise it is one of the two GraphBLAS-failure errors. Requires
`n ≤ 2^60` (`GRB_DIM_MAX`, which `GrB_Vector_new` enforces, so `Alloc.bfsFull`
discharges it) so that `k : i64` never wraps (`k_increment_no_overflow`). -/
theorem pushPull_correct {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) (h0 : H)
    {A AT : Pat} (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n ≤ GRB_DIM_MAX) (ml : Int) :
    (∀ w, bfsLevelsPushPull g hr h0 A AT n src ml = .ok w → Levels A src ml w) ∧
      NoFuel (bfsLevelsPushPull g hr h0 A AT n src ml) :=
  run_correct hg hr hT hb hs hn ml (n + 1) _ 1 (init_inv hs A ml h0) (by omega)

/-- The result does not depend on the heuristic at all: any two heuristics
(any two push/pull sequences) give the same `Ok` level vector. -/
theorem pushPull_heuristic_irrelevant {H₁ H₂ : Type} {n : Nat} {g : GB} (hg : GBSpec n g)
    (hr₁ : Heur H₁) (hr₂ : Heur H₂) (h₁ : H₁) (h₂ : H₂)
    {A AT : Pat} (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n ≤ GRB_DIM_MAX) (ml : Int) {w₁ w₂ : LVec}
    (e₁ : bfsLevelsPushPull g hr₁ h₁ A AT n src ml = .ok w₁)
    (e₂ : bfsLevelsPushPull g hr₂ h₂ A AT n src ml = .ok w₂) : w₁ = w₂ := by
  have L₁ := (pushPull_correct hg hr₁ h₁ hT hb hs hn ml).1 w₁ e₁
  have L₂ := (pushPull_correct hg hr₂ h₂ hT hb hs hn ml).1 w₂ e₂
  funext j
  cases hw : w₁ j with
  | none =>
    cases hw' : w₂ j with
    | none => rfl
    | some l => have := (L₁ j l).2 ((L₂ j l).1 hw'); rw [hw] at this; exact absurd this (by simp)
  | some l => exact ((L₂ j l).2 ((L₁ j l).1 hw)).symm

/-- In particular the Rust heuristic, over any float model. -/
theorem pushPull_correct_rust (F : Fl) (eu0 : F.T) (nvals : Nat) {n : Nat} {g : GB} (hg : GBSpec n g)
    {A AT : Pat} (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n ≤ GRB_DIM_MAX) (ml : Int) :
    (∀ w, bfsLevelsPushPull g (rustHeur F n nvals) eu0 A AT n src ml = .ok w → Levels A src ml w) ∧
      NoFuel (bfsLevelsPushPull g (rustHeur F n nvals) eu0 A AT n src ml) :=
  pushPull_correct hg _ eu0 hT hb hs hn ml

/-- If every GraphBLAS call succeeds, the result is `Ok`. -/
theorem pushPull_ok_of_total {H : Type} {n : Nat} {g : GB} (hg : GBSpec n g) (hr : Heur H) (h0 : H)
    {A AT : Pat} (hT : ∀ i j, AT j i = A i j) (hb : Bounded A n) {src : Nat} (hs : src < n)
    (hn : n ≤ GRB_DIM_MAX) (ml : Int)
    (hok : ∀ r, bfsLevelsPushPull g hr h0 A AT n src ml ≠ .error r) :
    ∃ w, bfsLevelsPushPull g hr h0 A AT n src ml = .ok w ∧ Levels A src ml w := by
  rcases (pushPull_correct hg hr h0 hT hb hs hn ml).2 with ⟨w, hw⟩ | h | h
  · exact ⟨w, hw, (pushPull_correct hg hr h0 hT hb hs hn ml).1 w hw⟩
  · exact absurd h (hok _)
  · exact absurd h (hok _)

end Pr3435.Bfs
