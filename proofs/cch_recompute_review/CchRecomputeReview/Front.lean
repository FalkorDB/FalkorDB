import CchRecomputeReview.Main
import CchRecomputeReview.Weights

/-!
# From `changed` pairs to the front, and the end-to-end statement

`recustomize_seeded` resolves its `changed` node pairs before writing anything (`partial.rs:106-126` @08c):
self-loops are skipped, each pair is ordered by rank, a pair with an arc enters the front as `(lo, slot)`, and
the front is sorted and deduplicated (`:125-126`). In rank space (`rank_of` is a bijection, I1) that is
`frontOf`. Not modelled here (and nothing is claimed about what they write): `NeedsRebuild` for `u >= n`
(`:115-116`), the `n/8` valve (`:131-133`, `NeedsFull`: the caller runs the full customization, Theorem 1), the
no-arc pairs (`:137-145`: ignored without an edge, `NeedsRebuild` with one, which is 03's premise "every edge has
an arc before the recompute"), and the `InvalidWeight` checks (`:150-161`).

`recustomize_seeded_exact`: if `changed` names (either way round) every arc whose seed changed, which is
`partial.rs:79-81`'s precondition and 03 §6b's "the front holds every changed seed", the scoped pass equals
`relax` for the new seeds, arc for arc, weights and middles.
-/

namespace CchRecompute

variable {W : Type}

/-- `partial.rs:119`: `if ru < rv { (ru, rv) } else { (rv, ru) }`. -/
def ordered (u v : Nat) : Nat × Nat := if u < v then (u, v) else (v, u)

/-- `partial.rs:111-124`: skip `u == v`, order by rank, keep a pair that has an arc. -/
def resolve (T : Topo) : List (Nat × Nat) → List (Nat × Nat)
  | [] => []
  | (u, v) :: t =>
    if u = v then resolve T t
    else if (ordered u v).2 ∈ T.up (ordered u v).1 then ordered u v :: resolve T t else resolve T t

/-- `front.sort_unstable(); front.dedup()`: the heap order does not matter (`PickSpec`), only the set. -/
def dedupL : List (Nat × Nat) → List (Nat × Nat)
  | [] => []
  | p :: t => if p ∈ t then dedupL t else p :: dedupL t

def frontOf (T : Topo) (changed : List (Nat × Nat)) : List (Nat × Nat) := dedupL (resolve T changed)

theorem mem_dedupL : ∀ {l : List (Nat × Nat)} {p : Nat × Nat}, p ∈ dedupL l ↔ p ∈ l
  | [], _ => by simp [dedupL]
  | q :: t, p => by
    simp only [dedupL]
    split
    · rename_i h
      rw [mem_dedupL]
      constructor
      · exact List.mem_cons_of_mem _
      · intro hp
        rcases List.mem_cons.mp hp with rfl | hp
        · exact h
        · exact hp
    · simp [mem_dedupL]

theorem dedupL_nodup : ∀ (l : List (Nat × Nat)), (dedupL l).Nodup
  | [] => List.nodup_nil
  | q :: t => by
    simp only [dedupL]
    split
    · exact dedupL_nodup t
    · rename_i h
      exact List.nodup_cons.mpr ⟨fun hm => h (mem_dedupL.mp hm), dedupL_nodup t⟩

theorem resolve_arc (T : Topo) : ∀ (l : List (Nat × Nat)), ∀ p ∈ resolve T l, p.2 ∈ T.up p.1
  | [], _, h => by simp [resolve] at h
  | (u, v) :: t, p, h => by
    simp only [resolve] at h
    split at h
    · exact resolve_arc T t p h
    · split at h
      · rcases List.mem_cons.mp h with rfl | h
        · assumption
        · exact resolve_arc T t p h
      · exact resolve_arc T t p h

theorem mem_resolve {T : Topo} (hV : T.Valid) {lo hi : Nat} (harc : hi ∈ T.up lo) :
    ∀ (l : List (Nat × Nat)), ((lo, hi) ∈ l ∨ (hi, lo) ∈ l) → (lo, hi) ∈ resolve T l
  | [], h => by simp at h
  | (u, v) :: t, h => by
    have hlt := hV.above lo hi harc
    simp only [resolve]
    have ih : ((lo, hi) ∈ t ∨ (hi, lo) ∈ t) → (lo, hi) ∈ resolve T t := mem_resolve hV harc t
    by_cases huv : u = v
    · simp only [huv, ↓reduceIte]
      apply ih
      rcases h with h | h <;> rcases List.mem_cons.mp h with h | h
      · cases h; omega
      · exact Or.inl h
      · cases h; omega
      · exact Or.inr h
    · simp only [huv, ↓reduceIte]
      by_cases hhead : (u, v) = (lo, hi) ∨ (u, v) = (hi, lo)
      · have hp : ordered u v = (lo, hi) := by
          rcases hhead with h | h <;> cases h
          · simp [ordered, hlt]
          · simp [ordered, Nat.not_lt.mpr (Nat.le_of_lt hlt)]
        rw [hp]
        simp [harc]
      · have ht : (lo, hi) ∈ t ∨ (hi, lo) ∈ t := by
          rcases h with h | h <;> rcases List.mem_cons.mp h with h | h
          · exact absurd (Or.inl h.symm) hhead
          · exact Or.inl h
          · exact absurd (Or.inr h.symm) hhead
          · exact Or.inr h
        split
        · exact List.mem_cons_of_mem _ (ih ht)
        · exact ih ht

theorem frontOf_arc (T : Topo) (l : List (Nat × Nat)) : ∀ p ∈ frontOf T l, p.2 ∈ T.up p.1 :=
  fun p hp => resolve_arc T l p (mem_dedupL.mp hp)

/-- **End to end** (`recustomize_seeded`, agreement form): the form that holds for the real code over `f64`,
whose change test `!=` is not sound for equality (R-F1). From the `changed` pairs, the result agrees with `relax`
for the new seeds: weights under `R` (for `f64`, `==` by P-f64), middles exactly. -/
theorem recustomize_seeded_agrees {o : Ops W} {R : W → W → Prop} (hA : Agree o R) {T : Topo}
    (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK) {S S' : Seeds W} {M0 F : Metric W}
    (hM0 : relax o T (seedM S) = some M0) (hF : relax o T (seedM S') = some F)
    (changed : List (Nat × Nat))
    (hchanged : ∀ lo hi, hi ∈ T.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ changed ∨ (hi, lo) ∈ changed)
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount T < fuel) :
    ∀ lo hi, hi ∈ T.up lo →
      ER R ((run o T S' pick fuel (init M0 (frontOf T changed))).M lo hi) (F lo hi) :=
  scoped_after_seed_change hV hC hD hA (relax_isFull o hV S hM0) (relax_isFull o hV S' hF)
    (frontOf_arc T changed) (dedupL_nodup _)
    (fun lo hi harc hne => mem_dedupL.mpr (mem_resolve hV harc changed (hchanged lo hi harc hne)))
    hpick hfuel

/-- **End to end** (`recustomize_seeded`, exact form). Needs a change test sound for equality: true for exact
arithmetic and for a `to_bits()` test, false for the code's `f64 !=` (R-F1). -/
theorem recustomize_seeded_exact {o : Ops W} (hbeq : ∀ a b, o.beq a b = true → a = b) {T : Topo}
    (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK) {S S' : Seeds W} {M0 F : Metric W}
    (hM0 : relax o T (seedM S) = some M0) (hF : relax o T (seedM S') = some F)
    (changed : List (Nat × Nat))
    (hchanged : ∀ lo hi, hi ∈ T.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ changed ∨ (hi, lo) ∈ changed)
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount T < fuel) :
    ∀ lo hi, hi ∈ T.up lo → (run o T S' pick fuel (init M0 (frontOf T changed))).M lo hi = F lo hi :=
  scoped_after_seed_change_exact hbeq hV hC hD hM0 hF (frontOf_arc T changed) (dedupL_nodup _)
    (fun lo hi harc hne => mem_dedupL.mpr (mem_resolve hV harc changed (hchanged lo hi harc hne)))
    hpick hfuel

/-- The same for exact arithmetic, with no premise on the weights left. -/
theorem recustomize_seeded_exact_NInf {T : Topo} (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK)
    {S S' : Seeds NInf} {M0 F : Metric NInf}
    (hM0 : relax NInf.ops T (seedM S) = some M0) (hF : relax NInf.ops T (seedM S') = some F)
    (changed : List (Nat × Nat))
    (hchanged : ∀ lo hi, hi ∈ T.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ changed ∨ (hi, lo) ∈ changed)
    {fuel : Nat} (hfuel : arcCount T < fuel) :
    ∀ lo hi, hi ∈ T.up lo →
      (run NInf.ops T S' pickMin fuel (init M0 (frontOf T changed))).M lo hi = F lo hi :=
  recustomize_seeded_exact (fun a b h => by simpa [NInf.ops] using h) hV hC hD hM0 hF changed hchanged
    pickMin_spec hfuel

end CchRecompute
