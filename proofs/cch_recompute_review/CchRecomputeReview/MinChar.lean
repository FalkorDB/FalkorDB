import CchRecomputeReview.Main
import CchRecomputeReview.Weights

/-!
# What the fold computes, for a strict total order

`IsFull` states 03 §6b's arc value as the fold the code runs. For a strict total order `<` this module proves
the closed form 03 §6b gives:

* `isFull_up_min`: an arc's upward weight is `min(seed, min over apexes x of dn(x,lo) + up(x,hi))`: it is the
  seed or some candidate, and nothing is strictly below it. `isFull_dn_min` is the mirror.
* `isFull_up_mid`: its middle is `none` iff no candidate beats the seed; it is `some x` iff `x` is the **first
  apex, in increasing `x`, whose candidate attains that minimum**, the minimum being strictly below the seed. (The
  literal "first `x` to win a strict `<`" is R-W1.)
* `deleted_node_arcs_top` (03 §6 "Node delete": "every arc at its rank resolves to `+∞` by induction on rank"):
  if every arc at rank `v` is seeded `+∞` both ways, every one of them is `(+∞, +∞, none, none)`.

`NInf.order` discharges the order laws for exact arithmetic. For `f64`, `<` is a strict total order up to
`==` (`±0.0`); the closed form then holds up to `==`, which is not proved here.
-/

namespace CchRecompute

variable {W : Type}

/-- A strict total order on `W`, as `Ops.lt`. -/
structure StrictTotal (o : Ops W) : Prop where
  irrefl : ∀ a, o.lt a a = false
  trans : ∀ a b c, o.lt a b = true → o.lt b c = true → o.lt a c = true
  total : ∀ a b, a ≠ b → o.lt a b = true ∨ o.lt b a = true

theorem StrictTotal.asymm {o : Ops W} (h : StrictTotal o) {a b : W} (hab : o.lt a b = true) :
    o.lt b a = false := by
  cases hba : o.lt b a
  · rfl
  · have := h.trans _ _ _ hab hba; rw [h.irrefl] at this; exact absurd this (by simp)

/-- One direction of the fold: keep the running value, replace it on a strict `<`, and record `x`. -/
def stepVM (o : Ops W) (c : Nat → W) (acc : W × Option Nat) (x : Nat) : W × Option Nat :=
  if o.lt (c x) acc.1 then (c x, some x) else acc

/-- The fold of `stepVM` from `(b0, m0)`: either nothing beats `b0`, or a unique last winner `x` that is the
first element attaining the minimum. -/
theorem foldl_stepVM {o : Ops W} (hO : StrictTotal o) (c : Nat → W) :
    ∀ (l : List Nat) (b0 : W) (m0 : Option Nat),
      ((l.foldl (stepVM o c) (b0, m0)) = (b0, m0) ∧ ∀ x ∈ l, o.lt (c x) b0 = false) ∨
      (∃ pre x post, l = pre ++ x :: post ∧ l.foldl (stepVM o c) (b0, m0) = (c x, some x) ∧
        o.lt (c x) b0 = true ∧ (∀ y ∈ pre, o.lt (c x) (c y) = true) ∧
        (∀ y ∈ post, o.lt (c y) (c x) = false))
  | [], b0, m0 => Or.inl ⟨rfl, by simp⟩
  | y :: t, b0, m0 => by
    simp only [List.foldl_cons]
    by_cases hy : o.lt (c y) b0 = true
    · have hs : stepVM o c (b0, m0) y = (c y, some y) := by simp [stepVM, hy]
      rw [hs]
      rcases foldl_stepVM hO c t (c y) (some y) with ⟨h1, h2⟩ | ⟨pre, x, post, e, h1, h2, h3, h4⟩
      · exact Or.inr ⟨[], y, t, rfl, h1, hy, by simp, h2⟩
      · refine Or.inr ⟨y :: pre, x, post, by rw [e]; rfl, h1, hO.trans _ _ _ h2 hy, ?_, h4⟩
        intro z hz
        rcases List.mem_cons.mp hz with rfl | hz
        · exact h2
        · exact h3 z hz
    · have hy' : o.lt (c y) b0 = false := by simpa using hy
      have hs : stepVM o c (b0, m0) y = (b0, m0) := by simp [stepVM, hy']
      rw [hs]
      rcases foldl_stepVM hO c t b0 m0 with ⟨h1, h2⟩ | ⟨pre, x, post, e, h1, h2, h3, h4⟩
      · refine Or.inl ⟨h1, ?_⟩
        intro z hz
        rcases List.mem_cons.mp hz with rfl | hz
        · exact hy'
        · exact h2 z hz
      · refine Or.inr ⟨y :: pre, x, post, by rw [e]; rfl, h1, h2, ?_, h4⟩
        intro z hz
        rcases List.mem_cons.mp hz with rfl | hz
        · -- `c z` did not beat `b0`, `c x` did: so `c x < c z`
          by_cases heq : c x = c z
          · rw [heq] at h2; rw [h2] at hy'; exact absurd hy' (by simp)
          · rcases hO.total _ _ heq with h | h
            · exact h
            · have := hO.trans _ _ _ h h2; rw [this] at hy'; exact absurd hy' (by simp)
        · exact h3 z hz

/-- The upward half of the entry fold is the `stepVM` fold of the upward candidates. -/
theorem foldl_upd_up (o : Ops W) (M : Metric W) (lo hi : Nat) :
    ∀ (l : List Nat) (e : Entry W),
      let r := l.foldl (fun e x => upd o M x lo hi e) e
      (r.up, r.mup) = l.foldl (stepVM o (fun x => o.add (M x lo).dn (M x hi).up)) (e.up, e.mup)
  | [], _ => rfl
  | x :: t, e => by
    simp only [List.foldl_cons]
    have := foldl_upd_up o M lo hi t (upd o M x lo hi e)
    simp only at this ⊢
    rw [this]
    congr 1
    simp only [stepVM, upd]
    split <;> rfl

/-- The downward half. -/
theorem foldl_upd_dn (o : Ops W) (M : Metric W) (lo hi : Nat) :
    ∀ (l : List Nat) (e : Entry W),
      let r := l.foldl (fun e x => upd o M x lo hi e) e
      (r.dn, r.mdn) = l.foldl (stepVM o (fun x => o.add (M x hi).dn (M x lo).up)) (e.dn, e.mdn)
  | [], _ => rfl
  | x :: t, e => by
    simp only [List.foldl_cons]
    have := foldl_upd_dn o M lo hi t (upd o M x lo hi e)
    simp only at this ⊢
    rw [this]
    congr 1
    simp only [stepVM, upd]
    split <;> rfl

/-- **Theorem 1, closed form (weights).** In a full customization an arc's upward weight is the seed or a
candidate `dn(x,lo) + up(x,hi)` of one of its apexes, and no candidate and not the seed is strictly below it:
`min(seed, min over x ∈ down(lo) ∩ down(hi) of dn(x,lo) + up(x,hi))`. -/
theorem isFull_up_min {o : Ops W} (hO : StrictTotal o) {T : Topo} {S : Seeds W} {F : Metric W}
    (hF : IsFull o T S F) {lo hi : Nat} (harc : hi ∈ T.up lo) :
    ((F lo hi).up = (S lo hi).1 ∨ ∃ x ∈ T.commonDown lo hi, (F lo hi).up = o.add (F x lo).dn (F x hi).up) ∧
    o.lt (S lo hi).1 (F lo hi).up = false ∧
    ∀ x ∈ T.commonDown lo hi, o.lt (o.add (F x lo).dn (F x hi).up) (F lo hi).up = false := by
  have h := foldl_upd_up o F lo hi (T.commonDown lo hi) (seedE S lo hi)
  simp only at h
  have hFe : (F lo hi).up = ((T.commonDown lo hi).foldl (fun e x => upd o F x lo hi e) (seedE S lo hi)).up := by
    rw [hF lo hi harc]; rfl
  rcases foldl_stepVM hO (fun x => o.add (F x lo).dn (F x hi).up) (T.commonDown lo hi) (S lo hi).1 none with
    ⟨h1, h2⟩ | ⟨pre, x, post, e, h1, h2, h3, h4⟩
  · have hv : (F lo hi).up = (S lo hi).1 := by
      rw [hFe]; have := congrArg Prod.fst (h.trans h1); simpa [seedE] using this
    refine ⟨Or.inl hv, by rw [hv]; exact hO.irrefl _, fun x hx => by rw [hv]; exact h2 x hx⟩
  · have hv : (F lo hi).up = o.add (F x lo).dn (F x hi).up := by
      rw [hFe]; have := congrArg Prod.fst (h.trans h1); simpa [seedE] using this
    have hx : x ∈ T.commonDown lo hi := by rw [e]; simp
    refine ⟨Or.inr ⟨x, hx, hv⟩, by rw [hv]; exact hO.asymm h2, ?_⟩
    intro y hy
    rw [hv]
    rw [e] at hy
    rcases List.mem_append.mp hy with hy | hy
    · exact hO.asymm (h3 y hy)
    · rcases List.mem_cons.mp hy with rfl | hy
      · exact hO.irrefl _
      · exact h4 y hy

/-- **Theorem 1, closed form (middle).** The middle is `none` iff no candidate beats the seed, and `some x`
iff `x` is an apex whose candidate is the arc's weight, strictly below the seed, and strictly below the candidate
of every lower apex: the first apex, in increasing `x`, attaining the minimum. -/
theorem isFull_up_mid {o : Ops W} (hO : StrictTotal o) {T : Topo} {S : Seeds W} {F : Metric W}
    (hF : IsFull o T S F) {lo hi : Nat} (harc : hi ∈ T.up lo) :
    ((F lo hi).mup = none ↔ ∀ x ∈ T.commonDown lo hi, o.lt (o.add (F x lo).dn (F x hi).up) (S lo hi).1 = false) ∧
    ∀ x, (F lo hi).mup = some x →
      x ∈ T.commonDown lo hi ∧ (F lo hi).up = o.add (F x lo).dn (F x hi).up ∧
      o.lt (F lo hi).up (S lo hi).1 = true ∧
      ∀ y ∈ T.commonDown lo hi, y < x → o.lt (F lo hi).up (o.add (F y lo).dn (F y hi).up) = true := by
  have h := foldl_upd_up o F lo hi (T.commonDown lo hi) (seedE S lo hi)
  simp only at h
  have hFe : F lo hi = (T.commonDown lo hi).foldl (fun e x => upd o F x lo hi e) (seedE S lo hi) :=
    hF lo hi harc
  have hsorted := commonDown_sorted T lo hi
  rcases foldl_stepVM hO (fun x => o.add (F x lo).dn (F x hi).up) (T.commonDown lo hi) (S lo hi).1 none with
    ⟨h1, h2⟩ | ⟨pre, x, post, e, h1, h2, h3, h4⟩
  · have hm : (F lo hi).mup = none := by
      rw [hFe]; have := congrArg Prod.snd (h.trans h1); simpa [seedE] using this
    exact ⟨⟨fun _ => h2, fun _ => hm⟩, fun x hx => by rw [hm] at hx; exact absurd hx (by simp)⟩
  · have hm : (F lo hi).mup = some x := by
      rw [hFe]; have := congrArg Prod.snd (h.trans h1); simpa [seedE] using this
    have hv : (F lo hi).up = o.add (F x lo).dn (F x hi).up := by
      rw [hFe]; have := congrArg Prod.fst (h.trans h1); simpa [seedE] using this
    have hx : x ∈ T.commonDown lo hi := by rw [e]; simp
    refine ⟨⟨fun hn => by rw [hm] at hn; exact absurd hn (by simp), fun hall => ?_⟩, ?_⟩
    · have := hall x hx; rw [h2] at this; exact absurd this (by simp)
    · intro x' hx'
      rw [hm] at hx'; cases hx'
      refine ⟨hx, hv, by rw [hv]; exact h2, ?_⟩
      intro y hy hyx
      rw [hv]
      rw [e] at hy hsorted
      rcases List.mem_append.mp hy with hy | hy
      · exact h3 y hy
      · -- `y` after `x` in an ascending list is above it
        exfalso
        have hs := List.pairwise_append.mp hsorted
        rcases List.mem_cons.mp hy with rfl | hy
        · exact Nat.lt_irrefl _ hyx
        · have := (List.pairwise_cons.mp hs.2.1).1 y hy; omega

/-- **Theorem 1, closed form (mirror).** The downward weight is the minimum of the mirror candidates. -/
theorem isFull_dn_min {o : Ops W} (hO : StrictTotal o) {T : Topo} {S : Seeds W} {F : Metric W}
    (hF : IsFull o T S F) {lo hi : Nat} (harc : hi ∈ T.up lo) :
    ((F lo hi).dn = (S lo hi).2 ∨ ∃ x ∈ T.commonDown lo hi, (F lo hi).dn = o.add (F x hi).dn (F x lo).up) ∧
    o.lt (S lo hi).2 (F lo hi).dn = false ∧
    ∀ x ∈ T.commonDown lo hi, o.lt (o.add (F x hi).dn (F x lo).up) (F lo hi).dn = false := by
  have h := foldl_upd_dn o F lo hi (T.commonDown lo hi) (seedE S lo hi)
  simp only at h
  have hFe : (F lo hi).dn = ((T.commonDown lo hi).foldl (fun e x => upd o F x lo hi e) (seedE S lo hi)).dn := by
    rw [hF lo hi harc]; rfl
  rcases foldl_stepVM hO (fun x => o.add (F x hi).dn (F x lo).up) (T.commonDown lo hi) (S lo hi).2 none with
    ⟨h1, h2⟩ | ⟨pre, x, post, e, h1, h2, h3, h4⟩
  · have hv : (F lo hi).dn = (S lo hi).2 := by
      rw [hFe]; have := congrArg Prod.fst (h.trans h1); simpa [seedE] using this
    refine ⟨Or.inl hv, by rw [hv]; exact hO.irrefl _, fun x hx => by rw [hv]; exact h2 x hx⟩
  · have hv : (F lo hi).dn = o.add (F x hi).dn (F x lo).up := by
      rw [hFe]; have := congrArg Prod.fst (h.trans h1); simpa [seedE] using this
    have hx : x ∈ T.commonDown lo hi := by rw [e]; simp
    refine ⟨Or.inr ⟨x, hx, hv⟩, by rw [hv]; exact hO.asymm h2, ?_⟩
    intro y hy
    rw [hv]
    rw [e] at hy
    rcases List.mem_append.mp hy with hy | hy
    · exact hO.asymm (h3 y hy)
    · rcases List.mem_cons.mp hy with rfl | hy
      · exact hO.irrefl _
      · exact h4 y hy

/-- `NInf`'s `<` is a strict total order. -/
theorem NInf.order : StrictTotal NInf.ops where
  irrefl a := by cases a <;> simp [NInf.ops, NInf.lt]
  trans a b c h1 h2 := by
    cases a <;> cases b <;> cases c <;> simp_all [NInf.ops, NInf.lt] <;> omega
  total a b h := by
    cases a <;> cases b <;> simp_all [NInf.ops, NInf.lt] <;> omega

/-! ## 03 §6 "Node delete" -/

theorem foldl_noop (f : Entry W → Nat → Entry W) :
    ∀ (l : List Nat) (e : Entry W), (∀ x ∈ l, ∀ a, f a x = a) → l.foldl f e = e
  | [], _, _ => rfl
  | x :: t, e, h => by
    simp only [List.foldl_cons, h x List.mem_cons_self e]
    exact foldl_noop f t e (fun y hy => h y (List.mem_cons_of_mem _ hy))

/-- **03 §6 "Node delete".** If every arc at rank `v` (as its lower or upper end) is seeded `+∞` both ways,
every arc at `v` is `(+∞, +∞, none, none)` in the full customization: each triangle through such an arc uses
another arc at `v` with a lower lower end. -/
theorem deleted_node_arcs_top {o : Ops W} (hTop : TopLaws o) {T : Topo} {S : Seeds W} {F : Metric W}
    (hF : IsFull o T S F) {v : Nat}
    (hS : ∀ lo hi, hi ∈ T.up lo → (lo = v ∨ hi = v) → S lo hi = (o.top, o.top)) :
    ∀ k lo hi, lo < k → hi ∈ T.up lo → (lo = v ∨ hi = v) → F lo hi = ⟨o.top, o.top, none, none⟩ := by
  intro k
  induction k with
  | zero => intro lo _ h; omega
  | succ k ih =>
    intro lo hi hk harc hv
    rw [hF lo hi harc, recomp, foldl_noop]
    · simp [seedE, hS lo hi harc hv]
    · intro x hx a
      obtain ⟨hxlo, h1, h2⟩ := mem_commonDown.mp hx
      apply upd_top_noop hTop
      rcases hv with rfl | rfl
      · have := ih x lo (by omega) h1 (Or.inr rfl); exact Or.inl ⟨by rw [this], by rw [this]⟩
      · have := ih x hi (by omega) h2 (Or.inr rfl); exact Or.inr ⟨by rw [this], by rw [this]⟩

end CchRecompute
