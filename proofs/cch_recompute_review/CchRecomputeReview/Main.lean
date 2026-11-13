import CchRecomputeReview.Relax
import CchRecomputeReview.Step

/-!
# Theorems 2 and 3 (Lemma A), in agreement form and exact form

* `scoped_after_seed_change` (**Theorem 2**): from a full customization for seeds `S`, with a front that holds
  every arc whose seed changed, the pop loop for `S′` ends at the full customization for `S′`.
* `lemmaA` (**Theorem 3**, 03 §6b Lemma A): from a full customization of `H`, grown to `H′ ⊇ H` with the new arcs
  preset to `+∞` and in the front together with the changed seeds, it ends at the full customization of `H′`.
* `*_exact`: with a sound change test (`beq a b = true → a = b`), equality: weights and middles.
* `*_bitwise`: with the change test made bitwise (`decide (a = b)`, i.e. `to_bits() ==` in Rust), equality for
  **any** `add` and `lt`, so IEEE rounding cannot break it.
-/

namespace CchRecompute

variable {W : Type}

theorem recomp_seed_ext (o : Ops W) (T : Topo) {S S' : Seeds W} (M : Metric W) {lo hi : Nat}
    (h : S' lo hi = S lo hi) : recomp o T S' M lo hi = recomp o T S M lo hi := by
  simp only [recomp, seedE, h]

/-- **Theorem 2** (agreement form). -/
theorem scoped_after_seed_change {o : Ops W} {T : Topo} (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK)
    {R : W → W → Prop} (hA : Agree o R) {S S' : Seeds W} {M0 F : Metric W}
    (hM0 : IsFull o T S M0) (hF : IsFull o T S' F)
    {fr : List (Nat × Nat)} (hfr_arc : ∀ p ∈ fr, p.2 ∈ T.up p.1) (hfr_nodup : fr.Nodup)
    (hfr : ∀ lo hi, hi ∈ T.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ fr)
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount T < fuel) :
    ∀ lo hi, hi ∈ T.up lo → ER R ((run o T S' pick fuel (init M0 fr)).M lo hi) (F lo hi) := by
  refine (scoped_agrees hV hC hD hA hF hfr_arc hfr_nodup ?_ hpick hfuel).2
  intro a b hab hn
  have hs : S' a b = S a b := Classical.byContradiction fun h => hn (hfr a b hab h)
  rw [recomp_seed_ext o T M0 hs, ← hM0 a b hab]
  exact ER.refl hA _

/-- **Theorem 2** (exact form), against `relax` itself: the scoped pass gives, arc for arc, what `relax` gives
for the new seeds, weights and middles. -/
theorem scoped_after_seed_change_exact {o : Ops W} (hbeq : ∀ a b, o.beq a b = true → a = b) {T : Topo}
    (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK) {S S' : Seeds W} {M0 F : Metric W}
    (hM0 : relax o T (seedM S) = some M0) (hF : relax o T (seedM S') = some F)
    {fr : List (Nat × Nat)} (hfr_arc : ∀ p ∈ fr, p.2 ∈ T.up p.1) (hfr_nodup : fr.Nodup)
    (hfr : ∀ lo hi, hi ∈ T.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ fr)
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount T < fuel) :
    ∀ lo hi, hi ∈ T.up lo → (run o T S' pick fuel (init M0 fr)).M lo hi = F lo hi := fun lo hi h =>
  ER_eq.mp (scoped_after_seed_change hV hC hD (agree_eq o hbeq) (relax_isFull o hV S hM0)
    (relax_isFull o hV S' hF) hfr_arc hfr_nodup hfr hpick hfuel lo hi h)

/-- The change test made bitwise: `best.to_bits() != old.to_bits()`. -/
def bitwise [DecidableEq W] (o : Ops W) : Ops W := { o with beq := fun a b => decide (a = b) }

theorem relaxZ_ops {o o' : Ops W} (ha : o.add = o'.add) (hl : o.lt = o'.lt) (T : Topo) (x y : Nat)
    (a b : W) : ∀ (Z : List Nat) (c : Nat) (M : Metric W), relaxZ o T x y a b c Z M = relaxZ o' T x y a b c Z M
  | [], _, _ => rfl
  | z :: zs, c, M => by
    simp only [relaxZ, ha, hl]
    split
    · rfl
    · exact relaxZ_ops ha hl T x y a b zs _ _

theorem relaxY_ops {o o' : Ops W} (ha : o.add = o'.add) (hl : o.lt = o'.lt) (T : Topo) (x : Nat) :
    ∀ (L : List Nat) (M : Metric W), relaxY o T x L M = relaxY o' T x L M
  | [], _ => rfl
  | y :: rest, M => by
    simp only [relaxY, relaxZ_ops ha hl]
    split
    · rfl
    · exact relaxY_ops ha hl T x rest _

theorem relaxX_ops {o o' : Ops W} (ha : o.add = o'.add) (hl : o.lt = o'.lt) (T : Topo) :
    ∀ (L : List Nat) (M : Metric W), relaxX o T L M = relaxX o' T L M
  | [], _ => rfl
  | x :: xs, M => by
    simp only [relaxX, relaxY_ops ha hl]
    split
    · rfl
    · exact relaxX_ops ha hl T xs _

/-- `relax` never reads `beq`. -/
theorem relax_bitwise [DecidableEq W] (o : Ops W) (T : Topo) (M : Metric W) :
    relax (bitwise o) T M = relax o T M :=
  relaxX_ops (o := bitwise o) (o' := o) rfl rfl T _ M

/-- **Theorem 2, bitwise.** With a bitwise change test the scoped pass is bit-identical to `relax`, for any
`add` and `lt` whatever: no IEEE law is used. -/
theorem scoped_after_seed_change_bitwise [DecidableEq W] (o : Ops W) {T : Topo}
    (hV : T.Valid) (hC : T.Chordal) (hD : T.DownOK) {S S' : Seeds W} {M0 F : Metric W}
    (hM0 : relax o T (seedM S) = some M0) (hF : relax o T (seedM S') = some F)
    {fr : List (Nat × Nat)} (hfr_arc : ∀ p ∈ fr, p.2 ∈ T.up p.1) (hfr_nodup : fr.Nodup)
    (hfr : ∀ lo hi, hi ∈ T.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ fr)
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount T < fuel) :
    ∀ lo hi, hi ∈ T.up lo → (run (bitwise o) T S' pick fuel (init M0 fr)).M lo hi = F lo hi :=
  scoped_after_seed_change_exact (fun a b h => by simpa [bitwise] using h) hV hC hD
    ((relax_bitwise o T _).trans hM0) ((relax_bitwise o T _).trans hF) hfr_arc hfr_nodup hfr hpick hfuel

/-! ## Lemma A -/

/-- An apex reached through a `+∞` arc never wins. -/
theorem upd_top_noop {o : Ops W} (hTop : TopLaws o) {M : Metric W} {x y z : Nat}
    (h : ((M x y).up = o.top ∧ (M x y).dn = o.top) ∨ ((M x z).up = o.top ∧ (M x z).dn = o.top))
    (e : Entry W) : upd o M x y z e = e := by
  have hcu : o.add (M x y).dn (M x z).up = o.top := by
    rcases h with ⟨_, h⟩ | ⟨h, _⟩
    · rw [h, hTop.top_add]
    · rw [h, hTop.add_top]
  have hcd : o.add (M x z).dn (M x y).up = o.top := by
    rcases h with ⟨h, _⟩ | ⟨_, h⟩
    · rw [h, hTop.add_top]
    · rw [h, hTop.top_add]
  cases e
  simp only [upd, hcu, hcd, hTop.top_not_lt, Bool.false_eq_true, ↓reduceIte]

theorem foldl_filter_noop (f : Entry W → Nat → Entry W) (p : Nat → Bool) :
    ∀ (l : List Nat) (e : Entry W), (∀ x ∈ l, p x = false → ∀ a, f a x = a) →
      l.foldl f e = (l.filter p).foldl f e
  | [], _, _ => rfl
  | x :: t, e, h => by
    have ih := foldl_filter_noop f p t
    cases hp : p x
    · rw [List.filter_cons_of_neg (by simp [hp]), List.foldl_cons, h x List.mem_cons_self hp e]
      exact ih e (fun y hy => h y (List.mem_cons_of_mem _ hy))
    · rw [List.filter_cons_of_pos hp, List.foldl_cons, List.foldl_cons]
      exact ih _ (fun y hy => h y (List.mem_cons_of_mem _ hy))

/-- 03 §6b Lemma A, first bullet ("a new arc has no old apex: the apex would have forced the arc already, by
I5"): true when `H` is chordal. Not needed by `lemmaA`, whose front holds every new arc anyway. -/
theorem new_arc_no_old_apex {H : Topo} (hCH : H.Chordal) {lo hi x : Nat} (hnew : hi ∉ H.up lo)
    (hx1 : lo ∈ H.up x) (hx2 : hi ∈ H.up x) (hlt : lo < hi) : False :=
  hnew (hCH x lo hi hx1 hx2 hlt)

/-- **Theorem 3, Lemma A** (agreement form). `H ⊆ H′`; the start metric holds the full customization of `H`
on `H`'s arcs and `+∞` both ways on the new arcs `N = H′ ∖ H` (their middles are free); the front holds `N` and
every old arc whose seed changed. Then the pop loop on `H′` ends at the full customization of `H′`. Only `H′`
needs P-rows, P-chordal and P-transpose; `H` enters only through its full customization `FH`. -/
theorem lemmaA {o : Ops W} {H H' : Topo} (hsub : ∀ x y, y ∈ H.up x → y ∈ H'.up x)
    (hV' : H'.Valid) (hC' : H'.Chordal) (hD' : H'.DownOK) {R : W → W → Prop} (hA : Agree o R)
    (hTop : TopLaws o) {S S' : Seeds W} {FH M0 F : Metric W}
    (hFH : IsFull o H S FH) (hF : IsFull o H' S' F)
    (hold : ∀ lo hi, hi ∈ H.up lo → M0 lo hi = FH lo hi)
    (hnew : ∀ lo hi, hi ∈ H'.up lo → hi ∉ H.up lo → (M0 lo hi).up = o.top ∧ (M0 lo hi).dn = o.top)
    {fr : List (Nat × Nat)} (hfr_arc : ∀ p ∈ fr, p.2 ∈ H'.up p.1) (hfr_nodup : fr.Nodup)
    (hfrN : ∀ lo hi, hi ∈ H'.up lo → hi ∉ H.up lo → (lo, hi) ∈ fr)
    (hfrS : ∀ lo hi, hi ∈ H.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ fr)
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount H' < fuel) :
    ∀ lo hi, hi ∈ H'.up lo → ER R ((run o H' S' pick fuel (init M0 fr)).M lo hi) (F lo hi) := by
  refine (scoped_agrees hV' hC' hD' hA hF hfr_arc hfr_nodup ?_ hpick hfuel).2
  intro a b hab' hn
  have hab : b ∈ H.up a := Classical.byContradiction fun h => hn (hfrN a b hab' h)
  have hs : S' a b = S a b := Classical.byContradiction fun h => hn (hfrS a b hab h)
  -- the old arc's value is what it reads in H …
  have h1 : M0 a b = recomp o H S M0 a b := by
    rw [hold a b hab, hFH a b hab]
    apply recomp_ext
    intro x hx
    obtain ⟨_, hxa, hxb⟩ := mem_commonDown.mp hx
    exact ⟨(hold x a hxa).symm, (hold x b hxb).symm⟩
  -- … and the apexes H′ adds go through a new, +∞ arc, so they never win
  have h2 : recomp o H' S' M0 a b = recomp o H S M0 a b := by
    rw [recomp_seed_ext o H' M0 hs]
    unfold recomp
    rw [foldl_filter_noop _ (H.apex a b) _ _ ?_]
    · congr 1
      unfold Topo.commonDown
      rw [List.filter_filter]
      apply List.filter_congr
      intro x _
      by_cases h1 : a ∈ H.up x <;> by_cases h2 : b ∈ H.up x <;>
        simp [Topo.apex, h1, h2, hsub x a, hsub x b]
    · intro x hx hp e
      obtain ⟨_, hxa, hxb⟩ := mem_commonDown.mp hx
      apply upd_top_noop hTop
      by_cases h1 : a ∈ H.up x
      · have h2 : b ∉ H.up x := fun h2 => by simp [Topo.apex, h1, h2] at hp
        exact Or.inr (hnew x b hxb h2)
      · exact Or.inl (hnew x a hxa h1)
  rw [h2, ← h1]
  exact ER.refl hA _

/-- **Theorem 3, Lemma A** (exact form), against `relax` on `H′`. -/
theorem lemmaA_exact {o : Ops W} (hbeq : ∀ a b, o.beq a b = true → a = b) {H H' : Topo}
    (hsub : ∀ x y, y ∈ H.up x → y ∈ H'.up x) (hVH : H.Valid)
    (hV' : H'.Valid) (hC' : H'.Chordal) (hD' : H'.DownOK) (hTop : TopLaws o) {S S' : Seeds W}
    {FH M0 F : Metric W} (hFH : relax o H (seedM S) = some FH) (hF : relax o H' (seedM S') = some F)
    (hold : ∀ lo hi, hi ∈ H.up lo → M0 lo hi = FH lo hi)
    (hnew : ∀ lo hi, hi ∈ H'.up lo → hi ∉ H.up lo → (M0 lo hi).up = o.top ∧ (M0 lo hi).dn = o.top)
    {fr : List (Nat × Nat)} (hfr_arc : ∀ p ∈ fr, p.2 ∈ H'.up p.1) (hfr_nodup : fr.Nodup)
    (hfrN : ∀ lo hi, hi ∈ H'.up lo → hi ∉ H.up lo → (lo, hi) ∈ fr)
    (hfrS : ∀ lo hi, hi ∈ H.up lo → S' lo hi ≠ S lo hi → (lo, hi) ∈ fr)
    {pick : List (Nat × Nat) → Option (Nat × Nat)} (hpick : PickSpec pick) {fuel : Nat}
    (hfuel : arcCount H' < fuel) :
    ∀ lo hi, hi ∈ H'.up lo → (run o H' S' pick fuel (init M0 fr)).M lo hi = F lo hi := fun lo hi h =>
  ER_eq.mp (lemmaA hsub hV' hC' hD' (agree_eq o hbeq) hTop (relax_isFull o hVH S hFH)
    (relax_isFull o hV' S' hF) hold hnew hfr_arc hfr_nodup hfrN hfrS hpick hfuel lo hi h)

end CchRecompute
