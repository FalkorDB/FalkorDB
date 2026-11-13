import CchRecomputeReview.Basic

/-!
# The apex merge of `recustomize_seeded` reads exactly what 03 §6b says

`partial.rs:188-219` @08c (the same at the stack tip `cch/10-rdb-v20`): a two-pointer merge of the ascending
rows `down(lo)` and `down(hi)`, relaxing at every common `x`. `mergeL`'s list arguments are `dlo[a..]` and
`dhi[b..]`; each round advances `a` or `b`, so `|dlo| + |dhi|` rounds of fuel finish it.

`merge_eq_recomp`: on a valid topology with **P-transpose**, the merge from the seed is `recomp`, the 03 §6b
fold over `down(lo) ∩ down(hi)` in increasing `x`.
-/

namespace CchRecompute

variable {W : Type}

/-- `partial.rs:189-219`. -/
def mergeL (o : Ops W) (M : Metric W) (lo hi : Nat) : Nat → List Nat → List Nat → Entry W → Entry W
  | 0, _, _, acc => acc
  | _ + 1, [], _, acc => acc
  | _ + 1, _ :: _, [], acc => acc
  | k + 1, xa :: ra, xb :: rb, acc =>
    if xa < xb then mergeL o M lo hi k ra (xb :: rb) acc
    else if xa > xb then mergeL o M lo hi k (xa :: ra) rb acc
    else mergeL o M lo hi k ra rb (upd o M xa lo hi acc)

theorem mergeL_eq (o : Ops W) (M : Metric W) (lo hi : Nat) :
    ∀ (k : Nat) (l1 l2 : List Nat) (acc : Entry W), l1.length + l2.length ≤ k →
      l1.Pairwise (· < ·) → l2.Pairwise (· < ·) →
      mergeL o M lo hi k l1 l2 acc =
        (l1.filter (fun x => decide (x ∈ l2))).foldl (fun e x => upd o M x lo hi e) acc
  | 0, l1, l2, acc, hk, _, _ => by
    have h1 : l1 = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst h1; simp [mergeL]
  | k + 1, [], l2, acc, _, _, _ => by simp [mergeL]
  | k + 1, xa :: ra, [], acc, _, _, _ => by
    have : ∀ l : List Nat, l.filter (fun _ => false) = [] := fun l => by
      induction l with
      | nil => rfl
      | cons a t ih => simp [ih]
    simp [mergeL, this]
  | k + 1, xa :: ra, xb :: rb, acc, hk, h1, h2 => by
    have h1' := List.pairwise_cons.mp h1
    have h2' := List.pairwise_cons.mp h2
    simp only [mergeL]
    by_cases hab : xa < xb
    · simp only [hab, ↓reduceIte]
      rw [mergeL_eq o M lo hi k ra (xb :: rb) acc (by simp at hk ⊢; omega) h1'.2 h2]
      have : ¬ (xa ∈ xb :: rb) := by
        intro hm
        rcases List.mem_cons.mp hm with h | h
        · omega
        · have := h2'.1 xa h; omega
      rw [List.filter_cons_of_neg (by simpa using this)]
    · simp only [hab, ↓reduceIte]
      by_cases hba : xa > xb
      · simp only [hba, ↓reduceIte]
        rw [mergeL_eq o M lo hi k (xa :: ra) rb acc (by simp at hk ⊢; omega) h1 h2'.2]
        congr 1
        apply List.filter_congr
        intro x hx
        have hx' : x ≠ xb := by
          rcases List.mem_cons.mp hx with h | h
          · omega
          · have := h1'.1 x h; omega
        simp [hx']
      · simp only [hba, ↓reduceIte]
        have heq : xa = xb := by omega
        subst heq
        rw [mergeL_eq o M lo hi k ra rb _ (by simp at hk ⊢; omega) h1'.2 h2'.2]
        rw [List.filter_cons_of_pos (by simp), List.foldl_cons]
        congr 1
        apply List.filter_congr
        intro x hx
        have hx' : x ≠ xa := by have := h1'.1 x hx; omega
        simp [hx']

/-- Shrinking the range of a filter whose predicate only holds below `m`. -/
theorem filter_range_shrink (p : Nat → Bool) {m : Nat} (hp : ∀ x, p x = true → x < m) :
    ∀ d, (List.range (m + d)).filter p = (List.range m).filter p
  | 0 => rfl
  | d + 1 => by
    rw [← Nat.add_assoc, List.range_succ, List.filter_append, filter_range_shrink p hp d]
    have : p (m + d) = false := by
      cases h : p (m + d)
      · rfl
      · have := hp _ h; omega
    simp [this]

theorem down_sorted {T : Topo} (hD : T.DownOK) (r : Nat) : (T.down r).Pairwise (· < ·) := by
  rw [hD r]
  apply List.Pairwise.filter
  rw [List.range_eq_range']
  exact List.pairwise_lt_range'

/-- `down(lo) ∩ down(hi)`, as the merge visits it, is `commonDown`. -/
theorem down_inter {T : Topo} (hV : T.Valid) (hD : T.DownOK) {lo hi : Nat} (harc : hi ∈ T.up lo) :
    (T.down lo).filter (fun x => decide (x ∈ T.down hi)) = T.commonDown lo hi := by
  have hlo : lo < T.n := Nat.lt_trans (hV.above lo hi harc) (hV.bound lo hi harc)
  rw [hD lo, List.filter_filter]
  have step1 : (List.range T.n).filter (fun x => decide (x ∈ T.down hi) && decide (lo ∈ T.up x)) =
      (List.range T.n).filter (T.apex lo hi) := by
    apply List.filter_congr
    intro x hx
    rw [hD hi]
    by_cases h1 : lo ∈ T.up x <;> by_cases h2 : hi ∈ T.up x <;> simp [Topo.apex, List.mem_filter, h1, h2, hx]
  rw [step1, Topo.commonDown]
  have hp : ∀ x, T.apex lo hi x = true → x < lo := fun x h => by
    simp only [Topo.apex, decide_eq_true_eq] at h
    exact hV.above x lo h.1
  have := filter_range_shrink (T.apex lo hi) hp (T.n - lo)
  rwa [Nat.add_sub_cancel' (Nat.le_of_lt hlo)] at this

/-- **The merge from the seed is `recomp`.** -/
theorem merge_eq_recomp (o : Ops W) {T : Topo} (hV : T.Valid) (hD : T.DownOK) (S : Seeds W) (M : Metric W)
    {lo hi : Nat} (harc : hi ∈ T.up lo) :
    mergeL o M lo hi ((T.down lo).length + (T.down hi).length) (T.down lo) (T.down hi) (seedE S lo hi) =
      recomp o T S M lo hi := by
  rw [mergeL_eq o M lo hi _ _ _ _ (Nat.le_refl _) (down_sorted hD lo) (down_sorted hD hi), down_inter hV hD harc]
  rfl

end CchRecompute
