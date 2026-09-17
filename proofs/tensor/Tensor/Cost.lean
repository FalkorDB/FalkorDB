/-
# Optimality of the fold point, and of iteration's output

Two results about *cost*, both of which need their scope stated before they are
stated, because "proved optimal" invites a reading neither supports.

Nothing here says how long anything takes. Lean proves what an operation
computes; running time is a property of machines, and every wall-clock and
instruction-count claim about this code rests on measurement, not on this file.

What can be proved is optimality *relative to an explicit model*:

* **`foldPoint_optimal`** — the square-root fold rule minimises the cost function
  the policy is derived from. (A minimiser, not proved unique: the AM--GM step
  would give uniqueness too, but the policy only needs that nothing beats it.) This is optimality of the
  *decision* given the model, and says nothing about whether the model describes
  the machine. The model's two constants are measured (`fold_cost_bench.rs`), and
  if those measurements are wrong the theorem is still true and the policy still
  wrong. What it does buy: the derivation in the paper is no longer a hand
  computation, and re-tuning the constants cannot invalidate the *form* of the
  rule.

* **`iterEdges_output_optimal`** — iteration emits every edge exactly once and
  nothing else. That is optimality in the output-sensitive sense: any correct
  enumerator must emit each edge at least once, so emitting each exactly once is
  the least possible output. It is a statement about the emitted multiset, not
  about the work done to produce it.
-/
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Tensor.Iter

namespace FalkorDB
namespace Tensor

/-! ## The fold point

The policy of `versioned_matrix.rs` folds a delta once it reaches
`sqrt(k · tx_added)`, where `k = 2F/w` is measured. Writing `D` for the fold
point, `t` for the per-transaction contribution, `w` for the per-entry cost a
transaction pays to touch a live delta and `F` for the fixed cost of one fold,
accumulating to `D` costs

    Σᵢ (i·t)·w + F  ≈  w·D²/(2t) + F,

so the amortised cost *per entry written* is `w·D/(2t) + F/D`: a rewrite tax
growing with `D` against a fold bill shrinking with it. -/

/-- Amortised cost per entry of running with fold point `D`. -/
noncomputable def foldCost (w F t D : ℝ) : ℝ := w * D / (2 * t) + F / D

/-- The balance point `D* = sqrt(2·(F/w)·t)`. -/
noncomputable def foldPoint (w F t : ℝ) : ℝ := Real.sqrt (2 * F * t / w)

/-- Both sides of the trade-off, in the form the proof uses: `c·D + F/D` with
`c = w/(2t)` the per-entry rewrite tax. -/
private theorem foldCost_eq (w F t D : ℝ) :
    foldCost w F t D = (w / (2 * t)) * D + F / D := by
  unfold foldCost
  ring

/-- **The trade-off is bounded below by `2√(cF)`.** This is AM--GM: the product
of the two terms is `cF`, independent of `D`, so their sum is least when they are
equal. -/
private theorem two_sqrt_le (hc : 0 < c) (hF : 0 < F) (hD : 0 < D) :
    2 * Real.sqrt (c * F) ≤ c * D + F / D := by
  have h1 : Real.sqrt (c * D) ^ 2 = c * D := Real.sq_sqrt (by positivity)
  have h2 : Real.sqrt (F / D) ^ 2 = F / D := Real.sq_sqrt (by positivity)
  have h3 : Real.sqrt (c * D) * Real.sqrt (F / D) = Real.sqrt (c * F) := by
    rw [← Real.sqrt_mul (by positivity)]
    congr 1
    field_simp
  calc 2 * Real.sqrt (c * F)
      = 2 * Real.sqrt (c * D) * Real.sqrt (F / D) := by rw [← h3]; ring
    _ ≤ Real.sqrt (c * D) ^ 2 + Real.sqrt (F / D) ^ 2 := two_mul_le_add_sq _ _
    _ = c * D + F / D := by rw [h1, h2]

/-- …and the bound is *attained* at `D = √(F/c)`, so it is the minimum rather
than merely a lower bound. -/
private theorem eq_at_point (hc : 0 < c) (hF : 0 < F) :
    c * Real.sqrt (F / c) + F / Real.sqrt (F / c) = 2 * Real.sqrt (c * F) := by
  have hFc : (0:ℝ) < F / c := by positivity
  have hs : 0 < Real.sqrt (F / c) := Real.sqrt_pos.mpr hFc
  have h1 : c * Real.sqrt (F / c) = Real.sqrt (c * F) := by
    have hcf : c * F = c ^ 2 * (F / c) := by field_simp
    rw [hcf, Real.sqrt_mul (by positivity), Real.sqrt_sq hc.le]
  have h2 : F / Real.sqrt (F / c) = Real.sqrt (c * F) := by
    rw [div_eq_iff hs.ne', ← Real.sqrt_mul (by positivity),
      show c * F * (F / c) = F ^ 2 by field_simp]
    exact (Real.sqrt_sq hF.le).symm
  rw [h1, h2]
  ring

/-- **The square-root rule is optimal for the cost model.** No fold point does
better than `D* = √(2Ft/w)`.

Read the quantifiers carefully: this is a statement about `foldCost`, which is a
*model* of the amortised per-entry cost, with `F` and `w` measured elsewhere. It
does not say the implementation is fast; it says that, granting the model, the
threshold the implementation uses is the one that minimises it. -/
theorem foldPoint_optimal {w F t : ℝ} (hw : 0 < w) (hF : 0 < F) (ht : 0 < t)
    {D : ℝ} (hD : 0 < D) :
    foldCost w F t (foldPoint w F t) ≤ foldCost w F t D := by
  have hc : 0 < w / (2 * t) := by positivity
  have hpt : foldPoint w F t = Real.sqrt (F / (w / (2 * t))) := by
    unfold foldPoint
    congr 1
    field_simp
  rw [foldCost_eq, foldCost_eq, hpt, eq_at_point hc hF]
  exact two_sqrt_le hc hF hD

/-- The minimum value itself, for reference: the best achievable amortised
per-entry cost is `2√(wF/(2t))`, which *falls* as the transaction size `t` grows
— the reason the rule depends on `tx_added` at all rather than being an absolute
delta cap. -/
theorem foldCost_foldPoint {w F t : ℝ} (hw : 0 < w) (hF : 0 < F) (ht : 0 < t) :
    foldCost w F t (foldPoint w F t) = 2 * Real.sqrt (w / (2 * t) * F) := by
  have hc : 0 < w / (2 * t) := by positivity
  have hpt : foldPoint w F t = Real.sqrt (F / (w / (2 * t))) := by
    unfold foldPoint
    congr 1
    field_simp
  rw [foldCost_eq, hpt, eq_at_point hc hF]

/-! ## The integer predicate implements the real rule

`fold_balance` tests `delta² ≥ k · tx_added` on `u64`, avoiding a square root in
the hot path. That is the same test as `delta ≥ √(k · tx_added)`, i.e. "have we
reached the balance point", which is what connects the theorem above to the code
that runs. -/

theorem sq_ge_iff_ge_sqrt (d k t : ℕ) :
    k * t ≤ d * d ↔ Real.sqrt ((k : ℝ) * t) ≤ (d : ℝ) := by
  rw [show ((d : ℝ)) = Real.sqrt ((d : ℝ) * d) by
    rw [Real.sqrt_mul_self (by positivity)]]
  rw [Real.sqrt_le_sqrt_iff (by positivity)]
  exact_mod_cast Iff.rfl

/-! ## Iteration is output-optimal -/

/-- **Iteration emits exactly the edges, each once.** Packaged from the two facts
that carry the content: membership is exactly "is an edge of this tensor", and
there are no duplicates.

This is optimality in the output-sensitive sense and nothing more. Any correct
enumerator must emit each edge at least once, so emitting each exactly once is
the least output any correct enumerator can produce. It says nothing about the
work done per emitted item — that the merge does no intermediate allocation is an
implementation property, argued in the paper's cost model rather than here. -/
theorem iterEdges_output_optimal (h : Inv t) :
    (∀ x : Nat × Nat × Nat, x ∈ iterEdges t ↔ x.2.2 ∈ t.edgesAt (x.1, x.2.1)) ∧
      (iterEdges t).Nodup :=
  ⟨fun _ => mem_iterEdges h, nodup_iterEdges h⟩

end Tensor
end FalkorDB
