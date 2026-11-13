import CchSupergraphReview.Topo
import CchSupergraphReview.LogLink
set_option linter.unusedVariables false

/-!
# Counterexamples: which premises are needed

Each is a concrete state, checked by `decide`, where one premise of a theorem in this project fails and the
theorem's conclusion fails with it. None is a bug in reachable code; the header of `CchSupergraphReview.lean`
says why for each.

* `asymmetric_misses_arc` (`order.rs:195-212` @main, Lean-only, not run live): `Topology::from_order` checks only
  `is_well_formed`, so on a well-formed skeleton that is not mirrored, `chordal_supergraph` returns no arc for an
  edge listed only in its higher end's row. `Topology::build`, its only non-test caller, never passes one:
  `nd::nested_dissection_order` refuses it first (`nd.rs:54-58`; `build_spec`).
* `masked_needs_promotion_completeness` (Masked.lean, design-level): the masked read means "has a live edge" only
  under the tensor's promotion completeness.
* `late_slab_breaks` (LogLink.lean, design-level): with `Records`, `EncSym`, `hcov` and `lo ≤ rv u` all holding, a
  slab read after `V_sg` makes `LogComplete` fail, so `logComplete_of_writes` needs `rv u ≤ V_sg`.
-/
namespace CchSG

/-- A well-formed skeleton on two nodes whose only entry, `0`, is in row `1` and not mirrored in row `0`. -/
def asymSk : NI.Cch.Csr := ⟨2, [0, 0, 1], [0]⟩

/-- **`from_order` needs a mirrored skeleton** (Lean-only; `from_order` is reached on such a skeleton only from
    tests). The skeleton is well formed, the identity order is a permutation, yet `chordal_supergraph` returns no
    arc, while the elimination game on the undirected graph `{0, 1}` has the arc `0 → 1`. -/
theorem asymmetric_misses_arc :
    asymSk.isWellFormed = true ∧ invert [0, 1] 2 = .ok [0, 1] ∧
    (chordalSupergraph asymSk [0, 1] [0, 1]).2 = [] ∧ gameRow 2 (rankAdj asymSk [0, 1] [0, 1]) 0 = [1] :=
  ⟨by decide, rfl, by decide, by decide⟩

end CchSG
