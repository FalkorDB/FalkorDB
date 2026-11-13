import VersionedMatrix.TensorProofs
set_option linter.unusedVariables false

/-!
# The masked read is live (03 §3 "Input skeleton"; §5 "The full-seed pass")

One `(src, dst)` pair of one relationship type's tensor is versioned_matrix's `VMTensor.PairSt`: its `m`, `dp`
and `dm` entries, its `mt` bit and its `me` row. Its live value is `VMTensor.effV` (`Tensor::eff_get`,
`tensor.rs:444` @main: "`dp` wins, else `m` unless its `dm` bit is set", `tensor.rs:52-53`), and its edges are
`VMTensor.edges` (`Tensor::get`, `:465`). versioned_matrix proves every tensor operation keeps `VMTensor.Inv`
(`addBatch_correct`, `removeBatch_correct`, `removeFast_correct`, `foldP_correct`).

* `maskedRead`: 03 §3's `pattern(m)⟨¬struct(dm)⟩ ∪ pattern(dp)` at the pair, as GraphBLAS defines a
  complemented structural mask and an eWiseAdd pattern: present iff (`m` present and `dm` absent) or `dp` present.
* `seedVal`: 03 §5's `m⟨¬struct(dm) ∧ ¬struct(dp)⟩ ∪ dp`, the full-seed pass's forward value.

Design-level: there is no Rust for either read yet.
-/
namespace CchSG

open VMTensor

/-- 03 §3: `pattern(m)⟨¬struct(dm)⟩ ∪ pattern(dp)` at one pair. -/
def maskedRead (s : PairSt) : Bool := (s.m.isSome && !s.dm) || s.dp.isSome

/-- **The masked read is the pattern of the live value, with no premise**: not `dm ⊆ m`, not `dp ∩ dm = ∅`, not
    cancel-to-clean. A stale `m` under `dp` (states H, I) is in both, and `dm` hides `m` in both. -/
theorem maskedRead_eq_eff (s : PairSt) : maskedRead s = (effV s).isSome := by
  obtain ⟨m, dp, dm, mt, me⟩ := s
  cases dp <;> cases dm <;> cases m <;> simp [maskedRead, effV]

/-- **The masked read is the set of pairs with a live edge**, given only promotion completeness (`effV = M`
    implies a non-empty `me` row; `VMTensor.Inv.multi_me`). -/
theorem maskedRead_live_of_multi_me (s : PairSt) (h : effV s = some .multi → 2 ≤ s.me.length) :
    maskedRead s = true ↔ edges s ≠ [] := by
  rw [maskedRead_eq_eff]
  unfold edges
  cases he : effV s with
  | none => simp
  | some v =>
    cases v with
    | id i => simp
    | multi =>
      have := h he
      simp only [Option.isSome_some, ne_eq, true_iff]
      intro h0; rw [h0] at this; simp at this

/-- **Theorem 1 (03 §3).** Under the tensor's invariants, a pair is in `pattern(m)⟨¬struct(dm)⟩ ∪ pattern(dp)`
    iff it has a live edge (`Tensor::get` is non-empty). -/
theorem maskedRead_live (s : PairSt) (h : Inv s) : maskedRead s = true ↔ edges s ≠ [] :=
  maskedRead_live_of_multi_me s h.multi_me

/-- **Promotion completeness is needed** (and is the only invariant used): a pair whose inline value is
    `MULTI_EDGE` with an empty `me` row is read as present but has no edge. versioned_matrix proves no tensor
    operation reaches it (`addBatch_correct`, `removeBatch_correct`, …; `Tensor::decode` refuses it, #3033). -/
theorem masked_needs_promotion_completeness :
    maskedRead ⟨none, some .multi, false, true, []⟩ = true ∧ edges ⟨none, some .multi, false, true, []⟩ = [] := by
  decide

/-! ## The full-seed pass (03 §5) reads the same pattern -/

/-- `m⟨¬struct(dm) ∧ ¬struct(dp)⟩`: the `m` part of the full-seed union. -/
def seedM (s : PairSt) : Option Val := if s.dm || s.dp.isSome then none else s.m

/-- 03 §5: the disjoint union `m⟨¬struct(dm) ∧ ¬struct(dp)⟩ ∪ dp`. -/
def seedVal (s : PairSt) : Option Val :=
  match s.dp with
  | some v => some v
  | none => seedM s

theorem seed_disjoint (s : PairSt) (h : (seedM s).isSome = true) : s.dp = none := by
  unfold seedM at h; cases hd : s.dp <;> simp_all

/-- **The full-seed union is exactly the live value, with no premise** (03 §5 cites the invariants for it; they
    are not needed: the `¬struct(dp)` mask already makes the union disjoint). -/
theorem seedVal_eq_eff (s : PairSt) : seedVal s = effV s := by
  obtain ⟨m, dp, dm, mt, me⟩ := s
  cases dp <;> cases dm <;> simp [seedVal, seedM, effV]

/-- **The skeleton read and the seed read have the same pattern** at the same version: every seeded pair is in
    the skeleton, and every skeleton pair is seeded. -/
theorem seed_pattern_eq_masked (s : PairSt) : (seedVal s).isSome = maskedRead s := by
  rw [seedVal_eq_eff, maskedRead_eq_eff]

/-! ## One slab of one type -/

/-- A graph at one version: per relationship type, per `(src, dst)`, the tensor's pair state. -/
abbrev Graph := Nat → Nat → Nat → PairSt

/-- Every pair of every type satisfies the tensor's invariants. -/
def GraphInv (g : Graph) : Prop := ∀ t u w, Inv (g t u w)

/-- The slab of rows `rows`, type `t`, read at the graph `g` of the slab's version. -/
def slabRead (g : Graph) (rows : Nat → Bool) (t u w : Nat) : Bool := rows u && maskedRead (g t u w)

/-- **Theorem 1, per type and slab**: the slab's masked read is exactly the slab's pairs with a live edge of that
    type at the slab's version. -/
theorem slabRead_live (g : Graph) (hg : GraphInv g) (rows : Nat → Bool) (t u w : Nat) :
    slabRead g rows t u w = true ↔ rows u = true ∧ edges (g t u w) ≠ [] := by
  simp only [slabRead, Bool.and_eq_true, maskedRead_live _ (hg t u w)]

end CchSG
