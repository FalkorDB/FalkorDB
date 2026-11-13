import CchRecomputeReview.Bridge

/-!
# `seed` meets `relax`'s input contract, so `customize`'s `seed` then `relax` is covered end to end

`relax_isFull` starts from `seedM S`: every arc's middles are `None`. "No index out of range" needs the four arrays
to have `arc_count` entries. `seed` (`customize.rs:204-240` @main) provides both:
* the allocation (`:208-214`): four arrays of `arc_count` entries, `+∞` and `None` (`seedAlloc`);
* the scatter (`:222-237`): it writes only the two weight arrays, at `slot = arc_slot(rank_u, rank_v)`
  (`seedLoop`). Self-loops are skipped, and `None` is the `NeedsRebuild` return.

Results:
* `arcSlot_lt`: every slot `arc_slot` returns is below `arc_count`, so no write is out of range.
* `seedLoop_shape`: the scatter keeps all four lengths, for any weights, and leaves the middles untouched.
* `toMetric_seed`: read by arc (RD1), the seeded arrays are `seedM` of the seeds they hold.
* `customize_build` composes all of it with `relax_build` (`customize.rs:187-196`, `seed` then `relax`): on the
  topology `Topology::build` returns, the seeded arrays have `arc_count` entries each, and `relax` returns the
  full customization of the seeds they hold.

`seed`'s own row stays where it is (NOT COVERED in `native_index`). This module models its shape, not every
reason it may return.
-/

namespace CchRecompute

variable {W : Type}

/-- The four arrays `seed` builds (`customize.rs:209-214`), indexed by slot. -/
structure Arrays (W : Type) where
  up : List W
  dn : List W
  mup : List (Option Nat)
  mdn : List (Option Nat)

/-- `customize.rs:208-214`: `vec![f64::INFINITY; arc_count]` twice, `vec![None; arc_count]` twice. -/
def seedAlloc (o : Ops W) (m : Nat) : Arrays W :=
  ⟨List.replicate m o.top, List.replicate m o.top, List.replicate m none, List.replicate m none⟩

/-- `customize.rs:222-237`: for each weight entry `(u, v, w)` in rank-resolved form. A self-loop is skipped
(`:223-225`). `arc_slot(rank_u, rank_v)` of `None` is the `NeedsRebuild` return (`:228-230`). Otherwise `w` goes to
`upward_weight[slot]` when `rank_u < rank_v`, else to `downward_weight[slot]` (`:232-236`). -/
def seedLoop (bs : List Nat → Nat → Option Nat) (t : CchSG.Topo) :
    List (Nat × Nat × W) → Arrays W → Option (Arrays W)
  | [], a => some a
  | (u, v, w) :: rest, a =>
    if u = v then seedLoop bs t rest a
    else
      match CchSG.arcSlot bs t (t.rankOf u) (t.rankOf v) with
      | none => none
      | some k =>
        if t.rankOf u < t.rankOf v then seedLoop bs t rest { a with up := a.up.set k w }
        else seedLoop bs t rest { a with dn := a.dn.set k w }

/-- `seed(topology, weights)`: allocation, then the scatter. -/
def seedArrays (o : Ops W) (bs : List Nat → Nat → Option Nat) (t : CchSG.Topo) (entries : List (Nat × Nat × W)) :
    Option (Arrays W) :=
  seedLoop bs t entries (seedAlloc o t.arcCount)

/-- The scatter keeps all four lengths and never touches the middles, whatever the weights and slots. -/
theorem seedLoop_shape (bs : List Nat → Nat → Option Nat) (t : CchSG.Topo) :
    ∀ (entries : List (Nat × Nat × W)) (a a' : Arrays W), seedLoop bs t entries a = some a' →
      a'.up.length = a.up.length ∧ a'.dn.length = a.dn.length ∧ a'.mup = a.mup ∧ a'.mdn = a.mdn
  | [], a, a', h => by simp [seedLoop] at h; subst h; simp
  | (u, v, w) :: rest, a, a', h => by
    simp only [seedLoop] at h
    split at h
    · exact seedLoop_shape bs t rest a a' h
    · split at h
      · simp at h
      · split at h
        · have := seedLoop_shape bs t rest _ a' h
          simp only [List.length_set] at this; exact this
        · have := seedLoop_shape bs t rest _ a' h
          simp only [List.length_set] at this; exact this

section
variable {sk : NI.Cch.Csr} {r2n : List Nat} {t : CchSG.Topo}

/-- **No out-of-range write**: every slot `arc_slot` returns is below `arc_count` (via `rd1_slots`, `arc_ids`). -/
theorem arcSlot_lt (h : CchSG.fromOrder sk r2n = .ok t) (hm : CchSG.Mirrored sk) {bs : List Nat → Nat → Option Nat}
    (hbs : CchSG.BsIdx bs) {e o' k : Nat} (hk : CchSG.arcSlot bs t e o' = some k) : k < t.arcCount := by
  have hV := (bridge h hm).1
  simp only [CchSG.arcSlot] at hk
  obtain ⟨i, hi, rfl⟩ := Option.map_eq_some_iff.1 hk
  have hrow : CchSG.slice t.upNb (t.upRange (if e < o' then e else o')) = t.up (if e < o' then e else o') := rfl
  rw [hrow] at hi
  have hget := (hbs _ _ (hV.sorted _) i).1 hi
  exact ((rd1_slots h hm hbs).1 _ i _ hget).1

end

/-- The arrays read by arc (RD1): arc `(lo, hi)` is slot `arc_slot(lo, hi)`. -/
def toMetric (o : Ops W) (bs : List Nat → Nat → Option Nat) (t : CchSG.Topo) (a : Arrays W) : Metric W :=
  fun lo hi =>
    match CchSG.arcSlot bs t lo hi with
    | some k => ⟨a.up.getD k o.top, a.dn.getD k o.top, a.mup.getD k none, a.mdn.getD k none⟩
    | none => ⟨o.top, o.top, none, none⟩

/-- The seeds the arrays hold. -/
def seedsOf (o : Ops W) (bs : List Nat → Nat → Option Nat) (t : CchSG.Topo) (a : Arrays W) : Seeds W :=
  fun lo hi => ((toMetric o bs t a lo hi).up, (toMetric o bs t a lo hi).dn)

theorem getD_replicate_none (m k : Nat) : (List.replicate m (none : Option Nat)).getD k none = none := by
  simp only [List.getD, List.getElem?_replicate]
  split <;> rfl

/-- With `None` middles, the arrays read by arc are `seedM` of their seeds: `relax`'s starting point. -/
theorem toMetric_seed (o : Ops W) (bs : List Nat → Nat → Option Nat) (t : CchSG.Topo) {a : Arrays W} {m : Nat}
    (hu : a.mup = List.replicate m none) (hd : a.mdn = List.replicate m none) :
    toMetric o bs t a = seedM (seedsOf o bs t a) := by
  funext lo hi
  simp only [seedM, seedE, seedsOf, toMetric]
  split
  · rw [hu, hd, getD_replicate_none]
  · rfl

theorem build_fromOrder (bs : List Nat → Nat → Bool) (hbs : NI.Cch.BsSpec bs)
    (backend : NI.Cch.Csr → Except CchSG.OErr (List Nat)) {sk : NI.Cch.Csr} {t : CchSG.Topo}
    (h : CchSG.build (CchSG.ndChecks bs backend) sk = .ok t) :
    ∃ ord, CchSG.fromOrder sk ord = .ok t ∧ CchSG.Mirrored sk := by
  unfold CchSG.build at h
  cases ho : CchSG.ndChecks bs backend sk with
  | error e => simp [ho] at h
  | ok ord =>
    simp only [ho] at h
    exact ⟨ord, h, CchSG.mirrored_of_symmetricSimple bs hbs sk (CchSG.ndChecks_symm bs backend sk ord ho)⟩

/-- **`customize`'s `seed` then `relax`** (`customize.rs:187-196`), on the topology `Topology::build` returns.
When `seed` returns, its arrays have `arc_count` entries each, every slot it may write is in range, and the
middles are `None`. `relax` on them then never panics and returns the full customization of the seeds they hold.
The premises are the two `binary_search` std contracts (`BsSpec` for `nd`'s check, `BsIdx` for `arc_slot`). -/
theorem customize_build (o : Ops W) (bsB : List Nat → Nat → Bool) (hbsB : NI.Cch.BsSpec bsB)
    (bs : List Nat → Nat → Option Nat) (hbs : CchSG.BsIdx bs)
    (backend : NI.Cch.Csr → Except CchSG.OErr (List Nat)) {sk : NI.Cch.Csr} {t : CchSG.Topo}
    (h : CchSG.build (CchSG.ndChecks bsB backend) sk = .ok t) (entries : List (Nat × Nat × W)) {a : Arrays W}
    (ha : seedArrays o bs t entries = some a) :
    a.up.length = t.arcCount ∧ a.dn.length = t.arcCount ∧
    a.mup = List.replicate t.arcCount none ∧ a.mdn = List.replicate t.arcCount none ∧
    (∀ e o' k, CchSG.arcSlot bs t e o' = some k → k < t.arcCount) ∧
    ∃ R, relax o (ofSG t) (toMetric o bs t a) = some R ∧ IsFull o (ofSG t) (seedsOf o bs t a) R := by
  obtain ⟨ord, hfo, hm⟩ := build_fromOrder bsB hbsB backend h
  obtain ⟨h1, h2, h3, h4⟩ := seedLoop_shape bs t entries _ a ha
  simp only [seedAlloc, List.length_replicate] at h1 h2 h3 h4
  refine ⟨h1, h2, h3, h4, fun e o' k hk => arcSlot_lt hfo hm hbs hk, ?_⟩
  rw [toMetric_seed o bs t h3 h4]
  exact relax_fromOrder o hfo hm _

end CchRecompute
