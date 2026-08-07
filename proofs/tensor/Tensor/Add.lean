/-
# `set_all_from_slices` — one edge at a time

`addEdge t p id` is the read-phase decision for a pair not yet seen in the batch
(`Entry::Vacant`) together with its write-phase effect.  Proved here:

* `inv_addEdge`  — every delta-layer invariant survives, for all three branches
  (already-multi / promote / first-edge) and all three write-phase shapes
  (plain `dp` set, un-mask, cancel-to-clean);
* `edgesAt_addEdge_self` — the pair gains exactly `id`;
* `edgesAt_addEdge_ne`   — no other pair changes;
* the same three facts for a whole list (`setAll`), by induction.

Preconditions, all of them real requirements of the Rust code:

* `InBounds t p` — `compound_key` asserts each side fits a `u32`, and GraphBLAS
  requires the coordinate to be inside the matrix (callers `resize` first);
* `ValidId id`   — an edge id is a GraphBLAS index (`≤ GrB_INDEX_MAX`), so it is
  never the `MULTI_EDGE` sentinel;
* `id ∉ t.edgesAt p` — edge ids are freshly allocated, so the same id is never
  inserted twice into one pair.
-/
import Tensor.Reads

namespace FalkorDB
namespace Tensor

variable {t : Tensor} {p : Pair} {id : Nat}

/-- The `m_masked` slot of the read phase is either the committed value or
`none`, so reading it back always agrees with `m`. -/
theorem mm_of_if {L : Layer Nat} {q : Pair} {P : Prop} [Decidable P] :
    ∀ c, (if P then L.get q else none) = some c → L.get q = some c := by
  intro c hc
  by_cases hP : P
  · rw [if_pos hP] at hc; exact hc
  · rw [if_neg hP] at hc; exact absurd hc (by simp)

/-! ## The write phase -/

section WriteInline

variable {mm : Option Nat}

@[simp] theorem writeInline_m : (writeInline t p id mm).m = t.m := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id <;> simp [writeInline, hc]

@[simp] theorem writeInline_me : (writeInline t p id mm).me = t.me := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id <;> simp [writeInline, hc]

@[simp] theorem writeInline_multiCount : (writeInline t p id mm).multiCount = t.multiCount := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id <;> simp [writeInline, hc]

@[simp] theorem writeInline_nrows : (writeInline t p id mm).nrows = t.nrows := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id <;> simp [writeInline, hc]

@[simp] theorem writeInline_ncols : (writeInline t p id mm).ncols = t.ncols := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id <;> simp [writeInline, hc]

@[simp] theorem writeInline_mt : (writeInline t p id mm).mt = insert (p.2, p.1) t.mt := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id <;> simp [writeInline, hc]

/-- `dm` only ever loses `p` (`dp ∩ dm = ∅` is restored). -/
theorem writeInline_mem_dm {q : Pair} :
    q ∈ (writeInline t p id mm).dm ↔ q ≠ p ∧ q ∈ t.dm ∨ mm = none ∧ q ∈ t.dm := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id <;> simp [writeInline, hc, Finset.mem_erase]

theorem writeInline_dm_subset : (writeInline t p id mm).dm ⊆ t.dm := by
  intro q hq
  rcases writeInline_mem_dm.mp hq with ⟨_, h⟩ | ⟨_, h⟩ <;> exact h

theorem writeInline_not_mem_dm (hnone : mm = none → p ∉ t.dm) :
    p ∉ (writeInline t p id mm).dm := by
  intro hp
  rcases writeInline_mem_dm.mp hp with ⟨hne, _⟩ | ⟨h0, hdm⟩
  · exact hne rfl
  · exact hnone h0 hdm

theorem writeInline_mem_dm_ne {q : Pair} (hq : q ≠ p) :
    q ∈ (writeInline t p id mm).dm ↔ q ∈ t.dm := by
  rw [writeInline_mem_dm]
  constructor
  · rintro (⟨_, h⟩ | ⟨_, h⟩) <;> exact h
  · intro h; exact Or.inl ⟨hq, h⟩

theorem writeInline_dp_get_ne {q : Pair} (hq : q ≠ p) :
    (writeInline t p id mm).dp.get q = t.dp.get q := by
  rcases mm with _ | c
  · simp [writeInline, hq]
  · by_cases hc : c = id <;> simp [writeInline, hc, hq]

theorem writeInline_dp_dom_subset : (writeInline t p id mm).dp.dom ⊆ insert p t.dp.dom := by
  rcases mm with _ | c
  · simp [writeInline]
  · by_cases hc : c = id
    · intro q hq
      have hq' : q ∈ t.dp.dom.erase p := by simpa [writeInline, hc] using hq
      exact Finset.mem_insert_of_mem (Finset.mem_of_mem_erase hq')
    · simp [writeInline, hc]

/-! The write phase's effect on `dp`/`dm` at the written pair, one lemma per
shape (used by the state-diagram proofs). -/

@[simp] theorem writeInline_none_dp : (writeInline t p id none).dp.get p = some id := by
  simp [writeInline]

@[simp] theorem writeInline_cancel_dp : (writeInline t p id (some id)).dp.get p = none := by
  simp [writeInline]

theorem writeInline_shadow_dp {c : Nat} (h : c ≠ id) :
    (writeInline t p id (some c)).dp.get p = some id := by
  simp [writeInline, h]

@[simp] theorem writeInline_none_dm : (writeInline t p id none).dm = t.dm := by
  simp [writeInline]

@[simp] theorem writeInline_some_dm {c : Nat} : (writeInline t p id (some c)).dm = t.dm.erase p := by
  by_cases hc : c = id <;> simp [writeInline, hc]

/-- Whatever shape the write phase takes, the pair's effective inline value is
the value that was queued for it.  (In the cancel-to-clean shape both deltas are
dropped and the committed value — equal to `id` — shows through.) -/
theorem writeInline_effGet_self (hmm : ∀ c, mm = some c → t.m.get p = some c) :
    (writeInline t p id mm).effGet p = some id := by
  rcases mm with _ | c
  · exact effGet_of_dp (by simp [writeInline])
  · by_cases hc : c = id
    · subst hc
      have h1 : (writeInline t p c (some c)).effGet p = (writeInline t p c (some c)).m.get p :=
        effGet_of_m (by simp [writeInline]) (by simp [writeInline])
      rw [h1, writeInline_m]
      exact hmm c rfl
    · exact effGet_of_dp (by simp [writeInline, hc])

theorem writeInline_effGet_ne {q : Pair} (hq : q ≠ p) :
    (writeInline t p id mm).effGet q = t.effGet q :=
  effGet_congr_at writeInline_m (writeInline_dp_get_ne hq) (writeInline_mem_dm_ne hq)

theorem writeInline_effDom (hmm : ∀ c, mm = some c → t.m.get p = some c) :
    (writeInline t p id mm).effDom = insert p t.effDom :=
  effDom_eq_of_effGet (by rw [writeInline_effGet_self hmm]; rfl)
    (fun _ hq => writeInline_effGet_ne hq)

/-- The three invariants about the forward layers, for any write-phase shape.
Only the three forward-layer invariants of the input are needed, so this applies
to the intermediate state of the promotion branch too (which has already updated
`me` and `multi_count`). -/
theorem writeInline_layer_inv (hsub : t.dm ⊆ t.m.dom) (hdisj : Disjoint t.dp.dom t.dm)
    (hcc : ∀ q ∈ t.dp.dom, t.m.get q ≠ some (t.dp.val q))
    (hmm : ∀ c, mm = some c → t.m.get p = some c)
    (hnone : mm = none → p ∉ t.dm ∧ t.m.get p ≠ some id) :
    (writeInline t p id mm).dm ⊆ (writeInline t p id mm).m.dom ∧
      Disjoint (writeInline t p id mm).dp.dom (writeInline t p id mm).dm ∧
      ∀ q ∈ (writeInline t p id mm).dp.dom,
        (writeInline t p id mm).m.get q ≠ some ((writeInline t p id mm).dp.val q) := by
  refine ⟨?_, ?_, ?_⟩
  · rw [writeInline_m]
    exact fun q hq => hsub (writeInline_dm_subset hq)
  · rw [Finset.disjoint_left]
    intro q hq hq'
    have hqdm : q ∈ t.dm := writeInline_dm_subset hq'
    by_cases hqp : q = p
    · subst hqp
      exact writeInline_not_mem_dm (fun h0 => (hnone h0).1) hq'
    · exact (Finset.disjoint_left.mp hdisj (by
        have := writeInline_dp_dom_subset hq
        rcases Finset.mem_insert.mp this with h' | h'
        · exact absurd h' hqp
        · exact h')) hqdm
  · intro q hq
    rw [writeInline_m]
    by_cases hqp : q = p
    · subst hqp
      rcases mm with _ | c
      · have hval : (writeInline t q id none).dp.val q = id := by simp [writeInline, Layer.set]
        rw [hval]
        exact (hnone rfl).2
      · by_cases hc : c = id
        · exact absurd hq (by simp [writeInline, hc])
        · have hval : (writeInline t q id (some c)).dp.val q = id := by
            simp [writeInline, hc, Layer.set]
          rw [hval]
          intro hcontra
          rw [hmm c rfl] at hcontra
          exact hc (Option.some_inj.mp hcontra)
    · have hq' : q ∈ t.dp.dom := by
        rcases Finset.mem_insert.mp (writeInline_dp_dom_subset hq) with h' | h'
        · exact absurd h' hqp
        · exact h'
      have hval : (writeInline t p id mm).dp.val q = t.dp.val q := by
        have := writeInline_dp_get_ne (t := t) (id := id) (mm := mm) hqp
        rw [Layer.get_of_mem hq, Layer.get_of_mem hq'] at this
        exact Option.some_inj.mp this
      rw [hval]
      exact hcc q hq'

end WriteInline

/-! ## `addEdge`

The three branches are handled separately; `inv_addEdge` and the two `edgesAt`
theorems below combine them. -/

theorem addEdge_multi_def (hv : t.effGet p = some MULTI) :
    addEdge t p id = { t with me := insert (key p, id) t.me } := by
  simp [addEdge, hv]

theorem addEdge_promote_def {v : Nat} (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    addEdge t p id =
      writeInline
        { t with me := insert (key p, v) (insert (key p, id) t.me),
                 multiCount := t.multiCount + 1 }
        p MULTI (if (t.dp.get p).isSome then t.m.get p else none) := by
  simp [addEdge, hv, hM]

theorem addEdge_first_def (hv : t.effGet p = none) :
    addEdge t p id = writeInline t p id (if p ∈ t.dm then t.m.get p else none) := by
  simp [addEdge, hv]

/-- Both non-trivial branches leave the capacity alone. -/
@[simp] theorem addEdge_nrows : (addEdge t p id).nrows = t.nrows := by
  unfold addEdge
  cases t.effGet p with
  | none => simp
  | some v => by_cases hM : v = MULTI <;> simp [hM]

@[simp] theorem addEdge_ncols : (addEdge t p id).ncols = t.ncols := by
  unfold addEdge
  cases t.effGet p with
  | none => simp
  | some v => by_cases hM : v = MULTI <;> simp [hM]

@[simp] theorem addEdge_m : (addEdge t p id).m = t.m := by
  unfold addEdge
  cases t.effGet p with
  | none => simp
  | some v => by_cases hM : v = MULTI <;> simp [hM]

/-! ### Branch 1: the pair is already multi-edge

`me.set(key, id)` and nothing else: the inline sentinel, `mt`, `dp`, `dm` and
`multi_count` are all already correct. -/

section Multi

private theorem multi_effGet (hv : t.effGet p = some MULTI) (q : Pair) :
    (addEdge t p id).effGet q = t.effGet q := by
  rw [addEdge_multi_def hv]; rfl

private theorem multi_effDom (hv : t.effGet p = some MULTI) :
    (addEdge t p id).effDom = t.effDom := by
  rw [addEdge_multi_def hv]; rfl

private theorem multi_multiCount (hv : t.effGet p = some MULTI) :
    (addEdge t p id).multiCount = t.multiCount := by
  rw [addEdge_multi_def hv]

private theorem multi_meRow_self (hv : t.effGet p = some MULTI) :
    (addEdge t p id).meRow (key p) = insert id (t.meRow (key p)) := by
  rw [addEdge_multi_def hv]; simp [meRow]

private theorem multi_meRow_ne (hv : t.effGet p = some MULTI) {q : Pair} (hq : key q ≠ key p) :
    (addEdge t p id).meRow (key q) = t.meRow (key q) := by
  rw [addEdge_multi_def hv]; simp [meRow, meRowOf_insert_ne hq]

/-- The pair gains exactly `id`. -/
theorem edgesAt_addEdge_multi_self (hv : t.effGet p = some MULTI) :
    (addEdge t p id).edgesAt p = insert id (t.edgesAt p) := by
  simp [edgesAt, multi_effGet hv, hv, multi_meRow_self hv]

/-- No other pair is touched: the new `me` entry sits in `p`'s row, and
`compound_key` is injective. -/
theorem edgesAt_addEdge_multi_ne (h : Inv t) (hbp : InBounds t p)
    (hv : t.effGet p = some MULTI) {q : Pair} (hq : q ≠ p) :
    (addEdge t p id).edgesAt q = t.edgesAt q := by
  simp only [edgesAt, multi_effGet hv]
  cases hgq : t.effGet q with
  | none => simp
  | some w =>
    by_cases hM : w = MULTI
    · have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by simp [hgq]))
      simp [hM, multi_meRow_ne hv (key_ne hbq hbp.1 hq)]
    · simp [hM]

theorem inv_addEdge_multi (h : Inv t) (hbp : InBounds t p) (hid : ValidId id)
    (hv : t.effGet p = some MULTI) : Inv (addEdge t p id) := by
  have hpdom : p ∈ t.effDom := mem_effDom_iff_isSome.mpr (by simp [hv])
  refine { dm_sub_m := ?_, dp_disj_dm := ?_, cancel_clean := ?_, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, bounded := ?_, in_range := ?_,
           multi_count_eq := ?_, valid_ids := ?_,
           mt_eq := by rw [multi_effDom hv, addEdge_multi_def hv]; exact h.mt_eq }
  · rw [addEdge_multi_def hv]; exact h.dm_sub_m
  · rw [addEdge_multi_def hv]; exact h.dp_disj_dm
  · rw [addEdge_multi_def hv]; exact h.cancel_clean
  · intro q hq
    rw [multi_effGet hv] at hq
    by_cases hqp : q = p
    · subst hqp
      rw [multi_meRow_self hv]
      have h1 := h.multi_iff q hq
      have h2 := Finset.card_le_card (Finset.subset_insert id (t.meRow (key q)))
      omega
    · have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by simp [hq]))
      rw [multi_meRow_ne hv (key_ne hbq hbp.1 hqp)]
      exact h.multi_iff q hq
  · intro q hbq hq
    rw [multi_effGet hv] at hq
    have hqp : q ≠ p := by rintro rfl; exact hq hv
    rw [multi_meRow_ne hv (key_ne hbq hbp.1 hqp)]
    exact h.row_empty q hbq hq
  · intro x hx
    rw [addEdge_multi_def hv] at hx
    rcases Finset.mem_insert.mp hx with rfl | hx
    · exact ⟨p, hbp.1, by rw [multi_effDom hv]; exact hpdom, rfl⟩
    · obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed x hx
      exact ⟨q, hbq, by rw [multi_effDom hv]; exact hqdom, hqk⟩
  · rw [multi_effDom hv]; exact h.bounded
  · rw [addEdge_multi_def hv]; exact h.in_range
  · rw [multi_multiCount hv, h.multi_count_eq]
    exact congrArg Finset.card (multiPairs_congr (multi_effDom hv) (multi_effGet hv)).symm
  · intro q i hi
    by_cases hqp : q = p
    · subst hqp
      rw [edgesAt_addEdge_multi_self hv] at hi
      rcases Finset.mem_insert.mp hi with rfl | hi
      · exact hid
      · exact h.valid_ids q i hi
    · rw [edgesAt_addEdge_multi_ne h hbp hv hqp] at hi
      exact h.valid_ids q i hi

end Multi


/-! ### Branch 3: the pair's first edge

`dp` (or the un-masked committed entry) carries the id inline; `me` stays empty
for this pair, which is what keeps a single-edge graph free of `me` rows. -/

section First

variable {mmF : Option Nat}

theorem first_mm :
    ∀ c, (if p ∈ t.dm then t.m.get p else none) = some c → t.m.get p = some c := by
  intro c hc
  by_cases hdm : p ∈ t.dm
  · rw [if_pos hdm] at hc; exact hc
  · rw [if_neg hdm] at hc; exact absurd hc (by simp)

private theorem first_hnone (hv : t.effGet p = none) (h : Inv t) :
    (if p ∈ t.dm then t.m.get p else none) = none → p ∉ t.dm ∧ t.m.get p ≠ some id := by
  intro h0
  by_cases hdm : p ∈ t.dm
  · rw [if_pos hdm] at h0
    exact absurd (h.dm_sub_m hdm) (Layer.get_eq_none.mp h0)
  · refine ⟨hdm, ?_⟩
    rw [m_get_eq_none_of_effGet_none hv hdm]
    simp

private theorem first_effGet_self (hv : t.effGet p = none) :
    (addEdge t p id).effGet p = some id := by
  rw [addEdge_first_def hv]
  exact writeInline_effGet_self (first_mm)

private theorem first_effGet_ne (hv : t.effGet p = none) {q : Pair} (hq : q ≠ p) :
    (addEdge t p id).effGet q = t.effGet q := by
  rw [addEdge_first_def hv]; exact writeInline_effGet_ne hq

private theorem first_effDom (hv : t.effGet p = none) :
    (addEdge t p id).effDom = insert p t.effDom := by
  rw [addEdge_first_def hv]; exact writeInline_effDom (first_mm)

private theorem first_me (hv : t.effGet p = none) : (addEdge t p id).me = t.me := by
  rw [addEdge_first_def hv]; exact writeInline_me

/-- The pair gains exactly `id` (it had none). -/
theorem edgesAt_addEdge_first_self (hid : ValidId id) (hv : t.effGet p = none) :
    (addEdge t p id).edgesAt p = insert id (t.edgesAt p) := by
  have h1 : t.edgesAt p = ∅ := by simp [edgesAt, hv]
  rw [h1]
  simp [edgesAt, first_effGet_self hv, hid.ne_multi]

theorem edgesAt_addEdge_first_ne (hv : t.effGet p = none) {q : Pair} (hq : q ≠ p) :
    (addEdge t p id).edgesAt q = t.edgesAt q :=
  edgesAt_eq_of_effGet_eq (first_me hv) (first_effGet_ne hv hq)

theorem inv_addEdge_first (h : Inv t) (hbp : InBounds t p) (hid : ValidId id)
    (hv : t.effGet p = none) : Inv (addEdge t p id) := by
  have hpdom : p ∉ t.effDom := effGet_eq_none_iff.mp hv
  obtain ⟨hsub, hdisj, hcc⟩ :=
    writeInline_layer_inv (t := t) (p := p) (id := id) (mm := if p ∈ t.dm then t.m.get p else none)
      h.dm_sub_m h.dp_disj_dm h.cancel_clean (first_mm) (first_hnone hv h)
  rw [← addEdge_first_def hv] at hsub hdisj hcc
  refine { dm_sub_m := hsub, dp_disj_dm := hdisj, cancel_clean := hcc, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, bounded := ?_, in_range := ?_,
           multi_count_eq := ?_, valid_ids := ?_,
           mt_eq := mt_eq_insert h.mt_eq (first_effDom hv)
             (by rw [addEdge_first_def hv]; exact writeInline_mt) }
  · intro q hq
    have hqp : q ≠ p := by
      rintro rfl
      rw [first_effGet_self hv] at hq
      exact hid.ne_multi (Option.some_inj.mp hq)
    rw [first_effGet_ne hv hqp] at hq
    rw [meRow, first_me hv]
    exact h.multi_iff q hq
  · intro q hbq hq
    rw [meRow, first_me hv]
    by_cases hqp : q = p
    · subst hqp
      exact h.row_empty q hbq (by rw [hv]; simp)
    · rw [first_effGet_ne hv hqp] at hq
      exact h.row_empty q hbq hq
  · intro x hx
    rw [first_me hv] at hx
    obtain ⟨q, hbq, hqdom, hqk⟩ := h.me_keyed x hx
    exact ⟨q, hbq, by rw [first_effDom hv]; exact Finset.mem_insert_of_mem hqdom, hqk⟩
  · intro q hq
    rw [first_effDom hv] at hq
    rcases Finset.mem_insert.mp hq with rfl | hq
    · exact hbp.1
    · exact h.bounded q hq
  · intro q hq
    rw [addEdge_nrows, addEdge_ncols]
    rw [addEdge_first_def hv] at hq
    rcases Finset.mem_union.mp hq with hq' | hq'
    · exact h.in_range q (Finset.mem_union_left _ (by simpa using hq'))
    · rcases Finset.mem_insert.mp (writeInline_dp_dom_subset hq') with rfl | hq''
      · exact ⟨hbp.2.1, hbp.2.2⟩
      · exact h.in_range q (Finset.mem_union_right _ hq'')
  · have hmc : (addEdge t p id).multiCount = t.multiCount := by
      rw [addEdge_first_def hv]; exact writeInline_multiCount
    rw [hmc, h.multi_count_eq]
    refine congrArg Finset.card (multiPairs_eq_of_not_multi hpdom (first_effDom hv) ?_
      (fun q hq => first_effGet_ne hv hq)).symm
    rw [first_effGet_self hv]
    exact fun hc => hid.ne_multi (Option.some_inj.mp hc)
  · intro q i hi
    by_cases hqp : q = p
    · subst hqp
      rw [edgesAt_addEdge_first_self hid hv] at hi
      rcases Finset.mem_insert.mp hi with rfl | hi
      · exact hid
      · exact h.valid_ids q i hi
    · rw [edgesAt_addEdge_first_ne hv hqp] at hi
      exact h.valid_ids q i hi

end First

/-! ### Branch 2: promotion of a single-edge pair

Both the existing inline id and the new one move into `me`, the inline slot gets
the `MULTI` sentinel and `multi_count` is bumped.  The write phase may *cancel*
instead of shadowing, when the committed value is already the sentinel (a pair
demoted earlier in the same transaction) — the case the `m_masked` bookkeeping in
`set_all_from_slices` exists for. -/

section Promote

variable {v : Nat}

/-- The intermediate state: `me` and `multi_count` updated, forward layers still
untouched (the read phase writes `me` eagerly). -/
private def promoted (t : Tensor) (p : Pair) (id v : Nat) : Tensor :=
  { t with me := insert (key p, v) (insert (key p, id) t.me),
           multiCount := t.multiCount + 1 }

@[simp] private theorem promoted_m : (promoted t p id v).m = t.m := rfl
@[simp] private theorem promoted_dm : (promoted t p id v).dm = t.dm := rfl
@[simp] private theorem promoted_dp : (promoted t p id v).dp = t.dp := rfl
@[simp] private theorem promoted_mt : (promoted t p id v).mt = t.mt := rfl
@[simp] private theorem promoted_nrows : (promoted t p id v).nrows = t.nrows := rfl
@[simp] private theorem promoted_ncols : (promoted t p id v).ncols = t.ncols := rfl
@[simp] private theorem promoted_multiCount :
    (promoted t p id v).multiCount = t.multiCount + 1 := rfl
@[simp] private theorem promoted_me :
    (promoted t p id v).me = insert (key p, v) (insert (key p, id) t.me) := rfl

private theorem promoted_effGet (q : Pair) : (promoted t p id v).effGet q = t.effGet q := rfl

private theorem promoted_effDom : (promoted t p id v).effDom = t.effDom := rfl

private theorem addEdge_promote_def' (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    addEdge t p id =
      writeInline (promoted t p id v) p MULTI
        (if (t.dp.get p).isSome then t.m.get p else none) := by
  rw [addEdge_promote_def hv hM]; rfl

private theorem promote_mm :
    ∀ c, (if (t.dp.get p).isSome then t.m.get p else none) = some c →
      (promoted t p id v).m.get p = some c := by
  intro c hc
  rw [promoted_m]
  by_cases hdp : (t.dp.get p).isSome
  · rw [if_pos hdp] at hc; exact hc
  · rw [if_neg hdp] at hc; exact absurd hc (by simp)

private theorem promote_hnone (h : Inv t) (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    (if (t.dp.get p).isSome then t.m.get p else none) = none →
      p ∉ (promoted t p id v).dm ∧ (promoted t p id v).m.get p ≠ some MULTI := by
  intro h0
  rw [promoted_dm, promoted_m]
  by_cases hdp : (t.dp.get p).isSome
  · rw [if_pos hdp] at h0
    refine ⟨Finset.disjoint_left.mp h.dp_disj_dm (Layer.get_isSome.mp hdp), ?_⟩
    rw [h0]; simp
  · have hdpn : t.dp.get p = none := by simpa using hdp
    have hnotdm : p ∉ t.dm := by
      intro hdm
      rw [effGet, hdpn] at hv
      simp [hdm] at hv
    have hmv : t.m.get p = some v := by rw [← effGet_of_m hdpn hnotdm]; exact hv
    refine ⟨hnotdm, ?_⟩
    rw [hmv]
    simpa using hM

private theorem promote_effGet_self (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    (addEdge t p id).effGet p = some MULTI := by
  rw [addEdge_promote_def' hv hM]
  exact writeInline_effGet_self promote_mm

private theorem promote_effGet_ne (hv : t.effGet p = some v) (hM : v ≠ MULTI) {q : Pair}
    (hq : q ≠ p) : (addEdge t p id).effGet q = t.effGet q := by
  rw [addEdge_promote_def' hv hM, writeInline_effGet_ne hq]
  exact promoted_effGet q

private theorem promote_effDom (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    (addEdge t p id).effDom = t.effDom := by
  have hp : p ∈ t.effDom := mem_effDom_iff_isSome.mpr (by rw [hv]; rfl)
  refine effDom_eq_of_effGet_of_mem hp ?_ (fun q hq => promote_effGet_ne hv hM hq)
  rw [promote_effGet_self hv hM]; rfl

private theorem promote_me (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    (addEdge t p id).me = insert (key p, v) (insert (key p, id) t.me) := by
  rw [addEdge_promote_def' hv hM, writeInline_me]; rfl

private theorem promote_multiCount (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    (addEdge t p id).multiCount = t.multiCount + 1 := by
  rw [addEdge_promote_def' hv hM, writeInline_multiCount]; rfl

/-- The old row was empty (the pair was single-edge), so after promotion the row
holds exactly the two ids. -/
private theorem promote_meRow_self (h : Inv t) (hbp : InBounds t p) (hv : t.effGet p = some v)
    (hM : v ≠ MULTI) : (addEdge t p id).meRow (key p) = {v, id} := by
  have hrow : t.meRow (key p) = ∅ := h.row_empty p hbp.1 (by rw [hv]; simpa using hM)
  rw [meRow, promote_me hv hM, meRowOf_insert_self, meRowOf_insert_self]
  rw [show meRowOf t.me (key p) = ∅ from hrow]
  rfl

private theorem promote_meRow_ne (hv : t.effGet p = some v) (hM : v ≠ MULTI) {q : Pair}
    (hkq : key q ≠ key p) : (addEdge t p id).meRow (key q) = t.meRow (key q) := by
  rw [meRow, promote_me hv hM, meRowOf_insert_ne hkq, meRowOf_insert_ne hkq]
  rfl

/-- The pair gains exactly `id`: `{v}` becomes `{v, id}`. -/
theorem edgesAt_addEdge_promote_self (h : Inv t) (hbp : InBounds t p) (hv : t.effGet p = some v)
    (hM : v ≠ MULTI) : (addEdge t p id).edgesAt p = insert id (t.edgesAt p) := by
  simp only [edgesAt, promote_effGet_self hv hM, hv, if_neg hM,
    promote_meRow_self h hbp hv hM]
  exact Finset.pair_comm v id

theorem edgesAt_addEdge_promote_ne (h : Inv t) (hbp : InBounds t p) (hv : t.effGet p = some v)
    (hM : v ≠ MULTI) {q : Pair} (hq : q ≠ p) :
    (addEdge t p id).edgesAt q = t.edgesAt q := by
  simp only [edgesAt, promote_effGet_ne hv hM hq]
  cases hgq : t.effGet q with
  | none => rfl
  | some w =>
    by_cases hMw : w = MULTI
    · have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by rw [hgq]; rfl))
      simp only [if_pos hMw, promote_meRow_ne hv hM (key_ne hbq hbp.1 hq)]
    · simp only [if_neg hMw]

theorem inv_addEdge_promote (h : Inv t) (hbp : InBounds t p) (hid : ValidId id)
    (hfresh : id ∉ t.edgesAt p) (hv : t.effGet p = some v) (hM : v ≠ MULTI) :
    Inv (addEdge t p id) := by
  have hpdom : p ∈ t.effDom := mem_effDom_iff_isSome.mpr (by rw [hv]; rfl)
  have hvid : id ≠ v := by
    intro hc
    exact hfresh (by simp only [edgesAt, hv, if_neg hM, hc]; simp)
  obtain ⟨hsub, hdisj, hcc⟩ :=
    writeInline_layer_inv (t := promoted t p id v) (p := p) (id := MULTI)
      (mm := if (t.dp.get p).isSome then t.m.get p else none)
      h.dm_sub_m h.dp_disj_dm h.cancel_clean promote_mm (promote_hnone h hv hM)
  rw [← addEdge_promote_def' hv hM] at hsub hdisj hcc
  refine { dm_sub_m := hsub, dp_disj_dm := hdisj, cancel_clean := hcc, multi_iff := ?_,
           row_empty := ?_, me_keyed := ?_, bounded := ?_, in_range := ?_,
           multi_count_eq := ?_, valid_ids := ?_,
           mt_eq := mt_eq_insert (p := p) h.mt_eq
             (by rw [promote_effDom hv hM, Finset.insert_eq_self.mpr hpdom])
             (by rw [addEdge_promote_def' hv hM, writeInline_mt]; rfl) }
  · intro q hq
    by_cases hqp : q = p
    · subst hqp
      rw [promote_meRow_self h hbp hv hM]
      rw [Finset.card_insert_of_notMem (by simpa using fun hc => hvid hc.symm)]
      simp
    · rw [promote_effGet_ne hv hM hqp] at hq
      have hbq : Bounded q := h.bounded q (mem_effDom_iff_isSome.mpr (by rw [hq]; rfl))
      rw [promote_meRow_ne hv hM (key_ne hbq hbp.1 hqp)]
      exact h.multi_iff q hq
  · intro q hbq hq
    have hqp : q ≠ p := by
      rintro rfl
      exact hq (promote_effGet_self hv hM)
    rw [promote_effGet_ne hv hM hqp] at hq
    rw [promote_meRow_ne hv hM (key_ne hbq hbp.1 hqp)]
    exact h.row_empty q hbq hq
  · intro x hx
    rw [promote_me hv hM] at hx
    rw [promote_effDom hv hM]
    rcases Finset.mem_insert.mp hx with rfl | hx
    · exact ⟨p, hbp.1, hpdom, rfl⟩
    · rcases Finset.mem_insert.mp hx with rfl | hx
      · exact ⟨p, hbp.1, hpdom, rfl⟩
      · exact h.me_keyed x hx
  · rw [promote_effDom hv hM]; exact h.bounded
  · intro q hq
    rw [addEdge_nrows, addEdge_ncols]
    rw [addEdge_promote_def' hv hM] at hq
    rcases Finset.mem_union.mp hq with hq' | hq'
    · exact h.in_range q (Finset.mem_union_left _ (by simpa using hq'))
    · rcases Finset.mem_insert.mp (writeInline_dp_dom_subset hq') with rfl | hq''
      · exact ⟨hbp.2.1, hbp.2.2⟩
      · exact h.in_range q (Finset.mem_union_right _ (by simpa using hq''))
  · rw [promote_multiCount hv hM, h.multi_count_eq,
      multiPairs_eq_insert (t := t) (t' := addEdge t p id)
        (by rw [promote_effDom hv hM, Finset.insert_eq_self.mpr hpdom])
        (promote_effGet_self hv hM) (fun q hq => promote_effGet_ne hv hM hq),
      Finset.card_insert_of_notMem (not_mem_multiPairs_of_ne (by rw [hv]; simpa using hM))]
  · intro q i hi
    by_cases hqp : q = p
    · subst hqp
      rw [edgesAt_addEdge_promote_self h hbp hv hM] at hi
      rcases Finset.mem_insert.mp hi with rfl | hi
      · exact hid
      · exact h.valid_ids q i hi
    · rw [edgesAt_addEdge_promote_ne h hbp hv hM hqp] at hi
      exact h.valid_ids q i hi

end Promote

/-! ## `addEdge`, all branches together -/

/-- **`set_all_from_slices` preserves the delta-layer invariants** (single edge). -/
theorem inv_addEdge (h : Inv t) (hbp : InBounds t p) (hid : ValidId id)
    (hfresh : id ∉ t.edgesAt p) : Inv (addEdge t p id) := by
  cases hv : t.effGet p with
  | none => exact inv_addEdge_first h hbp hid hv
  | some v =>
    by_cases hM : v = MULTI
    · subst hM; exact inv_addEdge_multi h hbp hid hv
    · exact inv_addEdge_promote h hbp hid hfresh hv hM

/-- **The pair gains exactly the inserted id** — whether it went inline, promoted
the pair to `me`, or was appended to an existing `me` row. -/
theorem edgesAt_addEdge_self (h : Inv t) (hbp : InBounds t p) (hid : ValidId id) :
    (addEdge t p id).edgesAt p = insert id (t.edgesAt p) := by
  cases hv : t.effGet p with
  | none => exact edgesAt_addEdge_first_self hid hv
  | some v =>
    by_cases hM : v = MULTI
    · subst hM; exact edgesAt_addEdge_multi_self hv
    · exact edgesAt_addEdge_promote_self h hbp hv hM

/-- **No other pair is affected.** -/
theorem edgesAt_addEdge_ne (h : Inv t) (hbp : InBounds t p) {q : Pair} (hq : q ≠ p) :
    (addEdge t p id).edgesAt q = t.edgesAt q := by
  cases hv : t.effGet p with
  | none => exact edgesAt_addEdge_first_ne hv hq
  | some v =>
    by_cases hM : v = MULTI
    · subst hM; exact edgesAt_addEdge_multi_ne h hbp hv hq
    · exact edgesAt_addEdge_promote_ne h hbp hv hM hq

theorem inBounds_addEdge {q : Pair} (hq : InBounds t q) : InBounds (addEdge t p id) q := by
  simpa [InBounds] using hq

/-! ## A whole batch

`set_all_from_slices` is handed parallel slices; here a batch is the list of
`(pair, edge id)` it denotes. -/

/-- The ids the batch adds at pair `q`. -/
def batchIds (l : List (Pair × Nat)) (q : Pair) : Finset Nat :=
  ((l.filter (fun e => e.1 = q)).map Prod.snd).toFinset

@[simp] theorem batchIds_nil {q : Pair} : batchIds [] q = ∅ := rfl

theorem batchIds_cons_self {e : Pair × Nat} {l : List (Pair × Nat)} :
    batchIds (e :: l) e.1 = insert e.2 (batchIds l e.1) := by
  simp [batchIds]

theorem batchIds_cons_ne {e : Pair × Nat} {l : List (Pair × Nat)} {q : Pair} (hq : q ≠ e.1) :
    batchIds (e :: l) q = batchIds l q := by
  simp [batchIds, Ne.symm hq]

/-- Edge ids are allocated fresh, so a batch never re-inserts an id that is
already stored, nor the same id twice. -/
def FreshBatch (t : Tensor) (l : List (Pair × Nat)) : Prop :=
  (l.map Prod.snd).Nodup ∧ ∀ e ∈ l, ∀ q : Pair, e.2 ∉ t.edgesAt q

/-- A batch the graph layer may hand to `set_all_from_slices`. -/
def WritableBatch (t : Tensor) (l : List (Pair × Nat)) : Prop :=
  ∀ e ∈ l, InBounds t e.1 ∧ ValidId e.2

@[simp] theorem setAll_nil : setAll t [] = t := rfl

@[simp] theorem setAll_cons {e : Pair × Nat} {l : List (Pair × Nat)} :
    setAll t (e :: l) = setAll (addEdge t e.1 e.2) l := rfl

@[simp] theorem setAll_nrows {l : List (Pair × Nat)} : (setAll t l).nrows = t.nrows := by
  induction l generalizing t with
  | nil => rfl
  | cons e l ih => rw [setAll_cons, ih, addEdge_nrows]

@[simp] theorem setAll_ncols {l : List (Pair × Nat)} : (setAll t l).ncols = t.ncols := by
  induction l generalizing t with
  | nil => rfl
  | cons e l ih => rw [setAll_cons, ih, addEdge_ncols]

/-- The step lemma of the batch induction: after inserting the head, the tail is
still a fresh, writable batch. -/
private theorem batch_step {e : Pair × Nat} {l : List (Pair × Nat)} (h : Inv t)
    (hb : WritableBatch t (e :: l)) (hf : FreshBatch t (e :: l)) :
    Inv (addEdge t e.1 e.2) ∧ WritableBatch (addEdge t e.1 e.2) l ∧
      FreshBatch (addEdge t e.1 e.2) l := by
  obtain ⟨hbe, hide⟩ := hb e (List.mem_cons_self ..)
  obtain ⟨hnd, hfresh⟩ := hf
  have hnd0 : (e.2 :: List.map Prod.snd l).Nodup := by rw [← List.map_cons]; exact hnd
  have hnd' := List.nodup_cons.mp hnd0
  refine ⟨inv_addEdge h hbe hide (hfresh e (List.mem_cons_self ..) e.1), ?_, ?_, ?_⟩
  · intro e' he'
    obtain ⟨hin, hval⟩ := hb e' (List.mem_cons_of_mem _ he')
    exact ⟨inBounds_addEdge hin, hval⟩
  · exact hnd'.2
  · intro e' he' q hq
    have hmem : e'.2 ∈ List.map Prod.snd l := List.mem_map_of_mem he'
    by_cases hqe : q = e.1
    · subst hqe
      rw [edgesAt_addEdge_self h hbe hide] at hq
      rcases Finset.mem_insert.mp hq with heq | hin
      · exact hnd'.1 (heq ▸ hmem)
      · exact hfresh e' (List.mem_cons_of_mem _ he') _ hin
    · rw [edgesAt_addEdge_ne h hbe hqe] at hq
      exact hfresh e' (List.mem_cons_of_mem _ he') _ hq

/-- **`set_all_from_slices` preserves every invariant.** -/
theorem inv_setAll {l : List (Pair × Nat)} (h : Inv t) (hb : WritableBatch t l)
    (hf : FreshBatch t l) : Inv (setAll t l) := by
  induction l generalizing t with
  | nil => simpa using h
  | cons e l ih =>
    obtain ⟨h', hb', hf'⟩ := batch_step h hb hf
    rw [setAll_cons]
    exact ih h' hb' hf'

/-- **`set_all_from_slices` adds exactly the batch's edges, at exactly the
batch's pairs.** -/
theorem edgesAt_setAll {l : List (Pair × Nat)} (h : Inv t) (hb : WritableBatch t l)
    (hf : FreshBatch t l) (q : Pair) :
    (setAll t l).edgesAt q = t.edgesAt q ∪ batchIds l q := by
  induction l generalizing t with
  | nil => simp
  | cons e l ih =>
    obtain ⟨h', hb', hf'⟩ := batch_step h hb hf
    obtain ⟨hbe, hide⟩ := hb e (List.mem_cons_self ..)
    rw [setAll_cons, ih h' hb' hf']
    by_cases hqe : q = e.1
    · subst hqe
      rw [edgesAt_addEdge_self h hbe hide, batchIds_cons_self]
      ext i
      simp only [Finset.mem_union, Finset.mem_insert]
      tauto
    · rw [edgesAt_addEdge_ne h hbe hqe, batchIds_cons_ne hqe]

end Tensor
end FalkorDB
