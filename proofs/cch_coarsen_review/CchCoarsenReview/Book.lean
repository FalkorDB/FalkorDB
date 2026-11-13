import CchCoarsenReview.CpackGB
import CchCoarsenReview.Hsplit
/-!
# `book`: capstop patches the numbering instead of numbering again

**Rust** (m1opt.patch @7df8e5887): `build_capstop`, m1_reuse_bench.rs:1245-1289. When a cell stops at a
level (it breaks the cap or its parent has no level), the switch-off path numbers the level again with
the smaller active set (`number(&p, par, Some(&active), ncells)`, :1286); under `book` it calls
`number_patch(&mut num, &p, np, &newly)` (:1281, body :736-797) on the numbering it already has
(`number_fast`, :385-528): parts whose cell stopped go and later parts move down (:743-764), clean ids
of stopped cells become holes (:765-773), and `cmap` drops the stopped cells' vertices and renumbers the
surviving parts (:774-786).

**The claim**: the patched numbering is the numbering of the smaller active set. Both follow from what
`number`/`number_fast` decide per parent id `q` (:250-275 / :400-421): its *members* (the counted
previous-level vertices mapped to `q`), whether it is *clean* (members in one cell, weights summing to
the parent's weight, `is_clean`, :277 / :424), and its *parts* (one per cell of a non-clean `q`,
:278-283 / :425-430). This file models those decisions and proves they restrict as the patch assumes:

* `members_restrict`: dropping cells `S` drops exactly the members in `S`;
* `clean_restrict`: a parent id that keeps a member is clean after the drop iff it was before
  (needs every member's weight positive and the members' total at most the parent's weight);
* `part_restrict`: `(q, c)` is a part after the drop iff it was a part and `c ∉ S` (what the patch keeps,
  in the same order, so `newid` is the renumbered id);
* `hole_restrict`: a clean id whose cell stopped has no member left (`number` leaves it a hole, as the
  patch does).

`book_zero_weight_breaks` shows the positivity premise is needed. It cannot fail in the bench (every
live weight is a sum of unit fine weights), but it is the module's obligation to keep.

**Switch dependency** (illustrated by `book_needs_pk` and `book_needs_hb2`; confirmed on the prototype for
`book`, by reading the code for `npar`): `number_fast` and `number_par` leave `ct_i/ct_j`, `cl_q/cl_c` (:513-516, :717-720) and, without stats, `ps_i/ps_j` empty,
and `number_patch` clears `ps_i/ps_j` (:796). The switch-off paths of `pk` (:946-949) and `hb2`
(:979-1010) read them. So `book` (and `npar`) are correct only together with `pk` and `hb2`. Every
measured step had them (`book` is step 6, after `hb2` at 1 and `pk` at 5), so no reported number is
affected. The run: `.handover/cch-redesign/repro/h2-switch-dependency/`.

Not modelled: the loops themselves (the sort of `(q, cell, x)` triples, the part runs, `ws` as `i32`,
`ccount`), so this is a model of the decisions, not a line-by-line model of `number_fast` and
`number_patch` (listed as a gap in the report).
-/

namespace CchCoarsen

/-- What `number`/`number_fast` read for a previous-level vertex `x`. -/
structure NumIn where
  nprev : Nat
  pcell : Nat → Nat
  /-- `par.cmap[pid(x)]`, `NONE` when the parent dropped it (:410-415). -/
  q : Nat → Nat
  pw : Nat → Int
  parw : Nat → Int

variable (I : NumIn)

/-- `x` is counted under the active cells `A` (:400-416): not a separator, active, mapped by the parent. -/
def touched (A : Nat → Bool) (x : Nat) : Prop :=
  x < I.nprev ∧ I.pcell x ≠ SEP ∧ A (I.pcell x) = true ∧ I.q x ≠ NONE

instance (A : Nat → Bool) : DecidablePred (touched I A) := fun _ => inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

/-- The members of parent id `q`. -/
def membersOf (A : Nat → Bool) (q : Nat) : List Nat :=
  (List.range I.nprev).filter (fun x => decide (touched I A x ∧ I.q x = q))

/-- `is_clean` (:277, :424): members in one cell (`f[q]` not `MULTI`), and their weight is the parent's. -/
def clean (A : Nat → Bool) (q : Nat) : Prop :=
  membersOf I A q ≠ [] ∧ (∀ x ∈ membersOf I A q, ∀ y ∈ membersOf I A q, I.pcell x = I.pcell y) ∧
    lsum (membersOf I A q) I.pw = I.parw q

/-- `(q, c)` is a part (:278-283, :425-430): `q` is not clean and has a member in cell `c`. -/
def isPart (A : Nat → Bool) (q c : Nat) : Prop :=
  (∃ x ∈ membersOf I A q, I.pcell x = c) ∧ ¬ clean I A q

/-- The active set after the stopped cells `S` leave. -/
def drop (A S : Nat → Bool) : Nat → Bool := fun c => A c && !S c

/-- The premises: every counted member has a positive weight, and a parent id's members weigh at most
the parent's weight (they are a subset of its fine vertices). -/
structure BookPre (A : Nat → Bool) : Prop where
  pos : ∀ x, touched I A x → 0 < I.pw x
  le : ∀ q, lsum (membersOf I A q) I.pw ≤ I.parw q

variable {I}

theorem mem_membersOf {A : Nat → Bool} {q x : Nat} :
    x ∈ membersOf I A q ↔ touched I A x ∧ I.q x = q := by
  simp only [membersOf, List.mem_filter, List.mem_range, decide_eq_true_eq]
  constructor
  · rintro ⟨_, h⟩; exact h
  · rintro h; exact ⟨h.1.1, h⟩

/-- Dropping cells `S` drops exactly the members in `S`. -/
theorem members_restrict (A S : Nat → Bool) (q : Nat) :
    membersOf I (drop A S) q = (membersOf I A q).filter (fun x => !S (I.pcell x)) := by
  unfold membersOf
  rw [List.filter_filter]
  apply List.filter_congr
  intro x _
  simp only [touched, drop, Bool.and_eq_true, Bool.not_eq_true']
  by_cases h1 : x < I.nprev <;> by_cases h2 : I.pcell x ≠ SEP <;> by_cases h3 : A (I.pcell x) = true <;>
    by_cases h4 : I.q x ≠ NONE <;> by_cases h5 : I.q x = q <;> cases hS : S (I.pcell x) <;> simp_all

theorem lsum_filter_lt {l : List Nat} {f : Nat → Int} (p : Nat → Bool) (hpos : ∀ x ∈ l, 0 < f x)
    (hx : ∃ x ∈ l, p x = false) : lsum (l.filter p) f < lsum l f := by
  induction l with
  | nil => obtain ⟨_, h, _⟩ := hx; simp at h
  | cons a l ih =>
    have ha := hpos a (by simp)
    by_cases hp : p a = true
    · simp only [List.filter_cons, hp, ite_true, lsum_cons]
      obtain ⟨x, hxl, hpx⟩ := hx
      rcases List.mem_cons.1 hxl with rfl | h
      · rw [hp] at hpx; exact absurd hpx (by simp)
      · have := ih (fun y hy => hpos y (List.mem_cons_of_mem _ hy)) ⟨x, h, hpx⟩; omega
    · simp only [List.filter_cons, hp, Bool.false_eq_true, ite_false, lsum_cons]
      have h1 := lsum_le (l := l.filter p) (f := f) (g := f) (fun _ _ => Int.le_refl _)
      have h2 : lsum (l.filter p) f ≤ lsum l f := by
        rw [lsum_filter]
        exact lsum_le (fun y hy => by
          by_cases h : p y = true
          · simp [h]
          · simp [h]; exact Int.le_of_lt (hpos y (List.mem_cons_of_mem _ hy)))
      omega

/-- **A parent id that keeps a member is clean after the drop iff it was clean before.** -/
theorem clean_restrict {A S : Nat → Bool} (hp : BookPre I A) {q : Nat}
    (hne : membersOf I (drop A S) q ≠ []) : clean I (drop A S) q ↔ clean I A q := by
  rw [clean, clean, members_restrict]
  rw [members_restrict] at hne
  constructor
  · rintro ⟨_, hsing, hw⟩
    -- no member was dropped: otherwise the kept weight is below the parent's
    have hall : ∀ x ∈ membersOf I A q, (!S (I.pcell x)) = true := by
      intro x hx
      by_cases h : (!S (I.pcell x)) = true
      · exact h
      · exfalso
        have hlt := lsum_filter_lt (fun x => !S (I.pcell x))
          (fun y hy => hp.pos y (mem_membersOf.1 hy).1) ⟨x, hx, by simpa using h⟩
        have := hp.le q; omega
    have heq : (membersOf I A q).filter (fun x => !S (I.pcell x)) = membersOf I A q :=
      List.filter_eq_self.2 hall
    rw [heq] at hsing hw hne
    exact ⟨hne, hsing, hw⟩
  · rintro ⟨_, hsing, hw⟩
    obtain ⟨x0, hx0⟩ := List.exists_mem_of_ne_nil _ hne
    have hx0' := (List.mem_filter.1 hx0)
    have hall : ∀ x ∈ membersOf I A q, (!S (I.pcell x)) = true := by
      intro x hx
      rw [hsing x hx x0 hx0'.1]; exact hx0'.2
    have heq : (membersOf I A q).filter (fun x => !S (I.pcell x)) = membersOf I A q :=
      List.filter_eq_self.2 hall
    rw [heq]
    exact ⟨by rw [← heq]; exact hne, hsing, hw⟩

/-- **The parts after the drop are the parts before whose cell did not stop**: what `number_patch`
keeps (:745-759), in the same order (both are the `(q, cell)` runs of one sorted list). -/
theorem part_restrict {A S : Nat → Bool} (hp : BookPre I A) (q c : Nat) :
    isPart I (drop A S) q c ↔ (isPart I A q c ∧ S c = false) := by
  unfold isPart
  constructor
  · rintro ⟨⟨x, hx, hc⟩, hncl⟩
    have hne : membersOf I (drop A S) q ≠ [] := List.ne_nil_of_mem hx
    rw [members_restrict] at hx
    have hx' := List.mem_filter.1 hx
    refine ⟨⟨⟨x, hx'.1, hc⟩, fun h => hncl ((clean_restrict hp hne).2 h)⟩, ?_⟩
    rw [← hc]; simpa using hx'.2
  · rintro ⟨⟨⟨x, hx, hc⟩, hncl⟩, hS⟩
    have hx2 : x ∈ membersOf I (drop A S) q := by
      rw [members_restrict]; exact List.mem_filter.2 ⟨hx, by rw [hc, hS]; rfl⟩
    exact ⟨⟨x, hx2, hc⟩, fun h => hncl ((clean_restrict hp (List.ne_nil_of_mem hx2)).1 h)⟩

/-- **A clean id whose cell stopped has no member left**: `number` of the smaller set leaves it a hole
(`f[q] = NONE`: cell `SEP`, weight 0, `pa = NONE`), as `number_patch` does (:765-773). -/
theorem hole_restrict {A S : Nat → Bool} {q : Nat} (hcl : clean I A q)
    (hS : ∃ x ∈ membersOf I A q, S (I.pcell x) = true) : membersOf I (drop A S) q = [] := by
  obtain ⟨x0, hx0, hs0⟩ := hS
  rw [members_restrict, List.filter_eq_nil_iff]
  intro x hx
  rw [hcl.2.1 x hx x0 hx0, hs0]; simp

/-- **The positivity premise is needed.** A parent id `0` of weight 1 with a weight-0 member in cell 0
and a weight-1 member in cell 1: with both cells active it straddles (part `(0, 1)`); when cell 0 stops,
the numbering of the smaller set finds it clean (one cell, full weight), while the patch keeps part
`(0, 1)`. -/
theorem book_zero_weight_breaks :
    let I : NumIn := ⟨2, fun x => x, fun _ => 0, fun x => if x = 0 then 0 else 1, fun _ => 1⟩
    let A : Nat → Bool := fun _ => true
    let S : Nat → Bool := fun c => c = 0
    isPart I A 0 1 ∧ S 1 = false ∧ ¬ isPart I (drop A S) 0 1 := by
  intro I A S
  have hm : membersOf I A 0 = [0, 1] := by decide
  have hm' : membersOf I (drop A S) 0 = [1] := by decide
  refine ⟨⟨⟨1, by rw [hm]; simp, rfl⟩, fun h => ?_⟩, rfl, fun ⟨_, h⟩ => h ?_⟩
  · have := h.2.1 0 (by rw [hm]; simp) 1 (by rw [hm]; simp)
    exact absurd this (by decide)
  · refine ⟨by rw [hm']; simp, fun x hx y hy => ?_, by rw [hm']; decide⟩
    rw [hm'] at hx hy; simp at hx hy; subst hx; subst hy; rfl

/-! ## The switch dependency -/

/-- *Illustration, true by construction* (not a model of `build_level`). **`book` needs `pk`**: under
`book` the numbering leaves `ct_i/ct_j` empty (:513-514), so the switch-off `Cᵀ = build(ct_i, ct_j)`
(:946-949) has no entry, while `pk`'s `Cᵀ` has one per kept vertex. Confirmed on the prototype:
`M1O=book` and `M1O=hb2,book` on PA fail "CHECK FAILED: projection lost vertices"
(m1_reuse_bench.rs:1140); `M1O=hb2,pk,book` matches 13/13 hashes. -/
theorem book_needs_pk (nprev : Nat) (cmap : Nat → Nat) {x : Nat} (hx : x < nprev) (hk : cmap x ≠ NONE) :
    buildS 1 [] x (cmap x) = none ∧ den (packCt cmap) x (cmap x) = some 1 := by
  refine ⟨rfl, ?_⟩
  rw [pk_eq_build nprev cmap x (cmap x) hx]
  have hm : (x, cmap x) ∈ ctTuples nprev cmap :=
    List.mem_map.2 ⟨x, List.mem_filter.2 ⟨List.mem_range.2 hx, by simpa using hk⟩, rfl⟩
  unfold buildS; split
  · rfl
  · next h => exact absurd hm h

/-- *Illustration, true by construction.* **`book` needs `hb2`**: with `cl_q/cl_c` empty (:515-516) the
switch-off `M` is empty, so the split `z = par.a⟨struct⟩ M ANY_PAIR Mᵀ` keeps nothing, whatever the
parent level holds. Confirmed on the prototype: `M1O=pk,book` on PA fails "CHECK FAILED: B1 != B2"
(:2344, exit 101) with 10 of 13 hashes differing; the control `hb2,pk,book` matches 13/13. -/
theorem book_needs_hb2 (ncells : Nat) (parA : Mat Int) (q q' : Nat) :
    (fun q q' => if (parA q q').isSome &&
      GBWDelta.mxmAny ncells (fun _ _ => false) (fun _ _ => false) q q' then some (1 : Int) else none) q q' = none := by
  simp [GBWDelta.mxmAny, GBWDelta.cnt]

end CchCoarsen
