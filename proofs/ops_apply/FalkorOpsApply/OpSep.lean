import FalkorOpsApply.VhjSep
/-
# Separability of the sub-plans Apply/Optional/SemiApply/OR-apply batch (closes the CONDITIONAL rows)

`Correlated`/`VhjSep` proved the batched idiom equal to the per-row reference *under* a hypothesis
on the sub-plan (`Separable`/`SepT`).  This file discharges it for the operators the real
`can_batch` guards admit (`apply.rs`, `optional.rs`), by modelling each one's
batched run on a tagged argument list `List (origin × row)` and proving separability composes
through the operator tree.

## Why `SepT` does not compose, and `DSep`

After a multi-output operator (a scan, a traverse) the origins repeat, so "a tagged batch with
distinct sequential origins" no longer describes what the next operator sees.  `DSep` tracks
origins instead of requiring them distinct: `DSep b f` says `f` is separable on every argument list
whose origins are pairwise distinct (`Nodup`).  Only the *argument* of a sub-plan needs distinct
origins; every operator above it consumes its child's output through a per-row `flatMap` that
carries each row's origin along (`BatchedResultEmitter`: "gather parent columns + origin per
result", `batched_result_emitter.rs`), so the property is stated, and proved, per operator.
The batches Apply/Optional/SemiApply/OR-apply build are `tag xs 0` (`Correlated.tag`), whose origins
are distinct (`tag_origins_nodup`), and a per-row Apply hands its sub-plan one row (`Nodup` trivially).

`b : Bool` is the *exactness* index: `Rel true` is list equality, `Rel false` is permutation
(`Rel`).  Every operator is separable exactly except the fused optional `CondTraverse`, whose
NULL-padded rows come after the matched rows of its input batch (`cond_traverse.rs`), so
a sub-plan containing one is separable up to permutation only (issue #3070's family).

## Operator inventory (`build_batch_op`, `runtime.rs`) and the combinator modelling it

NOT separable, with a counterexample theorem each:
* `Aggregate`, `CartesianProduct`, `Sort`, `Limit`, `Skip`, `Distinct`, `Union`, `Merge`, `Optional`,
  `Apply` (batched): blacklisted by Apply's guard (`applyGuard`); `Correlated.count_not_separable`,
  `limitF_not_separable`.
* anonymous-bidirectional chained `CondTraverse` (`bidir_dedup`, `cond_traverse.rs`):
  `dedupF_not_separable` (**known bug #3106**, confirmed live: see the report; also wrong at top level).
* the writing operators (`Create`, `Delete`, `Set`, `Remove`, `Commit`, `ForEach`, `CreateIndex`,
  `DropIndex`, a write `ProcedureCall`): their rows are not row-local in effect, so since #2968 (fixes
  #2738) Apply's guard blacklists them and a writing sub-plan runs per row
  (`RowWrites.applyW_write_eq_ref`).

`Optional`'s guard is wider than Apply's: it blacklists only `Aggregate` and `CartesianProduct`.  The
extra kinds it admits (`Sort`, `Limit`, `Skip`, `Distinct`, `Union`, `Merge`, `Apply`, `Optional`)
are not produced by the planner inside an optional sub-plan, whose body is `plan_match` output
(scans, traverses, `Filter`, `SemiApply`/`OrApply`, pattern-comprehension `Apply`s ending in
`Aggregate`, `ValueHashJoin`); `SemiApply`/`OrApply` have no guard at all and are fed `plan_match`
of one path pattern.  That planner-output fact is not machine-checked here (`optionalExtra_kinds`
lists the kinds); every kind the real shapes contain is covered by `Adm`.

That last sentence is the undischarged hypothesis of the CONDITIONAL rows: `guard_admits_classified` gives each
admitted kind a verdict, but bidirectional dedup has no `Adm` constructor, and nothing proves the planner's
sub-plans are `Adm` trees.
-/
namespace OpSep
open Correlated VhjSep

variable {ρ : Type}

/-! ## `Rel b`: list equality (`b = true`) or permutation (`b = false`) -/

def Rel {β : Type} : Bool → List β → List β → Prop
  | true, l, l' => l = l'
  | false, l, l' => l.Perm l'

namespace Rel
variable {β γ : Type} {b : Bool}

theorem refl (l : List β) : Rel b l l := by
  cases b
  · exact List.Perm.refl l
  · exact Eq.refl l

theorem symm {l l' : List β} (h : Rel b l l') : Rel b l' l := by
  cases b
  · exact List.Perm.symm h
  · exact Eq.symm h

theorem trans {l l' l'' : List β} (h : Rel b l l') (h' : Rel b l' l'') : Rel b l l'' := by
  cases b
  · exact List.Perm.trans h h'
  · exact Eq.trans h h'

theorem of_eq {l l' : List β} (h : l = l') : Rel b l l' := h ▸ refl l

theorem perm {l l' : List β} (h : Rel b l l') : l.Perm l' := by
  cases b
  · exact h
  · exact h ▸ List.Perm.refl _

theorem weaken {l l' : List β} (h : Rel true l l') : Rel false l l' :=
  (h : l = l') ▸ List.Perm.refl _

theorem map (f : β → γ) {l l' : List β} (h : Rel b l l') : Rel b (l.map f) (l'.map f) := by
  cases b
  · exact List.Perm.map f h
  · exact congrArg (List.map f) h

theorem filter (p : β → Bool) {l l' : List β} (h : Rel b l l') :
    Rel b (l.filter p) (l'.filter p) := by
  cases b
  · exact List.Perm.filter p h
  · exact congrArg (List.filter p) h

theorem filterMap (f : β → Option γ) {l l' : List β} (h : Rel b l l') :
    Rel b (l.filterMap f) (l'.filterMap f) := by
  cases b
  · exact List.Perm.filterMap f h
  · exact congrArg (List.filterMap f) h

theorem append {l l' t t' : List β} (h : Rel b l l') (h' : Rel b t t') :
    Rel b (l ++ t) (l' ++ t') := by
  cases b
  · exact List.Perm.append h h'
  · exact (h : l = l') ▸ (h' : t = t') ▸ rfl

theorem flatMap_l (f : β → List γ) {l l' : List β} (h : Rel b l l') :
    Rel b (l.flatMap f) (l'.flatMap f) := by
  cases b
  · exact List.Perm.flatMap_right f h
  · exact congrArg (List.flatMap f) h

theorem flatMap_r {f g : β → List γ} :
    ∀ {l : List β}, (∀ x ∈ l, Rel b (f x) (g x)) → Rel b (l.flatMap f) (l.flatMap g)
  | [], _ => refl _
  | x :: xs, h => by
    simp only [List.flatMap_cons]
    exact append (h x (by simp)) (flatMap_r (fun y hy => h y (by simp [hy])))

end Rel

/-! ## Separability on distinct-origin arguments -/

/-- The per-row reference of a sub-plan on a tagged argument list. -/
def rowsOf (f : SubPlan ρ ρ) (args : List (Nat × ρ)) : List (Nat × ρ) :=
  args.flatMap (fun p => (single f p.2).map (fun y => (p.1, y)))

/-- **Separable on distinct-origin arguments**, exactly (`b = true`) or up to permutation. -/
def DSep (b : Bool) (f : SubPlan ρ ρ) : Prop :=
  ∀ args : List (Nat × ρ), (args.map Prod.fst).Nodup → Rel b (f args) (rowsOf f args)

theorem DSep.single_at {b : Bool} {f : SubPlan ρ ρ} (h : DSep b f) (o : Nat) (x : ρ) :
    Rel b (f [(o, x)]) ((single f x).map (fun y => (o, y))) := by
  have := h [(o, x)] (by simp)
  simpa [rowsOf] using this

theorem DSep.weaken {f : SubPlan ρ ρ} (h : DSep true f) : DSep false f :=
  fun args hn => Rel.weaken (h args hn)

theorem tag_origins_nodup {α : Type} : ∀ (xs : List α) (n : Nat), ((tag xs n).map Prod.fst).Nodup
  | [], _ => by simp [tag]
  | x :: xs, n => by
    simp only [tag, List.map_cons, List.nodup_cons]
    refine ⟨?_, tag_origins_nodup xs (n + 1)⟩
    intro hmem
    obtain ⟨⟨i, y⟩, hq, hi⟩ := List.mem_map.1 hmem
    have := (mem_tag hq).1
    simp at hi; omega

theorem tag_map {α β : Type} (f : α → β) :
    ∀ (xs : List α) (n : Nat), tag (xs.map f) n = (tag xs n).map (fun q => (q.1, f q.2))
  | [], _ => rfl
  | x :: xs, n => by simp [tag, tag_map f xs (n + 1)]

/-- `DSep` of a plan only depends on it up to permutation of its results. -/
theorem dsep_congr {f f' : SubPlan ρ ρ} (hperm : ∀ a, (f a).Perm (f' a)) (h : DSep false f') :
    DSep false f := by
  intro args hn
  have h2 : (f' args).Perm (rowsOf f' args) := h args hn
  have h3 : (rowsOf f' args).Perm (rowsOf f args) := by
    have : ∀ p ∈ args, ((single f' p.2).map (fun y => (p.1, y))).Perm
        ((single f p.2).map (fun y => (p.1, y))) := by
      intro p _
      exact (((hperm [(0, p.2)]).symm).map Prod.snd).map _
    exact Rel.flatMap_r (b := false) this
  exact ((hperm args).trans h2).trans h3

/-! ## Combinators -/

/-- Row-local expansion with the parent origin carried (every `BatchedResultEmitter` operator,
`Filter`, `Project`, `PathBuilder`, `ProcedureCall`). -/
def expandF (g : ρ → List ρ) (c : SubPlan ρ ρ) : SubPlan ρ ρ :=
  fun a => (c a).flatMap (fun p => (g p.2).map (fun y => (p.1, y)))

theorem expandF_dsep (b : Bool) (g : ρ → List ρ) (c : SubPlan ρ ρ) (hc : DSep b c) :
    DSep b (expandF g c) := by
  intro args hn
  have h1 : Rel b (expandF g c args)
      ((rowsOf c args).flatMap (fun p => (g p.2).map (fun y => (p.1, y)))) :=
    Rel.flatMap_l _ (hc args hn)
  have h2 : (rowsOf c args).flatMap (fun p => (g p.2).map (fun y => (p.1, y)))
      = args.flatMap (fun p => ((single c p.2).flatMap g).map (fun y => (p.1, y))) := by
    simp only [rowsOf, List.flatMap_assoc, List.flatMap_map, List.map_flatMap]
  refine Rel.trans h1 (h2 ▸ ?_)
  refine Rel.flatMap_r (fun p _ => Rel.map _ ?_)
  have h3 := Rel.map Prod.snd (Rel.flatMap_l (fun p : Nat × ρ => (g p.2).map (fun y => (p.1, y)))
    (hc.single_at 0 p.2))
  have h4 : (((single c p.2).map (fun y => ((0 : Nat), y))).flatMap
      (fun q => (g q.2).map (fun y => (q.1, y)))).map Prod.snd = (single c p.2).flatMap g := by
    simp [List.flatMap_map, List.map_flatMap, Function.comp_def]
  rw [h4] at h3
  exact Rel.symm h3

/-- Fused optional `CondTraverse` (`cond_traverse.rs`): matched rows of the input
batch first, then ONE NULL-padded batch (`null_pad`) for the rows that matched nothing. -/
def optF (g : ρ → List ρ) (nf : ρ → ρ) (c : SubPlan ρ ρ) : SubPlan ρ ρ :=
  fun a =>
    (c a).flatMap (fun p => (g p.2).map (fun y => (p.1, y))) ++
      ((c a).filter (fun p => (g p.2).isEmpty)).map (fun p => (p.1, nf p.2))

theorem optF_perm (g : ρ → List ρ) (nf : ρ → ρ) (c : SubPlan ρ ρ) (a : List (Nat × ρ)) :
    (optF g nf c a).Perm
      (expandF (fun y => if (g y).isEmpty then [nf y] else g y) c a) := by
  unfold optF expandF
  generalize c a = cs
  induction cs with
  | nil => simp
  | cons p cs ih =>
    by_cases h : (g p.2).isEmpty = true
    · have hg : g p.2 = [] := List.isEmpty_iff.1 h
      simp only [List.flatMap_cons, List.filter_cons, hg, List.isEmpty_nil, if_true, List.map_nil,
        List.nil_append, List.map_cons, decide_true]
      exact (List.perm_middle).trans (List.Perm.cons _ ih)
    · have hg : (g p.2).isEmpty = false := by simpa using h
      simp only [List.flatMap_cons, List.filter_cons, hg, Bool.false_eq_true, ite_false,
        List.append_assoc, decide_false]
      exact List.Perm.append_left _ ih

theorem optF_dsep (g : ρ → List ρ) (nf : ρ → ρ) (c : SubPlan ρ ρ) (hc : DSep false c) :
    DSep false (optF g nf c) :=
  dsep_congr (optF_perm g nf c) (expandF_dsep false _ c hc)

/-- `SemiApplyOp::next` as a sub-plan step (`semi_apply.rs`): the sub-plan runs once over
the child's rows tagged `0..n` (`clone_active_rows_seq_origin`); a child row passes iff its
position has a result, xor `is_anti`; the passing rows keep their OWN (outer) origins. -/
def semiF (anti : Bool) (c sub : SubPlan ρ ρ) : SubPlan ρ ρ :=
  fun a =>
    ((tag (c a) 0).filter (fun q =>
        ((sub (tag ((c a).map Prod.snd) 0)).map Prod.fst).contains q.1 ^^ anti)).map Prod.snd

/-- With a (permutation-)separable sub-plan, matched origin `i` iff row `i` alone has results. -/
theorem matched_iff_P {sub : SubPlan ρ ρ} (hs : DSep false sub) (ys : List ρ) (i : Nat) (x : ρ)
    (h : (i, x) ∈ tag ys 0) :
    ((sub (tag ys 0)).map Prod.fst).contains i = !(single sub x).isEmpty := by
  have hp := hs (tag ys 0) (tag_origins_nodup ys 0)
  simp only [Rel, Bool.false_eq_true, if_false] at hp
  have hc : ((sub (tag ys 0)).map Prod.fst).contains i
      = ((rowsOf sub (tag ys 0)).map Prod.fst).contains i := by
    apply Bool.eq_iff_iff.mpr
    simp only [List.contains_iff_mem]
    exact (hp.map Prod.fst).mem_iff
  rw [hc]
  exact contains_iff_generic (single sub) (tag ys 0) i x h (fun x' h' => tag_unique h' h)

theorem filter_eq_flatMap (Q : ρ → Bool) :
    ∀ cs : List (Nat × ρ), cs.filter (fun p => Q p.2)
      = cs.flatMap (fun p => ((if Q p.2 then [p.2] else []).map (fun y => (p.1, y))))
  | [] => rfl
  | p :: cs => by
    by_cases h : Q p.2 = true <;> simp [List.filter_cons, h, filter_eq_flatMap Q cs]

theorem semiF_eq (anti : Bool) (c sub : SubPlan ρ ρ) (hs : DSep false sub) :
    semiF anti c sub
      = expandF (fun y => if (!(single sub y).isEmpty ^^ anti) then [y] else []) c := by
  funext a
  unfold semiF expandF
  have key := filter_tag_eq (α := Nat × ρ)
    (fun i => ((sub (tag ((c a).map Prod.snd) 0)).map Prod.fst).contains i ^^ anti)
    (fun p => !(single sub p.2).isEmpty ^^ anti) (c a) 0 (by
      intro i p hp
      have hm : (i, p.2) ∈ tag ((c a).map Prod.snd) 0 := by
        rw [tag_map]; exact List.mem_map.2 ⟨(i, p), hp, rfl⟩
      simp only [matched_iff_P hs _ i p.2 hm])
  rw [key]
  exact filter_eq_flatMap (fun y => !(single sub y).isEmpty ^^ anti) (c a)

theorem semiF_dsep (b : Bool) (anti : Bool) (c sub : SubPlan ρ ρ) (hc : DSep b c)
    (hs : DSep false sub) : DSep b (semiF anti c sub) := by
  rw [semiF_eq anti c sub hs]; exact expandF_dsep b _ c hc

/-- `OrApplyMultiplexerOp::next` (`or_apply_multiplexer.rs`): every branch runs over a fresh
tagged copy of the child's rows; a row passes iff some branch has `matched ^^ anti_flag`. -/
def orF (c : SubPlan ρ ρ) (brs : List (SubPlan ρ ρ × Bool)) : SubPlan ρ ρ :=
  fun a =>
    ((tag (c a) 0).filter (fun q => brs.any (fun b =>
        ((b.1 (tag ((c a).map Prod.snd) 0)).map Prod.fst).contains q.1 ^^ b.2))).map Prod.snd

theorem orF_eq (c : SubPlan ρ ρ) (brs : List (SubPlan ρ ρ × Bool))
    (hs : ∀ b ∈ brs, DSep false b.1) :
    orF c brs = expandF (fun y => if brs.any (fun b => !(single b.1 y).isEmpty ^^ b.2) then [y] else []) c := by
  funext a
  unfold orF expandF
  have key := filter_tag_eq (α := Nat × ρ)
    (fun i => brs.any (fun b => ((b.1 (tag ((c a).map Prod.snd) 0)).map Prod.fst).contains i ^^ b.2))
    (fun p => brs.any (fun b => !(single b.1 p.2).isEmpty ^^ b.2)) (c a) 0 (by
      intro i p hp
      have hm : (i, p.2) ∈ tag ((c a).map Prod.snd) 0 := by
        rw [tag_map]; exact List.mem_map.2 ⟨(i, p), hp, rfl⟩
      exact any_congr_mem brs (fun b hb => by simp only [matched_iff_P (hs b hb) _ i p.2 hm]))
  rw [key]
  exact filter_eq_flatMap (fun y => brs.any (fun b => !(single b.1 y).isEmpty ^^ b.2)) (c a)

theorem orF_dsep (b : Bool) (c : SubPlan ρ ρ) (brs : List (SubPlan ρ ρ × Bool)) (hc : DSep b c)
    (hs : ∀ b ∈ brs, DSep false b.1) : DSep b (orF c brs) := by
  rw [orF_eq c brs hs]; exact expandF_dsep b _ c hc

/-- A nested `Apply` in per-row mode (`next_per_row`, `apply.rs`): each row of the child
runs a fresh one-row sub-plan, results merged into that row, which keeps its outer origin. -/
def perRowF (mg : ρ → ρ → ρ) (c sub : SubPlan ρ ρ) : SubPlan ρ ρ :=
  fun a => (c a).flatMap (fun p => (single sub p.2).map (fun y => (p.1, mg p.2 y)))

theorem perRowF_eq (mg : ρ → ρ → ρ) (c sub : SubPlan ρ ρ) :
    perRowF mg c sub = expandF (fun x => (single sub x).map (mg x)) c := by
  funext a
  simp [perRowF, expandF, List.map_map, Function.comp_def]

/-- Per-row Apply is separable for EVERY sub-plan (no hypothesis on `sub`). -/
theorem perRowF_dsep (b : Bool) (mg : ρ → ρ → ρ) (c sub : SubPlan ρ ρ) (hc : DSep b c) :
    DSep b (perRowF mg c sub) := by
  rw [perRowF_eq]; exact expandF_dsep b _ c hc

/-- The per-origin `ValueHashJoin` (#3035, `value_hash_join.rs`): a left row joins the
right rows carrying the same origin whose keys match (`m`); the output keeps the left origin. -/
def joinO (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (L R : List (Nat × ρ)) : List (Nat × ρ) :=
  L.flatMap (fun lp => (R.filter (fun rp => rp.1 == lp.1 && m lp.2 rp.2)).map
    (fun rp => (lp.1, mg lp.2 rp.2)))

def vhjF (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (l r : SubPlan ρ ρ) : SubPlan ρ ρ :=
  fun a => joinO m mg (l a) (r a)

theorem joinO_rel (b : Bool) (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) {L L' R R' : List (Nat × ρ)}
    (hL : Rel b L L') (hR : Rel b R R') : Rel b (joinO m mg L R) (joinO m mg L' R') := by
  unfold joinO
  refine Rel.trans (Rel.flatMap_l _ hL) (Rel.flatMap_r (fun lp _ => ?_))
  exact Rel.map _ (Rel.filter _ hR)

/-- One row's join: its left rows against its right rows. -/
def rowJ (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (l r : SubPlan ρ ρ) (x : ρ) : List ρ :=
  (single l x).flatMap (fun a => ((single r x).filter (m a)).map (mg a))

theorem joinO_const (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (o : Nat) (A B : List ρ) :
    joinO m mg (A.map (fun y => (o, y))) (B.map (fun y => (o, y)))
      = (A.flatMap (fun a => (B.filter (m a)).map (mg a))).map (fun y => (o, y)) := by
  simp [joinO, List.flatMap_map, List.filter_map, List.map_map, Function.comp_def,
    List.map_flatMap]

theorem mem_rowsOf_origin (f : SubPlan ρ ρ) (ps : List (Nat × ρ)) (q : Nat × ρ)
    (h : q ∈ rowsOf f ps) : q.1 ∈ ps.map Prod.fst := by
  simp only [rowsOf, List.mem_flatMap, List.mem_map] at h ⊢
  obtain ⟨p, hp, y, _, rfl⟩ := h
  exact ⟨p, hp, rfl⟩

theorem joinO_left_only (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (o : Nat)
    (Ls Rs Rps : List (Nat × ρ)) (hL : ∀ lp ∈ Ls, lp.1 = o) (hR : ∀ rp ∈ Rps, rp.1 ≠ o) :
    joinO m mg Ls (Rs ++ Rps) = joinO m mg Ls Rs := by
  unfold joinO
  refine Rel.flatMap_r (b := true) (fun lp hlp => Rel.of_eq ?_)
  rw [List.filter_append]
  have : Rps.filter (fun rp => rp.1 == lp.1 && m lp.2 rp.2) = [] := by
    rw [List.filter_eq_nil_iff]
    intro rp hrp
    have h1 := hR rp hrp
    have h2 := hL lp hlp
    simp only [Bool.and_eq_true, beq_iff_eq, not_and]
    intro h; exact absurd (h.trans h2) h1
  rw [this, List.append_nil]

theorem joinO_right_skip (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (o : Nat)
    (Lps Rs Rps : List (Nat × ρ)) (hL : ∀ lp ∈ Lps, lp.1 ≠ o) (hR : ∀ rp ∈ Rs, rp.1 = o) :
    joinO m mg Lps (Rs ++ Rps) = joinO m mg Lps Rps := by
  unfold joinO
  refine Rel.flatMap_r (b := true) (fun lp hlp => Rel.of_eq ?_)
  rw [List.filter_append]
  have : Rs.filter (fun rp => rp.1 == lp.1 && m lp.2 rp.2) = [] := by
    rw [List.filter_eq_nil_iff]
    intro rp hrp
    have h1 := hL lp hlp
    have h2 := hR rp hrp
    simp only [Bool.and_eq_true, beq_iff_eq, not_and]
    intro h; exact absurd (h.symm.trans h2) h1
  rw [this, List.nil_append]

theorem joinO_append_l (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (L L' R : List (Nat × ρ)) :
    joinO m mg (L ++ L') R = joinO m mg L R ++ joinO m mg L' R := by
  unfold joinO; rw [List.flatMap_append]

theorem joinO_rows (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (l r : SubPlan ρ ρ) :
    ∀ args : List (Nat × ρ), (args.map Prod.fst).Nodup →
      joinO m mg (rowsOf l args) (rowsOf r args)
        = args.flatMap (fun p => (rowJ m mg l r p.2).map (fun y => (p.1, y)))
  | [], _ => by simp [joinO, rowsOf]
  | (o, x) :: ps, hn => by
    simp only [List.map_cons, List.nodup_cons] at hn
    obtain ⟨hno, hnd⟩ := hn
    have ih := joinO_rows m mg l r ps hnd
    have hL : rowsOf l ((o, x) :: ps) = (single l x).map (fun y => (o, y)) ++ rowsOf l ps := by
      simp [rowsOf]
    have hR : rowsOf r ((o, x) :: ps) = (single r x).map (fun y => (o, y)) ++ rowsOf r ps := by
      simp [rowsOf]
    rw [hL, hR, joinO_append_l]
    have hLs : ∀ lp ∈ (single l x).map (fun y => (o, y)), lp.1 = o := by
      intro lp hlp; obtain ⟨y, _, rfl⟩ := List.mem_map.1 hlp; rfl
    have hRs : ∀ rp ∈ (single r x).map (fun y => (o, y)), rp.1 = o := by
      intro rp hrp; obtain ⟨y, _, rfl⟩ := List.mem_map.1 hrp; rfl
    have hRps : ∀ rp ∈ rowsOf r ps, rp.1 ≠ o := by
      intro rp hrp h
      exact hno (h ▸ mem_rowsOf_origin r ps rp hrp)
    have hLps : ∀ lp ∈ rowsOf l ps, lp.1 ≠ o := by
      intro lp hlp h
      exact hno (h ▸ mem_rowsOf_origin l ps lp hlp)
    rw [joinO_left_only m mg o _ _ _ hLs hRps, joinO_right_skip m mg o _ _ _ hLps hRs, ih,
      joinO_const]
    simp [rowJ]

theorem vhjF_dsep (b : Bool) (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) (l r : SubPlan ρ ρ)
    (hl : DSep b l) (hr : DSep b r) : DSep b (vhjF m mg l r) := by
  intro args hn
  have h1 : Rel b (vhjF m mg l r args) (joinO m mg (rowsOf l args) (rowsOf r args)) :=
    joinO_rel b m mg (hl args hn) (hr args hn)
  rw [joinO_rows m mg l r args hn] at h1
  refine Rel.trans h1 (Rel.flatMap_r (fun p _ => Rel.map _ ?_))
  have h3 := Rel.map Prod.snd (joinO_rel b m mg (hl.single_at 0 p.2) (hr.single_at 0 p.2))
  rw [joinO_const] at h3
  have h4 : ((((single l p.2).flatMap (fun a => ((single r p.2).filter (m a)).map (mg a))).map
      (fun y => ((0 : Nat), y)))).map Prod.snd = rowJ m mg l r p.2 := by
    simp [rowJ, List.map_map, Function.comp_def]
  rw [h4] at h3
  exact Rel.symm h3

/-! ## The admitted class -/

/-- The operator trees Apply/Optional/SemiApply/OR-apply may batch, one constructor per modelled
operator (see the inventory in the header).  Index: exactness (`true` = no fused optional
traverse inside, so the batched run is the per-row run exactly; `false` = up to permutation).
Premises on `SemiApply`/`OrApply` branches ask only `Adm false` (only the matched set is used). -/
inductive Adm : Bool → SubPlan ρ ρ → Prop
  | argument (b : Bool) : Adm b (fun a => a)
  | expand (b : Bool) (g : ρ → List ρ) {c : SubPlan ρ ρ} : Adm b c → Adm b (expandF g c)
  | optExpand (g : ρ → List ρ) (nf : ρ → ρ) {c : SubPlan ρ ρ} :
      Adm false c → Adm false (optF g nf c)
  | semi (b : Bool) (anti : Bool) {c sub : SubPlan ρ ρ} :
      Adm b c → Adm false sub → Adm b (semiF anti c sub)
  | orApply (b : Bool) {c : SubPlan ρ ρ} {brs : List (SubPlan ρ ρ × Bool)} :
      Adm b c → (∀ br ∈ brs, Adm false br.1) → Adm b (orF c brs)
  | vhj (b : Bool) (m : ρ → ρ → Bool) (mg : ρ → ρ → ρ) {l r : SubPlan ρ ρ} :
      Adm b l → Adm b r → Adm b (vhjF m mg l r)
  | perRow (b : Bool) (mg : ρ → ρ → ρ) {c : SubPlan ρ ρ} (sub : SubPlan ρ ρ) :
      Adm b c → Adm b (perRowF mg c sub)

theorem Adm.mono {b : Bool} {f : SubPlan ρ ρ} (h : Adm b f) : Adm false f := by
  induction h with
  | argument => exact .argument _
  | expand _ g _ ih => exact .expand _ g ih
  | optExpand g nf h _ => exact .optExpand g nf h
  | semi _ anti _ hs ih => exact .semi _ anti ih hs
  | orApply _ _ hs ih => exact .orApply _ ih hs
  | vhj _ m mg _ _ ihl ihr => exact .vhj _ m mg ihl ihr
  | perRow _ mg sub _ ih => exact .perRow _ mg sub ih

/-- **Every admitted operator tree is separable on distinct-origin arguments.** -/
theorem Adm.dsep {b : Bool} {f : SubPlan ρ ρ} (h : Adm b f) : DSep b f := by
  induction h with
  | argument b =>
    intro args _
    apply Rel.of_eq
    unfold rowsOf single
    simp only [List.map_cons, List.map_nil]
    induction args with
    | nil => rfl
    | cons p ps ih => simp [ih]
  | expand b g _ ih => exact expandF_dsep b g _ ih
  | optExpand g nf _ ih => exact optF_dsep g nf _ ih
  | semi b anti _ hs ih ihs => exact semiF_dsep b anti _ _ ih ihs
  | orApply b _ hs ih ihs => exact orF_dsep b _ _ ih ihs
  | vhj b m mg _ _ ihl ihr => exact vhjF_dsep b m mg _ _ ihl ihr
  | perRow b mg sub _ ih => exact perRowF_dsep b mg _ sub ih


/-! ## The six CONDITIONAL rows, instantiated -/

variable {γ : Type}

/-- **`ApplyOp::next_batched`** (`apply.rs`): for every admitted sub-plan, batched Apply
returns the per-row reference's rows (same multiset; in the same order when the plan has no fused
optional traverse). -/
theorem applyBatched_rel {b : Bool} (merge : ρ → ρ → γ) {f : SubPlan ρ ρ} (hf : Adm b f)
    (xs : List ρ) : Rel b (applyBatched merge f xs) (applyRef merge f xs) := by
  have h := hf.dsep (tag xs 0) (tag_origins_nodup xs 0)
  unfold applyBatched
  have e : (rowsOf f (tag xs 0)).filterMap (fun p => xs[p.1]?.map (fun x => merge x p.2))
      = applyRef merge f xs := by
    unfold rowsOf applyRef
    have := lookup_tagged merge (single f) [] xs
    simpa using this
  exact Rel.trans (Rel.filterMap _ h) (Rel.of_eq e)

theorem applyBatched_perm_ref_adm {b : Bool} (merge : ρ → ρ → γ) {f : SubPlan ρ ρ} (hf : Adm b f)
    (xs : List ρ) : (applyBatched merge f xs).Perm (applyRef merge f xs) :=
  (applyBatched_rel merge hf xs).perm

/-- Without a fused optional traverse the batched run is the reference EXACTLY. -/
theorem applyBatched_eq_ref_adm (merge : ρ → ρ → γ) {f : SubPlan ρ ρ} (hf : Adm true f)
    (xs : List ρ) : applyBatched merge f xs = applyRef merge f xs :=
  applyBatched_rel merge hf xs

/-- `ApplyOp::next` (`apply.rs`): the `can_batch` dispatch. -/
def applyNext (cb : Bool) (merge : ρ → ρ → γ) (f : SubPlan ρ ρ) (xs : List ρ) : List γ :=
  if cb then applyBatched merge f xs else applyPerRow merge f xs

/-- **`ApplyOp::next`**: whichever branch `can_batch` picks, the rows are the reference's (the
per-row branch needs nothing; the batched branch needs the sub-plan to be admitted). -/
theorem applyNext_perm_ref {b : Bool} (cb : Bool) (merge : ρ → ρ → γ) (f : SubPlan ρ ρ)
    (hadm : cb = true → Adm b f) (xs : List ρ) :
    (applyNext cb merge f xs).Perm (applyRef merge f xs) := by
  unfold applyNext
  cases cb
  · exact (applyPerRow_eq_ref merge f xs) ▸ List.Perm.refl _
  · exact applyBatched_perm_ref_adm merge (hadm rfl) xs

theorem applyNext_eq_ref (cb : Bool) (merge : ρ → ρ → γ) (f : SubPlan ρ ρ)
    (hadm : cb = true → Adm true f) (xs : List ρ) :
    applyNext cb merge f xs = applyRef merge f xs := by
  unfold applyNext
  cases cb
  · exact applyPerRow_eq_ref merge f xs
  · exact applyBatched_eq_ref_adm merge (hadm rfl) xs

/-- Under a permutation-separable sub-plan the NULL-fallback rows are the per-row fallbacks. -/
theorem fallback_eq_P (nullFill : ρ → γ) (f : SubPlan ρ ρ) (hf : DSep false f) (xs : List ρ) :
    ((tag xs 0).filter (fun p => !(matchedOrigins f xs).contains p.1)).map (fun p => nullFill p.2)
      = xs.flatMap (fun x => if single f x = [] then [nullFill x] else []) := by
  have key : ∀ (ys : List ρ) (n : Nat), (∀ i x, (i, x) ∈ tag ys n → (i, x) ∈ tag xs 0) →
      ((tag ys n).filter (fun p => !(matchedOrigins f xs).contains p.1)).map
          (fun p => nullFill p.2)
        = ys.flatMap (fun x => if single f x = [] then [nullFill x] else []) := by
    intro ys
    induction ys with
    | nil => intro n _; simp [tag]
    | cons y ys ih =>
      intro n hsub
      simp only [tag, List.filter_cons, List.flatMap_cons]
      have hy : (matchedOrigins f xs).contains n = !(single f y).isEmpty :=
        matched_iff_P hf xs n y (hsub n y (by simp [tag]))
      have ih' := ih (n + 1) (fun i x h => hsub i x (by simp [tag, h]))
      rw [hy, ← ih']
      cases hs : single f y <;> simp
  exact key xs 0 (fun _ _ h => h)

theorem optAux (merge : ρ → ρ → γ) (nullFill : ρ → γ) (f : SubPlan ρ ρ) :
    ∀ xs : List ρ,
      (xs.flatMap (fun x => (single f x).map (merge x)) ++
        xs.flatMap (fun x => if single f x = [] then [nullFill x] else [])).Perm
      (xs.flatMap (fun x => if single f x = [] then [nullFill x] else (single f x).map (merge x)))
  | [] => by simp
  | x :: xs => by
    have ih := optAux merge nullFill f xs
    simp only [List.flatMap_cons]
    by_cases h : single f x = []
    · simp only [h, List.map_nil, List.nil_append, ite_true]
      exact (List.perm_middle).trans (List.Perm.cons _ ih)
    · simp only [h, ite_false, List.nil_append, List.append_assoc]
      exact List.Perm.append_left _ ih

/-- **`OptionalOp::next_batched`** (`optional.rs`; the Optional arm of `apply.rs`):
a permutation of the per-row reference for every admitted sub-plan. -/
theorem optionalBatched_perm_ref_adm {b : Bool} (merge : ρ → ρ → γ) (nullFill : ρ → γ)
    {f : SubPlan ρ ρ} (hf : Adm b f) (xs : List ρ) :
    (optionalBatched merge nullFill f xs).Perm (optionalRef merge nullFill f xs) := by
  unfold optionalBatched
  rw [fallback_eq_P nullFill f hf.mono.dsep]
  have hA := applyBatched_perm_ref_adm merge hf xs
  refine (List.Perm.append_right _ hA).trans ?_
  unfold applyRef optionalRef
  exact optAux merge nullFill f xs

/-- `OptionalOp::next` (`optional.rs`): the `can_batch` dispatch. -/
def optionalNext (cb : Bool) (merge : ρ → ρ → γ) (nullFill : ρ → γ) (f : SubPlan ρ ρ)
    (xs : List ρ) : List γ :=
  if cb then optionalBatched merge nullFill f xs else optionalPerRow merge nullFill f xs

/-- **`OptionalOp::next`**. -/
theorem optionalNext_perm_ref {b : Bool} (cb : Bool) (merge : ρ → ρ → γ) (nullFill : ρ → γ)
    (f : SubPlan ρ ρ) (hadm : cb = true → Adm b f) (xs : List ρ) :
    (optionalNext cb merge nullFill f xs).Perm (optionalRef merge nullFill f xs) := by
  unfold optionalNext
  cases cb
  · exact (optionalPerRow_eq_ref merge nullFill f xs) ▸ List.Perm.refl _
  · exact optionalBatched_perm_ref_adm merge nullFill (hadm rfl) xs

/-- `ApplyOp::next` in both of its modes (`apply.rs`): `opt = none` is a plain Apply (`optional_vars` is
`None`), `opt = some nullFill` is the Optional arm (`optional_vars` is `Some`: the sub-plan sits under an
`IR::Optional`, and an input row with no sub-plan result is emitted once with the optional variables
NULL, `nullFill`). The `can_batch` guard is Apply's in both modes. -/
def applyOpNext (opt : Option (ρ → γ)) (cb : Bool) (merge : ρ → ρ → γ) (f : SubPlan ρ ρ)
    (xs : List ρ) : List γ :=
  match opt with
  | none => applyNext cb merge f xs
  | some nullFill => optionalNext cb merge nullFill f xs

/-- The per-row reference for either mode: plain Apply or OPTIONAL MATCH. -/
def applyOpRef (opt : Option (ρ → γ)) (merge : ρ → ρ → γ) (f : SubPlan ρ ρ) (xs : List ρ) : List γ :=
  match opt with
  | none => applyRef merge f xs
  | some nullFill => optionalRef merge nullFill f xs

/-- **`ApplyOp::next`, both modes**: whichever mode `optional_vars` selects and whichever branch
`can_batch` picks, the rows are a permutation of the per-row reference (the per-row branch needs nothing; the
batched branch needs the admitted sub-plan to be `Adm`, a premise about the planner's output: see the header). -/
theorem applyOpNext_perm_ref {b : Bool} (opt : Option (ρ → γ)) (cb : Bool) (merge : ρ → ρ → γ)
    (f : SubPlan ρ ρ) (hadm : cb = true → Adm b f) (xs : List ρ) :
    (applyOpNext opt cb merge f xs).Perm (applyOpRef opt merge f xs) := by
  cases opt with
  | none => exact applyNext_perm_ref cb merge f hadm xs
  | some nullFill => exact optionalNext_perm_ref cb merge nullFill f hadm xs

/-- **The Optional arm keeps an unmatched input row**: if the sub-plan returns nothing for every input row
(`MATCH (a:A) OPTIONAL MATCH (a)-[:R]->(b)` with no `R` edge), `ApplyOp::next` in Optional mode emits each
input row once with the optional variables NULL (`nullFill`), in either branch. -/
theorem applyOpNext_optional_unmatched {b : Bool} (nullFill : ρ → γ) (cb : Bool) (merge : ρ → ρ → γ)
    (f : SubPlan ρ ρ) (hadm : cb = true → Adm b f) (xs : List ρ) (hnone : ∀ x ∈ xs, single f x = []) :
    (applyOpNext (some nullFill) cb merge f xs).Perm (xs.map nullFill) := by
  refine (applyOpNext_perm_ref (some nullFill) cb merge f hadm xs).trans ?_
  have : optionalRef merge nullFill f xs = xs.map nullFill := by
    unfold optionalRef
    induction xs with
    | nil => rfl
    | cons x xs ih =>
      simp only [List.flatMap_cons, List.map_cons, hnone x (by simp), ite_true]
      rw [ih (fun y hy => hnone y (by simp [hy]))]
      rfl
  simp only [applyOpRef, this]
  exact List.Perm.refl _

/-- **`SemiApplyOp::next`** (`semi_apply.rs`): (anti-)semi-apply over an admitted
sub-plan is exactly the per-row EXISTS filter. -/
theorem semiBatched_eq_ref_adm {b : Bool} (isAnti : Bool) {f : SubPlan ρ ρ} (hf : Adm b f)
    (xs : List ρ) : semiBatched isAnti f xs = semiRef isAnti f xs := by
  unfold semiBatched semiRef
  exact filter_tag_eq (fun i => (matchedOrigins f xs).contains i ^^ isAnti) _ xs 0
    (fun i x h => by simp only [matchedOrigins, matched_iff_P hf.mono.dsep xs i x h])

/-- **`OrApplyMultiplexerOp::next`** (`or_apply_multiplexer.rs`): over admitted branches,
exactly the per-row disjunction. -/
theorem orApply_eq_ref_adm (branches : List (SubPlan ρ ρ × Bool))
    (hs : ∀ b ∈ branches, ∃ e, Adm e b.1) (xs : List ρ) :
    orApply branches xs = orRef branches xs := by
  unfold orApply orRef
  exact filter_tag_eq (fun i => branches.any (fun b => (matchedOrigins b.1 xs).contains i ^^ b.2))
    _ xs 0 (fun i x h => any_congr_mem branches (fun b hb => by
      obtain ⟨e, he⟩ := hs b hb
      simp only [matchedOrigins, matched_iff_P he.mono.dsep xs i x h]))

/-! ## The guards and the operator inventory -/

/-- The IR variants of `planner/mod.rs`, with the fused optional and anonymous-bidirectional
`CondTraverse` split out (they behave differently). -/
inductive IRv
  | argument | optional | procedureCall | unwind | create | merge | delete | set | remove
  | allNodeScan | nodeByLabelScan | includePending | nodeByIndexScan | edgeByIndexScan
  | nodeByFulltextScan | edgeByFulltextScan | nodeByVectorScan | edgeByVectorScan
  | nodeByLabelAndIdScan | nodeByIdSeek | condTraverse | condTraverseOptional
  | condTraverseBidirDedup | condVarLenTraverse | allShortestPaths | expandInto | pathBuilder
  | filter | cartesianProduct | valueHashJoin | apply | semiApply | antiSemiApply
  | orApplyMultiplexer | loadCsv | sort | skip | limit | aggregate | project | distinct
  | nestedPlans | union | commit | forEach | createIndex | dropIndex
  /-- `IR::ProcedureCall { func, .. }` with `func.write` (`db.idx.fulltext.createNodeIndex`). -/
  | procedureCallWrite
  deriving DecidableEq, Repr

/-- `ApplyOp::new`'s blacklist (`apply.rs`): every writing operator since #2968, and the
non-separable read operators. -/
def applyBlacklisted : IRv → Bool
  | .procedureCallWrite | .create | .set | .remove | .delete | .forEach | .createIndex
  | .dropIndex | .commit => true
  | .aggregate | .cartesianProduct | .optional | .apply | .merge | .union | .sort | .limit
  | .skip | .distinct => true
  | _ => false

/-- `OptionalOp::new`'s blacklist (`optional.rs`). -/
def optionalBlacklisted : IRv → Bool
  | .aggregate | .cartesianProduct => true
  | _ => false

inductive Verdict
  | exact          -- `Adm true` combinator
  | permOnly       -- `Adm false` only (fused optional traverse)
  | bug (issue : Nat)
  | guarded        -- on Apply's blacklist
  | mergeOnly      -- appears only under `Merge` (`planner/mod.rs`)
  deriving DecidableEq, Repr

def verdict : IRv → Verdict
  | .argument | .procedureCall | .unwind | .allNodeScan | .nodeByLabelScan | .nodeByIndexScan
  | .edgeByIndexScan | .nodeByFulltextScan | .edgeByFulltextScan | .nodeByVectorScan
  | .edgeByVectorScan | .nodeByLabelAndIdScan | .nodeByIdSeek | .condTraverse
  | .condVarLenTraverse | .allShortestPaths | .expandInto | .pathBuilder | .filter
  | .valueHashJoin | .semiApply | .antiSemiApply | .orApplyMultiplexer | .loadCsv | .project
  | .nestedPlans => .exact
  | .condTraverseOptional => .permOnly
  | .condTraverseBidirDedup => .bug 3106
  | .includePending => .mergeOnly
  | .create | .delete | .set | .remove | .commit | .forEach | .createIndex | .dropIndex
  | .procedureCallWrite => .guarded
  | .aggregate | .cartesianProduct | .optional | .apply | .merge | .union | .sort | .limit
  | .skip | .distinct => .guarded

/-- Every kind Apply's guard admits has a verdict other than `guarded`, and the kinds it blocks
are exactly the `guarded` ones: the inventory above is complete for the guard. -/
theorem applyGuard_iff_guarded (k : IRv) : applyBlacklisted k = true ↔ verdict k = .guarded := by
  cases k <;> decide

/-- The kinds `OptionalOp` admits beyond Apply's guard (the planner never puts them in an
optional sub-plan, whose body is `plan_match` output: no clause there writes; see the header). -/
theorem optionalExtra_kinds (k : IRv) (h1 : optionalBlacklisted k = false)
    (h2 : applyBlacklisted k = true) :
    k = .optional ∨ k = .apply ∨ k = .merge ∨ k = .union ∨ k = .sort ∨ k = .limit ∨ k = .skip
      ∨ k = .distinct ∨ k = .create ∨ k = .delete ∨ k = .set ∨ k = .remove ∨ k = .commit
      ∨ k = .forEach ∨ k = .createIndex ∨ k = .dropIndex ∨ k = .procedureCallWrite := by
  cases k <;> simp_all [optionalBlacklisted, applyBlacklisted]

/-- Every kind the guard admits is covered: modelled exactly, up to permutation, or classified
as a confirmed bug with its issue number. Since #2968 no writing operator is admitted (no
`bug 2738` verdict is left). -/
theorem guard_admits_classified (k : IRv) (h : applyBlacklisted k = false) :
    verdict k = .exact ∨ verdict k = .permOnly ∨ verdict k = .bug 3106
      ∨ verdict k = .mergeOnly := by
  cases k <;> simp_all [applyBlacklisted, verdict]

/-! ## What is NOT separable (counterexamples) -/

/-- The anonymous-bidirectional chained `CondTraverse` (`cond_traverse.rs`),
KNOWN BUG #3106: a `(source, destination)` pair set shared by every row of the input batch
(cleared only per batch), keyed by the source column value, never by origin. -/
def dedupRow (src : Nat) (kd : ρ → Nat) (seen : List (Nat × Nat)) :
    List ρ → List ρ × List (Nat × Nat)
  | [] => ([], seen)
  | y :: ys =>
    if (src, kd y) ∈ seen then dedupRow src kd seen ys
    else
      let r := dedupRow src kd ((src, kd y) :: seen) ys
      (y :: r.1, r.2)

def dedupGo (ks : ρ → Nat) (kd : ρ → Nat) (g : ρ → List ρ) :
    List (Nat × Nat) → List (Nat × ρ) → List (Nat × ρ)
  | _, [] => []
  | seen, p :: ps =>
    let r := dedupRow (ks p.2) kd seen (g p.2)
    r.1.map (fun y => (p.1, y)) ++ dedupGo ks kd g r.2 ps

def dedupF (ks : ρ → Nat) (kd : ρ → Nat) (g : ρ → List ρ) (c : SubPlan ρ ρ) : SubPlan ρ ρ :=
  fun a => dedupGo ks kd g [] (c a)

/-- Two outer rows with the same scan source: the second row's results are all deduplicated away.
Live (graph `CREATE (:P)-[:R]->(:Q)-[:R]->(:S)`):
`MATCH (a:P) UNWIND [1,2] AS x MATCH (a)--()--(b) RETURN x, labels(b)` → Rust 2 rows (both `x=1`),
C 4; `CALL { WITH a MATCH (a)--()--(b) RETURN b }` and `OPTIONAL MATCH` forms likewise. -/
theorem dedupF_not_separable :
    ¬ DSep false (dedupF (fun x => x) (fun y => y) (fun _ => [7, 8]) (fun a => a) : SubPlan Nat Nat) := by
  intro h
  have := (Rel.perm (h (tag [1, 1] 0) (tag_origins_nodup [1, 1] 0))).length_eq
  revert this; decide

/-- `Limit` (blacklisted by Apply, admitted by `OptionalOp`'s guard): one cap over all rows. -/
def limitF (n : Nat) (c : SubPlan ρ ρ) : SubPlan ρ ρ := fun a => (c a).take n

theorem limitF_not_separable :
    ¬ DSep false (limitF 1 (fun a => a) : SubPlan Nat Nat) := by
  intro h
  have := (Rel.perm (h (tag [1, 2] 0) (tag_origins_nodup [1, 2] 0))).length_eq
  revert this; decide

/-- The fused optional traverse reorders: the NULL-padded row of the FIRST outer row comes after
the matched row of the second.  Live (`CREATE (:A {v:1}), (:A {v:2})-[:R]->(:B {v:7})`):
`MATCH (a:A) WITH a ORDER BY a.v CALL { WITH a OPTIONAL MATCH (a)-[r:R]->(b) RETURN b.v AS bv }
RETURN a.v, bv` → Rust `(2,7),(1,null)`, C `(1,null),(2,7)` (issue #3070's family; order only). -/
theorem optF_not_exact :
    applyBatched (fun x y => (x, y))
        (optF (fun x => if x = 1 then [] else [5]) (fun x => 100 + x) (fun a => a) : SubPlan Nat Nat)
        [1, 2] = [(2, 5), (1, 101)] ∧
    applyRef (fun x y => (x, y))
        (optF (fun x => if x = 1 then [] else [5]) (fun x => 100 + x) (fun a => a) : SubPlan Nat Nat)
        [1, 2] = [(1, 101), (2, 5)] := by decide

end OpSep
