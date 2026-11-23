import OptimizerScan.IndexPipeline
/-!
# Array-contains (`value IN n.prop`) through `try_filter_pushdown`

`utilize_sound` (`IndexPipeline`) assumes `goodA` conjuncts, which exclude `x IN n.k`. This file
closes that gap for every branch (the `try_filter_pushdown/array-contains` row): an array-contains
is pushed as `IndexQuery::ArrayContains` (`Q.contains`) and, since #2975, **its filter is kept
whatever the branch** (`pushed_needs_post_filter`, `utilize_index.rs`).

* `contains_exact` (#2975): the array tree answers membership exactly — a Bool member has a TAG
  key of its own (`tag_key`), so `1 IN [true]` is no longer a false positive (`contains_no_fn`:
  it never loses a row).
* `pushAnd_spec_ac`: the AND branch with array-contains conjuncts selects exactly the conjunction,
  still on `L`, well-formed and runtime-usable.
* `utilize_sound_ac` (headline): a rewritten single array-contains atom, AND or OR of covered atoms
  and array-contains atoms selects exactly what `Filter → NodeByLabelScan` does, on nodes whose
  values are Null/Bool/Int/String/lists (`Faithful`).
* `tryPushdown_or_ac_keeps_filter` / `Cex.C12`: the **OR** branch now keeps the filter of an
  array-contains disjunct (C12 / W9-scan-1, fixed by #2975).
-/
namespace OptimizerScan.Index

section
variable (idx : Nat → List Nat) (L : Nat)

/-- Literals the array tree answers exactly for membership: non-lossy Int, String, Bool
(`Q.contains` is built for `b`/`i`/`s`, `build_query_node` `ArrayContains` arm). -/
def goodC : V → Bool
  | .i x => !lossy x
  | .s _ => true
  | .b _ => true
  | _ => false

/-- `x IN n.k`, with a literal `x` and the bare property on the right. -/
def goodAC : A → Bool
  | .inn (.lit c) (.term (.prop _)) => goodC c
  | _ => false

theorem isArrayContains_ac (a : A) (h : goodAC a = true) : isArrayContains a = true := by
  match a, h with
  | .inn (.lit _) (.term (.prop _)), _ => rfl

theorem good_not_ac (a : A) (h : goodA a = true) : goodAC a = false := by
  match a, h with
  | .cmp _ (.prop _) (.lit _), _ => rfl
  | .cmp _ (.lit _) (.prop _), _ => rfl
  | .inn (.prop _) (.list _), _ => rfl
  | .opq _, _ => rfl

/-- What `try_single_filter_scan` returns for an array-contains atom. -/
theorem trySingle_ac (a : A) (h : goodAC a = true) (L' : Nat) (q : IQ)
    (hs : trySingle idx [L] a = some (L', q)) :
    ∃ k c, a = .inn (.lit c) (.term (.prop k)) ∧ L' = L ∧ q = .contains k (.lit c) ∧ k ∈ idx L ∧
      goodC c = true := by
  match a, h with
  | .inn (.lit c) (.term (.prop k)), hg =>
    simp only [goodAC] at hg
    simp only [trySingle, hasT, hasR, findLabel, List.find?, Option.bind_eq_bind] at hs
    by_cases hk : k ∈ idx L
    · simp [hk] at hs
      obtain ⟨rfl, rfl⟩ := hs
      exact ⟨k, c, rfl, rfl, rfl, hk, hg⟩
    · simp [hk] at hs

/-- **PROVEN**: the index never loses a row of an array-contains. If `x IN n.k` is `true`, the
array tree of `k` holds `x`'s key (`1` and `true` share one, so it may hold more). -/
theorem contains_no_fn (opq : Nat → Node → Bool) (k : Nat) (c : V) (hc : goodC c = true)
    (n : Node) (hk : k ∈ idx L) (hL : L ∈ n.labels)
    (h : evalA opq n (.inn (.lit c) (.term (.prop k))) = true) :
    qsem idx L (.contains k (.lit c)) n = true := by
  simp only [evalA, evalT] at h
  cases hp : n.prop k with
  | arr xs =>
    rw [hp] at h
    simp only [inList, List.any_eq_true, List.mem_map] at h
    obtain ⟨y, ⟨p, hpx, rfl⟩, hy⟩ := h
    have hceq : c = p.toV := by
      cases c <;> cases p <;> simp_all [eq3, goodC, P.toV]
    unfold qsem idxSel bsel
    simp only [evalIQ, evalC, evalT, hL, decide_true, Bool.true_and, build, hk, ite_true]
    cases c <;> cases p <;> simp_all [goodC, P.toV, arrEnc, encP]
    all_goals first | exact ⟨_, hpx, rfl⟩ | skip
  | null => rw [hp] at h; simp [eq3] at h
  | b _ => rw [hp] at h; simp at h
  | i _ => rw [hp] at h; simp at h
  | s _ => rw [hp] at h; simp at h
  | date _ => rw [hp] at h; simp at h

/-- **PROVEN** (#2975): the array tree answers membership exactly. Every member has its own key
(`encP`: NUMERIC for an Int, a string TAG, a Bool TAG apart from both), so the index holds `x`'s
key iff `x IN n.k`. -/
theorem contains_exact (opq : Nat → Node → Bool) (k : Nat) (c : V) (hc : goodC c = true)
    (n : Node) (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    qsem idx L (.contains k (.lit c)) n = evalA opq n (.inn (.lit c) (.term (.prop k))) := by
  unfold qsem idxSel bsel
  simp only [evalIQ, evalC, evalT, hL, decide_true, Bool.true_and, build, hk, ite_true, evalA]
  have hmem : ∀ (f : Fld), (∀ p : P, (encP p == f) = (eq3 c p.toV == some true)) →
      ∀ xs : List P, (xs.map encP).contains f = inList c (xs.map P.toV) := by
    intro f hf xs
    induction xs with
    | nil => rfl
    | cons p ps ih =>
      simp only [List.map_cons, List.contains_cons, inList, List.any_cons] at ih ⊢
      rw [ih, ← hf p]
      congr 1
      rw [Bool.eq_iff_iff]; simp only [beq_iff_eq]; exact eq_comm
  cases hp : n.prop k with
  | arr xs =>
    cases c <;> simp only [goodC] at hc <;> (try contradiction) <;> simp only [arrEnc, hp] <;>
      apply hmem <;> intro p <;> cases p <;> simp [encP, P.toV, eq3, eq_comm]
  | _ => cases c <;> simp_all [goodC, arrEnc]

theorem cu_contains (k : Nat) (c : V) (hc : goodC c = true) :
    canUtilize (evalIQ (.contains k (.lit c))) = true := by
  cases c <;> simp_all [goodC, evalIQ, evalC, evalT, canUtilize, isIndexable]

/-- The conjuncts the AND branch handles here: a covered atom (no strict string bound) or an
array-contains. -/
def okA (a : A) : Prop := (goodA a = true ∧ strictStr a = false) ∨ goodAC a = true

/-- **PROVEN** (the AND branch of `try_filter_pushdown` with array-contains conjuncts): on a
single-label scan, the merged query and the leftover conjuncts together select exactly the
conjunction; the merged query stays on `L`, well-formed, and usable by the runtime when no pushed
conjunct is an `IN` list. -/
theorem pushAnd_spec_ac (opq : Nat → Node → Bool) :
    ∀ (as : List A) (m : Option (Nat × IQ)) (rem : List A),
    (∀ a ∈ as, okA a) →
    (∀ L0 q0, m = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) →
    (∀ L0 q0, (pushAnd idx [L] as m rem).1 = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels →
      (mSem idx L (pushAnd idx [L] as m rem).1 n && (pushAnd idx [L] as m rem).2.all (evalA opq n))
        = (mSem idx L m n && rem.all (evalA opq n) && as.all (evalA opq n))) ∧
    ((∀ a ∈ as, isInn a = false ∨ goodAC a = true) →
      (∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
      ∀ L0 q0, (pushAnd idx [L] as m rem).1 = some (L0, q0) → canUtilize (evalIQ q0) = true)
  | [], m, rem, _, hm => by
    refine ⟨hm, fun n _ _ => by simp [pushAnd], fun _ h => h⟩
  | a :: as, m, rem, hg, hm => by
    have hga := hg a (by simp)
    have hgs : ∀ b ∈ as, okA b := fun b hb => hg b (by simp [hb])
    simp only [pushAnd]
    split
    · next L' q hs =>
      -- what the pushed conjunct contributes: its query, exact, on `L`, well-formed, usable
      obtain ⟨hL', hex, hro, hcq⟩ : L' = L ∧
          (∀ n : Node, Faithful n → L ∈ n.labels → qsem idx L q n = evalA opq n a) ∧
          rangeOK idx L q = true ∧
          ((isInn a = false ∨ goodAC a = true) → canUtilize (evalIQ q) = true) := by
        rcases hga with hga | hac
        · obtain ⟨hL', hex, hro, hcu, -⟩ := trySingle_spec idx L opq a L' q hga.1 hs
          refine ⟨hL', hex, hro hga.2, fun h => ?_⟩
          rcases h with hi | hi
          · exact hcu hi
          · rw [good_not_ac a hga.1] at hi; exact absurd hi (by simp)
        · obtain ⟨k, c, rfl, hL', rfl, hk, hcg⟩ := trySingle_ac idx L _ hac L' q hs
          exact ⟨hL', fun n _ hL => contains_exact idx L opq k c hcg n hk hL, rfl,
            fun _ => cu_contains k c hcg⟩
      subst L'
      have hm' : ∀ L0 q0, mergeInto m L q = some (L0, q0) →
            L0 = L ∧ rangeOK idx L q0 = true := by
        intro L0 q0 h
        cases m with
        | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨rfl, hro⟩
        | some p =>
          obtain ⟨L1, q1⟩ := p
          simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
          have := hm L1 q1 rfl
          exact ⟨this.1, rangeOK_merge idx L _ _ this.2 hro⟩
      obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec_ac opq as _ rem hgs hm'
      refine ⟨ih1, fun n hn hL => ?_, fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) ?_⟩
      · rw [ih2 n hn hL]
        cases m with
        | none => simp [mSem, mergeInto, hex n hn hL]; cases evalA opq n a <;> simp
        | some p =>
          obtain ⟨L1, q1⟩ := p
          have := hm L1 q1 rfl
          simp only [mSem, mergeInto]
          rw [sel_merge idx L n _ _ this.2 hro hn hL, hex n hn hL]
          simp only [List.all_cons]
          cases qsem idx L q1 n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp
      · intro L0 q0 h
        have hcq := hcq (hni a (by simp))
        cases m with
        | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact hcq
        | some p =>
          obtain ⟨L1, q1⟩ := p
          simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
          exact cu_merge _ _ (hcm L1 q1 rfl) hcq
    · obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec_ac opq as m (rem ++ [a]) hgs hm
      refine ⟨ih1, fun n hn hL => ?_, fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) hcm⟩
      rw [ih2 n hn hL]
      simp only [List.all_append, List.all_cons, List.all_nil, Bool.and_true]
      cases mSem idx L m n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp

/-- An array-contains always keeps its filter (`pushed_needs_post_filter`). -/
theorem pushedNeedsPost_ac (a : A) (h : goodAC a = true) : pushedNeedsPost a = true := by
  simp [pushedNeedsPost, isArrayContains_ac a h]

/-- **PROVEN** (the OR branch with array-contains disjuncts): every disjunct converted, the union
selects exactly the disjunction. -/
theorem pushOr_spec_ac (opq : Nat → Node → Bool) :
    ∀ (as : List A) (lab : Option Nat) (qs : List IQ),
    (∀ a ∈ as, goodA a = true ∨ goodAC a = true) → pushOr idx [L] as = some (lab, qs) →
    (∀ L0, lab = some L0 → L0 = L) ∧ (qs = [] ↔ as = []) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels →
      (evalIQs qs).any (fun q => bsel (idx L) q n) = as.any (evalA opq n))
  | [], lab, qs, _, h => by
    simp [pushOr] at h; obtain ⟨rfl, rfl⟩ := h; simp [evalIQs]
  | a :: as, lab, qs, hg, h => by
    simp only [pushOr] at h
    split at h
    · next L' q _ qs' hs hr =>
      simp at h; obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hL', hex⟩ : L' = L ∧
          (∀ n : Node, Faithful n → L ∈ n.labels → qsem idx L q n = evalA opq n a) := by
        rcases hg a (by simp) with hga | hac
        · obtain ⟨hL', hex, -⟩ := trySingle_spec idx L opq a L' q hga hs
          exact ⟨hL', hex⟩
        · obtain ⟨k, c, rfl, hL', rfl, hk, hcg⟩ := trySingle_ac idx L _ hac L' q hs
          exact ⟨hL', fun n _ hL => contains_exact idx L opq k c hcg n hk hL⟩
      subst L'
      obtain ⟨-, -, ih⟩ := pushOr_spec_ac opq as _ qs' (fun b hb => hg b (by simp [hb])) hr
      refine ⟨fun L0 h => by simp at h; rw [← h], by simp, fun n hn hL => ?_⟩
      simp only [evalIQs, List.any_cons]
      rw [ih n hn hL]
      have := hex n hn hL
      simp only [qsem, idxSel, hL, decide_true, Bool.true_and] at this
      rw [this]
    · simp at h

/-- The filter shapes `utilize_sound_ac` adds: a single array-contains atom, an AND whose conjuncts
are covered atoms (no strict string bound) or array-contains atoms, or an OR of covered atoms and
array-contains atoms. -/
def goodFAC : F → Bool
  | .atom a => goodAC a
  | .and as => as.all (fun a => (goodA a && !strictStr a) || goodAC a)
  | .or as => as.all (fun a => goodA a || goodAC a)

/-- **PROVEN** (headline for array-contains, #3736, OR since #2975): on a single-label scan whose
property values are Null/Bool/Int/String/lists of scalars, a filter that is a single `x IN n.k`,
an AND of covered atoms and `x IN n.k` atoms, or an OR of them (literal `x`: non-lossy Int, String,
Bool) is rewritten into a plan that selects exactly the nodes the original
`Filter → NodeByLabelScan` does. -/
theorem utilize_sound_ac (opq : Nat → Node → Bool) (f : F) (hf : goodFAC f = true)
    (n : Node) (hn : Faithful n) :
    (utilize idx [L] f).sel idx opq n = (reference [L] f).sel idx opq n := by
  have hok : ∀ a ∈ (match f with | .and as => as | .atom a => [a] | .or _ => []), okA a := by
    cases f with
    | atom a => intro b hb; simp at hb; subst hb; exact Or.inr (by simpa [goodFAC] using hf)
    | and as =>
      intro b hb
      have := List.all_eq_true.mp hf b hb
      simp only [Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true'] at this
      exact this
    | or _ => simp
  have hor : ∀ as, f = .or as → ∀ a ∈ as, goodA a = true ∨ goodAC a = true := by
    intro as hfa a ha
    subst hfa
    have := List.all_eq_true.mp hf a ha
    simpa using this
  by_cases hL : L ∈ n.labels
  case neg =>
    have hr : (reference [L] f).sel idx opq n = false := by
      simp [reference, Plan.sel, hasAll_single, hL]
    rw [hr]
    unfold utilize
    split
    · simp [Plan.sel, hasAll_single, hL]
    · next L' q rem hp =>
      have hL' : L' = L := by
        unfold tryPushdown at hp
        cases f with
        | atom a =>
          simp only at hp; split at hp
          · next L2 q2 hs =>
            simp at hp; obtain ⟨rfl, -⟩ := hp
            obtain ⟨-, -, -, hl, -⟩ := trySingle_ac idx L a (by simpa [goodFAC] using hf) _ _ hs
            exact hl
          · simp at hp
        | and as =>
          simp only at hp; split at hp
          · next L2 q2 rem2 he =>
            simp at hp; obtain ⟨rfl, -⟩ := hp
            exact ((pushAnd_spec_ac idx L opq as none [] hok (by simp)).1 L2 q2 (by rw [he])).1
          · simp at hp
        | or as =>
          simp only at hp; split at hp
          · next L2 qs he =>
            simp at hp; obtain ⟨rfl, -⟩ := hp
            exact (pushOr_spec_ac idx L opq as _ qs (hor as rfl) he).1 L2 rfl
          · simp at hp
      subst L'
      simp only
      split <;> (try split) <;> simp [Plan.sel, idxSel, hasAll_single, hL]
  case pos =>
    have hsel : ∀ q, (Plan.idxScan [L] L q).sel idx opq n =
        (if canUtilize (evalIQ q) then qsem idx L q n else true) := by
      intro q
      simp [Plan.sel, hL, qsem, hasAll]
    have href : (reference [L] f).sel idx opq n = evalF opq n f := by
      simp [reference, Plan.sel, hasAll_single, hL]
    rw [href]
    cases hp : tryPushdown idx [L] f with
    | none => simp [utilize, hp, Plan.sel, hasAll_single, hL]
    | some t =>
      obtain ⟨L', q, rem⟩ := t
      rw [utilize_some idx [L] f L' q rem hp]
      have key : L' = L ∧ (∀ m : Node, Faithful m → L ∈ m.labels →
            (qsem idx L q m && rem.all (evalA opq m)) = evalF opq m f) ∧
          (pushedPost idx [L] f = false → canUtilize (evalIQ q) = true) := by
        cases f with
        | atom a =>
          have hac : goodAC a = true := by simpa [goodFAC] using hf
          obtain ⟨hs, hrem⟩ := tp_atom idx [L] a L' q rem hp
          obtain ⟨k, c, rfl, hLL, rfl, hk, hcg⟩ := trySingle_ac idx L a hac L' q hs
          subst L'
          refine ⟨rfl, fun m hm hLm => ?_, fun _ => cu_contains k c hcg⟩
          simp [hrem, contains_exact idx L opq k c hcg m hk hLm, evalF]
        | and as =>
          obtain ⟨h1, h2⟩ := tp_and idx [L] as L' q rem hp
          obtain ⟨hlab, heq, hcu⟩ := pushAnd_spec_ac idx L opq as none [] hok (by simp)
          obtain ⟨rfl, -⟩ := hlab L' q h1
          refine ⟨rfl, fun m hm hLm => ?_, fun hk => ?_⟩
          · have := heq m hm hLm
            rw [h1, h2] at this
            simpa [mSem, evalF] using this
          · -- no pushed conjunct keeps its filter: none is an array-contains, every `IN` list is
            -- served (`trySingle_cu`), so the pushed conjuncts are those of a covered AND
            have hpk : ∀ a ∈ as, (trySingle idx [L'] a).isSome = true → pushedNeedsPost a = false := by
              intro a ha hs
              simp only [pushedPost, List.any_eq_false] at hk
              have := hk a ha
              simp only [hs, Bool.true_and, Bool.not_eq_true] at this
              exact this
            -- split the conjuncts into the pushed ones (all covered) and the rest
            have hpush : ∀ (bs : List A) (m : Option (Nat × IQ)) (r : List A), (∀ b ∈ bs, okA b) →
                (∀ b ∈ bs, (trySingle idx [L'] b).isSome = true → pushedNeedsPost b = false) →
                (∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
                ∀ L0 q0, (pushAnd idx [L'] bs m r).1 = some (L0, q0) → canUtilize (evalIQ q0) = true := by
              intro bs
              induction bs with
              | nil => intro m r _ _ hm; simpa [pushAnd] using hm
              | cons b bs ih =>
                intro m r hb hbp hm
                simp only [pushAnd]
                split
                · next L2 q2 hs =>
                  have hnp := hbp b (by simp) (by simp [hs])
                  have hcq : canUtilize (evalIQ q2) = true := by
                    rcases hb b (by simp) with hg | hac
                    · exact trySingle_cu idx L' opq b L2 q2 hg.1 hs hnp
                    · rw [pushedNeedsPost_ac b hac] at hnp; exact absurd hnp (by simp)
                  refine ih _ r (fun c hc => hb c (by simp [hc])) (fun c hc => hbp c (by simp [hc])) ?_
                  intro L0 q0 h
                  cases m with
                  | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact hcq
                  | some p =>
                    obtain ⟨L1, q1⟩ := p
                    simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
                    exact cu_merge _ _ (hm L1 q1 rfl) hcq
                · exact ih m (r ++ [b]) (fun c hc => hb c (by simp [hc]))
                    (fun c hc => hbp c (by simp [hc])) hm
            exact hpush as none [] hok hpk (by simp) L' q h1
        | or as =>
          obtain ⟨qs, hpo, rfl, rfl⟩ := tp_or idx [L] as L' q rem hp
          obtain ⟨hlab, hqe, hany⟩ := pushOr_spec_ac idx L opq as _ qs (hor as rfl) hpo
          obtain rfl := hlab L' rfl
          have hne : qs ≠ [] := by
            intro h
            have := hqe.mp h
            subst this
            simp [pushOr] at hpo
          refine ⟨rfl, fun m hm hLm => ?_, fun hk => ?_⟩
          · simp only [qsem, idxSel, evalIQ, bsel, build, hLm, decide_true, Bool.true_and,
              List.all_nil, Bool.and_true, evalF]
            rw [buildSome_any, hany m hm hLm]
          · simp only [pushedPost, List.any_eq_false, Bool.not_eq_true] at hk
            have hg : ∀ a ∈ as, goodA a = true := by
              intro a ha
              rcases hor as rfl a ha with h | h
              · exact h
              · have := hk a ha; rw [pushedNeedsPost_ac a h] at this; exact absurd this (by simp)
            simp only [evalIQ, canUtilize]
            rw [pushOr_cu idx L' opq as _ qs hg hk hpo]
            cases qs with
            | nil => exact absurd rfl hne
            | cons _ _ => rfl
      obtain ⟨rfl, hsem, hcu⟩ := key
      have hs := hsem n hn hL
      have sel_filter : ∀ (g : F) (p : Plan), (Plan.filter g p).sel idx opq n =
          (p.sel idx opq n && evalF opq n g) := fun _ _ => rfl
      cases hk : pushedPost idx [L'] f <;> cases hr : rem.isEmpty <;>
        simp only [ite_true, ite_false, Bool.false_eq_true, sel_filter, hsel]
      · rw [ite_eq_left (hcu hk)]; exact hs
      · have : rem = [] := List.isEmpty_iff.mp hr
        subst this
        rw [ite_eq_left (hcu hk)]; simpa using hs
      · split <;> cases hq : qsem idx L' q n <;> cases hf' : evalF opq n f <;> simp_all
      · split <;> cases hq : qsem idx L' q n <;> cases hf' : evalF opq n f <;> simp_all

/-- **OR branch keeps the array-contains filter** (C12 / W9-scan-1, fixed by #2975): `pushOr`
converts the disjunct `1 IN n.arr` into the index query `contains`, and `pushed_needs_post_filter`
flags it, so the original Filter stays above the scan. Keys: `3 = arr`, `0 = k`. -/
theorem tryPushdown_or_ac_keeps_filter :
    tryPushdown (fun _ => [3, 0]) [0]
        (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))]) =
      some (0, .or [.contains 3 (.lit (.i 1)), .eq 0 (.lit (.i 9))], []) ∧
    pushedPost (fun _ => [3, 0]) [0]
      (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))]) = true :=
  ⟨rfl, rfl⟩

/-- C12, fixed by #2975: `1 IN n.arr OR n.k = 9` on `{arr: [true], k: 0}` — the plan and the
reference both reject the row (before, the plan had no Filter and the array tree held `true` under
the key of `1`). -/
example : (utilize (fun _ => [3, 0]) [0]
      (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))])).sel
      (fun _ => [3, 0]) (fun _ _ => true) ⟨[0], fun k => if k = 3 then .arr [.b true] else if k = 0 then .i 0 else .null⟩ = false ∧
    (reference [0]
      (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))])).sel
      (fun _ => [3, 0]) (fun _ _ => true) ⟨[0], fun k => if k = 3 then .arr [.b true] else if k = 0 then .i 0 else .null⟩ = false := by
  decide

end
end OptimizerScan.Index
