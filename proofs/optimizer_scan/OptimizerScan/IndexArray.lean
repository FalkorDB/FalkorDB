import OptimizerScan.IndexPipeline
/-!
# Array-contains (`value IN n.prop`) through `try_filter_pushdown`

`utilize_sound` (`IndexPipeline`) assumes `goodA` conjuncts, which exclude `x IN n.k`. This file
closes that gap for every branch (the `try_filter_pushdown/array-contains` row): an array-contains
conjunct or disjunct is pushed as `IndexQuery::ArrayContains` (`Q.contains`) **and the whole
original filter is kept** (`pushed_needs_post_filter`, `utilize_index.rs`, #3064), because the
array tree answers with false positives (`1 IN [true]`: Bool and Int share the numeric key, `arrEnc`).

* `contains_no_fn`: the index never loses a row — if `x IN n.k` is true, the array tree has it.
* `pushAnd_spec_ac` / `pushOr_spec_ac`: the AND and OR branches with array-contains atoms never lose
  a row, and set the flag whenever an array-contains is pushed.
* `utilize_sound_ac` (headline): a rewritten AND or OR of covered atoms and array-contains atoms, or a
  single array-contains atom, selects exactly what `Filter → NodeByLabelScan` does, on nodes whose
  values are Null/Int/String/lists (`Faithful`).
* `tryPushdown_or_ac_keeps_filter` / `Cex.C12`: the **OR** branch keeps the filter of an
  array-contains disjunct since #3064 (before, it kept none: `1 IN n.arr OR n.k = 9` also returned
  `{arr:[true], k:0}`).
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

theorem cu_contains (k : Nat) (c : V) (hc : goodC c = true) :
    canUtilize (evalIQ (.contains k (.lit c))) = true := by
  cases c <;> simp_all [goodC, evalIQ, evalC, evalT, canUtilize, isIndexable]

theorem bool_ac_none (x r e s : Bool) (h : e = true → x = true) :
    (x && (r && e) && s) = (true && r && (e && s)) := by
  cases x <;> cases r <;> cases e <;> cases s <;> simp_all

theorem bool_ac_some (q x r e s : Bool) (h : e = true → x = true) :
    (q && x && (r && e) && s) = (q && r && (e && s)) := by
  cases q <;> cases x <;> cases r <;> cases e <;> cases s <;> simp_all

theorem pna_ac (a : A) (h : goodAC a = true) : pushedNeedsPost a = true := by
  simp [pushedNeedsPost, isArrayContains_ac a h]

/-- The conjuncts the AND branch handles here: a covered atom (no strict string bound) or an
array-contains. -/
def okA (a : A) : Prop := (goodA a = true ∧ strictStr a = false) ∨ goodAC a = true

/-- **PROVEN** (the AND branch of `try_filter_pushdown` with array-contains conjuncts): on a
single-label scan the merged query stays on `L` and well-formed, and with the leftovers never loses
a row of the conjunction; when the returned flag is false (no array-contains was pushed) they select
exactly the conjunction and the merged query is runtime-usable. -/
theorem pushAnd_spec_ac (opq : Nat → Node → Bool) :
    ∀ (as : List A) (m : Option (Nat × IQ)) (rem : List A) (keep : Bool),
    (∀ a ∈ as, okA a) →
    (∀ L0 q0, m = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) →
    (∀ L0 q0, (pushAnd idx [L] as m rem keep).1 = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels →
      (mSem idx L m n && rem.all (evalA opq n) && as.all (evalA opq n)) = true →
      (mSem idx L (pushAnd idx [L] as m rem keep).1 n &&
        (pushAnd idx [L] as m rem keep).2.1.all (evalA opq n)) = true) ∧
    ((pushAnd idx [L] as m rem keep).2.2 = false → keep = false ∧
      (∀ n : Node, Faithful n → L ∈ n.labels →
        (mSem idx L (pushAnd idx [L] as m rem keep).1 n &&
          (pushAnd idx [L] as m rem keep).2.1.all (evalA opq n))
        = (mSem idx L m n && rem.all (evalA opq n) && as.all (evalA opq n))) ∧
      ((∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
        ∀ L0 q0, (pushAnd idx [L] as m rem keep).1 = some (L0, q0) → canUtilize (evalIQ q0) = true))
  | [], m, rem, keep, _, hm => by
    refine ⟨hm, fun n _ _ h => by simpa [pushAnd] using h,
      fun hk => ⟨hk, fun n _ _ => by simp [pushAnd], fun h => h⟩⟩
  | a :: as, m, rem, keep, hg, hm => by
    have hga := hg a (by simp)
    have hgs : ∀ b ∈ as, okA b := fun b hb => hg b (by simp [hb])
    -- the conjunct stays among the leftovers
    have hleft : ∀ (r : Option (Nat × IQ) × List A × Bool), r = pushAnd idx [L] as m (rem ++ [a]) keep →
        (∀ L0 q0, r.1 = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) ∧
        (∀ n : Node, Faithful n → L ∈ n.labels →
          (mSem idx L m n && rem.all (evalA opq n) && (a :: as).all (evalA opq n)) = true →
          (mSem idx L r.1 n && r.2.1.all (evalA opq n)) = true) ∧
        (r.2.2 = false → keep = false ∧
          (∀ n : Node, Faithful n → L ∈ n.labels →
            (mSem idx L r.1 n && r.2.1.all (evalA opq n))
            = (mSem idx L m n && rem.all (evalA opq n) && (a :: as).all (evalA opq n))) ∧
          ((∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
            ∀ L0 q0, r.1 = some (L0, q0) → canUtilize (evalIQ q0) = true)) := by
      intro r hr; subst hr
      obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec_ac opq as m (rem ++ [a]) keep hgs hm
      refine ⟨ih1, fun n hn hL h => ih2 n hn hL ?_, fun hk => ?_⟩
      · revert h
        simp only [List.all_append, List.all_cons, List.all_nil, Bool.and_true]
        cases mSem idx L m n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp
      · obtain ⟨hk1, ih3a, ih3b⟩ := ih3 hk
        refine ⟨hk1, fun n hn hL => ?_, ih3b⟩
        rw [ih3a n hn hL]
        simp only [List.all_append, List.all_cons, List.all_nil, Bool.and_true]
        cases mSem idx L m n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp
    simp only [pushAnd]
    split
    · next L' q hs =>
      split
      · rcases hga with hga | hac
        · -- a covered atom: exactly as in `pushAnd_spec`
          obtain ⟨hL', hex, hro, -, -⟩ := trySingle_spec idx L opq a L' q hga.1 hs
          subst L'
          have hm' : ∀ L0 q0, mergeInto m L q = some (L0, q0) →
                L0 = L ∧ rangeOK idx L q0 = true := by
            intro L0 q0 h
            cases m with
            | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨rfl, hro hga.2⟩
            | some p =>
              obtain ⟨L1, q1⟩ := p
              simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
              have := hm L1 q1 rfl
              exact ⟨this.1, rangeOK_merge idx L _ _ this.2 (hro hga.2)⟩
          have hmsem : ∀ n : Node, Faithful n → L ∈ n.labels →
              (mSem idx L (mergeInto m L q) n && rem.all (evalA opq n) && as.all (evalA opq n)) =
                (mSem idx L m n && rem.all (evalA opq n) && (a :: as).all (evalA opq n)) := by
            intro n hn hL
            cases m with
            | none => simp [mSem, mergeInto, hex n hn hL]; cases evalA opq n a <;> simp
            | some p =>
              obtain ⟨L1, q1⟩ := p
              have := hm L1 q1 rfl
              simp only [mSem, mergeInto]
              rw [sel_merge idx L n _ _ this.2 (hro hga.2) hn hL, hex n hn hL]
              simp only [List.all_cons]
              cases qsem idx L q1 n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp
          obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec_ac opq as _ rem (keep || pushedNeedsPost a) hgs hm'
          refine ⟨ih1, fun n hn hL h => ih2 n hn hL (by rw [hmsem n hn hL]; exact h), fun hk => ?_⟩
          obtain ⟨hk1, ih3a, ih3b⟩ := ih3 hk
          simp only [Bool.or_eq_false_iff] at hk1
          refine ⟨hk1.1, fun n hn hL => by rw [ih3a n hn hL, hmsem n hn hL], fun hcm => ih3b ?_⟩
          have hcq := pna_cu idx L a L q hga.1 hs hk1.2
          intro L0 q0 h
          cases m with
          | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact hcq
          | some p =>
            obtain ⟨L1, q1⟩ := p
            simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
            exact cu_merge _ _ (hcm L1 q1 rfl) hcq
        · -- an array-contains: pushed, and the flag is set
          obtain ⟨k, c, rfl, hL', rfl, hk, hcg⟩ := trySingle_ac idx L a hac L' q hs
          subst L'
          have hro : rangeOK idx L (.contains k (.lit c)) = true := rfl
          have hm' : ∀ L0 q0, mergeInto m L (.contains k (.lit c)) = some (L0, q0) →
                L0 = L ∧ rangeOK idx L q0 = true := by
            intro L0 q0 h
            cases m with
            | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨rfl, hro⟩
            | some p =>
              obtain ⟨L1, q1⟩ := p
              simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
              have := hm L1 q1 rfl
              exact ⟨this.1, rangeOK_merge idx L _ _ this.2 hro⟩
          obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec_ac opq as _ rem (keep || pushedNeedsPost
            (.inn (.lit c) (.term (.prop k)))) hgs hm'
          refine ⟨ih1, fun n hn hL h => ih2 n hn hL ?_, fun hkf => ?_⟩
          · have hex := contains_no_fn idx L opq k c hcg n hk hL
            revert h
            cases m with
            | none =>
              simp only [mSem, mergeInto, List.all_cons]
              cases he : evalA opq n (.inn (.lit c) (.term (.prop k)))
              · simp
              · simp [hex he]
            | some p =>
              obtain ⟨L1, q1⟩ := p
              have := hm L1 q1 rfl
              simp only [mSem, mergeInto]
              rw [sel_merge idx L n _ _ this.2 hro hn hL]
              simp only [List.all_cons]
              cases he : evalA opq n (.inn (.lit c) (.term (.prop k)))
              · simp
              · simp [hex he]
          · obtain ⟨hk1, -, -⟩ := ih3 hkf
            simp [pna_ac _ hac] at hk1
      · exact hleft _ rfl
    · exact hleft _ rfl

/-- **PROVEN** (the OR branch with array-contains disjuncts): every branch on `L`, the union never
loses a row of the disjunction, and the flag is false only when every branch is a covered atom (so
`pushOr_spec` makes the union exact). -/
theorem pushOr_spec_ac (opq : Nat → Node → Bool) :
    ∀ (as : List A) (lab : Option Nat) (qs : List IQ) (k : Bool),
    (∀ a ∈ as, goodA a = true ∨ goodAC a = true) → pushOr idx [L] as = some (lab, qs, k) →
    (∀ L0, lab = some L0 → L0 = L) ∧ (qs = [] ↔ as = []) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels → as.any (evalA opq n) = true →
      (evalIQs qs).any (fun q => bsel (idx L) q n) = true) ∧
    (k = false → ∀ a ∈ as, goodA a = true)
  | [], lab, qs, k, _, h => by
    simp [pushOr] at h; obtain ⟨rfl, rfl, rfl⟩ := h; simp
  | a :: as, lab, qs, k, hg, h => by
    simp only [pushOr] at h
    split at h
    · next L' q lab' qs' k' hs hr =>
      split at h
      · simp at h; obtain ⟨rfl, rfl, rfl⟩ := h
        have hL' : L' = L := by
          rcases hg a (by simp) with hga | hac
          · exact (trySingle_spec idx L opq a L' q hga hs).1
          · obtain ⟨_, _, _, h, _⟩ := trySingle_ac idx L a hac L' q hs; exact h
        subst L'
        have hq : ∀ n : Node, Faithful n → L ∈ n.labels → evalA opq n a = true →
            bsel (idx L) (evalIQ q) n = true := by
          intro n hn hL he
          rcases hg a (by simp) with hga | hac
          · have := (trySingle_spec idx L opq a L q hga hs).2.1 n hn hL
            rw [he] at this
            simpa [qsem, idxSel, hL] using this
          · obtain ⟨k, c, rfl, -, rfl, hk, hcg⟩ := trySingle_ac idx L a hac L q hs
            have := contains_no_fn idx L opq k c hcg n hk hL he
            simpa [qsem, idxSel, hL] using this
        obtain ⟨-, -, ih3, ih4⟩ :=
          pushOr_spec_ac opq as _ qs' k' (fun b hb => hg b (by simp [hb])) hr
        refine ⟨fun L0 h => by simp at h; rw [← h], by simp, fun n hn hL h => ?_, fun hk => ?_⟩
        · simp only [List.any_cons, Bool.or_eq_true] at h
          simp only [evalIQs, List.any_cons, Bool.or_eq_true]
          rcases h with h | h
          · exact Or.inl (hq n hn hL h)
          · exact Or.inr (ih3 n hn hL h)
        · simp only [Bool.or_eq_false_iff] at hk
          intro b hb
          simp only [List.mem_cons] at hb
          rcases hb with rfl | hb
          · rcases hg b (by simp) with hga | hac
            · exact hga
            · rw [pna_ac b hac] at hk; simp at hk
          · exact ih4 hk.2 b hb
      · simp at h
    · simp at h

/-- The filter shapes `utilize_sound_ac` adds: a single array-contains atom, an AND whose
conjuncts are covered atoms (no strict string bound) or array-contains atoms, or an OR whose
branches are covered atoms or array-contains atoms. -/
def goodFAC : F → Bool
  | .atom a => goodAC a
  | .and as => as.all (fun a => (goodA a && !strictStr a) || goodAC a)
  | .or as => as.all (fun a => goodA a || goodAC a)

/-- **PROVEN** (headline for array-contains, #3736, OR since #3064): on a single-label scan whose
property values are Null/Int/String/lists of scalars, a filter that is a single `x IN n.k`, an AND of
covered atoms and `x IN n.k` atoms, or an OR of them (literal `x`: non-lossy Int, String, Bool) is
rewritten into a plan that selects exactly the nodes the original `Filter → NodeByLabelScan` does.
The array tree's false positives (`1 IN [true]`) are removed by the kept filter: this is what
`pushed_needs_post_filter` buys. -/
theorem utilize_sound_ac (opq : Nat → Node → Bool) (f : F) (hf : goodFAC f = true)
    (n : Node) (hn : Faithful n) :
    (utilize idx [L] f).sel idx opq n = (reference [L] f).sel idx opq n := by
  have key : ∀ L' q rem keep, tryPushdown idx [L] f = some (L', q, rem, keep) →
      L' = L ∧ (∀ m : Node, Faithful m → L ∈ m.labels → evalF opq m f = true → qsem idx L q m = true) ∧
      (keep = false → canUtilize (evalIQ q) = true ∧ ∀ m : Node, Faithful m → L ∈ m.labels →
        (qsem idx L q m && rem.all (evalA opq m)) = evalF opq m f) := by
    intro L' q rem keep hp
    cases f with
    | atom a =>
      have hac : goodAC a = true := by simpa [goodFAC] using hf
      obtain ⟨hs, -, hkeep⟩ := tp_atom idx [L] a L' q rem keep hp
      obtain ⟨k, c, rfl, hLL, rfl, hk, hcg⟩ := trySingle_ac idx L a hac L' q hs
      subst L'
      refine ⟨rfl, fun m hm hLm h => contains_no_fn idx L opq k c hcg m hk hLm h, fun hk' => ?_⟩
      rw [hkeep, pna_ac _ hac] at hk'; simp at hk'
    | and as =>
      have hok : ∀ a ∈ as, okA a := by
        intro b hb
        have := List.all_eq_true.mp hf b hb
        simp only [Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true'] at this
        exact this
      obtain ⟨h1, h2, h3⟩ := tp_and idx [L] as L' q rem keep hp
      obtain ⟨hlab, hnf, hex⟩ := pushAnd_spec_ac idx L opq as none [] false hok (by simp)
      obtain ⟨hL', -⟩ := hlab L' q h1
      subst L'
      refine ⟨rfl, fun m hm hLm h => ?_, fun hk => ?_⟩
      · have := hnf m hm hLm (by simpa [mSem, evalF] using h)
        rw [h1] at this
        simp only [mSem, Bool.and_eq_true] at this
        exact this.1
      · obtain ⟨-, heq, hcu⟩ := hex (by rw [h3]; exact hk)
        refine ⟨hcu (by simp) L q h1, fun m hm hLm => ?_⟩
        have := heq m hm hLm
        rw [h1, h2] at this
        simpa [mSem, evalF] using this
    | or as =>
      obtain ⟨qs, hpo, rfl, rfl⟩ := tp_or idx [L] as L' q rem keep hp
      have hg : ∀ a ∈ as, goodA a = true ∨ goodAC a = true := fun a ha => by
        have := List.all_eq_true.mp hf a ha
        simpa using this
      obtain ⟨hlab, hqe, hnf, hgood⟩ := pushOr_spec_ac idx L opq as _ qs keep hg hpo
      have hL' := hlab L' rfl
      subst L'
      have hne : qs ≠ [] := by
        intro h
        have := hqe.mp h
        subst this
        simp [pushOr] at hpo
      refine ⟨rfl, fun m hm hLm h => ?_, fun hk => ?_⟩
      · simp only [qsem, idxSel, evalIQ, bsel, build, hLm, decide_true, Bool.true_and]
        rw [buildSome_any]
        exact hnf m hm hLm (by simpa [evalF] using h)
      · obtain ⟨-, -, -, hany, -, hcan⟩ := pushOr_spec idx L opq as _ qs keep (hgood hk) hpo
        refine ⟨?_, fun m hm hLm => ?_⟩
        · simp only [evalIQ, canUtilize]
          rw [hcan hk]
          cases qs with
          | nil => exact absurd rfl hne
          | cons _ _ => rfl
        · simp only [qsem, idxSel, evalIQ, bsel, build, hLm, decide_true, Bool.true_and,
            List.all_nil, Bool.and_true, evalF]
          rw [buildSome_any, hany m hm hLm]
  by_cases hL : L ∈ n.labels
  case neg => exact utilize_off_label idx L opq f n hL (fun L' q rem keep hp => (key L' q rem keep hp).1)
  case pos =>
    cases hp : tryPushdown idx [L] f with
    | none => simp [utilize, hp, reference, Plan.sel, hasAll_single, hL]
    | some t =>
      obtain ⟨L', q, rem, keep⟩ := t
      obtain ⟨hL', hnf, hex⟩ := key L' q rem keep hp
      subst L'
      exact utilize_on_label idx L opq f n hL q rem keep hp (hnf n hn hL)
        (fun hk => ⟨(hex hk).1, (hex hk).2 n hn hL⟩)

/-- **PROVEN** (#3064): an array-contains branch sets the OR branch's flag. -/
theorem pushOr_keep_of_ac (ls : List Nat) : ∀ (as : List A) (lab : Option Nat) (qs : List IQ) (k : Bool),
    pushOr idx ls as = some (lab, qs, k) → (∃ a ∈ as, isArrayContains a = true) → k = true
  | [], _, _, _, _, ⟨_, ha, _⟩ => by simp at ha
  | a :: as, lab, qs, k, h, ⟨b, hb, hac⟩ => by
    simp only [pushOr] at h
    split at h
    · next L' q lab' qs' k' hs hr =>
      split at h
      · simp at h; obtain ⟨rfl, rfl, rfl⟩ := h
        simp only [List.mem_cons] at hb
        rcases hb with rfl | hb
        · simp [pushedNeedsPost, hac]
        · simp [pushOr_keep_of_ac ls as lab' qs' k' hr ⟨b, hb, hac⟩]
      · simp at h
    · simp at h

/-- **PROVEN** (#3064, W2-scan-7 fixed): an OR with an array-contains branch keeps the whole
original filter above the index scan, whatever the labels and shapes. -/
theorem tryPushdown_or_ac_keeps_filter (ls : List Nat) (as : List A) (L' : Nat) (q : IQ) (rem : List A)
    (keep : Bool) (h : tryPushdown idx ls (.or as) = some (L', q, rem, keep))
    (hac : ∃ a ∈ as, isArrayContains a = true) :
    utilize idx ls (.or as) = .filter (.or as) (.idxScan ls L' q) := by
  obtain ⟨qs, hpo, rfl, rfl⟩ := tp_or idx ls as L' q rem keep h
  have := pushOr_keep_of_ac idx ls as _ qs keep hpo hac
  subst this
  rw [utilize_some idx ls _ L' _ [] true h]
  simp

/-- C12 — FIXED by #3064 (was confirmed live, shared with C): `1 IN n.arr OR n.k = 9` on
`{arr: [true], k: 0}`. The OR branch now returns the flag, the Filter stays, and the false positive
of the array tree is removed: the scan agrees. -/
example : tryPushdown (fun _ => [3, 0]) [0]
        (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))]) =
      some (0, .or [.contains 3 (.lit (.i 1)), .eq 0 (.lit (.i 9))], [], true) ∧
    (utilize (fun _ => [3, 0]) [0]
      (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))])).sel
      (fun _ => [3, 0]) (fun _ _ => true) ⟨[0], fun k => if k = 3 then .arr [.b true] else if k = 0 then .i 0 else .null⟩ = false ∧
    (reference [0]
      (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))])).sel
      (fun _ => [3, 0]) (fun _ _ => true) ⟨[0], fun k => if k = 3 then .arr [.b true] else if k = 0 then .i 0 else .null⟩ = false := by
  exact ⟨rfl, by decide, by decide⟩

end
end OptimizerScan.Index
