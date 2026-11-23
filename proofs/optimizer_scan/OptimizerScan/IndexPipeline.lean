import OptimizerScan.IndexSound
/-!
# End-to-end soundness of `utilize_index` on a single-label scan
-/
namespace OptimizerScan.Index

section
variable (idx : Nat → List Nat) (L : Nat)

theorem rangeOK_merge (a b : IQ) (ha : rangeOK idx L a = true) (hb : rangeOK idx L b = true) :
    rangeOK idx L (mergeRange a b) = true := by
  cases a <;> cases b <;> simp only [mergeRange] <;> (try rfl)
  rename_i k lo hi il ih k' lo' hi' il' ih'
  split
  · next hkk =>
    subst hkk
    split
    · rfl
    · next hc =>
      simp only [rangeOK, Bool.and_eq_true, decide_eq_true_eq] at ha hb
      rcases lo with _ | l <;> rcases lo' with _ | l' <;> rcases hi with _ | h <;>
        rcases hi' with _ | h' <;> simp_all [rangeOK]
  · rfl

theorem cu_and2 (a b : IQ) :
    canUtilize (evalIQ (.and [a, b])) = (canUtilize (evalIQ a) && canUtilize (evalIQ b)) := by
  simp [evalIQ, evalIQs, canUtilize, canAll]

theorem cu_merge (a b : IQ) (ha : canUtilize (evalIQ a) = true) (hb : canUtilize (evalIQ b) = true) :
    canUtilize (evalIQ (mergeRange a b)) = true := by
  cases a <;> cases b <;> simp only [mergeRange] <;> (try (rw [cu_and2, ha, hb]; rfl))
  rename_i k lo hi il ih k' lo' hi' il' ih'
  split
  · split
    · rw [cu_and2, ha, hb]; rfl
    · simp only [evalIQ, canUtilize, Bool.and_eq_true] at ha hb ⊢
      rcases lo with _ | l <;> rcases lo' with _ | l' <;> rcases hi with _ | h <;>
        rcases hi' with _ | h' <;> simp_all
  · rw [cu_and2, ha, hb]; rfl

theorem isArrayContains_good (a : A) (h : goodA a = true) : isArrayContains a = false := by
  match a, h with
  | .cmp _ (.prop _) (.lit _), _ => rfl
  | .cmp _ (.lit _) (.prop _), _ => rfl
  | .inn (.prop _) (.list _), _ => rfl
  | .opq _, _ => rfl

def mSem (m : Option (Nat × IQ)) (n : Node) : Bool :=
  match m with
  | none => true
  | some (_, q) => qsem idx L q n

theorem cu_inList (k : Nat) (ts : List T) (hne : ts ≠ []) (hts : ts.all scalarLit = true) :
    canUtilize (evalIQ (.inList k (.list ts))) = true := by
  have hgood : ∀ t ∈ ts, notNull (evalC t) = true ∧ isIndexable (evalC t) = true := by
    intro t ht
    have := List.all_eq_true.mp hts t ht
    rcases t with _ | ⟨_ | _ | _ | _ | _ | _⟩ | _ | _ | _ <;>
      simp_all [scalarLit, evalC, evalT, notNull, isIndexable]
  have hfilt : (ts.map evalC).filter notNull = ts.map evalC :=
    List.filter_eq_self.mpr (fun c hc => by
      obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hc; exact (hgood t ht).1)
  simp only [evalIQ, listVals, hfilt, canUtilize]
  have hall : (ts.map evalC).all isIndexable = true :=
    List.all_eq_true.mpr (fun c hc => by
      obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hc; exact (hgood t ht).2)
  rw [hall]
  cases ts with
  | nil => exact absurd rfl hne
  | cons _ _ => rfl

/-- A covered conjunct that `pushed_needs_post_filter` does not flag is answered by the runtime:
the evaluated query is `can_be_served`. -/
theorem pna_cu (a : A) (L' : Nat) (q : IQ) (hg : goodA a = true) (hs : trySingle idx [L] a = some (L', q))
    (hk : pushedNeedsPost a = false) : canUtilize (evalIQ q) = true := by
  obtain ⟨-, -, -, hcu, hin⟩ := trySingle_spec idx L (fun _ _ => true) a L' q hg hs
  cases hia : isInn a
  · exact hcu hia
  · match a, hg, hia with
    | .inn (.prop k) (.list ts), _, _ =>
      rw [hin ts k rfl]
      simp only [pushedNeedsPost, isArrayContains, hasT, Bool.not_true, Bool.false_and,
        Bool.false_or, needsPost] at hk
      split at hk
      · next hc =>
        simp only [Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff] at hc
        exact cu_inList k ts hc.1 hc.2
      · simp at hk

/-- **PROVEN** (the AND branch of `try_filter_pushdown`): on a single-label scan, the merged
query and the leftover conjuncts together select exactly the conjunction; every merged query
stays on `L` and well-formed, and is usable by the runtime when no conjunct is an `IN`, or
whenever the returned flag (`pushed_needs_post_filter` of the pushed conjuncts) is false. -/
theorem pushAnd_spec (opq : Nat → Node → Bool) :
    ∀ (as : List A) (m : Option (Nat × IQ)) (rem : List A) (keep : Bool),
    (∀ a ∈ as, goodA a = true ∧ strictStr a = false) →
    (∀ L0 q0, m = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) →
    (∀ L0 q0, (pushAnd idx [L] as m rem keep).1 = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels →
      (mSem idx L (pushAnd idx [L] as m rem keep).1 n &&
          (pushAnd idx [L] as m rem keep).2.1.all (evalA opq n))
        = (mSem idx L m n && rem.all (evalA opq n) && as.all (evalA opq n))) ∧
    ((∀ a ∈ as, isInn a = false) → (∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
      ∀ L0 q0, (pushAnd idx [L] as m rem keep).1 = some (L0, q0) → canUtilize (evalIQ q0) = true) ∧
    ((pushAnd idx [L] as m rem keep).2.2 = false → keep = false ∧
      ((∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
        ∀ L0 q0, (pushAnd idx [L] as m rem keep).1 = some (L0, q0) → canUtilize (evalIQ q0) = true))
  | [], m, rem, keep, _, hm => by
    refine ⟨hm, fun n _ _ => by simp [pushAnd], fun _ h => h, fun hk => ⟨hk, fun h => h⟩⟩
  | a :: as, m, rem, keep, hg, hm => by
    have hga := hg a (by simp)
    have hgs : ∀ b ∈ as, goodA b = true ∧ strictStr b = false := fun b hb => hg b (by simp [hb])
    simp only [pushAnd]
    split
    · next L' q hs =>
      obtain ⟨hL', hex, hro, hcu, -⟩ := trySingle_spec idx L opq a L' q hga.1 hs
      subst L'
      split
      · have hm' : ∀ L0 q0, mergeInto m L q = some (L0, q0) →
              L0 = L ∧ rangeOK idx L q0 = true := by
          intro L0 q0 h
          cases m with
          | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨rfl, hro hga.2⟩
          | some p =>
            obtain ⟨L1, q1⟩ := p
            simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
            have := hm L1 q1 rfl
            exact ⟨this.1, rangeOK_merge idx L _ _ this.2 (hro hga.2)⟩
        have hmerge : ∀ (hcq : canUtilize (evalIQ q) = true),
            (∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
            ∀ L0 q0, mergeInto m L q = some (L0, q0) → canUtilize (evalIQ q0) = true := by
          intro hcq hcm L0 q0 h
          cases m with
          | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact hcq
          | some p =>
            obtain ⟨L1, q1⟩ := p
            simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
            exact cu_merge _ _ (hcm L1 q1 rfl) hcq
        obtain ⟨ih1, ih2, ih3, ih4⟩ := pushAnd_spec opq as _ rem (keep || pushedNeedsPost a) hgs hm'
        refine ⟨ih1, fun n hn hL => ?_,
          fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) (hmerge (hcu (hni a (by simp))) hcm),
          fun hk => ?_⟩
        · rw [ih2 n hn hL]
          cases m with
          | none => simp [mSem, mergeInto, hex n hn hL]; cases evalA opq n a <;> simp
          | some p =>
            obtain ⟨L1, q1⟩ := p
            have := hm L1 q1 rfl
            simp only [mSem, mergeInto]
            rw [sel_merge idx L n _ _ this.2 (hro hga.2) hn hL, hex n hn hL]
            simp only [List.all_cons]
            cases qsem idx L q1 n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp
        · obtain ⟨hk1, ih4'⟩ := ih4 hk
          simp only [Bool.or_eq_false_iff] at hk1
          exact ⟨hk1.1, fun hcm => ih4' (hmerge (pna_cu idx L a L q hga.1 hs hk1.2) hcm)⟩
      · obtain ⟨ih1, ih2, ih3, ih4⟩ := pushAnd_spec opq as m (rem ++ [a]) keep hgs hm
        refine ⟨ih1, fun n hn hL => ?_, fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) hcm, ih4⟩
        rw [ih2 n hn hL]
        simp only [List.all_append, List.all_cons, List.all_nil, Bool.and_true]
        cases mSem idx L m n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp
    · obtain ⟨ih1, ih2, ih3, ih4⟩ := pushAnd_spec opq as m (rem ++ [a]) keep hgs hm
      refine ⟨ih1, fun n hn hL => ?_, fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) hcm, ih4⟩
      rw [ih2 n hn hL]
      simp only [List.all_append, List.all_cons, List.all_nil, Bool.and_true]
      cases mSem idx L m n <;> cases rem.all (evalA opq n) <;> cases evalA opq n a <;> simp

theorem buildSome_any (F : List Nat) (n : Node) :
    ∀ qs : List Q, (buildSome F qs).any (· n) = qs.any (fun q => bsel F q n)
  | [] => by simp [buildSome]
  | q :: qs => by
    simp only [buildSome, List.any_cons]
    cases h : build F q <;> simp [bsel, h, buildSome_any F n qs]

/-- **PROVEN** (the OR branch): every disjunct converted, the union selects exactly the
disjunction; the runtime serves it when no disjunct is an `IN`, or whenever the returned flag is
false. -/
theorem pushOr_spec (opq : Nat → Node → Bool) :
    ∀ (as : List A) (lab : Option Nat) (qs : List IQ) (k : Bool),
    (∀ a ∈ as, goodA a = true) → pushOr idx [L] as = some (lab, qs, k) →
    (lab = none ↔ as = []) ∧ (∀ L0, lab = some L0 → L0 = L) ∧ (qs = [] ↔ as = []) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels →
      (evalIQs qs).any (fun q => bsel (idx L) q n) = as.any (evalA opq n)) ∧
    ((∀ a ∈ as, isInn a = false) → canAll (evalIQs qs) = true) ∧
    (k = false → canAll (evalIQs qs) = true)
  | [], lab, qs, k, _, h => by
    simp [pushOr] at h; obtain ⟨rfl, rfl, rfl⟩ := h; simp [evalIQs, canAll]
  | a :: as, lab, qs, k, hg, h => by
    simp only [pushOr] at h
    split at h
    · next L' q lab' qs' k' hs hr =>
      split at h
      · simp at h; obtain ⟨rfl, rfl, rfl⟩ := h
        obtain ⟨hL', hex, -, hcu, -⟩ := trySingle_spec idx L opq a L' q (hg a (by simp)) hs
        subst L'
        obtain ⟨-, -, -, ih4, ih5, ih6⟩ :=
          pushOr_spec opq as _ qs' k' (fun b hb => hg b (by simp [hb])) hr
        refine ⟨by simp, fun L0 h => by simp at h; rw [← h], by simp, fun n hn hL => ?_,
          fun hni => ?_, fun hk => ?_⟩
        · simp only [evalIQs, List.any_cons, List.any_cons]
          rw [ih4 n hn hL]
          have := hex n hn hL
          simp only [qsem, idxSel, hL, decide_true, Bool.true_and] at this
          rw [this]
        · simp only [evalIQs, canAll]
          rw [hcu (hni a (by simp)), ih5 (fun b hb => hni b (by simp [hb]))]; rfl
        · simp only [Bool.or_eq_false_iff] at hk
          simp only [evalIQs, canAll]
          rw [pna_cu idx L a L q (hg a (by simp)) hs hk.1, ih6 hk.2]; rfl
      · simp at h
    · simp at h

/-- The filter shapes the headline theorem covers: AND of covered atoms with no strict string
comparison, OR of covered atoms, or a single covered atom. -/
def goodF : F → Bool
  | .atom a => goodA a
  | .and as => as.all (fun a => goodA a && !strictStr a)
  | .or as => as.all goodA

theorem hasAll_single (n : Node) : hasAll [L] n = decide (L ∈ n.labels) := by
  simp [hasAll]

theorem utilize_some (ls : List Nat) (f : F) (L' : Nat) (q : IQ) (rem : List A) (keep : Bool)
    (h : tryPushdown idx ls f = some (L', q, rem, keep)) :
    utilize idx ls f =
      (if rem.isEmpty then (if keep then .filter f (.idxScan ls L' q) else .idxScan ls L' q)
       else if keep then .filter f (.idxScan ls L' q) else .filter (.and rem) (.idxScan ls L' q)) := by
  simp [utilize, h]

theorem tp_atom (ls : List Nat) (a : A) (L' : Nat) (q : IQ) (rem : List A) (keep : Bool)
    (h : tryPushdown idx ls (.atom a) = some (L', q, rem, keep)) :
    trySingle idx ls a = some (L', q) ∧ rem = [] ∧ keep = pushedNeedsPost a := by
  simp only [tryPushdown] at h
  cases hs : trySingle idx ls a with
  | none => simp [hs] at h
  | some p => obtain ⟨L2, q2⟩ := p; simp [hs] at h; obtain ⟨rfl, rfl, rfl, rfl⟩ := h; simp

theorem tp_and (ls : List Nat) (as : List A) (L' : Nat) (q : IQ) (rem : List A) (keep : Bool)
    (h : tryPushdown idx ls (.and as) = some (L', q, rem, keep)) :
    (pushAnd idx ls as none [] false).1 = some (L', q) ∧ (pushAnd idx ls as none [] false).2.1 = rem ∧
      (pushAnd idx ls as none [] false).2.2 = keep := by
  simp only [tryPushdown] at h
  cases hs : pushAnd idx ls as none [] false with
  | mk m r =>
    obtain ⟨r, k⟩ := r
    cases m with
    | none => simp [hs] at h
    | some p => obtain ⟨L2, q2⟩ := p; simp [hs] at h; obtain ⟨rfl, rfl, rfl, rfl⟩ := h; simp

theorem tp_or (ls : List Nat) (as : List A) (L' : Nat) (q : IQ) (rem : List A) (keep : Bool)
    (h : tryPushdown idx ls (.or as) = some (L', q, rem, keep)) :
    ∃ qs, pushOr idx ls as = some (some L', qs, keep) ∧ q = .or qs ∧ rem = [] := by
  simp only [tryPushdown] at h
  cases hs : pushOr idx ls as with
  | none => simp [hs] at h
  | some p =>
    obtain ⟨lab, qs, k⟩ := p
    cases lab with
    | none => simp [hs] at h
    | some L2 => simp [hs] at h; obtain ⟨rfl, rfl, rfl, rfl⟩ := h; exact ⟨qs, rfl, rfl, rfl⟩

/-- Off the label, the rewritten plan selects nothing, as the reference does. -/
theorem utilize_off_label (opq : Nat → Node → Bool) (f : F) (n : Node) (hL : L ∉ n.labels)
    (hlab : ∀ L' q rem keep, tryPushdown idx [L] f = some (L', q, rem, keep) → L' = L) :
    (utilize idx [L] f).sel idx opq n = (reference [L] f).sel idx opq n := by
  have hr : (reference [L] f).sel idx opq n = false := by
    simp [reference, Plan.sel, hasAll_single, hL]
  rw [hr]
  unfold utilize
  split
  · simp [Plan.sel, hasAll_single, hL]
  · next L' q rem keep hp =>
    obtain rfl := hlab L' q rem keep hp
    simp only
    split <;> (try split) <;> simp [Plan.sel, idxSel, hasAll_single, hL]

/-- On the label: the rewritten plan selects what `f` does as soon as the index never loses a
row of `f`, and, when the filter is dropped (`keep` false), the runtime serves the query and the
query with the leftovers is exactly `f`. -/
theorem utilize_on_label (opq : Nat → Node → Bool) (f : F) (n : Node) (hL : L ∈ n.labels)
    (q : IQ) (rem : List A) (keep : Bool) (hp : tryPushdown idx [L] f = some (L, q, rem, keep))
    (hnf : evalF opq n f = true → qsem idx L q n = true)
    (hex : keep = false → canUtilize (evalIQ q) = true ∧
      (qsem idx L q n && rem.all (evalA opq n)) = evalF opq n f) :
    (utilize idx [L] f).sel idx opq n = (reference [L] f).sel idx opq n := by
  have hsel : (Plan.idxScan [L] L q).sel idx opq n =
      (if canUtilize (evalIQ q) then qsem idx L q n else true) := by
    simp [Plan.sel, hL, qsem, hasAll]
  have href : (reference [L] f).sel idx opq n = evalF opq n f := by
    simp [reference, Plan.sel, hasAll_single, hL]
  rw [href, utilize_some idx [L] f L q rem keep hp]
  have sel_filter : ∀ (g : F) (p : Plan), (Plan.filter g p).sel idx opq n =
      (p.sel idx opq n && evalF opq n g) := fun _ _ => rfl
  cases hk : keep
  · obtain ⟨hcu, hs⟩ := hex hk
    cases hr : rem.isEmpty <;>
      simp only [ite_true, ite_false, Bool.false_eq_true, sel_filter, hsel, hcu]
    · simpa [evalF] using hs
    · have : rem = [] := List.isEmpty_iff.mp hr
      subst this
      simpa using hs
  · have : ∀ x : Plan, (if rem.isEmpty then (if true = true then Plan.filter f x else x)
        else if true = true then Plan.filter f x else Plan.filter (.and rem) x) = Plan.filter f x := by
      intro x; split <;> rfl
    rw [this, sel_filter, hsel]
    cases hf' : evalF opq n f
    · simp
    · rw [hnf hf']; split <;> rfl

/-- **PROVEN** (headline, `utilize_index` soundness): on a single-label scan whose property
values are Null/Int/String, for filters built from `n.k op literal`, `literal op n.k`,
`n.k IN [literals]` (non-lossy Int or String literals) and unindexable conjuncts, combined by
AND (no strict string bound) or OR, the rewritten plan selects exactly the nodes the original
`Filter → NodeByLabelScan` does. The one counterexample left in `IndexCex` breaks the
`Faithful` hypothesis: a Bool value (C1). The others now agree: C2 (temporal, #3076), C7, C7', C8
(computed attribute side, #3068), C10 (array-contains in an AND, #3736), C11 (strict string bound,
#3072), and C3, C4, C5, C6, C9, C12 (folded `date()`, computed constant, two labels, a Date inside
IN, array-contains in an OR: #3064). Array-contains is covered by `IndexArray.utilize_sound_ac`. -/
theorem utilize_sound (opq : Nat → Node → Bool) (f : F) (hf : goodF f = true)
    (n : Node) (hn : Faithful n) :
    (utilize idx [L] f).sel idx opq n = (reference [L] f).sel idx opq n := by
  -- facts common to all three shapes
  have key : ∀ L' q rem keep, tryPushdown idx [L] f = some (L', q, rem, keep) →
      L' = L ∧ (∀ m : Node, Faithful m → L ∈ m.labels →
            (qsem idx L q m && rem.all (evalA opq m)) = evalF opq m f) ∧
          (keep = false → canUtilize (evalIQ q) = true) := by
    intro L' q rem keep hp
    cases f with
    | atom a =>
      obtain ⟨hs, hrem, hkeep⟩ := tp_atom idx [L] a L' q rem keep hp
      obtain ⟨rfl, hex, -, -, -⟩ := trySingle_spec idx L opq a L' q hf hs
      refine ⟨rfl, fun m hm hLm => ?_, fun hk => pna_cu idx L' a L' q hf hs (hkeep ▸ hk)⟩
      simp [hrem, hex m hm hLm, evalF]
    | and as =>
      obtain ⟨h1, h2, h3⟩ := tp_and idx [L] as L' q rem keep hp
      have hg : ∀ a ∈ as, goodA a = true ∧ strictStr a = false := fun a ha => by
        have := List.all_eq_true.mp hf a ha; simp_all
      obtain ⟨hlab, heq, -, hcu⟩ := pushAnd_spec idx L opq as none [] false hg (by simp)
      obtain ⟨rfl, -⟩ := hlab L' q h1
      refine ⟨rfl, fun m hm hLm => ?_, fun hk => (hcu (h3 ▸ hk)).2 (by simp) L' q h1⟩
      have := heq m hm hLm
      rw [h1, h2] at this
      simpa [mSem, evalF] using this
    | or as =>
      obtain ⟨qs, hpo, rfl, rfl⟩ := tp_or idx [L] as L' q rem keep hp
      obtain ⟨-, hlab, hqe, hany, -, hcan⟩ :=
        pushOr_spec idx L opq as _ qs keep (fun a ha => List.all_eq_true.mp hf a ha) hpo
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
      · simp only [evalIQ, canUtilize]
        rw [hcan hk]
        cases qs with
        | nil => exact absurd rfl hne
        | cons _ _ => rfl
  by_cases hL : L ∈ n.labels
  case neg => exact utilize_off_label idx L opq f n hL (fun L' q rem keep hp => (key L' q rem keep hp).1)
  case pos =>
    cases hp : tryPushdown idx [L] f with
    | none =>
      simp [utilize, hp, reference, Plan.sel, hasAll_single, hL]
    | some t =>
      obtain ⟨L', q, rem, keep⟩ := t
      obtain ⟨rfl, hsem, hcu⟩ := key L' q rem keep hp
      have hs := hsem n hn hL
      refine utilize_on_label idx L' opq f n hL q rem keep hp (fun hfv => ?_) (fun hk => ⟨hcu hk, hs⟩)
      rw [← hs] at hfv
      simp only [Bool.and_eq_true] at hfv
      exact hfv.1

end
end OptimizerScan.Index
