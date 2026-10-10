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

def mSem (m : Option (Nat × IQ)) (n : Node) : Bool :=
  match m with
  | none => true
  | some (_, q) => qsem idx L q n

/-- **PROVEN** (the AND branch of `try_filter_pushdown`): on a single-label scan, the merged
query and the leftover conjuncts together select exactly the conjunction; every merged query
stays on `L` and well-formed, and is usable by the runtime when no conjunct is an `IN`. -/
theorem pushAnd_spec (opq : Nat → Node → Bool) :
    ∀ (as : List A) (m : Option (Nat × IQ)) (rem : List A),
    (∀ a ∈ as, goodA a = true ∧ strictStr a = false) →
    (∀ L0 q0, m = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) →
    (∀ L0 q0, (pushAnd idx [L] as m rem).1 = some (L0, q0) → L0 = L ∧ rangeOK idx L q0 = true) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels →
      (mSem idx L (pushAnd idx [L] as m rem).1 n && (pushAnd idx [L] as m rem).2.all (evalA opq n))
        = (mSem idx L m n && rem.all (evalA opq n) && as.all (evalA opq n))) ∧
    ((∀ a ∈ as, isInn a = false) → (∀ L0 q0, m = some (L0, q0) → canUtilize (evalIQ q0) = true) →
      ∀ L0 q0, (pushAnd idx [L] as m rem).1 = some (L0, q0) → canUtilize (evalIQ q0) = true)
  | [], m, rem, _, hm => by
    refine ⟨hm, fun n _ _ => by simp [pushAnd], fun _ h => h⟩
  | a :: as, m, rem, hg, hm => by
    have hga := hg a (by simp)
    have hgs : ∀ b ∈ as, goodA b = true ∧ strictStr b = false := fun b hb => hg b (by simp [hb])
    simp only [pushAnd]
    split
    · next L' q hs =>
      obtain ⟨hL', hex, hro, hcu, -⟩ := trySingle_spec idx L opq a L' q hga.1 hs
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
      obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec opq as _ rem hgs hm'
      refine ⟨ih1, fun n hn hL => ?_, fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) ?_⟩
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
      · intro L0 q0 h
        have hcq := hcu (hni a (by simp))
        cases m with
        | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact hcq
        | some p =>
          obtain ⟨L1, q1⟩ := p
          simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
          exact cu_merge _ _ (hcm L1 q1 rfl) hcq
    · obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec opq as m (rem ++ [a]) hgs hm
      refine ⟨ih1, fun n hn hL => ?_, fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) hcm⟩
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
disjunction. -/
theorem pushOr_spec (opq : Nat → Node → Bool) :
    ∀ (as : List A) (lab : Option Nat) (qs : List IQ),
    (∀ a ∈ as, goodA a = true) → pushOr idx [L] as = some (lab, qs) →
    (lab = none ↔ as = []) ∧ (∀ L0, lab = some L0 → L0 = L) ∧ (qs = [] ↔ as = []) ∧
    (∀ n : Node, Faithful n → L ∈ n.labels →
      (evalIQs qs).any (fun q => bsel (idx L) q n) = as.any (evalA opq n)) ∧
    ((∀ a ∈ as, isInn a = false) → canAll (evalIQs qs) = true)
  | [], lab, qs, _, h => by
    simp [pushOr] at h; obtain ⟨rfl, rfl⟩ := h; simp [evalIQs, canAll]
  | a :: as, lab, qs, hg, h => by
    simp only [pushOr] at h
    split at h
    · next L' q _ qs' hs hr =>
      simp at h; obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hL', hex, -, hcu, -⟩ := trySingle_spec idx L opq a L' q (hg a (by simp)) hs
      subst L'
      obtain ⟨-, -, -, ih4, ih5⟩ :=
        pushOr_spec opq as _ qs' (fun b hb => hg b (by simp [hb])) hr
      refine ⟨by simp, fun L0 h => by simp at h; rw [← h], by simp, fun n hn hL => ?_,
        fun hni => ?_⟩
      · simp only [evalIQs, List.any_cons, List.any_cons]
        rw [ih4 n hn hL]
        have := hex n hn hL
        simp only [qsem, idxSel, hL, decide_true, Bool.true_and] at this
        rw [this]
      · simp only [evalIQs, canAll]
        rw [hcu (hni a (by simp)), ih5 (fun b hb => hni b (by simp [hb]))]; rfl
    · simp at h

theorem isArrayContains_good (a : A) (h : goodA a = true) : isArrayContains a = false := by
  match a, h with
  | .cmp _ (.prop _) (.lit _), _ => rfl
  | .cmp _ (.lit _) (.prop _), _ => rfl
  | .inn (.prop _) (.list _), _ => rfl
  | .opq _, _ => rfl

theorem needsPost_inn (a : A) (h : goodA a = true) (hi : isInn a = true) : nonIdxA a = true := by
  match a, h with
  | .inn (.prop _) (.list _), _ => rfl
  | .cmp _ (.prop _) (.lit _), _ => simp [isInn] at hi
  | .cmp _ (.lit _) (.prop _), _ => simp [isInn] at hi
  | .opq _, _ => simp [isInn] at hi

theorem cu_inList (k : Nat) (ts : List T) (hne : ts ≠ []) (hts : ts.all scalarLit = true) :
    canUtilize (evalIQ (.inList k (.list ts))) = true := by
  have hgood : ∀ t ∈ ts, isPrim (evalC t) = true ∧ isIndexable (evalC t) = true := by
    intro t ht
    have := List.all_eq_true.mp hts t ht
    rcases t with _ | ⟨_ | _ | _ | _ | _ | _⟩ | _ | _ | _ <;>
      simp_all [scalarLit, evalC, evalT, isPrim, isIndexable]
  have hfilt : (ts.map evalC).filter isPrim = ts.map evalC :=
    List.filter_eq_self.mpr (fun c hc => by
      obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hc; exact (hgood t ht).1)
  simp only [evalIQ, listVals, hfilt, canUtilize]
  have hall : ∀ cs : List V, (∀ c ∈ cs, isIndexable c = true) →
      canAll (cs.map (Q.eq k)) = true := by
    intro cs hcs
    induction cs with
    | nil => rfl
    | cons c cs ih =>
      simp only [List.map_cons, canAll, canUtilize, hcs c (by simp), Bool.true_and]
      exact ih (fun d hd => hcs d (by simp [hd]))
  rw [hall _ (fun c hc => by obtain ⟨t, ht, rfl⟩ := List.mem_map.mp hc; exact (hgood t ht).2)]
  cases ts with
  | nil => exact absurd rfl hne
  | cons _ _ => rfl

/-- The filter shapes the headline theorem covers: AND of covered atoms with no strict string
comparison, OR of covered atoms, or a single covered atom. -/
def goodF : F → Bool
  | .atom a => goodA a
  | .and as => as.all (fun a => goodA a && !strictStr a)
  | .or as => as.all goodA

theorem hasAll_single (n : Node) : hasAll [L] n = decide (L ∈ n.labels) := by
  simp [hasAll]

theorem utilize_some (ls : List Nat) (f : F) (L' : Nat) (q : IQ) (rem : List A)
    (h : tryPushdown idx ls f = some (L', q, rem)) :
    utilize idx ls f =
      (if rem.isEmpty then (if needsPost f then .filter f (.idxScan ls L' q) else .idxScan ls L' q)
       else if needsPost f then .filter f (.idxScan ls L' q) else .filter (.and rem) (.idxScan ls L' q)) := by
  simp [utilize, h]

theorem tp_atom (ls : List Nat) (a : A) (L' : Nat) (q : IQ) (rem : List A)
    (h : tryPushdown idx ls (.atom a) = some (L', q, rem)) :
    trySingle idx ls a = some (L', q) ∧ rem = (if isArrayContains a then [a] else []) := by
  simp only [tryPushdown] at h
  cases hs : trySingle idx ls a with
  | none => simp [hs] at h
  | some p => obtain ⟨L2, q2⟩ := p; simp [hs] at h; obtain ⟨rfl, rfl, rfl⟩ := h; simp

theorem tp_and (ls : List Nat) (as : List A) (L' : Nat) (q : IQ) (rem : List A)
    (h : tryPushdown idx ls (.and as) = some (L', q, rem)) :
    (pushAnd idx ls as none []).1 = some (L', q) ∧ (pushAnd idx ls as none []).2 = rem := by
  simp only [tryPushdown] at h
  cases hs : pushAnd idx ls as none [] with
  | mk m r =>
    cases m with
    | none => simp [hs] at h
    | some p => obtain ⟨L2, q2⟩ := p; simp [hs] at h; obtain ⟨rfl, rfl, rfl⟩ := h; simp

theorem tp_or (ls : List Nat) (as : List A) (L' : Nat) (q : IQ) (rem : List A)
    (h : tryPushdown idx ls (.or as) = some (L', q, rem)) :
    ∃ qs, pushOr idx ls as = some (some L', qs) ∧ q = .or qs ∧ rem = [] := by
  simp only [tryPushdown] at h
  cases hs : pushOr idx ls as with
  | none => simp [hs] at h
  | some p =>
    obtain ⟨lab, qs⟩ := p
    cases lab with
    | none => simp [hs] at h
    | some L2 => simp [hs] at h; obtain ⟨rfl, rfl, rfl⟩ := h; exact ⟨qs, rfl, rfl, rfl⟩

/-- **PROVEN** (headline, `utilize_index` soundness): on a single-label scan whose property
values are Null/Int/String, for filters built from `n.k op literal`, `literal op n.k`,
`n.k IN [literals]` (non-lossy Int or String literals) and unindexable conjuncts, combined by
AND (no strict string bound) or OR, the rewritten plan selects exactly the nodes the original
`Filter → NodeByLabelScan` does. Every counterexample in `IndexCex` breaks one hypothesis:
two labels (C5, C6), a Bool value (C1; C2, temporal, fixed by #3076), a Date literal or computed
constant (C3, C4), a computed attribute side (C7, C7', C8), a Date inside IN (C9), array-contains
(C10); C11 (strict string bound in an AND) was fixed by #3072. -/
theorem utilize_sound (opq : Nat → Node → Bool) (f : F) (hf : goodF f = true)
    (n : Node) (hn : Faithful n) :
    (utilize idx [L] f).sel idx opq n = (reference [L] f).sel idx opq n := by
  by_cases hL : L ∈ n.labels
  case neg =>
    have hr : (reference [L] f).sel idx opq n = false := by
      simp [reference, Plan.sel, hasAll_single, hL]
    rw [hr]
    unfold utilize
    have hsc : ∀ L' q, L' = L → (Plan.idxScan [L] L' q).sel idx opq n = false := by
      intro L' q h; subst h
      simp [Plan.sel, idxSel, hasAll_single, hL]
    split
    · simp [Plan.sel, hasAll_single, hL]
    · next L' q rem hp =>
      have hL' : L' = L := by
        unfold tryPushdown at hp
        cases f with
        | atom a =>
          simp only at hp; split at hp
          · next L2 q2 hs => simp at hp; obtain ⟨rfl, -⟩ := hp; exact (trySingle_spec idx L opq a _ _ hf hs).1
          · simp at hp
        | and as =>
          simp only at hp; split at hp
          · next L2 q2 rem2 he =>
            simp at hp; obtain ⟨rfl, -⟩ := hp
            have := (pushAnd_spec idx L opq as none [] (fun a ha => by
              have := List.all_eq_true.mp hf a ha; simp_all) (by simp)).1 L2 q2 (by rw [he])
            exact this.1
          · simp at hp
        | or as =>
          simp only at hp; split at hp
          · next L2 qs he =>
            simp at hp; obtain ⟨rfl, -⟩ := hp
            exact (pushOr_spec idx L opq as _ qs (fun a ha => List.all_eq_true.mp hf a ha) he).2.1 L2 rfl
          · simp at hp
      subst L'
      simp only
      split <;> (try split) <;> simp [Plan.sel, idxSel, hasAll_single, hL]
  case pos =>
    have hsel : ∀ q, (Plan.idxScan [L] L q).sel idx opq n =
        (if canUtilize (evalIQ q) then qsem idx L q n else true) := by
      intro q
      simp [Plan.sel, hasAll_single, hL, qsem, hasAll]
    have href : (reference [L] f).sel idx opq n = evalF opq n f := by
      simp [reference, Plan.sel, hasAll_single, hL]
    rw [href]
    cases hp : tryPushdown idx [L] f with
    | none => simp [utilize, hp, Plan.sel, hasAll_single, hL]
    | some t =>
      obtain ⟨L', q, rem⟩ := t
      rw [utilize_some idx [L] f L' q rem hp]
      -- facts common to all three shapes
      have key : L' = L ∧ (∀ m : Node, Faithful m → L ∈ m.labels →
            (qsem idx L q m && rem.all (evalA opq m)) = evalF opq m f) ∧
          (needsPost f = false → canUtilize (evalIQ q) = true) := by
        cases f with
        | atom a =>
          obtain ⟨hs, hrem⟩ := tp_atom idx [L] a L' q rem hp
          obtain ⟨rfl, hex, -, hcu, hin⟩ := trySingle_spec idx L opq a L' q hf hs
          refine ⟨rfl, fun m hm hLm => ?_, fun hk => ?_⟩
          · simp [hrem, isArrayContains_good a hf, hex m hm hLm, evalF]
          · cases hia : isInn a
            · exact hcu hia
            · match a, hf, hia with
              | .inn (.prop k) (.list ts), hg, _ =>
                rw [hin ts k rfl]
                simp only [needsPost] at hk
                split at hk
                · next hc =>
                  simp only [Bool.and_eq_true, Bool.not_eq_true', List.isEmpty_eq_false_iff] at hc
                  exact cu_inList k ts hc.1 hc.2
                · simp at hk
        | and as =>
          obtain ⟨h1, h2⟩ := tp_and idx [L] as L' q rem hp
          have hg : ∀ a ∈ as, goodA a = true ∧ strictStr a = false := fun a ha => by
            have := List.all_eq_true.mp hf a ha; simp_all
          obtain ⟨hlab, heq, hcu⟩ := pushAnd_spec idx L opq as none [] hg (by simp)
          obtain ⟨rfl, -⟩ := hlab L' q h1
          refine ⟨rfl, fun m hm hLm => ?_, fun hk => ?_⟩
          · have := heq m hm hLm
            rw [h1, h2] at this
            simpa [mSem, evalF] using this
          · refine hcu (fun a ha => ?_) (by simp) L' q h1
            cases hi : isInn a
            · rfl
            · have := needsPost_inn a (hg a ha).1 hi
              simp only [needsPost, List.any_eq_false] at hk
              exact absurd this (by simpa using hk a ha)
        | or as =>
          obtain ⟨qs, hpo, rfl, rfl⟩ := tp_or idx [L] as L' q rem hp
          obtain ⟨-, hlab, hqe, hany, hcan⟩ :=
            pushOr_spec idx L opq as _ qs (fun a ha => List.all_eq_true.mp hf a ha) hpo
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
            rw [hcan (fun a ha => ?_)]
            · cases qs with
              | nil => exact absurd rfl hne
              | cons _ _ => rfl
            · cases hi : isInn a
              · rfl
              · have := needsPost_inn a (List.all_eq_true.mp hf a ha) hi
                simp only [needsPost, List.any_eq_false, Bool.not_eq_true] at hk
                exact absurd this (by simpa using hk a ha)
      obtain ⟨rfl, hsem, hcu⟩ := key
      have hs := hsem n hn hL
      have sel_filter : ∀ (g : F) (p : Plan), (Plan.filter g p).sel idx opq n =
          (p.sel idx opq n && evalF opq n g) := fun _ _ => rfl
      cases hk : needsPost f <;> cases hr : rem.isEmpty <;>
        simp only [hk, hr, ite_true, ite_false, Bool.false_eq_true, sel_filter, hsel]
      · rw [if_pos (hcu hk)]; exact hs
      · have : rem = [] := List.isEmpty_iff.mp hr
        subst this
        rw [if_pos (hcu hk)]; simpa using hs
      · split <;> cases hq : qsem idx L' q n <;> cases hf' : evalF opq n f <;> simp_all
      · split <;> cases hq : qsem idx L' q n <;> cases hf' : evalF opq n f <;> simp_all

end
end OptimizerScan.Index
