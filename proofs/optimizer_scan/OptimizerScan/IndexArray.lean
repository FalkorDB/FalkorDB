import OptimizerScan.IndexPipeline
/-!
# Array-contains (`value IN n.prop`) through `try_filter_pushdown`

`utilize_sound` (`IndexPipeline`) assumes `goodA` conjuncts, which exclude `x IN n.k`. This file
closes that gap for the AND branch and the single-atom branch (the `try_filter_pushdown/array-contains`
row): an array-contains conjunct is pushed as `IndexQuery::ArrayContains` (`Q.contains`) **and its
filter is kept** (`is_array_contains`, `utilize_index.rs`), because the array tree answers with
false positives (`1 IN [true]`: Bool and Int share the numeric key, `arrEnc`).

* `contains_no_fn`: the index never loses a row — if `x IN n.k` is true, the array tree has it.
* `pushAnd_spec_ac`: the AND branch with array-contains conjuncts selects exactly the conjunction
  (the kept filter removes the false positives), still on `L`, well-formed and runtime-usable.
* `utilize_sound_ac` (headline): a rewritten AND of covered atoms and array-contains atoms, or a
  single array-contains atom, selects exactly what `Filter → NodeByLabelScan` does, on nodes whose
  values are Null/Int/String/lists (`Faithful`).
* `tryPushdown_or_ac_drops_filter` / `Cex.C12`: the **OR** branch does not keep the filter of an
  array-contains disjunct (`rem = []`, `needs_post_filter` false), so the false positive survives:
  CONFIRMED live (`1 IN n.arr OR n.k = 9`, node `{arr:[true], k:0}`).
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
  cases c <;> simp_all [goodC, evalIQ, evalC, evalT, canUtilize]

theorem bool_ac_none (x r e s : Bool) (h : e = true → x = true) :
    (x && (r && e) && s) = (true && r && (e && s)) := by
  cases x <;> cases r <;> cases e <;> cases s <;> simp_all

theorem bool_ac_some (q x r e s : Bool) (h : e = true → x = true) :
    (q && x && (r && e) && s) = (q && r && (e && s)) := by
  cases q <;> cases x <;> cases r <;> cases e <;> cases s <;> simp_all

/-- The conjuncts the AND branch handles here: a covered atom (no strict string bound) or an
array-contains. -/
def okA (a : A) : Prop := (goodA a = true ∧ strictStr a = false) ∨ goodAC a = true

/-- **PROVEN** (the AND branch of `try_filter_pushdown` with array-contains conjuncts): on a
single-label scan, the merged query and the leftover conjuncts (which now include every pushed
array-contains conjunct) together select exactly the conjunction; the merged query stays on `L`,
well-formed, and usable by the runtime when no pushed conjunct is an `IN` list. -/
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
      rcases hga with hga | hac
      · -- a covered atom: exactly as in `pushAnd_spec`
        obtain ⟨hL', hex, hro, hcu, -⟩ := trySingle_spec idx L opq a L' q hga.1 hs
        subst L'
        simp only [isArrayContains_good a hga.1, Bool.false_eq_true, ite_false]
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
        obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec_ac opq as _ rem hgs hm'
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
          have hcq : canUtilize (evalIQ q) = true := by
            rcases hni a (by simp) with hi | hi
            · exact hcu hi
            · rw [good_not_ac a hga.1] at hi; exact absurd hi (by simp)
          cases m with
          | none => simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h; exact hcq
          | some p =>
            obtain ⟨L1, q1⟩ := p
            simp [mergeInto] at h; obtain ⟨rfl, rfl⟩ := h
            exact cu_merge _ _ (hcm L1 q1 rfl) hcq
      · -- an array-contains: pushed, and kept among the leftovers
        obtain ⟨k, c, rfl, hL', rfl, hk, hcg⟩ := trySingle_ac idx L a hac L' q hs
        subst L'
        simp only [isArrayContains_ac _ hac, ite_true]
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
        obtain ⟨ih1, ih2, ih3⟩ := pushAnd_spec_ac opq as _ (rem ++ [.inn (.lit c) (.term (.prop k))]) hgs hm'
        refine ⟨ih1, fun n hn hL => ?_, fun hni hcm => ih3 (fun b hb => hni b (by simp [hb])) ?_⟩
        · rw [ih2 n hn hL]
          have hex := contains_no_fn idx L opq k c hcg n hk hL
          cases m with
          | none =>
            simp only [mSem, mergeInto, List.all_append, List.all_cons, List.all_nil, Bool.and_true]
            exact bool_ac_none _ _ _ _ hex
          | some p =>
            obtain ⟨L1, q1⟩ := p
            have := hm L1 q1 rfl
            simp only [mSem, mergeInto]
            rw [sel_merge idx L n _ _ this.2 hro hn hL]
            simp only [List.all_append, List.all_cons, List.all_nil, Bool.and_true]
            exact bool_ac_some _ _ _ _ _ hex
        · intro L0 q0 h
          have hcq := cu_contains k c hcg
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

/-- The filter shapes `utilize_sound_ac` adds: a single array-contains atom, or an AND whose
conjuncts are covered atoms (no strict string bound) or array-contains atoms. -/
def goodFAC : F → Bool
  | .atom a => goodAC a
  | .and as => as.all (fun a => (goodA a && !strictStr a) || goodAC a)
  | .or _ => false

/-- **PROVEN** (headline for array-contains, #3736): on a single-label scan whose property values
are Null/Int/String/lists of scalars, a filter that is a single `x IN n.k` or an AND of covered atoms
and `x IN n.k` atoms (literal `x`: non-lossy Int, String, Bool) is rewritten into a plan that
selects exactly the nodes the original `Filter → NodeByLabelScan` does. The array tree's false
positives (`1 IN [true]`) are removed by the kept filter: this is what `pushAnd`'s
`rem ++ [a]` buys. The OR branch is NOT covered, because it does not keep it (`C12`). -/
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
        | or as => simp [goodFAC] at hf
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
          (needsPost f = false → canUtilize (evalIQ q) = true) := by
        cases f with
        | atom a =>
          have hac : goodAC a = true := by simpa [goodFAC] using hf
          obtain ⟨hs, hrem⟩ := tp_atom idx [L] a L' q rem hp
          obtain ⟨k, c, rfl, hLL, rfl, hk, hcg⟩ := trySingle_ac idx L a hac L' q hs
          subst L'
          refine ⟨rfl, fun m hm hLm => ?_, fun _ => cu_contains k c hcg⟩
          have hex := contains_no_fn idx L opq k c hcg m hk hLm
          simp only [hrem, isArrayContains_ac _ hac, ite_true, List.all_cons, List.all_nil,
            Bool.and_true, evalF]
          cases ha : evalA opq m (.inn (.lit c) (.term (.prop k))) <;> simp_all
        | and as =>
          obtain ⟨h1, h2⟩ := tp_and idx [L] as L' q rem hp
          obtain ⟨hlab, heq, hcu⟩ := pushAnd_spec_ac idx L opq as none [] hok (by simp)
          obtain ⟨rfl, -⟩ := hlab L' q h1
          refine ⟨rfl, fun m hm hLm => ?_, fun hk => ?_⟩
          · have := heq m hm hLm
            rw [h1, h2] at this
            simpa [mSem, evalF] using this
          · refine hcu (fun a ha => ?_) (by simp) L' q h1
            have hoka := hok a ha
            rcases hoka with hg | hac
            · cases hi : isInn a
              · exact Or.inl rfl
              · have := needsPost_inn a hg.1 hi
                simp only [needsPost, List.any_eq_false] at hk
                exact absurd this (by simpa using hk a ha)
            · exact Or.inr hac
        | or as => simp [goodFAC] at hf
      obtain ⟨rfl, hsem, hcu⟩ := key
      have hs := hsem n hn hL
      have sel_filter : ∀ (g : F) (p : Plan), (Plan.filter g p).sel idx opq n =
          (p.sel idx opq n && evalF opq n g) := fun _ _ => rfl
      cases hk : needsPost f <;> cases hr : rem.isEmpty <;>
        simp only [ite_true, ite_false, Bool.false_eq_true, sel_filter, hsel]
      · rw [ite_eq_left (hcu hk)]; exact hs
      · have : rem = [] := List.isEmpty_iff.mp hr
        subst this
        rw [ite_eq_left (hcu hk)]; simpa using hs
      · split <;> cases hq : qsem idx L' q n <;> cases hf' : evalF opq n f <;> simp_all
      · split <;> cases hq : qsem idx L' q n <;> cases hf' : evalF opq n f <;> simp_all

/-- **OR branch does not keep the array-contains filter** (confirmed bug, shared with C): `pushOr`
converts the disjunct `1 IN n.arr` into the index query `contains` and `tryPushdown` returns no
leftover filter, while `needs_post_filter` has nothing to flag. Keys: `3 = arr`, `0 = k`. -/
theorem tryPushdown_or_ac_drops_filter :
    tryPushdown (fun _ => [3, 0]) [0]
        (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))]) =
      some (0, .or [.contains 3 (.lit (.i 1)), .eq 0 (.lit (.i 9))], []) ∧
    needsPost (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))]) = false :=
  ⟨rfl, rfl⟩

/-- C12 (CONFIRMED, shared with C): `1 IN n.arr OR n.k = 9` on `{arr: [true], k: 0}` — the plan has
no Filter, the array tree holds `true` under the key of `1`, the row is returned with the index and
not without (Rust with index `[true],[1],[2]`; without `[1],[2]`; C with index `[true],[1],[2]`). -/
example : (utilize (fun _ => [3, 0]) [0]
      (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))])).sel
      (fun _ => [3, 0]) (fun _ _ => true) ⟨[0], fun k => if k = 3 then .arr [.b true] else if k = 0 then .i 0 else .null⟩ = true ∧
    (reference [0]
      (.or [.inn (.lit (.i 1)) (.term (.prop 3)), .cmp .eq (.prop 0) (.lit (.i 9))])).sel
      (fun _ => [3, 0]) (fun _ _ => true) ⟨[0], fun k => if k = 3 then .arr [.b true] else if k = 0 then .i 0 else .null⟩ = false := by
  decide

end
end OptimizerScan.Index
