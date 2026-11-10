import VersionedMatrix.Tensor

namespace VMTensor

/-! ### `me` row as a set -/

def insAll (me : List Nat) (is : List Nat) : List Nat := is.foldl meIns me

theorem mem_meIns {me : List Nat} {i x : Nat} : x ∈ meIns me i ↔ x ∈ me ∨ x = i := by
  unfold meIns; split <;> simp_all <;> grind

theorem nodup_meIns {me : List Nat} (h : me.Nodup) (i : Nat) : (meIns me i).Nodup := by
  unfold meIns; split
  · exact h
  · rename_i hi
    exact List.nodup_append.2 ⟨h, by simp, by simp; intro a ha e; subst e; exact hi ha⟩

theorem length_meIns {me : List Nat} {i : Nat} (h : i ∉ me) : (meIns me i).length = me.length + 1 := by
  simp [meIns, h]

theorem insAll_spec : ∀ (is : List Nat) (me : List Nat), me.Nodup →
    (∀ x, x ∈ insAll me is ↔ x ∈ me ∨ x ∈ is) ∧ (insAll me is).Nodup ∧
    me.length ≤ (insAll me is).length ∧
    (is.Nodup → (∀ x ∈ is, x ∉ me) → (insAll me is).length = me.length + is.length)
  | [], me, h => by simp [insAll, h]
  | i :: is, me, h => by
    have ih := insAll_spec is (meIns me i) (nodup_meIns h i)
    have e : insAll me (i :: is) = insAll (meIns me i) is := rfl
    rw [e]
    have hl : me.length ≤ (meIns me i).length := by unfold meIns; split <;> simp
    refine ⟨fun x => by rw [ih.1, mem_meIns]; simp; grind, ih.2.1, Nat.le_trans hl ih.2.2.1, ?_⟩
    intro hnd hdis
    have hi : i ∉ me := hdis i (by simp)
    rw [ih.2.2.2 (List.nodup_cons.1 hnd).2 (by
      intro x hx hm; rw [mem_meIns] at hm
      rcases hm with hm | rfl
      · exact hdis x (by simp [hx]) hm
      · exact (List.nodup_cons.1 hnd).1 hx), length_meIns hi]
    simp; omega

theorem fold_occ_known : ∀ (is : List Nat) (me : List Nat) (e : Option Entry),
    is.foldl occupied (me, e, true) = (insAll me is, e, true)
  | [], _, _ => rfl
  | i :: is, me, e => by
    simp only [List.foldl]
    have : occupied (me, e, true) i = (meIns me i, e, true) := by
      unfold occupied; split <;> simp_all
    rw [this, fold_occ_known is]; rfl

theorem fold_occ_pending (me : List Nat) (f : Nat) (msk : Option Val) :
    ∀ (is : List Nat), is.foldl occupied (me, some ⟨.id f, msk⟩, false) =
      match is with
      | [] => (me, some ⟨.id f, msk⟩, false)
      | j :: js => (insAll (meIns (meIns me f) j) js, some ⟨.multi, msk⟩, true)
  | [] => rfl
  | j :: js => by
    simp only [List.foldl]
    have : occupied (me, some ⟨.id f, msk⟩, false) j = (meIns (meIns me f) j, some ⟨.multi, msk⟩, true) := by
      simp [occupied]
    rw [this, fold_occ_known]

/-! ### Add -/

/-- The edge-list reference for an add: the old edges plus the new ids. -/
def AddOk (s : PairSt) (ids : List Nat) : Prop := ids.Nodup ∧ ∀ x ∈ ids, x ∉ edges s

/-- A row built from `base` (duplicate-free) by inserting a batch. -/
theorem row_facts (base is : List Nat) (hb : base.Nodup) (hnd : is.Nodup)
    (hdis : ∀ x ∈ is, x ∉ base) :
    ∃ ME, insAll base is = ME ∧ ME.Nodup ∧ ME.length = base.length + is.length ∧
      ∀ x, x ∈ ME ↔ x ∈ base ∨ x ∈ is := by
  have sp := insAll_spec is base hb
  exact ⟨_, rfl, sp.2.1, sp.2.2.2 hnd hdis, sp.1⟩

macro "close_leaf" : tactic =>
  `(tactic| (refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, fun x => ?_⟩ <;>
     simp_all [effV, edges, meIns] <;> grind))

/-- **Add is correct.** Adding any batch of fresh ids to a pair in any
reachable state keeps every invariant, and the pair's edges become exactly
the old ones plus the batch. -/
theorem addBatch_correct (s : PairSt) (h : Inv s) (ids : List Nat) (hok : AddOk s ids) :
    Inv (addBatch s ids) ∧ (∀ x, x ∈ edges (addBatch s ids) ↔ x ∈ edges s ∨ x ∈ ids) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  obtain ⟨hnd, hdis⟩ := hok
  cases ids with
  | nil => exact ⟨⟨h1, h2, h3, h4, h5, h6, h7⟩, by simp [addBatch]⟩
  | cons i is =>
  have hi_is : i ∉ is := (List.nodup_cons.1 hnd).1
  have hnd_is : is.Nodup := (List.nodup_cons.1 hnd).2
  obtain ⟨m, dp, dm, mt, me⟩ := s
  cases dp with
  | some v =>
    have hdm : dm = false := h2 rfl
    subst hdm
    cases v with
    | multi =>
      have hl : 2 ≤ me.length := h4 (by simp [effV])
      have hdis' : ∀ x ∈ i :: is, x ∉ me := by simpa [edges, effV] using hdis
      obtain ⟨ME, hME, n1, l1, m1⟩ := row_facts me (i :: is) h6 hnd hdis'
      have e : addBatch ⟨m, some .multi, false, mt, me⟩ (i :: is) = ⟨m, some .multi, false, mt, ME⟩ := by
        simp only [addBatch, vacant, fold_occ_known, writeEntry]; rw [← hME]; rfl
      rw [e]; simp at l1; close_leaf
    | id c =>
      have hme : me = [] := h5 (by simp [effV])
      subst hme
      have hdis' : ∀ x ∈ i :: is, x ∉ [c] := by simpa [edges, effV] using hdis
      obtain ⟨ME, hME, n1, l1, m1⟩ := row_facts [c] (i :: is) (by simp) hnd hdis'
      simp at l1
      have e : addBatch ⟨m, some (.id c), false, mt, []⟩ (i :: is) =
          writeEntry ⟨m, some (.id c), false, mt, []⟩ ME (some ⟨.multi, m⟩) := by
        simp only [addBatch, vacant, Option.isSome_some, ite_true, fold_occ_known]
        rw [← hME]; rfl
      rw [e]
      cases m with
      | none => simp only [writeEntry]; close_leaf
      | some w => cases w <;> simp only [writeEntry, reduceCtorEq, ite_true, ite_false] <;> close_leaf
  | none =>
    cases hdmv : dm with
    | true =>
      have hm : m.isSome = true := h1 hdmv
      subst hdmv
      obtain ⟨w, rfl⟩ := Option.isSome_iff_exists.1 hm
      have hme : me = [] := h5 (by simp [effV])
      subst hme
      have hdis' : ∀ x ∈ i :: is, x ∉ ([] : List Nat) := by simp
      cases is with
      | nil =>
        have e : addBatch ⟨some w, none, true, mt, []⟩ [i] =
            writeEntry ⟨some w, none, true, mt, []⟩ [] (some ⟨.id i, some w⟩) := by
          simp [addBatch, vacant]
        rw [e]; cases w <;> simp only [writeEntry] <;> split <;> close_leaf
      | cons j js =>
        obtain ⟨ME, hME, n1, l1, m1⟩ := row_facts [] (i :: j :: js) (by simp) hnd hdis'
        simp at l1
        have e : addBatch ⟨some w, none, true, mt, []⟩ (i :: j :: js) =
            writeEntry ⟨some w, none, true, mt, []⟩ ME (some ⟨.multi, some w⟩) := by
          simp only [addBatch, vacant, ite_true, fold_occ_pending]
          rw [← hME]; rfl
        rw [e]; cases w <;> simp only [writeEntry] <;> split <;> close_leaf
    | false =>
      subst hdmv
      cases m with
      | none =>
        have hme : me = [] := h5 (by simp [effV])
        subst hme
        have hdis' : ∀ x ∈ i :: is, x ∉ ([] : List Nat) := by simp
        cases is with
        | nil =>
          have e : addBatch ⟨none, none, false, mt, []⟩ [i] = ⟨none, some (.id i), false, true, []⟩ := by
            simp [addBatch, vacant, writeEntry]
          rw [e]; close_leaf
        | cons j js =>
          obtain ⟨ME, hME, n1, l1, m1⟩ := row_facts [] (i :: j :: js) (by simp) hnd hdis'
          simp at l1
          have e : addBatch ⟨none, none, false, mt, []⟩ (i :: j :: js) =
              ⟨none, some .multi, false, true, ME⟩ := by
            simp only [addBatch, vacant, Bool.false_eq_true, ite_false, fold_occ_pending, writeEntry]
            rw [← hME]; rfl
          rw [e]; close_leaf
      | some w =>
        cases w with
        | multi =>
          have hl : 2 ≤ me.length := h4 (by simp [effV])
          have hdis' : ∀ x ∈ i :: is, x ∉ me := by simpa [edges, effV] using hdis
          obtain ⟨ME, hME, n1, l1, m1⟩ := row_facts me (i :: is) h6 hnd hdis'
          have e : addBatch ⟨some .multi, none, false, mt, me⟩ (i :: is) =
              ⟨some .multi, none, false, mt, ME⟩ := by
            simp only [addBatch, vacant, Bool.false_eq_true, ite_false, fold_occ_known, writeEntry]
            rw [← hME]; rfl
          rw [e]; simp at l1; close_leaf
        | id c =>
          have hme : me = [] := h5 (by simp [effV])
          subst hme
          have hdis' : ∀ x ∈ i :: is, x ∉ [c] := by simpa [edges, effV] using hdis
          obtain ⟨ME, hME, n1, l1, m1⟩ := row_facts [c] (i :: is) (by simp) hnd hdis'
          simp at l1
          have e : addBatch ⟨some (.id c), none, false, mt, []⟩ (i :: is) =
              ⟨some (.id c), some .multi, false, true, ME⟩ := by
            simp only [addBatch, vacant, Bool.false_eq_true, ite_false, Option.isSome_none,
              fold_occ_known, writeEntry]
            rw [← hME]; rfl
          rw [e]; close_leaf

/-! ### Remove -/

def rem (L pre : List Nat) : List Nat := L.filter (fun x => !(decide (x ∈ pre)))

theorem mem_rem {L pre : List Nat} {x : Nat} : x ∈ rem L pre ↔ x ∈ L ∧ x ∉ pre := by
  simp [rem]

theorem rem_nodup {L : List Nat} (h : L.Nodup) (pre : List Nat) : (rem L pre).Nodup :=
  List.Pairwise.filter _ h

theorem rem_step {L : List Nat} (h : L.Nodup) (pre : List Nat) (id : Nat) :
    rem L (pre ++ [id]) = (rem L pre).erase id := by
  rw [List.Nodup.erase_eq_filter (rem_nodup h pre), rem, rem, List.filter_filter]
  apply List.filter_congr; intro x _; simp; grind

/-- What `remove_all`'s plan replay has worked out for a multi-edge pair whose
`me` row is `L`, after replaying the removals `pre`. -/
structure Q (L pre : List Nat) (st : Plan × List Nat) : Prop where
  pre_sub : ∀ x ∈ pre, x ∈ L
  many : 2 ≤ (rem L pre).length → st.1 = .multi (rem L pre) ∧ ∀ x, x ∈ st.2 ↔ x ∈ pre
  one : ∀ last, rem L pre = [last] →
    st.1 = .single last true ∧ ∀ x, x ∈ st.2 ↔ x ∈ pre ∨ x = last
  none : rem L pre = [] → st.1 = .emptied ∧ ∀ x, x ∈ st.2 ↔ x ∈ L

theorem q_step {L : List Nat} (hL : L.Nodup) {pre : List Nat} {st : Plan × List Nat}
    (hq : Q L pre st) {id : Nat} (hid : id ∈ L) (hnp : id ∉ pre) :
    Q L (pre ++ [id]) (planStep st id) := by
  obtain ⟨ps, qm, q1, q0⟩ := hq
  have hin : id ∈ rem L pre := mem_rem.2 ⟨hid, hnp⟩
  have hstep := rem_step hL pre id
  have hlen : (rem L (pre ++ [id])).length + 1 = (rem L pre).length := by
    rw [hstep, List.length_erase_of_mem hin]
    have : 0 < (rem L pre).length := List.length_pos_of_mem hin
    omega
  have ps' : ∀ x ∈ pre ++ [id], x ∈ L := by
    intro x hx; simp at hx; rcases hx with hx | rfl
    · exact ps x hx
    · exact hid
  by_cases h2 : 2 ≤ (rem L pre).length
  · obtain ⟨hP, hdel⟩ := qm h2
    obtain ⟨P, del⟩ := st
    simp only at hP hdel; subst hP
    simp only [planStep]; rw [if_pos hin]
    generalize hE : (rem L pre).erase id = E at hstep
    refine ⟨ps', ?_, ?_, ?_⟩ <;> simp only [hstep]
    · intro h2'
      cases E with
      | nil => simp at h2'
      | cons a t =>
        cases t with
        | nil => simp at h2'
        | cons b u => exact ⟨rfl, fun x => by simp [hdel]⟩
    · intro last hl; rw [hl]; exact ⟨rfl, fun x => by simp [hdel]; grind⟩
    · intro h0; rw [hstep, h0] at hlen; simp at hlen; omega
  · by_cases h1 : (rem L pre).length = 1
    · obtain ⟨last, hl⟩ := List.length_eq_one_iff.1 h1
      obtain ⟨hP, hdel⟩ := q1 last hl
      have hlast : id = last := by rw [hl] at hin; simpa using hin
      subst hlast
      obtain ⟨P, del⟩ := st
      simp only at hP hdel; subst hP
      have h0 : rem L (pre ++ [id]) = [] := by rw [hstep, hl]; simp
      refine ⟨ps', fun h => by rw [h0] at h; simp at h, fun l h => by rw [h0] at h; simp at h, ?_⟩
      intro _
      refine ⟨by simp [planStep], fun x => ?_⟩
      simp only [planStep, ite_true]
      rw [hdel]
      constructor
      · rintro (h | rfl)
        · exact ps x h
        · exact hid
      · intro hx
        by_cases hxp : x ∈ pre
        · exact Or.inl hxp
        · right
          have : x ∈ rem L pre := mem_rem.2 ⟨hx, hxp⟩
          rw [hl] at this; simpa using this
    · have : (rem L pre).length = 0 := by omega
      have := List.length_pos_of_mem hin; omega

theorem q_fold {L : List Nat} (hL : L.Nodup) : ∀ (ids pre : List Nat) (st : Plan × List Nat),
    Q L pre st → ids.Nodup → (∀ x ∈ ids, x ∈ L) → (∀ x ∈ ids, x ∉ pre) →
    Q L (pre ++ ids) (ids.foldl planStep st)
  | [], pre, st, hq, _, _, _ => by simpa using hq
  | id :: ids, pre, st, hq, hnd, hsub, hdis => by
    have h1 := q_step hL hq (hsub id (by simp)) (hdis id (by simp))
    have := q_fold hL ids (pre ++ [id]) _ h1 (List.nodup_cons.1 hnd).2
      (fun x hx => hsub x (by simp [hx]))
      (fun x hx hp => by
        simp at hp; rcases hp with hp | rfl
        · exact hdis x (by simp [hx]) hp
        · exact (List.nodup_cons.1 hnd).1 hx)
    simpa using this

/-- The ids removed must be live edges of the pair, without repeats
(`delete_relationships` resolves each id once, `refuse_recycled` rejects
dead ones). -/
def RemOk (s : PairSt) (ids : List Nat) : Prop := ids.Nodup ∧ ∀ x ∈ ids, x ∈ edges s

theorem foldl_plan_absent : ∀ (ids : List Nat) (del : List Nat),
    ids.foldl planStep (.absent, del) = (.absent, del)
  | [], _ => rfl
  | _ :: ids, del => by simp only [List.foldl, planStep]; exact foldl_plan_absent ids del

theorem foldl_plan_emptied : ∀ (ids : List Nat) (del : List Nat),
    ids.foldl planStep (.emptied, del) = (.emptied, del)
  | [], _ => rfl
  | _ :: ids, del => by simp only [List.foldl, planStep]; exact foldl_plan_emptied ids del

/-- **Remove is correct.** Removing any batch of live edges keeps every
invariant, and the pair's edges become exactly the old ones minus the batch —
including the demotion of a multi-edge pair back to one inline id, and the
cancel back to a clean committed state. -/
theorem removeBatch_correct (s : PairSt) (h : Inv s) (ids : List Nat) (hok : RemOk s ids) :
    Inv (removeBatch s ids) ∧
      (∀ x, x ∈ edges (removeBatch s ids) ↔ x ∈ edges s ∧ x ∉ ids) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  obtain ⟨hnd, hsub⟩ := hok
  obtain ⟨m, dp, dm, mt, me⟩ := s
  cases he : effV ⟨m, dp, dm, mt, me⟩ with
  | none =>
    have hme : me = [] := h5 (by simp [he])
    subst hme
    have hids : ids = [] := by
      cases ids with
      | nil => rfl
      | cons x _ => have := hsub x (by simp); simp [edges, he] at this
    subst hids
    simp only [removeBatch, initPlan, he, List.foldl, writePlan, dropMe]
    refine ⟨⟨h1, h2, h3, h4, h5, h6, h7⟩, fun x => by simp [edges, he]⟩
  | some v =>
    cases v with
    | id c =>
      have hme : me = [] := h5 (by simp [he])
      subst hme
      have hsub' : ∀ x ∈ ids, x = c := by intro x hx; have := hsub x hx; simpa [edges, he] using this
      cases ids with
      | nil =>
        simp only [removeBatch, initPlan, he, List.foldl, writePlan, dropMe]
        exact ⟨⟨h1, h2, h3, h4, h5, h6, h7⟩, fun x => by simp [edges, he]⟩
      | cons x xs =>
        have hx := hsub' x (by simp); subst hx
        have hxs : xs = [] := by
          cases xs with
          | nil => rfl
          | cons y _ =>
            have := hsub' y (by simp); subst this
            simp at hnd
        subst hxs
        have e : removeBatch ⟨m, dp, dm, mt, []⟩ [x] =
            ⟨m, none, if m.isSome then true else dm, false, []⟩ := by
          simp [removeBatch, initPlan, he, planStep, writePlan, dropMe]
        rw [e]
        simp only [effV] at he
        refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, fun y => ?_⟩ <;>
          (cases m <;> cases dp <;> cases dm <;> simp_all [effV, edges])
    | multi =>
      have hl : 2 ≤ me.length := h4 he
      have hsubL : ∀ x ∈ ids, x ∈ me := by intro x hx; have := hsub x hx; simpa [edges, he] using this
      have hq0 : Q me [] (.multi me, []) := by
        refine ⟨by simp, fun _ => ?_, fun last hl' => ?_, fun h0 => ?_⟩
        · have : rem me [] = me := by simp [rem]
          rw [this]; exact ⟨rfl, by simp⟩
        · have : rem me [] = me := by simp [rem]
          rw [this] at hl'; subst hl'; simp at hl
        · have : rem me [] = me := by simp [rem]
          rw [this] at h0; subst h0; simp at hl
      have hq := q_fold h6 ids [] _ hq0 hnd hsubL (by simp)
      simp only [List.nil_append] at hq
      obtain ⟨_, qm, q1, q0⟩ := hq
      have hr : removeBatch ⟨m, dp, dm, mt, me⟩ ids =
          writePlan ⟨m, dp, dm, mt, me⟩ (ids.foldl planStep (.multi me, [])).2
            (ids.foldl planStep (.multi me, [])).1 := by
        simp [removeBatch, initPlan, he]
      rw [hr]
      generalize ids.foldl planStep (.multi me, []) = st at qm q1 q0
      obtain ⟨P, del⟩ := st
      simp only at qm q1 q0
      have hmt : mt = true := by have := h7; simp only at this; rw [he] at this; simpa using this
      have hdel_me : ∀ x, x ∈ dropMe me del ↔ x ∈ me ∧ x ∉ del := by
        intro x; simp [dropMe]
      have hremL : ∀ x, x ∈ rem me ids ↔ x ∈ me ∧ x ∉ ids := fun x => mem_rem
      -- the effective state is multi: dp = M (dm clear), or m = M with no dp and no dm
      have hshape : (dp = some .multi ∧ dm = false) ∨ (dp = none ∧ dm = false ∧ m = some .multi) := by
        cases dp with
        | some w => simp [effV] at he; exact Or.inl ⟨by simp [he], h2 rfl⟩
        | none => cases dm <;> simp_all [effV]
      by_cases hk2 : 2 ≤ (rem me ids).length
      · obtain ⟨rfl, hd⟩ := qm hk2
        simp only [writePlan]
        have hdm' : ∀ x, x ∈ dropMe me del ↔ x ∈ rem me ids := by
          intro x; rw [hdel_me, hremL, hd]
        have hlen : (dropMe me del).length = (rem me ids).length := by
          apply List.Perm.length_eq
          exact (List.perm_ext_iff_of_nodup (List.Pairwise.filter _ h6) (rem_nodup h6 _)).2 hdm'
        refine ⟨⟨h1, h2, h3, fun _ => by dsimp only; omega, fun hne => absurd he hne,
          List.Pairwise.filter _ h6, by simpa [effV] using h7⟩, fun x => ?_⟩
        simp only [edges, effV] at he ⊢
        simp only [he]; rw [hdm', hremL]
      · by_cases hk1 : (rem me ids).length = 1
        · obtain ⟨last, hl1⟩ := List.length_eq_one_iff.1 hk1
          obtain ⟨rfl, hd⟩ := q1 last hl1
          have hlast : last ∈ me ∧ last ∉ ids := (hremL last).1 (by rw [hl1]; simp)
          have hempty : dropMe me del = [] := by
            apply List.eq_nil_iff_forall_not_mem.2
            intro x hx
            rw [hdel_me, hd] at hx
            have : x ∉ ids := fun h => hx.2 (Or.inl h)
            have : x ∈ rem me ids := (hremL x).2 ⟨hx.1, this⟩
            rw [hl1] at this; simp at this; exact hx.2 (Or.inr this)
          simp only [writePlan]
          split
          · rename_i hm
            rcases hshape with ⟨rfl, rfl⟩ | ⟨_, _, hm'⟩
            · rw [hempty]
              refine ⟨⟨by simp [hm], by simp, by simp, by simp [effV, hm], by simp, by simp,
                by simp [effV, hm, hmt]⟩, fun x => ?_⟩
              simp [edges, effV, hm]; rw [← hremL, hl1]; simp
            · rw [hm] at hm'; cases hm'
          · rename_i hm
            rw [hempty]
            rcases hshape with ⟨rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩
            · refine ⟨⟨by simp, by simp, fun v hv => by simp at hv; subst hv; exact hm,
                by simp [effV], by simp, by simp, by simp [effV, hmt]⟩, fun x => ?_⟩
              simp [edges, effV]; rw [← hremL, hl1]; simp
            · refine ⟨⟨by simp, by simp, fun v hv => by simp at hv; subst hv; simp,
                by simp [effV], by simp, by simp, by simp [effV, hmt]⟩, fun x => ?_⟩
              simp [edges, effV]; rw [← hremL, hl1]; simp
        · have hk0 : rem me ids = [] := by
            apply List.eq_nil_of_length_eq_zero; omega
          obtain ⟨rfl, hd⟩ := q0 hk0
          have hempty : dropMe me del = [] := by
            apply List.eq_nil_iff_forall_not_mem.2
            intro x hx; rw [hdel_me, hd] at hx; exact hx.2 hx.1
          simp only [writePlan]
          rw [hempty]
          rcases hshape with ⟨rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩
          · have hall : ∀ x, x ∈ me → x ∈ ids := by
              intro x hx; have : x ∈ rem me ids → False := by rw [hk0]; simp
              exact Classical.byContradiction fun hn => this ((hremL x).2 ⟨hx, hn⟩)
            refine ⟨⟨?_, by simp, by simp, ?_, by simp, by simp, ?_⟩, fun x => ?_⟩ <;>
              cases m <;> simp [effV, edges] <;> exact hall x
          · refine ⟨⟨by simp, by simp, by simp, by simp [effV], by simp, by simp, by simp [effV]⟩,
              fun x => ?_⟩
            simp [edges, effV]
            intro hx; have : x ∈ rem me ids → False := by rw [hk0]; simp
            exact Classical.byContradiction fun hn => this ((hremL x).2 ⟨hx, hn⟩)

/-! ### Encoding, fast remove, fold, count, backward iteration -/

/-- **Single vs multi encoding.** Under the invariants the inline value is
`MULTI_EDGE` exactly when the pair has ≥ 2 edges, an inline id exactly when
it has one, absent exactly when it has none; the edge list has no repeats. -/
theorem encoding (s : PairSt) (h : Inv s) :
    (effV s = some .multi ↔ 2 ≤ (edges s).length) ∧
    ((∃ i, effV s = some (.id i)) ↔ (edges s).length = 1) ∧
    (effV s = none ↔ edges s = []) ∧ (edges s).Nodup := by
  have hm := h.multi_me
  unfold edges
  cases he : effV s with
  | none => simp
  | some v =>
    cases v with
    | id i => simp
    | multi =>
      have := hm he
      refine ⟨by simp [this], ?_, ?_, h.me_nd⟩
      · simp; omega
      · simp; intro h0; simp [h0] at this

/-- **Fast remove.** When no pair has an `me` row (`!has_multi_edge()`), every
pair is single or absent, so tombstoning it deletes exactly its edges. -/
theorem removeFast_correct (s : PairSt) (h : Inv s) (hme : s.me = []) (ids : List Nat)
    (hok : RemOk s ids) (hne : ids ≠ []) :
    Inv (removeFast s) ∧ edges (removeFast s) = [] ∧ ∀ x, x ∈ edges s ↔ x ∈ ids := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  obtain ⟨hnd, hsub⟩ := hok
  obtain ⟨m, dp, dm, mt, me⟩ := s
  simp only at hme; subst hme
  have hnm : effV ⟨m, dp, dm, mt, []⟩ ≠ some .multi := fun e => by have := h4 e; simp at this
  refine ⟨⟨?_, by simp [removeFast], by simp [removeFast], ?_, by simp [removeFast], by simp [removeFast], ?_⟩, ?_, ?_⟩
  · cases m <;> cases dm <;> simp [removeFast] <;> exact absurd (h1 rfl) (by simp)
  · cases m <;> cases dm <;> simp [removeFast, effV]
  · cases m <;> cases dm <;> simp [removeFast, effV]
  · cases m <;> cases dm <;> simp [removeFast, edges, effV]
  · intro x
    obtain ⟨y, ys, rfl⟩ := List.exists_cons_of_ne_nil hne
    have hy := hsub y (by simp)
    constructor
    · intro hx
      revert hx hy; unfold edges; split <;> simp_all
    · intro hx; exact (by
        have := hsub x hx
        revert this hy; unfold edges; split <;> simp_all)

/-- **Fold** (`Tensor::flush`) never changes the effective value or the edges,
and keeps every invariant, for any fold decisions. -/
theorem foldP_correct (s : PairSt) (h : Inv s) (a b : Bool) :
    Inv (foldP s a b) ∧ effV (foldP s a b) = effV s ∧ edges (foldP s a b) = edges s := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  obtain ⟨m, dp, dm, mt, me⟩ := s
  have he : effV (foldP ⟨m, dp, dm, mt, me⟩ a b) = effV ⟨m, dp, dm, mt, me⟩ := by
    cases a <;> cases b <;> cases dp <;> cases dm <;> simp_all [foldP, effV, dpOr]
  have hed : edges (foldP ⟨m, dp, dm, mt, me⟩ a b) = edges ⟨m, dp, dm, mt, me⟩ := by
    unfold edges; rw [he]; cases a <;> cases b <;> rfl
  have hme : (foldP ⟨m, dp, dm, mt, me⟩ a b).me = me := by cases a <;> cases b <;> rfl
  have hmt : (foldP ⟨m, dp, dm, mt, me⟩ a b).mt = mt := by cases a <;> cases b <;> rfl
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, he, hed⟩
  · cases a <;> cases b <;> cases dp <;> cases dm <;> cases m <;> simp [foldP, dpOr] at h1 ⊢
  · cases a <;> cases b <;> cases dp <;> cases dm <;> simp [foldP, dpOr] at h2 ⊢
  · intro v hv
    cases a <;> cases b <;> cases dp <;> cases dm <;> simp [foldP, dpOr] at hv h2 h3 ⊢ <;>
      first | exact h3 v hv | (subst hv; exact h3 _ rfl) | grind
  · rw [he, hme]; exact h4
  · rw [he, hme]; exact h5
  · rw [hme]; exact h6
  · rw [he, hmt]; exact h7

/-- **`edge_count`.** For every pair in a reachable state, each running total
of `|m| + |dp| − |dm| − |dp ∩ m| − multi + |me|` is non-negative (so none of
the `checked_sub`s panics) and the last one is exactly the pair's number of
edges. Summing over pairs (all six terms are sums over pairs; `me` rows are
per pair by `compoundKey_inv`) gives the global identity. -/
theorem count_correct (s : PairSt) (h : Inv s) :
    (∀ t ∈ partials s, 0 ≤ t) ∧ (partials s).getLast? = some ((edges s).length : Int) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  obtain ⟨m, dp, dm, mt, me⟩ := s
  cases dp with
  | some v =>
    have hdm : dm = false := h2 rfl
    subst hdm
    cases v with
    | multi =>
      have := h4 (by simp [effV])
      cases m <;> simp [partials, b2i, effV, edges] <;> omega
    | id i =>
      have := h5 (by simp [effV]); subst this
      cases m <;> simp [partials, b2i, effV, edges]
  | none =>
    cases dm with
    | true =>
      have := h1 rfl
      have hme := h5 (by simp [effV]); simp only at hme; subst hme
      cases m <;> simp_all [partials, b2i, effV, edges]
    | false =>
      cases m with
      | none =>
        have hme := h5 (by simp [effV]); simp only at hme; subst hme
        simp [partials, b2i, effV, edges]
      | some w =>
        cases w with
        | multi =>
          have := h4 (by simp [effV])
          simp [partials, b2i, effV, edges] <;> omega
        | id i =>
          have hme := h5 (by simp [effV]); simp only at hme; subst hme
          simp [partials, b2i, effV, edges]

/-- Backward iteration (`Iter::next`, :1720-1732, and `col_degree`): a pair
reached through `mt` always has a forward inline value, so the
`unreachable!` there is unreachable. -/
theorem mt_has_forward (s : PairSt) (h : Inv s) (hmt : s.mt = true) : (effV s).isSome = true := by
  rw [← h.mt_eff]; exact hmt

/-! ### `compound_key` -/

theorem compoundKey_row_lt (s d : Nat) : (compoundKey s d).2 < 2 ^ 60 := by
  have h1 : s % B < B := Nat.mod_lt _ (by decide)
  have h2 : d % B < B := Nat.mod_lt _ (by decide)
  have : (2 : Nat) ^ 60 = B * B := by decide
  simp only [compoundKey]; rw [this]
  have : s % B * B + d % B < (s % B + 1) * B := by rw [Nat.succ_mul]; omega
  exact Nat.lt_of_lt_of_le this (Nat.mul_le_mul_right _ h1)

/-- `compound_key_inverse ∘ compound_key = id`: the key is injective, so
distinct pairs never share an `me` row (`ME_DIM = 2^60` rows hold every key,
`compoundKey_row_lt`). -/
theorem compoundKey_inv (s d : Nat) :
    compoundKeyInv (compoundKey s d).1 (compoundKey s d).2 = (s, d) := by
  have hB : 0 < B := by decide
  have h2 : d % B < B := Nat.mod_lt _ hB
  simp only [compoundKey, compoundKeyInv]
  have e1 : (s % B * B + d % B) / B = s % B := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt h2, Nat.zero_add]
  have e2 : (s % B * B + d % B) % B = d % B := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h2]
  rw [e1, e2, Nat.mul_comm (s / B), Nat.mul_comm (d / B), Nat.div_add_mod, Nat.div_add_mod]

theorem compoundKey_injective {s d s' d' : Nat} (h : compoundKey s d = compoundKey s' d') :
    s = s' ∧ d = d' := by
  have := compoundKey_inv s d
  rw [h, compoundKey_inv] at this
  simp at this; exact ⟨this.1.symm, this.2.symm⟩

/-- The Rust bit operations are the arithmetic above: `x & (2^30-1) = x % 2^30`,
`x << 30 = x * 2^30`, `x >> 30 = x / 2^30`, and `|` of a value shifted past
the other operand's bits is `+`. -/
theorem compoundKey_bits (s d : Nat) :
    compoundKey s d = ((s >>> 30, d >>> 30), ((s &&& (2^30 - 1)) <<< 30) ||| (d &&& (2^30 - 1))) := by
  simp only [compoundKey, Nat.shiftRight_eq_div_pow, Nat.and_two_pow_sub_one_eq_mod]
  rw [← Nat.shiftLeft_add_eq_or_of_lt (Nat.mod_lt _ (by decide)), Nat.shiftLeft_eq]
  rfl

end VMTensor
