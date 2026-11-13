import CchTopologyReview.Props

/-!
# The fold changes the representation and nothing else (03 §6 "Fold job")

The job forks at `F` (a published state `tF`), builds new bases from `base ⊕ delta_F` with GraphBLAS, and
publishes onto the current published copy `t` (or, for a writer that overtook it, onto that writer's
copy: 03 §6 "A writer that overtook a fold applies the same function to its own copy"). Both are
`WLin tF t`: writers only since `F`, each on a private copy above `F`. That is premise **FoldFit**, taken
as a hypothesis here. Its write-log half is formal in `cch_redesign_review`'s model (FitBridge.lean): between a
fitted job's fork and its publish, every step is a writer step or leaves the entry unchanged
(`fitted_job_only_writer_steps`), and every writer there starts from the published entry without a rebase
(`fitted_window_writer_no_rebase`). Argued, not proved: only the cross-model step, that those writer steps are
this model's `fork` + `Step`s (the two models' states differ and are not composed).

The GraphBLAS calls are the hypothesis structure `AddSpec` (C API, axiomatised as data).
-/

namespace CchTopo

variable {α : Type}

/-- **GraphBLAS contract** (C, a hypothesis structure, not an axiom): `GrB_eWiseAdd` of a read-only load of
    a base and of a matrix built from a delta whose keys are not in the base, then unloaded. The result
    holds exactly the entries of both (`eWiseAdd`'s pattern is the union, and with disjoint patterns no
    operator is applied, so each value, the id, is carried over), and its rows are strictly ascending (a
    matrix holds one entry per `(i, j)`, and the unload is not jumbled). -/
structure AddSpec (b : Rows) (d : List Ent) (out : Rows) : Prop where
  mem : (∀ e ∈ d, ∀ i, ¬ inB b ⟨e.row, e.col, i⟩) → ∀ e, inB out e ↔ inB b e ∨ e ∈ d
  strict : ∀ r, StrictCol (bRow out r)

theorem keepAfter_split (F : Nat) (a ext : List (Nat × List Ent)) (ha : ∀ p ∈ a, p.1 ≤ F)
    (he : ∀ p ∈ ext, F < p.1) : keepAfter F (a ++ ext) = ext := by
  unfold keepAfter
  rw [List.filter_append]
  have h1 : a.filter (fun p => decide (F < p.1)) = [] :=
    List.filter_eq_nil_iff.2 (fun p hp => by have := ha p hp; simp; omega)
  have h2 : ext.filter (fun p => decide (F < p.1)) = ext :=
    List.filter_eq_self.2 (fun p hp => by simp [he p hp])
  rw [h1, h2]; rfl

theorem mem_map_tr (l : List Ent) (e : Ent) : e ∈ l.map tr ↔ tr e ∈ l := by
  constructor
  · intro h; obtain ⟨u, hu, rfl⟩ := List.mem_map.1 h; rw [tr_tr]; exact hu
  · intro h; exact List.mem_map.2 ⟨tr e, h, tr_tr e⟩

/-- **Everything a fold changes, and keeps.** `t` is the copy the fold publishes onto. -/
theorem fold_facts (inf : α) {tF t : Cch α} {nu nd : Rows} (hI : Inv tF) (hw : WLin inf tF t)
    (hU : AddSpec tF.upB tF.upD nu) (hD : AddSpec tF.downB tF.downD nd) :
    (∀ e, upE (fold tF.ver nu nd t) e ↔ upE t e) ∧
    (∀ e, downE (fold tF.ver nu nd t) e ↔ downE t e) ∧
    (∀ e, e ∈ (fold tF.ver nu nd t).upD ↔ e ∈ t.upD ∧ e ∉ tF.upD) ∧
    (∀ e, e ∈ (fold tF.ver nu nd t).upD ↔ ∃ p ∈ t.added, tF.ver < p.1 ∧ e ∈ p.2) ∧
    Inv (fold tF.ver nu nd t) ∧ Inv t := by
  obtain ⟨hb1, hb2, hv, hc, ⟨ext, hext, hseq⟩, hIt⟩ := wlin_shape inf hw hI
  have haF := hI.2.1
  have hat := hIt.2.1
  have hkeep : keepAfter tF.ver t.added = ext := by rw [hext]; exact keepAfter_split _ _ _ haF.added_seq hseq
  have hents : ents t.added = ents tF.added ++ ents ext := by rw [hext, ents_append]
  have hnd : (ents tF.added ++ ents ext).Nodup := hents ▸ hat.added_nodup
  rw [List.nodup_append] at hnd
  have hFupD : ∀ e, e ∈ tF.upD ↔ e ∈ ents tF.added := haF.added_eq
  have htupD : ∀ e, e ∈ t.upD ↔ e ∈ tF.upD ∨ e ∈ ents ext := by
    intro e
    rw [show e ∈ t.upD ↔ e ∈ ents t.added from hat.added_eq e, hents, List.mem_append, hFupD]
  have hKsub : ∀ e ∈ ents ext, e ∈ t.upD := fun e he => (htupD e).2 (Or.inr he)
  have hKkey : (ents ext).Pairwise (fun a b => ¬ sameKey a b) :=
    hnd.2.1.imp_of_mem (fun ha hb hne hs => hne (hat.delta_key_unique _ (hKsub _ ha) _ (hKsub _ hb) hs))
  have hmK : ∀ e, e ∈ insAll [] (ents ext) ↔ e ∈ ents ext := by
    intro e; rw [mem_insAll_fresh _ [] hKkey (by simp)]; simp
  have hKkeyT : ((ents ext).map tr).Pairwise (fun a b => ¬ sameKey a b) := by
    rw [List.pairwise_map]; exact hKkey.imp (fun h hs => h ⟨hs.2, hs.1⟩)
  have hmKT : ∀ e, e ∈ insAll [] ((ents ext).map tr) ↔ e ∈ (ents ext).map tr := by
    intro e; rw [mem_insAll_fresh _ [] hKkeyT (by simp)]; simp
  have hKF : ∀ e ∈ ents ext, e ∉ tF.upD := fun e he hF => hnd.2.2 e ((hFupD e).1 hF) e he rfl
  have hnu : ∀ e, inB nu e ↔ inB tF.upB e ∨ e ∈ tF.upD := hU.mem haF.up_disj
  have hnd' : ∀ e, inB nd e ↔ inB tF.downB e ∨ e ∈ tF.downD := hD.mem haF.down_disj
  have fupD : (fold tF.ver nu nd t).upD = insAll [] (ents ext) := by simp only [fold, hkeep]
  have fdownD : (fold tF.ver nu nd t).downD = insAll [] ((ents ext).map tr) := by simp only [fold, hkeep]
  have fadded : (fold tF.ver nu nd t).added = ext := by simp only [fold, hkeep]
  have fupB : (fold tF.ver nu nd t).upB = nu := rfl
  have fdownB : (fold tF.ver nu nd t).downB = nd := rfl
  -- the arcs
  have hup : ∀ e, upE (fold tF.ver nu nd t) e ↔ upE t e := by
    intro e
    unfold upE
    rw [fupB, fupD, hmK, hnu, htupD, hb1]
    simp only [or_assoc]
  have htdownD : ∀ e, e ∈ t.downD ↔ e ∈ tF.downD ∨ e ∈ (ents ext).map tr := by
    intro e
    have h1 : e ∈ t.downD ↔ tr e ∈ t.upD := by
      have := hat.i7D (tr e); simp only [Cch.arcs_upD, Cch.arcs_downD, tr_tr] at this; exact this.symm
    have h2 : tr e ∈ tF.upD ↔ e ∈ tF.downD := by
      have := haF.i7D (tr e); simp only [Cch.arcs_upD, Cch.arcs_downD, tr_tr] at this; exact this
    rw [h1, htupD, h2, mem_map_tr]
  have hdown : ∀ e, downE (fold tF.ver nu nd t) e ↔ downE t e := by
    intro e
    unfold downE
    rw [fdownB, fdownD, hmKT, hnd', htdownD, hb2]
    simp only [or_assoc]
  have hexact : ∀ e, e ∈ (fold tF.ver nu nd t).upD ↔ e ∈ t.upD ∧ e ∉ tF.upD := by
    intro e
    rw [fupD, hmK, htupD]
    constructor
    · intro he; exact ⟨Or.inr he, hKF e he⟩
    · rintro ⟨h | h, hn⟩
      · exact absurd h hn
      · exact h
  have hnew : ∀ e, e ∈ (fold tF.ver nu nd t).upD ↔ ∃ p ∈ t.added, tF.ver < p.1 ∧ e ∈ p.2 := by
    intro e
    rw [fupD, hmK, ← hkeep, mem_ents]
    unfold keepAfter
    simp only [List.mem_filter, decide_eq_true_eq]
    constructor
    · rintro ⟨p, ⟨hp, hs⟩, he⟩; exact ⟨p, hp, hs, he⟩
    · rintro ⟨p, hp, hs, he⟩; exact ⟨p, ⟨hp, hs⟩, he⟩
  -- the invariant
  have hupdisj : ∀ e ∈ (fold tF.ver nu nd t).upD, ∀ i, ¬ inB nu ⟨e.row, e.col, i⟩ := by
    intro e he i hb
    rw [fupD, hmK] at he
    have het : e ∈ t.upD := hKsub e he
    rcases (hnu _).1 hb with h | h
    · have h' : upE t ⟨e.row, e.col, i⟩ := Or.inl (hb1 ▸ h)
      have := hat.up_key_unique _ _ h' (Or.inr het) ⟨rfl, rfl⟩
      rw [this] at h
      exact hat.up_disj e het e.id (by rw [Cch.arcs_upB, hb1]; exact h)
    · have h' : upE t ⟨e.row, e.col, i⟩ := Or.inr ((htupD _).2 (Or.inl h))
      have := hat.up_key_unique _ _ h' (Or.inr het) ⟨rfl, rfl⟩
      rw [this] at h
      exact hKF e he h
  have hI' : Inv (fold tF.ver nu nd t) := by
    refine ⟨hIt.1, ?_, fun e he => hIt.2.2 e ((hup e).1 he)⟩
    refine
      { up_lt := fun e he => hat.up_lt e ((hup e).1 he)
        upB_strict := hU.strict
        downB_strict := hD.strict
        upD_sorted := by
          show KSorted (fold tF.ver nu nd t).upD
          rw [fupD]; exact ksorted_insAll _ _ (by simp [KSorted])
        downD_sorted := by
          show KSorted (fold tF.ver nu nd t).downD
          rw [fdownD]; exact ksorted_insAll _ _ (by simp [KSorted])
        up_disj := hupdisj
        down_disj := ?_
        i7B := ?_
        i7D := ?_
        id_lt := fun e he => hat.id_lt e ((hup e).1 he)
        id_inj := fun e e' he he' hid => hat.id_inj e e' ((hup e).1 he) ((hup e').1 he') hid
        met_len := hat.met_len
        added_eq := ?_
        added_nodup := ?_
        added_seq := ?_ }
    · intro e he i hb
      have he' : e ∈ (fold tF.ver nu nd t).downD := he
      rw [fdownD, hmKT, mem_map_tr] at he'
      have hb' : inB nd ⟨e.row, e.col, i⟩ := hb
      rcases (hnd' _).1 hb' with h | h
      · have : inB tF.upB ⟨e.col, e.row, i⟩ := (haF.i7B ⟨e.col, e.row, i⟩).2 h
        exact hupdisj (tr e) (by rw [fupD, hmK]; exact he') i ((hnu _).2 (Or.inl this))
      · have : (⟨e.col, e.row, i⟩ : Ent) ∈ tF.upD := (haF.i7D ⟨e.col, e.row, i⟩).2 h
        exact hupdisj (tr e) (by rw [fupD, hmK]; exact he') i ((hnu _).2 (Or.inr this))
    · intro e
      show inB nu e ↔ inB nd (tr e)
      rw [hnu, hnd']
      have h1 : inB tF.upB e ↔ inB tF.downB (tr e) := haF.i7B e
      have h2 : e ∈ tF.upD ↔ tr e ∈ tF.downD := haF.i7D e
      rw [h1, h2]
    · intro e
      show e ∈ (fold tF.ver nu nd t).upD ↔ tr e ∈ (fold tF.ver nu nd t).downD
      rw [fupD, fdownD, hmK, hmKT, mem_map_tr, tr_tr]
    · intro e
      show e ∈ (fold tF.ver nu nd t).upD ↔ e ∈ ents (fold tF.ver nu nd t).added
      rw [fupD, fadded, hmK]
    · show (ents (fold tF.ver nu nd t).added).Nodup
      rw [fadded]; exact hnd.2.1
    · intro p hp
      have hp' : p ∈ ext := by rw [← fadded]; exact hp
      exact hat.added_seq p (by rw [Cch.arcs_added, hext]; exact List.mem_append_right _ hp')
  exact ⟨hup, hdown, hexact, hnew, hI', hIt⟩

/-- **Fold equivalence**: after a fold, every rank's `up_ids` and `down_ids` are what they were (same ranks,
    same ids, same order). -/
theorem fold_rows (inf : α) {tF t : Cch α} {nu nd : Rows} (hI : Inv tF) (hw : WLin inf tF t)
    (hU : AddSpec tF.upB tF.upD nu) (hD : AddSpec tF.downB tF.downD nd) :
    (∀ r, upIds (fold tF.ver nu nd t) r = upIds t r) ∧ (∀ r, downIds (fold tF.ver nu nd t) r = downIds t r) := by
  obtain ⟨hup, hdown, _, _, hI', hIt⟩ := fold_facts inf hI hw hU hD
  constructor
  · intro r
    refine strict_ext (upIds_strict hI' r) (upIds_strict hIt r) (fun ⟨c, i⟩ => ?_)
    rw [mem_upIds, mem_upIds, hup]
  · intro r
    refine strict_ext (downIds_strict hI' r) (downIds_strict hIt r) (fun ⟨c, i⟩ => ?_)
    rw [mem_downIds, mem_downIds, hdown]

/-- **The fold leaves everything else alone**: the metric, `rank_of`, `node_of`, `free_ranks`,
    `arc_capacity`, `LineageStats` and the version come from the copy it publishes onto. -/
theorem fold_keeps (F : Nat) (nu nd : Rows) (t : Cch α) :
    (fold F nu nd t).met = t.met ∧ (fold F nu nd t).nodeOf = t.nodeOf ∧ (fold F nu nd t).rankOf = t.rankOf ∧
    (fold F nu nd t).free = t.free ∧ (fold F nu nd t).cap = t.cap ∧ (fold F nu nd t).stats = t.stats ∧
    (fold F nu nd t).nr = t.nr ∧ (fold F nu nd t).ver = t.ver ∧ (fold F nu nd t).reserve = t.reserve :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- What a reader of a version reads: every row of `Up` and `Down`, the metric, and both rank maps. -/
structure ReaderView (t t' : Cch α) : Prop where
  up : ∀ r, upIds t r = upIds t' r
  down : ∀ r, downIds t r = downIds t' r
  met : t.met = t'.met
  nodeOf : t.nodeOf = t'.nodeOf
  rankOf : t.rankOf = t'.rankOf
  free : t.free = t'.free

/-- **A fold is invisible to readers**: the folded copy reads exactly as the copy it replaces. -/
theorem fold_view (inf : α) {tF t : Cch α} {nu nd : Rows} (hI : Inv tF) (hw : WLin inf tF t)
    (hU : AddSpec tF.upB tF.upD nu) (hD : AddSpec tF.downB tF.downD nd) :
    ReaderView (fold tF.ver nu nd t) t :=
  ⟨(fold_rows inf hI hw hU hD).1, (fold_rows inf hI hw hU hD).2, rfl, rfl, rfl, rfl⟩

/-- **The new delta is exactly the arcs added after the fold's fork**: in the published copy and not in
    the fork's delta, which are the arcs of `added_since_fold` with `seq > F`. -/
theorem fold_new_delta (inf : α) {tF t : Cch α} {nu nd : Rows} (hI : Inv tF) (hw : WLin inf tF t)
    (hU : AddSpec tF.upB tF.upD nu) (hD : AddSpec tF.downB tF.downD nd) (e : Ent) :
    (e ∈ (fold tF.ver nu nd t).upD ↔ e ∈ t.upD ∧ e ∉ tF.upD) ∧
    (e ∈ (fold tF.ver nu nd t).upD ↔ ∃ p ∈ t.added, tF.ver < p.1 ∧ e ∈ p.2) ∧
    (fold tF.ver nu nd t).added = keepAfter tF.ver t.added := by
  obtain ⟨_, _, h1, h2, _, _⟩ := fold_facts inf hI hw hU hD
  exact ⟨h1 e, h2 e, rfl⟩

/-- **A fold of a published state equals that state as every reader sees it**, with empty deltas. -/
theorem fold_of_published (inf : α) {tF : Cch α} {nu nd : Rows} (hI : Inv tF)
    (hU : AddSpec tF.upB tF.upD nu) (hD : AddSpec tF.downB tF.downD nd) :
    ReaderView (fold tF.ver nu nd tF) tF ∧ (fold tF.ver nu nd tF).upD = [] ∧ (fold tF.ver nu nd tF).downD = [] ∧
    Inv (fold tF.ver nu nd tF) := by
  obtain ⟨_, _, hx, _, hI', _⟩ := fold_facts inf hI (WLin.refl) hU hD
  have hu : (fold tF.ver nu nd tF).upD = [] :=
    List.eq_nil_iff_forall_not_mem.2 (fun e he => ((hx e).1 he).2 ((hx e).1 he).1)
  refine ⟨fold_view inf hI WLin.refl hU hD, hu, ?_, hI'⟩
  apply List.eq_nil_iff_forall_not_mem.2
  intro e he
  have : tr e ∈ (fold tF.ver nu nd tF).upD := by
    have := (hI'.2.1.i7D (tr e)).2; simp only [Cch.arcs_upD, Cch.arcs_downD, tr_tr] at this; exact this he
  rw [hu] at this; simp at this

end CchTopo
