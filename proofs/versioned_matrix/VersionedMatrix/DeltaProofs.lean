import VersionedMatrix.Delta

/-!
# Correctness of the delta layers

* every operation preserves `Inv` (`dp ∩ m = ∅`, `dm ⊆ m`, hence `dp ∩ dm = ∅`,
  no duplicates, everything in bounds);
* every operation moves the logical matrix `eff` exactly as the reference
  moves (`step_repr`), so after any sequence of operations the logical matrix
  *is* the reference (`run_repr`);
* folding/flushing never changes the logical matrix (`fold_eff`, `flush_eff`);
* `get` agrees with `eff` (`get_iff_eff`), and `nvals`' unsigned
  `|m| + |dp| − |dm|` neither underflows nor miscounts (`nvals_eq`).
-/

namespace VM

open List

theorem mem_filter_not {l : Layer} {d : Layer} {p : Pair} :
    p ∈ l.filter (fun q => !(decide (q ∈ d))) ↔ p ∈ l ∧ p ∉ d := by
  simp only [List.mem_filter, Bool.not_eq_true', decide_eq_false_iff_not]

theorem mem_filter_dec {l : Layer} {f : Pair → Prop} [DecidablePred f] {p : Pair} :
    p ∈ l.filter (fun q => decide (f q)) ↔ p ∈ l ∧ f p := by
  simp only [List.mem_filter, decide_eq_true_eq]

theorem nodup_app {a b : Layer} (ha : a.Nodup) (hb : b.Nodup) (hd : ∀ p, p ∈ b → p ∉ a) :
    (a ++ b).Nodup :=
  List.nodup_append.2 ⟨ha, hb, fun x hx y hy hxy => by subst hxy; exact hd _ hy hx⟩

/-! ### fold / flush -/

theorem fold_inv {v : VM} (h : Inv v) (a b : Bool) : Inv (fold v a b) := by
  obtain ⟨h1, h2, n1, n2, n3, b1, b2⟩ := h
  cases a <;> cases b <;> constructor <;>
    simp only [fold, mem_filter_not, List.mem_append, List.not_mem_nil, ite_true, ite_false,
      Bool.false_eq_true] <;> (try intro p) <;> (try intro hp)
  all_goals first
    | assumption
    | exact nodup_filter n1 _
    | exact nodup_app n1 n2 h1
    | exact nodup_filter (nodup_app n1 n2 h1) _
    | exact List.nodup_nil
    | grind

theorem fold_eff {v : VM} (h : Inv v) (a b : Bool) (p : Pair) :
    eff (fold v a b) p ↔ eff v p := by
  have h1 := h.dp_m p; have h2 := h.dm_m p
  cases a <;> cases b <;>
    simp only [eff, fold, mem_filter_not, List.mem_append, List.not_mem_nil, ite_true,
      ite_false, Bool.false_eq_true] <;> grind

theorem flush_inv {v : VM} (h : Inv v) : Inv (flush v) := by
  cases hv : v.armed with
  | none => simp only [flush, hv]; exact h
  | some ab =>
    obtain ⟨a, b⟩ := ab
    simp only [flush, hv]
    have := fold_inv h a b
    exact ⟨this.1, this.2, this.3, this.4, this.5, this.6, this.7⟩

theorem flush_eff {v : VM} (h : Inv v) (p : Pair) : eff (flush v) p ↔ eff v p := by
  unfold flush; split
  · exact Iff.rfl
  · exact fold_eff h _ _ p

@[simp] theorem flush_nrows (v : VM) : (flush v).nrows = v.nrows := by
  unfold flush; split <;> simp [fold]
@[simp] theorem flush_ncols (v : VM) : (flush v).ncols = v.ncols := by
  unfold flush; split <;> simp [fold]
@[simp] theorem flush_armed (v : VM) : (flush v).armed = none := by
  unfold flush; split <;> simp_all

theorem flush_flush (v : VM) : flush (flush v) = flush v := by
  have := flush_armed v
  generalize flush v = w at this ⊢
  simp [flush, this]

/-! ### set / remove -/

theorem set_inv {v : VM} (h : Inv v) {p : Pair} (hb : inBounds v.nrows v.ncols p) :
    Inv (set v p) := by
  have hf := flush_inv h
  obtain ⟨h1, h2, n1, n2, n3, b1, b2⟩ := hf
  have hb' : inBounds (flush v).nrows (flush v).ncols p := by simpa using hb
  unfold set; simp only
  by_cases hm : p ∈ (flush v).m
  · simp only [hm, ite_true]
    exact ⟨h1, fun q hq => h2 q (mem_del.1 hq).2, n1, n2, nodup_del n3 _, b1, b2⟩
  · simp only [hm, ite_false]
    refine ⟨?_, h2, n1, nodup_ins n2 _, n3, b1, ?_⟩
    · intro q hq; rcases mem_ins.1 hq with rfl | hq
      · exact hm
      · exact h1 q hq
    · intro q hq; rcases mem_ins.1 hq with rfl | hq
      · exact hb'
      · exact b2 q hq

theorem set_eff {v : VM} (h : Inv v) (p q : Pair) : eff (set v p) q ↔ q = p ∨ eff v q := by
  have hf := flush_inv h
  rw [← flush_eff h q]
  have hdm := hf.dm_m p; have hdp := hf.dp_m p
  unfold set; simp only
  by_cases hm : p ∈ (flush v).m
  · simp only [hm, ite_true, eff, mem_del]; grind
  · simp only [hm, ite_false, eff, mem_ins]; grind

theorem remove_inv {v : VM} (h : Inv v) (p : Pair) : Inv (remove v p) := by
  have hf := flush_inv h
  obtain ⟨h1, h2, n1, n2, n3, b1, b2⟩ := hf
  unfold remove; simp only
  by_cases hm : p ∈ (flush v).m
  · simp only [hm, ite_true]
    refine ⟨h1, ?_, n1, n2, nodup_ins n3 _, b1, b2⟩
    intro q hq; rcases mem_ins.1 hq with rfl | hq
    · exact hm
    · exact h2 q hq
  · simp only [hm, ite_false]
    exact ⟨fun q hq => h1 q (mem_del.1 hq).2, h2, n1, nodup_del n2 _, n3, b1,
      fun q hq => b2 q (mem_del.1 hq).2⟩

theorem remove_eff {v : VM} (h : Inv v) (p q : Pair) :
    eff (remove v p) q ↔ q ≠ p ∧ eff v q := by
  have hf := flush_inv h
  rw [← flush_eff h q]
  have hdm := hf.dm_m p; have hdp := hf.dp_m p
  unfold remove; simp only
  by_cases hm : p ∈ (flush v).m
  · simp only [hm, ite_true, eff, mem_ins]; grind
  · simp only [hm, ite_false, eff, mem_del]; grind

@[simp] theorem set_nrows (v : VM) (p : Pair) : (set v p).nrows = v.nrows := by
  unfold set; simp only; by_cases hm : p ∈ (flush v).m <;> simp [hm]
@[simp] theorem set_ncols (v : VM) (p : Pair) : (set v p).ncols = v.ncols := by
  unfold set; simp only; by_cases hm : p ∈ (flush v).m <;> simp [hm]

/-- Inserting a pair not in the committed base straight into `dp` (the body of
the fast loops) keeps the invariants. -/
theorem insDp_inv {v : VM} (h : Inv v) {p : Pair} (hpm : p ∉ v.m)
    (hb : inBounds v.nrows v.ncols p) : Inv { v with dp := ins v.dp p } := by
  obtain ⟨h1, h2, n1, n2, n3, b1, b2⟩ := h
  refine ⟨?_, h2, n1, nodup_ins n2 _, n3, b1, ?_⟩
  · intro q hq; rcases mem_ins.1 hq with rfl | hq
    · exact hpm
    · exact h1 q hq
  · intro q hq; rcases mem_ins.1 hq with rfl | hq
    · exact hb
    · exact b2 q hq

/-! ### bulk set -/

theorem setEach_inv_eff : ∀ (es : List Pair) {v : VM}, Inv v →
    (∀ p ∈ es, inBounds v.nrows v.ncols p) →
    Inv (setEach v es) ∧ (setEach v es).nrows = v.nrows ∧ (setEach v es).ncols = v.ncols ∧
      ∀ q, (eff (setEach v es) q ↔ q ∈ es ∨ eff v q)
  | [], v, h, _ => ⟨h, rfl, rfl, by simp [setEach]⟩
  | p :: ps, v, h, hb => by
    have h1 := set_inv h (hb p (by simp))
    have ih := setEach_inv_eff ps h1 (by intro q hq; simpa using hb q (by simp [hq]))
    refine ⟨ih.1, by simp [setEach, ih.2.1], by simp [setEach, ih.2.2.1], fun q => ?_⟩
    simp only [setEach]; rw [ih.2.2.2 q, set_eff h]; simp only [List.mem_cons]; grind

theorem setAllFast_inv_eff (NEW : Bool) : ∀ (es : List Pair) {v : VM}, Inv v → v.dm = [] →
    (∀ p ∈ es, inBounds v.nrows v.ncols p) → (NEW = true → ∀ p ∈ es, p ∉ v.m) →
    Inv (setAllFast NEW v es) ∧ (setAllFast NEW v es).nrows = v.nrows ∧
      (setAllFast NEW v es).ncols = v.ncols ∧
      ∀ q, (eff (setAllFast NEW v es) q ↔ q ∈ es ∨ eff v q)
  | [], v, h, _, _, _ => ⟨h, rfl, rfl, by simp [setAllFast]⟩
  | p :: ps, v, h, hdm, hb, hn => by
    simp only [setAllFast]
    by_cases hc : (!NEW && decide (p ∈ v.m)) = true
    · simp only [hc, ite_true]
      have hm : p ∈ v.m := by simp at hc; exact hc.2
      have ih := setAllFast_inv_eff NEW ps h hdm (fun q hq => hb q (by simp [hq]))
        (fun hN q hq => hn hN q (by simp [hq]))
      refine ⟨ih.1, ih.2.1, ih.2.2.1, fun q => ?_⟩
      rw [ih.2.2.2 q]; simp only [eff, hdm, List.mem_cons, List.not_mem_nil]
      grind
    · simp only [hc, Bool.false_eq_true, ite_false]
      have hpm : p ∉ v.m := by
        cases NEW
        · simpa using hc
        · exact hn rfl p (by simp)
      have hv' := insDp_inv h hpm (hb p (by simp))
      have ih := setAllFast_inv_eff NEW ps hv' hdm (fun q hq => hb q (by simp [hq]))
        (fun hN q hq => hn hN q (by simp [hq]))
      refine ⟨ih.1, ih.2.1, ih.2.2.1, fun q => ?_⟩
      rw [ih.2.2.2 q]; simp only [eff, mem_ins, List.mem_cons]; grind

theorem setAll_inv_eff {v : VM} (h : Inv v) (NEW : Bool) (es : List Pair)
    (hb : ∀ p ∈ es, inBounds v.nrows v.ncols p)
    (hn : NEW = true → ∀ p ∈ es, p ∉ (flush v).m) :
    Inv (setAll NEW v es) ∧ (setAll NEW v es).nrows = v.nrows ∧
      (setAll NEW v es).ncols = v.ncols ∧
      ∀ q, (eff (setAll NEW v es) q ↔ q ∈ es ∨ eff v q) := by
  have hf := flush_inv h
  have hb' : ∀ p ∈ es, inBounds (flush v).nrows (flush v).ncols p := by simpa using hb
  unfold setAll; simp only
  split
  · have := setAllFast_inv_eff NEW es hf ‹_› hb' hn
    exact ⟨this.1, by simp [this.2.1], by simp [this.2.2.1],
      fun q => by rw [this.2.2.2 q, flush_eff h]⟩
  · have := setEach_inv_eff es hf hb'
    exact ⟨this.1, by simp [this.2.1], by simp [this.2.2.1],
      fun q => by rw [this.2.2.2 q, flush_eff h]⟩

/-- The one-call `GrB_assign` of `set_product`'s fast path, pair by pair. -/
theorem foldIns_inv_eff : ∀ (es : List Pair) {v : VM}, Inv v →
    (∀ p ∈ es, inBounds v.nrows v.ncols p) → (∀ p ∈ es, p ∉ v.m) →
    let w := es.foldl (fun w p => { w with dp := ins w.dp p }) v
    Inv w ∧ w.nrows = v.nrows ∧ w.ncols = v.ncols ∧ ∀ q, (eff w q ↔ q ∈ es ∨ eff v q)
  | [], v, h, _, _ => by simp [h]
  | p :: ps, v, h, hb, hn => by
    simp only [List.foldl]
    have hv' := insDp_inv h (hn p (by simp)) (hb p (by simp))
    have ih := foldIns_inv_eff ps hv' (fun q hq => hb q (by simp [hq]))
      (fun q hq => hn q (by simp [hq]))
    refine ⟨ih.1, ih.2.1, ih.2.2.1, fun q => ?_⟩
    rw [ih.2.2.2 q]; simp only [eff, mem_ins, List.mem_cons]; grind

theorem setProduct_inv_eff {v : VM} (h : Inv v) (NEW : Bool) (rs cs : List Nat)
    (hb : ∀ p ∈ product rs cs, inBounds v.nrows v.ncols p)
    (hn : NEW = true → ∀ p ∈ product rs cs, p ∉ (flush v).m) :
    Inv (setProduct NEW v rs cs) ∧ (setProduct NEW v rs cs).nrows = v.nrows ∧
      (setProduct NEW v rs cs).ncols = v.ncols ∧
      ∀ q, (eff (setProduct NEW v rs cs) q ↔ q ∈ product rs cs ∨ eff v q) := by
  unfold setProduct
  split
  · rename_i he
    refine ⟨h, rfl, rfl, fun q => ?_⟩
    have : q ∉ product rs cs := by
      rcases he with rfl | rfl <;> simp [product]
    simp [this]
  · simp only
    have hf := flush_inv h
    have hb' : ∀ p ∈ product rs cs, inBounds (flush v).nrows (flush v).ncols p := by
      simpa using hb
    split
    · have := setAll_inv_eff hf NEW (product rs cs) hb' (by rw [flush_flush]; exact hn)
      exact ⟨this.1, by simp [this.2.1], by simp [this.2.2.1],
        fun q => by rw [this.2.2.2 q, flush_eff h]⟩
    · rename_i hc
      have hN : NEW = true := by cases NEW <;> simp_all
      have := foldIns_inv_eff (product rs cs) hf hb' (hn hN)
      exact ⟨this.1, by simp [this.2.1], by simp [this.2.2.1],
        fun q => by rw [this.2.2.2 q, flush_eff h]⟩

/-! ### remove_mask -/

theorem removeMask_inv_eff {v : VM} (h : Inv v) (mask : List Pair) :
    Inv (removeMask v mask) ∧ (removeMask v mask).nrows = v.nrows ∧
      (removeMask v mask).ncols = v.ncols ∧
      ∀ q, (eff (removeMask v mask) q ↔ q ∉ mask ∧ eff v q) := by
  have hf := flush_inv h
  obtain ⟨h1, h2, n1, n2, n3, b1, b2⟩ := hf
  have fm : ∀ q, q ∈ (flush v).m.filter (fun p => decide (p ∈ mask) && !decide (p ∈ (flush v).dm))
      ↔ q ∈ (flush v).m ∧ q ∈ mask ∧ q ∉ (flush v).dm := by
    intro q; simp only [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true',
      decide_eq_false_iff_not]
  refine ⟨⟨?_, ?_, n1, nodup_filter n2 _, ?_, ?_, ?_⟩, by simp [removeMask],
    by simp [removeMask], fun q => ?_⟩
  · intro q hq; simp only [removeMask, mem_filter_not] at hq; exact h1 q hq.1
  · intro q hq; simp only [removeMask, List.mem_append, fm] at hq
    rcases hq with hq | hq
    · exact h2 q hq
    · exact hq.1
  · simp only [removeMask]
    refine nodup_app n3 (nodup_filter n1 _) ?_
    intro x hx; rw [fm] at hx; exact hx.2.2
  · intro q hq; simp only [removeMask] at hq ⊢; simpa using b1 q hq
  · intro q hq; simp only [removeMask, mem_filter_not] at hq ⊢; simpa using b2 q hq.1
  · rw [← flush_eff h q]
    have hdm := h2 q; have hdp := h1 q
    simp only [eff, removeMask, List.mem_append, fm, mem_filter_not]; grind

/-! ### resize -/

theorem grow_inv_eff {v : VM} (h : Inv v) {r c : Nat} (hr : v.nrows ≤ r) (hc : v.ncols ≤ c) :
    Inv (grow v r c) ∧ (grow v r c).nrows = r ∧ (grow v r c).ncols = c ∧
      ∀ q, (eff (grow v r c) q ↔ eff v q) := by
  obtain ⟨h1, h2, n1, n2, n3, b1, b2⟩ := h
  have mono : ∀ p, inBounds v.nrows v.ncols p → inBounds r c p := by
    intro p hp; exact ⟨Nat.lt_of_lt_of_le hp.1 hr, Nat.lt_of_lt_of_le hp.2 hc⟩
  unfold grow
  split
  · exact ⟨⟨h1, h2, n1, n2, n3, fun p hp => mono p (b1 p hp), fun p hp => mono p (b2 p hp)⟩,
      rfl, rfl, fun q => Iff.rfl⟩
  · refine ⟨⟨by simp, by simp, ?_, List.nodup_nil, List.nodup_nil, ?_, by simp⟩, rfl, rfl,
      fun q => ?_⟩
    · refine nodup_app (nodup_filter n1 _) n2 ?_
      intro x hx hx'; rw [mem_filter_not] at hx'; exact h1 _ hx hx'.1
    · intro p hp; simp only [List.mem_append, mem_filter_not] at hp
      rcases hp with ⟨hm, _⟩ | hd
      · exact mono p (b1 p hm)
      · exact mono p (b2 p hd)
    · have := h1 q
      simp only [eff, List.mem_append, mem_filter_not, List.not_mem_nil]; grind

theorem shrink_inv_eff {v : VM} (h : Inv v) (r c : Nat) :
    Inv (shrink v r c) ∧ (shrink v r c).nrows = r ∧ (shrink v r c).ncols = c ∧
      ∀ q, (eff (shrink v r c) q ↔ inBounds r c q ∧ eff v q) := by
  have hf := flush_inv h
  obtain ⟨h1, h2, n1, n2, n3, b1, b2⟩ := hf
  refine ⟨⟨?_, ?_, nodup_filter n1 _, nodup_filter n2 _, nodup_filter n3 _, ?_, ?_⟩, rfl, rfl,
    fun q => ?_⟩
  · intro q hq; simp only [shrink, mem_filter_dec] at hq ⊢; intro hm; exact h1 q hq.1 hm.1
  · intro q hq; simp only [shrink, mem_filter_dec] at hq ⊢; exact ⟨h2 q hq.1, hq.2⟩
  · intro q hq; simp only [shrink, mem_filter_dec] at hq ⊢; exact hq.2
  · intro q hq; simp only [shrink, mem_filter_dec] at hq ⊢; exact hq.2
  · rw [← flush_eff h q]
    simp only [eff, shrink, mem_filter_dec]; grind

/-! ### one step, and any sequence -/

/-- A `VM` *represents* a reference when the logical matrix is the reference set
and the dimensions agree. -/
def Repr (v : VM) (R : Ref) : Prop :=
  Inv v ∧ R.nrows = v.nrows ∧ R.ncols = v.ncols ∧ ∀ q, eff v q ↔ R.s q

/-- **One step.** Every operation, under its documented caller contract,
preserves the invariants and moves the logical matrix exactly as the
reference moves. -/
theorem step_repr {v : VM} {R : Ref} (h : Repr v R) (op : Op) (hp : Pre v op) :
    Repr (step v op) (R.step op) := by
  obtain ⟨hi, hr, hc, he⟩ := h
  cases op with
  | set p =>
    exact ⟨set_inv hi hp, by simp [step, Ref.step, hr], by simp [step, Ref.step, hc],
      fun q => by simp only [step, Ref.step]; rw [set_eff hi, he]⟩
  | remove p =>
    have hd : (remove v p).nrows = v.nrows ∧ (remove v p).ncols = v.ncols := by
      unfold remove; simp only; by_cases hm : p ∈ (flush v).m <;> simp [hm]
    exact ⟨remove_inv hi p, by simp [step, Ref.step, hr, hd.1], by simp [step, Ref.step, hc, hd.2],
      fun q => by simp only [step, Ref.step]; rw [remove_eff hi, he]⟩
  | setAll n es =>
    have := setAll_inv_eff hi n es hp.1 hp.2
    exact ⟨this.1, by simp [step, Ref.step, hr, this.2.1], by simp [step, Ref.step, hc, this.2.2.1],
      fun q => by simp only [step, Ref.step]; rw [this.2.2.2, he]⟩
  | setProduct n rs cs =>
    have := setProduct_inv_eff hi n rs cs hp.1 hp.2
    exact ⟨this.1, by simp [step, Ref.step, hr, this.2.1], by simp [step, Ref.step, hc, this.2.2.1],
      fun q => by simp only [step, Ref.step]; rw [this.2.2.2, he]⟩
  | removeMask msk =>
    have := removeMask_inv_eff hi msk
    exact ⟨this.1, by simp [step, Ref.step, hr, this.2.1], by simp [step, Ref.step, hc, this.2.2.1],
      fun q => by simp only [step, Ref.step]; rw [this.2.2.2, he]⟩
  | resize r c =>
    simp only [step, resize, Ref.step, hr, hc]
    split
    · have := shrink_inv_eff hi r c
      exact ⟨this.1, this.2.1.symm, this.2.2.1.symm, fun q => by rw [this.2.2.2, he]⟩
    · have := grow_inv_eff hi (r := r) (c := c) (by omega) (by omega)
      exact ⟨this.1, this.2.1.symm, this.2.2.1.symm, fun q => by rw [this.2.2.2, he]⟩
  | dup a b =>
    exact ⟨⟨hi.1, hi.2, hi.3, hi.4, hi.5, hi.6, hi.7⟩, by simp [step, Ref.step, dup, hr],
      by simp [step, Ref.step, dup, hc], fun q => he q⟩
  | foldNow a b =>
    have := fold_inv hi a b
    refine ⟨⟨this.1, this.2, this.3, this.4, this.5, this.6, this.7⟩,
      by simp [step, Ref.step, foldNow, flush, fold, hr], by simp [step, Ref.step, foldNow, flush, fold, hc],
      fun q => ?_⟩
    simp only [step, Ref.step, foldNow, flush]; rw [← he q]; exact fold_eff hi a b q
  | flush =>
    exact ⟨flush_inv hi, by simp [step, Ref.step, hr], by simp [step, Ref.step, hc],
      fun q => by simp only [step, Ref.step]; rw [flush_eff hi, he]⟩
  | wait => exact ⟨hi, hr, hc, he⟩

/-- Run a list of operations. -/
def run (v : VM) : List Op → VM
  | [] => v
  | op :: ops => run (step v op) ops

def Ref.run (R : Ref) : List Op → Ref
  | [] => R
  | op :: ops => Ref.run (R.step op) ops

/-- Every step of the sequence respects its caller contract in the state it runs on. -/
def PreRun (v : VM) : List Op → Prop
  | [] => True
  | op :: ops => Pre v op ∧ PreRun (step v op) ops

/-- **Main theorem.** After *any* sequence of set / remove / bulk set / bulk
product / masked remove / grow / shrink / dup / fold / flush / wait, with
*any* fold decisions, the logical matrix equals the reference and every
layer invariant holds. -/
theorem run_repr : ∀ (ops : List Op) {v : VM} {R : Ref}, Repr v R → PreRun v ops →
    Repr (run v ops) (R.run ops)
  | [], _, _, h, _ => h
  | op :: ops, _, _, h, hp => run_repr ops (step_repr h op hp.1) hp.2

theorem run_from_empty (r c : Nat) (ops : List Op) (hp : PreRun (empty r c) ops) :
    ∀ q, eff (run (empty r c) ops) q ↔ (Ref.run ⟨fun _ => False, r, c⟩ ops).s q :=
  (run_repr ops ⟨inv_empty r c, rfl, rfl, fun q => by simp [empty, eff]⟩ hp).2.2.2

/-! ### `get` and `nvals` -/

/-- `get` probes `m` first; under `Inv` that is the logical matrix. -/
theorem get_iff_eff {v : VM} (h : Inv v) (p : Pair) : get v p = true ↔ eff v p := by
  have := h.dp_m p
  unfold get eff; split <;> simp_all

/-- Without `dp ∩ m = ∅` the two read paths disagree (`get` says absent, the
iterator/`extract` formula says present) — the invariant is load-bearing. -/
theorem get_eff_disagree_without_inv :
    let v : VM := ⟨[(0,0)], [(0,0)], [(0,0)], 1, 1, none⟩
    get v (0,0) = false ∧ eff v (0,0) := by
  simp [get, eff]

/-- The materialized logical matrix (`extract`, :773). -/
def effList (v : VM) : List Pair := v.m.filter (fun p => !(decide (p ∈ v.dm))) ++ v.dp

theorem mem_effList {v : VM} (p : Pair) : p ∈ effList v ↔ eff v p := by
  simp only [effList, List.mem_append, mem_filter_not, eff]

theorem effList_nodup {v : VM} (h : Inv v) : (effList v).Nodup := by
  refine nodup_app (nodup_filter h.nd_m _) h.nd_dp ?_
  intro x hx hx'; rw [mem_filter_not] at hx'; exact h.dp_m _ hx hx'.1

theorem len_filter_mem {m d : Layer} (hm : m.Nodup) (hd : d.Nodup) (hsub : ∀ p ∈ d, p ∈ m) :
    (m.filter (fun p => decide (p ∈ d))).length = d.length := by
  apply List.Perm.length_eq
  refine (List.perm_ext_iff_of_nodup (nodup_filter hm _) hd).2 ?_
  intro a; rw [mem_filter_dec]; exact ⟨fun h => h.2, fun h => ⟨hsub a h, h⟩⟩

/-- **`nvals`** (:793): `|m| + |dp| − |dm|` is the size of the logical matrix,
and the `u64` subtraction cannot underflow (`|dm| ≤ |m|`). -/
theorem nvals_eq {v : VM} (h : Inv v) :
    v.dm.length ≤ v.m.length ∧
      v.m.length + v.dp.length - v.dm.length = (effList v).length := by
  have hsplit := List.length_eq_countP_add_countP (fun p => decide (p ∈ v.dm)) (l := v.m)
  rw [List.countP_eq_length_filter, List.countP_eq_length_filter,
    len_filter_mem h.nd_m h.nd_dm h.dm_m] at hsplit
  have : (v.m.filter (fun p => !(decide (p ∈ v.dm)))).length =
      (v.m.filter (fun a => decide ¬(decide (a ∈ v.dm)) = true)).length := by
    congr 1; apply List.filter_congr; intro x _; simp
  simp only [effList, List.length_append]
  omega

/-! ### set-after-remove, remove-after-set: the cancel cases, layer by layer -/

/-- From a committed entry (tombstoned or not), `remove` then `set` leaves no delta on it. -/
theorem remove_set_committed {v : VM} (h : Inv v) (hn : v.armed = none) {p : Pair}
    (hm : p ∈ v.m) :
    let w := set (remove v p) p
    p ∈ w.m ∧ p ∉ w.dm ∧ p ∉ w.dp := by
  have hdp := h.dp_m p
  simp only [set, remove, flush, hn, hm, ite_true, mem_del, mem_ins]
  exact ⟨trivial, fun h => h.1 rfl, fun hp => hdp hp hm⟩

/-- From an absent entry, `set` then `remove` leaves no delta on it. -/
theorem set_remove_absent {v : VM} (hn : v.armed = none) {p : Pair} (hm : p ∉ v.m) :
    let w := remove (set v p) p
    p ∉ w.m ∧ p ∉ w.dp := by
  simp [set, remove, flush, hn, hm, mem_del, mem_ins]

/-! ### the `NEW` contract is load-bearing -/

/-- Violating `set_all::<true>`'s contract (a pair already live in `m`) puts it
in both `m` and `dp`, and `nvals` then double-counts it: 2 instead of 1. -/
theorem setAll_new_contract_needed :
    let v : VM := ⟨[(0,0)], [], [], 1, 1, none⟩
    let w := setAll true v [(0,0)]
    w.dp = [(0,0)] ∧ w.m = [(0,0)] ∧ w.m.length + w.dp.length - w.dm.length = 2 := by
  decide

end VM
