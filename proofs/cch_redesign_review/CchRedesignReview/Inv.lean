import CchRedesignReview.Model
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# The invariants

`PInv B w S`: what holds of every published version `S` (the current one and every earlier one, since a writer
or a job may still hold any of them as its fork), given the committed writes `w`:

* `nlw` (03 claim, task property 1): every write of `p` at `s ≤ S.ver` is resolved in `S`'s structure
  (`s ≤ res p`) or still in `S`'s log;
* `built` (property 2): a `Built` entry has every write resolved, with no log needed;
* `noOv`, `upNone`, `upSome` (property 3): a `Built` entry has no Overflow; its log is empty without `upkeep`, and
  with `upkeep = Some{since}` holds only nodes with `seq > since` and every write after `since`;
* `bound` (property 5): a `Building` entry's log is under the bound or holds an Overflow;
* `cnt`, `sound`, `seq_le`, `res_le`: the counter is exact, every logged pair is a committed write at its seq,
  nothing is logged or resolved from the future.

`XRel B cur S0`: how an earlier published `S0` relates to the current one (what a fork may assume):

* `same_ver`: within one data version the generation is constant, and the batch count names the entry;
* `same_gen`: within a generation `indexer_commits` never decreases (no publish is reverted), and once `Built`
  stays `Built`;
* `chain`: while `Building` with the same generation and count, the current entry `Ext`ends `S0`'s: only writers
  moved it, by appending nodes after `S0.ver` (or overflowing);
* `ticket`: a ticket names one `upkeep`, set within one generation and one batch count: between the version
  that set it and any later one that still has it, no batch or job of this index published.

`Inv.upw`: while `upkeep = Some u` is published, every version published at or after `u.since` has the same
generation, count and `upkeep`, so no writer escalating then is overtaken.
-/
namespace CchReview

structure PInv (B : Nat) (w : List (Nat × Pair)) (S : Pub) : Prop where
  res_le : ∀ p, S.e.res p ≤ S.ver
  seq_le : ∀ n ∈ S.e.log, n.seq ≤ S.ver
  cnt : S.e.pairs = sumPairs S.e.log
  sound : ∀ n ∈ S.e.log, ∀ p ∈ n.ps, (n.seq, p) ∈ w
  nlw : ∀ s p, (s, p) ∈ w → s ≤ S.ver → s ≤ S.e.res p ∨ covered S.e.log s p
  built : S.e.isBuilt = true → ∀ s p, (s, p) ∈ w → s ≤ S.ver → s ≤ S.e.res p
  noOv : S.e.isBuilt = true → hasOv S.e.log = false
  upNone : S.e.st = .built none → S.e.log = []
  upSome : ∀ u, S.e.st = .built (some u) → u.since ≤ S.ver ∧ (∀ n ∈ S.e.log, u.since < n.seq) ∧
    ∀ s p, (s, p) ∈ w → u.since < s → s ≤ S.ver → ∃ n ∈ S.e.log, n.seq = s ∧ p ∈ n.ps
  bound : S.e.isBuilt = false → S.e.pairs ≤ B ∨ hasOv S.e.log = true

/-- Ids in use are below the counter. -/
structure Fresh (nx : Nat) (e : Entry) : Prop where
  gen : e.gen < nx
  lineage : e.lineage < nx
  ticket : ∀ u, e.upkeep = some u → u.ticket < nx

/-- `A` extends `A0` by writers after `F`: the same phase state, structure, generation and count; the log is new
    nodes (all after `F`) on top of `A0`'s, and then under the bound unless nothing was added; or an Overflow
    replaced it. -/
def Ext (B F : Nat) (A0 A : Entry) : Prop :=
  A.st = A0.st ∧ A.res = A0.res ∧ A.gen = A0.gen ∧ A.commits = A0.commits ∧ A.lineage = A0.lineage ∧
  ∃ N : List Node, (∀ n ∈ N, F < n.seq) ∧
    ((A.log = N ++ A0.log ∧ (N = [] ∨ sumPairs A.log ≤ B)) ∨ (A.log = N ∧ hasOv N = true))

structure XRel (B : Nat) (cur S0 : Pub) : Prop where
  ver_le : S0.ver < cur.ver ∨ (S0.ver = cur.ver ∧ S0.batches < cur.batches)
  same_ver : S0.ver = cur.ver → S0.e.gen = cur.e.gen ∧ (S0.e.commits = cur.e.commits → S0.e = cur.e)
  same_gen : S0.e.gen = cur.e.gen → S0.e.commits ≤ cur.e.commits ∧ (S0.e.isBuilt = true → cur.e.isBuilt = true) ∧
    (S0.e.commits = cur.e.commits → S0.e.isBuilt = cur.e.isBuilt)
  chain : S0.e.isBuilt = false → cur.e.isBuilt = false → S0.e.gen = cur.e.gen → S0.e.commits = cur.e.commits →
    Ext B S0.ver S0.e cur.e
  ticket : ∀ u0 u, S0.e.st = .built (some u0) → cur.e.st = .built (some u) → u0.ticket = u.ticket →
    u0 = u ∧ S0.e.gen = cur.e.gen ∧ S0.e.commits = cur.e.commits

structure Inv (B : Nat) (σ : Sys) : Prop where
  pub : ∀ S ∈ σ.cur :: σ.hist, PInv B σ.writes S ∧ Fresh σ.next S.e
  rel : ∀ S0 ∈ σ.hist, XRel B σ.cur S0
  wver : ∀ x ∈ σ.writes, x.1 ≤ σ.cur.ver
  /-- While `upkeep = Some u` is published, every version published at or after `u.since` has the same
      generation, count and `upkeep`: since the ticket was set, only writers that kept it ran. -/
  upw : ∀ u, σ.cur.e.st = .built (some u) → ∀ S ∈ σ.hist, u.since ≤ S.ver →
    S.e.gen = σ.cur.e.gen ∧ S.e.commits = σ.cur.e.commits ∧ S.e.st = .built (some u)

/-! ## Basic lemmas -/

theorem isBuilt_built (e : Entry) (u : Option Upkeep) (h : e.st = .built u) : e.isBuilt = true := by
  simp [Entry.isBuilt, h]

theorem isBuilt_building (e : Entry) (ph : Phase) (h : e.st = .building ph) : e.isBuilt = false := by
  simp [Entry.isBuilt, h]

theorem isBuilt_false_iff (e : Entry) : e.isBuilt = false ↔ ∃ ph, e.st = .building ph := by
  cases h : e.st <;> simp [Entry.isBuilt, h]

theorem isBuilt_true_iff (e : Entry) : e.isBuilt = true ↔ ∃ u, e.st = .built u := by
  cases h : e.st <;> simp [Entry.isBuilt, h]

theorem upkeep_some (e : Entry) (u : Upkeep) : e.upkeep = some u ↔ e.st = .built (some u) := by
  cases h : e.st <;> simp [Entry.upkeep, h]

/-- `Ext` is reflexive. -/
theorem Ext.refl (B F : Nat) (A : Entry) : Ext B F A A :=
  ⟨rfl, rfl, rfl, rfl, rfl, [], by simp, Or.inl ⟨by simp, Or.inl rfl⟩⟩

/-- `Ext` is weakened by an earlier fork. -/
theorem Ext.weaken {B F F' : Nat} {A0 A : Entry} (h : Ext B F A0 A) (hF : F' ≤ F) : Ext B F' A0 A := by
  obtain ⟨h1, h2, h3, h4, h5, N, hN, hl⟩ := h
  exact ⟨h1, h2, h3, h4, h5, N, fun n hn => Nat.lt_of_le_of_lt hF (hN n hn), hl⟩

/-- `Ext` composes. -/
theorem Ext.trans {B F : Nat} {A0 A1 A2 : Entry} (h1 : Ext B F A0 A1) (h2 : Ext B F A1 A2) : Ext B F A0 A2 := by
  obtain ⟨a1, b1, c1, d1, l1, N1, hN1, hl1⟩ := h1
  obtain ⟨a2, b2, c2, d2, l2, N2, hN2, hl2⟩ := h2
  refine ⟨a2.trans a1, b2.trans b1, c2.trans c1, d2.trans d1, l2.trans l1, ?_⟩
  rcases hl2 with ⟨e2, h2'⟩ | ⟨e2, ov2⟩
  · rcases hl1 with ⟨e1, h1'⟩ | ⟨e1, ov1⟩
    · refine ⟨N2 ++ N1, ?_, Or.inl ⟨by rw [e2, e1, List.append_assoc], ?_⟩⟩
      · intro n hn; rcases List.mem_append.1 hn with h | h
        · exact hN2 n h
        · exact hN1 n h
      · rcases h2' with rfl | h2'
        · rcases h1' with rfl | h1'
          · exact Or.inl rfl
          · right; rw [e2]; simpa using h1'
        · exact Or.inr h2'
    · refine ⟨N2 ++ N1, ?_, Or.inr ⟨by rw [e2, e1], ?_⟩⟩
      · intro n hn; rcases List.mem_append.1 hn with h | h
        · exact hN2 n h
        · exact hN1 n h
      · simp [hasOv_append, ov1]
  · exact ⟨N2, hN2, Or.inr ⟨e2, ov2⟩⟩

/-- Writes with seqs past `S.ver` do not touch what `PInv` says of `S`. -/
theorem PInv.mono {B : Nat} {w w' : List (Nat × Pair)} {S : Pub} (h : PInv B w S)
    (hsub : ∀ x ∈ w, x ∈ w') (hw : ∀ x ∈ w', x ∈ w ∨ S.ver < x.1) : PInv B w' S where
  res_le := h.res_le
  seq_le := h.seq_le
  cnt := h.cnt
  sound := fun n hn p hp => hsub _ (h.sound n hn p hp)
  nlw := by
    intro s p hsp hs
    rcases hw _ hsp with h' | h'
    · exact h.nlw s p h' hs
    · simp at h'; omega
  built := by
    intro hb s p hsp hs
    rcases hw _ hsp with h' | h'
    · exact h.built hb s p h' hs
    · simp at h'; omega
  noOv := h.noOv
  upNone := h.upNone
  upSome := by
    intro u hu
    obtain ⟨a, b, c⟩ := h.upSome u hu
    refine ⟨a, b, fun s p hsp h1 h2 => ?_⟩
    rcases hw _ hsp with h' | h'
    · exact c s p h' h1 h2
    · simp at h'; omega
  bound := h.bound

end CchReview
