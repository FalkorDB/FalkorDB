/-
# Sorted merges with exact-duplicate collapsing
(`merge_sorted` `leaf/mod.rs:50`, `merge_walk` `leaf/mod.rs:97`, the distinct
merge of `CompactIndexedLeaf::merge` `leaf/compact_indexed.rs`), and
`partition_point` (`leaf/mod.rs:77`) / `gallop_lower_bound`.

One generic algorithm over a linear order `le`: take the left head when it is
`<=` the right head, skip a value equal to the last one emitted.
-/
namespace IndexLayer.Leaf

section
variable {α : Type} [DecidableEq α] (le : α → α → Bool)

def emit (x : α) (rest : List α) (last : Option α) : List α :=
  if last = some x then rest else x :: rest

def mergeD : List α → List α → Option α → List α
  | [], [], _ => []
  | a :: as, [], l => emit a (mergeD as [] (some a)) l
  | [], b :: bs, l => emit b (mergeD [] bs (some b)) l
  | a :: as, b :: bs, l =>
    if le a b then emit a (mergeD as (b :: bs) (some a)) l
    else emit b (mergeD (a :: as) bs (some b)) l
termination_by l r => l.length + r.length

structure LinOrd : Prop where
  refl : ∀ a, le a a = true
  trans : ∀ a b c, le a b = true → le b c = true → le a c = true
  total : ∀ a b, le a b = true ∨ le b a = true
  antisymm : ∀ a b, le a b = true → le b a = true → a = b

def lt (a b : α) : Prop := le a b = true ∧ a ≠ b

/-- The four facts the merge guarantees. -/
def MSpec (l r : List α) (last : Option α) (out : List α) : Prop :=
  (∀ y ∈ out, y ∈ l ∨ y ∈ r) ∧ (∀ y ∈ out, last ≠ some y) ∧
  (∀ y, y ∈ l ∨ y ∈ r → y ∈ out ∨ last = some y) ∧ out.Pairwise (lt le)

theorem emit_spec (O : LinOrd le) (a : α) (src : List α) (last : Option α) (rest : List α)
    (hsrc : a ∈ src) (habove : ∀ y ∈ src, le a y = true) (hlast : last.all (le · a) = true)
    (h1 : ∀ y ∈ rest, y ∈ src) (h2 : ∀ y ∈ rest, some a ≠ some y)
    (h3 : ∀ y ∈ src, y ∈ rest ∨ some a = some y) (h4 : rest.Pairwise (lt le)) :
    (∀ y ∈ emit a rest last, y ∈ src) ∧ (∀ y ∈ emit a rest last, last ≠ some y) ∧
    (∀ y ∈ src, y ∈ emit a rest last ∨ last = some y) ∧ (emit a rest last).Pairwise (lt le) := by
  unfold emit
  split
  · next he =>
    subst he
    exact ⟨h1, h2, fun y hy => (h3 y hy), h4⟩
  · next he =>
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro y hy; rcases List.mem_cons.mp hy with rfl | hy
      · exact hsrc
      · exact h1 y hy
    · intro y hy; rcases List.mem_cons.mp hy with rfl | hy
      · exact he
      · intro hl; subst hl
        have hy' := h1 y hy
        have h_ya : le y a = true := by simpa using hlast
        have := O.antisymm _ _ h_ya (habove y hy')
        exact h2 y hy (by rw [this])
    · intro y hy
      rcases h3 y hy with h | h
      · exact Or.inl (List.mem_cons_of_mem _ h)
      · simp at h; subst h; exact Or.inl (by simp)
    · refine List.Pairwise.cons (fun y hy => ⟨habove y (h1 y hy), fun e => h2 y hy (by rw [e])⟩) h4

/-- **PROVEN**: on non-decreasing inputs, all at or above `last`, the merge
emits exactly the union (every value at most once, `last` itself excluded),
strictly increasing. -/
theorem mergeD_spec (O : LinOrd le) : ∀ (l r : List α) (last : Option α),
    l.Pairwise (fun a b => le a b = true) → r.Pairwise (fun a b => le a b = true) →
    (∀ y, y ∈ l ∨ y ∈ r → last.all (le · y) = true) →
    MSpec le l r last (mergeD le l r last)
  | [], [], last, _, _, _ => by simp [mergeD, MSpec]
  | a :: as, [], last, hl, hr, hb => by
    rw [mergeD]
    have hl' := List.pairwise_cons.mp hl
    obtain ⟨i1, i2, i3, i4⟩ := mergeD_spec O as [] (some a) hl'.2 List.Pairwise.nil
      (fun y hy => (hy).elim (fun hy => by simpa using hl'.1 y hy) (fun hy => by simp at hy))
    have := emit_spec le O a (a :: as) last (mergeD le as [] (some a)) (by simp)
      (fun y hy => (List.mem_cons.mp hy).elim (fun h => by subst h; exact O.refl _) (fun hy => by exact hl'.1 y hy))
      (hb a (Or.inl (by simp)))
      (fun y hy => (i1 y hy).elim (fun h => by exact List.mem_cons_of_mem _ h) (fun h => by simp at h))
      i2
      (fun y hy => by rcases List.mem_cons.mp hy with rfl | hy
                      · exact Or.inr rfl
                      · exact i3 y (Or.inl hy))
      i4
    obtain ⟨e1, e2, e3, e4⟩ := this
    exact ⟨fun y hy => Or.inl (e1 y hy), e2, fun y hy => by
      exact hy.elim (fun hy => e3 y hy) (fun hy => by simp at hy), e4⟩
  | [], b :: bs, last, hl, hr, hb => by
    rw [mergeD]
    have hr' := List.pairwise_cons.mp hr
    obtain ⟨i1, i2, i3, i4⟩ := mergeD_spec O [] bs (some b) List.Pairwise.nil hr'.2
      (fun y hy => (hy).elim (fun hy => by simp at hy) (fun hy => by simpa using hr'.1 y hy))
    have := emit_spec le O b (b :: bs) last (mergeD le [] bs (some b)) (by simp)
      (fun y hy => (List.mem_cons.mp hy).elim (fun h => by subst h; exact O.refl _) (fun hy => by exact hr'.1 y hy))
      (hb b (Or.inr (by simp)))
      (fun y hy => (i1 y hy).elim (fun h => by simp at h) (fun h => by exact List.mem_cons_of_mem _ h))
      i2
      (fun y hy => by rcases List.mem_cons.mp hy with rfl | hy
                      · exact Or.inr rfl
                      · exact i3 y (Or.inr hy))
      i4
    obtain ⟨e1, e2, e3, e4⟩ := this
    exact ⟨fun y hy => Or.inr (e1 y hy), e2, fun y hy => by
      exact hy.elim (fun hy => by simp at hy) (fun hy => e3 y hy), e4⟩
  | a :: as, b :: bs, last, hl, hr, hb => by
    rw [mergeD]
    have hl' := List.pairwise_cons.mp hl
    have hr' := List.pairwise_cons.mp hr
    split
    · next hab =>
      have above : ∀ y, y ∈ as ∨ y ∈ b :: bs → le a y = true := by
        intro y hy; rcases hy with hy | hy
        · exact hl'.1 y hy
        · rcases List.mem_cons.mp hy with rfl | hy
          · exact hab
          · exact O.trans _ _ _ hab (hr'.1 y hy)
      obtain ⟨i1, i2, i3, i4⟩ := mergeD_spec O as (b :: bs) (some a) hl'.2 hr
        (fun y hy => by simpa using above y hy)
      have := emit_spec le O a (a :: as ++ b :: bs) last (mergeD le as (b :: bs) (some a)) (by simp)
        (fun y hy => by
          rcases List.mem_append.mp hy with hy | hy
          · exact (List.mem_cons.mp hy).elim (fun h => by subst h; exact O.refl _) (fun hy => above y (Or.inl hy))
          · exact above y (Or.inr hy))
        (hb a (Or.inl (by simp)))
        (fun y hy => by rcases i1 y hy with h | h
                        · exact List.mem_append_left _ (List.mem_cons_of_mem _ h)
                        · exact List.mem_append_right _ h)
        i2
        (fun y hy => by
          rcases List.mem_append.mp hy with hy | hy
          · rcases List.mem_cons.mp hy with rfl | hy
            · exact Or.inr rfl
            · exact i3 y (Or.inl hy)
          · exact i3 y (Or.inr hy))
        i4
      obtain ⟨e1, e2, e3, e4⟩ := this
      exact ⟨fun y hy => List.mem_append.mp (e1 y hy), e2,
        fun y hy => e3 y (List.mem_append.mpr hy), e4⟩
    · next hab =>
      have hba : le b a = true := by
        exact (O.total a b).elim (fun h => by simp_all) id
      have above : ∀ y, y ∈ a :: as ∨ y ∈ bs → le b y = true := by
        intro y hy; rcases hy with hy | hy
        · rcases List.mem_cons.mp hy with rfl | hy
          · exact hba
          · exact O.trans _ _ _ hba (hl'.1 y hy)
        · exact hr'.1 y hy
      obtain ⟨i1, i2, i3, i4⟩ := mergeD_spec O (a :: as) bs (some b) hl hr'.2
        (fun y hy => by simpa using above y hy)
      have := emit_spec le O b (a :: as ++ b :: bs) last (mergeD le (a :: as) bs (some b)) (by simp)
        (fun y hy => by
          rcases List.mem_append.mp hy with hy | hy
          · exact above y (Or.inl hy)
          · exact (List.mem_cons.mp hy).elim (fun h => by subst h; exact O.refl _) (fun hy => above y (Or.inr hy)))
        (hb b (Or.inr (by simp)))
        (fun y hy => by rcases i1 y hy with h | h
                        · exact List.mem_append_left _ h
                        · exact List.mem_append_right _ (List.mem_cons_of_mem _ h))
        i2
        (fun y hy => by
          rcases List.mem_append.mp hy with hy | hy
          · exact i3 y (Or.inl hy)
          · rcases List.mem_cons.mp hy with rfl | hy
            · exact Or.inr rfl
            · exact i3 y (Or.inr hy))
        i4
      obtain ⟨e1, e2, e3, e4⟩ := this
      exact ⟨fun y hy => List.mem_append.mp (e1 y hy), e2,
        fun y hy => e3 y (List.mem_append.mpr hy), e4⟩
termination_by l r => l.length + r.length

/-- `merge_walk` additionally reports which side each emitted value came from
(`Some(leaf_index)` / `None`); its value sequence is `mergeD`. -/
def mergeW : List α → List α → Option α → List (α × Bool)
  | [], [], _ => []
  | a :: as, [], l => (if l = some a then [] else [(a, true)]) ++ mergeW as [] (some a)
  | [], b :: bs, l => (if l = some b then [] else [(b, false)]) ++ mergeW [] bs (some b)
  | a :: as, b :: bs, l =>
    if le a b then (if l = some a then [] else [(a, true)]) ++ mergeW as (b :: bs) (some a)
    else (if l = some b then [] else [(b, false)]) ++ mergeW (a :: as) bs (some b)
termination_by l r => l.length + r.length

theorem mergeW_fst : ∀ (l r : List α) (last : Option α), (mergeW le l r last).map (·.1) = mergeD le l r last
  | [], [], _ => by simp [mergeW, mergeD]
  | a :: as, [], l => by
    rw [mergeW, mergeD, List.map_append, mergeW_fst as [] (some a)]; unfold emit; split <;> simp
  | [], b :: bs, l => by
    rw [mergeW, mergeD, List.map_append, mergeW_fst [] bs (some b)]; unfold emit; split <;> simp
  | a :: as, b :: bs, l => by
    rw [mergeW, mergeD]
    split
    · rw [List.map_append, mergeW_fst as (b :: bs) (some a)]; unfold emit; split <;> simp
    · rw [List.map_append, mergeW_fst (a :: as) bs (some b)]; unfold emit; split <;> simp
termination_by l r => l.length + r.length

/-- Values tagged `true` come from the left (leaf) input. -/
theorem mergeW_left : ∀ (l r : List α) (last : Option α) (x : α),
    (x, true) ∈ mergeW le l r last → x ∈ l
  | [], [], _, _, h => by simp [mergeW] at h
  | a :: as, [], l, x, h => by
    rw [mergeW] at h
    rcases List.mem_append.mp h with h | h
    · split at h <;> simp at h; subst h; simp
    · exact List.mem_cons_of_mem _ (mergeW_left as [] _ x h)
  | [], b :: bs, l, x, h => by
    rw [mergeW] at h
    rcases List.mem_append.mp h with h | h
    · split at h <;> simp at h
    · exact mergeW_left [] bs _ x h
  | a :: as, b :: bs, l, x, h => by
    rw [mergeW] at h
    split at h
    · rcases List.mem_append.mp h with h | h
      · split at h <;> simp at h; subst h; simp
      · exact List.mem_cons_of_mem _ (mergeW_left as (b :: bs) _ x h)
    · rcases List.mem_append.mp h with h | h
      · split at h <;> simp at h
      · exact mergeW_left (a :: as) bs _ x h
termination_by l r => l.length + r.length

/-- `merge_walk` exactly: leaf entries carry their leaf index (`Some(i)`), batch
entries `None`. `k` is the index of the current leaf head. -/
def mergeWI : List α → Nat → List α → Option α → List (α × Option Nat)
  | [], _, [], _ => []
  | a :: as, k, [], l => (if l = some a then [] else [(a, some k)]) ++ mergeWI as (k + 1) [] (some a)
  | [], k, b :: bs, l => (if l = some b then [] else [(b, none)]) ++ mergeWI [] k bs (some b)
  | a :: as, k, b :: bs, l =>
    if le a b then (if l = some a then [] else [(a, some k)]) ++ mergeWI as (k + 1) (b :: bs) (some a)
    else (if l = some b then [] else [(b, none)]) ++ mergeWI (a :: as) k bs (some b)
termination_by l _ r => l.length + r.length

theorem mergeWI_fst : ∀ (l : List α) (k : Nat) (r : List α) (last : Option α),
    (mergeWI le l k r last).map (·.1) = mergeD le l r last
  | [], _, [], _ => by simp [mergeWI, mergeD]
  | a :: as, k, [], l => by
    rw [mergeWI, mergeD, List.map_append, mergeWI_fst as (k + 1) [] (some a)]; unfold emit; split <;> simp
  | [], k, b :: bs, l => by
    rw [mergeWI, mergeD, List.map_append, mergeWI_fst [] k bs (some b)]; unfold emit; split <;> simp
  | a :: as, k, b :: bs, l => by
    rw [mergeWI, mergeD]
    split
    · rw [List.map_append, mergeWI_fst as (k + 1) (b :: bs) (some a)]; unfold emit; split <;> simp
    · rw [List.map_append, mergeWI_fst (a :: as) k bs (some b)]; unfold emit; split <;> simp
termination_by l _ r => l.length + r.length

/-- A leaf-tagged emission is the leaf entry at that index. -/
theorem mergeWI_idx : ∀ (l : List α) (k : Nat) (r : List α) (last : Option α) (x : α) (i : Nat),
    (x, some i) ∈ mergeWI le l k r last → ∃ h : i - k < l.length, k ≤ i ∧ l[i - k] = x
  | [], _, [], _, _, _, h => by simp [mergeWI] at h
  | a :: as, k, [], l, x, i, h => by
    rw [mergeWI] at h
    rcases List.mem_append.mp h with h | h
    · split at h <;> simp at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨by simp, Nat.le_refl _, by simp⟩
    · obtain ⟨h1, h2, h3⟩ := mergeWI_idx as (k + 1) [] _ x i h
      refine ⟨by simp; omega, by omega, ?_⟩
      have e : i - k = (i - (k + 1)) + 1 := by omega
      simp only [e, List.getElem_cons_succ]; exact h3
  | [], k, b :: bs, l, x, i, h => by
    rw [mergeWI] at h
    rcases List.mem_append.mp h with h | h
    · split at h <;> simp at h
    · exact mergeWI_idx [] k bs _ x i h
  | a :: as, k, b :: bs, l, x, i, h => by
    rw [mergeWI] at h
    split at h
    · rcases List.mem_append.mp h with h | h
      · split at h <;> simp at h; obtain ⟨rfl, rfl⟩ := h; exact ⟨by simp, Nat.le_refl _, by simp⟩
      · obtain ⟨h1, h2, h3⟩ := mergeWI_idx as (k + 1) (b :: bs) _ x i h
        refine ⟨by simp; omega, by omega, ?_⟩
        have e : i - k = (i - (k + 1)) + 1 := by omega
        simp only [e, List.getElem_cons_succ]; exact h3
    · rcases List.mem_append.mp h with h | h
      · split at h <;> simp at h
      · exact mergeWI_idx (a :: as) k bs _ x i h
termination_by l _ r => l.length + r.length

/-- The per-emission byte source of a merge: raw leaf bytes or a fresh encoding. -/
def pickEnc {β : Type} (raw : Nat → List β) (encN : α → List β) (e : α × Option Nat) : List β :=
  match e.2 with | some i => raw i | none => encN e.1

/-- Copying the raw encoding of a leaf entry is the same as re-encoding it. -/
theorem flatMap_raw {β : Type} (out : List (α × Option Nat)) (l : List α) (enc : α → List β)
    (raw : Nat → List β) (hraw : ∀ i (hi : i < l.length), raw i = enc l[i])
    (hout : ∀ x i, (x, some i) ∈ out → ∃ h : i < l.length, l[i] = x)
    (encN : α → List β) (hN : ∀ x, encN x = enc x) :
    out.flatMap (pickEnc raw encN) = out.flatMap (fun e => enc e.1) := by
  induction out with
  | nil => rfl
  | cons e es ih =>
    simp only [List.flatMap_cons]
    rw [ih (fun x i h => hout x i (List.mem_cons_of_mem _ h))]
    obtain ⟨x, oi⟩ := e
    cases oi with
    | none => simp only [pickEnc]; rw [hN]
    | some i =>
      obtain ⟨h1, h2⟩ := hout x i (by simp)
      simp only [pickEnc]; rw [hraw i h1, h2]

end

/-! ## `partition_point` and `gallop_lower_bound` -/

/-- `partition_point(lo..hi, p)` (binary search), with fuel `hi - lo + 1`. -/
def pp (p : Nat → Bool) : Nat → Nat → Nat → Nat
  | 0, lo, _ => lo
  | f + 1, lo, hi =>
    if lo < hi then
      let mid := lo + (hi - lo) / 2
      if p mid then pp p f (mid + 1) hi else pp p f lo mid
    else lo

def partitionPoint (p : Nat → Bool) (lo hi : Nat) : Nat := pp p (hi - lo + 1) lo hi

/-- `p` holds on a prefix of `[lo, hi)` and fails after it. -/
def Mono (p : Nat → Bool) (lo hi : Nat) : Prop := ∀ i j, lo ≤ i → i ≤ j → j < hi → p j = true → p i = true

theorem pp_spec (p : Nat → Bool) : ∀ (f lo hi : Nat), hi - lo < f → lo ≤ hi → Mono p lo hi →
    lo ≤ pp p f lo hi ∧ pp p f lo hi ≤ hi ∧ (∀ i, lo ≤ i → i < pp p f lo hi → p i = true) ∧
      (∀ i, pp p f lo hi ≤ i → i < hi → p i = false)
  | 0, _, _, h, _, _ => absurd h (Nat.not_lt_zero _)
  | f + 1, lo, hi, hf, hle, hm => by
    simp only [pp]
    split
    · next hlt =>
      have hmid1 : lo ≤ lo + (hi - lo) / 2 := Nat.le_add_right _ _
      have hmid2 : lo + (hi - lo) / 2 < hi := by omega
      split
      · next hp =>
        obtain ⟨a1, a2, a3, a4⟩ := pp_spec p f (lo + (hi - lo) / 2 + 1) hi (by omega) (by omega)
          (fun i j hi' hij hj hpj => hm i j (by omega) hij hj hpj)
        refine ⟨by omega, a2, fun i hi1 hi2 => ?_, a4⟩
        by_cases hc : i ≤ lo + (hi - lo) / 2
        · exact hm i _ hi1 hc hmid2 hp
        · exact a3 i (by omega) hi2
      · next hp =>
        obtain ⟨a1, a2, a3, a4⟩ := pp_spec p f lo (lo + (hi - lo) / 2) (by omega) hmid1
          (fun i j hi' hij hj hpj => hm i j hi' hij (by omega) hpj)
        refine ⟨a1, by omega, a3, fun i hi1 hi2 => ?_⟩
        by_cases hc : i < lo + (hi - lo) / 2
        · exact a4 i hi1 hc
        · cases hpi : p i
          · rfl
          · exact absurd (hm _ i hmid1 (by omega) hi2 hpi) (by simp [hp])
    · next hlt => refine ⟨Nat.le_refl _, hle, fun i h1 h2 => absurd h2 (by omega), fun i h1 h2 => by omega⟩

/-- **PROVEN**: on a predicate true on a prefix of `[lo, hi)`, `partition_point`
returns the first index where it fails (`hi` if none). -/
theorem partitionPoint_spec (p : Nat → Bool) (lo hi : Nat) (hle : lo ≤ hi) (hm : Mono p lo hi) :
    lo ≤ partitionPoint p lo hi ∧ partitionPoint p lo hi ≤ hi ∧
    (∀ i, lo ≤ i → i < partitionPoint p lo hi → p i = true) ∧
    (∀ i, partitionPoint p lo hi ≤ i → i < hi → p i = false) :=
  pp_spec p _ lo hi (by omega) hle hm

/-- `gallop_lower_bound(start, count, p)` (`leaf/compact_indexed.rs`): double the
step while `p(start + step)` holds, then binary-search the last window. -/
def gallopStep (p : Nat → Bool) (start count : Nat) : Nat → Nat → Nat
  | 0, step => step
  | f + 1, step => if start + step < count ∧ p (start + step) = true then gallopStep p start count f (2 * step) else step

def gallop (p : Nat → Bool) (start count : Nat) : Nat :=
  let step := gallopStep p start count count 1
  partitionPoint p (start + step / 2) (min (start + step) count)

theorem gallopStep_spec (p : Nat → Bool) (start count : Nat) : ∀ f step, 1 ≤ step →
    (step = 1 ∨ (start + step / 2 < count ∧ p (start + step / 2) = true)) →
    let s := gallopStep p start count f step
    1 ≤ s ∧ (s = 1 ∨ (start + s / 2 < count ∧ p (start + s / 2) = true)) ∧
      (¬ (start + s < count ∧ p (start + s) = true) ∨ 2 ^ f * step ≤ s)
  | 0, step, h1, h2 => by simp [gallopStep]; exact ⟨h1, h2⟩
  | f + 1, step, h1, h2 => by
    simp only [gallopStep]
    split
    · next hc =>
      have := gallopStep_spec p start count f (2 * step) (by omega)
        (Or.inr (by rw [Nat.mul_div_cancel_left _ (by omega)]; exact hc))
      simp only at this
      refine ⟨this.1, this.2.1, ?_⟩
      rcases this.2.2 with h | h
      · exact Or.inl h
      · right; rw [Nat.pow_succ, Nat.mul_assoc]; exact h
    · next hc => exact ⟨h1, h2, Or.inl hc⟩

/-- **PROVEN**: on a predicate true on a prefix of `[start, count)`, the gallop
returns the first index `≥ start` where it fails (or `count`). -/
theorem gallop_spec (p : Nat → Bool) (start count : Nat) (hs : start ≤ count) (hm : Mono p start count) :
    start ≤ gallop p start count ∧ gallop p start count ≤ count ∧
    (∀ i, start ≤ i → i < gallop p start count → p i = true) ∧
    (∀ i, gallop p start count ≤ i → i < count → p i = false) := by
  unfold gallop
  simp only
  generalize hst : gallopStep p start count count 1 = s
  have hg := gallopStep_spec p start count count 1 (Nat.le_refl _) (Or.inl rfl)
  simp only [hst] at hg
  obtain ⟨g1, g2, g3⟩ := hg
  -- the window [start + s/2, min (start+s) count)
  have hlo : start + s / 2 ≤ min (start + s) count := by
    rcases g2 with h | h
    · subst h; simp; omega
    · have : s / 2 ≤ s := Nat.div_le_self _ _
      omega
  -- elements below the window satisfy p
  have below : ∀ i, start ≤ i → i < start + s / 2 → p i = true := by
    intro i h1 h2
    rcases g2 with h | h
    · subst h; omega
    · exact hm i (start + s / 2) h1 (by omega) h.1 h.2
  -- elements at or after the window end fail p
  have after : ∀ i, min (start + s) count ≤ i → i < count → p i = false := by
    intro i h1 h2
    have hc : ¬ (start + s < count ∧ p (start + s) = true) := by
      rcases g3 with h | h
      · exact h
      · have : count < 2 ^ count := Nat.lt_two_pow_self
        omega
    cases hpi : p i
    · rfl
    · exfalso; apply hc
      have hlt : start + s < count := by
        by_cases hmin : start + s ≤ count
        · rw [Nat.min_eq_left hmin] at h1
          rcases Nat.lt_or_ge (start + s) count with h | h
          · exact h
          · omega
        · rw [Nat.min_eq_right (by omega)] at h1; omega
      exact ⟨hlt, hm _ i (by omega) (by rw [Nat.min_eq_left (by omega)] at h1; exact h1) h2 hpi⟩
  obtain ⟨a1, a2, a3, a4⟩ := partitionPoint_spec p (start + s / 2) (min (start + s) count) hlo
    (fun i j h1 h2 h3 h4 => hm i j (by omega) h2 (by omega) h4)
  refine ⟨by omega, by omega, fun i h1 h2 => ?_, fun i h1 h2 => ?_⟩
  · by_cases hc : i < start + s / 2
    · exact below i h1 hc
    · exact a3 i (by omega) h2
  · by_cases hc : i < min (start + s) count
    · exact a4 i h1 hc
    · exact after i (by omega) h2

end IndexLayer.Leaf
