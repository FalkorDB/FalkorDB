/-
# Iteration as the merge that computes it

`Iter.lean` characterises the iterators by their *result* — the effective set, as
a multiset. That is what a caller needs, and it is what the ordering-free theorems
are about. It leaves out the part where the delicate index reasoning lives:
`Tensor::iter` does not build the effective set and then walk it, it walks three
ascending GraphBLAS row iterators at once and merges them, which is what makes a
scan allocation-free.

This file models that merge and proves the two things the result-level statement
cannot say:

* [`mem_merge3`] — the merge emits exactly `(m ∖ dm) ∪ dp`, with `dp` winning at a
  shared position (the *shadow rule*), and
* [`sortedBy_merge3`] — it emits in strictly ascending `(src, dst)` order.

The ordering half is the one worth having. Downstream operators assume a scan
arrives sorted, and nothing in this development checked it until now.

## The three cursors

```rust
// forward layers, all three already ascending
m  : (src, dst) -> edge id      // committed base
dm : (src, dst)                 // deletion mask, a lookahead over m
dp : (src, dst) -> edge id      // pending, wins ties with m
```

`dm` is consumed as a lookahead that drops the `m` entry it matches; a `dp` entry
at the same position as an `m` entry wins and suppresses it. Those are the two
rules the paper's merge figure draws, and they are the two branches of [`merge3`]
that do something other than emit and advance.

## Why the merge needs Invariant *purity*

At a tie the algorithm emits the `dp` entry and advances the `m` cursor **without
advancing `dm`**. That is sound only because a shadowed position carries no
tombstone — Invariant purity, `dp ∩ dm = ∅`. Violate it and `dm`'s head stays
pinned at the shadowed position while `m` moves past, so the *next* masked entry
is compared against a stale head, fails the match, and is emitted: a deleted edge
appearing in a scan. `hpure` below is that invariant, and the tie case is where it
is used.

The paper motivates purity as keeping `dm`'s meaning crisp for the fold and
removal paths. The iterator depends on it too, and more sharply.

## The shape of the recursion

The natural way to write this merge — "if `dp` is behind, emit it and advance
`dp`" — recurses on the pending list while holding the base fixed, so it needs a
combined termination measure and compiles by well-founded recursion, whose
equation lemmas are then awkward to rewrite with. Flushing the pending cursor up
front instead makes the recursion structural on the base list, and is closer to
what the loop does anyway: *flush every pending entry below the current base
position, then decide about the base position*. [`takeLt`] is that flush.

## What the model is, and is not

Positions compare lexicographically by `(src, dst)`, which is the order GraphBLAS
row iterators yield. The lists carry one entry per position, mirroring one row
iterator each; the model is over lists rather than over the GraphBLAS iterators
themselves, so what is proved is that *this merge algorithm* is correct, not that
GraphBLAS's iterators are ascending — that remains an assumption, and it is the
same one the Rust makes.
-/
import Tensor.Iter

namespace FalkorDB
namespace Tensor

/-! ## Lexicographic order on positions -/

/-- `(src, dst)` ascending: the order GraphBLAS row iterators yield. -/
def plt (a b : Pair) : Prop := a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

instance (a b : Pair) : Decidable (plt a b) :=
  decidable_of_iff (a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)) Iff.rfl

theorem plt_irrefl (a : Pair) : ¬ plt a a := by
  simp only [plt, lt_irrefl, false_or, and_false, or_self, not_false_eq_true]

theorem plt_trans {a b c : Pair} (h1 : plt a b) (h2 : plt b c) : plt a c := by
  rcases h1 with h1 | ⟨h1e, h1l⟩ <;> rcases h2 with h2 | ⟨h2e, h2l⟩
  · exact Or.inl (lt_trans h1 h2)
  · exact Or.inl (h2e ▸ h1)
  · exact Or.inl (h1e ▸ h2)
  · exact Or.inr ⟨h1e.trans h2e, lt_trans h1l h2l⟩

theorem plt_total {a b : Pair} (h : a ≠ b) : plt a b ∨ plt b a := by
  rcases lt_trichotomy a.1 b.1 with h1 | h1 | h1
  · exact Or.inl (Or.inl h1)
  · rcases lt_trichotomy a.2 b.2 with h2 | h2 | h2
    · exact Or.inl (Or.inr ⟨h1, h2⟩)
    · exact absurd (Prod.ext h1 h2) h
    · exact Or.inr (Or.inr ⟨h1.symm, h2⟩)
  · exact Or.inr (Or.inl h1)

theorem plt_asymm {a b : Pair} (h : plt a b) : ¬ plt b a := fun hc =>
  plt_irrefl a (plt_trans h hc)

theorem plt_ne {a b : Pair} (h : plt a b) : a ≠ b := by
  rintro rfl; exact plt_irrefl a h

/-- Below-then-not-below is strictly below: what lets a flushed prefix sit ahead
of everything the rest of the merge emits. -/
theorem plt_of_lt_of_not_lt {a b c : Pair} (h1 : plt a b) (h2 : ¬ plt c b) : plt a c := by
  by_cases hcb : c = b
  · exact hcb ▸ h1
  · rcases plt_total hcb with h | h
    · exact absurd h h2
    · exact plt_trans h1 h

/-- Strictly ascending by position. -/
def SortedBy (l : List (Pair × Nat)) : Prop := l.Pairwise (fun x y => plt x.1 y.1)

def SortedP (l : List Pair) : Prop := l.Pairwise plt

theorem head_lt_tail {m : Pair × Nat} {ms : List (Pair × Nat)} (h : SortedBy (m :: ms))
    {x : Pair × Nat} (hx : x ∈ ms) : plt m.1 x.1 :=
  (List.pairwise_cons.mp h).1 x hx

theorem head_lt_tailP {d : Pair} {ds : List Pair} (h : SortedP (d :: ds))
    {e : Pair} (he : e ∈ ds) : plt d e :=
  (List.pairwise_cons.mp h).1 e he

/-! ## Flushing the pending cursor

`takeLt b ps` splits the pending list at the first position not below `b`. -/

def takeLt (b : Pair) : List (Pair × Nat) → List (Pair × Nat) × List (Pair × Nat)
  | [] => ([], [])
  | pp :: ps => if plt pp.1 b then (pp :: (takeLt b ps).1, (takeLt b ps).2) else ([], pp :: ps)

@[simp] theorem takeLt_nil (b : Pair) : takeLt b [] = ([], []) := rfl

theorem takeLt_append (b : Pair) (ps : List (Pair × Nat)) :
    (takeLt b ps).1 ++ (takeLt b ps).2 = ps := by
  induction ps with
  | nil => rfl
  | cons pp ps ih =>
    by_cases h : plt pp.1 b
    · simp only [takeLt, if_pos h, List.cons_append, ih]
    · simp [takeLt, if_neg h]

theorem mem_takeLt (b : Pair) (ps : List (Pair × Nat)) (x : Pair × Nat) :
    x ∈ ps ↔ x ∈ (takeLt b ps).1 ∨ x ∈ (takeLt b ps).2 := by
  rw [← List.mem_append, takeLt_append]

theorem takeLt_fst_lt (b : Pair) (ps : List (Pair × Nat)) :
    ∀ x ∈ (takeLt b ps).1, plt x.1 b := by
  induction ps with
  | nil => simp
  | cons pp ps ih =>
    by_cases h : plt pp.1 b
    · simp only [takeLt, if_pos h]
      intro x hx
      rcases List.mem_cons.mp hx with rfl | hx'
      · exact h
      · exact ih x hx'
    · simp [takeLt, if_neg h]

theorem takeLt_snd_not_lt {b : Pair} {ps : List (Pair × Nat)} (hps : SortedBy ps) :
    ∀ x ∈ (takeLt b ps).2, ¬ plt x.1 b := by
  induction ps with
  | nil => simp
  | cons pp ps ih =>
    by_cases h : plt pp.1 b
    · simp only [takeLt, if_pos h]
      exact ih (List.Pairwise.of_cons hps)
    · simp only [takeLt, if_neg h]
      intro x hx
      rcases List.mem_cons.mp hx with rfl | hx'
      · exact h
      · exact fun hc => h (plt_trans (head_lt_tail hps hx') hc)

theorem sortedBy_takeLt_fst {b : Pair} {ps : List (Pair × Nat)} (hps : SortedBy ps) :
    SortedBy (takeLt b ps).1 := by
  induction ps with
  | nil => simp [SortedBy]
  | cons pp ps ih =>
    by_cases h : plt pp.1 b
    · simp only [takeLt, if_pos h]
      refine List.pairwise_cons.mpr ⟨?_, ih (List.Pairwise.of_cons hps)⟩
      intro y hy
      exact head_lt_tail hps ((mem_takeLt b ps y).mpr (Or.inl hy))
    · simp [takeLt, if_neg h, SortedBy]

theorem sortedBy_takeLt_snd {b : Pair} {ps : List (Pair × Nat)} (hps : SortedBy ps) :
    SortedBy (takeLt b ps).2 := by
  induction ps with
  | nil => simp [SortedBy]
  | cons pp ps ih =>
    by_cases h : plt pp.1 b
    · simp only [takeLt, if_pos h]; exact ih (List.Pairwise.of_cons hps)
    · simpa [takeLt, if_neg h] using hps

/-! ## The merge

Structural on the base list: flush every pending entry below the base position,
then decide about the base position itself. The two interesting branches are the
tie (the shadow rule) and the mask. -/

mutual

/-- The merge. Structural on the base list: flush every pending entry below the
base position, then hand the rest to [`mergeTail`]. -/
def merge3 (ms : List (Pair × Nat)) (ds : List Pair) (ps : List (Pair × Nat)) :
    List (Pair × Nat) :=
  match ms with
  | [] => ps
  | m :: ms' => (takeLt m.1 ps).1 ++ mergeTail m ms' ds (takeLt m.1 ps).2
termination_by 2 * ms.length + 1

/-- The decision at the base position, with the pending cursor already flushed
past everything below it. Splitting this out keeps `merge3`'s own equation
unconditional, which is what makes it rewritable. -/
def mergeTail (m : Pair × Nat) (ms' : List (Pair × Nat)) (ds : List Pair)
    (rest : List (Pair × Nat)) : List (Pair × Nat) :=
  match rest with
  | pp :: rest' =>
      if pp.1 = m.1 then
        -- the shadow rule: `dp` wins and suppresses the `m` entry
        pp :: merge3 ms' ds rest'
      else
        match ds with
        | d :: ds' =>
            if d = m.1 then merge3 ms' ds' (pp :: rest')
            else m :: merge3 ms' ds (pp :: rest')
        | [] => m :: merge3 ms' [] (pp :: rest')
  | [] =>
      match ds with
      | d :: ds' => if d = m.1 then merge3 ms' ds' [] else m :: merge3 ms' ds []
      | [] => m :: merge3 ms' [] []
termination_by 2 * ms'.length + 2

end

/-! ## Provenance

Everything the merge emits came from one of its inputs. That one fact gives both
bounds the ordering proof needs, so neither is a separate induction. -/

theorem mergeTail_mem_src (m : Pair × Nat) (ms' : List (Pair × Nat)) (ds : List Pair)
    (rest : List (Pair × Nat))
    (ih : ∀ (ds : List Pair) (ps : List (Pair × Nat)), ∀ y ∈ merge3 ms' ds ps,
      y ∈ ms' ∨ y ∈ ps) :
    ∀ y ∈ mergeTail m ms' ds rest, y ∈ m :: ms' ∨ y ∈ rest := by
  intro y hy
  rcases rest with _ | ⟨pp, rest'⟩
  · rw [mergeTail.eq_def] at hy
    simp only at hy
    rcases ds with _ | ⟨d, ds'⟩
    · rcases List.mem_cons.mp hy with rfl | hy'
      · exact Or.inl (by simp)
      · rcases ih [] [] y hy' with h | h
        · exact Or.inl (by simp [h])
        · exact absurd h (by simp)
    · by_cases hd : d = m.1
      · simp only [if_pos hd] at hy
        rcases ih ds' [] y hy with h | h
        · exact Or.inl (by simp [h])
        · exact absurd h (by simp)
      · simp only [if_neg hd] at hy
        rcases List.mem_cons.mp hy with rfl | hy'
        · exact Or.inl (by simp)
        · rcases ih (d :: ds') [] y hy' with h | h
          · exact Or.inl (by simp [h])
          · exact absurd h (by simp)
  · rw [mergeTail.eq_def] at hy
    simp only at hy
    by_cases hq : pp.1 = m.1
    · simp only [if_pos hq] at hy
      rcases List.mem_cons.mp hy with rfl | hy'
      · exact Or.inr (by simp)
      · rcases ih ds rest' y hy' with h | h
        · exact Or.inl (by simp [h])
        · exact Or.inr (by simp [h])
    · simp only [if_neg hq] at hy
      rcases ds with _ | ⟨d, ds'⟩
      · rcases List.mem_cons.mp hy with rfl | hy'
        · exact Or.inl (by simp)
        · rcases ih [] (pp :: rest') y hy' with h | h
          · exact Or.inl (by simp [h])
          · exact Or.inr h
      · by_cases hd : d = m.1
        · simp only [if_pos hd] at hy
          rcases ih ds' (pp :: rest') y hy with h | h
          · exact Or.inl (by simp [h])
          · exact Or.inr h
        · simp only [if_neg hd] at hy
          rcases List.mem_cons.mp hy with rfl | hy'
          · exact Or.inl (by simp)
          · rcases ih (d :: ds') (pp :: rest') y hy' with h | h
            · exact Or.inl (by simp [h])
            · exact Or.inr h

theorem merge3_mem_src : ∀ (ms : List (Pair × Nat)) (ds : List Pair) (ps : List (Pair × Nat)),
    ∀ y ∈ merge3 ms ds ps, y ∈ ms ∨ y ∈ ps := by
  intro ms
  induction ms with
  | nil => intro ds ps y hy; rw [merge3] at hy; exact Or.inr hy
  | cons m ms' ih =>
    intro ds ps y hy
    rw [merge3] at hy
    rcases List.mem_append.mp hy with h | h
    · exact Or.inr ((mem_takeLt m.1 ps y).mpr (Or.inl h))
    · rcases mergeTail_mem_src m ms' ds _ ih y h with h' | h'
      · exact Or.inl h'
      · exact Or.inr ((mem_takeLt m.1 ps y).mpr (Or.inr h'))

theorem merge3_gt (ms : List (Pair × Nat)) (ds : List Pair) (ps : List (Pair × Nat))
    (b : Pair) (hm : ∀ y ∈ ms, plt b y.1) (hp : ∀ y ∈ ps, plt b y.1) :
    ∀ y ∈ merge3 ms ds ps, plt b y.1 := by
  intro y hy
  rcases merge3_mem_src ms ds ps y hy with h | h
  · exact hm y h
  · exact hp y h

theorem merge3_ge (ms : List (Pair × Nat)) (ds : List Pair) (ps : List (Pair × Nat))
    (b : Pair) (hm : ∀ y ∈ ms, ¬ plt y.1 b) (hp : ∀ y ∈ ps, ¬ plt y.1 b) :
    ∀ y ∈ merge3 ms ds ps, ¬ plt y.1 b := by
  intro y hy
  rcases merge3_mem_src ms ds ps y hy with h | h
  · exact hm y h
  · exact hp y h

/-- If `dm`'s head is not the base's head, nothing masks the base's head: `dm` is
a subset of the base's positions and both ascend, so its head is the smallest
mask there is. -/
theorem head_not_mem_ds {m : Pair × Nat} {ms' : List (Pair × Nat)} {d : Pair}
    {ds' : List Pair} (hms : SortedBy (m :: ms')) (hds : SortedP (d :: ds'))
    (hsub : ∀ e ∈ d :: ds', ∃ y ∈ m :: ms', y.1 = e) (hd : d ≠ m.1) :
    m.1 ∉ d :: ds' := by
  have hdm : plt m.1 d := by
    obtain ⟨y, hy, hy2⟩ := hsub d (by simp)
    rcases List.mem_cons.mp hy with rfl | hy'
    · exact absurd hy2.symm hd
    · have := head_lt_tail hms hy'; rwa [hy2] at this
  intro hc
  rcases List.mem_cons.mp hc with hc' | hc'
  · rw [hc'] at hdm; exact plt_irrefl d hdm
  · exact plt_asymm hdm (head_lt_tailP hds hc')

/-! ## What the merge emits

The effective set, spelled out: a `dp` entry always survives, and an `m` entry
survives exactly when `dm` does not mask it and `dp` does not shadow it.

`hpure` is Invariant purity, and the tie branch is where it is used — see the
header for why the algorithm is wrong without it. -/

/-- The statement of [`mem_merge3`], as a predicate, so the induction hypothesis
can be passed to [`mem_mergeTail`] without restating it. -/
def MemSpec (ms : List (Pair × Nat)) : Prop :=
  ∀ (ds : List Pair) (ps : List (Pair × Nat)), SortedBy ms → SortedP ds → SortedBy ps →
    (∀ d ∈ ds, ∃ y ∈ ms, y.1 = d) → (∀ d ∈ ds, d ∉ ps.map Prod.fst) → ∀ x,
    (x ∈ merge3 ms ds ps ↔ x ∈ ps ∨ (x ∈ ms ∧ x.1 ∉ ds ∧ x.1 ∉ ps.map Prod.fst))

theorem mem_mergeTail {m : Pair × Nat} {ms' : List (Pair × Nat)} {ds : List Pair}
    {rest : List (Pair × Nat)} (ih : MemSpec ms')
    (hms : SortedBy (m :: ms')) (hds : SortedP ds) (hrest : SortedBy rest)
    (hsub : ∀ d ∈ ds, ∃ y ∈ m :: ms', y.1 = d)
    (hpure : ∀ d ∈ ds, d ∉ rest.map Prod.fst)
    (hge : ∀ y ∈ rest, ¬ plt y.1 m.1) (x : Pair × Nat) :
    x ∈ mergeTail m ms' ds rest ↔
      x ∈ rest ∨ (x ∈ m :: ms' ∧ x.1 ∉ ds ∧ x.1 ∉ rest.map Prod.fst) := by
  have hms' : SortedBy ms' := List.Pairwise.of_cons hms
  have hmne : ∀ y ∈ ms', y.1 ≠ m.1 := fun y hy => (plt_ne (head_lt_tail hms hy)).symm
  rcases rest with _ | ⟨pp, rest'⟩
  · rcases ds with _ | ⟨d, ds'⟩
    · rw [mergeTail.eq_def]; simp only
      rw [List.mem_cons, ih [] [] hms' (by simp [SortedP]) (by simp [SortedBy])
        (by simp) (by simp) x]
      simp only [List.not_mem_nil, or_false, List.mem_cons, List.map_nil, and_true,
        false_or, List.mem_map]
      constructor
      · rintro (rfl | h) <;> simp_all
      · rintro (rfl | h) <;> simp_all
    · by_cases hd : d = m.1
      · rw [mergeTail.eq_def]; simp only [if_pos hd]
        have hsub' : ∀ e ∈ ds', ∃ y ∈ ms', y.1 = e := by
          intro e he
          obtain ⟨y, hy, hy2⟩ := hsub e (by simp [he])
          rcases List.mem_cons.mp hy with rfl | hy'
          · exact absurd (hy2 ▸ head_lt_tailP hds he) (by rw [hd] at *; exact plt_irrefl _)
          · exact ⟨y, hy', hy2⟩
        rw [ih ds' [] hms' (List.Pairwise.of_cons hds) (by simp [SortedBy]) hsub' (by simp) x]
        simp only [List.not_mem_nil, or_false, List.map_nil, and_true, false_or,
          not_false_eq_true]
        constructor
        · rintro ⟨hx, hxd⟩
          exact ⟨by simp [hx], by simp [hxd, hmne x hx, hd]⟩
        · rintro ⟨hx, hxd⟩
          rcases List.mem_cons.mp hx with rfl | hx'
          · exact absurd (by simp [hd]) hxd
          · exact ⟨hx', fun hc => hxd (by simp [hc])⟩
      · rw [mergeTail.eq_def]; simp only [if_neg hd]
        have hnm := head_not_mem_ds hms hds hsub hd
        have hsub' : ∀ e ∈ d :: ds', ∃ y ∈ ms', y.1 = e := by
          intro e he
          obtain ⟨y, hy, hy2⟩ := hsub e he
          rcases List.mem_cons.mp hy with rfl | hy'
          · exact absurd (hy2 ▸ he) hnm
          · exact ⟨y, hy', hy2⟩
        rw [List.mem_cons, ih (d :: ds') [] hms' hds (by simp [SortedBy]) hsub' (by simp) x]
        simp only [List.not_mem_nil, or_false, List.map_nil, and_true, false_or,
          not_false_eq_true, List.mem_cons]
        constructor
        · rintro (rfl | ⟨hx, hxd⟩)
          · exact ⟨Or.inl rfl, by simpa using hnm⟩
          · exact ⟨Or.inr hx, hxd⟩
        · rintro ⟨hx | hx, hxd⟩
          · exact Or.inl hx
          · exact Or.inr ⟨hx, hxd⟩
  · have hppm : ¬ plt pp.1 m.1 := hge pp (by simp)
    by_cases hq : pp.1 = m.1
    · -- the tie: `dp` wins. Purity is what makes advancing `m` without `dm` sound.
      rw [mergeTail.eq_def]; simp only [if_pos hq]
      have hmnd : m.1 ∉ ds := fun hc => hpure m.1 hc (by simp [hq])
      have hsub' : ∀ e ∈ ds, ∃ y ∈ ms', y.1 = e := by
        intro e he
        obtain ⟨y, hy, hy2⟩ := hsub e he
        rcases List.mem_cons.mp hy with rfl | hy'
        · exact absurd (hy2 ▸ he) hmnd
        · exact ⟨y, hy', hy2⟩
      rw [List.mem_cons, ih ds rest' hms' hds (List.Pairwise.of_cons hrest) hsub'
        (fun d hd hc => hpure d hd (by simp [hc])) x]
      constructor
      · rintro (rfl | h | ⟨hx, hxd, hxp⟩)
        · exact Or.inl (by simp)
        · exact Or.inl (by simp [h])
        · refine Or.inr ⟨by simp [hx], hxd, ?_⟩
          simp only [List.map_cons, List.mem_cons, not_or]
          exact ⟨by rw [hq]; exact hmne x hx, hxp⟩
      · rintro (h | ⟨hx, hxd, hxp⟩)
        · rcases List.mem_cons.mp h with rfl | h'
          · exact Or.inl rfl
          · exact Or.inr (Or.inl h')
        · rcases List.mem_cons.mp hx with rfl | hx'
          · exact absurd (by simp [hq]) hxp
          · exact Or.inr (Or.inr ⟨hx', hxd, fun hc => hxp (by simp [hc])⟩)
    · have hmlt : plt m.1 pp.1 := by
        rcases plt_total (a := m.1) (b := pp.1) (fun hc => hq hc.symm) with h | h
        · exact h
        · exact absurd h hppm
      have hmp : m.1 ∉ (pp :: rest').map Prod.fst := by
        simp only [List.map_cons, List.mem_cons, not_or]
        refine ⟨fun hc => plt_irrefl m.1 (hc ▸ hmlt), ?_⟩
        intro hc
        obtain ⟨y, hy, hy2⟩ := List.mem_map.mp hc
        have := head_lt_tail hrest hy
        rw [hy2] at this
        exact plt_asymm hmlt this
      rcases ds with _ | ⟨d, ds'⟩
      · rw [mergeTail.eq_def]; simp only [if_neg hq]
        rw [List.mem_cons, ih [] (pp :: rest') hms' (by simp [SortedP]) hrest
          (by simp) (by simp) x]
        constructor
        · rintro (rfl | h | ⟨hx, hxd, hxp⟩)
          · exact Or.inr ⟨by simp, by simp, hmp⟩
          · exact Or.inl h
          · exact Or.inr ⟨by simp [hx], by simp, hxp⟩
        · rintro (h | ⟨hx, hxd, hxp⟩)
          · exact Or.inr (Or.inl h)
          · rcases List.mem_cons.mp hx with rfl | hx'
            · exact Or.inl rfl
            · exact Or.inr (Or.inr ⟨hx', by simp, hxp⟩)
      · by_cases hd : d = m.1
        · rw [mergeTail.eq_def]; simp only [if_neg hq, if_pos hd]
          have hsub' : ∀ e ∈ ds', ∃ y ∈ ms', y.1 = e := by
            intro e he
            obtain ⟨y, hy, hy2⟩ := hsub e (by simp [he])
            rcases List.mem_cons.mp hy with rfl | hy'
            · exact absurd (hy2 ▸ head_lt_tailP hds he) (by rw [hd] at *; exact plt_irrefl _)
            · exact ⟨y, hy', hy2⟩
          rw [ih ds' (pp :: rest') hms' (List.Pairwise.of_cons hds) hrest hsub'
            (fun e he => hpure e (by simp [he])) x]
          constructor
          · rintro (h | ⟨hx, hxd, hxp⟩)
            · exact Or.inl h
            · exact Or.inr ⟨by simp [hx], by simp [hxd, hmne x hx, hd], hxp⟩
          · rintro (h | ⟨hx, hxd, hxp⟩)
            · exact Or.inl h
            · rcases List.mem_cons.mp hx with rfl | hx'
              · exact absurd (by simp [hd]) hxd
              · exact Or.inr ⟨hx', fun hc => hxd (by simp [hc]), hxp⟩
        · rw [mergeTail.eq_def]; simp only [if_neg hq, if_neg hd]
          have hnm := head_not_mem_ds hms hds hsub hd
          have hsub' : ∀ e ∈ d :: ds', ∃ y ∈ ms', y.1 = e := by
            intro e he
            obtain ⟨y, hy, hy2⟩ := hsub e he
            rcases List.mem_cons.mp hy with rfl | hy'
            · exact absurd (hy2 ▸ he) hnm
            · exact ⟨y, hy', hy2⟩
          rw [List.mem_cons, ih (d :: ds') (pp :: rest') hms' hds hrest hsub' hpure x]
          constructor
          · rintro (rfl | h | ⟨hx, hxd, hxp⟩)
            · exact Or.inr ⟨by simp, hnm, hmp⟩
            · exact Or.inl h
            · exact Or.inr ⟨by simp [hx], hxd, hxp⟩
          · rintro (h | ⟨hx, hxd, hxp⟩)
            · exact Or.inr (Or.inl h)
            · rcases List.mem_cons.mp hx with rfl | hx'
              · exact Or.inl rfl
              · exact Or.inr (Or.inr ⟨hx', hxd, hxp⟩)

/-- **The merge emits exactly the effective set**, with `dp` winning at a shared
position. -/
theorem mem_merge3 : ∀ ms : List (Pair × Nat), MemSpec ms := by
  intro ms
  induction ms with
  | nil =>
    intro ds ps _ _ _ _ _ x
    rw [merge3]; simp
  | cons m ms' ih =>
    intro ds ps hms hds hps hsub hpure x
    rw [merge3, List.mem_append,
      mem_mergeTail ih hms hds (sortedBy_takeLt_snd hps) hsub
        (fun d hd hc => hpure d hd (by
          obtain ⟨y, hy, hy2⟩ := List.mem_map.mp hc
          exact List.mem_map.mpr ⟨y, (mem_takeLt m.1 ps y).mpr (Or.inr hy), hy2⟩))
        (takeLt_snd_not_lt hps) x]
    have hsplit := mem_takeLt m.1 ps x
    have hmapsplit : ∀ z : Pair, z ∈ ps.map Prod.fst ↔
        z ∈ (takeLt m.1 ps).1.map Prod.fst ∨ z ∈ (takeLt m.1 ps).2.map Prod.fst := by
      intro z
      constructor
      · intro hz
        obtain ⟨y, hy, hy2⟩ := List.mem_map.mp hz
        rcases (mem_takeLt m.1 ps y).mp hy with h | h
        · exact Or.inl (List.mem_map.mpr ⟨y, h, hy2⟩)
        · exact Or.inr (List.mem_map.mpr ⟨y, h, hy2⟩)
      · rintro (hz | hz)
        · obtain ⟨y, hy, hy2⟩ := List.mem_map.mp hz
          exact List.mem_map.mpr ⟨y, (mem_takeLt m.1 ps y).mpr (Or.inl hy), hy2⟩
        · obtain ⟨y, hy, hy2⟩ := List.mem_map.mp hz
          exact List.mem_map.mpr ⟨y, (mem_takeLt m.1 ps y).mpr (Or.inr hy), hy2⟩
    constructor
    · rintro (h | h | ⟨hx, hxd, hxp⟩)
      · exact Or.inl (hsplit.mpr (Or.inl h))
      · exact Or.inl (hsplit.mpr (Or.inr h))
      · refine Or.inr ⟨hx, hxd, ?_⟩
        rw [hmapsplit]
        refine not_or.mpr ⟨?_, hxp⟩
        intro hc
        obtain ⟨y, hy, hy2⟩ := List.mem_map.mp hc
        have hylt : plt y.1 m.1 := takeLt_fst_lt m.1 ps y hy
        rcases List.mem_cons.mp hx with rfl | hx'
        · rw [hy2] at hylt; exact plt_irrefl _ hylt
        · rw [hy2] at hylt; exact plt_asymm (head_lt_tail hms hx') hylt
    · rintro (h | ⟨hx, hxd, hxp⟩)
      · rcases hsplit.mp h with h' | h'
        · exact Or.inl h'
        · exact Or.inr (Or.inl h')
      · exact Or.inr (Or.inr ⟨hx, hxd, fun hc => hxp (hmapsplit x.1 |>.mpr (Or.inr hc))⟩)

/-! ## The merge emits in order

The ordering half, and the one downstream operators assume. `SortedBy` is
*strict*, so this says the output has no duplicate positions either. -/

/-- The statement of [`sortedBy_merge3`], as a predicate, so the induction
hypothesis can be handed to the tail case. -/
def SortSpec (ms : List (Pair × Nat)) : Prop :=
  ∀ (ds : List Pair) (ps : List (Pair × Nat)), SortedBy ms → SortedBy ps →
    SortedBy (merge3 ms ds ps)

theorem sortedBy_mergeTail {m : Pair × Nat} {ms' : List (Pair × Nat)} {ds : List Pair}
    {rest : List (Pair × Nat)} (ih : SortSpec ms')
    (hms : SortedBy (m :: ms')) (hrest : SortedBy rest)
    (hge : ∀ y ∈ rest, ¬ plt y.1 m.1) :
    SortedBy (mergeTail m ms' ds rest) ∧ ∀ y ∈ mergeTail m ms' ds rest, ¬ plt y.1 m.1 := by
  have hms' : SortedBy ms' := List.Pairwise.of_cons hms
  -- everything the recursive calls can emit is at or beyond `m`
  have hgt : ∀ (ds₀ : List Pair) (ps₀ : List (Pair × Nat)), (∀ y ∈ ps₀, plt m.1 y.1) →
      ∀ y ∈ merge3 ms' ds₀ ps₀, plt m.1 y.1 := by
    intro ds₀ ps₀ hp y hy
    exact merge3_gt ms' ds₀ ps₀ m.1 (fun z hz => head_lt_tail hms hz) hp y hy
  have hnotlt : ∀ {y : Pair × Nat}, plt m.1 y.1 → ¬ plt y.1 m.1 := fun h => plt_asymm h
  rcases rest with _ | ⟨pp, rest'⟩
  · rcases ds with _ | ⟨d, ds'⟩
    · rw [mergeTail.eq_def]; simp only
      refine ⟨List.pairwise_cons.mpr ⟨fun y hy => hgt [] [] (by simp) y hy,
        ih [] [] hms' (by simp [SortedBy])⟩, ?_⟩
      intro y hy
      rcases List.mem_cons.mp hy with rfl | hy'
      · exact plt_irrefl _
      · exact hnotlt (hgt [] [] (by simp) y hy')
    · by_cases hd : d = m.1
      · rw [mergeTail.eq_def]; simp only [if_pos hd]
        refine ⟨ih ds' [] hms' (by simp [SortedBy]), ?_⟩
        intro y hy
        exact hnotlt (hgt ds' [] (by simp) y hy)
      · rw [mergeTail.eq_def]; simp only [if_neg hd]
        refine ⟨List.pairwise_cons.mpr ⟨fun y hy => hgt (d :: ds') [] (by simp) y hy,
          ih (d :: ds') [] hms' (by simp [SortedBy])⟩, ?_⟩
        intro y hy
        rcases List.mem_cons.mp hy with rfl | hy'
        · exact plt_irrefl _
        · exact hnotlt (hgt (d :: ds') [] (by simp) y hy')
  · have hppm : ¬ plt pp.1 m.1 := hge pp (by simp)
    have hrest'gt : ∀ y ∈ rest', plt pp.1 y.1 := fun y hy => head_lt_tail hrest hy
    by_cases hq : pp.1 = m.1
    · rw [mergeTail.eq_def]; simp only [if_pos hq]
      have hgt' : ∀ y ∈ rest', plt m.1 y.1 := by
        intro y hy; rw [← hq]; exact hrest'gt y hy
      refine ⟨List.pairwise_cons.mpr ⟨?_, ih ds rest' hms' (List.Pairwise.of_cons hrest)⟩, ?_⟩
      · intro y hy
        have := hgt ds rest' hgt' y hy
        rwa [hq]
      · intro y hy
        rcases List.mem_cons.mp hy with rfl | hy'
        · rw [hq]; exact plt_irrefl _
        · exact hnotlt (hgt ds rest' hgt' y hy')
    · have hmlt : plt m.1 pp.1 := by
        rcases plt_total (a := m.1) (b := pp.1) (fun hc => hq hc.symm) with h | h
        · exact h
        · exact absurd h hppm
      have hgt' : ∀ y ∈ pp :: rest', plt m.1 y.1 := by
        intro y hy
        rcases List.mem_cons.mp hy with rfl | hy'
        · exact hmlt
        · exact plt_trans hmlt (hrest'gt y hy')
      rcases ds with _ | ⟨d, ds'⟩
      · rw [mergeTail.eq_def]; simp only [if_neg hq]
        refine ⟨List.pairwise_cons.mpr ⟨fun y hy => hgt [] (pp :: rest') hgt' y hy,
          ih [] (pp :: rest') hms' hrest⟩, ?_⟩
        intro y hy
        rcases List.mem_cons.mp hy with rfl | hy'
        · exact plt_irrefl _
        · exact hnotlt (hgt [] (pp :: rest') hgt' y hy')
      · by_cases hd : d = m.1
        · rw [mergeTail.eq_def]; simp only [if_neg hq, if_pos hd]
          refine ⟨ih ds' (pp :: rest') hms' hrest, ?_⟩
          intro y hy
          exact hnotlt (hgt ds' (pp :: rest') hgt' y hy)
        · rw [mergeTail.eq_def]; simp only [if_neg hq, if_neg hd]
          refine ⟨List.pairwise_cons.mpr ⟨fun y hy => hgt (d :: ds') (pp :: rest') hgt' y hy,
            ih (d :: ds') (pp :: rest') hms' hrest⟩, ?_⟩
          intro y hy
          rcases List.mem_cons.mp hy with rfl | hy'
          · exact plt_irrefl _
          · exact hnotlt (hgt (d :: ds') (pp :: rest') hgt' y hy')

/-- **The merge emits in strictly ascending position order.** -/
theorem sortedBy_merge3 : ∀ ms : List (Pair × Nat), SortSpec ms := by
  intro ms
  induction ms with
  | nil => intro ds ps _ hps; rw [merge3]; exact hps
  | cons m ms' ih =>
    intro ds ps hms hps
    rw [merge3]
    obtain ⟨htail, hge⟩ := sortedBy_mergeTail ih hms (sortedBy_takeLt_snd hps)
      (takeLt_snd_not_lt hps)
    refine List.pairwise_append.mpr ⟨sortedBy_takeLt_fst hps, htail, ?_⟩
    intro a ha b hb
    exact plt_of_lt_of_not_lt (takeLt_fst_lt m.1 ps a ha) (hge b hb)

/-! ## The merge computes `effGet`

The bridge back to the tensor, and the point of the file: instantiate the three
cursors with the three forward layers as GraphBLAS presents them — ascending, one
entry per stored position — and the merge's output is exactly the effective view.

`hsub` is Invariant `dm ⊆ dom m` and `hpure` is Invariant purity; both are
hypotheses here because both are hypotheses of the algorithm. -/

theorem merge3_effGet {t : Tensor} {ms : List (Pair × Nat)} {ds : List Pair}
    {ps : List (Pair × Nat)}
    (hms : SortedBy ms) (hds : SortedP ds) (hps : SortedBy ps)
    (hm : ∀ x : Pair × Nat, x ∈ ms ↔ t.m.get x.1 = some x.2)
    (hd : ∀ q, q ∈ ds ↔ q ∈ t.dm)
    (hp : ∀ x : Pair × Nat, x ∈ ps ↔ t.dp.get x.1 = some x.2)
    (hsub : ∀ d ∈ ds, ∃ y ∈ ms, y.1 = d) (hpure : ∀ d ∈ ds, d ∉ ps.map Prod.fst)
    (x : Pair × Nat) :
    x ∈ merge3 ms ds ps ↔ t.effGet x.1 = some x.2 := by
  rw [mem_merge3 ms ds ps hms hds hps hsub hpure x]
  constructor
  · rintro (h | ⟨hx, hxd, hxp⟩)
    · exact effGet_of_dp ((hp x).mp h)
    · have hdpnone : t.dp.get x.1 = none := by
        by_contra hc
        obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.mp hc
        exact hxp (List.mem_map.mpr ⟨(x.1, v), (hp (x.1, v)).mpr hv, rfl⟩)
      have hdmnot : x.1 ∉ t.dm := fun hc => hxd ((hd x.1).mpr hc)
      rw [effGet_of_m hdpnone hdmnot]
      exact (hm x).mp hx
  · intro h
    by_cases hdp : t.dp.get x.1 = some x.2
    · exact Or.inl ((hp x).mpr hdp)
    · have hdpnone : t.dp.get x.1 = none := by
        by_contra hc
        obtain ⟨v, hv⟩ := Option.ne_none_iff_exists'.mp hc
        rw [effGet_of_dp hv] at h
        exact hdp (by rw [hv, Option.some_inj.mp h])
      have hdmnot : x.1 ∉ t.dm := by
        intro hc
        have hnone : t.effGet x.1 = none := by
          simp only [Tensor.effGet, hdpnone, if_pos hc]
        rw [hnone] at h
        exact absurd h (by simp)
      refine Or.inr ⟨(hm x).mpr (by rw [← effGet_of_m hdpnone hdmnot]; exact h), ?_, ?_⟩
      · exact fun hc => hdmnot ((hd x.1).mp hc)
      · intro hc
        obtain ⟨y, hy, hy2⟩ := List.mem_map.mp hc
        have hdpy := (hp y).mp hy
        rw [hy2, hdpnone] at hdpy
        exact absurd hdpy (by simp)

end Tensor
end FalkorDB
