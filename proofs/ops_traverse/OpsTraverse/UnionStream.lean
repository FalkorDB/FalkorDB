import OpsTraverse.Basic
import VersionedMatrix.Iter
import VersionedMatrix.Delta
/-
# `UnionIter`: the streamed union of several types' pair matrices (#3492)

Model of `UnionIter` (`cond_traverse.rs:245-278`), the k-way merge that replaced the per-execution
materialised union `build_relationship_matrix_unrestricted`.

| here | there |
| --- | --- |
| `Cur`, `Cur.seek`, `Cur.pop` | one `Iter` of `UnionIter::iters` with its `heads[i]` slot; `it.seek(..)` then `*head = it.next()` |
| `new` | `UnionIter::new` (:245): heads start empty |
| `seek` | `UnionIter::seek` (:250) |
| `next`, `minHead` | `impl Iterator for UnionIter` (:265): the smallest head, then advance every iterator sitting on it |
| `drain` | `Iterator::collect` |

A pair is `(row, col)`, ordered lexicographically as the tuples `Option<(u64, u64)>::min` compares.
`Cur.src` is everything the layer iterator would yield for the whole matrix; a seek rebinds it
to the row range `p` (the `versioned_matrix::Iter::seek` contract, `VMIterSeek.seek_abs`: a seek
is a fresh iterator on the new range).
-/

namespace OpsTraverse.UnionStream

abbrev P := Nat × Nat

def lt (a b : P) : Prop := a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

instance (a b : P) : Decidable (lt a b) := by unfold lt; infer_instance

theorem lt_irrefl (a : P) : ¬ lt a a := by unfold lt; omega
theorem lt_trans {a b c : P} (h1 : lt a b) (h2 : lt b c) : lt a c := by unfold lt at *; omega
theorem lt_total (a b : P) : lt a b ∨ a = b ∨ lt b a := by
  rcases a with ⟨a1, a2⟩; rcases b with ⟨b1, b2⟩
  unfold lt
  simp only [Prod.mk.injEq]
  omega

/-- Strictly ascending: what `versioned_matrix::Iter` yields. -/
def Sorted (l : List P) : Prop := l.Pairwise lt

theorem Sorted.tail {a : P} {l : List P} (h : Sorted (a :: l)) : Sorted l :=
  (List.pairwise_cons.1 h).2

theorem Sorted.filter (p : P → Bool) {l : List P} (h : Sorted l) : Sorted (l.filter p) :=
  List.Pairwise.filter p h

theorem Sorted.nodup {l : List P} (h : Sorted l) : l.Nodup :=
  List.Pairwise.imp (fun {a b} (hl : lt a b) (heq : a = b) => by subst heq; exact lt_irrefl _ hl) h

/-- One iterator of the union with its head slot. -/
structure Cur where
  /-- what its layer iterator would yield over the whole matrix -/
  src : List P
  /-- what it has not yet produced behind `head` -/
  rem : List P
  head : Option P

/-- `it.seek(min_row, max_row); *head = it.next()` (:250-258), `p` = the row-range test. -/
def Cur.seek (p : P → Bool) (c : Cur) : Cur :=
  { c with rem := (c.src.filter p).tail, head := (c.src.filter p).head? }

/-- `*head = it.next()`. -/
def Cur.pop (c : Cur) : Cur := { c with rem := c.rem.tail, head := c.rem.head? }

def Cur.stream (c : Cur) : List P := c.head.toList ++ c.rem

/-- `UnionIter::new` (:245): the heads start empty. -/
def new (srcs : List (List P)) : List Cur := srcs.map fun s => ⟨s, s, none⟩

def seek (p : P → Bool) (st : List Cur) : List Cur := st.map (Cur.seek p)

def minP (a b : P) : P := if lt b a then b else a

/-- `self.heads.iter().flatten().min().copied()` -/
def minHead : List (Option P) → Option P
  | [] => none
  | none :: t => minHead t
  | some a :: t => some (match minHead t with | none => a | some b => minP a b)

/-- `UnionIter::next` (:265-272). -/
def next (st : List Cur) : Option P × List Cur :=
  match minHead (st.map (·.head)) with
  | none => (none, st)
  | some m => (some m, st.map fun c => if c.head = some m then c.pop else c)

def meas (st : List Cur) : Nat := (st.map fun c => c.stream.length).sum

def drain : Nat → List Cur → List P
  | 0, _ => []
  | n + 1, st =>
    match next st with
    | (none, _) => []
    | (some m, st') => m :: drain n st'

/-! ## The merge -/

theorem minHead_none {hs : List (Option P)} (h : minHead hs = none) : ∀ x ∈ hs, x = none := by
  induction hs with
  | nil => simp
  | cons a t ih =>
    cases a with
    | none => intro x hx; rcases List.mem_cons.1 hx with rfl | hx; rfl; exact ih (by simpa [minHead] using h) x hx
    | some a => simp [minHead] at h

theorem minHead_some : ∀ {hs : List (Option P)} {m : P}, minHead hs = some m →
    some m ∈ hs ∧ ∀ x, some x ∈ hs → ¬ lt x m
  | [], m, h => by simp [minHead] at h
  | none :: t, m, h => by
    have ih := minHead_some (m := m) (by simpa [minHead] using h)
    refine ⟨List.mem_cons_of_mem _ ih.1, fun x hx => ih.2 x ?_⟩
    rcases List.mem_cons.1 hx with h | h
    · exact absurd h (by simp)
    · exact h
  | some a :: t, m, h => by
    simp only [minHead, Option.some.injEq] at h
    cases hm : minHead t with
    | none =>
      rw [hm] at h; simp only at h; subst h
      refine ⟨List.mem_cons_self, fun x hx => ?_⟩
      rcases List.mem_cons.1 hx with h | h
      · simp only [Option.some.injEq] at h; subst h; exact lt_irrefl _
      · have := minHead_none hm _ h; simp at this
    | some b =>
      rw [hm] at h; simp only at h
      have ih := minHead_some hm
      unfold minP at h
      by_cases hlt : lt b a
      · rw [if_pos hlt] at h; subst h
        refine ⟨List.mem_cons_of_mem _ ih.1, fun x hx => ?_⟩
        rcases List.mem_cons.1 hx with h | h
        · simp only [Option.some.injEq] at h; subst h
          intro hxb; exact lt_irrefl _ (lt_trans hxb hlt)
        · exact ih.2 x h
      · rw [if_neg hlt] at h; subst h
        refine ⟨List.mem_cons_self, fun x hx => ?_⟩
        rcases List.mem_cons.1 hx with h | h
        · simp only [Option.some.injEq] at h; subst h; exact lt_irrefl _
        · intro hxa
          exact ih.2 x h (by
            rcases lt_total b a with h1 | h1 | h1
            · exact absurd h1 hlt
            · subst h1; exact hxa
            · exact lt_trans hxa h1 |> fun h2 => by
                -- x < a and a < b gives x < b
                exact lt_trans hxa h1)

/-- Per-iterator invariant: the stream (head, then the rest) is ascending, and an empty head
means the iterator is exhausted. -/
def Good (c : Cur) : Prop := Sorted c.stream ∧ (c.head = none → c.rem = [])

def AllGood (st : List Cur) : Prop := ∀ c ∈ st, Good c

/-- The union of what the iterators still have to produce. -/
def inU (st : List Cur) (k : P) : Prop := ∃ c ∈ st, k ∈ c.stream

theorem good_seek (p : P → Bool) {c : Cur} (h : Sorted c.src) : Good (c.seek p) := by
  have hs : Sorted (c.src.filter p) := h.filter p
  unfold Good Cur.seek Cur.stream
  cases hf : c.src.filter p with
  | nil => simp [Sorted, List.Pairwise.nil]
  | cons a t => simp only [hf] at hs; exact ⟨hs, by simp⟩

theorem mem_seek_stream (p : P → Bool) (c : Cur) (k : P) :
    k ∈ (c.seek p).stream ↔ k ∈ c.src ∧ p k = true := by
  unfold Cur.seek Cur.stream
  cases hf : c.src.filter p with
  | nil =>
    have : k ∉ c.src.filter p := by rw [hf]; simp
    simp only [List.mem_filter, not_and] at this
    simp only [Option.toList, List.head?_nil, List.nil_append, List.tail_nil, List.not_mem_nil, false_iff, not_and]
    intro h1 h2; exact absurd h2 (this h1) |> fun x => x
  | cons a t =>
    have : k ∈ a :: t ↔ k ∈ c.src ∧ p k = true := by
      rw [← hf, List.mem_filter]
    simpa using this

theorem seek_good (p : P → Bool) {st : List Cur} (h : ∀ c ∈ st, Sorted c.src) : AllGood (seek p st) := by
  intro c hc
  simp only [seek, List.mem_map] at hc
  obtain ⟨c0, hc0, rfl⟩ := hc
  exact good_seek p (h c0 hc0)

/-- **Heads start empty**: before the first `seek`, `next` yields nothing, however full the
matrices are. Every consumer (`expand_union_row`, :1121) seeks before reading. -/
theorem next_new (srcs : List (List P)) : (next (new srcs)).1 = none := by
  unfold next
  have : minHead ((new srcs).map (·.head)) = none := by
    induction srcs with
    | nil => rfl
    | cons s t ih => simpa [new, minHead] using ih
  rw [this]


theorem sum_le_aux (l : List Cur) (f h : Cur → Nat) (hf : ∀ c ∈ l, f c ≤ h c) :
    (l.map f).sum ≤ (l.map h).sum := by
  induction l with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.sum_cons]
    have := hf a (by simp)
    have := ih (fun c hc => hf c (List.mem_cons_of_mem _ hc))
    omega

theorem sum_lt (l : List Cur) (g : Cur → Cur)
    (hle : ∀ c ∈ l, (g c).stream.length ≤ c.stream.length)
    (hlt : ∃ c ∈ l, (g c).stream.length < c.stream.length) :
    (l.map fun c => (g c).stream.length).sum < (l.map fun c => c.stream.length).sum := by
  induction l with
  | nil => simp at hlt
  | cons a t ih =>
    simp only [List.map_cons, List.sum_cons]
    have h1 := hle a (by simp)
    have hle' : ∀ c ∈ t, (g c).stream.length ≤ c.stream.length :=
      fun c hc => hle c (List.mem_cons_of_mem _ hc)
    obtain ⟨c, hc, hlc⟩ := hlt
    rcases List.mem_cons.1 hc with rfl | hc
    · have := sum_le_aux t (fun c => (g c).stream.length) (fun c => c.stream.length) hle'
      omega
    · have := ih hle' ⟨c, hc, hlc⟩
      omega

theorem pop_stream {c : Cur} {m : P} (h : c.head = some m) (hs : Sorted c.stream) :
    c.pop.stream = c.rem ∧ c.stream = m :: c.rem := by
  have e : c.stream = m :: c.rem := by simp [Cur.stream, h]
  refine ⟨?_, e⟩
  cases hr : c.rem with
  | nil => simp [Cur.pop, Cur.stream, hr]
  | cons a t => simp [Cur.pop, Cur.stream, hr]

/-- Heads of a good iterator: the head, if any, is below everything else it will yield. -/
theorem good_head {c : Cur} (h : Good c) {a : P} (ha : c.head = some a) :
    ∀ x ∈ c.rem, lt a x := by
  have e : c.stream = a :: c.rem := by simp [Cur.stream, ha]
  have := h.1
  rw [e] at this
  exact fun x hx => (List.pairwise_cons.1 this).1 x hx

theorem next_some {st st' : List Cur} {m : P} (hg : AllGood st) (h : next st = (some m, st')) :
    inU st m ∧ (∀ k, inU st k → ¬ lt k m) ∧ AllGood st' ∧
      (∀ k, inU st' k ↔ inU st k ∧ k ≠ m) ∧ meas st' < meas st := by
  unfold next at h
  cases hm : minHead (st.map (·.head)) with
  | none => rw [hm] at h; simp at h
  | some m' =>
    rw [hm] at h
    simp only [Prod.mk.injEq, Option.some.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨hmem, hmin⟩ := minHead_some hm
    simp only [List.mem_map] at hmem
    obtain ⟨c0, hc0, hh0⟩ := hmem
    have hmin' : ∀ c ∈ st, ∀ x, c.head = some x → ¬ lt x m' := fun c hc x hx =>
      hmin x (List.mem_map.2 ⟨c, hc, hx⟩)
    -- no iterator that is not on `m'` can still yield it
    have hnot : ∀ c ∈ st, c.head ≠ some m' → m' ∉ c.stream := by
      intro c hc hne hin
      cases hh : c.head with
      | none => have := (hg c hc).2 hh; simp [Cur.stream, hh, this] at hin
      | some a =>
        have hne' : a ≠ m' := fun e => hne (by rw [hh, e])
        have hla := good_head (hg c hc) hh
        have hnl := hmin' c hc a hh
        have hmla : lt m' a := by
          rcases lt_total a m' with h1 | h1 | h1
          · exact absurd h1 hnl
          · exact absurd h1 hne'
          · exact h1
        simp only [Cur.stream, hh, Option.toList, List.singleton_append, List.mem_cons] at hin
        rcases hin with e | hin
        · exact hne' e.symm
        · exact lt_irrefl _ (lt_trans (lt_trans hmla (hla m' hin)) hmla |> fun _ => lt_trans hmla (hla m' hin))
    refine ⟨⟨c0, hc0, by simp [Cur.stream, hh0]⟩, ?_, ?_, ?_, ?_⟩
    · rintro k ⟨c, hc, hk⟩ hkm
      cases hh : c.head with
      | none => have := (hg c hc).2 hh; simp [Cur.stream, hh, this] at hk
      | some a =>
        simp only [Cur.stream, hh, Option.toList, List.singleton_append, List.mem_cons] at hk
        rcases hk with rfl | hk
        · exact hmin' c hc _ hh hkm
        · exact hmin' c hc a hh (lt_trans (good_head (hg c hc) hh k hk) hkm)
    · intro c' hc'
      simp only [List.mem_map] at hc'
      obtain ⟨c, hc, rfl⟩ := hc'
      by_cases hch : c.head = some m'
      · simp only [hch, if_true]
        obtain ⟨h1, h2⟩ := pop_stream hch (hg c hc).1
        refine ⟨by rw [h1]; have := (hg c hc).1; rw [h2] at this; exact this.tail, ?_⟩
        intro hnone
        have : c.rem.head? = none := hnone
        simp only [Cur.pop, this]
        have hr : c.rem = [] := by cases hr : c.rem <;> simp_all
        simp [hr]
      · simp only [hch, if_false]; exact hg c hc
    · intro k
      constructor
      · rintro ⟨c', hc', hk⟩
        simp only [List.mem_map] at hc'
        obtain ⟨c, hc, rfl⟩ := hc'
        by_cases hch : c.head = some m'
        · simp only [hch, if_true] at hk
          obtain ⟨h1, h2⟩ := pop_stream hch (hg c hc).1
          rw [h1] at hk
          have hs := (hg c hc).1; rw [h2] at hs
          refine ⟨⟨c, hc, by rw [h2]; exact List.mem_cons_of_mem _ hk⟩, ?_⟩
          rintro rfl
          exact lt_irrefl _ ((List.pairwise_cons.1 hs).1 _ hk)
        · simp only [hch, if_false] at hk
          exact ⟨⟨c, hc, hk⟩, fun e => hnot c hc hch (e ▸ hk)⟩
      · rintro ⟨⟨c, hc, hk⟩, hne⟩
        by_cases hch : c.head = some m'
        · obtain ⟨h1, h2⟩ := pop_stream hch (hg c hc).1
          rw [h2] at hk
          rcases List.mem_cons.1 hk with e | hk
          · exact absurd e hne
          · exact ⟨_, List.mem_map.2 ⟨c, hc, rfl⟩, by simp only [hch, if_true]; rw [h1]; exact hk⟩
        · exact ⟨_, List.mem_map.2 ⟨c, hc, rfl⟩, by simp only [hch, if_false]; exact hk⟩
    · unfold meas
      rw [List.map_map]
      apply sum_lt st (fun c => if c.head = some m' then c.pop else c)
      · intro c hc
        by_cases hch : c.head = some m'
        · obtain ⟨h1, h2⟩ := pop_stream hch (hg c hc).1
          simp only [hch, if_true, h1, h2]; simp
        · simp [hch]
      · refine ⟨c0, hc0, ?_⟩
        obtain ⟨h1, h2⟩ := pop_stream hh0 (hg c0 hc0).1
        simp only [hh0, if_true, h1, h2]; simp

theorem next_none {st st' : List Cur} (hg : AllGood st) (h : next st = (none, st')) :
    ∀ k, ¬ inU st k := by
  unfold next at h
  cases hm : minHead (st.map (·.head)) with
  | none =>
    intro k ⟨c, hc, hk⟩
    have := minHead_none hm c.head (List.mem_map.2 ⟨c, hc, rfl⟩)
    have h2 := (hg c hc).2 this
    simp [Cur.stream, this, h2] at hk
  | some m => rw [hm] at h; simp at h

/-- **The drain is the sorted union.** From any good state, draining yields a strictly
ascending list (so no pair twice) containing exactly what some iterator still has to yield. -/
theorem drain_spec : ∀ (n : Nat) (st : List Cur), AllGood st → meas st < n →
    Sorted (drain n st) ∧ ∀ k, k ∈ drain n st ↔ inU st k
  | 0, st, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, st, hg, hn => by
    unfold drain
    cases hx : next st with
    | mk r st' =>
      cases r with
      | none =>
        have := next_none hg hx
        exact ⟨List.Pairwise.nil, fun k => ⟨fun h => by simp at h, fun h => absurd h (this k)⟩⟩
      | some m =>
        obtain ⟨ha, hb, hc, hd, he⟩ := next_some hg hx
        obtain ⟨ih1, ih2⟩ := drain_spec n st' hc (by omega)
        refine ⟨List.pairwise_cons.2 ⟨?_, ih1⟩, ?_⟩
        · intro y hy
          have hy' := (hd y).1 ((ih2 y).1 hy)
          rcases lt_total m y with h | h | h
          · exact h
          · exact absurd h.symm hy'.2
          · exact absurd h (hb y hy'.1)
        · intro k
          rw [List.mem_cons, ih2, hd]
          constructor
          · rintro (rfl | h); exact ha; exact h.1
          · intro h
            by_cases e : k = m
            · exact Or.inl e
            · exact Or.inr ⟨h, e⟩


/-! ## Seek, then drain: exactly the union of the per-type streams in the range -/

theorem sorted_ext {a b : List P} (ha : Sorted a) (hb : Sorted b) (h : ∀ k, k ∈ a ↔ k ∈ b) :
    a = b := by
  have hp : a.Perm b := (List.perm_ext_iff_of_nodup ha.nodup hb.nodup).2 h
  exact List.Perm.eq_of_pairwise (le := lt)
    (fun x y _ _ h1 h2 => absurd (lt_trans h1 h2) (lt_irrefl x)) ha hb hp

/-- **`UnionIter` after `seek(p)`** (rows `p`), each iterator ascending: the stream is
strictly ascending (each pair once) and holds exactly the pairs of some type's stream in range. -/
theorem union_stream (srcs : List (List P)) (hs : ∀ s ∈ srcs, Sorted s) (p : P → Bool) :
    let st := seek p (new srcs)
    Sorted (drain (meas st + 1) st) ∧ (drain (meas st + 1) st).Nodup ∧
      ∀ k, k ∈ drain (meas st + 1) st ↔ (∃ s ∈ srcs, k ∈ s) ∧ p k = true := by
  intro st
  have hg : AllGood st := seek_good p (by
    intro c hc
    simp only [new, List.mem_map] at hc
    obtain ⟨s, hsm, rfl⟩ := hc
    exact hs s hsm)
  obtain ⟨h1, h2⟩ := drain_spec (meas st + 1) st hg (Nat.lt_succ_self _)
  refine ⟨h1, h1.nodup, fun k => ?_⟩
  rw [h2]
  constructor
  · rintro ⟨c, hc, hk⟩
    simp only [st, seek, new, List.map_map, List.mem_map] at hc
    obtain ⟨s, hsm, rfl⟩ := hc
    have := (mem_seek_stream p ⟨s, s, none⟩ k).1 hk
    exact ⟨⟨s, hsm, this.1⟩, this.2⟩
  · rintro ⟨⟨s, hsm, hk⟩, hp⟩
    refine ⟨Cur.seek p ⟨s, s, none⟩, ?_, (mem_seek_stream p _ k).2 ⟨hk, hp⟩⟩
    simp only [st, seek, new, List.map_map, List.mem_map]
    exact ⟨s, hsm, rfl⟩

/-- **Same pairs, same order as the merged matrix it replaces**: any ascending list holding
the pairs of the types in range (the merged matrix's own iterator) *is* the union stream. -/
theorem union_stream_eq_merged (srcs : List (List P)) (hs : ∀ s ∈ srcs, Sorted s) (p : P → Bool)
    (merged : List P) (hm : Sorted merged)
    (hmem : ∀ k, k ∈ merged ↔ (∃ s ∈ srcs, k ∈ s) ∧ p k = true) :
    drain (meas (seek p (new srcs)) + 1) (seek p (new srcs)) = merged := by
  obtain ⟨h1, _, h3⟩ := union_stream srcs hs p
  exact sorted_ext h1 hm fun k => (h3 k).trans (hmem k).symm

/-- A single type is the identity: the stream of one iterator in range. -/
theorem union_stream_single (s : List P) (hs : Sorted s) (p : P → Bool) :
    drain (meas (seek p (new [s])) + 1) (seek p (new [s])) = s.filter p := by
  apply union_stream_eq_merged [s] (by simpa using hs) p (s.filter p) (hs.filter p)
  intro k; simp [List.mem_filter]

/-! ## The per-type streams are ascending: discharged from the layer merge

`versioned_matrix::Iter` merges a type's `m`, `dp`, `dm` layers (`VMIter`, keyed by the
row-major position `row * N + col`). Given the C row iterator's order on each layer (the
GraphBLAS contract, an FFI hypothesis), its output is strictly ascending, so the premise of
`union_stream` holds for the real per-type iterators. -/

def enc (N : Nat) (p : P) : Nat := p.1 * N + p.2

theorem enc_lt {N : Nat} {a b : P} (ha : a.2 < N) (hb : b.2 < N) : lt a b ↔ enc N a < enc N b := by
  unfold lt enc
  constructor
  · rintro (h | ⟨h1, h2⟩)
    · have := Nat.mul_le_mul_right N (Nat.succ_le_of_lt h)
      rw [Nat.succ_mul] at this
      omega
    · rw [h1]; omega
  · intro h
    by_cases h1 : a.1 < b.1
    · exact Or.inl h1
    · by_cases h2 : a.1 = b.1
      · right; refine ⟨h2, ?_⟩; rw [h2] at h; omega
      · exfalso
        have h3 : b.1 < a.1 := by omega
        have := Nat.mul_le_mul_right N (Nat.succ_le_of_lt h3)
        rw [Nat.succ_mul] at this
        omega

theorem enc_inj {N : Nat} {a b : P} (ha : a.2 < N) (hb : b.2 < N) (h : enc N a = enc N b) : a = b := by
  rcases lt_total a b with h1 | h1 | h1
  · have := (enc_lt ha hb).1 h1; omega
  · exact h1
  · have := (enc_lt hb ha).1 h1; omega

/-- The merged stream of one type: `VMIter.drain` over the encoded layers, payload = the pair. -/
def layerStream (N : Nat) (M D Pp : List P) : List P :=
  (VMIter.drain (M.length + Pp.length + 1)
    (⟨M.map fun p => (enc N p, p), D.map (enc N), Pp.map fun p => (enc N p, p)⟩ : VMIter.St P)).map (·.2)

/-- Layers in the row-iterator order (`Sorted`), columns below `N`. -/
structure LayersOk (N : Nat) (M D Pp : List P) : Prop where
  sm : Sorted M
  sd : Sorted D
  sp : Sorted Pp
  bm : ∀ p ∈ M, p.2 < N
  bd : ∀ p ∈ D, p.2 < N
  bp : ∀ p ∈ Pp, p.2 < N

theorem layerStream_spec {N : Nat} {M D Pp : List P} (h : LayersOk N M D Pp) :
    Sorted (layerStream N M D Pp) ∧
      ∀ k, k ∈ layerStream N M D Pp ↔ (k ∈ M ∧ k ∉ D) ∨ k ∈ Pp := by
  have hM : VMIter.Sorted (M.map fun p => (enc N p, p)) := by
    unfold VMIter.Sorted
    rw [List.pairwise_map]
    exact List.Pairwise.imp_of_mem (fun {a b} ha hb hab => (enc_lt (h.bm a ha) (h.bm b hb)).1 hab) h.sm
  have hP : VMIter.Sorted (Pp.map fun p => (enc N p, p)) := by
    unfold VMIter.Sorted
    rw [List.pairwise_map]
    exact List.Pairwise.imp_of_mem (fun {a b} ha hb hab => (enc_lt (h.bp a ha) (h.bp b hb)).1 hab) h.sp
  have hD : VMIter.SortedN (D.map (enc N)) := by
    unfold VMIter.SortedN
    rw [List.pairwise_map]
    exact List.Pairwise.imp_of_mem (fun {a b} ha hb hab => (enc_lt (h.bd a ha) (h.bd b hb)).1 hab) h.sd
  have key := VMIter.iter_correct _ _ _ hM hD hP
  simp only [List.length_map] at key
  obtain ⟨k1, k2, k3⟩ := key
  unfold layerStream
  refine ⟨?_, ?_⟩
  · unfold Sorted
    rw [List.pairwise_map]
    exact List.Pairwise.imp_of_mem (fun {a b} ha hb hab => by
      -- both payloads come from a layer entry, whose key is `enc` of the payload
      have ha' := k2 a ha
      have hb' := k2 b hb
      have ea : a.1 = enc N a.2 := by
        rcases ha' with ⟨ha1, _⟩ | ha1 <;> simp only [List.mem_map] at ha1 <;>
          obtain ⟨q, _, rfl⟩ := ha1 <;> rfl
      have eb : b.1 = enc N b.2 := by
        rcases hb' with ⟨hb1, _⟩ | hb1 <;> simp only [List.mem_map] at hb1 <;>
          obtain ⟨q, _, rfl⟩ := hb1 <;> rfl
      have bda : a.2.2 < N := by
        rcases ha' with ⟨ha1, _⟩ | ha1 <;> simp only [List.mem_map] at ha1 <;>
          obtain ⟨q, hq, rfl⟩ := ha1
        · exact h.bm q hq
        · exact h.bp q hq
      have bdb : b.2.2 < N := by
        rcases hb' with ⟨hb1, _⟩ | hb1 <;> simp only [List.mem_map] at hb1 <;>
          obtain ⟨q, hq, rfl⟩ := hb1
        · exact h.bm q hq
        · exact h.bp q hq
      exact (enc_lt bda bdb).2 (by rw [← ea, ← eb]; exact hab)) k1
  · intro k
    simp only [List.mem_map]
    constructor
    · rintro ⟨e, he, rfl⟩
      rcases k2 e he with ⟨h1, h2⟩ | h1
      · simp only [List.mem_map] at h1
        obtain ⟨q, hq, rfl⟩ := h1
        left
        refine ⟨hq, fun hd => h2 ?_⟩
        exact List.mem_map.2 ⟨q, hd, rfl⟩
      · simp only [List.mem_map] at h1
        obtain ⟨q, hq, rfl⟩ := h1
        exact Or.inr hq
    · rintro (⟨hm, hd⟩ | hp)
      · obtain ⟨e, he, hek⟩ := k3 (enc N k) (Or.inl ⟨(enc N k, k), List.mem_map.2 ⟨k, hm, rfl⟩, rfl, by
          intro hmem
          obtain ⟨q, hq, hqe⟩ := List.mem_map.1 hmem
          exact hd (enc_inj (h.bd q hq) (h.bm k hm) hqe ▸ hq)⟩)
        refine ⟨e, he, ?_⟩
        rcases k2 e he with ⟨h1, _⟩ | h1 <;> simp only [List.mem_map] at h1 <;>
          obtain ⟨q, hq, rfl⟩ := h1
        · exact enc_inj (h.bm q hq) (h.bm k hm) hek
        · exact enc_inj (h.bp q hq) (h.bm k hm) hek
      · obtain ⟨e, he, hek⟩ := k3 (enc N k) (Or.inr ⟨(enc N k, k), List.mem_map.2 ⟨k, hp, rfl⟩, rfl⟩)
        refine ⟨e, he, ?_⟩
        rcases k2 e he with ⟨h1, _⟩ | h1 <;> simp only [List.mem_map] at h1 <;>
          obtain ⟨q, hq, rfl⟩ := h1
        · exact enc_inj (h.bm q hq) (h.bp k hp) hek
        · exact enc_inj (h.bp q hq) (h.bp k hp) hek

end OpsTraverse.UnionStream
