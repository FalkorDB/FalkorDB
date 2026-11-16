import OpsTraverse.UnionGlue
/-
# Destination-bound expansion of an untyped / single-type pattern (#3591)

| here | there |
| --- | --- |
| `maxStreamedBwdTypes` | `MAX_STREAMED_BWD_TYPES` (`cond_traverse.rs:339`) |
| `BwdIter`, `BwdSt` | `enum BwdIter { One(Iter), Union(UnionIter) }` (:343-346) |
| `BwdSt.seek` | `BwdIter::seek` (:349-358) |
| `BwdSt.next`, `BwdSt.drain` | `impl Iterator for BwdIter` (:361-370); `drain` = reading it to the end |
| `buildTransposed` | `build_transposed_iter` (:381-403) |

`build_transposed_iter` used to transpose the adjacency matrix for an untyped pattern. It now
merges every relationship type's maintained transpose (`Tensor::matrix_t`) through `UnionIter`
when there are at most `MAX_STREAMED_BWD_TYPES` types, and keeps the adjacency transpose
otherwise. A single type is still its own `matrix_t` (`BwdIter::One`). `expand_row` seeks the
iterator to the bound destination before every read (:1062-1064, :1104-1106), so the empty heads
of `UnionIter::new` are never read.
-/

namespace OpsTraverse.BwdGlue
open OpsTraverse OpsTraverse.UnionStream OpsTraverse.UnionGlue

/-- `MAX_STREAMED_BWD_TYPES` (:339). -/
def maxStreamedBwdTypes : Nat := 16

/-- `BwdIter` over abstract iterators `I`. -/
inductive BwdIter (I : Type) where
  | one (it : I)
  | union (its : List I)

/-- `build_transposed_iter` (:381): `adjT` the adjacency transpose's iterator, `allT` every
type's `matrix_t` iterator (`g.relationship_tensors()`, in type order), `rel` a named type's
`matrix_t` iterator (`None` when the type does not exist). Several types never reach it
(`debug_assert!(types.len() <= 1)`). -/
def buildTransposed {I Ty : Type} (adjT : I) (allT : List I) (rel : Ty → Option I) :
    List Ty → Option (BwdIter I)
  | [] => if allT.length ≤ maxStreamedBwdTypes then some (.union allT) else some (.one adjT)
  | t :: _ => (rel t).map .one

theorem buildTransposed_spec {I Ty : Type} (adjT : I) (allT : List I) (rel : Ty → Option I)
    (t : Ty) :
    (allT.length ≤ maxStreamedBwdTypes → buildTransposed adjT allT rel [] = some (.union allT)) ∧
    (maxStreamedBwdTypes < allT.length → buildTransposed adjT allT rel [] = some (.one adjT)) ∧
    buildTransposed adjT allT rel [t] = (rel t).map .one := by
  refine ⟨fun h => by simp [buildTransposed, h], fun h => ?_, rfl⟩
  have : ¬ allT.length ≤ maxStreamedBwdTypes := Nat.not_le.2 h
  simp [buildTransposed, this]

/-- Outside the untyped few-types case the builder is the old one (`buildUnrestricted`) wrapped
in `BwdIter::One`. -/
theorem buildTransposed_eq_old {I Ty : Type} (adjT : I) (allT : List I) (rel : Ty → Option I)
    (types : List Ty) (h : types ≠ [] ∨ maxStreamedBwdTypes < allT.length) :
    buildTransposed adjT allT rel types = (buildUnrestricted adjT rel types).map .one := by
  cases types with
  | nil =>
    rcases h with h | h
    · exact absurd rfl h
    · have : ¬ allT.length ≤ maxStreamedBwdTypes := Nat.not_le.2 h
      simp [buildTransposed, buildUnrestricted, this]
  | cons t _ => rfl

/-! ## The stream: `BwdIter::seek` / `next` -/

/-- A `BwdIter` over pair streams: one iterator (its `head` the next pair, as `Cur`), or a
`UnionIter`. -/
inductive BwdSt where
  | one (c : Cur)
  | union (st : List Cur)

/-- `BwdIter::new` states: a fresh `Iter` over the whole matrix, or `UnionIter::new`. -/
def BwdSt.ofIter (s : List P) : BwdSt := .one ⟨s, s.tail, s.head?⟩

def BwdSt.ofUnion (srcs : List (List P)) : BwdSt := .union (new srcs)

/-- `BwdIter::seek` (:349): dispatch to `Iter::seek` / `UnionIter::seek`. -/
def BwdSt.seek (p : P → Bool) : BwdSt → BwdSt
  | .one c => .one (c.seek p)
  | .union st => .union (UnionStream.seek p st)

/-- `BwdIter::next` (:364): dispatch to `Iter::next` / `UnionIter::next`. -/
def BwdSt.next : BwdSt → Option P × BwdSt
  | .one c => (c.head, .one c.pop)
  | .union st => match UnionStream.next st with
    | (o, st') => (o, .union st')

def BwdSt.drain : Nat → BwdSt → List P
  | 0, _ => []
  | n + 1, b =>
    match b.next with
    | (none, _) => []
    | (some m, b') => m :: BwdSt.drain n b'

def BwdSt.meas : BwdSt → Nat
  | .one c => c.stream.length
  | .union st => UnionStream.meas st

theorem drain_one (c : Cur) (hc : c.head = none → c.rem = []) :
    ∀ n, c.stream.length ≤ n → BwdSt.drain n (.one c) = c.stream := by
  intro n
  induction n generalizing c with
  | zero =>
    intro h
    have : c.stream = [] := List.eq_nil_of_length_eq_zero (Nat.le_zero.1 h)
    rw [this]; rfl
  | succ n ih =>
    intro h
    cases hh : c.head with
    | none =>
      simp only [BwdSt.drain, BwdSt.next, hh]
      simp [Cur.stream, hh, hc hh]
    | some a =>
      simp only [BwdSt.drain, BwdSt.next, hh]
      have hp : c.pop.head = none → c.pop.rem = [] := by
        intro h0
        simp only [Cur.pop] at h0 ⊢
        cases hr : c.rem with
        | nil => rfl
        | cons x xs => simp [hr] at h0
      have hs : c.pop.stream = c.rem := by
        simp only [Cur.pop, Cur.stream]
        cases c.rem <;> rfl
      have hl : c.pop.stream.length ≤ n := by
        rw [hs]
        simp only [Cur.stream, hh, Option.toList_some, List.length_append, List.length_singleton] at h
        omega
      rw [ih c.pop hp hl, hs]
      simp [Cur.stream, hh]

/-- **`BwdIter::One` after `seek(p)`**: exactly the iterator's pairs in range, in its order
(what the pre-#3591 single `Iter` gave). -/
theorem one_seek_drain (s : List P) (p : P → Bool) :
    let b := (BwdSt.ofIter s).seek p
    BwdSt.drain (b.meas + 1) b = s.filter p := by
  intro b
  have hc : (Cur.seek p ⟨s, s.tail, s.head?⟩).head = none →
      (Cur.seek p ⟨s, s.tail, s.head?⟩).rem = [] := by
    simp only [Cur.seek]
    cases s.filter p <;> simp
  have := drain_one _ hc (b.meas + 1) (by simp [b, BwdSt.meas, BwdSt.seek, BwdSt.ofIter])
  change BwdSt.drain (b.meas + 1) (.one (Cur.seek p ⟨s, s.tail, s.head?⟩)) = _
  rw [this]
  simp only [Cur.seek, Cur.stream]
  cases s.filter p <;> rfl

theorem drain_union (st : List Cur) :
    ∀ n, BwdSt.drain n (.union st) = UnionStream.drain n st := by
  intro n
  induction n generalizing st with
  | zero => rfl
  | succ n ih =>
    simp only [BwdSt.drain, BwdSt.next, UnionStream.drain]
    cases UnionStream.next st with
    | mk o st' => cases o with
      | none => rfl
      | some m => simp only [ih]

/-- **`BwdIter::Union` after `seek(p)`** is the `UnionIter` stream: strictly ascending, each pair
once, exactly the pairs of some type in range (`union_stream`). -/
theorem union_seek_drain (srcs : List (List P)) (hs : ∀ s ∈ srcs, Sorted s) (p : P → Bool) :
    let b := (BwdSt.ofUnion srcs).seek p
    Sorted (BwdSt.drain (b.meas + 1) b) ∧ (BwdSt.drain (b.meas + 1) b).Nodup ∧
      ∀ k, k ∈ BwdSt.drain (b.meas + 1) b ↔ (∃ s ∈ srcs, k ∈ s) ∧ p k = true := by
  intro b
  simp only [b, BwdSt.ofUnion, BwdSt.seek, BwdSt.meas, drain_union]
  exact union_stream srcs hs p

/-- **The untyped few-types stream equals the adjacency transpose's stream it replaces**: if the
adjacency transpose's iterator `adjT` is ascending and holds exactly the pairs some type's
`matrix_t` holds (the adjacency matrix is the union of the tensors' structures, which
`Graph::delete_relationships` keeps by clearing a pair only once no tensor has it), then after the
same seek both arms of `BwdIter` yield the same pairs in the same order. -/
theorem union_eq_adjacency (srcs : List (List P)) (hs : ∀ s ∈ srcs, Sorted s) (adjT : List P)
    (ha : Sorted adjT) (hmem : ∀ k, k ∈ adjT ↔ ∃ s ∈ srcs, k ∈ s) (p : P → Bool) :
    let bu := (BwdSt.ofUnion srcs).seek p
    let bo := (BwdSt.ofIter adjT).seek p
    BwdSt.drain (bu.meas + 1) bu = BwdSt.drain (bo.meas + 1) bo := by
  intro bu bo
  rw [one_seek_drain adjT p]
  simp only [bu, BwdSt.ofUnion, BwdSt.seek, BwdSt.meas, drain_union]
  apply union_stream_eq_merged srcs hs p _ (ha.filter p)
  intro k
  rw [List.mem_filter, hmem]

/-- **Untyped, destination-bound, few types**: the pairs `expand_row` reads after
`seek(d, d)` and swaps back to `(src, dst)` are `transposedPairs` over every type: each pair once,
and exactly the `(s, d)` joined by an edge of some type. -/
theorem untyped_bwd_pairs {S : Graph → List (Nat × Nat)} (hS : StructIterT S) (gs : List Graph)
    (d : Nat) :
    let b := (BwdSt.ofUnion (gs.map S)).seek (fun q => q.1 == d)
    (BwdSt.drain (b.meas + 1) b).map Prod.swap = transposedPairs S gs (fun q => q.1 == d) ∧
      ((BwdSt.drain (b.meas + 1) b).map Prod.swap).Nodup ∧
      ∀ s d', (s, d') ∈ (BwdSt.drain (b.meas + 1) b).map Prod.swap ↔
        (∃ x ∈ gs.flatten, x.src = s ∧ x.dst = d') ∧ d' = d := by
  intro b
  have he : (BwdSt.drain (b.meas + 1) b).map Prod.swap = transposedPairs S gs (fun q => q.1 == d) := by
    simp only [b, BwdSt.ofUnion, BwdSt.seek, BwdSt.meas, drain_union, transposedPairs, unionStream]
  obtain ⟨h1, h2⟩ := transposedPairs_spec hS gs (fun q => q.1 == d)
  refine ⟨he, he ▸ h1, fun s d' => ?_⟩
  rw [he, h2]
  simp

end OpsTraverse.BwdGlue
