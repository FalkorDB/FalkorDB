import FalkorNativeIndex.Rows

/-!
# `DistinctIter` with its `dedupe` flag (#3738, `numeric.rs`)

Since #3738 `DistinctIter` carries `dedupe`: `DistinctIter::new` sets it (a union across columns,
where a doc may satisfy several members), `DistinctIter::disjoint` clears it (the kinds of one
column, `RangeIndex::lookup`'s `Scalars { numeric, tag }` arm, whose streams share no doc). `next`
now yields a doc when `!self.dedupe || self.seen.insert(doc)`, so with `dedupe` off nothing is
remembered and nothing skipped.

`distinctGoD` is that loop. Proven:
* `distinctGoD_true`: with `dedupe` on it is `distinctGo`, so `distinctGo_spec`,
  `distinctIter_next_spec` and `distinctRem_initial` describe `DistinctIter::new`'s iterator as before;
* `distinctGoD_false_spec`: with `dedupe` off each `next` yields the head of what is left of the
  chained streams (`chainRem`), leaves `seen` untouched, and `None` only when nothing is left;
* `chainRem_disjoint`: when the streams hold no doc twice (disjoint, each distinct) the chain is
  exactly what the deduplicating iterator yields, so skipping the set loses nothing.
-/

namespace NI

/-- `DistinctIter::next`'s loop since #3738 (`numeric.rs`): `!dedupe || seen.insert(doc)`. -/
def distinctGoD (dedupe : Bool) : Nat → List Nat → List (List Nat) → List Nat →
    Option (Nat × List (List Nat) × List Nat × List Nat)
  | 0, _, _, _ => none
  | n + 1, seen, streams, d :: ds =>
    if !dedupe then some (d, streams, ds, seen)
    else if d ∈ seen then distinctGoD dedupe n seen streams ds else some (d, streams, ds, d :: seen)
  | _ + 1, _, [], [] => none
  | n + 1, seen, s :: ss, [] => distinctGoD dedupe n seen ss s

/-- With `dedupe` on (`DistinctIter::new`) the loop is the modelled `distinctGo`. -/
theorem distinctGoD_true : ∀ (fuel : Nat) (seen : List Nat) (streams : List (List Nat)) (cur : List Nat),
    distinctGoD true fuel seen streams cur = distinctGo fuel seen streams cur
  | 0, _, _, _ => rfl
  | n + 1, seen, streams, d :: ds => by
    simp only [distinctGoD, distinctGo, Bool.not_true, Bool.false_eq_true, ite_false]
    rw [distinctGoD_true n seen streams ds]
  | _ + 1, _, [], [] => rfl
  | n + 1, seen, s :: ss, [] => by
    simp only [distinctGoD, distinctGo]
    exact distinctGoD_true n seen ss s

/-- What a `disjoint` iterator still has to yield: the rest of the current stream, then the later
    streams, as they are. -/
def chainRem (streams : List (List Nat)) (cur : List Nat) : List Nat := cur ++ streams.flatten

/-- **`DistinctIter::next` with `dedupe` off yields the head of the chained streams and leaves its
    tail**, remembering nothing; `None` only when every stream is exhausted. -/
theorem distinctGoD_false_spec : ∀ (fuel : Nat) (seen : List Nat) (streams : List (List Nat)) (cur : List Nat),
    cur.length + (streams.map (fun s => s.length + 1)).sum < fuel →
    (∀ d st cu se, distinctGoD false fuel seen streams cur = some (d, st, cu, se) →
      se = seen ∧ chainRem streams cur = d :: chainRem st cu) ∧
    (distinctGoD false fuel seen streams cur = none → chainRem streams cur = []) := by
  intro fuel
  induction fuel with
  | zero => intro seen streams cur h; omega
  | succ n ih =>
    intro seen streams cur hf
    cases cur with
    | cons d ds =>
      simp only [distinctGoD, Bool.not_false, ite_true]
      refine ⟨fun d' st cu se h => ?_, fun h => by simp at h⟩
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      exact ⟨rfl, by simp [chainRem]⟩
    | nil =>
      cases streams with
      | nil =>
        simp only [distinctGoD]
        exact ⟨fun d st cu se h => by simp at h, fun _ => by simp [chainRem]⟩
      | cons s ss =>
        have hf' : s.length + (ss.map (fun s => s.length + 1)).sum < n := by
          simp only [List.map_cons, List.sum_cons, List.length_nil] at hf; omega
        obtain ⟨h1, h2⟩ := ih seen ss s hf'
        have hrem : chainRem (s :: ss) [] = chainRem ss s := by simp [chainRem]
        simp only [distinctGoD]
        refine ⟨fun d' st cu se h => ?_, fun h => ?_⟩
        · obtain ⟨a, b⟩ := h1 d' st cu se h
          exact ⟨a, by rw [hrem]; exact b⟩
        · rw [hrem]; exact h2 h

theorem eraseDups_of_nodup : ∀ (l : List Nat), l.Nodup → l.eraseDups = l
  | [], _ => rfl
  | a :: l, h => by
    rw [List.eraseDups_cons]
    have ha : a ∉ l := (List.nodup_cons.1 h).1
    have hf : l.filter (fun b => !b == a) = l :=
      List.filter_eq_self.2 (fun x hx => by
        have : x ≠ a := fun e => ha (e ▸ hx)
        simp [this])
    rw [hf, eraseDups_of_nodup l (List.nodup_cons.1 h).2]

/-- **The `disjoint` chain loses nothing**: when no doc occurs twice across the streams (the kinds
    of one column share no doc, and each kind's stream is distinct), the docs it yields are exactly
    those the deduplicating iterator yields from the same state with nothing seen. -/
theorem chainRem_disjoint (streams : List (List Nat)) (cur : List Nat)
    (h : (cur ++ streams.flatten).Nodup) :
    chainRem streams cur = distinctRem [] streams cur := by
  unfold chainRem distinctRem
  rw [List.filter_eq_self.2 (fun x _ => by simp), eraseDups_of_nodup _ h]

/-- Started from nothing, a `disjoint` iterator over streams with no doc in common yields
    `distinctDocs streams`, as `DistinctIter::new` does (`distinctRem_initial`). -/
theorem chainRem_initial (streams : List (List Nat)) (h : streams.flatten.Nodup) :
    chainRem streams [] = distinctDocs streams := by
  rw [chainRem_disjoint streams [] (by simpa using h)]
  exact (distinctRem_initial streams).2

end NI
