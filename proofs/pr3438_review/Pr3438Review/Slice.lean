import Pr3438Review.NthLive
import PendingCommit.StoreCodec
/-
# `AttributeStore::encode_with_range` on the PR (attribute_store.rs:1425-1474, 2fabe1a8a)

```rust
if count == 0 { return; }
let Some(start) = nth_live_id(deleted, max_id, offset) else { return; };
let mut deleted_iter = deleted.iter();
deleted_iter.advance_to(start);
let mut next_deleted = deleted_iter.next();
let mut encoded = 0u64;
for id in start..=max_id {
    if next_deleted == Some(id) { next_deleted = deleted_iter.next(); continue; }
    w.write_unsigned(id); /* span length, (attr_id, value)… — unchanged from main */
    encoded += 1;
    if encoded >= count { break; }
}
```

| here | there |
| --- | --- |
| `walk ent count ids q enc` | the `for id in start..=max_id` loop (:1452-1473); the cursor state is `q`, the ids the iterator has still to yield *including* `next_deleted = q.head?` (`next()` = advance to `q.tail`) |
| `prEncode` | `encode_with_range` (:1425) |
| `ent id` | the per-entity writes (:1458-1467), the same code as main's: abstract, then instantiated with pending_commit's `(id, dread d id)` |
| `PendingCommit.Content.encode` | **main's** `encode_with_range` (9212add39 :1425; pending_commit `StoreCodec.lean`) |

* `walk_spec` — with the cursor at the deleted ids `≥ id`, the loop writes the
  live ids of the remaining range, first `count - encoded` of them.
* **`prEncode_spec`** — the PR writes `((liveL deleted max_id).drop offset).take count`
  (nothing when `count = 0`), for every deleted set, `offset`, `count`, `max_id`.
* **`prEncode_eq_main`** — for `count > 0` it is *exactly* main's entity sequence
  (same ids, same order, same per-entity writes): `prEncode … = Content.encode …`.
* **`count_zero_differs`** — for `count = 0` main still writes one entity (its
  `break` follows the write: `window` takes `max count 1`), the PR writes none;
  `count_zero_witness` is a concrete store where they differ. No caller passes
  `count = 0` (`Layout.lean`, `encoder_counts_pos`).
-/
namespace Pr3438
open PendingCommit.Content

/-- The PR's loop (:1452-1473). -/
def walk {E : Type} (ent : Nat → E) (count : Nat) : List Nat → List Nat → Nat → List E
  | [], _, _ => []
  | id :: ids, q, enc =>
    if q.head? = some id then walk ent count ids q.tail enc
    else if enc + 1 ≥ count then [ent id]
    else ent id :: walk ent count ids q (enc + 1)

/-- `encode_with_range` on the PR (:1425). -/
def prEncode {E : Type} (ent : Nat → E) (r : RSet) (maxId count offset : Nat) : List E :=
  if count = 0 then []
  else match nthLive (r.rank) maxId offset with
    | none => []
    | some start => walk ent count (List.range' start (maxId + 1 - start)) (r.iterFrom start) 0

theorem walk_spec {E : Type} (ent : Nat → E) (count : Nat) (l : List Nat) (hl : l.Pairwise (· < ·)) :
    ∀ (k a enc : Nat), enc < count →
      walk ent count (List.range' a k) (l.filter (fun d => decide (a ≤ d))) enc =
        (((List.range' a k).filter (fun i => !decide (i ∈ l))).take (count - enc)).map ent := by
  intro k
  induction k with
  | zero => intro a enc _; simp [walk]
  | succ k ih =>
    intro a enc henc
    have hr : List.range' a (k + 1) = a :: List.range' (a + 1) k := rfl
    rw [hr, walk]
    have hstep := filter_ge_step l hl a
    by_cases ha : a ∈ l
    · rw [if_pos ha] at hstep
      rw [hstep]
      simp only [List.head?_cons, List.tail_cons, ite_true]
      rw [ih (a + 1) enc henc]
      simp [ha]
    · rw [if_neg ha] at hstep
      rw [hstep]
      have hne : (l.filter (fun d => decide (a + 1 ≤ d))).head? ≠ some a := by
        intro h; have := head_filter_ge l (a + 1) a h; omega
      rw [if_neg hne]
      have hfc : (a :: List.range' (a + 1) k).filter (fun i => !decide (i ∈ l)) =
          a :: (List.range' (a + 1) k).filter (fun i => !decide (i ∈ l)) := by
        simp [ha]
      rw [hfc]
      by_cases hc : enc + 1 ≥ count
      · rw [if_pos hc]
        have : count - enc = 0 + 1 := by omega
        rw [this]; simp
      · rw [if_neg hc, ← hstep, ← filter_ge_step_neg l hl a ha, ih (a + 1) (enc + 1) (by omega)]
        have : count - enc = (count - (enc + 1)) + 1 := by omega
        rw [this]; simp
where
  filter_ge_step_neg (l : List Nat) (hl : l.Pairwise (· < ·)) (a : Nat) (ha : a ∉ l) :
      l.filter (fun d => decide (a + 1 ≤ d)) = l.filter (fun d => decide (a ≤ d)) := by
    rw [filter_ge_step l hl a, if_neg ha]

/-- **`prEncode_spec`**: the entities the PR writes. -/
theorem prEncode_spec {E : Type} (ent : Nat → E) (r : RSet) (maxId count offset : Nat) :
    prEncode ent r maxId count offset =
      if count = 0 then [] else ((((liveL (r.elems) maxId).drop offset).take count).map ent) := by
  unfold prEncode
  by_cases hc : count = 0
  · simp [hc]
  · rw [if_neg hc, if_neg hc]
    cases hn : nthLive (r.rank) maxId offset with
    | none =>
      have := nthLive_none r maxId offset hn
      simp [List.drop_eq_nil_of_le this]
    | some s =>
      simp only
      rw [RSet.iterFrom_spec, walk_spec ent count _ (r.sorted) _ s 0 (by omega), nthLive_drop r maxId offset s hn]
      simp

/-- main's window is the same live list. -/
theorem window_eq (l : List Nat) (maxId count offset : Nat) :
    window (fun i => decide (i ∈ l)) maxId count offset = (((liveL l maxId).drop offset).take (max count 1)) := rfl

/-- **`prEncode_eq_main`**: for every deleted set, `max_id`, `offset` and
`count > 0`, the PR writes exactly main's entity sequence. -/
theorem prEncode_eq_main (r : RSet) (d : D) (maxId count offset : Nat)
    (hc : 0 < count) :
    prEncode (fun id => (id, dread d id)) r maxId count offset =
      encode d (fun i => decide (i ∈ r.elems)) maxId count offset := by
  rw [prEncode_spec, if_neg (by omega), encode_spec, window_eq, Nat.max_eq_left hc]

/-- **`count = 0`**: the PR writes nothing; main writes the first live entity at
or after `offset`, if there is one. -/
theorem count_zero_differs (r : RSet) (d : D) (maxId offset : Nat) :
    prEncode (fun id => (id, dread d id)) r maxId 0 offset = [] ∧
      encode d (fun i => decide (i ∈ r.elems)) maxId 0 offset =
        (((liveL (r.elems) maxId).drop offset).take 1).map (fun id => (id, dread d id)) := by
  refine ⟨by simp [prEncode], ?_⟩
  rw [encode_spec, window_eq]; rfl

/-- A concrete difference at `count = 0`: no deleted ids, `max_id = 0`, `offset = 0` —
main writes entity 0, the PR nothing. -/
theorem count_zero_witness (d : D) :
    prEncode (fun id => (id, dread d id)) ⟨[], by simp, by simp⟩ 0 0 0 = [] ∧
      encode d (fun i => decide (i ∈ ([] : List Nat))) 0 0 0 = [(0, dread d 0)] := by
  refine ⟨by simp [prEncode], ?_⟩
  rw [encode_spec]; rfl

/-- The round trip pending_commit proves for main's encoder holds verbatim for
the PR's, for every `count > 0`. -/
theorem pr_encode_decode_roundtrip (r : RSet) (d : D) (h : SInv d) (hs : SOK d)
    (limit : Nat) (hlim : ∀ id, ∀ e ∈ dread d id, e.1 < limit ∧ e.1 < 65536)
    (hfit : ∀ id, (dread d id).length ≤ 65535) (maxId count offset : Nat) (hc : 0 < count) :
    ∀ id, dread (decode emptyD limit (prEncode (fun id => (id, dread d id)) r maxId count offset)) id =
      if id ∈ window (fun i => decide (i ∈ r.elems)) maxId count offset then dread d id else [] := by
  rw [prEncode_eq_main r d maxId count offset hc]
  exact encode_decode_roundtrip d h hs limit hlim hfit _ maxId count offset

end Pr3438
