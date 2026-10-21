import Pr3438Review.Roaring
/-
# `nth_live_id` (attribute_store.rs:1483-1504 on the PR branch, 2fabe1a8a)

```rust
fn nth_live_id(deleted: &RoaringTreemap, max_id: u64, n: u64) -> Option<u64> {
    let live_through = |x: u64| u128::from(x) + 1 - u128::from(deleted.rank(x));
    let n128 = u128::from(n);
    if live_through(max_id) <= n128 { return None; }
    let (mut lo, mut hi) = (n, max_id);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if live_through(mid) > n128 { hi = mid; } else { lo = mid + 1; }
    }
    Some(lo)
}
```

| here | there |
| --- | --- |
| `cntLive l x` | live ids `< x`: `((0..x).filter(!deleted.contains)).count()` |
| `liveL l maxId` | the ids main's walk visits and does not skip as deleted (`0..=max_id` minus `deleted`) |
| `bsearch` | the `while lo < hi` loop (:1495-1502) |
| `nthLive` | `nth_live_id` (:1483) |

* `liveThrough_eq` — `x + 1 - rank(x)` is the number of live ids in `0..=x`,
  and `rank_le` — `rank(x) ≤ x + 1`, so the `u128` subtraction never underflows
  (and `x + 1 ≤ 2^64` never overflows `u128`).
* `mid_bounds` — `lo ≤ mid < hi`, so `mid + 1 ≤ hi ≤ max_id`: no `u64` overflow.
* **`nthLive_spec`** — `nth_live_id(deleted, max_id, n)` is the `n`-th (0-based)
  id of `0..=max_id` not in `deleted`, or `None` when there are not that many:
  `nthLive … = (liveL …)[n]?`, for every deleted set, `max_id` and `n`
  (empty deleted set, `n` past the live count and `max_id = u64::MAX` included).
* `nthLive_drop` — when it returns `some s`, the live ids from position `n` on
  are exactly the live ids of `s..=max_id`: what the PR's walk then visits.
-/
namespace Pr3438

/-- Live ids below `x`. -/
def cntLive (l : List Nat) (x : Nat) : Nat := ((List.range x).filter (fun i => !decide (i ∈ l))).length

/-- Deleted ids below `x`. -/
def cntDel (l : List Nat) (x : Nat) : Nat := (l.filter (fun d => decide (d < x))).length

/-- Every live id of `0..=max_id`, ascending. -/
def liveL (l : List Nat) (maxId : Nat) : List Nat := (List.range (maxId + 1)).filter (fun i => !decide (i ∈ l))

theorem liveL_length (l : List Nat) (maxId : Nat) : (liveL l maxId).length = cntLive l (maxId + 1) := rfl

theorem cntLive_succ (l : List Nat) (x : Nat) :
    cntLive l (x + 1) = cntLive l x + (if x ∈ l then 0 else 1) := by
  simp only [cntLive, List.range_succ, List.filter_append, List.length_append]
  by_cases h : x ∈ l <;> simp [h]

theorem cntDel_succ : ∀ (l : List Nat), l.Nodup → ∀ x,
    cntDel l (x + 1) = cntDel l x + (if x ∈ l then 1 else 0)
  | [], _, x => by simp [cntDel]
  | a :: t, h, x => by
    have hat : a ∉ t := (List.nodup_cons.1 h).1
    have ih := cntDel_succ t (List.nodup_cons.1 h).2 x
    simp only [cntDel] at ih ⊢
    by_cases hlt : a < x
    · have h1 : a < x + 1 := by omega
      have hne : x ≠ a := by omega
      simp only [List.filter_cons, h1, hlt, decide_true, ite_true, List.length_cons, List.mem_cons, hne,
        false_or]
      rw [ih]; omega
    · by_cases heq : a = x
      · subst heq
        have h1 : a < a + 1 := by omega
        simp only [List.filter_cons, h1, hlt, decide_true, decide_false, ite_true, Bool.false_eq_true,
          ite_false, List.length_cons, List.mem_cons, true_or]
        rw [ih]; simp [hat]
      · have h1 : ¬ a < x + 1 := by omega
        have hne : x ≠ a := by omega
        simp only [List.filter_cons, h1, hlt, decide_false, Bool.false_eq_true, ite_false, List.mem_cons,
          hne, false_or]
        exact ih

/-- Live + deleted below `x` is `x` (deleted ids are distinct). -/
theorem cnt_sum (l : List Nat) (hl : l.Nodup) : ∀ x, cntLive l x + cntDel l x = x
  | 0 => by simp [cntLive, cntDel]
  | x + 1 => by
    rw [cntLive_succ, cntDel_succ l hl, ← Nat.add_assoc]
    have := cnt_sum l hl x
    by_cases h : x ∈ l <;> simp [h] <;> omega

theorem cntLive_le_succ (l : List Nat) (x : Nat) : cntLive l (x + 1) ≤ cntLive l x + 1 := by
  rw [cntLive_succ]; split <;> omega

theorem cntLive_mono (l : List Nat) {x y : Nat} (h : x ≤ y) : cntLive l x ≤ cntLive l y := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => rw [cntLive_succ]; omega

theorem cntLive_le (l : List Nat) (x : Nat) : cntLive l x ≤ x := by
  have := List.length_filter_le (fun i => !decide (i ∈ l)) (List.range x)
  simpa [cntLive] using this

/-- `rank(x)` counts the deleted ids `≤ x`, i.e. `< x + 1`. -/
theorem rank_eq {R : Type} (H : RoaringContract R) (r : R) (x : Nat) :
    H.rank r x = cntDel (H.elems r) (x + 1) := by
  have : (fun d => decide (d ≤ x)) = (fun d => decide (d < x + 1)) := by
    funext d; simp only [decide_eq_decide]; omega
  rw [H.rank_spec, cntDel, this]

theorem elems_nodup {R : Type} (H : RoaringContract R) (r : R) : (H.elems r).Nodup :=
  (H.sorted r).imp (fun h => Nat.ne_of_lt h)

/-- `rank(x) ≤ x + 1`: the `u128` subtraction in `live_through` never underflows. -/
theorem rank_le {R : Type} (H : RoaringContract R) (r : R) (x : Nat) : H.rank r x ≤ x + 1 := by
  rw [rank_eq]; have := cnt_sum (H.elems r) (elems_nodup H r) (x + 1); omega

/-- `live_through(x) = x + 1 - rank(x)` is the number of live ids in `0..=x`. -/
theorem liveThrough_eq {R : Type} (H : RoaringContract R) (r : R) (x : Nat) :
    x + 1 - H.rank r x = cntLive (H.elems r) (x + 1) := by
  rw [rank_eq]; have := cnt_sum (H.elems r) (elems_nodup H r) (x + 1); omega

/-- No `u128` overflow: `x + 1 ≤ 2^64 < 2^128` for every `u64` `x`. -/
theorem liveThrough_no_overflow (x : Nat) (hx : x < 2 ^ 64) : x + 1 < 2 ^ 128 := by
  have : (2 : Nat) ^ 64 < 2 ^ 128 := Nat.pow_lt_pow_right (by decide) (by decide)
  omega

/-! ## The binary search -/

/-- The `while lo < hi` loop (:1495-1502). -/
def bsearch (f : Nat → Nat) (n : Nat) (lo hi : Nat) : Nat :=
  if lo < hi then
    if f (lo + (hi - lo) / 2) > n then bsearch f n lo (lo + (hi - lo) / 2)
    else bsearch f n (lo + (hi - lo) / 2 + 1) hi
  else lo
termination_by hi - lo
decreasing_by all_goals omega

/-- `mid = lo + (hi - lo) / 2` lies in `[lo, hi)`: `mid + 1 ≤ hi`, so neither
`lo + (hi - lo) / 2` nor `mid + 1` overflows `u64` (`hi ≤ max_id < 2^64`). -/
theorem mid_bounds (lo hi : Nat) (h : lo < hi) : lo ≤ lo + (hi - lo) / 2 ∧ lo + (hi - lo) / 2 < hi := by
  omega

theorem bsearch_spec (f : Nat → Nat) (hmono : ∀ x y, x ≤ y → f x ≤ f y) (n : Nat) :
    ∀ lo hi, lo ≤ hi → (∀ x, x < lo → f x ≤ n) → n < f hi →
      lo ≤ bsearch f n lo hi ∧ bsearch f n lo hi ≤ hi ∧ n < f (bsearch f n lo hi) ∧
        ∀ x, x < bsearch f n lo hi → f x ≤ n := by
  intro lo hi
  induction h : hi - lo using Nat.strongRecOn generalizing lo hi with
  | _ d ih =>
    intro hle hlo hhi
    rw [bsearch]
    by_cases hlt : lo < hi
    · rw [if_pos hlt]
      by_cases hf : f (lo + (hi - lo) / 2) > n
      · rw [if_pos hf]
        obtain ⟨a, b, c, e⟩ := ih ((lo + (hi - lo) / 2) - lo) (by omega) lo (lo + (hi - lo) / 2) rfl
          (by omega) hlo hf
        exact ⟨a, by omega, c, e⟩
      · rw [if_neg hf]
        have hlo' : ∀ x, x < lo + (hi - lo) / 2 + 1 → f x ≤ n := by
          intro x hx
          by_cases hxl : x < lo
          · exact hlo x hxl
          · have := hmono x (lo + (hi - lo) / 2) (by omega); omega
        obtain ⟨a, b, c, e⟩ := ih (hi - (lo + (hi - lo) / 2 + 1)) (by omega) _ hi rfl (by omega) hlo' hhi
        exact ⟨by omega, b, c, e⟩
    · rw [if_neg hlt]
      have : lo = hi := by omega
      subst this
      exact ⟨Nat.le_refl _, Nat.le_refl _, hhi, hlo⟩

/-- `nth_live_id(deleted, max_id, n)` (:1483). -/
def nthLive (rank : Nat → Nat) (maxId n : Nat) : Option Nat :=
  if maxId + 1 - rank maxId ≤ n then none
  else some (bsearch (fun x => x + 1 - rank x) n n maxId)

/-- The search result `s`: `n` live ids lie below it, it is live itself, and it is
at most `max_id`. -/
theorem nthLive_some {R : Type} (H : RoaringContract R) (r : R) (maxId n s : Nat)
    (h : nthLive (H.rank r) maxId n = some s) :
    s ≤ maxId ∧ cntLive (H.elems r) s = n ∧ s ∉ H.elems r := by
  unfold nthLive at h
  split at h
  · cases h
  · rename_i hgt
    simp only [Option.some.injEq] at h
    subst h
    have hf : ∀ x, x + 1 - H.rank r x = cntLive (H.elems r) (x + 1) := liveThrough_eq H r
    have hmono : ∀ x y, x ≤ y → (fun x => x + 1 - H.rank r x) x ≤ (fun x => x + 1 - H.rank r x) y := by
      intro x y hxy; simp only [hf]; exact cntLive_mono _ (by omega)
    have hlo : ∀ x, x < n → (fun x => x + 1 - H.rank r x) x ≤ n := by
      intro x hx; simp only [hf]; have := cntLive_le (H.elems r) (x + 1); omega
    have hle : n ≤ maxId := by
      have := cntLive_le (H.elems r) (maxId + 1); rw [hf] at hgt; omega
    obtain ⟨_, hs, hgt', hbelow⟩ := bsearch_spec _ hmono n n maxId hle hlo (by omega)
    generalize bsearch (fun x => x + 1 - H.rank r x) n n maxId = s at hs hgt' hbelow
    simp only [hf] at hgt' hbelow
    have hcs : cntLive (H.elems r) s ≤ n := by
      cases s with
      | zero => simp [cntLive]
      | succ k => exact hbelow k (by omega)
    have hstep := cntLive_succ (H.elems r) s
    refine ⟨hs, ?_, ?_⟩
    · have := cntLive_le_succ (H.elems r) s; omega
    · intro hm; simp [hm] at hstep; omega

theorem nthLive_none {R : Type} (H : RoaringContract R) (r : R) (maxId n : Nat)
    (h : nthLive (H.rank r) maxId n = none) : (liveL (H.elems r) maxId).length ≤ n := by
  unfold nthLive at h
  split at h
  · rename_i hle; rw [liveL_length, ← liveThrough_eq]; exact hle
  · cases h

/-- `0..=max_id` splits at `s`. -/
theorem range_split (s maxId : Nat) (h : s ≤ maxId) :
    List.range (maxId + 1) = List.range s ++ List.range' s (maxId + 1 - s) := by
  rw [List.range_eq_range', List.range_eq_range']
  have := List.range'_append (s := 0) (m := s) (n := maxId + 1 - s) (step := 1)
  simp only [Nat.one_mul, Nat.zero_add] at this
  rw [this]; congr 1; omega

/-- **The seek.** When `nth_live_id` returns `some s`, the live ids from
position `n` on are exactly the live ids of `s..=max_id`. -/
theorem nthLive_drop {R : Type} (H : RoaringContract R) (r : R) (maxId n s : Nat)
    (h : nthLive (H.rank r) maxId n = some s) :
    (liveL (H.elems r) maxId).drop n =
      (List.range' s (maxId + 1 - s)).filter (fun i => !decide (i ∈ H.elems r)) := by
  obtain ⟨hs, hc, _⟩ := nthLive_some H r maxId n s h
  rw [liveL, range_split s maxId hs, List.filter_append]
  have hlen : ((List.range s).filter (fun i => !decide (i ∈ H.elems r))).length = n := hc
  rw [← hlen, List.drop_left]

/-- **`nthLive_spec`**: `nth_live_id` returns the `n`-th live id of `0..=max_id`,
or `None` when there are not that many. -/
theorem nthLive_spec {R : Type} (H : RoaringContract R) (r : R) (maxId n : Nat) :
    nthLive (H.rank r) maxId n = (liveL (H.elems r) maxId)[n]? := by
  cases h : nthLive (H.rank r) maxId n with
  | none =>
    have := nthLive_none H r maxId n h
    rw [List.getElem?_eq_none this]
  | some s =>
    obtain ⟨hs, _, hlive⟩ := nthLive_some H r maxId n s h
    have hd := nthLive_drop H r maxId n s h
    rw [← List.head?_drop, hd]
    have : List.range' s (maxId + 1 - s) = s :: List.range' (s + 1) (maxId - s) := by
      rw [show maxId + 1 - s = (maxId - s) + 1 by omega]; rfl
    rw [this]; simp [hlive]

/-! Boundary cases, by the spec. -/

theorem nthLive_empty_deleted {R : Type} (H : RoaringContract R) (r : R) (he : H.elems r = [])
    (maxId n : Nat) : nthLive (H.rank r) maxId n = if n ≤ maxId then some n else none := by
  rw [nthLive_spec, he]
  have hf : (List.range (maxId + 1)).filter (fun i => !decide (i ∈ ([] : List Nat))) = List.range (maxId + 1) :=
    List.filter_eq_self.2 (by simp)
  rw [liveL, hf]
  by_cases h : n ≤ maxId
  · rw [if_pos h, List.getElem?_range (by omega)]
  · rw [if_neg h, List.getElem?_eq_none (by simp; omega)]

theorem nthLive_past_end {R : Type} (H : RoaringContract R) (r : R) (maxId n : Nat)
    (h : (liveL (H.elems r) maxId).length ≤ n) : nthLive (H.rank r) maxId n = none := by
  rw [nthLive_spec]; exact List.getElem?_eq_none h

end Pr3438
