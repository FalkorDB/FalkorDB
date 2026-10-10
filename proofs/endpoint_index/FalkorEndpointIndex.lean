/-
# The edge-endpoint index answers exactly what the edge set says

A model of `EndpointIndex` (`graph/src/graph/endpoint_index.rs`), the graph-wide
reverse index `edge_id -> (src, dst)` that every materialised relationship reads,
and a machine-checked proof that after **any** sequence of `set` / `clear` /
`prepare` calls — interleaved with the tier promotions they trigger — `get`
returns precisely what a flat map `Nat → Option (src, dst)` would.

## What it indexes

Edge id → `(src, dst)`. Not node → incident edges, and not the type: the type
lives in `relationship_type_matrix`, the incidence in the per-type tensors. The
index holds one slot per edge id ever reserved; a deleted edge is a tombstone
(both fields = the width's `EMPTY`), and nothing ever shrinks, so there is no
swap-remove or compaction to get wrong. The only structural rewrite is
*promotion*: re-encoding a narrow tier into a wider one when a reused id's new
endpoints no longer fit.

## What is modelled, and where it lives in the tree

| here | there (`graph/src/graph/endpoint_index.rs` unless noted) |
| --- | --- |
| `bits`, `EMPTY`, `putF`         | `impl Endpoint for u16 / U24 / u32 / u64` (:81-124) — `put` truncates (`v as u16`, 3 LE bytes, ...) |
| `decode`                        | `get`'s `if s.vacant() && d.vacant() { None }` (:468) |
| `rankFor`                       | `EndpointIndex::rank_for` (:391) |
| `Ix` (`w16 w24 w32 w64`)        | `struct EndpointIndex` (:326); a `Paged<T>` is modelled as its flat slot list (see `Paged` section for the paging lemma) |
| `Ix.tier`, `Ix.setTier`         | the `match rank { 0 => w16, ..., _ => w64 }` arms of `tier_len` / `with_tier` (:537, :549) |
| `Ix.withTier`                   | `with_tier` (:549): `get_or_insert_with(Arc::default)` + `Arc::make_mut` + `f` |
| `growTo`, `putAt`, `vacate`     | `TierOps for Paged<T>`: `grow_to` (:732), `put` (:741), `vacate` (:750) |
| `Ix.appendRank`                 | `append_rank` (:402) |
| `Ix.len`, `Ix.tierLen`          | `len` (:412), `tier_len` (:537) |
| `Ix.get` / `walk`               | `get` (:459) — the `each_tier!` walk subtracting tier lengths |
| `Ix.locate` / `locAux`          | `locate` (:626) |
| `widen`, `Ix.promote`           | `promote` (:336) — `from.take()`, empty early-return, re-emitted sentinel |
| `Ix.ensure`                     | `get_or_insert_with(Arc::default)` inside `raise_to` |
| `Ix.raiseTo`                    | `raise_to` (:486) |
| `Ix.set`                        | `set` (:642) — the three `match self.locate(slot)` arms |
| `Ix.clear`                      | `clear` (:692) |
| `Ix.prepare`                    | `prepare` (:520) — `reserve_exact` has no semantic effect and is dropped |
| `Ix.prepareTiers`               | `prepare_tiers` (:605) |
| `Op`, `run`, `refRun`           | the call sites: `Graph::create_relationships_bulk` (`graph.rs:2582-2593`, `prepare` then `set` per edge), `delete_relationships` (`graph.rs:2844`) and `delete_implicit_edges` (`graph.rs:2991`) via `clear_edge_endpoint`, `Graph::restore` (`graph.rs:904-909`, `prepare_tiers` then `set`) |

## What is proved (all without `sorry`; see `#print axioms` at the end)

* `get_view` — `get` is lookup in the concatenation of the four decoded tiers.
* `locate_some` / `locate_none` — `locate` names an in-range offset whose tier
  prefix reconstructs the id, or the id is past the end.
* `raise1..3`, `raiseTo_spec` — promotion never changes what any id reads
  (tombstones and gaps included), keeps the width invariant, empties every tier
  below the target (so it starts at id 0) and leaves the tiers above alone.
* `view_set` — `set e s d` is "pad with absent slots to `e`, then overwrite `e`",
  in all three arms (in place, reuse-with-promotion, append / new tier), and
  preserves the invariant.
* `view_clear`, `view_prepare`, `view_prepareTiers` — tombstoning, batch sizing
  and the restore layout.
* `get_set`, `get_clear`, `get_prepare` — the pointwise spec.
* `run_refines` — **headline**: for any op sequence whose endpoints are 64-bit
  and not both `u64::MAX`, `get` after the run equals the reference map.
* `restore_agrees` — the restore path (`prepare_tiers` then `set` of every edge)
  yields exactly the edge set, whatever tier boundaries it is given.
* `paged_at`, `ensurePages_spec` — `pages[i / P][i % P]` addresses the flat list
  under the page-shape invariant, and `ensure_pages` keeps the shape, only
  appends `EMPTY` slots and covers the requested length.
* `mem_collect`, `collect_nodup`, `detach_delete_index` — `delete_implicit_edges`'
  per-type collection is exactly "incident to a deleted node, not explicit",
  never lists an id twice, and clearing it leaves no stale slot.
* `u64_max_pair_reads_as_absent` — the one boundary where the spec needs its
  hypothesis: an edge `(u64::MAX, u64::MAX)` reads back as deleted.

## Companion: `EndpointPaged.lean`

`Arc` sharing, page layout and page-level writes are proved there: `pageMut_spec`
(a write through one version changes only its page `p`, no other live version, and keeps
strong counts exact — the `Arc::get_mut`-or-copy of `page_mut`, and equally `make_mut`
on a tier), `cloneV_spec`, `pushFresh_spec`, `replace_spec` (`regrow`), `fromSlots_spec`
/ `promote_paged` (page layout of `from_slots`), `flatten_pagePut` / `put_spec` /
`vacate_spec` / `growTo_spec` (the paged `TierOps` are the flat ones), `reserve_covers`,
the memory accounting (`bytes_per_edge_bounds`), and `set_*_in_range` (every `put` the
three `set` arms issue is in range, which the flat `List.set` does not check).
-/

set_option linter.unusedSimpArgs false

namespace Falkor.EndpointIndex

abbrev Slot := Nat × Nat

/-! ## Field types -/

/-- Field width in bits: `u16`, `U24`, `u32`, `u64`. -/
def bits : Nat → Nat
  | 0 => 16
  | 1 => 24
  | 2 => 32
  | _ => 64

/-- `Endpoint::EMPTY` = all-ones of the width. Also `Endpoint::CEILING` for every
width (`u16::MAX as u64`, `0x00FF_FFFF`, `u32::MAX as u64`, `u64::MAX`). -/
def EMPTY : Nat → Nat
  | 0 => 65535
  | 1 => 16777215
  | 2 => 4294967295
  | _ => 18446744073709551615

/-- `Endpoint::put` — a truncating cast to the width. -/
def putF (r v : Nat) : Nat := v % 2 ^ bits r

/-- The read side of `EndpointIndex::get` for one slot (:467-472). -/
def decode (r : Nat) (p : Slot) : Option Slot :=
  if p.1 = EMPTY r ∧ p.2 = EMPTY r then none else some p

/-- `EndpointIndex::rank_for` (:391): `0..u16::CEILING => 0`, ... -/
def rankFor (v : Nat) : Nat :=
  if v < 65535 then 0
  else if v < 16777215 then 1
  else if v < 4294967295 then 2
  else 3

/-! ## `TierOps for Paged<T>`, on the flat slot list -/

def growTo (r n : Nat) (v : List Slot) : List Slot :=
  if n > v.length then v ++ List.replicate (n - v.length) (EMPTY r, EMPTY r) else v

def putAt (r at_ s d : Nat) (v : List Slot) : List Slot :=
  v.set at_ (putF r s, putF r d)

def vacate (r at_ : Nat) (v : List Slot) : List Slot :=
  if at_ < v.length then v.set at_ (EMPTY r, EMPTY r) else v

/-! ## The index -/

structure Ix where
  w16 : Option (List Slot) := none
  w24 : Option (List Slot) := none
  w32 : Option (List Slot) := none
  w64 : Option (List Slot) := none

namespace Ix

def tier (ix : Ix) : Nat → Option (List Slot)
  | 0 => ix.w16
  | 1 => ix.w24
  | 2 => ix.w32
  | _ => ix.w64

def setTier (ix : Ix) : Nat → Option (List Slot) → Ix
  | 0, o => { ix with w16 := o }
  | 1, o => { ix with w24 := o }
  | 2, o => { ix with w32 := o }
  | _, o => { ix with w64 := o }

def tierLen (ix : Ix) (r : Nat) : Nat := ((ix.tier r).map List.length).getD 0

def len (ix : Ix) : Nat := ix.tierLen 0 + ix.tierLen 1 + ix.tierLen 2 + ix.tierLen 3

/-- `with_tier`: `get_or_insert_with(Arc::default)`, `make_mut`, then `f`. -/
def withTier (ix : Ix) (r : Nat) (f : List Slot → List Slot) : Ix :=
  ix.setTier r (some (f ((ix.tier r).getD [])))

def appendRank (ix : Ix) : Nat :=
  if ix.w64.isSome then 3 else if ix.w32.isSome then 2 else if ix.w24.isSome then 1 else 0

/-- The `each_tier!` walk of `get`. -/
def walk (ix : Ix) : List Nat → Nat → Option Slot
  | [], _ => none
  | r :: rs, slot =>
    match ix.tier r with
    | none => walk ix rs slot
    | some v =>
      if h : slot < v.length then decode r v[slot] else walk ix rs (slot - v.length)

def get (ix : Ix) (e : Nat) : Option Slot := ix.walk [0, 1, 2, 3] e

def locAux (ix : Ix) : List Nat → Nat → Nat → Option (Nat × Nat)
  | [], _, _ => none
  | r :: rs, base, slot =>
    if slot < base + ix.tierLen r then some (r, slot - base)
    else locAux ix rs (base + ix.tierLen r) slot

def locate (ix : Ix) (slot : Nat) : Option (Nat × Nat) := ix.locAux [0, 1, 2, 3] 0 slot

end Ix

/-- The slot re-encoding inside `promote` (:353-359). -/
def widen (a b : Nat) (p : Slot) : Slot :=
  if p.1 = EMPTY a ∧ p.2 = EMPTY a then (EMPTY b, EMPTY b) else (putF b p.1, putF b p.2)

namespace Ix

def ensure (ix : Ix) (r : Nat) : Ix := ix.setTier r (some ((ix.tier r).getD []))

/-- `promote(from = tier a, into = tier b)`: `from.take()`, return if it was
`None` or empty, else `into := widened(from) ++ into`. -/
def promote (ix : Ix) (a b : Nat) : Ix :=
  match ix.tier a with
  | none => ix
  | some old =>
    let ix1 := ix.setTier a none
    if old.isEmpty then ix1
    else ix1.setTier b (some (old.map (widen a b) ++ (ix1.tier b).getD []))

def raiseTo (ix : Ix) : Nat → Ix
  | 1 => (ix.ensure 1).promote 0 1
  | 2 => (((ix.ensure 2).promote 1 2).promote 0 2)
  | 3 => ((((ix.ensure 3).promote 2 3).promote 1 3).promote 0 3)
  | _ => ix.ensure 0

def set (ix : Ix) (e s d : Nat) : Ix :=
  let slot := e
  let needed := rankFor (max s d)
  match ix.locate slot with
  | some (rank, at_) =>
    if needed > rank then (ix.raiseTo needed).withTier needed (putAt needed slot s d)
    else ix.withTier rank (putAt rank at_ s d)
  | none =>
    let rank := max needed ix.appendRank
    let at_ := slot - (ix.len - ix.tierLen rank)
    ix.withTier rank (fun v => putAt rank at_ s d (growTo rank (at_ + 1) v))

def clear (ix : Ix) (e : Nat) : Ix :=
  match ix.locate e with
  | some (rank, at_) => ix.withTier rank (vacate rank at_)
  | none => ix

def prepare (ix : Ix) (maxEdgeId maxEndpoint : Nat) : Ix :=
  let rank := max (rankFor maxEndpoint) ix.appendRank
  let needed := maxEdgeId + 1
  let below := ix.len - ix.tierLen rank
  let want := needed - below
  ix.withTier rank (growTo rank want)

/-- `prepare_tiers`: one `with_tier(rank, grow_to(n))` per non-empty tier. -/
def prepareTiers (ix : Ix) (s1 s2 s3 len : Nat) : Ix :=
  let s1 := min s1 len
  let s2 := min s2 len
  let s3 := min s3 len
  let step (ix : Ix) (rn : Nat × Nat) : Ix :=
    if rn.2 = 0 then ix else ix.withTier rn.1 (growTo rn.1 rn.2)
  [(0, s1), (1, s2 - s1), (2, s3 - s2), (3, len - s3)].foldl step ix

end Ix


/-! # Proofs -/

/-! ## The flat view -/

/-- What one tier answers, slot by slot. -/
def viewT (r : Nat) (o : Option (List Slot)) : List (Option Slot) := (o.getD []).map (decode r)

/-- What the whole index answers: the tiers in id order, decoded. -/
def view (ix : Ix) : List (Option Slot) :=
  viewT 0 ix.w16 ++ viewT 1 ix.w24 ++ viewT 2 ix.w32 ++ viewT 3 ix.w64

/-- Lookup in a flat answer list: past the end is absent. -/
def lk (l : List (Option Slot)) (i : Nat) : Option Slot := l[i]?.join

theorem walk_flat (ix : Ix) (rs : List Nat) (slot : Nat) :
    ix.walk rs slot = lk (rs.flatMap (fun r => viewT r (ix.tier r))) slot := by
  induction rs generalizing slot with
  | nil => simp [Ix.walk, lk]
  | cons r rs ih =>
    simp only [Ix.walk, List.flatMap_cons]
    cases h : ix.tier r with
    | none => simp [viewT, ih]
    | some v =>
      simp only
      split
      · rename_i hs
        simp only [lk, viewT, h, Option.getD_some]
        rw [List.getElem?_append_left (by simpa using hs)]
        simp [hs]
      · rename_i hs
        rw [ih]
        simp only [lk, viewT, h, Option.getD_some]
        rw [List.getElem?_append_right (by simp; omega)]
        simp

theorem get_view (ix : Ix) (i : Nat) : ix.get i = lk (view ix) i := by
  simp [Ix.get, walk_flat, view, Ix.tier, List.append_assoc]


/-! ## Splicing a tier out of the view -/

/-- The answers of every tier below `r`. -/
def pre (ix : Ix) : Nat → List (Option Slot)
  | 0 => []
  | 1 => viewT 0 ix.w16
  | 2 => viewT 0 ix.w16 ++ viewT 1 ix.w24
  | _ => viewT 0 ix.w16 ++ viewT 1 ix.w24 ++ viewT 2 ix.w32

/-- The answers of every tier above `r`. -/
def suf (ix : Ix) : Nat → List (Option Slot)
  | 0 => viewT 1 ix.w24 ++ viewT 2 ix.w32 ++ viewT 3 ix.w64
  | 1 => viewT 2 ix.w32 ++ viewT 3 ix.w64
  | 2 => viewT 3 ix.w64
  | _ => []

theorem rank_cases {r : Nat} (h : r ≤ 3) : r = 0 ∨ r = 1 ∨ r = 2 ∨ r = 3 := by omega

theorem splice (ix : Ix) {r : Nat} (h : r ≤ 3) :
    view ix = pre ix r ++ viewT r (ix.tier r) ++ suf ix r := by
  rcases rank_cases h with rfl | rfl | rfl | rfl <;> simp [view, pre, suf, Ix.tier]

theorem pre_setTier (ix : Ix) {r : Nat} (h : r ≤ 3) (o) : pre (ix.setTier r o) r = pre ix r := by
  rcases rank_cases h with rfl | rfl | rfl | rfl <;> rfl

theorem suf_setTier (ix : Ix) {r : Nat} (h : r ≤ 3) (o) : suf (ix.setTier r o) r = suf ix r := by
  rcases rank_cases h with rfl | rfl | rfl | rfl <;> rfl

theorem tier_setTier (ix : Ix) {r : Nat} (h : r ≤ 3) (o) : (ix.setTier r o).tier r = o := by
  rcases rank_cases h with rfl | rfl | rfl | rfl <;> rfl

theorem tier_setTier_ne (ix : Ix) {r k : Nat} (h : r ≤ 3) (hk : k ≤ 3) (hne : k ≠ r) (o) :
    (ix.setTier r o).tier k = ix.tier k := by
  rcases rank_cases h with rfl | rfl | rfl | rfl <;>
  rcases rank_cases hk with rfl | rfl | rfl | rfl <;> first | rfl | omega

theorem view_setTier (ix : Ix) {r : Nat} (h : r ≤ 3) (o) :
    view (ix.setTier r o) = pre ix r ++ viewT r o ++ suf ix r := by
  rw [splice _ h, pre_setTier _ h, suf_setTier _ h, tier_setTier _ h]

theorem view_withTier (ix : Ix) {r : Nat} (h : r ≤ 3) (f) :
    view (ix.withTier r f) = pre ix r ++ ((f ((ix.tier r).getD [])).map (decode r)) ++ suf ix r := by
  simp [Ix.withTier, view_setTier _ h, viewT]

theorem tierLen_eq (ix : Ix) (r : Nat) : ix.tierLen r = (viewT r (ix.tier r)).length := by
  cases h : ix.tier r <;> simp [Ix.tierLen, viewT, h]

theorem len_view (ix : Ix) : ix.len = (view ix).length := by
  simp [Ix.len, view, tierLen_eq, Ix.tier]; omega

theorem pre_length (ix : Ix) : ∀ r, (pre ix r).length + ix.tierLen r ≤ ix.len := by
  intro r
  rcases Nat.lt_or_ge r 4 with h | h
  · rcases rank_cases (by omega : r ≤ 3) with rfl | rfl | rfl | rfl <;>
      simp [pre, Ix.len, tierLen_eq, Ix.tier] <;> omega
  · have : ∀ k, 3 < k → pre ix k = pre ix 3 ∧ ix.tierLen k = ix.tierLen 3 := by
      intro k hk
      match k, hk with
      | k + 4, _ => exact ⟨rfl, rfl⟩
    rw [(this r (by omega)).1, (this r (by omega)).2]
    simp [pre, Ix.len, tierLen_eq, Ix.tier]; omega

/-- `locate` (:626): a hit names a tier `r ≤ 3`, an in-range offset, and that
offset plus the lengths below reconstructs the id; a miss means past the end. -/
theorem locate_some (ix : Ix) {slot r at_ : Nat} (h : ix.locate slot = some (r, at_)) :
    r ≤ 3 ∧ at_ < ix.tierLen r ∧ slot = (pre ix r).length + at_ := by
  simp only [Ix.locate, Ix.locAux] at h
  simp only [tierLen_eq, Ix.tier] at h ⊢
  split at h
  · simp at h; obtain ⟨rfl, rfl⟩ := h; simp [pre]; omega
  split at h
  · simp at h; obtain ⟨rfl, rfl⟩ := h; simp [pre]; omega
  split at h
  · simp at h; obtain ⟨rfl, rfl⟩ := h; simp [pre]; omega
  split at h
  · simp at h; obtain ⟨rfl, rfl⟩ := h; simp [pre]; omega
  · simp at h

theorem locate_none (ix : Ix) {slot : Nat} (h : ix.locate slot = none) : ix.len ≤ slot := by
  simp only [Ix.locate, Ix.locAux] at h
  simp only [Ix.len]
  repeat (split at h; · simp at h)
  omega


/-! ## The width invariant -/

/-- A slot of tier `r < 3` is either the sentinel pair or fits strictly below the
sentinel in both fields. `rank_for` is what guarantees the second; the `u64`
tier needs nothing, because nothing is ever widened out of it. -/
def ok (r : Nat) (p : Slot) : Prop :=
  r < 3 → (p.1 = EMPTY r ∧ p.2 = EMPTY r) ∨ (p.1 < EMPTY r ∧ p.2 < EMPTY r)

def TierOK (r : Nat) (o : Option (List Slot)) : Prop := ∀ p ∈ o.getD [], ok r p

def WF (ix : Ix) : Prop := ∀ r, r ≤ 3 → TierOK r (ix.tier r)

theorem ok_empty (r : Nat) : ok r (EMPTY r, EMPTY r) := fun _ => Or.inl ⟨rfl, rfl⟩

theorem decode_empty (r : Nat) : decode r (EMPTY r, EMPTY r) = none := by simp [decode]

/-- A value `rank_for` placed at or below `r` is stored unchanged, reads back as
live, and satisfies the invariant. For the `u64` tier the pair must not be the
sentinel pair `(u64::MAX, u64::MAX)`. -/
theorem fits {r s d : Nat} (hr : r ≤ 3) (hn : rankFor (max s d) ≤ r)
    (hs : s < 2 ^ 64) (hd : d < 2 ^ 64) (hm : ¬ (s = EMPTY 3 ∧ d = EMPTY 3)) :
    putF r s = s ∧ putF r d = d ∧ decode r (s, d) = some (s, d) ∧ ok r (s, d) := by
  simp only [rankFor] at hn
  simp only [EMPTY] at hm
  rcases rank_cases hr with rfl | rfl | rfl | rfl <;>
  · simp only [putF, bits, decode, ok, EMPTY]
    refine ⟨?_, ?_, ?_, ?_⟩
    all_goals (repeat' split at hn) <;> (try simp) <;> omega

theorem widen_ok {a b : Nat} (hab : a < b) (hb : b ≤ 3) {p : Slot} (hp : ok a p) :
    decode b (widen a b p) = decode a p ∧ ok b (widen a b p) := by
  have ha : a < 3 := by omega
  have hp := hp ha
  obtain ⟨p1, p2⟩ := p
  rcases rank_cases (by omega : a ≤ 3) with rfl | rfl | rfl | rfl <;>
  rcases rank_cases hb with rfl | rfl | rfl | rfl <;> (try omega) <;>
  · simp only [widen, decode, ok, putF, bits, EMPTY] at hp ⊢
    rcases hp with ⟨h1, h2⟩ | ⟨h1, h2⟩
    · simp [h1, h2]
    · refine ⟨?_, ?_⟩ <;> (repeat' split) <;> (try simp) <;> omega

theorem tierOK_withTier {r : Nat} {v : List Slot} (f : List Slot → List Slot)
    (hf : ∀ p ∈ f v, p ∈ v ∨ ok r p) (hv : ∀ p ∈ v, ok r p) : ∀ p ∈ f v, ok r p := by
  intro p hp
  rcases hf p hp with h | h
  · exact hv p h
  · exact h

theorem wf_setTier {ix : Ix} (h : WF ix) {r : Nat} (hr : r ≤ 3) {o} (ho : TierOK r o) :
    WF (ix.setTier r o) := by
  intro k hk
  by_cases hkr : k = r
  · subst hkr; rw [tier_setTier _ hr]; exact ho
  · rw [tier_setTier_ne _ hr hk hkr]; exact h k hk

theorem mem_set_or {l : List Slot} {i : Nat} {x p : Slot} (h : p ∈ l.set i x) : p ∈ l ∨ p = x :=
  List.mem_or_eq_of_mem_set h

theorem mem_growTo {r n : Nat} {v : List Slot} {p : Slot} (h : p ∈ growTo r n v) :
    p ∈ v ∨ p = (EMPTY r, EMPTY r) := by
  unfold growTo at h
  split at h
  · rcases List.mem_append.1 h with h | h
    · exact Or.inl h
    · exact Or.inr (List.eq_of_mem_replicate h)
  · exact Or.inl h

theorem viewT_growTo (r n : Nat) (v : List Slot) :
    (growTo r n v).map (decode r) = v.map (decode r) ++ List.replicate (n - v.length) none := by
  unfold growTo
  split
  · simp [List.map_replicate, decode_empty]
  · have : n - v.length = 0 := by omega
    simp [this]


/-! ## Promotion preserves every answer -/

theorem map_widen {a b : Nat} (hab : a < b) (hb : b ≤ 3) {old : List Slot}
    (h : ∀ p ∈ old, ok a p) :
    (old.map (widen a b)).map (decode b) = old.map (decode a) ∧
    ∀ p ∈ old.map (widen a b), ok b p := by
  refine ⟨?_, ?_⟩
  · simp only [List.map_map]
    apply List.map_congr_left
    intro p hp
    exact (widen_ok hab hb (h p hp)).1
  · intro p hp
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
    exact (widen_ok hab hb (h q hq)).2

theorem ok3 (p : Slot) : ok 3 p := fun h => absurd h (by decide)

theorem wf_mk {w16 w24 w32 w64 : Option (List Slot)} :
    WF ⟨w16, w24, w32, w64⟩ ↔
      (∀ p ∈ w16.getD [], ok 0 p) ∧ (∀ p ∈ w24.getD [], ok 1 p) ∧ (∀ p ∈ w32.getD [], ok 2 p) := by
  constructor
  · intro h; exact ⟨h 0 (by decide), h 1 (by decide), h 2 (by decide)⟩
  · rintro ⟨h0, h1, h2⟩ r hr
    rcases rank_cases hr with rfl | rfl | rfl | rfl
    · exact h0
    · exact h1
    · exact h2
    · exact fun p _ => ok3 p

/-! Closed forms of `raise_to`: promotion is `from.take()` then
"widened slots go to the front of `into`", tier by tier from the top down.
The empty-tier early return in `promote` and the `None` case both agree with
the closed form, which is what the case split checks. -/

theorem raise1 (w16 w24 w32 w64 : Option (List Slot)) :
    Ix.raiseTo ⟨w16, w24, w32, w64⟩ 1 =
      ⟨none, some ((w16.getD []).map (widen 0 1) ++ w24.getD []), w32, w64⟩ := by
  rcases w16 with _ | ⟨_ | ⟨a, l0⟩⟩ <;>
    simp [Ix.raiseTo, Ix.promote, Ix.ensure, Ix.setTier, Ix.tier]

theorem raise2 (w16 w24 w32 w64 : Option (List Slot)) :
    Ix.raiseTo ⟨w16, w24, w32, w64⟩ 2 =
      ⟨none, none, some ((w16.getD []).map (widen 0 2) ++
        ((w24.getD []).map (widen 1 2) ++ w32.getD [])), w64⟩ := by
  rcases w16 with _ | ⟨_ | ⟨a, l0⟩⟩ <;> rcases w24 with _ | ⟨_ | ⟨b, l1⟩⟩ <;>
    simp [Ix.raiseTo, Ix.promote, Ix.ensure, Ix.setTier, Ix.tier]

theorem raise3 (w16 w24 w32 w64 : Option (List Slot)) :
    Ix.raiseTo ⟨w16, w24, w32, w64⟩ 3 =
      ⟨none, none, none, some ((w16.getD []).map (widen 0 3) ++
        ((w24.getD []).map (widen 1 3) ++ ((w32.getD []).map (widen 2 3) ++ w64.getD [])))⟩ := by
  rcases w16 with _ | ⟨_ | ⟨a, l0⟩⟩ <;> rcases w24 with _ | ⟨_ | ⟨b, l1⟩⟩ <;>
    rcases w32 with _ | ⟨_ | ⟨c, l2⟩⟩ <;>
    simp [Ix.raiseTo, Ix.promote, Ix.ensure, Ix.setTier, Ix.tier]

/-- `raise_to(n)` (:486), `n ∈ {1,2,3}`: every answer is unchanged (live pairs,
tombstones and gaps alike), the invariant holds, every tier below `n` is gone
— so tier `n` now starts at edge id 0, which is why `set` may index it by the
raw id — and every tier above `n` is untouched. -/
theorem raiseTo_spec (ix : Ix) {n : Nat} (h1 : 1 ≤ n) (h3 : n ≤ 3) (hwf : WF ix) :
    view (ix.raiseTo n) = view ix ∧ WF (ix.raiseTo n) ∧
    pre (ix.raiseTo n) n = [] ∧ suf (ix.raiseTo n) n = suf ix n := by
  obtain ⟨w16, w24, w32, w64⟩ := ix
  obtain ⟨h0, h1', h2⟩ := wf_mk.1 hwf
  rcases rank_cases h3 with rfl | rfl | rfl | rfl
  · omega
  · have m01 := map_widen (by decide : 0 < 1) (by decide) h0
    rw [raise1]
    refine ⟨?_, ?_, rfl, rfl⟩
    · simp [view, viewT, m01.1]
    · refine wf_mk.2 ⟨by simp, ?_, h2⟩
      intro p hp
      simp only [Option.getD_some, List.mem_append] at hp
      rcases hp with hp | hp
      · exact m01.2 p hp
      · exact h1' p hp
  · have m02 := map_widen (by decide : 0 < 2) (by decide) h0
    have m12 := map_widen (by decide : 1 < 2) (by decide) h1'
    rw [raise2]
    refine ⟨?_, ?_, rfl, rfl⟩
    · simp [view, viewT, m02.1, m12.1]
    · refine wf_mk.2 ⟨by simp, by simp, ?_⟩
      intro p hp
      simp only [Option.getD_some, List.mem_append] at hp
      rcases hp with hp | hp | hp
      · exact m02.2 p hp
      · exact m12.2 p hp
      · exact h2 p hp
  · have m03 := map_widen (by decide : 0 < 3) (by decide) h0
    have m13 := map_widen (by decide : 1 < 3) (by decide) h1'
    have m23 := map_widen (by decide : 2 < 3) (by decide) h2
    rw [raise3]
    refine ⟨?_, ?_, rfl, rfl⟩
    · simp [view, viewT, m03.1, m13.1, m23.1]
    · exact wf_mk.2 ⟨by simp, by simp, by simp⟩


/-! ## `set` -/

/-- The reference behaviour of one write on the flat answer list: pad with
absent slots up to `e`, then overwrite slot `e`. -/
def padSet (l : List (Option Slot)) (e : Nat) (x : Option Slot) : List (Option Slot) :=
  (l ++ List.replicate (e + 1 - l.length) none).set e x

theorem lk_padSet (l : List (Option Slot)) (e i : Nat) (x : Option Slot) :
    lk (padSet l e x) i = if i = e then x else lk l i := by
  simp only [lk, padSet, List.getElem?_set, List.length_append, List.length_replicate]
  by_cases hie : i = e
  · subst hie; simp; split <;> first | rfl | omega
  · have : ¬ e = i := fun h => hie h.symm
    simp only [this, ite_false, hie]
    by_cases hi : i < l.length
    · rw [List.getElem?_append_left hi]
    · rw [List.getElem?_append_right (by omega), List.getElem?_replicate,
        List.getElem?_eq_none (by omega)]
      split <;> rfl

theorem padSet_lt {l : List (Option Slot)} {e : Nat} (h : e < l.length) (x) :
    padSet l e x = l.set e x := by
  simp [padSet, show e + 1 - l.length = 0 by omega]

theorem set_mid {P M S : List (Option Slot)} {at_ : Nat} (h : at_ < M.length) (x) :
    (P ++ M ++ S).set (P.length + at_) x = P ++ M.set at_ x ++ S := by
  rw [List.append_assoc, List.set_append, if_neg (by omega), Nat.add_sub_cancel_left,
    List.set_append, if_pos h, List.append_assoc]

theorem pad_append (P M : List (Option Slot)) {at_ : Nat} (h : M.length ≤ at_) (x) :
    P ++ (M ++ List.replicate (at_ + 1 - M.length) none).set at_ x =
      padSet (P ++ M) (P.length + at_) x := by
  unfold padSet
  rw [List.length_append, List.append_assoc, List.set_append (s := P), if_neg (by omega),
    Nat.add_sub_cancel_left]
  rw [show P.length + at_ + 1 - (P.length + M.length) = at_ + 1 - M.length by omega]

theorem rankFor_le (v : Nat) : rankFor v ≤ 3 := by
  unfold rankFor; repeat' split
  all_goals omega

theorem appendRank_le (ix : Ix) : ix.appendRank ≤ 3 := by
  unfold Ix.appendRank; repeat' split
  all_goals omega

/-- Nothing lies above the append tier: it is the widest `Some`. -/
theorem suf_nil (ix : Ix) {r : Nat} (hr : r ≤ 3) (h : ix.appendRank ≤ r) : suf ix r = [] := by
  obtain ⟨w16, w24, w32, w64⟩ := ix
  rcases w24 with _ | l1 <;> rcases w32 with _ | l2 <;> rcases w64 with _ | l3 <;>
    simp [Ix.appendRank] at h <;>
    rcases rank_cases hr with rfl | rfl | rfl | rfl <;> first | omega | simp [suf, viewT]

theorem len_splice (ix : Ix) {r : Nat} (hr : r ≤ 3) :
    ix.len = (pre ix r).length + ix.tierLen r + (suf ix r).length := by
  rw [len_view, splice ix hr, tierLen_eq]; simp; omega

theorem pre_mono (ix : Ix) {r n : Nat} (h : r < n) (hn : n ≤ 3) :
    (pre ix r).length + ix.tierLen r ≤ (pre ix n).length := by
  rcases rank_cases hn with rfl | rfl | rfl | rfl <;> (try omega) <;>
  rcases rank_cases (by omega : r ≤ 3) with rfl | rfl | rfl | rfl <;> (try omega) <;>
  simp [pre, tierLen_eq, Ix.tier] <;> omega

/-- Endpoints the engine can hand to `set`: 64-bit, and not the one pair that
coincides with the `u64` tier's tombstone. -/
def Valid (s d : Nat) : Prop := s < 2 ^ 64 ∧ d < 2 ^ 64 ∧ ¬ (s = EMPTY 3 ∧ d = EMPTY 3)

theorem tierOK_set {r : Nat} {v : List Slot} {i : Nat} {x : Slot}
    (hv : ∀ p ∈ v, ok r p) (hx : ok r x) : ∀ p ∈ v.set i x, ok r p := by
  intro p hp
  rcases mem_set_or hp with h | h
  · exact hv p h
  · exact h ▸ hx

theorem tier_ok (ix : Ix) (hwf : WF ix) {r : Nat} (hr : r ≤ 3) : ∀ p ∈ (ix.tier r).getD [], ok r p :=
  hwf r hr

/-- **`set` (:642), all three arms.** Reusing an id in place, reusing a low id
whose new endpoints force a promotion, and appending past the end (into the
widest active tier or a fresh wider one) all have the same observable effect:
pad, then overwrite slot `e` with `(s, d)`. And the invariant survives. -/
theorem view_set (ix : Ix) (hwf : WF ix) (e s d : Nat) (hv : Valid s d) :
    view (ix.set e s d) = padSet (view ix) e (some (s, d)) ∧ WF (ix.set e s d) := by
  obtain ⟨hs, hd, hm⟩ := hv
  unfold Ix.set
  simp only
  split
  · rename_i rank at_ hloc
    obtain ⟨hr, hat, he⟩ := locate_some ix hloc
    split
    · -- reuse with promotion
      rename_i hgt
      have hn3 := rankFor_le (max s d)
      have hn1 : 1 ≤ rankFor (max s d) := by omega
      obtain ⟨hvw, hwf', hpre, hsuf⟩ := raiseTo_spec ix hn1 hn3 hwf
      obtain ⟨p1, p2, pdec, pok⟩ := fits hn3 (Nat.le_refl _) hs hd hm
      have hsp := splice (ix.raiseTo (rankFor (max s d))) hn3
      rw [hpre, hsuf, hvw] at hsp
      have hlt : e < (viewT (rankFor (max s d)) ((ix.raiseTo (rankFor (max s d))).tier (rankFor (max s d)))).length := by
        have h1 := pre_mono ix hgt hn3
        have h2 := len_splice ix hn3
        have h3 := congrArg List.length hsp
        rw [← len_view] at h3
        simp at h3
        omega
      refine ⟨?_, ?_⟩
      · rw [view_withTier _ hn3, hpre, hsuf, padSet_lt (by rw [hsp]; simp; omega), hsp]
        simp only [putAt, p1, p2, List.map_set, pdec, List.nil_append]
        rw [List.set_append]
        simp only [viewT, List.length_map] at hlt ⊢
        simp [hlt]
      · exact wf_setTier hwf' hn3 (tierOK_set (tier_ok _ hwf' hn3) (by simpa [p1, p2] using pok))
    · -- in place
      rename_i hle
      have hle : rankFor (max s d) ≤ rank := by omega
      obtain ⟨p1, p2, pdec, pok⟩ := fits hr hle hs hd hm
      refine ⟨?_, ?_⟩
      · rw [view_withTier _ hr, padSet_lt (by rw [← len_view, len_splice ix hr]; omega),
          splice ix hr, he]
        simp only [putAt, p1, p2, List.map_set, pdec]
        rw [tierLen_eq] at hat
        exact (set_mid (by simpa [viewT] using hat) _).symm
      · exact wf_setTier hwf hr (tierOK_set (tier_ok _ hwf hr) (by simpa [p1, p2] using pok))
  · -- append
    rename_i hloc
    have hlen := locate_none ix hloc
    have hr : max (rankFor (max s d)) ix.appendRank ≤ 3 :=
      Nat.max_le.2 ⟨rankFor_le _, appendRank_le _⟩
    have hsuf := suf_nil ix hr (Nat.le_max_right _ _)
    have hls := len_splice ix hr
    rw [hsuf] at hls
    simp at hls
    have hat0 : ix.len - ix.tierLen (max (rankFor (max s d)) ix.appendRank) =
        (pre ix (max (rankFor (max s d)) ix.appendRank)).length := by omega
    obtain ⟨p1, p2, pdec, pok⟩ := fits hr (Nat.le_max_left _ _) hs hd hm
    refine ⟨?_, ?_⟩
    · rw [view_withTier _ hr, hsuf, splice ix hr, hsuf]
      simp only [List.append_nil, putAt, p1, p2, List.map_set, pdec, viewT_growTo]
      generalize hR : max (rankFor (max s d)) ix.appendRank = R at *
      have hT := tierLen_eq ix R
      simp only [viewT] at hT
      rw [hat0]
      have he : e = (pre ix R).length + (e - (pre ix R).length) := by omega
      have hls' := len_splice ix hr
      rw [suf_nil ix hr (hR ▸ Nat.le_max_right _ _)] at hls'
      have hT' : ix.tierLen R = ((ix.tier R).getD []).length := by
        simp only [Ix.tierLen]; cases ix.tier R <;> rfl
      simp only [List.length_nil, Nat.add_zero] at hls'
      have hge : ((ix.tier R).getD []).length ≤ e - (pre ix R).length := by omega
      rw [show padSet (pre ix R ++ viewT R (ix.tier R)) e (some (s, d)) =
          padSet (pre ix R ++ viewT R (ix.tier R)) ((pre ix R).length + (e - (pre ix R).length))
            (some (s, d)) from by rw [← he]]
      simp only [viewT]
      rw [show ((ix.tier R).getD []).length = (List.map (decode R) ((ix.tier R).getD [])).length by simp]
      exact pad_append (pre ix R) (List.map (decode R) ((ix.tier R).getD [])) (by simpa using hge) _
    · refine wf_setTier hwf hr ?_
      exact tierOK_set (fun p hp => by
        rcases mem_growTo hp with h | h
        · exact tier_ok _ hwf hr p h
        · exact h ▸ ok_empty _) (by simpa [p1, p2] using pok)


/-! ## `clear`, `prepare`, `prepare_tiers` -/

/-- `clear` (:692): an id inside some tier becomes a tombstone; an id past the
end is left alone (and already reads as absent). -/
theorem view_clear (ix : Ix) (hwf : WF ix) (e : Nat) :
    (∀ i, lk (view (ix.clear e)) i = if i = e then none else lk (view ix) i) ∧
    WF (ix.clear e) := by
  unfold Ix.clear
  split
  · rename_i r at_ hloc
    obtain ⟨hr, hat, he⟩ := locate_some ix hloc
    have hv : at_ < ((ix.tier r).getD []).length := by
      rw [tierLen_eq] at hat; simpa [viewT] using hat
    have hview : view (ix.withTier r (vacate r at_)) = (view ix).set e none := by
      rw [view_withTier _ hr, splice ix hr, he]
      simp only [vacate, if_pos hv, List.map_set, decode_empty]
      exact (set_mid (by simpa [viewT] using hv) _).symm
    refine ⟨fun i => ?_, ?_⟩
    · rw [hview]
      have : e < (view ix).length := by rw [← len_view, len_splice ix hr]; omega
      simp only [lk, List.getElem?_set]
      by_cases hie : i = e
      · subst hie; simp [this]
      · simp [hie, Ne.symm hie]
    · refine wf_setTier hwf hr ?_
      intro p hp
      simp only [Option.getD_some, vacate, if_pos hv] at hp
      rcases mem_set_or hp with h | h
      · exact tier_ok _ hwf hr p h
      · exact h ▸ ok_empty _
  · rename_i hloc
    have hlen := locate_none ix hloc
    refine ⟨fun i => ?_, hwf⟩
    by_cases hie : i = e
    · subst hie
      simp only [if_true, lk]
      rw [List.getElem?_eq_none (by rw [← len_view]; omega)]
      rfl
    · simp [hie]

theorem lk_append_none (l : List (Option Slot)) (k i : Nat) :
    lk (l ++ List.replicate k none) i = lk l i := by
  simp only [lk]
  by_cases hi : i < l.length
  · rw [List.getElem?_append_left hi]
  · rw [List.getElem?_append_right (by omega), List.getElem?_replicate,
      List.getElem?_eq_none (by omega)]
    split <;> rfl

/-- Growing a tier appends absent slots and nothing else. -/
theorem view_grow (ix : Ix) (hwf : WF ix) {r : Nat} (hr : r ≤ 3) (n : Nat) :
    view (ix.withTier r (growTo r n)) =
      pre ix r ++ (viewT r (ix.tier r) ++
        List.replicate (n - ((ix.tier r).getD []).length) none) ++ suf ix r ∧
    WF (ix.withTier r (growTo r n)) := by
  refine ⟨?_, ?_⟩
  · rw [view_withTier _ hr, viewT_growTo]; rfl
  · refine wf_setTier hwf hr ?_
    intro p hp
    rcases mem_growTo hp with h | h
    · exact tier_ok _ hwf hr p h
    · exact h ▸ ok_empty _

/-- `prepare` (:520) only appends absent slots: it changes the cost of the
batch that follows, never an answer. -/
theorem view_prepare (ix : Ix) (hwf : WF ix) (m n : Nat) :
    (∃ k, view (ix.prepare m n) = view ix ++ List.replicate k none) ∧ WF (ix.prepare m n) := by
  unfold Ix.prepare
  simp only
  have hr : max (rankFor n) ix.appendRank ≤ 3 := Nat.max_le.2 ⟨rankFor_le _, appendRank_le _⟩
  have hsuf := suf_nil ix hr (Nat.le_max_right _ _)
  obtain ⟨hv, hw⟩ := view_grow ix hwf hr (m + 1 - (ix.len - ix.tierLen (max (rankFor n) ix.appendRank)))
  refine ⟨⟨m + 1 - (ix.len - ix.tierLen (max (rankFor n) ix.appendRank)) -
      ((ix.tier (max (rankFor n) ix.appendRank)).getD []).length, ?_⟩, hw⟩
  rw [hv, hsuf, splice ix hr, hsuf]
  simp only [List.append_nil, List.append_assoc]

theorem empty_wf : WF ({} : Ix) := by
  intro r hr p hp; rcases rank_cases hr with rfl | rfl | rfl | rfl <;> simp [Ix.tier] at hp

/-- A view that answers "absent" everywhere. -/
def AllNone (l : List (Option Slot)) : Prop := ∀ x ∈ l, x = none

theorem allNone_grow (ix : Ix) (hwf : WF ix) {r : Nat} (hr : r ≤ 3) (n : Nat)
    (h : AllNone (view ix)) : AllNone (view (ix.withTier r (growTo r n))) := by
  rw [(view_grow ix hwf hr n).1]
  rw [splice ix hr] at h
  intro x hx
  simp only [List.mem_append] at hx
  rcases hx with (hx | hx | hx) | hx
  · exact h x (by simp [hx])
  · exact h x (by simp [hx])
  · exact List.eq_of_mem_replicate hx
  · exact h x (by simp [hx])

/-- `prepare_tiers` (:605), from an empty index as `Graph::restore` calls it:
every slot it lays out reads as absent, whatever the boundaries. -/
theorem view_prepareTiers (s1 s2 s3 len : Nat) :
    AllNone (view (({} : Ix).prepareTiers s1 s2 s3 len)) ∧
    WF (({} : Ix).prepareTiers s1 s2 s3 len) := by
  unfold Ix.prepareTiers
  simp only
  have key : ∀ (rs : List (Nat × Nat)) (ix : Ix), (∀ rn ∈ rs, rn.1 ≤ 3) → WF ix →
      AllNone (view ix) →
      AllNone (view (rs.foldl (fun ix rn => if rn.2 = 0 then ix else ix.withTier rn.1 (growTo rn.1 rn.2)) ix)) ∧
      WF (rs.foldl (fun ix rn => if rn.2 = 0 then ix else ix.withTier rn.1 (growTo rn.1 rn.2)) ix) := by
    intro rs
    induction rs with
    | nil => intro ix _ hw ha; exact ⟨ha, hw⟩
    | cons rn rs ih =>
      intro ix hrs hw ha
      simp only [List.foldl_cons]
      have h1 : rn.1 ≤ 3 := hrs rn (List.mem_cons_self ..)
      apply ih _ (fun x hx => hrs x (List.mem_cons_of_mem _ hx))
      · split
        · exact hw
        · exact (view_grow ix hw h1 _).2
      · split
        · exact ha
        · exact allNone_grow ix hw h1 _ ha
  apply key
  · intro rn hrn; simp only [List.mem_cons, List.not_mem_nil, or_false] at hrn
    rcases hrn with h | h | h | h <;> subst h <;> simp
  · exact empty_wf
  · intro x hx; simp [view, viewT] at hx

theorem lk_allNone {l : List (Option Slot)} (h : AllNone l) (i : Nat) : lk l i = none := by
  simp only [lk]
  cases hl : l[i]? with
  | none => rfl
  | some x =>
    have := h x (List.mem_of_getElem? hl)
    simp [this]

/-! ## The pointwise spec -/

theorem get_set (ix : Ix) (hwf : WF ix) (e s d : Nat) (hv : Valid s d) (i : Nat) :
    (ix.set e s d).get i = if i = e then some (s, d) else ix.get i := by
  rw [get_view, get_view, (view_set ix hwf e s d hv).1, lk_padSet]

theorem get_clear (ix : Ix) (hwf : WF ix) (e i : Nat) :
    (ix.clear e).get i = if i = e then none else ix.get i := by
  rw [get_view, get_view, (view_clear ix hwf e).1]

theorem get_prepare (ix : Ix) (hwf : WF ix) (m n i : Nat) : (ix.prepare m n).get i = ix.get i := by
  obtain ⟨⟨k, hk⟩, _⟩ := view_prepare ix hwf m n
  rw [get_view, get_view, hk, lk_append_none]

/-! ## Any sequence of operations -/

/-- What the graph does to the index: `create_relationships_bulk` is one
`prepare` then a `set` per edge; both delete paths are `clear` per edge. -/
inductive Op where
  | set (e s d : Nat)
  | clear (e : Nat)
  | prepare (maxEdgeId maxEndpoint : Nat)

def Op.apply (ix : Ix) : Op → Ix
  | .set e s d => ix.set e s d
  | .clear e => ix.clear e
  | .prepare m n => ix.prepare m n

def Op.valid : Op → Prop
  | .set _ s d => Valid s d
  | _ => True

/-- The reference: a flat map from edge id to endpoints. -/
def Op.ref (f : Nat → Option Slot) : Op → (Nat → Option Slot)
  | .set e s d => fun i => if i = e then some (s, d) else f i
  | .clear e => fun i => if i = e then none else f i
  | .prepare _ _ => f

def run (ix : Ix) (ops : List Op) : Ix := ops.foldl Op.apply ix
def refRun (f : Nat → Option Slot) (ops : List Op) : Nat → Option Slot := ops.foldl Op.ref f

theorem step_refines (ix : Ix) (hwf : WF ix) (f : Nat → Option Slot) (hf : ∀ i, ix.get i = f i)
    (op : Op) (hop : op.valid) :
    WF (op.apply ix) ∧ ∀ i, (op.apply ix).get i = op.ref f i := by
  cases op with
  | set e s d =>
    exact ⟨(view_set ix hwf e s d hop).2, fun i => by
      simp only [Op.apply, Op.ref]; rw [get_set ix hwf e s d hop, hf]⟩
  | clear e =>
    exact ⟨(view_clear ix hwf e).2, fun i => by
      simp only [Op.apply, Op.ref]; rw [get_clear ix hwf, hf]⟩
  | prepare m n =>
    exact ⟨(view_prepare ix hwf m n).2, fun i => by
      simp only [Op.apply, Op.ref]; rw [get_prepare ix hwf, hf]⟩

/-- **Headline.** Starting from any well-formed index that agrees with a
reference map, every sequence of `set` / `clear` / `prepare` — with whatever
promotions, new tiers and tail growth they trigger internally — leaves an
index that still agrees with the reference, id by id. In particular: no stale
answer survives a delete and an id reuse (the reuse overwrites, the delete
tombstones), no duplicate slot exists for one id (the map is a function), and
multi-edges between the same pair are separate ids and never interact. -/
theorem run_refines (ops : List Op) : ∀ (ix : Ix) (f : Nat → Option Slot),
    WF ix → (∀ i, ix.get i = f i) → (∀ op ∈ ops, op.valid) →
    WF (run ix ops) ∧ ∀ i, (run ix ops).get i = refRun f ops i := by
  induction ops with
  | nil => intro ix f hw hf _; exact ⟨hw, hf⟩
  | cons op ops ih =>
    intro ix f hw hf hv
    obtain ⟨hw', hf'⟩ := step_refines ix hw f hf op (hv op (List.mem_cons_self ..))
    exact ih _ _ hw' hf' (fun o ho => hv o (List.mem_cons_of_mem _ ho))

theorem empty_get (i : Nat) : ({} : Ix).get i = none := by
  simp [Ix.get, Ix.walk, Ix.tier]

/-- From a fresh graph (`Graph::new`, `edge_endpoints: EndpointIndex::default()`). -/
theorem fresh_refines (ops : List Op) (hv : ∀ op ∈ ops, op.valid) (i : Nat) :
    (run {} ops).get i = refRun (fun _ => none) ops i :=
  (run_refines ops {} _ empty_wf empty_get hv).2 i

/-- MVCC: a snapshot is the value it was when taken. In this model that is by
construction (states are values); the Rust counterpart is the `Arc` audit in
the header. Stated so the claim is explicit rather than implicit. -/
theorem snapshot_isolated (ops₁ ops₂ : List Op) (hv₁ : ∀ op ∈ ops₁, op.valid) (i : Nat) :
    let snap := run {} ops₁
    let _live := run snap ops₂
    snap.get i = refRun (fun _ => none) ops₁ i :=
  fresh_refines ops₁ hv₁ i

/-- `Graph::restore` (`graph.rs:904-909`): `prepare_tiers` with any boundaries,
then `set` for every edge of every tensor. The index answers exactly the
restored edge set (the last `set` of an id wins, which cannot matter since a
tensor holds each edge id once). -/
theorem restore_agrees (s1 s2 s3 len : Nat) (edges : List (Nat × Nat × Nat))
    (hv : ∀ x ∈ edges, Valid x.2.1 x.2.2) (i : Nat) :
    (run (({} : Ix).prepareTiers s1 s2 s3 len) (edges.map fun x => Op.set x.1 x.2.1 x.2.2)).get i =
      refRun (fun _ => none) (edges.map fun x => Op.set x.1 x.2.1 x.2.2) i := by
  obtain ⟨hall, hwf⟩ := view_prepareTiers s1 s2 s3 len
  refine (run_refines _ _ _ hwf (fun j => by rw [get_view, lk_allNone hall]) ?_).2 i
  intro op hop
  obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hop
  exact hv x hx


/-! ## `Paged<T>`: the page arithmetic addresses the flat list

The model above treats a tier as its flat slot list. That is sound because of
the two facts here: with every page but the last full (`Shape`),
`pages[i / P][i % P]` (`Paged::at`, :192) is the `i`-th slot of the
concatenation; and `ensure_pages` (:256) -- the only thing that adds pages --
keeps `Shape`, only appends `EMPTY` slots, and covers the requested length. So
`grow_to` is exactly the flat `growTo`, including the slack a doubling tail
leaves past `len`, which is still `EMPTY` when a later `grow_to` exposes it. -/

/-- Every page holds at most `P` slots and every page but the last exactly `P`. -/
def Shape {α} (P : Nat) : List (List α) → Prop
  | [] => True
  | [p] => p.length ≤ P
  | p :: q :: ps => p.length = P ∧ Shape P (q :: ps)

theorem paged_at {α} (P : Nat) (hP : 0 < P) :
    ∀ (pages : List (List α)), Shape P pages → ∀ i, i < pages.flatten.length →
    pages.flatten[i]? = (pages[i / P]?).bind (·[i % P]?) := by
  intro pages
  induction pages with
  | nil => intro _ i hi; simp at hi
  | cons p ps ih =>
    intro hs i hi
    cases ps with
    | nil =>
      simp only [Shape] at hs
      simp only [List.flatten_cons, List.flatten_nil, List.append_nil, List.length_cons] at hi ⊢
      have h1 : i / P = 0 := Nat.div_eq_of_lt (by omega)
      have h2 : i % P = i := Nat.mod_eq_of_lt (by omega)
      simp [h1, h2]
    | cons q qs =>
      simp only [Shape] at hs
      obtain ⟨hp, hs'⟩ := hs
      simp only [List.flatten_cons] at hi ⊢
      by_cases hiP : i < P
      · have h1 : i / P = 0 := Nat.div_eq_of_lt hiP
        have h2 : i % P = i := Nat.mod_eq_of_lt hiP
        rw [List.getElem?_append_left (by omega), h1, h2]
        simp
      · rw [List.getElem?_append_right (by omega), Nat.div_eq_sub_div hP (by omega),
          Nat.mod_eq_sub_mod (by omega), hp]
        have := ih hs' (i - P) (by simp [List.length_append] at hi ⊢; omega)
        simp only [List.flatten_cons] at this
        rw [this]
        simp


def regrow {α} (fill : α) (page : List α) (n : Nat) : List α :=
  let held := min page.length n
  page.take held ++ List.replicate (n - held) fill

/-- `Paged::ensure_pages` (:256), for a page size `P` (`P` in the engine). -/
def ensurePages {α} (P : Nat) (fill : α) (pages : List (List α)) (slots : Nat) : List (List α) :=
  if slots = 0 then pages else
  let want := (slots + P - 1) / P
  let tail := slots - (want - 1) * P
  if pages.length < want then
    let pages1 := match pages.getLast? with
      | some last => if last.length < P then pages.dropLast ++ [regrow fill last P] else pages
      | none => pages
    let pages2 := pages1 ++ List.replicate (want - 1 - pages1.length) (List.replicate P fill)
    pages2 ++ [List.replicate tail fill]
  else
    match pages.getLast? with
    | some last =>
      if last.length < tail then
        pages.dropLast ++ [regrow fill last (min (max tail (last.length * 2)) P)]
      else pages
    | none => pages

theorem shape_concat {α} (P : Nat) (ps : List (List α)) (l : List α) :
    Shape P (ps ++ [l]) ↔ (∀ p ∈ ps, p.length = P) ∧ l.length ≤ P := by
  induction ps with
  | nil => simp [Shape]
  | cons p ps ih =>
    cases ps with
    | nil => simp [Shape]
    | cons q qs =>
      simp only [List.cons_append, Shape] at ih ⊢
      rw [ih]; simp [and_assoc]

theorem flatten_full_len {α} (P : Nat) (ps : List (List α)) (h : ∀ p ∈ ps, p.length = P) :
    ps.flatten.length = ps.length * P := by
  induction ps with
  | nil => simp
  | cons p ps ih =>
    simp only [List.flatten_cons, List.length_append, List.length_cons]
    rw [h p (List.mem_cons_self ..), ih (fun q hq => h q (List.mem_cons_of_mem _ hq)), Nat.succ_mul]
    omega

theorem regrow_ext {α} (fill : α) (l : List α) {n : Nat} (h : l.length ≤ n) :
    regrow fill l n = l ++ List.replicate (n - l.length) fill := by
  simp [regrow, Nat.min_eq_left h, List.take_of_length_le (Nat.le_refl _)]

theorem ensurePages_spec {α} (P : Nat) (hP : 0 < P) (fill : α) (pages : List (List α)) (slots : Nat)
    (hs : Shape P pages) :
    Shape P (ensurePages P fill pages slots) ∧
    (∃ k, (ensurePages P fill pages slots).flatten = pages.flatten ++ List.replicate k fill) ∧
    slots ≤ (ensurePages P fill pages slots).flatten.length := by
  unfold ensurePages
  by_cases h0 : slots = 0
  · simp only [h0, if_true]; exact ⟨hs, ⟨0, by simp⟩, Nat.zero_le _⟩
  simp only [h0, if_false]
  generalize hw : (slots + P - 1) / P = want
  have hd1 : (slots + P - 1) / P * P ≤ slots + P - 1 := Nat.div_mul_le_self _ _
  have hd2 : slots + P - 1 < (slots + P - 1) / P * P + P := Nat.lt_div_mul_add hP
  rw [hw] at hd1 hd2
  have hsub : (want - 1) * P = want * P - P := Nat.sub_one_mul _ _
  have hw1 : 1 ≤ want := by
    rcases Nat.eq_zero_or_pos want with h | h
    · subst h; simp at hd2; omega
    · exact h
  have hw2 : (want - 1) * P < slots := by omega
  have hw3 : slots ≤ want * P := by omega
  have hrep : ∀ n m, (List.replicate n (List.replicate m fill)).flatten = List.replicate (n * m) fill :=
    fun _ _ => List.flatten_replicate_replicate
  have hrr : ∀ a b, List.replicate a fill ++ List.replicate b fill = List.replicate (a + b) fill :=
    fun a b => by simp [List.replicate_append_replicate]
  have hfullrep : ∀ n, ∀ p ∈ List.replicate n (List.replicate P fill), p.length = P := by
    intro n p hp; rw [List.eq_of_mem_replicate hp]; simp
  rcases List.eq_nil_or_concat pages with rfl | ⟨ps, l, rfl⟩
  all_goals try rw [List.concat_eq_append] at hs ⊢
  · have hlt : ([] : List (List α)).length < want := by simp; omega
    rw [if_pos hlt]
    simp only [List.getLast?_nil, List.length_nil, Nat.sub_zero]
    refine ⟨(shape_concat _ _ _).2 ⟨hfullrep _, by simp; omega⟩, ⟨(want - 1) * P +
      (slots - (want - 1) * P), ?_⟩, ?_⟩
    · simp [hrep, hrr]
    · simp [hrep]; omega
  · obtain ⟨hps, hl⟩ := (shape_concat _ _ _).1 hs
    have hpsl := flatten_full_len _ ps hps
    simp only [List.getLast?_concat, List.dropLast_concat, List.length_append, List.length_cons,
      List.length_nil]
    by_cases hlt : ps.length + (0 + 1) < want
    · rw [if_pos hlt]
      have hl' : (if l.length < P then ps ++ [regrow fill l P] else ps ++ [l]) =
          ps ++ [l ++ List.replicate (P - l.length) fill] := by
        split
        · rw [regrow_ext _ _ hl]
        · have : P - l.length = 0 := by omega
          simp [this]
      rw [hl']
      simp only [List.length_append, List.length_cons, List.length_nil]
      refine ⟨?_, ⟨P - l.length + (want - 1 - (ps.length + (0 + 1))) * P +
        (slots - (want - 1) * P), ?_⟩, ?_⟩
      · refine (shape_concat _ _ _).2 ⟨?_, by (try simp only [List.length_replicate]); omega⟩
        intro p hp
        simp only [List.mem_append, List.mem_singleton] at hp
        rcases hp with (hp | hp) | hp
        · exact hps p hp
        · subst hp; simp; omega
        · exact hfullrep _ p hp
      · simp [hrep, hrr, List.append_assoc]; omega
      · simp [hrep, hpsl]
        have e1 : (want - 1 - (ps.length + 1)) * P = (want - 1) * P - (ps.length + 1) * P :=
          Nat.sub_mul _ _ _
        have e2 : (ps.length + 1) * P = ps.length * P + P := Nat.succ_mul _ _
        have e3 : (ps.length + 1) * P ≤ (want - 1) * P := Nat.mul_le_mul_right _ (by omega)
        omega
    · rw [if_neg hlt]
      split
      · rename_i hlt2
        rw [regrow_ext _ _ (by omega)]
        refine ⟨(shape_concat _ _ _).2 ⟨hps, by simp only [List.length_append, List.length_replicate]; omega⟩,
          ⟨min (max (slots - (want - 1) * P) (l.length * 2)) P - l.length, by simp⟩, ?_⟩
        have e3 : (want - 1) * P ≤ ps.length * P := Nat.mul_le_mul_right _ (by omega)
        simp [hpsl]
        omega
      · refine ⟨hs, ⟨0, by simp⟩, ?_⟩
        have e3 : (want - 1) * P ≤ ps.length * P := Nat.mul_le_mul_right _ (by omega)
        simp [hpsl]
        omega

/-! ## Detach-delete: which edges `delete_implicit_edges` clears

`Graph::delete_implicit_edges` (`graph.rs:2882-2997`) builds, per relationship
type, the list of edges to remove for a set `N` of deleted nodes: for each
`n ∈ N`, the outgoing edges of `n`, then the incoming edges of `n` whose source
is neither `n` (a self-loop, already seen outgoing) nor another deleted node
(already seen from its own outgoing side), skipping ids in `explicit_rels`.
It then subtracts `all_implicit.len()` from `relationship_count` and clears
each id from the index — so a duplicate would double-count, and a miss would
leave a stale slot pointing at a deleted node id that is about to be recycled. -/

/-- An edge of one tensor: `(edge_id, src, dst)`. -/
abbrev Edge := Nat × Nat × Nat

def collect (N X : List Nat) (E : List Edge) : List Edge :=
  N.flatMap fun n =>
    E.filter (fun e => e.2.1 == n && !(X.contains e.1)) ++
    E.filter (fun e => e.2.2 == n && e.2.1 != n && !(N.contains e.2.1) && !(X.contains e.1))

/-- Exactly the edges incident to a deleted node and not deleted explicitly. -/
theorem mem_collect (N X : List Nat) (E : List Edge) (e : Edge) :
    e ∈ collect N X E ↔ e ∈ E ∧ (e.2.1 ∈ N ∨ e.2.2 ∈ N) ∧ e.1 ∉ X := by
  simp only [collect, List.mem_flatMap, List.mem_append, List.mem_filter, Bool.and_eq_true,
    beq_iff_eq, Bool.not_eq_true', List.contains_iff_mem, bne_iff_ne, ne_eq,
    Bool.not_eq_eq_eq_not, Bool.not_true, decide_eq_false_iff_not]
  constructor
  · rintro ⟨n, hn, (⟨he, rfl, hx⟩ | ⟨he, ⟨⟨rfl, _⟩, _⟩, hx⟩)⟩
    · exact ⟨he, Or.inl hn, by simpa using hx⟩
    · exact ⟨he, Or.inr hn, by simpa using hx⟩
  · rintro ⟨he, hN, hx⟩
    have hx' : ¬ (List.contains X e.1 = true) := by simpa using hx
    by_cases hs : e.2.1 ∈ N
    · exact ⟨e.2.1, hs, Or.inl ⟨he, rfl, by simpa using hx'⟩⟩
    · rcases hN with h | h
      · exact absurd h hs
      · refine ⟨e.2.2, h, Or.inr ⟨he, ⟨⟨rfl, ?_⟩, by simpa using hs⟩, by simpa using hx'⟩⟩
        intro heq; exact hs (heq ▸ h)

/-- No edge id is collected twice, provided the tensor holds each id once and
the deleted-node set has no repeats (it is a `RoaringTreemap`). -/
theorem collect_nodup (N X : List Nat) (E : List Edge) (hN : N.Nodup)
    (hE : E.Pairwise (fun a b => a.1 ≠ b.1)) :
    (collect N X E).Pairwise (fun a b => a.1 ≠ b.1) := by
  have hNp := List.nodup_iff_pairwise_ne.1 hN
  have hid : ∀ a ∈ E, ∀ b ∈ E, a.1 = b.1 → a = b := by
    intro a ha b hb h
    induction E with
    | nil => simp at ha
    | cons x xs ih =>
      rw [List.pairwise_cons] at hE
      simp only [List.mem_cons] at ha hb
      rcases ha with rfl | ha <;> rcases hb with rfl | hb
      · rfl
      · exact absurd h (hE.1 b hb)
      · exact absurd h.symm (hE.1 a ha)
      · exact ih hE.2 ha hb
  unfold collect
  rw [List.pairwise_flatMap]
  refine ⟨fun n _ => ?_, ?_⟩
  · rw [List.pairwise_append]
    refine ⟨List.Pairwise.filter _ hE, List.Pairwise.filter _ hE, ?_⟩
    intro a ha b hb hab
    simp only [List.mem_filter, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at ha hb
    have := hid a ha.1 b hb.1 hab
    subst this
    exact hb.2.1.1.2 ha.2.1
  · refine List.Pairwise.imp_of_mem ?_ hNp
    intro n1 n2 hn1 _ hne a ha b hb hab
    simp only [List.mem_append, List.mem_filter, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq,
      Bool.not_eq_true', List.contains_iff_mem] at ha hb
    have hab' : a = b := by
      rcases ha with ha | ha <;> rcases hb with hb | hb <;> exact hid a ha.1 b hb.1 hab
    subst hab'
    rcases ha with ⟨_, h1, _⟩ | ⟨_, ⟨⟨h1, h1'⟩, h1n⟩, _⟩ <;>
    rcases hb with ⟨_, h2, _⟩ | ⟨_, ⟨⟨h2, h2'⟩, h2n⟩, _⟩
    · exact hne (h1.symm.trans h2)
    · simp [h1 ▸ hn1] at h2n
    · simp [h2 ▸ ‹n2 ∈ N›] at h1n
    · exact hne (h1.symm.trans h2)

/-- Clearing a list of ids answers `none` for exactly those ids. -/
theorem refRun_clears (ids : List Nat) : ∀ (f : Nat → Option Slot) (i : Nat),
    refRun f (ids.map Op.clear) i = if i ∈ ids then none else f i := by
  induction ids with
  | nil => intro f i; simp [refRun]
  | cons x xs ih =>
    intro f i
    simp only [List.map_cons, refRun, List.foldl_cons] at ih ⊢
    rw [ih]
    simp only [Op.ref, List.mem_cons]
    by_cases h1 : i = x <;> by_cases h2 : i ∈ xs <;> simp [h1, h2]

/-- Detach-delete end to end on the index: after the collected ids are cleared,
every edge incident to a deleted node (and not explicit) reads as absent, and
every other edge reads as before. -/
theorem detach_delete_index (ix : Ix) (hwf : WF ix) (N X : List Nat) (E : List Edge) (i : Nat) :
    (run ix ((collect N X E).map (fun e => Op.clear e.1))).get i =
      if i ∈ (collect N X E).map Prod.fst then none else ix.get i := by
  have := (run_refines ((collect N X E).map (fun e => Op.clear e.1)) ix ix.get hwf (fun _ => rfl)
    (by intro op hop; obtain ⟨_, _, rfl⟩ := List.mem_map.1 hop; trivial)).2 i
  rw [this]
  have hm : (collect N X E).map (fun e => Op.clear e.1) = ((collect N X E).map Prod.fst).map Op.clear := by
    simp [List.map_map, Function.comp_def]
  rw [hm, refRun_clears]

/-! ## The boundary, and executable checks -/

/-- The only pair the spec has to exclude: in the `u64` tier the tombstone is
`(u64::MAX, u64::MAX)`, so an edge with both endpoints `u64::MAX` reads back as
deleted. Unreachable in practice (it needs a node id of 2^64 - 1), and
documented in `endpoint_index.rs` only for the narrow tiers. -/
theorem u64_max_pair_reads_as_absent :
    (({} : Ix).set 0 (2 ^ 64 - 1) (2 ^ 64 - 1)).get 0 = none := by decide

/-- One endpoint at `u64::MAX` is fine: a slot is empty only when both are. -/
theorem u64_max_one_side_is_live :
    (({} : Ix).set 0 (2 ^ 64 - 1) 7).get 0 = some (2 ^ 64 - 1, 7) := by decide

-- The Rust unit test `promotion_preserves_pairs_and_tombstones_at_every_tier`, replayed.
#guard
  let ix := (((({} : Ix).set 0 1 2).set 1 3 4).clear 1).set 2 300 400
  let ix := ix.set 3 70000 4
  let ix := ix.set 4 (2 ^ 25) 5
  let ix := ix.set 5 (2 ^ 33) (2 ^ 34 + 7)
  ix.get 0 == some (1, 2) && ix.get 1 == none && ix.get 2 == some (300, 400) &&
  ix.get 5 == some (2 ^ 33, 2 ^ 34 + 7)

-- Reuse a low id with wide endpoints after a delete: promotion, no stale answer.
#guard
  let ix := (List.range 10).foldl (fun ix e => ix.set e e (e + 1)) ({} : Ix)
  let ix := (ix.clear 3).clear 7
  let ix := ix.set 7 (2 ^ 40) 70000
  ix.get 3 == none && ix.get 7 == some (2 ^ 40, 70000) && ix.get 9 == some (9, 10) &&
  ix.w16.isNone && ix.len == 10

-- Values sitting on each sentinel go one tier up and read back live.
#guard [65535, 16777215, 4294967295].all fun v => (({} : Ix).set 0 v v).get 0 == some (v, v)

#print axioms run_refines
#print axioms restore_agrees
#print axioms detach_delete_index
#print axioms paged_at
#print axioms ensurePages_spec
-- PROOFS

end Falkor.EndpointIndex
