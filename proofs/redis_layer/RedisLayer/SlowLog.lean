/-!
# `src/slow_log.rs` — per-graph slow log

| here | there (`origin/main`) |
| --- | --- |
| `boundary`, `slice`   | `str::is_char_boundary`, `str` indexing `&s[..n]` (panics off a char boundary) |
| `floorCB`             | `str::floor_char_boundary` (std contract: the greatest `x ≤ index` with `is_char_boundary(x)`; `len` when `index ≥ len`) |
| `prefix`              | `prefix` `:32` (`&s[..s.floor_char_boundary(STR_MAX_LEN)]`) |
| `truncate`            | `truncate` `:38` |
| `hashInput`           | `entry_hash` `:51` (hashes `prefix(query)`, `:58`) |
| `E`, `add`, `reset`   | `SlowLog::new` `:81`, `add` `:97`, `reset` `:171` |
| `replyLen`            | `SlowLog::reply` `:181` (5-element rows) |
| `MinCache`, `SlowestKept`, `add` | #3442 (fixed by #3456): the `min_latency` cache (`:77`, `:114`, `:162-169`); main's update arm `:121-139` falls through to the recompute (b84473a7f returned early there) |

**Fixed crash** (W4-redis-1, fixed by #3234 `3660a5fc3`): `entry_hash` and `truncate`
used to cut the query at *byte* 2048, so a query longer than 2048 bytes with a multi-byte
character across byte 2048 panicked (counterexamples removed; in 17ccc5412). Both now
go through `prefix`, which backs up to `floor_char_boundary(2048)`:
`hashInput_total`/`truncate_total` (never panics, for every byte string),
`prefix_le`/`prefix_isPrefix`/`prefix_boundary`/`prefix_longest` (the cut is the longest
char-boundary prefix of at most 2048 bytes), `prefix_short` (≤ 2048 bytes: untouched).
-/
namespace RedisLayer.SlowLog

abbrev Bytes := List Nat

/-- UTF-8 continuation byte. -/
def cont (b : Nat) : Bool := 0x80 ≤ b && b < 0xC0

/-- `str::is_char_boundary(n)`. -/
def boundary (s : Bytes) (n : Nat) : Bool :=
  n = 0 || s.length ≤ n || !(cont (s.getD n 0))

/-- `&s[..n]`: `none` is the panic. -/
def slice (s : Bytes) (n : Nat) : Option Bytes :=
  if boundary s n then some (s.take n) else none

def STR_MAX_LEN := 2048

/-- Greatest `x ≤ n` with `boundary s x` (0 always is one). -/
def floorB (s : Bytes) : Nat → Nat
  | 0 => 0
  | n + 1 => if boundary s (n + 1) then n + 1 else floorB s n

/-- `str::floor_char_boundary(index)`. -/
def floorCB (s : Bytes) (n : Nat) : Nat :=
  if n ≥ s.length then s.length else floorB s n

/-- `prefix` (`:32`): `&s[..s.floor_char_boundary(STR_MAX_LEN)]`. -/
def strPrefix (s : Bytes) : Option Bytes := slice s (floorCB s STR_MAX_LEN)

/-- `entry_hash`'s hashed slice (`prefix(query)`). -/
def hashInput (q : Bytes) : Option Bytes := strPrefix q

/-- `truncate`. -/
def truncate (s : Bytes) : Option Bytes :=
  if s.length > STR_MAX_LEN then (strPrefix s).map (· ++ [46, 46, 46]) else some s

theorem floorB_le (s : Bytes) : ∀ n, floorB s n ≤ n
  | 0 => by simp [floorB]
  | n + 1 => by
    unfold floorB; split
    · omega
    · have := floorB_le s n; omega

theorem floorB_boundary (s : Bytes) : ∀ n, boundary s (floorB s n) = true
  | 0 => by simp [floorB, boundary]
  | n + 1 => by
    unfold floorB; split
    · assumption
    · exact floorB_boundary s n

/-- No boundary strictly between `floorB s n` and `n`. -/
theorem floorB_greatest (s : Bytes) : ∀ n m, floorB s n < m → m ≤ n → boundary s m = false
  | 0, m, h1, h2 => by simp [floorB] at h1; omega
  | n + 1, m, h1, h2 => by
    unfold floorB at h1; split at h1
    · omega
    · rename_i hb
      by_cases hm : m = n + 1
      · subst hm; simpa using hb
      · exact floorB_greatest s n m h1 (by omega)

theorem floorB_succ (s : Bytes) (n : Nat) :
    floorB s (n + 1) = if boundary s (n + 1) then n + 1 else floorB s n := rfl

theorem floorCB_boundary (s : Bytes) (n : Nat) : boundary s (floorCB s n) = true := by
  unfold floorCB; split
  · simp [boundary]
  · exact floorB_boundary s n

theorem floorCB_le (s : Bytes) (n : Nat) : floorCB s n ≤ n ∧ floorCB s n ≤ s.length := by
  unfold floorCB; split
  · omega
  · have := floorB_le s n; omega

/-- The slice in `prefix` is always on a char boundary: `prefix` never panics. -/
theorem prefix_eq (s : Bytes) : strPrefix s = some (s.take (floorCB s STR_MAX_LEN)) := by
  simp [strPrefix, slice, floorCB_boundary]

theorem prefix_total (s : Bytes) : (strPrefix s).isSome := by simp [prefix_eq]

/-- `entry_hash` never panics, for any query text. -/
theorem hashInput_total (q : Bytes) : (hashInput q).isSome := prefix_total q

/-- `truncate` never panics. -/
theorem truncate_total (s : Bytes) : (truncate s).isSome := by
  unfold truncate; split
  · simp [prefix_eq]
  · rfl

/-- The prefix is at most 2048 bytes … -/
theorem prefix_le (s p : Bytes) (h : strPrefix s = some p) : p.length ≤ STR_MAX_LEN := by
  rw [prefix_eq] at h; cases h
  have := floorCB_le s STR_MAX_LEN; simp; omega

/-- … is a prefix of `s` … -/
theorem prefix_isPrefix (s p : Bytes) (h : strPrefix s = some p) : p <+: s := by
  rw [prefix_eq] at h; cases h; exact List.take_prefix _ _

/-- … ends on a char boundary … -/
theorem prefix_boundary (s p : Bytes) (h : strPrefix s = some p) : boundary s p.length = true := by
  rw [prefix_eq] at h; cases h
  have := floorCB_le s STR_MAX_LEN
  rw [List.length_take, Nat.min_eq_left this.2]; exact floorCB_boundary s _

/-- … and is the longest such: no longer cut within the limit is a char boundary. -/
theorem prefix_longest (s p : Bytes) (h : strPrefix s = some p) (m : Nat)
    (h1 : p.length < m) (h2 : m ≤ STR_MAX_LEN) (h3 : m ≤ s.length) : boundary s m = false := by
  rw [prefix_eq] at h; cases h
  have hle := floorCB_le s STR_MAX_LEN
  rw [List.length_take, Nat.min_eq_left hle.2] at h1
  unfold floorCB at h1; split at h1
  · omega
  · exact floorB_greatest s _ m h1 h2

/-- Queries of at most 2048 bytes are hashed / kept whole. -/
theorem prefix_short (q : Bytes) (h : q.length ≤ STR_MAX_LEN) : strPrefix q = some q := by
  rw [prefix_eq]; unfold floorCB; rw [if_pos h]; simp

theorem hashInput_short (q : Bytes) (h : q.length ≤ STR_MAX_LEN) : hashInput q = some q :=
  prefix_short q h

theorem truncate_short (s : Bytes) (h : s.length ≤ STR_MAX_LEN) : truncate s = some s := by
  simp [truncate]; omega

/-- ASCII text is cut at exactly 2048 bytes, as C does. -/
theorem prefix_ascii (q : Bytes) (h : ∀ b ∈ q, b < 0x80) (hl : STR_MAX_LEN < q.length) :
    strPrefix q = some (q.take STR_MAX_LEN) := by
  have hl' : 2048 < q.length := hl
  rw [prefix_eq]; unfold floorCB; rw [if_neg (by omega)]
  have hg : q.getD 2048 0 = q[2048]'hl' := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hl']; rfl
  have hb := h _ (List.getElem_mem hl')
  have : boundary q 2048 = true := by
    simp only [boundary, cont, hg]; simp; omega
  rw [show STR_MAX_LEN = 2047 + 1 from rfl, floorB_succ, if_pos this]

/-- 2047 ASCII bytes then `é` (`C3 A9`): byte 2048 is a continuation byte. -/
def bad : Bytes := List.replicate 2047 97 ++ [0xC3, 0xA9]

theorem getD_bad (p t : Bytes) (hp : p.length = 2047) : (p ++ 0xC3 :: 0xA9 :: t).getD 2048 0 = 0xA9 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), hp]; rfl

/-- The old crash input now hashes its first 2047 bytes (the `é` is left out whole). -/
theorem hashInput_bad (p t : Bytes) (hp : p.length = 2047) :
    hashInput (p ++ 0xC3 :: 0xA9 :: t) = some p := by
  unfold hashInput; rw [prefix_eq]; unfold floorCB
  rw [if_neg (by simp [STR_MAX_LEN, hp]; omega)]
  have hb : boundary (p ++ 0xC3 :: 0xA9 :: t) (2047 + 1) = false := by
    unfold boundary; rw [show 2047 + 1 = 2048 from rfl, getD_bad p t hp]; simp [cont, hp]; omega
  have hb' : boundary (p ++ 0xC3 :: 0xA9 :: t) (2046 + 1) = true := by
    unfold boundary
    rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), hp]
    simp [cont]
  rw [show STR_MAX_LEN = 2047 + 1 from rfl, floorB_succ, if_neg (by simp [hb]),
    show (2047 : Nat) = 2046 + 1 from rfl, floorB_succ, if_pos hb']
  simp [hp]

/-! ## Historical: before #3234 (`3660a5fc3`)

The old code sliced at byte 2048 directly (`query[..min(len, 2048)]` and
`s[..STR_MAX_LEN]`). Kept as the record of W4-redis-1. -/

/-! ## Entries -/

def MAX_ENTRIES := 10
def MIN_LATENCY := 10

structure E where
  hash : Nat
  latency : Nat
  deriving DecidableEq, Repr

structure SL where
  entries : List E
  minLat : Nat

def SL.new : SL := ⟨[], 0⟩
def reset (_ : SL) : SL := ⟨[], 0⟩

def minOf (l : List E) : Nat := l.foldl (fun m e => min m e.latency) (2^64)

/-- Index of the lowest latency (`min_by` keeps the first minimum). -/
def argmin : List E → Nat
  | [] => 0
  | e :: es => match es with
    | [] => 0
    | _ => let j := argmin es; if (es.getD j e).latency < e.latency then j + 1 else 0

/-- `SlowLog::add` on origin/main (5e475e6c9, #3456; `slow_log.rs:97-175`): an update that
does not raise the latency returns unchanged; one that does raises it and falls through to
the `min_latency` recompute, like every other arm. -/
def add (s : SL) (h lat : Nat) : SL :=
  if lat < MIN_LATENCY then s
  else if s.entries.length ≥ MAX_ENTRIES ∧ lat ≤ s.minLat then s
  else match s.entries.find? (·.hash = h) with
    | some e =>
      if lat ≤ e.latency then s
      else
        let es := s.entries.map fun e =>
          if e.hash = h ∧ lat > e.latency then { e with latency := lat } else e
        ⟨es, minOf es⟩
    | none =>
      let es := if s.entries.length < MAX_ENTRIES then s.entries ++ [⟨h, lat⟩]
        else s.entries.set (argmin s.entries) ⟨h, lat⟩
      ⟨es, minOf es⟩

/-- The log never holds more than `MAX_ENTRIES`. -/
theorem add_bounded (s : SL) (h lat : Nat) (hs : s.entries.length ≤ MAX_ENTRIES) :
    (add s h lat).entries.length ≤ MAX_ENTRIES := by
  unfold add
  split; · exact hs
  split; · exact hs
  split
  · split
    · exact hs
    · simpa using hs
  · simp only; split
    · simp; omega
    · simpa using hs

theorem nodup_map_set {α β} (f : α → β) (l : List α) (i : Nat) (y : α)
    (hl : (l.map f).Nodup) (hy : ∀ x ∈ l, f x ≠ f y) : ((l.set i y).map f).Nodup := by
  induction l generalizing i with
  | nil => simp
  | cons a as ih =>
    have hl' : (as.map f).Nodup := (List.nodup_cons.1 (by simpa using hl)).2
    have ha : f a ∉ as.map f := (List.nodup_cons.1 (by simpa using hl)).1
    cases i with
    | zero =>
      simp only [List.set_cons_zero, List.map_cons]
      refine List.nodup_cons.2 ⟨?_, hl'⟩
      intro hm; obtain ⟨x, hx, e⟩ := List.mem_map.1 hm
      exact hy x (by simp [hx]) e
    | succ i =>
      simp only [List.set_cons_succ, List.map_cons]
      refine List.nodup_cons.2 ⟨?_, ih i hl' (fun x hx => hy x (by simp [hx]))⟩
      intro hm; obtain ⟨x, hx, e⟩ := List.mem_map.1 hm
      rcases List.mem_or_eq_of_mem_set hx with hx | rfl
      · exact ha (List.mem_map.2 ⟨x, hx, e⟩)
      · exact hy a (by simp) e.symm

/-- One entry per (command, query) hash. -/
theorem add_nodup (s : SL) (h lat : Nat) (hs : (s.entries.map (·.hash)).Nodup)
    (hs2 : s.entries.length ≤ MAX_ENTRIES) :
    ((add s h lat).entries.map (·.hash)).Nodup := by
  unfold add
  split; · exact hs
  split; · exact hs
  split
  · rename_i e he
    split
    · exact hs
    have : (s.entries.map fun e => if e.hash = h ∧ lat > e.latency then { e with latency := lat } else e).map (·.hash)
        = s.entries.map (·.hash) := by
      rw [List.map_map]; apply List.map_congr_left; intro x _; simp; split <;> rfl
    simpa [this] using hs
  · rename_i hn
    have hnot : h ∉ s.entries.map (·.hash) := by
      intro hm
      obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hm
      have := List.find?_eq_none.1 hn x hx
      simp at this
    simp only; split
    · simp [List.nodup_append, hs, hnot]
      intro x hx e; exact hnot (List.mem_map.2 ⟨x, hx, e⟩)
    · exact nodup_map_set _ _ _ _ hs (fun x hx e => hnot (List.mem_map.2 ⟨x, hx, e⟩))

/-- Below the threshold nothing is logged. -/
theorem add_fast (s : SL) (h lat : Nat) (hl : lat < MIN_LATENCY) : add s h lat = s := by
  simp [add, hl]

theorem nodup_map_inj_E (l : List E) (hn : (l.map (·.hash)).Nodup) (a b : E) (ha : a ∈ l) (hb : b ∈ l)
    (h : a.hash = b.hash) : a = b := by
  induction l with
  | nil => simp at ha
  | cons x xs ih =>
    rw [List.map_cons] at hn
    have hn' := List.nodup_cons.1 hn
    rcases List.mem_cons.1 ha with rfl | ha' <;> rcases List.mem_cons.1 hb with rfl | hb'
    · rfl
    · exact absurd (List.mem_map.2 ⟨b, hb', h.symm⟩) hn'.1
    · exact absurd (List.mem_map.2 ⟨a, ha', h⟩) hn'.1
    · exact ih hn'.2 ha' hb'

/-- A repeat keeps the worst latency (the log holds one entry per hash, `add_nodup`). -/
theorem add_repeat_max (s : SL) (h lat : Nat) (e : E) (he : e ∈ s.entries) (heh : e.hash = h)
    (hnd : (s.entries.map (·.hash)).Nodup)
    (hl : MIN_LATENCY ≤ lat) (hc : ¬(s.entries.length ≥ MAX_ENTRIES ∧ lat ≤ s.minLat)) :
    ∃ e' ∈ (add s h lat).entries, e'.hash = h ∧ e'.latency = max e.latency lat := by
  have hf : (s.entries.find? (·.hash = h)).isSome := by
    rw [List.find?_isSome]; exact ⟨e, he, by simp [heh]⟩
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hf
  have hxm : x ∈ s.entries := List.mem_of_find?_eq_some hx
  have hxh : x.hash = h := by simpa using List.find?_some hx
  have hxe : x = e := nodup_map_inj_E _ hnd x e hxm he (by rw [hxh, heh])
  subst hxe
  unfold add
  rw [if_neg (by omega), if_neg hc, hx]
  simp only
  split
  · rename_i hle
    exact ⟨x, he, heh, by omega⟩
  · rename_i hle
    refine ⟨{ x with latency := lat }, List.mem_map.2 ⟨x, he, ?_⟩, heh, ?_⟩
    · simp [heh]; omega
    · show lat = max x.latency lat; omega

theorem new_empty : SL.new.entries = [] ∧ SL.new.minLat = 0 := ⟨rfl, rfl⟩

theorem reset_empty (s : SL) : (reset s).entries = [] := rfl

/-- `reply`: one 5-element row per entry. -/
def replyLen (s : SL) : List Nat := s.entries.map fun _ => 5
theorem replyLen_len (s : SL) : (replyLen s).length = s.entries.length := by simp [replyLen]

/-! ## Gap closed: #3442 (fixed by #3456) — the update arm left `min_latency` stale

On b84473a7f `SlowLog::add`'s existing-entry branch (`src/slow_log.rs:120-132`) raised the
entry's latency and `return`ed before the recompute at `:172-177`, leaving `minLat` alone. Once the cached minimum was below the real
one, the pre-check at `:114` admitted a query faster than every retained entry and the
replace path evicted the real fastest entry. C (`SlowLog_Add`) sends the update path through
`cleanup`, which recomputes the minimum. #3456 makes the branch fall through to the recompute
(`:120-139`, `:178`); `add` above is that main code. The refutations were removed (in 17ccc5412); `add_minCache` and `fixed3442_trace` prove the fix on main. -/

/-- Cache invariant: a non-empty log caches the minimum of its latencies. -/
def MinCache (s : SL) : Prop := s.entries ≠ [] → s.minLat = minOf s.entries

instance (s : SL) : Decidable (MinCache s) := by unfold MinCache; infer_instance

/-- Replay a trace of `(hash, latency_ms)` additions through main's `add`. -/
def run (s : SL) (tr : List (Nat × Nat)) : SL := tr.foldl (fun s p => add s p.1 p.2) s

/-- Worst latency the trace logged for hash `h` (only latencies ≥ `MIN_LATENCY` count). -/
def best (tr : List (Nat × Nat)) (h : Nat) : Nat :=
  tr.foldl (fun m p => if p.1 = h ∧ MIN_LATENCY ≤ p.2 then max m p.2 else m) 0

/-- Spec (`SlowestKept`, what C keeps): a query the trace logged that is no longer in the
log was no slower than any query still in it. -/
def slowestKeptB (tr : List (Nat × Nat)) (s : SL) : Bool :=
  tr.all fun p => !(MIN_LATENCY ≤ p.2) || s.entries.any (·.hash = p.1) ||
    s.entries.all fun e => decide (best tr p.1 ≤ e.latency)
def SlowestKept (tr : List (Nat × Nat)) (s : SL) : Prop := slowestKeptB tr s = true

instance (tr : List (Nat × Nat)) (s : SL) : Decidable (SlowestKept tr s) := by
  unfold SlowestKept; infer_instance

/-- The issue's trace: `Q1..Q10` at 20..110 ms, `Q1` again at 200 ms, then `Q11` at 25 ms. -/
def trace3442 : List (Nat × Nat) :=
  (List.range 10).map (fun i => (i + 1, 20 + 10 * i)) ++ [(1, 200), (11, 25)]

/-- Main: after the same ten entries and the update of `Q1`, the cache is the real minimum (30). -/
theorem fixed3442_update_keeps_min_cache :
    let s := run SL.new ((List.range 10).map (fun i => (i + 1, 20 + 10 * i)))
    MinCache s ∧ MinCache (add s 1 200) ∧ (add s 1 200).minLat = 30 := by
  decide

/-- **The fix**: main's `add` preserves the cache invariant, for every state and input. -/
theorem add_minCache (s : SL) (h lat : Nat) (hs : MinCache s) : MinCache (add s h lat) := by
  unfold add; split; · exact hs
  split; · exact hs
  split
  · split
    · exact hs
    · exact fun _ => rfl
  · exact fun _ => rfl

/-- With the fix the issue's trace keeps `Q2` and rejects `Q11`, as C does. -/
theorem fixed3442_trace :
    let s := run SL.new trace3442
    s.entries.any (·.hash = 11) = false ∧ s.entries.any (·.hash = 2) = true ∧
      SlowestKept trace3442 s := by
  decide

end RedisLayer.SlowLog
