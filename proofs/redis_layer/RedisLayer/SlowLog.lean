/-!
# `src/slow_log.rs` — per-graph slow log

| here | there (`origin/main`) |
| --- | --- |
| `boundary`, `slice`   | `str::is_char_boundary`, `str` indexing `&s[..n]` (panics off a char boundary) |
| `floorCB`             | `str::floor_char_boundary` (std contract: the greatest `x ≤ index` with `is_char_boundary(x)`; `len` when `index ≥ len`) |
| `prefix`              | `prefix` `:32` (`&s[..s.floor_char_boundary(STR_MAX_LEN)]`) |
| `truncate`            | `truncate` `:38` |
| `hashInput`           | `entry_hash` `:51` (hashes `prefix(query)`, `:58`) |
| `E`, `add`, `reset`   | `SlowLog::new` `:81`, `add` `:97`, `reset` `:181` |
| `replyLen`            | `SlowLog::reply` `:191` (5-element rows) |

**Fixed crash** (W4-redis-1, fixed by #3234 `3660a5fc3`): `entry_hash` and `truncate`
used to cut the query at *byte* 2048, so a query longer than 2048 bytes with a multi-byte
character across byte 2048 panicked (`pre3234_entry_hash_panics`, `pre3234_truncate_panics`,
kept as historical notes on the old code `pre3234_hashInput`/`pre3234_truncate`). Both now
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

def pre3234_hashInput (q : Bytes) : Option Bytes := slice q (min q.length STR_MAX_LEN)

def pre3234_truncate (s : Bytes) : Option Bytes :=
  if s.length > STR_MAX_LEN then (slice s STR_MAX_LEN).map (· ++ [46, 46, 46]) else some s

theorem pre3234_entry_hash_panics (p t : Bytes) (hp : p.length = 2047) :
    pre3234_hashInput (p ++ 0xC3 :: 0xA9 :: t) = none := by
  have hm : min (p ++ 0xC3 :: 0xA9 :: t).length STR_MAX_LEN = 2048 := by
    simp [STR_MAX_LEN, hp]; omega
  unfold pre3234_hashInput slice boundary
  rw [hm, getD_bad p t hp]
  simp [cont, hp]; omega

theorem pre3234_truncate_panics (p t : Bytes) (hp : p.length = 2047) :
    pre3234_truncate (p ++ 0xC3 :: 0xA9 :: t) = none := by
  unfold pre3234_truncate slice boundary
  rw [if_pos (by simp [STR_MAX_LEN, hp]; omega)]
  show (if _ then _ else none).map _ = none
  rw [show STR_MAX_LEN = 2048 from rfl, getD_bad p t hp]
  simp [cont, hp]; omega

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

/-- `SlowLog::add` with the hash of the (possibly panicking) slice already computed. -/
def add (s : SL) (h lat : Nat) : SL :=
  if lat < MIN_LATENCY then s
  else if s.entries.length ≥ MAX_ENTRIES ∧ lat ≤ s.minLat then s
  else match s.entries.find? (·.hash = h) with
    | some _ => { s with entries := s.entries.map fun e =>
        if e.hash = h ∧ lat > e.latency then { e with latency := lat } else e }
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

/-- A repeat keeps the worst latency. -/
theorem add_repeat_max (s : SL) (h lat : Nat) (e : E) (he : e ∈ s.entries) (heh : e.hash = h)
    (hl : MIN_LATENCY ≤ lat) (hc : ¬(s.entries.length ≥ MAX_ENTRIES ∧ lat ≤ s.minLat)) :
    ∃ e' ∈ (add s h lat).entries, e'.hash = h ∧ e'.latency = max e.latency lat := by
  have hf : (s.entries.find? (·.hash = h)).isSome := by
    rw [List.find?_isSome]; exact ⟨e, he, by simp [heh]⟩
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 hf
  unfold add
  rw [if_neg (by omega), if_neg hc, hx]
  refine ⟨if e.hash = h ∧ lat > e.latency then { e with latency := lat } else e,
    List.mem_map.2 ⟨e, he, rfl⟩, by split <;> simp [heh], ?_⟩
  split
  · rename_i hh; show lat = max e.latency lat; omega
  · rename_i hh
    have : ¬ lat > e.latency := fun hg => hh ⟨heh, hg⟩
    show e.latency = max e.latency lat; omega

theorem new_empty : SL.new.entries = [] ∧ SL.new.minLat = 0 := ⟨rfl, rfl⟩

theorem reset_empty (s : SL) : (reset s).entries = [] := rfl

/-- `reply`: one 5-element row per entry. -/
def replyLen (s : SL) : List Nat := s.entries.map fun _ => 5
theorem replyLen_len (s : SL) : (replyLen s).length = s.entries.length := by simp [replyLen]

end RedisLayer.SlowLog
