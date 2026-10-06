/-!
# `src/slow_log.rs` — per-graph slow log

| here | there (`origin/main`) |
| --- | --- |
| `boundary`, `slice`   | `str` indexing `&s[..n]` (panics off a char boundary) |
| `truncate`            | `truncate` `:29` |
| `hashInput`           | `entry_hash` `:41` (`query[..min(len, 2048)]`, `:48-49`) |
| `E`, `add`, `reset`   | `SlowLog::new` `:72`, `add` `:88`, `reset` `:172` |
| `replyLen`            | `SlowLog::reply` `:182` (5-element rows) |

**Confirmed crash** (`entry_hash_panics`): `entry_hash` and `truncate` cut the query at
*byte* 2048. For a query text longer than 2048 bytes whose byte 2048 falls inside a
multi-byte character, `query[..2048]` panics and the panic hook exits the server. Any
query slower than 10 ms reaches it: `slowlog_utf8.py` (scratch) sent
`UNWIND range(1,3000000) AS x WITH count(x) AS c RETURN c, 'aaaa…é' AS s` (2047 bytes
before `é`): Rust died (`FalkorDB panic: panicked at src/slow_log.rs:49:10`), C answered
`c=3000000`. (`telemetry::truncate` counts chars and is safe.) Fix: back up to
`floor_char_boundary(2048)` in both places.
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

/-- `entry_hash`'s hashed slice. -/
def hashInput (q : Bytes) : Option Bytes := slice q (min q.length STR_MAX_LEN)

/-- `truncate`. -/
def truncate (s : Bytes) : Option Bytes :=
  if s.length > STR_MAX_LEN then (slice s STR_MAX_LEN).map (· ++ [46, 46, 46]) else some s

/-- 2047 ASCII bytes then `é` (`C3 A9`): byte 2048 is a continuation byte. -/
def bad : Bytes := List.replicate 2047 97 ++ [0xC3, 0xA9]

theorem getD_bad (p t : Bytes) (hp : p.length = 2047) : (p ++ 0xC3 :: 0xA9 :: t).getD 2048 0 = 0xA9 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_append_right (by omega), hp]; rfl

theorem entry_hash_panics (p t : Bytes) (hp : p.length = 2047) :
    hashInput (p ++ 0xC3 :: 0xA9 :: t) = none := by
  have hm : min (p ++ 0xC3 :: 0xA9 :: t).length STR_MAX_LEN = 2048 := by
    simp [STR_MAX_LEN, hp]; omega
  unfold hashInput slice boundary
  rw [hm, getD_bad p t hp]
  simp [cont, hp]; omega

theorem truncate_panics (p t : Bytes) (hp : p.length = 2047) :
    truncate (p ++ 0xC3 :: 0xA9 :: t) = none := by
  unfold truncate slice boundary
  rw [if_pos (by simp [STR_MAX_LEN, hp]; omega)]
  show (if _ then _ else none).map _ = none
  rw [show STR_MAX_LEN = 2048 from rfl, getD_bad p t hp]
  simp [cont, hp]; omega

/-- ASCII queries (or any query ≤ 2048 bytes) are safe. -/
theorem hashInput_short (q : Bytes) (h : q.length ≤ STR_MAX_LEN) : hashInput q = some q := by
  simp [hashInput, slice, boundary, Nat.min_eq_left h]

theorem hashInput_ascii (q : Bytes) (h : ∀ b ∈ q, b < 0x80) : (hashInput q).isSome := by
  unfold hashInput slice boundary
  by_cases hl : q.length ≤ STR_MAX_LEN
  · simp [Nat.min_eq_left hl]
  · have hm : min q.length STR_MAX_LEN = STR_MAX_LEN := by omega
    rw [hm]
    have hlt : STR_MAX_LEN < q.length := by omega
    have hg : q.getD STR_MAX_LEN 0 = q[STR_MAX_LEN] := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt]; rfl
    have hb := h _ (List.getElem_mem hlt)
    simp only [cont, hg]
    simp; omega

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
