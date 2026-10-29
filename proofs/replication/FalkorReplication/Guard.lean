/-
# `src/divergence_guard.rs`, modelled function by function

Every Redis module API call the guard makes is a boundary, not logic: its
behaviour is taken as an *input* (the flags a `Context` reports, what
`INFO replication` answers, whether a `ctx.call` errored). The guard's own logic
— which of those inputs lead to "return", "exit(1)", or "schedule a resync" —
is then a total function here, and the theorems say exactly what it does.

| here | there |
| --- | --- |
| `Flags`, `isReplayed`   | `is_replayed` (`src/divergence_guard.rs:41`) — `REPLICATED \| LOADING` intersect |
| `onFailure`             | `on_failure` (`:53`) |
| `forceFullResync`       | `force_full_resync` (`:104`) |
| `masterAddress`         | `master_address` (`:142`) |
| `logPayload`            | `log_payload` (`:168`) |
| `clip`, `clipEnd`       | `clip` (`:192`), `LOG_LINE_BUDGET = 880` (`:208`) |

Boundaries (inputs, not axioms): `Context::get_flags`, `ctx.server_info(..).field`,
`ctx.call("REPLICAOF", ..)` success/failure, `EffectsPayload::describe` (the
list of lines it yields; it is modelled in proofs/effects_codec), `log_warning`
(recorded as an `Act.log`), `std::process::exit(1)` (`Act.exit`), and
`ctx.create_timer(0, ..)` (`Act.timer`).
-/
namespace Falkor.Guard

/-- The two `ContextFlags` bits the guard looks at. -/
structure Flags where
  replicated : Bool
  loading    : Bool
deriving DecidableEq, Repr

/-- `is_replayed` (`divergence_guard.rs:41`):
`flags.intersects(REPLICATED | LOADING)`. -/
def isReplayed (f : Flags) : Bool := f.replicated || f.loading

/-- Observable effects, in order. -/
inductive Act where
  | log   (s : String)
  | exit
  | timer (graph : String)
  | call  (args : List String)
deriving DecidableEq, Repr

/-- `log_payload` (`:168`): `None` logs nothing; otherwise one warning per
description line, each passed through `clip` (modelled abstractly as `c`, the
concrete `clip` is below). `lines` is `EffectsPayload::describe(buf)`. -/
def logPayload (c : String → String) (graph : String) (lines : Option (List String)) : List Act :=
  match lines with
  | none    => []
  | some ls => ls.map fun l => .log ("Diverged payload on '" ++ graph ++ "': " ++ c l)

/-- `on_failure` (`:53`). The `msg*` are the formatted warning texts; the model
keeps their order relative to the other effects, which is what matters. -/
def onFailure (c : String → String) (f : Flags) (graph msgLoad msgRepl : String)
    (lines : Option (List String)) : List Act :=
  if !isReplayed f then []
  else if f.loading then
    [.log msgLoad] ++ logPayload c graph lines ++ [.exit]
  else
    [.log msgRepl] ++ logPayload c graph lines ++ [.timer graph]

/-- `master_address` (`:142`): both fields must be present and non-empty. -/
def masterAddress (host port : Option String) : Option (String × String) :=
  match host, port with
  | some h, some p => if h.isEmpty || p.isEmpty then none else some (h, p)
  | _, _ => none

/-- `force_full_resync` (`:104`). `ok1` / `ok2`: whether each `REPLICAOF` call
succeeded (a Redis boundary). Every failure path ends in `exit`. -/
def forceFullResync (host port : Option String) (ok1 ok2 : Bool) : List Act :=
  match masterAddress host port with
  | none => [.log "no master", .exit]
  | some (h, p) =>
    if !ok1 then [.call ["REPLICAOF", "NO", "ONE"], .log "no one failed", .exit]
    else if !ok2 then
      [.call ["REPLICAOF", "NO", "ONE"], .call ["REPLICAOF", h, p], .log "reattach failed", .exit]
    else
      [.call ["REPLICAOF", "NO", "ONE"], .call ["REPLICAOF", h, p], .log "resync initiated"]

theorem getLast?_cons_append_single {α} (a x : α) (l : List α) :
    (a :: (l ++ [x])).getLast? = some x := by
  rw [← List.cons_append]; exact List.getLast?_concat

/-! ### Theorems: on_failure / is_replayed -/

theorem isReplayed_iff (f : Flags) : isReplayed f = true ↔ (f.replicated = true ∨ f.loading = true) := by
  simp [isReplayed]

/-- A client-sent command (neither flag) has no effect at all: no log, no exit,
no resync. This is the "any client can take a replica down" guard. -/
theorem onFailure_client_noop (c) (f : Flags) (g m1 m2 l) (h : isReplayed f = false) :
    onFailure c f g m1 m2 l = [] := by
  simp [onFailure, h]

/-- Under `LOADING` the guard always ends in `exit(1)`, and never schedules a resync. -/
theorem onFailure_loading_exits (c) (f : Flags) (g m1 m2 l) (h : f.loading = true) :
    (onFailure c f g m1 m2 l).getLast? = some .exit ∧
    ∀ g', Act.timer g' ∉ onFailure c f g m1 m2 l := by
  have hr : isReplayed f = true := by simp [isReplayed, h]
  refine ⟨?_, ?_⟩
  · simp only [onFailure, hr, h]
    simp only [Bool.not_true, Bool.false_eq_true, ite_false, ite_true]; exact List.getLast?_concat
  · intro g' hm
    simp [onFailure, hr, h, logPayload] at hm
    cases l <;> simp_all

/-- Replicated and not loading: log, then exactly one resync timer for this
graph as the final action, and never `exit`. -/
theorem onFailure_replicated_resyncs (c) (f : Flags) (g m1 m2 l)
    (hr : f.replicated = true) (hl : f.loading = false) :
    (onFailure c f g m1 m2 l).getLast? = some (.timer g) ∧ Act.exit ∉ onFailure c f g m1 m2 l := by
  have hr' : isReplayed f = true := by simp [isReplayed, hr]
  refine ⟨?_, ?_⟩
  · simp only [onFailure, hr', hl]
    simp only [Bool.not_true, Bool.false_eq_true, ite_false, ite_true]; exact List.getLast?_concat
  · intro hm
    simp [onFailure, hr', hl, logPayload] at hm
    cases l <;> simp_all

/-- The payload is logged only inside the gate. -/
theorem onFailure_logs_payload_only_when_replayed (c) (f : Flags) (g m1 m2 l) :
    onFailure c f g m1 m2 l ≠ [] → isReplayed f = true := by
  intro h; cases hf : isReplayed f <;> simp_all [onFailure]

/-! ### log_payload -/

theorem logPayload_none (c g) : logPayload c g none = [] := rfl

theorem logPayload_length (c g ls) : (logPayload c g (some ls)).length = ls.length := by
  simp [logPayload]

/-! ### master_address -/

theorem masterAddress_some (h p : Option String) (a b : String) :
    masterAddress h p = some (a, b) ↔ h = some a ∧ p = some b ∧ a ≠ "" ∧ b ≠ "" := by
  cases h <;> cases p <;> simp [masterAddress]
  rename_i x y
  by_cases hx : x.isEmpty <;> by_cases hy : y.isEmpty <;>
    simp_all [String.isEmpty_iff]
  rintro rfl rfl; exact ⟨hx, hy⟩

theorem masterAddress_none_missing (p : Option String) : masterAddress none p = none := by
  cases p <;> rfl

/-! ### force_full_resync -/

/-- Every failure path exits; the success path never does. -/
theorem forceFullResync_exit_iff (h p ok1 ok2) :
    Act.exit ∈ forceFullResync h p ok1 ok2 ↔ (masterAddress h p = none ∨ ok1 = false ∨ ok2 = false) := by
  unfold forceFullResync
  cases hm : masterAddress h p with
  | none => simp
  | some hp => obtain ⟨a, b⟩ := hp; cases ok1 <;> cases ok2 <;> simp

/-- When it proceeds, `REPLICAOF NO ONE` is always the first call — the one that
drops the cached replication id so the reconnect cannot be a partial resync. -/
theorem forceFullResync_no_one_first (h p ok1 ok2) (a b : String) (hm : masterAddress h p = some (a, b)) :
    (forceFullResync h p ok1 ok2).head? = some (.call ["REPLICAOF", "NO", "ONE"]) := by
  simp only [forceFullResync, hm]; cases ok1 <;> cases ok2 <;> rfl

/-- The reattach targets exactly the address `INFO replication` reported. -/
theorem forceFullResync_success (h p) (a b : String) (hm : masterAddress h p = some (a, b)) :
    forceFullResync h p true true =
      [.call ["REPLICAOF", "NO", "ONE"], .call ["REPLICAOF", a, b], .log "resync initiated"] := by
  simp [forceFullResync, hm]

/-! ### clip (`:192`)

`s.len()` is the UTF-8 byte length; `is_char_boundary(i)` holds exactly at the
byte offsets of char starts and at `len`. A string is modelled as its
`List Char`; `bytes cs = Σ utf8Size`. -/

def LOG_LINE_BUDGET : Nat := 880

def bytes (cs : List Char) : Nat := (cs.map Char.utf8Size).sum

/-- Byte offsets that are char boundaries: prefix sums, including `0` and `len`. -/
def isBoundary (cs : List Char) (i : Nat) : Bool :=
  (List.range (cs.length + 1)).any fun k => bytes (cs.take k) == i

/-- `(0..=B).rev().find(|&i| s.is_char_boundary(i)).unwrap_or_default()` —
the largest boundary `≤ B`. `findRev B` scans `B, B-1, …, 0`. -/
def findRev (cs : List Char) : Nat → Nat
  | 0     => 0     -- 0 is always a boundary; `unwrap_or_default` also gives 0
  | i + 1 => if isBoundary cs (i + 1) then i + 1 else findRev cs i

/-- `clip`: `none` = returned unchanged (`s.len() <= B`); `some (end, cut)` =
`&s[..end]` kept and `cut = s.len() - end` bytes reported cut. -/
def clip (cs : List Char) : Option (Nat × Nat) :=
  if bytes cs ≤ LOG_LINE_BUDGET then none
  else let e := findRev cs LOG_LINE_BUDGET; some (e, bytes cs - e)

theorem isBoundary_zero (cs : List Char) : isBoundary cs 0 = true := by
  simp only [isBoundary, List.any_eq_true, List.mem_range, beq_iff_eq]
  exact ⟨0, by omega, by simp [bytes]⟩

theorem findRev_le (cs : List Char) : ∀ i, findRev cs i ≤ i
  | 0 => by simp [findRev]
  | i + 1 => by
    simp only [findRev]; split
    · omega
    · have := findRev_le cs i; omega

theorem findRev_boundary (cs : List Char) : ∀ i, isBoundary cs (findRev cs i) = true
  | 0 => by simp [findRev, isBoundary_zero]
  | i + 1 => by
    simp only [findRev]; split
    · assumption
    · exact findRev_boundary cs i

/-- `findRev` is the *largest* boundary not above `i`. -/
theorem findRev_max (cs : List Char) : ∀ i j, j ≤ i → isBoundary cs j = true → j ≤ findRev cs i
  | 0, j, hj, _ => by simp [findRev]; omega
  | i + 1, j, hj, hb => by
    simp only [findRev]; split
    · exact hj
    · rename_i hn
      rcases Nat.lt_or_ge j (i + 1) with h | h
      · exact findRev_max cs i j (by omega) hb
      · have : j = i + 1 := by omega
        subst this; simp_all

theorem take_bytes_le (cs : List Char) (k : Nat) : bytes (cs.take k) ≤ bytes cs := by
  induction cs generalizing k with
  | nil => simp [bytes]
  | cons c t ih => cases k <;> simp_all [bytes]

/-- `clip` is correct: a line within budget is untouched; otherwise the kept
prefix ends on a char boundary, fits the budget, is the longest such prefix,
and the reported cut is exactly the rest (so kept + cut = original length). -/
theorem clip_spec (cs : List Char) :
    (bytes cs ≤ LOG_LINE_BUDGET → clip cs = none) ∧
    (bytes cs > LOG_LINE_BUDGET → ∃ e, clip cs = some (e, bytes cs - e) ∧
        isBoundary cs e = true ∧ e ≤ LOG_LINE_BUDGET ∧ e ≤ bytes cs ∧
        (∀ j, j ≤ LOG_LINE_BUDGET → isBoundary cs j = true → j ≤ e)) := by
  refine ⟨fun h => by simp [clip, h], fun h => ?_⟩
  refine ⟨findRev cs LOG_LINE_BUDGET, by simp [clip]; omega, findRev_boundary _ _,
    findRev_le _ _, by have := findRev_le cs LOG_LINE_BUDGET; omega,
    fun j hj hb => findRev_max _ _ _ hj hb⟩

/-- The kept prefix is a prefix of whole characters (so slicing never panics). -/
theorem clip_prefix_chars (cs : List Char) (e c : Nat) (h : clip cs = some (e, c)) :
    ∃ k ≤ cs.length, bytes (cs.take k) = e := by
  simp only [clip] at h; split at h
  · cases h
  · simp only [Option.some.injEq, Prod.mk.injEq] at h
    have hb := findRev_boundary cs LOG_LINE_BUDGET
    rw [h.1] at hb
    simp only [isBoundary, List.any_eq_true, List.mem_range, beq_iff_eq] at hb
    obtain ⟨k, hk, he⟩ := hb; exact ⟨k, by omega, he⟩

-- Concrete checks against the Rust behaviour.
#guard clip ("é".toList ++ List.replicate 880 'a') == some (880, 2)
#guard clip (List.replicate 881 'a') == some (880, 1)
#guard clip (List.replicate 880 'a') == none
-- 'é' is 2 bytes; 'é' ++ 879 'a' = 881 bytes: boundaries 0,2,3,…,881 → 880 kept, 1 cut
#guard clip ("é".toList ++ List.replicate 879 'a') == some (880, 1)
-- 440 two-byte chars then one more: 882 bytes, boundaries even → end 880
#guard clip (List.replicate 441 'é') == some (880, 2)
-- a 3-byte char straddling 880: 879 'a' then '€' (3 bytes) = 882 → end 879
#guard clip (List.replicate 879 'a' ++ ['€']) == some (879, 3)

end Falkor.Guard
