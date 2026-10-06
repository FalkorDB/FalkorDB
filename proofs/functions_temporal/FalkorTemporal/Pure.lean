import FalkorTemporal.Parse
/-!
# Temporal constructors: `*_pure`, `*_struct_pure`, the clock `*_fn`s and `register`
(functions/temporal.rs:488-823)

`construct_duration_secs` (PROVEN in Duration.lean as `constructDuration`) and
`date_from_components` (Constructors.lean) are parameters. `unreachable!()` is the result
`Res.unreachable`; the theorems show it is dead for every argument the declared type
admits. The clock is an argument (`now`, seconds; `nowMs`, milliseconds).
-/
namespace FalkorTemporal.Fns

inductive Res (α : Type) where
  | ok (a : α) | err (s : String) | unreachable

def ofExcept {α β : Type} (f : α → β) : Except String α → Res β
  | .ok a => .ok (f a)
  | .error e => .err e

variable {F : Type} (N : Num F) (C : Chrono) (dfc : DFC)

/-- `construct_duration_secs` signature (PROVEN as `constructDuration`). -/
abbrev CDS := Int → Int → Int → Int → Int → Int → Int → Except String Int

/-- `date.and_hms_opt(0,0,0).unwrap().and_utc().timestamp()`: midnight of day `d`
(`unwrap` cannot fail: 00:00:00 is always valid). -/
def midnight (d : Int) : Int := d * 86400

/-- `date_pure` (temporal.rs:488). -/
def datePure : List (TV F) → Res (TV F)
  | .map m :: _ => ofExcept (fun d => .date (midnight d)) (dateFromMap N dfc m)
  | .str s :: _ => ofExcept (fun d => .date (midnight d)) (parseDate C s)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `localtime_pure` (temporal.rs:505): the time on 1970-01-01, i.e. its second of day. -/
def localtimePure : List (TV F) → Res (TV F)
  | .map m :: _ => ofExcept (fun (t : Nat) => TV.time (t : Int)) (timeFromMap N m)
  | .str s :: _ => ofExcept (fun (t : Nat) => TV.time (t : Int)) (parseTime C s)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `localdatetime_pure` (temporal.rs:524). -/
def localdatetimePure : List (TV F) → Res (TV F)
  | .map m :: _ =>
    match dateFromMap N dfc m with
    | .error e => .err e
    | .ok d => ofExcept (fun (t : Nat) => TV.datetime (midnight d + t)) (timeFromMap N m)
  | .str s :: _ => ofExcept (fun p => .datetime (midnight p.1 + p.2)) (parseDatetime C s)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- The seven duration fields, in `DURATION_SLOTS` order. -/
def durKeys : List String := ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]

/-- The `for (k, v) in map.iter()` loop of `duration_pure` (temporal.rs:544-566): a numeric
value overwrites its field, anything else is skipped (`continue`), unknown keys ignored. -/
def durLoop : List (String × TV F) → (String → Int) → (String → Int)
  | [], acc => acc
  | (k, v) :: kvs, acc =>
    match slotToInt N v with
    | none => durLoop kvs acc
    | some n => durLoop kvs (if k ∈ durKeys then (fun k' => if k' = k then n else acc k') else acc)

def applyCds (cds : CDS) (f : String → Int) : Except String Int :=
  cds (f "years") (f "months") (f "weeks") (f "days") (f "hours") (f "minutes") (f "seconds")

/-- `duration_pure` (temporal.rs:541). -/
def durationPure (cds : CDS) : List (TV F) → Res (TV F)
  | .map m :: _ => ofExcept .duration (applyCds cds (durLoop N m fun _ => 0))
  | .str s :: _ =>
    match parseDuration C s with
    | .error e => .err e
    | .ok (y, mo, w, d, h, mi, se) => ofExcept .duration (cds y mo w d h mi se)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- Slot `i` read with `slot_to_int` (missing slots are `Null`). -/
def slot (args : List (TV F)) (i : Nat) : Option Int := slotToInt N (args.getD i .null)

/-- `duration_struct_pure` (temporal.rs:621). -/
def durationStructPure (cds : CDS) (args : List (TV F)) : Res (TV F) :=
  ofExcept .duration (cds ((slot N args 0).getD 0) ((slot N args 1).getD 0) ((slot N args 2).getD 0)
    ((slot N args 3).getD 0) ((slot N args 4).getD 0) ((slot N args 5).getD 0) ((slot N args 6).getD 0))

/-- `date_struct_pure` (temporal.rs:634). -/
def dateStructPure (args : List (TV F)) : Res (TV F) :=
  ofExcept (fun d => .date (midnight d)) (dfc (slot N args 0) (slot N args 1) (slot N args 2)
    (slot N args 3) (slot N args 4) (slot N args 5) (slot N args 6))

/-- `localtime_struct_pure` (temporal.rs:649). -/
def localtimeStructPure (args : List (TV F)) : Res (TV F) :=
  ofExcept (fun (t : Nat) => TV.time (t : Int)) (timeFromComponents (slot N args 0) (slot N args 1) (slot N args 2))

/-- `localdatetime_struct_pure` (temporal.rs:661). -/
def localdatetimeStructPure (args : List (TV F)) : Res (TV F) :=
  match dfc (slot N args 0) (slot N args 1) (slot N args 2) (slot N args 3) (slot N args 4)
      (slot N args 5) (slot N args 6) with
  | .error e => .err e
  | .ok d => ofExcept (fun (t : Nat) => TV.datetime (midnight d + t))
      (timeFromComponents (slot N args 7) (slot N args 8) (slot N args 9))

/-! ### Theorems: dispatch, dead `unreachable!()`, map form = slot form -/

/-- The declared argument type `Map | String | Null`. -/
def MSN : TV F → Prop
  | .map _ | .str _ | .null => True
  | _ => False

theorem pure_not_unreachable (cds : CDS) (a : TV F) (rest : List (TV F)) (h : MSN a) :
    datePure N C dfc (a :: rest) ≠ .unreachable ∧ localtimePure N C (a :: rest) ≠ .unreachable ∧
    localdatetimePure N C dfc (a :: rest) ≠ .unreachable ∧
    durationPure N C cds (a :: rest) ≠ .unreachable := by
  cases a <;> simp [MSN] at h <;>
    simp only [datePure, localtimePure, localdatetimePure, durationPure] <;>
    refine ⟨?_, ?_, ?_, ?_⟩ <;> (try split) <;> (try split) <;>
    simp [ofExcept] <;> (try split) <;> simp

theorem pure_null (cds : CDS) (rest : List (TV F)) :
    datePure N C dfc (.null :: rest) = .ok .null ∧ localtimePure N C (.null :: rest) = .ok .null ∧
    localdatetimePure N C dfc (.null :: rest) = .ok .null ∧
    durationPure N C cds (.null :: rest) = .ok .null := ⟨rfl, rfl, rfl, rfl⟩

/-- A `date(...)` value is always a midnight timestamp. -/
theorem datePure_midnight (args : List (TV F)) (t : Int) (h : datePure N C dfc args = .ok (.date t)) :
    t % 86400 = 0 := by
  unfold datePure at h
  split at h <;> (try cases h)
  all_goals (unfold ofExcept at h; split at h <;> cases h; simp [midnight])

/-- A `localtime(...)` value is a second of the day. -/
theorem localtimePure_range (args : List (TV F)) (t : Int)
    (h : localtimePure N C args = .ok (.time t)) : 0 ≤ t ∧ t < 86400 := by
  unfold localtimePure at h
  split at h <;> (try cases h)
  · unfold ofExcept at h; split at h <;> cases h
    rename_i t' ht; have := (timeFromComponents_ok _ _ _ _ ht).1; omega
  · unfold ofExcept at h; split at h <;> cases h
    rename_i t' ht; have := parseTime_ok C _ _ ht; omega

theorem bind_slot (o : Option (TV F)) : o.bind (slotToInt N) = slotToInt N (o.getD .null) := by
  cases o <;> rfl

/-- The map form and the positional-slot form agree on dates (the binder rewrites constant
map literals into the slot form, so both must give the same value). -/
theorem dateStruct_eq_map (m : List (String × TV F)) :
    datePure N C dfc [.map m] = dateStructPure N dfc
      ((["year", "month", "day", "week", "dayOfWeek", "quarter", "dayOfQuarter"]).map
        fun k => (getStr m k).getD .null) := by
  simp only [datePure, dateStructPure, dateFromMap_eq_slots, slot, bind_slot]; rfl

theorem localtimeStruct_eq_map (m : List (String × TV F)) :
    localtimePure N C [.map m] = localtimeStructPure N
      ((["hour", "minute", "second"]).map fun k => (getStr m k).getD .null) := by
  simp only [localtimePure, localtimeStructPure, timeFromMap_eq_slots, slot, bind_slot]; rfl


theorem lookup_none_of_not_mem {β : Type} (k : String) (l : List (String × β))
    (h : k ∉ l.map Prod.fst) : l.lookup k = none := by
  induction l with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨a, b⟩ := p
    have hne : k ≠ a := fun e => h (by simp [e])
    have : (k == a) = false := by simp [hne]
    simp only [List.lookup, this]
    exact ih (fun hm => h (by simp [hm]))

/-- The `duration_pure` map loop reads each known field as its (unique) map binding:
with distinct keys (an `OrderMap`), field `k` ends as `slot_to_int` of `map[k]`, or 0. -/
theorem durLoop_get (k : String) (hk : k ∈ durKeys) :
    ∀ (m : List (String × TV F)) (acc : String → Int), (m.map Prod.fst).Nodup →
      durLoop N m acc k = (((m.lookup k).bind (slotToInt N)).getD (acc k)) := by
  intro m
  induction m with
  | nil => intro acc _; rfl
  | cons p kvs ih =>
    intro acc hnd
    obtain ⟨k', v⟩ := p
    simp only [List.map, List.nodup_cons] at hnd
    obtain ⟨hk', hnd⟩ := hnd
    simp only [durLoop]
    by_cases e : k = k'
    · subst e
      have hl : kvs.lookup k = none := lookup_none_of_not_mem k kvs hk'
      simp only [List.lookup, beq_self_eq_true]
      cases hs : slotToInt N v with
      | none => dsimp only; rw [ih acc hnd, hl]; simp [hs]
      | some n => dsimp only; rw [ih _ hnd, hl]; simp [hk, hs]
    · have : (k == k') = false := by simp [e]
      simp only [List.lookup, this]
      cases hs : slotToInt N v with
      | none => dsimp only; exact ih acc hnd
      | some n =>
        dsimp only; rw [ih _ hnd]
        split <;> simp [e]

/-- Hence `duration({..})` through the map loop equals the 7-slot form (the binder's
rewrite of constant map literals preserves the value). -/
theorem durationStruct_eq_map (cds : CDS) (m : List (String × TV F)) (hnd : (m.map Prod.fst).Nodup) :
    durationPure N C cds [.map m] = durationStructPure N cds (durKeys.map fun k => (getStr m k).getD .null) := by
  simp only [durationPure, durationStructPure, applyCds, slot, getStr]
  rw [durLoop_get N "years" (by decide) m _ hnd, durLoop_get N "months" (by decide) m _ hnd,
    durLoop_get N "weeks" (by decide) m _ hnd, durLoop_get N "days" (by decide) m _ hnd,
    durLoop_get N "hours" (by decide) m _ hnd, durLoop_get N "minutes" (by decide) m _ hnd,
    durLoop_get N "seconds" (by decide) m _ hnd]
  simp only [bind_slot]; rfl

/-! ### Clock-reading functions (temporal.rs:687-805) -/

/-- `Utc::now()` / `transaction_timestamp` as a timestamp in seconds. -/
def todayMidnight (now : Int) : Int := midnight (now / 86400)   -- `date_naive()` (floor)
def timeOfDay (now : Int) : Int := now % 86400                  -- `.time()` on 1970-01-01

/-- `timestamp_fn` (temporal.rs:687): `now.timestamp_millis()`. -/
def timestampFn (nowMs : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.int nowMs) else .unreachable   -- `debug_assert!(args.is_empty())`

/-- `date_fn` (temporal.rs:703). -/
def dateFn (now : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.date (todayMidnight now)) else datePure N C dfc args

/-- `localtime_fn` (temporal.rs:724). -/
def localtimeFn (now : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.time (timeOfDay now)) else localtimePure N C args

/-- `localdatetime_fn` (temporal.rs:747). -/
def localdatetimeFn (now : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.datetime now) else localdatetimePure N C dfc args

/-- `duration_fn` (temporal.rs:763). -/
def durationFn (cds : CDS) (args : List (TV F)) : Res (TV F) := durationPure N C cds args

/-- `date.transaction` / `localtime.transaction` / `localdatetime.transaction`
(temporal.rs:773, :785, :799), reading the runtime's `transaction_timestamp`. -/
def dateTransactionFn (txn : Int) : Res (TV F) := .ok (.date (todayMidnight txn))
def localtimeTransactionFn (txn : Int) : Res (TV F) := .ok (.time (timeOfDay txn))
def localdatetimeTransactionFn (txn : Int) : Res (TV F) := .ok (.datetime txn)

theorem clock_fns_shape (now : Int) :
    dateFn N C dfc now ([] : List (TV F)) = .ok (.date (todayMidnight now)) ∧
    todayMidnight now % 86400 = 0 ∧ todayMidnight now ≤ now ∧ now < todayMidnight now + 86400 ∧
    0 ≤ timeOfDay now ∧ timeOfDay now < 86400 ∧ todayMidnight now + timeOfDay now = now := by
  refine ⟨by simp [dateFn], ?_⟩
  simp only [todayMidnight, timeOfDay, midnight]
  refine ⟨by simp, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- With arguments the clock is not read: the `*_fn` is the pure constructor. -/
theorem fns_with_args (now : Int) (args : List (TV F)) (h : args ≠ []) :
    dateFn N C dfc now args = datePure N C dfc args ∧ localtimeFn N C now args = localtimePure N C args ∧
    localdatetimeFn N C dfc now args = localdatetimePure N C dfc args := by
  simp [dateFn, localtimeFn, localdatetimeFn, h]

/-- The transaction functions are consistent within a query: they read one timestamp, and
`localdatetime.transaction() = date.transaction() + localtime.transaction()`. -/
theorem transaction_consistent (txn : Int) :
    (dateTransactionFn txn : Res (TV F)) = .ok (.date (todayMidnight txn)) ∧
    (localtimeTransactionFn txn : Res (TV F)) = .ok (.time (timeOfDay txn)) ∧
    todayMidnight txn + timeOfDay txn = txn := by
  refine ⟨rfl, rfl, ?_⟩; simp only [todayMidnight, timeOfDay, midnight]; omega

/-! ### `register` (temporal.rs:681) -/

/-- What `register` adds: `cypher_fn!` names with (arity range, non-deterministic flag),
then `set_pure_fn` and `set_struct_fn` (with the slot list) on four of them. -/
def registered : List (String × Nat × Nat × Bool) :=
  [("timestamp", 0, 0, true), ("date", 0, 1, true), ("localtime", 0, 1, true),
   ("localdatetime", 0, 1, true), ("duration", 1, 1, false), ("date.transaction", 0, 0, true),
   ("localtime.transaction", 0, 0, true), ("localdatetime.transaction", 0, 0, true)]
def pureFns : List String := ["date", "localtime", "localdatetime", "duration"]
def structSlots : List (String × List String) :=
  [("date", ["year", "month", "day", "week", "dayOfWeek", "quarter", "dayOfQuarter"]),
   ("localtime", ["hour", "minute", "second"]),
   ("localdatetime", ["year", "month", "day", "week", "dayOfWeek", "quarter", "dayOfQuarter",
     "hour", "minute", "second"]),
   ("duration", durKeys)]

/-- Names are distinct, every pure/struct target was registered first, and each struct
form's slot count is the arity its `*_struct_pure` asserts (7, 3, 10, 7). -/
theorem register_wf :
    (registered.map (·.1)).Nodup ∧ (∀ n ∈ pureFns, n ∈ registered.map (·.1)) ∧
    (∀ p ∈ structSlots, p.1 ∈ pureFns) ∧
    structSlots.map (fun p => p.2.length) = [7, 3, 10, 7] := by decide

/-- Only `duration` is deterministic (constant-foldable without arguments). -/
theorem register_nondet : registered.filter (fun r => !r.2.2.2) = [("duration", 1, 1, false)] := by
  decide

end FalkorTemporal.Fns
