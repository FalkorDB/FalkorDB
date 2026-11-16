import FalkorTemporal.Parse
/-!
# Temporal constructors: `*_pure`, `*_struct_pure`, the clock `*_fn`s and `register`
(functions/temporal.rs:660-1020)

`construct_duration_secs` (PROVEN in Duration.lean as `constructDuration`) and
`date_from_components` (Constructors.lean) are parameters, as is
`construct_duration_secs_float` (`cdsF`: f32/f64 arithmetic, NOT COVERED). `unreachable!()`
is the result `Res.unreachable`; the theorems show it is dead for every argument the
declared type admits. The clock is an argument (`now`, seconds; `nowMs`, milliseconds).

Since #3607 the binder's struct rewrite passes one child per slot plus a presence mask
(`rewriteArgs`); `*Struct_eq_map` prove the rewritten call equals the map call for
`date`, `localtime` and `duration`. For `localdatetime` they differ
(`pr3607_localdatetime_map_ne_struct`): the map form checks the date before validating
the time components, the struct form validates all ten first.
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

/-- `construct_duration_secs_float` signature (temporal.rs:822, f32 arithmetic; a parameter). -/
abbrev CDSF (F : Type) := List F → Except String Int

/-- `date.and_hms_opt(0,0,0).unwrap().and_utc().timestamp()`: midnight of day `d`
(`unwrap` cannot fail: 00:00:00 is always valid). -/
def midnight (d : Int) : Int := d * 86400

/-- `date_pure` (temporal.rs:667). -/
def datePure : List (TV F) → Res (TV F)
  | .map m :: _ => ofExcept (fun d => .date (midnight d))
      (match mapComponents m dateSlots [] false with
       | .error e => .error e
       | .ok vals => dateFromValues dfc vals)
  | .str s :: _ => ofExcept (fun d => .date (midnight d)) (parseDate C s)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `localtime_pure` (temporal.rs:684): the time on 1970-01-01, i.e. its second of day. -/
def localtimePure : List (TV F) → Res (TV F)
  | .map m :: _ => ofExcept (fun (t : Nat) => TV.time (t : Int))
      (match mapComponents m localtimeSlots subsecondKeys false with
       | .error e => .error e
       | .ok vals => timeFromValues vals)
  | .str s :: _ => ofExcept (fun (t : Nat) => TV.time (t : Int)) (parseTime C s)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `localdatetime_pure` (temporal.rs:708): `vals[7..]` is `fun i => vals (i + 7)`. -/
def localdatetimePure : List (TV F) → Res (TV F)
  | .map m :: _ =>
    match mapComponents m localdatetimeSlots subsecondKeys false with
    | .error e => .err e
    | .ok vals =>
      match dateFromValues dfc vals with
      | .error e => .err e
      | .ok d => ofExcept (fun (t : Nat) => TV.datetime (midnight d + t))
          (timeFromValues fun i => vals (i + 7))
  | .str s :: _ => ofExcept (fun p => .datetime (midnight p.1 + p.2)) (parseDatetime C s)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- The seven duration fields, in `DURATION_SLOTS` order. -/
def durKeys : List String := ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]

def upd {α : Type} (f : Nat → α) (i : Nat) (x : α) : Nat → α := fun j => if j = i then x else f j

/-- The loop of `duration_from_values` (temporal.rs:791-805): `ints`, `floats`, `any_float`. -/
def dvLoop (vals : Nat → Option (TV F)) : List String → Nat → (Nat → Int) → (Nat → F) → Bool →
    Except String ((Nat → Int) × (Nat → F) × Bool)
  | [], _, ints, fl, af => .ok (ints, fl, af)
  | key :: ks, i, ints, fl, af =>
    match vals i with
    | none => dvLoop vals ks (i + 1) ints fl af
    | some (.int n) => dvLoop vals ks (i + 1) (upd ints i n) (upd fl i (N.ofInt n)) af
    | some (.float f) => dvLoop vals ks (i + 1) ints (upd fl i f) true
    | some _ => .error s!"{key} must be a numerical value"

/-- `duration_from_values` (temporal.rs:787). -/
def durationFromValues (cds : CDS) (cdsF : CDSF F) (vals : Nat → Option (TV F)) : Res (TV F) :=
  match dvLoop N vals durKeys 0 (fun _ => 0) (fun _ => N.ofInt 0) false with
  | .error e => .err e
  | .ok (ints, fl, af) =>
    ofExcept .duration
      (if af then cdsF ((List.range 7).map fl)
       else cds (ints 0) (ints 1) (ints 2) (ints 3) (ints 4) (ints 5) (ints 6))

/-- `duration_pure` (temporal.rs:726). -/
def durationPure (cds : CDS) (cdsF : CDSF F) : List (TV F) → Res (TV F)
  | .map m :: _ =>
    match mapComponents m durKeys [] true with
    | .error e => .err e
    | .ok vals => durationFromValues N cds cdsF vals
  | .str s :: _ =>
    match parseDuration C s with
    | .error e => .err e
    | .ok (y, mo, w, d, h, mi, se) => ofExcept .duration (cds y mo w d h mi se)
  | .null :: _ => .ok .null
  | _ => .unreachable

/-- `duration_struct_pure` (temporal.rs:845). -/
def durationStructPure (cds : CDS) (cdsF : CDSF F) (args : List (TV F)) : Res (TV F) :=
  durationFromValues N cds cdsF (structComponents args)

/-- `date_struct_pure` (temporal.rs:850). -/
def dateStructPure (args : List (TV F)) : Res (TV F) :=
  ofExcept (fun d => .date (midnight d))
    (match structInts args dateSlots with
     | .error e => .error e
     | .ok c => dfc (c 0) (c 1) (c 2) (c 3) (c 4) (c 5) (c 6))

/-- `localtime_struct_pure` (temporal.rs:858). -/
def localtimeStructPure (args : List (TV F)) : Res (TV F) :=
  ofExcept (fun (t : Nat) => TV.time (t : Int))
    (match structInts args localtimeSlots with
     | .error e => .error e
     | .ok c => timeFromComponents (c 0) (c 1) (c 2))

/-- `localdatetime_struct_pure` (temporal.rs:867): all ten components validated first. -/
def localdatetimeStructPure (args : List (TV F)) : Res (TV F) :=
  match structInts args localdatetimeSlots with
  | .error e => .err e
  | .ok c =>
    match dfc (c 0) (c 1) (c 2) (c 3) (c 4) (c 5) (c 6) with
    | .error e => .err e
    | .ok d => ofExcept (fun (t : Nat) => TV.datetime (midnight d + t))
        (timeFromComponents (c 7) (c 8) (c 9))

/-! ### Theorems: dispatch, dead `unreachable!()` -/

/-- The declared argument type `Map | String | Null`. -/
def MSN : TV F → Prop
  | .map _ | .str _ | .null => True
  | _ => False

theorem ofExcept_ne {α β : Type} (f : α → β) (e : Except String α) : ofExcept f e ≠ .unreachable := by
  cases e <;> (intro h; cases h)

theorem pure_not_unreachable (cds : CDS) (cdsF : CDSF F) (a : TV F) (rest : List (TV F)) (h : MSN a) :
    datePure C dfc (a :: rest) ≠ .unreachable ∧ localtimePure C (a :: rest) ≠ .unreachable ∧
    localdatetimePure C dfc (a :: rest) ≠ .unreachable ∧
    durationPure N C cds cdsF (a :: rest) ≠ .unreachable := by
  cases a <;> simp [MSN] at h <;>
    simp only [datePure, localtimePure, localdatetimePure, durationPure, durationFromValues] <;>
    refine ⟨?_, ?_, ?_, ?_⟩ <;> (repeat' split) <;>
    first | exact ofExcept_ne _ _ | (intro h; cases h)

theorem pure_null (cds : CDS) (cdsF : CDSF F) (rest : List (TV F)) :
    datePure C dfc (.null :: rest) = .ok .null ∧ localtimePure C (.null :: rest) = .ok .null ∧
    localdatetimePure C dfc (.null :: rest) = .ok .null ∧
    durationPure N C cds cdsF (.null :: rest) = .ok .null := ⟨rfl, rfl, rfl, rfl⟩

/-- A `date(...)` value is always a midnight timestamp. -/
theorem datePure_midnight (args : List (TV F)) (t : Int) (h : datePure C dfc args = .ok (.date t)) :
    t % 86400 = 0 := by
  unfold datePure at h
  split at h <;> (try cases h)
  all_goals (unfold ofExcept at h; split at h <;> cases h; simp [midnight])

theorem timeFromValues_ok (vals : Nat → Option (TV F)) (t : Nat) (h : timeFromValues vals = .ok t) :
    t < 86400 := by
  unfold timeFromValues at h; split at h
  · cases h
  · exact (timeFromComponents_ok _ _ _ _ h).1

/-- A `localtime(...)` value is a second of the day. -/
theorem localtimePure_range (args : List (TV F)) (t : Int)
    (h : localtimePure C args = .ok (.time t)) : 0 ≤ t ∧ t < 86400 := by
  unfold localtimePure at h
  split at h <;> (try cases h)
  · unfold ofExcept at h; split at h <;> cases h
    rename_i t' ht
    split at ht
    · cases ht
    · have := timeFromValues_ok _ _ ht; omega
  · unfold ofExcept at h; split at h <;> cases h
    rename_i t' ht; have := parseTime_ok C _ _ ht; omega

/-! ### The binder's rewrite (#3607): map form = slot form -/

/-- The presence mask the binder appends (`mask |= 1 << i` for each written slot `i`;
the bits are distinct, so `|` is `+`). -/
def maskFor (m : List (String × TV F)) : List String → Int
  | [] => 0
  | s :: ss => (if (m.lookup s).isSome then 1 else 0) + 2 * maskFor m ss

/-- The children `rewrite_struct_constructor` produces for a literal map `m` whose keys are
all slots (binder.rs:2558-2580): one per slot (`Constant(Null)` when absent), then the mask.
eval.rs evaluates them in place (a non-null constant is its value). -/
def rewriteArgs (keys : List String) (m : List (String × TV F)) : List (TV F) :=
  keys.map (fun s => (m.lookup s).getD .null) ++ [.int (maskFor m keys)]

theorem bitAt_maskFor (m : List (String × TV F)) :
    ∀ (keys : List String) (i : Nat),
      bitAt (maskFor m keys) i = (keys[i]?.any fun s => (m.lookup s).isSome)
  | [], i => by
    cases i <;> simp [maskFor, bitAt]
    rename_i i
    induction i with
    | zero => rfl
    | succ j ih => simp [bitAt, ih]
  | s :: ss, 0 => by
    simp only [maskFor, bitAt, List.getElem?_cons_zero, Option.any_some]
    cases (m.lookup s).isSome
    · simp
    · have : (1 + 2 * maskFor m ss) % 2 = 1 := by omega
      simp [this]
  | s :: ss, i + 1 => by
    simp only [maskFor, bitAt, List.getElem?_cons_succ]
    rw [← bitAt_maskFor m ss i]
    congr 1
    split <;> omega

theorem splitMask_append (xs : List (TV F)) (k : Int) : splitMask (xs ++ [.int k]) = (k, xs) := by
  simp [splitMask]

theorem lastMask_append (xs : List (TV F)) (k : Int) : lastMask (xs ++ [.int k]) = k := by
  simp [lastMask]

/-- The rewritten call's components are the map's: written slots `Some`, the rest `None`. -/
theorem structComponents_rewrite (keys : List String) (m : List (String × TV F)) :
    structComponents (rewriteArgs keys m) = fun i => keys[i]?.bind (List.lookup · m) := by
  funext i
  simp only [structComponents, rewriteArgs, splitMask_append, bitAt_maskFor, List.getElem?_map]
  cases keys[i]? with
  | none => simp
  | some s => cases h : m.lookup s <;> simp [h]

theorem siLoop_eq_icLoop (xs : List (TV F)) (k : Int) :
    ∀ (keys : List String) (i : Nat) (out : Nat → Option Int), i + keys.length ≤ xs.length →
      siLoop (xs ++ [.int k]) k keys i out = icLoop (structComponents (xs ++ [.int k])) keys i out
  | [], _, _, _ => rfl
  | key :: ks, i, out, h => by
    simp only [List.length_cons] at h
    have hi : i < xs.length := by omega
    have hg : (xs ++ [TV.int k]).getD i .null = xs[i] := by
      simp [List.getD, List.getElem?_append_left hi, List.getElem?_eq_getElem hi]
    simp only [siLoop, icLoop, structComponents, splitMask_append, hg]
    cases hb : bitAt k i with
    | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      exact siLoop_eq_icLoop xs k ks (i + 1) out (by omega)
    | true =>
      simp only [↓reduceIte, List.getElem?_eq_getElem hi]
      cases intComponent xs[i] key with
      | error e => rfl
      | ok n => exact siLoop_eq_icLoop xs k ks (i + 1) _ (by omega)

/-- `struct_ints` is `int_components` of `struct_components` (the fused per-row path). -/
theorem structInts_rewrite (keys : List String) (m : List (String × TV F)) :
    structInts (rewriteArgs keys m) keys = intComponents (fun i => keys[i]?.bind (List.lookup · m)) keys := by
  rw [← structComponents_rewrite keys m]
  simp only [structInts, intComponents, rewriteArgs, lastMask_append]
  exact siLoop_eq_icLoop _ _ keys 0 _ (by simp)

theorem position_of_mem (k : String) :
    ∀ (keys : List String), k ∈ keys → (∀ a ∈ keys, ieq a k = true → a = k) →
      ∃ i, position (fun s => ieq s k) keys = some i ∧ keys[i]? = some k
  | [], h, _ => by simp at h
  | s :: ss, h, hd => by
    simp only [position]
    by_cases e : ieq s k = true
    · have := hd s (by simp) e; subst this; exact ⟨0, by simp [e]⟩
    · have hs : s ≠ k := fun h' => by subst h'; exact e (ieq_refl s)
      have hk : k ∈ ss := by
        rcases List.mem_cons.1 h with h' | h'
        · exact absurd h'.symm hs
        · exact h'
      obtain ⟨i, hp, hi⟩ := position_of_mem k ss hk (fun a ha => hd a (by simp [ha]))
      exact ⟨i + 1, by simp [e, hp], by simpa using hi⟩

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

theorem getElem?_inj_of_nodup {α : Type} :
    ∀ (l : List α) (i j : Nat) (a : α), l.Nodup → l[i]? = some a → l[j]? = some a → i = j
  | [], _, _, _, _, h, _ => by simp at h
  | x :: xs, i, j, a, hn, hi, hj => by
    rw [List.nodup_cons] at hn
    cases i with
    | zero =>
      cases j with
      | zero => rfl
      | succ j =>
        simp at hi hj; subst hi
        exact absurd (List.mem_of_getElem? hj) hn.1
    | succ i =>
      cases j with
      | zero =>
        simp at hi hj; subst hj
        exact absurd (List.mem_of_getElem? hi) hn.1
      | succ j =>
        simp at hi hj
        rw [getElem?_inj_of_nodup xs i j a hn.2 hi hj]

/-- The `map_components` loop on a map whose keys are distinct slot names. -/
theorem mcLoop_exact (keys extra : List String) (hkn : keys.Nodup)
    (hdist : ∀ a ∈ keys, ∀ b ∈ keys, ieq a b = true → a = b) (seen : Nat → Bool) :
    ∀ (m : List (String × TV F)) (out : Nat → Option (TV F)),
      (m.map Prod.fst).Nodup → (∀ k ∈ m.map Prod.fst, k ∈ keys) →
      (∀ j k, keys[j]? = some k → k ∈ m.map Prod.fst → out j = none) →
      mcLoop keys extra m out false seen = .ok (fun j => (keys[j]?.bind (List.lookup · m)).or (out j), false)
  | [], out, _, _, _ => by simp [mcLoop]
  | (k, v) :: kvs, out, hnd, hk, hout => by
    simp only [List.map_cons, List.nodup_cons] at hnd
    have hkk : k ∈ keys := hk k (by simp)
    obtain ⟨i, hp, hi⟩ := position_of_mem k keys hkk (fun a ha => hdist a ha k hkk)
    have hoi : out i = none := hout i k hi (by simp)
    simp only [mcLoop, hp, hoi]
    rw [mcLoop_exact keys extra hkn hdist seen kvs (setAt out i v) hnd.2
      (fun k' hk' => hk k' (by simp [hk']))]
    · congr 1; congr 1
      funext j
      by_cases hj : j = i
      · subst hj
        rw [hi]
        simp [setAt, lookup_none_of_not_mem k kvs hnd.1, List.lookup]
      · simp only [setAt, hj, ↓reduceIte]
        cases hkj : keys[j]? with
        | none => rfl
        | some s =>
          have hs : s ≠ k := fun e => hj (getElem?_inj_of_nodup keys j i k hkn (e ▸ hkj) hi)
          have : (s == k) = false := by simp [hs]
          simp [List.lookup, this]
    · intro j k' hj hk'
      simp only [setAt]
      split
      · rename_i e; subst e
        rw [hi] at hj; cases hj; exact absurd hk' hnd.1
      · exact hout j k' hj (by simp [hk'])

/-- `map_components` on such a map: no error, slot `i` holds `map[keys[i]]`. -/
theorem mapComponents_exact (keys extra : List String) (numeric : Bool) (hkn : keys.Nodup)
    (hdist : ∀ a ∈ keys, ∀ b ∈ keys, ieq a b = true → a = b)
    (m : List (String × TV F)) (hnd : (m.map Prod.fst).Nodup) (hk : ∀ k ∈ m.map Prod.fst, k ∈ keys) :
    mapComponents m keys extra numeric = .ok fun i => keys[i]?.bind (List.lookup · m) := by
  unfold mapComponents
  rw [mcLoop_exact keys extra hkn hdist _ m _ hnd hk (fun _ _ _ _ => rfl)]
  simp

theorem dateSlots_wf : dateSlots.Nodup ∧ ∀ a ∈ dateSlots, ∀ b ∈ dateSlots, ieq a b = true → a = b := by
  decide
theorem localtimeSlots_wf :
    localtimeSlots.Nodup ∧ ∀ a ∈ localtimeSlots, ∀ b ∈ localtimeSlots, ieq a b = true → a = b := by
  decide
theorem durKeys_wf : durKeys.Nodup ∧ ∀ a ∈ durKeys, ∀ b ∈ durKeys, ieq a b = true → a = b := by
  decide
theorem localdatetimeSlots_wf : localdatetimeSlots.Nodup ∧
    ∀ a ∈ localdatetimeSlots, ∀ b ∈ localdatetimeSlots, ieq a b = true → a = b := by
  decide

/-- The map form and the positional-slot form agree on dates: for a literal map with
distinct slot keys (the only maps the binder rewrites, `rewriteStruct_unknown`), the
rewritten call (slots + presence mask) gives the map call's value or error. -/
theorem dateStruct_eq_map (m : List (String × TV F)) (hnd : (m.map Prod.fst).Nodup)
    (hk : ∀ k ∈ m.map Prod.fst, k ∈ dateSlots) :
    datePure C dfc [.map m] = dateStructPure dfc (rewriteArgs dateSlots m) := by
  simp only [datePure, dateStructPure, structInts_rewrite,
    mapComponents_exact dateSlots [] false dateSlots_wf.1 dateSlots_wf.2 m hnd hk, dateFromValues]
  rfl

theorem localtimeStruct_eq_map (m : List (String × TV F)) (hnd : (m.map Prod.fst).Nodup)
    (hk : ∀ k ∈ m.map Prod.fst, k ∈ localtimeSlots) :
    localtimePure C [.map m] = localtimeStructPure (rewriteArgs localtimeSlots m) := by
  simp only [localtimePure, localtimeStructPure, structInts_rewrite,
    mapComponents_exact localtimeSlots subsecondKeys false localtimeSlots_wf.1 localtimeSlots_wf.2
      m hnd hk, timeFromValues]
  rfl

/-- Hence `duration({..})` through the map equals the slot form (same components, same
`duration_from_values`). -/
theorem durationStruct_eq_map (cds : CDS) (cdsF : CDSF F) (m : List (String × TV F))
    (hnd : (m.map Prod.fst).Nodup) (hk : ∀ k ∈ m.map Prod.fst, k ∈ durKeys) :
    durationPure N C cds cdsF [.map m] = durationStructPure N cds cdsF (rewriteArgs durKeys m) := by
  simp only [durationPure, durationStructPure, structComponents_rewrite,
    mapComponents_exact durKeys [] true durKeys_wf.1 durKeys_wf.2 m hnd hk]

/-- Integer components keep the integer path: `duration_from_values` on written `Int`s
calls `construct_duration_secs` with them (absent = 0). -/
theorem durationFromValues_ints (cds : CDS) (cdsF : CDSF F) (n : Nat → Int) :
    durationFromValues N cds cdsF (fun i => if i < 7 then some (.int (n i)) else none) =
      ofExcept .duration (cds (n 0) (n 1) (n 2) (n 3) (n 4) (n 5) (n 6)) := by
  simp [durationFromValues, dvLoop, durKeys, upd]

/-- A written `null` is an error, not a default (#3607), on both forms. -/
theorem date_null_component_errors :
    datePure C dfc [.map [("month", (.null : TV F))]] = .err "month must be an integer value" ∧
    dateStructPure dfc (rewriteArgs dateSlots [("month", (.null : TV F))]) =
      .err "month must be an integer value" := by
  constructor <;> rfl

/-- An unknown key is an error on the map form (the binder leaves such calls alone). -/
theorem date_unknown_key_errors :
    datePure C dfc [.map [("year", (.int 2020 : TV F)), ("foo", .int 5)]] = .err unknownKey := by
  rfl

/-! ### Counterexample (#3607): `localdatetime` map form ≠ slot form -/

/-- The map `{month: 13, hour: null}`. -/
def ldtMap : List (String × TV F) := [("month", .int 13), ("hour", .null)]

/-- The slot form validates all ten components first, so it reports the `null` hour. -/
theorem ldt_struct_hour_error :
    localdatetimeStructPure dfc (rewriteArgs localdatetimeSlots (ldtMap : List (String × TV F))) =
      .err "hour must be an integer value" := by
  rfl

/-- The map form builds the date first, so it reports the invalid month 13 (whatever
`date_from_components` says about it) and never looks at the hour. -/
theorem ldt_map_date_error (e : String)
    (hd : dfc none (some 13) none none none none none = .error e) :
    localdatetimePure C dfc [.map (ldtMap : List (String × TV F))] = .err e := by
  have h := mapComponents_exact localdatetimeSlots subsecondKeys false localdatetimeSlots_wf.1
    localdatetimeSlots_wf.2 (ldtMap : List (String × TV F)) (by simp [ldtMap])
    (by simp [ldtMap, localdatetimeSlots, dateSlots, localtimeSlots])
  simp only [localdatetimePure, h]
  have : dateFromValues dfc (fun i => localdatetimeSlots[i]?.bind (List.lookup · (ldtMap : List (String × TV F))))
      = dfc none (some 13) none none none none none := rfl
  rw [this, hd]

/-- **Counterexample.** The binder rewrite no longer preserves `localdatetime`'s result
(the `dateStruct_eq_map` / `localtimeStruct_eq_map` / `durationStruct_eq_map` property):
`localdatetime({month: 13, hour: null})` errors with "hour must be an integer value",
`WITH {month: 13, hour: null} AS m RETURN localdatetime(m)` with
"Invalid date: year=1970, month=13, day=1" (any message other than the hour's). Before
#3607 both gave the date error. -/
theorem pr3607_localdatetime_map_ne_struct (e : String)
    (hd : dfc none (some 13) none none none none none = .error e)
    (he : e ≠ "hour must be an integer value") :
    localdatetimePure C dfc [.map (ldtMap : List (String × TV F))] ≠
      localdatetimeStructPure dfc (rewriteArgs localdatetimeSlots (ldtMap : List (String × TV F))) := by
  rw [ldt_map_date_error C dfc e hd, ldt_struct_hour_error dfc]
  intro h; cases h; exact he rfl

/-! ### Clock-reading functions (temporal.rs:882-1004) -/

/-- `Utc::now()` / `transaction_timestamp` as a timestamp in seconds. -/
def todayMidnight (now : Int) : Int := midnight (now / 86400)   -- `date_naive()` (floor)
def timeOfDay (now : Int) : Int := now % 86400                  -- `.time()` on 1970-01-01

/-- `timestamp_fn` (temporal.rs:882): `now.timestamp_millis()`. -/
def timestampFn (nowMs : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.int nowMs) else .unreachable   -- `debug_assert!(args.is_empty())`

/-- `date_fn` (temporal.rs:898). -/
def dateFn (now : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.date (todayMidnight now)) else datePure C dfc args

/-- `localtime_fn` (temporal.rs:919). -/
def localtimeFn (now : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.time (timeOfDay now)) else localtimePure C args

/-- `localdatetime_fn` (temporal.rs:942). -/
def localdatetimeFn (now : Int) (args : List (TV F)) : Res (TV F) :=
  if args = [] then .ok (.datetime now) else localdatetimePure C dfc args

/-- `duration_fn` (temporal.rs:958). -/
def durationFn (cds : CDS) (cdsF : CDSF F) (args : List (TV F)) : Res (TV F) :=
  durationPure N C cds cdsF args

/-- `date.transaction` / `localtime.transaction` / `localdatetime.transaction`
(temporal.rs:968, :980, :994), reading the runtime's `transaction_timestamp`. -/
def dateTransactionFn (txn : Int) : Res (TV F) := .ok (.date (todayMidnight txn))
def localtimeTransactionFn (txn : Int) : Res (TV F) := .ok (.time (timeOfDay txn))
def localdatetimeTransactionFn (txn : Int) : Res (TV F) := .ok (.datetime txn)

theorem clock_fns_shape (now : Int) :
    dateFn C dfc now ([] : List (TV F)) = .ok (.date (todayMidnight now)) ∧
    todayMidnight now % 86400 = 0 ∧ todayMidnight now ≤ now ∧ now < todayMidnight now + 86400 ∧
    0 ≤ timeOfDay now ∧ timeOfDay now < 86400 ∧ todayMidnight now + timeOfDay now = now := by
  refine ⟨by simp [dateFn], ?_⟩
  simp only [todayMidnight, timeOfDay, midnight]
  refine ⟨by simp, ?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- With arguments the clock is not read: the `*_fn` is the pure constructor. -/
theorem fns_with_args (now : Int) (args : List (TV F)) (h : args ≠ []) :
    dateFn C dfc now args = datePure C dfc args ∧ localtimeFn C now args = localtimePure C args ∧
    localdatetimeFn C dfc now args = localdatetimePure C dfc args := by
  simp [dateFn, localtimeFn, localdatetimeFn, h]

/-- The transaction functions are consistent within a query: they read one timestamp, and
`localdatetime.transaction() = date.transaction() + localtime.transaction()`. -/
theorem transaction_consistent (txn : Int) :
    (dateTransactionFn txn : Res (TV F)) = .ok (.date (todayMidnight txn)) ∧
    (localtimeTransactionFn txn : Res (TV F)) = .ok (.time (timeOfDay txn)) ∧
    todayMidnight txn + timeOfDay txn = txn := by
  refine ⟨rfl, rfl, ?_⟩; simp only [todayMidnight, timeOfDay, midnight]; omega

/-! ### `register` (temporal.rs:876) -/

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

/-- #3607: all four struct forms are masked (`set_struct_fn(.., true)`), so each takes its
slot count plus the presence mask: the arities `*_struct_pure` asserts (8, 4, 11, 8). -/
def structMasked : List (String × Bool) :=
  [("date", true), ("localtime", true), ("localdatetime", true), ("duration", true)]

theorem register_masked :
    structMasked.map (·.1) = structSlots.map (·.1) ∧ structMasked.all (·.2) ∧
    structSlots.map (fun p => p.2.length + 1) = [8, 4, 11, 8] := by decide

/-- Only `duration` is deterministic (constant-foldable without arguments). -/
theorem register_nondet : registered.filter (fun r => !r.2.2.2) = [("duration", 1, 1, false)] := by
  decide

end FalkorTemporal.Fns
