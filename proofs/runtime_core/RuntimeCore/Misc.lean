/-
# Limits, `ops/mod.rs` helpers, index options

| here | there |
| --- | --- |
| `checkTimeout`       | `Runtime::check_timeout`, `runtime/runtime.rs:467-474` (deadline = `Instant::now() + timeout_ms` from `Runtime::new`, `:457`) |
| `asI64`, `checkMem`  | `Runtime::check_mem_capacity`, `runtime/runtime.rs:477-485` (`usage_fn() as i64`) |
| `drainPending`       | `drain_pending`, `runtime/ops/mod.rs:127-139` |
| `edgeAlreadyUsed`    | `edge_already_used`, `runtime/ops/mod.rs:155-172` |
| `natOpt`, `vectorOptions` | the `IndexType::Vector` arm of `map_to_index_options`, `runtime/runtime.rs:1970-2073` |
| `fulltextUnknown`    | the unknown-key refusal of the `IndexType::Fulltext` arm, `runtime.rs:1894-1903` |
| `phonetic`           | the `phonetic` key of the `IndexType::Fulltext` arm, `runtime.rs:1924-1941` |
-/
namespace RuntimeCore.Misc

/-! ### Timeout and memory cap -/

/-- `check_timeout`: `deadline` and `now` as monotonic ticks. -/
def checkTimeout (deadline : Option Nat) (now : Nat) : Except String Unit :=
  match deadline with
  | some d => if now ≥ d then .error "Query timed out" else .ok ()
  | none => .ok ()

theorem checkTimeout_spec (deadline : Option Nat) (now : Nat) :
    (checkTimeout deadline now = .error "Query timed out") ↔ ∃ d, deadline = some d ∧ d ≤ now := by
  cases deadline with
  | none => simp [checkTimeout]
  | some d =>
    simp only [checkTimeout]
    by_cases h : now ≥ d <;> simp [h] <;> omega

/-- `usize as i64` (two's complement). -/
def asI64 (u : Nat) : Int := if u % 2 ^ 64 < 2 ^ 63 then (u % 2 ^ 64 : Nat) else (u % 2 ^ 64 : Nat) - 2 ^ 64

/-- `check_mem_capacity`. -/
def checkMem (cap : Int) (usage : Option Nat) : Bool :=
  -- true = error "Query's mem consumption exceeded capacity"
  cap > 0 && match usage with
    | some u => decide (asI64 u > cap)
    | none => false

/-- **PROVEN**: for any realistic usage (`< 2^63` bytes) the cap triggers exactly when the
usage exceeds a positive capacity; capacity `≤ 0` or no usage function disables it. -/
theorem checkMem_spec (cap : Int) (u : Nat) (hu : u < 2 ^ 63) :
    checkMem cap (some u) = true ↔ 0 < cap ∧ cap < u := by
  have h1 : u % 2 ^ 64 = u := Nat.mod_eq_of_lt (by omega)
  simp only [checkMem, asI64, h1, if_pos hu, Bool.and_eq_true, decide_eq_true_eq]

theorem checkMem_disabled (cap : Int) (usage : Option Nat) (h : cap ≤ 0 ∨ usage = none) :
    checkMem cap usage = false := by
  rcases h with h | h
  · simp [checkMem]; omega
  · subst h; simp [checkMem]

/-- The `as i64` wrap (latent: needs ≥ 8 EiB of usage): usage `2^63` reads as negative
and never exceeds a positive cap. -/
theorem checkMem_wrap : checkMem 1 (some (2 ^ 63)) = false := by decide

/-! ### `ops/mod.rs` -/

def BATCH_SIZE : Nat := 1024

/-- `drain_pending`: move rows from the front of `pending` while `builder.len() < BATCH_SIZE`. -/
def drainPending {α : Type} (fuel : Nat) (pending builder : List α) : List α × List α :=
  match fuel with
  | 0 => (pending, builder)
  | fuel + 1 =>
    if builder.length < BATCH_SIZE then
      match pending with
      | [] => ([], builder)
      | r :: rs => drainPending fuel rs (builder ++ [r])
    else (pending, builder)

/-- **PROVEN**: `drain_pending` moves exactly `min(|pending|, BATCH_SIZE - |builder|)` rows,
in order, from the front of the queue (fuel = |pending| suffices). -/
theorem drainPending_spec {α : Type} :
    ∀ (pending builder : List α), builder.length ≤ BATCH_SIZE →
      drainPending pending.length pending builder =
        (pending.drop (BATCH_SIZE - builder.length),
         builder ++ pending.take (BATCH_SIZE - builder.length))
  | [], builder, _ => by
    simp [drainPending]
  | r :: rs, builder, h => by
    simp only [List.length_cons, drainPending]
    split
    · next hl =>
      have := drainPending_spec rs (builder ++ [r]) (by simp; omega)
      rw [this]
      have e : BATCH_SIZE - builder.length = (BATCH_SIZE - (builder ++ [r]).length) + 1 := by
        simp; omega
      rw [e]; simp
    · next hl =>
      have e : BATCH_SIZE - builder.length = 0 := by omega
      simp [e]

/-- A row's value at a slot: only relationship ids matter here. -/
inductive Slot where
  | rel (id : Nat)
  | other

/-- `edge_already_used`. -/
def edgeAlreadyUsed (env : Nat → Option Slot) (edge own : Nat) : List Nat → Bool
  | [] => false
  | s :: ss =>
    if s = own then edgeAlreadyUsed env edge own ss
    else match env s with
      | some (.rel r) => if r = edge then true else edgeAlreadyUsed env edge own ss
      | _ => edgeAlreadyUsed env edge own ss

/-- **PROVEN**: `edge_already_used` is exactly "some *other* sibling relationship variable
of the same MATCH component is already bound to this edge" (Cypher relationship
uniqueness). -/
theorem edgeAlreadyUsed_iff (env : Nat → Option Slot) (edge own : Nat) :
    ∀ sib : List Nat, edgeAlreadyUsed env edge own sib = true ↔
      ∃ s ∈ sib, s ≠ own ∧ env s = some (.rel edge)
  | [] => by simp [edgeAlreadyUsed]
  | s :: ss => by
    have ih := edgeAlreadyUsed_iff env edge own ss
    simp only [edgeAlreadyUsed]
    by_cases hs : s = own
    · simp only [hs, ite_true, ih, List.mem_cons]
      constructor
      · rintro ⟨x, hx, h1, h2⟩; exact ⟨x, Or.inr hx, h1, h2⟩
      · rintro ⟨x, hx | hx, h1, h2⟩
        · exact absurd hx h1
        · exact ⟨x, hx, h1, h2⟩
    · simp only [hs, ite_false]
      cases he : env s with
      | none =>
        simp only [ih, List.mem_cons]
        constructor
        · rintro ⟨x, hx, h1, h2⟩; exact ⟨x, Or.inr hx, h1, h2⟩
        · rintro ⟨x, hx | hx, h1, h2⟩
          · subst hx; rw [he] at h2; simp at h2
          · exact ⟨x, hx, h1, h2⟩
      | some sl =>
        cases sl with
        | other =>
          simp only [ih, List.mem_cons]
          constructor
          · rintro ⟨x, hx, h1, h2⟩; exact ⟨x, Or.inr hx, h1, h2⟩
          · rintro ⟨x, hx | hx, h1, h2⟩
            · subst hx; rw [he] at h2; simp at h2
            · exact ⟨x, hx, h1, h2⟩
        | rel r =>
          by_cases hr : r = edge
          · subst hr; simp only [ite_true, true_iff]; exact ⟨s, by simp, hs, he⟩
          · simp only [hr, ite_false, ih, List.mem_cons]
            constructor
            · rintro ⟨x, hx, h1, h2⟩; exact ⟨x, Or.inr hx, h1, h2⟩
            · rintro ⟨x, hx | hx, h1, h2⟩
              · subst hx; rw [he] at h2; simp at h2; exact absurd h2 hr
              · exact ⟨x, hx, h1, h2⟩

/-! ### `map_to_index_options` (vector arm, phonetic) -/

inductive V where
  | int (i : Int)
  | float
  | bool (b : Bool)
  | str (s : String)
  | other

/-- One integer option (`dimension`, `M`, `efConstruction`, `efRuntime`): absent → `none`,
non-negative `Int` → its value, negative → "must be a non-negative integer", anything
else → "must be an integer". -/
def natOpt (key : String) : Option V → Except String (Option Nat)
  | none => .ok none
  | some (.int n) => if n < 0 then .error s!"{key}: negative" else .ok (some n.toNat)
  | some _ => .error s!"{key}: not an integer"

structure VecOpts where
  dimension : Nat
  similarity : Option String
  m : Option Nat
  efConstruction : Option Nat
  efRuntime : Option Nat

/-- The `IndexType::Vector` arm (`runtime.rs:1970-2073`, #3094 / `fe619ac5f`): `dimension`
and `similarityFunction` are **required** (they used to default to 0 / `None`), and the
similarity name is stored as `to_ascii_lowercase()` (Lean's `String.toLower` is ASCII-only). -/
def vectorOptions (get : String → Option V) : Except String VecOpts := do
  let d ← match get "dimension" with
    | some (.int n) => if n < 0 then throw "dimension: negative" else pure n.toNat
    | none => throw "dimension is required"
    | some _ => throw "dimension: not an integer"
  let sf ← match get "similarityFunction" with
    | some (.str s) => pure (some s.toLower)
    | none => throw "similarityFunction is required"
    | some _ => throw "similarityFunction must be a string"
  let m ← natOpt "M" (get "M")
  let efc ← natOpt "efConstruction" (get "efConstruction")
  let efr ← natOpt "efRuntime" (get "efRuntime")
  pure ⟨d, sf, m, efc, efr⟩

/-- **PROVEN** (#3094, fixes #3091): no `dimension`, no vector index. -/
theorem vector_requires_dimension (get : String → Option V) (h : get "dimension" = none) :
    vectorOptions get = .error "dimension is required" := by
  simp [vectorOptions, h, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]

/-- **PROVEN** (#3094): no `similarityFunction`, no vector index. -/
theorem vector_requires_similarity (get : String → Option V) (n : Int) (hn : 0 ≤ n)
    (hd : get "dimension" = some (.int n)) (h : get "similarityFunction" = none) :
    vectorOptions get = .error "similarityFunction is required" := by
  have : ¬ n < 0 := by omega
  simp [vectorOptions, hd, h, this, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure,
    Except.pure]

/-- **PROVEN** (#3094): an accepted similarity name is stored lower-cased, so
`'Euclidean'` reaches `register_fields` as `"euclidean"` (C's `strcasecmp`). -/
theorem vector_similarity_lowered (get : String → Option V) (o : VecOpts)
    (h : vectorOptions get = .ok o) : ∃ s, get "similarityFunction" = some (.str s) ∧ o.similarity = some s.toLower := by
  unfold vectorOptions at h
  cases hd : get "dimension" with
  | none => simp [hd, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  | some dv =>
    cases hs : get "similarityFunction" with
    | none =>
      cases dv <;> simp [hd, hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure] at h
      split at h <;> simp at h
    | some sv =>
      cases sv with
      | str s =>
        refine ⟨s, rfl, ?_⟩
        cases dv <;> simp [hd, hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure] at h
        split at h
        · simp at h
        · revert h
          cases natOpt "M" (get "M") <;> cases natOpt "efConstruction" (get "efConstruction") <;>
            cases natOpt "efRuntime" (get "efRuntime") <;> simp
          intro h; rw [← h]
      | _ =>
        cases dv <;> simp [hd, hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure] at h
        all_goals (split at h <;> simp at h)

/-- `FULLTEXT_OPTIONS` (`runtime.rs:1894-1895`) and the unknown-key refusal
(`runtime.rs:1896-1903`, #3094): the first key of the map, in order, outside it. -/
def fulltextOptions : List String := ["weight", "nostem", "phonetic", "language", "stopwords"]

def fulltextUnknown (keys : List String) : Option String := keys.find? (!fulltextOptions.contains ·)

theorem fulltextUnknown_none_iff (keys : List String) :
    fulltextUnknown keys = none ↔ ∀ k ∈ keys, k ∈ fulltextOptions := by
  simp [fulltextUnknown, List.find?_eq_none]

theorem natOpt_ok (key : String) (v : Option V) (r : Option Nat) :
    natOpt key v = .ok r ↔ (v = none ∧ r = none) ∨ ∃ n : Int, 0 ≤ n ∧ v = some (.int n) ∧ r = some n.toNat := by
  cases v with
  | none => simp [natOpt]; constructor <;> intro h <;> simp_all
  | some x =>
    cases x with
    | int n =>
      simp only [natOpt]
      by_cases hn : n < 0
      · simp [hn]; intro _ h; omega
      · simp [hn]; constructor
        · intro h; exact ⟨n, by omega, rfl, h.symm⟩
        · rintro ⟨_, _, rfl, rfl⟩; rfl
    | float | bool _ | str _ | other => simp [natOpt]

/-- **PROVEN**: every negative integer option is refused — the `*n as u64` / `as usize`
casts after the check never see a negative number. -/
theorem vector_negative_refused (get : String → Option V) (key : String)
    (hk : key ∈ ["dimension", "M", "efConstruction", "efRuntime"]) (n : Int) (hn : n < 0)
    (hg : get key = some (.int n)) : ∃ e, vectorOptions get = .error e := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  have hbad : natOpt key (get key) = .error s!"{key}: negative" := by simp [natOpt, hg, hn]
  simp only [vectorOptions]
  rcases hk with rfl | rfl | rfl | rfl <;>
  · simp only [bind, Except.bind, hbad]
    repeat' split
    all_goals (first | exact ⟨_, rfl⟩ | simp_all)

/-- The `phonetic` key (`runtime.rs:1924-1941`), with `eq_ignore_ascii_case` abstracted
as `isDmEn`. -/
def phonetic (isDmEn : String → Bool) : Option V → Except String (Option String)
  | none => .ok none
  | some (.bool true) => .ok (some "dm:en")
  | some (.bool false) => .ok (some "")
  | some (.str s) => if isDmEn s then .ok (some "dm:en") else .error s!"Unsupported phonetic algorithm '{s}'"
  | some _ => .error "Phonetic must be bool or string"

theorem phonetic_codes (isDmEn : String → Bool) (v : Option V) (s : String)
    (h : phonetic isDmEn v = .ok (some s)) : s = "dm:en" ∨ s = "" := by
  cases v with
  | none => simp [phonetic] at h
  | some x =>
    cases x with
    | bool b => cases b <;> simp [phonetic] at h <;> simp [h]
    | str t => simp only [phonetic] at h; split at h <;> simp_all
    | int _ | float | other => simp [phonetic] at h

end RuntimeCore.Misc
