/-
# Limits, `ops/mod.rs` helpers, index options

| here | there |
| --- | --- |
| `checkTimeout`       | `Runtime::check_timeout`, `runtime/runtime.rs:530-537` (deadline = `Instant::now() + timeout_ms` from `Runtime::new`, `:457`) |
| `asI64`, `checkMem`  | `Runtime::check_mem_capacity`, `runtime/runtime.rs:540-548` (`usage_fn() as i64`) |
| `drainPending`       | `drain_pending`, `runtime/ops/mod.rs:127-138` |
| `edgeAlreadyUsed`    | `edge_already_used`, `runtime/ops/mod.rs:261-295` (#3054; `edgeAlreadyUsedPre`: the pre-#3054 body) |
| `Parallel.hasAlternative` | `Parallel::has_alternative`, `runtime/ops/mod.rs:183-234` |
| `natOpt`, `vectorOptions` | the `IndexType::Vector` arm of `map_to_index_options`, `runtime/runtime.rs:2100-2178` |
| `fulltextUnknown`    | the unknown-key refusal of the `IndexType::Fulltext` arm, `runtime.rs:2021-2030` |
| `phonetic`           | the `phonetic` key of the `IndexType::Fulltext` arm, `runtime.rs:2051-2067` |
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
deriving DecidableEq

/-- `edge_already_used` before #3054 (`ops/mod.rs:155-172` at 343e077f2): plain relationship
uniqueness. The historical model; on #3054's code it is the case with no collapsed edge
(`edgeAlreadyUsed_emit`). -/
def edgeAlreadyUsedPre (env : Nat → Option Slot) (edge own : Nat) : List Nat → Bool
  | [] => false
  | s :: ss =>
    if s = own then edgeAlreadyUsedPre env edge own ss
    else match env s with
      | some (.rel r) => if r = edge then true else edgeAlreadyUsedPre env edge own ss
      | _ => edgeAlreadyUsedPre env edge own ss

/-- **PROVEN** (pre-#3054 model): "some *other* sibling relationship variable of the same MATCH
component is already bound to this edge" (Cypher relationship uniqueness). -/
theorem edgeAlreadyUsedPre_iff (env : Nat → Option Slot) (edge own : Nat) :
    ∀ sib : List Nat, edgeAlreadyUsedPre env edge own sib = true ↔
      ∃ s ∈ sib, s ≠ own ∧ env s = some (.rel edge)
  | [] => by simp [edgeAlreadyUsedPre]
  | s :: ss => by
    have ih := edgeAlreadyUsedPre_iff env edge own ss
    simp only [edgeAlreadyUsedPre]
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

/-! #### #3054: collapsed siblings and `Parallel::has_alternative` -/

/-- `COLLAPSED_SIBLING` / `COLLAPSED_SIBLING_ANY_TYPE` (`ops/mod.rs:146,153`): flag bits of a
`u32` `sibling_edges` entry. -/
def COLLAPSED_SIBLING : Nat := 2 ^ 31
def COLLAPSED_SIBLING_ANY_TYPE : Nat := 2 ^ 30

/-- `entry & !SIBLING_FLAGS` on a `u32` entry (`ops/mod.rs:269`). -/
def entryId (entry : Nat) : Nat := entry % 2 ^ 30
/-- `entry & COLLAPSED_SIBLING_ANY_TYPE != 0`. -/
def anyFlag (entry : Nat) : Bool := entry / 2 ^ 30 % 2 == 1
/-- `entry & COLLAPSED_SIBLING != 0`. -/
def oneFlag (entry : Nat) : Bool := entry / 2 ^ 31 % 2 == 1

/-- A stored relationship: id, endpoints (`endpoints_for_edge`) and tensor index (its type,
`relationship_type_matrix_iter(raw, raw)`). -/
structure Rel where
  id : Nat
  src : Nat
  dst : Nat
  ty : Nat

/-- `Parallel` (`ops/mod.rs:160-167`): the graph's relationships, the number of relationship
tensors, the hop's own tensors when it is collapsed (`None` when it emits), and `deleted`. -/
structure Parallel where
  rels : List Rel
  ntensors : Nat
  ownTypes : Option (List Nat)
  deleted : Nat → Bool

/-- `Matching` (`ops/mod.rs:171-179`). -/
inductive Matching where
  | tensors (ts : List Nat)
  | typeOfEdge
  | anyType

/-- Live edges of `(s, d)` in tensor `t`, uncapped. -/
def Parallel.liveRaw (P : Parallel) (s d t : Nat) : Nat :=
  if t < P.ntensors then
    (P.rels.filter fun x => x.ty == t && x.src == s && x.dst == d && !P.deleted x.id).length
  else 0

/-- The `live` closure (`ops/mod.rs:197-205`): `get(tidx).map_or(0, …)`, `.take(2).count()`. -/
def Parallel.live (P : Parallel) (s d t : Nat) : Nat := min 2 (P.liveRaw s d t)

/-- `Parallel::has_alternative` (`ops/mod.rs:183-234`). The `break` once `total > 1` only
stops adding (non-negative) counts, so it does not change `total > 1`; the sums below run to
the end. -/
def Parallel.hasAlternative (P : Parallel) (edge : Nat) (m : Matching) : Bool :=
  match P.rels.find? (·.id == edge) with
  | none => false
  | some x =>
    let total := match m with
      | .tensors ts => (ts.map (P.live x.src x.dst)).sum
      | .typeOfEdge => P.live x.src x.dst x.ty
      | .anyType => ((List.range P.ntensors).map (P.live x.src x.dst)).sum
    decide (1 < total)

theorem min2_sum (l : List Nat) : min 2 (l.map (min 2)).sum = min 2 l.sum := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [List.map_cons, List.sum_cons]; omega

/-- **PROVEN**: the `take(2)` cap does not change the answer: `has_alternative` holds iff more
than one live edge of the matched types joins the edge's endpoints. -/
theorem hasAlternative_uncapped (P : Parallel) (edge : Nat) (m : Matching) :
    P.hasAlternative edge m = match P.rels.find? (·.id == edge) with
      | none => false
      | some x => decide (1 < match m with
        | .tensors ts => (ts.map (P.liveRaw x.src x.dst)).sum
        | .typeOfEdge => P.liveRaw x.src x.dst x.ty
        | .anyType => ((List.range P.ntensors).map (P.liveRaw x.src x.dst)).sum) := by
  unfold Parallel.hasAlternative
  split
  · rfl
  · rename_i x _
    have key : ∀ l : List Nat, (1 < (l.map (P.live x.src x.dst)).sum ↔
        1 < (l.map (P.liveRaw x.src x.dst)).sum) := by
      intro l
      have h := min2_sum (l.map (P.liveRaw x.src x.dst))
      have e : (l.map (P.live x.src x.dst)) = (l.map (P.liveRaw x.src x.dst)).map (min 2) := by
        simp [List.map_map, Function.comp_def, Parallel.live]
      rw [e]; omega
    cases m with
    | tensors ts => simp only [decide_eq_decide]; exact key ts
    | typeOfEdge =>
      have hm : 1 < P.live x.src x.dst x.ty ↔ 1 < P.liveRaw x.src x.dst x.ty := by
        simp only [Parallel.live]; omega
      simp only [hm]
    | anyType => simp only [decide_eq_decide]; exact key _

/-- `own_avoids` (`ops/mod.rs:278-280`). -/
def ownAvoids (P : Parallel) (edge : Nat) : Bool :=
  match P.ownTypes with
  | some ts => P.hasAlternative edge (.tensors ts)
  | none => false

/-- `sibling_avoids` (`ops/mod.rs:281-287`). -/
def sibAvoids (P : Parallel) (edge entry : Nat) : Bool :=
  if anyFlag entry then P.hasAlternative edge .anyType
  else if oneFlag entry then P.hasAlternative edge .typeOfEdge
  else false

/-- `edge_already_used` (`ops/mod.rs:261-295`, #3054): a collision with a sibling is ignored when
the expanding (collapsed) hop or the colliding (collapsed) sibling has an alternative edge. -/
def edgeAlreadyUsed (P : Parallel) (env : Nat → Option Slot) (edge own : Nat) : List Nat → Bool
  | [] => false
  | entry :: ss =>
    if entryId entry = own then edgeAlreadyUsed P env edge own ss
    else match env (entryId entry) with
      | some (.rel r) =>
        if r = edge then
          if ownAvoids P edge || sibAvoids P edge entry then edgeAlreadyUsed P env edge own ss
          else true
        else edgeAlreadyUsed P env edge own ss
      | _ => edgeAlreadyUsed P env edge own ss

/-- **PROVEN** (#3054): `edge_already_used` is "some other sibling is bound to this edge, and
neither the expanding hop nor that sibling could have bound a parallel edge instead". -/
theorem edgeAlreadyUsed_spec (P : Parallel) (env : Nat → Option Slot) (edge own : Nat) :
    ∀ sib : List Nat, edgeAlreadyUsed P env edge own sib = true ↔
      ∃ s ∈ sib, entryId s ≠ own ∧ env (entryId s) = some (.rel edge) ∧
        ownAvoids P edge = false ∧ sibAvoids P edge s = false
  | [] => by simp [edgeAlreadyUsed]
  | s :: ss => by
    have ih := edgeAlreadyUsed_spec P env edge own ss
    simp only [edgeAlreadyUsed, List.mem_cons]
    have tail : (∃ x, (x = s ∨ x ∈ ss) ∧ entryId x ≠ own ∧ env (entryId x) = some (.rel edge) ∧
        ownAvoids P edge = false ∧ sibAvoids P edge x = false) ↔
        ((entryId s ≠ own ∧ env (entryId s) = some (.rel edge) ∧
          ownAvoids P edge = false ∧ sibAvoids P edge s = false) ∨
        ∃ x ∈ ss, entryId x ≠ own ∧ env (entryId x) = some (.rel edge) ∧
          ownAvoids P edge = false ∧ sibAvoids P edge x = false) := by
      constructor
      · rintro ⟨x, rfl | hx, h⟩
        · exact Or.inl h
        · exact Or.inr ⟨x, hx, h⟩
      · rintro (h | ⟨x, hx, h⟩)
        · exact ⟨s, Or.inl rfl, h⟩
        · exact ⟨x, Or.inr hx, h⟩
    rw [tail, ← ih]
    by_cases hs : entryId s = own
    · simp [hs]
    · simp only [hs, ite_false, ne_eq, not_false_eq_true, true_and]
      cases he : env (entryId s) with
      | none => simp
      | some sl =>
        cases sl with
        | other => simp
        | rel r =>
          by_cases hr : r = edge
          · subst hr
            cases ho : ownAvoids P r <;> cases hsa : sibAvoids P r s <;> simp
          · simp [hr]

/-- **PROVEN**: when the expanding hop emits and no sibling entry is flagged (no collapsed edge in
the MATCH component), #3054's `edge_already_used` is the pre-#3054 one, hence exactly Cypher
relationship uniqueness (`edgeAlreadyUsedPre_iff`). -/
theorem edgeAlreadyUsed_emit (P : Parallel) (env : Nat → Option Slot) (edge own : Nat)
    (hown : P.ownTypes = none) :
    ∀ sib : List Nat, (∀ s ∈ sib, s < 2 ^ 30) →
      edgeAlreadyUsed P env edge own sib = edgeAlreadyUsedPre env edge own sib
  | [], _ => rfl
  | s :: ss, h => by
    have ih := edgeAlreadyUsed_emit P env edge own hown ss (fun x hx => h x (List.mem_cons_of_mem _ hx))
    have hs : s < 2 ^ 30 := h s (by simp)
    have hid : entryId s = s := Nat.mod_eq_of_lt hs
    have hsa : sibAvoids P edge s = false := by
      have h1 : s / 2 ^ 30 = 0 := Nat.div_eq_of_lt hs
      have h2 : s / 2 ^ 31 = 0 := Nat.div_eq_of_lt (by omega)
      simp [sibAvoids, anyFlag, oneFlag, h1, h2]
    have hoa : ownAvoids P edge = false := by simp [ownAvoids, hown]
    simp only [edgeAlreadyUsed, edgeAlreadyUsedPre, hid, hsa, hoa, Bool.or_false, ih]
    rfl

/-! #### Counterexample (#3054)

`(a)-[r:R]->(x), (a)-[s:R]->(x), (a)-[t:R]->(x)` with none of `r, s, t` read: all three
collapse (`reduce_expand_into`), each carries `COLLAPSED_SIBLING` in the others' sibling lists.
On `a⇉x` (edges 1 and 2, type 0) Cypher has no match: three distinct edges out of two. Each
collapsed hop binds the first candidate `edge_already_used` accepts (`expand_into.rs:210-229`,
`cond_traverse.rs:1295-1318`). Before #3054: `r = 1`, `s = 2`, `t` finds none (0 rows). On
#3054: `s` takes `1` (it collides with `r`, but edge `2` is an alternative), then `t` takes `1`
as well (it collides with `r` and `s`, and `2` is still "an alternative", though `s` could only
have avoided `r` by taking it): one row. `has_alternative` asks whether *another* parallel
edge exists, not whether one is still free. -/

/-- `a⇉x`: relationships 1 and 2, both `0 → 1` of type 0. -/
def pTwo : Parallel := ⟨[⟨1, 0, 1, 0⟩, ⟨2, 0, 1, 0⟩], 1, some [0], fun _ => false⟩

/-- Slots `r = 0`, `s = 1`, `t = 2`, each flagged `COLLAPSED_SIBLING` (one type). -/
def sibTriple : List Nat := [COLLAPSED_SIBLING + 0, COLLAPSED_SIBLING + 1, COLLAPSED_SIBLING + 2]

/-- The row after binding `r` (slot 0) and then `s` (slot 1). -/
def envOf (binds : List (Nat × Nat)) (slot : Nat) : Option Slot :=
  (binds.find? (·.1 == slot)).map fun b => .rel b.2

/-- A collapsed hop's representative: the first candidate not already used. -/
def pick (P : Parallel) (env : Nat → Option Slot) (own : Nat) (sib cands : List Nat) :
    Option Nat :=
  cands.find? fun e => !edgeAlreadyUsed P env e own sib

def pickPre (env : Nat → Option Slot) (own : Nat) (sib cands : List Nat) : Option Nat :=
  cands.find? fun e => !edgeAlreadyUsedPre env e own sib

/-- **Counterexample (#3054)**: the three collapsed hops all bind edge 1 and the row survives,
where the pre-#3054 code (correctly) finds no edge for `t`, and no three distinct edges exist. -/
theorem collapsed_triple_reuses_edge :
    pick pTwo (envOf []) 0 sibTriple [1, 2] = some 1 ∧
    pick pTwo (envOf [(0, 1)]) 1 sibTriple [1, 2] = some 1 ∧
    pick pTwo (envOf [(0, 1), (1, 1)]) 2 sibTriple [1, 2] = some 1 ∧
    pickPre (envOf []) 0 [0, 1, 2] [1, 2] = some 1 ∧
    pickPre (envOf [(0, 1)]) 1 [0, 1, 2] [1, 2] = some 2 ∧
    pickPre (envOf [(0, 1), (1, 2)]) 2 [0, 1, 2] [1, 2] = none ∧
    ¬ ∃ r ∈ [1, 2], ∃ s ∈ [1, 2], ∃ t ∈ [1, 2], r ≠ s ∧ r ≠ t ∧ s ≠ t := by
  decide

/-- **Counterexample (#3054)** to `edgeAlreadyUsed_iff` (proved before #3054, now
`edgeAlreadyUsedPre_iff`): on #3054's code `edge_already_used` is no longer relationship
uniqueness. At the `t` hop above, siblings `r` and `s` are both bound to edge 1, yet edge 1 is
reported unused. -/
theorem edgeAlreadyUsed_iff_fails :
    ¬ ∀ (P : Parallel) (env : Nat → Option Slot) (edge own : Nat) (sib : List Nat),
      edgeAlreadyUsed P env edge own sib = true ↔
        ∃ s ∈ sib, entryId s ≠ own ∧ env (entryId s) = some (.rel edge) := by
  intro h
  have := (h pTwo (envOf [(0, 1), (1, 1)]) 1 2 sibTriple).2
    ⟨COLLAPSED_SIBLING + 0, by decide, by decide, by decide⟩
  revert this; decide

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

/-- The `IndexType::Vector` arm (`runtime.rs:2100-2178`, #3094 / `fe619ac5f`): `dimension`
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

/-- `FULLTEXT_OPTIONS` (`runtime.rs:2021-2022`) and the unknown-key refusal
(`runtime.rs:2023-2030`, #3094): the first key of the map, in order, outside it. -/
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

/-- The `phonetic` key (`runtime.rs:2051-2067`), with `eq_ignore_ascii_case` abstracted
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
