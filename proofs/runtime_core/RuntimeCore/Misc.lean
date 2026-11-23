/-
# Limits, `ops/mod.rs` helpers, index options

| here | there |
| --- | --- |
| `checkTimeout`       | `Runtime::check_timeout`, `runtime/runtime.rs` (deadline = `Instant::now() + timeout_ms` from `Runtime::new`) |
| `asI64`, `checkMem`  | `Runtime::check_mem_capacity`, `runtime/runtime.rs` (`usage_fn() as i64`) |
| `drainPending`       | `drain_pending`, `runtime/ops/mod.rs` |
| `edgeAlreadyUsed`    | `edge_already_used`, `runtime/ops/mod.rs`, when nothing is collapsed (`edgeAlreadyUsedP_plain`) |
| `edgeAlreadyUsedP`, `hasAlt`, `typedDemandExceeded` | `edge_already_used`, `Parallel::has_alternative` and `Parallel::typed_demand_exceeded` since #3054, `runtime/ops/mod.rs` |
| `natOpt`, `vectorOptions`, `simNames` | the `IndexType::Vector` arm of `map_to_index_options`, `runtime/runtime.rs` (name check) |
| `fulltextUnknown`    | the unknown-key refusal of the `IndexType::Fulltext` arm, `runtime.rs` |
| `phonetic`           | the `phonetic` key of the `IndexType::Fulltext` arm, `runtime.rs` |
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
  simp only [checkMem, asI64, h1, ite_eq_left hu, Bool.and_eq_true, decide_eq_true_eq]

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

/-- The scan of `edge_already_used` over plain (unflagged) siblings: the whole function
when nothing is collapsed (`edgeAlreadyUsedP_plain`). -/
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

/-! ### `edge_already_used` with collapsed edges (#3054)

Since #3054 `edge_already_used` also takes a `Parallel` and `sibling_edges`
entries may carry a flag (`COLLAPSED_SIBLING`, bit 31: a collapsed edge of one
type; `COLLAPSED_SIBLING_ANY_TYPE`, bit 30: a collapsed untyped edge). An entry
is modelled as its alias id (`entry & !SIBLING_FLAGS`) and its flag; alias ids
are below `2^30`, so the two never overlap. `edgeAlreadyUsed` above is the
scan with no flag and no collapsed own hop (`edgeAlreadyUsedP_plain`). -/

inductive Flag where
  | none
  | single
  | any
deriving DecidableEq, Repr

structure Entry where
  id : Nat
  flag : Flag
deriving DecidableEq, Repr

/-- A stored relationship: id, endpoints, and the tensor (type) it lives in. -/
structure Rel where
  id : Nat
  src : Nat
  dst : Nat
  ty : Nat
deriving DecidableEq, Repr

/-- `Parallel`: the graph (`rels`, `ntypes` = `relationship_tensors().len()`),
`own_types` (the tensors of the expanding hop when it is collapsed) and
`deleted` (the query's pending relationship deletes). -/
structure Par where
  rels : List Rel
  ntypes : Nat
  ownTypes : Option (List Nat)
  deleted : Nat → Bool

/-- `Matching`. -/
inductive Matching where
  | tensors (ts : List Nat)
  | typeOfEdge
  | anyType

/-- `endpoints_for_edge` and the edge's tensor (`relationship_type_matrix_iter`). -/
def Par.find (P : Par) (e : Nat) : Option Rel := P.rels.find? (·.id == e)

/-- The `matches` closure of `has_alternative`. -/
def Matching.ok (P : Par) (ownT : Option Nat) (t : Nat) : Matching → Bool
  | .tensors ts => ts.contains t
  | .typeOfEdge => ownT == some t
  | .anyType => decide (t < P.ntypes)

/-- The `taken` loop of `has_alternative`: other hops bound to a live edge of
the pair `(src, dst)` (stored direction) whose type matches. -/
def taken (P : Par) (env : Nat → Option Slot) (src dst own : Nat) (ownT : Option Nat)
    (m : Matching) (sibs : List Entry) : Nat :=
  sibs.countP fun en =>
    en.id != own && match env en.id with
      | some (.rel b) => !P.deleted b && match P.find b with
          | some rb => rb.src == src && rb.dst == dst && m.ok P ownT rb.ty
          | none => false
      | _ => false

/-- The `live` closure: live edges of tensor `t` on the pair, stopping at `k + 1`. -/
def live (P : Par) (src dst k t : Nat) : Nat :=
  if t < P.ntypes then
    min ((P.rels.filter fun r => r.src == src && r.dst == dst && r.ty == t && !P.deleted r.id).length)
      (k + 1)
  else 0

/-- `total += live(tidx); if total > taken { break }`. -/
def sumBreak (k : Nat) : List Nat → Nat → Nat
  | [], acc => acc
  | x :: xs, acc => if acc + x > k then acc + x else sumBreak k xs (acc + x)

def total (P : Par) (src dst k : Nat) (ownT : Option Nat) : Matching → Nat
  | .tensors ts => sumBreak k (ts.map (live P src dst k)) 0
  | .typeOfEdge => match ownT with
    | some t => live P src dst k t
    | none => 0
  | .anyType => sumBreak k ((List.range P.ntypes).map (live P src dst k)) 0

/-- `Parallel::has_alternative`. -/
def hasAlt (P : Par) (env : Nat → Option Slot) (edge own : Nat) (sibs : List Entry)
    (m : Matching) : Bool :=
  match P.find edge with
  | none => false
  | some e =>
    let k := taken P env e.src e.dst own (some e.ty) m sibs
    if k = 0 then true else decide (total P e.src e.dst k (some e.ty) m > k)

/-- Live edges of tensor `t` on the pair, uncapped (spec side). -/
def liveAll (P : Par) (src dst t : Nat) : Nat :=
  if t < P.ntypes then
    (P.rels.filter fun r => r.src == src && r.dst == dst && r.ty == t && !P.deleted r.id).length
  else 0

/-- Live edges of the pair whose type matches, uncapped (spec side). -/
def totalAll (P : Par) (src dst : Nat) (ownT : Option Nat) : Matching → Nat
  | .tensors ts => (ts.map (liveAll P src dst)).sum
  | .typeOfEdge => match ownT with
    | some t => liveAll P src dst t
    | none => 0
  | .anyType => ((List.range P.ntypes).map (liveAll P src dst)).sum

theorem live_eq (P : Par) (src dst k t : Nat) :
    live P src dst k t = min (liveAll P src dst t) (k + 1) := by
  unfold live liveAll; split <;> simp

theorem sumBreak_gt (k : Nat) (f : Nat → Nat) :
    ∀ (xs : List Nat) (acc : Nat),
      sumBreak k (xs.map fun t => min (f t) (k + 1)) acc > k ↔ acc + (xs.map f).sum > k
  | [], acc => by simp [sumBreak]
  | x :: xs, acc => by
    have ih := sumBreak_gt k f xs (acc + min (f x) (k + 1))
    simp only [List.map_cons, sumBreak, List.sum_cons]
    split
    · next h => constructor <;> intro <;> omega
    · next h =>
      rw [ih]
      have : min (f x) (k + 1) = f x := by omega
      rw [this]; omega

/-- **PROVEN**: the early exits of `has_alternative` (`take(taken + 1)` and the `break`)
do not change its answer: it holds iff the edge exists and either no other hop is bound
to a matching edge of its pair, or the pair has more live matching edges than those hops. -/
theorem hasAlt_spec (P : Par) (env : Nat → Option Slot) (edge own : Nat) (sibs : List Entry)
    (m : Matching) :
    hasAlt P env edge own sibs m = true ↔
      ∃ e, P.find edge = some e ∧
        (taken P env e.src e.dst own (some e.ty) m sibs = 0 ∨
          taken P env e.src e.dst own (some e.ty) m sibs < totalAll P e.src e.dst (some e.ty) m) := by
  unfold hasAlt
  cases hf : P.find edge with
  | none => simp
  | some e =>
    simp only [Option.some.injEq, exists_eq_left']
    generalize taken P env e.src e.dst own (some e.ty) m sibs = k
    by_cases hk : k = 0
    · simp [hk]
    · simp only [hk, ite_false, decide_eq_true_eq, false_or]
      have hl : live P e.src e.dst k = fun t => min (liveAll P e.src e.dst t) (k + 1) := by
        funext t; exact live_eq P e.src e.dst k t
      cases m with
      | tensors ts =>
        simp only [total, totalAll, hl]
        rw [sumBreak_gt k (liveAll P e.src e.dst) ts 0]; omega
      | typeOfEdge =>
        simp only [total, totalAll, hl]; omega
      | anyType =>
        simp only [total, totalAll, hl]
        rw [sumBreak_gt k (liveAll P e.src e.dst) _ 0]; omega

/-- `own_avoids`. -/
def ownAvoids (P : Par) (env : Nat → Option Slot) (edge own : Nat) (sibs : List Entry) : Bool :=
  match P.ownTypes with
  | some ts => hasAlt P env edge own sibs (.tensors ts)
  | none => false

/-- `sibling_avoids`. -/
def sibAvoids (P : Par) (env : Nat → Option Slot) (edge own : Nat) (sibs : List Entry)
    (en : Entry) : Bool :=
  match en.flag with
  | .any => hasAlt P env edge own sibs .anyType
  | .single => hasAlt P env edge own sibs .typeOfEdge
  | .none => false

/-- The loop of `edge_already_used`: `true` at the first collision neither side avoids. -/
def collisions (P : Par) (env : Nat → Option Slot) (edge own : Nat) (full : List Entry) :
    List Entry → Bool
  | [] => false
  | en :: rest =>
    if en.id = own then collisions P env edge own full rest
    else match env en.id with
      | some (.rel r) =>
        if r = edge then
          if ownAvoids P env edge own full || sibAvoids P env edge own full en then
            collisions P env edge own full rest
          else true
        else collisions P env edge own full rest
      | _ => collisions P env edge own full rest

/-- The types of the final `has_alternative` call. -/
def finalMatching (P : Par) (sibs : List Entry) : Matching :=
  match P.ownTypes with
  | some ts => .tensors ts
  | none => if sibs.any (·.flag == .any) then .anyType else .typeOfEdge

def anyCollapsed (P : Par) (sibs : List Entry) : Bool :=
  P.ownTypes.isSome || sibs.any (·.flag != .none)

/-- The `demand` loop of `Parallel::typed_demand_exceeded`: this hop plus every other
`COLLAPSED_SIBLING` (single-type) hop bound to a live edge of the pair `(src, dst)` of
tensor `t`. -/
def typedDemand (P : Par) (env : Nat → Option Slot) (src dst own t : Nat)
    (sibs : List Entry) : Nat :=
  1 + sibs.countP fun en =>
    en.id != own && en.flag == .single && match env en.id with
      | some (.rel b) => !P.deleted b && match P.find b with
          | some rb => rb.src == src && rb.dst == dst && rb.ty == t
          | none => false
      | _ => false

/-- The early `return false` of `typed_demand_exceeded`: the hop is collapsed over a
number of tensors other than one. -/
def multiOwn (P : Par) : Bool :=
  match P.ownTypes with
  | some ts => ts.length != 1
  | none => false

/-- `Parallel::typed_demand_exceeded`: the live edges of `edge`'s type on its pair,
counted up to `demand` (`take(demand)`), are fewer than the hops that need one. -/
def typedDemandExceeded (P : Par) (env : Nat → Option Slot) (edge own : Nat)
    (sibs : List Entry) : Bool :=
  if multiOwn P then false
  else match P.find edge with
    | none => false
    | some e =>
      let d := typedDemand P env e.src e.dst own e.ty sibs
      decide (d > live P e.src e.dst (d - 1) e.ty)

/-- **PROVEN**: the `take(demand)` early exit does not change the answer:
`typed_demand_exceeded` holds iff the hop is not collapsed over several tensors, the edge
exists, and the pair holds fewer live edges of its type than the single-type hops bound
to it (this one included). -/
theorem typedDemandExceeded_spec (P : Par) (env : Nat → Option Slot) (edge own : Nat)
    (sibs : List Entry) :
    typedDemandExceeded P env edge own sibs = true ↔
      multiOwn P = false ∧
      ∃ e, P.find edge = some e ∧
        liveAll P e.src e.dst e.ty < typedDemand P env e.src e.dst own e.ty sibs := by
  unfold typedDemandExceeded
  cases hm : multiOwn P
  · simp only [Bool.false_eq_true, ite_false, true_and]
    cases P.find edge with
    | none => simp
    | some e =>
      simp only [Option.some.injEq, exists_eq_left', decide_eq_true_eq, live_eq]
      have : 1 ≤ typedDemand P env e.src e.dst own e.ty sibs := by
        unfold typedDemand; omega
      omega
  · simp

/-- `edge_already_used` (#3054). -/
def edgeAlreadyUsedP (P : Par) (env : Nat → Option Slot) (edge own : Nat) (sibs : List Entry) :
    Bool :=
  collisions P env edge own sibs sibs ||
    (anyCollapsed P sibs && (typedDemandExceeded P env edge own sibs ||
      !hasAlt P env edge own sibs (finalMatching P sibs)))

theorem collisions_iff (P : Par) (env : Nat → Option Slot) (edge own : Nat) (full : List Entry) :
    ∀ L : List Entry, collisions P env edge own full L = true ↔
      ∃ en ∈ L, en.id ≠ own ∧ env en.id = some (.rel edge) ∧
        ownAvoids P env edge own full = false ∧ sibAvoids P env edge own full en = false
  | [] => by simp [collisions]
  | en :: rest => by
    have ih := collisions_iff P env edge own full rest
    simp only [collisions]
    by_cases hs : en.id = own
    · simp only [hs, ite_true, ih, List.mem_cons]
      constructor
      · rintro ⟨x, hx, h⟩; exact ⟨x, Or.inr hx, h⟩
      · rintro ⟨x, hx | hx, h1, h⟩
        · subst hx; exact absurd hs h1
        · exact ⟨x, hx, h1, h⟩
    · simp only [hs, ite_false]
      have tail : (∃ x ∈ en :: rest, x.id ≠ own ∧ env x.id = some (.rel edge) ∧
          ownAvoids P env edge own full = false ∧ sibAvoids P env edge own full x = false) ↔
          ((env en.id = some (.rel edge) ∧ ownAvoids P env edge own full = false ∧
            sibAvoids P env edge own full en = false) ∨
          ∃ x ∈ rest, x.id ≠ own ∧ env x.id = some (.rel edge) ∧
            ownAvoids P env edge own full = false ∧ sibAvoids P env edge own full x = false) := by
        constructor
        · rintro ⟨x, hx, h⟩
          rcases List.mem_cons.1 hx with rfl | hx
          · exact Or.inl h.2
          · exact Or.inr ⟨x, hx, h⟩
        · rintro (h | ⟨x, hx, h⟩)
          · exact ⟨en, List.mem_cons_self, hs, h⟩
          · exact ⟨x, List.mem_cons_of_mem _ hx, h⟩
      rw [tail, ← ih]
      cases he : env en.id with
      | none => simp
      | some sl =>
        cases sl with
        | other => simp
        | rel r =>
          by_cases hr : r = edge
          · subst hr
            cases ho : ownAvoids P env r own full <;>
              cases hsa : sibAvoids P env r own full en <;> simp
          · simp [hr]

/-- **PROVEN**: `edge_already_used` since #3054 rejects `edge` exactly when some other
hop of the clause is bound to it and neither the expanding hop (when collapsed) nor that
sibling (when flagged collapsed) has an alternative edge on the pair, or when some hop is
collapsed and either the single-type hops bound to the pair need more edges of the edge's
type than it holds (`typed_demand_exceeded`) or the pair has no edge left for one more hop
(`has_alternative`). -/
theorem edgeAlreadyUsedP_iff (P : Par) (env : Nat → Option Slot) (edge own : Nat)
    (sibs : List Entry) :
    edgeAlreadyUsedP P env edge own sibs = true ↔
      (∃ en ∈ sibs, en.id ≠ own ∧ env en.id = some (.rel edge) ∧
        ownAvoids P env edge own sibs = false ∧ sibAvoids P env edge own sibs en = false) ∨
      (anyCollapsed P sibs = true ∧ (typedDemandExceeded P env edge own sibs = true ∨
        hasAlt P env edge own sibs (finalMatching P sibs) = false)) := by
  simp only [edgeAlreadyUsedP, Bool.or_eq_true, Bool.and_eq_true, Bool.not_eq_true',
    collisions_iff]

theorem collisions_plain (P : Par) (env : Nat → Option Slot) (edge own : Nat) (full : List Entry)
    (ho : P.ownTypes = none) :
    ∀ L : List Entry, (∀ en ∈ L, en.flag = .none) →
      collisions P env edge own full L = edgeAlreadyUsed env edge own (L.map (·.id))
  | [], _ => rfl
  | en :: rest, hf => by
    have ih := collisions_plain P env edge own full ho rest
      (fun x hx => hf x (List.mem_cons_of_mem _ hx))
    have hen : en.flag = .none := hf en List.mem_cons_self
    have hown : ownAvoids P env edge own full = false := by simp [ownAvoids, ho]
    have hsib : sibAvoids P env edge own full en = false := by simp [sibAvoids, hen]
    simp only [collisions, edgeAlreadyUsed, List.map_cons, hown, hsib, Bool.false_or, ih]
    split <;> simp

/-- **PROVEN**: when nothing is collapsed (the expanding hop emits and no sibling entry is
flagged) the new `edge_already_used` is the plain scan, so `edgeAlreadyUsed_iff` (Cypher
relationship uniqueness) holds unchanged for it. -/
theorem edgeAlreadyUsedP_plain (P : Par) (env : Nat → Option Slot) (edge own : Nat)
    (sibs : List Entry) (ho : P.ownTypes = none) (hf : ∀ en ∈ sibs, en.flag = .none) :
    edgeAlreadyUsedP P env edge own sibs = edgeAlreadyUsed env edge own (sibs.map (·.id)) := by
  have hc : anyCollapsed P sibs = false := by
    simp only [anyCollapsed, ho, Option.isSome_none, Bool.false_or, List.any_eq_false]
    intro x hx; simp [hf x hx]
  simp only [edgeAlreadyUsedP, collisions_plain P env edge own sibs ho sibs hf, hc,
    Bool.false_and, Bool.or_false]

theorem edgeAlreadyUsedP_plain_iff (P : Par) (env : Nat → Option Slot) (edge own : Nat)
    (sibs : List Entry) (ho : P.ownTypes = none) (hf : ∀ en ∈ sibs, en.flag = .none) :
    edgeAlreadyUsedP P env edge own sibs = true ↔
      ∃ en ∈ sibs, en.id ≠ own ∧ env en.id = some (.rel edge) := by
  rw [edgeAlreadyUsedP_plain P env edge own sibs ho hf, edgeAlreadyUsed_iff]
  constructor
  · rintro ⟨_, hm, h1, h2⟩
    obtain ⟨en, hen, rfl⟩ := List.mem_map.1 hm
    exact ⟨en, hen, h1, h2⟩
  · rintro ⟨en, hen, h1, h2⟩
    exact ⟨en.id, List.mem_map.2 ⟨en, hen, rfl⟩, h1, h2⟩

/-! #### The #3047 repro: `MATCH (a)-[r]->(x)<-[s]-(c)` with `r`, `s` collapsed

`e0, e1 : 1→2`, `e2 : 2→3`, one type. `r` (alias 0, untyped: flag `any`) is bound to its
representative; `s` (alias 1) is collapsed over tensor 0. -/

def g3047 (pending : Nat → Bool) : Par :=
  ⟨[⟨0, 1, 2, 0⟩, ⟨1, 1, 2, 0⟩, ⟨2, 2, 3, 0⟩], 1, some [0], pending⟩

def envR (e : Nat) : Nat → Option Slot := fun v => if v = 0 then some (.rel e) else none

def sibs3047 : List Entry := [⟨0, .any⟩]

/-- **PROVEN** (#3047 fixed): `r`'s representative `e0` on a pair that also holds `e1` no
longer blocks `s`, whichever of the two `r` picked, so `s` binds a row for `(1,1)` either way. -/
theorem repro3047_parallel_free :
    edgeAlreadyUsedP (g3047 fun _ => false) (envR 0) 0 1 sibs3047 = false ∧
    edgeAlreadyUsedP (g3047 fun _ => false) (envR 1) 1 1 sibs3047 = false := by decide

/-- **PROVEN**: on the single-edge pair `2→3` the representative is the edge itself and
uniqueness still applies (no `(2,2)` row, as before #3054). -/
theorem repro3047_single_kept :
    edgeAlreadyUsedP (g3047 fun _ => false) (envR 2) 2 1 sibs3047 = true := by decide

/-- **PROVEN**: a parallel edge deleted by the query is no alternative. -/
theorem repro3047_deleted_parallel :
    edgeAlreadyUsedP (g3047 fun e => e == 1) (envR 0) 0 1 sibs3047 = true := by decide

/-! #### Still open: a typed hop next to a collapsed untyped one

`R1 = e0 : 1→2` of type 0 (`R`), `T1 = e1 : 1→2` of type 1 (`T`).
`MATCH (a)-[s]->(x)<-[r:R]-(a)`: `s` (alias 0, untyped, collapsed: flag `any`) is
traversed first and binds one representative edge of the pair; `r` (alias 1, tensor 0,
collapsed) is expanded next. The final `has_alternative` counts `s` as holding the pair's
only `R` edge, although `s` could have taken `T1`. -/

def gRT : Par := ⟨[⟨0, 1, 2, 0⟩, ⟨1, 1, 2, 1⟩], 2, some [0], fun _ => false⟩

/-- **Open** (W2-traverse-12): whether `r` finds an edge depends on which representative
`s` holds: with `s = R1`, `R1` is rejected and `r` has no other `R` edge, so the row is
lost; with `s = T1` it is kept. The pattern has a match (`s = T1`, `r = R1`). Confirmed
live on the base (41d2f20ce) and on #3054's head: on `CREATE (a)-[:R]->(x), (a)-[:T]->(x)`
the runtime binds `s` to the `T` edge, so `MATCH (a)-[s]->(x)<-[r:T]-(a) RETURN count(*)`
gives 0 (expected 1) while the `r:R` query gives 1. Not a regression of #3054. -/
theorem typed_hop_depends_on_representative :
    edgeAlreadyUsedP gRT (envR 0) 0 1 [⟨0, .any⟩] = true ∧
    edgeAlreadyUsedP gRT (envR 1) 0 1 [⟨0, .any⟩] = false := by decide

/-! #### Three `R` hops over two `R` edges next to an untyped hop (W2-traverse-13, fixed)

`R0 = e0`, `R1 = e1` of type 0 (`R`), `T2 = e2`, `T3 = e3` of type 1 (`T`), all `1→2`.
`MATCH (a)-[r:R]->(x), (a)-[s:R]->(x), (a)-[t]->(x), (a)-[u:R]->(x) RETURN id(u)`:
`r`, `s` (aliases 0, 1, collapsed, flag `single`), `t` (alias 2, untyped, collapsed, flag
`any`) and `u` (alias 3, read, emitted), expanded in that order. Three `R` hops need three
distinct `R` edges and the pair has two, so the pattern has no match. `s` may still share
`R0` with `r` (`own_avoids`: the pair holds `R1`), but `typed_demand_exceeded` charges `r`
and `s` against the `R` edges, so `u` finds none. -/

def gRRTT (own : Option (List Nat)) : Par :=
  ⟨[⟨0, 1, 2, 0⟩, ⟨1, 1, 2, 0⟩, ⟨2, 1, 2, 1⟩, ⟨3, 1, 2, 1⟩], 2, own, fun _ => false⟩

/-- The row's relationship slots, `alias ↦ edge`. -/
def envL (l : List (Nat × Nat)) : Nat → Option Slot := fun v => (l.lookup v).map .rel

/-- **PROVEN** (W2-traverse-13 fixed by `typed_demand_exceeded`): with `r` and `s` sharing
`R0`, `u` rejects both `R` edges, whichever edge the untyped `t` took (`R0` or `T2`), so
the clause emits no row, as Cypher requires (three `R` hops, two `R` edges). -/
theorem shared_representative_no_row :
    -- `s` (tensor `R` only) may still share `R0` with `r`
    edgeAlreadyUsedP (gRRTT (some [0])) (envL [(0, 0)]) 0 1
        [⟨0, .single⟩, ⟨1, .none⟩, ⟨2, .any⟩, ⟨3, .none⟩] = false ∧
    -- the pattern has no match: three `R` hops, two `R` edges
    ((gRRTT none).rels.filter (·.ty == 0)).length = 2 ∧
    -- `t` on `R0`: `u` rejects `R0` and `R1`
    edgeAlreadyUsedP (gRRTT none) (envL [(0, 0), (1, 0), (2, 0)]) 0 3
        [⟨0, .single⟩, ⟨1, .single⟩, ⟨2, .any⟩, ⟨3, .none⟩] = true ∧
    edgeAlreadyUsedP (gRRTT none) (envL [(0, 0), (1, 0), (2, 0)]) 1 3
        [⟨0, .single⟩, ⟨1, .single⟩, ⟨2, .any⟩, ⟨3, .none⟩] = true ∧
    -- `t` on `T2`: `u` rejects `R0` and `R1`
    edgeAlreadyUsedP (gRRTT none) (envL [(0, 0), (1, 0), (2, 2)]) 0 3
        [⟨0, .single⟩, ⟨1, .single⟩, ⟨2, .any⟩, ⟨3, .none⟩] = true ∧
    edgeAlreadyUsedP (gRRTT none) (envL [(0, 0), (1, 0), (2, 2)]) 1 3
        [⟨0, .single⟩, ⟨1, .single⟩, ⟨2, .any⟩, ⟨3, .none⟩] = true := by
  decide

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

/-- The similarity names the vector arm accepts (`runtime.rs`; `ip` is a Rust
extension over C). -/
def simNames : List String := ["euclidean", "cosine", "ip"]

/-- The `IndexType::Vector` arm (`runtime.rs`): `dimension`
and `similarityFunction` are **required**, and the
similarity name is stored as `to_ascii_lowercase()` (Lean's `String.toLower` is ASCII-only);
a lower-cased name outside `simNames` is refused here (`runtime.rs`). -/
def vectorOptions (get : String → Option V) : Except String VecOpts := do
  let d ← match get "dimension" with
    | some (.int n) => if n < 0 then throw "dimension: negative" else pure n.toNat
    | none => throw "dimension is required"
    | some _ => throw "dimension: not an integer"
  let sf ← match get "similarityFunction" with
    | some (.str s) =>
      if simNames.contains s.toLower then pure (some s.toLower)
      else throw s!"unknown similarityFunction '{s}'"
    | none => throw "similarityFunction is required"
    | some _ => throw "similarityFunction must be a string"
  let m ← natOpt "M" (get "M")
  let efc ← natOpt "efConstruction" (get "efConstruction")
  let efr ← natOpt "efRuntime" (get "efRuntime")
  pure ⟨d, sf, m, efc, efr⟩

/-- **PROVEN**: no `dimension`, no vector index. -/
theorem vector_requires_dimension (get : String → Option V) (h : get "dimension" = none) :
    vectorOptions get = .error "dimension is required" := by
  simp [vectorOptions, h, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]

/-- **PROVEN**: no `similarityFunction`, no vector index. -/
theorem vector_requires_similarity (get : String → Option V) (n : Int) (hn : 0 ≤ n)
    (hd : get "dimension" = some (.int n)) (h : get "similarityFunction" = none) :
    vectorOptions get = .error "similarityFunction is required" := by
  have : ¬ n < 0 := by omega
  simp [vectorOptions, hd, h, this, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure,
    Except.pure]

/-- **PROVEN**: an accepted similarity name is stored lower-cased, so
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
        · split at h
          · revert h
            cases natOpt "M" (get "M") <;> cases natOpt "efConstruction" (get "efConstruction") <;>
              cases natOpt "efRuntime" (get "efRuntime") <;> simp
            intro h; rw [← h]
          · simp at h
      | _ =>
        cases dv <;> simp [hd, hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure] at h
        all_goals (split at h <;> simp at h)

/-- **PROVEN**: an unknown similarity name is refused, even with `dimension: 0`
(`OPTIONS {dimension:0, similarityFunction:'bogus'}`). -/
theorem vector_unknown_similarity_refused (get : String → Option V) (s : String)
    (hs : get "similarityFunction" = some (.str s)) (hk : s.toLower ∉ simNames) :
    ∃ e, vectorOptions get = .error e := by
  unfold vectorOptions
  cases hd : get "dimension" with
  | none => simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw]
  | some dv =>
    cases dv <;> simp [hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure,
      Except.pure]
    split <;> simp

/-- **PROVEN**: an accepted similarity name is a known one, whatever the
dimension (`0` included). -/
theorem vector_similarity_known (get : String → Option V) (o : VecOpts)
    (h : vectorOptions get = .ok o) : ∃ s, o.similarity = some s ∧ s ∈ simNames := by
  obtain ⟨s, hs, ho⟩ := vector_similarity_lowered get o h
  refine ⟨s.toLower, ho, ?_⟩
  apply Classical.byContradiction
  intro hk
  obtain ⟨e, he⟩ := vector_unknown_similarity_refused get s hs hk
  rw [he] at h; cases h

/-- `FULLTEXT_OPTIONS` (`runtime.rs`) and the unknown-key refusal
(`runtime.rs`): the first key of the map, in order, outside it. -/
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

/-- The `phonetic` key (`runtime.rs`), with `eq_ignore_ascii_case` abstracted
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
