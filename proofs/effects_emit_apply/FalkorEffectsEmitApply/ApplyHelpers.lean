import FalkorEffectsEmitApply.Digests
/-!
# `apply.rs`'s checks and plumbing, function by function

| here | there (`graph/src/effects/v3/apply.rs`) |
| --- | --- |
| `AErr`, `ofString` | `ApplyError`, `From<String>` (`:43`) |
| `idSpaceErrorMap`, `fromNodeOp` | `id_space_error_map` (`:109`), `From<NodeOpError>` (`:96`) (was `node_op`, #2846) |
| `validateG` | `Graph::validate` → `verify_id_batches` (`graph.rs:1640,1649`), as `apply_effects` consumes it |
| `verifyId` | `verify_id` (`:516`) |
| `resolved`, `verifySchema`, `verifyAttribute` | `:571`, `:539`, `:559` |
| `applyAddSchema` | `apply_add_schema` (`:496`) — get-or-create, then `verify_id` |
| `resolveType`, `checkedTypeId`, `checkedLabelIds` | `:593`, `:617`, `:633` |
| `checkAttrShape`, `attrMap` | `:663`, `:707` |
| `indexOptions`, `singleIndexLabel` | `:750`, `:781` |
| `applyEffects` | `apply_effects` (`:53`) |
-/
namespace FalkorEA

inductive Kind | label | relType | attribute | node | relationship deriving DecidableEq, Repr

inductive AErr where
  | graph (s : String)
  | idMismatch (k : Kind) (name : Name) (expected assigned : Nat) (local_ : Option Name)
  | nameMismatch (k : Kind) (name : Name) (id : Nat) (local_ : Name)
  | unresolved (k : Kind) (name : Name) (id : Nat)
  | idOutOfRange (k : Kind) (id : Nat)
  | attrIdsNotAscending (a b : Nat)
  | shapeMismatch (entities width values : Nat)
  | multiSchema (count : Nat)
  | alreadyLive (k : Kind) (id firstUnallocated : Nat)
  | notLive (k : Kind) (id : Nat) (reason : String)
  | idsHaveAHole (k : Kind) (entry highest created : Nat)
  | idPastEnd (k : Kind) (id : Nat)
deriving DecidableEq, Repr

/-- `impl From<String> for ApplyError` (`:43`). -/
def ofString (s : String) : AErr := .graph s
theorem ofString_spec (s : String) : ofString s = .graph s := rfl

/-- `IdSpaceError` (`graph/id_space.rs:87`, main @ 2c874022a), as `apply.rs`
consumes it. `Miscounted` is gone; `Inconsistent` and `AlreadyTaken` are new (#2846). -/
inductive IdSpaceErr where
  | inconsistent (live recycled bound entry taken expected : Nat)
  | neverCreated (id : Nat)
  | hole (entry highest created : Nat)
  | alreadyRecycled (id : Nat)
  | alreadyLive (id entry : Nat)
  | alreadyTaken (id : Nat)
  | idOutOfRange (id : Nat)

/-- The `&'static str` kind `NodeOpError::node`/`relationship` carry. -/
def Kind.name : Kind → String
  | .node => "node" | .relationship => "relationship" | .label => "label"
  | .relType => "relationship type" | .attribute => "attribute"

/-- `Display` of the two internal-fault variants (`id_space.rs:95-97,132`). -/
def IdSpaceErr.display : IdSpaceErr → String
  | .inconsistent l r b eb t ex =>
    s!"the id space contradicts itself: {l} live + {r} free puts the boundary at {b}, but a batch opened at {eb} having taken {t} puts it at {ex}"
  | .alreadyTaken id => s!"{id} was already taken by this batch"
  | _ => ""

/-- A refusal that is a claim about the buffer (a divergence), as opposed to the
replica's own id space being at fault. -/
def IdSpaceErr.divergence : IdSpaceErr → Bool
  | .inconsistent .. | .alreadyTaken _ => false
  | _ => true

/-- `id_space_error_map` (`:109`). -/
def idSpaceErrorMap (k : Kind) : IdSpaceErr → AErr
  | .alreadyLive id eb => .alreadyLive k id eb
  | .alreadyRecycled id => .notLive k id "it is already in the recycle bin"
  | .neverCreated id => .notLive k id "it was never allocated here"
  | .hole eb hi c => .idsHaveAHole k eb hi c
  | .idOutOfRange id => .idPastEnd k id
  | e@(.inconsistent ..) | e@(.alreadyTaken _) => .graph (k.name ++ " " ++ e.display)

/-- Every divergence refusal keeps its kind and the ids it names, and distinct
divergences stay distinct (the mapping is injective on them). -/
theorem idSpaceErrorMap_injective (k : Kind) (a b : IdSpaceErr) (ha : a.divergence) (hb : b.divergence)
    (h : idSpaceErrorMap k a = idSpaceErrorMap k b) : a = b := by
  cases a <;> cases b <;> simp_all [idSpaceErrorMap, IdSpaceErr.divergence]

/-- The two internal faults are rendered as `ApplyError::Graph("{kind} {e}")`,
never as a divergence. -/
theorem idSpaceErrorMap_internal (k : Kind) (e : IdSpaceErr) (he : e.divergence = false) :
    idSpaceErrorMap k e = .graph (k.name ++ " " ++ e.display) := by
  cases e <;> simp_all [idSpaceErrorMap, IdSpaceErr.divergence]

theorem idSpaceErrorMap_divergence_not_graph (k : Kind) (e : IdSpaceErr) (he : e.divergence) (s : String) :
    idSpaceErrorMap k e ≠ .graph s := by
  cases e <;> simp_all [idSpaceErrorMap, IdSpaceErr.divergence]

/-- `NodeOpError` (`graph.rs:274`, #2846: the id-space arm carries its `kind`)
and `impl From<NodeOpError> for ApplyError` (`:96`). -/
inductive NodeOpErr | graph (s : String) | idSpace (k : Kind) (e : IdSpaceErr)
def fromNodeOp : NodeOpErr → AErr
  | .graph s => .graph s
  | .idSpace k e => idSpaceErrorMap k e
theorem fromNodeOp_spec (k : Kind) (s : String) (e : IdSpaceErr) :
    fromNodeOp (.graph s) = .graph s ∧ fromNodeOp (.idSpace k e) = idSpaceErrorMap k e := ⟨rfl, rfl⟩

/-- `Graph::validate` = `verify_id_batches`: the node space, then the
relationship space, each refusal wrapped with its kind (`NodeOpError::node`,
`::relationship`). -/
def validateG {G} (verifyN verifyE : G → Except IdSpaceErr Unit) (g : G) : Except NodeOpErr Unit :=
  match verifyN g with
  | .error e => .error (.idSpace .node e)
  | .ok () => match verifyE g with
    | .error e => .error (.idSpace .relationship e)
    | .ok () => .ok ()

theorem validateG_ok {G} (verifyN verifyE : G → Except IdSpaceErr Unit) (g : G) :
    validateG verifyN verifyE g = .ok () ↔ verifyN g = .ok () ∧ verifyE g = .ok () := by
  unfold validateG
  cases verifyN g <;> simp
  cases verifyE g <;> simp

/-! ### Name ↔ id checks -/

/-- `verify_id` (`:516`). -/
def verifyId (k : Kind) (name : Name) (expected assigned : Nat) (local_ : Option Name) : Except AErr Unit :=
  if assigned = expected then .ok () else .error (.idMismatch k name expected assigned local_)

theorem verifyId_ok (k name e a l) : verifyId k name e a l = .ok () ↔ a = e := by
  unfold verifyId; split <;> simp_all

/-- `resolved` (`:571`). -/
def resolved (k : Kind) (name : Name) (id : Nat) (local_ : Option Name) : Except AErr Unit :=
  match local_ with
  | some l => if l = name then .ok () else .error (.nameMismatch k name id l)
  | none => .error (.unresolved k name id)

theorem resolved_ok (k name id l) : resolved k name id l = .ok () ↔ l = some name := by
  unfold resolved; split <;> (try split) <;> simp_all

/-- `verify_schema` (`:539`): the replica's own dictionary must hold `name` at `id`. -/
def verifySchema (labels types : List Name) (node : Bool) (id : Nat) (name : Name) : Except AErr Unit :=
  if node then resolved .label name id labels[id]? else resolved .relType name id types[id]?

/-- `verify_attribute` (`:559`). -/
def verifyAttribute (attrs : List Name) (id : Nat) (name : Name) : Except AErr Unit :=
  resolved .attribute name id attrs[id]?

theorem verifySchema_ok (labels types node id name) :
    verifySchema labels types node id name = .ok () ↔ (if node then labels else types)[id]? = some name := by
  unfold verifySchema; cases node <;> simp [resolved_ok]

theorem verifyAttribute_ok (attrs id name) : verifyAttribute attrs id name = .ok () ↔ attrs[id]? = some name :=
  resolved_ok _ _ _ _

/-- `get_label_id_mut` / `get_type_id_mut` / `add_node_attribute_name`:
get-or-create (append-only, id = index). -/
def intern (dict : List Name) (n : Name) : Nat × List Name :=
  match idxOf dict n with
  | some i => (i, dict)
  | none => (dict.length, dict ++ [n])

/-- `apply_add_schema` (`:496`): intern, then compare the assigned id with the
record's. The dictionary is updated *before* the check (as in the Rust); a
refusal fails the whole payload, so that write never survives. -/
def applyAddSchema (dict : List Name) (id : Nat) (name : Name) (k : Kind) : List Name × Except AErr Unit :=
  let (assigned, dict') := intern dict name
  (dict', verifyId k name id assigned dict[id]?)

/-- It agrees with the replication proof's `applyAddName`. -/
theorem applyAddSchema_eq_applyAddName (dict : List Name) (id : Nat) (name : Name) (k : Kind) :
    (match applyAddSchema dict id name k with
     | (d, .ok ()) => some d
     | (_, .error _) => none) = applyAddName dict id name := by
  unfold applyAddSchema intern applyAddName verifyId
  cases h : idxOf dict name with
  | some i => by_cases hi : i = id <;> simp [hi]
  | none => by_cases hl : dict.length = id <;> simp [hl]

/-! ### Id bounds -/

/-- `resolve_type` (`:593`). -/
def resolveType (types : List Name) (rid : Nat) : Except AErr Name :=
  match types[rid]? with | some t => .ok t | none => .error (.idOutOfRange .relType rid)
/-- `checked_type_id` (`:617`). -/
def checkedTypeId (types : List Name) (rid : Nat) : Except AErr Nat :=
  match resolveType types rid with | .ok _ => .ok rid | .error e => .error e

theorem checkedTypeId_ok (types rid) : checkedTypeId types rid = .ok rid ↔ rid < types.length := by
  unfold checkedTypeId resolveType
  cases h : types[rid]? with
  | none => simp; exact List.getElem?_eq_none_iff.mp h
  | some t => simp; exact (List.getElem?_eq_some_iff.mp h).1

/-- `checked_label_ids` (`:633`): every label id below the dictionary size, else
the first offender. -/
def checkedLabelIds (bound : Nat) : List Nat → Except AErr (List Nat)
  | [] => .ok []
  | l :: ls => if l ≥ bound then .error (.idOutOfRange .label l) else
    match checkedLabelIds bound ls with | .ok r => .ok (l :: r) | .error e => .error e

theorem checkedLabelIds_ok (bound : Nat) : ∀ ls, checkedLabelIds bound ls = .ok ls ↔ ∀ l ∈ ls, l < bound := by
  intro ls; induction ls with
  | nil => simp [checkedLabelIds]
  | cons l ls ih =>
    unfold checkedLabelIds
    by_cases h : l ≥ bound
    · simp only [h, ite_true]
      constructor
      · intro e; cases e
      · intro hall; have := hall l (by simp); omega
    · simp only [h, ite_false]
      cases hr : checkedLabelIds bound ls with
      | error e =>
        constructor
        · intro e'; cases e'
        · intro hall; rw [hr] at ih
          have := ih.mpr (fun x hx => hall x (by simp [hx])); cases this
      | ok r =>
        rw [hr] at ih
        constructor
        · intro e; simp only [Except.ok.injEq, List.cons.injEq, true_and] at e; subst e
          intro x hx; rcases List.mem_cons.mp hx with rfl | hx
          · omega
          · exact ih.mp rfl x hx
        · intro hall
          have := ih.mpr (fun x hx => hall x (by simp [hx]))
          simp only [Except.ok.injEq] at this; subst this; rfl

/-- `check_attr_shape` (`:663`): in range, strictly ascending, `rows = ids × width`. -/
def firstNotAsc : List Nat → Option (Nat × Nat)
  | a :: b :: t => if a ≥ b then some (a, b) else firstNotAsc (b :: t)
  | _ => none

def checkAttrShape (bound nIds : Nat) (attrIds : List Nat) (nRows : Nat) : Except AErr Unit :=
  match attrIds.find? (· ≥ bound) with
  | some a => .error (.idOutOfRange .attribute a)
  | none => match firstNotAsc attrIds with
    | some (a, b) => .error (.attrIdsNotAscending a b)
    | none => if nRows ≠ nIds * attrIds.length then .error (.shapeMismatch nIds attrIds.length nRows) else .ok ()

theorem firstNotAsc_none : ∀ l : List Nat, firstNotAsc l = none ↔ l.Pairwise (· < ·) := by
  intro l
  induction l with
  | nil => simp [firstNotAsc]
  | cons a t ih =>
    cases t with
    | nil => simp [firstNotAsc]
    | cons b t =>
      simp only [firstNotAsc]
      by_cases h : a ≥ b
      · simp only [h, ite_true, reduceCtorEq, false_iff]
        intro hp; have := List.rel_of_pairwise_cons hp (List.mem_cons_self); omega
      · simp only [h, ite_false, ih]
        constructor
        · intro hp
          refine List.Pairwise.cons ?_ hp
          intro x hx
          rcases List.mem_cons.mp hx with rfl | hx
          · omega
          · have := List.rel_of_pairwise_cons hp hx; omega
        · intro hp; exact hp.of_cons

/-- **`check_attr_shape` accepts exactly** in-range, strictly ascending attribute
ids and a rows block of `ids × width` values. -/
theorem checkAttrShape_ok (bound n attrIds nRows) :
    checkAttrShape bound n attrIds nRows = .ok () ↔
      ((∀ a ∈ attrIds, a < bound) ∧ attrIds.Pairwise (· < ·) ∧ nRows = n * attrIds.length) := by
  unfold checkAttrShape
  cases hf : attrIds.find? (· ≥ bound) with
  | some a =>
    simp only [reduceCtorEq, false_iff, not_and]
    intro hall; have := List.find?_some hf; have := hall a (List.mem_of_find?_eq_some hf); simp at *; omega
  | none =>
    have hall : ∀ a ∈ attrIds, a < bound := by
      intro a ha; have := List.find?_eq_none.mp hf a ha; simp at this; omega
    cases hn : firstNotAsc attrIds with
    | some p =>
      obtain ⟨a, b⟩ := p
      simp only [reduceCtorEq, false_iff, not_and]
      intro _ hp; rw [(firstNotAsc_none _).mpr hp] at hn; cases hn
    | none =>
      have hp := (firstNotAsc_none _).mp hn
      simp only
      by_cases hr : nRows ≠ n * attrIds.length
      · rw [if_pos hr]; simp only [reduceCtorEq, false_iff, not_and]; intro _ _ h; exact hr h
      · rw [if_neg hr]; simp only [true_iff]; exact ⟨hall, hp, by omega⟩

/-- `attr_map` (`:707`): check, then `id ↦ [(attr_id, rows[row*w + col])]`.
`FxHashMap::insert` keeps the last write for a repeated id. -/
def attrMap {V} [Inhabited V] (bound : Nat) (ids attrIds : List Nat) (rows : List V) : Except AErr (List (Nat × List (Nat × V))) :=
  match checkAttrShape bound ids.length attrIds rows.length with
  | .error e => .error e
  | .ok () => .ok ((ids.zipIdx).map fun (id, r) =>
      (id, attrIds.zipIdx.map fun (a, c) => (a, rows[r * attrIds.length + c]!)))

theorem map_fst_pair {β} (f : Nat × Nat → β) : ∀ (l : List Nat) (n : Nat),
    ((l.zipIdx n).map (fun x => (x.1, f x))).map (·.1) = l := by
  intro l; induction l with
  | nil => intro n; rfl
  | cons a t ih => intro n; simp [List.zipIdx_cons, ih]

/-- Row `r` of the map belongs to the `r`-th id and holds exactly that row's slice. -/
theorem attrMap_row {V} [Inhabited V] (bound : Nat) (ids attrIds : List Nat) (rows : List V) (m)
    (h : attrMap bound ids attrIds rows = .ok m) :
    m.map (·.1) = ids ∧ ∀ r (hr : r < m.length),
      (m[r].2).map (·.2) = (rows.drop (r * attrIds.length)).take attrIds.length := by
  unfold attrMap at h
  split at h
  · cases h
  · rename_i hok
    have hs := (checkAttrShape_ok _ _ _ _).mp hok
    cases h
    refine ⟨map_fst_pair _ _ _, fun r hr => ?_⟩
    simp only [List.getElem_map, List.getElem_zipIdx, List.map_map]
    simp at hr
    apply List.ext_getElem
    · simp; have := hs.2.2; rw [Nat.mul_comm] at this
      have : (r + 1) * attrIds.length ≤ rows.length := by rw [hs.2.2]; exact Nat.mul_le_mul_right _ hr
      rw [Nat.min_eq_left (by rw [Nat.add_mul] at this; omega)]
    · intro c h1 h2
      simp only [List.getElem_map, List.getElem_zipIdx, Function.comp, List.getElem_take, List.getElem_drop]
      simp at h1
      have : r * attrIds.length + c < rows.length := by
        rw [hs.2.2]; have := Nat.mul_le_mul_right attrIds.length (show r + 1 ≤ ids.length by omega)
        rw [Nat.add_mul] at this; omega
      simp only [Nat.zero_add]; rw [getElem!_pos rows (r * attrIds.length + c) this]

/-! ### Index plumbing -/

inductive IdxType | range | fulltext | vector deriving DecidableEq, Repr

/-- `index_options` (`:750`). -/
def indexOptions {T Vo} (t : IdxType) (text : T) (vec : Option Vo) : Option (Sum T Vo) :=
  match t with
  | .vector => vec.map .inr
  | .fulltext => some (.inl text)
  | .range => none

theorem indexOptions_spec {T Vo} (text : T) (vec : Option Vo) :
    indexOptions .range text vec = none ∧ indexOptions .fulltext text vec = some (.inl text) ∧
    indexOptions .vector text vec = vec.map .inr := ⟨rfl, rfl, rfl⟩

/-- `single_index_label` (`:781`): exactly one schema. -/
def singleIndexLabel (schemas : List (Nat × Name)) : Except AErr Name :=
  match schemas with
  | [s] => .ok s.2
  | ss => .error (.multiSchema ss.length)

theorem singleIndexLabel_ok (schemas : List (Nat × Name)) (n : Name) :
    singleIndexLabel schemas = .ok n ↔ ∃ id, schemas = [(id, n)] := by
  unfold singleIndexLabel
  split
  · rename_i s; simp; constructor
    · rintro rfl; exact ⟨s.1, rfl⟩
    · rintro ⟨id, rfl⟩; rfl
  · rename_i h; simp; intro id he; exact h _ he

/-! ### `apply_effects` (`:53`) -/

/-- Open, apply every record in order threading the `IndexDocs` (stop at the
first refusal), `g.validate()?` (the batch `Graph::new_version` opened: both id
spaces verified, refusals through `From<NodeOpError>`), then commit the index
docs. Graph steps are parameters. Since #2846 the batch lives on the graph
rather than in a local `BufferOps`. -/
def applyEffects {G R D} (openP : Except AErr (List R)) (applyRec : G → D → R → Except AErr (G × D))
    (validate : G → Except NodeOpErr Unit) (commit : G → D → G) (g : G) (d : D) :
    Except AErr G :=
  match openP with
  | .error e => .error e
  | .ok recs =>
    match recs.foldlM (fun (st : G × D) r => applyRec st.1 st.2 r) (g, d) with
    | .error e => .error e
    | .ok (g', d') =>
      match validate g' with
      | .error e => .error (fromNodeOp e)
      | .ok () => .ok (commit g' d')

/-- **`apply_effects` succeeds only if every record applied and the graph
validated**, and then the result is the committed state after all records. -/
theorem applyEffects_ok {G R D} (openP applyRec validate commit) (g : G) (d : D) (g'' : G)
    (h : applyEffects (R := R) openP applyRec validate commit g d = .ok g'') :
    ∃ recs g' d', openP = .ok recs ∧ recs.foldlM (fun (st : G × D) r => applyRec st.1 st.2 r) (g, d) = .ok (g', d') ∧
      validate g' = .ok () ∧ g'' = commit g' d' := by
  unfold applyEffects at h
  cases ho : openP with
  | error e => rw [ho] at h; cases h
  | ok recs =>
    rw [ho] at h; simp only at h
    cases hf : recs.foldlM (fun (st : G × D) r => applyRec st.1 st.2 r) (g, d) with
    | error e => rw [hf] at h; cases h
    | ok p =>
      obtain ⟨g', d'⟩ := p
      rw [hf] at h; simp only at h
      cases h1 : validate g' with
      | error e => rw [h1] at h; cases h
      | ok u => cases u; rw [h1] at h; cases h; exact ⟨recs, g', d', rfl, hf, h1, rfl⟩

/-- A buffer whose records all apply but which leaves an impossible id space is
refused with the id space's own refusal, kind attached. -/
theorem applyEffects_validate_refused {G R D} (openP applyRec commit) (verifyN verifyE : G → Except IdSpaceErr Unit)
    (g : G) (d : D) (recs : List R) (g' : G) (d' : D) (e : NodeOpErr)
    (ho : openP = .ok recs) (hf : recs.foldlM (fun (st : G × D) r => applyRec st.1 st.2 r) (g, d) = .ok (g', d'))
    (hv : validateG verifyN verifyE g' = .error e) :
    applyEffects openP applyRec (validateG verifyN verifyE) commit g d = .error (fromNodeOp e) := by
  simp [applyEffects, ho, hf, hv]

end FalkorEA
