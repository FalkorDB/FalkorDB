import FalkorEffectsEmitApply.ApplyHelpers
/-!
# `emit.rs`: schema additions, DDL buffers, small helpers

| here | there (`graph/src/effects/v3/emit.rs`) |
| --- | --- |
| `schemaId` | `schema_id` (`:56`) — `u32::try_from(..).expect` |
| `schemaAdditions` (Emit.lean), `replayNames` | `emit_schema_additions` (`:61`) |
| `encodeAll` | `encode_all` (`:187`) |
| `indexFieldFlags` | `index_field_flags` (`:205`) |
| `buildIndexRec`, `buildConstraintRec` | `build_index_buffer` (`:227`), `build_constraint_buffer` (`:111`) |
| `wireIndexOptions` | `wire_index_options` (`:2080`) |
-/
namespace FalkorEA

/-- `schema_id` (`:56`): `none` is the `expect` panic. -/
def schemaId (id : Nat) : Option Nat := if id < 2 ^ 32 then some id else none
theorem schemaId_spec (id : Nat) : schemaId id = some id ↔ id < 2 ^ 32 := by
  unfold schemaId; split <;> simp_all

/-! ### `emit_schema_additions` replays on the replica -/

theorem idxOf_none_of_not_mem (l : List Name) (n : Name) (h : n ∉ l) : idxOf l n = none := by
  unfold idxOf
  rw [List.findIdx?_eq_none_iff.mpr (fun x hx => by
    have : x ≠ n := fun e => h (e ▸ hx)
    simpa using this)]

/-- Apply a run of `ADD_SCHEMA`-style records with `applyAddName`. -/
def replayNames : List Name → Nat → List Name → Option (List Name)
  | dict, _, [] => some dict
  | dict, i, n :: ns => match applyAddName dict i n with
    | some d => replayNames d (i + 1) ns
    | none => none

/-- **A replica whose dictionary is the master's baseline prefix accepts every
`ADD_SCHEMA`/`ADD_ATTRIBUTE` the master emits and ends with the master's
dictionary** (names are unique, as the interning dictionaries guarantee). -/
theorem replayNames_suffix : ∀ (pre suf : List Name), (pre ++ suf).Nodup →
    replayNames pre pre.length suf = some (pre ++ suf)
  | pre, [], _ => by simp [replayNames]
  | pre, n :: ns, h => by
    have hn : n ∉ pre := by
      intro hm; have := List.nodup_append.mp h; exact this.2.2 n hm n (by simp) rfl
    simp only [replayNames, applyAddName, idxOf_none_of_not_mem pre n hn, ite_true]
    have := replayNames_suffix (pre ++ [n]) ns (by simpa using h)
    simpa using this

/-- The record ids `emit_schema_additions` assigns are the dictionary offsets. -/
theorem suffixRecs_ids (mk : Nat → Name → Rec) : ∀ (base : Nat) (ns : List Name),
    suffixRecs mk base ns = (ns.zipIdx base).map fun (n, i) => mk i n := by
  intro base ns; induction ns generalizing base with
  | nil => rfl
  | cons n ns ih => simp [suffixRecs, ih, List.zipIdx_cons]

/-! ### `encode_all` (`:187`): schema additions, stopping at the first failure -/

def encodeAll {R} (enc : R → Except String (List Nat)) : List R → Option String → List Nat → (List Nat × Option String)
  | [], failed, buf => (buf, failed)
  | r :: rs, failed, buf => match failed with
    | some e => encodeAll enc rs (some e) buf
    | none => match enc r with
      | .ok b => encodeAll enc rs none (buf ++ b)
      | .error e => encodeAll enc rs (some e) buf

theorem encodeAll_ok {R} (enc : R → Except String (List Nat)) :
    ∀ (rs : List R) (bss : List (List Nat)) (buf : List Nat), rs.length = bss.length →
    (∀ i (h1 : i < rs.length) (h2 : i < bss.length), enc rs[i] = .ok bss[i]) →
    encodeAll enc rs none buf = (buf ++ bss.flatten, none)
  | [], [], _, _, _ => by simp [encodeAll]
  | r :: rs, b :: bs, buf, hl, h => by
    have h0 := h 0 (by simp) (by simp)
    simp only [List.getElem_cons_zero] at h0
    simp only [encodeAll, h0]
    rw [encodeAll_ok enc rs bs (buf ++ b) (by simpa using hl)
      (fun i h1 h2 => by have := h (i + 1) (by simp; omega) (by simp; omega); simpa using this)]
    simp
  | [], _ :: _, _, hl, _ => by simp at hl
  | _ :: _, [], _, hl, _ => by simp at hl

theorem encodeAll_failed {R} (enc : R → Except String (List Nat)) :
    ∀ (rs : List R) (e : String) (buf : List Nat), encodeAll enc rs (some e) buf = (buf, some e)
  | [], _, _ => rfl
  | _ :: rs, e, buf => by simp only [encodeAll]; exact encodeAll_failed enc rs e buf

/-! ### Index / constraint DDL -/

/-- `index_field_flags` (`:205`). -/
def indexFieldFlags : IdxType → Nat
  | .range => 0x02 ||| 0x04 ||| 0x08
  | .fulltext => 0x01
  | .vector => 0x10

theorem indexFieldFlags_spec :
    indexFieldFlags .range = 0x0E ∧ indexFieldFlags .fulltext = 1 ∧ indexFieldFlags .vector = 0x10 := by decide

/-- `names.iter().position(|n| n == x)`. -/
def position (names : List Name) (x : Name) : Option Nat := names.findIdx? (· == x)

theorem position_get (names : List Name) (x : Name) (i : Nat) (h : position names x = some i) :
    names[i]? = some x := by
  unfold position at h
  obtain ⟨hi, hp, -⟩ := List.findIdx?_eq_some_iff_getElem.mp h
  simp at hp; simp [List.getElem?_eq_getElem hi, hp]

/-- The resolved `(label id, field ids)` an index/constraint record carries. -/
def resolveRefs (dict attrs : List Name) (label : Name) (fields : List Name) :
    Except String (Nat × List (Nat × Name)) :=
  match position dict label with
  | none => .error s!"label '{label}' is not registered"
  | some lid =>
    match fields.mapM (fun f => (position attrs f).map (fun i => (i, f))) with
    | none => .error "field is not registered"
    | some fs => .ok (lid, fs)

/-- `build_index_buffer` (`:227`): schema additions, then the record. -/
def buildIndexRec (dict attrs : List Name) (create : Bool) (label : Name) (fields : List Name) (t : IdxType) :
    Except String (Bool × List (Nat × Name) × Nat × List (Nat × Name)) :=
  match resolveRefs dict attrs label fields with
  | .error e => .error e
  | .ok (lid, fs) => .ok (create, [(lid, label)], indexFieldFlags t, fs)

/-- `build_constraint_buffer` (`:111`): a create must carry a status. -/
def buildConstraintRec {S} (dict attrs : List Name) (create : Bool) (status : Option S) (label : Name)
    (props : List Name) : Except String (Option S × Nat × Name × List (Nat × Name)) :=
  match resolveRefs dict attrs label props with
  | .error e => .error e
  | .ok (lid, ps) =>
    if create then match status with
      | none => .error s!"constraint create on '{label}' carries no status to replicate"
      | some st => .ok (some st, lid, label, ps)
    else .ok (none, lid, label, ps)

theorem mapM_position (attrs fields : List Name) (fs : List (Nat × Name))
    (h : fields.mapM (fun f => (position attrs f).map (fun i => (i, f))) = some fs) :
    fs.map (·.2) = fields ∧ ∀ p ∈ fs, attrs[p.1]? = some p.2 := by
  induction fields generalizing fs with
  | nil => simp at h; subst h; simp
  | cons f t ih =>
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨⟨i, f'⟩, ⟨i', hi, he⟩, rest, hr, hfs⟩ := h
    cases he
    simp only [Option.pure_def, Option.some.injEq] at hfs; subst hfs
    obtain ⟨h1, h2⟩ := ih rest hr
    refine ⟨by simp [h1], ?_⟩
    intro p hp; rcases List.mem_cons.mp hp with rfl | hp
    · exact position_get _ _ _ hi
    · exact h2 p hp

/-- **An emitted index record passes the replica's checks** when the replica's
dictionaries are the master's: `verify_schema` on its one schema and
`verify_attribute` on every field accept, and `single_index_label` succeeds. -/
theorem buildIndexRec_verifies (labels types attrs : List Name) (node create : Bool) (label : Name)
    (fields : List Name) (t : IdxType) (r) (h : buildIndexRec (if node then labels else types) attrs create label fields t = .ok r) :
    (∀ s ∈ r.2.1, verifySchema labels types node s.1 s.2 = .ok ()) ∧
    (∀ f ∈ r.2.2.2, verifyAttribute attrs f.1 f.2 = .ok ()) ∧ singleIndexLabel r.2.1 = .ok label ∧
    r.2.2.2.map (·.2) = fields := by
  unfold buildIndexRec resolveRefs at h
  cases hpos : position (if node then labels else types) label with
  | none => simp [hpos] at h
  | some lid =>
    cases hm : fields.mapM (fun f => (position attrs f).map (fun i => (i, f))) with
    | none => simp [hpos, hm] at h
    | some fs =>
      obtain ⟨h1, h2⟩ := mapM_position _ _ _ hm
      simp only [hpos, hm, Except.ok.injEq] at h; subst h
      refine ⟨?_, fun f hf => (verifyAttribute_ok _ _ _).mpr (h2 f hf), rfl, h1⟩
      intro s hs; simp at hs; subst hs
      rw [verifySchema_ok]; exact position_get _ _ _ hpos

/-- The same for constraints, and a create always carries its status. -/
theorem buildConstraintRec_verifies {S} (labels types attrs : List Name) (node create : Bool) (status : Option S)
    (label : Name) (props : List Name) (r) (h : buildConstraintRec (if node then labels else types) attrs create status label props = .ok r) :
    verifySchema labels types node r.2.1 r.2.2.1 = .ok () ∧ (∀ p ∈ r.2.2.2, verifyAttribute attrs p.1 p.2 = .ok ()) ∧
    (create = true → r.1.isSome) ∧ r.2.2.2.map (·.2) = props := by
  unfold buildConstraintRec resolveRefs at h
  cases hpos : position (if node then labels else types) label with
  | none => simp [hpos] at h
  | some lid =>
    cases hm : props.mapM (fun f => (position attrs f).map (fun i => (i, f))) with
    | none => simp [hpos, hm] at h
    | some ps =>
      obtain ⟨h1, h2⟩ := mapM_position _ _ _ hm
      simp only [hpos, hm] at h
      have hv : verifySchema labels types node lid label = .ok () := by
        rw [verifySchema_ok]; exact position_get _ _ _ hpos
      cases create with
      | false => simp at h; subst h; exact ⟨hv, fun p hp => (verifyAttribute_ok _ _ _).mpr (h2 p hp), by simp, h1⟩
      | true =>
        cases status with
        | none => simp at h
        | some st => simp at h; subst h; exact ⟨hv, fun p hp => (verifyAttribute_ok _ _ _).mpr (h2 p hp), by simp, h1⟩

/-- `wire_index_options` (`:2080`): unparsable or absent options are "none
given"; text options fill the text half; vector options the vector half. -/
def wireIndexOptions {T V} (defaultText : T) (parsed : Option (Sum T V)) : T × Option V :=
  match parsed with
  | none => (defaultText, none)
  | some (.inl t) => (t, none)
  | some (.inr v) => (defaultText, some v)

theorem wireIndexOptions_spec {T V} (d : T) (t : T) (v : V) :
    wireIndexOptions d none = (d, (none : Option V)) ∧ wireIndexOptions (V := V) d (some (.inl t)) = (t, none) ∧
    wireIndexOptions d (some (.inr v)) = (d, some v) := ⟨rfl, rfl, rfl⟩

end FalkorEA
