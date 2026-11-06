import FalkorEffectsCodec.RecBlocks
/-!
# The DDL blocks of `records.rs` and the tags of `v3/mod.rs`

| here | there |
| --- | --- |
| `Entity`, `schemaTag`/`entityFromSchemaTag` | `schema_tag`, `entity_from_schema_tag` (`v3/mod.rs`) |
| `entityTag`/`entityFromTag`                 | `entity_tag`, `entity_from_tag` |
| `CType`, `CStatus` + tags                   | `constraint_tag`, `constraint_from_tag`, `constraint_status_tag`, `…_from_tag` |
| `indexTypeOf`                               | `index_type_of`, `field_type_bits` |
| `bOpts`                                     | `IndexFieldOptions::{encode_sized, decode_sized}` (`records.rs:162`, `:217`) |
| `bAttrRef`, `bFields`, `bProps`             | `AttrRef`, `IndexFields` (`:324`, `:346`), `read_constraint_props` (`:818`) |
| `bSchemas`                                  | `IndexSchemas` (`:388`, `:421`) — the encoder sorts by id |
-/
namespace FalkorCodec

inductive Entity | node | rel deriving DecidableEq, Repr
inductive CType | unique | mandatory deriving DecidableEq, Repr
inductive CStatus | operational | underConstruction | failed deriving DecidableEq, Repr

def schemaTag : Entity → Nat | .node => 0 | .rel => 1
def entityFromSchemaTag (v : Nat) : Except DErr Entity :=
  if v = 0 then .ok .node else if v = 1 then .ok .rel else .error (.badSchemaType v)
def entityTag : Entity → Nat | .node => 1 | .rel => 2
def entityFromTag (v : Nat) : Except DErr Entity :=
  if v = 1 then .ok .node else if v = 2 then .ok .rel else .error (.badEntityType v)
def ctypeTag : CType → Nat | .unique => 0 | .mandatory => 1
def ctypeFromTag (v : Nat) : Except DErr CType :=
  if v = 0 then .ok .unique else if v = 1 then .ok .mandatory else .error (.badConstraintType v)
def statusTag : CStatus → Nat | .operational => 0 | .underConstruction => 1 | .failed => 2
def statusFromTag (v : Nat) : Except DErr CStatus :=
  if v = 0 then .ok .operational else if v = 1 then .ok .underConstruction
  else if v = 2 then .ok .failed else .error (.badConstraintStatus v)

def bSchemaTag : Blk Entity := bTag schemaTag entityFromSchemaTag (by intro a; cases a <;> rfl) (by intro a; cases a <;> decide)
def bEntityTag : Blk Entity := bTag entityTag entityFromTag (by intro a; cases a <;> rfl) (by intro a; cases a <;> decide)
def bCType : Blk CType := bTag ctypeTag ctypeFromTag (by intro a; cases a <;> rfl) (by intro a; cases a <;> decide)
def bStatus : Blk CStatus := bTag statusTag statusFromTag (by intro a; cases a <;> rfl) (by intro a; cases a <;> decide)

/-- The tag decoders reject exactly the values no tag maps to. -/
theorem entityFromSchemaTag_err (v : Nat) : (∃ e, entityFromSchemaTag v = .ok e) ↔ v ≤ 1 := by
  unfold entityFromSchemaTag; by_cases h0 : v = 0 <;> by_cases h1 : v = 1 <;> simp [h0, h1] <;> omega
theorem statusFromTag_err (v : Nat) : (∃ e, statusFromTag v = .ok e) ↔ v ≤ 2 := by
  unfold statusFromTag
  by_cases h0 : v = 0 <;> by_cases h1 : v = 1 <;> by_cases h2 : v = 2 <;> simp [h0, h1, h2] <;> omega

/-! ### `index_type_of` -/

inductive IdxType | fulltext | range | vector deriving DecidableEq, Repr

/-- `IndexFieldBit::try_from(bit)?.index_type()`. -/
def bitKind (b : Nat) : Option IdxType :=
  if b = 1 then some .fulltext else if b = 2 ∨ b = 4 ∨ b = 8 then some .range
  else if b = 16 then some .vector else none

/-- `field_type_bits`: the set bits, lowest first. -/
def setBits (ft : Nat) : List Nat := (List.range 32).filterMap fun i =>
  if ft / 2 ^ i % 2 = 1 then some (2 ^ i) else none

/-- `index_type_of` (`v3/mod.rs`): the first bit's kind; an unknown bit or a
second kind is refused; no bits is `Range`. -/
def indexTypeOf (ft : Nat) : Except EErr IdxType :=
  go (setBits ft) none
where
  go : List Nat → Option IdxType → Except EErr IdxType
    | [], first => .ok (first.getD .range)
    | b :: bs, first =>
      match bitKind b with
      | none => .error .badFieldType
      | some k => match first with
        | none => go bs (some k)
        | some f => if f ≠ k then .error .badFieldType else go bs (some f)

#guard (indexTypeOf 0).toOption == some .range
#guard (indexTypeOf 0x0E).toOption == some .range
#guard (indexTypeOf 0x10).toOption == some .vector
#guard (indexTypeOf 0x11).toOption == none
#guard (indexTypeOf 0x20).toOption == none

def INDEX_FLD_VECTOR : Nat := 0x10
def hasVector (ft : Nat) : Bool := ft / 16 % 2 = 1

/-- `field_type`: `buf.u32(ft)`; decode `r.u32()?` then `index_type_of(ft)?`. -/
def bFieldType : Blk Nat where
  enc ft := .ok (w32 ft)
  dec := do
    let ft ← u32
    match indexTypeOf ft with | .ok _ => pure ft | .error _ => fail (.badFieldType ft)
  ok ft := ft < 2 ^ 32 ∧ ∃ k, indexTypeOf ft = .ok k
  rt := by
    rintro ft bs rest ⟨h, k, hk⟩ e; cases e
    simp only [bind_apply]; rw [u32_w32 h]; simp only [hk]; rfl

/-! ### Items -/

structure Ref where
  id : Nat
  name : Bytes
deriving DecidableEq, Repr

theorem wString_len (s : Bytes) : 9 ≤ (wString s).length := by simp [wString, w64]; omega

/-- `AttrRef { id: u16, name }` — at least 11 bytes. -/
def bAttrRef (utf8 : Bytes → Bool) : Blk Ref :=
  { enc := fun r => eseq (.ok (w16 r.id)) (.ok (wString r.name))
    dec := do let id ← u16; let name ← rString utf8; pure ⟨id, name⟩
    ok := fun r => r.id < 2 ^ 16 ∧ r.name.length + 1 < 2 ^ 64 ∧ utf8 r.name = true
    rt := by
      rintro ⟨id, name⟩ bs rest ⟨h1, h2, h3⟩ e
      simp only [eseq, Except.ok.injEq] at e; subst e
      simp only [bind_apply, List.append_assoc]; rw [u16_w16 h1]; simp only
      rw [rString_wString utf8 name rest h2 h3]; rfl }

/-- `SchemaRef { id: u32, name }` — at least 13 bytes. -/
def bSchemaRef (utf8 : Bytes → Bool) : Blk Ref :=
  { enc := fun r => eseq (.ok (w32 r.id)) (.ok (wString r.name))
    dec := do let id ← u32; let name ← rString utf8; pure ⟨id, name⟩
    ok := fun r => r.id < 2 ^ 32 ∧ r.name.length + 1 < 2 ^ 64 ∧ utf8 r.name = true
    rt := by
      rintro ⟨id, name⟩ bs rest ⟨h1, h2, h3⟩ e
      simp only [eseq, Except.ok.injEq] at e; subst e
      simp only [bind_apply, List.append_assoc]; rw [u32_w32 h1]; simp only
      rw [rString_wString utf8 name rest h2 h3]; rfl }

/-- `IndexFields` (`records.rs:324/337`): `u16` count (`BlockCountTooLarge`
past it), `guard_count(n, 11)`, items. -/
def bFields (utf8 : Bytes → Bool) : Blk (List Ref) :=
  bItems 2 11 .blockCountTooLarge (bAttrRef utf8) (by
    intro r bs e; simp only [bAttrRef, eseq, Except.ok.injEq] at e; subst e
    have := wString_len r.name; simp [w16]; omega)

/-- `read_constraint_props` (`:818`) / `write_constraint_tail` (`:850`): `u8`
count — the encoder `expect`s ≤ 255 (`panic`). -/
def bProps (utf8 : Bytes → Bool) : Blk (List Ref) :=
  bItems 1 11 .panic (bAttrRef utf8) (by
    intro r bs e; simp only [bAttrRef, eseq, Except.ok.injEq] at e; subst e
    have := wString_len r.name; simp [w16]; omega)

def bSchemaItems (utf8 : Bytes → Bool) : Blk (List Ref) :=
  bItems 2 13 .blockCountTooLarge (bSchemaRef utf8) (by
    intro r bs e; simp only [bSchemaRef, eseq, Except.ok.injEq] at e; subst e
    have := wString_len r.name; simp [w32]; omega)

/-- `sort_unstable_by_key(|s| s.id)`; stable here, which agrees whenever ids are
distinct (they are schema ids). -/
def sortById (xs : List Ref) : List Ref := xs.mergeSort (fun a b => decide (a.id ≤ b.id))

theorem readN_length {α} (A : R α) : ∀ (n : Nat) (r : Bytes) (l : List α) (r' : Bytes),
    readN n A r = .ok (l, r') → l.length = n := by
  intro n; induction n with
  | zero => intro r l r' h; simp [readN] at h; simp [h.1]
  | succ n ih =>
    intro r l r' h
    simp only [readN, bind_apply] at h
    split at h; · cases h
    split at h; · cases h
    rename_i _ _ _ _ _ _ h2
    simp only [pure_apply, Except.ok.injEq, Prod.mk.injEq] at h
    rw [← h.1]; simp [ih _ _ _ h2]

/-- `IndexSchemas` (`:379/:412`): the encoder sorts by id; the decoder refuses an
empty list. -/
def bSchemas (utf8 : Bytes → Bool) : Blk (List Ref) where
  enc xs := (bSchemaItems utf8).enc (sortById xs)
  dec := do
    let n ← u16
    let n ← guardCount n 13
    if n = 0 then fail .emptyIndexSchemaList else readN n (bSchemaRef utf8).dec
  ok xs := (bSchemaItems utf8).ok xs ∧ xs ≠ [] ∧ xs.Pairwise (fun a b => a.id ≤ b.id)
  rt := by
    intro xs bs rest ⟨hok, hne, hs⟩ e
    have hsort : sortById xs = xs := List.mergeSort_of_pairwise (by
      exact hs.imp (fun h => by simpa using h))
    rw [hsort] at e
    have := (bSchemaItems utf8).rt xs bs rest hok e
    simp only [bSchemaItems, bItems, readItems, bind_apply, u16] at this ⊢
    revert this
    cases hu : rdU 2 (bs ++ rest) with
    | error e => intro h; exact h
    | ok p =>
      obtain ⟨n, r1⟩ := p
      simp only
      cases hg : guardCount n 13 r1 with
      | error e => intro h; exact h
      | ok q =>
        obtain ⟨m, r2⟩ := q
        simp only
        intro h
        have hm : m = xs.length := (readN_length _ _ _ _ _ h).symm
        have : m ≠ 0 := by rw [hm]; exact fun h0 => hne (List.length_eq_zero_iff.mp h0)
        simp only [this, ite_false]; exact h

/-- The encoder's sort is what reaches the wire: an unsorted list encodes as
its sorted permutation. -/
theorem bSchemas_enc_sorted (utf8 : Bytes → Bool) (xs : List Ref) :
    (bSchemas utf8).enc xs = (bSchemaItems utf8).enc (sortById xs) := rfl

end FalkorCodec
