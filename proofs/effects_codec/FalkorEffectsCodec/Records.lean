import FalkorEffectsCodec.Value
/-
# Record framing: `graph/src/effects/v3/records.rs`, `blocks.rs`

| here | there |
| --- | --- |
| `IdCodec`        | `IdList` + `read_ids` (`id_list.rs`) — **abstract**: its roundtrip is a hypothesis owned by the IdList proof |
| `wLabelSet`/`rLabelSet` | `LabelSet` (`blocks.rs:54-81`) |
| `wAttrIds`/`rAttrIds`   | `AttrIds` (`blocks.rs:100-124`) |
| `wRows`/`rRows`         | `AttrValues` (`blocks.rs:154-189`) |
| `Rec`            | `Record` (`records.rs:489`) — batchable variants and `ADD_ATTRIBUTE` |
| `encRec`         | `impl EffectEncode<3> for Record` (`records.rs:917`) |
| `readRec`        | `read_record` (`records.rs:611`) |
| `readAll`        | `Records::next` (`records.rs:1242`) |

Modelled: `CREATE_NODE`, `UPDATE_NODE`, `DELETE_NODE`, `SET_LABELS`,
`REMOVE_LABELS`, `CREATE_EDGE`, `DELETE_EDGE`, `ADD_ATTRIBUTE`. Not modelled:
`UPDATE_EDGE` (same shape as `UPDATE_NODE` with a `u32` in the label slot),
`ADD_SCHEMA`, and the index/constraint DDL — see `REPORT.md`.
-/

namespace FalkorCodec

/-- What this proof needs from the `IdList` codec, and nothing more. -/
structure IdCodec where
  T : Type
  count : T → Nat
  enc : T → Except EErr Bytes
  dec : Nat → R T
  roundtrip : ∀ t bs rest, enc t = .ok bs → dec (count t) (bs ++ rest) = .ok (t, rest)
  good : ∀ n, Good (dec n)

variable (C : IdCodec) (utf8 I : Bytes → Bool)

def wLabelSet (ls : List Nat) : Except EErr Bytes :=
  if ls.length < 2 ^ 16 then .ok (w16 ls.length ++ ls.flatMap w32) else .error .blockCountTooLarge

def wAttrIds (as : List Nat) : Except EErr Bytes :=
  if as.length < 2 ^ 16 then .ok (w16 as.length ++ as.flatMap w16) else .error .blockCountTooLarge

def rLabelSet : R (List Nat) := do
  let n ← u16; let n ← guardCount n 4; readN n u32
def rAttrIds : R (List Nat) := do
  let n ← u16; let n ← guardCount n 2; readN n u16

/-- `AttrValues` writes every row it holds; **it does not check** that there
are `count × attrs` of them (`blocks.rs:155-168`). -/
def wRows : List Val → Except EErr Bytes := encL I

/-- `AttrValues::decode_sized` (`blocks.rs:177-188`). -/
def rRows (count attrs : Nat) : R (List Val) := do
  let total ← guardCount (count * attrs) MIN_VALUE_BYTES
  readN total (decodeV utf8)

inductive Rec (T : Type) where
  | createNode (ids : T) (labels attrIds : List Nat) (rows : List Val)
  | addAttribute (id : Nat) (name : Bytes)

def OP_CREATE_NODE : Nat := 3
def OP_ADD_ATTRIBUTE : Nat := 10

def seqE (xs : List (Except EErr Bytes)) : Except EErr Bytes :=
  xs.foldr (fun x acc => match x, acc with
    | .ok a, .ok b => .ok (a ++ b)
    | .error e, _ => .error e
    | _, .error e => .error e) (.ok [])

/-- `records.rs:948-960`: `check_row_shape` (#2916), header `opcode · count`
from `ids.count()` (`write_header` refuses a zero count since #2916), then the
blocks. Both conditions the decoder needs are now checked here. -/
def encRec : Rec C.T → Except EErr Bytes
  | .createNode ids ls as rows =>
    if C.count ids * as.length ≠ rows.length then .error .rowShapeMismatch
    else if C.count ids = 0 then .error .emptyRecord else
    seqE [.ok (w32 OP_CREATE_NODE ++ w32 (C.count ids)), wLabelSet ls, wAttrIds as, C.enc ids,
      wRows I rows]
  | .addAttribute id name => .ok (w32 OP_ADD_ATTRIBUTE ++ w16 id ++ wString name)

def readRec : R (Rec C.T) := do
  let op ← u32
  if op = OP_CREATE_NODE then do
    let count ← u32
    if count = 0 then fail (.emptyRecord op) else do
    let ls ← rLabelSet
    let as ← rAttrIds
    let ids ← C.dec count
    let rows ← rRows utf8 count as.length
    pure (.createNode ids ls as rows)
  else if op = OP_ADD_ATTRIBUTE then do
    let id ← u16
    let name ← rString utf8
    pure (.addAttribute id name)
  else fail (.badOpcode op)

theorem readN_u32 (ls : List Nat) (h : ∀ x ∈ ls, x < 2 ^ 32) (rest : Bytes) :
    readN ls.length u32 (ls.flatMap w32 ++ rest) = .ok (ls, rest) := by
  induction ls with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.length_cons, readN, List.flatMap_cons, List.append_assoc, bind_apply]
    rw [u32_w32 (h x (by simp))]
    simp only [bind_apply]
    rw [ih (fun y hy => h y (by simp [hy]))]
    rfl

theorem readN_u16 (ls : List Nat) (h : ∀ x ∈ ls, x < 2 ^ 16) (rest : Bytes) :
    readN ls.length u16 (ls.flatMap w16 ++ rest) = .ok (ls, rest) := by
  induction ls with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.length_cons, readN, List.flatMap_cons, List.append_assoc, bind_apply]
    rw [u16_w16 (h x (by simp))]
    simp only [bind_apply]
    rw [ih (fun y hy => h y (by simp [hy]))]
    rfl

theorem flatMap_w32_length (ls : List Nat) : (ls.flatMap w32).length = 4 * ls.length := by
  induction ls with
  | nil => rfl
  | cons x xs ih => rw [List.flatMap_cons, List.length_append, ih]; simp [w32]; omega

theorem flatMap_w16_length (ls : List Nat) : (ls.flatMap w16).length = 2 * ls.length := by
  induction ls with
  | nil => rfl
  | cons x xs ih => rw [List.flatMap_cons, List.length_append, ih]; simp [w16]; omega

/-- `AttrValues` roundtrip: `readN` over `decodeV`, one value at a time. -/
theorem readN_decodeV (xs : List Val) (bs rest : Bytes) (h : encL I xs = .ok bs)
    (hw : WFL utf8 xs) : readN xs.length (decodeV utf8) (bs ++ rest) = .ok (xs, rest) := by
  induction xs generalizing bs with
  | nil => simp [encL] at h; subst h; rfl
  | cons x xs ih =>
    obtain ⟨a, b', ha, hb, rfl⟩ := encL_cons h
    simp only [WFL] at hw
    simp only [List.length_cons, readN, List.append_assoc, bind_apply]
    rw [decodeV_encV utf8 I x a _ ha hw.1]
    simp only [bind_apply]
    rw [ih b' hb hw.2]
    rfl

/-- The bytes `encRec` writes for a `CREATE_NODE`, block by block. -/
theorem encRec_createNode (ids : C.T) (ls as : List Nat) (rows : List Val) (idb rb : Bytes)
    (hl : ls.length < 2 ^ 16) (ha : as.length < 2 ^ 16)
    (hid : C.enc ids = .ok idb) (hrb : encL I rows = .ok rb)
    (hrows : rows.length = C.count ids * as.length) (hc : 1 ≤ C.count ids) :
    encRec C I (.createNode ids ls as rows) =
      .ok (w32 OP_CREATE_NODE ++ w32 (C.count ids) ++ (w16 ls.length ++ ls.flatMap w32 ++
        (w16 as.length ++ as.flatMap w16 ++ (idb ++ (rb ++ []))))) := by
  have h1 : ¬ (C.count ids * as.length ≠ rows.length) := by omega
  have h2 : C.count ids ≠ 0 := by omega
  simp [encRec, seqE, wLabelSet, wAttrIds, wRows, hl, ha, hid, hrb, h1, h2]

/-- **The encoder refuses exactly the two shapes the reader cannot take**
(#2916): a row block that is not `count × attrs`, and a record with no ids. -/
theorem encRec_refuses_shape (ids : C.T) (ls as : List Nat) (rows : List Val) :
    (rows.length ≠ C.count ids * as.length →
      encRec C I (.createNode ids ls as rows) = .error .rowShapeMismatch) ∧
    (rows.length = C.count ids * as.length → C.count ids = 0 →
      encRec C I (.createNode ids ls as rows) = .error .emptyRecord) := by
  refine ⟨fun h => ?_, fun h h0 => ?_⟩
  · simp [encRec]; omega
  · have : rows.length = 0 := by rw [h, h0]; simp
    simp [encRec, h0, this]

/-- **Record roundtrip for `CREATE_NODE`.** `hc` and `hrows` are now the
encoder's own checks (`encRec_createNode` needs them to produce these bytes);
before #2916 they were conditions only the reader enforced. -/
theorem readRec_createNode (ids : C.T) (ls as : List Nat) (rows : List Val) (idb rb rest : Bytes)
    (hl : ls.length < 2 ^ 16) (ha : as.length < 2 ^ 16)
    (hid : C.enc ids = .ok idb) (hrb : encL I rows = .ok rb)
    (hc : 1 ≤ C.count ids) (hc32 : C.count ids < 2 ^ 32)
    (hls : ∀ x ∈ ls, x < 2 ^ 32) (has : ∀ x ∈ as, x < 2 ^ 16)
    (hrows : rows.length = C.count ids * as.length) (hw : WFL utf8 rows) :
    readRec C utf8 ((w32 OP_CREATE_NODE ++ w32 (C.count ids) ++ (w16 ls.length ++ ls.flatMap w32 ++
        (w16 as.length ++ as.flatMap w16 ++ (idb ++ (rb ++ []))))) ++ rest) =
      .ok (.createNode ids ls as rows, rest) := by
  have e1 := flatMap_w32_length ls
  have e2 := flatMap_w16_length as
  have hlen := (encL_len I rows rb hrb).2
  unfold readRec
  simp only [List.append_nil, List.append_assoc, bind_apply]
  rw [u32_w32 (by decide)]
  simp only [OP_CREATE_NODE, ↓reduceIte, bind_apply]
  rw [u32_w32 hc32]
  simp only [Nat.pos_iff_ne_zero.mp hc, ite_false, rLabelSet, bind_apply]
  rw [u16_w16 hl]
  simp only [bind_apply, guardCount, flatMap_w32_length]
  rw [if_neg (by simp only [List.length_append, e1, e2, w16, w32, le_length]; omega)]
  simp only
  rw [readN_u32 ls hls]
  simp only [rAttrIds, bind_apply]
  rw [u16_w16 ha]
  simp only [bind_apply, guardCount, flatMap_w16_length]
  rw [if_neg (by simp only [List.length_append, e1, e2, w16, w32, le_length]; omega)]
  simp only
  rw [readN_u16 as has]
  simp only [bind_apply]
  rw [C.roundtrip ids idb _ hid]
  simp only [rRows, bind_apply, guardCount]
  have hlen := (encL_len I rows rb hrb).2
  rw [if_neg (by simp [MIN_VALUE_BYTES]; omega)]
  simp only
  rw [← hrows, readN_decodeV utf8 I rows rb rest hrb hw]
  rfl

theorem readRec_addAttribute (id : Nat) (name rest : Bytes) (hid : id < 2 ^ 16)
    (hn : name.length + 1 < 2 ^ 64) (hv : utf8 name = true) :
    readRec C utf8 ((w32 OP_ADD_ATTRIBUTE ++ w16 id ++ wString name) ++ rest) =
      .ok (.addAttribute id name, rest) := by
  unfold readRec
  simp only [List.append_assoc, bind_apply]
  rw [u32_w32 (by decide)]
  simp only [OP_CREATE_NODE, OP_ADD_ATTRIBUTE, Nat.reduceEqDiff, ↓reduceIte, bind_apply]
  rw [u16_w16 hid]
  simp only [bind_apply]
  rw [rString_wString utf8 name rest hn hv]
  rfl

/-! ## Framing: a payload is its records, back to back -/

/-- `Records::next` (`records.rs:1242-1249`): stop when the reader is empty,
yield one record otherwise, fuse on the first error. -/
def readAll : Nat → R (List (Rec C.T))
  | 0 => fun _ => .error .outOfFuel
  | fuel + 1 => fun r =>
    if r = [] then .ok ([], [])
    else match readRec C utf8 r with
      | .error e => .error e
      | .ok (x, r') => match readAll fuel r' with
        | .error e => .error e
        | .ok (xs, r'') => .ok (x :: xs, r'')

/-- **Record framing roundtrips**: if each record's bytes decode to that
record whatever follows them — which `readRec_createNode` and
`readRec_addAttribute` establish — then the concatenation decodes to the
list, in order, with nothing left over and nothing skipped. -/
theorem readAll_concat :
    ∀ (items : List (Bytes × Rec C.T)),
    (∀ p ∈ items, p.1 ≠ [] ∧ ∀ rest, readRec C utf8 (p.1 ++ rest) = .ok (p.2, rest)) →
    ∀ fuel, items.length < fuel →
    readAll C utf8 fuel (items.map Prod.fst).flatten = .ok (items.map Prod.snd, [])
  | [], _, fuel, hf => by
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    simp [readAll]
  | (b, x) :: items, h, fuel, hf => by
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    have hb := h (b, x) (by simp)
    have ih := readAll_concat items (fun p hp => h p (by simp [hp])) f (by simp at hf; omega)
    simp only [List.map_cons, List.flatten_cons, readAll]
    rw [if_neg (by simp [hb.1])]
    rw [hb.2]
    simp only [ih]

/-! ## The shapes the encoder refuses (formerly: counterexamples) -/

/-- A toy `IdList` for the `#guard`s only: one `u32` id per record. -/
def toyIds : IdCodec where
  T := BitVec 32
  count := fun _ => 1
  enc := fun t => .ok (w32 t.toNat)
  dec := fun _ => f32
  roundtrip := by intro t bs rest h; cases h; exact f32_w32 t rest
  good := fun _ => good_f32

def encToy (r : Rec (BitVec 32)) : Except EErr Bytes := encRec toyIds (fun _ => false) r
def readToy : R (Rec (BitVec 32)) := readRec toyIds (fun _ => true)

def isOk {α} : Except DErr α → Bool
  | .ok _ => true
  | .error _ => false

-- The wave-1 counterexample (a CREATE_NODE with one id, one attribute and
-- **two** rows was written, and its spare row's tag `T_INT64 = 0x2000` was then
-- read as the next opcode) no longer holds: #2916 (`2c874022a`) refuses it.
def eerr {α} : Except EErr α → Option EErr | .error e => some e | .ok _ => none
#guard eerr (encToy (.createNode 7 [] [0] [.int 1, .int 2])) == some .rowShapeMismatch
#guard eerr (encToy (.createNode 7 [] [0, 1] [.int 1])) == some .rowShapeMismatch
-- The right shape is written and reads back, with a record after it.
#guard match encToy (.createNode 7 [] [0] [.int 1]) with
  | .ok bs =>
    match readToy (bs ++ w32 OP_ADD_ATTRIBUTE ++ w16 0 ++ wString [0x61]) with
    | .ok (.createNode _ [] [0] [.int 1], rest) => isOk (readToy rest)
    | _ => false
  | _ => false
#guard (OP_CREATE_NODE, T_ARRAY) = (3, 8)

end FalkorCodec
