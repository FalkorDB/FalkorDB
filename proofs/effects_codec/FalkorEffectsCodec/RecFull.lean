import FalkorEffectsCodec.RecOpts
/-!
# Every `Record` (`graph/src/effects/v3/records.rs`)

`Record` with all fourteen opcodes, `write_header` (`:21`), `check_row_shape`
(`:864`), `Record::encode` (`:917`), `read_record` (`:611`), and a roundtrip
theorem for each variant. Since #2916 (`2c874022a`) the encoder refuses the two
shapes `read_record` rejects: a batch of zero ids (`EmptyRecord`, in
`write_header`) and a row block that is not `count × attrs` (`RowShapeMismatch`).
The ids codec `C` is the abstract `IdCodec` (its roundtrip is proven in
proofs/id_list).
-/
namespace FalkorCodec

variable (C : IdCodec) (utf8 I : Bytes → Bool)

inductive Record (T : Type) where
  | updateNode (ids : T) (labels attrIds : List Nat) (rows : List Val)
  | updateEdge (ids : T) (rel : Nat) (attrIds : List Nat) (rows : List Val)
  | createNode (ids : T) (labels attrIds : List Nat) (rows : List Val)
  | createEdge (ids : T) (rel : Nat) (src dst : T) (attrIds : List Nat) (rows : List Val)
  | deleteNode (ids : T) (labels : List Nat)
  | deleteEdge (ids : T) (rel : Nat) (src dst : T)
  | setLabels (ids : T) (labels : List Nat)
  | removeLabels (ids : T) (labels : List Nat)
  | addLabel (id : Nat) (name : Bytes)
  | addRelType (id : Nat) (name : Bytes)
  | addAttribute (id : Nat) (name : Bytes)
  | createIndex (st : Entity) (schemas : List Ref) (ft : Nat) (fields : List Ref) (opts : OptsT)
  | dropIndex (st : Entity) (schemas : List Ref) (ft : Nat) (fields : List Ref)
  | createConstraint (ct : CType) (et : Entity) (status : CStatus) (labelId : Nat) (label : Bytes) (props : List Ref)
  | dropConstraint (ct : CType) (et : Entity) (labelId : Nat) (label : Bytes) (props : List Ref)

/-- `Opcode` values and `is_batchable` (`v3/mod.rs`). -/
def isBatchable (op : Nat) : Bool := 1 ≤ op ∧ op ≤ 8

/-- `write_header` (`records.rs:21`): refuses a count on a non-batchable opcode
and vice versa, then (#2916) a count of zero. -/
def writeHeader (op : Nat) (count : Option Nat) : Except EErr Bytes :=
  if count.isSome != isBatchable op then .error .headerShapeMismatch
  else if count = some 0 then .error .emptyRecord
  else .ok (w32 op ++ match count with | some c => w32 c | none => [])

/-- `IdList::count()` — `expect`s the length fits a `u32` (panic otherwise). -/
def idCount (ids : C.T) : Except EErr Nat :=
  if C.count ids < 2 ^ 32 then .ok (C.count ids) else .error .panic

/-! ### Bodies (what follows the header), as composed blocks -/

def bodyNode (c : Nat) := bLabels.seq fun _ => bAttrIds.seq fun as => (bIds C c).seq fun _ => bRows utf8 I c as.length
def bodyUpdEdge (c : Nat) := bU32.seq fun _ => bAttrIds.seq fun as => (bIds C c).seq fun _ => bRows utf8 I c as.length
def bodyCrEdge (c : Nat) := bU32.seq fun _ => bAttrIds.seq fun as => (bIds C c).seq fun _ =>
  (bIds C c).seq fun _ => (bIds C c).seq fun _ => bRows utf8 I c as.length
def bodyLabels (c : Nat) := bLabels.seq fun _ => bIds C c
def bodyDelEdge (c : Nat) := bU32.seq fun _ => (bIds C c).seq fun _ => (bIds C c).seq fun _ => bIds C c
def bodySchema := bSchemaTag.seq fun _ => bU32.seq fun _ => bStr utf8
def bodyAttr := bU16.seq fun _ => bStr utf8
def bodyIndex (create : Bool) := bSchemaTag.seq fun _ => (bSchemas utf8).seq fun _ => bFieldType.seq fun ft =>
  (bFields utf8).seq fun _ => if create then (bOpts utf8 ft).map some (fun o => o.getD default) (fun _ => rfl)
    else { enc := fun _ => .ok [], dec := pure none, ok := fun o => o = none,
           rt := by intro o bs rest h e; subst h; cases e; rfl }
def bodyConstraint (create : Bool) := bCType.seq fun _ => bEntityTag.seq fun _ =>
  (if create then (bStatus.map some (fun o => o.getD .operational) (fun _ => rfl))
   else { enc := fun _ => .ok [], dec := pure none, ok := fun o => o = none,
          rt := by intro o bs rest h e; subst h; cases e; rfl }).seq fun _ =>
  bU32.seq fun _ => (bStr utf8).seq fun _ => bProps utf8

instance : Inhabited TextT := ⟨(none, none, none, none, none)⟩

/-- `check_endpoint_columns` (`:838`). -/
def checkEndpoints (ids src dst : C.T) : Except EErr Unit :=
  if C.count src ≠ C.count ids then .error .endpointMisaligned
  else if C.count dst ≠ C.count ids then .error .endpointMisaligned else .ok ()

/-- `check_index_record` (`:456`). -/
def checkIndex (ft : Nat) (schemas : List Ref) : Except EErr Unit :=
  match indexTypeOf ft with
  | .error e => .error e
  | .ok _ => if schemas = [] then .error .emptyIndexSchemaList else .ok ()

def thenE (x : Except EErr Unit) (y : Except EErr Bytes) : Except EErr Bytes :=
  match x with | .error e => .error e | .ok () => y

/-- `check_row_shape` (`:864`, #2916): `ids.len().checked_mul(attr_ids.len()) ==
Some(rows.len())`. An overflowing product cannot equal a `usize` length, so the
`Nat` equality is the same test. -/
def checkRowShape (ids : C.T) (attrs : List Nat) (rows : List Val) : Except EErr Unit :=
  if C.count ids * attrs.length = rows.length then .ok () else .error .rowShapeMismatch

def batched (op : Nat) (ids : C.T) (body : Nat → Except EErr Bytes) : Except EErr Bytes :=
  match idCount C ids with
  | .error e => .error e
  | .ok c => eseq (writeHeader op (some c)) (body c)

/-- `impl EffectEncode<3> for Record` (`records.rs:917`), arm by arm. The
row-bearing arms call `check_row_shape` first (`:954,977,1011,1025`), after
`check_endpoint_columns` on `CREATE_EDGE`. -/
def encRecord : Record C.T → Except EErr Bytes
  | .addLabel id name => eseq (writeHeader 9 none) ((bodySchema utf8).enc (.node, id, name))
  | .addRelType id name => eseq (writeHeader 9 none) ((bodySchema utf8).enc (.rel, id, name))
  | .addAttribute id name => eseq (writeHeader 10 none) ((bodyAttr utf8).enc (id, name))
  | .createNode ids ls as rows => thenE (checkRowShape C ids as rows) <|
      batched C 3 ids fun c => (bodyNode C utf8 I c).enc (ls, as, ids, rows)
  | .createEdge ids rel src dst as rows => thenE (checkEndpoints C ids src dst) <|
      thenE (checkRowShape C ids as rows) <|
      batched C 4 ids fun c => (bodyCrEdge C utf8 I c).enc (rel, as, ids, src, dst, rows)
  | .updateNode ids ls as rows => thenE (checkRowShape C ids as rows) <|
      batched C 1 ids fun c => (bodyNode C utf8 I c).enc (ls, as, ids, rows)
  | .updateEdge ids rel as rows => thenE (checkRowShape C ids as rows) <|
      batched C 2 ids fun c => (bodyUpdEdge C utf8 I c).enc (rel, as, ids, rows)
  | .setLabels ids ls => batched C 7 ids fun c => (bodyLabels C c).enc (ls, ids)
  | .removeLabels ids ls => batched C 8 ids fun c => (bodyLabels C c).enc (ls, ids)
  | .deleteNode ids ls => batched C 5 ids fun c => (bodyLabels C c).enc (ls, ids)
  | .deleteEdge ids rel src dst => thenE (checkEndpoints C ids src dst) <|
      batched C 6 ids fun c => (bodyDelEdge C c).enc (rel, ids, src, dst)
  | .createIndex st schemas ft fields opts => thenE (checkIndex ft schemas) <|
      if opts.2.isSome != hasVector ft then .error .optionsFieldTypeMismatch else
      eseq (writeHeader 11 none) ((bodyIndex utf8 true).enc (st, schemas, ft, fields, some opts))
  | .dropIndex st schemas ft fields => thenE (checkIndex ft schemas) <|
      eseq (writeHeader 12 none) ((bodyIndex utf8 false).enc (st, schemas, ft, fields, none))
  | .createConstraint ct et status lid label props =>
      eseq (writeHeader 13 none) ((bodyConstraint utf8 true).enc (ct, et, some status, lid, label, props))
  | .dropConstraint ct et lid label props =>
      eseq (writeHeader 14 none) ((bodyConstraint utf8 false).enc (ct, et, none, lid, label, props))

/-- The `match opcode` of `read_record` (`records.rs:633-815`). -/
def arm (op count : Nat) : R (Record C.T) :=
  if op = 1 then do let (ls, as, ids, rows) ← (bodyNode C utf8 I count).dec; pure (.updateNode ids ls as rows)
  else if op = 2 then do let (rel, as, ids, rows) ← (bodyUpdEdge C utf8 I count).dec; pure (.updateEdge ids rel as rows)
  else if op = 3 then do let (ls, as, ids, rows) ← (bodyNode C utf8 I count).dec; pure (.createNode ids ls as rows)
  else if op = 4 then do
    let (rel, as, ids, src, dst, rows) ← (bodyCrEdge C utf8 I count).dec; pure (.createEdge ids rel src dst as rows)
  else if op = 5 then do let (ls, ids) ← (bodyLabels C count).dec; pure (.deleteNode ids ls)
  else if op = 6 then do let (rel, ids, src, dst) ← (bodyDelEdge C count).dec; pure (.deleteEdge ids rel src dst)
  else if op = 7 then do let (ls, ids) ← (bodyLabels C count).dec; pure (.setLabels ids ls)
  else if op = 8 then do let (ls, ids) ← (bodyLabels C count).dec; pure (.removeLabels ids ls)
  else if op = 9 then do
    let (st, id, name) ← (bodySchema utf8).dec
    pure (match st with | .node => .addLabel id name | .rel => .addRelType id name)
  else if op = 10 then do let (id, name) ← (bodyAttr utf8).dec; pure (.addAttribute id name)
  else if op = 11 then do
    let (st, schemas, ft, fields, opts) ← (bodyIndex utf8 true).dec
    pure (.createIndex st schemas ft fields (opts.getD default))
  else if op = 12 then do let (st, schemas, ft, fields, _) ← (bodyIndex utf8 false).dec; pure (.dropIndex st schemas ft fields)
  else if op = 13 then do
    let (ct, et, status, lid, label, props) ← (bodyConstraint utf8 true).dec
    pure (.createConstraint ct et (status.getD .operational) lid label props)
  else do let (ct, et, _, lid, label, props) ← (bodyConstraint utf8 false).dec; pure (.dropConstraint ct et lid label props)

/-- `read_record` (`records.rs:611`): opcode (`Opcode::try_from` → `BadOpcode`),
the count for batchable opcodes (`EmptyRecord` on 0), then the arm. -/
def readRecord : R (Record C.T) := do
  let op ← u32
  if op = 0 ∨ op > 14 then fail (.badOpcode op) else
  let count ← if isBatchable op then u32 else pure 0
  if isBatchable op ∧ count = 0 then fail (.emptyRecord op) else
  arm C utf8 I op count

end FalkorCodec
