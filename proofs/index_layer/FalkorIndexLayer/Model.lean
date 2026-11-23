/-
# The model: `graph/src/index/{mod.rs,indexer.rs}` line by line

Executable definitions only; proofs are in `Proofs.lean`, counterexamples in
`Bugs.lean`.  Every definition cites the Rust it follows.

## Abstractions (and why they are sound for the stated properties)

* **Bytes are `Nat`s.** A Rust `&[u8]` is a `List Nat`; theorems that need
  it carry `∀ b ∈ s, b < 256`.
* **f64 is `ENum`.** Every property below is about `=`, `<`, `<=` between
  doubles, never arithmetic. Any finite set of doubles embeds order-
  isomorphically into `Int`, so a finite double is `ENum.fin (i : Int)`;
  `±inf` and `NaN` are separate constructors with IEEE comparison rules
  (NaN compares false with everything). `-0.0` and `0.0` are the same `fin 0`
  (IEEE `==` says they are equal, and both RediSearch and Cypher use `==`).
* **`i as f64` is exact.** The only callers (`value_to_numeric`,
  `Document::set`) are reached for query constants only after
  `int_loses_f64_precision` said "exact" (`int_loses_f64_precision_sound`
  proves that is right); stored ints above 2^53 are a known modelling gap.
* **RediSearch is a specification, not code.** `rsMatch` states the LLAPI
  query semantics documented by RediSearch (numeric range filter, TAG exact
  token, TAG lexicographic range, union/intersection, empty node) and the
  documented indexing rule that an empty TAG value is not indexed unless the
  field was created with `INDEXEMPTY` (which FalkorDB never sets). That is the
  axiomatised boundary: it is written as ordinary definitions, so the project
  contains no Lean `axiom`.
* **A document is a map field → stored values.** `Document::set` appends
  `(field, value)` pairs; RediSearch groups them per field. Because distinct
  attributes get distinct field names (`range:{attr}`, `range:{attr}:numeric:arr`,
  `range:{attr}:string:arr`), `docOf` computes the per-field lists directly.
-/

namespace IndexLayer

abbrev Bytes := List Nat
abbrev Attr := Nat

/-! ## Doubles -/

/-- An IEEE double up to order-isomorphism (see header). -/
inductive ENum where
  | negInf
  | fin (i : Int)
  | posInf
  | nan
  deriving DecidableEq, Repr

namespace ENum
/-- IEEE `<` (false whenever a NaN is involved). -/
def lt : ENum → ENum → Bool
  | negInf, fin _ => true
  | negInf, posInf => true
  | fin a, fin b => decide (a < b)
  | fin _, posInf => true
  | _, _ => false

/-- IEEE `==`. -/
def eq : ENum → ENum → Bool
  | negInf, negInf => true
  | posInf, posInf => true
  | fin a, fin b => decide (a = b)
  | _, _ => false

def le (a b : ENum) : Bool := lt a b || eq a b

def isFinite : ENum → Bool
  | fin _ => true
  | _ => false
end ENum

/-! ## Cypher values (the subset stored in / compared against a range index) -/

/-- `runtime::value::Value`, restricted to what can be stored as a property.
`temporal k ts`: `Datetime/Date/Time/Duration` (kind `k`, raw `ts`). -/
inductive Val where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | flt (f : ENum)
  | str (s : Bytes)
  | temporal (kind : Nat) (ts : Int)
  | point (lat lon : Int)
  | list (xs : List Val)
  deriving Repr

/-- Byte-wise lexicographic `<` (Rust `str::cmp`, RediSearch TrieMap order). -/
def lexLt : Bytes → Bytes → Bool
  | [], [] => false
  | [], _ :: _ => true
  | _ :: _, [] => false
  | a :: as, b :: bs => if a < b then true else if a = b then lexLt as bs else false

def numOf : Val → Option ENum
  | .int i => some (.fin i)
  | .flt f => some f
  | _ => none

/-- Cypher `=` (three-valued: `none` is null). Mirrors `Value` equality in
`runtime/value.rs` / C `SIValue_Compare` for scalars: Int and Float compare
numerically, Bool only equals Bool, temporals only the same kind. Lists are
compared element-wise only for the one-level case needed here. -/
def cyEqScalar : Val → Val → Option Bool
  | .null, _ => none
  | _, .null => none
  | .bool a, .bool b => some (a == b)
  | .str a, .str b => some (a == b)
  | .temporal k a, .temporal k' b => some (k == k' && a == b)
  | .point a b, .point c d => some (a == c && b == d)
  | .list _, .list _ => none
  | a, b => match numOf a, numOf b with
    | some x, some y => some (ENum.eq x y)
    | _, _ => some false

/-- Cypher `<` for the comparable pairs; `none` (null) for incomparable
types, e.g. `true < 1`, `'a' < 1`, a date vs a number. -/
def cyLt : Val → Val → Option Bool
  | .bool a, .bool b => some (!a && b)
  | .str a, .str b => some (lexLt a b)
  | .temporal k a, .temporal k' b => if k = k' then some (decide (a < b)) else none
  | a, b => match numOf a, numOf b with
    | some x, some y => some (ENum.lt x y)
    | _, _ => none

def cyLe (a b : Val) : Option Bool :=
  match cyLt a b, cyEqScalar a b with
  | some l, some e => some (l || e)
  | _, _ => none

/-! ## `IndexQuery<Value>` (`index/mod.rs`) and its Cypher meaning -/

inductive IQ where
  | equal (key : Attr) (v : Val)
  | range (key : Attr) (min max : Option Val) (imin imax : Bool)
  | and (qs : List IQ)
  | or (qs : List IQ)
  | inList (key : Attr) (list : Val)
  | arrayContains (key : Attr) (v : Val)
  deriving Repr

/-- An entity's properties (`none` = absent). -/
abbrev Props := Attr → Option Val

def propOr (p : Props) (k : Attr) : Val := (p k).getD .null

def isTrue : Option Bool → Bool
  | some true => true
  | _ => false

/-- The Cypher predicate the optimizer replaced by the index query
(`planner/optimizer/utilize_index.rs` `build_op_query`): `n.k = v`,
`c < n.k` (`min`), `n.k < c` (`max`), `n.k IN list`, `v IN n.k`. -/
def holds (p : Props) : IQ → Bool
  | .equal k v => isTrue (cyEqScalar (propOr p k) v)
  | .range k mn mx imn imx =>
      (match mn with
       | none => true
       | some m => isTrue (if imn then cyLe m (propOr p k) else cyLt m (propOr p k))) &&
      (match mx with
       | none => true
       | some m => isTrue (if imx then cyLe (propOr p k) m else cyLt (propOr p k) m))
  | .and qs => qs.attach.all (fun ⟨q, _⟩ => holds p q)
  | .or qs => qs.attach.any (fun ⟨q, _⟩ => holds p q)
  | .inList k (.list xs) => xs.any (fun x => isTrue (cyEqScalar (propOr p k) x))
  | .inList _ _ => false
  | .arrayContains k v => match propOr p k with
      | .list ys => ys.any (fun y => isTrue (cyEqScalar y v))
      | _ => false

/-! ## `tag_encode_lower` and `tag_key` (`index/mod.rs`, #2975) -/

def HEX : List Nat := [48,49,50,51,52,53,54,55,56,57,97,98,99,100,101,102]  -- "0123456789abcdef"
def hexDigit (n : Nat) : Nat := HEX.getD n 0

/-- The escaped bytes: `0..=0x21`, `[` and `\` (`tag_encode_lower`). -/
def escaped (b : Nat) : Bool := b ≤ 0x21 || b == 0x5b || b == 0x5c

/-- One byte of `tag_encode_lower`: `0..=0x21` ↦ `0x21, 0x22 + b`; `[` ↦ `[[`; `\` ↦ `[]`;
every other byte passes through. -/
def encByte (b : Nat) : Bytes :=
  if b ≤ 0x21 then [0x21, 0x22 + b]
  else if b = 0x5b then [0x5b, 0x5b]
  else if b = 0x5c then [0x5b, 0x5d]
  else [b]

/-- The byte loop of `tag_encode_lower`. -/
def tagEncB : Bytes → Bytes
  | [] => []
  | b :: bs => encByte b ++ tagEncB bs

/-- `tag_encode_lower`: the empty string is the lone byte `0x21`, which no other string
encodes to and which sorts before every other key. -/
def tagEnc (s : Bytes) : Bytes := if s = [] then [0x21] else tagEncB s

/-- `BOOL_TRUE_KEY` / `BOOL_FALSE_KEY`: `0xFF, 't'` / `0xFF, 'f'`. -/
def boolKey (b : Bool) : Bytes := [0xFF, if b then 0x74 else 0x66]

/-- `STRING_RANGE_END`: closes an open-ended string range below the Bool keys. -/
def STRING_RANGE_END : Bytes := [0xFE]

/-- `hex_nibble` (`mod.rs`). -/
def hexNibble (c : Nat) : Nat :=
  if 48 ≤ c ∧ c ≤ 57 then c - 48 else if 97 ≤ c ∧ c ≤ 102 then c - 97 + 10 else 0

/-- A decoder of the byte loop (none exists in Rust; used only to prove `tagEnc` injective). -/
def tagDecB : Bytes → Bytes
  | 0x21 :: c :: rest => (c - 0x22) :: tagDecB rest
  | 0x5b :: 0x5b :: rest => 0x5b :: tagDecB rest
  | 0x5b :: 0x5d :: rest => 0x5c :: tagDecB rest
  | c :: rest => c :: tagDecB rest
  | [] => []

def tagDec (t : Bytes) : Bytes := if t = [0x21] then [] else tagDecB t

/-! ## Document keys (`hex_encode_into` `mod.rs`, `decode_id` `mod.rs`) -/

/-- `u64::to_le_bytes` (little-endian: byte `i` is `id / 256^i % 256`). -/
def leBytesN : Nat → Nat → List Nat
  | 0, _ => []
  | n + 1, id => id % 256 :: leBytesN n (id / 256)
def leBytes (id : Nat) : List Nat := leBytesN 8 id

def hexEncode : List Nat → List Nat
  | [] => []
  | b :: bs => hexDigit (b / 16) :: hexDigit (b % 16) :: hexEncode bs

/-- Pair up hex chars into bytes (the loop in `decode_id`). -/
def hexDecode : List Nat → List Nat
  | h :: l :: rest => (hexNibble h * 16 + hexNibble l) :: hexDecode rest
  | _ => []

/-- `u64::from_le_bytes`. -/
def fromLe : List Nat → Nat
  | [] => 0
  | b :: bs => b + 256 * fromLe bs

def nodeKey (id : Nat) : List Nat := hexEncode (leBytes id)
def decodeId (key : List Nat) : Nat := fromLe (hexDecode key)
def edgeKey (src dst eid : Nat) : List Nat := nodeKey src ++ nodeKey dst ++ nodeKey eid

/-! ## `int_loses_f64_precision` (`mod.rs`) -/

def MASK : Nat := 0x7FF0000000000000
/-- `i.unsigned_abs() & 0x7FF0_0000_0000_0000 != 0`. -/
def intLosesPrecision (i : Int) : Bool := i.natAbs &&& MASK != 0

/-! ## RediSearch documents and queries (the axiomatised boundary) -/

/-- Field of the RS spec. `main` = `range:{attr}` (NUMERIC|TAG, created
at `mod.rs`), `geo` = `range:{attr}:geo` (GEO, #2975), `numArr` = `range:{attr}:numeric:arr`,
`strArr` = `range:{attr}:string:arr` (`mod.rs`). -/
inductive Sub where
  | main | geo | numArr | strArr
  deriving DecidableEq, Repr

/-- A value stored under one field. -/
inductive RsV where
  | num (v : ENum)
  | tag (t : Bytes)
  | geo (lat lon : Int)
  deriving DecidableEq, Repr

/-- RediSearch LLAPI query nodes built by `build_query_node`. -/
inductive QN where
  /-- `RediSearch_CreateNumericNode(field, max, min, include_max, include_min)` -/
  | numeric (k : Attr) (s : Sub) (lo hi : ENum) (ilo ihi : Bool)
  /-- `RediSearch_CreateTagNode(field)` + `RediSearch_CreateTagTokenNode(tok)` -/
  | tagToken (k : Attr) (s : Sub) (tok : Bytes)
  /-- `CreateTagNode` + `RediSearch_CreateTagLexRangeNode(min, max, imin, imax)`,
  `none` = the NULL pointer = open bound. -/
  | tagLex (k : Attr) (lo hi : Option Bytes) (ilo ihi : Bool)
  | inter (cs : List QN)
  | union (cs : List QN)
  | empty
  deriving Repr

/-- RS indexing rule: a TAG value is stored as one token (separator is `\x01`,
case-sensitive, `mod.rs`); an empty TAG value is not indexed
(RediSearch: "empty values are not indexed unless INDEXEMPTY"). Numeric and
geo values are stored as is. -/
def rsStore : RsV → List RsV
  | .tag [] => []
  | v => [v]

def inNum (lo hi : ENum) (ilo ihi : Bool) (v : ENum) : Bool :=
  (if ilo then ENum.le lo v else ENum.lt lo v) && (if ihi then ENum.le v hi else ENum.lt v hi)

def inLex (lo hi : Option Bytes) (ilo ihi : Bool) (t : Bytes) : Bool :=
  (match lo with | none => true | some l => if ilo then lexLt l t || l == t else lexLt l t) &&
  (match hi with | none => true | some h => if ihi then lexLt t h || t == h else lexLt t h)

/-- A document: stored values per `(attr, sub-field)`. -/
abbrev Doc := Attr → Sub → List RsV

/-- RediSearch query semantics over one document (the documented spec). -/
def rsMatch (d : Doc) : QN → Bool
  | .numeric k s lo hi ilo ihi => (d k s).any fun | .num v => inNum lo hi ilo ihi v | _ => false
  | .tagToken k s tok => (d k s).any fun | .tag t => t == tok | _ => false
  | .tagLex k lo hi ilo ihi => (d k .main).any fun | .tag t => inLex lo hi ilo ihi t | _ => false
  | .inter cs => cs.attach.all fun ⟨c, _⟩ => rsMatch d c
  | .union cs => cs.attach.any fun ⟨c, _⟩ => rsMatch d c
  | .empty => false

/-! ## `Document::set` (`mod.rs`) for a Range field -/

/-- What `Document::set` hands RediSearch for one attribute value of a Range
field, per sub-field (before `rsStore`). `Null/Map/Node/..` are
`unreachable!()` in Rust; callers never pass them (attributes are never
stored as null), so they produce nothing here. -/
def setRange (v : Val) (s : Sub) : List RsV :=
  match s, v with
  | .main, .bool b => [.tag (boolKey b)]                           -- mod.rs (#2975: a TAG key)
  | .main, .int i => [.num (.fin i)]                               -- mod.rs
  | .main, .flt f => [.num f]                                      -- mod.rs
  | .main, .str t => [.tag (tagEnc t)]                             -- mod.rs
  | .main, .temporal _ _ => []                                     -- mod.rs (#3076: not indexed)
  | .geo, .point la lo => [.geo la lo]                             -- mod.rs (#2975: `geo_name`)
  | .numArr, .list xs => xs.filterMap fun                          -- mod.rs
      | .int i => some (.num (.fin i))
      | .flt f => some (.num f)
      | _ => none
  | .strArr, .list xs => xs.filterMap fun                          -- mod.rs
      | .bool b => some (.tag (boolKey b))
      | .str t => some (.tag (tagEnc t))
      | _ => none
  | _, _ => []

/-- The RS document for an entity: every indexed attribute's value, run
through `Document::set` and the RS store rule. `fields k` = the label has a
Range field for attribute `k`. -/
def docOf (fields : Attr → Bool) (p : Props) : Doc := fun k s =>
  if fields k then
    match p k with
    | some v => (setRange v s).flatMap rsStore
    | none => []
  else []

/-! ## `build_query_node` (`mod.rs`) -/

/-- `value_to_numeric` (`mod.rs`). -/
def valueToNumeric : Val → Option ENum
  | .int i => some (.fin i)
  | .flt f => some f
  | _ => none

/-- `build_numeric_range_node` (`mod.rs`). Open bounds become
`RSRANGE_NEG_INF`/`RSRANGE_INF`, and since #3081 (`ff3d24ba7`) an absent
bound is inclusive whatever the caller's flag (`include_min || min.is_none()`,
`mod.rs`), so the infinity standing in for it is itself selected. -/
def buildNumRange (fields : Attr → Bool) (k : Attr) (mn mx : Option Val) (imn imx : Bool) :
    Option QN := do
  let lo ← match mn with
    | some v => valueToNumeric v
    | none => some .negInf
  let hi ← match mx with
    | some v => valueToNumeric v
    | none => some .posInf
  if fields k then some (.numeric k .main lo hi (imn || mn.isNone) (imx || mx.isNone)) else none

/-- `build_string_range_node` (`mod.rs`). An open upper end is `STRING_RANGE_END`, inclusive
(#2975), so the range stops below the Bool keys. -/
def buildStrRange (fields : Attr → Bool) (k : Attr) (mn mx : Option Bytes) (imn imx : Bool) :
    Option QN :=
  if fields k then
    match mn, mx with
    | some lo, some hi =>
      if lo = hi then
        if !(imn && imx) then some .empty                                        -- mod.rs (#3072)
        else some (.tagToken k .main (tagEnc lo))                                -- mod.rs
      else some (.tagLex k (some (tagEnc lo)) (some (tagEnc hi)) imn imx)
    | lo, hi => some (.tagLex k (lo.map tagEnc) (some ((hi.map tagEnc).getD STRING_RANGE_END))
        imn (imx || hi.isNone))
  else none

def strBound : Option Val → Option (Option Bytes)
  | none => some none
  | some (.str s) => some (some s)
  | some _ => none

def isStrVal : Option Val → Bool
  | some (.str _) => true
  | _ => false

/-- `tag_key` (`mod.rs`): the TAG key a String or Bool is indexed and looked up under. -/
def tagKey : Val → Option Bytes
  | .str s => some (tagEnc s)
  | .bool b => some (boolKey b)
  | _ => none

/-- The `IndexQuery::Equal` arms of `build_query_node` (`mod.rs`),
factored out because `InList` re-enters them (`mod.rs`). -/
def buildEq (fields : Attr → Bool) (k : Attr) (v : Val) : Option QN :=
  match valueToNumeric v with
  | some d => if fields k then some (.numeric k .main d d true true) else none      -- mod.rs
  | none => match tagKey v with
    | some t => if fields k then some (.tagToken k .main t) else none               -- mod.rs
    | none => none                                                                   -- mod.rs

/-- `build_query_node`. `none` is Rust's null pointer. -/
def buildQ (fields : Attr → Bool) : IQ → Option QN
  | .equal k v => buildEq fields k v
  | .range k mn mx imn imx =>
      if isStrVal mn || isStrVal mx then
        match strBound mn, strBound mx with
        | some a, some b => buildStrRange fields k a b imn imx
        | _, _ => none                                                               -- mod.rs
      else buildNumRange fields k mn mx imn imx
  | .and qs =>                                                                       -- mod.rs
      (qs.attach.mapM (m := Option) (fun (x : {q // q ∈ qs}) =>
        have := List.sizeOf_lt_of_mem x.2; buildQ fields x.1)).map QN.inter
  | .or [] => some .empty                                                            -- mod.rs
  | .or (q :: qs) => some (.union ((q :: qs).attach.filterMap (fun (x : {q' // q' ∈ q :: qs}) =>
        have := List.sizeOf_lt_of_mem x.2; buildQ fields x.1)))                      -- mod.rs
  | .inList k (.list xs) =>
      if xs.isEmpty then some .empty
      else some (.union (xs.filterMap fun x => buildEq fields k x))                  -- mod.rs
  | .inList _ _ => none
  | .arrayContains k v =>                                                            -- mod.rs
      if fields k then
        match v with
        | .int i => some (.numeric k .numArr (.fin i) (.fin i) true true)
        | .flt f => some (.numeric k .numArr f f true true)
        | .bool b => some (.tagToken k .strArr (boolKey b))
        | .str s => some (.tagToken k .strArr (tagEnc s))
        | _ => none
      else none
decreasing_by
  all_goals simp_wf
  all_goals (first | omega | (simp only [List.cons.sizeOf_spec] at *; omega))

/-- `Index::query` (`mod.rs`): a null node yields the empty iterator. -/
def indexHit (fields : Attr → Bool) (p : Props) (q : IQ) : Bool :=
  match buildQ fields q with
  | some n => rsMatch (docOf fields p) n
  | none => false

/-! ## The runtime side (`runtime/ops/node_by_index_scan.rs`) -/

def notNull : Val → Bool
  | .null => false
  | _ => true

/-- `IndexQuery::in_list` (`mod.rs`), called by `evaluate_index_query`
(`node_by_index_scan.rs`): one `InList` of the non-null items (#2975: a `date(...)` item is
kept, so `can_be_served` sees it). -/
def expandIn (k : Attr) : Val → Option IQ
  | .list xs => some (.inList k (.list (xs.filter notNull)))
  | _ => none

/-- `is_indexable`, local to `IndexQuery::can_be_served` (`mod.rs`), on the RediSearch build
(a Bool has a TAG key of its own). A temporal, a Point or a list has no Equal arm. -/
def indexable : Val → Bool
  | .int i => !intLosesPrecision i
  | .flt _ | .str _ | .bool _ | .null => true
  | _ => false

def isBool : Val → Bool
  | .bool _ => true
  | _ => false

/-- `IndexQuery::can_be_served` (`mod.rs`, #2975). -/
def canUtilize : IQ → Bool
  | .equal _ v => indexable v
  | .range _ mn mx _ _ =>
      (mn.all fun v => indexable v && !isBool v) && (mx.all fun v => indexable v && !isBool v)
  | .and qs => !qs.isEmpty && qs.attach.all fun ⟨q, _⟩ => canUtilize q
  | .or qs => !qs.isEmpty && qs.attach.all fun ⟨q, _⟩ => canUtilize q
  | .arrayContains _ v => notNull v && indexable v
  | .inList _ (.list xs) => !xs.isEmpty && xs.all indexable
  | .inList _ _ => false

/-- What `NodeByIndexScanOp` emits for one entity of the label, followed by
the post-filter when the optimizer kept it (`keep`). -/
def scanEmits (fields : Attr → Bool) (keep : Bool) (p : Props) (q : IQ) : Bool :=
  (if canUtilize q then indexHit fields p q else true) && (!keep || holds p q)

/-! ## Pending population tickets (`mod.rs` `PendingSlots`, `mod.rs`) -/

structure Slots where
  gen : Nat
  cur : Int
  stale : Int
  deriving DecidableEq, Repr

/-- `increment_pending_for_generation` (`mod.rs`). -/
def Slots.inc (s : Slots) (g : Nat) : Slots :=
  if g = s.gen then { s with cur := s.cur + 1 } else { s with stale := s.stale + 1 }

/-- `try_decrement_pending_for_generation` (`mod.rs`). -/
def Slots.dec (s : Slots) (g : Nat) : Slots :=
  if g = s.gen then (if s.cur > 0 then { s with cur := s.cur - 1 } else s)
  else (if s.stale > 0 then { s with stale := s.stale - 1 } else s)

/-- `bump_id` (`mod.rs`): new generation, current work becomes stale. -/
def Slots.bump (s : Slots) (g' : Nat) : Slots :=
  { gen := g', cur := 0, stale := s.stale + s.cur }

/-- `pending_count_for_generation` (`mod.rs`). -/
def Slots.countFor (s : Slots) (g : Nat) : Int := if g = s.gen then s.cur else s.stale

/-! ## Index maintenance: `Indexer::commit` (`indexer.rs`) -/

/-- RS doc table for one label: entity id ↦ stored document. -/
abbrev Table := Nat → Option Doc

/-- `add_document` with `REDISEARCH_ADD_REPLACE` (`mod.rs`). -/
def Table.add (t : Table) (id : Nat) (d : Doc) : Table := fun j => if j = id then some d else t j
/-- `delete_document` (`mod.rs`). -/
def Table.del (t : Table) (id : Nat) : Table := fun j => if j = id then none else t j

/-- `Indexer::commit`: all adds of the label first, then all removes
(`indexer.rs`). Documents are built from the *committed* graph
(`graph.rs` `commit_index_kind`, `doc.set`). -/
def commit (t : Table) (build : Nat → Doc) (adds removes : List Nat) : Table :=
  removes.foldl Table.del (adds.foldl (fun t id => t.add id (build id)) t)

/-- What a query sees for an id: an absent document matches nothing. -/
def Table.sees (t : Table) (id : Nat) (q : QN) : Bool :=
  match t id with
  | some d => rsMatch d q
  | none => false

end IndexLayer
