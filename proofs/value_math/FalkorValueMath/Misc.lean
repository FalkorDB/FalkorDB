import FalkorValueMath.SelfCompare
import FalkorValueMath.Conversion
/-!
# Formatting, memory accounting, attribute access and the v19 value codec

| Lean | Rust (value.rs) |
| --- | --- |
| `fmtDuration` | `Value::format_duration` :296 |
| `HV`, `heapSize`, `amortized` | `Value::heap_size` :342, `amortized_heap_size` :388 |
| `getAttr`, `pointComponent`, `durationComponentPre` | `get_attr` :439, `get_point_component` :458, `get_duration_component` :608 (pre-chrono part) |
| `partialCmp` | `PartialOrd for Value` :1788 |
| `getNumeric` | `Value::get_numeric` :246 |
| `Tok`, `encode`, `decode` | `Encode<19>`/`Decode<19> for Value` :1899/:1985 |
| `Deduper` | `ValuesDeduper` :1853-1896 |
| `validatePoint` | `Point::validate` :144 |
| `jsonFloat` | `DisplayJson` Float arm :1591 |
-/

namespace ValueMath

/-! ## `format_duration` -/

/- `format_duration` after `decompose_duration` returned `(years, months, remaining)`
(`remaining ≥ 0`: it is `ts - anchor` with the anchor the first of the month). -/
def fmtDuration (years months remaining : Int) : String :=
  let days := remaining / 86400
  let r1 := remaining % 86400
  let hours := r1 / 3600
  let r2 := r1 % 3600
  let minutes := r2 / 60
  let seconds := r2 % 60
  let s := "P" ++ (if years ≠ 0 then toString years ++ "Y" else "")
      ++ (if months ≠ 0 then toString months ++ "M" else "")
      ++ (if days ≠ 0 then toString days ++ "D" else "")
      ++ (if hours ≠ 0 ∨ minutes ≠ 0 ∨ seconds ≠ 0 then
            "T" ++ (if hours ≠ 0 then toString hours ++ "H" else "")
                ++ (if minutes ≠ 0 then toString minutes ++ "M" else "")
                ++ (if seconds ≠ 0 then toString seconds ++ "S" else "")
          else "")
  if s.length = 1 then s ++ "T0S" else s

/-- The clock fields are the base-(86400, 3600, 60) digits of `remaining`. -/
theorem fmtDuration_fields (y mo d h mi se : Int) (hd : 0 ≤ d) (hh : 0 ≤ h ∧ h < 24)
    (hm : 0 ≤ mi ∧ mi < 60) (hs : 0 ≤ se ∧ se < 60) :
    let r := d * 86400 + h * 3600 + mi * 60 + se
    r / 86400 = d ∧ r % 86400 / 3600 = h ∧ r % 86400 % 3600 / 60 = mi ∧ r % 86400 % 3600 % 60 = se := by
  intro r; refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- The zero duration prints `PT0S` (ISO 8601). C prints `P` (live: `toString(duration('PT0S'))`). -/
theorem fmtDuration_zero : fmtDuration 0 0 0 = "PT0S" := by decide

theorem fmtDuration_ex : fmtDuration 1 2 (3 * 86400 + 4 * 3600 + 5 * 60 + 6) = "P1Y2M3DT4H5M6S" := by decide
theorem fmtDuration_ex2 : fmtDuration 0 0 90000 = "P1DT1H" := by decide

/-! ## Memory accounting -/

/-- Heap view of a value: payload lengths, capacities and `Arc` strong counts. -/
inductive HV where
  | scalar                                              -- Null..Node: inline
  | rel                                                 -- Relationship
  | str (len cap rc : Nat)
  | list (items : List HV) (cap rc : Nat)               -- List and Path
  | map (entries : List (Nat × Nat × Nat × HV)) (rc : Nat) -- (key len, key cap, key rc, value)
  | vec (len cap rc : Nat)

/-- `size_of` constants on 64-bit targets (checked by `sizes_are_as_modelled` in
graph/tests/lean_value_math.rs). -/
def SZ_VALUE : Nat := 16
def SZ_STRING : Nat := 24
def SZ_ARC : Nat := 8
def SZ_REL : Nat := 24
def ARC_HDR : Nat := 16

mutual
def heapSize : HV → Nat
  | .scalar => 0
  | .rel => SZ_REL
  | .str len _ _ => SZ_STRING + len
  | .list items _ _ => items.length * SZ_VALUE + heapSizeList items
  | .map entries _ => heapSizeMap entries
  | .vec len _ _ => len * 4
def heapSizeList : List HV → Nat
  | [] => 0
  | x :: xs => heapSize x + heapSizeList xs
def heapSizeMap : List (Nat × Nat × Nat × HV) → Nat
  | [] => 0
  | (klen, _, _, v) :: xs => (SZ_ARC + SZ_STRING + klen + SZ_VALUE + heapSize v) + heapSizeMap xs
end

mutual
def amortized : HV → Nat
  | .scalar => 0
  | .rel => SZ_REL
  | .str _ cap rc => (ARC_HDR + SZ_STRING + cap) / rc
  | .list items cap rc => (ARC_HDR + cap * SZ_VALUE + amortizedList items) / rc
  | .map entries rc => (ARC_HDR + amortizedMap entries) / rc
  | .vec _ cap rc => (ARC_HDR + cap * 4) / rc
def amortizedList : List HV → Nat
  | [] => 0
  | x :: xs => amortized x + amortizedList xs
def amortizedMap : List (Nat × Nat × Nat × HV) → Nat
  | [] => 0
  | (_, kcap, krc, v) :: xs => (SZ_ARC + (ARC_HDR + SZ_STRING + kcap) / krc + SZ_VALUE + amortized v) + amortizedMap xs
end

/- Unshared, well-formed (capacity ≥ length) heap views. -/
mutual
def Unshared : HV → Prop
  | .scalar | .rel => True
  | .str len cap rc => rc = 1 ∧ len ≤ cap
  | .list items cap rc => rc = 1 ∧ items.length ≤ cap ∧ UnsharedList items
  | .map entries rc => rc = 1 ∧ UnsharedMap entries
  | .vec len cap rc => rc = 1 ∧ len ≤ cap
def UnsharedList : List HV → Prop
  | [] => True
  | x :: xs => Unshared x ∧ UnsharedList xs
def UnsharedMap : List (Nat × Nat × Nat × HV) → Prop
  | [] => True
  | (klen, kcap, krc, v) :: xs => krc = 1 ∧ klen ≤ kcap ∧ Unshared v ∧ UnsharedMap xs
end

/- For values nobody shares, `amortized_heap_size ≥ heap_size`: the amortized figure
additionally charges the `Arc` headers and spare capacity. -/
theorem heap_le_amortized : ∀ (n : Nat) (h : HV), sizeOf h < n → Unshared h → heapSize h ≤ amortized h := by
  intro n
  induction n with
  | zero => intro h hs; omega
  | succ n ih =>
    intro h hs hu
    have hl : ∀ items : List HV, sizeOf items < n → UnsharedList items →
        heapSizeList items ≤ amortizedList items := by
      intro items
      induction items with
      | nil => intro _ _; simp [heapSizeList, amortizedList]
      | cons x xs ihx =>
        intro hsz hux
        simp only [UnsharedList] at hux
        simp only [heapSizeList, amortizedList]
        have := ih x (by simp at hsz; omega) hux.1
        have := ihx (by simp at hsz; omega) hux.2
        omega
    have hm : ∀ es : List (Nat × Nat × Nat × HV), sizeOf es < n → UnsharedMap es →
        heapSizeMap es ≤ amortizedMap es := by
      intro es
      induction es with
      | nil => intro _ _; simp [heapSizeMap, amortizedMap]
      | cons e es ihe =>
        obtain ⟨klen, kcap, krc, v⟩ := e
        intro hsz hue
        simp only [UnsharedMap] at hue
        obtain ⟨hk, hlen, hv, hrest⟩ := hue
        subst hk
        simp only [heapSizeMap, amortizedMap, Nat.div_one]
        have := ih v (by simp at hsz; omega) hv
        have := ihe (by simp at hsz; omega) hrest
        simp only [SZ_ARC, SZ_STRING, SZ_VALUE, ARC_HDR] at *
        omega
    cases h with
    | scalar => simp [heapSize, amortized]
    | rel => simp [heapSize, amortized]
    | str len cap rc =>
      simp only [Unshared] at hu; obtain ⟨rfl, hc⟩ := hu
      simp [heapSize, amortized, SZ_STRING, ARC_HDR]; omega
    | vec len cap rc =>
      simp only [Unshared] at hu; obtain ⟨rfl, hc⟩ := hu
      simp [heapSize, amortized, ARC_HDR]; omega
    | list items cap rc =>
      simp only [Unshared] at hu; obtain ⟨rfl, hc, hi⟩ := hu
      have := hl items (by simp at hs; omega) hi
      simp only [heapSize, amortized, Nat.div_one, SZ_VALUE, ARC_HDR] at *
      have : items.length * 16 ≤ cap * 16 := Nat.mul_le_mul_right _ hc
      omega
    | map entries rc =>
      simp only [Unshared] at hu; obtain ⟨rfl, he⟩ := hu
      have := hm entries (by simp at hs; omega) he
      simp only [heapSize, amortized, Nat.div_one, ARC_HDR] at *
      omega

/-- Sharing never over-counts: the `rc` holders of one allocation of `x` bytes are
charged `rc * (x / rc) ≤ x` in total (and lose less than `rc` bytes to rounding). -/
theorem shares_le (x rc : Nat) (h : 0 < rc) : rc * (x / rc) ≤ x ∧ x < rc * (x / rc) + rc := by
  constructor
  · exact Nat.mul_div_le x rc
  · have := Nat.div_add_mod x rc
    have := Nat.mod_lt x h
    rw [Nat.mul_comm] at *; omega

/-- A `Relationship` value owns no heap (it is an id, value.rs:200), yet both
estimates charge it 24 bytes — a fixed over-count (suspicion; no user-visible effect
known, relationships are never stored as attributes). -/
theorem rel_charged : heapSize .rel = 24 ∧ amortized .rel = 24 := ⟨rfl, rfl⟩

/-! ## `get_attr` / components -/

variable {F : Type}

def asciiLowerStr (s : List Char) : List Char := s.map lowerAscii

def eqIC (a : String) (b : String) : Bool := asciiLowerStr a.toList == asciiLowerStr b.toList

/-- `get_point_component` (value.rs:458): ASCII case-insensitive. -/
def pointComponent (widen : F → F) (la lo : F) (attr : String) : V F :=
  if eqIC attr "latitude" then .float (widen la)
  else if eqIC attr "longitude" then .float (widen lo)
  else .null

/-- `p.LATITUDE` is the latitude in Rust; C's `Point` accessor is case-sensitive and
returns null (live). -/
theorem point_component_case (widen : F → F) (la lo : F) :
    pointComponent widen la lo "LATITUDE" = .float (widen la) := by
  simp [pointComponent, eqIC, asciiLowerStr] <;> decide

/-- The part of `get_duration_component` (value.rs:608-634) before chrono is consulted:
unknown names error, `weeks` is always `0.0`. -/
def durationComponentPre (zero : F) (c : String) : Option (Except String (V F)) :=
  let known := ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]
  if !(known.any (fun k => eqIC c k)) then some (.error s!"unknown duration component {c}")
  else if eqIC c "weeks" then some (.ok (.float zero))
  else none

/-- `duration({weeks:2}).weeks` is `0.0` (the weeks are folded into days; C does the same,
live; openCypher expects 2). -/
theorem duration_weeks_zero (zero : F) : durationComponentPre zero "weeks" = some (.ok (.float zero)) := by
  simp [durationComponentPre, eqIC, asciiLowerStr] <;> decide

/-- `get_attr` dispatch (value.rs:439); temporal component extractors are parameters
(chrono-backed). -/
def getAttr (widen : F → F) (dt d t du : Int → String → Except String (V F)) (v : V F) (attr : String) :
    Except String (V F) :=
  match v with
  | .map m => .ok ((lookup m attr).getD .null)
  | .point la lo => .ok (pointComponent widen la lo attr)
  | .datetime ts => dt ts attr
  | .date ts => d ts attr
  | .time ts => t ts attr
  | .duration x => du x attr
  | .null => .ok .null
  | v => .error s!"Type mismatch: expected Map, Node, Edge, Datetime, Date, Time, Duration, Null, or Point but was {v.name}"

theorem getAttr_null (widen : F → F) (dt d t du : Int → String → Except String (V F)) (a : String) :
    getAttr widen dt d t du .null a = .ok .null := rfl

theorem getAttr_map_missing (widen : F → F) (dt d t du : Int → String → Except String (V F))
    (m : List (String × V F)) (a : String) (h : lookup m a = none) :
    getAttr widen dt d t du (.map m) a = .ok .null := by simp [getAttr, h]

/-! ## `get_numeric`, `partial_cmp` -/

def getNumeric (ofInt : Int → F) (zero : F) : V F → Option F
  | .int i => some (ofInt i)
  | .float f => some f
  | .null => some zero
  | _ => none   -- unreachable!() panics

/-- `validate_args_domain` (mod.rs:862) calls `get_numeric` on `args[1]` of
`percentile*` after `validate_args_type`: with the declared `Int | Float` slot (and the
explicit Null test) the panic arm is dead. -/
theorem getNumeric_after_validation (ofInt : Int → F) (zero : F) (v : V F)
    (h : Accepts v (.union [.int, .float, .null])) : (getNumeric ofInt zero v).isSome := by
  rw [← valueOfType_none_iff] at h
  cases v <;> simp_all [getNumeric, vot_union, unionLoop, valueOfType, tagMatch, Ty.isAny]

def partialCmp (fc : FloatCmp F) (a b : V F) : Option Ordering :=
  let (o, d) := cmpV fc a b
  if d = .comparedNull then none else some o

theorem partialCmp_null (fc : FloatCmp F) (a : V F) : partialCmp fc .null a = none := by
  cases a <;> simp [partialCmp, cmpV]

/-! ## `Point::validate` (value.rs:144) at class level -/

def validatePoint (lat lon : FC) (latInRange lonInRange : Bool) : Bool :=
  !(lat = .nan ∨ lat = .pinf ∨ lat = .ninf) && !(lon = .nan ∨ lon = .pinf ∨ lon = .ninf) &&
  latInRange && lonInRange

theorem validatePoint_rejects_nan (lon : FC) (a b : Bool) : validatePoint .nan lon a b = false := by
  simp [validatePoint]

/-! ## JSON of a Float (value.rs:1591) -/

/-- NaN/±∞ are emitted as JSON `null` (valid JSON; C emits `nan`, live). -/
def jsonFloat (display : FC → String) : FC → String
  | .nan | .pinf | .ninf => "null"
  | f => display f

theorem jsonFloat_nonfinite (d : FC → String) : jsonFloat d .nan = "null" ∧ jsonFloat d .pinf = "null" :=
  ⟨rfl, rfl⟩

/-! ## `ValuesDeduper` -/

/-- `check_and_insert_hash`: `!set.insert(h)`. -/
def dedupStep (seen : List Nat) (h : Nat) : Bool × List Nat :=
  if h ∈ seen then (true, seen) else (false, h :: seen)

theorem dedup_second_seen (seen : List Nat) (h : Nat) :
    (dedupStep (dedupStep seen h).2 h).1 = true := by
  unfold dedupStep; split <;> simp_all

theorem dedup_first_unseen (seen : List Nat) (h : Nat) (hn : h ∉ seen) : (dedupStep seen h).1 = false := by
  simp [dedupStep, hn]

/-! ## v19 value codec (`Encode<19>`/`Decode<19>`, value.rs:1899-2047) -/

def T_NULL : Nat := 2 ^ 15
def T_BOOL : Nat := 2 ^ 12
def T_INT64 : Nat := 2 ^ 13
def T_DOUBLE : Nat := 2 ^ 14
def T_STRING : Nat := 2 ^ 11
def T_INTERN : Nat := 2 ^ 19
def T_ARRAY : Nat := 2 ^ 3
def T_POINT : Nat := 2 ^ 17
def T_VECTOR_F32 : Nat := 2 ^ 18
def T_DATETIME : Nat := 2 ^ 5
def T_DATE : Nat := 2 ^ 7
def T_TIME : Nat := 2 ^ 8
def T_DURATION : Nat := 2 ^ 10

/-- Writer calls as tokens: `write_unsigned`, `write_signed`, `write_double`, `write_buffer`. -/
inductive Tok (F : Type) where
  | u (n : Nat)
  | s (i : Int)
  | d (f : F)
  | buf (bytes : List Nat)

/-- The byte/float layer: UTF-8 of strings, f32 ↔ f64 and f32 bit patterns. -/
structure CodecOps (F : Type) where
  utf8 : String → List Nat
  fromUtf8Lossy : List Nat → String
  lossy_utf8 : ∀ s, fromUtf8Lossy (utf8 s) = s
  widen : F → F            -- f64::from(f32)
  narrow : F → F           -- `as f32`
  narrow_widen : ∀ x, narrow (widen x) = x
  bits : F → Nat           -- f32::to_bits
  ofBits : Nat → F
  bits_lt : ∀ x, bits x < 2 ^ 32
  ofBits_bits : ∀ x, ofBits (bits x) = x

variable (co : CodecOps F)

def le4 (n : Nat) : List Nat := [n % 256, n / 256 % 256, n / 65536 % 256, n / 16777216 % 256]
def fromLe4 : List Nat → Nat
  | [a, b, c, d] => a + 256 * b + 65536 * c + 16777216 * d
  | _ => 0

theorem fromLe4_le4 (n : Nat) (h : n < 2 ^ 32) : fromLe4 (le4 n) = n := by
  simp [le4, fromLe4]; omega

def vecBytes (v : List F) : List Nat := le4 v.length ++ (v.map (fun x => le4 (co.bits x))).flatten

/- `write_buffer(bytes ++ [0])` for strings; the interned flag does not change the value. -/
mutual
def encode : V F → List (Tok F)
  | .bool b => [.u T_BOOL, .s (if b then 1 else 0)]
  | .int i => [.u T_INT64, .s i]
  | .float f => [.u T_DOUBLE, .d f]
  | .str s => [.u T_STRING, .buf (co.utf8 s ++ [0])]
  | .list l => [.u T_ARRAY, .u l.length] ++ encodeList l
  | .point la lo => [.u T_POINT, .d (co.widen la), .d (co.widen lo)]
  | .vec v => [.u T_VECTOR_F32, .buf (vecBytes co v)]
  | .datetime t => [.u T_DATETIME, .s t]
  | .date t => [.u T_DATE, .s t]
  | .time t => [.u T_TIME, .s t]
  | .duration t => [.u T_DURATION, .s t]
  | .null => [.u T_NULL]
  | .map _ | .node _ | .rel _ | .path _ => [.u T_NULL]   -- debug_assert!(false) in debug
def encodeList : List (V F) → List (Tok F)
  | [] => []
  | x :: xs => encode x ++ encodeList xs
end

def readVec (bytes : List Nat) : Option (List F) :=
  if bytes.length < 4 then none else
  let dim := fromLe4 (bytes.take 4)
  let body := bytes.drop 4
  if body.length < 4 * dim then none
  else some ((List.range dim).map (fun i => co.ofBits (fromLe4 ((body.drop (4 * i)).take 4))))

/- `Decode<19>` with a fuel bound (every token consumes fuel). -/
def decode : Nat → List (Tok F) → Option (V F × List (Tok F))
  | 0, _ => none
  | fuel + 1, .u tag :: rest =>
    if tag = T_NULL then some (.null, rest)
    else if tag = T_BOOL then match rest with
      | .s i :: r => some (.bool (i ≠ 0), r) | _ => none
    else if tag = T_INT64 then match rest with
      | .s i :: r => some (.int i, r) | _ => none
    else if tag = T_DOUBLE then match rest with
      | .d f :: r => some (.float f, r) | _ => none
    else if tag = T_STRING ∨ tag = T_INTERN + T_STRING then match rest with
      | .buf b :: r =>
        some (.str (co.fromUtf8Lossy (if b.getLast? = some 0 then b.dropLast else b)), r)
      | _ => none
    else if tag = T_ARRAY then match rest with
      | .u len :: r => (decodeN fuel len r).map (fun (l, r') => (.list l, r'))
      | _ => none
    else if tag = T_POINT then match rest with
      | .d a :: .d b :: r => some (.point (co.narrow a) (co.narrow b), r) | _ => none
    else if tag = T_VECTOR_F32 then match rest with
      | .buf b :: r => (readVec co b).map (fun v => (.vec v, r)) | _ => none
    else if tag = T_DATETIME then match rest with
      | .s i :: r => some (.datetime i, r) | _ => none
    else if tag = T_DATE then match rest with
      | .s i :: r => some (.date i, r) | _ => none
    else if tag = T_TIME then match rest with
      | .s i :: r => some (.time i, r) | _ => none
    else if tag = T_DURATION then match rest with
      | .s i :: r => some (.duration i, r) | _ => none
    else none
  | _ + 1, _ => none
where
  decodeN : Nat → Nat → List (Tok F) → Option (List (V F) × List (Tok F))
  | _, 0, r => some ([], r)
  | fuel, n + 1, r => match decode fuel r with
    | some (x, r') => (decodeN fuel n r').map (fun (xs, r'') => (x :: xs, r''))
    | none => none

/- Values that `Encode<19>` stores faithfully. -/
mutual
def Storable : V F → Prop
  | .map _ | .node _ | .rel _ | .path _ => False
  | .list l => StorableList l
  | .vec v => v.length < 2 ^ 32
  | _ => True
def StorableList : List (V F) → Prop
  | [] => True
  | x :: xs => Storable x ∧ StorableList xs
end

theorem utf8_zero_strip (b : List Nat) : ((b ++ [0]).getLast? = some 0) ∧ (b ++ [0]).dropLast = b := by
  simp

theorem le4_length (n : Nat) : (le4 n).length = 4 := rfl

theorem chunks_length (v : List F) : ((v.map (fun x => le4 (co.bits x))).flatten).length = 4 * v.length := by
  induction v with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.map_cons, List.flatten_cons, List.length_append, le4_length, List.length_cons] at *
    omega

theorem chunks_get (v : List F) (i : Nat) (hi : i < v.length) :
    (((v.map (fun x => le4 (co.bits x))).flatten).drop (4 * i)).take 4 = le4 (co.bits v[i]) := by
  induction v generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    cases i with
    | zero => simp [List.flatten_cons, le4]
    | succ i =>
      simp only [List.map_cons, List.flatten_cons]
      rw [show 4 * (i + 1) = (le4 (co.bits x)).length + 4 * i by simp [le4_length]; omega,
        ← List.drop_drop, List.drop_left]
      simpa using ih i (by simpa using hi)

theorem readVec_vecBytes (v : List F) (hl : v.length < 2 ^ 32) : readVec co (vecBytes co v) = some v := by
  unfold readVec vecBytes
  have hlen : (le4 v.length ++ (v.map (fun x => le4 (co.bits x))).flatten).length = 4 + 4 * v.length := by
    rw [List.length_append, le4_length, chunks_length]
  have ht : (le4 v.length ++ (v.map (fun x => le4 (co.bits x))).flatten).take 4 = le4 v.length := by
    simp [List.take_left' (le4_length _)]
  have hd : (le4 v.length ++ (v.map (fun x => le4 (co.bits x))).flatten).drop 4 =
      (v.map (fun x => le4 (co.bits x))).flatten := List.drop_left' (le4_length _)
  rw [if_neg (by omega), ht, hd, fromLe4_le4 _ hl, if_neg (by rw [chunks_length]; omega)]
  congr 1
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_range]
  rw [chunks_get co v i h2, fromLe4_le4 _ (co.bits_lt _), co.ofBits_bits]

/-- **Codec roundtrip**: every storable value decodes to itself, leaving the rest of
the stream untouched (fuel: any bound above the value's size). -/
theorem decode_encode : ∀ (fuel : Nat) (v : V F) (rest : List (Tok F)), sizeOf v < fuel → Storable v →
    decode co fuel (encode co v ++ rest) = some (v, rest) := by
  intro fuel
  induction fuel with
  | zero => intro v _ h; omega
  | succ f ih =>
    intro v rest hs hst
    have hN : ∀ (l : List (V F)) (r : List (Tok F)), sizeOf l < f → StorableList l →
        decode.decodeN co f l.length (encodeList co l ++ r) = some (l, r) := by
      intro l
      induction l with
      | nil => intro r _ _; simp [encodeList, decode.decodeN]
      | cons x xs ihl =>
        intro r hsz hsl
        simp only [StorableList] at hsl
        simp only [encodeList, List.length_cons, List.append_assoc, decode.decodeN]
        rw [ih x _ (by simp at hsz; omega) hsl.1]
        simp [ihl r (by simp at hsz; omega) hsl.2]
    cases v with
    | bool b => cases b <;> simp [encode, decode, T_NULL, T_BOOL]
    | int i => simp [encode, decode, T_NULL, T_BOOL, T_INT64]
    | float x => simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE]
    | str s => simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, co.lossy_utf8]
    | list l =>
      simp only [Storable] at hst
      simp only [encode, List.append_assoc, List.cons_append, List.nil_append]
      simp only [decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY]
      simp [hN l rest (by simp at hs; omega) hst]
    | point la lo =>
      simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY, T_POINT,
        co.narrow_widen]
    | vec v =>
      simp only [Storable] at hst
      simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY, T_POINT,
        T_VECTOR_F32, readVec_vecBytes co v hst]
    | datetime t => simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY,
        T_POINT, T_VECTOR_F32, T_DATETIME]
    | date t => simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY,
        T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE]
    | time t => simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY,
        T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME]
    | duration t => simp [encode, decode, T_NULL, T_BOOL, T_INT64, T_DOUBLE, T_STRING, T_INTERN, T_ARRAY,
        T_POINT, T_VECTOR_F32, T_DATETIME, T_DATE, T_TIME, T_DURATION]
    | null =>
      simp only [encode, List.cons_append, List.nil_append]
      unfold decode
      simp only [if_pos (show T_NULL = T_NULL from rfl), if_true]
    | map _ => simp [Storable] at hst
    | node _ => simp [Storable] at hst
    | rel _ => simp [Storable] at hst
    | path _ => simp [Storable] at hst

/-- Non-storable values are written as `NULL` (after a `debug_assert!`), so a Map
attribute would silently come back as null in release builds. -/
theorem encode_map_is_null (m : List (String × V F)) : encode co (.map m) = [.u T_NULL] := rfl

end ValueMath
