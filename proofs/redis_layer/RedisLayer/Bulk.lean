/-!
# `GRAPH.BULK` binary format (`src/commands/bulk_insert.rs`)

* `go` / `attach` — `read_property` (`bulk_insert.rs:94-169`): the explicit-stack reader.
  `go` is the outer `loop` decoding one type byte, `attach` the inner loop that pushes
  a finished value into the innermost open array and pops every array it completes.
  The input is the remaining byte list rather than `(data, idx)`; `idx` is
  `data.length - rest.length`.
* `cstr` — `read_cstring` (`:30-44`), `R8` — `read_u64_ne`/`read_i64_ne`/`read_f64_ne`
  (`:58-92`): every one bounds-checks before indexing.
* `records` — the record loop of `process_node_token` (`:384-408`).
* `parseCount` — `parse_count` (`:607-629`).

`i64::from_ne_bytes`/`to_ne_bytes` and `str::from_utf8` are Rust `core` functions; their
specification is taken as the hypothesis structure `Codec` (bytes ↔ value, 8 bytes wide;
a UTF-8 check that accepts every encoded string). No `axiom` is used.

**Totality.** `go` and `attach` are total Lean functions: Lean accepted their
termination proof (every `go` step consumes the type byte; every `attach` step pops a
frame), and every byte access is a pattern match on the list, so there is no index to
get wrong. The one arithmetic hazard in the Rust loop, `*remaining -= 1`, is modelled
as `Nat` subtraction; it never saturates because a frame is pushed only with a count
≥ 1 (`len = 0` attaches an empty list at once) and re-pushed only when `k - 1 ≠ 0`.
-/

namespace RedisLayer.Bulk

structure Codec where
  enc8 : Int → List Nat
  dec8 : List Nat → Int
  len8 : ∀ i, (enc8 i).length = 8
  dec_enc8 : ∀ i r, dec8 ((enc8 i ++ r).take 8) = i
  utf8ok : List Nat → Bool

/-- Bulk values (`BI_NULL` … `BI_ARRAY`). Doubles are carried as their `i64` bit
pattern — the reader never interprets them. -/
inductive BV where
  | null
  | bool (b : Bool)
  | dbl (bits : Int)
  | str (s : List Nat)
  | long (i : Int)
  | arr (xs : List BV)
  deriving Repr

inductive BErr where
  | eof | unterminated | badUtf8 | negLen | unknownType
  deriving DecidableEq, Repr

variable (C : Codec)

/-- `read_cstring`: bytes up to the first NUL, which is consumed. -/
def cstr : List Nat → Option (List Nat × List Nat)
  | [] => none
  | 0 :: r => some ([], r)
  | c :: r => (cstr r).map fun (s, r') => (c :: s, r')

theorem cstr_len (r s r' : List Nat) (h : cstr r = some (s, r')) : r'.length < r.length := by
  induction r generalizing s r' with
  | nil => simp [cstr] at h
  | cons c r ih =>
    cases c with
    | zero => simp [cstr] at h; obtain ⟨-, rfl⟩ := h; simp
    | succ n =>
      simp only [cstr, Option.map_eq_some_iff] at h
      obtain ⟨⟨a, b⟩, h1, h2⟩ := h
      simp at h2; obtain ⟨-, rfl⟩ := h2
      have := ih _ _ h1; simp; omega

abbrev Frame := List BV × Nat

mutual
def go : List Nat → List Frame → Except BErr (BV × List Nat)
  | [], _ => .error .eof
  | t :: r, st =>
    if t = 0 then attach .null r st
    else if t = 1 then
      match r with
      | [] => .error .eof
      | x :: r' => attach (.bool (x ≠ 0)) r' st
    else if t = 2 then
      if r.length < 8 then .error .eof else attach (.dbl (C.dec8 (r.take 8))) (r.drop 8) st
    else if t = 4 then
      if r.length < 8 then .error .eof else attach (.long (C.dec8 (r.take 8))) (r.drop 8) st
    else if t = 3 then
      match h : cstr r with
      | none => .error .unterminated
      | some (s, r') =>
        have := cstr_len r s r' h
        if C.utf8ok s then attach (.str s) r' st else .error .badUtf8
    else if t = 5 then
      if r.length < 8 then .error .eof
      else
        let len := C.dec8 (r.take 8)
        if len < 0 then .error .negLen
        else if len = 0 then attach (.arr []) (r.drop 8) st
        else go (r.drop 8) (([], len.toNat) :: st)
    else .error .unknownType
  termination_by r _ => (r.length, 0)
  decreasing_by all_goals (simp_wf; first | omega | (apply Prod.Lex.left; simp; omega))
def attach : BV → List Nat → List Frame → Except BErr (BV × List Nat)
  | v, r, [] => .ok (v, r)
  | v, r, (arr, k) :: st =>
    if k - 1 = 0 then attach (.arr (arr ++ [v])) r st
    else go r ((arr ++ [v], k - 1) :: st)
  termination_by _ r st => (r.length, st.length + 1)
  decreasing_by all_goals (simp_wf; first | (apply Prod.Lex.right; omega) | (apply Prod.Lex.right'; omega) | omega)
end

def readProperty (bytes : List Nat) : Except BErr (BV × List Nat) := go C bytes []

/-! ## Round trip -/

mutual
/-- What the Python bulk loader writes for a value. -/
def enc : BV → List Nat
  | .null => [0]
  | .bool x => [1, if x then 1 else 0]
  | .dbl i => 2 :: C.enc8 i
  | .str s => 3 :: s ++ [0]
  | .long i => 4 :: C.enc8 i
  | .arr xs => 5 :: C.enc8 xs.length ++ encs xs
def encs : List BV → List Nat
  | [] => []
  | x :: xs => enc x ++ encs xs
end

mutual
/-- A value the loader can write: strings hold no NUL and pass the UTF-8 check. -/
def wf : BV → Prop
  | .str s => (∀ c ∈ s, c ≠ 0) ∧ C.utf8ok s = true
  | .arr xs => wfs xs
  | _ => True
def wfs : List BV → Prop
  | [] => True
  | x :: xs => wf x ∧ wfs xs
end

theorem cstr_enc (s r : List Nat) (h : ∀ c ∈ s, c ≠ 0) : cstr (s ++ 0 :: r) = some (s, r) := by
  induction s with
  | nil => rfl
  | cons c s ih =>
    have hc : c ≠ 0 := h c (List.mem_cons_self ..)
    obtain ⟨n, rfl⟩ : ∃ n, c = n + 1 := ⟨c - 1, by omega⟩
    simp [cstr, ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]

theorem take8 (i : Int) (r : List Nat) : (C.enc8 i ++ r).take 8 = C.enc8 i := by
  rw [List.take_append_of_le_length (by simp [C.len8])]; exact List.take_of_length_le (by simp [C.len8])

theorem drop8 (i : Int) (r : List Nat) : (C.enc8 i ++ r).drop 8 = r := by
  rw [List.drop_append_of_le_length (by simp [C.len8]), List.drop_of_length_le (by simp [C.len8])]; rfl

theorem de8 (i : Int) : C.dec8 (C.enc8 i) = i := by
  have := C.dec_enc8 i []
  rwa [List.append_nil, List.take_of_length_le (by simp [C.len8])] at this

mutual
/-- **The reader reads back exactly what the loader wrote**, from any open-array
context. -/
theorem go_enc : ∀ (v : BV) (r : List Nat) (st : List Frame), wf C v →
    go C (enc C v ++ r) st = attach C v r st
  | .null, r, st, _ => by rw [enc, List.singleton_append, go.eq_def]; simp
  | .bool x, r, st, _ => by
    cases x <;> (simp only [enc, List.cons_append, List.nil_append]; rw [go.eq_def]; simp)
  | .dbl i, r, st, _ => by
    have := C.dec_enc8 i r
    simp only [enc, List.cons_append]; rw [go.eq_def]
    simp [C.len8, de8]
    all_goals (intro h; omega)
  | .long i, r, st, _ => by
    have := C.dec_enc8 i r
    simp only [enc, List.cons_append]; rw [go.eq_def]
    simp [C.len8, de8]
    all_goals (intro h; omega)
  | .str s, r, st, h => by
    obtain ⟨h1, h2⟩ := h
    simp only [enc, List.cons_append, List.append_assoc, List.singleton_append]
    rw [go.eq_def]
    simp only [show (3:Nat) ≠ 0 by omega, show (3:Nat) ≠ 1 by omega, show (3:Nat) ≠ 2 by omega,
      show (3:Nat) ≠ 4 by omega, ite_false, ite_true]
    split
    · rename_i hc; simp only [List.nil_append] at hc; rw [cstr_enc s r h1] at hc; simp at hc
    · rename_i s' r' hc
      simp only [List.nil_append] at hc; rw [cstr_enc s r h1] at hc; simp at hc; obtain ⟨rfl, rfl⟩ := hc
      simp [h2]
  | .arr xs, r, st, h => by
    have hd := C.dec_enc8 xs.length (encs C xs ++ r)
    simp only [enc, List.cons_append, List.append_assoc]
    rw [go.eq_def]
    simp only [show (5:Nat) ≠ 0 by omega, show (5:Nat) ≠ 1 by omega, show (5:Nat) ≠ 2 by omega,
      show (5:Nat) ≠ 4 by omega, show (5:Nat) ≠ 3 by omega, ite_false, ite_true,
      List.length_append, C.len8, take8, drop8, hd, de8]
    rw [if_neg (by omega)]
    cases xs with
    | nil => simp [encs, attach]
    | cons x xs =>
      rw [if_neg (by simp; omega), if_neg (by simp; omega)]
      simp only [Int.toNat_natCast]
      exact go_encs (x :: xs) r st [] (by simp) h
theorem go_encs : ∀ (xs : List BV) (r : List Nat) (st : List Frame) (acc : List BV),
    xs ≠ [] → wfs C xs →
    go C (encs C xs ++ r) ((acc, xs.length) :: st) = attach C (.arr (acc ++ xs)) r st
  | [], _, _, _, h, _ => absurd rfl h
  | [x], r, st, acc, _, h => by
    simp only [encs, List.append_nil]
    rw [go_enc x r _ h.1, attach]; simp
  | x :: y :: ys, r, st, acc, _, h => by
    simp only [encs, List.append_assoc]
    rw [go_enc x _ _ h.1, attach]
    simp only [List.length_cons]
    rw [if_neg (by omega)]
    have := go_encs (y :: ys) r st (acc ++ [x]) (by simp) h.2
    simp only [List.length_cons, encs, List.append_assoc] at this
    rw [show ys.length + 1 + 1 - 1 = ys.length + 1 by omega, this]
    simp
end

/-- Top level: `read_property` returns the value and leaves exactly the bytes after it. -/
theorem readProperty_enc (v : BV) (r : List Nat) (h : wf C v) :
    readProperty C (enc C v ++ r) = .ok (v, r) := by
  rw [readProperty, go_enc C v r [] h, attach]

/-- Every encoded value costs at least one byte (the bound `max_records` relies on). -/
theorem enc_len_pos (v : BV) : 1 ≤ (enc C v).length := by
  cases v <;> simp [enc]

/-! ## Records of a node token -/

/-- `process_node_token`'s loop: `while idx < data.len()`, read `P` properties per
record, drop `Null`s (`:398-403`). -/
def readN : Nat → List Nat → Except BErr (List BV × List Nat)
  | 0, r => .ok ([], r)
  | n + 1, r =>
    match readProperty C r with
    | .error e => .error e
    | .ok (v, r') =>
      match readN n r' with
      | .error e => .error e
      | .ok (vs, r'') => .ok (v :: vs, r'')

def records (P : Nat) (r : List Nat) (fuel : Nat) : Except BErr (List (List BV)) :=
  match fuel with
  | 0 => .ok []
  | fuel + 1 =>
    if r = [] then .ok []
    else match readN C P r with
      | .error e => .error e
      | .ok (vs, r') => (records P r' fuel).map (vs :: ·)

def encRec (vs : List BV) : List Nat := vs.flatMap (enc C)

theorem readN_enc (vs : List BV) (r : List Nat) (h : ∀ v ∈ vs, wf C v) :
    readN C vs.length (encRec C vs ++ r) = .ok (vs, r) := by
  induction vs with
  | nil => rfl
  | cons v vs ih =>
    simp only [encRec, List.flatMap_cons, List.append_assoc, List.length_cons, readN]
    rw [readProperty_enc C v _ (h v (List.mem_cons_self ..))]
    simp only
    rw [show vs.flatMap (enc C) = encRec C vs from rfl, ih (fun x hx => h x (List.mem_cons_of_mem _ hx))]

theorem encRec_ne_nil (vs : List BV) (hP : vs ≠ []) : encRec C vs ≠ [] := by
  cases vs with
  | nil => exact absurd rfl hP
  | cons v vs =>
    have := enc_len_pos C v
    simp [encRec]; intro h; rw [h] at this; simp at this

/-- **A node token imports exactly the records it encodes**: for a header with `P ≥ 1`
properties and a body that is the concatenation of `rs` (each with `P` values), the
record loop returns `rs`, in order — so with `node_count = rs.length` every record gets
exactly one reserved id and none is left over (the `node_id_cursor` check at `:385`). -/
theorem records_enc (P : Nat) (hP : 1 ≤ P) (rs : List (List BV))
    (hlen : ∀ vs ∈ rs, vs.length = P) (hwf : ∀ vs ∈ rs, ∀ v ∈ vs, wf C v) (fuel : Nat)
    (hfuel : rs.length ≤ fuel) :
    records C P (rs.flatMap (encRec C)) fuel = .ok rs := by
  induction rs generalizing fuel with
  | nil => cases fuel <;> simp [records]
  | cons vs rs ih =>
    obtain ⟨f, rfl⟩ : ∃ f, fuel = f + 1 := ⟨fuel - 1, by simp at hfuel; omega⟩
    have hv := hlen vs (List.mem_cons_self ..)
    have hne : vs ≠ [] := by intro h; rw [h] at hv; simp at hv; omega
    simp only [records, List.flatMap_cons]
    rw [if_neg (by simp [encRec_ne_nil C vs hne])]
    have hr := readN_enc C vs (rs.flatMap (encRec C)) (hwf vs (List.mem_cons_self ..))
    rw [hv] at hr
    rw [hr]
    simp only
    rw [ih (fun x hx => hlen x (List.mem_cons_of_mem _ hx))
      (fun x hx => hwf x (List.mem_cons_of_mem _ hx)) f (by simp at hfuel; omega)]
    rfl

theorem encRec_len (vs : List BV) : vs.length ≤ (encRec C vs).length := by
  induction vs with
  | nil => simp [encRec]
  | cons v vs ih =>
    have := enc_len_pos C v
    have e : encRec C (v :: vs) = enc C v ++ encRec C vs := by simp [encRec]
    rw [e, List.length_append, List.length_cons]; omega

/-- The `max_records` ceiling (`bulk_insert.rs:643-654`) never rejects an honest
payload: `P` properties cost at least `P` bytes per record. -/
theorem ceiling_sound (P : Nat) (rs : List (List BV)) (hlen : ∀ vs ∈ rs, vs.length = P) :
    rs.length * P ≤ (rs.flatMap (encRec C)).length := by
  induction rs with
  | nil => simp
  | cons vs rs ih =>
    have hv := hlen vs (List.mem_cons_self ..)
    have ih' := ih (fun x hx => hlen x (List.mem_cons_of_mem _ hx))
    have := encRec_len C vs
    rw [List.flatMap_cons, List.length_append, List.length_cons, Nat.succ_mul]; omega

/-! ## `parse_count` (`bulk_insert.rs:607-629`) -/

def parseCount (s : List Nat) : Option Nat :=
  if s = [] then none
  else if !(s.all fun c => 48 ≤ c && c ≤ 57) then none
  else if s.length > 1 ∧ s.head? = some 48 then none
  else
    let v := s.foldl (fun a d => a * 10 + (d - 48)) 0
    if v ≤ 2 ^ 63 - 1 then some v else none

/-- `parse_count` is total and refuses signs, padding and the empty string. -/
theorem parseCount_rejects :
    parseCount [] = none ∧ parseCount [43, 49] = none ∧ parseCount [45, 49] = none ∧
    parseCount [48, 49] = none ∧ parseCount [48] = some 0 ∧ parseCount [49, 48] = some 10 := by
  decide

/-! ## Attribute resolution: where the imported entity differs from the encoded one -/

/-- `get_or_create` (`graph/src/graph/attribute_store.rs:224-238`) over a dictionary
modelled as a list of names. -/
def getOrCreate (dict : List (List Nat)) (name : List Nat) : List (List Nat) × Nat :=
  match dict.idxOf? name with
  | some i => (dict, i)
  | none => if dict.length ≥ 65535 then (dict, 65535) else (dict ++ [name], dict.length)

/-- The attribute ids a header resolves to (`process_node_token`, `:362-365`). -/
def resolve : List (List Nat) → List (List Nat) → List Nat
  | _, [] => []
  | d, n :: ns => let (d', i) := getOrCreate d n; i :: resolve d' ns

/-- **Divergence (confirmed live).** A header naming `p` twice resolves both columns to
attribute id 0, and the record's span stores `(0, v₁), (0, v₂)`: `properties(n)` then
shows `{p: 1, p: 3, q: 2}`, `n.p` reads 3, `MATCH (n {p:3})` misses it and
`REMOVE n.p` drops both. C keeps one `p` (the last). -/
theorem duplicate_header_ids : resolve [] [[112], [113], [112]] = [0, 1, 0] := by decide

/-- `Slot.len` is a `u16` written as `n as u16` (`attribute_store.rs:598`, `:609-610`).
A record with 65536 non-null properties gets `len = 0`: every property is silently
dropped (confirmed live via `GRAPH.BULK`, `size(keys(n)) = 0`; 65537 keeps 1). -/
def slotLen (n : Nat) : Nat := n % 65536

theorem slot_len_truncates : slotLen 65536 = 0 ∧ slotLen 65537 = 1 ∧ slotLen 65535 = 65535 := by
  decide

end RedisLayer.Bulk
