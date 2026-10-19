/-!
# `GRAPH.BULK` binary format (`src/commands/bulk_insert.rs`)

* `go` / `attach` — `read_property` (`bulk_insert.rs:95-176`, depth cap since #3339 b080a44c1): the explicit-stack reader.
  `go` is the outer `loop` decoding one type byte, `attach` the inner loop that pushes
  a finished value into the innermost open array and pops every array it completes.
  The input is the remaining byte list rather than `(data, idx)`; `idx` is
  `data.length - rest.length`.
* `cstr` — `read_cstring` (`:31-45`), `readNe` — `read_u32_ne`/`read_u64_ne`/`read_i64_ne`/
  `read_f64_ne` (`:47-93`): every one bounds-checks before indexing (`readNe_ok`).
* `records` — the record loop of `process_node_token` (`:419-440`); each record's values
  are read by `PropColumns::read_record`, which consumes one value per header column.
* `PropColumns.new` / `PropColumns.readRecord` / `keepFirst` — `PropColumns::new`/`read_record`
  (`:178-235`, #3026 `fb3d516d0`): one entry per attribute, the first non-null in header order.
* `parseCount` — `parse_count` (`:647-666`).

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
  | tooDeep   -- `value_nesting_error()` (#3339, `bulk_insert.rs:132-134`)
  deriving DecidableEq, Repr

/-- `MAX_VALUE_NESTING` (`graph/src/runtime/value.rs:62`). -/
def MAX_VALUE_NESTING : Nat := 4096

variable (C : Codec)

/-- `read_u32_ne` (`:46`, `w = 4`), `read_u64_ne`/`read_i64_ne`/`read_f64_ne` (`:58`, `:70`,
`:82`, `w = 8`): bounds check `*idx + w > data.len()`, then `from_ne_bytes` of
`data[*idx..*idx + w]` (`dec`: the core decoder, any total function) and `*idx += w`. -/
def readNe {α : Type} (w : Nat) (dec : List Nat → α) (data : List Nat) (idx : Nat) :
    Except BErr (α × Nat) :=
  if idx + w > data.length then .error .eof else .ok (dec ((data.drop idx).take w), idx + w)

/-- Reads succeed exactly when `w` bytes remain. -/
theorem readNe_ok_iff {α : Type} (w : Nat) (dec : List Nat → α) (data : List Nat) (idx : Nat) :
    (∃ r, readNe w dec data idx = .ok r) ↔ idx + w ≤ data.length := by
  unfold readNe; split
  · simp; omega
  · simp; omega

/-- On success the slice handed to `from_ne_bytes` has exactly `w` bytes (so
`try_into().unwrap()` never panics), it is the bytes at `idx`, and the cursor moves by
`w` and stays within the data. -/
theorem readNe_ok {α : Type} (w : Nat) (dec : List Nat → α) (data : List Nat) (idx : Nat)
    (v : α) (idx' : Nat) (h : readNe w dec data idx = .ok (v, idx')) :
    ((data.drop idx).take w).length = w ∧ v = dec ((data.drop idx).take w) ∧
      idx' = idx + w ∧ idx' ≤ data.length := by
  unfold readNe at h; split at h
  · cases h
  · cases h; refine ⟨?_, rfl, rfl, by omega⟩
    simp; omega

/-- Reading an encoded 8-byte value back yields it (`i64`/`u64`/`f64` bit patterns). -/
theorem readNe_enc8 (pre r : List Nat) (i : Int) :
    readNe 8 C.dec8 (pre ++ C.enc8 i ++ r) pre.length = .ok (i, pre.length + 8) := by
  unfold readNe
  rw [if_neg (by simp [C.len8])]
  have : (List.drop pre.length (pre ++ C.enc8 i ++ r)) = C.enc8 i ++ r := by
    rw [List.append_assoc, List.drop_left]
  rw [this, C.dec_enc8]

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
      -- `stack.len() >= MAX_VALUE_NESTING` (#3339, `:132`), before the length is read
      if st.length ≥ MAX_VALUE_NESTING then .error .tooDeep
      else if r.length < 8 then .error .eof
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
/-- Array nesting depth: how many `BI_ARRAY` frames are open at the deepest point. -/
def depth : BV → Nat
  | .arr xs => depths xs + 1
  | _ => 0
def depths : List BV → Nat
  | [] => 0
  | x :: xs => max (depth x) (depths xs)
end

mutual
/-- **The reader reads back exactly what the loader wrote**, from any open-array
context, for any value whose arrays fit under `MAX_VALUE_NESTING` open frames (#3339). -/
theorem go_enc : ∀ (v : BV) (r : List Nat) (st : List Frame), wf C v →
    depth v + st.length ≤ MAX_VALUE_NESTING →
    go C (enc C v ++ r) st = attach C v r st
  | .null, r, st, _, _ => by rw [enc, List.singleton_append, go.eq_def]; simp
  | .bool x, r, st, _, _ => by
    cases x <;> (simp only [enc, List.cons_append, List.nil_append]; rw [go.eq_def]; simp)
  | .dbl i, r, st, _, _ => by
    have := C.dec_enc8 i r
    simp only [enc, List.cons_append]; rw [go.eq_def]
    simp [C.len8, de8]
    all_goals (intro h; omega)
  | .long i, r, st, _, _ => by
    have := C.dec_enc8 i r
    simp only [enc, List.cons_append]; rw [go.eq_def]
    simp [C.len8, de8]
    all_goals (intro h; omega)
  | .str s, r, st, h, _ => by
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
  | .arr xs, r, st, h, hd0 => by
    have hd := C.dec_enc8 xs.length (encs C xs ++ r)
    have hdep : depths xs + 1 + st.length ≤ MAX_VALUE_NESTING := by
      simp only [depth] at hd0; exact hd0
    simp only [enc, List.cons_append, List.append_assoc]
    rw [go.eq_def]
    simp only [show (5:Nat) ≠ 0 by omega, show (5:Nat) ≠ 1 by omega, show (5:Nat) ≠ 2 by omega,
      show (5:Nat) ≠ 4 by omega, show (5:Nat) ≠ 3 by omega, ite_false, ite_true]
    rw [if_neg (by omega)]
    simp only [List.length_append, C.len8, take8, drop8, hd, de8]
    rw [if_neg (by omega)]
    cases xs with
    | nil => simp [encs, attach]
    | cons x xs =>
      rw [if_neg (by simp; omega), if_neg (by simp; omega)]
      simp only [Int.toNat_natCast]
      exact go_encs (x :: xs) r st [] (by simp) h (by omega)
theorem go_encs : ∀ (xs : List BV) (r : List Nat) (st : List Frame) (acc : List BV),
    xs ≠ [] → wfs C xs → depths xs + st.length + 1 ≤ MAX_VALUE_NESTING →
    go C (encs C xs ++ r) ((acc, xs.length) :: st) = attach C (.arr (acc ++ xs)) r st
  | [], _, _, _, h, _, _ => absurd rfl h
  | [x], r, st, acc, _, h, hd => by
    simp only [encs, List.append_nil]
    have : depth x + ((acc, [x].length) :: st).length ≤ MAX_VALUE_NESTING := by
      simp only [depths] at hd; simp; omega
    rw [go_enc x r _ h.1 this, attach]; simp
  | x :: y :: ys, r, st, acc, _, h, hd => by
    simp only [encs, List.append_assoc]
    have hx : depth x + ((acc, (x :: y :: ys).length) :: st).length ≤ MAX_VALUE_NESTING := by
      simp only [depths] at hd; simp; omega
    rw [go_enc x _ _ h.1 hx, attach]
    simp only [List.length_cons]
    rw [if_neg (by omega)]
    have hys : depths (y :: ys) + st.length + 1 ≤ MAX_VALUE_NESTING := by
      simp only [depths] at hd ⊢; omega
    have := go_encs (y :: ys) r st (acc ++ [x]) (by simp) h.2 hys
    simp only [List.length_cons, encs, List.append_assoc] at this
    rw [show ys.length + 1 + 1 - 1 = ys.length + 1 by omega, this]
    simp
end

/-- Top level: `read_property` returns the value and leaves exactly the bytes after it. -/
theorem readProperty_enc (v : BV) (r : List Nat) (h : wf C v) (hd : depth v ≤ MAX_VALUE_NESTING) :
    readProperty C (enc C v ++ r) = .ok (v, r) := by
  rw [readProperty, go_enc C v r [] h (by simpa using hd), attach]

/-- **Depth cap** (#3339): an array opened with `MAX_VALUE_NESTING` frames already open is
refused before its length is read, so no value deeper than the cap is ever built. -/
theorem deep_refused (r : List Nat) (st : List Frame) (h : MAX_VALUE_NESTING ≤ st.length) :
    go C (5 :: r) st = .error .tooDeep := by
  rw [go.eq_def]; simp [h]

/-- Historical (before #3339, 4a4f2a72f): no cap — any depth was accepted, which the
recursive drop/save/reply paths then walked. Witness: the deep-array prefix is no longer
read as an open frame once the stack is full. -/
theorem pre3339_uncapped_depth_now_refused :
    go C (5 :: C.enc8 1) (List.replicate MAX_VALUE_NESTING ([], 1)) = .error .tooDeep :=
  deep_refused C _ _ (by simp)

/-- Every encoded value costs at least one byte (the bound `max_records` relies on). -/
theorem enc_len_pos (v : BV) : 1 ≤ (enc C v).length := by
  cases v <;> simp [enc]

/-! ## Records of a node token -/

/-- `process_node_token`'s loop: `while idx < data.len()`, read `P` properties per
record (`read_record`, `:433-438`; which values it keeps is `keepFirst`, below). -/
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

theorem readN_enc (vs : List BV) (r : List Nat) (h : ∀ v ∈ vs, wf C v)
    (hd : ∀ v ∈ vs, depth v ≤ MAX_VALUE_NESTING) :
    readN C vs.length (encRec C vs ++ r) = .ok (vs, r) := by
  induction vs with
  | nil => rfl
  | cons v vs ih =>
    simp only [encRec, List.flatMap_cons, List.append_assoc, List.length_cons, readN]
    rw [readProperty_enc C v _ (h v (List.mem_cons_self ..)) (hd v (List.mem_cons_self ..))]
    simp only
    rw [show vs.flatMap (enc C) = encRec C vs from rfl, ih (fun x hx => h x (List.mem_cons_of_mem _ hx))
      (fun x hx => hd x (List.mem_cons_of_mem _ hx))]

theorem encRec_ne_nil (vs : List BV) (hP : vs ≠ []) : encRec C vs ≠ [] := by
  cases vs with
  | nil => exact absurd rfl hP
  | cons v vs =>
    have := enc_len_pos C v
    simp [encRec]; intro h; rw [h] at this; simp at this

/-- **A node token imports exactly the records it encodes**: for a header with `P ≥ 1`
properties and a body that is the concatenation of `rs` (each with `P` values), the
record loop returns `rs`, in order — so with `node_count = rs.length` every record gets
exactly one reserved id and none is left over (the `node_id_cursor` check at `:420`). -/
theorem records_enc (P : Nat) (hP : 1 ≤ P) (rs : List (List BV))
    (hlen : ∀ vs ∈ rs, vs.length = P) (hwf : ∀ vs ∈ rs, ∀ v ∈ vs, wf C v)
    (hdep : ∀ vs ∈ rs, ∀ v ∈ vs, depth v ≤ MAX_VALUE_NESTING) (fuel : Nat)
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
      (hdep vs (List.mem_cons_self ..))
    rw [hv] at hr
    rw [hr]
    simp only
    rw [ih (fun x hx => hlen x (List.mem_cons_of_mem _ hx))
      (fun x hx => hwf x (List.mem_cons_of_mem _ hx)) (fun x hx => hdep x (List.mem_cons_of_mem _ hx))
      f (by simp at hfuel; omega)]
    rfl

theorem encRec_len (vs : List BV) : vs.length ≤ (encRec C vs).length := by
  induction vs with
  | nil => simp [encRec]
  | cons v vs ih =>
    have := enc_len_pos C v
    have e : encRec C (v :: vs) = enc C v ++ encRec C vs := by simp [encRec]
    rw [e, List.length_append, List.length_cons]; omega

/-- The `max_records` ceiling (`bulk_insert.rs:683-695`) never rejects an honest
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

/-! ## `parse_count` (`bulk_insert.rs:647-669`) -/

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

/-- The attribute ids a header resolves to (`process_node_token`, `:406-410`). -/
def resolve : List (List Nat) → List (List Nat) → List Nat
  | _, [] => []
  | d, n :: ns => let (d', i) := getOrCreate d n; i :: resolve d' ns

/-- A header naming `p` twice resolves both columns to attribute id 0. Before #3063 the
record's span then stored `(0, v₁), (0, v₂)` (**divergence, confirmed live**:
`properties(n)` showed `{p: 1, p: 3, q: 2}`, `n.p` read 3; C keeps one `p`, `n.p` = 1).
Since #3063 (`0f45a8685`) `import_attrs_resolved` keeps the first column only (proofs/
pending_commit `Content.normResolved_get`, `resolved_duplicate_fixed`), and since #3026
(`fb3d516d0`) the record itself already holds one entry per attribute
(`readRecord_enc`, `keepFirst_nodup`; history `pre3026_two_entries_for_one_attr`). -/
theorem duplicate_header_ids : resolve [] [[112], [113], [112]] = [0, 1, 0] := by decide

/-- `Slot.len` is a `u16` written as `n as u16` (`attribute_store.rs:598`, `:609-610`).
Before #3063 a record with 65536 non-null properties got `len = 0`: every property was
silently dropped (confirmed live via `GRAPH.BULK`, `size(keys(n)) = 0`; 65537 kept 1).
The cast is unchanged, but since #3063 (`0f45a8685`) the span `GRAPH.BULK` hands to
`set_span` is deduplicated and free of `ATTRIBUTE_ID_NONE`, hence ≤ 65,535 entries
(proofs/pending_commit `Content.normResolved_fits`). -/
def slotLen (n : Nat) : Nat := n % 65536

theorem slot_len_truncates : slotLen 65536 = 0 ∧ slotLen 65537 = 1 ∧ slotLen 65535 = 65535 := by
  decide

/-! ## Header columns: `PropColumns` (#3026, `fb3d516d0`, closes #3024)

`PropColumns::new` (`bulk_insert.rs:197-210`) and `PropColumns::read_record` (`:218-234`),
used by `process_node_token` (`:411`, `:433-438`) and `process_edge_token` (`:495`, `:517-522`).
Attribute ids are `Nat` (Rust `u16`). The `HashMap<u16, usize>` from attribute to slot is
modelled by its key list in insertion order: `entry(a).or_insert(slots.len())` gives a new key
the next index, so a key's slot is its index in that list (`slotOf`). -/

/-- `matches!(val, Value::Null)`. -/
def BV.isNull : BV → Bool
  | .null => true
  | _ => false

/-- One step of `new`'s `map`: `next = slots.len()`, `*slots.entry(a).or_insert(next)`. -/
def slotOf (ks : List Nat) (a : Nat) : List Nat × Nat :=
  if a ∈ ks then (ks, ks.idxOf a) else (ks ++ [a], ks.length)

/-- The `map … collect` over `attr_ids`, threading the slot map. -/
def buildCols : List Nat → List Nat → List Nat × List (Nat × Nat)
  | ks, [] => (ks, [])
  | ks, a :: as =>
    let (ks', s) := slotOf ks a
    let (kf, cs) := buildCols ks' as
    (kf, (a, s) :: cs)

structure PropColumns where
  cols : List (Nat × Nat)
  filled : List Bool

/-- `PropColumns::new`: `filled = vec![false; slots.len()]`. -/
def PropColumns.new (ids : List Nat) : PropColumns :=
  let (ks, cols) := buildCols [] ids
  ⟨cols, List.replicate ks.length false⟩

/-- `PropColumns::is_empty`: `self.cols.is_empty()`. -/
def PropColumns.isEmpty (pc : PropColumns) : Bool := pc.cols.isEmpty

/-- The `for &(attr_id, slot) in &self.cols` loop: read one value (`?` propagates the error),
skip it when it is null or `filled[slot]`, else set `filled[slot]` and push. -/
def recLoop : List (Nat × Nat) → List Bool → List (Nat × BV) → List Nat →
    Except BErr (List (Nat × BV) × List Bool × List Nat)
  | [], filled, acc, r => .ok (acc, filled, r)
  | (a, s) :: cs, filled, acc, r =>
    match readProperty C r with
    | .error e => .error e
    | .ok (v, r') =>
      if v.isNull || filled.getD s false then recLoop cs filled acc r'
      else recLoop cs (filled.set s true) (acc ++ [(a, v)]) r'

/-- `PropColumns::read_record`: `self.filled.fill(false)`, then the loop. Returns the entries,
the updated columns (`&mut self`) and the remaining bytes (`*idx`). -/
def PropColumns.readRecord (pc : PropColumns) (r : List Nat) :
    Except BErr (List (Nat × BV) × PropColumns × List Nat) :=
  match recLoop C pc.cols (pc.filled.map fun _ => false) [] r with
  | .error e => .error e
  | .ok (es, filled, r') => .ok (es, ⟨pc.cols, filled⟩, r')

/-- **Specification**: walk `(attribute, value)` pairs in header order, dropping nulls and any
attribute already kept in this record (`seen`). -/
def keepFirst (seen : List Nat) : List (Nat × BV) → List (Nat × BV)
  | [] => []
  | (a, v) :: xs =>
    if v.isNull = true ∨ a ∈ seen then keepFirst seen xs else (a, v) :: keepFirst (a :: seen) xs

/-- The per-column read before #3026: one entry per non-null column. -/
def plainRead (xs : List (Nat × BV)) : List (Nat × BV) := xs.filter fun p => !p.2.isNull

/-! ### `new` -/

theorem buildCols_spec (ks ids : List Nat) :
    ∃ t, (buildCols ks ids).1 = ks ++ t ∧ (∀ a ∈ ids, a ∈ (buildCols ks ids).1) ∧
      (buildCols ks ids).2 = ids.map fun a => (a, (buildCols ks ids).1.idxOf a) := by
  induction ids generalizing ks with
  | nil => exact ⟨[], by simp [buildCols]⟩
  | cons a as ih =>
    by_cases ha : a ∈ ks
    · obtain ⟨t, h1, h2, h3⟩ := ih ks
      have e : buildCols ks (a :: as) = ((buildCols ks as).1, (a, ks.idxOf a) :: (buildCols ks as).2) := by
        simp [buildCols, slotOf, ha]
      refine ⟨t, by rw [e]; exact h1, ?_, ?_⟩
      · intro b hb; rw [e]; simp at hb
        rcases hb with rfl | hb
        · rw [h1]; simp [ha]
        · exact h2 b hb
      · rw [e, h3, h1]; simp [List.idxOf_append, ha]
    · obtain ⟨t, h1, h2, h3⟩ := ih (ks ++ [a])
      have e : buildCols ks (a :: as) =
          ((buildCols (ks ++ [a]) as).1, (a, ks.length) :: (buildCols (ks ++ [a]) as).2) := by
        simp [buildCols, slotOf, ha]
      refine ⟨[a] ++ t, by rw [e, h1]; simp, ?_, ?_⟩
      · intro b hb; rw [e]; simp at hb
        rcases hb with rfl | hb
        · rw [h1]; simp
        · exact h2 b hb
      · rw [e, h3, h1]; simp [List.idxOf_append, ha]

/-- The slot map's keys: the distinct attributes in first-occurrence order. -/
def keys (ids : List Nat) : List Nat := (buildCols [] ids).1

/-- **`new`**: column `i` holds attribute `ids[i]` and that attribute's slot, the slot of an
attribute is its index among the distinct attributes, and `filled` has one entry per slot. -/
theorem new_spec (ids : List Nat) :
    (PropColumns.new ids).cols = ids.map (fun a => (a, (keys ids).idxOf a)) ∧
    (PropColumns.new ids).filled = List.replicate (keys ids).length false ∧
    ∀ a ∈ ids, a ∈ keys ids := by
  obtain ⟨_, _, h2, h3⟩ := buildCols_spec [] ids
  exact ⟨by simp only [PropColumns.new]; exact h3, rfl, h2⟩

/-- `is_empty` holds exactly when the header names no property (the record loop then reads
nothing). -/
theorem new_isEmpty (ids : List Nat) : (PropColumns.new ids).isEmpty = ids.isEmpty := by
  rw [PropColumns.isEmpty, (new_spec ids).1]; cases ids <;> rfl

/-- Two columns share a slot exactly when they name the same attribute (`filled[slot]` is
per attribute, and every slot index is in bounds of `filled`). -/
theorem slot_inj (K : List Nat) (a b : Nat) (ha : a ∈ K) (hb : b ∈ K) :
    K.idxOf a = K.idxOf b ↔ a = b := by
  constructor
  · intro h
    have e1 := List.getElem_idxOf (List.idxOf_lt_length_of_mem ha)
    have e2 := List.getElem_idxOf (List.idxOf_lt_length_of_mem hb)
    simp only [h] at e1; rw [← e1, e2]
  · rintro rfl; rfl

/-! ### `read_record` -/

theorem getD_set_true (f : List Bool) (i j : Nat) (hi : i < f.length) :
    (f.set i true).getD j false = (i = j || f.getD j false) := by
  simp only [List.getD_eq_getElem?_getD, List.getElem?_set]
  by_cases h : i = j
  · subst h; simp [hi]
  · simp [h]

/-- The loop computes `keepFirst`: invariant — for every attribute `a` of the token,
`filled[slot a]` says whether `a` is in `seen`. -/
theorem recLoop_keepFirst (K : List Nat) :
    ∀ (cs : List (Nat × Nat)) (vs : List BV) (filled : List Bool) (seen : List Nat)
      (acc : List (Nat × BV)) (r : List Nat),
    (∀ p ∈ cs, p.1 ∈ K ∧ p.2 = K.idxOf p.1) → filled.length = K.length →
    (∀ a ∈ K, filled.getD (K.idxOf a) false = decide (a ∈ seen)) →
    vs.length = cs.length → (∀ v ∈ vs, wf C v) → (∀ v ∈ vs, depth v ≤ MAX_VALUE_NESTING) →
    ∃ f', f'.length = filled.length ∧
      recLoop C cs filled acc (encRec C vs ++ r) =
        .ok (acc ++ keepFirst seen ((cs.map Prod.fst).zip vs), f', r)
  | [], vs, filled, seen, acc, r, _, _, _, hl, _, _ => by
    cases vs with
    | nil => exact ⟨filled, rfl, by simp [recLoop, encRec, keepFirst]⟩
    | cons _ _ => simp at hl
  | (a, s) :: cs, [], _, _, _, _, _, _, _, hl, _, _ => by simp at hl
  | (a, s) :: cs, v :: vs, filled, seen, acc, r, hcs, hfl, hinv, hl, hw, hd => by
    have ⟨haK, hs⟩ := hcs (a, s) (List.mem_cons_self ..)
    simp only at haK hs
    have hcs' : ∀ p ∈ cs, p.1 ∈ K ∧ p.2 = K.idxOf p.1 := fun p hp => hcs p (List.mem_cons_of_mem _ hp)
    have hl' : vs.length = cs.length := by simpa using hl
    have hw' := fun x hx => hw x (List.mem_cons_of_mem _ hx)
    have hd' := fun x hx => hd x (List.mem_cons_of_mem _ hx)
    have henc : encRec C (v :: vs) ++ r = enc C v ++ (encRec C vs ++ r) := by
      simp [encRec]
    rw [henc, recLoop, readProperty_enc C v _ (hw v (List.mem_cons_self ..)) (hd v (List.mem_cons_self ..))]
    simp only [List.map_cons, List.zip_cons_cons, keepFirst]
    have hget : filled.getD s false = decide (a ∈ seen) := by rw [hs]; exact hinv a haK
    by_cases hskip : v.isNull = true ∨ a ∈ seen
    · have hc : (v.isNull || filled.getD s false) = true := by
        rw [hget]; rcases hskip with h | h <;> simp [h]
      rw [if_pos hc, if_pos hskip]
      exact recLoop_keepFirst K cs vs filled seen acc r hcs' hfl hinv hl' hw' hd'
    · have hc : ¬ (v.isNull || filled.getD s false) = true := by
        rw [hget]; simp only [not_or] at hskip; simp [hskip.1, hskip.2]
      rw [if_neg hc, if_neg hskip]
      have hlt : s < filled.length := by rw [hfl, hs]; exact List.idxOf_lt_length_of_mem haK
      have hinv' : ∀ b ∈ K, (filled.set s true).getD (K.idxOf b) false = decide (b ∈ a :: seen) := by
        intro b hb
        rw [getD_set_true _ _ _ hlt, hinv b hb, hs]
        by_cases hab : a = b
        · subst hab; simp
        · have hne : K.idxOf a ≠ K.idxOf b := fun e => hab ((slot_inj K a b haK hb).1 e)
          simp [hne, Ne.symm hab]
      obtain ⟨f', hf'l, hf'⟩ := recLoop_keepFirst K cs vs (filled.set s true) (a :: seen)
        (acc ++ [(a, v)]) r hcs' (by simp [hfl]) hinv' hl' hw' hd'
      refine ⟨f', by simpa using hf'l, ?_⟩
      rw [hf']; simp

/-- **`read_record` reads the specification**, whatever `filled` the previous record left
(`fill(false)` resets it): on the encoding of one record whose column values are `vs`, it
returns `keepFirst [] (ids.zip vs)`, leaves exactly the bytes after the record, and keeps the
columns and the length of `filled` (so the next record starts from the same state). -/
theorem readRecord_enc (ids : List Nat) (filled : List Bool)
    (hf : filled.length = (keys ids).length) (vs : List BV) (r : List Nat)
    (hl : vs.length = ids.length) (hw : ∀ v ∈ vs, wf C v)
    (hd : ∀ v ∈ vs, depth v ≤ MAX_VALUE_NESTING) :
    ∃ pc', pc'.cols = (PropColumns.new ids).cols ∧ pc'.filled.length = (keys ids).length ∧
      PropColumns.readRecord C ⟨(PropColumns.new ids).cols, filled⟩ (encRec C vs ++ r) =
        .ok (keepFirst [] (ids.zip vs), pc', r) := by
  obtain ⟨hcols, -, hmem⟩ := new_spec ids
  have hcs : ∀ p ∈ (PropColumns.new ids).cols, p.1 ∈ keys ids ∧ p.2 = (keys ids).idxOf p.1 := by
    intro p hp; rw [hcols] at hp; simp at hp
    obtain ⟨a, ha, rfl⟩ := hp; exact ⟨hmem a ha, rfl⟩
  have hfst : (PropColumns.new ids).cols.map Prod.fst = ids := by
    rw [hcols]; simp [Function.comp_def]
  obtain ⟨f', hf'l, h⟩ := recLoop_keepFirst C (keys ids) (PropColumns.new ids).cols vs
    (filled.map fun _ => false) [] [] r hcs (by simp [hf])
    (by
      intro a _; rw [List.getD_eq_getElem?_getD, List.getElem?_map]
      cases filled[(keys ids).idxOf a]? <;> rfl)
    (by rw [hcols, List.length_map, hl]) hw hd
  refine ⟨⟨(PropColumns.new ids).cols, f'⟩, rfl, by simp [hf'l, hf], ?_⟩
  simp only [PropColumns.readRecord, h, hfst, List.nil_append]

/-! ### Properties of the specification -/

theorem keepFirst_fresh (seen : List Nat) (xs : List (Nat × BV)) :
    ∀ a ∈ (keepFirst seen xs).map Prod.fst, a ∉ seen := by
  induction xs generalizing seen with
  | nil => simp [keepFirst]
  | cons p xs ih =>
    obtain ⟨b, v⟩ := p
    simp only [keepFirst]
    split
    · exact ih seen
    · rename_i h
      intro a ha; simp only [List.map_cons, List.mem_cons] at ha
      rcases ha with rfl | ha
      · exact fun hm => h (Or.inr hm)
      · exact fun hm => ih (b :: seen) a ha (List.mem_cons_of_mem _ hm)

/-- **Each attribute appears at most once per record.** -/
theorem keepFirst_nodup (seen : List Nat) (xs : List (Nat × BV)) :
    ((keepFirst seen xs).map Prod.fst).Nodup := by
  induction xs generalizing seen with
  | nil => simp [keepFirst]
  | cons p xs ih =>
    obtain ⟨b, v⟩ := p
    simp only [keepFirst]
    split
    · exact ih seen
    · simp only [List.map_cons, List.nodup_cons]
      exact ⟨fun hm => keepFirst_fresh (b :: seen) xs b hm (List.mem_cons_self ..), ih _⟩

/-- **Header order is kept**: the entries are a subsequence of the columns. -/
theorem keepFirst_sublist (seen : List Nat) (xs : List (Nat × BV)) :
    (keepFirst seen xs).Sublist xs := by
  induction xs generalizing seen with
  | nil => simp [keepFirst]
  | cons p xs ih =>
    obtain ⟨b, v⟩ := p
    simp only [keepFirst]
    split
    · exact (ih seen).cons _
    · exact (ih _).cons₂ _

/-- **The first non-null value wins**: the entry kept for `a` is the first column naming `a`
whose value is not null (and there is none exactly when every such column is null). -/
theorem keepFirst_find (seen : List Nat) (xs : List (Nat × BV)) (a : Nat) (ha : a ∉ seen) :
    (keepFirst seen xs).find? (fun p => p.1 == a) =
      xs.find? (fun p => p.1 == a && !p.2.isNull) := by
  induction xs generalizing seen with
  | nil => simp [keepFirst]
  | cons p xs ih =>
    obtain ⟨b, v⟩ := p
    simp only [keepFirst]
    split
    · rename_i h
      rw [ih seen ha, List.find?_cons]
      have : (b == a && !v.isNull) = false := by
        rcases h with h | h
        · simp [h]
        · have : b ≠ a := fun e => ha (e ▸ h)
          simp [this]
      rw [this]
    · rename_i h
      simp only [not_or] at h
      rw [List.find?_cons, List.find?_cons]
      by_cases hba : b = a
      · subst hba; simp [h.1]
      · have hf : (b == a) = false := by simp [hba]
        simp only [hf, Bool.false_and]
        exact ih (b :: seen) (by simp [Ne.symm hba, ha])

/-- **Without repeated names nothing changes**: the result is the plain per-column read. -/
theorem keepFirst_nodup_eq (seen : List Nat) (xs : List (Nat × BV))
    (hnd : (xs.map Prod.fst).Nodup) (hs : ∀ p ∈ xs, p.1 ∉ seen) :
    keepFirst seen xs = plainRead xs := by
  induction xs generalizing seen with
  | nil => rfl
  | cons p xs ih =>
    obtain ⟨b, v⟩ := p
    simp only [List.map_cons, List.nodup_cons] at hnd
    have hb : b ∉ seen := hs (b, v) (List.mem_cons_self ..)
    simp only [keepFirst, plainRead, List.filter_cons]
    by_cases hv : v.isNull = true
    · rw [if_pos (Or.inl hv), if_neg (by simp [hv])]
      exact ih seen hnd.2 (fun p hp => hs p (List.mem_cons_of_mem _ hp))
    · have hv' : v.isNull = false := by simpa using hv
      rw [if_neg (by simp [hv', hb]), if_pos (by simp [hv'])]
      congr 1
      refine ih (b :: seen) hnd.2 ?_
      intro p hp hm
      simp only [List.mem_cons] at hm
      rcases hm with h | h
      · exact hnd.1 (h ▸ List.mem_map_of_mem hp)
      · exact hs p (List.mem_cons_of_mem _ hp) h

theorem keepFirst_plain_of_nodup (ids : List Nat) (vs : List BV) (hl : ids.length ≤ vs.length)
    (hnd : ids.Nodup) : keepFirst [] (ids.zip vs) = plainRead (ids.zip vs) :=
  keepFirst_nodup_eq [] _ (by rw [List.map_fst_zip hl]; exact hnd) (by simp)

/-- Dropping the nulls first changes nothing. With `keepFirst_nodup_eq`'s ingredients this
says `read_record` equals what #3063's `import_attrs_resolved` made of the old per-column
read (it kept the first entry of each attribute), so #3026 changes no stored entity. -/
theorem keepFirst_plainRead (seen : List Nat) (xs : List (Nat × BV)) :
    keepFirst seen (plainRead xs) = keepFirst seen xs := by
  induction xs generalizing seen with
  | nil => rfl
  | cons p xs ih =>
    obtain ⟨b, v⟩ := p
    simp only [plainRead] at ih ⊢
    by_cases hv : v.isNull = true
    · simp [List.filter_cons, hv, keepFirst, ih]
    · simp only [List.filter_cons, hv, Bool.not_false, if_true, keepFirst]
      split <;> simp [ih]

/-- **End to end**: a record of a token whose header names no attribute twice reads exactly
the per-column non-null values. -/
theorem readRecord_plain (ids : List Nat) (hnd : ids.Nodup) (filled : List Bool)
    (hf : filled.length = (keys ids).length) (vs : List BV) (r : List Nat)
    (hl : vs.length = ids.length) (hw : ∀ v ∈ vs, wf C v)
    (hd : ∀ v ∈ vs, depth v ≤ MAX_VALUE_NESTING) :
    ∃ pc', PropColumns.readRecord C ⟨(PropColumns.new ids).cols, filled⟩ (encRec C vs ++ r) =
        .ok (plainRead (ids.zip vs), pc', r) := by
  obtain ⟨pc', -, -, h⟩ := readRecord_enc C ids filled hf vs r hl hw hd
  exact ⟨pc', by rw [h, keepFirst_plain_of_nodup ids vs (by omega) hnd]⟩

/-- Header `p, q, p` (ids `0, 1, 0`), record `1, 2, 3`: one `p`, holding the first value. -/
theorem repeated_header_first_wins :
    keepFirst [] ([0, 1, 0].zip [.long 1, .long 2, .long 3]) = [(0, .long 1), (1, .long 2)] := by
  simp [keepFirst, BV.isNull]

/-- Header `p, p`, record `null, 3`: a null in the first column lets the later one supply the
value (why the header cannot simply be deduplicated). -/
theorem repeated_header_null_first :
    keepFirst [] ([0, 0].zip [.null, .long 3]) = [(0, .long 3)] := by
  simp [keepFirst, BV.isNull]

/-- Historical (before #3026, `fb3d516d0`): `process_node_token`/`process_edge_token` pushed
one entry per non-null column, so the record of header `p, q, p` carried two entries for `p`.
(#3063 already deduplicated it downstream in `import_attrs_resolved`, first entry kept — see
`keepFirst_plainRead` — so no stored entity changes; #3026 makes the parse itself correct.) -/
theorem pre3026_two_entries_for_one_attr :
    (plainRead ([0, 1, 0].zip [.long 1, .long 2, .long 3])).map Prod.fst = [0, 1, 0] ∧
    ((keepFirst [] ([0, 1, 0].zip [.long 1, .long 2, .long 3])).map Prod.fst) = [0, 1] := by
  simp [plainRead, keepFirst, BV.isNull]

end RedisLayer.Bulk
