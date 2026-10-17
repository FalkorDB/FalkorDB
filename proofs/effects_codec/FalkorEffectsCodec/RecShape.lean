import FalkorEffectsCodec.RecRT
/-!
# #2916: the encoder refuses exactly the shapes `read_record` rejects

Before `2c874022a` `Record::encode` wrote a batch of zero ids and a row block of
any length; `read_record` refuses the first (`EmptyRecord`) and misreads the
second (it takes `count × attrs` values, so a longer block's spare value is
parsed as the next opcode — the wave-1 counterexample, which no longer holds).
Now `write_header` refuses a zero count and `check_row_shape` a row block that
is not `count × attrs`, so:

* `encRecord_ok_shape` — a record the encoder accepts has ≥ 1 id, the right
  row count, and aligned endpoint columns;
* `encRecord_rowShape_refused`, `encRecord_empty_refused`,
  `encRecord_empty_never_ok` — and a record without them is refused, naming why;
* `readRecord_empty` — the reader's half: a batchable header with count 0;
* `readRecord_encRecord_wf` — **every record the encoder writes reads back**,
  under only the per-value conditions (widths, UTF-8) — no shape hypotheses.
-/
namespace FalkorCodec

variable (C : IdCodec) (utf8 I : Bytes → Bool)

/-- The shape facts `read_record` needs and the encoder now guarantees. -/
def Shape : Record C.T → Prop
  | .createNode ids _ as rows | .updateNode ids _ as rows | .updateEdge ids _ as rows =>
      0 < C.count ids ∧ rows.length = C.count ids * as.length
  | .createEdge ids _ src dst as rows =>
      0 < C.count ids ∧ rows.length = C.count ids * as.length ∧
        C.count src = C.count ids ∧ C.count dst = C.count ids
  | .deleteEdge ids _ src dst =>
      0 < C.count ids ∧ C.count src = C.count ids ∧ C.count dst = C.count ids
  | .deleteNode ids _ | .setLabels ids _ | .removeLabels ids _ => 0 < C.count ids
  | _ => True

theorem checkRowShape_ok {ids : C.T} {as : List Nat} {rows : List Val}
    (h : checkRowShape C ids as rows = .ok ()) : rows.length = C.count ids * as.length := by
  unfold checkRowShape at h; split at h <;> simp_all

theorem thenE_split {x : Except EErr Unit} {y : Except EErr Bytes} {bs : Bytes}
    (h : thenE x y = .ok bs) : x = .ok () ∧ y = .ok bs := by
  unfold thenE at h; split at h <;> simp_all

/-- **What the encoder accepts has the shape the reader needs.** -/
theorem encRecord_ok_shape (r : Record C.T) (bs : Bytes) (h : encRecord C utf8 I r = .ok bs) :
    Shape C r := by
  cases r with
  | createNode ids ls as rows =>
    obtain ⟨h1, h2⟩ := thenE_split h
    exact ⟨(batched_ok C (op := 3) (by omega) h2).1, checkRowShape_ok C h1⟩
  | updateNode ids ls as rows =>
    obtain ⟨h1, h2⟩ := thenE_split h
    exact ⟨(batched_ok C (op := 1) (by omega) h2).1, checkRowShape_ok C h1⟩
  | updateEdge ids rel as rows =>
    obtain ⟨h1, h2⟩ := thenE_split h
    exact ⟨(batched_ok C (op := 2) (by omega) h2).1, checkRowShape_ok C h1⟩
  | createEdge ids rel src dst as rows =>
    obtain ⟨h0, h12⟩ := thenE_split h
    obtain ⟨h1, h2⟩ := thenE_split h12
    have := (checkEndpoints_ok C ids src dst).mp h0
    exact ⟨(batched_ok C (op := 4) (by omega) h2).1, checkRowShape_ok C h1, this.1, this.2⟩
  | deleteEdge ids rel src dst =>
    obtain ⟨h0, h2⟩ := thenE_split h
    have := (checkEndpoints_ok C ids src dst).mp h0
    exact ⟨(batched_ok C (op := 6) (by omega) h2).1, this.1, this.2⟩
  | deleteNode ids ls => exact (batched_ok C (op := 5) (by omega) h).1
  | setLabels ids ls => exact (batched_ok C (op := 7) (by omega) h).1
  | removeLabels ids ls => exact (batched_ok C (op := 8) (by omega) h).1
  | _ => trivial

/-- A batchable record with no ids is never written. -/
theorem encRecord_empty_never_ok (r : Record C.T) (bs : Bytes)
    (h0 : match r with
      | .createNode ids .. | .updateNode ids .. | .updateEdge ids .. | .createEdge ids ..
      | .deleteEdge ids .. | .deleteNode ids _ | .setLabels ids _ | .removeLabels ids _ => C.count ids = 0
      | _ => False) :
    encRecord C utf8 I r ≠ .ok bs := by
  intro h
  have hs := encRecord_ok_shape C utf8 I r bs h
  cases r <;> simp only [Shape] at hs h0 <;> omega

/-- …and when nothing else is wrong with it, the refusal is `EmptyRecord`
(here for the label-shaped records, whose only other check is the count). -/
theorem encRecord_empty_refused (ids : C.T) (ls : List Nat) (h0 : C.count ids = 0) :
    encRecord C utf8 I (.deleteNode ids ls) = .error .emptyRecord ∧
    encRecord C utf8 I (.setLabels ids ls) = .error .emptyRecord ∧
    encRecord C utf8 I (.removeLabels ids ls) = .error .emptyRecord := by
  simp [encRecord, batched, idCount, h0, eseq, writeHeader, isBatchable]

/-- A row block that is not `count × attrs` is refused as `RowShapeMismatch`
(on `CREATE_EDGE`, once its endpoint columns are aligned). -/
theorem encRecord_rowShape_refused (ids src dst : C.T) (ls as : List Nat) (rel : Nat) (rows : List Val)
    (hr : rows.length ≠ C.count ids * as.length) :
    encRecord C utf8 I (.createNode ids ls as rows) = .error .rowShapeMismatch ∧
    encRecord C utf8 I (.updateNode ids ls as rows) = .error .rowShapeMismatch ∧
    encRecord C utf8 I (.updateEdge ids rel as rows) = .error .rowShapeMismatch ∧
    (checkEndpoints C ids src dst = .ok () →
      encRecord C utf8 I (.createEdge ids rel src dst as rows) = .error .rowShapeMismatch) := by
  have hc : checkRowShape C ids as rows = .error .rowShapeMismatch := by
    unfold checkRowShape; rw [if_neg (fun e => hr e.symm)]
  refine ⟨?_, ?_, ?_, fun he => ?_⟩
  · simp [encRecord, thenE, hc]
  · simp [encRecord, thenE, hc]
  · simp [encRecord, thenE, hc]
  · simp [encRecord, thenE, hc, he]

/-- The reader's half: a batchable header with a zero count is `EmptyRecord`. -/
theorem readRecord_empty (op : Nat) (hop : 1 ≤ op ∧ op ≤ 8) (rest : Bytes) :
    readRecord C utf8 I (w32 op ++ w32 0 ++ rest) = .error (.emptyRecord op) := by
  have hb : isBatchable op = true := by simp [isBatchable]; omega
  simp only [readRecord, bind_apply, List.append_assoc]
  rw [u32_w32 (by omega)]
  have : ¬ (op = 0 ∨ op > 14) := by omega
  simp only [this, ite_false, hb, ite_true, bind_apply]
  rw [u32_w32 (by omega)]
  simp [fail]

/-- The per-value conditions the encoder still does not check (field widths,
UTF-8, value well-formedness): `RecOk` without its shape half. -/
def RecWF : Record C.T → Prop
  | .createNode _ ls as rows | .updateNode _ ls as rows => bLabels.ok ls ∧ bAttrIds.ok as ∧ WFL utf8 rows
  | .updateEdge _ rel as rows | .createEdge _ rel _ _ as rows => bU32.ok rel ∧ bAttrIds.ok as ∧ WFL utf8 rows
  | .deleteNode _ ls | .setLabels _ ls | .removeLabels _ ls => bLabels.ok ls
  | .deleteEdge _ rel _ _ => bU32.ok rel
  | r => RecOk C utf8 I r

theorem recOk_of (r : Record C.T) (hs : Shape C r) (hw : RecWF C utf8 I r) : RecOk C utf8 I r := by
  cases r <;> simp_all [Shape, RecWF, RecOk, bodyNode, bodyUpdEdge, bodyCrEdge, bodyLabels, bodyDelEdge,
    Blk.seq, bIds, bRows]

/-- **Every record the encoder writes reads back as itself**, whatever follows
it — with no hypothesis about its shape, which the encoder now enforces. -/
theorem readRecord_encRecord_wf (r : Record C.T) (bs rest : Bytes)
    (h : encRecord C utf8 I r = .ok bs) (hw : RecWF C utf8 I r) :
    readRecord C utf8 I (bs ++ rest) = .ok (r, rest) :=
  readRecord_encRecord C utf8 I r bs rest h (recOk_of C utf8 I r (encRecord_ok_shape C utf8 I r bs h) hw)

end FalkorCodec
