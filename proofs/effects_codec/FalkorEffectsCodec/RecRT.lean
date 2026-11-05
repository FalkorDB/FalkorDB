import FalkorEffectsCodec.RecFull
import FalkorEffectsCodec.Cursor
/-! # `read_record ∘ Record::encode = id`, for every variant -/
namespace FalkorCodec

variable (C : IdCodec) (utf8 I : Bytes → Bool)

theorem readRecord_batched (op c : Nat) (hop : 1 ≤ op ∧ op ≤ 8) (hc : 0 < c ∧ c < 2 ^ 32) (body : Bytes) :
    readRecord C utf8 I (w32 op ++ w32 c ++ body) = arm C utf8 I op c body := by
  have hb : isBatchable op = true := by simp [isBatchable]; omega
  simp only [readRecord, bind_apply, List.append_assoc]
  rw [u32_w32 (by omega)]
  have : ¬ (op = 0 ∨ op > 14) := by omega
  simp only [this, ite_false, hb, ite_true, bind_apply]
  rw [u32_w32 hc.2]
  have hc0 : c ≠ 0 := by omega
  simp only [true_and, hc0, ite_false]

theorem readRecord_single (op : Nat) (hop : 9 ≤ op ∧ op ≤ 14) (body : Bytes) :
    readRecord C utf8 I (w32 op ++ body) = arm C utf8 I op 0 body := by
  have hb : isBatchable op = false := by simp [isBatchable]; omega
  simp only [readRecord, bind_apply]
  rw [u32_w32 (by omega)]
  have : ¬ (op = 0 ∨ op > 14) := by omega
  simp only [this, ite_false, hb, Bool.false_eq_true, false_and, bind_apply, pure_apply]

/-- What the encoder does not check but the decoder needs, per variant: each
block's `ok` (fits its width, valid UTF-8, rows = count × attrs, schemas sorted
by id, similarity name canonical), and a batch of at least one id. -/
def RecOk : Record C.T → Prop
  | .createNode ids ls as rows | .updateNode ids ls as rows =>
      0 < C.count ids ∧ (bodyNode C utf8 I (C.count ids)).ok (ls, as, ids, rows)
  | .updateEdge ids rel as rows =>
      0 < C.count ids ∧ (bodyUpdEdge C utf8 I (C.count ids)).ok (rel, as, ids, rows)
  | .createEdge ids rel src dst as rows =>
      0 < C.count ids ∧ (bodyCrEdge C utf8 I (C.count ids)).ok (rel, as, ids, src, dst, rows)
  | .deleteNode ids ls | .setLabels ids ls | .removeLabels ids ls =>
      0 < C.count ids ∧ (bodyLabels C (C.count ids)).ok (ls, ids)
  | .deleteEdge ids rel src dst => 0 < C.count ids ∧ (bodyDelEdge C (C.count ids)).ok (rel, ids, src, dst)
  | .addLabel id name => (bodySchema utf8).ok (.node, id, name)
  | .addRelType id name => (bodySchema utf8).ok (.rel, id, name)
  | .addAttribute id name => (bodyAttr utf8).ok (id, name)
  | .createIndex st schemas ft fields opts => (bodyIndex utf8 true).ok (st, schemas, ft, fields, some opts)
  | .dropIndex st schemas ft fields => (bodyIndex utf8 false).ok (st, schemas, ft, fields, none)
  | .createConstraint ct et status lid label props =>
      (bodyConstraint utf8 true).ok (ct, et, some status, lid, label, props)
  | .dropConstraint ct et lid label props => (bodyConstraint utf8 false).ok (ct, et, none, lid, label, props)

theorem batched_ok {op : Nat} {ids : C.T} {body : Nat → Except EErr Bytes} {bs : Bytes}
    (hop : 1 ≤ op ∧ op ≤ 8) (h : batched C op ids body = .ok bs) :
    0 < C.count ids ∧ C.count ids < 2 ^ 32 ∧
      ∃ b, body (C.count ids) = .ok b ∧ bs = w32 op ++ w32 (C.count ids) ++ b := by
  unfold batched idCount at h
  by_cases hc : C.count ids < 2 ^ 32
  · simp only [hc, ite_true] at h
    obtain ⟨a, b, ha, hb, rfl⟩ := eseq_ok h
    have : isBatchable op = true := by simp [isBatchable]; omega
    by_cases h0 : C.count ids = 0
    · simp [writeHeader, this, h0] at ha
    · simp [writeHeader, this, h0] at ha
      exact ⟨by omega, hc, b, hb, by rw [← ha]⟩
  · simp [hc] at h

theorem single_ok {op : Nat} {x : Except EErr Bytes} {bs : Bytes} (hop : 9 ≤ op ∧ op ≤ 14)
    (h : eseq (writeHeader op none) x = .ok bs) : ∃ b, x = .ok b ∧ bs = w32 op ++ b := by
  obtain ⟨a, b, ha, hb, rfl⟩ := eseq_ok h
  have : isBatchable op = false := by simp [isBatchable]; omega
  simp [writeHeader, this] at ha
  exact ⟨b, hb, by rw [← ha]; try simp⟩

theorem thenE_ok {x : Except EErr Unit} {y : Except EErr Bytes} {bs : Bytes} (h : thenE x y = .ok bs) :
    y = .ok bs := by
  unfold thenE at h; split at h <;> simp_all

/-- **Every record round-trips**: whatever follows it, `read_record` gives back
exactly the record `Record::encode` wrote, for all fourteen opcodes. -/
theorem readRecord_encRecord (r : Record C.T) (bs rest : Bytes)
    (h : encRecord C utf8 I r = .ok bs) (hw : RecOk C utf8 I r) :
    readRecord C utf8 I (bs ++ rest) = .ok (r, rest) := by
  cases r with
  | createNode ids ls as rows =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 3) (by omega) (thenE_ok h)
    rw [List.append_assoc, readRecord_batched C utf8 I 3 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyNode C utf8 I _).rt _ b rest hw.2 hb]; rfl
  | updateNode ids ls as rows =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 1) (by omega) (thenE_ok h)
    rw [List.append_assoc, readRecord_batched C utf8 I 1 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyNode C utf8 I _).rt _ b rest hw.2 hb]; rfl
  | updateEdge ids rel as rows =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 2) (by omega) (thenE_ok h)
    rw [List.append_assoc, readRecord_batched C utf8 I 2 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyUpdEdge C utf8 I _).rt _ b rest hw.2 hb]; rfl
  | createEdge ids rel src dst as rows =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 4) (by omega) (thenE_ok (thenE_ok h))
    rw [List.append_assoc, readRecord_batched C utf8 I 4 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyCrEdge C utf8 I _).rt _ b rest hw.2 hb]; rfl
  | deleteNode ids ls =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 5) (by omega) h
    rw [List.append_assoc, readRecord_batched C utf8 I 5 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyLabels C _).rt _ b rest hw.2 hb]; rfl
  | deleteEdge ids rel src dst =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 6) (by omega) (thenE_ok h)
    rw [List.append_assoc, readRecord_batched C utf8 I 6 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyDelEdge C _).rt _ b rest hw.2 hb]; rfl
  | setLabels ids ls =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 7) (by omega) h
    rw [List.append_assoc, readRecord_batched C utf8 I 7 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyLabels C _).rt _ b rest hw.2 hb]; rfl
  | removeLabels ids ls =>
    obtain ⟨_, hc, b, hb, rfl⟩ := batched_ok C (op := 8) (by omega) h
    rw [List.append_assoc, readRecord_batched C utf8 I 8 _ (by omega) ⟨hw.1, hc⟩]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyLabels C _).rt _ b rest hw.2 hb]; rfl
  | addLabel id name =>
    simp only [encRecord] at h
    obtain ⟨b, hb, rfl⟩ := single_ok (op := 9) (by omega) h
    rw [List.append_assoc, readRecord_single C utf8 I 9 (by omega)]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodySchema utf8).rt _ b rest hw hb]; rfl
  | addRelType id name =>
    simp only [encRecord] at h
    obtain ⟨b, hb, rfl⟩ := single_ok (op := 9) (by omega) h
    rw [List.append_assoc, readRecord_single C utf8 I 9 (by omega)]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodySchema utf8).rt _ b rest hw hb]; rfl
  | addAttribute id name =>
    simp only [encRecord] at h
    obtain ⟨b, hb, rfl⟩ := single_ok (op := 10) (by omega) h
    rw [List.append_assoc, readRecord_single C utf8 I 10 (by omega)]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyAttr utf8).rt _ b rest hw hb]; rfl
  | createIndex st schemas ft fields opts =>
    simp only [encRecord] at h
    have h := thenE_ok h
    split at h; · cases h
    obtain ⟨b, hb, rfl⟩ := single_ok (op := 11) (by omega) h
    rw [List.append_assoc, readRecord_single C utf8 I 11 (by omega)]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyIndex utf8 true).rt _ b rest hw hb]; rfl
  | dropIndex st schemas ft fields =>
    simp only [encRecord] at h
    obtain ⟨b, hb, rfl⟩ := single_ok (op := 12) (by omega) (thenE_ok h)
    rw [List.append_assoc, readRecord_single C utf8 I 12 (by omega)]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyIndex utf8 false).rt _ b rest hw hb]; rfl
  | createConstraint ct et status lid label props =>
    simp only [encRecord] at h
    obtain ⟨b, hb, rfl⟩ := single_ok (op := 13) (by omega) h
    rw [List.append_assoc, readRecord_single C utf8 I 13 (by omega)]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyConstraint utf8 true).rt _ b rest hw hb]; rfl
  | dropConstraint ct et lid label props =>
    simp only [encRecord] at h
    obtain ⟨b, hb, rfl⟩ := single_ok (op := 14) (by omega) h
    rw [List.append_assoc, readRecord_single C utf8 I 14 (by omega)]
    simp only [arm, Nat.reduceEqDiff, ↓reduceIte, bind_apply]; rw [(bodyConstraint utf8 false).rt _ b rest hw hb]; rfl

/-- `write_header` refuses exactly a shape mismatch; every `encode` arm calls
it with the right shape, so it never refuses there. -/
theorem writeHeader_spec (op : Nat) (c : Option Nat) :
    (writeHeader op c = .error .headerShapeMismatch ↔ c.isSome ≠ isBatchable op) := by
  unfold writeHeader; split
  · simp_all
  · split <;> simp_all

/-- …and, of a correctly shaped header, exactly a zero count (#2916). -/
theorem writeHeader_empty (op : Nat) (c : Option Nat) (hs : c.isSome = isBatchable op) :
    (writeHeader op c = .error .emptyRecord ↔ c = some 0) := by
  unfold writeHeader; simp only [hs, bne_self_eq_false, Bool.false_eq_true, ite_false]
  split <;> simp_all

/-- `check_endpoint_columns` refuses exactly misaligned endpoint columns. -/
theorem checkEndpoints_ok (ids src dst : C.T) :
    checkEndpoints C ids src dst = .ok () ↔ (C.count src = C.count ids ∧ C.count dst = C.count ids) := by
  unfold checkEndpoints; split <;> (try split) <;> simp_all

/-- `check_index_record` refuses an unknown/mixed field type or no schemas. -/
theorem checkIndex_ok (ft : Nat) (schemas : List Ref) :
    checkIndex ft schemas = .ok () ↔ ((∃ k, indexTypeOf ft = .ok k) ∧ schemas ≠ []) := by
  unfold checkIndex; split <;> rename_i h <;> simp [h]

/-- `RelType` (`blocks.rs:29`/`:39`): `schema_id` / `u32`. -/
theorem relType_rt (v : Nat) (hv : v < 2 ^ 32) (rest : Bytes) :
    bU32.dec ((schemaId vecWrite [] v) ++ rest) = .ok (v, rest) := by
  simp [schemaId, vecWrite]; exact u32_w32 hv rest

end FalkorCodec
