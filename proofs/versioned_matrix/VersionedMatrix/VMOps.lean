import VersionedMatrix.DeltaBook
/-
# `VersionedMatrix<bool>` with its bookkeeping: the remaining methods

`BVM` is `struct VersionedMatrix` (`versioned_matrix.rs:644`) with the full
`Delta` bookkeeping of `DeltaBook.lean` (the set-level layer algebra of
`set`/`remove`/`flush`/… is `Delta.lean`/`DeltaProofs.lean`).

| here | there |
| --- | --- |
| `getM`/`getDp`/`getDm`/`nrows`/`ncols` | accessors :678-703 |
| `wait`       | `wait` (:709) |
| `waitBase`/`waitAll`/`isSynced` | :744 / :750 / :759 |
| `memoryUsage`| :764 (`memM` = `GxB_Matrix_memoryUsage`, FFI) |
| `print`      | :798 (the sequence of layers printed) |
| `clearDeltas`| :925 |
| `fromMatrix` | :1041 |
| `transpose`  | :1280 |
| `clone`      | :661 |
| `encode`/`decode` | :1293 / :1304 over a token stream of matrices |
-/
namespace VMOps
open VMMat VMRowFilter VMPolicy VMDelta

structure BVM where
  m : Mat Unit
  dp : Delta Unit
  dm : Delta Unit
  nf : Bool

/-- The logical matrix `(m ∖ dm) ∪ dp`. -/
def effK (v : BVM) (p : Pair) : Prop :=
  (p ∈ keys v.m ∧ p ∉ keys v.dm.layer) ∨ p ∈ keys v.dp.layer

def getM (v : BVM) := v.m
def getDp (v : BVM) := v.dp.layer
def getDm (v : BVM) := v.dm.layer
def nrows (v : BVM) := v.m.nrows
def ncols (v : BVM) := v.m.ncols

theorem accessors (v : BVM) : getM v = v.m ∧ getDp v = v.dp.layer ∧ getDm v = v.dm.layer ∧
    nrows v = v.m.nrows ∧ ncols v = v.m.ncols := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `wait` (:709). -/
def wait (v : BVM) : BVM :=
  if v.dp.layer.synced && v.dm.layer.synced then v else
  let dp := resync v.dp
  let dm := resync v.dm
  let base := nvals v.m
  { v with dp := latch dp (foldDecision dp shouldFoldRead base),
           dm := latch dm (foldDecision dm shouldFoldRead base) }

theorem wait_keys (v : BVM) :
    keys (wait v).m = keys v.m ∧ keys (wait v).dp.layer = keys v.dp.layer ∧
    keys (wait v).dm.layer = keys v.dm.layer := by
  unfold wait; split
  · exact ⟨rfl, rfl, rfl⟩
  · simp [latch_layer, resync, VMMat.wait]; exact ⟨rfl, rfl⟩

theorem wait_eff (v : BVM) (p : Pair) : effK (wait v) p ↔ effK v p := by
  obtain ⟨h1, h2, h3⟩ := wait_keys v; simp [effK, h1, h2, h3]

/-- `wait` never arms the fold (the deliberate mid-transaction deferral). -/
theorem wait_nf (v : BVM) : (wait v).nf = v.nf := by unfold wait; split <;> rfl

/-- A latched decision is never revoked by `wait`. -/
theorem wait_latch_mono (v : BVM) (h : v.dp.fold = true) : (wait v).dp.fold = true := by
  unfold wait; split
  · exact h
  · exact latch_monotone _ _ h

/-- When `wait` runs, both counters are exact afterwards. -/
theorem wait_exact (v : BVM) (h : (v.dp.layer.synced && v.dm.layer.synced) = false) :
    Exact (wait v).dp ∧ Exact (wait v).dm := by
  unfold wait; rw [if_neg (by simp [h])]
  constructor
  · show (latch _ _).count = nvals (latch _ _).layer
    rw [latch_layer]; unfold latch; split <;> rfl
  · show (latch _ _).count = nvals (latch _ _).layer
    rw [latch_layer]; unfold latch; split <;> rfl

theorem wait_rowsOk (v : BVM) (h1 : RowsOk v.dp) (h2 : RowsOk v.dm) :
    RowsOk (wait v).dp ∧ RowsOk (wait v).dm := by
  unfold wait; split
  · exact ⟨h1, h2⟩
  · exact ⟨latch_rowsOk (resync_rowsOk h1) _, latch_rowsOk (resync_rowsOk h2) _⟩

def waitBase (v : BVM) : BVM := { v with m := VMMat.wait v.m }
def waitAll (v : BVM) : BVM :=
  { v with m := VMMat.wait v.m, dp := { v.dp with layer := VMMat.wait v.dp.layer },
           dm := { v.dm with layer := VMMat.wait v.dm.layer } }
def isSynced (v : BVM) : Bool := v.m.synced && v.dp.layer.synced && v.dm.layer.synced

theorem waitAll_synced (v : BVM) : isSynced (waitAll v) = true := rfl
theorem waitAll_eff (v : BVM) (p : Pair) : effK (waitAll v) p ↔ effK v p := Iff.rfl
theorem waitBase_synced (v : BVM) : (waitBase v).m.synced = true ∧ effK (waitBase v) = effK v := ⟨rfl, rfl⟩
theorem isSynced_iff (v : BVM) :
    isSynced v = true ↔ v.m.synced = true ∧ v.dp.layer.synced = true ∧ v.dm.layer.synced = true := by
  simp [isSynced, and_assoc]

def memoryUsage (memM : Mat Unit → Nat) (v : BVM) : Nat := memM v.m + memM v.dp.layer + memM v.dm.layer
theorem memoryUsage_eq (memM : Mat Unit → Nat) (v : BVM) :
    memoryUsage memM v = memM v.m + memM v.dp.layer + memM v.dm.layer := rfl

def print (v : BVM) : List (Mat Unit) := [v.m, v.dp.layer, v.dm.layer]
theorem print_order (v : BVM) : print v = [getM v, getDp v, getDm v] := rfl

/-- `clear_deltas` (:925). -/
def clearDeltas (v : BVM) (r c : Nat) : BVM :=
  { v with dp := clear v.dp r c, dm := clear v.dm r c, nf := false }
theorem clearDeltas_spec (v : BVM) (r c : Nat) :
    (clearDeltas v r c).nf = false ∧ (clearDeltas v r c).dp.layer.ents = [] ∧
    (clearDeltas v r c).dm.layer.ents = [] ∧ RowsOk (clearDeltas v r c).dp ∧
    RowsOk (clearDeltas v r c).dm ∧ Exact (clearDeltas v r c).dp ∧ Exact (clearDeltas v r c).dm :=
  ⟨rfl, rfl, rfl, clear_rowsOk v.dp _ _, clear_rowsOk v.dm _ _, rfl, rfl⟩
theorem clearDeltas_eff (v : BVM) (r c : Nat) (p : Pair) : effK (clearDeltas v r c) p ↔ p ∈ keys v.m := by
  simp [effK, clearDeltas, clear, keys, empty]

/-- `from_matrix` (:1041). -/
def fromMatrix (m : Mat Unit) : BVM :=
  let m := VMMat.wait m
  ⟨m, ofLayer (empty m.nrows m.ncols), ofLayer (empty m.nrows m.ncols), false⟩
theorem fromMatrix_spec (m : Mat Unit) (p : Pair) :
    (effK (fromMatrix m) p ↔ p ∈ keys m) ∧ (fromMatrix m).nf = false ∧ (fromMatrix m).m.synced = true ∧
    RowsOk (fromMatrix m).dp ∧ RowsOk (fromMatrix m).dm := by
  refine ⟨?_, rfl, rfl, ofLayer_rowsOk _, ofLayer_rowsOk _⟩
  simp [effK, fromMatrix, ofLayer, keys, empty, VMMat.wait]

/-- `transpose` (:1280). -/
def transpose (v : BVM) : BVM := ⟨transposeM v.m, transposed v.dp, transposed v.dm, v.nf⟩
theorem transpose_eff (v : BVM) (i j : Nat) : effK (transpose v) (i, j) ↔ effK v (j, i) := by
  have hm : (i, j) ∈ keys (transposeM v.m) ↔ (j, i) ∈ keys v.m := by
    simpa [transposed, relayer] using transposed_keys ⟨v.m, 0, 0, false, .empty⟩ i j
  simp only [effK, transpose, transposed_keys, hm]
theorem transpose_bookkeeping (v : BVM) :
    (transpose v).nf = v.nf ∧ (transpose v).dp.count = v.dp.count ∧ (transpose v).dm.fold = v.dm.fold ∧
    RowsOk (transpose v).dp ∧ RowsOk (transpose v).dm :=
  ⟨rfl, rfl, rfl, sound_unknown _, sound_unknown _⟩

/-- `Clone` (:661): shares the handles; observationally the same value
(the sharing hazard is `VMCow.clone_of_owned_is_not_isolated`). -/
def clone (v : BVM) : BVM := ⟨v.m, VMDelta.clone v.dp, VMDelta.clone v.dm, v.nf⟩
theorem clone_eq (v : BVM) : clone v = v := rfl

/-! ## Encode / decode (:1293 / :1304)

The writer is modelled at the granularity of whole matrices: `Matrix::encode`
emits one token, `Matrix::decode` consumes one (its byte-level round trip is
`graphblas_wrappers`, `Index`/`Decode`). A decoded matrix is synced. -/

def encode (v : BVM) : List (Mat Unit) := [v.m, v.dp.layer, v.dm.layer]

def decodeMat : List (Mat Unit) → Except String (Mat Unit × List (Mat Unit))
  | [] => .error "eof"
  | x :: xs => .ok (VMMat.wait x, xs)

def decode (s : List (Mat Unit)) : Except String (BVM × List (Mat Unit)) := do
  let (m, s) ← decodeMat s
  let (dp, s) ← decodeMat s
  let (dm, s) ← decodeMat s
  let base := nvals m
  let dp := ofLayer dp
  let dm := ofLayer dm
  let dp := latch dp (foldDecision dp shouldFold base)
  let dm := latch dm (foldDecision dm shouldFold base)
  pure (⟨m, dp, dm, folding dp || folding dm⟩, s)

theorem decode_encode (v : BVM) (rest : List (Mat Unit)) :
    ∃ w, decode (encode v ++ rest) = .ok (w, rest) ∧
      keys w.m = keys v.m ∧ keys w.dp.layer = keys v.dp.layer ∧ keys w.dm.layer = keys v.dm.layer ∧
      (∀ p, effK w p ↔ effK v p) ∧ RowsOk w.dp ∧ RowsOk w.dm ∧ Exact w.dp ∧ Exact w.dm ∧
      w.dp.tx = 0 ∧ w.dm.tx = 0 ∧ w.nf = (w.dp.fold || w.dm.fold) := by
  refine ⟨_, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [latch_layer]; rfl
  · simp only [latch_layer]; rfl
  · intro p; simp only [effK, latch_layer]; rfl
  · exact latch_rowsOk (ofLayer_rowsOk _) _
  · exact latch_rowsOk (ofLayer_rowsOk _) _
  · show (latch _ _).count = nvals (latch _ _).layer
    rw [latch_layer]; unfold latch; split <;> rfl
  · show (latch _ _).count = nvals (latch _ _).layer
    rw [latch_layer]; unfold latch; split <;> rfl
  · show (latch _ _).tx = 0; unfold latch; split <;> rfl
  · show (latch _ _).tx = 0; unfold latch; split <;> rfl
  · rfl

/-- A decoded delta is all "added by this transaction" (`tx_nvals = 0`), so the
write policy sees it whole on the first flush. -/
theorem decode_decision (m dp : Mat Unit) :
    foldDecision (ofLayer dp) shouldFold (nvals m) = shouldFold (nvals dp) (nvals dp) (nvals m) := by
  simp [foldDecision, ofLayer]

theorem decode_short (a b : Mat Unit) : decode [a, b] = .error "eof" := rfl

end VMOps
