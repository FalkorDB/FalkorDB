import VersionedMatrix.TensorBlocks
import VersionedMatrix.Policy
/-
# `Tensor`: queries, degrees, maintenance, resize, encode, iterator plumbing

| here | there (`tensor.rs`) |
| --- | --- |
| `fwdIter`/`structuralIter` | `fwd_iter` (:1057) / `structural_iter` (:1045) |
| `meDegree`/`degLoop`/`rowDegree` | `me_degree` (:1106) / `row_degree` (:1082) |
| `colDegree`   | `col_degree` (:1134) (`none` = the `unreachable!`) |
| `extract`     | `extract` (:977) |
| `rebuildBackward` | `rebuild_backward` (:998) |
| `waitFwd`/`waitT`/`waitBase`/`waitAll`/`isSyncedT` | :421 / :1403 / :1414 / :1423 / :1436 |
| `memoryUsage` | :1445 |
| `flushT`/`foldLatched`/`foldOversized` | :880 / :937 / :955 |
| `dupT`        | `dup` (:1009) |
| `resizeT`     | `resize` (:809) |
| `encVal`/`decVal` | the inline value `encode` (:1464) writes / C's MSB reading of it |
| `EdgeIds`/`sizeHint` | `enum EdgeIds` (:1648) / `size_hint` (:1663) |
| `TIt`/`newTIt`/`seekTIt` | `Iter` (:1680) / `Iter::new` (:1691) / `Iter::seek` (:1723) |

`mt`/`me` are `VersionedMatrix`es whose own `wait`/`flush`/`fold_*` never
change their logical contents (`VMOps.wait_eff`, `DeltaProofs.flush_eff`,
`fold_eff`), so in the per-pair model they are untouched by those calls.
-/
namespace VMTensorOps
open VMTensor VMPolicy

def inRow (lo hi : Nat) (r : Nat) : Bool := decide (lo ≤ r) && decide (r ≤ hi)

/-- `fwd_iter` (:1057): effective forward pairs in range, with their inline value. -/
def fwdIter (t : TT) (lo hi : Nat) : List (Pair × Val) :=
  (t.ps.filter (fun p => inRow lo hi p.1)).filterMap (fun p => (effV (t.st p)).map (fun v => (p, v)))

def structuralIter (t : TT) (lo hi : Nat) : List Pair := (fwdIter t lo hi).map (·.1)

theorem mem_fwdIter (t : TT) (lo hi : Nat) (p : Pair) (v : Val) :
    (p, v) ∈ fwdIter t lo hi ↔ p ∈ t.ps ∧ inRow lo hi p.1 = true ∧ effV (t.st p) = some v := by
  simp only [fwdIter, List.mem_filterMap, List.mem_filter, Option.map_eq_some_iff]
  constructor
  · rintro ⟨q, ⟨hq, hr⟩, w, hw, he⟩; injection he with h1 h2; subst h1; subst h2; exact ⟨hq, hr, hw⟩
  · rintro ⟨hq, hr, hw⟩; exact ⟨p, ⟨hq, hr⟩, v, hw, rfl⟩

theorem mem_structuralIter (t : TT) (lo hi : Nat) (p : Pair) :
    p ∈ structuralIter t lo hi ↔ p ∈ t.ps ∧ inRow lo hi p.1 = true ∧ (effV (t.st p)).isSome = true := by
  simp only [structuralIter, List.mem_map]
  constructor
  · rintro ⟨⟨q, v⟩, h, rfl⟩; have := (mem_fwdIter t lo hi q v).1 h; exact ⟨this.1, this.2.1, by simp [this.2.2]⟩
  · rintro ⟨h1, h2, h3⟩
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 h3
    exact ⟨(p, v), (mem_fwdIter t lo hi p v).2 ⟨h1, h2, hv⟩, rfl⟩

/-! ## Degrees -/

/-- `me_degree` (:1106): the pair's `me` row length through the shared cursor
(re-seeked when on the same block — `VMIterSeek.seek_abs` makes the reseeked
cursor yield what a fresh one would). Returns the count and the cursor's block. -/
def meDegree (t : TT) (cur : Option Blk) (p : Pair) : Nat × Option Blk :=
  if blkOf p ∈ names t then ((t.st p).me.length, some (blkOf p)) else (0, cur)

theorem meDegree_cursor_irrelevant (t : TT) (c1 c2 : Option Blk) (p : Pair) :
    (meDegree t c1 p).1 = (meDegree t c2 p).1 := by unfold meDegree; split <;> rfl

def contrib (t : TT) (pv : Pair × Val) : Nat :=
  if pv.2 ≠ .multi then 1 else (meDegree t none pv.1).1

def degLoop (t : TT) : List (Pair × Val) → Option Blk → Nat → Nat
  | [], _, d => d
  | (p, v) :: xs, cur, d =>
    if v ≠ .multi then degLoop t xs cur (d + 1)
    else let r := meDegree t cur p; degLoop t xs r.2 (d + r.1)

theorem degLoop_eq (t : TT) : ∀ (xs : List (Pair × Val)) (cur : Option Blk) (d : Nat),
    degLoop t xs cur d = d + (xs.map (contrib t)).sum
  | [], _, d => by simp [degLoop]
  | (p, v) :: xs, cur, d => by
    simp only [degLoop, List.map_cons, List.sum_cons, contrib]
    split
    · rw [degLoop_eq t xs]; omega
    · rw [degLoop_eq t xs, meDegree_cursor_irrelevant t cur none p]; omega

def rowDegree (t : TT) (r : Nat) : Nat := degLoop t (fwdIter t r r) none 0

def AllInv (t : TT) : Prop := ∀ p, Inv (t.st p)

theorem contrib_edges {t : TT} (hb : BlkInv t) (hi : AllInv t) {p : Pair} {v : Val}
    (hv : effV (t.st p) = some v) : contrib t (p, v) = (edges (t.st p)).length := by
  unfold contrib edges; rw [hv]
  cases v with
  | id i => simp
  | multi =>
    have hne : (t.st p).me ≠ [] := (me_nonempty_iff_multi _ (hi p)).2 hv
    simp [meDegree, hb.present p hne]

theorem sum_filterMap_edges {t : TT} (hb : BlkInv t) (hi : AllInv t) : ∀ (l : List Pair),
    ((l.filterMap (fun p => (effV (t.st p)).map (fun v => (p, v)))).map (contrib t)).sum =
      (l.map (fun p => (edges (t.st p)).length)).sum
  | [] => rfl
  | p :: ps => by
    rw [List.filterMap_cons]
    cases hv : effV (t.st p) with
    | none =>
      have : edges (t.st p) = [] := by unfold edges; rw [hv]
      simp [this, sum_filterMap_edges hb hi ps]
    | some v =>
      simp [contrib_edges hb hi hv, sum_filterMap_edges hb hi ps]

/-- `row_degree` counts every edge leaving the row, without materialising ids. -/
theorem rowDegree_eq {t : TT} (hb : BlkInv t) (hi : AllInv t) (r : Nat) :
    rowDegree t r = ((t.ps.filter (fun p => inRow r r p.1)).map (fun p => (edges (t.st p)).length)).sum := by
  unfold rowDegree fwdIter; rw [degLoop_eq, sum_filterMap_edges hb hi]; simp

/-- `col_degree` (:1134); `none` is the `unreachable!` (mt hit without a forward value). -/
def colDegree (t : TT) (c : Nat) : Option Nat :=
  ((t.ps.filter (fun p => p.2 == c && (t.st p).mt)).foldl
    (fun acc p => acc.bind (fun d =>
      match effV (t.st p) with
      | none => none
      | some v => some (d + contrib t (p, v)))) (some 0))

theorem colDegree_fold {t : TT} (hb : BlkInv t) (hi : AllInv t) : ∀ (l : List Pair) (d : Nat),
    (∀ p ∈ l, (t.st p).mt = true) →
    l.foldl (fun acc p => acc.bind (fun d =>
      match effV (t.st p) with
      | none => none
      | some v => some (d + contrib t (p, v)))) (some d) =
      some (d + (l.map (fun p => (edges (t.st p)).length)).sum)
  | [], d, _ => by simp
  | p :: ps, d, hmt => by
    have hs := mt_has_forward _ (hi p) (hmt p List.mem_cons_self)
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hs
    simp only [List.foldl_cons, Option.bind_some, hv]
    rw [colDegree_fold hb hi ps _ (fun q hq => hmt q (List.mem_cons_of_mem _ hq)), contrib_edges hb hi hv]
    simp; omega

/-- `col_degree` never reaches its `unreachable!`, and counts every incoming edge. -/
theorem colDegree_eq {t : TT} (hb : BlkInv t) (hi : AllInv t) (c : Nat) :
    colDegree t c = some (((t.ps.filter (fun p => p.2 == c && (t.st p).mt)).map
      (fun p => (edges (t.st p)).length)).sum) := by
  unfold colDegree; rw [colDegree_fold hb hi _ 0]; · simp
  intro p hp; simp at hp; exact hp.2.2

/-! ## extract / rebuild_backward -/

def extract (t : TT) : List Pair := t.ps.filter (fun p => (effV (t.st p)).isSome)
theorem mem_extract (t : TT) (p : Pair) :
    p ∈ extract t ↔ p ∈ t.ps ∧ (effV (t.st p)).isSome = true := by simp [extract]

/-- `rebuild_backward` (:998): `mt := from_matrix(extract().transpose())`. -/
def rebuildBackward (t : TT) : TT :=
  { t with st := fun p => { t.st p with mt := (effV (t.st p)).isSome }, mtSynced := true }

theorem rebuildBackward_mt (t : TT) (p : Pair) :
    ((rebuildBackward t).st p).mt = (effV ((rebuildBackward t).st p)).isSome := rfl
theorem rebuildBackward_edges (t : TT) (p : Pair) : edges ((rebuildBackward t).st p) = edges (t.st p) := rfl
/-- Rebuilding restores `mt_eff` even on a state where it was broken. -/
theorem rebuildBackward_inv (t : TT) (p : Pair)
    (h : ∀ q, Inv { t.st q with mt := (effV (t.st q)).isSome } ) : Inv ((rebuildBackward t).st p) := h p

/-! ## wait family (bookkeeping only) -/

def dpCount (t : TT) : Nat := (t.ps.filter (fun p => (t.st p).dp.isSome)).length
def dmCount (t : TT) : Nat := (t.ps.filter (fun p => (t.st p).dm)).length
def mCount (t : TT) : Nat := (t.ps.filter (fun p => (t.st p).m.isSome)).length

def latchB (b : Bk) (d : Bool) : Bk := if d then { b with fold := true } else b
def decideB (b : Bk) (policy : Nat → Nat → Nat → Bool) (base : Nat) : Bool :=
  b.fold || policy b.count (b.count - b.tx) base

/-- `wait_fwd` (:421). -/
def waitFwd (t : TT) : TT :=
  if t.dpB.synced && t.dmB.synced then t else
  let dp : Bk := { t.dpB with count := dpCount t, synced := true }
  let dm : Bk := { t.dmB with count := dmCount t, synced := true }
  { t with dpB := latchB dp (decideB dp shouldFoldRead (mCount t)),
           dmB := latchB dm (decideB dm shouldFoldRead (mCount t)) }

theorem waitFwd_st (t : TT) : (waitFwd t).st = t.st ∧ (waitFwd t).nf = t.nf ∧ (waitFwd t).me = t.me := by
  unfold waitFwd; split <;> exact ⟨rfl, rfl, rfl⟩
theorem latchB_mono (b : Bk) (d : Bool) (h : b.fold = true) : (latchB b d).fold = true := by
  unfold latchB; split <;> simp [h]
theorem waitFwd_latch_mono (t : TT) (h : t.dpB.fold = true) : (waitFwd t).dpB.fold = true := by
  unfold waitFwd; split; exact h; exact latchB_mono _ _ h
theorem waitFwd_exact (t : TT) (h : (t.dpB.synced && t.dmB.synced) = false) :
    (waitFwd t).dpB.count = dpCount t ∧ (waitFwd t).dmB.count = dmCount t := by
  have hc : ∀ (b : Bk) d, (latchB b d).count = b.count := by intro b d; unfold latchB; split <;> rfl
  unfold waitFwd; rw [if_neg (by simp [h])]; exact ⟨hc _ _, hc _ _⟩

def syncAll (me : List MeB) : List MeB := me.map (fun e => { e with synced := true })

/-- `Tensor::wait` (:1403). -/
def waitT (t : TT) : TT := { waitFwd t with mtSynced := true, me := syncAll t.me }
/-- `wait_base` (:1414): bases only (`m`; `mt.m`, `me.m` are inside the VMs). -/
def waitBase (t : TT) : TT := { t with mSynced := true }
/-- `wait_all` (:1423). -/
def waitAll (t : TT) : TT :=
  { t with mSynced := true, dpB := { t.dpB with synced := true }, dmB := { t.dmB with synced := true },
           mtSynced := true, me := syncAll t.me }
def isSyncedT (t : TT) : Bool :=
  t.mSynced && t.dpB.synced && t.dmB.synced && t.mtSynced && t.me.all (·.synced)

theorem waitAll_synced (t : TT) : isSyncedT (waitAll t) = true := by
  simp [isSyncedT, waitAll, syncAll]
theorem waitAll_st (t : TT) : (waitAll t).st = t.st := rfl
theorem waitBase_st (t : TT) : (waitBase t).st = t.st ∧ (waitBase t).mSynced = true := ⟨rfl, rfl⟩
theorem waitT_st (t : TT) : (waitT t).st = t.st ∧ (waitT t).nf = t.nf := by
  simp only [waitT]; exact ⟨(waitFwd_st t).1, (waitFwd_st t).2.1⟩
theorem waitT_me_synced (t : TT) : ∀ e ∈ (waitT t).me, e.synced = true := by
  intro e he; simp [waitT, syncAll] at he; obtain ⟨a, _, rfl⟩ := he; rfl

def memoryUsage (fwd mt : Nat) (meMem : MeB → Nat) (t : TT) : Nat := fwd + mt + (t.me.map meMem).sum
theorem memoryUsage_eq (fwd mt : Nat) (meMem : MeB → Nat) (t : TT) :
    memoryUsage fwd mt meMem t = fwd + mt + (t.me.map meMem).sum := rfl

/-! ## flush / fold_latched / fold_oversized / dup -/

/-- `Tensor::flush` (:880): `take_fold` both, `foldP` every pair, clear the folded deltas. -/
def flushT (t : TT) : TT :=
  if t.nf then
    let a := t.dpB.fold && decide (0 < dpCount t)
    let b := t.dmB.fold && decide (0 < dmCount t)
    { t with st := fun p => foldP (t.st p) a b,
             dpB := if a then ⟨0, 0, false, true⟩ else { t.dpB with fold := false },
             dmB := if b then ⟨0, 0, false, true⟩ else { t.dmB with fold := false },
             nf := false }
  else t

theorem flushT_correct {t : TT} (hi : AllInv t) :
    AllInv (flushT t) ∧ ∀ p, edges ((flushT t).st p) = edges (t.st p) := by
  unfold flushT; split
  · exact ⟨fun p => (foldP_correct _ (hi p) _ _).1, fun p => (foldP_correct _ (hi p) _ _).2.2⟩
  · exact ⟨hi, fun _ => rfl⟩

theorem flushT_me (t : TT) : (flushT t).me = t.me ∧ ∀ p, ((flushT t).st p).me = (t.st p).me := by
  unfold flushT; split
  · refine ⟨rfl, fun p => ?_⟩; simp only; cases t.dpB.fold && decide (0 < dpCount t) <;>
      cases t.dmB.fold && decide (0 < dmCount t) <;> rfl
  · exact ⟨rfl, fun _ => rfl⟩

/-- `fold_latched` (:937). -/
def foldLatched (t : TT) : TT :=
  let t := waitFwd t
  if t.dpB.fold || t.dmB.fold then flushT { t with nf := true } else t

/-- `fold_oversized` (:955). -/
def foldOversized (t : TT) : TT :=
  let base := mCount t
  let a := deltaDominatesBase t.dpB.count base
  let b := deltaDominatesBase t.dmB.count base
  if a || b then flushT { t with dpB := latchB t.dpB a, dmB := latchB t.dmB b, nf := true } else t

theorem foldLatched_correct {t : TT} (hi : AllInv t) :
    AllInv (foldLatched t) ∧ ∀ p, edges ((foldLatched t).st p) = edges (t.st p) := by
  have hw := (waitFwd_st t).1
  have hi' : AllInv (waitFwd t) := by rw [AllInv, hw]; exact hi
  unfold foldLatched; simp only; split
  · have := flushT_correct (t := { waitFwd t with nf := true }) hi'
    exact ⟨this.1, fun p => by rw [this.2 p]; simp [hw]⟩
  · exact ⟨hi', fun p => by rw [hw]⟩

theorem foldOversized_correct {t : TT} (hi : AllInv t) :
    AllInv (foldOversized t) ∧ ∀ p, edges ((foldOversized t).st p) = edges (t.st p) := by
  unfold foldOversized; simp only; split
  · exact flushT_correct (t := { t with dpB := _, dmB := _, nf := true }) hi
  · exact ⟨hi, fun _ => rfl⟩

/-- `fold_oversized` folds nothing unless a delta dominates the base. -/
theorem foldOversized_lazy (t : TT) (h1 : deltaDominatesBase t.dpB.count (mCount t) = false)
    (h2 : deltaDominatesBase t.dmB.count (mCount t) = false) : foldOversized t = t := by
  simp [foldOversized, h1, h2]

/-- `dup` (:1009). -/
def dupT (t : TT) : TT :=
  let a := decideB t.dpB shouldFold (mCount t)
  let b := decideB t.dmB shouldFold (mCount t)
  { t with dpB := { t.dpB with tx := t.dpB.count, fold := a },
           dmB := { t.dmB with tx := t.dmB.count, fold := b }, nf := a || b }

theorem dupT_spec (t : TT) :
    (dupT t).st = t.st ∧ (dupT t).me = t.me ∧ (dupT t).dpB.tx = t.dpB.count ∧
    (dupT t).nf = ((dupT t).dpB.fold || (dupT t).dmB.fold) := ⟨rfl, rfl, rfl, rfl⟩

/-- A fresh version (nothing added yet) inherits exactly the latched decisions. -/
theorem dupT_fresh (t : TT) : (dupT (dupT t)).dpB.fold = (dupT t).dpB.fold := by
  simp [dupT, decideB, shouldFold, foldBalance_readOnly]

end VMTensorOps
