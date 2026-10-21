import VersionedMatrix
/-! # Claim 1: `Tensor::clean_fwd` / `Tensor::clean_bwd` (PR #3435)

PR branch `perf/bfs-pushpull` 28ab35368, `graph/src/graph/graphblas/tensor.rs`:

| here | there |
| --- | --- |
| `cleanFwd`   | `clean_fwd` :989-992 — `wait_fwd()`, then `Some(&m)` iff `dp.nvals() == 0 && dm.nvals() == 0` |
| `cleanBwd`   | `clean_bwd` :997-1000 — `mt.wait()`, then `Some(mt.m())` iff `mt.dp().nvals() == 0 && mt.dm().nvals() == 0` |
| `mPattern`   | the stored pattern of the committed base `m` (:272) |
| `VMTensorOps.extract` | `extract` :1006 (owner: versioned_matrix, `mem_extract`) |

The forward tensor is the owner's whole-tensor model `VMTensorOps.TT`
(versioned_matrix `TensorBlocks.lean`): a support `ps` and one `PairSt` per pair
holding the `m`/`dp`/`dm` entries. `nvals` of `dp`/`dm` is the owner's
`dpCount`/`dmCount` (the number of support pairs whose layer holds an entry).
`wait_fwd` is the owner's `waitFwd`, which never moves an entry (`waitFwd_st`).

The backward matrix `mt` is a `VersionedMatrix<bool>`, modelled by the owner's
three-layer `VM.VM` (versioned_matrix `Delta.lean`): stored pattern lists `m`,
`dp`, `dm`, logical contents `VM.eff = (m ∖ dm) ∪ dp`, `nvals` = list length of
a duplicate-free layer. `VersionedMatrix::wait` only latches fold decisions
(owner: `Op.wait`, "never moves an entry"), so it is the identity on layers.
-/
namespace Pr3435.Clean
open VMTensor VMTensorOps

abbrev Pair := Nat × Nat

/-- The stored pattern of the committed forward base `m`. -/
def mPattern (t : TT) : List Pair := t.ps.filter (fun p => (t.st p).m.isSome)

/-- `clean_fwd` (tensor.rs:989-992). -/
def cleanFwd (t : TT) : Option (List Pair) :=
  let t' := waitFwd t
  if dpCount t' = 0 ∧ dmCount t' = 0 then some (mPattern t') else none

theorem waitFwd_ps (t : TT) : (waitFwd t).ps = t.ps := by
  unfold waitFwd; split <;> rfl

theorem filter_len_zero {l : List Pair} {f : Pair → Bool} (h : (l.filter f).length = 0) :
    ∀ p ∈ l, f p = false := by
  intro p hp
  have : l.filter f = [] := List.eq_nil_of_length_eq_zero h
  cases hf : f p
  · rfl
  · exact absurd (List.mem_filter.2 ⟨hp, hf⟩) (by simp [this])

/-- **`clean_fwd` answers `Some` exactly when both deltas are empty.** -/
theorem cleanFwd_isSome_iff (t : TT) :
    (cleanFwd t).isSome = true ↔ dpCount t = 0 ∧ dmCount t = 0 := by
  have hst := (waitFwd_st t).1
  have hps := waitFwd_ps t
  have e1 : dpCount (waitFwd t) = dpCount t := by simp [dpCount, hst, hps]
  have e2 : dmCount (waitFwd t) = dmCount t := by simp [dmCount, hst, hps]
  unfold cleanFwd; simp only [e1, e2]; split <;> simp_all

/-- When `clean_fwd` returns `Some a`, no support pair has a `dp` or `dm` entry. -/
theorem cleanFwd_some_deltas_empty {t : TT} {a : List Pair} (h : cleanFwd t = some a) :
    ∀ p ∈ t.ps, (t.st p).dp = none ∧ (t.st p).dm = false := by
  have hst := (waitFwd_st t).1
  have hps := waitFwd_ps t
  unfold cleanFwd at h; dsimp only at h
  split at h
  · rename_i hc
    obtain ⟨h1, h2⟩ := hc
    simp only [dpCount, dmCount, hst, hps] at h1 h2
    intro p hp
    have a1 := filter_len_zero h1 p hp
    have a2 := filter_len_zero h2 p hp
    refine ⟨?_, by simpa using a2⟩
    cases hd : (t.st p).dp with
    | none => rfl
    | some _ => simp [hd] at a1
  · exact absurd h (by simp)

/-- **Claim 1, forward.** When `clean_fwd` lends the base `a`, its pattern is
exactly the effective forward structure `(m ∖ dm) ∪ dp` that `extract`
materializes — equal as lists, in the same order. So BFS over the lent base
sees the same edge set as BFS over `extract()` (= `build_adjacency_matrix` for
one type, graph.rs:4731-4736). -/
theorem cleanFwd_eq_extract {t : TT} {a : List Pair} (h : cleanFwd t = some a) :
    a = extract t := by
  have hd := cleanFwd_some_deltas_empty h
  have hst := (waitFwd_st t).1
  have hps := waitFwd_ps t
  unfold cleanFwd at h; dsimp only at h
  split at h
  · simp only [Option.some.injEq] at h
    subst h
    unfold mPattern extract
    simp only [hst, hps]
    apply List.filter_congr
    intro p hp
    obtain ⟨h1, h2⟩ := hd p hp
    simp [effV, h1, h2]
  · exact absurd h (by simp)

theorem cleanFwd_mem {t : TT} {a : List Pair} (h : cleanFwd t = some a) (p : Pair) :
    p ∈ a ↔ p ∈ t.ps ∧ (effV (t.st p)).isSome = true := by
  rw [cleanFwd_eq_extract h]; exact mem_extract t p

/-! ## `clean_bwd` -/

/-- `clean_bwd` (tensor.rs:997-1000). `mt.wait()` leaves every layer as it is. -/
def cleanBwd (mt : VM.VM) : Option (List Pair) :=
  if mt.dp.length = 0 ∧ mt.dm.length = 0 then some mt.m else none

theorem cleanBwd_isSome_iff (mt : VM.VM) :
    (cleanBwd mt).isSome = true ↔ mt.dp = [] ∧ mt.dm = [] := by
  unfold cleanBwd; split <;> simp_all [List.length_eq_zero_iff]

/-- **Claim 1, backward (unconditional part).** When `clean_bwd` lends `mt`'s
base `b`, `b`'s pattern is exactly `mt`'s logical contents. -/
theorem cleanBwd_eff {mt : VM.VM} {b : List Pair} (h : cleanBwd mt = some b) (p : Pair) :
    p ∈ b ↔ VM.eff mt p := by
  unfold cleanBwd at h; split at h
  · rename_i hc
    simp only [Option.some.injEq] at h; subst h
    have hdp : mt.dp = [] := List.eq_nil_of_length_eq_zero hc.1
    have hdm : mt.dm = [] := List.eq_nil_of_length_eq_zero hc.2
    simp [VM.eff, hdp, hdm]
  · exact absurd h (by simp)

/-- How versioned_matrix abstracts `mt`: the per-pair bit `(t.st (i,j)).mt` *is*
`mt`'s logical contents at the transposed coordinate `(j,i)` (the owner models
`mt`'s `VersionedMatrix<bool>` by its logical set, `VM.run_repr`; `mt` is
`ncols × nrows`, `Tensor::new` :346, `resize` :876). Pairs outside the support
have no `mt` entry. -/
def MtRepr (t : TT) (mt : VM.VM) : Prop :=
  ∀ i j, VM.eff mt (j, i) ↔ ((i, j) ∈ t.ps ∧ (t.st (i, j)).mt = true)

/-- **Claim 1, transpose.** With `mt` representing the tensor (`MtRepr`) and the
tensor invariant `Inv.mt_eff` on every pair (`AllInv`; see `Reach.allInv` for
how the tensor's own mutators keep it), the two lent bases are transposes:
`(j,i) ∈ clean_bwd ↔ (i,j) ∈ clean_fwd`. -/
theorem clean_transpose {t : TT} {mt : VM.VM} {a b : List Pair}
    (hf : cleanFwd t = some a) (hb : cleanBwd mt = some b)
    (hr : MtRepr t mt) (hi : AllInv t) (i j : Nat) :
    (j, i) ∈ b ↔ (i, j) ∈ a := by
  rw [cleanBwd_eff hb, hr, cleanFwd_mem hf, (hi (i, j)).mt_eff]

/-- The transpose holds as soon as `mt` itself is clean, whatever the forward
deltas: `clean_bwd`'s base is the transpose of `extract()`. -/
theorem cleanBwd_transpose_extract {t : TT} {mt : VM.VM} {b : List Pair}
    (hb : cleanBwd mt = some b) (hr : MtRepr t mt) (hi : AllInv t) (i j : Nat) :
    (j, i) ∈ b ↔ (i, j) ∈ extract t := by
  rw [cleanBwd_eff hb, hr, mem_extract, (hi (i, j)).mt_eff]

/-! ## `AllInv` is the tensor's own invariant

Every tensor history the owner models keeps `AllInv` (`Inv` on every pair,
including `mt_eff`). `Reach` lists the whole-tensor transitions of tensor.rs the
owner models: `Tensor::new`, a per-pair add (`set_all_from_slices`, owner
`addBatch_correct`, precondition `AddOk`: fresh, distinct edge ids — the caller's
id allocator) or remove (`remove_all`, slow `removeBatch` / fast `removeFast`,
precondition `RemOk`), `flush`, `fold_latched`, `fold_oversized`, `resize`, `dup`,
`wait`, and decode followed by `rebuild_backward` (graph.rs:1002; decode's
checks give every invariant but `mt_eff`, which the rebuild restores). -/

def setPair (t : TT) (p : Pair) (s : PairSt) : TT :=
  { t with st := fun q => if q = p then s else t.st q }

inductive Reach : TT → Prop
  | new (r c : Nat) : Reach (newT r c)
  | add {t} (p : Pair) (ids : List Nat) : Reach t → AddOk (t.st p) ids →
      Reach (setPair t p (addBatch (t.st p) ids))
  | remove {t} (p : Pair) (ids : List Nat) : Reach t → RemOk (t.st p) ids →
      Reach (setPair t p (removeBatch (t.st p) ids))
  | removeFast {t} (p : Pair) (ids : List Nat) : Reach t → (t.st p).me = [] → RemOk (t.st p) ids →
      ids ≠ [] → Reach (setPair t p (removeFast (t.st p)))
  | flush {t} : Reach t → Reach (flushT t)
  | foldLatched {t} : Reach t → Reach (foldLatched t)
  | foldOversized {t} : Reach t → Reach (foldOversized t)
  | resize {t} (r c : Nat) : Reach t → Reach (resizeT t r c)
  | dup {t} : Reach t → Reach (dupT t)
  | wait {t} : Reach t → Reach (waitT t)
  | decoded (t : TT) : (∀ q, Inv { t.st q with mt := (effV (t.st q)).isSome }) →
      Reach (rebuildBackward t)

theorem setPair_allInv {t : TT} {p : Pair} {s : PairSt} (hi : AllInv t) (hs : Inv s) :
    AllInv (setPair t p s) := by
  intro q; unfold setPair; dsimp only; split
  · exact hs
  · exact hi q

theorem emptyP_inv : Inv emptyP := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [emptyP, effV]

/-- **`mt_eff` is maintained**: every modelled tensor history satisfies `AllInv`. -/
theorem Reach.allInv {t : TT} (h : Reach t) : AllInv t := by
  induction h with
  | new r c => intro p; exact emptyP_inv
  | add p ids _ hok ih => exact setPair_allInv ih (addBatch_correct _ (ih p) ids hok).1
  | remove p ids _ hok ih => exact setPair_allInv ih (removeBatch_correct _ (ih p) ids hok).1
  | removeFast p ids _ hme hok hne ih =>
    exact setPair_allInv ih (removeFast_correct _ (ih p) hme ids hok hne).1
  | flush _ ih => exact (flushT_correct ih).1
  | foldLatched _ ih => exact (foldLatched_correct ih).1
  | foldOversized _ ih => exact (foldOversized_correct ih).1
  | resize r c _ ih => exact resizeT_allInv ih r c
  | dup _ ih => intro p; rw [(dupT_spec _).1]; exact ih p
  | wait _ ih => intro p; rw [(waitT_st _).1]; exact ih p
  | decoded t hd => intro p; exact rebuildBackward_inv t p hd

/-- Claim 1 for every reachable tensor: only `MtRepr` (the owner's `mt` layer
abstraction) remains as a premise. -/
theorem clean_transpose_reach {t : TT} {mt : VM.VM} {a b : List Pair}
    (hR : Reach t) (hf : cleanFwd t = some a) (hb : cleanBwd mt = some b)
    (hr : MtRepr t mt) (i j : Nat) : (j, i) ∈ b ↔ (i, j) ∈ a :=
  clean_transpose hf hb hr hR.allInv i j

end Pr3435.Clean
