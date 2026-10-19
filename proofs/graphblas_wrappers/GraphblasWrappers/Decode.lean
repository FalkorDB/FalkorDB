/-
# `Tensor::decode` and promotion completeness (`edge_count` underflow, #3033)

**Status on main 73077bb4a: FIXED by #3033.** #2996 added a size check of the
forward layers (below, `checkLayers_*`) and made a corrupt id blob an `Err`
(`Codes.decodeBlob_never_dies`). #3033 adds the promotion-completeness check:
every tensor-section pair must land on a `MULTI_EDGE` forward entry
(tensor.rs:1632-1636) and the number of `me` pairs with ids must equal the
number of markers, counted by `Matrix::<u64>::count_value_eq` (matrix.rs:1232,
`GrB_select` with `GrB_VALUEEQ_UINT64`) (tensor.rs:1617, 1665-1670). The section
"Promotion completeness on decode" below proves it refuses exactly the payloads
that break the invariant (`decodeCheck_ok_iff`) and accepts every encoded tensor
(`decodeCheck_accepts_encoded`). The text that follows is the pre-#3033 state;
its counterexamples are kept as `pre3033_*`.

`Tensor::decode` (tensor.rs:1553) rebuilds the inline forward matrix `m` from the
on-disk forward layers, then fills the multi-edge store `me` from the tensor
section (then tensor.rs:1609-1647). Before #3033 it never checked the
*promotion-completeness* invariant that ties them together:

> `eff_get (src,dst) = MULTI_EDGE`  ⟺  row `κ(src,dst)` of `me` is non-empty
> (tensor.rs module docs; `single_me` / `multi_me` in `proofs/versioned_matrix`).

A crafted payload can put a pair in the tensor section (giving it an `me` row)
without a matching `MULTI_EDGE` forward entry. Then `edge_count` (tensor.rs:1231),
which is *checked* precisely because a decoded blob can break an invariant,
computes `multi` (number of `me` rows) greater than the effective forward
pattern and the subtraction underflows — a deliberate `panic!`. Reached from the
very next `Tensor::encode` (BGSAVE / GRAPH.COPY), so a restored graph can no
longer be saved.

`edge_count`'s identity (tensor.rs:1231):
`|E| = |m| + |dp| − |dm| − |dp∩m| − multi + |me|`, each step checked.

Confirmed: `bug_tensor_decode_accepts_orphan_me_row_then_edge_count_panics`
(`graph/tests/lean_graphblas_wrappers.rs`) prints
`Tensor::edge_count: multi exceeds the effective pattern
(promotion-completeness is broken). |m|=0 |dp|=0 |dm|=0 |dp∩m|=0 multi=1 |me|=1`.
-/

namespace GBW

/-- Checked `u64` subtraction, matching `checked_sub().unwrap_or_else(panic)`
(tensor.rs:1265-1285): `none` = the panic. -/
def csub (x y : Nat) : Option Nat := if y ≤ x then some (x - y) else none

/-- Checked add — the two additions in `edge_count`. Overflow needs an `nvals`
that is not a real matrix size (`> u64`); modelled as always-`some` at the `Nat`
level, since the additions cannot underflow. -/
def cadd (x y : Nat) : Option Nat := some (x + y)

/-- `Tensor::edge_count` (tensor.rs:1231), the exact left-to-right checked
sequence. `some n` = returns `n`; `none` = panics (server crash). -/
def edgeCount (m dp dm shadow multi me : Nat) : Option Nat :=
  (cadd m dp).bind fun a =>
  (csub a dm).bind fun a =>
  (csub a shadow).bind fun a =>
  (csub a multi).bind fun a =>
  cadd a me

/-- The delta/promotion invariants `edge_count`'s doc-comment lists as bounding
each subtraction (tensor.rs:1231-1265):
`dm ⊆ m` ⇒ `dm ≤ m`; `dp ∩ dm = ∅` bounds the pattern so `shadow ≤ m+dp−dm`;
promotion-completeness ⇒ `multi ≤ effective pattern`. -/
structure Wf (m dp dm shadow multi me : Nat) : Prop where
  dm_le : dm ≤ m
  shadow_le : shadow ≤ m + dp - dm
  multi_le : multi ≤ m + dp - dm - shadow

/-- **No underflow while the invariant holds.** For a tensor whose layers
satisfy `Wf`, `edge_count` returns a value and never panics — the machine-checked
core of the `# Panics` doc-comment. -/
theorem edgeCount_no_panic (m dp dm shadow multi me : Nat) (h : Wf m dp dm shadow multi me) :
    (edgeCount m dp dm shadow multi me).isSome := by
  obtain ⟨h1, h2, h3⟩ := h
  unfold edgeCount cadd csub
  have e1 : dm ≤ m + dp := by omega
  simp only [Option.bind, if_pos e1]
  have e2 : shadow ≤ m + dp - dm := h2
  simp only [if_pos e2]
  have e3 : multi ≤ m + dp - dm - shadow := h3
  simp only [if_pos e3]
  rfl

/-- The value it returns is the edge-count identity. -/
theorem edgeCount_value (m dp dm shadow multi me : Nat) (h : Wf m dp dm shadow multi me) :
    edgeCount m dp dm shadow multi me = some (m + dp - dm - shadow - multi + me) := by
  obtain ⟨h1, h2, h3⟩ := h
  unfold edgeCount cadd csub
  have e1 : dm ≤ m + dp := by omega
  simp only [Option.bind, if_pos e1, if_pos h2, if_pos h3]

/-- **The decode gap (before #3033).** An orphan `me` row — a multi-edge pair the decoder
accepted with no forward entry — has `m = dp = dm = shadow = 0`, `multi = 1`,
`me = 1`. That violates `Wf.multi_le` (`1 ≤ 0` is false) and `edge_count`
panics. This is the exact state the confirmed repro prints. -/
theorem pre3033_orphan_row_breaks_wf : ¬ Wf 0 0 0 0 1 1 := by
  intro h
  have := h.multi_le
  omega

theorem pre3033_orphan_row_panics : edgeCount 0 0 0 0 1 1 = none := by decide

/-- More generally, any decoded tensor with more `me` rows than its effective
forward pattern panics — the class of the bug, not just the one witness. -/
theorem pre3033_more_me_rows_than_pattern_panics
    (m dp dm shadow multi me : Nat)
    (hdm : dm ≤ m + dp) (hsh : shadow ≤ m + dp - dm)
    (hbad : m + dp - dm - shadow < multi) :
    edgeCount m dp dm shadow multi me = none := by
  unfold edgeCount cadd csub
  simp only [Option.bind, if_pos hdm, if_pos hsh]
  have : ¬ multi ≤ m + dp - dm - shadow := by omega
  simp only [if_neg this]

/-! ## Forward-layer sizes (#2996, tensor.rs:1562-1573)

`Tensor::decode` merges the on-disk delta-plus / delta-minus layers into a base
built at the forward base's size. Since #2996 it refuses a payload whose
deltas are not exactly that size; before, a larger delta-plus was merged and
`m.set` wrote outside `m`. Dims are `(nrows, ncols)`. -/

abbrev Dims := Nat × Nat

def inDims (d : Dims) (e : Nat × Nat) : Prop := e.1 < d.1 ∧ e.2 < d.2

/-- tensor.rs:1562-1573 on main: `Err` unless both deltas have the base's dims. -/
def checkLayers (m dp dm : Dims) : Except String Unit :=
  if dp ≠ m then .error "Tensor decode: forward delta-plus size"
  else if dm ≠ m then .error "Tensor decode: forward delta-minus size"
  else .ok ()

/-- Before #2996: no check. -/
def pre2996_checkLayers (_m _dp _dm : Dims) : Except String Unit := .ok ()

theorem checkLayers_ok_iff (m dp dm : Dims) : checkLayers m dp dm = .ok () ↔ dp = m ∧ dm = m := by
  unfold checkLayers
  by_cases h1 : dp = m <;> by_cases h2 : dm = m <;> simp [h1, h2]

/-- **Main.** A decode that passes the check only merges delta entries that lie
inside the base: every stored delta entry is within its own layer's dims. -/
theorem checkLayers_merge_in_bounds (m dp dm : Dims) (h : checkLayers m dp dm = .ok ())
    (e : Nat × Nat) (he : inDims dp e) : inDims m e := by
  have := ((checkLayers_ok_iff m dp dm).1 h).1; subst this; exact he

/-- History: before #2996 a delta-plus entry outside the base was accepted
(main's test `decode_rejects_forward_layers_of_different_sizes`: base 4x4,
delta-plus 8x8). -/
theorem pre2996_accepts_out_of_bounds_delta :
    pre2996_checkLayers (4, 4) (8, 8) (4, 4) = .ok () ∧ inDims (8, 8) (5, 5) ∧ ¬ inDims (4, 4) (5, 5) := by
  refine ⟨rfl, ?_, ?_⟩ <;> simp [inDims]

theorem checkLayers_rejects_main_test : checkLayers (4, 4) (8, 8) (4, 4) ≠ .ok () := by
  rw [Ne, checkLayers_ok_iff]; simp

/-! ## Promotion completeness on decode (#3033, tensor.rs:1608-1670)

After the forward merge `Tensor::decode` waits `m` (:1610) and counts its
`MULTI_EDGE` markers with `Matrix::<u64>::count_value_eq` (:1617). Then every
tensor-section entry `(src, dst, ids)` (TM then TDP, one loop here) is refused
unless `src < nrows ∧ dst < ncols ∧ m.get(src,dst) = Some(MULTI_EDGE)`
(:1632-1636), and its ids are streamed into the pair's `me` row. At the end the
number of `me` pairs with ids (`multi_pairs_in`, versioned_matrix
`multiPairsIn_eq`) must equal the marker count (:1665-1670).

Model. `m` is the merged forward matrix as its stored entries (coordinates
distinct, the GraphBLAS matrix invariant `hk`). `me` rows are identified by
their pair: `compound_key` is a bijection (`compound_key_inverse`). A row exists
iff some entry for the pair carried a non-empty id blob (an empty blob streams
nothing, so `me` gets no row). -/

abbrev Pair := Nat × Nat

/-- `MULTI_EDGE` (tensor.rs:300) = `u64::MAX`. -/
def MULTI_EDGE : Nat := 2 ^ 64 - 1
/-- `MSB_MASK` (tensor.rs:1470): single inline ids have it clear. -/
def MSB : Nat := 2 ^ 63

/-- A `Matrix<u64>` as its stored entries. -/
abbrev MatU := List (Pair × Nat)

/-- `Matrix::<u64>::get` (`GrB_Matrix_extractElement_UINT64`; `None` on `NO_VALUE`). -/
def mget (m : MatU) (p : Pair) : Option Nat := List.lookup p m

/-- The C contract of `count_value_eq` (matrix.rs:1232): `GrB_Matrix_select_UINT64`
with `GrB_VALUEEQ_UINT64`, no mask/accum/descriptor, into a fresh matrix of the
same dims, keeps exactly the entries `A(i,j) == y` (C API spec, `GrB_select` with
the `VALUEEQ` index-unary op); `GrB_Matrix_nvals` is its entry count. A hypothesis
structure, not an axiom (as `Bulk.BulkSpec`). Its `GrB_Info` is
`debug_assert`ed (`Disposition.debugAssert`). -/
structure SelectValueEqSpec where
  select : MatU → Nat → MatU
  select_eq : ∀ A y, select A y = A.filter (fun e => e.2 == y)

/-- `Matrix::<u64>::count_value_eq` (matrix.rs:1232): select, then `nvals`. -/
def countValueEq (S : SelectValueEqSpec) (A : MatU) (y : Nat) : Nat := (S.select A y).length

theorem countValueEq_spec (S : SelectValueEqSpec) (A : MatU) (y : Nat) :
    countValueEq S A y = (A.filter (fun e => e.2 == y)).length := by
  simp [countValueEq, S.select_eq]

/-- The marker pairs of `m`. -/
def markerPairs (m : MatU) : List Pair := (m.filter (fun e => e.2 == MULTI_EDGE)).map Prod.fst

theorem countValueEq_markers (S : SelectValueEqSpec) (m : MatU) :
    countValueEq S m MULTI_EDGE = (markerPairs m).length := by
  simp [countValueEq_spec, markerPairs]

theorem mem_markerPairs (m : MatU) (p : Pair) : p ∈ markerPairs m ↔ (p, MULTI_EDGE) ∈ m := by
  simp [markerPairs]

theorem markerPairs_nodup (m : MatU) (hk : (m.map Prod.fst).Nodup) : (markerPairs m).Nodup :=
  List.Nodup.sublist ((List.filter_sublist).map Prod.fst) hk

/-- `get` on distinct coordinates returns the stored value. -/
theorem mget_eq_some_iff : ∀ (m : MatU) (p : Pair) (v : Nat), (m.map Prod.fst).Nodup →
    (mget m p = some v ↔ (p, v) ∈ m)
  | [], p, v, _ => by simp [mget]
  | (q, w) :: t, p, v, hk => by
    have hk' := List.nodup_cons.1 hk
    have ih := mget_eq_some_iff t p v hk'.2
    by_cases hpq : p = q
    · subst hpq
      have hnot : ∀ x, (p, x) ∉ t := fun x hx => hk'.1 (List.mem_map.2 ⟨(p, x), hx, rfl⟩)
      simp only [mget, List.lookup_cons, beq_self_eq_true, Option.some.injEq, List.mem_cons,
        Prod.mk.injEq, true_and]
      constructor
      · intro h; exact Or.inl h.symm
      · rintro (h | h)
        · exact h.symm
        · exact absurd h (hnot v)
    · have hb : (p == q) = false := by simpa using hpq
      simp only [mget, List.lookup_cons, hb, List.mem_cons, Prod.mk.injEq, hpq, false_and,
        false_or]
      exact ih

/-! ### The tensor-section loop -/

/-- One entry's effect on `me`: a row for `p` iff its id blob is non-empty
(`me[at].1.set(key, edge_id, true)` per streamed id). -/
def addRow (rows : List Pair) (p : Pair) (ids : List Nat) : List Pair :=
  if ids = [] then rows else if p ∈ rows then rows else p :: rows

/-- The `me` rows built from the tensor-section entries, with no checks — also
exactly what decode built before #3033. -/
def rowsFrom (rows : List Pair) (ts : List (Pair × List Nat)) : List Pair :=
  ts.foldl (fun r e => addRow r e.1 e.2) rows

/-- tensor.rs:1632: the per-entry check. -/
def entryOk (nrows ncols : Nat) (m : MatU) (p : Pair) : Bool :=
  decide (p.1 < nrows) && decide (p.2 < ncols) && (mget m p == some MULTI_EDGE)

/-- tensor.rs:1622-1662: the TM/TDP loop; `.error` = the early `return Err`. -/
def tensorLoop (nrows ncols : Nat) (m : MatU) :
    List (Pair × List Nat) → List Pair → Except String (List Pair)
  | [], rows => .ok rows
  | (p, ids) :: rest, rows =>
    if entryOk nrows ncols m p then tensorLoop nrows ncols m rest (addRow rows p ids)
    else .error "Tensor decode: edge ids for a pair which is not a multi-edge pair"

/-- `Tensor::decode` from the marker count to the final check (tensor.rs:1617-1670);
`.ok rows` = decode returns a tensor whose `me` rows are `rows`. -/
def decodeCheck (S : SelectValueEqSpec) (nrows ncols : Nat) (m : MatU)
    (ts : List (Pair × List Nat)) : Except String (List Pair) :=
  let markers := countValueEq S m MULTI_EDGE
  match tensorLoop nrows ncols m ts [] with
  | .error e => .error e
  | .ok rows =>
    if rows.length = markers then .ok rows
    else .error "Tensor decode: multi-edge pairs and pairs with edge ids differ"

/-- Before #3033: no check; the `me` rows are `rowsFrom [] ts`, whatever `m` is. -/
def pre3033_decodeCheck (_nrows _ncols : Nat) (_m : MatU)
    (ts : List (Pair × List Nat)) : Except String (List Pair) :=
  .ok (rowsFrom [] ts)

/-- The invariant: a pair is a forward marker iff it has an `me` row. -/
def PromotionComplete (m : MatU) (rows : List Pair) : Prop :=
  ∀ p, mget m p = some MULTI_EDGE ↔ p ∈ rows

theorem mem_addRow (rows : List Pair) (p q : Pair) (ids : List Nat) :
    q ∈ addRow rows p ids ↔ q ∈ rows ∨ (q = p ∧ ids ≠ []) := by
  unfold addRow
  by_cases h1 : ids = [] <;> by_cases h2 : p ∈ rows <;> simp [h1, h2]
  · intro h; subst h; exact h2
  · exact Or.comm

theorem nodup_addRow (rows : List Pair) (p : Pair) (ids : List Nat) (h : rows.Nodup) :
    (addRow rows p ids).Nodup := by
  unfold addRow
  by_cases h1 : ids = [] <;> by_cases h2 : p ∈ rows <;> simp [h1, h2, h]

theorem mem_rowsFrom : ∀ (ts : List (Pair × List Nat)) (rows : List Pair) (q : Pair),
    q ∈ rowsFrom rows ts ↔ q ∈ rows ∨ ∃ ids, (q, ids) ∈ ts ∧ ids ≠ []
  | [], rows, q => by simp [rowsFrom]
  | (p, ids) :: rest, rows, q => by
    have ih := mem_rowsFrom rest (addRow rows p ids) q
    simp only [rowsFrom, List.foldl_cons] at ih ⊢
    rw [ih, mem_addRow]
    constructor
    · rintro ((h | ⟨h1, h2⟩) | ⟨i, hi, hne⟩)
      · exact Or.inl h
      · subst h1; exact Or.inr ⟨ids, List.mem_cons_self, h2⟩
      · exact Or.inr ⟨i, List.mem_cons_of_mem _ hi, hne⟩
    · rintro (h | ⟨i, hi, hne⟩)
      · exact Or.inl (Or.inl h)
      · rcases List.mem_cons.1 hi with h | h
        · cases h; exact Or.inl (Or.inr ⟨rfl, hne⟩)
        · exact Or.inr ⟨i, h, hne⟩

theorem nodup_rowsFrom : ∀ (ts : List (Pair × List Nat)) (rows : List Pair),
    rows.Nodup → (rowsFrom rows ts).Nodup
  | [], _, h => h
  | (p, ids) :: rest, rows, h => nodup_rowsFrom rest _ (nodup_addRow rows p ids h)

theorem tensorLoop_ok_iff (nrows ncols : Nat) (m : MatU) :
    ∀ (ts : List (Pair × List Nat)) (rows r : List Pair),
    tensorLoop nrows ncols m ts rows = .ok r ↔
      (∀ e ∈ ts, entryOk nrows ncols m e.1 = true) ∧ r = rowsFrom rows ts
  | [], rows, r => by
    simp only [tensorLoop, Except.ok.injEq, List.not_mem_nil, false_implies, implies_true,
      true_and, rowsFrom, List.foldl_nil]
    exact ⟨fun h => h.symm, fun h => h.symm⟩
  | (p, ids) :: rest, rows, r => by
    by_cases hp : entryOk nrows ncols m p = true
    · simp only [tensorLoop, hp, ite_true]
      rw [tensorLoop_ok_iff nrows ncols m rest (addRow rows p ids) r]
      simp [rowsFrom, hp]
    · simp only [tensorLoop, hp]
      constructor
      · intro h; cases h
      · rintro ⟨h, -⟩; exact absurd (h (p, ids) List.mem_cons_self) hp

theorem tensorLoop_error_of_bad (nrows ncols : Nat) (m : MatU) (ts : List (Pair × List Nat))
    (rows : List Pair) (hbad : ∃ e ∈ ts, entryOk nrows ncols m e.1 = false) :
    ∃ err, tensorLoop nrows ncols m ts rows = .error err := by
  cases h : tensorLoop nrows ncols m ts rows with
  | error err => exact ⟨err, rfl⟩
  | ok r =>
    obtain ⟨e, he, hf⟩ := hbad
    have := ((tensorLoop_ok_iff nrows ncols m ts rows r).1 h).1 e he
    rw [hf] at this; cases this

/-! ### Counting distinct pairs -/

theorem len_le_of_nodup_subset : ∀ {l1 l2 : List Pair}, l1.Nodup → (∀ x ∈ l1, x ∈ l2) →
    l1.length ≤ l2.length
  | [], _, _, _ => Nat.zero_le _
  | a :: t, l2, hn, hs => by
    have ha : a ∈ l2 := hs a List.mem_cons_self
    have hn' := List.nodup_cons.1 hn
    have hsub : ∀ x ∈ t, x ∈ l2.erase a := fun x hx =>
      (List.mem_erase_of_ne (fun hxa => hn'.1 (by subst hxa; exact hx))).2 (hs x (List.mem_cons_of_mem _ hx))
    have ih := len_le_of_nodup_subset hn'.2 hsub
    rw [List.length_erase_of_mem ha] at ih
    have := List.length_pos_of_mem ha
    simp only [List.length_cons]; omega

/-- Equal sizes: a duplicate-free subset of a list covers it. -/
theorem cover_of_len_eq {l1 l2 : List Pair} (hn : l1.Nodup) (hs : ∀ x ∈ l1, x ∈ l2)
    (hl : l1.length = l2.length) : ∀ x ∈ l2, x ∈ l1 := by
  intro b hb
  refine Classical.byContradiction fun hnb => ?_
  have hsub : ∀ x ∈ l1, x ∈ l2.erase b := fun x hx =>
    (List.mem_erase_of_ne (fun hxb => hnb (by subst hxb; exact hx))).2 (hs x hx)
  have := len_le_of_nodup_subset hn hsub
  rw [List.length_erase_of_mem hb] at this
  have := List.length_pos_of_mem hb
  omega

/-! ### Main theorems -/

/-- **#3033, exact characterisation.** `Tensor::decode` accepts iff every tensor
entry is an in-bounds marker pair and the `me` rows it builds are promotion
complete. -/
theorem decodeCheck_ok_iff (S : SelectValueEqSpec) (nrows ncols : Nat) (m : MatU)
    (ts : List (Pair × List Nat)) (hk : (m.map Prod.fst).Nodup) :
    (∃ rows, decodeCheck S nrows ncols m ts = .ok rows) ↔
      (∀ e ∈ ts, entryOk nrows ncols m e.1 = true) ∧ PromotionComplete m (rowsFrom [] ts) := by
  have hnd : (rowsFrom [] ts).Nodup := nodup_rowsFrom ts [] List.nodup_nil
  have hmk := markerPairs_nodup m hk
  -- every row is a marker once the per-entry check passed
  have hsub : (∀ e ∈ ts, entryOk nrows ncols m e.1 = true) →
      ∀ x ∈ rowsFrom [] ts, x ∈ markerPairs m := by
    intro hall x hx
    rcases (mem_rowsFrom ts [] x).1 hx with h | ⟨ids, hi, -⟩
    · cases h
    · have := hall (x, ids) hi
      simp only [entryOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at this
      exact (mem_markerPairs m x).2 ((mget_eq_some_iff m x _ hk).1 this.2)
  constructor
  · rintro ⟨rows, h⟩
    unfold decodeCheck at h
    cases hl : tensorLoop nrows ncols m ts [] with
    | error e => rw [hl] at h; cases h
    | ok r =>
      rw [hl] at h
      obtain ⟨hall, hr⟩ := (tensorLoop_ok_iff nrows ncols m ts [] r).1 hl
      subst hr
      by_cases hlen : (rowsFrom [] ts).length = countValueEq S m MULTI_EDGE
      · refine ⟨hall, fun p => ⟨fun hp => ?_, fun hp => ?_⟩⟩
        · rw [countValueEq_markers] at hlen
          exact cover_of_len_eq hnd (hsub hall) hlen p
            ((mem_markerPairs m p).2 ((mget_eq_some_iff m p _ hk).1 hp))
        · exact (mget_eq_some_iff m p _ hk).2 ((mem_markerPairs m p).1 (hsub hall p hp))
      · simp only [hlen, ite_false] at h; cases h
  · rintro ⟨hall, hpc⟩
    refine ⟨rowsFrom [] ts, ?_⟩
    have hl := (tensorLoop_ok_iff nrows ncols m ts [] (rowsFrom [] ts)).2 ⟨hall, rfl⟩
    have h1 := len_le_of_nodup_subset hnd (hsub hall)
    have h2 := len_le_of_nodup_subset hmk (fun x hx =>
      (hpc x).1 ((mget_eq_some_iff m x _ hk).2 ((mem_markerPairs m x).1 hx)))
    have hlen : (rowsFrom [] ts).length = countValueEq S m MULTI_EDGE := by
      rw [countValueEq_markers]; omega
    unfold decodeCheck
    simp only [hl, hlen, ite_true]

/-- **Main (#3033).** Every input whose `me` rows would break promotion
completeness is refused. -/
theorem decodeCheck_refuses_incomplete (S : SelectValueEqSpec) (nrows ncols : Nat) (m : MatU)
    (ts : List (Pair × List Nat)) (hk : (m.map Prod.fst).Nodup)
    (hbad : ¬ PromotionComplete m (rowsFrom [] ts)) :
    ∃ err, decodeCheck S nrows ncols m ts = .error err := by
  cases h : decodeCheck S nrows ncols m ts with
  | error err => exact ⟨err, rfl⟩
  | ok rows => exact absurd ((decodeCheck_ok_iff S nrows ncols m ts hk).1 ⟨rows, h⟩).2 hbad

/-- An accepted decode yields exactly the unchecked rows, and they are complete. -/
theorem decodeCheck_ok_complete (S : SelectValueEqSpec) (nrows ncols : Nat) (m : MatU)
    (ts : List (Pair × List Nat)) (hk : (m.map Prod.fst).Nodup) (rows : List Pair)
    (h : decodeCheck S nrows ncols m ts = .ok rows) :
    rows = rowsFrom [] ts ∧ PromotionComplete m rows := by
  have hiff := (decodeCheck_ok_iff S nrows ncols m ts hk).1 ⟨rows, h⟩
  have hr : rows = rowsFrom [] ts := by
    unfold decodeCheck at h
    have hl := (tensorLoop_ok_iff nrows ncols m ts [] (rowsFrom [] ts)).2 ⟨hiff.1, rfl⟩
    rw [hl] at h
    by_cases hlen : (rowsFrom [] ts).length = countValueEq S m MULTI_EDGE
    · simp only [hlen, ite_true, Except.ok.injEq] at h; exact h.symm
    · simp only [hlen, ite_false] at h; cases h
  subst hr; exact ⟨rfl, hiff.2⟩

/-- **The bug is gone.** A decoded tensor (`dp = dm = ∅`, so `shadow = 0`) has
`multi = rows.length ≤ |m|`, so `edge_count` cannot panic, whatever `|me|`. -/
theorem decodeCheck_ok_edgeCount_no_panic (S : SelectValueEqSpec) (nrows ncols : Nat)
    (m : MatU) (ts : List (Pair × List Nat)) (rows : List Pair)
    (h : decodeCheck S nrows ncols m ts = .ok rows) (me : Nat) :
    (edgeCount m.length 0 0 0 rows.length me).isSome := by
  have hlen : rows.length = countValueEq S m MULTI_EDGE := by
    unfold decodeCheck at h
    cases hl : tensorLoop nrows ncols m ts [] with
    | error e => rw [hl] at h; cases h
    | ok r =>
      rw [hl] at h
      by_cases hc : r.length = countValueEq S m MULTI_EDGE
      · simp only [hc, ite_true, Except.ok.injEq] at h; subst h; exact hc
      · simp only [hc, ite_false] at h; cases h
  have hle : countValueEq S m MULTI_EDGE ≤ m.length := by
    rw [countValueEq_spec]; exact List.length_filter_le _ _
  exact edgeCount_no_panic _ _ _ _ _ _ ⟨Nat.zero_le _, by omega, by omega⟩

/-! ### Every encoded tensor is accepted -/

/-- A stored pair as `encode` sees it (deltas folded): one inline id, or the ids
of a multi-edge pair. -/
inductive Slot where
  | single (id : Nat)
  | multi (ids : List Nat)

/-- A committed tensor: its pairs, distinct and in bounds; single ids have the MSB
clear (they are written verbatim and read back by `value & MSB_MASK == 0`), and a
multi-edge pair has ids (`PairPlan`: at least two). -/
structure TState where
  nrows : Nat
  ncols : Nat
  pairs : List (Pair × Slot)

def TState.Wf (t : TState) : Prop :=
  (t.pairs.map Prod.fst).Nodup ∧
  ∀ e ∈ t.pairs, e.1.1 < t.nrows ∧ e.1.2 < t.ncols ∧
    (match e.2 with
     | .single id => id < MSB
     | .multi ids => ids ≠ [])

def slotVal : Slot → Nat
  | .single id => id
  | .multi _ => MULTI_EDGE

/-- The forward matrix decode rebuilds from `encode`'s forward layer: a single id
verbatim, `count | MSB` mapped back to `MULTI_EDGE` (tensor.rs:1595-1607). -/
def encFwd (t : TState) : MatU := t.pairs.map fun e => (e.1, slotVal e.2)

def slotIds : Pair × Slot → Option (Pair × List Nat)
  | (_, .single _) => none
  | (p, .multi ids) => some (p, ids)

/-- The tensor section `encode` writes: one `(src, dst, ids)` per multi-edge pair. -/
def encTs (t : TState) : List (Pair × List Nat) := t.pairs.filterMap slotIds

theorem mem_encFwd (t : TState) (p : Pair) (v : Nat) :
    (p, v) ∈ encFwd t ↔ ∃ s, (p, s) ∈ t.pairs ∧ slotVal s = v := by
  simp only [encFwd, List.mem_map]
  constructor
  · rintro ⟨⟨q, s⟩, hm, he⟩
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl⟩ := he
    exact ⟨s, hm, rfl⟩
  · rintro ⟨s, hm, rfl⟩
    exact ⟨(p, s), hm, rfl⟩

theorem mem_encTs (t : TState) (p : Pair) (ids : List Nat) :
    (p, ids) ∈ encTs t ↔ (p, Slot.multi ids) ∈ t.pairs := by
  simp only [encTs, List.mem_filterMap]
  constructor
  · rintro ⟨⟨q, s⟩, hm, hs⟩
    cases s with
    | single _ => simp [slotIds] at hs
    | multi l => simp only [slotIds, Option.some.injEq, Prod.mk.injEq] at hs
                 obtain ⟨rfl, rfl⟩ := hs; exact hm
  · intro h; exact ⟨_, h, rfl⟩

/-- **Main (#3033), no false refusal.** `decode` accepts every tensor `encode`
writes. -/
theorem decodeCheck_accepts_encoded (S : SelectValueEqSpec) (t : TState) (h : t.Wf) :
    ∃ rows, decodeCheck S t.nrows t.ncols (encFwd t) (encTs t) = .ok rows := by
  obtain ⟨hk0, hwf⟩ := h
  have hk : ((encFwd t).map Prod.fst).Nodup := by
    simpa [encFwd, List.map_map, Function.comp_def] using hk0
  have hmulti : ∀ p, (p, MULTI_EDGE) ∈ encFwd t ↔ ∃ ids, (p, Slot.multi ids) ∈ t.pairs := by
    intro p
    rw [mem_encFwd]
    constructor
    · rintro ⟨s, hs, hv⟩
      cases s with
      | multi ids => exact ⟨ids, hs⟩
      | single id =>
        have := (hwf _ hs).2.2
        simp only [slotVal] at hv this
        simp [MSB, MULTI_EDGE] at this hv; omega
    · rintro ⟨ids, hs⟩; exact ⟨_, hs, rfl⟩
  refine (decodeCheck_ok_iff S _ _ _ _ hk).2 ⟨?_, ?_⟩
  · rintro ⟨p, ids⟩ he
    have hs := (mem_encTs t p ids).1 he
    have hb := hwf _ hs
    simp only [entryOk, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨hb.1, hb.2.1⟩, (mget_eq_some_iff _ p _ hk).2 ((hmulti p).2 ⟨ids, hs⟩)⟩
  · intro p
    rw [mget_eq_some_iff _ p _ hk, hmulti, mem_rowsFrom]
    constructor
    · rintro ⟨ids, hs⟩
      exact Or.inr ⟨ids, (mem_encTs t p ids).2 hs, (hwf _ hs).2.2⟩
    · rintro (h | ⟨ids, hi, -⟩)
      · cases h
      · exact ⟨ids, (mem_encTs t p ids).1 hi⟩

/-! ### main's tests, and the pre-#3033 counterexample -/

/-- `decode_rejects_edge_ids_for_a_pair_that_is_not_multi_edge`, absent pair
(7, 9) — the orphan row of `pre3033_orphan_row_panics`. -/
theorem orphan_row_refused (S : SelectValueEqSpec) :
    ∃ err, decodeCheck S 16 16 [((1, 2), 5)] [((7, 9), [3])] = .error err :=
  ⟨_, rfl⟩

/-- Same test, single-edge pair (1, 2). -/
theorem single_pair_ids_refused (S : SelectValueEqSpec) :
    ∃ err, decodeCheck S 16 16 [((1, 2), 5)] [((1, 2), [3])] = .error err :=
  ⟨_, rfl⟩

/-- `decode_rejects_a_multi_edge_pair_without_edge_ids`: two markers, ids for one. -/
theorem marker_without_ids_refused (S : SelectValueEqSpec) :
    ∃ err, decodeCheck S 16 16 [((1, 2), MULTI_EDGE), ((3, 4), MULTI_EDGE)]
      [((1, 2), [10, 11])] = .error err := by
  refine ⟨"Tensor decode: multi-edge pairs and pairs with edge ids differ", ?_⟩
  have : countValueEq S [((1, 2), MULTI_EDGE), ((3, 4), MULTI_EDGE)] MULTI_EDGE = 2 := by
    rw [countValueEq_spec]; rfl
  unfold decodeCheck
  rw [this]
  rfl

/-- History: before #3033 the orphan payload was accepted, with a row that breaks
promotion completeness (then `pre3033_orphan_row_panics`). -/
theorem pre3033_orphan_row_accepted :
    pre3033_decodeCheck 16 16 [((1, 2), 5)] [((7, 9), [3])] = .ok [(7, 9)] ∧
    ¬ PromotionComplete [((1, 2), 5)] [(7, 9)] := by
  refine ⟨rfl, fun h => ?_⟩
  have := (h (7, 9)).2 (List.mem_singleton.2 rfl)
  revert this; decide

end GBW
