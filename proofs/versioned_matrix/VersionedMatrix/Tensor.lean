/-
# `Tensor`: the inline / `MULTI_EDGE` encoding of multi-edges

Model of `graph/src/graph/graphblas/tensor.rs`, one `(src, dst)` pair at a
time. Pairs are independent: every per-pair decision in
`set_all_from_slices` / `remove_all` reads and writes only that pair's
forward entry, its `mt` mirror and its `me` row (`compound_key` is injective,
`compoundKey_inv` below), and a batch's `FxHashMap` groups a pair's edges in
input order, which is what the per-pair functions here consume.

`me` and `mt` are `VersionedMatrix<bool>`s, whose logical contents behave as
plain sets by `VM.run_repr` (`DeltaProofs.lean`), so here they *are* their
logical contents: the list of ids in the pair's `me` row (set semantics,
`meIns`), and one bit.

| here | there (`tensor.rs`) |
| --- | --- |
| `Val`          | the UINT64 forward value: an edge id, or `MULTI_EDGE = u64::MAX` (:300) |
| `PairSt`       | one pair's `m`/`dp`/`dm` entries (:272-276), `mt` bit (:280), `me` row (:291) |
| `effV`         | `eff_get` (:442): `dp` wins, else `m` unless `dm` |
| `edges`        | `get` (:463) — the inline id, or the `me` row |
| `vacant`/`occupied`/`writeEntry`/`addBatch` | `set_all_from_slices` (:497): read phase (:536-596), write phase (:603-621) |
| `initPlan`/`planStep`/`writePlan`/`removeBatch` | `remove_all` slow path (:672-806) |
| `removeFast`   | `remove_all` fast path (:642-670), taken when no `me` row exists anywhere |
| `foldP`        | `Tensor::flush` fold (:880-931), eWiseAdd `SECOND` so `dp` wins |
| `countTerms`   | the pair's terms in `edge_count` (:1221-1275) |
| `compoundKey`  | `compound_key` (:222); `compoundKeyInv` = `compound_key_inverse` (:241) |
-/

namespace VMTensor

inductive Val where
  | id (n : Nat)
  | multi
deriving DecidableEq, Repr

structure PairSt where
  m : Option Val
  dp : Option Val
  dm : Bool
  mt : Bool
  me : List Nat
deriving DecidableEq, Repr

/-- `eff_get` (:442). -/
def effV (s : PairSt) : Option Val :=
  match s.dp with
  | some v => some v
  | none => if s.dm then none else s.m

/-- `get` (:463): the pair's edge ids. -/
def edges (s : PairSt) : List Nat :=
  match effV s with
  | some (.id i) => [i]
  | some .multi => s.me
  | none => []

/-- `me.set(key, id, true)`: set semantics on the row. -/
def meIns (me : List Nat) (i : Nat) : List Nat := if i ∈ me then me else me ++ [i]

/-- The module-doc invariants (:50-70). -/
structure Inv (s : PairSt) : Prop where
  /-- `dm ⊆ m` -/
  dm_m : s.dm = true → s.m.isSome = true
  /-- `dp ∩ dm = ∅`: `dm` marks pure deletions only -/
  dp_dm : s.dp.isSome = true → s.dm = false
  /-- cancel-to-clean: `dp` never repeats the committed value -/
  cancel : ∀ v, s.dp = some v → s.m ≠ some v
  /-- promotion completeness, both directions -/
  multi_me : effV s = some .multi → 2 ≤ s.me.length
  single_me : effV s ≠ some .multi → s.me = []
  me_nd : s.me.Nodup
  /-- `mt` mirrors the effective forward structure -/
  mt_eff : s.mt = (effV s).isSome

/-! ### Add (`set_all_from_slices`) -/

/-- The write-phase entry the read phase queues for the pair
(`m_ids[idx]`, `m_masked[idx]`), if any. -/
structure Entry where
  val : Val
  masked : Option Val

/-- Read phase for the pair's first edge in the batch (`Entry::Vacant`,
:553-594). Returns the `me` row, the queued entry, and whether the pair is
known multi (`batch` slot = `usize::MAX`). -/
def vacant (s : PairSt) (i : Nat) : List Nat × Option Entry × Bool :=
  let cur := match s.dp with
    | some v => some v
    | none => if s.dm then none else s.m
  match cur with
  | some .multi => (meIns s.me i, none, true)
  | some (.id c) =>
    (meIns (meIns s.me c) i, some ⟨.multi, if s.dp.isSome then s.m else none⟩, true)
  | none => (s.me, some ⟨.id i, if s.dm then s.m else none⟩, false)

/-- Read phase for a later edge of the same pair (`Entry::Occupied`,
:539-552): the first occupied hit promotes the pending inline slot. -/
def occupied (st : List Nat × Option Entry × Bool) (i : Nat) : List Nat × Option Entry × Bool :=
  match st with
  | (me, some ⟨.id first, msk⟩, false) => (meIns (meIns me first) i, some ⟨.multi, msk⟩, true)
  | (me, e, k) => (meIns me i, e, k)

/-- Write phase (:603-621). -/
def writeEntry (s : PairSt) (me : List Nat) : Option Entry → PairSt
  | none => { s with me := me }
  | some e =>
    match e.masked with
    | some committed =>
      if committed = e.val then { s with me := me, mt := true, dm := false, dp := none }
      else { s with me := me, mt := true, dm := false, dp := some e.val }
    | none => { s with me := me, mt := true, dp := some e.val }

def addBatch (s : PairSt) : List Nat → PairSt
  | [] => s
  | i :: is =>
    let st := is.foldl occupied (vacant s i)
    writeEntry s st.1 st.2.1

/-! ### Remove (`remove_all`, slow path) -/

inductive Plan where
  | multi (ids : List Nat)
  | single (id : Nat) (demoted : Bool)
  | emptied
  | absent

/-- The initial plan, from the pair's effective state (:700-720). -/
def initPlan (s : PairSt) : Plan :=
  match effV s with
  | some .multi => .multi s.me
  | some (.id i) => .single i false
  | none => .absent

/-- One removal replayed against the plan (:722-763), accumulating the `me`
entries to drop. `ids.len() == 0` is `unreachable!` in Rust. -/
def planStep (p : Plan × List Nat) (id : Nat) : Plan × List Nat :=
  match p with
  | (.multi ids, del) =>
    if id ∈ ids then
      match ids.erase id with
      | [last] => (.single last true, del ++ [id, last])
      | ids' => (.multi ids', del ++ [id])
    else (.multi ids, del)
  | (.single inline d, del) => if inline = id then (.emptied, del) else (.single inline d, del)
  | other => other

def dropMe (me del : List Nat) : List Nat := me.filter (fun x => !(decide (x ∈ del)))

/-- Write phase (:773-805). -/
def writePlan (s : PairSt) (del : List Nat) : Plan → PairSt
  | .emptied =>
    { s with me := dropMe s.me del, dp := none,
             dm := if s.m.isSome then true else s.dm, mt := false }
  | .single id true =>
    if s.m = some (.id id) then { s with me := dropMe s.me del, dp := none }
    else { s with me := dropMe s.me del, dp := some (.id id) }
  | _ => { s with me := dropMe s.me del }

def removeBatch (s : PairSt) (ids : List Nat) : PairSt :=
  let r := ids.foldl planStep (initPlan s, [])
  writePlan s r.2 r.1

/-- Fast path (:642-670): `dm<mask> = mask ∩ m`, `dp &= ¬mask`,
`mt.remove_mask`. Taken only when *no* pair has an `me` row. -/
def removeFast (s : PairSt) : PairSt :=
  { s with dm := s.dm || s.m.isSome, dp := none, mt := false }

/-! ### Fold (`Tensor::flush`) -/

def dpOr (s : PairSt) : Option Val :=
  match s.dp with
  | some v => some v
  | none => s.m

def foldP (s : PairSt) (a b : Bool) : PairSt :=
  match a, b with
  | true, true => { s with m := if s.dm then none else dpOr s, dp := none, dm := false }
  | true, false => { s with m := dpOr s, dp := none }
  | false, true => { s with m := if s.dm then none else s.m, dm := false }
  | false, false => s

/-! ### `edge_count` -/

def b2i (b : Bool) : Int := if b then 1 else 0

/-- The running totals of `edge_count` (:1248-1274) restricted to one pair:
`|m| + |dp|`, `− |dm|`, `− |dp ∩ m|`, `− multi`, `+ |me|`. -/
def partials (s : PairSt) : List Int :=
  let a := b2i s.m.isSome + b2i s.dp.isSome
  let b := a - b2i s.dm
  let c := b - b2i (s.dp.isSome && s.m.isSome)
  let d := c - b2i (decide (effV s = some .multi))
  [a, b, c, d, d + s.me.length]

/-! ### `compound_key` -/

def B : Nat := 2 ^ 30

/-- `compound_key` (:222), on values `< 2^64` (no `u64` shift overflows:
`(src & MASK) << 30 < 2^60`). `x & (2^30 - 1) = x % 2^30`, `<< 30 = * 2^30`,
`>> 30 = / 2^30`, and `|` of disjoint bit ranges is `+`
(`Nat.shiftLeft_add_eq_or_of_lt`). -/
def compoundKey (s d : Nat) : (Nat × Nat) × Nat :=
  ((s / B, d / B), (s % B) * B + d % B)

/-- `compound_key_inverse` (:241). -/
def compoundKeyInv (blk : Nat × Nat) (row : Nat) : Nat × Nat :=
  (blk.1 * B + row / B, blk.2 * B + row % B)

end VMTensor
