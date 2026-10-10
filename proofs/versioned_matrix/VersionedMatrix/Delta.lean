/-
# `VersionedMatrix<bool>`: the three-layer delta structure

Model of `graph/src/graph/graphblas/versioned_matrix.rs` — the logical
matrix `(m ∖ dm) ∪ dp` and every mutation that touches its layers.

## GraphBLAS, abstracted

A `Matrix<bool>` is modelled as the *set of its stored coordinates*, a
`List Pair` read with set semantics (`p ∈ l`). Bool layers only ever store
`true` (`Delta::insert` writes `true`, `tombstone_masked` writes
`ANY_PAIR = true`), so pattern = contents and no value is lost. The only
GraphBLAS primitives used are:

| primitive | modelled as | Rust |
| --- | --- | --- |
| `GrB_Matrix_setElement(true)` | `ins`  | `Matrix::set` |
| `GrB_Matrix_removeElement`    | `del`  | `Matrix::remove` |
| `GrB_Matrix_extractElement`   | `p ∈ l` | `Matrix::get` |
| `eWiseAdd` (LOR), `<!dm,replace>` | union / set difference | `element_wise_add`, `select` |
| `eWiseMult(ANY_PAIR)<mask>`   | `mask ∩ m`, unioned in (no REPLACE: entries outside the mask are kept) | `tombstone_masked` |
| `GrB_transpose(C<!M,replace>, A, T0)` | `A ∖ M` | `remove_all` |
| `GrB_Matrix_resize`           | restriction to the new bounds | `resize` |
| `GrB_Matrix_nvals`            | `length` of a duplicate-free list | `nvals` |

No `axiom` is needed: each primitive is a *definition* whose meaning is the
one the GraphBLAS spec gives it; the justification is the table above.

Pending tuples / `wait` / `resync` / the approximate counters / the row
filter never change *which* coordinates a layer stores, so they are not in
the model (see REPORT.md, "modelling gaps"). The fold *policy*
(`should_fold`, `should_fold_read`, `delta_dominates_base`) is performance
only: every theorem below quantifies over **all** fold decisions.

## What is modelled, and where it lives

| here | there (`graph/src/graph/graphblas/versioned_matrix.rs`) |
| --- | --- |
| `VM`            | `struct VersionedMatrix` (:645) — `m`, `dp`, `dm`, `needs_flush` + latched `Delta::fold` |
| `VM.armed`      | `needs_flush` (:656) together with the two latched `fold` flags (:321) |
| `eff`           | `Iter` / `extract` (:774): `(m ∖ dm) ∪ dp` |
| `get`           | `VersionedMatrix::get` (:1009) |
| `fold`          | the body of `flush` (:1082-1127) for a given `(fold_dp, fold_dm)` |
| `flush`         | `VersionedMatrix::flush` (:1082) |
| `set`           | `VersionedMatrix::set` (:1034) |
| `remove`        | `VersionedMatrix::remove` (:970) |
| `setAll`        | `VersionedMatrix::set_all::<NEW>` (:1242) |
| `setProduct`    | `VersionedMatrix::set_product::<NEW>` (:1193) |
| `removeMask`    | `VersionedMatrix::remove_mask` (:989) |
| `grow`          | `VersionedMatrix::resize`, grow branch (:866-945) |
| `shrink`        | `VersionedMatrix::resize`, shrink branch (:851-864) |
| `dup`           | `Dup for VersionedMatrix::dup` (:1287) |
| `foldNow`       | `fold_latched` (:1138) / `fold_oversized` (:1162) |
| `Op.wait`       | `wait` (:710) — latches only, never arms, never moves an entry |
-/

namespace VM

abbrev Pair := Nat × Nat
abbrev Layer := List Pair

/-- `GrB_Matrix_setElement(true)`: set semantics, so no duplicate is created. -/
def ins (s : Layer) (p : Pair) : Layer := if p ∈ s then s else p :: s
/-- `GrB_Matrix_removeElement`. -/
def del (s : Layer) (p : Pair) : Layer := s.filter (fun q => q != p)

@[simp] theorem mem_ins {s : Layer} {p q : Pair} : q ∈ ins s p ↔ q = p ∨ q ∈ s := by
  unfold ins; split <;> simp_all

@[simp] theorem mem_del {s : Layer} {p q : Pair} : q ∈ del s p ↔ q ≠ p ∧ q ∈ s := by
  unfold del; simp [List.mem_filter]; exact And.comm

theorem nodup_ins {s : Layer} (h : s.Nodup) (p : Pair) : (ins s p).Nodup := by
  unfold ins; split
  · exact h
  · exact List.nodup_cons.2 ⟨by assumption, h⟩

theorem nodup_filter {s : Layer} (h : s.Nodup) (f : Pair → Bool) : (s.filter f).Nodup :=
  List.Pairwise.filter f h

theorem nodup_del {s : Layer} (h : s.Nodup) (p : Pair) : (del s p).Nodup := nodup_filter h _

structure VM where
  m : Layer
  dp : Layer
  dm : Layer
  nrows : Nat
  ncols : Nat
  /-- `needs_flush` with the fold decisions it will execute (`None` = not armed). -/
  armed : Option (Bool × Bool)

/-- The logical matrix: what `Iter` and `extract` produce. -/
def eff (v : VM) (p : Pair) : Prop := (p ∈ v.m ∧ p ∉ v.dm) ∨ p ∈ v.dp

/-- `VersionedMatrix::get` (:1009): probes `m` first and only then a delta. -/
def get (v : VM) (p : Pair) : Bool :=
  if p ∈ v.m then !(decide (p ∈ v.dm)) else decide (p ∈ v.dp)

def inBounds (r c : Nat) (p : Pair) : Prop := p.1 < r ∧ p.2 < c

instance (r c : Nat) (p : Pair) : Decidable (inBounds r c p) := by
  unfold inBounds; infer_instance

/-- The layer invariants `set`/`remove` rely on (doc comments :789-792, :963-969). -/
structure Inv (v : VM) : Prop where
  dp_m   : ∀ p, p ∈ v.dp → p ∉ v.m
  dm_m   : ∀ p, p ∈ v.dm → p ∈ v.m
  nd_m   : v.m.Nodup
  nd_dp  : v.dp.Nodup
  nd_dm  : v.dm.Nodup
  bm     : ∀ p, p ∈ v.m → inBounds v.nrows v.ncols p
  bdp    : ∀ p, p ∈ v.dp → inBounds v.nrows v.ncols p

theorem Inv.dp_dm {v : VM} (h : Inv v) : ∀ p, p ∈ v.dp → p ∉ v.dm :=
  fun p hp hm => h.dp_m p hp (h.dm_m p hm)

/-- The empty matrix `VersionedMatrix::new` (:1050). -/
def empty (r c : Nat) : VM := ⟨[], [], [], r, c, none⟩

theorem inv_empty (r c : Nat) : Inv (empty r c) := by
  constructor <;> simp [empty]

/-! ## Flush / fold -/

/-- One fold with decisions `(fdp, fdm)`: the `match (fold_dp, fold_dm)` of
`flush` (:1100-1116) followed by the `clear`s (:1119-1124). -/
def fold (v : VM) (fdp fdm : Bool) : VM :=
  { v with
    m := match fdp, fdm with
      | true,  true  => (v.m ++ v.dp).filter (fun p => !(decide (p ∈ v.dm)))   -- new_m<!dm,replace> = m ∪ dp
      | true,  false => v.m ++ v.dp                                            -- m ∪ dp  (dp ∩ m = ∅)
      | false, true  => v.m.filter (fun p => !(decide (p ∈ v.dm)))            -- select(!dm, m)
      | false, false => v.m
    dp := if fdp then [] else v.dp
    dm := if fdm then [] else v.dm }

/-- `flush` (:1082): runs the armed fold, if any, and disarms. -/
def flush (v : VM) : VM :=
  match v.armed with
  | none => v
  | some (a, b) => { fold v a b with armed := none }

/-! ## Mutations -/

/-- `set` (:1034). Reads only the committed base. -/
def set (v : VM) (p : Pair) : VM :=
  let v := flush v
  if p ∈ v.m then { v with dm := del v.dm p } else { v with dp := ins v.dp p }

/-- `remove` (:970). Reads only the committed base. -/
def remove (v : VM) (p : Pair) : VM :=
  let v := flush v
  if p ∈ v.m then { v with dm := ins v.dm p } else { v with dp := del v.dp p }

/-- The `dm`-empty loop of `set_all` (:1255-1265). `NEW` skips the `m` probe. -/
def setAllFast (NEW : Bool) (v : VM) : List Pair → VM
  | [] => v
  | p :: ps =>
    if !NEW && decide (p ∈ v.m) then setAllFast NEW v ps
    else setAllFast NEW { v with dp := ins v.dp p } ps

/-- Per-entry `set`, the `dm`-non-empty branch of `set_all` (:1267-1269). -/
def setEach (v : VM) : List Pair → VM
  | [] => v
  | p :: ps => setEach (set v p) ps

/-- `set_all::<NEW>` (:1242). -/
def setAll (NEW : Bool) (v : VM) (es : List Pair) : VM :=
  let v := flush v
  if v.dm = [] then setAllFast NEW v es else setEach v es

def product (rows cols : List Nat) : List Pair :=
  rows.flatMap (fun i => cols.map (fun j => (i, j)))

/-- `set_product::<NEW>` (:1193). The fast path is one `GrB_assign` of the
product into `dp` (`Delta::insert_product`, :574) — the same layer effect as
inserting every product pair. -/
def setProduct (NEW : Bool) (v : VM) (rows cols : List Nat) : VM :=
  if rows = [] ∨ cols = [] then v else
  let v := flush v
  if !NEW || v.dm ≠ [] then setAll NEW v (product rows cols)
  else (product rows cols).foldl (fun w p => { w with dp := ins w.dp p }) v

/-- `remove_mask` (:989): `dm<mask> = mask ∩ m` (existing tombstones kept),
then `dp &= ¬mask`. -/
def removeMask (v : VM) (mask : List Pair) : VM :=
  let v := flush v
  { v with
    dm := v.dm ++ (v.m.filter (fun p => decide (p ∈ mask) && !(decide (p ∈ v.dm))))
    dp := v.dp.filter (fun p => !(decide (p ∈ mask))) }

/-- Grow branch of `resize` (:866-945): with both deltas empty, only the
dimensions change (`grown` + `clear_deltas`); otherwise the merge
`(m ∖ dm) ∪ dp` is streamed into a fresh base and both deltas are cleared.
Either way `needs_flush` is cleared (`clear_deltas`, :958). -/
def grow (v : VM) (r c : Nat) : VM :=
  if v.dp = [] ∧ v.dm = [] then { v with nrows := r, ncols := c, armed := none }
  else
    { m := v.m.filter (fun p => !(decide (p ∈ v.dm))) ++ v.dp
      dp := [], dm := [], nrows := r, ncols := c, armed := none }

/-- Shrink branch of `resize` (:851-864): `flush`, then `GrB_Matrix_resize`
drops every out-of-range entry from each of the three layers. -/
def shrink (v : VM) (r c : Nat) : VM :=
  let v := flush v
  let keep := fun (p : Pair) => decide (inBounds r c p)
  { v with m := v.m.filter keep, dp := v.dp.filter keep, dm := v.dm.filter keep,
           nrows := r, ncols := c }

/-- `dup` (:1287): same layers (COW-shared), arms the fold `(a, b)` that the
write policy decided — any pair of booleans. -/
def dup (v : VM) (a b : Bool) : VM :=
  { v with armed := if a || b then some (a, b) else none }

/-- `fold_latched` (:1138) / `fold_oversized` (:1162): arm some decision
and flush immediately. -/
def foldNow (v : VM) (a b : Bool) : VM := flush { v with armed := some (a, b) }

/-! ## The operation language and the reference -/

inductive Op where
  | set (p : Pair)
  | remove (p : Pair)
  | setAll (NEW : Bool) (es : List Pair)
  | setProduct (NEW : Bool) (rows cols : List Nat)
  | removeMask (mask : List Pair)
  | resize (r c : Nat)
  | dup (a b : Bool)
  | foldNow (a b : Bool)
  | flush
  | wait

/-- `resize` dispatch (:851): shrink if either dimension shrinks. -/
def resize (v : VM) (r c : Nat) : VM :=
  if r < v.nrows ∨ c < v.ncols then shrink v r c else grow v r c

def step (v : VM) : Op → VM
  | .set p => set v p
  | .remove p => remove v p
  | .setAll n es => setAll n v es
  | .setProduct n rs cs => setProduct n v rs cs
  | .removeMask msk => removeMask v msk
  | .resize r c => resize v r c
  | .dup a b => dup v a b
  | .foldNow a b => foldNow v a b
  | .flush => flush v
  | .wait => v

/-- The reference: a plain set of coordinates with dimensions. -/
structure Ref where
  s : Pair → Prop
  nrows : Nat
  ncols : Nat

def Ref.step (R : Ref) : Op → Ref
  | .set p => { R with s := fun q => q = p ∨ R.s q }
  | .remove p => { R with s := fun q => q ≠ p ∧ R.s q }
  | .setAll _ es => { R with s := fun q => q ∈ es ∨ R.s q }
  | .setProduct _ rs cs => { R with s := fun q => q ∈ product rs cs ∨ R.s q }
  | .removeMask msk => { R with s := fun q => q ∉ msk ∧ R.s q }
  | .resize r c =>
    if r < R.nrows ∨ c < R.ncols then ⟨fun q => inBounds r c q ∧ R.s q, r, c⟩
    else ⟨R.s, r, c⟩
  | _ => R

/-- The caller contracts the Rust code documents (debug_asserts that release
builds skip): every written coordinate is in range, and `NEW` coordinates
are not live in the committed base (`set_all` doc, :1233-1241). -/
def Pre (v : VM) : Op → Prop
  | .set p => inBounds v.nrows v.ncols p
  | .remove _ => True
  | .setAll n es => (∀ p ∈ es, inBounds v.nrows v.ncols p) ∧ (n = true → ∀ p ∈ es, p ∉ (flush v).m)
  | .setProduct n rs cs => (∀ p ∈ product rs cs, inBounds v.nrows v.ncols p) ∧
      (n = true → ∀ p ∈ product rs cs, p ∉ (flush v).m)
  | .removeMask _ => True
  | .resize _ _ => True
  | _ => True

end VM
