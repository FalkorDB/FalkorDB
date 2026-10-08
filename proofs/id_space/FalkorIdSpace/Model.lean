import FalkorIdSpace.Card
/-!
# The `IdSpace` state machine (`graph/src/graph/id_space.rs`, main @ e8f8a3017)

Since #2846 (`157d42ec1`) `IdSpace` *owns* the id space: `live` and `recycled`
are the entity count and the free set (`Graph` keeps neither), `entry_bound`
and `taken` are the open batch. Since #3022 (`18fc277b9`) a fifth field,
`released`, records what the open batch freed (outside the count equation;
cleared with `taken`).

A `RoaringTreemap` is an `IdSet` read over the universe `[0, N)` (`N = 2^64`
in the engine). Its operations are the pointwise Boolean ones; `len = cnt N`,
`min`/`max` are the first/last of the ascending listing `asc N`, `rank(b-1)` is
`cnt b` (members `≤ b-1`), `select(k)` is the k-th of `asc N`, and `iter()` is
`asc N`.

Mutators return the state *after* the call together with the result: the Rust
mutates `&mut self` and then runs `checked()`, so an `Inconsistent` refusal
leaves the mutated state behind (the caller discards the MVCC version). Every
other refusal returns before any mutation, and the model returns `sp` unchanged.
-/
namespace IdSpaceModel

variable (N : Nat)

/-- `u64::MAX`. -/
def umax : Nat := N - 1

def sMin (s : IdSet) : Option Nat := (asc N s).head?
def sMax (s : IdSet) : Option Nat := (asc N s).getLast?
def sLen (s : IdSet) : Nat := cnt N s
/-- `select(k)`: the k-th smallest member (0-based), `None` past the end. -/
def sSelect (s : IdSet) (k : Nat) : Option Nat := (asc N s)[k]?
def sDiff (a b : IdSet) : IdSet := fun i => a i && !b i
def sInter (a b : IdSet) : IdSet := fun i => a i && b i
def sUnion (a b : IdSet) : IdSet := fun i => a i || b i
/-- `insert(id)`. -/
def sIns (a : IdSet) (id : Nat) : IdSet := fun i => a i || decide (i = id)
/-- `RoaringTreemap::new()` / `clear()`. -/
def sEmpty : IdSet := fun _ => false
/-- `is_disjoint`. -/
def sDisjoint (a b : IdSet) : Bool := (asc N (sInter a b)).isEmpty
/-- `u64::checked_add`. -/
def checkedAdd (a b : Nat) : Option Nat := if a + b < N then some (a + b) else none

/-- `above` (`id_space.rs:236`): `ids.len() - (if bound == 0 {0} else {ids.rank(bound-1)})`. -/
def above (ids : IdSet) (bound : Nat) : Nat :=
  let below := if bound = 0 then 0 else cnt bound ids
  sLen N ids - below

/-- `reclaim_ids` (`id_space.rs:244`): `free = pool - taken - issued`;
`out.extend(free.iter().take(count))`; return the number appended. -/
def freeOf (pool taken issued : IdSet) : IdSet := sDiff (sDiff pool taken) issued

def reclaimIds (pool taken issued : IdSet) (count : Nat) (out : List Nat) : List Nat × Nat :=
  let out' := out ++ (asc N (freeOf pool taken issued)).take count
  (out', out'.length - out.length)

/-- `IdSpace { live, recycled, entry_bound, taken, released }` (`:164`). -/
structure IdSpace where
  live     : Nat
  recycled : IdSet
  eb       : Nat
  taken    : IdSet
  /-- `released` (`:207`, #3022): every id the open batch freed. -/
  released : IdSet := fun _ => false

/-- `IdSpaceError` (`:104`). `Miscounted` is gone (#2846); `Inconsistent` and
`AlreadyTaken` are new. -/
inductive IdSpaceError where
  | inconsistent (live recycled bound eb taken expected : Nat)
  | neverCreated (id : Nat)
  | hole (eb highest created : Nat)
  | alreadyRecycled (id : Nat)
  | alreadyLive (id eb : Nat)
  | alreadyTaken (id : Nat)
  | idOutOfRange (id : Nat)
deriving DecidableEq, Repr

/-- `IdSpace::new` (`:266`), also `Default::default` (`:258`). -/
def IdSpace.new : IdSpace := ⟨0, sEmpty, 0, sEmpty, sEmpty⟩

/-- `IdSpace::restored` (`:280`): `entry_bound = live + recycled.len()`. -/
def IdSpace.restored (live : Nat) (recycled : IdSet) : IdSpace :=
  ⟨live, recycled, live + sLen N recycled, sEmpty, sEmpty⟩

/-- `taken` (`:311`) — a field read. -/
def IdSpace.takenSet (sp : IdSpace) : IdSet := sp.taken
/-- `released` (`:317`) — a field read. -/
def IdSpace.releasedSet (sp : IdSpace) : IdSet := sp.released

/-- `recycled_count` (`:323`). -/
def IdSpace.recycledCount (sp : IdSpace) : Nat := sLen N sp.recycled
/-- `is_free` (`:329`). -/
def IdSpace.isFree (sp : IdSpace) (id : Nat) : Bool := sp.recycled id
/-- `bound` (`:343`): `live + recycled.len()`. -/
def IdSpace.bound (sp : IdSpace) : Nat := sp.live + sLen N sp.recycled
/-- `max_id` (`:353`). -/
def IdSpace.maxId (sp : IdSpace) : Nat := if sp.live = 0 then 0 else sp.bound N - 1
/-- `new_version` (`:367`). -/
def IdSpace.newVersion (sp : IdSpace) : IdSpace := IdSpace.restored N sp.live sp.recycled

/-- `checked` (`:403`). -/
def IdSpace.checked (sp : IdSpace) : Except IdSpaceError Unit :=
  let taken := above N sp.taken sp.eb
  let bound := sp.bound N
  match checkedAdd N sp.eb taken with
  | some expected =>
    if expected = bound then .ok ()
    else .error (.inconsistent sp.live (sLen N sp.recycled) bound sp.eb taken expected)
  | none => .error (.inconsistent sp.live (sLen N sp.recycled) bound sp.eb taken (umax N))

/-- `open_batch` (`:447`): `checked()?` then re-anchor and clear `taken` and
(since #3022) `released`. -/
def IdSpace.openBatch (sp : IdSpace) : Except IdSpaceError IdSpace :=
  match sp.checked N with
  | .error e => .error e
  | .ok () => .ok { sp with eb := sp.bound N, taken := sEmpty, released := sEmpty }

/-- `reserve` (`:483`). `allocOk` is whether `try_reserve_exact(count)`
succeeded (an allocator boundary). -/
def IdSpace.reserve (sp : IdSpace) (allocOk : Bool) (count : Nat) (issued : IdSet) :
    Except String (List Nat) :=
  if !allocOk then .error s!"failed to reserve {count} ids" else
  let (ids, reclaimed) := reclaimIds N sp.recycled sp.taken issued count []
  let start := sp.eb + above N sp.taken sp.eb + above N issued sp.eb
  .ok (ids ++ List.range' start (count - reclaimed))

/-- `cancel` (`:529`): `if !taken.insert(id) {AlreadyTaken}; recycled.insert(id); checked()`. -/
def IdSpace.cancel (sp : IdSpace) (id : Nat) : IdSpace × Except IdSpaceError Unit :=
  if sp.taken id then (sp, .error (.alreadyTaken id)) else
  let sp' := { sp with taken := sIns sp.taken id, recycled := sIns sp.recycled id }
  (sp', sp'.checked N)

/-- `refuse_recycled` (`:550`). -/
def IdSpace.refuseRecycled (sp : IdSpace) (nodes : IdSet) : Except IdSpaceError Unit :=
  match sMin N (sInter nodes sp.recycled) with
  | some id => .error (.alreadyRecycled id)
  | none    => .ok ()

/-- `ID_LIMIT` (`id_space.rs:100`) = `GrB_INDEX_MAX` = `2^60 - 1` (`tensor.rs:144`): one past
the highest id a batch may create. The model takes it as a parameter `L`; the
theorems need only `L < N` (it is below `u64::MAX`). -/
def idLimit : Nat := 2 ^ 60 - 1

/-- `create` (`:582`). Since #2911 (`fe619ac5f`) the first refusal is
`nodes.max().filter(|&id| id >= ID_LIMIT)` → `IdOutOfRange(id)` (it used to be
`nodes.contains(u64::MAX)`). -/
def IdSpace.create (L : Nat) (sp : IdSpace) (nodes : IdSet) : IdSpace × Except IdSpaceError Unit :=
  match (sMax N nodes).filter (· ≥ L) with
  | some id => (sp, .error (.idOutOfRange id))
  | none =>
    let claimed := sDiff nodes sp.recycled
    match (sMin N claimed).filter (· < sp.eb) with
    | some id => (sp, .error (.alreadyLive id sp.eb))
    | none =>
      if !sDisjoint N sp.taken claimed then
        match sMin N (sInter claimed sp.taken) with
        | some id => (sp, .error (.alreadyLive id sp.eb))
        | none => (sp, .error (.alreadyLive 0 sp.eb))   -- `.expect` panic: unreachable (create_expect_unreachable)
      else
        let sp' := { sp with recycled := sDiff sp.recycled nodes,
                             live := sp.live + sLen N nodes,
                             taken := sUnion sp.taken nodes }
        (sp', sp'.checked N)

/-- `refuse_undeletable` (`:638`). -/
def IdSpace.refuseUndeletable (sp : IdSpace) (nodes : IdSet) : Except IdSpaceError Unit :=
  if (sMax N nodes).all (· < sp.eb) then .ok () else
  match (sMax N (sDiff nodes sp.taken)).filter (· ≥ sp.eb) with
  | some id => .error (.neverCreated id)
  | none    => .ok ()

/-- `refuse_not_live` (`:679`, #3022): `refuse_recycled(ids)?;
refuse_undeletable(ids)`. Changes nothing. -/
def IdSpace.refuseNotLive (sp : IdSpace) (ids : IdSet) : Except IdSpaceError Unit :=
  match sp.refuseRecycled N ids with
  | .error e => .error e
  | .ok () => sp.refuseUndeletable N ids

/-- `release` (`:703`): since #3022 the two refusals are `refuse_not_live`, and
the freed ids are also added to `released`. The `debug_assert!(freed ⊆ requested)` is a hypothesis
of the theorems, not a branch (it is compiled out of release builds). `live -=
freed.len()` is `u64`; `release_no_underflow` proves it never wraps in a
reachable state, so `Nat` subtraction is faithful there. -/
def IdSpace.release (sp : IdSpace) (requested freed : IdSet) : IdSpace × Except IdSpaceError Unit :=
  match sp.refuseNotLive N requested with
  | .error e => (sp, .error e)
  | .ok () =>
    let sp' := { sp with recycled := sUnion sp.recycled freed, live := sp.live - sLen N freed,
                         released := sUnion sp.released freed }
    (sp', sp'.checked N)

/-- `verify`'s hole test (`:748-782`) over the part of `taken` at or above the
boundary: `lowest_above = taken.select(taken.len() - created)`,
`.zip(taken.max()).filter(lowest != entry || created - 1 != highest - entry)`. -/
def IdSpace.holeCheck (sp : IdSpace) : Option Nat :=
  let created := above N sp.taken sp.eb
  match sSelect N sp.taken (sLen N sp.taken - created), sMax N sp.taken with
  | some lo, some hi =>
    if lo != sp.eb || created - 1 != hi - sp.eb then some hi else none
  | _, _ => none

/-- `verify` (`:737`): `checked()?`, then the hole test. -/
def IdSpace.verify (sp : IdSpace) : Except IdSpaceError Unit :=
  match sp.checked N with
  | .error e => .error e
  | .ok () =>
    match sp.holeCheck N with
    | some hi => .error (.hole sp.eb hi (above N sp.taken sp.eb))
    | none => .ok ()

end IdSpaceModel
