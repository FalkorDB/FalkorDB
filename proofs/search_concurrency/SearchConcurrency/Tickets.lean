/-
# Index population tickets

| here | there |
| --- | --- |
| `Slots`                 | `graph/src/index/mod.rs:987` `PendingSlots` (`current_generation`, `current_pending`, `stale_pending`) |
| `inc`                   | `mod.rs:2110` `increment_pending_for_generation` |
| `dec`                   | `mod.rs:2128` `try_decrement_pending_for_generation` (`if prev > 0 { -= 1 }` = `Nat` truncated `-`) |
| `pendingFor`            | `mod.rs:2150` `pending_count_for_generation` |
| `Slots.cur = 0`         | `mod.rs:2083` `is_operational` / `indexer.rs:704` `enabled` (`pending_count() == 0`) |
| `bump`                  | `mod.rs:1048` `bump_id` (called only from `recreate_index`, i.e. when a *vector* field is added) |
| `acquire`/`release`     | `indexer.rs:626` `acquire_population_snapshot`, `:670` `release_population_ticket` |
| `batch`                 | `graph/src/graph/graph.rs:521` `populate_index_batch` (the three exits at `:551`, `:562`, `:726`) |

The counters are `i32` in Rust; `2^31` outstanding tickets is not reachable (one
ticket per populate job), so `Nat` is faithful.
-/

namespace SC.Tickets

structure Slots where
  gen : Nat
  cur : Nat
  stale : Nat
deriving DecidableEq, Repr

def inc (s : Slots) (g : Nat) : Slots :=
  if g = s.gen then { s with cur := s.cur + 1 } else { s with stale := s.stale + 1 }

def dec (s : Slots) (g : Nat) : Slots :=
  if g = s.gen then { s with cur := s.cur - 1 } else { s with stale := s.stale - 1 }

def pendingFor (s : Slots) (g : Nat) : Nat := if g = s.gen then s.cur else s.stale

def bump (s : Slots) (g' : Nat) : Slots := { gen := g', cur := 0, stale := s.stale + s.cur }

/-- `T` is the multiset of outstanding tickets (their generations). -/
def Inv (s : Slots) (T : List Nat) : Prop :=
  s.cur = T.count s.gen ∧ s.stale + s.cur = T.length

theorem inv_empty (g : Nat) : Inv { gen := g, cur := 0, stale := 0 } [] := by
  simp [Inv]

theorem inc_inv {s : Slots} {T : List Nat} (h : Inv s T) (g : Nat) : Inv (inc s g) (g :: T) := by
  obtain ⟨h1, h2⟩ := h
  unfold inc Inv
  by_cases hg : g = s.gen
  · have hb : (g == s.gen) = true := by simpa using hg
    simp only [hg, ite_true, List.count_cons, beq_self_eq_true, List.length_cons]
    constructor <;> (try simp) <;> omega
  · have hb : (g == s.gen) = false := by simpa using hg
    simp only [hg, ite_false, List.count_cons, hb, List.length_cons]
    constructor <;> (try simp) <;> omega

theorem dec_inv {s : Slots} {T : List Nat} (h : Inv s T) {g : Nat} (hg : g ∈ T) :
    Inv (dec s g) (T.erase g) := by
  obtain ⟨h1, h2⟩ := h
  have hlen := List.length_erase_of_mem hg
  have hpos := List.length_pos_of_mem hg
  unfold dec Inv
  by_cases he : g = s.gen
  · have hc : 0 < T.count g := List.count_pos_iff.mpr hg
    rw [he] at hc hlen
    have hb : (s.gen == s.gen) = true := by simp
    simp only [he, ite_true, List.count_erase, hb]
    constructor <;> omega
  · have hb : (g == s.gen) = false := by simpa using he
    have hce : (T.erase g).count s.gen = T.count s.gen := by
      rw [List.count_erase]; simp [hb]
    have hle := List.count_le_length (a := s.gen) (l := T.erase g)
    simp only [he, ite_false, hce]
    constructor <;> omega

theorem bump_inv {s : Slots} {T : List Nat} (h : Inv s T) {g' : Nat} (hfresh : g' ∉ T) :
    Inv (bump s g') T := by
  obtain ⟨_, h2⟩ := h
  unfold bump Inv
  exact ⟨by simp [List.count_eq_zero_of_not_mem hfresh], by simp; omega⟩

/-- A released ticket is always counted: `dec` never saturates on a live ticket, so
releases are exact (the `prev > 0` guard is never the thing that saves us). -/
theorem release_exact {s : Slots} {T : List Nat} (h : Inv s T) {g : Nat} (hg : g ∈ T) :
    0 < pendingFor s g := by
  obtain ⟨h1, h2⟩ := h
  unfold pendingFor
  by_cases he : g = s.gen
  · simp only [he, ite_true, h1]; exact List.count_pos_iff.mpr (he ▸ hg)
  · simp only [he, ite_false]
    have hb : (g == s.gen) = false := by simpa using he
    have hce : (T.erase g).count s.gen = T.count s.gen := by
      rw [List.count_erase]; simp [hb]
    have := List.count_le_length (a := s.gen) (l := T.erase g)
    have := List.length_erase_of_mem hg
    have := List.length_pos_of_mem hg
    omega

/-- `is_operational` ⇔ no outstanding ticket of the current generation. -/
theorem operational_iff {s : Slots} {T : List Nat} (h : Inv s T) :
    s.cur = 0 ↔ s.gen ∉ T := by
  rw [h.1]; exact ⟨fun h0 hm => by have := List.count_pos_iff.mpr hm; omega, List.count_eq_zero_of_not_mem⟩

/-- After a recreate (`bump`), stale workers' releases never touch the fresh
generation's counter. -/
theorem stale_release_isolated (s : Slots) (g g' : Nat) (hne : g ≠ g') :
    (dec (bump s g') g).cur = 0 := by
  simp [dec, bump, hne]

/-! ## The population protocol, and the lost second index

A worker carries its ticket's generation, the field snapshot it was spawned with,
and its cursor. One step of `populate_index_batch`: -/

structure Worker where
  gen : Nat
  attrs : List String
  cursor : Nat
  alive : Bool := true
deriving DecidableEq, Repr

structure PState where
  slots : Slots
  /-- the label's current field set (what `db.indexes` reports) -/
  fields : List String
  /-- for each batch, which attributes have been written into its documents -/
  docs : List (List String)
  workers : List Worker
deriving DecidableEq, Repr

def setAt (l : List (List String)) (i : Nat) (f : List String → List String) : List (List String) :=
  l.modify i f

/-- `populate_index_batch` for worker `i` over `nb` batches (graph.rs:551-728). -/
def batch (nb : Nat) (s : PState) (i : Nat) : PState :=
  match s.workers[i]? with
  | none => s
  | some w =>
    if !w.alive then s
    else if w.gen ≠ s.slots.gen then          -- `!is_ticket_current`
      { s with slots := dec s.slots w.gen, workers := s.workers.set i { w with alive := false } }
    else if pendingFor s.slots w.gen > 1 then  -- `ticket_pending_changes(&ticket) > 1`
      { s with slots := dec s.slots w.gen, workers := s.workers.set i { w with alive := false } }
    else
      let docs := setAt s.docs w.cursor (fun a => a ++ w.attrs)
      if w.cursor + 1 ≥ nb then                -- exhausted: release
        { s with docs, slots := dec s.slots w.gen, workers := s.workers.set i { w with alive := false } }
      else
        { s with docs, workers := s.workers.set i { w with cursor := w.cursor + 1 } }

/-- `CREATE INDEX` of a range field on a label that already has an RS spec:
`register_fields` on the shared spec, **no** `bump_id` (indexer.rs:319-331), then
`populate_index` acquires a ticket with the *new* field snapshot. -/
def createRange (s : PState) (attr : String) : PState :=
  let fields := s.fields ++ [attr]
  { s with fields, slots := inc s.slots s.slots.gen,
           workers := s.workers ++ [{ gen := s.slots.gen, attrs := fields, cursor := 0 }] }

/-- Every batch carries every field of the index. -/
def complete (s : PState) : Bool := s.docs.all fun d => s.fields.all (d.contains ·)

def operational (s : PState) : Bool := s.slots.cur = 0

def init (nb : Nat) : PState :=
  { slots := { gen := 1, cur := 0, stale := 0 }, fields := [], docs := List.replicate nb [], workers := [] }

/-- The live-server trace (`twoidx.py`, 300k nodes = 30 batches; here 3):
`CREATE INDEX … (n.a)`, one batch of `a`, `CREATE INDEX … (n.b)`; the second
worker's first batch sees two tickets and bails; the first finishes with its
`{a}` snapshot. The index is OPERATIONAL and `b` is in no document. -/
def lostSecondIndex : PState :=
  let s := createRange (init 3) "a"
  let s := batch 3 s 0            -- P1 batch 0 with {a}
  let s := createRange s "b"      -- P2 spawned with {a,b}, pending = 2
  let s := batch 3 s 1            -- P2: pending 2 > 1 → bail
  let s := batch 3 s 0            -- P1 batch 1 with {a}
  batch 3 s 0                     -- P1 batch 2 → exhausted, release

theorem second_index_never_populated :
    operational lostSecondIndex = true ∧ complete lostSecondIndex = false ∧
    lostSecondIndex.docs = [["a"], ["a"], ["a"]] := by decide

/-- The other interleaving is fine: if the *older* worker reaches its check first,
it bails and the newer one reindexes everything. So the outcome is a scheduling
race, and on the live server the loser is the newer worker (it is already parked
on the indexer lock when the older one releases it). -/
def luckyOrder : PState :=
  let s := createRange (init 3) "a"
  let s := batch 3 s 0
  let s := createRange s "b"
  let s := batch 3 s 0            -- P1 sees pending 2 → bails
  let s := batch 3 s 1
  let s := batch 3 s 1
  batch 3 s 1

theorem lucky_order_complete : operational luckyOrder = true ∧ complete luckyOrder = true := by decide

/-- Fix: bump the generation on every schema change (what `recreate_index` already
does for vector fields), so the older worker fails `is_ticket_current` and the
newer one owns the index. -/
def createRangeFixed (s : PState) (attr : String) (g' : Nat) : PState :=
  let fields := s.fields ++ [attr]
  let slots := inc (bump s.slots g') g'
  { s with fields, slots, workers := s.workers ++ [{ gen := g', attrs := fields, cursor := 0 }] }

def fixedTrace : PState :=
  let s := createRange (init 3) "a"
  let s := batch 3 s 0
  let s := createRangeFixed s "b" 2
  let s := batch 3 s 1            -- the order that lost `b` above
  let s := batch 3 s 0            -- P1: stale ticket → bails
  let s := batch 3 s 1
  batch 3 s 1

theorem fixed_trace_complete : operational fixedTrace = true ∧ complete fixedTrace = true := by decide

end SC.Tickets
