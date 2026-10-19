/-
# Index population tickets

| here | there |
| --- | --- |
| `Slots`                 | `graph/src/index/mod.rs:987` `PendingSlots` (`current_generation`, `current_pending`, `stale_pending`) |
| `inc`                   | `mod.rs:2111` `increment_pending_for_generation` |
| `dec`                   | `mod.rs:2129` `try_decrement_pending_for_generation` (`if prev > 0 { -= 1 }` = `Nat` truncated `-`) |
| `pendingFor`            | `mod.rs:2151` `pending_count_for_generation` |
| `Slots.cur = 0`         | `mod.rs:2084` `is_operational` / `indexer.rs:720` `enabled` (`pending_count() == 0`) |
| `bump`                  | `mod.rs:1049` `bump_id` (from `recreate_index`, and since #3086 from `create_index` whenever the label already has an entry, `indexer.rs:357-359`) |
| `acquire`/`release`     | `indexer.rs:642` `acquire_population_snapshot`, `:686` `release_population_ticket` |
| `batch`                 | `graph/src/graph/graph.rs:534` `populate_index_batch` (the three exits: `is_ticket_current` `:565`, `ticket_pending_changes > 1` `:577`, exhausted `:740`) |
| `store`                 | `indexer.rs:181` `create_index` @ 73077bb4a (fields appended, then `bump_id` if `existing.is_some()`) |
| `MState`/`step`/`Reach` | `graph.rs:3585` `Graph::create_index`: `create_index` (store) then `populate_index` (acquire), on the serialized writer |
| `pre3086_createRange`   | `create_index` @ 1d15bf713 (`indexer.rs:319-331`): `register_fields`, no `bump_id` |

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

/-- `populate_index_batch` for worker `i` over `nb` batches (graph.rs:565-728). -/
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

/-- HISTORICAL (1d15bf713, before #3086): `CREATE INDEX` of a range field on a label
that already has an RS spec: `register_fields` on the shared spec, **no** `bump_id`
(indexer.rs:319-331), then
`populate_index` acquires a ticket with the *new* field snapshot. -/
def pre3086_createRange (s : PState) (attr : String) : PState :=
  let fields := s.fields ++ [attr]
  { s with fields, slots := inc s.slots s.slots.gen,
           workers := s.workers ++ [{ gen := s.slots.gen, attrs := fields, cursor := 0 }] }

/-- Every batch carries every field of the index. -/
def complete (s : PState) : Bool := s.docs.all fun d => s.fields.all (d.contains ·)

def operational (s : PState) : Bool := s.slots.cur = 0

def init (nb : Nat) : PState :=
  { slots := { gen := 1, cur := 0, stale := 0 }, fields := [], docs := List.replicate nb [], workers := [] }

/-- HISTORICAL (BUG 2, fixed by #3086). The live-server trace (`twoidx.py`, 300k nodes = 30 batches; here 3):
`CREATE INDEX … (n.a)`, one batch of `a`, `CREATE INDEX … (n.b)`; the second
worker's first batch sees two tickets and bails; the first finishes with its
`{a}` snapshot. The index is OPERATIONAL and `b` is in no document. -/
def pre3086_lostSecondIndex : PState :=
  let s := pre3086_createRange (init 3) "a"
  let s := batch 3 s 0            -- P1 batch 0 with {a}
  let s := pre3086_createRange s "b"      -- P2 spawned with {a,b}, pending = 2
  let s := batch 3 s 1            -- P2: pending 2 > 1 → bail
  let s := batch 3 s 0            -- P1 batch 1 with {a}
  batch 3 s 0                     -- P1 batch 2 → exhausted, release

theorem pre3086_second_index_never_populated :
    operational pre3086_lostSecondIndex = true ∧ complete pre3086_lostSecondIndex = false ∧
    pre3086_lostSecondIndex.docs = [["a"], ["a"], ["a"]] := by decide

/-- The other interleaving is fine: if the *older* worker reaches its check first,
it bails and the newer one reindexes everything. So the outcome is a scheduling
race, and on the live server the loser is the newer worker (it is already parked
on the indexer lock when the older one releases it). -/
def pre3086_luckyOrder : PState :=
  let s := pre3086_createRange (init 3) "a"
  let s := batch 3 s 0
  let s := pre3086_createRange s "b"
  let s := batch 3 s 0            -- P1 sees pending 2 → bails
  let s := batch 3 s 1
  let s := batch 3 s 1
  batch 3 s 1

theorem pre3086_lucky_order_complete :
    operational pre3086_luckyOrder = true ∧ complete pre3086_luckyOrder = true := by decide

/-- Fix: bump the generation on every schema change (what `recreate_index` already
does for vector fields), so the older worker fails `is_ticket_current` and the
newer one owns the index. -/
def createRangeFixed (s : PState) (attr : String) (g' : Nat) : PState :=
  let fields := s.fields ++ [attr]
  let slots := inc (bump s.slots g') g'
  { s with fields, slots, workers := s.workers ++ [{ gen := g', attrs := fields, cursor := 0 }] }

def fixedTrace : PState :=
  let s := pre3086_createRange (init 3) "a"
  let s := batch 3 s 0
  let s := createRangeFixed s "b" 2
  let s := batch 3 s 1            -- the order that lost `b` above
  let s := batch 3 s 0            -- P1: stale ticket → bails
  let s := batch 3 s 1
  batch 3 s 1

theorem fixed_trace_complete : operational fixedTrace = true ∧ complete fixedTrace = true := by decide

/-! ## On main (#3086, 73077bb4a): every field addition starts a new generation

`Graph::create_index` (graph.rs:3585) runs two steps on the serialized writer:
`Indexer::create_index` stores the entry (fields appended; since #3086 `bump_id`
when the label already had one, indexer.rs:357-359), then `populate_index`
acquires a ticket of the current generation with the current field snapshot
(`acquire_population_snapshot`, indexer.rs:642). Batches of every job interleave
arbitrarily with both. `NEXT_INDEX_ID` is a monotone static counter, so the bumped
generation is fresh; `gen + 1` stands for its next value. -/

/-- `Indexer::create_index` @ 73077bb4a: the store, with the #3086 bump. -/
def store (s : PState) (attr : String) : PState :=
  { s with fields := s.fields ++ [attr],
           slots := if s.fields = [] then s.slots else bump s.slots (s.slots.gen + 1) }

/-- `populate_index` → `acquire_population_snapshot`: one ticket of the current
generation and the current field set, in one critical section. -/
def acquire (s : PState) : PState :=
  { s with slots := inc s.slots s.slots.gen,
           workers := s.workers ++ [{ gen := s.slots.gen, attrs := s.fields, cursor := 0 }] }

structure MState where
  p : PState
  /-- the entry is stored but `populate_index` has not acquired its ticket yet -/
  owed : Bool

inductive Ev
  | create (attr : String)
  | populate
  | batch (i : Nat)

def step (nb : Nat) (m : MState) : Ev → Option MState
  | .create a => if m.owed then none else some ⟨store m.p a, true⟩
  | .populate => if m.owed then some ⟨acquire m.p, false⟩ else none
  | .batch i => some ⟨batch nb m.p i, m.owed⟩

/-- Every state reachable by any interleaving of index creations and batches. -/
inductive Reach (nb : Nat) : MState → Prop
  | init : Reach nb ⟨init nb, false⟩
  | step {m m' : MState} (e : Ev) : Reach nb m → step nb m e = some m' → Reach nb m'

/-- A worker that owns the current generation. -/
def Live (s : PState) (w : Worker) : Prop := w.alive = true ∧ w.gen = s.slots.gen

def dget (l : List (List String)) (j : Nat) : List String := (l[j]?).getD []

/-- Batch `j` carries every field of the index. -/
def Full (s : PState) (j : Nat) : Prop := ∀ f ∈ s.fields, f ∈ dget s.docs j

def NoLive (s : PState) : Prop := ∀ (i : Nat) (w : Worker), s.workers[i]? = some w → ¬ Live s w

structure MInv (nb : Nat) (m : MState) : Prop where
  len : m.p.docs.length = nb
  gen_le : ∀ (i : Nat) (w : Worker), m.p.workers[i]? = some w → w.gen ≤ m.p.slots.gen
  uniq : ∀ (i j : Nat) (wi wj : Worker), m.p.workers[i]? = some wi → m.p.workers[j]? = some wj →
    Live m.p wi → Live m.p wj → i = j
  cur_live : ∀ (i : Nat) (w : Worker), m.p.workers[i]? = some w → Live m.p w → m.p.slots.cur = 1
  cur_none : NoLive m.p → m.p.slots.cur = 0
  live_ok : ∀ (i : Nat) (w : Worker), m.p.workers[i]? = some w → Live m.p w →
    w.attrs = m.p.fields ∧ ∀ j, j < w.cursor → Full m.p j
  owed_none : m.owed = true → NoLive m.p
  owed_fields : m.owed = true → m.p.fields ≠ []
  empty : m.p.fields = [] → m.p.workers = []
  done : m.owed = false → NoLive m.p → ∀ j, j < nb → Full m.p j

theorem dget_setAt_mono (l : List (List String)) (i j : Nat) (x : List String) {f : String}
    (h : f ∈ dget l j) : f ∈ dget (setAt l i (· ++ x)) j := by
  unfold dget setAt at *
  rw [List.getElem?_modify]
  cases hj : l[j]? with
  | none => simp [hj] at h
  | some d => simp only [hj, Option.getD_some] at h; simp only [Option.map_eq_map, Option.map_some, Option.getD_some]; split <;> simp [h]

theorem dget_setAt_hit (l : List (List String)) (i : Nat) (x : List String) {f : String}
    (hi : i < l.length) (h : f ∈ x) : f ∈ dget (setAt l i (· ++ x)) i := by
  unfold dget setAt
  rw [List.getElem?_modify, List.getElem?_eq_getElem hi]
  simp [h]

theorem length_setAt (l : List (List String)) (i : Nat) (g : List String → List String) :
    (setAt l i g).length = l.length := List.length_modify _ _ _

theorem getElem?_set_cases {l : List Worker} {i j : Nat} {a w : Worker}
    (h : (l.set i a)[j]? = some w) : (j = i ∧ w = a) ∨ (j ≠ i ∧ l[j]? = some w) := by
  rw [List.getElem?_set] at h
  by_cases hij : i = j
  · subst hij; simp only [ite_true] at h; split at h
    · cases h; exact Or.inl ⟨rfl, rfl⟩
    · cases h
  · simp only [hij, ite_false] at h; exact Or.inr ⟨fun e => hij e.symm, h⟩

theorem getElem?_append_one {l : List Worker} {a w : Worker} {j : Nat}
    (h : (l ++ [a])[j]? = some w) : l[j]? = some w ∨ (j = l.length ∧ w = a) := by
  by_cases hj : j < l.length
  · rw [List.getElem?_append_left hj] at h; exact Or.inl h
  · rw [List.getElem?_append_right (by omega)] at h
    have : j - l.length = 0 := by
      cases hk : j - l.length with
      | zero => rfl
      | succ k => rw [hk] at h; simp at h
    rw [this] at h; simp at h; exact Or.inr ⟨by omega, h.symm⟩

/-- The four exits of one `populate_index_batch` step. -/
theorem batch_cases (nb : Nat) (s : PState) (i : Nat) :
    batch nb s i = s ∨
    (∃ w, s.workers[i]? = some w ∧ w.alive = true ∧ w.gen ≠ s.slots.gen ∧
      batch nb s i = { s with slots := dec s.slots w.gen, workers := s.workers.set i { w with alive := false } }) ∨
    (∃ w, s.workers[i]? = some w ∧ w.alive = true ∧ w.gen = s.slots.gen ∧ pendingFor s.slots w.gen > 1) ∨
    (∃ w, s.workers[i]? = some w ∧ w.alive = true ∧ w.gen = s.slots.gen ∧ w.cursor + 1 ≥ nb ∧
      batch nb s i = { s with docs := setAt s.docs w.cursor (· ++ w.attrs), slots := dec s.slots w.gen,
                              workers := s.workers.set i { w with alive := false } }) ∨
    (∃ w, s.workers[i]? = some w ∧ w.alive = true ∧ w.gen = s.slots.gen ∧ w.cursor + 1 < nb ∧
      batch nb s i = { s with docs := setAt s.docs w.cursor (· ++ w.attrs),
                              workers := s.workers.set i { w with cursor := w.cursor + 1 } }) := by
  cases hw : s.workers[i]? with
  | none => exact Or.inl (by simp [batch, hw])
  | some w =>
    by_cases ha : w.alive = true
    · by_cases hg : w.gen ≠ s.slots.gen
      · exact Or.inr (Or.inl ⟨w, rfl, ha, hg, by simp [batch, hw, ha, hg]⟩)
      · have hg' : w.gen = s.slots.gen := by simpa using hg
        by_cases hp : pendingFor s.slots w.gen > 1
        · exact Or.inr (Or.inr (Or.inl ⟨w, rfl, ha, hg', hp⟩))
        · have hp' : ¬ pendingFor s.slots s.slots.gen > 1 := hg' ▸ hp
          by_cases he : w.cursor + 1 ≥ nb
          · exact Or.inr (Or.inr (Or.inr (Or.inl ⟨w, rfl, ha, hg', he,
              by simp only [batch, hw, ha, hg', hp', he]; simp⟩)))
          · exact Or.inr (Or.inr (Or.inr (Or.inr ⟨w, rfl, ha, hg', by omega,
              by simp only [batch, hw, ha, hg', hp', he]; simp⟩)))
    · have : w.alive = false := by simpa using ha
      exact Or.inl (by simp [batch, hw, this])

theorem lt_of_getElem?_some {α : Type} {l : List α} {i : Nat} {a : α} (h : l[i]? = some a) :
    i < l.length := by
  by_cases hc : i < l.length
  · exact hc
  · rw [List.getElem?_eq_none (by omega)] at h; cases h

theorem inv_init (nb : Nat) : MInv nb ⟨init nb, false⟩ := by
  refine ⟨by simp [init], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [init, NoLive, Full]

/-- Changes that keep the field set, the documents, the generation and its
counter, and every live worker in place, adding no live worker. -/
theorem inv_transfer {nb : Nat} {m m' : MState} (h : MInv nb m) (hf : m'.p.fields = m.p.fields)
    (hd : m'.p.docs = m.p.docs) (hg : m'.p.slots.gen = m.p.slots.gen)
    (hc : m'.p.slots.cur = m.p.slots.cur) (ho : m'.owed = m.owed)
    (hfw : ∀ (j : Nat) (v : Worker), m'.p.workers[j]? = some v → v.gen ≤ m.p.slots.gen ∧ (Live m'.p v → m.p.workers[j]? = some v))
    (hbw : ∀ (j : Nat) (v : Worker), m.p.workers[j]? = some v → Live m.p v → m'.p.workers[j]? = some v)
    (hem : m.p.fields ≠ []) : MInv nb m' := by
  obtain ⟨hlen, hle, huq, hcl, hcn, hlo, hon, hof, -, hdn⟩ := h
  have hL : ∀ v, Live m'.p v ↔ Live m.p v := by intro v; simp [Live, hg]
  have hN : NoLive m'.p ↔ NoLive m.p := by
    constructor
    · intro hn j v hv hl; exact hn j v (hbw j v hv hl) ((hL v).2 hl)
    · intro hn j v hv hl; exact hn j v ((hfw j v hv).2 hl) ((hL v).1 hl)
  have hF : ∀ j, Full m'.p j ↔ Full m.p j := by intro j; simp [Full, hf, hd]
  refine ⟨hd ▸ hlen, fun j v hv => hg ▸ (hfw j v hv).1, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro a b va vb ha hb hla hlb
    exact huq a b va vb ((hfw a va ha).2 hla) ((hfw b vb hb).2 hlb) ((hL _).1 hla) ((hL _).1 hlb)
  · intro j v hv hl; rw [hc]; exact hcl j v ((hfw j v hv).2 hl) ((hL _).1 hl)
  · intro hn; rw [hc]; exact hcn (hN.1 hn)
  · intro j v hv hl
    obtain ⟨h1, h2⟩ := hlo j v ((hfw j v hv).2 hl) ((hL _).1 hl)
    exact ⟨hf ▸ h1, fun k hk => (hF k).2 (h2 k hk)⟩
  · intro hw; exact hN.2 (hon (ho ▸ hw))
  · intro hw; rw [hf]; exact hof (ho ▸ hw)
  · intro h0; exact absurd (hf ▸ h0) hem
  · intro hw hn k hk; exact (hF k).2 (hdn (ho ▸ hw) (hN.1 hn) k hk)

theorem inv_batch {nb : Nat} {m : MState} (h : MInv nb m) (i : Nat) :
    MInv nb ⟨batch nb m.p i, m.owed⟩ := by
  rcases batch_cases nb m.p i with hb | ⟨w, hw, ha, hg, hb⟩ | ⟨w, hw, ha, hg, hp⟩ | ⟨w, hw, ha, hg, he, hb⟩ | ⟨w, hw, ha, hg, he, hb⟩
  · rw [hb]; exact h
  · -- stale exit: `w` is not live; only the stale counter moves
    have hne : m.p.fields ≠ [] := fun h0 => by rw [h.empty h0] at hw; simp at hw
    apply inv_transfer h <;> (try simp only [hb]) <;> (try simp [dec, hg])
    · intro j v hv
      rcases getElem?_set_cases hv with ⟨rfl, rfl⟩ | ⟨-, hv⟩
      · exact ⟨h.gen_le _ w hw, fun hl => by simp [Live] at hl⟩
      · exact ⟨h.gen_le _ _ hv, fun _ => hv⟩
    · intro j v hv hl
      rw [List.getElem?_set]
      have : i ≠ j := fun e => by subst e; rw [hw] at hv; cases hv; exact hg hl.2
      simp [this, hv]
    · exact hne
  · -- the `> 1` bail cannot fire: a live worker's generation has exactly one ticket
    have := h.cur_live i w hw ⟨ha, hg⟩
    simp [pendingFor, hg, this] at hp
  · -- last batch: write it, release the ticket; no worker is live any more
    have hl : Live m.p w := ⟨ha, hg⟩
    have hc1 := h.cur_live i w hw hl
    obtain ⟨hat, hfull⟩ := h.live_ok i w hw hl
    have hown : m.owed = false := by
      cases ho : m.owed
      · rfl
      · exact absurd hl (h.owed_none ho i w hw)
    have hne : m.p.fields ≠ [] := fun h0 => by rw [h.empty h0] at hw; simp at hw
    have hnl : NoLive (batch nb m.p i) := by
      intro j v hv hlv
      rw [hb] at hv hlv
      rcases getElem?_set_cases hv with ⟨rfl, rfl⟩ | ⟨hji, hv⟩
      · simp [Live] at hlv
      · exact hji (h.uniq j i v w hv hw (by simpa [Live, dec, hg] using hlv) hl)
    rw [hb] at hnl ⊢
    refine ⟨by simp [length_setAt, h.len], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro j v hv
      simp only [dec, hg, ite_true]
      rcases getElem?_set_cases hv with ⟨rfl, rfl⟩ | ⟨-, hv⟩
      · exact h.gen_le _ w hw
      · exact h.gen_le _ _ hv
    · intro a b va vb ha' _ hla _; exact absurd hla (hnl a va ha')
    · intro a va ha' hla; exact absurd hla (hnl a va ha')
    · intro _; simp [dec, hg, hc1]
    · intro a va ha' hla; exact absurd hla (hnl a va ha')
    · intro ho; simp [hown] at ho
    · intro ho; simp [hown] at ho
    · intro h0; exact absurd h0 hne
    · intro _ _ j hj f hf
      have hlen := h.len
      by_cases hjc : j < w.cursor
      · exact dget_setAt_mono _ _ _ _ (hfull j hjc f hf)
      · have : j = w.cursor := by omega
        subst this
        exact dget_setAt_hit _ _ _ (by omega) (hat ▸ hf)
  · -- middle batch: write it, advance the cursor; `w` stays the only live worker
    have hl : Live m.p w := ⟨ha, hg⟩
    have hc1 := h.cur_live i w hw hl
    obtain ⟨hat, hfull⟩ := h.live_ok i w hw hl
    have hown : m.owed = false := by
      cases ho : m.owed
      · rfl
      · exact absurd hl (h.owed_none ho i w hw)
    have hne : m.p.fields ≠ [] := fun h0 => by rw [h.empty h0] at hw; simp at hw
    have hil : i < m.p.workers.length := lt_of_getElem?_some hw
    have honly : ∀ j v, (m.p.workers.set i { w with cursor := w.cursor + 1 })[j]? = some v →
        Live m.p v → j = i ∧ v = { w with cursor := w.cursor + 1 } := by
      intro j v hv hlv
      rcases getElem?_set_cases hv with ⟨rfl, rfl⟩ | ⟨hji, hv⟩
      · exact ⟨rfl, rfl⟩
      · exact absurd (h.uniq j i v w hv hw hlv hl) hji
    rw [hb]
    refine ⟨by simp [length_setAt, h.len], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro j v hv
      rcases getElem?_set_cases hv with ⟨rfl, rfl⟩ | ⟨-, hv⟩
      · exact h.gen_le _ w hw
      · exact h.gen_le _ _ hv
    · intro a b va vb ha' hb' hla hlb
      rw [(honly a va ha' hla).1, (honly b vb hb' hlb).1]
    · intro _ _ _ _; exact hc1
    · intro hn; exact (hn i { w with cursor := w.cursor + 1 } (by simp [hil]) ⟨ha, hg⟩).elim
    · intro j v hv hlv
      obtain ⟨rfl, rfl⟩ := honly j v hv hlv
      refine ⟨hat, fun k hk f hf => ?_⟩
      by_cases hkc : k < w.cursor
      · exact dget_setAt_mono _ _ _ _ (hfull k hkc f hf)
      · have : k = w.cursor := by simp at hk; omega
        subst this
        exact dget_setAt_hit _ _ _ (by rw [h.len]; omega) (hat ▸ hf)
    · intro ho; simp [hown] at ho
    · intro ho; simp [hown] at ho
    · intro h0; exact absurd h0 hne
    · intro _ hn; exact (hn i { w with cursor := w.cursor + 1 } (by simp [hil]) ⟨ha, hg⟩).elim

theorem inv_create {nb : Nat} {m : MState} (h : MInv nb m) (_ho : m.owed = false) (a : String) :
    MInv nb ⟨store m.p a, true⟩ := by
  by_cases h0 : m.p.fields = []
  · have hw := h.empty h0
    have hc := h.cur_none (by intro j v hv; simp [hw] at hv)
    refine ⟨by simp [store, h.len], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      simp [store, hw, h0, hc, NoLive]
  · have hnl : NoLive (store m.p a) := by
      intro j v hv hl
      have := h.gen_le j v (by simpa [store] using hv)
      simp [Live, store, h0, bump] at hl; omega
    refine ⟨by simp [store, h.len], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro j v hv
      have := h.gen_le j v (by simpa [store] using hv)
      simp [store, h0, bump]; omega
    · intro x y vx vy hx _ hlx _; exact absurd hlx (hnl x vx hx)
    · intro x vx hx hlx; exact absurd hlx (hnl x vx hx)
    · intro _; simp [store, h0, bump]
    · intro x vx hx hlx; exact absurd hlx (hnl x vx hx)
    · intro _; exact hnl
    · intro _; simp [store]
    · intro h1; simp [store] at h1
    · intro h1; simp at h1

theorem inv_populate {nb : Nat} {m : MState} (h : MInv nb m) (ho : m.owed = true) :
    MInv nb ⟨acquire m.p, false⟩ := by
  have hn := h.owed_none ho
  have hc := h.cur_none hn
  have hne := h.owed_fields ho
  let nw : Worker := { gen := m.p.slots.gen, attrs := m.p.fields, cursor := 0 }
  have hws : (acquire m.p).workers = m.p.workers ++ [nw] := rfl
  have hgen : (acquire m.p).slots.gen = m.p.slots.gen := by simp [acquire, inc]
  have hcur : (acquire m.p).slots.cur = 1 := by simp [acquire, inc, hc]
  have hnew : (acquire m.p).workers[m.p.workers.length]? = some nw := by
    rw [hws, List.getElem?_append_right (by omega)]; simp
  have honly : ∀ j v, (acquire m.p).workers[j]? = some v → Live (acquire m.p) v →
      j = m.p.workers.length ∧ v = nw := by
    intro j v hv hl
    rw [hws] at hv
    rcases getElem?_append_one hv with hv | hv
    · exact absurd (by simpa [Live, hgen] using hl) (hn j v hv)
    · exact hv
  have hlnw : Live (acquire m.p) nw := ⟨rfl, hgen.symm⟩
  refine ⟨by simp [acquire, h.len], ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro j v hv
    rw [hgen]; rw [hws] at hv
    rcases getElem?_append_one hv with hv | ⟨-, rfl⟩
    · exact h.gen_le _ _ hv
    · exact Nat.le_refl _
  · intro a b va vb ha hb hla hlb; rw [(honly a va ha hla).1, (honly b vb hb hlb).1]
  · intro _ _ _ _; exact hcur
  · intro hn'; exact absurd hlnw (hn' _ _ hnew)
  · intro j v hv hl
    obtain ⟨-, rfl⟩ := honly j v hv hl
    exact ⟨rfl, fun k hk => absurd hk (Nat.not_lt_zero _)⟩
  · intro h1; simp at h1
  · intro h1; simp at h1
  · intro h1; exact absurd h1 (by simpa [acquire] using hne)
  · intro _ hn'; exact absurd hlnw (hn' _ _ hnew)

theorem reach_inv {nb : Nat} {m : MState} (h : Reach nb m) : MInv nb m := by
  induction h with
  | init => exact inv_init nb
  | step e _ hs ih =>
    cases e with
    | create a =>
      simp only [step] at hs; split at hs
      · cases hs
      · next ho => cases hs; exact inv_create ih (by simpa using ho) a
    | populate =>
      simp only [step] at hs; split at hs
      · next ho => cases hs; exact inv_populate ih ho
      · cases hs
    | batch i => simp only [step] at hs; cases hs; exact inv_batch ih i

/-- **#3086, main never reports an unpopulated field as OPERATIONAL.** Over every
interleaving of `CREATE INDEX` on one label (any number, any attributes) and the
batches of every populate job, whenever no creation is between its store and its
`populate_index`, an operational index has every field in every batch. -/
theorem main_operational_complete {nb : Nat} {m : MState} (h : Reach nb m)
    (ho : m.owed = false) (hop : operational m.p = true) : complete m.p = true := by
  have hi := reach_inv h
  have hc : m.p.slots.cur = 0 := by simpa [operational] using hop
  have hn : NoLive m.p := fun j v hv hl => by have := hi.cur_live j v hv hl; omega
  have hd := hi.done ho hn
  unfold complete
  rw [List.all_eq_true]
  intro d hdm
  obtain ⟨j, hj⟩ := List.mem_iff_getElem?.mp hdm
  have hjl : j < nb := by
    rw [← hi.len]; exact lt_of_getElem?_some hj
  rw [List.all_eq_true]
  intro f hf
  have := hd j hjl f hf
  simp only [dget, hj, Option.getD_some] at this
  exact List.contains_iff_mem.mpr this

/-- The interleaving that lost `b` before #3086 (`pre3086_lostSecondIndex`), on
main: P2 runs first and no longer bails, P1 fails `is_ticket_current`. -/
def mainLostOrder : PState :=
  let s := acquire (store (init 3) "a")
  let s := batch 3 s 0            -- P1 batch 0 with {a}
  let s := acquire (store s "b")  -- bump, then P2 with {a,b}: pending of the new generation = 1
  let s := batch 3 s 1            -- P2 batch 0
  let s := batch 3 s 0            -- P1: stale ticket, bails
  let s := batch 3 s 1
  batch 3 s 1

theorem main_lost_order_complete :
    operational mainLostOrder = true ∧ complete mainLostOrder = true ∧
    mainLostOrder.docs = [["a", "a", "b"], ["a", "b"], ["a", "b"]] := by decide

/-- The hypothesis `owed = false` is needed: between `Indexer::create_index`
storing the entry (bumped, pending 0) and `populate_index` acquiring its ticket,
the entry is OPERATIONAL with the new field in no document. A concurrent reader's
`is_label_indexed` (indexer.rs:454) can plan an index scan in that window. -/
def storeWindow : PState :=
  let s := acquire (store (init 3) "a")
  let s := batch 3 (batch 3 (batch 3 s 0) 0) 0
  store s "b"

theorem store_window_operational_incomplete :
    operational storeWindow = true ∧ complete storeWindow = false := by decide

end SC.Tickets
