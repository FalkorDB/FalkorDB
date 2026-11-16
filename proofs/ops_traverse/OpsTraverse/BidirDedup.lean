import OpsTraverse.Basic
/-
# The bidirectional dedup of chained anonymous undirected hops

## Before #3110 (historical, W2-traverse-2)

`dedupRows`, `dedupBatches` and the three counterexamples below model the code *before*
PR #3110 (fixes #3106). They are kept as the record of W2-traverse-2; the code they describe
is gone. The current code is modelled in `Lin` at the end of this file.

| here | there (pre-#3110, base 343e077f2) |
| --- | --- |
| `dedupRows`    | the `bidir_dedup` loop of `CondTraverseOp::expand_row` (`cond_traverse.rs:1087-1110`): key `(src_key, out.to)`, `swap_remove` on a repeat |
| `dedupBatches` | `seen.clear()` on every new child batch (`cond_traverse.rs:1467-1469`) |
| key source     | `dedup_source_alias` = the *child* CT's `from` alias (`cond_traverse.rs:397-407`) |

A result row is `(outer, key source, destination)`; `outer` stands for every
other column of the row (an `UNWIND` variable, an earlier MATCH's node, or the
real scan source when the key source is an intermediate node). The intended
semantics (C FalkorDB's algebraic expression) is: one row per distinct *row*
`(outer, source, destination)`. The model shows the three ways Rust's key
diverges from it. (`swap_remove` reorders the survivors; order is not modelled,
only membership and multiplicity.)
-/

namespace OpsTraverse.BidirDedup

abbrev Row := Nat × Nat × Nat   -- (outer, keySource, dest)

def key (r : Row) : Nat × Nat := (r.2.1, r.2.2)

def dedupRows : List (Nat × Nat) → List Row → List Row
  | _, [] => []
  | seen, r :: rs =>
    if key r ∈ seen then dedupRows seen rs else r :: dedupRows (key r :: seen) rs

def dedupBatches (bs : List (List Row)) : List Row := bs.flatMap (dedupRows [])

theorem dedupRows_sub : ∀ (seen : List (Nat × Nat)) (rs : List Row) (r : Row),
    r ∈ dedupRows seen rs → r ∈ rs
  | _, [], _, h => by simp [dedupRows] at h
  | seen, x :: rs, r, h => by
    unfold dedupRows at h
    split at h
    · exact List.mem_cons_of_mem _ (dedupRows_sub _ _ _ h)
    · rcases List.mem_cons.1 h with rfl | h
      · exact List.mem_cons_self ..
      · exact List.mem_cons_of_mem _ (dedupRows_sub _ _ _ h)

theorem dedupRows_key_fresh : ∀ (seen : List (Nat × Nat)) (rs : List Row) (r : Row),
    r ∈ dedupRows seen rs → key r ∉ seen
  | _, [], _, h => by simp [dedupRows] at h
  | seen, x :: rs, r, h => by
    unfold dedupRows at h
    split at h
    · exact dedupRows_key_fresh _ _ _ h
    · rename_i hx
      rcases List.mem_cons.1 h with rfl | h
      · exact hx
      · have := dedupRows_key_fresh _ _ _ h
        exact fun hm => this (List.mem_cons_of_mem _ hm)

/-- Within one batch the surviving keys are pairwise distinct. -/
theorem dedupRows_keys_nodup : ∀ (seen : List (Nat × Nat)) (rs : List Row),
    ((dedupRows seen rs).map key).Nodup
  | _, [] => by simp [dedupRows]
  | seen, x :: rs => by
    unfold dedupRows
    split
    · exact dedupRows_keys_nodup _ _
    · simp only [List.map_cons, List.nodup_cons, List.mem_map]
      refine ⟨?_, dedupRows_keys_nodup _ _⟩
      rintro ⟨r, hr, hk⟩
      exact dedupRows_key_fresh _ _ _ hr (hk ▸ List.mem_cons_self ..)

/-- A key-complete dedup never loses a row: every input key is still present. -/
theorem dedupRows_keys_complete : ∀ (seen : List (Nat × Nat)) (rs : List Row) (r : Row),
    r ∈ rs → key r ∉ seen → ∃ r' ∈ dedupRows seen rs, key r' = key r
  | _, [], _, h, _ => by simp at h
  | seen, x :: rs, r, h, hs => by
    unfold dedupRows
    split
    · rename_i hx
      rcases List.mem_cons.1 h with rfl | h
      · exact absurd hx hs
      · exact dedupRows_keys_complete _ _ _ h hs
    · by_cases hk : key r = key x
      · exact ⟨x, List.mem_cons_self .., hk.symm⟩
      · rcases List.mem_cons.1 h with rfl | h
        · exact absurd rfl hk
        · have hs' : key r ∉ key x :: seen := by
            simp only [List.mem_cons, not_or]; exact ⟨hk, hs⟩
          obtain ⟨r', hr', he⟩ := dedupRows_keys_complete _ _ _ h hs'
          exact ⟨r', List.mem_cons_of_mem _ hr', he⟩

/-! ## Counterexamples (each reproduced against the engine)

The `lean_ops_traverse::bug_*` Rust tests cited below are not in this PR: as for every area
(see the FINDINGS.md header) they were run locally against the real engine and are not
committed, because many `bug_*` tests fail on purpose. The Rust and C counts quoted are from
those runs; the tests land with their fixes. -/

/-- (a) Rows that differ only in an outer column collapse:
`UNWIND [1,2] AS x MATCH (a)-[]-()-[]-(b)` loses every `x = 2` row.
`lean_ops_traverse::bug_bidir_dedup_drops_rows_differing_in_outer_column`
(Rust 5 rows, C 10). -/
theorem drops_outer_rows :
    dedupRows [] [(1, 1, 3), (2, 1, 3)] = [(1, 1, 3)] := by decide

/-- (b) With three hops the key source is the intermediate node, so start
nodes 1 and 4 reaching destination 3 through the same intermediate 2 collide.
`lean_ops_traverse::bug_bidir_dedup_three_hops_keys_on_intermediate` (Rust 7 rows, C 12). -/
theorem three_hops_collide :
    -- rows are (start, intermediate, dest)
    dedupRows [] [(1, 2, 3), (4, 2, 3)] = [(1, 2, 3)] := by decide

/-- (c) Clearing `seen` per input batch makes the answer depend on where the
child's output was cut into `BATCH_SIZE` batches.
`lean_ops_traverse::bug_bidir_dedup_depends_on_batch_size` (Rust count 3/4, C 1). -/
theorem batch_split_changes_result :
    dedupBatches [[(0, 1, 1), (0, 1, 1)]] ≠ dedupBatches [[(0, 1, 1)], [(0, 1, 1)]] := by
  decide


/-! ## Since #3110 (fixes #3106): the dedup is keyed on the record that entered the chain

| here | there |
| --- | --- |
| `Lin.St`, `Lin.init` | `struct BidirDedup { record, seen }`, `#[derive(Default)]` (`cond_traverse.rs:166-170`) |
| `Lin.enter` | `BidirDedup::enter` (`cond_traverse.rs:175-186`), `CLEAR_AT = 4096` (`:173`) |
| `Lin.filterDests`, `Lin.dedupRow` | the dedup block of `expand_row` (`cond_traverse.rs:1146-1161`): skipped when the row produced nothing (`out.len() > start`), `enter(record)`, then keep `out[i]` iff `seen.insert((record, source, to))` |
| `Lin.run` | that block over every input row the CT expands, in order: the state is never cleared between child batches (`next`, `:1571-1587` only installs `in_lineage`) |
| `Lin.runBatches` | the same stream cut into child batches |
| `Lin.stampStart`, `Lin.stampFwd` | `CondTraverseOp::stamp_lineage` (`cond_traverse.rs:1375-1400`): a chain start numbers its seeded rows from `lineage_base` (`:1584-1587`), a dedup CT forwards `in_lineage` |

`seen` (an `FxHashSet`) is a list with no duplicate (`filterDests` only adds a key it does not
hold), so its length is the set's `len()`. `swap_remove` reorders the survivors of one row; order
is not modelled, only which destinations survive.

`run_spec`: when the rows of one record arrive contiguously (the Rust comment's premise),
the dedup keeps every offered `(record, source, destination)` key exactly once, whatever the
batch cut (`runBatches_eq`) and whether `seen` was cleared at a record change. `records_blocks`
discharges the premise from the record sequence a chain start produces: the emitter emits each
parent row's results contiguously and in row order (proofs/columnar `Emitter.emitAll_spec`,
proofs/runtime_core `Emitter.drive_flatten`), and `stampStart_spec` numbers them `base + row`,
distinct rows giving distinct records (`lineage_fresh` across batches).
-/
namespace Lin

abbrev Key := Nat × Nat × Nat      -- (record, source, destination)
abbrev Lineage := Nat × Nat        -- (record, source)
abbrev Row := Lineage × List Nat   -- an input row's lineage and the `to` endpoints it produced

def CLEAR_AT : Nat := 4096

structure St where
  record : Nat
  seen : List Key

def init : St := ⟨0, []⟩

/-- `BidirDedup::enter` (`cond_traverse.rs:175-186`). -/
def enter (s : St) (record : Nat) : St :=
  if record ≠ s.record then
    { record, seen := if s.seen.length ≥ CLEAR_AT then [] else s.seen }
  else s

/-- The `while i < out.len()` loop: keep a destination iff its key was newly inserted. -/
def filterDests (rec src : Nat) : List Key → List Nat → List Key × List Nat
  | seen, [] => (seen, [])
  | seen, d :: ds =>
    if (rec, src, d) ∈ seen then filterDests rec src seen ds
    else
      let r := filterDests rec src ((rec, src, d) :: seen) ds
      (r.1, d :: r.2)

/-- The dedup block of `expand_row` for one input row with lineage `lin` that produced the
destinations `ds` (`cond_traverse.rs:1146-1161`). -/
def dedupRow (s : St) (lin : Lineage) (ds : List Nat) : St × List Nat :=
  if ds = [] then (s, [])
  else
    let s1 := enter s lin.1
    let r := filterDests lin.1 lin.2 s1.seen ds
    (⟨s1.record, r.1⟩, r.2)

def keysOf (lin : Lineage) (ds : List Nat) : List Key := ds.map fun d => (lin.1, lin.2, d)

def run : St → List Row → St × List Key
  | s, [] => (s, [])
  | s, x :: xs =>
    let r := dedupRow s x.1 x.2
    let t := run r.1 xs
    (t.1, keysOf x.1 r.2 ++ t.2)

def runBatches : St → List (List Row) → St × List Key
  | s, [] => (s, [])
  | s, b :: bs =>
    let r := run s b
    let t := runBatches r.1 bs
    (t.1, r.2 ++ t.2)

/-- Every key the rows offer, in order. -/
def cand (xs : List Row) : List Key := xs.flatMap fun x => keysOf x.1 x.2

/-- Reference: the first occurrence of each key (C's `F · (A+Aᵀ) · …` with one row of `F` per
record: one result per `(record, destination)`). -/
def nubK : List Key → List Key → List Key × List Key
  | acc, [] => (acc, [])
  | acc, k :: ks =>
    if k ∈ acc then nubK acc ks
    else
      let r := nubK (k :: acc) ks
      (r.1, k :: r.2)

theorem nubK_spec : ∀ (acc ks : List Key),
    (nubK acc ks).2.Nodup ∧ (∀ k, k ∈ (nubK acc ks).2 ↔ k ∈ ks ∧ k ∉ acc) ∧
    (∀ k, k ∈ (nubK acc ks).1 ↔ k ∈ acc ∨ k ∈ ks)
  | acc, [] => by simp [nubK]
  | acc, k :: ks => by
    unfold nubK
    split
    · rename_i h
      obtain ⟨h1, h2, h3⟩ := nubK_spec acc ks
      refine ⟨h1, fun x => ?_, fun x => ?_⟩
      · rw [h2]; by_cases hx : x = k
        · subst hx; simp [h]
        · simp [hx]
      · rw [h3]; by_cases hx : x = k
        · subst hx; simp [h]
        · simp [hx]
    · rename_i h
      obtain ⟨h1, h2, h3⟩ := nubK_spec (k :: acc) ks
      refine ⟨?_, fun x => ?_, fun x => ?_⟩
      · refine List.nodup_cons.2 ⟨fun hm => ?_, h1⟩
        exact ((h2 k).1 hm).2 (List.mem_cons_self ..)
      · simp only [List.mem_cons, h2]
        by_cases hx : x = k
        · subst hx; simp [h]
        · simp [hx]
      · simp only [h3, List.mem_cons]
        by_cases hx : x = k <;> simp [hx]

theorem nubK_append : ∀ (acc a b : List Key),
    nubK acc (a ++ b) = ((nubK (nubK acc a).1 b).1, (nubK acc a).2 ++ (nubK (nubK acc a).1 b).2)
  | acc, [], b => by simp [nubK]
  | acc, k :: a, b => by
    simp only [List.cons_append, nubK]
    split
    · exact nubK_append acc a b
    · rw [nubK_append (k :: acc) a b]; simp

theorem run_append : ∀ (s : St) (a b : List Row),
    run s (a ++ b) = ((run (run s a).1 b).1, (run s a).2 ++ (run (run s a).1 b).2)
  | s, [], b => by simp [run]
  | s, x :: a, b => by
    simp only [List.cons_append, run]
    rw [run_append _ a b]; simp

/-- Child batch boundaries do not matter: the state is threaded through them. -/
theorem runBatches_eq : ∀ (s : St) (bs : List (List Row)), runBatches s bs = run s bs.flatten
  | s, [] => by simp [runBatches, run]
  | s, b :: bs => by
    simp only [runBatches, List.flatten_cons]
    rw [run_append, runBatches_eq]

theorem cand_append (a b : List Row) : cand (a ++ b) = cand a ++ cand b := by
  simp [cand]

theorem mem_cand {k : Key} {xs : List Row} (h : k ∈ cand xs) : ∃ x ∈ xs, k.1 = x.1.1 := by
  simp only [cand, keysOf, List.mem_flatMap, List.mem_map] at h
  obtain ⟨x, hx, d, -, rfl⟩ := h
  exact ⟨x, hx, rfl⟩

theorem enter_spec (s : St) (r : Nat) :
    (enter s r).record = r ∧ (∀ x ∈ (enter s r).seen, x ∈ s.seen) ∧ (s.record = r → enter s r = s) := by
  unfold enter
  by_cases h : r = s.record
  · simp [h]
  · split
    · refine ⟨rfl, fun x hx => ?_, fun h' => absurd h'.symm h⟩
      split at hx
      · simp at hx
      · exact hx
    · rename_i hc; exact absurd h hc

theorem filterDests_spec (rec src : Nat) : ∀ (acc seen : List Key) (ds : List Nat),
    (∀ x ∈ seen, x ∈ acc) → (∀ x : Key, x.1 = rec → x ∈ acc → x ∈ seen) →
    keysOf (rec, src) (filterDests rec src seen ds).2 = (nubK acc (keysOf (rec, src) ds)).2 ∧
    (∀ x ∈ (filterDests rec src seen ds).1, x ∈ (nubK acc (keysOf (rec, src) ds)).1) ∧
    (∀ x : Key, x.1 = rec → x ∈ (nubK acc (keysOf (rec, src) ds)).1 →
      x ∈ (filterDests rec src seen ds).1)
  | acc, seen, [], h1, h2 => by simp only [filterDests, keysOf, List.map_nil, nubK]; exact ⟨by first | rfl | trivial, h1, h2⟩
  | acc, seen, d :: ds, h1, h2 => by
    have hk : (rec, src, d) ∈ seen ↔ (rec, src, d) ∈ acc := ⟨h1 _, h2 _ rfl⟩
    have ek : keysOf (rec, src) (d :: ds) = (rec, src, d) :: keysOf (rec, src) ds := rfl
    rw [ek]
    by_cases hs : (rec, src, d) ∈ seen
    · simp only [filterDests, hs, ite_true, nubK, hk.1 hs]
      exact filterDests_spec rec src acc seen ds h1 h2
    · have ha : (rec, src, d) ∉ acc := fun h => hs (hk.2 h)
      simp only [filterDests, hs, ite_false, nubK, ha]
      have h1' : ∀ x ∈ (rec, src, d) :: seen, x ∈ (rec, src, d) :: acc := by
        intro x hx
        rcases List.mem_cons.1 hx with rfl | hx
        · exact List.mem_cons_self ..
        · exact List.mem_cons_of_mem _ (h1 x hx)
      have h2' : ∀ x : Key, x.1 = rec → x ∈ (rec, src, d) :: acc → x ∈ (rec, src, d) :: seen := by
        intro x hr hx
        rcases List.mem_cons.1 hx with rfl | hx
        · exact List.mem_cons_self ..
        · exact List.mem_cons_of_mem _ (h2 x hr hx)
      obtain ⟨e1, e2, e3⟩ := filterDests_spec rec src _ _ ds h1' h2'
      refine ⟨?_, e2, e3⟩
      have : keysOf (rec, src) (d :: (filterDests rec src ((rec, src, d) :: seen) ds).2) =
          (rec, src, d) :: keysOf (rec, src) (filterDests rec src ((rec, src, d) :: seen) ds).2 := rfl
      rw [this, e1]

/-- Within the block of record `r`: `seen` holds only accepted keys, and every accepted key of
`r` (the current record, once one was accepted). -/
def Inv (r : Nat) (s : St) (acc : List Key) : Prop :=
  (∀ x ∈ s.seen, x ∈ acc) ∧ (∀ x : Key, x.1 = r → x ∈ acc → x ∈ s.seen) ∧
  (∀ x ∈ acc, x.1 = r → s.record = r)

theorem dedupRow_spec (r : Nat) (s : St) (acc : List Key) (x : Row) (hi : Inv r s acc)
    (hr : x.1.1 = r) :
    keysOf x.1 (dedupRow s x.1 x.2).2 = (nubK acc (keysOf x.1 x.2)).2 ∧
    Inv r (dedupRow s x.1 x.2).1 (nubK acc (keysOf x.1 x.2)).1 := by
  obtain ⟨⟨rec, src⟩, ds⟩ := x
  simp only at hr; subst hr
  obtain ⟨i1, i2, i3⟩ := hi
  unfold dedupRow
  by_cases hd : ds = []
  · subst hd; simp only [ite_true, keysOf, List.map_nil, nubK]; exact ⟨by first | rfl | trivial, i1, i2, i3⟩
  · simp only [hd, ite_false]
    obtain ⟨er, es, ei⟩ := enter_spec s rec
    have e1 : ∀ y ∈ (enter s rec).seen, y ∈ acc := fun y hy => i1 y (es y hy)
    have e2 : ∀ y : Key, y.1 = rec → y ∈ acc → y ∈ (enter s rec).seen := by
      intro y hy hya
      rw [ei (i3 y hya hy)]; exact i2 y hy hya
    obtain ⟨f1, f2, f3⟩ := filterDests_spec rec src acc _ ds e1 e2
    exact ⟨f1, f2, f3, fun _ _ _ => er⟩

theorem run_block (r : Nat) : ∀ (s : St) (acc : List Key) (xs : List Row),
    Inv r s acc → (∀ x ∈ xs, x.1.1 = r) →
    (run s xs).2 = (nubK acc (cand xs)).2 ∧ Inv r (run s xs).1 (nubK acc (cand xs)).1
  | s, acc, [], hi, _ => by simp only [run, cand, List.flatMap_nil, nubK]; exact ⟨by first | rfl | trivial, hi⟩
  | s, acc, x :: xs, hi, hr => by
    have ec : cand (x :: xs) = keysOf x.1 x.2 ++ cand xs := by simp [cand]
    obtain ⟨d1, d2⟩ := dedupRow_spec r s acc x hi (hr x (List.mem_cons_self ..))
    obtain ⟨r1, r2⟩ := run_block r _ _ xs d2 (fun y hy => hr y (List.mem_cons_of_mem _ hy))
    simp only [run, ec, nubK_append]
    exact ⟨by rw [d1, r1], r2⟩

theorem run_blocks : ∀ (blocks : List (Nat × List Row)) (s : St) (acc : List Key) (done : List Nat),
    (∀ x ∈ s.seen, x ∈ acc) → (∀ x ∈ acc, x.1 ∈ done) →
    (∀ b ∈ blocks, b.1 ∉ done) → (∀ b ∈ blocks, ∀ x ∈ b.2, x.1.1 = b.1) →
    (blocks.map Prod.fst).Nodup →
    (run s (blocks.flatMap Prod.snd)).2 = (nubK acc (cand (blocks.flatMap Prod.snd))).2
  | [], s, acc, done, _, _, _, _, _ => by simp [run, cand, nubK]
  | (r, xs) :: bs, s, acc, done, g1, g2, hdone, hb, hnd => by
    simp only [List.flatMap_cons]
    rw [run_append, cand_append, nubK_append]
    have hrd : r ∉ done := hdone _ (List.mem_cons_self ..)
    have hi : Inv r s acc :=
      ⟨g1, fun x hx hxa => absurd (hx ▸ g2 x hxa) hrd, fun x hxa hx => absurd (hx ▸ g2 x hxa) hrd⟩
    obtain ⟨b1, b2, -, -⟩ := run_block r s acc xs hi (hb _ (List.mem_cons_self ..))
    dsimp only
    rw [b1]
    simp only [List.map_cons, List.nodup_cons, List.mem_map] at hnd
    obtain ⟨hn1, hn2⟩ := hnd
    refine congrArg _ (run_blocks bs _ _ (r :: done) b2 (fun x hx => ?_) (fun b hb' => ?_)
      (fun b hb' => hb b (List.mem_cons_of_mem _ hb')) hn2)
    · rcases ((nubK_spec acc (cand xs)).2.2 x).1 hx with hx | hx
      · exact List.mem_cons_of_mem _ (g2 x hx)
      · obtain ⟨y, hy, he⟩ := mem_cand hx
        rw [he, hb _ (List.mem_cons_self ..) y hy]; exact List.mem_cons_self ..
    · simp only [List.mem_cons, not_or]
      exact ⟨fun h => hn1 ⟨b, hb', h⟩, hdone b (List.mem_cons_of_mem _ hb')⟩

/-- **The lineage dedup is exact** (C's per-record matrix product): when the input rows come
in blocks of one record each, distinct across blocks, the surviving keys are every offered
`(record, source, destination)` exactly once. -/
theorem run_spec (blocks : List (Nat × List Row)) (hb : ∀ b ∈ blocks, ∀ x ∈ b.2, x.1.1 = b.1)
    (hnd : (blocks.map Prod.fst).Nodup) :
    (run init (blocks.flatMap Prod.snd)).2.Nodup ∧
    ∀ k, k ∈ (run init (blocks.flatMap Prod.snd)).2 ↔ k ∈ cand (blocks.flatMap Prod.snd) := by
  rw [run_blocks blocks init [] [] (by simp [init]) (by simp) (by simp) hb hnd]
  obtain ⟨h1, h2, -⟩ := nubK_spec [] (cand (blocks.flatMap Prod.snd))
  exact ⟨h1, fun k => by rw [h2]; simp⟩

/-- The same across child batches. -/
theorem runBatches_spec (bs : List (List Row)) (blocks : List (Nat × List Row))
    (he : bs.flatten = blocks.flatMap Prod.snd)
    (hb : ∀ b ∈ blocks, ∀ x ∈ b.2, x.1.1 = b.1) (hnd : (blocks.map Prod.fst).Nodup) :
    (runBatches init bs).2.Nodup ∧
    ∀ k, k ∈ (runBatches init bs).2 ↔ k ∈ cand bs.flatten := by
  rw [runBatches_eq, he]; exact run_spec blocks hb hnd

/-- **The premise holds for the rows a chain start emits**: a record sequence made of runs of
distinct records (one run per parent row, in emitter order) splits into such blocks. -/
theorem records_blocks : ∀ (ps : List (Nat × Nat)) (xs : List Row),
    xs.map (·.1.1) = ps.flatMap (fun p => List.replicate p.2 p.1) →
    ∃ blocks : List (Nat × List Row), xs = blocks.flatMap Prod.snd ∧
      (∀ b ∈ blocks, ∀ x ∈ b.2, x.1.1 = b.1) ∧ blocks.map Prod.fst = ps.map Prod.fst
  | [], xs, h => ⟨[], by simpa using h, by simp, by simp⟩
  | (r, n) :: ps, xs, h => by
    simp only [List.flatMap_cons] at h
    have ht : (xs.take n).map (·.1.1) = List.replicate n r := by
      rw [List.map_take, h, List.take_left']; simp
    have hdr : (xs.drop n).map (·.1.1) = ps.flatMap (fun p => List.replicate p.2 p.1) := by
      rw [List.map_drop, h, List.drop_left']; simp
    obtain ⟨bs, e, hb, hn⟩ := records_blocks ps (xs.drop n) hdr
    refine ⟨(r, xs.take n) :: bs, ?_, ?_, by simp [hn]⟩
    · simp only [List.flatMap_cons]; rw [← e, List.take_append_drop]
    · intro b hb'
      rcases List.mem_cons.1 hb' with rfl | hb'
      · intro x hx
        have : x.1.1 ∈ List.replicate n r := ht ▸ List.mem_map_of_mem hx
        exact List.eq_of_mem_replicate this
      · exact hb b hb'

/-- **Discharged form**: rows whose records are runs of distinct records (what a chain start
emits, see `records_blocks`) are deduplicated to every offered key exactly once, across any
cut into child batches. -/
theorem dedup_exact (ps : List (Nat × Nat)) (hnd : (ps.map Prod.fst).Nodup) (bs : List (List Row))
    (h : bs.flatten.map (·.1.1) = ps.flatMap (fun p => List.replicate p.2 p.1)) :
    (runBatches init bs).2.Nodup ∧
    ∀ k, k ∈ (runBatches init bs).2 ↔ k ∈ cand bs.flatten := by
  obtain ⟨blocks, e, hb, hm⟩ := records_blocks ps _ h
  exact runBatches_spec bs blocks e hb (hm ▸ hnd)

/-! ### `stamp_lineage` (`cond_traverse.rs:1375-1400`) -/

def U64_MAX : Nat := 2 ^ 64 - 1

/-- Chain start: output row `i` gathered from seeded row `parentRows[i]` gets record
`lineage_base + parentRows[i]` and the `from` node bound on it (`u64::MAX` when not a node). -/
def stampStart (base : Nat) (fromNode : Nat → Option Nat) (parentRows : List Nat) : List Lineage :=
  parentRows.mapIdx fun i r => (base + r, (fromNode i).getD U64_MAX)

/-- A dedup CT forwards the lineage of the seeded row each output row came from. -/
def stampFwd (inLin : List Lineage) (parentRows : List Nat) : List Lineage :=
  parentRows.map fun r => inLin.getD r (0, 0)

theorem stampStart_spec (base : Nat) (fromNode : Nat → Option Nat) (rows : List Nat) :
    (stampStart base fromNode rows).length = rows.length ∧
    ∀ i (h : i < rows.length),
      (stampStart base fromNode rows)[i]'(by simp [stampStart, h]) =
        (base + rows[i], (fromNode i).getD U64_MAX) := by
  refine ⟨by simp [stampStart], fun i h => ?_⟩
  simp [stampStart]

theorem stampFwd_spec (inLin : List Lineage) (rows : List Nat) (hin : ∀ r ∈ rows, r < inLin.length) :
    (stampFwd inLin rows).length = rows.length ∧
    ∀ i (h : i < rows.length),
      (stampFwd inLin rows)[i]'(by simp [stampFwd, h]) =
        inLin[rows[i]]'(hin _ (List.getElem_mem h)) := by
  refine ⟨by simp [stampFwd], fun i h => ?_⟩
  simp [stampFwd, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (hin _ (List.getElem_mem h))]

/-- The records of a chain start's seeded batch are `[base, base + len)`; the next batch starts
at `base + len` (`lineage_next += b.len()`), so records of different batches never collide. -/
theorem lineage_fresh (base len r1 r2 : Nat) (h : r1 < len) : base + r1 ≠ (base + len) + r2 := by
  omega

/-! ### The W2-traverse-2 inputs on the new model: C's answers -/

/-- `UNWIND [1,2] AS x MATCH (a)-[]-()-[]-(b)`: two records reaching the same `(a, b)` both
keep their row (pre-#3110: `drops_outer_rows`). -/
theorem outer_rows_kept :
    (run init [((0, 1), [3]), ((1, 1), [3])]).2 = [(0, 1, 3), (1, 1, 3)] := by decide

/-- Three hops: start nodes 1 and 4 (records 0 and 1) reach 3 through the same intermediate,
and both rows stay (pre-#3110: `three_hops_collide`). -/
theorem three_hops_keyed_on_start :
    (run init [((0, 1), [3]), ((1, 4), [3])]).2 = [(0, 1, 3), (1, 4, 3)] := by decide

/-- One record whose duplicates are split across child batches is still deduplicated
(pre-#3110: `batch_split_changes_result`). -/
theorem batch_split_same :
    (runBatches init [[((0, 1), [1]), ((0, 1), [1])]]).2 =
      (runBatches init [[((0, 1), [1])], [((0, 1), [1])]]).2 := by decide

end Lin

end OpsTraverse.BidirDedup
