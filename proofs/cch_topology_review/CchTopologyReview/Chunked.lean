import CchTopologyReview.Basic

/-!
# `WeightArr`: copy-on-write at two levels, with `Arc::make_mut` modelled (MVCC isolation, 03 §6b "MVCC")

`WeightArr` is on the stack, not on main: `origin/cch/12a-chunked-weights`
`graph/src/index/falkordb/cch/customize.rs:112-240`, identical at `cch/08c-build-catch-up` (03 cites
`customize.rs:182-189` @08c). 03 §2 builds `rank_of`, `node_of` (`ChunkedArr`) and the metric's four
arrays on it, "plus append".

| Rust (@12a) | Lean |
|---|---|
| `struct WeightArr { chunks: Arc<[Arc<[T]>]>, len }` (`:112-115`) | `WA` (`tab` = the table's address), `Heap` (`tbl`: tables, `chk`: chunks) |
| `CHUNK_BITS = 11`, `CHUNK_LEN = 1 << CHUNK_BITS` (`:126-127`) | `CHUNK_BITS`, `CHUNK_LEN` |
| `Index::index`: `chunks[slot >> CHUNK_BITS][slot & (CHUNK_LEN - 1)]` (`:158-163`) | `read` (`none` is the panic) |
| `set`: `Arc::make_mut(&mut self.chunks)[slot >> CHUNK_BITS]`, then `Arc::make_mut(chunk)[slot & …] = value` (`:182-189`) | `setW` = `mmTab`, index, `mmChunk`, write |
| `new_version` = `clone` (`:192-194`) | `newVersion` |
| dropping a version | `dropV` |
| append (design: 03 §2 "plus append", PR 2's `ChunkedArr`) | `appendW` |

**`Arc::make_mut`** (std): if the `Arc` is the only strong reference (and no `Weak` exists; there are
none here) it hands out the value in place; otherwise it clones the value into a new allocation, points
this `Arc` at it and drops one reference to the old one. `Arc::strong_count` is the number of `Arc`
handles to the allocation, so it is *derived* here, not stored: a table's handles are the versions that
hold it (`cntT`); a chunk's handles are the slots of live tables that hold it (`cntC`). Cloning a table
(`mmTab`'s copy branch) clones each `Arc<[T]>` in it, which the derived count sees as the new table's
slots. The only thing assumed is std's contract that `make_mut` clones exactly when the count is not 1.

A state is the heap, every live `WeightArr` value (`vs`; the written one first, which loses nothing
since versions are a multiset: `perm` reorders) and the live tables (`ts`). `WF` says the live tables are
exactly those some version holds, once each, and every address in use is below the allocator's next.

Theorems: `setW_spec`, `appendW_spec`, `newVersion_spec`, `dropV_spec`, `perm_wf` (and the halves
`setW_isolated`, `appendW_isolated`): every other version reads exactly what it read before, at every slot,
and `WF` is kept. `setW_spec` also gives the written version's reads: the new value at the slot and its old
values elsewhere. `appendW_spec` gives only the new length for the appended version; what it reads would need
a shape invariant (table length `⌈len / CHUNK_LEN⌉`, full chunks but the last), which is not modelled.
-/

namespace CchTopo.Chunked

open CchTopo

/-- `CHUNK_BITS` (`customize.rs:126` @12a). -/
def CHUNK_BITS : Nat := 11
/-- `CHUNK_LEN = 1 << CHUNK_BITS` (`:127`). -/
def CHUNK_LEN : Nat := 1 <<< CHUNK_BITS

theorem chunk_len_eq : CHUNK_LEN = 2 ^ CHUNK_BITS := by unfold CHUNK_LEN; rw [Nat.one_shiftLeft]

/-- `slot >> CHUNK_BITS`. -/
def cix (s : Nat) : Nat := s >>> CHUNK_BITS
/-- `slot & (CHUNK_LEN - 1)`. -/
def oix (s : Nat) : Nat := s &&& (CHUNK_LEN - 1)

theorem cix_eq (s : Nat) : cix s = s / CHUNK_LEN := by
  unfold cix; rw [Nat.shiftRight_eq_div_pow, chunk_len_eq]

theorem oix_eq (s : Nat) : oix s = s % CHUNK_LEN := by
  unfold oix; rw [chunk_len_eq, Nat.and_two_pow_sub_one_eq_mod]

/-- Different slots differ in chunk or in offset. -/
theorem slot_inj {s s' : Nat} (hc : cix s = cix s') (ho : oix s = oix s') : s = s' := by
  rw [cix_eq, cix_eq] at hc
  rw [oix_eq, oix_eq] at ho
  rw [← Nat.div_add_mod s CHUNK_LEN, ← Nat.div_add_mod s' CHUNK_LEN, hc, ho]

variable {T : Type}

/-- The allocations: tables (`Arc<[Arc<[T]>]>`, as the chunk addresses they hold) and chunks
    (`Arc<[T]>`), with the next fresh address of each. -/
structure Heap (T : Type) where
  tbl : Nat → List Nat
  chk : Nat → List T
  nT : Nat
  nC : Nat

/-- One `WeightArr` value. -/
structure WA where
  tab : Nat
  len : Nat

/-- `Index::index`; `none` is the out-of-range panic. -/
def read (h : Heap T) (v : WA) (s : Nat) : Option T :=
  match (h.tbl v.tab)[cix s]? with
  | none => none
  | some c => (h.chk c)[oix s]?

structure St (T : Type) where
  h : Heap T
  vs : List WA
  ts : List Nat

/-- `Arc::strong_count` of table `p`: the versions holding it. -/
def cntT (vs : List WA) (p : Nat) : Nat := vs.countP (fun v => decide (v.tab = p))
/-- `Arc::strong_count` of chunk `c`: the slots of live tables holding it. -/
def cntC (h : Heap T) (ts : List Nat) (c : Nat) : Nat := (ts.map (fun p => (h.tbl p).count c)).sum

structure WF (st : St T) : Prop where
  ts_nodup : st.ts.Nodup
  ts_iff : ∀ p, p ∈ st.ts ↔ ∃ v ∈ st.vs, v.tab = p
  ts_lt : ∀ p ∈ st.ts, p < st.h.nT
  chk_lt : ∀ p ∈ st.ts, ∀ c ∈ st.h.tbl p, c < st.h.nC

/-- `Arc::make_mut(&mut self.chunks)` for the version `A` (the head of `vs`): in place if no other version
    holds the table; otherwise a copy at a fresh address (each chunk `Arc` in it cloned). -/
def mmTab (st : St T) (A : WA) (os : List WA) : St T × WA :=
  if cntT (A :: os) A.tab = 1 then (st, A)
  else
    (⟨⟨fun q => if q = st.h.nT then st.h.tbl A.tab else st.h.tbl q, st.h.chk, st.h.nT + 1, st.h.nC⟩,
      ⟨st.h.nT, A.len⟩ :: os, st.h.nT :: st.ts⟩, ⟨st.h.nT, A.len⟩)

/-- `Arc::make_mut(chunk)` for slot `k` of table `p`, which holds chunk `c` there: in place if no other
    slot of a live table holds it; otherwise a copy at a fresh address, stored into slot `k`. -/
def mmChunk (st : St T) (p k c : Nat) : St T × Nat :=
  if cntC st.h st.ts c = 1 then (st, c)
  else
    (⟨⟨fun q => if q = p then (st.h.tbl p).set k st.h.nC else st.h.tbl q,
       fun q => if q = st.h.nC then st.h.chk c else st.h.chk q, st.h.nT, st.h.nC + 1⟩, st.vs, st.ts⟩, st.h.nC)

/-- The write, after both `make_mut`s: `chunk[slot & (CHUNK_LEN - 1)] = value`. -/
def writeAt (st2 : St T) (c2 o : Nat) (x : T) : St T :=
  ⟨⟨st2.h.tbl, fun q => if q = c2 then (st2.h.chk c2).set o x else st2.h.chk q, st2.h.nT, st2.h.nC⟩, st2.vs, st2.ts⟩

/-- `set` after the table's `make_mut`: index the table (panic if out of range), `make_mut` the chunk,
    write (panic if the offset is past the chunk). -/
def setW2 (st1 : St T) (A1 : WA) (s : Nat) (x : T) : Option (St T) :=
  match (st1.h.tbl A1.tab)[cix s]? with
  | none => none
  | some c =>
    if oix s < ((mmChunk st1 A1.tab (cix s) c).1.h.chk (mmChunk st1 A1.tab (cix s) c).2).length then
      some (writeAt (mmChunk st1 A1.tab (cix s) c).1 (mmChunk st1 A1.tab (cix s) c).2 (oix s) x)
    else none

/-- **`WeightArr::set`** (`customize.rs:182-189` @12a) on the head version; `none` is a panic. -/
def setW (st : St T) (s : Nat) (x : T) : Option (St T) :=
  match st.vs with
  | [] => none
  | A :: os => setW2 (mmTab st A os).1 (mmTab st A os).2 s x

/-- Extend the (now private) last chunk `c2` by `x`. -/
def pushAt (st2 : St T) (c2 : Nat) (x : T) (A1 : WA) (os : List WA) : St T :=
  ⟨⟨st2.h.tbl, fun q => if q = c2 then st2.h.chk c2 ++ [x] else st2.h.chk q, st2.h.nT, st2.h.nC⟩,
   ⟨A1.tab, A1.len + 1⟩ :: os, st2.ts⟩

def appendW2 (st1 : St T) (A1 : WA) (os : List WA) (x : T) : Option (St T) :=
  if oix A1.len = 0 then
    some ⟨⟨fun q => if q = A1.tab then st1.h.tbl A1.tab ++ [st1.h.nC] else st1.h.tbl q,
           fun q => if q = st1.h.nC then [x] else st1.h.chk q, st1.h.nT, st1.h.nC + 1⟩,
          ⟨A1.tab, A1.len + 1⟩ :: os, st1.ts⟩
  else
    match (st1.h.tbl A1.tab)[cix A1.len]? with
    | none => none
    | some c => some (pushAt (mmChunk st1 A1.tab (cix A1.len) c).1 (mmChunk st1 A1.tab (cix A1.len) c).2 x A1 os)

/-- **Append** (design: 03 §2 "plus append"; 03 §2 "The metric grows by appending chunks"): copy the table
    if shared; a full last chunk (or none) is followed by a new chunk `[x]`; otherwise the last chunk is
    copied if shared, then extended. -/
def appendW (st : St T) (x : T) : Option (St T) :=
  match st.vs with
  | [] => none
  | A :: os => appendW2 (mmTab st A os).1 (mmTab st A os).2 os x

/-- `new_version` (`:192-194`): a clone of the head version, sharing everything. -/
def newVersion (st : St T) : St T :=
  match st.vs with
  | [] => st
  | A :: os => { st with vs := A :: A :: os }

/-- Dropping the head version; a table no version holds any more is freed. -/
def dropV (st : St T) : St T :=
  match st.vs with
  | [] => st
  | _ :: os => { st with vs := os, ts := st.ts.filter (fun p => os.any (fun v => decide (v.tab = p))) }

/-! ## Lemmas -/

theorem read_frame (h h' : Heap T) (v : WA) (ht : h'.tbl v.tab = h.tbl v.tab)
    (hc : ∀ c ∈ h.tbl v.tab, h'.chk c = h.chk c) (s : Nat) : read h' v s = read h v s := by
  unfold read
  rw [ht]
  cases e : (h.tbl v.tab)[cix s]? with
  | none => rfl
  | some c => simp only; rw [hc c (List.mem_of_getElem? e)]

theorem cntT_cons_self (A : WA) (os : List WA) : cntT (A :: os) A.tab = cntT os A.tab + 1 := by
  simp [cntT]

theorem other_holds_of_shared (A : WA) (os : List WA) (h : cntT (A :: os) A.tab ≠ 1) : ∃ B ∈ os, B.tab = A.tab := by
  rw [cntT_cons_self] at h
  have : 0 < cntT os A.tab := by omega
  unfold cntT at this
  rw [List.countP_pos_iff] at this
  obtain ⟨B, hB, hb⟩ := this
  exact ⟨B, hB, by simpa using hb⟩

theorem unique_tab (A : WA) (os : List WA) (h : cntT (A :: os) A.tab = 1) : ∀ B ∈ os, B.tab ≠ A.tab := by
  intro B hB he
  rw [cntT_cons_self] at h
  have : 0 < cntT os A.tab := by
    unfold cntT; rw [List.countP_pos_iff]; exact ⟨B, hB, by simp [he]⟩
  omega

theorem count_le_cntC (h : Heap T) {ts : List Nat} {p : Nat} (hp : p ∈ ts) (c : Nat) : (h.tbl p).count c ≤ cntC h ts c :=
  le_sum_of_mem' (List.mem_map_of_mem hp)

theorem count_zero_of_unique (h : Heap T) {ts : List Nat} (hn : ts.Nodup) {p q c : Nat} (hp : p ∈ ts) (hq : q ∈ ts)
    (hpq : p ≠ q) (hc : c ∈ h.tbl p) (h1 : cntC h ts c = 1) : c ∉ h.tbl q := by
  intro hcq
  have := two_le_sum (fun p => (h.tbl p).count c) hn hp hq hpq
  have a := List.count_pos_iff.2 hc
  have b := List.count_pos_iff.2 hcq
  unfold cntC at h1
  omega

theorem count_one_index {l : List Nat} {c : Nat} (h1 : l.count c ≤ 1) :
    ∀ {k k' : Nat}, l[k]? = some c → l[k']? = some c → k = k' := by
  induction l with
  | nil => intro k k' hk; simp at hk
  | cons a l ih =>
    intro k k' hk hk'
    cases k with
    | zero =>
      cases k' with
      | zero => rfl
      | succ k' =>
        simp at hk hk'
        subst hk
        have := List.count_pos_iff.2 (List.mem_of_getElem? hk')
        simp at h1; omega
    | succ k =>
      cases k' with
      | zero =>
        simp at hk hk'
        subst hk'
        have := List.count_pos_iff.2 (List.mem_of_getElem? hk)
        simp at h1; omega
      | succ k' =>
        simp at hk hk'
        have : l.count c ≤ 1 := by
          have := List.count_le_count_cons (a := c) (b := a) (l := l); omega
        exact congrArg (· + 1) (ih this hk hk')

theorem count_set_fresh {c : Nat} : ∀ {l : List Nat} {k : Nat}, c ∉ l → (l.set k c).count c ≤ 1
  | [], _, _ => by simp
  | a :: l, 0, h => by
    have hl : c ∉ l := fun e => h (List.mem_cons_of_mem _ e)
    simp [List.count_eq_zero.2 hl]
  | a :: l, k + 1, h => by
    have ha : a ≠ c := fun e => h (e ▸ List.mem_cons_self ..)
    have hl : c ∉ l := fun e => h (List.mem_cons_of_mem _ e)
    simp only [List.set_cons_succ, List.count_cons]
    have := count_set_fresh (k := k) hl
    simp [ha]; omega

/-! ## `mmTab` -/

/-- What `mmTab` gives: the head now holds a table no other version holds, with the same contents, and
    the old tables and every chunk are untouched. -/
theorem mmTab_spec (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st) :
    (mmTab st A os).1.vs = (mmTab st A os).2 :: os ∧ (mmTab st A os).2.len = A.len ∧ WF (mmTab st A os).1 ∧
    (∀ B ∈ os, B.tab ≠ (mmTab st A os).2.tab) ∧
    (mmTab st A os).1.h.tbl (mmTab st A os).2.tab = st.h.tbl A.tab ∧ (mmTab st A os).1.h.chk = st.h.chk ∧
    (mmTab st A os).1.h.nC = st.h.nC ∧
    (∀ q ∈ st.ts, (mmTab st A os).1.h.tbl q = st.h.tbl q) := by
  unfold mmTab
  split
  · rename_i h1
    exact ⟨hv, rfl, hw, unique_tab A os h1, rfl, rfl, rfl, fun _ _ => rfl⟩
  · rename_i h1
    dsimp only
    have hA : A.tab ∈ st.ts := (hw.ts_iff A.tab).2 ⟨A, by rw [hv]; exact List.mem_cons_self .., rfl⟩
    have hfresh : st.h.nT ∉ st.ts := fun h => by have := hw.ts_lt _ h; omega
    have hne : ∀ q ∈ st.ts, q ≠ st.h.nT := fun q hq e => hfresh (by rw [← e]; exact hq)
    refine ⟨rfl, rfl, ?_, ?_, by simp, rfl, rfl, ?_⟩
    · refine ⟨List.nodup_cons.2 ⟨hfresh, hw.ts_nodup⟩, ?_, ?_, ?_⟩
      · intro q
        rw [List.mem_cons]
        constructor
        · rintro (h | h)
          · exact ⟨⟨st.h.nT, A.len⟩, List.mem_cons_self .., h.symm⟩
          · obtain ⟨v, hv', hq⟩ := (hw.ts_iff q).1 h
            rw [hv] at hv'
            rcases List.mem_cons.1 hv' with e | e
            · subst e
              obtain ⟨B, hB, hb⟩ := other_holds_of_shared v os h1
              exact ⟨B, List.mem_cons_of_mem _ hB, hb.trans hq⟩
            · exact ⟨v, List.mem_cons_of_mem _ e, hq⟩
        · rintro ⟨v, hv', hq⟩
          rcases List.mem_cons.1 hv' with e | e
          · subst e; exact Or.inl hq.symm
          · exact Or.inr ((hw.ts_iff q).2 ⟨v, by rw [hv]; exact List.mem_cons_of_mem _ e, hq⟩)
      · intro q hq
        dsimp only
        rcases List.mem_cons.1 hq with h | h
        · rw [h]; exact Nat.lt_succ_self _
        · have := hw.ts_lt q h; omega
      · intro q hq c hc
        dsimp only at hc ⊢
        rcases List.mem_cons.1 hq with h | h
        · rw [h, ite_eq_left rfl] at hc; exact hw.chk_lt _ hA c hc
        · rw [ite_eq_right (hne q h)] at hc; exact hw.chk_lt q h c hc
    · intro B hB e
      have : B.tab ∈ st.ts := (hw.ts_iff B.tab).2 ⟨B, by rw [hv]; exact List.mem_cons_of_mem _ hB, rfl⟩
      exact hne _ this e
    · intro q hq
      rw [ite_eq_right (hne q hq)]

/-! ## `mmChunk` -/

/-- What `mmChunk` gives, for slot `k` of the head's table (held by no other version): a chunk `c2` that no
    other version's table holds, held by the head's table at `k` only, with `c`'s contents; every other
    table, and every chunk other than `c2`, is untouched. -/
theorem mmChunk_spec (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st)
    (huniq : ∀ B ∈ os, B.tab ≠ A.tab) (k c : Nat) (hk : (st.h.tbl A.tab)[k]? = some c) :
    (mmChunk st A.tab k c).1.vs = st.vs ∧ (mmChunk st A.tab k c).1.ts = st.ts ∧ WF (mmChunk st A.tab k c).1 ∧
    (mmChunk st A.tab k c).1.h.nT = st.h.nT ∧ st.h.nC ≤ (mmChunk st A.tab k c).1.h.nC ∧
    ((mmChunk st A.tab k c).1.h.tbl A.tab)[k]? = some (mmChunk st A.tab k c).2 ∧
    ((mmChunk st A.tab k c).1.h.tbl A.tab).count (mmChunk st A.tab k c).2 ≤ 1 ∧
    (mmChunk st A.tab k c).1.h.chk (mmChunk st A.tab k c).2 = st.h.chk c ∧
    (∀ k', k' ≠ k → ((mmChunk st A.tab k c).1.h.tbl A.tab)[k']? = (st.h.tbl A.tab)[k']?) ∧
    (∀ B ∈ os, (mmChunk st A.tab k c).1.h.tbl B.tab = st.h.tbl B.tab ∧ (mmChunk st A.tab k c).2 ∉ st.h.tbl B.tab) ∧
    (∀ q, q ≠ (mmChunk st A.tab k c).2 → (mmChunk st A.tab k c).1.h.chk q = st.h.chk q) ∧
    (∀ q, q ≠ A.tab → (mmChunk st A.tab k c).1.h.tbl q = st.h.tbl q) := by
  have hA : A.tab ∈ st.ts := (hw.ts_iff A.tab).2 ⟨A, by rw [hv]; exact List.mem_cons_self .., rfl⟩
  have hB : ∀ B ∈ os, B.tab ∈ st.ts :=
    fun B hB => (hw.ts_iff B.tab).2 ⟨B, by rw [hv]; exact List.mem_cons_of_mem _ hB, rfl⟩
  have hcm : c ∈ st.h.tbl A.tab := List.mem_of_getElem? hk
  unfold mmChunk
  split
  · rename_i h1
    dsimp only
    refine ⟨rfl, rfl, hw, rfl, Nat.le_refl _, hk, ?_, rfl, fun _ _ => rfl, ?_, fun _ _ => rfl, fun _ _ => rfl⟩
    · have := count_le_cntC st.h hA c; omega
    · intro B hb
      exact ⟨rfl, count_zero_of_unique st.h hw.ts_nodup hA (hB B hb) (Ne.symm (huniq B hb)) hcm h1⟩
  · dsimp only
    have hklt : k < (st.h.tbl A.tab).length := (List.getElem?_eq_some_iff.1 hk).1
    have hnot : ∀ q ∈ st.ts, st.h.nC ∉ st.h.tbl q := fun q hq h => by have := hw.chk_lt q hq _ h; omega
    refine ⟨rfl, rfl, ?_, rfl, Nat.le_succ _, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · refine ⟨hw.ts_nodup, hw.ts_iff, hw.ts_lt, ?_⟩
      intro q hq c' hc'
      dsimp only at hc' ⊢
      by_cases e : q = A.tab
      · rw [ite_eq_left e] at hc'
        subst e
        rcases List.mem_or_eq_of_mem_set hc' with h | h
        · have := hw.chk_lt _ hq c' h; omega
        · omega
      · rw [ite_eq_right e] at hc'
        have := hw.chk_lt q hq c' hc'; omega
    · rw [ite_eq_left rfl]; simp [hklt]
    · rw [ite_eq_left rfl]; exact count_set_fresh (hnot _ hA)
    · rw [ite_eq_left rfl]
    · intro k' hk'
      rw [ite_eq_left rfl, List.getElem?_set_ne (Ne.symm hk')]
    · intro B hb
      refine ⟨?_, hnot _ (hB B hb)⟩
      rw [ite_eq_right (huniq B hb)]
    · intro q hq
      rw [ite_eq_right hq]
    · intro q hq
      rw [ite_eq_right hq]

/-! ## `set` -/

/-- **`WeightArr::set` is isolated**: on success, every other version reads exactly what it read before,
    at every slot; the written version reads `x` at `s` and its old values everywhere else; `WF` holds. -/
theorem setW_spec (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st) (s : Nat) (x : T)
    (st' : St T) (he : setW st s x = some st') :
    ∃ A', st'.vs = A' :: os ∧ A'.len = A.len ∧ WF st' ∧
      (∀ B ∈ os, ∀ s', read st'.h B s' = read st.h B s') ∧
      read st'.h A' s = some x ∧ (∀ s', s' ≠ s → read st'.h A' s' = read st.h A s') := by
  have F1 := mmTab_spec st A os hv hw
  have hB0 : ∀ B ∈ os, B.tab ∈ st.ts :=
    fun B hB => (hw.ts_iff B.tab).2 ⟨B, by rw [hv]; exact List.mem_cons_of_mem _ hB, rfl⟩
  have hs : setW st s x = setW2 (mmTab st A os).1 (mmTab st A os).2 s x := by simp [setW, hv]
  rw [hs] at he
  generalize mmTab st A os = m at F1 he
  obtain ⟨st1, A1⟩ := m
  obtain ⟨hv1, hlen, hw1, huniq, htab, hchk, _, hold⟩ := F1
  dsimp only at hv1 hlen hw1 huniq htab hchk hold he
  unfold setW2 at he
  cases hc : (st1.h.tbl A1.tab)[cix s]? with
  | none => rw [hc] at he; cases he
  | some c =>
    rw [hc] at he
    dsimp only at he
    have F2 := mmChunk_spec st1 A1 os hv1 hw1 huniq (cix s) c hc
    generalize mmChunk st1 A1.tab (cix s) c = m2 at F2 he
    obtain ⟨st2, c2⟩ := m2
    obtain ⟨g1, g2, g3, _, _, g6, g7, g8, g9, g10, g11, _⟩ := F2
    dsimp only at g1 g2 g3 g6 g7 g8 g9 g10 g11 he
    split at he
    · rename_i hlt
      cases he
      refine ⟨A1, by simp [writeAt, g1, hv1], hlen, ⟨g3.ts_nodup, g3.ts_iff, g3.ts_lt, g3.chk_lt⟩, ?_, ?_, ?_⟩
      · intro B hB s'
        have ht : st2.h.tbl B.tab = st.h.tbl B.tab := (g10 B hB).1.trans (hold B.tab (hB0 B hB))
        refine read_frame st.h (writeAt st2 c2 (oix s) x).h B ht (fun c' hc' => ?_) s'
        have hne : c' ≠ c2 := by
          intro e; subst e
          exact (g10 B hB).2 (by rw [hold B.tab (hB0 B hB)]; exact hc')
        show (if c' = c2 then _ else st2.h.chk c') = st.h.chk c'
        rw [ite_eq_right hne, g11 c' hne, hchk]
      · unfold read
        show (match (st2.h.tbl A1.tab)[cix s]? with
          | none => none
          | some c => (if c = c2 then (st2.h.chk c2).set (oix s) x else st2.h.chk c)[oix s]?) = some x
        rw [g6]
        simp only
        simp [hlt]
      · intro s' hs'
        unfold read
        show (match (st2.h.tbl A1.tab)[cix s']? with
          | none => none
          | some c => (if c = c2 then (st2.h.chk c2).set (oix s) x else st2.h.chk c)[oix s']?) = _
        by_cases hk : cix s' = cix s
        · have ho : oix s' ≠ oix s := fun ho => hs' (slot_inj hk ho)
          rw [hk, g6, ← htab, hc]
          dsimp only
          rw [ite_eq_left rfl, List.getElem?_set_ne (Ne.symm ho), g8, hchk]
        · rw [g9 _ hk, ← htab]
          cases hc' : (st1.h.tbl A1.tab)[cix s']? with
          | none => rfl
          | some c' =>
            have hne : c' ≠ c2 := by
              intro e; subst e
              exact hk (count_one_index g7 (by rw [g9 _ hk]; exact hc') g6)
            simp only
            rw [ite_eq_right hne, g11 c' hne, hchk]
    · cases he

/-! ## append (design) -/

/-- **Append is isolated**: on success, every other version reads exactly what it read before, the head's
    length grows by one, and `WF` holds. -/
theorem appendW_spec (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st) (x : T)
    (st' : St T) (he : appendW st x = some st') :
    ∃ A', st'.vs = A' :: os ∧ A'.len = A.len + 1 ∧ WF st' ∧ (∀ B ∈ os, ∀ s', read st'.h B s' = read st.h B s') := by
  have F1 := mmTab_spec st A os hv hw
  have hB0 : ∀ B ∈ os, B.tab ∈ st.ts :=
    fun B hB => (hw.ts_iff B.tab).2 ⟨B, by rw [hv]; exact List.mem_cons_of_mem _ hB, rfl⟩
  have hs : appendW st x = appendW2 (mmTab st A os).1 (mmTab st A os).2 os x := by simp [appendW, hv]
  rw [hs] at he
  generalize mmTab st A os = m at F1 he
  obtain ⟨st1, A1⟩ := m
  obtain ⟨hv1, hlen, hw1, huniq, htab, hchk, hnC, hold⟩ := F1
  dsimp only at hv1 hlen hw1 huniq htab hchk hnC hold he
  have hA1 : A1.tab ∈ st1.ts := (hw1.ts_iff A1.tab).2 ⟨A1, by rw [hv1]; exact List.mem_cons_self .., rfl⟩
  have hB1 : ∀ B ∈ os, B.tab ∈ st1.ts :=
    fun B hB => (hw1.ts_iff B.tab).2 ⟨B, by rw [hv1]; exact List.mem_cons_of_mem _ hB, rfl⟩
  unfold appendW2 at he
  split at he
  · cases he
    refine ⟨⟨A1.tab, A1.len + 1⟩, rfl, by simp [hlen], ?_, ?_⟩
    · refine ⟨hw1.ts_nodup, fun p => ?_, hw1.ts_lt, ?_⟩
      · rw [hw1.ts_iff p, hv1]
        constructor
        · rintro ⟨v, hv', hp⟩
          rcases List.mem_cons.1 hv' with e | e
          · subst e; exact ⟨_, List.mem_cons_self .., hp⟩
          · exact ⟨v, List.mem_cons_of_mem _ e, hp⟩
        · rintro ⟨v, hv', hp⟩
          rcases List.mem_cons.1 hv' with e | e
          · subst e; exact ⟨A1, List.mem_cons_self .., hp⟩
          · exact ⟨v, List.mem_cons_of_mem _ e, hp⟩
      · intro q hq c hc
        dsimp only at hc ⊢
        by_cases e : q = A1.tab
        · rw [ite_eq_left e] at hc
          rcases List.mem_append.1 hc with h | h
          · have := hw1.chk_lt q hq c (e ▸ h); omega
          · simp at h; omega
        · rw [ite_eq_right e] at hc
          have := hw1.chk_lt q hq c hc; omega
    · intro B hB s'
      have hne : B.tab ≠ A1.tab := huniq B hB
      refine read_frame st.h _ B ?_ (fun c' hc' => ?_) s'
      · show (if B.tab = A1.tab then _ else st1.h.tbl B.tab) = _
        rw [ite_eq_right hne, hold B.tab (hB0 B hB)]
      · have : c' < st1.h.nC := hw1.chk_lt _ (hB1 B hB) c' (by rw [hold B.tab (hB0 B hB)]; exact hc')
        show (if c' = st1.h.nC then _ else st1.h.chk c') = _
        rw [ite_eq_right (by omega), hchk]
  · cases hc : (st1.h.tbl A1.tab)[cix A1.len]? with
    | none => rw [hc] at he; cases he
    | some c =>
      rw [hc] at he
      dsimp only at he
      have F2 := mmChunk_spec st1 A1 os hv1 hw1 huniq (cix A1.len) c hc
      generalize mmChunk st1 A1.tab (cix A1.len) c = m2 at F2 he
      obtain ⟨st2, c2⟩ := m2
      obtain ⟨g1, g2, g3, _, _, _, _, _, _, g10, g11, _⟩ := F2
      dsimp only at g1 g2 g3 g10 g11 he
      cases he
      refine ⟨⟨A1.tab, A1.len + 1⟩, rfl, by simp [hlen], ?_, ?_⟩
      · refine ⟨g3.ts_nodup, fun p => ?_, g3.ts_lt, g3.chk_lt⟩
        show p ∈ st2.ts ↔ ∃ v, v ∈ (⟨A1.tab, A1.len + 1⟩ :: os : List WA) ∧ v.tab = p
        rw [g3.ts_iff p, g1, hv1]
        constructor
        · rintro ⟨v, hv', hp⟩
          rcases List.mem_cons.1 hv' with e | e
          · subst e; exact ⟨_, List.mem_cons_self .., hp⟩
          · exact ⟨v, List.mem_cons_of_mem _ e, hp⟩
        · rintro ⟨v, hv', hp⟩
          rcases List.mem_cons.1 hv' with e | e
          · subst e; exact ⟨A1, List.mem_cons_self .., hp⟩
          · exact ⟨v, List.mem_cons_of_mem _ e, hp⟩
      · intro B hB s'
        have ht : st2.h.tbl B.tab = st.h.tbl B.tab := (g10 B hB).1.trans (hold B.tab (hB0 B hB))
        refine read_frame st.h (pushAt st2 c2 x A1 os).h B ht (fun c' hc' => ?_) s'
        have hne : c' ≠ c2 := by
          intro e; subst e
          exact (g10 B hB).2 (by rw [hold B.tab (hB0 B hB)]; exact hc')
        show (if c' = c2 then _ else st2.h.chk c') = st.h.chk c'
        rw [ite_eq_right hne, g11 c' hne, hchk]

/-! ## Taking, dropping and reordering versions -/

/-- `new_version`: the clone reads as its source, and nothing else changes. -/
theorem newVersion_spec (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st) :
    (newVersion st).vs = A :: A :: os ∧ (newVersion st).h = st.h ∧ WF (newVersion st) := by
  have e : newVersion st = { st with vs := A :: A :: os } := by unfold newVersion; rw [hv]
  rw [e]
  refine ⟨rfl, rfl, hw.ts_nodup, fun p => ?_, hw.ts_lt, hw.chk_lt⟩
  rw [hw.ts_iff p, hv]
  simp only [List.mem_cons]
  constructor
  · rintro ⟨v, hv', hp⟩; exact ⟨v, Or.inr hv', hp⟩
  · rintro ⟨v, hv' | hv', hp⟩
    · exact ⟨v, Or.inl hv', hp⟩
    · exact ⟨v, hv', hp⟩

/-- Dropping a version frees its table if no one else holds it; the others read as before. -/
theorem dropV_spec (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st) :
    (dropV st).vs = os ∧ (dropV st).h = st.h ∧ WF (dropV st) := by
  have e : dropV st = { st with vs := os, ts := st.ts.filter (fun p => os.any (fun v => decide (v.tab = p))) } := by
    unfold dropV; rw [hv]
  rw [e]
  refine ⟨rfl, rfl, hw.ts_nodup.sublist List.filter_sublist, fun p => ?_,
    fun p hp => hw.ts_lt p (List.mem_filter.1 hp).1, fun p hp => hw.chk_lt p (List.mem_filter.1 hp).1⟩
  rw [List.mem_filter, hw.ts_iff p, hv]
  simp only [List.any_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨_, h⟩; exact h
  · rintro ⟨v, hv', hp⟩; exact ⟨⟨v, List.mem_cons_of_mem _ hv', hp⟩, ⟨v, hv', hp⟩⟩

/-- Versions are a multiset: reordering them changes nothing. -/
theorem perm_wf (st : St T) (vs' : List WA) (hp : vs'.Perm st.vs) (hw : WF st) : WF { st with vs := vs' } := by
  refine ⟨hw.ts_nodup, fun p => ?_, hw.ts_lt, hw.chk_lt⟩
  rw [hw.ts_iff p]
  constructor
  · rintro ⟨v, hv, h⟩; exact ⟨v, hp.mem_iff.2 hv, h⟩
  · rintro ⟨v, hv, h⟩; exact ⟨v, hp.mem_iff.1 hv, h⟩

/-- **Isolation across any history.** Under `WF`, any sequence of `set`s, appends, `new_version`s, drops
    and reorderings changes no read of a version it does not write: each step above leaves every other
    version's reads unchanged, and keeps `WF`. Stated per step (`setW_spec`, `appendW_spec`,
    `newVersion_spec`, `dropV_spec`, `perm_wf`); the read of a version `B` only depends on the heap through
    `B`'s table and its chunks (`read_frame`). -/
theorem setW_isolated (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st) (s : Nat) (x : T)
    (st' : St T) (he : setW st s x = some st') (B : WA) (hB : B ∈ os) (s' : Nat) :
    read st'.h B s' = read st.h B s' := by
  obtain ⟨_, _, _, _, h, _⟩ := setW_spec st A os hv hw s x st' he
  exact h B hB s'

theorem appendW_isolated (st : St T) (A : WA) (os : List WA) (hv : st.vs = A :: os) (hw : WF st) (x : T)
    (st' : St T) (he : appendW st x = some st') (B : WA) (hB : B ∈ os) (s' : Nat) :
    read st'.h B s' = read st.h B s' := by
  obtain ⟨_, _, _, _, h⟩ := appendW_spec st A os hv hw x st' he
  exact h B hB s'

end CchTopo.Chunked
