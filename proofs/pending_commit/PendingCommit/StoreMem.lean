import PendingCommit.StoreCodec
/-
# `DataBlock::trim` and the memory partition (attribute_store.rs:1038-1145)

See `StoreCodec.lean`'s header for the statements; `trim` is modelled with an
`owned bi` predicate standing for `Arc::get_mut` succeeding on the root, the
page and the block (shared levels belong to another version and are skipped);
the `shrink_to_fit` calls change capacities only, which `structural_partition`
takes as parameters.
-/
namespace PendingCommit.Content
open Arena (Slot)

/-- Commit-time compaction rule inside `trim` (note `>=`, unlike
`maybe_compact`'s `>`). -/
def trimB (b : CB) : CB :=
  if (b.dead + b.slack) * 2 ≥ b.arena.length ∧ b.arena.length > 256 then compactC b else b

def dtrim (d : D) (owned : Nat → Bool) : D :=
  ⟨d.root.mapIdx (fun di op => op.map (fun pg =>
    pg.mapIdx (fun si ob => ob.map (fun b => if owned (di * 64 + si) then trimB b else b))))⟩

theorem blockAt_trim (d : D) (owned : Nat → Bool) (bi : Nat) :
    blockAt (dtrim d owned) bi = (blockAt d bi).map (fun b => if owned bi then trimB b else b) := by
  have hbi : bi / 64 * 64 + bi % 64 = bi := by omega
  simp only [blockAt, dtrim, List.getD_eq_getElem?_getD, List.getElem?_mapIdx]
  cases e : d.root[bi / 64]? with
  | none => simp
  | some op =>
    cases op with
    | none => simp
    | some pg =>
      simp only [Option.map_some, Option.getD_some, List.getElem?_mapIdx, hbi]
      cases pg[bi % 64]? with
      | none => simp
      | some ob => cases ob <;> simp

theorem trimB_ok (b : CB) (h : CInv b) : CInv (trimB b) ∧ ∀ j, span (trimB b) j = span b j := by
  unfold trimB; split
  · obtain ⟨a, c, -⟩ := compactC_ok b h; exact ⟨a, c⟩
  · exact ⟨h, fun _ => rfl⟩

/-- **`dtrim_ok`**: `trim` keeps every read and `SInv`. -/
theorem dtrim_ok (d : D) (h : SInv d) (owned : Nat → Bool) :
    SInv (dtrim d owned) ∧ ∀ id, dread (dtrim d owned) id = dread d id := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro pg hm
    simp only [dtrim, List.mem_mapIdx] at hm
    obtain ⟨i, hi, e⟩ := hm
    cases eo : d.root[i] with
    | none => rw [eo] at e; simp at e
    | some pg0 =>
      rw [eo] at e; simp only [Option.map_some, Option.some.injEq] at e
      subst e; simp only [List.length_mapIdx]
      exact h.1 pg0 (by rw [← eo]; exact List.getElem_mem hi)
  · intro bi b e
    rw [blockAt_trim] at e
    cases e' : blockAt d bi with
    | none => rw [e'] at e; simp at e
    | some b0 =>
      rw [e'] at e; simp only [Option.map_some, Option.some.injEq] at e; subst e
      have := h.2 bi b0 e'
      split
      · exact (trimB_ok b0 this).1
      · exact this
  · intro id
    simp only [dread, blockOf, blockAt_trim]
    cases e' : blockAt d (id / 64) with
    | none => rfl
    | some b0 =>
      simp only [Option.map_some, Option.getD_some]
      split
      · exact (trimB_ok b0 (h.2 _ b0 e')).2 _
      · rfl

/-! ## Memory partition -/

theorem heap_len_partition (H : Pack.Heap) (refs : List Nat) (hI : Pack.HInv H refs) :
    H.heap.length = refs.length + H.free.length := by
  have hnd : (refs ++ H.free).Nodup :=
    List.nodup_append.2 ⟨hI.refs_nodup, hI.free_nodup, fun a ha b hb e => hI.disj a ha (e ▸ hb)⟩
  have hp : (refs ++ H.free).Perm (List.range H.heap.length) :=
    (List.perm_ext_iff_of_nodup hnd List.nodup_range).2 (fun a => by
      rw [List.mem_append, List.mem_range, hI.cover a])
  have := hp.length_eq
  simp at this; omega

/-- `structural_memory_usage`'s per-block term (:1121-1139). -/
def structBlock (ARC SB SV slotsCap arenaCap heapCap freeCap : Nat) (b : CB) (heapLen freeLen : Nat) : Nat :=
  ARC + SB + slotsCap * 8 + (arenaCap - (b.arena.length - b.dead - b.slack)) * 12 +
    (heapCap - (heapLen - freeLen)) * SV + freeCap * 4

/-- **`structural_partition`**: the block's structural bytes plus its entities'
`heap_bytes` (`Σ len·12 + |refs|·SV + Σ amortized`) is exactly the block's
allocation — nothing counted twice, nothing missed, no underflow. -/
theorem structural_partition (ARC SB SV slotsCap arenaCap heapCap freeCap amort : Nat) (b : CB) (h : CInv b)
    (H : Pack.Heap) (refs : List Nat) (hI : Pack.HInv H refs)
    (hac : b.arena.length ≤ arenaCap) (hhc : H.heap.length ≤ heapCap) :
    structBlock ARC SB SV slotsCap arenaCap heapCap freeCap b H.heap.length H.free.length +
      (Arena.sumBy (·.len) b.slots * 12 + refs.length * SV + amort) =
    ARC + SB + slotsCap * 8 + arenaCap * 12 + heapCap * SV + freeCap * 4 + amort := by
  have hl := Arena.live_eq_sum_len h.arena
  simp only [Arena.live, toBlk] at hl
  have hp := heap_len_partition H refs hI
  have hle : Arena.sumBy (·.len) b.slots ≤ arenaCap := by omega
  unfold structBlock
  rw [hl, hp, Nat.add_sub_cancel]
  have e1 : (arenaCap - Arena.sumBy (·.len) b.slots) * 12 + Arena.sumBy (·.len) b.slots * 12 = arenaCap * 12 := by
    rw [← Nat.add_mul, Nat.sub_add_cancel hle]
  have e2 : (heapCap - refs.length) * SV + refs.length * SV = heapCap * SV := by
    rw [← Nat.add_mul, Nat.sub_add_cancel (by omega)]
  omega

end PendingCommit.Content
