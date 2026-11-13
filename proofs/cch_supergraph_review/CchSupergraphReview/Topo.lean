import CchSupergraphReview.Invert
import CchSupergraphReview.Transpose
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# `Topology` (`order.rs:100-314` @main), arc slots and arc ids (03 §2 "Arc identity", §4 step 4, I7)

| Lean | Rust (`order.rs` @main unless noted) |
| --- | --- |
| `rankNew`, `rankIndex` | `Rank::new` (`:51-57`), `Rank::index` (`:60-62`): stored as `index + 1` |
| `idIndex` | `id_index` (`cch/mod.rs:58-60`), `index_id` (`:66-68`), `node_index` (`order.rs:317-319`) |
| `Topo` | `Topology` (`:129-139`) with its two `RankRows` (`:142-148`) as `(offsets, neighbors)` |
| `fromOrder` | `Topology::from_order` (`:195-212`) |
| `build` | `Topology::build` (`:180-182`); `nd` is `nd::nested_dissection_order`, of which only its symmetry check (`nd.rs:51-58`) is used |
| `Topo.nodeCount`, `ranks`, `arcCount`, `nodeOf`, `rankOf`, `up`, `upRange`, `down` | `:214-277` |
| `arcSlot` | `Topology::arc_slot` (`:287-302`), `binary_search` by its std contract (`BsIdx`) |
| `RowsOk`, `rowSlots`/`slice` | `RankRows::slots` (`:152-157`), `RankRows::row` (`:159-164`), `row_slots` (`:324-329`) |
-/
namespace CchSG

open NI.Cch (Csr)

/-! ## `Rank` and the index casts -/

/-- `Rank::new(index)` (`order.rs:51-57`): the stored `index + 1`, or the panic past `usize::MAX`. -/
def rankNew (index : Nat) : Option Nat := if index + 1 < 2 ^ 64 then some (index + 1) else none

/-- `Rank::index` (`order.rs:60-62`): `stored - 1`. -/
def rankIndex (stored : Nat) : Nat := stored - 1

/-- `Rank::new` never panics below `MAX_NODES` (`2^60 - 1`), round-trips through `index`, and keeps the order, so
    `Rank`'s derived `Ord` compares indices. -/
theorem rank_spec {i j : Nat} (hi : i < 2 ^ 60) (hj : j < 2 ^ 60) :
    (∃ s, rankNew i = some s ∧ rankIndex s = i) ∧
    (∀ s t, rankNew i = some s → rankNew j = some t → (s < t ↔ i < j)) := by
  have hi' : i + 1 < 2 ^ 64 := by have : (2 : Nat) ^ 60 < 2 ^ 64 := by decide
                                  omega
  refine ⟨⟨i + 1, by simp [rankNew, hi'], by simp [rankIndex]⟩, fun s t hs ht => ?_⟩
  simp only [rankNew] at hs ht
  split at hs <;> split at ht <;> simp_all <;> omega

/-- `impl Debug for Rank` (`order.rs:66-73`): `Rank(<index>)`, the index and not the stored `index + 1`.
Models the `{:?}` form only; the `{:#?}` form and the `fmt::Result` error path are not modelled. -/
def rankDebug (stored : Nat) : String := "Rank(" ++ toString (rankIndex stored) ++ ")"

theorem rankDebug_spec (i : Nat) : rankDebug (i + 1) = "Rank(" ++ toString i ++ ")" := by
  simp [rankDebug, rankIndex]

/-- `id_index` / `index_id` / `node_index`: `usize::try_from(id).expect(..)` on a target with `bits`-bit
    `usize`. -/
def idIndex (bits id : Nat) : Option Nat := if id < 2 ^ bits then some id else none

/-- On a 64-bit target every `u64` id casts to itself: the `expect` never fires. -/
theorem idIndex_64 {id : Nat} (h : id < 2 ^ 64) : idIndex 64 id = some id := by simp [idIndex, h]

/-! ## `Topology` -/

/-- `Topology` (`order.rs:129-139`): the order both ways and the two `RankRows`. -/
structure Topo where
  r2n : List Nat
  n2r : List Nat
  upOff : List Nat
  upNb : List Nat
  downOff : List Nat
  downNb : List Nat

/-- `Topology::from_order` (`order.rs:195-212`). -/
def fromOrder (sk : Csr) (r2n : List Nat) : Except OErr Topo :=
  if !sk.isWellFormed then .error .malformedSkeleton
  else match invert r2n sk.n with
    | .error e => .error e
    | .ok n2r =>
      let up := chordalSupergraph sk r2n n2r
      let down := transpose up.1 up.2
      .ok ⟨r2n, n2r, up.1, up.2, down.1, down.2⟩

/-- `Topology::build` (`order.rs:180-182`). -/
def build (nd : Csr → Except OErr (List Nat)) (sk : Csr) : Except OErr Topo :=
  match nd sk with
  | .error e => .error e
  | .ok o => fromOrder sk o

def Topo.nodeCount (t : Topo) : Nat := t.r2n.length
def Topo.ranks (t : Topo) : List Nat := List.range t.nodeCount
def Topo.arcCount (t : Topo) : Nat := t.upNb.length
def Topo.nodeOf (t : Topo) (r : Nat) : Nat := t.r2n.getD r 0
def Topo.rankOf (t : Topo) (v : Nat) : Nat := t.n2r.getD v 0
def Topo.upRange (t : Topo) (r : Nat) : Nat × Nat := rowSlots t.upOff r
def Topo.up (t : Topo) (r : Nat) : List Nat := slice t.upNb (t.upRange r)
def Topo.down (t : Topo) (r : Nat) : List Nat := slice t.downNb (rowSlots t.downOff r)

/-- `slice::binary_search(..).ok()` by the standard library's contract: on a strictly sorted slice it is
    `Some(i)` exactly at the index holding the value. -/
def BsIdx (bs : List Nat → Nat → Option Nat) : Prop :=
  ∀ l x, CowBTree.Sorted l → ∀ i, bs l x = some i ↔ l[i]? = some x

/-- `Topology::arc_slot` (`order.rs:287-302`). -/
def arcSlot (bs : List Nat → Nat → Option Nat) (t : Topo) (e o : Nat) : Option Nat :=
  let lower := if e < o then e else o
  let higher := if e < o then o else e
  let slots := t.upRange lower
  (bs (slice t.upNb slots) higher).map (fun off => slots.1 + off)

/-! ## `from_order` plays the game -/

/-- `from_order` refuses a malformed skeleton and a non-permutation, and nothing else. -/
theorem fromOrder_errors (sk : Csr) (r2n : List Nat) :
    (sk.isWellFormed = false → fromOrder sk r2n = .error .malformedSkeleton) ∧
    (sk.isWellFormed = true → ∀ e, invert r2n sk.n = .error e → fromOrder sk r2n = .error e) := by
  refine ⟨fun h => by simp [fromOrder, h], fun h e he => by simp [fromOrder, h, he]⟩

/-- **Theorem 3 for `Topology::from_order`.** On a mirrored skeleton, an `Ok` topology has: `node_of`/`rank_of`
    inverse; `up(r)` = the elimination game's upward row; `down(h)` = every `l` with `h ∈ up(l)`, ascending
    (`transpose`); the arc count = the sum of the rows' lengths. -/
theorem fromOrder_spec {sk : Csr} {r2n : List Nat} {t : Topo} (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) :
    let G := rankAdj sk r2n t.n2r
    OrderOk sk r2n t.n2r ∧ t.nodeCount = sk.n ∧
    (∀ r, r < sk.n → t.up r = gameRow sk.n G r) ∧
    (∀ h, h < sk.n → t.down h = lowersK (gameRow sk.n G) sk.n h) ∧
    t.upOff = offsOf (gameRow sk.n G) sk.n ∧ t.upNb = (rowsOf (gameRow sk.n G) sk.n).flatten ∧
    t.arcCount = (offsOf (gameRow sk.n G) sk.n).getD sk.n 0 := by
  intro G
  unfold fromOrder at h
  cases hwf : sk.isWellFormed
  · simp [hwf] at h
  · simp only [hwf, Bool.not_true, Bool.false_eq_true, ite_false] at h
    cases hinv : invert r2n sk.n with
    | error e => simp [hinv] at h
    | ok n2r =>
      simp only [hinv, Except.ok.injEq] at h
      subst h
      have hok := orderOk_of_invert hwf hinv
      have hc := chordalSupergraph_spec hok hm
      have hs := rankAdj_supp hok
      have ht := transpose_spec (R := gameRow sk.n (rankAdj sk r2n n2r)) (n := sk.n)
        (fun l _ => gameRow_sorted _ _ _) (fun l _ x hx => (up_supp hs ((mem_gameRow hs).1 hx)).2)
      simp only [hc] at ht ⊢
      refine ⟨hok, hok.len_r2n, fun r hr => slice_rowOf _ hr, fun h hh => ?_, rfl, rfl, ?_⟩
      · simp only [Topo.down, ht]; exact slice_rowOf _ hh
      · simp only [Topo.arcCount]; exact flatten_length _ _

/-- **`Topology::build` plays the game**: `nd::nested_dissection_order` returns `Ok` only for a skeleton that is
    `is_symmetric_simple` (`nd.rs:54-58`), which discharges `Mirrored`. -/
theorem build_spec (bs : List Nat → Nat → Bool) (hbs : NI.Cch.BsSpec bs) (nd : Csr → Except OErr (List Nat))
    (hnd : ∀ sk o, nd sk = .ok o → sk.isSymmetricSimple bs = true) {sk : Csr} {t : Topo}
    (h : build nd sk = .ok t) :
    ∀ r, r < sk.n → t.up r = gameRow sk.n (rankAdj sk t.r2n t.n2r) r := by
  unfold build at h
  cases ho : nd sk with
  | error e => simp [ho] at h
  | ok o =>
    simp only [ho] at h
    have hm := mirrored_of_symmetricSimple bs hbs sk (hnd sk o ho)
    have hr : t.r2n = o := by
      unfold fromOrder at h
      cases hwf : sk.isWellFormed <;> simp [hwf] at h
      cases hinv : invert o sk.n <;> simp [hinv] at h
      rw [← h]
    have := (fromOrder_spec h hm).2.2.1
    rw [hr]; exact this

/-- **Theorem 3 for the topology `from_order` returns.** Chordal in the CCH sense (if `y < z` are both in
    `up(x)`, then `z ∈ up(y)`), every skeleton edge is an arc of its lower end, and every row is the game's. -/
theorem topo_chordal {sk : Csr} {r2n : List Nat} {t : Topo} (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) :
    (∀ x y z, x < sk.n → y < sk.n → y ∈ t.up x → z ∈ t.up x → y < z → z ∈ t.up y) ∧
    (∀ u v, u < sk.n → v ∈ sk.rowD u → t.rankOf u < t.rankOf v → t.rankOf v ∈ t.up (t.rankOf u)) := by
  obtain ⟨hok, _, hup, _, _, _, _⟩ := fromOrder_spec h hm
  have hs := rankAdj_supp hok
  refine ⟨fun x y z hx hy hyx hzx hyz => ?_, fun u v hu hv hlt => ?_⟩
  · rw [hup x hx] at hyx hzx
    rw [hup y hy, mem_gameRow hs]
    exact clique ((mem_gameRow hs).1 hyx) ((mem_gameRow hs).1 hzx) hyz
  · have hru : t.rankOf u < sk.n := hok.n2r_lt u hu
    rw [hup _ hru, mem_gameRow hs]
    apply skeleton_in_up hlt
    have : rankAdj sk r2n t.n2r (t.rankOf u) (t.rankOf v) = true := by
      rw [rankAdj_iff, hok.len_r2n]
      exact ⟨hru, v, by simp only [Topo.rankOf]; rw [hok.inv2 u hu]; exact hv, rfl⟩
    simp [this]

/-- **Theorem 5 (the column counts, 03 §3 step 8 and §4 step 2).** `up(r)` is the ascending list of every
    `w > r` that is a skeleton neighbour of `r` or of an elimination-tree descendant of `r` (Liu's etree: the
    parent is the row's first entry, `par`). So its length is the Gilbert–Ng–Peyton column count of `r`, and the
    counts sum to the arc count. -/
theorem counts_spec {sk : Csr} {r2n : List Nat} {t : Topo} (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk)
    {r : Nat} (hr : r < sk.n) :
    CowBTree.Sorted (t.up r) ∧
    (∀ w, w ∈ t.up r ↔ r < w ∧ ∃ k, Desc (par sk.n (rankAdj sk r2n t.n2r)) k r ∧
      (rankAdj sk r2n t.n2r k w || rankAdj sk r2n t.n2r w k) = true) ∧
    t.arcCount = ((rowsOf (fun r => (t.up r)) sk.n).map List.length).sum := by
  obtain ⟨hok, _, hup, _, _, hnb, _⟩ := fromOrder_spec h hm
  have hs := rankAdj_supp hok
  refine ⟨by rw [hup r hr]; exact gameRow_sorted _ _ _, fun w => by rw [hup r hr, mem_gameRow hs, row_subtree hs],
    ?_⟩
  have e : rowsOf (fun r => t.up r) sk.n = rowsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n := by
    simp only [rowsOf]; exact List.map_congr_left (fun r hr' => hup r (List.mem_range.1 hr'))
  rw [e, Topo.arcCount, hnb, List.length_flatten]

/-- The checks `nd::nested_dissection_order` runs before its backend (`nd.rs:48-58`): too large, malformed, not
    symmetric and simple. The backend (`nd.rs:60-91`, feral) is any function: `from_order` checks what it returns. -/
def ndChecks (bs : List Nat → Nat → Bool) (backend : Csr → Except OErr (List Nat)) (sk : Csr) :
    Except OErr (List Nat) :=
  if sk.n > 2 ^ 60 - 1 then .error .tooLarge
  else if !sk.isWellFormed then .error .malformedSkeleton
  else if !sk.isSymmetricSimple bs then .error .malformedSkeleton
  else backend sk

theorem ndChecks_symm (bs : List Nat → Nat → Bool) (backend : Csr → Except OErr (List Nat)) :
    ∀ sk o, ndChecks bs backend sk = .ok o → sk.isSymmetricSimple bs = true := by
  intro sk o h
  unfold ndChecks at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · rename_i h3; simpa using h3

/-- **`Topology::build` plays the game whatever order the backend returns** (`order.rs:180-182`). -/
theorem build_game (bs : List Nat → Nat → Bool) (hbs : NI.Cch.BsSpec bs) (backend : Csr → Except OErr (List Nat))
    {sk : Csr} {t : Topo} (h : build (ndChecks bs backend) sk = .ok t) :
    ∀ r, r < sk.n → t.up r = gameRow sk.n (rankAdj sk t.r2n t.n2r) r :=
  build_spec bs hbs _ (ndChecks_symm bs backend) h

/-- The accessors of an `Ok` topology (`order.rs:214-268`): `node_of`/`rank_of` are inverse, `ranks` is
    `0 .. n`, and `up_range(r)` is `offsets[r] .. offsets[r+1]`. -/
theorem fromOrder_accessors {sk : Csr} {r2n : List Nat} {t : Topo} (h : fromOrder sk r2n = .ok t)
    (hm : Mirrored sk) :
    t.r2n = r2n ∧ t.ranks = List.range sk.n ∧
    (∀ r, r < sk.n → t.nodeOf r < sk.n ∧ t.rankOf (t.nodeOf r) = r) ∧
    (∀ v, v < sk.n → t.rankOf v < sk.n ∧ t.nodeOf (t.rankOf v) = v) ∧
    (∀ r, r < sk.n → t.upRange r = ((offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD r 0,
      (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD (r + 1) 0)) := by
  have hr : t.r2n = r2n := by
    unfold fromOrder at h
    cases hwf : sk.isWellFormed <;> simp [hwf] at h
    cases hinv : invert r2n sk.n <;> simp [hinv] at h
    rw [← h]
  obtain ⟨hok, hn, _, _, hoff, _, _⟩ := fromOrder_spec h hm
  refine ⟨hr, by simp [Topo.ranks, hn], fun r hr' => ?_, fun v hv => ?_, fun r _ => by simp [Topo.upRange, rowSlots, hoff]⟩
  · simp only [Topo.nodeOf, Topo.rankOf, hr]; exact ⟨hok.r2n_lt r hr', hok.inv1 r hr'⟩
  · simp only [Topo.nodeOf, Topo.rankOf, hr]; exact ⟨hok.n2r_lt v hv, hok.inv2 v hv⟩

/-! ## Arc slots and ids (theorem 4) -/

section Arcs
variable {sk : Csr} {r2n : List Nat} {t : Topo}

/-- **`arc_slot` finds exactly the arcs**, at `offsets[lower] + i` where `up(lower)[i] = higher`. -/
theorem arcSlot_spec (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) {bs : List Nat → Nat → Option Nat}
    (hbs : BsIdx bs) {e o : Nat} (he : e < sk.n) (ho : o < sk.n) (k : Nat) :
    arcSlot bs t e o = some k ↔ ∃ i,
      (gameRow sk.n (rankAdj sk r2n t.n2r) (if e < o then e else o))[i]? = some (if e < o then o else e) ∧
      k = (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD (if e < o then e else o) 0 + i := by
  obtain ⟨hok, _, hup, _, hoff, _, _⟩ := fromOrder_spec h hm
  have hl : (if e < o then e else o) < sk.n := by split <;> omega
  have hrow := hup _ hl
  have hstart : (t.upRange (if e < o then e else o)).1 =
      (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD (if e < o then e else o) 0 := by
    simp [Topo.upRange, rowSlots, hoff]
  simp only [arcSlot]
  unfold Topo.up at hrow
  rw [hrow, hstart]
  constructor
  · intro hk
    obtain ⟨i, hi, rfl⟩ := Option.map_eq_some_iff.1 hk
    exact ⟨i, (hbs _ _ (gameRow_sorted _ _ _) i).1 hi, rfl⟩
  · rintro ⟨i, hi, rfl⟩
    exact Option.map_eq_some_iff.2 ⟨i, (hbs _ _ (gameRow_sorted _ _ _) i).2 hi, rfl⟩

/-- `arc_slot` is `Some` exactly for the chordal supergraph's arcs, in either argument order. -/
theorem arcSlot_isSome (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) {bs : List Nat → Nat → Option Nat}
    (hbs : BsIdx bs) {e o : Nat} (he : e < sk.n) (ho : o < sk.n) (heo : e < o) :
    (arcSlot bs t e o).isSome = up (rankAdj sk r2n t.n2r) e o ∧ arcSlot bs t e o = arcSlot bs t o e := by
  have hsym : arcSlot bs t e o = arcSlot bs t o e := by
    simp only [arcSlot, heo, ite_true, show ¬ o < e by omega, ite_false]
  refine ⟨?_, hsym⟩
  have hs := rankAdj_supp (fromOrder_spec h hm).1
  apply Bool.eq_iff_iff.2
  rw [Option.isSome_iff_exists]
  constructor
  · rintro ⟨k, hk⟩
    obtain ⟨i, hi, _⟩ := (arcSlot_spec h hm hbs he ho k).1 hk
    simp only [heo, ite_true] at hi
    exact (mem_gameRow hs).1 (List.mem_of_getElem? hi)
  · intro hu
    obtain ⟨i, hie⟩ := List.mem_iff_getElem?.1 ((mem_gameRow hs).2 hu)
    exact ⟨_, (arcSlot_spec h hm hbs he ho _).2 ⟨i, by simp only [heo, ite_true]; exact hie, rfl⟩⟩

/-- **Row-major ids (03 §2 "Fresh build: `id = Ap[lo] + i`", §4 step 4).** With `Ax[k] = k`, entry `i` of row
    `lo` holds `up(lo)[i]` at position `Ap[lo] + i`, which is its id, and `arc_slot` returns that id. Every id
    below the arc count belongs to exactly one arc. -/
theorem arc_ids (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk) {bs : List Nat → Nat → Option Nat}
    (hbs : BsIdx bs) :
    (∀ lo i, lo < sk.n → i < (gameRow sk.n (rankAdj sk r2n t.n2r) lo).length →
      t.upNb[(offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD lo 0 + i]? =
        (gameRow sk.n (rankAdj sk r2n t.n2r) lo)[i]? ∧
      ∀ hi, (gameRow sk.n (rankAdj sk r2n t.n2r) lo)[i]? = some hi →
        arcSlot bs t lo hi = some ((offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD lo 0 + i)) ∧
    (∀ k, k < t.arcCount → ∃ lo i, lo < sk.n ∧ i < (gameRow sk.n (rankAdj sk r2n t.n2r) lo).length ∧
      k = (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD lo 0 + i) ∧
    (∀ lo lo' i i', lo < sk.n → lo' < sk.n → i < (gameRow sk.n (rankAdj sk r2n t.n2r) lo).length →
      i' < (gameRow sk.n (rankAdj sk r2n t.n2r) lo').length →
      (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD lo 0 + i =
        (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD lo' 0 + i' → lo = lo' ∧ i = i') := by
  obtain ⟨hok, _, hup, _, hoff, hnb, hcnt⟩ := fromOrder_spec h hm
  have hs := rankAdj_supp hok
  refine ⟨fun lo i hlo hi => ⟨by rw [hnb]; exact flatten_getElem? _ hlo hi, fun hi' hget => ?_⟩,
    fun k hk => ?_, fun lo lo' i i' hlo hlo' hi hi' he => pos_inj _ hlo hlo' hi hi' he⟩
  · have hu : up (rankAdj sk r2n t.n2r) lo hi' = true := (mem_gameRow hs).1 (List.mem_of_getElem? hget)
    have hlt := up_lt hu
    have hhi : hi' < sk.n := (up_supp hs hu).2
    refine (arcSlot_spec h hm hbs hlo hhi _).2 ⟨i, ?_, ?_⟩
    · simp only [hlt, ite_true]; exact hget
    · simp only [hlt, ite_true]
  · rw [hcnt] at hk
    obtain ⟨lo, hlo, h1, h2⟩ := locate _ hk
    exact ⟨lo, k - (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD lo 0, hlo, by omega, by omega⟩

end Arcs

/-! ## `Down` holds the same ids (I7) -/

/-- **I7 for a host transpose that carries ids**: `transposeG` writing each arc's id (its slot, `Ax[k] = k`)
    gives `Down` rows whose `j`-th entry is the `j`-th lower end `l` of `h` and carries `slotOf l h`, the id of
    `Up(l, h)`. -/
theorem down_ids (R : Nat → List Nat) (n : Nat) (hsort : ∀ l, l < n → CowBTree.Sorted (R l))
    (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n) (dflt : Nat) :
    transposeG (fun _ slot => slot) dflt (offsOf R n) (rowsOf R n).flatten =
      (offsOf (lowersK R n) n, (rowsOf (fun h => (lowersK R n h).map (fun l => slotOf R n l h)) n).flatten) := by
  rw [transposeG_spec hsort hlt]; rfl

/-- `slotOf l h` is the position of `h` in `Up`'s row `l`: the id `Up(l, h)` holds on a fresh build. -/
theorem slotOf_is_up_id (R : Nat → List Nat) {n l h : Nat} (hl : l < n) (hh : h ∈ R l) :
    (rowsOf R n).flatten[slotOf R n l h]? = some h := by
  have hi : (R l).idxOf h < (R l).length := List.idxOf_lt_length_iff.2 hh
  rw [slotOf, flatten_getElem? R hl hi, List.getElem?_eq_getElem hi, List.getElem_idxOf hi]

/-- 03 §4 step 4 builds `Down.base` with `GrB_transpose` of a read-only load. Its C contract (GraphBLAS
    `GrB_transpose`: `C(j,i) = A(i,j)`, values carried), on the entries of the two matrices. -/
def TransposeContract (upEntry downEntry : Nat → Nat → Nat → Prop) : Prop :=
  ∀ h l x, downEntry h l x ↔ upEntry l h x

/-- `Up.base`'s entries with 03 §4 step 4's row-major ids (`Ax[k] = k`): `(l, h)` holds `x` when position `x` is
    in row `l`'s slots and `upward.neighbors[x] = h`. -/
def upEntry (t : Topo) (l h x : Nat) : Prop :=
  l < t.nodeCount ∧ (t.upRange l).1 ≤ x ∧ x < (t.upRange l).2 ∧ t.upNb[x]? = some h

/-- **I7 for the row-major ids** (03 §2 I7, §4 step 4). If `Down` is `GrB_transpose` of `Up` with `Ax[k] = k`,
    then `Down(h, l)` holds `x` exactly when `h ∈ up(l)` and `x` is that arc's id, the one `arc_slot(l, h)`
    returns. -/
theorem i7_of_contract {sk : Csr} {r2n : List Nat} {t : Topo} (h : fromOrder sk r2n = .ok t) (hm : Mirrored sk)
    {bs : List Nat → Nat → Option Nat} (hbs : BsIdx bs) (downEntry : Nat → Nat → Nat → Prop)
    (hc : TransposeContract (upEntry t) downEntry) (hh l x : Nat) :
    downEntry hh l x ↔ l < sk.n ∧ hh ∈ t.up l ∧ arcSlot bs t l hh = some x := by
  obtain ⟨hok, hn, hup, _, _, hnb, _⟩ := fromOrder_spec h hm
  obtain ⟨_, _, _, _, hrange⟩ := fromOrder_accessors h hm
  obtain ⟨hids, _, _⟩ := arc_ids h hm hbs
  have hs := rankAdj_supp hok
  rw [hc, upEntry, hn]
  constructor
  · rintro ⟨hl, h1, h2, h3⟩
    rw [hrange l hl] at h1 h2
    simp only at h1 h2
    rw [offsOf_getD_succ _ hl] at h2
    have hi : x - (offsOf (gameRow sk.n (rankAdj sk r2n t.n2r)) sk.n).getD l 0 <
        (gameRow sk.n (rankAdj sk r2n t.n2r) l).length := by omega
    obtain ⟨e1, e2⟩ := hids l _ hl hi
    rw [Nat.add_sub_cancel' h1] at e1 e2
    rw [e1] at h3
    refine ⟨hl, by rw [hup l hl]; exact List.mem_of_getElem? h3, ?_⟩
    exact e2 hh h3
  · rintro ⟨hl, hm', hslot⟩
    rw [hup l hl, mem_gameRow hs] at hm'
    have hlt := up_lt hm'
    have hhn : hh < sk.n := (up_supp hs hm').2
    obtain ⟨i, hi, hx⟩ := (arcSlot_spec h hm hbs hl hhn x).1 hslot
    simp only [hlt, ite_true] at hi hx
    subst hx
    have hilen : i < (gameRow sk.n (rankAdj sk r2n t.n2r) l).length := (List.getElem?_eq_some_iff.1 hi).1
    rw [hrange l hl]
    dsimp only
    refine ⟨hl, Nat.le_add_right _ _, ?_, ?_⟩
    · rw [offsOf_getD_succ _ hl]; omega
    · rw [(hids l i hl hilen).1]; exact hi

/-! ## The contracts are satisfiable -/

/-- The standard library's behaviour satisfies `BsIdx`: on a strictly sorted slice, `Some` of the value's index. -/
theorem bsIdx_satisfiable : BsIdx (fun l x => if x ∈ l then some (l.idxOf x) else none) := by
  intro l x hl i
  have hnd : l.Nodup := hl.imp Nat.ne_of_lt
  by_cases hx : x ∈ l
  · simp only [hx, ite_true, Option.some.injEq]
    have hlt : l.idxOf x < l.length := List.idxOf_lt_length_iff.2 hx
    constructor
    · rintro rfl; rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    · intro hi
      obtain ⟨hil, hget⟩ := List.getElem?_eq_some_iff.1 hi
      rw [← hget]; exact List.Nodup.idxOf_getElem hnd i hil
  · simp only [hx, ite_false, reduceCtorEq, false_iff]
    intro hi; exact hx (List.mem_of_getElem? hi)

/-- A `Down` that is exactly the transpose satisfies `TransposeContract`. -/
example (t : Topo) : TransposeContract (upEntry t) (fun h l x => upEntry t l h x) := fun _ _ _ => Iff.rfl

end CchSG
