import CchSupergraphReview.Csr
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# `transpose`, the counting sort (`order.rs:439-467` @main)

| Lean | Rust (`order.rs` @main) |
| --- | --- |
| `countLoop` | `offsets[higher.index() + 1] += 1` over `upward.neighbors` (`:444-447`) |
| `prefixLoop` | the running total, in place (`:448-452`) |
| `fillRow`, `fillLoop` | `neighbors[next_free[higher]] = lower; next_free[higher] += 1` per row (`:454-461`) |
| `transposeG val` | the whole function, writing `val lower slot` where Rust writes `lower` (`val = fun l _ => l`) |
| `transpose` | `transpose` (`:441-467`): `val` is `lower`, the array starts as `Rank::new(0)` |

`transposeG` takes the value to write as a parameter so that one proof covers today's `transpose` (it writes the
lower rank) and a transpose that carries each arc's id (it writes the arc's slot in `Up`, which is its id on a
fresh build, 03 §2): `transposeG_spec`.

Proved for any upward rows `R` that are strictly increasing with entries below `n` (`chordal_supergraph`'s rows,
`gameRow`, are; `gameRow_sorted`, `up_supp`): the result's row `h` is `lowersK R n h`, every `l` with `h ∈ R l`,
ascending, each carrying `val l (slot of (l, h) in Up)`. Indexing is `getD`, `List.set` and `List.modify`, which
do nothing out of range where Rust panics; `transpose_in_range` proves every index is in range when it is used.
-/
namespace CchSG

/-- `offsets[higher + 1] += 1` for each entry (`order.rs:445-447`). -/
def countLoop : List Nat → List Nat → List Nat
  | offs, [] => offs
  | offs, h :: t => countLoop (offs.modify (h + 1) (· + 1)) t

/-- The running total (`order.rs:448-452`). -/
def prefixLoop : Nat → List Nat → List Nat
  | _, [] => []
  | rt, o :: os => (rt + o) :: prefixLoop (rt + o) os

/-- `for &higher in upward.row(lower)` (`order.rs:457-460`); `slot` is the entry's position in `upward.neighbors`. -/
def fillRow (val : Nat → Nat → Nat) (lower : Nat) : Nat → List Nat → List Nat × List Nat → List Nat × List Nat
  | _, [], s => s
  | slot, h :: hs, (nb, nf) =>
    fillRow val lower (slot + 1) hs (nb.set (nf.getD h 0) (val lower slot), nf.modify h (· + 1))

/-- `for lower in (0..rank_count).map(Rank::new)` (`order.rs:456`). -/
def fillLoop (val : Nat → Nat → Nat) (upOff upNb : List Nat) : List Nat → List Nat × List Nat → List Nat × List Nat
  | [], s => s
  | lower :: ls, s =>
    fillLoop val upOff upNb ls (fillRow val lower (upOff.getD lower 0) (slice upNb (rowSlots upOff lower)) s)

def transposeG (val : Nat → Nat → Nat) (dflt : Nat) (upOff upNb : List Nat) : List Nat × List Nat :=
  let rc := upOff.length - 1
  let offsets := prefixLoop 0 (countLoop (List.replicate (rc + 1) 0) upNb)
  (offsets, (fillLoop val upOff upNb (List.range rc) (List.replicate upNb.length dflt, offsets)).1)

/-- `transpose` (`order.rs:441-467`). -/
def transpose (upOff upNb : List Nat) : List Nat × List Nat := transposeG (fun lower _ => lower) 0 upOff upNb

/-! ## The rows of the transpose -/

/-- The lower ends among `0 .. k-1` adjacent to `h`, ascending: row `h` of the transpose once `k` rows are done. -/
def lowersK (R : Nat → List Nat) (k h : Nat) : List Nat := (List.range k).filter (fun l => (R l).contains h)

/-- The slot of arc `(l, h)` in `Up`: `offsets[l] + index of h in row l`. -/
def slotOf (R : Nat → List Nat) (n l h : Nat) : Nat := (offsOf R n).getD l 0 + (R l).idxOf h

/-- Row `h` of the transpose's values once `k` rows are done. -/
def valsK (val : Nat → Nat → Nat) (R : Nat → List Nat) (n k h : Nat) : List Nat :=
  (lowersK R k h).map (fun l => val l (slotOf R n l h))

section
variable (R : Nat → List Nat)

theorem lowersK_succ (k h : Nat) :
    lowersK R (k + 1) h = lowersK R k h ++ (if h ∈ R k then [k] else []) := by
  simp only [lowersK, List.range_succ, List.filter_append, List.filter_cons, List.filter_nil, List.contains_iff_mem]

theorem mem_lowersK {k h l : Nat} : l ∈ lowersK R k h ↔ l < k ∧ h ∈ R l := by
  simp [lowersK]

theorem lowersK_length_mono {k k' h : Nat} (hk : k ≤ k') : (lowersK R k h).length ≤ (lowersK R k' h).length := by
  induction k' with
  | zero => have : k = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ k' ih =>
    by_cases e : k = k' + 1
    · subst e; exact Nat.le_refl _
    · rw [lowersK_succ]; simp only [List.length_append]; have := ih (by omega); omega

theorem lowersK_length_lt {k n h : Nat} (hk : k < n) (hh : h ∈ R k) :
    (lowersK R k h).length < (lowersK R n h).length := by
  have := lowersK_length_mono R (k := k + 1) (k' := n) (h := h) hk
  rw [lowersK_succ, ite_eq_left hh] at this; simp at this; omega

theorem valsK_succ (val : Nat → Nat → Nat) (n k h : Nat) :
    valsK val R n (k + 1) h = valsK val R n k h ++ (if h ∈ R k then [val k (slotOf R n k h)] else []) := by
  simp only [valsK, lowersK_succ, List.map_append]; split <;> simp

theorem valsK_length (val : Nat → Nat → Nat) (n k h : Nat) : (valsK val R n k h).length = (lowersK R k h).length := by
  simp [valsK]
end

/-! ## Counting -/

theorem countLoop_spec : ∀ (L offs : List Nat), (∀ x ∈ L, x + 1 < offs.length) →
    (countLoop offs L).length = offs.length ∧
    ∀ i, i < offs.length → (countLoop offs L)[i]? = some (offs.getD i 0 + if i = 0 then 0 else L.count (i - 1))
  | [], offs, _ => ⟨rfl, fun i hi => by simp [countLoop, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi]⟩
  | x :: L, offs, hx => by
    have hx0 : x + 1 < offs.length := hx x (by simp)
    obtain ⟨h1, h2⟩ := countLoop_spec L (offs.modify (x + 1) (· + 1))
      (fun y hy => by rw [List.length_modify]; exact hx y (by simp [hy]))
    refine ⟨by rw [countLoop, h1, List.length_modify], fun i hi => ?_⟩
    rw [countLoop, h2 i (by rw [List.length_modify]; exact hi)]
    simp only [List.getD_eq_getElem?_getD, List.getElem?_modify, List.getElem?_eq_getElem hi, Option.map_eq_map,
      Option.map_some, Option.getD_some, List.count_cons, beq_iff_eq]
    by_cases h0 : i = 0
    · subst h0; simp
    · simp only [h0, ite_false]
      by_cases hxi : x + 1 = i
      · subst hxi; simp; omega
      · have : ¬ x = i - 1 := by omega
        simp [hxi, this]

theorem count_flatten_rows (R : Nat → List Nat) (h : Nat) : ∀ k, (∀ l, l < k → (R l).Nodup) →
    ((rowsOf R k).flatten).count h = (lowersK R k h).length
  | 0, _ => by simp [rowsOf, lowersK]
  | k + 1, hnd => by
    rw [rowsOf_succ, List.flatten_append, List.count_append, count_flatten_rows R h k (fun l hl => hnd l (by omega)),
      lowersK_succ]
    simp only [List.flatten_cons, List.flatten_nil, List.append_nil, List.length_append]
    rw [List.Nodup.count (hnd k (by omega))]
    split <;> simp

theorem prefixLoop_spec : ∀ (cs : List Nat) (rt i : Nat),
    (prefixLoop rt cs)[i]? = if i < cs.length then some (rt + (cs.take (i + 1)).sum) else none
  | [], rt, i => by simp [prefixLoop]
  | c :: cs, rt, 0 => by simp [prefixLoop]
  | c :: cs, rt, i + 1 => by
    simp only [prefixLoop, List.getElem?_cons_succ, prefixLoop_spec cs (rt + c) i, List.length_cons,
      List.take_succ_cons, List.sum_cons]
    split <;> split <;> first | omega | simp [Nat.add_assoc]

theorem sum_map_add (f g : Nat → Nat) : ∀ (l : List Nat), (l.map (fun h => f h + g h)).sum = (l.map f).sum + (l.map g).sum
  | [] => rfl
  | a :: l => by simp only [List.map_cons, List.sum_cons, sum_map_add f g l]; omega

theorem sum_ind (x : Nat) : ∀ m, ((List.range m).map (fun h => if x = h then 1 else 0)).sum = if x < m then 1 else 0
  | 0 => rfl
  | m + 1 => by
    rw [List.range_succ, List.map_append, List.sum_append, sum_ind x m]
    by_cases e : x = m
    · subst e; simp
    · by_cases hlt : x < m
      · simp [e, hlt, show x < m + 1 by omega]
      · simp [e, hlt, show ¬ x < m + 1 by omega]

theorem sum_zero (l : List Nat) : (l.map (fun _ => (0 : Nat))).sum = 0 := by
  induction l with
  | nil => rfl
  | cons a l ih => simp only [List.map_cons, List.sum_cons, ih]

theorem sum_count_range (L : List Nat) : ∀ n, (∀ x ∈ L, x < n) → ((List.range n).map (fun h => L.count h)).sum = L.length := by
  induction L with
  | nil => intro n _; simp only [List.count_nil, sum_zero, List.length_nil]
  | cons x L ih =>
    intro n hlt
    have hx : x < n := hlt x (by simp)
    have e : (List.range n).map (fun h => (x :: L).count h) =
        (List.range n).map (fun h => L.count h + if x = h then 1 else 0) :=
      List.map_congr_left (fun h _ => by simp [List.count_cons, beq_iff_eq])
    rw [e, sum_map_add, sum_ind, ih n (fun y hy => hlt y (by simp [hy])), ite_eq_left hx]; simp

/-! ## The invariant of the fill -/

section Fill
variable (val : Nat → Nat → Nat) (R : Nat → List Nat) (n : Nat)

/-- The transpose's offsets: `oD h` is where row `h` starts. -/
abbrev oD (h : Nat) : Nat := (offsOf (lowersK R n) n).getD h 0

/-- After rows `0 .. k-1`: `next_free[h] = oD h + |lowersK k h|`, and those positions hold `valsK k h`. -/
structure FI (T k : Nat) (s : List Nat × List Nat) : Prop where
  nbLen : s.1.length = T
  nfLen : s.2.length = n + 1
  nf : ∀ h, h < n → s.2.getD h 0 = oD R n h + (lowersK R k h).length
  nb : ∀ h, h < n → ∀ j, j < (lowersK R k h).length → s.1[oD R n h + j]? = (valsK val R n k h)[j]?

/-- Inside row `k`, after its entries `pre`. -/
structure FIr (T k : Nat) (pre : List Nat) (s : List Nat × List Nat) : Prop where
  nbLen : s.1.length = T
  nfLen : s.2.length = n + 1
  nf : ∀ h, h < n → s.2.getD h 0 = oD R n h + (lowersK R k h).length + (if h ∈ pre then 1 else 0)
  nb : ∀ h, h < n → ∀ j, j < (lowersK R k h).length → s.1[oD R n h + j]? = (valsK val R n k h)[j]?
  done : ∀ h ∈ pre, s.1[oD R n h + (lowersK R k h).length]? = some (val k (slotOf R n k h))
end Fill

section FillProof
variable {val : Nat → Nat → Nat} {R : Nat → List Nat} {n : Nat}

/-- Rows of the transpose are disjoint: a position in row `h` is in no other row. -/
theorem oD_inj {h h' j j' : Nat} (hh : h < n) (hh' : h' < n) (hj : j < (lowersK R n h).length)
    (hj' : j' < (lowersK R n h').length) (he : oD R n h + j = oD R n h' + j') : h = h' ∧ j = j' :=
  pos_inj (lowersK R n) hh hh' hj hj' he

/-- **Every write of the fill is in range**: in a state inside row `k` (`FIr`), the next entry `x` of the row
    has `x < next_free.len()` and `next_free[x] < neighbors.len()` (`order.rs:458-459`). -/
theorem fillRow_write_in_range (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n)
    {T k : Nat} (hk : k < n) (hT : oD R n n ≤ T) {pre : List Nat} {s : List Nat × List Nat}
    (hI : FIr val R n T k pre s) {x : Nat} (hxR : x ∈ R k) (hxpre : x ∉ pre) :
    x < s.2.length ∧ s.2.getD x 0 = oD R n x + (lowersK R k x).length ∧
      oD R n x + (lowersK R k x).length < s.1.length := by
  have hxn : x < n := hlt k hk x hxR
  have hltx := lowersK_length_lt R hk hxR
  refine ⟨by rw [hI.nfLen]; omega, by rw [hI.nf x hxn, ite_eq_right hxpre]; simp, ?_⟩
  have h1 : oD R n (x + 1) = oD R n x + (lowersK R n x).length := offsOf_getD_succ (lowersK R n) hxn
  have h2 : oD R n (x + 1) ≤ oD R n n := offsOf_mono (lowersK R n) (by omega : x + 1 ≤ n) (Nat.le_refl n)
  rw [hI.nbLen]; omega

/-- The inner loop over any stretch `rest` of row `k` that follows `pre` keeps the invariant. -/
theorem fillRow_spec (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n)
    {T k : Nat} (hk : k < n) (hT : oD R n n ≤ T) :
    ∀ (rest pre suf : List Nat) (s : List Nat × List Nat), pre ++ rest ++ suf = R k → FIr val R n T k pre s →
      FIr val R n T k (pre ++ rest) (fillRow val k ((offsOf R n).getD k 0 + pre.length) rest s)
  | [], pre, suf, s, hpre, hI => by simpa [fillRow] using hI
  | x :: rest, pre, suf, (nb, nf), hpre, hI => by
    have hnd : (R k).Nodup := (hsort k hk).imp Nat.ne_of_lt
    have hxR : x ∈ R k := by rw [← hpre]; simp
    have hxn : x < n := hlt k hk x hxR
    have hxpre : x ∉ pre := by
      rw [← hpre, List.append_assoc] at hnd
      exact fun h => (List.nodup_append.1 hnd).2.2 x h x (by simp) rfl
    have hidx : (R k).idxOf x = pre.length := by
      have hlen : pre.length < (R k).length := by rw [← hpre]; simp
      have hget : (R k)[pre.length] = x := by simp only [← hpre]; simp
      rw [← hget]; exact List.Nodup.idxOf_getElem hnd _ hlen
    have hltx := lowersK_length_lt R hk hxR
    obtain ⟨_, hpos, hposT⟩ := fillRow_write_in_range hsort hlt hk hT hI hxR hxpre
    simp only at hpos hposT
    have hnext := fillRow_spec hsort hlt hk hT rest (pre ++ [x]) suf
      (nb.set (nf.getD x 0) (val k ((offsOf R n).getD k 0 + pre.length)), nf.modify x (· + 1))
      (by rw [← hpre]; simp) ?_
    · simpa [fillRow, Nat.add_assoc] using hnext
    refine ⟨by simp [hI.nbLen], by simp [hI.nfLen], fun h hh => ?_, fun h hh j hj => ?_, fun h hh => ?_⟩
    · simp only [List.getD_eq_getElem?_getD, List.getElem?_modify]
      have := hI.nf h hh
      simp only [List.getD_eq_getElem?_getD] at this
      have hhl : h < nf.length := by rw [hI.nfLen]; omega
      rw [List.getElem?_eq_getElem hhl] at this ⊢
      simp only [Option.map_eq_map, Option.map_some, Option.getD_some] at this ⊢
      by_cases e : x = h
      · subst e; simp [hxpre] at this ⊢; omega
      · simp [e, Ne.symm e] at this ⊢; omega
    · rw [hpos, List.getElem?_set_ne]
      · exact hI.nb h hh j hj
      · intro he
        have hj' : j < (lowersK R n h).length :=
          Nat.lt_of_lt_of_le hj (lowersK_length_mono R (Nat.le_of_lt hk))
        obtain ⟨rfl, rfl⟩ := oD_inj hxn hh hltx hj' he
        omega
    · rw [hpos]
      rcases List.mem_append.1 hh with hh | hh
      · have hhR : h ∈ R k := by rw [← hpre]; simp [hh]
        have hhn : h < n := hlt k hk h hhR
        rw [List.getElem?_set_ne]
        · exact hI.done h hh
        · intro he
          obtain ⟨rfl, _⟩ := oD_inj hxn hhn hltx (lowersK_length_lt R hk hhR) he
          exact hxpre hh
      · simp at hh; subst hh
        rw [List.getElem?_set_self hposT, slotOf, hidx]

/-- One row of the fill (`order.rs:457-460` for `lower = k`) takes the invariant from `k` rows to `k + 1`. -/
theorem fi_row (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n)
    {T k : Nat} (hT : oD R n n ≤ T) (hkn : k < n) {s : List Nat × List Nat} (hI : FI val R n T k s) :
    FI val R n T (k + 1) (fillRow val k ((offsOf R n).getD k 0)
      (slice (rowsOf R n).flatten (rowSlots (offsOf R n) k)) s) := by
  have hrow : slice (rowsOf R n).flatten (rowSlots (offsOf R n) k) = R k := slice_rowOf R hkn
  have hstart : FIr val R n T k [] s :=
    ⟨hI.nbLen, hI.nfLen, fun h hh => by rw [hI.nf h hh]; simp, hI.nb, fun h hh => by simp at hh⟩
  have hrowI := fillRow_spec hsort hlt hkn hT (R k) [] [] s (by simp) hstart
  simp only [List.length_nil, Nat.add_zero, List.nil_append] at hrowI
  rw [hrow]
  refine ⟨hrowI.nbLen, hrowI.nfLen, fun h hh => ?_, fun h hh j hj => ?_⟩
  · rw [hrowI.nf h hh, lowersK_succ]; split <;> simp <;> omega
  · rw [lowersK_succ] at hj
    rw [valsK_succ]
    by_cases hj0 : j < (lowersK R k h).length
    · rw [hrowI.nb h hh j hj0, List.getElem?_append_left (by rw [valsK_length]; exact hj0)]
    · by_cases hhR : h ∈ R k
      · simp only [hhR, ite_true, List.length_append, List.length_singleton] at hj
        have hje : j = (lowersK R k h).length := by omega
        subst hje
        rw [hrowI.done h hhR, List.getElem?_append_right (by rw [valsK_length]; exact Nat.le_refl _),
          ite_eq_left hhR, valsK_length]
        simp
      · simp only [hhR, ite_false, List.length_append, List.length_nil, Nat.add_zero] at hj; omega

theorem fillLoop_spec (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n)
    {T : Nat} (hT : oD R n n ≤ T) :
    ∀ (ls : List Nat) (k : Nat) (s : List Nat × List Nat), ls = List.range' k (n - k) → k ≤ n →
      FI val R n T k s →
      FI val R n T n (fillLoop val (offsOf R n) (rowsOf R n).flatten ls s)
  | [], k, s, hls, hk, hI => by
    have : n - k = 0 := by
      cases h : n - k with
      | zero => rfl
      | succ m => rw [h] at hls; simp [List.range'] at hls
    have : k = n := by omega
    subst this; exact hI
  | l :: ls, k, s, hls, hk, hI => by
    have hkn : k < n := by
      cases h : n - k with
      | zero => rw [h] at hls; simp [List.range'] at hls
      | succ m => omega
    have hl : l = k := by
      obtain ⟨m, hm⟩ : ∃ m, n - k = m + 1 := ⟨n - k - 1, by omega⟩
      rw [hm] at hls; simp [List.range'] at hls; exact hls.1
    have hls' : ls = List.range' (k + 1) (n - (k + 1)) := by
      obtain ⟨m, hm⟩ : ∃ m, n - k = m + 1 := ⟨n - k - 1, by omega⟩
      rw [hm] at hls; simp [List.range'] at hls
      rw [hls.2]; congr 1; omega
    subst hl
    exact fillLoop_spec hsort hlt hT ls (l + 1) _ hls' (by omega) (fi_row hsort hlt hT hkn hI)

theorem fillLoop_append (upOff upNb : List Nat) : ∀ (l1 l2 : List Nat) (s : List Nat × List Nat),
    fillLoop val upOff upNb (l1 ++ l2) s = fillLoop val upOff upNb l2 (fillLoop val upOff upNb l1 s)
  | [], l2, s => rfl
  | x :: l1, l2, s => by simp only [List.cons_append, fillLoop]; exact fillLoop_append upOff upNb l1 l2 _

/-- The fill's state after rows `0 .. k-1` satisfies the invariant. -/
theorem fi_prefix (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n)
    {T : Nat} (hT : oD R n n ≤ T) {s0 : List Nat × List Nat} (h0 : FI val R n T 0 s0) :
    ∀ k, k ≤ n → FI val R n T k (fillLoop val (offsOf R n) (rowsOf R n).flatten (List.range k) s0)
  | 0, _ => h0
  | k + 1, hk => by
    rw [List.range_succ, fillLoop_append]
    exact fi_row hsort hlt hT (by omega) (fi_prefix hsort hlt hT h0 k (by omega))

/-- The total: the transpose's rows hold as many entries as `upward.neighbors` (double counting). -/
theorem oD_total (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n) :
    oD R n n = (rowsOf R n).flatten.length := by
  have hnd : ∀ l, l < n → (R l).Nodup := fun l hl => (hsort l hl).imp Nat.ne_of_lt
  have hall : ∀ x ∈ (rowsOf R n).flatten, x < n := by
    intro x hx
    simp only [rowsOf, List.mem_flatten, List.mem_map, List.mem_range] at hx
    obtain ⟨_, ⟨l, hl, rfl⟩, hx⟩ := hx
    exact hlt l hl x hx
  have e : (List.range n).map (fun h => ((rowsOf R n).flatten).count h) =
      (List.range n).map (fun h => (lowersK R n h).length) :=
    List.map_congr_left (fun h _ => count_flatten_rows R h n hnd)
  show (offsOf (lowersK R n) n).getD n 0 = _
  rw [← flatten_length (lowersK R n) n, ← sum_count_range _ n hall, e]
  simp [rowsOf, List.length_flatten, List.map_map, Function.comp_def]

/-- **The counting sort is the transpose** (`order.rs:441-467`). For strictly increasing rows `R` with entries
    below `n`, `transposeG val` returns, for every `h`, the lower ends of `h`'s arcs in increasing order
    (`lowersK R n h`), each carrying `val l (slotOf R n l h)`, under the offsets of those rows. -/
theorem entries_lt (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n) : ∀ x ∈ (rowsOf R n).flatten, x < n := by
  intro x hx
  simp only [rowsOf, List.mem_flatten, List.mem_map, List.mem_range] at hx
  obtain ⟨_, ⟨l, hl, rfl⟩, hx⟩ := hx
  exact hlt l hl x hx

/-- **The counting loop and the running total give the transpose's offsets** (`order.rs:444-452`). -/
theorem transpose_offsets (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n) :
    prefixLoop 0 (countLoop (List.replicate (n + 1) 0) (rowsOf R n).flatten) = offsOf (lowersK R n) n := by
  have hnd : ∀ l, l < n → (R l).Nodup := fun l hl => (hsort l hl).imp Nat.ne_of_lt
  have hall := entries_lt (R := R) hlt
  -- the counts
  have hcount : countLoop (List.replicate (n + 1) 0) (rowsOf R n).flatten =
      0 :: (List.range n).map (fun h => (lowersK R n h).length) := by
    obtain ⟨h1, h2⟩ := countLoop_spec (rowsOf R n).flatten (List.replicate (n + 1) 0)
      (fun x hx => by simp; exact hall x hx)
    apply List.ext_getElem?
    intro i
    by_cases hi : i < n + 1
    · rw [h2 i (by simp; exact hi)]
      cases i with
      | zero => simp [List.getD_eq_getElem?_getD]
      | succ j =>
        simp [List.getD_eq_getElem?_getD, List.getElem?_replicate, count_flatten_rows R j n hnd,
          show j < n by omega]
    · rw [List.getElem?_eq_none (by rw [h1]; simp; omega), List.getElem?_eq_none (by simp; omega)]
  -- the offsets
  have hoffs : prefixLoop 0 (0 :: (List.range n).map (fun h => (lowersK R n h).length)) =
      offsOf (lowersK R n) n := by
    apply List.ext_getElem?
    intro i
    rw [prefixLoop_spec]
    simp only [List.length_cons, List.length_map, List.length_range, List.take_succ_cons, List.sum_cons,
      Nat.zero_add]
    by_cases hi : i < n + 1
    · rw [ite_eq_left hi, offsOf, List.getElem?_map, List.getElem?_range hi, ← List.map_take, List.take_range,
        Nat.min_eq_left (by omega)]
      simp [rowsOf, List.length_flatten, List.map_map, Function.comp_def]
    · rw [ite_eq_right hi, List.getElem?_eq_none (by rw [offsOf_length]; omega)]
  rw [hcount, hoffs]

theorem transposeG_spec (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n)
    (dflt : Nat) :
    transposeG val dflt (offsOf R n) (rowsOf R n).flatten =
      (offsOf (lowersK R n) n, (rowsOf (valsK val R n n) n).flatten) := by
  have hrc : (offsOf R n).length - 1 = n := by rw [offsOf_length]; omega
  -- the total
  have hT : oD R n n = (rowsOf R n).flatten.length := oD_total hsort hlt
  -- the fill
  have hfill := fillLoop_spec (val := val) hsort hlt (T := (rowsOf R n).flatten.length) (Nat.le_of_eq hT)
    (List.range n) 0 (List.replicate (rowsOf R n).flatten.length dflt, offsOf (lowersK R n) n)
    (by simp [List.range_eq_range']) (Nat.zero_le _)
    ⟨by simp, by rw [offsOf_length], fun h hh => by simp [lowersK], fun h hh j hj => by simp [lowersK] at hj⟩
  simp only [transposeG, hrc, transpose_offsets hsort hlt, Prod.mk.injEq, true_and]
  have hlen : (fillLoop val (offsOf R n) (rowsOf R n).flatten (List.range n)
      (List.replicate (rowsOf R n).flatten.length dflt, offsOf (lowersK R n) n)).1.length =
      (offsOf (valsK val R n n) n).getD n 0 := by
    rw [hfill.nbLen, ← hT, ← flatten_length]
    show (offsOf (lowersK R n) n).getD n 0 = _
    rw [← flatten_length]
    simp [rowsOf, List.length_flatten, List.map_map, Function.comp_def, valsK_length]
  apply eq_flatten_of_rows (valsK val R n n) _ hlen
  intro h hh j hj
  have e : (offsOf (valsK val R n n) n).getD h 0 = oD R n h := by
    rw [oD, offsOf_getD _ (Nat.le_of_lt hh), offsOf_getD _ (Nat.le_of_lt hh)]
    simp [rowsOf, List.length_flatten, List.map_map, Function.comp_def, valsK_length]
  rw [e]; exact hfill.nb h hh j (by rw [← valsK_length R val]; exact hj)

/-- **Today's `transpose` returns the downward rows**: row `h` is every lower `l` with `h ∈ up(l)`, ascending. -/
theorem transpose_spec (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n) :
    transpose (offsOf R n) (rowsOf R n).flatten = (offsOf (lowersK R n) n, (rowsOf (lowersK R n) n).flatten) := by
  rw [transpose, transposeG_spec hsort hlt]
  congr 3; funext h; simp [valsK]

/-- The fill's state when it is about to write entry `x` of row `k`, after rows `0 .. k-1` and the entries `pre`
    before `x`, from the real initial state of `transposeG` (`rank_count = n`, `transpose_offsets`). -/
def fillState (val : Nat → Nat → Nat) (R : Nat → List Nat) (n dflt k : Nat) (pre : List Nat) :
    List Nat × List Nat :=
  fillRow val k ((offsOf R n).getD k 0) pre
    (fillLoop val (offsOf R n) (rowsOf R n).flatten (List.range k)
      (List.replicate (rowsOf R n).flatten.length dflt,
       prefixLoop 0 (countLoop (List.replicate (n + 1) 0) (rowsOf R n).flatten)))

/-- **`transpose` never indexes out of range** (`order.rs:441-467`) for strictly increasing rows with entries
    below `n`: `rank_count` is `n`; every count `offsets[higher + 1] += 1` (`:446`) and every row read
    `upward.row(lower)` (`:457`) is in range; and every write `neighbors[next_free[higher]] = ..;
    next_free[higher] += 1` (`:458-459`) is in range in the state where it happens (`fillState`). -/
theorem transpose_in_range (hsort : ∀ l, l < n → CowBTree.Sorted (R l)) (hlt : ∀ l, l < n → ∀ h ∈ R l, h < n)
    (dflt : Nat) :
    (offsOf R n).length - 1 = n ∧
    (∀ x ∈ (rowsOf R n).flatten, x + 1 < (List.replicate (n + 1) 0).length) ∧
    (∀ k, k < n → k + 1 < (offsOf R n).length) ∧
    (∀ k, k < n → ∀ pre x rest, pre ++ x :: rest = R k →
      x < (fillState val R n dflt k pre).2.length ∧
      (fillState val R n dflt k pre).2.getD x 0 < (fillState val R n dflt k pre).1.length) := by
  have hT := oD_total hsort hlt
  refine ⟨by rw [offsOf_length]; omega, fun x hx => by simp; exact entries_lt hlt x hx,
    fun k hk => by rw [offsOf_length]; omega, fun k hk pre x rest hpre => ?_⟩
  unfold fillState
  rw [transpose_offsets hsort hlt]
  have h0 : FI val R n (rowsOf R n).flatten.length 0
      (List.replicate (rowsOf R n).flatten.length dflt, offsOf (lowersK R n) n) :=
    ⟨by simp, by rw [offsOf_length], fun h hh => by simp [lowersK], fun h hh j hj => by simp [lowersK] at hj⟩
  have hk' := fi_prefix (val := val) hsort hlt (Nat.le_of_eq hT) h0 k (Nat.le_of_lt hk)
  have hstart : FIr val R n (rowsOf R n).flatten.length k []
      (fillLoop val (offsOf R n) (rowsOf R n).flatten (List.range k)
        (List.replicate (rowsOf R n).flatten.length dflt, offsOf (lowersK R n) n)) :=
    ⟨hk'.nbLen, hk'.nfLen, fun h hh => by rw [hk'.nf h hh]; simp, hk'.nb, fun h hh => by simp at hh⟩
  have hpreI := fillRow_spec (val := val) hsort hlt hk (Nat.le_of_eq hT) pre [] (x :: rest) _
    (by simpa using hpre) hstart
  simp only [List.nil_append, List.length_nil, Nat.add_zero] at hpreI
  have hnd : (R k).Nodup := (hsort k hk).imp Nat.ne_of_lt
  have hxR : x ∈ R k := by rw [← hpre]; simp
  have hxpre : x ∉ pre := by
    rw [← hpre] at hnd
    exact fun h => (List.nodup_append.1 hnd).2.2 x h x (by simp) rfl
  obtain ⟨h1, h2, h3⟩ := fillRow_write_in_range hsort hlt hk (Nat.le_of_eq hT) hpreI hxR hxpre
  exact ⟨h1, by rw [h2]; exact h3⟩

end FillProof

end CchSG
