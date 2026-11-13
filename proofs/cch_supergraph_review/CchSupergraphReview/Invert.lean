import CchSupergraphReview.Order
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-!
# `invert` (`order.rs:331-360` @main): the order is checked, not trusted

`invert(rank_to_node, node_count)` returns `node_to_rank` or `OrderError::NotAPermutation`. Its loop fills a
`Vec<Option<Rank>>` (`:342-353`): an empty slot takes the rank, a filled one is "lists a node twice", a missing one
is "lists a node the graph does not have". Then `.expect("a permutation sets every slot")` (`:356-359`).

* `invert_ok_spec`: an `Ok` result is the inverse permutation over `0 .. n`, every `expect` succeeds, and
  `OrderOk` holds for any well-formed skeleton of `n` nodes.
* `invert_ok_of_perm`: a permutation (length `n`, no repeat, every id below `n`) always gives `Ok`.
-/
namespace CchSG

/-- `OrderError` (`order.rs:84-98`), one constructor per message this file can produce. -/
inductive OErr where
  | tooLarge
  | malformedSkeleton
  | notAPermutationLen
  | notAPermutationTwice
  | notAPermutationRange
deriving DecidableEq, Repr

/-- The loop (`order.rs:343-353`): `node_to_rank.get_mut(node)`. -/
def invertLoop : Nat → List Nat → List (Option Nat) → Except OErr (List (Option Nat))
  | _, [], acc => .ok acc
  | i, node :: rest, acc =>
    match acc[node]? with
    | some none => invertLoop (i + 1) rest (acc.set node (some i))
    | some (some _) => .error .notAPermutationTwice
    | none => .error .notAPermutationRange

/-- `invert` (`order.rs:333-360`). The `expect` is `getD 0`: `invert_ok_spec` shows every slot is set. -/
def invert (r2n : List Nat) (n : Nat) : Except OErr (List Nat) :=
  if r2n.length ≠ n then .error .notAPermutationLen
  else match invertLoop 0 r2n (List.replicate n none) with
    | .ok acc => .ok (acc.map (fun o => o.getD 0))
    | .error e => .error e

/-- The loop invariant after `i` nodes of `L`: slot `v` holds `j` iff `L[j] = v` for some `j < i`. -/
structure J (L : List Nat) (n i : Nat) (acc : List (Option Nat)) : Prop where
  len : acc.length = n
  lt : ∀ j, j < i → L.getD j 0 < n
  slot : ∀ v, v < n → ∀ j, acc[v]? = some (some j) ↔ (j < i ∧ j < L.length ∧ L.getD j 0 = v)

theorem J_init (L : List Nat) (n : Nat) : J L n 0 (List.replicate n none) :=
  ⟨by simp, fun j h => absurd h (Nat.not_lt_zero _), fun v hv j => by simp [List.getElem?_replicate, hv]⟩

theorem J_step {L : List Nat} {n i : Nat} {acc : List (Option Nat)} (hJ : J L n i acc) (hi : i < L.length)
    (hnone : acc[L.getD i 0]? = some none) : J L n (i + 1) (acc.set (L.getD i 0) (some i)) := by
  have hv : L.getD i 0 < n := by
    have := (List.getElem?_eq_some_iff.1 hnone).1; rw [hJ.len] at this; exact this
  refine ⟨by simp [hJ.len], fun j hj => ?_, fun v hvn j => ?_⟩
  · by_cases hji : j = i
    · subst hji; exact hv
    · exact hJ.lt j (by omega)
  · by_cases hve : L.getD i 0 = v
    · subst hve
      rw [List.getElem?_set_self (by rw [hJ.len]; exact hv)]
      constructor
      · intro h; cases h; exact ⟨by omega, hi, rfl⟩
      · rintro ⟨h1, h2, h3⟩
        by_cases hji : j = i
        · subst hji; rfl
        · have := (hJ.slot _ hv j).2 ⟨by omega, h2, h3⟩
          rw [hnone] at this; cases this
    · rw [List.getElem?_set_ne hve, hJ.slot v hvn j]
      constructor
      · rintro ⟨h1, h2, h3⟩; exact ⟨by omega, h2, h3⟩
      · rintro ⟨h1, h2, h3⟩
        by_cases hji : j = i
        · subst hji; exact absurd h3 hve
        · exact ⟨by omega, h2, h3⟩

theorem getD_drop_head {L rest : List Nat} {i node : Nat} (h : node :: rest = L.drop i) :
    L.getD i 0 = node ∧ rest = L.drop (i + 1) ∧ i < L.length := by
  have h0 : (L.drop i)[0]? = some node := by rw [← h]; rfl
  simp only [List.getElem?_drop, Nat.add_zero] at h0
  refine ⟨by simp [List.getD_eq_getElem?_getD, h0], ?_, (List.getElem?_eq_some_iff.1 h0).1⟩
  have := congrArg List.tail h
  simpa [List.tail_drop] using this

/-- The loop's result keeps the invariant at the end. -/
theorem invertLoop_ok {L : List Nat} {n : Nat} : ∀ (rest : List Nat) (i : Nat) (acc acc' : List (Option Nat)),
    J L n i acc → rest = L.drop i → invertLoop i rest acc = .ok acc' → J L n L.length acc'
  | [], i, acc, acc', hJ, hr, h => by
    have hi : L.length ≤ i := by
      have := congrArg List.length hr; simp at this; omega
    simp only [invertLoop, Except.ok.injEq] at h; subst h
    refine ⟨hJ.len, fun j hj => hJ.lt j (by omega), fun v hv j => ?_⟩
    rw [hJ.slot v hv j]
    exact ⟨fun ⟨_, h2, h3⟩ => ⟨h2, h2, h3⟩, fun ⟨_, h2, h3⟩ => ⟨by omega, h2, h3⟩⟩
  | node :: rest, i, acc, acc', hJ, hr, h => by
    obtain ⟨hnode, hrest, hi⟩ := getD_drop_head hr
    subst hnode
    simp only [invertLoop] at h
    split at h
    · rename_i hn
      exact invertLoop_ok rest (i + 1) _ acc' (J_step hJ hi hn) hrest h
    · cases h
    · cases h

/-- **A permutation never errors**. -/
theorem invertLoop_perm {L : List Nat} {n : Nat} (hnd : L.Nodup) (hlt : ∀ x ∈ L, x < n) :
    ∀ (rest : List Nat) (i : Nat) (acc : List (Option Nat)),
    J L n i acc → rest = L.drop i → ∃ acc', invertLoop i rest acc = .ok acc'
  | [], i, acc, hJ, hr => ⟨acc, rfl⟩
  | node :: rest, i, acc, hJ, hr => by
    obtain ⟨hnode, hrest, hi⟩ := getD_drop_head hr
    have hv : L.getD i 0 < n := hlt _ (by simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi])
    have hnone : acc[L.getD i 0]? = some none := by
      cases hs : acc[L.getD i 0]? with
      | none => rw [List.getElem?_eq_none_iff, hJ.len] at hs; omega
      | some o =>
        cases o with
        | none => rfl
        | some j =>
          obtain ⟨hj, hjl, hjv⟩ := (hJ.slot _ hv j).1 hs
          have : j = i := by
            have h1 : L[j] = L[i] := by
              simp only [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjl,
                List.getElem?_eq_getElem hi, Option.getD_some] at hjv
              exact hjv
            exact (List.Nodup.getElem_inj hnd).1 h1
          omega
    subst hnode
    simp only [invertLoop, hnone]
    exact invertLoop_perm hnd hlt rest (i + 1) _ (J_step hJ hi hnone) hrest

/-! ## Pigeonhole -/

theorem nodup_length_le : ∀ (n : Nat) (l : List Nat), l.Nodup → (∀ x ∈ l, x < n) → l.length ≤ n
  | 0, l, _, h => by
    cases l with
    | nil => simp
    | cons a t => exact absurd (h a (by simp)) (Nat.not_lt_zero _)
  | n + 1, l, hnd, h => by
    have h1 := nodup_length_le n (l.erase n) (hnd.erase n) (fun x hx => by
      have := (hnd.mem_erase_iff).1 hx; have := h x this.2; omega)
    have h2 := @List.length_erase _ _ _ n l
    split at h2 <;> omega

theorem nodup_map_range {f : Nat → Nat} : ∀ (n : Nat), (∀ i j, i < n → j < n → f i = f j → i = j) →
    ((List.range n).map f).Nodup
  | 0, _ => by simp
  | n + 1, hinj => by
    rw [List.range_succ, List.map_append, List.nodup_append]
    refine ⟨nodup_map_range n (fun i j hi hj => hinj i j (by omega) (by omega)), by simp, ?_⟩
    intro a ha b hb
    simp only [List.mem_map, List.mem_range] at ha
    simp only [List.map_cons, List.map_nil, List.mem_singleton] at hb
    obtain ⟨i, hi, rfl⟩ := ha
    subst hb
    intro h; have := hinj i n (by omega) (by omega) h; omega

/-- An injection of `0 .. n` into `0 .. n` reaches everything. -/
theorem inj_surj {f : Nat → Nat} {n : Nat} (hinj : ∀ i j, i < n → j < n → f i = f j → i = j)
    (hlt : ∀ i, i < n → f i < n) (v : Nat) (hv : v < n) : ∃ i, i < n ∧ f i = v := by
  apply Classical.byContradiction
  intro hno
  have hnot : v ∉ (List.range n).map f := by
    simp only [List.mem_map, List.mem_range]; exact fun ⟨i, hi, he⟩ => hno ⟨i, hi, he⟩
  have hnd : (v :: (List.range n).map f).Nodup := List.nodup_cons.2 ⟨hnot, nodup_map_range n hinj⟩
  have := nodup_length_le n _ hnd (fun x hx => by
    rcases List.mem_cons.1 hx with rfl | hx
    · exact hv
    · obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx; exact hlt i (List.mem_range.1 hi))
  simp at this; omega

/-! ## `invert` -/

/-- **An `Ok` from `invert` is the inverse permutation**, and no `expect` fails: every slot was set. -/
theorem invert_ok_spec {r2n : List Nat} {n : Nat} {n2r : List Nat} (h : invert r2n n = .ok n2r) :
    r2n.length = n ∧ n2r.length = n ∧
    (∀ r, r < n → r2n.getD r 0 < n ∧ n2r.getD (r2n.getD r 0) 0 = r) ∧
    (∀ v, v < n → n2r.getD v 0 < n ∧ r2n.getD (n2r.getD v 0) 0 = v) ∧
    (∀ acc, invertLoop 0 r2n (List.replicate n none) = .ok acc → ∀ v, v < n → ∃ j, acc[v]? = some (some j)) := by
  unfold invert at h
  by_cases hl : r2n.length ≠ n
  · simp [hl] at h
  · have hl' : r2n.length = n := by omega
    simp only [hl, ite_false] at h
    cases hloop : invertLoop 0 r2n (List.replicate n none) with
    | error e => simp [hloop] at h
    | ok acc =>
      simp only [hloop, Except.ok.injEq] at h
      subst h
      have hJ := invertLoop_ok r2n 0 _ acc (J_init r2n n) (by simp) hloop
      rw [hl'] at hJ
      -- injective on `0 .. n`
      have hinj : ∀ i j, i < n → j < n → r2n.getD i 0 = r2n.getD j 0 → i = j := by
        intro i j hi hj he
        have h1 := (hJ.slot _ (hJ.lt i hi) i).2 ⟨hi, by omega, rfl⟩
        have h2 := (hJ.slot _ (hJ.lt i hi) j).2 ⟨hj, by omega, he.symm⟩
        rw [h1] at h2; cases h2; rfl
      have hall : ∀ v, v < n → ∃ j, j < n ∧ acc[v]? = some (some j) ∧ r2n.getD j 0 = v := by
        intro v hv
        obtain ⟨j, hj, hjv⟩ := inj_surj hinj hJ.lt v hv
        exact ⟨j, hj, (hJ.slot v hv j).2 ⟨hj, by omega, hjv⟩, hjv⟩
      have hget : ∀ v, v < n → ∀ j, acc[v]? = some (some j) → (acc.map (fun o => o.getD 0)).getD v 0 = j := by
        intro v hv j hj
        simp [List.getD_eq_getElem?_getD, hj]
      refine ⟨hl', by simp [hJ.len], fun r hr => ?_, fun v hv => ?_, fun acc' hacc v hv => ?_⟩
      · have hlt := hJ.lt r hr
        exact ⟨hlt, hget _ hlt r ((hJ.slot _ hlt r).2 ⟨hr, by omega, rfl⟩)⟩
      · obtain ⟨j, hj, hs, hjv⟩ := hall v hv
        rw [hget v hv j hs]; exact ⟨hj, hjv⟩
      · cases hacc
        obtain ⟨j, _, hs, _⟩ := hall v hv; exact ⟨j, hs⟩

/-- **A permutation of `0 .. n` is always accepted.** -/
theorem invert_ok_of_perm {r2n : List Nat} {n : Nat} (hl : r2n.length = n) (hnd : r2n.Nodup)
    (hlt : ∀ x ∈ r2n, x < n) : ∃ n2r, invert r2n n = .ok n2r := by
  obtain ⟨acc, hacc⟩ := invertLoop_perm hnd hlt r2n 0 _ (J_init r2n n) (by simp)
  exact ⟨acc.map (fun o => o.getD 0), by simp [invert, hl, hacc]⟩

/-- `invert`'s result discharges `OrderOk` for a well-formed skeleton of `n` nodes. -/
theorem orderOk_of_invert {sk : NI.Cch.Csr} {r2n n2r : List Nat} (hwf : sk.isWellFormed = true)
    (h : invert r2n sk.n = .ok n2r) : OrderOk sk r2n n2r := by
  obtain ⟨h1, h2, h3, h4, _⟩ := invert_ok_spec h
  exact ⟨hwf, h1, h2, fun r hr => (h3 r hr).1, fun v hv => (h4 v hv).1, fun r hr => (h3 r hr).2,
    fun v hv => (h4 v hv).2⟩

end CchSG
