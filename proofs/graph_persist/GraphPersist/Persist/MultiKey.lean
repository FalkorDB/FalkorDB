import GraphPersist.Persist.RoundTrip
/-! # `build_multi_key_payloads` (encoder/mod.rs:95-189)

The greedy split of the four entity kinds over `key_count` RDB keys of at most `vkey_max`
entities each, modelled loop for loop: `fillKey` is the inner `while` (:135-153), one
call per key; `multiKey` is the outer `for key_idx` (:129-186). A kind still being filled
is `(state, total, offset)`; the list holds the kinds not yet finished, in encoding order.

* `fillKey_spec` — one key takes `min(cap, remaining)` entities, as consecutive slices.
* `multi_key_tiles` — over all keys, the entity slices, expanded to `(kind, id)` pairs, are
  exactly every id of every kind once, in order (what the decoder re-accumulates).
* `multi_key_bound` — no key holds more than `vkey_max` entities.
* `multi_key_all_placed` — `ceil(total / vkey_max)` keys hold everything.
* `multi_key_matrices` — key 0 carries every matrix payload, no other key carries any.
-/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

/-- A kind being distributed: `(state, total, offset)`. -/
abbrev Kind := St × Nat × Nat

def kindRem (k : Kind) : Nat := k.2.1 - k.2.2
def rem (ts : List Kind) : Nat := (ts.map kindRem).sum

/-- `(kind, id)` for each id still to place. -/
def seqOf (ts : List Kind) : List (St × Nat) :=
  ts.flatMap fun k => (List.range (k.2.1 - k.2.2)).map fun i => (k.1, k.2.2 + i)

/-- `(kind, id)` for each id an entry carries. -/
def expand (es : List Entry) : List (St × Nat) :=
  es.flatMap fun e => (List.range e.count).map fun i => (e.st, e.offset + i)

def KInv (ts : List Kind) : Prop := ∀ k ∈ ts, k.2.2 < k.2.1

/-- The inner `while remaining_capacity > 0 && type_idx < entity_types.len()` loop:
`take = min(remaining, available)`, push it if non-zero, advance the offset, move to the
next kind once this one is exhausted. -/
def fillKey (cap : Nat) : List Kind → List Entry × List Kind
  | [] => ([], [])
  | (st, tot, off) :: rest =>
    if h0 : cap = 0 then ([], (st, tot, off) :: rest)
    else if hd : off + min cap (tot - off) ≥ tot then
      let r := fillKey (cap - min cap (tot - off)) rest
      ((if min cap (tot - off) > 0 then [⟨st, min cap (tot - off), off⟩] else []) ++ r.1, r.2)
    else
      let r := fillKey (cap - min cap (tot - off)) ((st, tot, off + min cap (tot - off)) :: rest)
      ((if min cap (tot - off) > 0 then [⟨st, min cap (tot - off), off⟩] else []) ++ r.1, r.2)
termination_by ts => cap + ts.length
decreasing_by
  all_goals simp_wf
  all_goals omega

theorem expand_append (a b : List Entry) : expand (a ++ b) = expand a ++ expand b := by
  simp [expand]

theorem range_split_map (a b off : Nat) (st : St) :
    (List.range (a + b)).map (fun i => (st, off + i)) =
      (List.range a).map (fun i => (st, off + i)) ++ (List.range b).map (fun i => (st, off + a + i)) := by
  rw [List.range_add, List.map_append, List.map_map]
  congr 1
  apply List.map_congr_left
  intro x _
  simp only [Function.comp]
  congr 1
  omega

/-- **One key.** It takes `min(cap, remaining)` entities; what it takes, expanded, followed
by what is left, is exactly what was there — consecutive slices, nothing skipped or
repeated; the invariant survives. -/
theorem expand_cons (e : Entry) (es : List Entry) :
    expand (e :: es) = (List.range e.count).map (fun i => (e.st, e.offset + i)) ++ expand es := by
  simp [expand]

theorem seqOf_cons (k : Kind) (ks : List Kind) :
    seqOf (k :: ks) = (List.range (k.2.1 - k.2.2)).map (fun i => (k.1, k.2.2 + i)) ++ seqOf ks := by
  simp [seqOf]

theorem fillKey_spec : ∀ (n cap : Nat) (ts : List Kind), cap + ts.length = n → KInv ts →
    ((fillKey cap ts).1.map Entry.count).sum = min cap (rem ts) ∧
    rem (fillKey cap ts).2 = rem ts - min cap (rem ts) ∧
    expand (fillKey cap ts).1 ++ seqOf (fillKey cap ts).2 = seqOf ts ∧
    KInv (fillKey cap ts).2 ∧
    ((fillKey cap ts).2 = [] ∨ ((fillKey cap ts).1.map Entry.count).sum = cap) := by
  intro n
  induction n using Nat.strongRecOn with
  | _ n ih =>
  intro cap ts hn hI
  match ts with
  | [] => simp [fillKey, rem, seqOf, expand, KInv]
  | (st, tot, off) :: rest =>
    have hk : off < tot := hI (st, tot, off) (by simp)
    have hrest : KInv rest := fun k hk' => hI k (by simp [hk'])
    have hrem0 : rem ((st, tot, off) :: rest) = (tot - off) + rem rest := by simp [rem, kindRem]
    simp only [List.length_cons] at hn
    by_cases h0 : cap = 0
    · subst h0; simp [fillKey, rem, seqOf, expand]; exact hI
    have htake : 0 < min cap (tot - off) := by omega
    by_cases hd : off + min cap (tot - off) ≥ tot
    · -- this kind is exhausted
      obtain ⟨h1, h2, h3, h4, h5⟩ := ih _ (by omega) (cap - min cap (tot - off)) rest rfl hrest
      have hmin : min cap (tot - off) = tot - off := by omega
      rw [fillKey, dif_neg h0, dif_pos hd]
      simp only [htake, ite_true, List.cons_append, List.nil_append, List.map_cons, List.sum_cons]
      refine ⟨?_, ?_, ?_, h4, ?_⟩
      · rw [h1, hrem0]; omega
      · rw [h2, hrem0]; omega
      · rw [expand_cons, List.append_assoc, h3, seqOf_cons]
        simp only [hmin]
      · rcases h5 with h5 | h5
        · exact Or.inl h5
        · right; rw [h5]; omega
    · -- still owing: the key is full
      have hmin : min cap (tot - off) = cap := by omega
      have hI' : KInv ((st, tot, off + min cap (tot - off)) :: rest) := by
        intro k hk'; simp at hk'; rcases hk' with rfl | hk'
        · simp; omega
        · exact hrest k hk'
      obtain ⟨h1, h2, h3, h4, h5⟩ := ih _ (by simp only [List.length_cons]; omega)
        (cap - min cap (tot - off)) _ rfl hI'
      rw [fillKey, dif_neg h0, dif_neg hd]
      simp only [htake, ite_true, List.cons_append, List.nil_append, List.map_cons, List.sum_cons]
      have hc0 : cap - min cap (tot - off) = 0 := by omega
      simp only [hc0, Nat.zero_min, Nat.sub_zero] at h1 h2 h3 h4 h5 ⊢
      refine ⟨?_, ?_, ?_, h4, ?_⟩
      · rw [h1, hrem0]; omega
      · rw [h2]; simp only [rem, kindRem, List.map_cons, List.sum_cons] at hrem0 ⊢; omega
      · rw [expand_cons, List.append_assoc, h3, seqOf_cons, seqOf_cons, ← List.append_assoc]
        congr 1
        simp only
        have hs := range_split_map (min cap (tot - off)) (tot - off - min cap (tot - off)) off st
        rw [show min cap (tot - off) + (tot - off - min cap (tot - off)) = tot - off by omega] at hs
        rw [hs, show tot - (off + min cap (tot - off)) = tot - off - min cap (tot - off) by omega]
      · right; rw [h1]; omega

/-- The outer loop: `n` keys, each filled from what the previous ones left. -/
def keysFrom (cap : Nat) : Nat → List Kind → List (List Entry)
  | 0, _ => []
  | n + 1, ts => (fillKey cap ts).1 :: keysFrom cap n (fillKey cap ts).2

def leftAfter (cap : Nat) : Nat → List Kind → List Kind
  | 0, ts => ts
  | n + 1, ts => leftAfter cap n (fillKey cap ts).2

theorem keysFrom_spec (cap : Nat) : ∀ (n : Nat) (ts : List Kind), KInv ts →
    expand (keysFrom cap n ts).flatten ++ seqOf (leftAfter cap n ts) = seqOf ts ∧
    (∀ es ∈ keysFrom cap n ts, (es.map Entry.count).sum ≤ cap) ∧
    rem (leftAfter cap n ts) = rem ts - n * cap ∧ KInv (leftAfter cap n ts)
  | 0, ts, hI => by simp [keysFrom, leftAfter, expand, hI]
  | n + 1, ts, hI => by
    obtain ⟨h1, h2, h3, h4, h5⟩ := fillKey_spec _ cap ts rfl hI
    obtain ⟨g1, g2, g3, g4⟩ := keysFrom_spec cap n _ h4
    refine ⟨?_, ?_, ?_, g4⟩
    · simp only [keysFrom, leftAfter, List.flatten_cons, expand_append, List.append_assoc]
      rw [g1, h3]
    · intro es hes
      simp only [keysFrom, List.mem_cons] at hes
      rcases hes with rfl | hes
      · rw [h1]; exact Nat.min_le_left _ _
      · exact g2 es hes
    · simp only [leftAfter]; rw [g3, h2, Nat.succ_mul]; omega

/-- `entity_types`: the four kinds in encoding order, empty ones dropped (`:112-120`). -/
def kindsOf (g : G M Tn Nm Ix Cn) : List Kind :=
  ([(St.nodes, g.nodeCount), (St.delNodes, g.delNodes.length), (St.edges, g.edgeCount),
    (St.delEdges, g.delEdges.length)].filter fun p => p.2 > 0).map fun p => (p.1, p.2, 0)

def U64MAX : Nat := 2 ^ 64 - 1

/-- `key_count` (`:104-109`). -/
def keyCount (total vmax : Nat) : Nat := if total = 0 ∨ vmax = 0 then 1 else (total + vmax - 1) / vmax

/-- The matrix payloads key 0 appends (`:155-183`). -/
def matrixEntries (g : G M Tn Nm Ix Cn) : List Entry :=
  (if g.labelMats.length > 0 then [⟨.labelsM, g.labelMats.length, 0⟩] else []) ++
  (if g.tensors.length > 0 then [⟨.relM, g.tensors.length, 0⟩] else []) ++
  [⟨.adj, 1, 0⟩, ⟨.lblsM, 1, 0⟩]

def total (g : G M Tn Nm Ix Cn) : Nat := g.nodeCount + g.edgeCount + g.delNodes.length + g.delEdges.length

/-- `build_multi_key_payloads(graph, vkey_max)`. -/
def buildMulti (g : G M Tn Nm Ix Cn) (vmax : Nat) : List (List Entry) :=
  let cap := if vmax = 0 then U64MAX else vmax
  match keysFrom cap (keyCount (total g) vmax) (kindsOf g) with
  | [] => []
  | k0 :: ks => (k0 ++ matrixEntries g) :: ks

theorem kindsOf_inv (g : G M Tn Nm Ix Cn) : KInv (kindsOf g) := by
  intro k hk
  simp only [kindsOf, List.mem_map, List.mem_filter] at hk
  obtain ⟨p, ⟨_, hp⟩, rfl⟩ := hk
  simpa using hp

theorem rem_filter (l : List (St × Nat)) :
    rem ((l.filter fun p => p.2 > 0).map fun p => (p.1, p.2, 0)) = (l.map Prod.snd).sum := by
  induction l with
  | nil => rfl
  | cons p ps ih =>
    by_cases h : p.2 > 0
    · simp only [List.filter_cons, h, decide_true, ite_true, List.map_cons, List.sum_cons]
      simp only [rem, List.map_cons, List.sum_cons] at ih ⊢
      rw [ih]; simp [kindRem]
    · simp only [List.filter_cons, h, decide_false, Bool.false_eq_true, ite_false, List.map_cons, List.sum_cons]
      rw [ih]; omega

theorem rem_kindsOf (g : G M Tn Nm Ix Cn) : rem (kindsOf g) = total g := by
  rw [kindsOf, rem_filter]; simp [total]; omega

theorem seqOf_filter (l : List (St × Nat)) :
    seqOf ((l.filter fun p => p.2 > 0).map fun p => (p.1, p.2, 0)) =
      l.flatMap fun p => (List.range p.2).map fun i => (p.1, i) := by
  induction l with
  | nil => rfl
  | cons p ps ih =>
    by_cases h : p.2 > 0
    · simp only [List.filter_cons, h, decide_true, ite_true, List.map_cons, List.flatMap_cons]
      simp only [seqOf, List.flatMap_cons] at ih ⊢
      rw [ih]; simp
    · simp only [List.filter_cons, h, decide_false, Bool.false_eq_true, ite_false, List.flatMap_cons]
      rw [ih]
      have : p.2 = 0 := by omega
      simp [this]

/-- The ids of each kind, in order, as `(kind, id)`. -/
def allIds (g : G M Tn Nm Ix Cn) : List (St × Nat) :=
  (List.range g.nodeCount).map (fun i => (St.nodes, i)) ++
  (List.range g.delNodes.length).map (fun i => (St.delNodes, i)) ++
  (List.range g.edgeCount).map (fun i => (St.edges, i)) ++
  (List.range g.delEdges.length).map (fun i => (St.delEdges, i))

theorem seqOf_kindsOf (g : G M Tn Nm Ix Cn) : seqOf (kindsOf g) = allIds g := by
  rw [kindsOf, seqOf_filter]; simp [allIds]

/-- **Every key fits.** With `vkey_max > 0`, no key carries more than `vkey_max` entities. -/
theorem multi_key_bound (g : G M Tn Nm Ix Cn) (vmax : Nat) (hv : 0 < vmax) :
    ∀ es ∈ keysFrom vmax (keyCount (total g) vmax) (kindsOf g), (es.map Entry.count).sum ≤ vmax :=
  (keysFrom_spec vmax _ _ (kindsOf_inv g)).2.1

/-- **Everything is placed.** `ceil(total / vkey_max)` keys leave nothing over (and with
`vkey_max = 0` the single `u64::MAX`-capacity key takes all of a graph below `2^64`
entities). -/
theorem multi_key_all_placed (g : G M Tn Nm Ix Cn) (vmax : Nat) (ht : total g ≤ U64MAX) :
    leftAfter (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g) = [] := by
  have hs := (keysFrom_spec (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)
    (kindsOf_inv g))
  have hrem : rem (leftAfter (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)) = 0 := by
    rw [hs.2.2.1, rem_kindsOf]
    simp only [keyCount]
    by_cases h0 : total g = 0
    · simp [h0]
    by_cases hv : vmax = 0
    · simp [hv]; omega
    · simp only [h0, hv, false_or, ite_false]
      have := Nat.lt_div_mul_add (a := total g + vmax - 1) (b := vmax) (by omega)
      have := Nat.div_mul_le_self (total g + vmax - 1) vmax
      omega
  -- nothing left: every kind on the list still owes at least one id
  have hI := hs.2.2.2
  generalize leftAfter _ _ _ = L at hrem hI
  cases L with
  | nil => rfl
  | cons k ks =>
    have := hI k (by simp)
    simp [rem, kindRem] at hrem
    omega

/-- **Tiling.** Concatenating every key's entity slices, in key order, and expanding them
gives each kind's ids `0, 1, …, total-1` exactly once, kinds in encoding order — the
multi-key analogue of `build_payloads`, and what the decoder re-accumulates. -/
theorem multi_key_tiles (g : G M Tn Nm Ix Cn) (vmax : Nat) (ht : total g ≤ U64MAX) :
    expand (keysFrom (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)).flatten =
      allIds g := by
  have hs := (keysFrom_spec (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)
    (kindsOf_inv g)).1
  rw [multi_key_all_placed g vmax ht, ← seqOf_kindsOf] at *
  simpa [seqOf] using hs

theorem fillKey_kinds (P : St → Prop) : ∀ (n cap : Nat) (ts : List Kind), cap + ts.length = n →
    (∀ k ∈ ts, P k.1) → (∀ e ∈ (fillKey cap ts).1, P e.st) ∧ (∀ k ∈ (fillKey cap ts).2, P k.1) := by
  intro n
  induction n using Nat.strongRecOn with
  | _ n ih =>
  intro cap ts hn hts
  match ts with
  | [] => simp [fillKey]
  | (st, tot, off) :: rest =>
    have hst : P st := hts (st, tot, off) (by simp)
    have hrest : ∀ k ∈ rest, P k.1 := fun k hk => hts k (by simp [hk])
    by_cases h0 : cap = 0
    · subst h0; simp [fillKey]; exact ⟨hst, fun a b c h => hrest (a, b, c) h⟩
    by_cases hd : off + min cap (tot - off) ≥ tot
    · obtain ⟨i1, i2⟩ := ih _ (by simp at hn; omega) (cap - min cap (tot - off)) rest rfl hrest
      rw [fillKey, dif_neg h0, dif_pos hd]
      refine ⟨fun e he => ?_, i2⟩
      simp only [List.mem_append] at he
      rcases he with he | he
      · split at he <;> simp at he; subst he; exact hst
      · exact i1 e he
    · have hts' : ∀ k ∈ (st, tot, off + min cap (tot - off)) :: rest, P k.1 := by
        intro k hk; simp at hk; rcases hk with rfl | hk
        · exact hst
        · exact hrest k hk
      obtain ⟨i1, i2⟩ := ih _ (by simp only [List.length_cons] at hn ⊢; omega) (cap - min cap (tot - off)) _ rfl hts'
      rw [fillKey, dif_neg h0, dif_neg hd]
      refine ⟨fun e he => ?_, i2⟩
      simp only [List.mem_append] at he
      rcases he with he | he
      · split at he <;> simp at he; subst he; exact hst
      · exact i1 e he

/-- **Matrices on key 0 only.** -/
theorem multi_key_matrices (g : G M Tn Nm Ix Cn) (vmax : Nat) :
    ∀ k ∈ buildMulti g vmax, ∀ e ∈ k, (e.st = .labelsM ∨ e.st = .relM ∨ e.st = .adj ∨ e.st = .lblsM) →
      (buildMulti g vmax).head? = some k ∧ e ∈ matrixEntries g := by
  intro k hk e he hst
  have hent : ∀ (cap n : Nat) (ts : List Kind), (∀ k ∈ ts, k.1 = .nodes ∨ k.1 = .delNodes ∨ k.1 = .edges ∨ k.1 = .delEdges) →
      ∀ es ∈ keysFrom cap n ts, ∀ e ∈ es, e.st = .nodes ∨ e.st = .delNodes ∨ e.st = .edges ∨ e.st = .delEdges := by
    intro cap n
    induction n with
    | zero => intro ts _ es hes; simp [keysFrom] at hes
    | succ n ih =>
      intro ts hts es hes e he
      have hfill := fun (c : Nat) (ts : List Kind) hts =>
        fillKey_kinds (fun st => st = .nodes ∨ st = .delNodes ∨ st = .edges ∨ st = .delEdges) _ c ts rfl hts
      simp only [keysFrom, List.mem_cons] at hes
      rcases hes with rfl | hes
      · exact (hfill cap ts hts).1 e he
      · exact ih _ (hfill cap ts hts).2 es hes e he
  have hkinds : ∀ k ∈ kindsOf g, k.1 = .nodes ∨ k.1 = .delNodes ∨ k.1 = .edges ∨ k.1 = .delEdges := by
    intro k hk; simp only [kindsOf, List.mem_map, List.mem_filter] at hk
    obtain ⟨p, ⟨hp, _⟩, rfl⟩ := hk
    simp at hp; rcases hp with rfl | rfl | rfl | rfl <;> simp
  simp only [buildMulti] at hk ⊢
  split at hk
  · simp at hk
  · rename_i k0 ks heq
    simp only [List.mem_cons] at hk
    rcases hk with rfl | hk
    · simp only [List.mem_append] at he
      rcases he with he | he
      · have := hent _ _ _ hkinds k0 (by rw [heq]; simp) e he
        rcases hst with h | h | h | h <;> rw [h] at this <;> simp at this
      · exact ⟨rfl, he⟩
    · have := hent _ _ _ hkinds k (by rw [heq]; simp [hk]) e he
      rcases hst with h | h | h | h <;> rw [h] at this <;> simp at this

end GraphPersist.Persist
