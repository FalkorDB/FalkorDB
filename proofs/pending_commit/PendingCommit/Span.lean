/-
# `Block::merge_span` / `set_span` at the logical level

`graph/src/graph/attribute_store.rs`. An entity's attributes are a *span*: a
list of `(attr_id, value)` strictly ascending by id, with no `Null` stored
(FalkorDB never stores a null — a null in an update means "remove").

`merge_span` (attribute_store.rs:636) has three code paths:

* fast path 1 (`:648-667`): every pair non-null and already present — patch
  in place, return `(pairs.len(), pairs.len())`;
* fast path 2 (`:671-696`): every pair null — shift the survivors left;
* the general two-pass merge (`:699-792`).

Each is modelled with the same loop structure as the Rust and proved to
compute the *same* span and the *same* `(nremoved, nset)` as one reference
(`refGet` / `refCounts`). The first pass of the general path (`newLen`) is
proved to equal the length of what the second pass emits, which is the
`debug_assert!(new_len == 0 || w - dst == new_len)` at `:765`.
-/
namespace PendingCommit

/-- Property values, as far as the store is concerned: the store only
distinguishes `Null` (a removal marker, never stored) from everything else.
`inl` stands for the inline scalar tags, `heap` for `Tag::Heap` values. -/
inductive Val where
  | null
  | inl (i : Int)
  | heap (s : String)
  deriving DecidableEq, Repr

def Val.isNull : Val → Bool
  | .null => true
  | _ => false

abbrev Entry := Nat × Val

/-- First-match lookup (`binary_search_by_key` on a strictly sorted span). -/
def get : List Entry → Nat → Option Val
  | [], _ => none
  | e :: es, k => if k = e.1 then some e.2 else get es k

def mem (l : List Entry) (k : Nat) : Bool := (get l k).isSome

/-- Strictly ascending by attribute id: `debug_assert!(attrs.windows(2).all(|w| w[0].0 < w[1].0))`
(pending.rs:431, `insert_attrs_rows` :1231). -/
def Sorted (l : List Entry) : Prop := List.Pairwise (fun a b => a.1 < b.1) l

/-- No stored nulls. -/
def NoNull (l : List Entry) : Prop := ∀ e ∈ l, e.2.isNull = false

/-! ## Reference semantics -/

/-- What an attribute reads as after applying `pairs` to `span`: the update
wins where it speaks (`Null` = absent), otherwise the old value. -/
def refGet (span pairs : List Entry) (k : Nat) : Option Val :=
  match get pairs k with
  | some .null => none
  | some v => some v
  | none => get span k

/-- Reference counts, with `AttributeStore::insert_attrs` semantics
(attribute_store.rs:1306-1313): `nremoved` = pairs whose id the span already
held (replaced *or* removed), `nset` = non-null pairs. -/
def refRemoved (span pairs : List Entry) : Nat :=
  (pairs.filter (fun p => mem span p.1)).length

def refSet (pairs : List Entry) : Nat :=
  (pairs.filter (fun p => !p.2.isNull)).length

/-! ## General path: `merge_span` second pass (attribute_store.rs:735-764) -/

/-- The emit loop. `c` is `scratch` (the old span), `p` the pairs. Returns
(emitted entries, nremoved, nset). Same branch order as the Rust:
`take_old` first, then "equal id ⇒ count a removal and skip the old entry",
then "non-null ⇒ emit". -/
def mergeGen : List Entry → List Entry → List Entry × Nat × Nat
  | [], [] => ([], 0, 0)
  | c :: cs, [] =>
      let r := mergeGen cs []
      (c :: r.1, r.2)
  | [], p :: ps =>
      let r := mergeGen [] ps
      if p.2.isNull then r else (p :: r.1, r.2.1, r.2.2 + 1)
  | c :: cs, p :: ps =>
      if c.1 < p.1 then
        let r := mergeGen cs (p :: ps)
        (c :: r.1, r.2)
      else if c.1 = p.1 then
        let r := mergeGen cs ps
        if p.2.isNull then (r.1, r.2.1 + 1, r.2.2)
        else (p :: r.1, r.2.1 + 1, r.2.2 + 1)
      else
        let r := mergeGen (c :: cs) ps
        if p.2.isNull then r else (p :: r.1, r.2.1, r.2.2 + 1)
termination_by c p => c.length + p.length

/-- The first pass (attribute_store.rs:706-729): merged length only. -/
def newLen : List Entry → List Entry → Nat
  | [], [] => 0
  | _ :: cs, [] => newLen cs [] + 1
  | [], p :: ps => newLen [] ps + (if p.2.isNull then 0 else 1)
  | c :: cs, p :: ps =>
      if c.1 < p.1 then newLen cs (p :: ps) + 1
      else if c.1 = p.1 then newLen cs ps + (if p.2.isNull then 0 else 1)
      else newLen (c :: cs) ps + (if p.2.isNull then 0 else 1)
termination_by c p => c.length + p.length

/-! ## Fast paths -/

/-- Fast-path-1 guard (`:653-656`). -/
def allPresentNonNull (span pairs : List Entry) : Bool :=
  pairs.all (fun p => !p.2.isNull && mem span p.1)

/-- Fast path 1 body: each pair overwrites its entry in place (`:657-665`). -/
def patch (span pairs : List Entry) : List Entry :=
  span.map (fun e => (e.1, match get pairs e.1 with
    | some v => v
    | none => e.2))

/-- Fast-path-2 guard (`:671`). -/
def allNull (pairs : List Entry) : Bool := pairs.all (fun p => p.2.isNull)

/-- Drop pairs with id below `k` — the `while ni < pairs.len() && pairs.id(ni) < e.id` loop. -/
def skipBelow (k : Nat) : List Entry → List Entry
  | [] => []
  | p :: ps => if p.1 < k then skipBelow k ps else p :: ps

/-- Fast path 2 body (`:672-685`): survivors and the removal count. -/
def pureRemove : List Entry → List Entry → List Entry × Nat
  | [], _ => ([], 0)
  | e :: es, ps =>
      let ps' := skipBelow e.1 ps
      match ps' with
      | p :: _ =>
          if p.1 = e.1 then
            let r := pureRemove es ps'
            (r.1, r.2 + 1)
          else
            let r := pureRemove es ps'
            (e :: r.1, r.2)
      | [] =>
          let r := pureRemove es ps'
          (e :: r.1, r.2)

/-- `Block::merge_span`, logical content: which path runs and what it
returns. `old = none` is `old.cap == 0`. -/
def mergeSpan (old : List Entry) (pairs : List Entry) : List Entry × Nat × Nat :=
  if old ≠ [] ∧ allPresentNonNull old pairs = true then
    (patch old pairs, pairs.length, pairs.length)
  else if old ≠ [] ∧ allNull pairs = true then
    let r := pureRemove old pairs
    (r.1, r.2, 0)
  else
    mergeGen old pairs

/-! ## Lemmas -/

theorem get_none_of_all_gt {l : List Entry} {k : Nat} (h : ∀ a ∈ l, k < a.1) :
    get l k = none := by
  induction l with
  | nil => rfl
  | cons e es ih =>
    have he := h e (by simp)
    simp only [get]
    rw [if_neg (by omega)]
    exact ih (fun a ha => h a (by simp [ha]))

theorem get_none_of_lt_head {p : Entry} {ps : List Entry} (h : Sorted (p :: ps))
    {k : Nat} (hk : k < p.1) : get (p :: ps) k = none := by
  apply get_none_of_all_gt
  intro a ha
  simp at ha
  rcases ha with rfl | ha
  · exact hk
  · have := (List.pairwise_cons.1 h).1 a ha; omega

theorem sorted_tail {e : Entry} {l : List Entry} (h : Sorted (e :: l)) : Sorted l :=
  (List.pairwise_cons.1 h).2

theorem sorted_head_lt {e : Entry} {l : List Entry} (h : Sorted (e :: l)) :
    ∀ a ∈ l, e.1 < a.1 := (List.pairwise_cons.1 h).1

theorem sorted_cons_cons {a b : Entry} {l : List Entry} (h : Sorted (a :: b :: l)) :
    Sorted (a :: l) := by
  unfold Sorted at *
  refine List.pairwise_cons.2 ⟨fun x hx => (List.pairwise_cons.1 h).1 x (by simp [hx]), ?_⟩
  exact (List.pairwise_cons.1 (List.pairwise_cons.1 h).2).2

theorem get_cons_ne {e : Entry} {l : List Entry} {k : Nat} (h : k ≠ e.1) :
    get (e :: l) k = get l k := by simp [get, h]

theorem get_cons_eq {e : Entry} {l : List Entry} : get (e :: l) e.1 = some e.2 := by
  simp [get]

/-- The first pass counts exactly what the second pass emits
(`debug_assert!(new_len == 0 || w - dst == new_len)`, attribute_store.rs:765). -/
theorem newLen_eq : ∀ (c p : List Entry), newLen c p = (mergeGen c p).1.length
  | [], [] => by simp [newLen, mergeGen]
  | c :: cs, [] => by
      simp only [newLen, mergeGen, List.length_cons]; rw [newLen_eq cs []]
  | [], p :: ps => by
      simp only [newLen, mergeGen]
      rw [newLen_eq [] ps]; cases p.2.isNull <;> simp
  | c :: cs, p :: ps => by
      simp only [newLen, mergeGen]
      by_cases h1 : c.1 < p.1
      · simp only [h1, ite_true, List.length_cons]; rw [newLen_eq cs (p :: ps)]
      · simp only [h1, ite_false]
        by_cases h2 : c.1 = p.1
        · simp only [h2, ite_true]; rw [newLen_eq cs ps]; cases p.2.isNull <;> simp
        · simp only [h2, ite_false]; rw [newLen_eq (c :: cs) ps]; cases p.2.isNull <;> simp
termination_by c p => c.length + p.length

/-- The general merge reads back as the reference, for every attribute id. -/
theorem mergeGen_get : ∀ (c p : List Entry), Sorted c → Sorted p →
    ∀ k, get (mergeGen c p).1 k = refGet c p k
  | [], [], _, _, k => by simp [mergeGen, refGet, get]
  | c :: cs, [], hc, _, k => by
      have ih := mergeGen_get cs [] (sorted_tail hc) (by simp [Sorted]) k
      simp only [mergeGen, refGet, get] at ih ⊢
      by_cases hk : k = c.1
      · simp [hk]
      · simp [hk]; simpa using ih
  | [], p :: ps, _, hp, k => by
      have ih := mergeGen_get [] ps (by simp [Sorted]) (sorted_tail hp) k
      simp only [mergeGen]
      unfold refGet at ih ⊢
      by_cases hk : k = p.1
      · subst hk
        have hn : get ps p.1 = none := by
          cases ps with
          | nil => rfl
          | cons q qs =>
            exact get_none_of_lt_head (sorted_tail hp) (sorted_head_lt hp q (by simp))
        rw [hn] at ih
        cases hv : p.2 with
        | null => simp [Val.isNull, hv, get_cons_eq]; simpa [get] using ih
        | inl i => simp [Val.isNull, hv, get]
        | heap s => simp [Val.isNull, hv, get]
      · rw [get_cons_ne hk]
        cases hv : p.2.isNull
        · simp only [Bool.false_eq_true, ite_false]; rw [get_cons_ne hk]; exact ih
        · simp only [ite_true]; exact ih
  | c :: cs, p :: ps, hc, hp, k => by
      simp only [mergeGen]
      by_cases h1 : c.1 < p.1
      · simp only [h1, ite_true]
        have ih := mergeGen_get cs (p :: ps) (sorted_tail hc) hp k
        unfold refGet at ih ⊢
        by_cases hk : k = c.1
        · subst hk
          rw [get_none_of_lt_head hp h1]; simp [get]
        · rw [get_cons_ne hk, get_cons_ne hk]; exact ih
      · simp only [h1, ite_false]
        by_cases h2 : c.1 = p.1
        · simp only [h2, ite_true]
          have ih := mergeGen_get cs ps (sorted_tail hc) (sorted_tail hp) k
          unfold refGet at ih ⊢
          by_cases hk : k = p.1
          · subst hk
            have hn1 : get cs p.1 = none := by
              cases cs with
              | nil => rfl
              | cons q qs =>
                exact get_none_of_lt_head (sorted_tail hc)
                  (by have := sorted_head_lt hc q (by simp); omega)
            have hn2 : get ps p.1 = none := by
              cases ps with
              | nil => rfl
              | cons q qs =>
                exact get_none_of_lt_head (sorted_tail hp) (sorted_head_lt hp q (by simp))
            rw [hn1, hn2] at ih
            cases hv : p.2 with
            | null => simp [Val.isNull, hv, get_cons_eq]; simpa using ih
            | inl i => simp [Val.isNull, hv, get]
            | heap s => simp [Val.isNull, hv, get]
          · have hkc : k ≠ c.1 := by omega
            rw [get_cons_ne hk, get_cons_ne hkc]
            cases hv : p.2.isNull
            · simp only [Bool.false_eq_true, ite_false]; rw [get_cons_ne hk]; exact ih
            · simp only [ite_true]; exact ih
        · simp only [h2, ite_false]
          have hlt : p.1 < c.1 := by omega
          have ih := mergeGen_get (c :: cs) ps hc (sorted_tail hp) k
          unfold refGet at ih ⊢
          by_cases hk : k = p.1
          · subst hk
            have hn2 : get ps p.1 = none := by
              cases ps with
              | nil => rfl
              | cons q qs =>
                exact get_none_of_lt_head (sorted_tail hp) (sorted_head_lt hp q (by simp))
            have hn1 : get (c :: cs) p.1 = none := get_none_of_lt_head hc hlt
            rw [hn2, hn1] at ih
            cases hv : p.2 with
            | null => simp [Val.isNull, hv, get_cons_eq]; simpa using ih
            | inl i => simp [Val.isNull, hv, get]
            | heap s => simp [Val.isNull, hv, get]
          · rw [get_cons_ne hk]
            cases hv : p.2.isNull
            · simp only [Bool.false_eq_true, ite_false]; rw [get_cons_ne hk]; exact ih
            · simp only [ite_true]; exact ih
termination_by c p => c.length + p.length


/-! ## Counts -/

theorem mem_cons_ne {e : Entry} {l : List Entry} {k : Nat} (h : k ≠ e.1) :
    mem (e :: l) k = mem l k := by simp [mem, get_cons_ne h]

theorem refRemoved_nil_left (p : List Entry) : refRemoved [] p = 0 := by
  simp [refRemoved, mem, get]

theorem refRemoved_cons_drop {c : Entry} {cs p : List Entry} (h : ∀ q ∈ p, q.1 ≠ c.1) :
    refRemoved (c :: cs) p = refRemoved cs p := by
  unfold refRemoved
  congr 1
  apply List.filter_congr
  intro q hq
  exact mem_cons_ne (h q hq)

theorem refRemoved_cons_right (c p : List Entry) (q : Entry) :
    refRemoved c (q :: p) = (if mem c q.1 then 1 else 0) + refRemoved c p := by
  unfold refRemoved
  by_cases h : mem c q.1 <;> simp [List.filter_cons, h] <;> omega

theorem refSet_cons (q : Entry) (p : List Entry) :
    refSet (q :: p) = (if q.2.isNull then 0 else 1) + refSet p := by
  unfold refSet
  cases h : q.2.isNull <;> simp [List.filter_cons, h] <;> omega

theorem mem_none_of_lt_head {c : Entry} {cs : List Entry} (h : Sorted (c :: cs)) {k : Nat}
    (hk : k < c.1) : mem (c :: cs) k = false := by
  simp [mem, get_none_of_lt_head h hk]

theorem mem_self {c : Entry} {cs : List Entry} : mem (c :: cs) c.1 = true := by
  simp [mem, get]

theorem mergeGen_removed : ∀ (c p : List Entry), Sorted c → Sorted p →
    (mergeGen c p).2.1 = refRemoved c p
  | [], [], _, _ => by simp [mergeGen, refRemoved]
  | c :: cs, [], hc, _ => by
      have ih := mergeGen_removed cs [] (sorted_tail hc) (by simp [Sorted])
      simp only [mergeGen]; rw [ih]; simp [refRemoved]
  | [], p :: ps, _, hp => by
      have ih := mergeGen_removed [] ps (by simp [Sorted]) (sorted_tail hp)
      simp only [mergeGen]; rw [refRemoved_nil_left]
      rw [refRemoved_nil_left] at ih
      cases p.2.isNull <;> simp [ih]
  | c :: cs, p :: ps, hc, hp => by
      simp only [mergeGen]
      by_cases h1 : c.1 < p.1
      · simp only [h1, ite_true]
        rw [mergeGen_removed cs (p :: ps) (sorted_tail hc) hp]
        rw [refRemoved_cons_drop]
        intro q hq; simp at hq
        rcases hq with rfl | hq
        · omega
        · have := sorted_head_lt hp q hq; omega
      · simp only [h1, ite_false]
        by_cases h2 : c.1 = p.1
        · simp only [h2, ite_true]
          have ih := mergeGen_removed cs ps (sorted_tail hc) (sorted_tail hp)
          rw [refRemoved_cons_right, refRemoved_cons_drop (cs := cs)]
          · rw [← h2, mem_self]
            cases p.2.isNull <;> simp [ih] <;> omega
          · intro q hq; have := sorted_head_lt hp q hq; omega
        · simp only [h2, ite_false]
          have ih := mergeGen_removed (c :: cs) ps hc (sorted_tail hp)
          rw [refRemoved_cons_right, mem_none_of_lt_head hc (by omega)]
          cases p.2.isNull <;> simp [ih]
termination_by c p => c.length + p.length

theorem mergeGen_set : ∀ (c p : List Entry), (mergeGen c p).2.2 = refSet p
  | [], [] => by simp [mergeGen, refSet]
  | c :: cs, [] => by
      simp only [mergeGen]; rw [mergeGen_set cs []]
  | [], p :: ps => by
      simp only [mergeGen]; rw [refSet_cons]
      have ih := mergeGen_set [] ps
      cases p.2.isNull <;> simp [ih] <;> omega
  | c :: cs, p :: ps => by
      simp only [mergeGen]
      by_cases h1 : c.1 < p.1
      · simp only [h1, ite_true]; rw [mergeGen_set cs (p :: ps)]
      · simp only [h1, ite_false]
        rw [refSet_cons]
        by_cases h2 : c.1 = p.1
        · simp only [h2, ite_true]
          have ih := mergeGen_set cs ps
          cases p.2.isNull <;> simp [ih] <;> omega
        · simp only [h2, ite_false]
          have ih := mergeGen_set (c :: cs) ps
          cases p.2.isNull <;> simp [ih] <;> omega
termination_by c p => c.length + p.length

/-! ## Output shape: strictly sorted, no stored null -/

theorem mergeGen_keys_gt : ∀ (c p : List Entry) (b : Nat),
    (∀ a ∈ c, b < a.1) → (∀ a ∈ p, b < a.1) → ∀ e ∈ (mergeGen c p).1, b < e.1
  | [], [], _, _, _ => by simp [mergeGen]
  | c :: cs, [], b, hc, hp => by
      simp only [mergeGen]; intro e he; simp at he
      rcases he with rfl | he
      · exact hc _ (by simp)
      · exact mergeGen_keys_gt cs [] b (fun a ha => hc a (by simp [ha])) hp e he
  | [], p :: ps, b, hc, hp => by
      simp only [mergeGen]
      have ih := mergeGen_keys_gt [] ps b hc (fun a ha => hp a (by simp [ha]))
      cases p.2.isNull
      · simp only [Bool.false_eq_true, ite_false]; intro e he; simp at he
        rcases he with rfl | he
        · exact hp _ (by simp)
        · exact ih e he
      · simpa using ih
  | c :: cs, p :: ps, b, hc, hp => by
      simp only [mergeGen]
      have hc' : ∀ a ∈ cs, b < a.1 := fun a ha => hc a (by simp [ha])
      have hp' : ∀ a ∈ ps, b < a.1 := fun a ha => hp a (by simp [ha])
      by_cases h1 : c.1 < p.1
      · simp only [h1, ite_true]; intro e he; simp at he
        rcases he with rfl | he
        · exact hc _ (by simp)
        · exact mergeGen_keys_gt cs (p :: ps) b hc' hp e he
      · simp only [h1, ite_false]
        by_cases h2 : c.1 = p.1
        · simp only [h2, ite_true]
          have ih := mergeGen_keys_gt cs ps b hc' hp'
          cases p.2.isNull
          · simp only [Bool.false_eq_true, ite_false]; intro e he; simp at he
            rcases he with rfl | he
            · exact hp _ (by simp)
            · exact ih e he
          · simpa using ih
        · simp only [h2, ite_false]
          have ih := mergeGen_keys_gt (c :: cs) ps b hc hp'
          cases p.2.isNull
          · simp only [Bool.false_eq_true, ite_false]; intro e he; simp at he
            rcases he with rfl | he
            · exact hp _ (by simp)
            · exact ih e he
          · simpa using ih
termination_by c p => c.length + p.length

theorem mergeGen_sorted : ∀ (c p : List Entry), Sorted c → Sorted p → Sorted (mergeGen c p).1
  | [], [], _, _ => by simp [mergeGen, Sorted]
  | c :: cs, [], hc, _ => by
      simp only [mergeGen, Sorted]
      refine List.pairwise_cons.2 ⟨?_, mergeGen_sorted cs [] (sorted_tail hc) (by simp [Sorted])⟩
      exact mergeGen_keys_gt cs [] c.1 (sorted_head_lt hc) (by simp)
  | [], p :: ps, _, hp => by
      simp only [mergeGen]
      have ih := mergeGen_sorted [] ps (by simp [Sorted]) (sorted_tail hp)
      cases p.2.isNull
      · simp only [Bool.false_eq_true, ite_false, Sorted]
        exact List.pairwise_cons.2 ⟨mergeGen_keys_gt [] ps p.1 (by simp) (sorted_head_lt hp), ih⟩
      · simpa using ih
  | c :: cs, p :: ps, hc, hp => by
      simp only [mergeGen]
      by_cases h1 : c.1 < p.1
      · simp only [h1, ite_true, Sorted]
        refine List.pairwise_cons.2 ⟨?_, mergeGen_sorted cs (p :: ps) (sorted_tail hc) hp⟩
        apply mergeGen_keys_gt cs (p :: ps) c.1 (sorted_head_lt hc)
        intro a ha; simp at ha
        rcases ha with rfl | ha
        · exact h1
        · have := sorted_head_lt hp a ha; omega
      · simp only [h1, ite_false]
        by_cases h2 : c.1 = p.1
        · simp only [h2, ite_true]
          have ih := mergeGen_sorted cs ps (sorted_tail hc) (sorted_tail hp)
          cases p.2.isNull
          · simp only [Bool.false_eq_true, ite_false, Sorted]
            refine List.pairwise_cons.2 ⟨?_, ih⟩
            apply mergeGen_keys_gt cs ps p.1 _ (sorted_head_lt hp)
            intro a ha; have := sorted_head_lt hc a ha; omega
          · simpa using ih
        · simp only [h2, ite_false]
          have ih := mergeGen_sorted (c :: cs) ps hc (sorted_tail hp)
          cases p.2.isNull
          · simp only [Bool.false_eq_true, ite_false, Sorted]
            refine List.pairwise_cons.2 ⟨?_, ih⟩
            apply mergeGen_keys_gt (c :: cs) ps p.1 _ (sorted_head_lt hp)
            intro a ha; simp at ha
            rcases ha with rfl | ha
            · omega
            · have := sorted_head_lt hc a ha; omega
          · simpa using ih
termination_by c p => c.length + p.length

theorem mergeGen_noNull : ∀ (c p : List Entry), NoNull c → NoNull (mergeGen c p).1
  | [], [], _ => by simp [mergeGen, NoNull]
  | c :: cs, [], hc => by
      simp only [mergeGen, NoNull]; intro e he; simp at he
      rcases he with rfl | he
      · exact hc _ (by simp)
      · exact mergeGen_noNull cs [] (fun a ha => hc a (by simp [ha])) e he
  | [], p :: ps, hc => by
      simp only [mergeGen]
      have ih := mergeGen_noNull [] ps hc
      cases hn : p.2.isNull
      · simp only [Bool.false_eq_true, ite_false, NoNull]; intro e he; simp at he
        rcases he with rfl | he
        · exact hn
        · exact ih e he
      · simpa using ih
  | c :: cs, p :: ps, hc => by
      simp only [mergeGen]
      have hc' : NoNull cs := fun a ha => hc a (by simp [ha])
      by_cases h1 : c.1 < p.1
      · simp only [h1, ite_true, NoNull]; intro e he; simp at he
        rcases he with rfl | he
        · exact hc _ (by simp)
        · exact mergeGen_noNull cs (p :: ps) hc' e he
      · simp only [h1, ite_false]
        by_cases h2 : c.1 = p.1
        · simp only [h2, ite_true]
          have ih := mergeGen_noNull cs ps hc'
          cases hn : p.2.isNull
          · simp only [Bool.false_eq_true, ite_false, NoNull]; intro e he; simp at he
            rcases he with rfl | he
            · exact hn
            · exact ih e he
          · simpa using ih
        · simp only [h2, ite_false]
          have ih := mergeGen_noNull (c :: cs) ps hc
          cases hn : p.2.isNull
          · simp only [Bool.false_eq_true, ite_false, NoNull]; intro e he; simp at he
            rcases he with rfl | he
            · exact hn
            · exact ih e he
          · simpa using ih
termination_by c p => c.length + p.length


/-! ## Fast path 1 -/

theorem get_patch (old pairs : List Entry) (k : Nat) :
    get (patch old pairs) k =
      (get old k).map (fun v0 => match get pairs k with | some v => v | none => v0) := by
  induction old with
  | nil => rfl
  | cons e es ih =>
    simp only [patch, List.map_cons, get] at ih ⊢
    by_cases hk : k = e.1
    · subst hk; simp
    · simp only [hk, ite_false]; exact ih

theorem patch_keys (old pairs : List Entry) : (patch old pairs).map (·.1) = old.map (·.1) := by
  simp [patch]

theorem get_some_mem {l : List Entry} {k : Nat} {v : Val} (h : get l k = some v) :
    ∃ e ∈ l, e.1 = k ∧ e.2 = v := by
  induction l with
  | nil => simp [get] at h
  | cons e es ih =>
    simp only [get] at h
    by_cases hk : k = e.1
    · simp [hk] at h; exact ⟨e, by simp, hk.symm, h⟩
    · simp [hk] at h; obtain ⟨x, hx, h1, h2⟩ := ih h; exact ⟨x, by simp [hx], h1, h2⟩

theorem patch_get (old pairs : List Entry) (h : allPresentNonNull old pairs = true) (k : Nat) :
    get (patch old pairs) k = refGet old pairs k := by
  rw [get_patch]; unfold refGet
  unfold allPresentNonNull at h
  rw [List.all_eq_true] at h
  cases hp : get pairs k with
  | none => cases get old k <;> rfl
  | some v =>
    obtain ⟨e, he, rfl, rfl⟩ := get_some_mem hp
    have := h e he
    simp [Bool.and_eq_true] at this
    obtain ⟨hn, hm⟩ := this
    unfold mem at hm
    obtain ⟨v0, hv0⟩ := Option.isSome_iff_exists.1 hm
    rw [hv0]
    cases hv : e.2 with
    | null => simp [hv, Val.isNull] at hn
    | inl i => rfl
    | heap s => rfl

theorem sorted_of_keys_eq {a b : List Entry} (h : a.map (·.1) = b.map (·.1)) (hb : Sorted b) :
    Sorted a := by
  induction a generalizing b with
  | nil => simp [Sorted]
  | cons x xs ih =>
    cases b with
    | nil => simp at h
    | cons y ys =>
      simp at h
      obtain ⟨h1, h2⟩ := h
      unfold Sorted
      refine List.pairwise_cons.2 ⟨?_, ih h2 (sorted_tail hb)⟩
      intro a ha
      have : a.1 ∈ xs.map (·.1) := List.mem_map.2 ⟨a, ha, rfl⟩
      rw [h2] at this
      obtain ⟨c, hc, hce⟩ := List.mem_map.1 this
      have := sorted_head_lt hb c hc
      omega

theorem patch_noNull (old pairs : List Entry) (h : allPresentNonNull old pairs = true)
    (hn : NoNull old) : NoNull (patch old pairs) := by
  unfold allPresentNonNull at h; rw [List.all_eq_true] at h
  have hnn : ∀ q ∈ pairs, q.2.isNull = false := by
    intro q hq; have := h q hq; simp [Bool.and_eq_true] at this; simpa using this.1
  clear h
  intro x hx
  simp only [patch, List.mem_map] at hx
  obtain ⟨e, he, rfl⟩ := hx
  cases hp : get pairs e.1 with
  | none => simpa [hp] using hn e he
  | some v =>
    obtain ⟨q, hq, _, rfl⟩ := get_some_mem hp
    simpa [hp] using hnn q hq

theorem patch_counts (old pairs : List Entry) (h : allPresentNonNull old pairs = true) :
    refRemoved old pairs = pairs.length ∧ refSet pairs = pairs.length := by
  unfold allPresentNonNull at h; rw [List.all_eq_true] at h
  constructor
  · unfold refRemoved; rw [List.filter_eq_self.2]
    intro a ha; have := h a ha; simp [Bool.and_eq_true] at this; exact this.2
  · unfold refSet; rw [List.filter_eq_self.2]
    intro a ha; have := h a ha; simp [Bool.and_eq_true] at this; simp [this.1]

/-! ## Fast path 2 -/

theorem get_skipBelow (b : Nat) (ps : List Entry) (j : Nat) (hj : b ≤ j) :
    get (skipBelow b ps) j = get ps j := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    simp only [skipBelow]
    by_cases h : p.1 < b
    · simp only [h, ite_true]; rw [ih, get_cons_ne (by omega)]
    · simp [h]

theorem skipBelow_ge (b : Nat) (ps : List Entry) : ∀ q ∈ (skipBelow b ps).head?, b ≤ q.1 := by
  induction ps with
  | nil => simp [skipBelow]
  | cons p ps ih =>
    simp only [skipBelow]
    by_cases h : p.1 < b
    · simp only [h, ite_true]; exact ih
    · simp [h]; omega

theorem skipBelow_sorted (b : Nat) (ps : List Entry) (h : Sorted ps) : Sorted (skipBelow b ps) := by
  induction ps with
  | nil => simp [skipBelow, Sorted]
  | cons p ps ih =>
    simp only [skipBelow]
    by_cases hp : p.1 < b
    · simp only [hp, ite_true]; exact ih (sorted_tail h)
    · simp [hp]; exact h

theorem mem_iff (l : List Entry) (k : Nat) : mem l k = true ↔ ∃ e ∈ l, e.1 = k := by
  induction l with
  | nil => simp [mem, get]
  | cons e es ih =>
    by_cases hk : k = e.1
    · subst hk; simp [mem, get]
    · rw [mem_cons_ne hk, ih]
      constructor
      · rintro ⟨x, hx, rfl⟩; exact ⟨x, by simp [hx], rfl⟩
      · rintro ⟨x, hx, rfl⟩
        simp at hx; rcases hx with rfl | hx
        · exact absurd rfl hk
        · exact ⟨x, hx, rfl⟩

theorem mem_skipBelow_head_ne {b : Nat} {ps : List Entry} (hs : Sorted ps) {p : Entry}
    {rest : List Entry} (hh : skipBelow b ps = p :: rest) (hne : p.1 ≠ b) : mem ps b = false := by
  have hge := skipBelow_ge b ps p (by simp [hh])
  have hsk := get_skipBelow b ps b (Nat.le_refl b)
  rw [hh] at hsk
  have hsorted : Sorted (p :: rest) := hh ▸ skipBelow_sorted b ps hs
  rw [get_none_of_lt_head hsorted (by omega)] at hsk
  simp [mem, ← hsk]

theorem mem_skipBelow_nil {b : Nat} {ps : List Entry} (hh : skipBelow b ps = []) :
    mem ps b = false := by
  have hsk := get_skipBelow b ps b (Nat.le_refl b)
  rw [hh] at hsk
  simp [mem, ← hsk, get]

theorem pureRemove_keys_gt : ∀ (old ps : List Entry) (b : Nat), (∀ a ∈ old, b < a.1) →
    ∀ e ∈ (pureRemove old ps).1, b < e.1
  | [], _, _, _ => by simp [pureRemove]
  | e :: es, ps, b, h => by
      have h' : ∀ a ∈ es, b < a.1 := fun a ha => h a (by simp [ha])
      simp only [pureRemove]
      split
      · split
        · exact pureRemove_keys_gt es _ b h'
        · intro x hx; simp at hx; rcases hx with rfl | hx
          · exact h _ (by simp)
          · exact pureRemove_keys_gt es _ b h' x hx
      · intro x hx; simp at hx; rcases hx with rfl | hx
        · exact h _ (by simp)
        · exact pureRemove_keys_gt es _ b h' x hx

theorem pureRemove_sublist : ∀ (old ps : List Entry), List.Sublist (pureRemove old ps).1 old
  | [], _ => by simp [pureRemove]
  | e :: es, ps => by
      simp only [pureRemove]
      split
      · split
        · exact (pureRemove_sublist es _).cons e
        · exact (pureRemove_sublist es _).cons₂ e
      · exact (pureRemove_sublist es _).cons₂ e

theorem pureRemove_get : ∀ (old ps : List Entry), Sorted old → Sorted ps → ∀ k,
    get (pureRemove old ps).1 k = if mem ps k then none else get old k
  | [], ps, _, _, k => by simp [pureRemove, get]
  | e :: es, ps, ho, hs, k => by
      have hes := sorted_tail ho
      have hs' := skipBelow_sorted e.1 ps hs
      have ih := pureRemove_get es (skipBelow e.1 ps) hes hs'
      have hkeys := pureRemove_keys_gt es (skipBelow e.1 ps) e.1 (sorted_head_lt ho)
      -- reading `k` below or at `e.1` never reaches the recursive part
      have hlow : ∀ j, j ≤ e.1 → get (pureRemove es (skipBelow e.1 ps)).1 j = none := by
        intro j hj
        apply get_none_of_all_gt
        intro a ha; have := hkeys a ha; omega
      have hmem : ∀ j, e.1 ≤ j → mem (skipBelow e.1 ps) j = mem ps j := by
        intro j hj; simp [mem, get_skipBelow e.1 ps j hj]
      simp only [pureRemove]
      split
      · rename_i p rest hh
        split
        · rename_i hpe
          by_cases hk : k = e.1
          · subst hk
            rw [hlow _ (Nat.le_refl _)]
            have : mem ps e.1 = true := by
              rw [← hmem _ (Nat.le_refl _), hh, ← hpe]; exact mem_self
            simp [this]
          · rw [get_cons_ne hk]
            by_cases hlt : k < e.1
            · rw [hlow _ (by omega)]
              have : get es k = none := by
                apply get_none_of_all_gt; intro a ha; have := sorted_head_lt ho a ha; omega
              simp [this]
            · rw [ih, hmem _ (by omega)]
        · rename_i hpe
          by_cases hk : k = e.1
          · subst hk
            rw [mem_skipBelow_head_ne hs hh (Ne.symm (Ne.symm hpe))]; simp [get]
          · rw [get_cons_ne hk, get_cons_ne hk]
            by_cases hlt : k < e.1
            · rw [hlow _ (by omega)]
              have : get es k = none := by
                apply get_none_of_all_gt; intro a ha; have := sorted_head_lt ho a ha; omega
              simp [this]
            · rw [ih, hmem _ (by omega)]
      · rename_i hh
        by_cases hk : k = e.1
        · subst hk
          rw [mem_skipBelow_nil hh]; simp [get]
        · rw [get_cons_ne hk, get_cons_ne hk]
          by_cases hlt : k < e.1
          · rw [hlow _ (by omega)]
            have : get es k = none := by
              apply get_none_of_all_gt; intro a ha; have := sorted_head_lt ho a ha; omega
            simp [this]
          · rw [ih, hmem _ (by omega)]


/-- Old entries whose id some pair names. -/
def cntL (old ps : List Entry) : Nat := (old.filter (fun e => mem ps e.1)).length

theorem cntL_cons (e : Entry) (es ps : List Entry) :
    cntL (e :: es) ps = (if mem ps e.1 then 1 else 0) + cntL es ps := by
  unfold cntL; by_cases h : mem ps e.1 <;> simp [List.filter_cons, h] <;> omega

theorem cntL_congr {es ps qs : List Entry} (h : ∀ a ∈ es, mem ps a.1 = mem qs a.1) :
    cntL es ps = cntL es qs := by
  unfold cntL; congr 1; exact List.filter_congr h

theorem pureRemove_count : ∀ (old ps : List Entry), Sorted old → Sorted ps →
    (pureRemove old ps).2 = cntL old ps
  | [], ps, _, _ => by simp [pureRemove, cntL]
  | e :: es, ps, ho, hs => by
      have hes := sorted_tail ho
      have hs' := skipBelow_sorted e.1 ps hs
      have ih := pureRemove_count es (skipBelow e.1 ps) hes hs'
      have hmem : ∀ a ∈ es, mem (skipBelow e.1 ps) a.1 = mem ps a.1 := by
        intro a ha
        have := sorted_head_lt ho a ha
        simp [mem, get_skipBelow e.1 ps a.1 (by omega)]
      rw [cntL_cons, ← cntL_congr hmem, ← ih]
      simp only [pureRemove]
      split
      · rename_i p rest hh
        split
        · rename_i hpe
          have : mem ps e.1 = true := by
            have h2 : mem (skipBelow e.1 ps) e.1 = mem ps e.1 := by
              simp [mem, get_skipBelow e.1 ps e.1 (Nat.le_refl _)]
            rw [← h2, hh, ← hpe]; exact mem_self
          simp [this]; omega
        · rename_i hpe
          rw [mem_skipBelow_head_ne hs hh (Ne.symm (Ne.symm hpe))]; simp
      · rename_i hh
        rw [mem_skipBelow_nil hh]; simp

/-- The two ways of counting a key intersection agree on sorted lists — the
pure-removal fast path's `nremoved` (counted over the span) equals
`insert_attrs`' `nremoved` (counted over the pairs). -/
theorem cntL_eq_refRemoved : ∀ (old ps : List Entry), Sorted old → Sorted ps →
    cntL old ps = refRemoved old ps
  | [], ps, _, _ => by simp [cntL, refRemoved_nil_left]
  | e :: es, [], _, _ => by simp [cntL, refRemoved, mem, get]
  | e :: es, p :: ps, ho, hs => by
      by_cases h1 : e.1 < p.1
      · rw [cntL_cons, refRemoved_cons_drop]
        · rw [mem_none_of_lt_head hs h1]; simp
          exact cntL_eq_refRemoved es (p :: ps) (sorted_tail ho) hs
        · intro q hq; simp at hq; rcases hq with rfl | hq
          · omega
          · have := sorted_head_lt hs q hq; omega
      · by_cases h2 : e.1 = p.1
        · rw [cntL_cons, refRemoved_cons_right, refRemoved_cons_drop (cs := es)]
          · have hm1 : mem (p :: ps) e.1 = true := by rw [h2]; exact mem_self
            have hm2 : mem (e :: es) p.1 = true := by rw [← h2]; exact mem_self
            rw [hm1, hm2]; simp
            rw [cntL_congr (qs := ps)]
            · exact cntL_eq_refRemoved es ps (sorted_tail ho) (sorted_tail hs)
            · intro a ha
              have := sorted_head_lt ho a ha
              rw [mem_cons_ne (by omega)]
          · intro q hq; have := sorted_head_lt hs q hq; omega
        · have hlt : p.1 < e.1 := by omega
          rw [refRemoved_cons_right, mem_none_of_lt_head ho hlt]; simp
          rw [cntL_congr (qs := ps)]
          · exact cntL_eq_refRemoved (e :: es) ps ho (sorted_tail hs)
          · intro a ha; simp at ha
            rcases ha with rfl | ha
            · rw [mem_cons_ne (by omega)]
            · have := sorted_head_lt ho a ha; rw [mem_cons_ne (by omega)]

theorem pureRemove_ref (old ps : List Entry) (hn : allNull ps = true) (hs : Sorted ps)
    (ho : Sorted old) (k : Nat) : get (pureRemove old ps).1 k = refGet old ps k := by
  rw [pureRemove_get old ps ho hs]; unfold refGet
  unfold allNull at hn; rw [List.all_eq_true] at hn
  cases hp : get ps k with
  | none => simp [mem, hp]
  | some v =>
    obtain ⟨q, hq, _, rfl⟩ := get_some_mem hp
    have := hn q hq
    cases hv : q.2 with
    | null => simp [mem, hp]
    | inl i => simp [hv, Val.isNull] at this
    | heap s => simp [hv, Val.isNull] at this

/-! ## `merge_span` as a whole -/

/-- **`mergeSpan_correct`**: whichever of the three paths runs, `merge_span`
leaves a strictly sorted, null-free span that reads back as the reference
`refGet`, and reports the reference `(nremoved, nset)`. -/
theorem mergeSpan_correct (old pairs : List Entry) (ho : Sorted old) (hn : NoNull old)
    (hp : Sorted pairs) :
    let r := mergeSpan old pairs
    (∀ k, get r.1 k = refGet old pairs k) ∧ Sorted r.1 ∧ NoNull r.1 ∧
      r.2.1 = refRemoved old pairs ∧ r.2.2 = refSet pairs := by
  simp only [mergeSpan]
  split
  · rename_i h
    obtain ⟨_, h⟩ := h
    obtain ⟨c1, c2⟩ := patch_counts old pairs h
    exact ⟨patch_get old pairs h, sorted_of_keys_eq (patch_keys old pairs) ho,
      patch_noNull old pairs h hn, c1.symm, c2.symm⟩
  · split
    · rename_i _ h
      obtain ⟨_, h⟩ := h
      refine ⟨pureRemove_ref old pairs h hp ho, ?_, ?_, ?_, ?_⟩
      · exact ho.sublist (pureRemove_sublist old pairs)
      · intro e he; exact hn e ((pureRemove_sublist old pairs).subset he)
      · rw [pureRemove_count old pairs ho hp, cntL_eq_refRemoved old pairs ho hp]
      · unfold allNull at h; rw [List.all_eq_true] at h
        unfold refSet; simp only
        rw [List.filter_eq_nil_iff.2]; · rfl
        intro a ha; simp [h a ha]
    · exact ⟨mergeGen_get old pairs ho hp, mergeGen_sorted old pairs ho hp,
        mergeGen_noNull old pairs hn, mergeGen_removed old pairs ho hp, mergeGen_set old pairs⟩

end PendingCommit
