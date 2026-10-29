import GraphPersist.Persist.MultiLoad
/-! # Multi-key save → load: the assembled theorem -/
namespace GraphPersist.Persist
open GraphPersist

variable {M Tn Nm Ix Cn : Type}

/-! ## Every entry `fillKey` emits is a non-empty slice, at most one per kind -/

theorem fillKey_pos : ∀ (n cap : Nat) (ts : List Kind), cap + ts.length = n →
    (∀ e ∈ (fillKey cap ts).1, 0 < e.count) ∧ (fillKey cap ts).1.length ≤ ts.length := by
  intro n
  induction n using Nat.strongRecOn with
  | _ n ih =>
  intro cap ts hn
  match ts with
  | [] => simp [fillKey]
  | (st, tot, off) :: rest =>
    simp only [List.length_cons] at hn
    by_cases h0 : cap = 0
    · subst h0; simp [fillKey]
    by_cases hd : off + min cap (tot - off) ≥ tot
    · obtain ⟨i1, i2⟩ := ih _ (by omega) (cap - min cap (tot - off)) rest rfl
      rw [fillKey, dif_neg h0, dif_pos hd]
      refine ⟨fun e he => ?_, ?_⟩
      · simp only [List.mem_append] at he
        rcases he with he | he
        · split at he <;> simp at he; subst he; assumption
        · exact i1 e he
      · simp only [List.length_append, List.length_cons]; split <;> simp <;> omega
    · obtain ⟨i1, i2⟩ := ih _ (by simp only [List.length_cons]; omega) (cap - min cap (tot - off))
        ((st, tot, off + min cap (tot - off)) :: rest) rfl
      have hc0 : cap - min cap (tot - off) = 0 := by omega
      have hnil : (fillKey (cap - min cap (tot - off)) ((st, tot, off + min cap (tot - off)) :: rest)).1 = [] := by
        rw [hc0]; simp [fillKey]
      rw [fillKey, dif_neg h0, dif_neg hd]
      refine ⟨fun e he => ?_, ?_⟩
      · simp only [List.mem_append, hnil] at he
        rcases he with he | he
        · split at he <;> simp at he; subst he; assumption
        · simp at he
      · simp only [List.length_append, hnil]; split <;> simp

theorem keysFrom_entries (cap : Nat) : ∀ (n : Nat) (ts : List Kind), ts.length ≤ 4 →
    ∀ es ∈ keysFrom cap n ts, (∀ e ∈ es, 0 < e.count) ∧ es.length ≤ 4
  | 0, _, _ => by simp [keysFrom]
  | n + 1, ts, hl => by
    intro es hes
    simp only [keysFrom, List.mem_cons] at hes
    obtain ⟨i1, i2⟩ := fillKey_pos _ cap ts rfl
    rcases hes with rfl | hes
    · exact ⟨i1, by omega⟩
    · have : (fillKey cap ts).2.length ≤ 4 := by
        have h := (fillKey_spec _ cap ts rfl)
        -- the leftover kinds are a suffix-shaped rewrite of `ts`: never more of them
        have hk : ∀ (m c : Nat) (ts : List Kind), c + ts.length = m → (fillKey c ts).2.length ≤ ts.length := by
          intro m
          induction m using Nat.strongRecOn with
          | _ m ihm =>
          intro c ts hm
          match ts with
          | [] => simp [fillKey]
          | (st, tot, off) :: rest =>
            simp only [List.length_cons] at hm
            by_cases h0 : c = 0
            · subst h0; simp [fillKey]
            by_cases hd : off + min c (tot - off) ≥ tot
            · rw [fillKey, dif_neg h0, dif_pos hd]
              have := ihm _ (by omega) (c - min c (tot - off)) rest rfl
              simp only [List.length_cons]; omega
            · rw [fillKey, dif_neg h0, dif_neg hd]
              have := ihm _ (by simp only [List.length_cons]; omega) (c - min c (tot - off))
                ((st, tot, off + min c (tot - off)) :: rest) rfl
              simpa using this
        clear h
        have := hk _ cap ts rfl; omega
      exact keysFrom_entries cap n _ this es hes

/-! ## Each kind is fully covered by the keys -/

theorem expand_mem (es : List Entry) (st : St) (o : Nat) :
    (st, o) ∈ expand es ↔ ∃ e ∈ es, e.st = st ∧ e.offset ≤ o ∧ o < e.offset + e.count := by
  simp only [expand, List.mem_flatMap, List.mem_map, List.mem_range, Prod.mk.injEq]
  constructor
  · rintro ⟨e, he, i, hi, h1, h2⟩; exact ⟨e, he, h1, by omega, by omega⟩
  · rintro ⟨e, he, h1, h2, h3⟩; exact ⟨e, he, o - e.offset, by omega, h1, by omega⟩

theorem covered_kind (g : G M Tn Nm Ix Cn) (vmax : Nat) (ht : total g ≤ U64MAX) (st : St) (o : Nat)
    (ho : o < kindTotal g st) (hst : IsEntity st) :
    ∃ e ∈ (keysFrom (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)).flatten,
      e.st = st ∧ e.offset ≤ o ∧ o < e.offset + e.count := by
  have := multi_key_tiles g vmax ht
  rw [← expand_mem, this]
  rcases hst with rfl | rfl | rfl | rfl <;> simp [allIds, kindTotal] at ho ⊢ <;> exact ho

theorem slice_in_kind (g : G M Tn Nm Ix Cn) (vmax : Nat) (ht : total g ≤ U64MAX) (e : Entry)
    (he : e ∈ (keysFrom (if vmax = 0 then U64MAX else vmax) (keyCount (total g) vmax) (kindsOf g)).flatten)
    (hc : 0 < e.count) : e.offset + e.count ≤ kindTotal g e.st := by
  have hmem : (e.st, e.offset + (e.count - 1)) ∈ allIds g := by
    rw [← multi_key_tiles g vmax ht, expand_mem]; exact ⟨e, he, rfl, by omega, by omega⟩
  obtain ⟨st, c, o⟩ := e
  simp only at hc ⊢
  cases st <;> simp [allIds, kindTotal] at hmem ⊢ <;> omega

end GraphPersist.Persist
