import GraphPersist.Persist.Reader
/-! # Deleted-id bitmaps and attribute stores -/
namespace GraphPersist.Persist
open GraphPersist

/-! ## The deleted-id bitmap (`impl Encode/Decode<19> for RoaringTreemap`, serialization.rs:172-241) -/

/-- `id.to_le_bytes()`. -/
def le8 (n : Nat) : List UInt8 :=
  [UInt8.ofNat n, UInt8.ofNat (n / 2^8), UInt8.ofNat (n / 2^16), UInt8.ofNat (n / 2^24),
   UInt8.ofNat (n / 2^32), UInt8.ofNat (n / 2^40), UInt8.ofNat (n / 2^48), UInt8.ofNat (n / 2^56)]

/-- `u64::from_le_bytes(bytes[i*8..(i+1)*8])` for each `i`. -/
def dec8 : List UInt8 → List Nat
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest =>
    (b0.toNat + 2^8 * b1.toNat + 2^16 * b2.toNat + 2^24 * b3.toNat + 2^32 * b4.toNat +
      2^40 * b5.toNat + 2^48 * b6.toNat + 2^56 * b7.toNat) :: dec8 rest
  | _ => []

theorem le8_length (n : Nat) : (le8 n).length = 8 := rfl

theorem dec8_le8 (n : Nat) (h : n < 2 ^ 64) (rest : List UInt8) : dec8 (le8 n ++ rest) = n :: dec8 rest := by
  simp only [le8, List.cons_append, List.nil_append, dec8, UInt8.toNat_ofNat', List.cons.injEq, and_true]
  omega

theorem dec8_flat (ids : List Nat) (h : ∀ i ∈ ids, i < 2 ^ 64) : dec8 (ids.flatMap le8) = ids := by
  induction ids with
  | nil => rfl
  | cons i is ih =>
    rw [List.flatMap_cons, dec8_le8 i (h i (by simp)), ih (fun j hj => h j (by simp [hj]))]

theorem flat_le8_length (ids : List Nat) : (ids.flatMap le8).length = ids.length * 8 := by
  induction ids with
  | nil => rfl
  | cons i is ih => simp [List.flatMap_cons, le8_length, ih, Nat.succ_mul]; omega

/-- `encode_with_range(count, offset)`: one buffer of the `count` ids after `offset`. -/
def encRoar (ids : List Nat) (count off : Nat) : Stream :=
  [.bytes (((ids.drop off).take count).flatMap le8)]

/-- `decode_with_count(count)`: the buffer must be exactly `count * 8` bytes. -/
def decRoar (count : Nat) : Dec (List Nat)
  | .bytes b :: r => if b.length = count * 8 then .ok (dec8 b) r else .err
  | [] => .eof
  | _ => .err

theorem decRoar_rt (l : List Nat) (h : ∀ i ∈ l, i < 2 ^ 64) :
    RT (decRoar l.length) [.bytes (l.flatMap le8)] l := by
  refine ⟨fun r => ?_, fun p hp => ?_⟩
  · simp only [List.cons_append, List.nil_append, decRoar, flat_le8_length, ite_true, dec8_flat l h]
  · obtain ⟨hp, hl⟩ := strictPre_take hp
    cases p with
    | nil => rfl
    | cons t ts => simp at hl

/-! ## Attribute spans (`AttributeStore::encode_with_range` / `decode_with_count`,
attribute_store.rs:1409-1497) -/

abbrev Span := List (Nat × Val)
/-- An attribute store: entity id ↦ its span (absent = no attributes). -/
abbrev Store := Nat → Option Span

def _root_.GraphPersist.Val.isNull : Val → Bool
  | .vnull => true
  | _ => false

/-- One entity on the wire: id, attribute count, then `(attr_id, value)` pairs. -/
def encEnt (S : Store) (i : Nat) : Stream :=
  encU i ++ encU ((S i).getD []).length ++ ((S i).getD []).flatMap fun (k, v) => encU k ++ v.encode

/-- One raw `(attr_id, value)`: `u16::try_from(id).ok()`, and the value is read
whether or not the id is usable. -/
def decAttr : Dec (Option Nat × Val) :=
  readU.bind fun k => decV.bind fun v => Dec.pure (if k < 65536 then some k else none, v)

/-- One entity: id, count, then that many raw attributes. -/
def decEnt : Dec (Nat × List (Option Nat × Val)) :=
  readU.bind fun i => readU.bind fun n => (rep decAttr n).bind fun es => Dec.pure (i, es)

/-- Keep `(id, v)` iff the id fits `u16`, is inside the dictionary, and `v` is not NULL. -/
def keep (limit : Nat) : Option Nat × Val → Option (Nat × Val)
  | (some k, v) => if k < limit && !v.isNull then some (k, v) else none
  | (none, _) => none

/-- Stable insertion of `x` by attribute id (`sort_by_key` is a stable sort; every stable
sort by the same key gives the same list). -/
def ins (x : Nat × Val) : Span → Span
  | [] => [x]
  | y :: ys => if y.1 < x.1 then y :: ins x ys else x :: y :: ys

def insSort : Span → Span
  | [] => []
  | x :: xs => ins x (insSort xs)

/-- The per-entity tail of `decode_with_count`: filter, sort, `set_span` if non-empty. -/
def applyEnt (limit : Nat) (S : Store) (e : Nat × List (Option Nat × Val)) : Store :=
  let sp := insSort (e.2.filterMap (keep limit))
  if sp = [] then S else fun j => if j = e.1 then some sp else S j

def upd {β : Type} (f : Nat → β) (a : Nat) (x : β) : Nat → β := fun b => if b = a then x else f b

/-- What a stored span must satisfy for `load (save)` to give it back unchanged:
non-empty (an empty span is not stored), strictly sorted by attribute id, ids inside the
dictionary, no NULL values (NULL is a removal), scalar values. -/
def GoodSpan (limit : Nat) (sp : Span) : Prop :=
  sp ≠ [] ∧ sp.Pairwise (fun a b => a.1 < b.1) ∧ ∀ kv ∈ sp, kv.1 < limit ∧ kv.2.isNull = false ∧
    kv.2.isScalar = true

theorem insSort_sorted : ∀ sp : Span, sp.Pairwise (fun a b => a.1 < b.1) → insSort sp = sp
  | [], _ => rfl
  | x :: xs, h => by
    have hx := List.pairwise_cons.1 h
    rw [insSort, insSort_sorted xs hx.2]
    cases xs with
    | nil => rfl
    | cons y ys =>
      have : ¬ y.1 < x.1 := by have := hx.1 y (by simp); omega
      simp [ins, this]

theorem filterMap_keep (limit : Nat) (sp : Span) (h : ∀ kv ∈ sp, kv.1 < limit ∧ kv.2.isNull = false)
    (hl : limit ≤ 65536) :
    (sp.map fun kv => ((some kv.1 : Option Nat), kv.2)).filterMap (keep limit) = sp := by
  induction sp with
  | nil => rfl
  | cons kv sp ih =>
    obtain ⟨k, v⟩ := kv
    have ⟨h1, h2⟩ := h (k, v) (by simp)
    simp only [List.map_cons, List.filterMap_cons, keep, h1, h2, decide_true, Bool.not_false,
      Bool.and_self, ite_true]
    rw [ih (fun kv hkv => h kv (by simp [hkv]))]

/-- One entity reads back as its span with every id `some`. -/
theorem decEnt_rt (S : Store) (i : Nat) (hi : i < 2 ^ 64) (limit : Nat) (hl : limit ≤ 65536)
    (hS : ∀ sp, S i = some sp → GoodSpan limit sp) :
    RT decEnt (encEnt S i) (i, ((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2)) := by
  have hsp : ∀ kv ∈ (S i).getD [], kv.1 < limit ∧ kv.2.isNull = false ∧ kv.2.isScalar = true := by
    intro kv hkv
    cases hs : S i with
    | none => simp [hs] at hkv
    | some sp => rw [hs] at hkv; exact (hS sp hs).2.2 kv hkv
  have hlen : ((S i).getD []).length < 2 ^ 64 := by
    -- a span's ids are distinct and below `limit ≤ 65536`
    cases hs : S i with
    | none => simp
    | some sp =>
      have hg := hS sp hs
      have hnd : (sp.map Prod.fst).Nodup := by
        have := hg.2.1
        rw [List.Nodup, List.pairwise_map]
        exact this.imp (fun h => Nat.ne_of_lt h)
      have hsub : (sp.map Prod.fst) ⊆ List.range limit := by
        intro k hk; obtain ⟨kv, hkv, rfl⟩ := List.mem_map.1 hk; simpa using (hg.2.2 kv hkv).1
      have := List.Nodup.length_le_of_subset hnd hsub
      simp at this ⊢; omega
  -- the attribute pairs
  have hattr : ∀ kv ∈ (S i).getD [],
      RT decAttr (encU kv.1 ++ kv.2.encode) ((some kv.1 : Option Nat), kv.2) := by
    intro kv hkv
    obtain ⟨h1, _, h3⟩ := hsp kv hkv
    have hk : kv.1 < 65536 := by omega
    have := RT_bind (readU_rt kv.1 (by omega))
      (f := fun k => decV.bind fun v => Dec.pure (if k < 65536 then some k else none, v))
      (RT_bind (decV_rt kv.2 h3) (f := fun v => Dec.pure ((if kv.1 < 65536 then some kv.1 else none), v))
        (RT_pure _))
    simpa [decAttr, hk] using this
  have hrep := RT_rep decAttr (fun ra => encU (ra.1.getD 0) ++ ra.2.encode)
    (fun ra => ∃ kv ∈ (S i).getD [], ra = ((some kv.1 : Option Nat), kv.2))
    (by
      rintro ra ⟨kv, hkv, rfl⟩
      simpa using hattr kv hkv)
    (((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2))
    (by intro ra hra; obtain ⟨kv, hkv, rfl⟩ := List.mem_map.1 hra; exact ⟨kv, hkv, rfl⟩)
  have := RT_bind (readU_rt i hi)
    (f := fun i => readU.bind fun n => (rep decAttr n).bind fun es => Dec.pure (i, es))
    (RT_bind (readU_rt _ hlen)
      (f := fun n => (rep decAttr n).bind fun es => Dec.pure (i, es))
      (RT_bind (by simpa using hrep) (f := fun es => Dec.pure (i, es)) (RT_pure _)))
  simpa [decEnt, encEnt, List.flatMap_map, List.append_assoc] using this

/-- **Store round trip.** Decoding the entities of `ids` into a store `S0` leaves each
`j ∈ ids` that has a span with exactly that span and everything else as it was. -/
theorem fold_ents (limit : Nat) (hl : limit ≤ 65536) (S : Store) (ids : List Nat)
    (hg : ∀ i ∈ ids, ∀ sp, S i = some sp → GoodSpan limit sp) :
    ∀ S0 : Store, (ids.map fun i => (i, ((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2))).foldl
        (applyEnt limit) S0 = fun j => if j ∈ ids ∧ S j ≠ none then S j else S0 j := by
  induction ids with
  | nil => intro S0; funext j; simp
  | cons i is ih =>
    intro S0
    simp only [List.map_cons, List.foldl_cons]
    rw [ih (fun k hk => hg k (by simp [hk]))]
    have hstep : applyEnt limit S0 (i, ((S i).getD []).map fun kv => ((some kv.1 : Option Nat), kv.2)) =
        fun j => if j = i ∧ S i ≠ none then S i else S0 j := by
      cases hs : S i with
      | none => funext j; simp [applyEnt, insSort]
      | some sp =>
        have hgs := hg i (by simp) sp hs
        have hk : ∀ kv ∈ sp, kv.1 < limit ∧ kv.2.isNull = false :=
          fun kv h => ⟨(hgs.2.2 kv h).1, (hgs.2.2 kv h).2.1⟩
        simp only [applyEnt, Option.getD_some, filterMap_keep limit sp hk hl, insSort_sorted sp hgs.2.1,
          if_neg hgs.1]
        funext j; simp
    rw [hstep]
    funext j
    by_cases hj : j = i
    · subst hj; by_cases hs : S j = none <;> simp [hs]
    · simp [hj]

end GraphPersist.Persist
