import GraphPersist.Codec
/-! # Reader outcomes, `RT`, words and scalar values -/
namespace GraphPersist.Persist
open GraphPersist

/-! ## Reader outcomes and the round-trip law -/

inductive Out (α : Type) where
  | ok (a : α) (rest : Stream)
  | err
  | eof
  deriving Repr

abbrev Dec (α : Type) := Stream → Out α

def Dec.pure {α} (a : α) : Dec α := fun s => .ok a s

def Dec.bind {α β} (d : Dec α) (f : α → Dec β) : Dec β := fun s =>
  match d s with
  | .ok a r => f a r
  | .err => .err
  | .eof => .eof

def Dec.fail {α} : Dec α := fun _ => .err

/-- `p` is `toks` cut short by at least one word. -/
def StrictPre (p toks : Stream) : Prop := ∃ q, q ≠ [] ∧ p ++ q = toks

/-- **Round trip with truncation.** -/
def RT {α} (d : Dec α) (toks : Stream) (a : α) : Prop :=
  (∀ r, d (toks ++ r) = .ok a r) ∧ ∀ p, StrictPre p toks → d p = .eof

theorem RT_pure {α} (a : α) : RT (Dec.pure a) [] a :=
  ⟨fun r => rfl, fun p ⟨q, hq, h⟩ => by simp at h; exact absurd h.2 hq⟩

theorem RT_bind {α β} {d : Dec α} {f : α → Dec β} {t1 t2 : Stream} {a : α} {b : β}
    (h1 : RT d t1 a) (h2 : RT (f a) t2 b) : RT (d.bind f) (t1 ++ t2) b := by
  refine ⟨fun r => ?_, fun p ⟨q, hq, hpq⟩ => ?_⟩
  · simp only [Dec.bind, List.append_assoc, h1.1, h2.1]
  · rcases List.append_eq_append_iff.1 hpq with ⟨as, h, h'⟩ | ⟨bs, h, h'⟩
    · -- t1 = p ++ as, q = as ++ t2
      by_cases has : as = []
      · subst has
        simp at h
        subst h
        have hq2 : t2 ≠ [] := by rintro rfl; simp at h'; exact hq h'
        have := h2.2 [] ⟨t2, hq2, rfl⟩
        have e := h1.1 []
        simp at e
        simp only [Dec.bind]
        rw [e]
        exact this
      · have := h1.2 p ⟨as, has, h.symm⟩
        simp only [Dec.bind, this]
    · -- p = t1 ++ bs, t2 = bs ++ q
      subst h
      have := h2.2 bs ⟨q, hq, h'.symm⟩
      simp only [Dec.bind, h1.1, this]

/-- Read `n` items with `d`. -/
def rep {α} (d : Dec α) : Nat → Dec (List α)
  | 0 => Dec.pure []
  | n + 1 => d.bind fun a => (rep d n).bind fun as => Dec.pure (a :: as)

theorem RT_rep {α} (d : Dec α) (enc : α → Stream) (good : α → Prop)
    (hd : ∀ a, good a → RT d (enc a) a) :
    ∀ xs : List α, (∀ x ∈ xs, good x) → RT (rep d xs.length) (xs.flatMap enc) xs
  | [], _ => RT_pure []
  | x :: xs, h => by
    have h1 := hd x (h x (by simp))
    have h2 := RT_rep d enc good hd xs (fun y hy => h y (by simp [hy]))
    have := RT_bind h1 (f := fun a => (rep d xs.length).bind fun as => Dec.pure (a :: as))
      (RT_bind h2 (f := fun as => Dec.pure (x :: as)) (RT_pure (x :: xs)))
    simpa [rep, List.flatMap_cons] using this

/-- A decoder that cannot read anything from the empty stream. -/
def NeedsInput {α} (d : Dec α) : Prop := d [] = .eof

/-! ## Words -/

/-- `read_unsigned`. -/
def readU : Dec Nat
  | .u v :: r => .ok v.toNat r
  | [] => .eof
  | _ => .err

/-- `write_unsigned(n)`. -/
def encU (n : Nat) : Stream := [.u (UInt64.ofNat n)]

theorem readU_rt (n : Nat) (h : n < 2 ^ 64) : RT readU (encU n) n := by
  refine ⟨fun r => ?_, fun p ⟨q, hq, hpq⟩ => ?_⟩
  · simp [readU, encU, UInt64.toNat_ofNat', Nat.mod_eq_of_lt h]
  · cases p with
    | nil => rfl
    | cons t ts =>
      simp [encU] at hpq
      obtain ⟨rfl, h2⟩ := hpq
      exact absurd h2.2 hq

/-! ## Scalar values (`impl Decode<19> for Value`, value.rs:2170) -/

/-- The tags `decOne` recognises that carry payload words (every scalar tag but NULL). -/
def payloadTags : List UInt64 :=
  [siType.T_BOOL, siType.T_INT64, siType.T_DOUBLE, siType.T_STRING,
   siType.T_INTERN ||| siType.T_STRING, siType.T_POINT, siType.T_DATETIME, siType.T_DATE,
   siType.T_TIME, siType.T_DURATION]

/-- Words still owed after a recognised tag (the stream ended inside a value). An unknown
one-word tag is a codec error, not a short read. -/
def owes : Stream → Bool
  | [] => true
  | [.u t] => payloadTags.contains t
  | [.u t, .d _] => t == siType.T_POINT
  | _ => false

/-- `Value::decode` as an `Out` reader: `decOne`, with "ran out of words" told apart. -/
def decV : Dec Val := fun s =>
  match decOne s with
  | some (v, r) => .ok v r
  | none => if owes s then .eof else .err

/-- A one-word stream whose tag is neither NULL nor a payload-bearing scalar tag is a codec
error, not `.eof`. -/
theorem decV_unknown_tag (t : UInt64) (h : payloadTags.contains t = false)
    (hn : (t == siType.T_NULL) = false) : decV [.u t] = .err := by
  simp only [payloadTags, List.contains_cons, List.contains_nil, Bool.or_eq_false_iff, Bool.or_false] at h
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := h
  simp at h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  simp [decV, decOne, owes, payloadTags, hn, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]

theorem strictPre_take {p toks : Stream} (h : StrictPre p toks) :
    p = toks.take p.length ∧ p.length < toks.length := by
  obtain ⟨q, hq, rfl⟩ := h
  refine ⟨by simp, ?_⟩
  have : 0 < q.length := List.length_pos_iff.2 hq
  simp; omega

theorem decV_rt (v : Val) (h : v.isScalar = true) : RT decV v.encode v := by
  refine ⟨fun r => by simp [decV, decOne_encode v r h], fun p hp => ?_⟩
  obtain ⟨hp, hl⟩ := strictPre_take hp
  rw [hp]
  generalize p.length = k at hl
  cases v <;> simp [Val.isScalar] at h <;> simp only [Val.encode, List.length_cons, List.length_nil] at hl ⊢ <;>
    (rcases k with _ | _ | _ | k) <;>
    first
    | omega
    | rfl
    | (cases ‹Bool› <;> simp (config := { decide := true }) [decV, decOne, owes])
    | simp (config := { decide := true }) [decV, decOne, owes]

end GraphPersist.Persist
