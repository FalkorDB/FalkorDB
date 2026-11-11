import FalkorEffectsCodec.Records
/-!
# Blocks with a proven roundtrip, and how they compose

A `Blk α` is a writer, a reader, and the condition under which the reader
inverts the writer *whatever bytes follow*. `Blk.seq` composes two blocks the
way a Rust `encode` body writes one field after another (the second may depend
on the first, as `IdList::decode_sized(r, count)` depends on the header), and
its roundtrip is proven once.
-/
namespace FalkorCodec

structure Blk (α : Type) where
  enc : α → Except EErr Bytes
  dec : R α
  ok  : α → Prop
  rt  : ∀ a bs rest, ok a → enc a = .ok bs → dec (bs ++ rest) = .ok (a, rest)

def eseq (x y : Except EErr Bytes) : Except EErr Bytes :=
  match x, y with
  | .ok a, .ok b => .ok (a ++ b)
  | .error e, _ => .error e
  | _, .error e => .error e

theorem eseq_ok {x y : Except EErr Bytes} {bs : Bytes} (h : eseq x y = .ok bs) :
    ∃ a b, x = .ok a ∧ y = .ok b ∧ bs = a ++ b := by
  unfold eseq at h; split at h <;> simp_all

/-- Field after field; the second block may depend on the first value. -/
def Blk.seq {α β} (A : Blk α) (B : α → Blk β) : Blk (α × β) where
  enc p := eseq (A.enc p.1) ((B p.1).enc p.2)
  dec := do let a ← A.dec; let b ← (B a).dec; pure (a, b)
  ok p := A.ok p.1 ∧ (B p.1).ok p.2
  rt := by
    rintro ⟨a, b⟩ bs rest ⟨ha, hb⟩ h
    obtain ⟨x, y, hx, hy, rfl⟩ := eseq_ok h
    simp only [bind_apply, List.append_assoc]
    rw [A.rt a x _ ha hx]; simp only
    rw [(B a).rt b y _ hb hy]; rfl

/-- Change the carried type along an injection (decoder maps forward). -/
def Blk.map {α β} (A : Blk α) (f : α → β) (g : β → α) (hgf : ∀ a, g (f a) = a) : Blk β where
  enc b := A.enc (g b)
  dec := do let a ← A.dec; pure (f a)
  ok b := A.ok (g b) ∧ f (g b) = b
  rt := by
    intro b bs rest ⟨hb, hfg⟩ h
    simp only [bind_apply]; rw [A.rt _ _ _ hb h]; simp [hfg]

/-! ## Primitive blocks -/

def bU (k : Nat) : Blk Nat where
  enc n := .ok (le k n)
  dec := rdU k
  ok n := n < 2 ^ (8 * k)
  rt := by intro n bs rest h e; cases e; exact rdU_le h rest

abbrev bU8 := bU 1
abbrev bU16 := bU 2
abbrev bU32 := bU 4
abbrev bU64 := bU 8

/-- `string` / `Reader::string`. -/
def bStr (utf8 : Bytes → Bool) : Blk Bytes where
  enc s := .ok (wString s)
  dec := rString utf8
  ok s := s.length + 1 < 2 ^ 64 ∧ utf8 s = true
  rt := by intro s bs rest ⟨h1, h2⟩ e; cases e; exact rString_wString utf8 s rest h1 h2

/-- A fixed value read back and checked (`Opcode::try_from`, tag decoders):
`dec` reads a `u32` and maps it through a partial inverse. -/
def bTag {α} (tag : α → Nat) (untag : Nat → Except DErr α) (h : ∀ a, untag (tag a) = .ok a)
    (hlt : ∀ a, tag a < 2 ^ 32) : Blk α where
  enc a := .ok (w32 (tag a))
  dec := do let v ← u32; match untag v with | .ok a => pure a | .error e => fail e
  ok _ := True
  rt := by
    intro a bs rest _ e; cases e
    simp only [bind_apply]; rw [u32_w32 (hlt a)]; simp only [h]; rfl

/-- `LabelSet` (`blocks.rs:54-81`). -/
def bLabels : Blk (List Nat) where
  enc := wLabelSet
  dec := rLabelSet
  ok ls := ls.length < 2 ^ 16 ∧ ∀ x ∈ ls, x < 2 ^ 32
  rt := by
    intro ls bs rest ⟨hl, hx⟩ e
    simp only [wLabelSet, hl, ite_true, Except.ok.injEq] at e; subst e
    simp only [rLabelSet, bind_apply, List.append_assoc]
    rw [u16_w16 hl]; simp only [bind_apply, guardCount]
    rw [if_neg (by simp only [List.length_append, flatMap_w32_length]; omega)]
    simp only; rw [readN_u32 ls hx]

/-- `AttrIds` (`blocks.rs:100-124`). -/
def bAttrIds : Blk (List Nat) where
  enc := wAttrIds
  dec := rAttrIds
  ok as := as.length < 2 ^ 16 ∧ ∀ x ∈ as, x < 2 ^ 16
  rt := by
    intro as bs rest ⟨hl, hx⟩ e
    simp only [wAttrIds, hl, ite_true, Except.ok.injEq] at e; subst e
    simp only [rAttrIds, bind_apply, List.append_assoc]
    rw [u16_w16 hl]; simp only [bind_apply, guardCount]
    rw [if_neg (by simp only [List.length_append, flatMap_w16_length]; omega)]
    simp only; rw [readN_u16 as hx]

/-- `IdList::encode` / `decode_sized(r, count)`. -/
def bIds (C : IdCodec) (count : Nat) : Blk C.T where
  enc := C.enc
  dec := C.dec count
  ok t := C.count t = count
  rt := by intro t bs rest h e; subst h; exact C.roundtrip t bs rest e

/-- `AttrValues` encode / `decode_sized(r, (count, attrs))`. -/
def bRows (utf8 I : Bytes → Bool) (count attrs : Nat) : Blk (List Val) where
  enc := encL I
  dec := rRows utf8 count attrs
  ok rows := rows.length = count * attrs ∧ WFL utf8 rows
  rt := by
    intro rows bs rest ⟨hl, hw⟩ e
    have hlen := (encL_len I rows bs e).2
    simp only [rRows, bind_apply, guardCount]
    rw [if_neg (by simp [MIN_VALUE_BYTES]; omega)]
    simp only; rw [← hl, readN_decodeV utf8 I rows bs rest e hw]

/-- `put_opt` (`records.rs:123`) / `take_opt` (`:137`). -/
def bOpt {α} (A : Blk α) : Blk (Option α) where
  enc
    | none => .ok (w8 0)
    | some x => eseq (.ok (w8 1)) (A.enc x)
  dec := do
    let t ← u8
    if t = 0 then pure none
    else if t = 1 then do let x ← A.dec; pure (some x)
    else fail (.badBool t)
  ok
    | none => True
    | some x => A.ok x
  rt := by
    intro o bs rest h e
    cases o with
    | none => cases e; simp only [bind_apply]; rw [u8_w8 (by decide)]; rfl
    | some x =>
      obtain ⟨a, b, ha, hb, rfl⟩ := eseq_ok e
      cases ha
      simp only [bind_apply, List.append_assoc]; rw [u8_w8 (by decide)]
      simp only [Nat.one_ne_zero, ite_false, ite_true, bind_apply]
      rw [A.rt x b rest h hb]; rfl

/-- `n` items after a count of `k` bytes (`u8`/`u16`), with
`guard_count(n, minEach)` first — `IndexFields`, `IndexSchemas`,
`read_constraint_props`. -/
def readItems {α} (k minEach : Nat) (A : R α) : R (List α) := do
  let n ← rdU k
  let n ← guardCount n minEach
  readN n A

/-- The per-item loop of the encoders (`for x in xs { … }`), appending. -/
def encItems {α} (A : α → Except EErr Bytes) : List α → Except EErr Bytes
  | [] => .ok []
  | x :: xs => eseq (A x) (encItems A xs)

theorem readN_items {α} (A : Blk α) : ∀ (xs : List α) (bs rest : Bytes), (∀ x ∈ xs, A.ok x) →
    encItems A.enc xs = .ok bs → readN xs.length A.dec (bs ++ rest) = .ok (xs, rest) := by
  intro xs
  induction xs with
  | nil => intro bs rest _ h; cases h; rfl
  | cons x xs ih =>
    intro bs rest hok h
    obtain ⟨a, b, ha, hb, rfl⟩ := eseq_ok h
    simp only [List.length_cons, readN, bind_apply, List.append_assoc]
    rw [A.rt x a _ (hok x (by simp)) ha]; simp only
    rw [ih b rest (fun y hy => hok y (by simp [hy])) hb]; rfl

theorem encItems_len {α} (A : α → Except EErr Bytes) (m : Nat) (hA : ∀ x bs, A x = .ok bs → m ≤ bs.length) :
    ∀ (xs : List α) bs, encItems A xs = .ok bs → m * xs.length ≤ bs.length := by
  intro xs; induction xs with
  | nil => intro bs _; simp
  | cons x xs ih =>
    intro bs h
    obtain ⟨a, b, ha, hb, rfl⟩ := eseq_ok h
    have := hA x a ha; have := ih b hb
    simp [Nat.mul_succ]; omega

/-- A counted list: `k`-byte count, `guard_count(n, minEach)`, then the items. -/
def bItems {α} (k minEach : Nat) (tooLong : EErr) (A : Blk α)
    (hmin : ∀ x bs, A.enc x = .ok bs → minEach ≤ bs.length) : Blk (List α) where
  enc xs := if xs.length < 2 ^ (8 * k) then eseq (.ok (le k xs.length)) (encItems A.enc xs)
    else .error tooLong
  dec := readItems k minEach A.dec
  ok xs := xs.length < 2 ^ (8 * k) ∧ ∀ x ∈ xs, A.ok x
  rt := by
    intro xs bs rest ⟨hl, hx⟩ e
    simp only [hl, ite_true] at e
    obtain ⟨a, b, ha, hb, rfl⟩ := eseq_ok e
    cases ha
    have hlen := encItems_len A.enc minEach hmin xs b hb
    simp only [readItems, bind_apply, List.append_assoc]
    rw [rdU_le hl]; simp only [bind_apply, guardCount]
    rw [if_neg (by simp only [List.length_append]; rw [Nat.mul_comm]; omega)]
    simp only; rw [readN_items A xs b rest hx hb]

end FalkorCodec
