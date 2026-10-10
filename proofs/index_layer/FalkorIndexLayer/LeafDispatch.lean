import FalkorIndexLayer.LeafIndexedMerge
/-
# The `Leaf` enum (`cow_btree/leaf/mod.rs`, origin/main 8743953a8): dispatch,
`from_pairs` (format choice), `from_parts`, `raw`, `pow2_bytes_for`.
-/
namespace IndexLayer.Leaf

inductive LeafV
  | aos (b : List Nat)
  | compact (b : List Nat)
  | indexed (b : List Nat)

inductive Fmt | aos | compact
  deriving DecidableEq

def LeafV.count : LeafV → Nat
  | .aos b => aosCount b | .compact b => cCount b | .indexed b => ciCount b
def LeafV.key : LeafV → Nat → Nat
  | .aos b => aosKey b | .compact b => cKey b | .indexed b => ciKey b
def LeafV.doc : LeafV → Nat → Nat
  | .aos b => aosDoc b | .compact b => cDoc b | .indexed b => ciDoc b
/-- `doc_layout`: `(FIELD, FIELD + DOC_BYTES, DOC_BYTES)` for AoS (`leaf/mod.rs:215`), here at `DOC_BYTES = 8`
(general `D`: `docLayoutD`, `LeafAosD.lean`). -/
def LeafV.docLayout : LeafV → Nat × Nat × Nat
  | .aos _ => (8, 16, 8) | .compact b => cDocLayout b | .indexed b => ciDocLayout b
/-- `iter` / `to_pairs`. -/
def LeafV.toPairs (l : LeafV) : List P := pairsOf l.count l.key l.doc
/-- `raw`. -/
def LeafV.raw : LeafV → List Nat
  | .aos b | .compact b | .indexed b => b
def LeafV.format : LeafV → Fmt
  | .aos _ => .aos | _ => .compact
/-- `from_parts`. -/
def fromParts (f : Fmt) (b : List Nat) : LeafV :=
  match f with
  | .aos => .aos b
  | .compact => if isIndexed b then .indexed b else .compact b

/-- `pow2_bytes_for` = `narrow_int::width_for`. -/
def pow2BytesFor (x : Nat) : Nat := widthFor x

theorem dispatch_spec (b : List Nat) :
    (LeafV.aos b).toPairs = aosPairs b ∧ (LeafV.compact b).toPairs = cPairs b ∧
    (LeafV.indexed b).toPairs = ciPairs b ∧ (LeafV.aos b).raw = b ∧ (LeafV.compact b).raw = b ∧
    (LeafV.indexed b).raw = b := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `doc_layout` agrees with `doc` for every format: `doc(i)` reads
`width` bytes at `base + i * stride` (the cursor relies on this). -/
theorem docLayout_spec (l : LeafV) (i : Nat) :
    l.doc i = readWidth l.raw (l.docLayout.1 + i * l.docLayout.2.1) l.docLayout.2.2 := by
  cases l with
  | aos b =>
    simp only [LeafV.doc, LeafV.raw, LeafV.docLayout, aosDoc, aosRead, readU64, readWidth]
    congr 1; omega
  | compact b => rfl
  | indexed b => rfl

theorem pow2BytesFor_spec (x : Nat) (hx : x < 2 ^ 64) : W (pow2BytesFor x) ∧ x < 256 ^ pow2BytesFor x :=
  ⟨widthFor_W x, lt_widthFor x hx⟩

/-! ## `from_pairs` -/

/-- The `windows(2)` scan: distinct-key count and the max doc over run ends. -/
def scan : List P → Nat → Nat → Nat × Nat
  | [], d, m => (d, m)
  | [_], d, m => (d, m)
  | a :: b :: r, d, m => if a.1 ≠ b.1 then scan (b :: r) (d + 1) (max m a.2) else scan (b :: r) d m

/-- `Leaf::from_pairs`: pick AoS, compact or compact-indexed by encoded size. -/
def fromPairs (ps : List P) : LeafV :=
  match ps.getLast? , ps.head? with
  | some last, some first =>
    let count := ps.length
    let (distinct, md0) := scan ps 1 0
    let maxDoc := max md0 last.2
    let minValue := first.1
    let vw := pow2BytesFor (last.1 - minValue)
    let dw := pow2BytesFor maxDoc
    let dedup := decide (distinct < count)
    let compactSize := 14 + distinct * vw + (if dedup then count else 0) + count * dw
    let aosSize := count * 16   -- `count * (FIELD + DOC_BYTES)` at `DOC_BYTES = 8` (`fromPairsD`)
    if compactSize + 8 * count ≤ aosSize then
      if dedup then .indexed (ciBuild ps minValue vw dw) else .compact (cBuild ps minValue vw dw)
    else .aos (aosBuild ps)
  | _, _ => .aos (aosBuild ps)

def Sorted2 (ps : List P) : Prop := ps.Pairwise (lt lexLe)

theorem lt_lex (a b : P) (h : lt lexLe a b) : a.1 < b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2) := by
  obtain ⟨h1, h2⟩ := h
  simp only [lexLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h1
  rcases h1 with h | ⟨h, h'⟩
  · exact Or.inl h
  · right; refine ⟨h, ?_⟩
    rcases Nat.lt_or_ge a.2 b.2 with h3 | h3
    · exact h3
    · exact absurd (Prod.ext h (by omega)) h2

/-- Every doc is `<=` the scanned max (docs grow within a run of equal keys). -/
theorem scan_doc : ∀ (ps : List P) (d m : Nat), Sorted2 ps →
    (∀ p ∈ ps, p.2 ≤ max (scan ps d m).2 ((ps.getLast?.map (·.2)).getD 0)) ∧ m ≤ (scan ps d m).2
  | [], d, m, _ => ⟨by simp, by simp [scan]⟩
  | [a], d, m, _ => ⟨fun p hp => by simp at hp; subst hp; exact Nat.le_max_right _ _, by simp [scan]⟩
  | a :: b :: r, d, m, hs => by
    have hs' := List.pairwise_cons.mp hs
    have hab := lt_lex a b (hs'.1 b (by simp))
    have hlast : (a :: b :: r).getLast? = (b :: r).getLast? := by simp [List.getLast?_cons_cons]
    simp only [scan, hlast]
    split
    · next hne =>
      obtain ⟨i1, i2⟩ := scan_doc (b :: r) (d + 1) (max m a.2) hs'.2
      refine ⟨fun p hp => ?_, by omega⟩
      rcases List.mem_cons.mp hp with rfl | hp
      · omega
      · exact i1 p hp
    · next heq =>
      obtain ⟨i1, i2⟩ := scan_doc (b :: r) d m hs'.2
      refine ⟨fun p hp => ?_, i2⟩
      rcases List.mem_cons.mp hp with rfl | hp
      · have := i1 b (by simp)
        rcases hab with h | ⟨_, h⟩
        · simp at heq; omega
        · omega
      · exact i1 p hp

/-- The scanned distinct count is the number of runs `build` creates. -/
theorem scan_distinct : ∀ (ps : List P) (d : List Nat) (n m : Nat), ps ≠ [] →
    d.getLast? = (ps.head?.map (·.1)) → n = d.length →
    (scan ps n m).1 = (runs ps.tail d).1.length
  | [], _, _, _, h, _, _ => absurd rfl h
  | [a], d, n, m, _, _, hn => by simp [scan, runs, hn]
  | a :: b :: r, d, n, m, _, hl, hn => by
    simp only [scan, List.tail_cons, runs]
    simp at hl
    split
    · next hne =>
      have hl' : d.getLast? ≠ some b.1 := by rw [hl]; simpa using hne
      simp only [hl', ite_false]
      have := scan_distinct (b :: r) (d ++ [b.1]) (n + 1) (max m a.2) (by simp) (by simp) (by simp [hn])
      simpa [runs] using this
    · next heq =>
      have hl' : d.getLast? = some b.1 := by rw [hl]; simpa using heq
      simp only [hl', ite_true]
      have := scan_distinct (b :: r) d n m (by simp) (by rw [hl]; simpa using heq) hn
      simpa [runs] using this

theorem scan_bound (B : Nat) : ∀ (ps : List P) (d m : Nat), (∀ p ∈ ps, p.2 < B) → m < B → (scan ps d m).2 < B
  | [], _, _, _, hm => hm
  | [_], _, _, _, hm => hm
  | a :: b :: r, d, m, h, hm => by
    simp only [scan]
    split
    · exact scan_bound B (b :: r) _ _ (fun p hp => h p (by simp [hp])) (by have := h a (by simp); omega)
    · exact scan_bound B (b :: r) _ _ (fun p hp => h p (by simp [hp])) hm

theorem sorted_bounds (ps : List P) (hs : Sorted2 ps) (first last : P) (hf : ps.head? = some first)
    (hl : ps.getLast? = some last) : ∀ p ∈ ps, first.1 ≤ p.1 ∧ p.1 ≤ last.1 := by
  have le_of : ∀ a b, lt lexLe a b → a.1 ≤ b.1 := fun a b h => by rcases lt_lex a b h with h | h <;> omega
  intro p hp
  constructor
  · cases ps with
    | nil => simp at hf
    | cons a r =>
      simp at hf; subst hf
      rcases List.mem_cons.mp hp with rfl | hp
      · exact Nat.le_refl _
      · exact le_of _ _ ((List.pairwise_cons.mp hs).1 p hp)
  · obtain ⟨init, hinit⟩ : ∃ init, ps = init ++ [last] := by
      rw [List.getLast?_eq_some_iff] at hl; exact hl
    subst hinit
    rcases List.mem_append.mp hp with hp | hp
    · exact le_of _ _ (List.pairwise_append.mp hs |>.2.2 p hp last (by simp))
    · simp at hp; subst hp; exact Nat.le_refl _

/-- **PROVEN** (`Leaf::from_pairs` round trip): for a strictly sorted entry list
of `u64`s with at most `LEAF_MAX <= 256` entries, whichever format the size
heuristic picks reads back as the input. -/
theorem fromPairs_roundtrip (ps : List P) (hs : Sorted2 ps) (hu : AllU64 ps) (hn : ps.length ≤ 256) :
    (fromPairs ps).toPairs = ps := by
  unfold fromPairs
  split
  · next last first hl hf =>
    obtain ⟨hdoc, -⟩ := scan_doc ps 1 0 hs
    simp only [hl, Option.map_some, Option.getD_some] at hdoc
    have hb := sorted_bounds ps hs first last hf hl
    have hlastin : last ∈ ps := List.mem_of_getLast? hl
    have hfirstin : first ∈ ps := List.mem_of_mem_head? hf
    have hvw := pow2BytesFor_spec (last.1 - first.1) (by have := (hu last hlastin).1; unfold U64 at this; omega)
    have hmd : max (scan ps 1 0).2 last.2 < 2 ^ 64 := by
      have h1 := (hu last hlastin).2
      have h2 := scan_bound (2 ^ 64) ps 1 0 (fun p hp => (hu p hp).2) (by decide)
      unfold U64 at h1; omega
    have hdw := pow2BytesFor_spec _ hmd
    have hfit : CFits first.1 (pow2BytesFor (last.1 - first.1)) (pow2BytesFor (max (scan ps 1 0).2 last.2)) ps := by
      intro p hp
      obtain ⟨b1, b2⟩ := hb p hp
      refine ⟨b1, by have := hvw.2; omega, by have := hdoc p hp; have := hdw.2; omega⟩
    have hmin : U64 first.1 := (hu first hfirstin).1
    generalize hsc : scan ps 1 0 = sc at hfit hdw
    obtain ⟨dc, md⟩ := sc
    simp only at hfit hdw ⊢
    repeat' split
    all_goals first
      | (rw [(dispatch_spec _).2.2.1]; exact ciBuild_roundtrip ps _ _ _ hn hmin hvw.1 hdw.1 hfit)
      | (rw [(dispatch_spec _).2.1]; exact cBuild_roundtrip ps _ _ _ (by omega) hmin hvw.1 hdw.1 hfit)
      | (rw [(dispatch_spec _).1]; exact (aos_roundtrip ps hu).2.2)
  · rw [(dispatch_spec _).1]; exact (aos_roundtrip ps hu).2.2

/-- **PROVEN** (`from_parts` recovers the variant from the persisted bytes): AoS
stays AoS; a compact page is re-recognised as plain or indexed from its header
(`distinct_count < count`). -/
theorem fromParts_spec (b : List Nat) (h : Hdr) (hok : h.ok) (ds os dv idx docs : List Nat) :
    fromParts .aos b = .aos b ∧
    (h.dc = h.n → fromParts .compact (cBytes h ds os) = .compact (cBytes h ds os)) ∧
    (h.dc < h.n → fromParts .compact (ciBytes h dv idx docs) = .indexed (ciBytes h dv idx docs)) := by
  refine ⟨rfl, fun hd => ?_, fun hd => ?_⟩
  · obtain ⟨r1, -, -, r4, -⟩ := cBytes_hdr h hok ds os
    simp [fromParts, isIndexed, r1, r4, hd]
  · obtain ⟨r1, -, -, r4, -⟩ := ciBytes_hdr h hok dv idx docs
    simp [fromParts, isIndexed, r1, r4, hd]

theorem isIndexed_spec (b : List Nat) : isIndexed b = decide (readU16 b 11 < readU16 b 0) := rfl

theorem fromParts_format (l : LeafV) : (fromParts l.format l.raw).format = l.format := by
  cases l with
  | aos b => rfl
  | compact b => simp only [LeafV.format, LeafV.raw, fromParts]; by_cases hi : isIndexed b = true <;> simp [hi]
  | indexed b => simp only [LeafV.format, LeafV.raw, fromParts]; by_cases hi : isIndexed b = true <;> simp [hi]

end IndexLayer.Leaf
