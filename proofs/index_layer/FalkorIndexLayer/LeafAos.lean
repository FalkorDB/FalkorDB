import FalkorIndexLayer.LeafBytes
import FalkorIndexLayer.LeafMerge
/-
# `AosLeaf` (`cow_btree/leaf/aos.rs`, origin/main 3fec7d7c9) and the `(key, doc)` order

| Lean | Rust |
| --- | --- |
| `lexLe` | `(u64, u64)` tuple `<=` |
| `aosEnc`, `aosBuild` | `AosLeaf::build` `aos.rs:46` |
| `aosCount`, `aosRead`, `aosKey`, `aosDoc` | `count` 14, `read` 21, `key` 30, `doc` 38 |
| `aosMerge` | `AosLeaf::merge_batch` 60 (via `merge_walk`) |
| `pairsOf` | `Leaf::iter` / `to_pairs` (`leaf/mod.rs:233,238`) |
-/
namespace IndexLayer.Leaf

abbrev P := Nat × Nat

def lexLe (a b : P) : Bool := decide (a.1 < b.1) || (decide (a.1 = b.1) && decide (a.2 ≤ b.2))

theorem lexLe_lin : LinOrd lexLe where
  refl a := by simp [lexLe]
  trans a b c h1 h2 := by
    simp only [lexLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at *; omega
  total a b := by simp only [lexLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq]; omega
  antisymm a b h1 h2 := by
    simp only [lexLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at *
    exact Prod.ext (by omega) (by omega)

def U64 (x : Nat) : Prop := x < 2 ^ 64

theorem u64_256 (x : Nat) (h : U64 x) : x < 256 ^ 8 := by unfold U64 at h; simpa using h

/-- Read back `n` entries through `key`/`doc` accessors. -/
def pairsOf (n : Nat) (key doc : Nat → Nat) : List P := (List.range n).map (fun i => (key i, doc i))

theorem pairsOf_eq (ps : List P) (key doc : Nat → Nat)
    (h : ∀ i (hi : i < ps.length), key i = ps[i].1 ∧ doc i = ps[i].2) : pairsOf ps.length key doc = ps := by
  apply List.ext_getElem (by simp [pairsOf])
  intro i h1 h2
  simp only [pairsOf, List.getElem_map, List.getElem_range]
  obtain ⟨a, b⟩ := h i h2
  rw [a, b]

def aosEnc (p : P) : List Nat := le 8 p.1 ++ le 8 p.2
theorem aosEnc_len (p : P) : (aosEnc p).length = 16 := by simp [aosEnc, le_length]

/-- `AosLeaf::build`. -/
def aosBuild (ps : List P) : List Nat := ps.flatMap aosEnc
/-- `AosLeaf::count`: `len / STRIDE`. -/
def aosCount (b : List Nat) : Nat := b.length / 16
/-- `AosLeaf::read`: `read_u64(bytes, STRIDE * i + off)`. -/
def aosRead (b : List Nat) (i off : Nat) : Nat := readU64 b (16 * i + off)
def aosKey (b : List Nat) (i : Nat) : Nat := aosRead b i 0
def aosDoc (b : List Nat) (i : Nat) : Nat := aosRead b i 8
def aosPairs (b : List Nat) : List P := pairsOf (aosCount b) (aosKey b) (aosDoc b)

def AllU64 (ps : List P) : Prop := ∀ p ∈ ps, U64 p.1 ∧ U64 p.2

/-- **PROVEN** (`AosLeaf` round trip): a built page has `len` entries and reads
back every `(key, doc)`. -/
theorem aos_roundtrip (ps : List P) (hu : AllU64 ps) :
    aosCount (aosBuild ps) = ps.length ∧
    (∀ i (hi : i < ps.length), aosKey (aosBuild ps) i = ps[i].1 ∧ aosDoc (aosBuild ps) i = ps[i].2) ∧
    aosPairs (aosBuild ps) = ps := by
  have hc : aosCount (aosBuild ps) = ps.length := by
    simp [aosCount, aosBuild, length_flatMap_const aosEnc 16 aosEnc_len]
  have hkd : ∀ i (hi : i < ps.length), aosKey (aosBuild ps) i = ps[i].1 ∧ aosDoc (aosBuild ps) i = ps[i].2 := by
    intro i hi
    have hp := hu ps[i] (List.getElem_mem hi)
    have e : aosBuild ps = [] ++ ps.flatMap aosEnc ++ [] := by simp [aosBuild]
    constructor
    · unfold aosKey aosRead readU64
      rw [e, rd_in_chunk aosEnc 16 aosEnc_len [] [] ps i hi 0 8 (by omega) _ (by simp; omega)]
      simp only [aosEnc, List.drop_zero]
      rw [List.take_left' (le_length 8 _), unle_le_of_lt 8 _ (u64_256 _ hp.1)]
    · unfold aosDoc aosRead readU64
      rw [e, rd_in_chunk aosEnc 16 aosEnc_len [] [] ps i hi 8 8 (by omega) _ (by simp; omega)]
      simp only [aosEnc]
      rw [List.drop_left' (le_length 8 _), List.take_of_length_le (by rw [le_length]; exact Nat.le_refl _),
        unle_le_of_lt 8 _ (u64_256 _ hp.2)]
  refine ⟨hc, hkd, ?_⟩
  unfold aosPairs; rw [hc]; exact pairsOf_eq ps _ _ hkd

/-- `AosLeaf::merge_batch`: the `merge_walk` emissions, each re-encoded. -/
def aosMerge (b : List Nat) (batch : List P) : List Nat :=
  (mergeW lexLe (aosPairs b) batch none).flatMap (fun e => aosEnc e.1)

/-- **PROVEN**: merging into a built AoS page yields the page of the merged
entry list. -/
theorem aosMerge_spec (ps batch : List P) (hu : AllU64 ps) :
    aosMerge (aosBuild ps) batch = aosBuild (mergeD lexLe ps batch none) := by
  unfold aosMerge
  rw [(aos_roundtrip ps hu).2.2, aosBuild, ← mergeW_fst lexLe ps batch none, List.flatMap_map]

end IndexLayer.Leaf
