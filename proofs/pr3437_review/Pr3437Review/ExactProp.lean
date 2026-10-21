import FalkorCore.Value
import FalkorCore.Float

/-!
# PR #3437: `materialize_node_property_exact` (runtime.rs:1566-1587)

| Lean | Rust |
| --- | --- |
| `Tag`, `Packed` | `enum Tag`, `struct PackedAttr` (attribute_store.rs:257-278); `payload` = `i64::from_le_bytes(payload)` |
| `entry` | `self.data.get(key).and_then(|span| binary_search_by_key(attr_idx).map(entries[pos]))` — the lookup both `get_int_attrs_by_idx_batch_into` (attribute_store.rs:1194-1200) and `SpanRef::get` (:856-865) perform |
| `unpack` | `Block::unpack` (attribute_store.rs:486-505): `Null → Null`, `Int → Int(payload)`, other tags opaque |
| `batchValues` | `get_attrs_by_idx_batch_into(keys, idx, &Null, out)` (attribute_store.rs:1221-1232) via `get_node_attributes_by_idx` (graph.rs:2587) |
| `getIntLoop` | `get_int_attrs_by_idx_batch_into` (attribute_store.rs:1186-1216) via `get_node_int_attributes_by_idx` (graph.rs:2602-2613); `on_null(i)` = `nulls.set(i)` |
| `matValues` | `materialize_node_property_values` (runtime.rs:1595-1633); the deleted/pending path is `slow` (unchanged) |
| `classifyExact` | `classify_exact_column` (batch.rs:295) = `NullBitmap::from_values` + `classify_numeric(_, true, Pure)` (batch.rs:191-231) |
| `exactPR` | `materialize_node_property_exact` (runtime.rs:1566-1587) |
| `evalPropPR`, `evalPropMain` | `VectorEval::eval_property`'s column (vector_expr.rs:337-353; main: `classify_exact_column(property_values(..)?)`, and `property_values` on `NodeIds` is `materialize_node_property_values` of the gathered ids, :379-386) |

A `NullBitmap` is compared bit by bit on `0..len` (`Bits`).

**`exact_eq_main`**: for every store, every guard outcome and every id list,
`materialize_node_property_exact` returns *exactly* main's
`classify_exact_column(materialize_node_property_values(..))` — on the integer fast path the
same `Ints` lane (a `0` under each null) and the same null bits; in every other case
(an attribute the graph never registered, deletions, pending attribute writes, or any
stored value that is neither an integer nor null) it *is* that call.
-/

namespace Pr3437.ExactProp
open FalkorCore

inductive Tag where
  | null | bool | int | float | point | datetime | date | time | duration | heap
  deriving DecidableEq

structure Packed where
  tag : Tag
  payload : Int

variable (entry : Nat → Option Packed) (unpackOther : Packed → V F)

def unpack (p : Packed) : V F :=
  match p.tag with
  | .null => .null
  | .int => .int p.payload
  | _ => unpackOther p

/-- `get_attrs_by_idx_batch_into` with `default = Null`. -/
def batchValues (keys : List Nat) : List (V F) :=
  keys.map fun k => ((entry k).map (unpack unpackOther)).getD .null

abbrev Bits := Nat → Bool

/-- The `for (i, &key) in keys.iter().enumerate()` loop; `none` is `return false`. -/
def getIntLoop : Nat → List Int → Bits → List Nat → Option (List Int × Bits)
  | _, out, bm, [] => some (out, bm)
  | i, out, bm, k :: ks =>
    match entry k with
    | some ⟨.int, p⟩ => getIntLoop (i + 1) (out ++ [p]) bm ks
    | none | some ⟨.null, _⟩ => getIntLoop (i + 1) (out ++ [0]) (fun j => j == i || bm j) ks
    | some _ => none

inductive Column where
  | ints (xs : List Int)
  | floats (xs : List F)
  | values (vs : List (V F))

def toIntN : V F → Option Int
  | .int i => some i
  | .null => some 0
  | _ => none

def toFloatN : V F → Option F
  | .float f => some f
  | .null => some (.num 0)
  | _ => none

def isNullB : V F → Bool
  | .null => true
  | _ => false

/-- `classify_exact_column`: the bitmap of `from_values` and `classify_numeric(.., true, Pure)`. -/
def classifyExact (vs : List (V F)) : Column × Bits :=
  let bm : Bits := fun j => match vs[j]? with | some v => isNullB v | none => false
  match vs.mapM toIntN with
  | some xs => (.ints xs, bm)
  | none => match vs.mapM toFloatN with
    | some fs => (.floats fs, bm)
    | none => (.values vs, bm)

/-- `materialize_node_property_values`. -/
def matValues (attrIdx : Option Nat) (deletedEmpty noPendingAttrs : Bool)
    (slow : List Nat → List (V F)) (keys : List Nat) : List (V F) :=
  if deletedEmpty && noPendingAttrs then
    match attrIdx with
    | some _ => batchValues entry unpackOther keys
    | none => List.replicate keys.length .null
  else slow keys

/-- `materialize_node_property_exact`. -/
def exactPR (attrIdx : Option Nat) (deletedEmpty noPendingAttrs : Bool)
    (slow : List Nat → List (V F)) (keys : List Nat) : Column × Bits :=
  if attrIdx.isSome && deletedEmpty && noPendingAttrs then
    match getIntLoop entry 0 [] (fun _ => false) keys with
    | some (ints, bm) => (.ints ints, bm)
    | none => classifyExact (matValues entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys)
  else classifyExact (matValues entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys)

/-- Loop invariant: what the integer loop returns, element by element. -/
theorem getIntLoop_spec : ∀ (ks : List Nat) (i : Nat) (out : List Int) (bm : Bits) (out' : List Int) (bm' : Bits),
    getIntLoop entry i out bm ks = some (out', bm') →
      (∃ xs, (batchValues entry unpackOther ks).mapM toIntN = some xs ∧ out' = out ++ xs) ∧
      ∀ j, bm' j = (bm j || (decide (i ≤ j) && match (batchValues entry unpackOther ks)[j - i]? with
        | some v => isNullB v | none => false))
  | [], i, out, bm, out', bm', h => by
    simp only [getIntLoop, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨⟨[], by simp [batchValues, pure], by simp⟩, fun j => ?_⟩
    simp [batchValues]
  | k :: ks, i, out, bm, out', bm', h => by
    simp only [getIntLoop] at h
    cases he : entry k with
    | none =>
      rw [he] at h
      obtain ⟨⟨xs, hxs, hout⟩, hbm⟩ := getIntLoop_spec ks (i + 1) _ _ out' bm' h
      refine ⟨⟨0 :: xs, ?_, by simp [hout]⟩, fun j => ?_⟩
      · simp [batchValues, he, toIntN] at hxs ⊢; simp [hxs]
      · rw [hbm j]
        by_cases hj : j = i
        · rw [hj]; have h1 : ¬ (i + 1 ≤ i) := by omega
          simp [batchValues, he, isNullB, h1]
        · by_cases hij : i + 1 ≤ j
          · have : j - i = (j - (i + 1)) + 1 := by omega
            have hb : (j == i) = false := by simpa using hj
            simp [batchValues, this, hb, show i ≤ j by omega, hij]
          · have : ¬ i ≤ j := by omega
            have hb : (j == i) = false := by simpa using hj
            simp [hb, this, hij]
    | some p =>
      rw [he] at h
      rcases p with ⟨tag, pl⟩
      cases tag
      case int =>
        simp only at h
        obtain ⟨⟨xs, hxs, hout⟩, hbm⟩ := getIntLoop_spec ks (i + 1) _ _ out' bm' h
        refine ⟨⟨pl :: xs, ?_, by simp [hout]⟩, fun j => ?_⟩
        · simp [batchValues, he, unpack, toIntN] at hxs ⊢; simp [hxs]
        · rw [hbm j]
          by_cases hj : j = i
          · rw [hj]; have h1 : ¬ (i + 1 ≤ i) := by omega
            simp [batchValues, he, unpack, isNullB, h1]
          · by_cases hij : i + 1 ≤ j
            · have : j - i = (j - (i + 1)) + 1 := by omega
              simp [batchValues, this, show i ≤ j by omega, hij]
            · have : ¬ i ≤ j := by omega
              simp [this, hij]
      case null =>
        simp only at h
        obtain ⟨⟨xs, hxs, hout⟩, hbm⟩ := getIntLoop_spec ks (i + 1) _ _ out' bm' h
        refine ⟨⟨0 :: xs, ?_, by simp [hout]⟩, fun j => ?_⟩
        · simp [batchValues, he, unpack, toIntN] at hxs ⊢; simp [hxs]
        · rw [hbm j]
          by_cases hj : j = i
          · rw [hj]; have h1 : ¬ (i + 1 ≤ i) := by omega
            simp [batchValues, he, unpack, isNullB, h1]
          · by_cases hij : i + 1 ≤ j
            · have : j - i = (j - (i + 1)) + 1 := by omega
              have hb : (j == i) = false := by simpa using hj
              simp [batchValues, this, hb, show i ≤ j by omega, hij]
            · have : ¬ i ≤ j := by omega
              have hb : (j == i) = false := by simpa using hj
              simp [hb, this, hij]
      all_goals simp at h

/-- **Claim 7.** The PR's typed fetch is main's `classify_exact_column(materialize_..)`,
column and null bits alike, in every case. -/
theorem exact_eq_main (attrIdx : Option Nat) (deletedEmpty noPendingAttrs : Bool)
    (slow : List Nat → List (V F)) (keys : List Nat) :
    exactPR entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys =
      classifyExact (matValues entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys) := by
  unfold exactPR
  split
  · rename_i hg
    cases hl : getIntLoop entry 0 [] (fun _ => false) keys with
    | none => rfl
    | some r =>
      obtain ⟨ints, bm⟩ := r
      simp only
      obtain ⟨⟨xs, hxs, hout⟩, hbm⟩ := getIntLoop_spec entry unpackOther keys 0 [] _ ints bm hl
      simp only [Bool.and_eq_true, Option.isSome_iff_exists] at hg
      obtain ⟨⟨⟨idx, hidx⟩, hd⟩, hp⟩ := hg
      have hmv : matValues entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys =
          batchValues entry unpackOther keys := by
        simp [matValues, hd, hp, hidx]
      rw [hmv]; simp only [classifyExact, hxs]
      simp only [List.nil_append] at hout; subst hout
      congr 1; funext j; rw [hbm j]; simp
  · rfl

/-- Main's column for an int/null property under the read-only path is the same `Ints`
lane, so an `eval_property` over a `NodeIds` column is unchanged by the PR. -/
def evalPropPR (attrIdx : Option Nat) (deletedEmpty noPendingAttrs : Bool)
    (slow : List Nat → List (V F)) (keys : List Nat) : Column × Bits :=
  exactPR entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys

def evalPropMain (attrIdx : Option Nat) (deletedEmpty noPendingAttrs : Bool)
    (slow : List Nat → List (V F)) (keys : List Nat) : Column × Bits :=
  classifyExact (matValues entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys)

theorem evalProp_eq (attrIdx : Option Nat) (deletedEmpty noPendingAttrs : Bool)
    (slow : List Nat → List (V F)) (keys : List Nat) :
    evalPropPR entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys =
      evalPropMain entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys :=
  exact_eq_main entry unpackOther attrIdx deletedEmpty noPendingAttrs slow keys

/-- Any non-int, non-null stored value makes the fast loop give up (and the caller
take main's path). -/
theorem getIntLoop_other (i : Nat) (out : List Int) (bm : Bits) (k : Nat) (ks : List Nat) (p : Packed)
    (he : entry k = some p) (ht : p.tag ≠ .int) (hn : p.tag ≠ .null) :
    getIntLoop entry i out bm (k :: ks) = none := by
  rcases p with ⟨tag, pl⟩
  simp only [getIntLoop, he]
  cases tag <;> simp_all

end Pr3437.ExactProp
