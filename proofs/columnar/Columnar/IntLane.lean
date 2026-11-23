import Columnar.RuntimeReads

/-!
# The integer-lane property fetch is `classify_exact_column` of the `Value` fetch (#3437)

`materialize_node_property_exact` (runtime.rs), which `eval_property` (vector_expr.rs) calls on a
node column, reads an attribute's packed entries straight into an `i64` lane on the read-only
path (`Graph::get_node_int_attributes_by_idx` → `AttributeStore::get_int_attrs_by_idx_batch_into`)
and otherwise falls back to `classify_exact_column(materialize_node_property_values(..))`.

| here | there |
| --- | --- |
| `Packed` | a `PackedAttr` of the attribute store (attribute_store.rs): `Tag::Int` with its payload, `Tag::Null`, or any other tag (`Bool`, `Float`, `Point`, temporal, `Heap`) as the value `unpack` gives |
| `Packed.unpack` | `Block::unpack` (attribute_store.rs) |
| `look id i` | `data.get(key).and_then(|span| binary_search_by_key(i).map(entries[pos]))`, the entry both `SpanRef::get` and `get_int_attrs_by_idx_batch_into` read; `get_attr_by_idx` is `(look id i).map unpack` (`hlook`) |
| `intBatch` | `get_int_attrs_by_idx_batch_into` (and its forward `get_node_int_attributes_by_idx`, graph.rs): `some (ints, flags)` with `flags[i]` true iff `on_null(i)` was called (the `NullBitmap::none(len)` the caller sets at those `i`), `none` for `return false` |
| `matNodeExact` | `materialize_node_property_exact` (runtime.rs) |

**`matNodeExact_eq`**: on every state, the result is `classify_exact_column` of
`materialize_node_property_values` — so `CoreBridge.evalProperty` (which reads it that way)
still models `eval_property`.
-/

namespace Columnar.IntLane
open FalkorCore (V)
open Columnar (FloatModel Column Bits classifyExact classifyNumeric intsOf)
open Columnar.RuntimeReads (St matNode hasNodeAttrs byIdxN)

variable {F : Type}

/-- A packed attribute entry. -/
inductive Packed (F : Type) where
  | int (i : Int)
  | null
  | other (v : V F)

/-- `Block::unpack` (attribute_store.rs). -/
def Packed.unpack : Packed F → V F
  | .int i => .int i
  | .null => .null
  | .other v => v

/-- `get_int_attrs_by_idx_batch_into` (attribute_store.rs), one entry per key. -/
def intBatch : List (Option (Packed F)) → Option (List Int × Bits)
  | [] => some ([], [])
  | some (.int i) :: r => (intBatch r).map fun p => (i :: p.1, false :: p.2)
  | none :: r => (intBatch r).map fun p => (0 :: p.1, true :: p.2)
  | some .null :: r => (intBatch r).map fun p => (0 :: p.1, true :: p.2)
  | some (.other _) :: _ => none

/-- The value `get_attrs_by_idx_batch_into` pushes for an entry (`Null` as default). -/
def readE (e : Option (Packed F)) : V F := (e.map Packed.unpack).getD .null

/-- An integer-lane batch is the `classify_numeric` int lane and the null bitmap of the values. -/
theorem intBatch_sound : ∀ {es : List (Option (Packed F))} {is : List Int} {ns : Bits},
    intBatch es = some (is, ns) →
      intsOf true (es.map readE) = some is ∧ Bits.ofValues (es.map readE) = ns
  | [], is, ns, h => by
    simp [intBatch] at h; obtain ⟨rfl, rfl⟩ := h; simp [intsOf, Bits.ofValues]
  | e :: es, is, ns, h => by
    match e, h with
    | some (.int i), h =>
      simp only [intBatch, Option.map_eq_some_iff] at h
      obtain ⟨⟨is', ns'⟩, h', he⟩ := h
      simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
      obtain ⟨h1, h2⟩ := intBatch_sound h'
      simp only [List.map_cons]
      refine ⟨?_, ?_⟩
      · simp [intsOf, readE, Packed.unpack, h1]
      · simp [Bits.ofValues, readE, Packed.unpack, V.isNull, ← h2]
    | none, h =>
      simp only [intBatch, Option.map_eq_some_iff] at h
      obtain ⟨⟨is', ns'⟩, h', he⟩ := h
      simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
      obtain ⟨h1, h2⟩ := intBatch_sound h'
      simp only [List.map_cons]
      refine ⟨?_, ?_⟩
      · simp [intsOf, readE, h1]
      · simp [Bits.ofValues, readE, V.isNull, ← h2]
    | some .null, h =>
      simp only [intBatch, Option.map_eq_some_iff] at h
      obtain ⟨⟨is', ns'⟩, h', he⟩ := h
      simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
      obtain ⟨h1, h2⟩ := intBatch_sound h'
      simp only [List.map_cons]
      refine ⟨?_, ?_⟩
      · simp [intsOf, readE, Packed.unpack, h1]
      · simp [Bits.ofValues, readE, Packed.unpack, V.isNull, ← h2]
    | some (.other _), h => simp [intBatch] at h

/-- `materialize_node_property_exact` (runtime.rs). -/
def matNodeExact [FloatModel F] (s : St F) (look : Nat → Nat → Option (Packed F))
    (ids : List Nat) (a : String) : Column F × Bits :=
  match (if s.delNodes.isEmpty && !hasNodeAttrs s then s.attrId a else none) with
  | some i =>
    match intBatch (ids.map fun id => look id i) with
    | some (is, ns) => (.ints is, ns)
    | none => classifyExact (matNode s ids a)
  | none => classifyExact (matNode s ids a)

/-- **The integer lane changes nothing**: `materialize_node_property_exact` is
`classify_exact_column(materialize_node_property_values(..))` on every state. -/
theorem matNodeExact_eq [FloatModel F] (s : St F) (look : Nat → Nat → Option (Packed F))
    (hlook : ∀ id i, byIdxN s id i = (look id i).map Packed.unpack) (ids : List Nat) (a : String) :
    matNodeExact s look ids a = classifyExact (matNode s ids a) := by
  unfold matNodeExact
  split
  · rename_i i hi
    split
    · rename_i is ns hb
      have hfast : (s.delNodes.isEmpty && !hasNodeAttrs s) = true := by
        cases hc : (s.delNodes.isEmpty && !hasNodeAttrs s) <;> simp_all
      have hai : s.attrId a = some i := by simpa [hfast] using hi
      have hm : matNode s ids a = (ids.map fun id => look id i).map readE := by
        simp only [matNode, hfast, ite_true, hai, List.map_map]
        apply List.map_congr_left; intro id _
        simp [hlook, readE]
      obtain ⟨h1, h2⟩ := intBatch_sound hb
      rw [hm]; simp only [classifyExact, classifyNumeric, h1, h2]
    · rfl
  · rfl

end Columnar.IntLane
