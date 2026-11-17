import GraphQueries.Ids
/-
# The schema hash (#3757): `graph/src/graph/schema_hash.rs` and its graph.rs writers

| here | there |
| --- | --- |
| `xxh` (parameter) | `xxhash_rust::xxh3::xxh3_64` (seed 0; an abstract function, like `format_g`) |
| `le64`, `nameBytes` | `u64::to_le_bytes`, `str::as_bytes` (UTF-8) |
| `push`            | `NameChain::<TAG>::push`, schema_hash.rs:47-58 |
| `chainOf`         | `NameChain::<TAG>::of`, :61-67 |
| `schemaHash`      | `schema_hash`, :72-78 (`wrapping_add` twice = sum mod 2^64) |
| `GC`, `Inv`       | `Graph`'s `label_chain`/`type_chain` fields (graph.rs:491-494) |
| `newGC`           | `Graph::new` (:882), chains `default()` = 0 |
| `restoreGC`       | `Graph::restore` (:929), chains rebuilt with `of` (:1023-1026) |
| `newVersionGC`    | `new_version` (:1105), chains copied (:1146-1147) |
| `internLabelC`    | `intern_label` (:1279-1292): `label_chain.push` with the list push |
| `registerRelTypeC`| `register_relationship_type` (:1403-1415): `type_chain.push` with the list push |
| `graphSchemaHash` | `Graph::schema_hash` (:1273-1275), key chain from `AttrNameMap::chain` |

Headlines: `chainOf_snoc` (`push_matches_of`: pushing names one by one = `of` the list);
every writer keeps `Inv` (each chain is `chainOf` of its list): `newGC_inv`,
`restoreGC_inv`, `newVersionGC_inv`, `internLabelC_inv`, `registerRelTypeC_inv`; each
chained writer does to the graph exactly what the unchained model does (`*_model`), so
every existing theorem about `internLabel`/`registerRelType`/`newVersion` still applies.
`graphSchemaHash_eq`: under `Inv` and the key-chain invariant (`PendingCommit.NameMapChain`),
`schema_hash()` is `listsHash` of the three name lists, so a restart (restore rebuilds the
chains) gives the same value as the writers built (`restore_same_hash`); `empty_hash`: an
empty schema hashes to 0.
-/
namespace GQ.SchemaHash
open GQ

variable {V : Type} (xxh : List Nat → Nat)

/-- `u64::to_le_bytes` (8 bytes). -/
def le64 (x : Nat) : List Nat := (List.range 8).map fun i => x / 256 ^ i % 256

/-- `str::as_bytes`. -/
def nameBytes (s : String) : List Nat := s.toUTF8.data.toList.map UInt8.toNat

/-- `NameChain::<TAG>::push` (schema_hash.rs:47-58): `xxh3_64(tag ‖ h ‖ len ‖ name)`. -/
def push (tag : Nat) (c : Nat) (s : String) : Nat :=
  xxh ([tag] ++ le64 c ++ le64 (nameBytes s).length ++ nameBytes s)

/-- `NameChain::<TAG>::of` (:61-67): `default()` (0), then `push` each name. -/
def chainOf (tag : Nat) (l : List String) : Nat := l.foldl (push xxh tag) 0

/-- `schema_hash` (:72-78). -/
def schemaHash (lc tc kc : Nat) : Nat := (lc + tc + kc) % 2 ^ 64

/-- `S` as a function of the three lists only (the module-doc definition). -/
def listsHash (labels types keys : List String) : Nat :=
  schemaHash (chainOf xxh 0 labels) (chainOf xxh 1 types) (chainOf xxh 2 keys)

/-- `push_matches_of`: extending a chain by a name is the chain of the extended list. -/
theorem chainOf_snoc (tag : Nat) (l : List String) (s : String) :
    push xxh tag (chainOf xxh tag l) s = chainOf xxh tag (l ++ [s]) := by
  simp [chainOf, List.foldl_append]

theorem chainOf_nil (tag : Nat) : chainOf xxh tag [] = 0 := rfl

/-- An empty schema hashes to 0 (`golden_values`' first case). -/
theorem empty_hash : listsHash xxh [] [] [] = 0 := by
  simp [listsHash, chainOf_nil, schemaHash]

/-! ## The chains on `Graph` -/

/-- `Graph` with its two chain fields. -/
structure GC (V : Type) where
  g : G V
  lc : Nat
  tc : Nat

/-- Each chain is the chain of its list. -/
def Inv (gc : GC V) : Prop := gc.lc = chainOf xxh 0 gc.g.labels ∧ gc.tc = chainOf xxh 1 gc.g.types

/-- `Graph::new` (:882; chains at :910-911). -/
def newGC (n e version : Nat) (name : String) : GC V := ⟨newG n e version name, 0, 0⟩

theorem newGC_inv (n e v : Nat) (name : String) : Inv xxh (newGC (V := V) n e v name) := by
  simp [Inv, newGC, newG, chainOf_nil]

/-- `Graph::restore` (:929): whatever graph it builds from the RDB's lists, the chains are
`of` those lists (:1023-1026). -/
def restoreGC (g : G V) : GC V := ⟨g, chainOf xxh 0 g.labels, chainOf xxh 1 g.types⟩

theorem restoreGC_inv (g : G V) : Inv xxh (restoreGC xxh g) := ⟨rfl, rfl⟩

/-- `new_version` (:1105): chains copied (:1146-1147). -/
def newVersionGC (gc : GC V) : GC V := { gc with g := newVersion gc.g }

theorem newVersionGC_model (gc : GC V) : (newVersionGC gc).g = newVersion gc.g := rfl

theorem newVersionGC_inv (gc : GC V) (h : Inv xxh gc) : Inv xxh (newVersionGC gc) := by
  simpa [Inv, newVersionGC, newVersion, setNodeIds, setRelIds] using h

/-- `intern_label` (:1279-1292): a new label pushes the chain, then the list. -/
def internLabelC (gc : GC V) (s : String) : GC V × Nat :=
  match gc.g.labelsIndex s with
  | some id => (gc, id)
  | none => let r := internLabel gc.g s; (⟨r.1, push xxh 0 gc.lc s, gc.tc⟩, r.2)

theorem internLabelC_model (gc : GC V) (s : String) :
    (internLabelC xxh gc s).1.g = (internLabel gc.g s).1 ∧
      (internLabelC xxh gc s).2 = (internLabel gc.g s).2 := by
  unfold internLabelC internLabel; split <;> simp_all

theorem internLabelC_inv (gc : GC V) (s : String) (h : Inv xxh gc) :
    Inv xxh (internLabelC xxh gc s).1 := by
  unfold internLabelC; split
  · exact h
  · rename_i hs
    refine ⟨?_, ?_⟩
    · simp only [internLabel, hs, h.1]
      exact chainOf_snoc xxh 0 _ s
    · simp only [internLabel, hs]; exact h.2

/-- `register_relationship_type` (:1403-1415): push the chain, then the list. -/
def registerRelTypeC (chunk : Nat) (hc : 0 < chunk) (gc : GC V) (s : String) : GC V × Nat :=
  let r := registerRelType chunk hc gc.g s
  (⟨r.1, gc.lc, push xxh 1 gc.tc s⟩, r.2)

theorem registerRelTypeC_model (chunk : Nat) (hc : 0 < chunk) (gc : GC V) (s : String) :
    (registerRelTypeC xxh chunk hc gc s).1.g = (registerRelType chunk hc gc.g s).1 ∧
      (registerRelTypeC xxh chunk hc gc s).2 = (registerRelType chunk hc gc.g s).2 := ⟨rfl, rfl⟩

theorem registerRelTypeC_inv (chunk : Nat) (hc : 0 < chunk) (gc : GC V) (s : String)
    (h : Inv xxh gc) : Inv xxh (registerRelTypeC xxh chunk hc gc s).1 := by
  obtain ⟨k1, -, -, k4, -⟩ := resize_keeps chunk hc
    ({ gc.g with types := gc.g.types ++ [s],
                 relMs := vecInsert gc.g.relMs ((gc.g.types ++ [s]).length - 1) [] } : G V)
  refine ⟨?_, ?_⟩
  · simp only [registerRelTypeC, registerRelType, k1]; exact h.1
  · simp only [registerRelTypeC, registerRelType, k4, h.2]
    exact chainOf_snoc xxh 1 _ s

/-- `Graph::schema_hash` (:1273-1275); `kc` is `attrs_name.chain()`. -/
def graphSchemaHash (gc : GC V) (kc : Nat) : Nat := schemaHash gc.lc gc.tc kc

/-- **`schema_hash()` depends on the three lists only**: under `Inv` and the key chain's
invariant (`kc = chainOf 2 attrs`, `PendingCommit.NameMapChain.Inv`). -/
theorem graphSchemaHash_eq (gc : GC V) (kc : Nat) (h : Inv xxh gc)
    (hk : kc = chainOf xxh 2 gc.g.attrs) :
    graphSchemaHash gc kc = listsHash xxh gc.g.labels gc.g.types gc.g.attrs := by
  simp [graphSchemaHash, listsHash, h.1, h.2, hk]

/-- A restart (RDB load rebuilds the chains) gives the value the writers built. -/
theorem restore_same_hash (gc : GC V) (h : Inv xxh gc) :
    graphSchemaHash (restoreGC xxh gc.g) (chainOf xxh 2 gc.g.attrs) =
      graphSchemaHash gc (chainOf xxh 2 gc.g.attrs) := by
  simp [graphSchemaHash, restoreGC, h.1, h.2]

/-- The hash is a 64-bit value, as `u64` is. -/
theorem schemaHash_lt (lc tc kc : Nat) : schemaHash lc tc kc < 2 ^ 64 := by
  unfold schemaHash; exact Nat.mod_lt _ (by decide)

end GQ.SchemaHash
