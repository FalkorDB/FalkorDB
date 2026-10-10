/-
# `serialization.rs`: `EncodeState::from_u64`, the `Encode`/`Decode` trait contract

| here | there (`graph/src/graph/graphblas/serialization.rs`) |
| --- | --- |
| `EncodeState`/`disc` | `enum EncodeState` (:130), `#[repr(u64)]` discriminants |
| `fromU64`     | `from_u64` (:146) |
| `Codec`       | `trait Encode::encode` (:34) + `trait Decode::decode` (:64): the contract every impl owes |
| `encodeWithRangeDefault` / `decodeWithCountDefault` | the `unimplemented!()` defaults (:40 / :76) |

`Codec` is the obligation the two required trait methods carry: decoding
what was encoded returns the value and the rest of the stream. Instances are
proved in their own projects (e.g. `VMOps.decode_encode` for
`VersionedMatrix`, `graph_persist` for the roaring set).
-/
namespace GBWSerial

inductive EncodeState
  | init | nodes | deletedNodes | edges | deletedEdges | graphSchema | labelsMatrices
  | relationMatrices | adjMatrix | lblsMatrix | final
  deriving DecidableEq, Repr

def disc : EncodeState → Nat
  | .init => 0 | .nodes => 1 | .deletedNodes => 2 | .edges => 3 | .deletedEdges => 4
  | .graphSchema => 5 | .labelsMatrices => 6 | .relationMatrices => 7 | .adjMatrix => 8
  | .lblsMatrix => 9 | .final => 10

def fromU64 (v : Nat) : Option EncodeState :=
  match v with
  | 0 => some .init | 1 => some .nodes | 2 => some .deletedNodes | 3 => some .edges
  | 4 => some .deletedEdges | 5 => some .graphSchema | 6 => some .labelsMatrices
  | 7 => some .relationMatrices | 8 => some .adjMatrix | 9 => some .lblsMatrix
  | 10 => some .final | _ => none

/-- `from_u64` inverts the `repr(u64)` discriminant. -/
theorem fromU64_disc (s : EncodeState) : fromU64 (disc s) = some s := by cases s <;> rfl

/-- …and accepts nothing else: every tag it accepts is a discriminant. -/
theorem fromU64_some {v : Nat} {s : EncodeState} (h : fromU64 v = some s) : disc s = v := by
  unfold fromU64 at h
  split at h <;> (try simp at h) <;> (subst h; rfl)

theorem fromU64_none {v : Nat} (h : 10 < v) : fromU64 v = none := by
  unfold fromU64; split <;> first | rfl | omega

/-- The contract of the required methods `encode` (:34) and `decode` (:64). -/
structure Codec (α τ : Type) where
  encode : α → List τ
  decode : List τ → Except String (α × List τ)
  roundtrip : ∀ a rest, decode (encode a ++ rest) = .ok (a, rest)

/-- Any codec reads back a sequence of its own encodings, in order. -/
theorem Codec.roundtrip2 {α τ : Type} (c : Codec α τ) (a b : α) (rest : List τ) :
    (c.decode (c.encode a ++ (c.encode b ++ rest))) = .ok (a, c.encode b ++ rest) ∧
    c.decode (c.encode b ++ rest) = .ok (b, rest) := ⟨c.roundtrip _ _, c.roundtrip _ _⟩

/-- The `unimplemented!()` defaults (:40, :76): a call panics (`none`), it never
silently writes or reads a partial stream. -/
def encodeWithRangeDefault {τ : Type} (_count _offset : Nat) : Option (List τ) := none
def decodeWithCountDefault (_count _attrLimit : Nat) : Option Unit := none
theorem defaults_panic {τ : Type} (c o l : Nat) :
    (encodeWithRangeDefault (τ := τ) c o) = none ∧ decodeWithCountDefault c l = none := ⟨rfl, rfl⟩

end GBWSerial
