/-!
# `schema::Key`: a name as a map key, looked up by `&str` (schema.rs:21-39)

```rust
#[derive(Clone, PartialEq, Eq)]
struct Key(Arc<String>);                                   // :23
impl Hash for Key { fn hash(&self, state) { self.0.as_str().hash(state); } }   // :27
impl Borrow<str> for Key { fn borrow(&self) -> &str { &self.0 } }              // :36
```

`NameList.ids` is a `FxHashMap<Key, usize>`; a lookup passes a `&str` (`ids.get(name)`), which is
sound only under std's `Borrow` contract: the borrowed form compares and hashes like the owned one.
`NameMap.lean` models the map as a function keyed by the name itself (`M.index : String → Option Nat`);
this file proves the contract that licenses that, for any hasher on `str`.

| here | there |
| --- | --- |
| `Key.s` | `Key.0.as_str()` |
| `Key.hash` | `Hash for Key` (:27): the hash of the `str` |
| `Key.borrow` | `Borrow<str> for Key` (:36) |
| `Key.eq` (`derive(PartialEq, Eq)`) | `Arc<String>: PartialEq` compares the strings |
| `lookup` | `HashMap<Key, usize>::get(&str)` as a lookup among the entries by `Borrow`ed equality |

* `hash_eq_str_hash` — a key hashes as the `str` it borrows as (what `HashMap::get(&str)` hashes).
* `borrow_eq_iff` — two keys are equal iff their borrowed `str`s are.
* `borrow_contract` — equal borrowed forms ⇒ equal keys ⇒ equal hashes: the `Borrow` requirement.
* `lookup_insert` — a `get(&str)` finds the entry inserted under the key whose `str` it is, and
  `lookup_other` — no other entry (so `NameMap.M.index` keyed by the name is exact).
-/
namespace PendingCommit.KeyBorrow

/-- `Key(Arc<String>)`: the string inside. -/
structure Key where
  s : String
  deriving DecidableEq

variable {H : Type}

/-- `Borrow<str> for Key` (:36). -/
def Key.borrow (k : Key) : String := k.s

/-- `Hash for Key` (:27): `self.0.as_str().hash(state)`, for any hasher on `str`. -/
def Key.hash (hs : String → H) (k : Key) : H := hs k.s

/-- `derive(PartialEq)`: contents compare. -/
def Key.eq (a b : Key) : Bool := a.s == b.s

theorem hash_eq_str_hash (hs : String → H) (k : Key) : k.hash hs = hs k.borrow := rfl

theorem borrow_eq_iff (a b : Key) : a.borrow = b.borrow ↔ a = b := by
  constructor
  · intro h; cases a; cases b; simp only [Key.borrow] at h; rw [h]
  · rintro rfl; rfl

theorem eq_iff (a b : Key) : a.eq b = true ↔ a = b := by
  unfold Key.eq; rw [beq_iff_eq]; exact (borrow_eq_iff a b).symm.symm

/-- The `Borrow` requirement: `x.borrow() == y.borrow()` implies `x == y` and the same hash. -/
theorem borrow_contract (hs : String → H) (a b : Key) (h : a.borrow = b.borrow) :
    a = b ∧ a.hash hs = b.hash hs := by
  have hab := (borrow_eq_iff a b).1 h
  exact ⟨hab, by rw [hab]⟩

/-- `HashMap<Key, usize>`, as its entries; `get(&str)` finds the entry whose key borrows as it. -/
def lookup (m : List (Key × Nat)) (n : String) : Option Nat :=
  (m.find? (fun p => p.1.borrow == n)).map (·.2)

theorem lookup_insert (m : List (Key × Nat)) (n : String) (i : Nat) :
    lookup ((⟨n⟩, i) :: m) n = some i := by
  simp [lookup, Key.borrow]

theorem lookup_other (m : List (Key × Nat)) (n n' : String) (i : Nat) (h : n' ≠ n) :
    lookup ((⟨n⟩, i) :: m) n' = lookup m n' := by
  simp [lookup, Key.borrow, List.find?_cons, h.symm, beq_iff_eq]

end PendingCommit.KeyBorrow
