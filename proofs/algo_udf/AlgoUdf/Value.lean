/-! Shared, cut-down model of `runtime::value::Value` (`graph/src/runtime/value.rs`)
restricted to the variants the algo.* argument parsers inspect. -/
namespace AlgoUdf

inductive Val where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (f : Float)
  | str (s : String)
  | list (xs : List Val)
  | map (kv : List (String × Val))
  | node (id : Nat)
  deriving Inhabited

/-- `OrderMap::get` on a map with unique keys: first match. -/
def lookup (m : List (String × Val)) (k : String) : Option Val :=
  (m.find? (·.1 == k)).map (·.2)

/-- Rust `i64 as i32` (two's complement truncation), on the mathematical integer. -/
def asI32 (n : Int) : Int :=
  let r := n % 4294967296
  if r ≥ 2147483648 then r - 4294967296 else r

/-- Rust `i64 as u32` for a non-negative i64. -/
def asU32 (n : Int) : Int := n % 4294967296

end AlgoUdf
