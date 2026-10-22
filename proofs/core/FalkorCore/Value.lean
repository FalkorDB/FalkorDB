/-!
# The engine `Value` (`graph/src/runtime/value.rs:180`)

`V F` is `enum Value` with the `f64` payload abstracted to a type `F` (instantiate it with
`FalkorCore.F`, `Float`, or a record of operations); `Arc`s are transparent. `V.order` is
`OrderedEnum::order` (value.rs:1178), `V.name` is `Value::name` (value.rs:1352).
From proofs/value_math (pilot of the shared core).
-/
namespace FalkorCore

inductive V (F : Type) where
  | null
  | bool (b : Bool)
  | int (i : Int)
  | float (f : F)
  | str (s : String)
  | list (l : List (V F))
  | map (m : List (String × V F))
  | node (id : Nat)
  | rel (id : Nat)
  | path (l : List (V F))
  | vec (v : List F)
  | point (lat lon : F)
  | datetime (t : Int)
  | date (t : Int)
  | time (t : Int)
  | duration (t : Int)
  deriving Inhabited

variable {F : Type}

/-- `Value::name` (value.rs:1387; C's `SIType_ToString` spellings since #2954). -/
def V.name : V F → String
  | .null => "Null" | .bool _ => "Boolean" | .int _ => "Integer" | .float _ => "Float"
  | .str _ => "String" | .list _ => "List" | .map _ => "Map" | .node _ => "Node"
  | .rel _ => "Edge" | .path _ => "Path" | .vec _ => "Vectorf32" | .point .. => "Point"
  | .datetime _ => "Datetime" | .date _ => "Date" | .time _ => "Time"
  | .duration _ => "Duration"

/-- `OrderedEnum::order` (value.rs:1178). -/
def V.order : V F → Nat
  | .null => 2 ^ 15 | .bool _ => 2 ^ 12 | .int _ => 2 ^ 13 | .float _ => 2 ^ 14
  | .str _ => 2 ^ 11 | .list _ => 2 ^ 3 | .map _ => 2 ^ 0 | .node _ => 2 ^ 1
  | .rel _ => 2 ^ 2 | .path _ => 2 ^ 4 | .point .. => 2 ^ 5 | .datetime _ => 2 ^ 6
  | .date _ => 2 ^ 7 | .time _ => 2 ^ 8 | .duration _ => 2 ^ 10 | .vec _ => 2 ^ 18

def V.isNull : V F → Bool
  | .null => true
  | _ => false

def V.isNum : V F → Bool
  | .int _ | .float _ => true
  | _ => false

def V.isList : V F → Bool
  | .list _ => true
  | _ => false

end FalkorCore
