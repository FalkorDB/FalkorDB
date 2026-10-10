import FalkorValueMath.TypeCheck
/-!
# `graph/src/runtime/functions/trig.rs` (origin/main 3fec7d7c9)

`f64` stays abstract (`F`); the libm functions and the three IEEE operations the file
uses (`*`, `/`, `-`) are fields of `TrigOps`. What is proved is everything the Rust adds
around them: argument dispatch, `Int → f64` promotion, `Null` propagation, the exact
formula each Cypher function evaluates (`cot = cos/sin`, `haversin = (1-cos)/2`,
`degrees = x·(180/π)`, `radians = x·(π/180)` — the last two are Rust std's own
`to_degrees`/`to_radians`, core/src/num/f64.rs:871-895), that every `unreachable!()` is
dead after `validate_args_type`, that the result type is the declared return type, and
that each function is **the same expression tree as C** (`AR_SIN` … `AR_HAVERSIN`,
`src/arithmetic/numeric_funcs/numeric_funcs.c:250-392` on master), so Rust = C
bit-for-bit given the same libm. Live (ports 18920/18921): 25 calls incl. `cot(0)=inf`,
`asin(2)=nan`, `atan2(0,-1)=π`, null/arity/type errors — identical.

| Lean | Rust (trig.rs) |
| --- | --- |
| `applyUnary` | `apply_unary_float` :31 |
| `tSin … tHaversin`, `tAtan2`, `tPi` | the `cypher_fn!` bodies :47 :53 :59 :65 :73 :79 :85 :94 :109 :115 :121 :130 |
| `trigSigs` | `register` :43 (names, `args:`, `ret:`) |
| `cUnary`, `cAtan2` | C `AR_*` (null check, `SI_GET_NUMERIC`) |
-/

namespace ValueMath.Trig

open ValueMath

variable {F : Type}

/-- libm + the IEEE operations trig.rs uses. `lit n` is the `f64` literal `n.0`. -/
structure TrigOps (F : Type) where
  ofInt : Int → F               -- `n as f64` (round to nearest; C `(double)n` is the same)
  sin : F → F
  cos : F → F
  tan : F → F
  asin : F → F
  acos : F → F
  atan : F → F
  atan2 : F → F → F
  mul : F → F → F
  div : F → F → F
  sub : F → F → F
  lit : Nat → F
  pi : F                        -- `std::f64::consts::PI` = C `M_PI` (same double)

variable (T : TrigOps F)

/-- `apply_unary_float` (:31). `none` = panic (`unreachable!()` or `args[0]` on `[]`). -/
def applyUnary (f : F → F) : List (V F) → Option (V F)
  | .int n :: _ => some (.float (f (T.ofInt n)))
  | .float v :: _ => some (.float (f v))
  | .null :: _ => some .null
  | _ => none

/-- The per-function `f64 → f64` maps, exactly as written in trig.rs. -/
def fCot (x : F) : F := T.div (T.cos x) (T.sin x)                          -- :66
def fDegrees (x : F) : F := T.mul x (T.div (T.lit 180) T.pi)               -- f64::to_degrees
def fRadians (x : F) : F := T.mul x (T.div T.pi (T.lit 180))               -- f64::to_radians
def fHaversin (x : F) : F := T.div (T.sub (T.lit 1) (T.cos x)) (T.lit 2)   -- :131

def tSin := applyUnary T T.sin
def tCos := applyUnary T T.cos
def tTan := applyUnary T T.tan
def tCot := applyUnary T (fCot T)
def tAsin := applyUnary T T.asin
def tAcos := applyUnary T T.acos
def tAtan := applyUnary T T.atan
def tDegrees := applyUnary T (fDegrees T)
def tRadians := applyUnary T (fRadians T)
def tHaversin := applyUnary T (fHaversin T)

/-- `atan2` (:94-102), arms in source order. -/
def tAtan2 : List (V F) → Option (V F)
  | .int y :: .int x :: _ => some (.float (T.atan2 (T.ofInt y) (T.ofInt x)))
  | .float y :: .float x :: _ => some (.float (T.atan2 y x))
  | .int y :: .float x :: _ => some (.float (T.atan2 (T.ofInt y) x))
  | .float y :: .int x :: _ => some (.float (T.atan2 y (T.ofInt x)))
  | .null :: _ :: _ => some .null
  | _ :: .null :: _ => some .null
  | _ => none

/-- `pi` (:121): `debug_assert!(args.is_empty())` — a panic in debug builds only. -/
def tPi (debug : Bool) (args : List (V F)) : Option (V F) :=
  if debug && !args.isEmpty then none else some (.float T.pi)

/-! ## C reference (numeric_funcs.c:250-392) -/

/-- `SI_GET_NUMERIC` on a non-null argument (`none` = undefined for non-numerics). -/
def cNum : V F → Option F
  | .int n => some (T.ofInt n)
  | .float v => some v
  | _ => none

/-- Every unary `AR_*`: `if(SIValue_IsNull(arg)) return SI_NullVal(); … SI_DoubleVal(g(value))`. -/
def cUnary (g : F → F) : List (V F) → Option (V F)
  | .null :: _ => some .null
  | v :: _ => (cNum T v).map (fun x => .float (g x))
  | [] => none

/-- `AR_ATAN2`: null if either is null, else `atan2(y, x)`. -/
def cAtan2 : List (V F) → Option (V F)
  | y :: x :: _ => if y.isNull || x.isNull then some .null else
      match cNum T y, cNum T x with
      | some a, some b => some (.float (T.atan2 a b))
      | _, _ => none
  | _ => none

/-- C's expressions: `cos(value)/sin(value)`, `value * (180/M_PI)` (int `180` → `180.0`),
`value * (M_PI / 180.0)`, `(1-cos(value))/2` (int literals promoted to double). -/
def cCot (x : F) : F := T.div (T.cos x) (T.sin x)
def cDegrees (x : F) : F := T.mul x (T.div (T.lit 180) T.pi)
def cRadians (x : F) : F := T.mul x (T.div T.pi (T.lit 180))
def cHaversin (x : F) : F := T.div (T.sub (T.lit 1) (T.cos x)) (T.lit 2)

/-! ## Registration (`register`, :43) -/

/-- `(name, args, ret)` for every `cypher_fn!` in `register`, in source order. -/
def trigSigs : List (String × List Ty × Ty) :=
  [("sin", [numArg], .union [.float, .null]), ("cos", [numArg], .union [.float, .null]),
   ("tan", [numArg], .union [.float, .null]), ("cot", [numArg], .union [.float, .null]),
   ("asin", [numArg], .union [.float, .null]), ("acos", [numArg], .union [.float, .null]),
   ("atan", [numArg], .union [.float, .null]), ("atan2", [numArg, numArg], .union [.float, .null]),
   ("degrees", [numArg], .union [.float, .null]), ("radians", [numArg], .union [.float, .null]),
   ("pi", [], .float), ("haversin", [numArg], .union [.float, .null])]

theorem trigSigs_names :
    trigSigs.map (·.1) = ["sin", "cos", "tan", "cot", "asin", "acos", "atan", "atan2",
      "degrees", "radians", "pi", "haversin"] := rfl

/-- Names are distinct (each `funcs.add` asserts absence, so `register` never panics on a
fresh registry for these keys). -/
theorem trigSigs_nodup : (trigSigs.map (·.1)).Nodup := by decide

/-- The ten unary functions, paired with the C function computing the same map. -/
def unaryPairs : List ((List (V F) → Option (V F)) × (F → F)) :=
  [(tSin T, T.sin), (tCos T, T.cos), (tTan T, T.tan), (tCot T, cCot T), (tAsin T, T.asin),
   (tAcos T, T.acos), (tAtan T, T.atan), (tDegrees T, cDegrees T), (tRadians T, cRadians T),
   (tHaversin T, cHaversin T)]

/-! ## Theorems -/

/-- The exact formulas (definitional). -/
theorem cot_formula (x : F) : tCot T [.float x] = some (.float (T.div (T.cos x) (T.sin x))) := rfl
theorem haversin_formula (x : F) :
    tHaversin T [.float x] = some (.float (T.div (T.sub (T.lit 1) (T.cos x)) (T.lit 2))) := rfl
theorem degrees_formula (x : F) :
    tDegrees T [.float x] = some (.float (T.mul x (T.div (T.lit 180) T.pi))) := rfl
theorem radians_formula (x : F) :
    tRadians T [.float x] = some (.float (T.mul x (T.div T.pi (T.lit 180)))) := rfl
theorem pi_value (d : Bool) : tPi T d ([] : List (V F)) = some (.float T.pi) := by
  simp [tPi]

/-- `apply_unary_float`: Null in → Null out; Int is promoted then mapped; never anything but
Float/Null; extra arguments are ignored. -/
theorem applyUnary_null (f : F → F) (rest : List (V F)) : applyUnary T f (.null :: rest) = some .null := rfl
theorem applyUnary_int (f : F → F) (n : Int) (rest : List (V F)) :
    applyUnary T f (.int n :: rest) = applyUnary T f [.float (T.ofInt n)] := rfl

/-- `unreachable!()` in `apply_unary_float` is dead after `validate_args_type` with
`[Int | Float | Null]`, and the result is accepted by the declared `ret`. -/
theorem applyUnary_no_panic (f : F → F) (v : V F)
    (h : validateArgsType (.fixed [numArg]) [v] = none) :
    ∃ r, applyUnary T f [v] = some r ∧ Accepts r (.union [.float, .null]) := by
  rcases numeric_body_cases v h with ⟨i, rfl⟩ | ⟨x, rfl⟩ | rfl
  · exact ⟨_, rfl, .union _ _ .float (by simp) (.tag _ _ rfl)⟩
  · exact ⟨_, rfl, .union _ _ .float (by simp) (.tag _ _ rfl)⟩
  · exact ⟨_, rfl, .union _ _ .null (by simp) (.tag _ _ rfl)⟩

/-- Every unary trig function = its C counterpart on every validated argument. -/
theorem unary_rust_eq_c (p : (List (V F) → Option (V F)) × (F → F)) (hp : p ∈ unaryPairs T)
    (v : V F) (h : validateArgsType (.fixed [numArg]) [v] = none) :
    p.1 [v] = cUnary T p.2 [v] := by
  simp only [unaryPairs, List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases numeric_body_cases v h with ⟨i, rfl⟩ | ⟨x, rfl⟩ | rfl <;>
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> rfl

/-- …including on the unvalidated `Null`. -/
theorem unary_null (p : (List (V F) → Option (V F)) × (F → F)) (hp : p ∈ unaryPairs T) :
    p.1 [.null] = some .null := by
  simp only [unaryPairs, List.mem_cons, List.mem_nil_iff, or_false] at hp
  rcases hp with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> rfl

theorem atan2_args (y x : V F) (h : validateArgsType (.fixed [numArg, numArg]) [y, x] = none) :
    ((∃ i, y = .int i) ∨ (∃ f, y = .float f) ∨ y = .null) ∧
    ((∃ i, x = .int i) ∨ (∃ f, x = .float f) ∨ x = .null) := by
  have h' := (validateArgsType_go_none [numArg, numArg] [y, x] 0).1 h
  simp only [ArgsOk] at h'
  exact ⟨(accepts_numArg y).1 h'.1, (accepts_numArg x).1 h'.2.1⟩

/-- `atan2`: no panic after validation; Null iff either argument is Null; otherwise
`atan2(y as f64, x as f64)`; and = C `AR_ATAN2`. -/
theorem atan2_spec (y x : V F) (h : validateArgsType (.fixed [numArg, numArg]) [y, x] = none) :
    tAtan2 T [y, x] = cAtan2 T [y, x] ∧ (tAtan2 T [y, x]).isSome ∧
    (tAtan2 T [y, x] = some .null ↔ (y.isNull || x.isNull) = true) := by
  obtain ⟨hy, hx⟩ := atan2_args y x h
  rcases hy with ⟨a, rfl⟩ | ⟨a, rfl⟩ | rfl <;> rcases hx with ⟨b, rfl⟩ | ⟨b, rfl⟩ | rfl <;>
    simp [tAtan2, cAtan2, cNum, V.isNull]

theorem atan2_null_first (x : V F) : tAtan2 T [.null, x] = some .null := by
  cases x <;> rfl

/-- The declared signatures type the results: every validated call returns a value the
declared `ret` accepts (unary and `atan2`: `Float | Null`; `pi`: `Float`). -/
theorem atan2_ret (y x : V F) (h : validateArgsType (.fixed [numArg, numArg]) [y, x] = none) :
    ∃ r, tAtan2 T [y, x] = some r ∧ Accepts r (.union [.float, .null]) := by
  obtain ⟨hy, hx⟩ := atan2_args y x h
  rcases hy with ⟨a, rfl⟩ | ⟨a, rfl⟩ | rfl <;> rcases hx with ⟨b, rfl⟩ | ⟨b, rfl⟩ | rfl <;>
    first
    | exact ⟨_, rfl, .union _ _ .float (by simp) (.tag _ _ rfl)⟩
    | exact ⟨_, rfl, .union _ _ .null (by simp) (.tag _ _ rfl)⟩

theorem pi_ret (d : Bool) : ∃ r, tPi T d ([] : List (V F)) = some r ∧ Accepts r .float :=
  ⟨_, pi_value T d, .tag _ _ rfl⟩

/-- `pi()` with arguments never reaches the body: arity is `validate (.fixed [])`, which
rejects one argument (live: "Received 1 arguments to function 'pi', expected at most 0",
same as C). -/
theorem pi_arity : validate (.fixed []) 1 = false := rfl

end ValueMath.Trig
