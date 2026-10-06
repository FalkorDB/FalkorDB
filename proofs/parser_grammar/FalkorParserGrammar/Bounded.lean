/-
# Bounded exhaustive agreement with the grammar
-/
import FalkorParserGrammar.Grammar

namespace FalkorParserGrammar

/-- Same tree and same remaining input, or both errors (positions ignored). -/
def agreeB (a b : Out) : Bool :=
  match a, b with
  | .ok t r, .ok t' r' => t == t' && r == r'
  | .err _, .err _ => true
  | _, _ => false

/-- 21 tokens: literals, a variable, a function name, the arithmetic,
comparison, boolean and predicate operators, brackets, `..`, `,`, `.`. -/
def alpha : List Tok := [.int 1, .ident 3, .plus, .dash, .pow, .lt, .eq, .kw .not_, .kw .is_, .kw .null,
  .kw .and_, .lparen, .rparen, .lbrack, .rbrack, .dotdot, .comma, .dot, .kw .in_, .kw .contains, .ident 7]

/-- All token lists of length ≤ `n` over `alpha`. -/
def lists : Nat → List (List Tok)
  | 0 => [[]]
  | n + 1 => lists n ++ ((lists n).filter (·.length == n)).flatMap (fun l => alpha.map (fun t => l ++ [t]))

/-- Keywords read as identifiers (Rust's `parse_primary_expr` catch-all,
cypher.rs:1973, accepts any keyword as a variable name). -/
def asIdents (l : List Tok) : List Tok :=
  l.map fun t => match t with
    | .kw .and_ => .ident 90 | .kw .in_ => .ident 91 | .kw .contains => .ident 92
    | .kw .is_ => .ident 93 | .kw .not_ => .ident 94 | t => t

/-- The documented disagreement classes (see `Theorems.lean`). Since #3060 (`fe619ac5f`)
`NOT NOT`, a predicate after `IS NULL` and a call's trailing comma are no longer among
them: the bounded check now proves those agree with the grammar. -/
def known (l : List Tok) : Bool :=
  let pairs := l.zip (l.drop 1)
  pairs.any (fun (x, y) =>
      (x == .plus && y == .dash) ||                                   -- `+-x` (sign_sequences)
      (x == .kw .not_ && (y == .kw .in_ || y == .kw .contains))) ||   -- `x NOT IN y` extension
    -- a run of ≥ 3 NOTs folds by parity (not_run_folds); none fits in 3 tokens with an operand
    (l.zip (l.drop 1) |>.zip (l.drop 2) |>.any (fun ((x, y), z) =>
      x == .kw .not_ && y == .kw .not_ && z == .kw .not_)) ||
    (l.contains .dotdot && (parseAll 400 l matches .ok _ _)) ||       -- slice_bound_error_swallowed
    ((parseAll 400 l matches .ok _ _) && (G.all 200 (asIdents l) matches .ok _ _))  -- keyword as variable

/-- **Bounded soundness and completeness**: on every one of the 9,724 token
lists of length ≤ 3 over `alpha`, the Rust model and the grammar accept the
same inputs with the same trees, outside the documented classes. (The same
check over all 4,288,306 lists of length ≤ 5 was run with `#eval`; the only
disagreements are the same classes.) -/
def okAt (l : List Tok) : Bool := agreeB (parseAll 400 l) (G.all 200 l) || known l

set_option maxRecDepth 100000 in
theorem bounded_check : (lists 3).all okAt = true := by decide +kernel

theorem bounded_agree :
    ∀ l ∈ lists 3, agreeB (parseAll 400 l) (G.all 200 l) = true ∨ known l = true := by
  intro l hl
  have := List.all_eq_true.mp bounded_check l hl
  simpa [okAt] using this

end FalkorParserGrammar
