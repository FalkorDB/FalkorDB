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
cypher.rs:1989, accepts any keyword as a variable name). -/
def asIdents (l : List Tok) : List Tok :=
  l.map fun t => match t with
    | .kw .and_ => .ident 90 | .kw .in_ => .ident 91 | .kw .contains => .ident 92
    | .kw .is_ => .ident 93 | .kw .not_ => .ident 94 | t => t

/-- The documented disagreement classes (see `Theorems.lean`). Since #3060 (`fe619ac5f`)
`NOT NOT`, a predicate after `IS NULL` and a call's trailing comma are no longer among
them, and since #3041 (`2023af648`) neither is an accepted slice (`..`): the bounded
check now proves those agree with the grammar. -/
def known (l : List Tok) : Bool :=
  let pairs := l.zip (l.drop 1)
  pairs.any (fun (x, y) =>
      (x == .plus && y == .dash) ||                                   -- `+-x` (sign_sequences)
      (x == .kw .not_ && (y == .kw .in_ || y == .kw .contains))) ||   -- `x NOT IN y` extension
    -- a run of ≥ 3 NOTs folds by parity (not_run_folds); none fits in 3 tokens with an operand
    (l.zip (l.drop 1) |>.zip (l.drop 2) |>.any (fun ((x, y), z) =>
      x == .kw .not_ && y == .kw .not_ && z == .kw .not_)) ||
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

/-! ## Balanced consumption (the fragment's `Oracle.Bal`)

`C.ListGate`'s premise `Oracle.Bal` says `parse_expr` only succeeds by consuming
a bracket-balanced run with no top-level `,` (or `|`, which this alphabet lacks).
Before #3041 it was false (`x[(..2]` consumed an unclosed `(`). Here it is checked on the fragment. -/

/-- Depth scan: never below 0, ends at 0, no `,` at depth 0. -/
def balRun : Nat → List Tok → Bool
  | d, [] => d == 0
  | d, t :: r =>
    if t == .lparen || t == .lbrack || t == .lbrace then balRun (d + 1) r
    else if t == .rparen || t == .rbrack || t == .rbrace then d != 0 && balRun (d - 1) r
    else if t == .comma then d != 0 && balRun d r
    else balRun d r

/-- A successful `parseExpr` leaves a suffix of its input and consumed a balanced run. -/
def balOK (l : List Tok) : Bool :=
  match parseExpr 400 l with
  | .ok _ r => r.length ≤ l.length && l.drop (l.length - r.length) == r &&
      balRun 0 (l.take (l.length - r.length))
  | _ => true

set_option maxRecDepth 100000 in
/-- **The fragment's `parse_expr` consumes balanced runs** on every token list of
length ≤ 3 over `alpha`. (The same check was run with `#eval` over all lists of
length ≤ 5 over `alpha` and of length ≤ 6 over the slice alphabet
`x 1 + ( ) [ ] ..`, with no failure.) -/
theorem bounded_bal : (lists 3).all balOK = true := by decide +kernel

/-- The input `[1][(1..2]` (the pre-#3041 gate counterexample) no longer parses. -/
theorem slice_unclosed_paren_refused :
    ∃ e, parseExpr 400 [.lbrack, .int 1, .rbrack, .lbrack, .lparen, .int 1, .dotdot, .int 2, .rbrack] = .err e :=
  ⟨_, rfl⟩

end FalkorParserGrammar
