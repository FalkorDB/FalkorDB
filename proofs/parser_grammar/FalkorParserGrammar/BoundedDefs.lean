/-
# Bounded exhaustive agreement with the grammar: definitions and the chunking lemma

`Bounded.lean` states the two checks over `lists 3`. The kernel needs ~50 s to run them, so
they are cut into 10 chunks of 1000 lists (`BoundedChunk0..9.lean`, one file each so Lake runs
them in parallel) and `all_chunks` puts the chunks back together.
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
      ((x == .plus || x == .dash) && (y == .plus || y == .dash)) ||    -- a run of signs (sign_runs)
      (x == .kw .not_ && (y == .kw .in_ || y == .kw .contains))) ||   -- `x NOT IN y` extension
    -- a run of ≥ 3 NOTs folds by parity (not_run_folds); none fits in 3 tokens with an operand
    (l.zip (l.drop 1) |>.zip (l.drop 2) |>.any (fun ((x, y), z) =>
      x == .kw .not_ && y == .kw .not_ && z == .kw .not_)) ||
    -- keyword as variable: the grammar reads each keyword as an identifier; the trees (and the
    -- rest of the input) must still be equal, not merely both accepted
    ((parseAll 400 l matches .ok _ _) && (G.all 200 (asIdents l) matches .ok _ _) &&
      agreeB (parseAll 400 l) (G.all 200 (asIdents l)))

/-- **Bounded soundness and completeness**: on every one of the 9,724 token
lists of length ≤ 3 over `alpha`, the Rust model and the grammar accept the
same inputs with the same trees, outside the documented classes. (The same
check over all 4,288,306 lists of length ≤ 5 was run with `#eval`; the only
disagreements are the same classes.) -/
def okAt (l : List Tok) : Bool := agreeB (parseAll 400 l) (G.all 200 l) || known l

/-! ## Balanced consumption (the fragment's `Oracle.Bal`)

`C.ListGate`'s premise `Oracle.Bal` says `parse_expr` only succeeds by consuming
a bracket-balanced run with no top-level `,` (or `|`, which this alphabet lacks).
Here it is checked on the fragment. -/

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

/-! ## Chunking: `lists 3` is the concatenation of ten chunks of 1000 -/

/-- Chunk `i` of a list: elements `1000 i .. 1000 i + 999`. -/
def chunk (l : List α) (i : Nat) : List α := (l.drop (1000 * i)).take 1000

theorem all_chunks (p : α → Bool) : ∀ (n : Nat) (l : List α), l.length ≤ 1000 * n →
    (∀ i, i < n → (chunk l i).all p = true) → l.all p = true := by
  intro n
  induction n with
  | zero =>
    intro l hl _
    have : l = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this; rfl
  | succ n ih =>
    intro l hl h
    have h0 := h 0 (by omega)
    have h1 : (l.drop 1000).all p = true := by
      apply ih
      · simp only [List.length_drop]; omega
      · intro i hi
        have := h (i + 1) (by omega)
        simpa [chunk, List.drop_drop, Nat.mul_add, Nat.add_comm] using this
    have e : l = l.take 1000 ++ l.drop 1000 := (List.take_append_drop 1000 l).symm
    rw [e, List.all_append]
    simp only [chunk, Nat.mul_zero, List.drop_zero] at h0
    simp [h0, h1]

theorem lists3_length : (lists 3).length ≤ 10000 := by decide +kernel

/-- Both checks hold on every list of `lists 3` once they hold on each of its ten chunks. -/
theorem all_lists3 (p : List Tok → Bool) (h : ∀ i, i < 10 → (chunk (lists 3) i).all p = true) :
    (lists 3).all p = true :=
  all_chunks p 10 _ lists3_length h

end FalkorParserGrammar
