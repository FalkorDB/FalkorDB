/-
# Precedence, associativity and agreement with the grammar
-/
import FalkorParserGrammar.Grammar

namespace FalkorParserGrammar

/-- Binary / postfix operator spellings, one per precedence family. -/
def binOps : List (List Tok) :=
  [[.kw .or_], [.kw .xor], [.kw .and_], [.eq], [.neq], [.lt], [.le], [.gt], [.ge],
   [.kw .in_], [.kw .starts, .kw .with_], [.kw .ends, .kw .with_], [.kw .contains], [.regex],
   [.plus], [.dash], [.star], [.slash], [.percent], [.pow]]

/-- `1 op1 2 op2 3`. -/
def triple (o1 o2 : List Tok) : List Tok := [.int 1] ++ o1 ++ [.int 2] ++ o2 ++ [.int 3]

/-- **Every pair of binary operators** is parsed by the Rust model exactly as
the openCypher grammar parses it (same tree, same associativity). -/
theorem binary_pairs_agree :
    ∀ o1 ∈ binOps, ∀ o2 ∈ binOps, parseAll 400 (triple o1 o2) = G.all 200 (triple o1 o2) := by
  decide

/-! ## Precedence and associativity for arbitrary literals

These hold for *every* integer literal `a b c` (the parser never looks at a
literal's value), by evaluation. -/

/-- `-a ^ b` is `(-a) ^ b`: unary minus binds tighter than `^` (Cypher.g4:342-347). -/
theorem neg_binds_tighter_than_pow (a b : Nat) :
    parseAll 400 [.dash, .int a, .pow, .int b] = .ok (.node .pow [.node .neg [lf (.int a)], lf (.int b)]) [] := by
  rfl

/-- `a ^ b ^ c` is left-associative: one `Pow[a,b,c]`, folded left by the evaluator. -/
theorem pow_left_assoc (a b c : Nat) :
    parseAll 400 [.int a, .pow, .int b, .pow, .int c] =
      .ok (.node .pow [lf (.int a), lf (.int b), lf (.int c)]) [] := by
  rfl

/-- `a - b - c` is `(a - b) - c`; `a - b + c` is `(a - b) + c`. -/
theorem sub_left_assoc (a b c : Nat) :
    parseAll 400 [.int a, .dash, .int b, .dash, .int c] =
      .ok (.node .sub [lf (.int a), lf (.int b), lf (.int c)]) [] ∧
    parseAll 400 [.int a, .dash, .int b, .plus, .int c] =
      .ok (.node .add [.node .sub [lf (.int a), lf (.int b)], lf (.int c)]) [] := by
  constructor <;> rfl

/-- `a * b + c * d` and `a + b * c`: `*` binds tighter than `+`. -/
theorem mul_over_add (a b c : Nat) :
    parseAll 400 [.int a, .plus, .int b, .star, .int c] =
      .ok (.node .add [lf (.int a), .node .mul [lf (.int b), lf (.int c)]]) [] := by
  rfl

/-- `a < b < c` is `a < b AND b < c`; a longer chain nests to the right. -/
theorem chained_comparison (a b c d : Nat) :
    parseAll 400 [.int a, .lt, .int b, .le, .int c] =
      .ok (.node .and_ [.node .lt [lf (.int a), lf (.int b)], .node .le [lf (.int b), lf (.int c)]]) [] ∧
    parseAll 400 [.int a, .lt, .int b, .eq, .int c, .gt, .int d] =
      .ok (.node .and_ [.node .lt [lf (.int a), lf (.int b)],
        .node .and_ [.node .eq [lf (.int b), lf (.int c)], .node .gt [lf (.int c), lf (.int d)]]]) [] := by
  constructor <;> rfl

/-- `NOT a = b` is `NOT (a = b)`. -/
theorem not_looser_than_comparison (a b : Nat) :
    parseAll 400 [.kw .not_, .int a, .eq, .int b] = .ok (.node .not_ [.node .eq [lf (.int a), lf (.int b)]]) [] := by
  rfl

/-- `a + b IS NULL` is `(a + b) IS NULL`, and `a = b IS NULL` is `a = (b IS NULL)`
(`oC_StringListNullPredicateExpression` sits between comparison and `+`). -/
theorem is_null_between_cmp_and_add (a b : Nat) :
    parseAll 400 [.int a, .plus, .int b, .kw .is_, .kw .null] =
      .ok (.node .isNull [lf (.bool false), .node .add [lf (.int a), lf (.int b)]]) [] ∧
    parseAll 400 [.int a, .eq, .int b, .kw .is_, .kw .null] =
      .ok (.node .eq [lf (.int a), .node .isNull [lf (.bool false), lf (.int b)]]) [] := by
  constructor <;> rfl

/-- `a + b STARTS WITH c` is `(a + b) STARTS WITH c` (FalkorDB C gets this wrong: `'a'+'b' STARTS WITH 'ab'` is `'afalse'` there). -/
theorem starts_with_looser_than_add (a b c : Nat) :
    parseAll 400 [.str a, .plus, .str b, .kw .starts, .kw .with_, .str c] =
      .ok (.node .startsWith [.node .add [lf (.str a), lf (.str b)], lf (.str c)]) [] := by
  rfl

/-- Property access and indexing bind tightest and apply to any expression:
`-x.p` is `-(x.p)`, `(x).p[i]` is `((x).p)[i]`, `x[i..j]` is a slice. -/
theorem postfix_tightest (x p i j : Nat) :
    parseAll 400 [.dash, .ident x, .dot, .ident p] = .ok (.node .neg [.node (.prop p) [lf (.var x)]]) [] ∧
    parseAll 400 [.lparen, .ident x, .rparen, .dot, .ident p, .lbrack, .int i, .rbrack] =
      .ok (.node .getElem [.node (.prop p) [.node .paren [lf (.var x)]], lf (.int i)]) [] ∧
    parseAll 400 [.ident x, .lbrack, .int i, .dotdot, .int j, .rbrack] =
      .ok (.node .getElems [lf (.var x), lf (.int i), lf (.int j)]) [] := by
  refine ⟨rfl, rfl, rfl⟩

/-- `OR` < `XOR` < `AND`. -/
theorem boolean_levels (a b c : Nat) :
    parseAll 400 [.ident a, .kw .or_, .ident b, .kw .xor, .ident c, .kw .and_, .ident a] =
      .ok (.node .or_ [lf (.var a), .node .xor [lf (.var b), .node .and_ [lf (.var c), lf (.var a)]]]) [] := by
  rfl

/-! ## Machine-checked disagreements with the grammar

Each is reproduced against the Rust in `graph/tests/lean_parser_grammar.rs`. -/

/-- **`NOT NOT x` keeps both `NOT`s** (#3060, `fe619ac5f`, cypher.rs:2193-2203), as the
grammar does — `NOT` type-checks its operand, so `RETURN NOT NOT 1` is a type error (C
too). Historical (2c874022a, `not_not_dropped`): the parity fold returned `x`. -/
theorem not_not_kept :
    parseAll 400 [.kw .not_, .kw .not_, .int 1] = .ok (.node .not_ [.node .not_ [lf (.int 1)]]) [] ∧
    G.all 200 [.kw .not_, .kw .not_, .int 1] = .ok (.node .not_ [.node .not_ [lf (.int 1)]]) [] := by
  decide

/-- A longer run still folds, by parity, to one or two `NOT`s (`NOT NOT NOT x` → `NOT x`,
four → two) — equal in value to the grammar's tree, since `NOT` type-checks at every
level; it is the one remaining (documented) `NOT` difference from the grammar. -/
theorem not_run_folds :
    parseAll 400 [.kw .not_, .kw .not_, .kw .not_, .int 1] = .ok (.node .not_ [lf (.int 1)]) [] ∧
    parseAll 400 [.kw .not_, .kw .not_, .kw .not_, .kw .not_, .int 1] =
      .ok (.node .not_ [.node .not_ [lf (.int 1)]]) [] := by
  decide

/-- **A predicate after `IS NULL` is accepted** with the grammar's tree (#3060,
cypher.rs:2410-2413 comes back to level 5). Historical (2c874022a,
`is_null_then_predicate_rejected`): it returned to the parent level and was rejected. -/
theorem is_null_then_predicate :
    parseAll 400 [.int 1, .kw .is_, .kw .null, .kw .in_, .ident 3] =
      .ok (.node .in_ [.node .isNull [lf (.bool false), lf (.int 1)], lf (.var 3)]) [] ∧
    G.all 200 [.int 1, .kw .is_, .kw .null, .kw .in_, .ident 3] =
      .ok (.node .in_ [.node .isNull [lf (.bool false), lf (.int 1)], lf (.var 3)]) [] := by
  decide

/-- A failed slice bound is silently replaced by `0` / `i64::MAX`
(cypher.rs:2124-2132): `x[1 + ..2]`, `x[abs()..2]`, `x[(..2]` are all
accepted as `x[0..2]`, where the grammar (and C) reject them. -/
theorem slice_bound_error_swallowed :
    parseAll 400 [.ident 3, .lbrack, .int 1, .plus, .dotdot, .int 2, .rbrack] =
      .ok (.node .getElems [lf (.var 3), lf (.int 0), lf (.int 2)]) [] ∧
    parseAll 400 [.ident 3, .lbrack, .ident 7, .lparen, .rparen, .dotdot, .int 2, .rbrack] =
      .ok (.node .getElems [lf (.var 3), lf (.int 0), lf (.int 2)]) [] ∧
    parseAll 400 [.ident 3, .lbrack, .lparen, .dotdot, .int 2, .rbrack] =
      .ok (.node .getElems [lf (.var 3), lf (.int 0), lf (.int 2)]) [] ∧
    (∃ e, G.all 200 [.ident 3, .lbrack, .int 1, .plus, .dotdot, .int 2, .rbrack] = .err e) ∧
    (∃ e, G.all 200 [.ident 3, .lbrack, .ident 7, .lparen, .rparen, .dotdot, .int 2, .rbrack] = .err e) := by
  refine ⟨rfl, rfl, rfl, ⟨_, rfl⟩, ⟨_, rfl⟩⟩

/-- **A trailing comma in a call is refused** (#3060, cypher.rs:2729-2739), as in the
grammar and C; `f()` and `f(1)` still parse. Historical (2c874022a, `call_trailing_comma`):
`abs(1,)` was accepted as `abs(1)`. -/
theorem call_trailing_comma_refused :
    (∃ e, parseAll 400 [.ident 7, .lparen, .int 1, .comma, .rparen] = .err e) ∧
    (∃ e, G.all 200 [.ident 7, .lparen, .int 1, .comma, .rparen] = .err e) ∧
    parseAll 400 [.ident 9, .lparen, .rparen] = .ok (lf (.func [9])) [] ∧
    parseAll 400 [.ident 7, .lparen, .int 1, .rparen] = .ok (.node (.func [7]) [lf (.int 1)]) [] := by
  refine ⟨⟨_, rfl⟩, ⟨_, rfl⟩, rfl, rfl⟩

/-- An empty-argument call cannot end on a comma either: `f(,)` is refused. -/
theorem exprList_comma_first (pe : List Tok → Out) (n : Nat) (r : List Tok)
    (h : ∀ ts, pe (.comma :: ts) = .err (.comma :: ts)) :
    exprList pe (n + 1) [] (.comma :: r) = .err (.comma :: r) := by
  simp [exprList, cur, exprItems, h]

/-- `+-x` is accepted (cypher.rs:2207-2208 reads an optional `+` then an
optional `-`), while `- -x` and `-+x` are rejected (#2309). The grammar
allows exactly one sign. -/
theorem sign_sequences :
    parseAll 400 [.plus, .dash, .int 1] = .ok (.node .neg [lf (.int 1)]) [] ∧
    (∃ e, parseAll 400 [.dash, .dash, .int 1] = .err e) ∧
    (∃ e, parseAll 400 [.dash, .plus, .int 1] = .err e) ∧
    (∃ e, G.all 200 [.plus, .dash, .int 1] = .err e) := by
  refine ⟨rfl, ⟨_, rfl⟩, ⟨_, rfl⟩, ⟨_, rfl⟩⟩

/-- The parse is deterministic: `parse_expr` is a function of its input
(no hidden state survives a `restore_state`; the model has none to begin
with, `anon_counter` is saved and restored by `save_state`/`restore_state`,
cypher.rs:341-354). -/
theorem parse_deterministic (n : Nat) (ts : List Tok) (r r' : Out)
    (h : parseAll n ts = r) (h' : parseAll n ts = r') : r = r' := h ▸ h'

end FalkorParserGrammar
