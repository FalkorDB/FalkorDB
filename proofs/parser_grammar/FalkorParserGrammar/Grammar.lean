/-
# The reference: openCypher's expression grammar (`graph/src/Cypher.g4:275-560`)

A direct transcription of the `oC_*Expression` rules as a recursive-descent
parser, one function per rule, over the same tokens and producing the same
tree shapes as the Rust parser for the same meaning:

* a run `a op b op c` of one left-associative operator is one n-ary node
  `op[a,b,c]` (Rust flattens it, macro.rs:140; the evaluator folds left,
  eval.rs:779-825), a change of operator nests on the left;
* a chain of comparisons `a < b <= c` is `a < b AND b <= c` (the chaining
  rule of the openCypher / Cypher manual), built as
  `And[c1, And[c2, ... ck]]`;
* `( ( x ) )` is `Paren[x]` (one level; the evaluator's `Paren` is identity);
* `+x` is `x`; `IS NULL` / `IS NOT NULL` are `isNull[bool, x]`.

Each function returns `Res` (from `Model`), `.fuel` when the budget runs out.
-/
import FalkorParserGrammar.Model

namespace FalkorParserGrammar

namespace G

/-- One step of `oC_ComparisonExpression`'s chain builder. -/
def chain : List RT → RT
  | [] => lf .null
  | [c] => c
  | c :: cs => .node .and_ [c, chain cs]

/-- Fold one more operand of a left-associative operator run (flattening a
run of the same operator). -/
def fold (op : Op) (acc : RT) (rhs : RT) : RT :=
  if acc.root = op then acc.addChild rhs else .node op [acc, rhs]

/-- `IS [NOT] NULL` suffixes. -/
def nullPred : List Tok → Option (Bool × List Tok)
  | .kw .is_ :: .kw .not_ :: .kw .null :: r => some (true, r)
  | .kw .is_ :: .kw .null :: r => some (false, r)
  | _ => none

/-- `oC_StringPredicateExpression` / `oC_ListPredicateExpression` heads. -/
def binPred : List Tok → Option (Op × List Tok)
  | .kw .starts :: .kw .with_ :: r => some (.startsWith, r)
  | .kw .ends :: .kw .with_ :: r => some (.endsWith, r)
  | .kw .contains :: r => some (.contains, r)
  | .kw .in_ :: r => some (.in_, r)
  | .regex :: r => some (.regex, r)      -- `=~` (in Cypher.g4's oC_StringPredicate family in Neo4j; FalkorDB extension)
  | _ => none

def addTok : Tok → Option Op
  | .plus => some .add | .dash => some .sub | _ => none
def mulTok : Tok → Option Op
  | .star => some .mul | .slash => some .div | .percent => some .mod | _ => none

mutual

/-- `oC_Expression` / `oC_OrExpression`. -/
def expr : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts => bin n 0 ts

/-- Levels 0-2 (`OR`, `XOR`, `AND`): `oC_XExpression ( X oC_YExpression )*`. -/
def bin : Nat → Nat → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, l, ts =>
    match sub n l ts with
    | .ok a r => binTail n l a r
    | o => o

/-- The operand rule of level `l`. -/
def sub : Nat → Nat → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, l, ts => if l < 2 then bin n (l + 1) ts else notE n ts

def binTail : Nat → Nat → RT → List Tok → Out
  | 0, _, _, _ => .fuel
  | n + 1, l, acc, ts =>
    match opTok l (cur ts) with
    | some op =>
      match sub n l (adv ts) with
      | .ok b r => binTail n l (fold op acc b) r
      | o => o
    | none => .ok acc ts

/-- `oC_NotExpression : ( NOT )* oC_ComparisonExpression`. -/
def notE : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, .kw .not_ :: r =>
    match notE n r with
    | .ok a r' => .ok (.node .not_ [a]) r'
    | o => o
  | n + 1, ts => cmpE n ts

/-- `oC_ComparisonExpression : SLN ( op SLN )*`, desugared as a chain. -/
def cmpE : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts =>
    match predE n ts with
    | .ok a r => cmpTail n a [] r
    | o => o

def cmpTail : Nat → RT → List RT → List Tok → Out
  | 0, _, _, _ => .fuel
  | n + 1, last, acc, ts =>
    match cmpTok (cur ts) with
    | some op =>
      match predE n (adv ts) with
      | .ok b r => cmpTail n b (acc ++ [.node op [last, b]]) r
      | o => o
    | none => .ok (if acc = [] then last else chain acc) ts

/-- `oC_StringListNullPredicateExpression : AddSub ( SP | LP | NP )*`. -/
def predE : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts =>
    match addE n ts with
    | .ok a r => predTail n a r
    | o => o

def predTail : Nat → RT → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, acc, ts =>
    match nullPred ts with
    | some (neg, r) => predTail n (.node .isNull [lf (.bool neg), acc]) r
    | none =>
      match binPred ts with
      | some (op, r) =>
        match addE n r with
        | .ok b r' => predTail n (.node op [acc, b]) r'
        | o => o
      | none => .ok acc ts

/-- `oC_AddOrSubtractExpression`. -/
def addE : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts =>
    match mulE n ts with
    | .ok a r => addTail n a r
    | o => o

def addTail : Nat → RT → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, acc, ts =>
    match addTok (cur ts) with
    | some op =>
      match mulE n (adv ts) with
      | .ok b r => addTail n (fold op acc b) r
      | o => o
    | none => .ok acc ts

/-- `oC_MultiplyDivideModuloExpression`. -/
def mulE : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts =>
    match powE n ts with
    | .ok a r => mulTail n a r
    | o => o

def mulTail : Nat → RT → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, acc, ts =>
    match mulTok (cur ts) with
    | some op =>
      match powE n (adv ts) with
      | .ok b r => mulTail n (fold op acc b) r
      | o => o
    | none => .ok acc ts

/-- `oC_PowerOfExpression : Unary ( '^' Unary )*` — left-associative. -/
def powE : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts =>
    match unE n ts with
    | .ok a r => powTail n a r
    | o => o

def powTail : Nat → RT → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, acc, ts =>
    match cur ts with
    | .pow =>
      match unE n (adv ts) with
      | .ok b r => powTail n (fold .pow acc b) r
      | o => o
    | _ => .ok acc ts

/-- `oC_UnaryAddOrSubtractExpression : NonArith | ('+'|'-') NonArith` — one sign. -/
def unE : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, .dash :: r =>
    match nonArith n r with
    | .ok a r' => .ok (.node .neg [a]) r'
    | o => o
  | n + 1, .plus :: r => nonArith n r
  | n + 1, ts => nonArith n ts

/-- `oC_NonArithmeticOperatorExpression : Atom ( ListOp | PropLookup )* NodeLabels?`. -/
def nonArith : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts =>
    match atom n ts with
    | .ok a r => postTail n a r
    | o => o

def postTail : Nat → RT → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, acc, ts =>
    match ts with
    | .dot :: t :: r =>
      match identOf t with
      | some p => postTail n (.node (.prop p) [acc]) r
      | none => .err (t :: r)
    | .lbrack :: .dotdot :: .rbrack :: r =>
      postTail n (.node .getElems [acc, lf (.int 0), lf (.int 9223372036854775807)]) r
    | .lbrack :: .dotdot :: r =>
      match expr n r with
      | .ok b (.rbrack :: r') =>
        postTail n (.node .getElems [acc, lf (.int 0), b]) r'
      | .ok _ r' => .err r'
      | o => o
    | .lbrack :: r =>
      match expr n r with
      | .ok a (.rbrack :: r') => postTail n (.node .getElem [acc, a]) r'
      | .ok a (.dotdot :: .rbrack :: r') =>
        postTail n (.node .getElems [acc, a, lf (.int 9223372036854775807)]) r'
      | .ok a (.dotdot :: r') =>
        match expr n r' with
        | .ok b (.rbrack :: r'') => postTail n (.node .getElems [acc, a, b]) r''
        | .ok _ r'' => .err r''
        | o => o
      | .ok _ r' => .err r'
      | o => o
    | .colon :: _ =>
      match labels [] ts with
      | .ok ls r => .ok (.node .hasLabels [acc, .node .list (ls.map (fun l => lf (.str l)))]) r
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
    | _ => .ok acc ts

/-- `oC_Atom` (the fragment): literal, parameter, list comprehension, list
literal, map literal, parenthesized expression, function invocation, variable. -/
def atom : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts =>
    match ts with
    | .kw .null :: r => .ok (lf .null) r
    | .kw .true_ :: r => .ok (lf (.bool true)) r
    | .kw .false_ :: r => .ok (lf (.bool false)) r
    | .param p :: r => .ok (lf (.param p)) r
    | .int k :: r => .ok (lf (.int k)) r
    | .str s :: r => .ok (lf (.str s)) r
    | .lparen :: r =>
      match expr n r with
      | .ok e (.rparen :: r') =>
        .ok (match e with | .node .paren cs => .node .paren cs | e => .node .paren [e]) r'
      | .ok _ r' => .err r'
      | o => o
    | .lbrack :: .rbrack :: r => .ok (lf .list) r
    | .lbrack :: t :: .kw .in_ :: r =>
      match identOf t with
      | some v =>
        -- oC_ListComprehension without `|` : '[' Variable IN Expression Where? ']'
        match expr n r with
        | .ok l (.kw .where_ :: r') =>
          match expr n r' with
          | .ok c (.rbrack :: r'') => .ok (.node (.listComp v) [l, c, lf (.var v)]) r''
          | .ok _ r'' => .err r''
          | o => o
        | .ok l (.rbrack :: r') => .ok (.node (.listComp v) [l, lf (.bool true), lf (.var v)]) r'
        | .ok _ r' => .err r'
        | o => o
      | none => listItems n [] (t :: .kw .in_ :: r)
    | .lbrack :: r => listItems n [] r
    | .lbrace :: .rbrace :: r => .ok (lf .map) r
    | .lbrace :: r => mapItems n [] r
    | t :: r =>
      match identOf t with
      | none => .err (t :: r)
      | some v =>
        match dotted [v] r with
        | .ok name (.lparen :: .rparen :: r') => call name [] r'
        | .ok name (.lparen :: r') => callArgs n name [] r'
        | .ok _ _ => .ok (lf (.var v)) r
        | .err e => .err e
        | .panic => .panic
        | .fuel => .fuel
    | [] => .err []

/-- `oC_ListLiteral`'s `expr (',' expr)* ']'`. -/
def listItems : Nat → List RT → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, acc, ts =>
    match expr n ts with
    | .ok e (.comma :: r) => listItems n (acc ++ [e]) r
    | .ok e (.rbrack :: r) => .ok (.node .list (acc ++ [e])) r
    | .ok _ r => .err r
    | o => o

/-- `oC_MapLiteral`'s `key ':' expr (',' key ':' expr)* '}'`. -/
def mapItems : Nat → List RT → List Tok → Out
  | 0, _, _ => .fuel
  | n + 1, acc, t :: .colon :: r =>
    match identOf t with
    | none => .err (t :: .colon :: r)
    | some k =>
      match expr n r with
      | .ok e (.comma :: r') => mapItems n (acc ++ [.node (.entry k) [e]]) r'
      | .ok e (.rbrace :: r') => .ok (.node .map (acc ++ [.node (.entry k) [e]])) r'
      | .ok _ r' => .err r'
      | o => o
  | _ + 1, _, ts => .err ts

/-- `oC_FunctionInvocation`'s `expr (',' expr)* ')'` — no trailing comma. -/
def callArgs : Nat → List Nat → List RT → List Tok → Out
  | 0, _, _, _ => .fuel
  | n + 1, name, acc, ts =>
    match expr n ts with
    | .ok e (.comma :: r) => callArgs n name (acc ++ [e]) r
    | .ok e (.rparen :: r) => call name (acc ++ [e]) r
    | .ok _ r => .err r
    | o => o

/-- Name resolution and arity (the same registry as the model). -/
def call : List Nat → List RT → List Tok → Out
  | name, args, r =>
    if name = [100] ∨ name = [101] ∨ name = [102] then .err r
    else match fnTable name with
      | some (lo, hi) => if lo ≤ args.length ∧ args.length ≤ hi then .ok (.node (.func name) args) r else .err r
      | none => .err r

end

/-- A whole expression followed by end of input. -/
def all (n : Nat) (ts : List Tok) : Out :=
  match expr n ts with
  | .ok t r => if cur r = .eof then .ok t r else .err r
  | o => o

end G

end FalkorParserGrammar
