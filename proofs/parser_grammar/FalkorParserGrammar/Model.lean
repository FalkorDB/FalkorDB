/-
# Model of the expression parser of `graph/src/parser/cypher.rs`

A line-by-line model of `Parser::parse_expr_inner` (cypher.rs:2170-2618), the
`parse_expr_return!` / `parse_operators!` macros (macro.rs:115-168) and the
primaries it calls. Line numbers refer to cypher.rs unless stated otherwise.

Scope of the token alphabet (see the header of `FalkorParserGrammar.lean`):
no `CASE`, quantifiers, `DISTINCT`, `|`, floats, `count(*)`, aggregates,
pattern predicates (`allow_pattern_predicate = false`). The absence of `|`
makes both pattern-comprehension attempts of
`parse_list_literal_or_comprehension` fail, which is what the model does.
Tree heights / `check_depth` and `nested` are *not* modelled here — they are
the subject of `proofs/lexer` (`height_sound`, `nested_depth_bound`).
-/

namespace FalkorParserGrammar

/-! ## Tokens (`graph/src/parser/lexer.rs` `Token`) -/

/-- The keywords of `Keyword` the fragment uses. -/
inductive Kw
  | or_ | xor | and_ | not_ | in_ | starts | ends | with_ | contains | is_
  | null | true_ | false_ | where_
  deriving DecidableEq, Repr

/-- `Token`. The Rust names `LBrace`/`RBrace` are `[`/`]` and
`LBracket`/`RBracket` are `{`/`}`; here they are named by their glyph. -/
inductive Tok
  | int (n : Nat)            -- Token::Integer (non-negative; i64::MIN handled in proofs/lexer)
  | str (s : Nat)            -- Token::String (abstract contents)
  | ident (s : Nat)          -- Token::IdentifierOrKeyword { keyword: None }
  | kw (k : Kw)              -- Token::IdentifierOrKeyword { keyword: Some k }
  | param (s : Nat)          -- Token::Parameter
  | lparen | rparen          -- ( )
  | lbrack | rbrack          -- [ ]   (Rust LBrace / RBrace)
  | lbrace | rbrace          -- { }   (Rust LBracket / RBracket)
  | dot | dotdot | comma | colon
  | eq | neq | lt | le | gt | ge | regex
  | plus | dash | star | slash | percent | pow
  | eof
  deriving DecidableEq, Repr

/-- The lexer's `current()`; running off the list is `EndOfFile`. -/
def cur : List Tok → Tok
  | [] => .eof
  | t :: _ => t

/-- The lexer's `next()`; at end of input it stays there (lexer.rs:295-300,
`get_token` returns `(EndOfFile, 0)` at the end, lexer.rs:512). -/
def adv : List Tok → List Tok
  | [] => []
  | _ :: r => r

/-- Keyword text as an identifier (`parse_ident` accepts keywords, cypher.rs:2644). -/
def kwCode : Kw → Nat
  | .or_ => 1000 | .xor => 1001 | .and_ => 1002 | .not_ => 1003 | .in_ => 1004
  | .starts => 1005 | .ends => 1006 | .with_ => 1007 | .contains => 1008
  | .is_ => 1009 | .null => 1010 | .true_ => 1011 | .false_ => 1012 | .where_ => 1013

/-- `Token::IdentifierOrKeyword { ident, .. }` → `ident`. -/
def identOf : Tok → Option Nat
  | .ident s => some s
  | .kw k => some (kwCode k)
  | _ => none

/-! ## Trees (`ExprIR` in `graph/src/parser/ast.rs`) -/

inductive Op
  | or_ | xor | and_ | not_ | eq | neq | lt | le | gt | ge
  | in_ | startsWith | endsWith | contains | regex | isNull
  | add | sub | mul | div | mod | pow | neg
  | prop (p : Nat) | getElem | getElems | hasLabels | paren | list
  | int (n : Int) | str (s : Nat) | bool (b : Bool) | null | var (v : Nat) | param (p : Nat)
  | func (f : List Nat) | listComp (v : Nat) | map | entry (k : Nat) | mapProj
  deriving DecidableEq, Repr

/-- `DynTree<ExprIR>`: an operator with its children, in order. -/
inductive RT
  | node (op : Op) (cs : List RT)
  deriving Repr

def RT.root : RT → Op | .node o _ => o
def RT.kids : RT → List RT | .node _ cs => cs

/-- `push_child_tree`: append a child. -/
def RT.addChild : RT → RT → RT | .node o cs, c => .node o (cs ++ [c])

mutual
def RT.beq : RT → RT → Bool
  | .node o cs, .node o' cs' => o == o' && RT.beqL cs cs'
def RT.beqL : List RT → List RT → Bool
  | [], [] => true
  | a :: as, b :: bs => RT.beq a b && RT.beqL as bs
  | _, _ => false
end

mutual
theorem RT.beq_eq : ∀ a b : RT, RT.beq a b = true → a = b
  | .node o cs, .node o' cs', h => by
    simp only [RT.beq, Bool.and_eq_true, beq_iff_eq] at h
    rw [h.1, RT.beqL_eq cs cs' h.2]
theorem RT.beqL_eq : ∀ a b : List RT, RT.beqL a b = true → a = b
  | [], [], _ => rfl
  | a :: as, b :: bs, h => by
    simp only [RT.beqL, Bool.and_eq_true] at h
    rw [RT.beq_eq a b h.1, RT.beqL_eq as bs h.2]
  | [], _ :: _, h => by simp [RT.beqL] at h
  | _ :: _, [], h => by simp [RT.beqL] at h
end

mutual
theorem RT.beq_refl : ∀ a : RT, RT.beq a a = true
  | .node o cs => by simp only [RT.beq, beq_self_eq_true, Bool.true_and]; exact RT.beqL_refl cs
theorem RT.beqL_refl : ∀ a : List RT, RT.beqL a a = true
  | [] => rfl
  | a :: as => by simp only [RT.beqL, Bool.and_eq_true]; exact ⟨RT.beq_refl a, RT.beqL_refl as⟩
end

instance : DecidableEq RT := fun a b =>
  if h : RT.beq a b then .isTrue (RT.beq_eq a b h)
  else .isFalse (fun e => h (e ▸ RT.beq_refl a))

/-- Leaves. -/
abbrev lf (o : Op) : RT := .node o []

/-! ## Results -/

/-- A sub-parser's result. `err` keeps the lexer position, because
`parse_list_operator_expression` carries on from wherever a failed
`parse_expr` stopped (cypher.rs:2124-2132). `panic` is an `unwrap()` /
`unreachable!()` / `child(0)` on a childless node; `fuel` is the model's
step budget running out (not a Rust behaviour). -/
inductive Res (α : Type)
  | ok (a : α) (ts : List Tok)
  | err (ts : List Tok)
  | panic
  | fuel
  deriving Repr, DecidableEq

/-- `parse_expr`'s result type. -/
abbrev Out := Res RT

def Res.rest {α} : Res α → List Tok → List Tok
  | .ok _ ts, _ => ts
  | .err ts, _ => ts
  | _, d => d

/-! ## `parse_expr_return!` (macro.rs:115-133) -/

/-- A stack frame `(precedence level, Option<tree>)`; the height component
is not modelled here. The stack is a list with its top at the head. -/
abbrev Frame := Nat × Option RT

/-- The machine state between iterations of `while let Some(..) = stack.pop()`. -/
inductive St
  | run (K : List Frame) (ts : List Tok)
  | done (t : RT) (ts : List Tok)
  | err (ts : List Tok)
  | panic
  | fuel

/-- `parse_expr_return!`: push `t` as a child of the top frame's tree, or
store it in a still-empty top frame, or return it when the stack is empty. -/
def ret (t : RT) : List Frame → List Tok → St
  | (l, some e) :: K, ts => .run ((l, some (e.addChild t)) :: K) ts
  | (l, none) :: K, ts => .run ((l, some t) :: K) ts
  | [], ts => .done t ts

/-! ## `parse_operators!` (macro.rs:135-168), levels 0, 1, 2, 6, 7, 8 -/

/-- The operator tokens of each binary level (cypher.rs:2244-2520). The
multi-token form tries its tokens in order and `continue`s on the first
match, so a lookup table is the same thing. -/
def opTok : Nat → Tok → Option Op
  | 0, .kw .or_ => some .or_
  | 1, .kw .xor => some .xor
  | 2, .kw .and_ => some .and_
  | 6, .plus => some .add
  | 6, .dash => some .sub
  | 7, .star => some .mul
  | 7, .slash => some .div
  | 7, .percent => some .mod
  | 8, .pow => some .pow
  | _, _ => none

/-- `parse_operators!`: on an operator, wrap the left operand (unless it is
already that operator — this flattens `a+b+c` into `Add[a,b,c]`) and parse
the right operand one level up; otherwise fold into the parent. -/
def binStep (l : Nat) (res : RT) (K : List Frame) (ts : List Tok) : St :=
  match opTok l (cur ts) with
  | some op =>
    .run ((l + 1, none) :: (l, some (if res.root = op then res else .node op [res])) :: K) (adv ts)
  | none => ret res K ts

/-! ## Level 4: comparison with chained-range desugaring (cypher.rs:2258-2329) -/

def cmpTok : Tok → Option Op
  | .eq => some .eq | .neq => some .neq | .lt => some .lt
  | .le => some .le | .gt => some .gt | .ge => some .ge
  | _ => none

def isCmp : Op → Bool
  | .eq | .neq | .lt | .le | .gt | .ge => true
  | _ => false

def RT.last? (t : RT) : Option RT := t.kids.getLast?

/-- `last_cmp_is_ordering` (2276-2295). -/
def lastCmpIsOrdering (res : RT) : Bool :=
  isCmp res.root ||
    (res.root == .and_ && (match res.last? with | some c => isCmp c.root | none => false))

def cmpStep (res : RT) (K : List Frame) (ts : List Tok) : St :=
  match cmpTok (cur ts) with
  | none => ret res K ts
  | some op =>
    let ts := adv ts
    -- last_cmp_is_ordering (2276-2295)
    if lastCmpIsOrdering res then
      -- last_cmp_node (2299-2303): `.children().last().unwrap()`
      match (if res.root = .and_ then res.last? else some res) with
      | none => .panic
      | some lcn =>
        -- middle_clone (2304-2305): `.children().last().unwrap()`
        match lcn.last? with
        | none => .panic
        | some mid =>
          let res' := if res.root = .and_ then res else .node .and_ [res]
          .run ((5, none) :: (4, some (.node op [mid])) :: (4, some res') :: K) ts
    else
      .run ((5, none) :: (4, some (.node op [res])) :: K) ts

/-! ## Level 5: string, list and null predicates (cypher.rs:2330-2509) -/

/-- The `while optional_match_token!(Is)` loop (2393-2409). -/
def isLoop (res : RT) : List Tok → Res RT
  | .kw .is_ :: r =>
    match r with
    | .kw .not_ :: .kw .null :: r' => isLoop (.node .isNull [lf (.bool true), res]) r'
    | .kw .not_ :: r' => .err r'
    | .kw .null :: r' => isLoop (.node .isNull [lf (.bool false), res]) r'
    | r' => .err r'
  | ts => .ok res ts

/-- `stack.push((current, Some(res))); stack.push((current + 1, None))` (2507-2508). -/
def push5 (res : RT) (K : List Frame) (ts : List Tok) : St :=
  .run ((6, none) :: (5, some res) :: K) ts

def predStep (res : RT) (K : List Frame) (ts : List Tok) : St :=
  match ts with
  | .kw .in_ :: r => push5 (.node .in_ [res]) K r
  | .kw .starts :: r =>
    match r with
    | .kw .with_ :: r' => push5 (.node .startsWith [res]) K r'
    | r' => .err r'
  | .kw .ends :: r =>
    match r with
    | .kw .with_ :: r' => push5 (.node .endsWith [res]) K r'
    | r' => .err r'
  | .kw .contains :: r => push5 (.node .contains [res]) K r
  | .regex :: r => push5 (.node .regex [res]) K r
  | .kw .is_ :: _ =>
    -- #3060 (`fe619ac5f`): `stack.push((current, Some(res), height)); continue;` (2412) —
    -- more predicates may follow (`x IS NULL IN [false]`); it used to `parse_expr_return!`.
    match isLoop res ts with
    | .ok res' ts' => .run ((5, some res') :: K) ts'
    | .err ts' => .err ts'
    | .panic => .panic
    | .fuel => .fuel
  | .kw .not_ :: r =>
    let K' := (5, some (.node .not_ [])) :: K
    match r with
    | .kw .in_ :: r' => push5 (.node .in_ [res]) K' r'
    | .kw .starts :: .kw .with_ :: r' => push5 (.node .startsWith [res]) K' r'
    | .kw .starts :: r' => .err r'
    | .kw .ends :: .kw .with_ :: r' => push5 (.node .endsWith [res]) K' r'
    | .kw .ends :: r' => .err r'
    | .kw .contains :: r' => push5 (.node .contains [res]) K' r'
    | r' => .err r'   -- "Invalid usage of 'NOT' filter"
  | _ => ret res K ts

/-! ## Primaries, lists, maps, calls -/

/-- `parse_ident` (2642-2654). Identifier-length validation is not modelled. -/
def parseIdent : List Tok → Res Nat
  | t :: r => match identOf t with
    | some v => .ok v r
    | none => .err (t :: r)
  | [] => .err []

/-- `parse_dotted_ident` (1099-1108): `parse_ident` inlined so the recursion is structural. -/
def dotted (acc : List Nat) : List Tok → Res (List Nat)
  | .dot :: t :: r =>
    match identOf t with
    | some v => dotted (acc ++ [v]) r
    | none => .err (t :: r)
  | [.dot] => .err []
  | ts => .ok acc ts

/-- The function registry `get_functions()`: name ↦ (min, max) arity.
The ids 100/101/102 stand for `reduce` / `shortestPath` / `allShortestPaths`,
which divert to their own parsers (1982-1994); those parsers are outside
the fragment and the model returns an error for them. -/
def fnTable : List Nat → Option (Nat × Nat)
  | [7] => some (1, 1)    -- abs
  | [8] => some (1, 1)    -- toUpper
  | [9] => some (0, 0)    -- rand
  | [10] => some (1, 3)   -- a three-argument function, e.g. substring
  | _ => none

/-- The `loop` of `parse_expression_list` (2732-2738) and its closing `)` (2741-2747):
after a `,` an expression must follow. -/
def exprItems (pe : List Tok → Out) (fuel : Nat) (acc : List RT) (ts : List Tok) :
    Res (List RT) :=
  match fuel with
  | 0 => .fuel
  | n + 1 =>
    match pe ts with
    | .ok e r =>
      match r with
      | .comma :: r' => exprItems pe n (acc ++ [e]) r'
      | .rparen :: r' => .ok (acc ++ [e]) r'
      | r' => .err r'
    | .err e => .err e
    | .panic => .panic
    | .fuel => .fuel

/-- `parse_expression_list(ZeroOrMoreClosedBy(RParen))` (2723-2750). #3060 (`fe619ac5f`):
only an *empty* list may end at once (`if !is_end_token(current) { loop {..} }`), so
`f(1,)` is an error; it used to re-test the end token after every `,`. -/
def exprList (pe : List Tok → Out) (fuel : Nat) (acc : List RT) (ts : List Tok) :
    Res (List RT) :=
  if cur ts = .rparen then .ok acc (adv ts) else exprItems pe fuel acc ts

/-- `parse_map` (3111-3141), entered at `{`. -/
def mapBody (pe : List Tok → Out) (fuel : Nat) (acc : List RT) (ts : List Tok) : Res RT :=
  match fuel with
  | 0 => .fuel
  | n + 1 =>
    match parseIdent ts with
    | .ok k r =>
      match r with
      | .colon :: r' =>
        match pe r' with
        | .ok v r'' =>
          match r'' with
          | .comma :: r3 => mapBody pe n (acc ++ [.node (.entry k) [v]]) r3
          | .rbrace :: r3 => .ok (.node .map (acc ++ [.node (.entry k) [v]])) r3
          | r3 => .err r3
        | .err e => .err e
        | .panic => .panic
        | .fuel => .fuel
      | r' => .err r'
    | .err e => .err e
    | .panic => .panic
    | .fuel => .fuel

def parseMap (pe : List Tok → Out) (fuel : Nat) : List Tok → Res RT
  | .lbrace :: .rbrace :: r => .ok (lf .map) r
  | .lbrace :: r => mapBody pe fuel [] r
  | ts => .ok (lf .map) ts

/-- `parse_list_comprehension` (2824-2859); `var IN` already consumed. No
`|` token exists in the fragment, so the projection is the variable. -/
def listCompTail (v : Nat) (l : RT) : Res RT → Res (RT × Bool)
  | .ok c (.rbrack :: r) => .ok (.node (.listComp v) [l, c, lf (.var v)], false) r
  | .ok _ r => .err r
  | .err e => .err e
  | .panic => .panic
  | .fuel => .fuel

def listComp (pe : List Tok → Out) (v : Nat) (ts : List Tok) : Res (RT × Bool) :=
  match pe ts with
  | .ok l (.kw .where_ :: r) => listCompTail v l (pe r)       -- WHERE cond
  | .ok l r => listCompTail v l (.ok (lf (.bool true)) r)     -- no WHERE: `true`
  | .err e => .err e
  | .panic => .panic
  | .fuel => .fuel

/-- `parse_list_literal_or_comprehension` (2778-2822), after `[`.
1) `ident IN` commits to a comprehension; 2) and 3) try a pattern
comprehension, which needs a `|` and so always fails and restores here;
4) a list literal, `recurse` unless it is `[]`. -/
def listDefault : List Tok → Res (RT × Bool)
  | .rbrack :: r => .ok (lf .list, false) r
  | ts => .ok (lf .list, true) ts

def listLit (pe : List Tok → Out) (ts : List Tok) : Res (RT × Bool) :=
  match ts with
  | t :: .kw .in_ :: r =>
    match identOf t with
    | some v => listComp pe v r
    | none => listDefault ts
  | _ => listDefault ts

/-- `parse_primary_expr` (1926-2117) with `allow_pattern_predicate = false`. -/
def primary (pe : List Tok → Out) (fuel : Nat) : List Tok → Res (RT × Bool)
  | .kw .null :: r => .ok (lf .null, false) r
  | .kw .true_ :: r => .ok (lf (.bool true), false) r
  | .kw .false_ :: r => .ok (lf (.bool false), false) r
  | .param p :: r => .ok (lf (.param p), false) r
  | .int n :: r => .ok (lf (.int n), false) r
  | .str s :: r => .ok (lf (.str s), false) r
  | .lbrack :: r => listLit pe r
  | .lbrace :: r =>
    match parseMap pe fuel (.lbrace :: r) with
    | .ok m r' => .ok (m, false) r'
    | .err e => .err e
    | .panic => .panic
    | .fuel => .fuel
  | .lparen :: r => .ok (lf .paren, true) r
  | t :: r =>
    match identOf t with
    | none => .err (t :: r)
    | some v =>
      match dotted [v] r with
      | .ok name (.lparen :: r') =>
        if name = [100] ∨ name = [101] ∨ name = [102] then .err r'  -- reduce/shortestPath: not modelled
        else
          match fnTable name with
          | none => .err r'                      -- "Unknown function" (2007, `?`)
          | some (lo, hi) =>
            match exprList pe fuel [] r' with
            | .ok args r'' =>
              if lo ≤ args.length ∧ args.length ≤ hi then .ok (.node (.func name) args, false) r''
              else .err r''                      -- func.validate (2066)
            | .err e => .err e
            | .panic => .panic
            | .fuel => .fuel
      | .ok _ _ => .ok (lf (.var v), false) r      -- restore_state; parse_ident (2074-2076)
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
  | [] => .err []

/-- `parse_list_operator_expression` (2120-2140), after `[`. A failed
`from`/`to` is *replaced by a default* rather than reported. -/
def orDefault (r : Out) (d : RT) : RT :=
  match r with
  | .ok t _ => t
  | _ => d

/-- After `..`: the `to` expression's result is `toR`, the lexer at `r`. -/
def listOpTo (lhs : RT) (fromR toR : Out) (r : List Tok) : Res RT :=
  match toR.rest r with
  | .rbrack :: r' =>
    .ok (.node .getElems [lhs, orDefault fromR (lf (.int 0)),
      orDefault toR (lf (.int 9223372036854775807))]) r'
  | r' => .err r'

/-- After the `from` expression, whose result is `fromR`. -/
def listOpFrom (pe : List Tok → Out) (lhs : RT) (fromR : Out) (ts : List Tok) : Res RT :=
  match fromR.rest ts with
  | .dotdot :: r =>
    match pe r with
    | .panic => .panic
    | .fuel => .fuel
    | .ok t r' => listOpTo lhs fromR (.ok t r') r
    | .err e => listOpTo lhs fromR (.err e) r
  | .rbrack :: r =>
    match fromR with
    | .ok f _ => .ok (.node .getElem [lhs, f]) r
    | _ => .err r
  | r => .err r

def listOp (pe : List Tok → Out) (lhs : RT) (ts : List Tok) : Res RT :=
  match pe ts with
  | .panic => .panic
  | .fuel => .fuel
  | .ok f r => listOpFrom pe lhs (.ok f r) ts
  | .err e => listOpFrom pe lhs (.err e) ts

-- `parse_map_projection` (3143-3199), after `{`:
/-- One item of a map projection: `.*`, `.prop`, `key: expr` or `var`. -/
def mapProjItem (pe : List Tok → Out) (ts : List Tok) : Res RT :=
  match ts with
  | .dot :: .star :: r => .ok (lf .mapProj) r
  | .dot :: r => match parseIdent r with
    | .ok p r' => .ok (lf (.prop p)) r'
    | .err e => .err e
    | .panic => .panic
    | .fuel => .fuel
  | _ => match parseIdent ts with
    | .ok k (.colon :: r) => match pe r with
      | .ok v r' => .ok (.node (.entry k) [v]) r'
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
    | .ok k r => .ok (.node (.entry k) [lf (.var k)]) r
    | .err e => .err e
    | .panic => .panic
    | .fuel => .fuel

def mapProjBody (pe : List Tok → Out) (fuel : Nat) (items : List RT) (ts : List Tok) : Res RT :=
  match fuel with
  | 0 => .fuel
  | n + 1 =>
    match mapProjItem pe ts with
    | .ok it r =>
      match r with
      | .comma :: r' => mapProjBody pe n (items ++ [it]) r'
      | .rbrace :: r' => .ok (.node .mapProj (items ++ [it])) r'
      | r' => .err r'
    | .err e => .err e
    | .panic => .panic
    | .fuel => .fuel

def mapProj (pe : List Tok → Out) (fuel : Nat) (base : RT) : List Tok → Res RT
  | .rbrace :: r => .ok (.node .mapProj [base]) r
  | ts => mapProjBody pe fuel [base] ts

/-- `parse_labels` (3102-3109); `OrderSet::insert` drops repeats. -/
def labels (acc : List Nat) : List Tok → Res (List Nat)
  | .colon :: t :: r =>
    match identOf t with
    | some l => labels (if l ∈ acc then acc else acc ++ [l]) r
    | none => .err (t :: r)
  | [.colon] => .err []
  | ts => .ok acc ts

/-- Level 10's postfix loop (2553-2577). -/
def postLoop (pe : List Tok → Out) (fuel : Nat) (res : RT) (ts : List Tok) : Res RT :=
  match fuel with
  | 0 => .fuel
  | n + 1 =>
    match ts with
    | .lbrack :: r =>
      match listOp pe res r with
      | .ok res' r' => postLoop pe n res' r'
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
    | .dot :: r =>
      match parseIdent r with
      | .ok p r' => postLoop pe n (.node (.prop p) [res]) r'
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
    | .lbrace :: r =>
      match mapProj pe n res r with
      | .ok res' r' => postLoop pe n res' r'
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
    | _ => .ok res ts

/-- Level 10 (2543-2591): the postfix loop, then an optional `:Label...`. -/
def postStep (pe : List Tok → Out) (fuel : Nat) (res : RT) (K : List Frame) (ts : List Tok) : St :=
  match postLoop pe fuel res ts with
  | .ok res' ts' =>
    match ts' with
    | .colon :: _ =>
      match labels [] ts' with
      | .ok ls r => ret (.node .hasLabels [res', .node .list (ls.map (fun l => lf (.str l)))]) K r
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
    | _ => ret res' K ts'
  | .err e => .err e
  | .panic => .panic
  | .fuel => .fuel

/-- Level 3's `while NOT` counter (2185-2193). -/
def countNots : List Tok → Nat × List Tok
  | .kw .not_ :: r => let p := countNots r; (p.1 + 1, p.2)
  | ts => (0, ts)

/-- The frames level 3 pushes for `n` leading `NOT`s (2193-2203). -/
def notFrames (n : Nat) (K : List Frame) : List Frame :=
  if n = 0 then (3, none) :: K
  else if n % 2 = 1 then (3, some (lf .not_)) :: K
  else (3, some (lf .not_)) :: (3, some (lf .not_)) :: K

/-- Level 9 prefix (2205-2221): an optional `+`, then an optional `-`. -/
def signs : List Tok → Bool × List Tok
  | .plus :: .dash :: r => (true, r)
  | .plus :: r => (false, r)
  | .dash :: r => (true, r)
  | ts => (false, ts)

/-- Level 11 with a tree (2592-2613). `take_out` of a `Paren` child moves
its children up in its place. -/
def closeStep (res : RT) (K : List Frame) (ts : List Tok) : St :=
  match res with
  | .node .paren cs =>
    match ts with
    | .rparen :: r =>
      match cs with
      | [] => .panic                                    -- `child(0)` (2598)
      | .node .paren gcs :: rest => ret (.node .paren (gcs ++ rest)) K r
      | c :: rest => ret (.node .paren (c :: rest)) K r
    | r => .err r
  | .node .list cs =>
    match ts with
    | .comma :: r => .run ((0, none) :: (11, some (.node .list cs)) :: K) r
    | .rbrack :: r => ret (.node .list cs) K r
    | r => .err r
  | _ => ret res K ts

/-- One iteration of `while let Some((current, res, _)) = stack.pop()`. -/
def step (pe : List Tok → Out) (fuel : Nat) : List Frame → List Tok → St
  | [], _ => .panic                                     -- `unreachable!()` (2617)
  | (l, none) :: K, ts =>
    if l < 3 ∨ (3 < l ∧ l < 9) ∨ l = 10 then
      .run ((l + 1, none) :: (l, none) :: K) ts
    else if l = 3 then
      -- #3060 (`fe619ac5f`, 2193-2203): no `NOT` pushes `(3, None)`; a run of `n ≥ 1`
      -- pushes `2 - n % 2` frames `(3, Some(Not))` — one `NOT` for odd `n`, two for even.
      let (n, ts') := countNots ts
      .run ((4, none) :: notFrames n K) ts'
    else if l = 9 then
      let (neg, ts') := signs ts
      .run ((10, none) :: (9, if neg then some (lf .neg) else none) :: K) ts'
    else
      match primary pe fuel ts with
      | .ok (res, true) ts' => .run ((0, none) :: (l, some res) :: K) ts'
      | .ok (res, false) ts' => ret res K ts'
      | .err e => .err e
      | .panic => .panic
      | .fuel => .fuel
  | (l, some res) :: K, ts =>
    match l with
    | 0 | 1 | 2 | 6 | 7 | 8 => binStep l res K ts
    | 3 => ret res K ts
    | 4 => cmpStep res K ts
    | 5 => predStep res K ts
    | 9 =>
      -- the i64::MIN special case (2524-2540) reads `child(0)` of a Negate
      match res with
      | .node .neg [] => .panic
      | _ => ret res K ts
    | 10 => postStep pe fuel res K ts
    | 11 => closeStep res K ts
    | _ => .panic                                       -- `unreachable!()` (2614)

/-- The loop, with a step budget. -/
def loop (pe : List Tok → Out) : Nat → St → Out
  | _, .done t ts => .ok t ts
  | _, .err e => .err e
  | _, .panic => .panic
  | _, .fuel => .fuel
  | 0, .run _ _ => .fuel
  | n + 1, .run K ts => loop pe n (step pe n K ts)

/-- `parse_expr` / `parse_expr_inner` (2159-2177): start from `[(0, None)]`.
Nested `parse_expr` calls (in calls, maps, indexes, comprehensions) use one
unit less of budget. -/
def parseExpr : Nat → List Tok → Out
  | 0, _ => .fuel
  | n + 1, ts => loop (parseExpr n) n (.run [(0, none)] ts)

/-- A whole `RETURN <expr>` style parse: the expression must be followed by `eof`. -/
def parseAll (n : Nat) (ts : List Tok) : Out :=
  match parseExpr n ts with
  | .ok t r => if cur r = .eof then .ok t r else .err r
  | o => o

end FalkorParserGrammar
