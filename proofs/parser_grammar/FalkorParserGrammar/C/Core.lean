/-
# Clause level: tokens, trees, the parser monad and its primitives

The clause parsers of `graph/src/parser/cypher.rs` (line numbers are from
origin/main, commit 3fec7d7c9; identical
to 55204c94b for every parser file) are modelled over a token type `CT`
that has every `Token` / `Keyword` of `lexer.rs:63-156`.

Expressions are an *oracle*: `parse_expr`, `parse_primary_expr(false)`,
`parse_map` and the projection-text slice are parameters (`Oracle`). Every
soundness theorem holds for every oracle, and every no-panic theorem holds
for every oracle that does not panic. The expression parser itself is the
subject of `FalkorParserGrammar.Model` (`parseExpr_no_panic`,
`bounded_agree`, ...).
-/
namespace FalkorParserGrammar.C

/-- `Keyword` (lexer.rs:63-123), all 59 of them. -/
inductive CK
  | call | yield_ | optional | match_ | unwind | merge | create | detach | delete | set
  | remove | where_ | with_ | return_ | as_ | null | or_ | xor | and_ | not_ | is_ | in_
  | starts | ends | contains | true_ | false_ | case_ | when | then | else_ | end_ | all
  | any | none | single | distinct | order | by | asc | ascending | desc | descending
  | skip | limit | load | csv | headers | from | fieldterminator | drop | index | fulltext
  | vector | options | for_ | foreach | on | union
  deriving DecidableEq, Repr

/-- `Token` (lexer.rs:125-157). `lbrack`/`rbrack` are `[ ]` (Rust `LBrace`/`RBrace`),
`lbrace`/`rbrace` are `{ }` (Rust `LBracket`/`RBracket`). `op` is any other
operator token (`+ / % ^ <> <= >= =~`). `float` carries no value. -/
inductive CT
  | kw (k : CK) | ident (v : Nat) | param (p : Nat) | int (i : Int) | float | str (s : Nat)
  | lbrack | rbrack | lbrace | rbrace | lparen | rparen
  | star | dash | eq | plusEq | lt | gt | comma | colon | dot | dotdot | pipe | semi
  | op (n : Nat) | eof
  deriving DecidableEq, Repr

/-- The identifier text of a keyword token (its spelling, as a name id). -/
def kwId (k : CK) : Nat := 1000000 + (match k with
  | .call => 0 | .yield_ => 1 | .optional => 2 | .match_ => 3 | .unwind => 4 | .merge => 5
  | .create => 6 | .detach => 7 | .delete => 8 | .set => 9 | .remove => 10 | .where_ => 11
  | .with_ => 12 | .return_ => 13 | .as_ => 14 | .null => 15 | .or_ => 16 | .xor => 17
  | .and_ => 18 | .not_ => 19 | .is_ => 20 | .in_ => 21 | .starts => 22 | .ends => 23
  | .contains => 24 | .true_ => 25 | .false_ => 26 | .case_ => 27 | .when => 28
  | .then => 29 | .else_ => 30 | .end_ => 31 | .all => 32 | .any => 33 | .none => 34
  | .single => 35 | .distinct => 36 | .order => 37 | .by => 38 | .asc => 39
  | .ascending => 40 | .desc => 41 | .descending => 42 | .skip => 43 | .limit => 44
  | .load => 45 | .csv => 46 | .headers => 47 | .from => 48 | .fieldterminator => 49
  | .drop => 50 | .index => 51 | .fulltext => 52 | .vector => 53 | .options => 54
  | .for_ => 55 | .foreach => 56 | .on => 57 | .union => 58)

/-- `Token::IdentifierOrKeyword { ident, .. }` ↦ `ident`. -/
def identOf : CT → Option Nat
  | .ident v => some v
  | .kw k => some (kwId k)
  | _ => none

/-- The name `format!("_anon_{n}")` (cypher.rs:2935, 2972, 3043) as a name id.
A user may spell the same identifier (`_anon_0` is a valid name). -/
def anonName (k : Nat) : Nat := 2000000 + k

/-! ## Expression trees (`ExprIR`, ast.rs) as far as the clause level looks into them -/

inductive EOp
  | var (v : Nat) | int (i : Int) | param (p : Nat) | str (s : Nat) | null | bool (b : Bool)
  | prop (p : Nat) | func (f : Nat) (agg : Bool) | list | map | paren | distinct
  | case_ (subject : Bool) | quant (q : Nat) (v : Nat) | listComp (v : Nat)
  | reduce (acc it : Nat) | patComp | shortest (types : List Nat) (mn : Nat) (mx : Option Nat) (directed : Bool)
  | pattern | other (n : Nat)
  deriving DecidableEq, Repr

inductive ET
  | node (op : EOp) (cs : List ET)
  deriving Repr

def ET.root : ET → EOp | .node o _ => o
def ET.kids : ET → List ET | .node _ cs => cs
abbrev leaf (o : EOp) : ET := .node o []

/- `find_aggregate_name` (cypher.rs:3442-3453): the first aggregate function
met by a preorder DFS. -/
mutual
def aggName : ET → Option Nat
  | .node (.func f true) _ => some f
  | .node _ cs => aggNameL cs
def aggNameL : List ET → Option Nat
  | [] => none
  | c :: cs => match aggName c with
    | some f => some f
    | none => aggNameL cs
end

/- The specification: an aggregate occurs somewhere in the tree. -/
mutual
inductive HasAgg : ET → Prop
  | here (f : Nat) (cs : List ET) : HasAgg (.node (.func f true) cs)
  | inner (o : EOp) (cs : List ET) : HasAggL cs → HasAgg (.node o cs)
inductive HasAggL : List ET → Prop
  | head (c : ET) (cs : List ET) : HasAgg c → HasAggL (c :: cs)
  | tail (c : ET) (cs : List ET) : HasAggL cs → HasAggL (c :: cs)
end

mutual
theorem aggName_sound : ∀ t : ET, (aggName t).isSome → HasAgg t
  | .node (.func f true) cs, _ => .here f cs
  | .node (.var _) cs, h | .node (.int _) cs, h | .node (.param _) cs, h
  | .node (.str _) cs, h | .node .null cs, h | .node (.bool _) cs, h
  | .node (.prop _) cs, h | .node (.func _ false) cs, h | .node .list cs, h
  | .node .map cs, h | .node .paren cs, h | .node .distinct cs, h
  | .node (.case_ _) cs, h | .node (.quant _ _) cs, h | .node (.listComp _) cs, h
  | .node (.reduce _ _) cs, h | .node .patComp cs, h | .node (.shortest _ _ _ _) cs, h
  | .node .pattern cs, h
  | .node (.other _) cs, h => .inner _ cs (aggNameL_sound cs (by simpa [aggName] using h))
theorem aggNameL_sound : ∀ cs : List ET, (aggNameL cs).isSome → HasAggL cs
  | [], h => by simp [aggNameL] at h
  | c :: cs, h => by
    unfold aggNameL at h
    cases hc : aggName c with
    | some f => exact .head c cs (aggName_sound c (by simp [hc]))
    | none => rw [hc] at h; exact .tail c cs (aggNameL_sound cs h)
end

mutual
theorem aggName_complete : ∀ t : ET, HasAgg t → (aggName t).isSome
  | _, .here f cs => by simp [aggName]
  | _, .inner o cs h => by
    have := aggNameL_complete cs h
    cases o with
    | func f b => cases b <;> simp [aggName, this]
    | _ => simp [aggName, this]
theorem aggNameL_complete : ∀ cs : List ET, HasAggL cs → (aggNameL cs).isSome
  | _, .head c cs h => by
    have := aggName_complete c h
    unfold aggNameL; cases hc : aggName c <;> simp_all
  | _, .tail c cs h => by
    have := aggNameL_complete cs h
    unfold aggNameL; cases hc : aggName c <;> simp_all
end

/-- **find_aggregate_name** is exactly "the tree contains an aggregate". -/
theorem find_aggregate_name_spec (t : ET) : (aggName t).isSome ↔ HasAgg t :=
  ⟨aggName_sound t, aggName_complete t⟩

/-! ## Parser state and results -/

/-- `Parser { lexer, anon_counter, .. }`: remaining tokens and the counter.
`depth`, `expr_height`, `max_child_height` are the subject of `proofs/lexer`. -/
abbrev PS := List CT × Nat

def PS.cur (s : PS) : CT := match s.1 with | [] => .eof | t :: _ => t
def PS.adv (s : PS) : PS := (s.1.tail, s.2)

inductive R (α : Type)
  | ok (a : α) (s : PS)
  | err
  | panic   -- `unreachable!()` / an `unwrap` that fails
  | fuel    -- the model's loop budget ran out (not a Rust behaviour)

def P (α : Type) := PS → R α

def P.pure {α} (a : α) : P α := fun s => .ok a s
def P.bind {α β} (x : P α) (f : α → P β) : P β := fun s =>
  match x s with
  | .ok a s' => f a s'
  | .err => .err
  | .panic => .panic
  | .fuel => .fuel

instance : Monad P where
  pure := P.pure
  bind := P.bind

def fail {α} : P α := fun _ => .err
def oops {α} : P α := fun _ => .panic
def getS : P PS := fun s => .ok s s
def setS (s : PS) : P Unit := fun _ => .ok () s

@[simp] theorem run_pure {α} (a : α) (s : PS) : (pure a : P α) s = .ok a s := rfl
@[simp] theorem run_bind {α β} (x : P α) (f : α → P β) (s : PS) :
    (x >>= f) s = (match x s with
      | .ok a s' => f a s' | .err => .err | .panic => .panic | .fuel => .fuel) := rfl
@[simp] theorem run_fail {α} (s : PS) : (fail : P α) s = .err := rfl
@[simp] theorem run_oops {α} (s : PS) : (oops : P α) s = .panic := rfl
@[simp] theorem run_getS (s : PS) : getS s = .ok s s := rfl
@[simp] theorem run_setS (s s' : PS) : setS s' s = .ok () s' := rfl

theorem bind_ok {α β} {x : P α} {f : α → P β} {s : PS} {b : β} {s' : PS} :
    (x >>= f) s = .ok b s' → ∃ a s1, x s = .ok a s1 ∧ f a s1 = .ok b s' := by
  intro h; simp only [run_bind] at h
  split at h <;> first | exact ⟨_, _, ‹_›, h⟩ | cases h

theorem bind_np {α β} {x : P α} {f : α → P β} {s : PS}
    (hx : x s ≠ .panic) (hf : ∀ a s1, x s = .ok a s1 → f a s1 ≠ .panic) :
    (x >>= f) s ≠ .panic := by
  simp only [run_bind]
  split
  · exact hf _ _ ‹_›
  · simp
  · exact absurd ‹_› hx
  · simp

/-- `peel h => a s hx`: split the first bind of `h : (x >>= f) s0 = .ok b s'`
into `hx : x s0 = .ok a s` and the rest `h : f a s = .ok b s'`. -/
syntax "peel " ident " => " ident ident ident : tactic
macro_rules
  | `(tactic| peel $h => $a $s $hx) =>
    `(tactic| (have hb := bind_ok $h; clear $h; have ⟨$a, $s, $hx, $h⟩ := hb; (try dsimp only at $h:ident); clear hb))

theorem pure_ok {α} {a b : α} {s s' : PS} (h : (pure a : P α) s = .ok b s') : a = b ∧ s = s' := by
  simp only [run_pure, R.ok.injEq] at h; exact h

/-- "never panics, from any state". -/
structure NP {α} (x : P α) : Prop where
  run : ∀ s, x s ≠ .panic

theorem np_bind {α β} {x : P α} {f : α → P β} (hx : NP x) (hf : ∀ a, NP (f a)) :
    NP (x >>= f) := ⟨fun s => bind_np (hx.run s) (fun a s1 _ => (hf a).run s1)⟩
theorem np_pure {α} (a : α) : NP (pure a : P α) := ⟨fun _ => by simp⟩
theorem np_fail {α} : NP (fail : P α) := ⟨fun _ => by simp⟩
theorem np_ite {α} {c : Prop} [Decidable c] {x y : P α} (hx : NP x) (hy : NP y) :
    NP (if c then x else y) := by split <;> assumption

/-! ## Token primitives (`match_token!` / `optional_match_token!`, macro.rs:51-113) -/

/-- `match_token!(lexer, T)` / `match_token!(lexer => K)`. -/
def tok (t : CT) : P Unit := fun s => if s.cur = t then .ok () s.adv else .err
/-- `optional_match_token!`. -/
def opt (t : CT) : P Bool := fun s => if s.cur = t then .ok true s.adv else .ok false s
/-- `self.lexer.current()` as a value. -/
def peek : P CT := fun s => .ok s.cur s
/-- `self.lexer.next()`. -/
def next : P Unit := fun s => .ok () s.adv

/-- `parse_ident` (cypher.rs:2642-2654); identifier-length validation
(`validate_identifier_len`) is not modelled (it can only add errors). -/
def ident : P Nat := fun s => match identOf s.cur with
  | some v => .ok v s.adv
  | none => .err

/-- `try_parse_ident` (2661-2671): `parse_ident` without the error. -/
def tryIdent : P (Option Nat) := fun s => match identOf s.cur with
  | some v => .ok (some v) s.adv
  | none => .ok none s

/-- `parse_property_name` (2673-2685): the same as `parse_ident`. -/
def propName : P Nat := ident

theorem np_tok (t : CT) : NP (tok t) := ⟨fun s => by unfold tok; split <;> simp⟩
theorem np_opt (t : CT) : NP (opt t) := ⟨fun s => by unfold opt; split <;> simp⟩
theorem np_peek : NP peek := ⟨fun _ => by simp [peek]⟩
theorem np_next : NP next := ⟨fun _ => by simp [next]⟩
theorem np_ident : NP ident := ⟨fun s => by unfold ident; split <;> simp⟩
theorem np_tryIdent : NP tryIdent := ⟨fun s => by unfold tryIdent; split <;> simp⟩

theorem tok_ok {t s u s'} : tok t s = .ok u s' → s.cur = t ∧ s' = s.adv := by
  unfold tok; split <;> simp_all
theorem opt_ok {t s b s'} : opt t s = .ok b s' →
    (b = true ∧ s.cur = t ∧ s' = s.adv) ∨ (b = false ∧ s.cur ≠ t ∧ s' = s) := by
  unfold opt; split
  · intro h; cases h; exact .inl ⟨rfl, ‹_›, rfl⟩
  · intro h; cases h; exact .inr ⟨rfl, ‹_›, rfl⟩
theorem ident_ok {s v s'} : ident s = .ok v s' → identOf s.cur = some v ∧ s' = s.adv := by
  unfold ident; split <;> simp_all
theorem tryIdent_ok {s o s'} : tryIdent s = .ok o s' →
    (∃ v, o = some v ∧ identOf s.cur = some v ∧ s' = s.adv) ∨
    (o = none ∧ identOf s.cur = none ∧ s' = s) := by
  unfold tryIdent; split
  · intro h; cases h; exact .inl ⟨_, rfl, ‹_›, rfl⟩
  · intro h; cases h; exact .inr ⟨rfl, ‹_›, rfl⟩
theorem peek_ok {s c s'} : peek s = .ok c s' → c = s.cur ∧ s' = s := by
  simp [peek]; intro h1 h2; exact ⟨h1.symm, h2.symm⟩
theorem next_ok {s u s'} : next s = .ok u s' → s' = s.adv := by
  simp [next]; intro h; exact h.symm

/-- **parse_ident / try_parse_ident / parse_property_name agree**: on an
identifier token all three consume it and return its name; otherwise
`parse_ident` fails and `try_parse_ident` consumes nothing. -/
theorem ident_tryIdent (s : PS) :
    (ident s = .err ∧ tryIdent s = .ok none s) ∨
    (∃ v, ident s = .ok v s.adv ∧ tryIdent s = .ok (some v) s.adv) := by
  unfold ident tryIdent; split <;> simp_all

/-! ## The expression oracle -/

/-- A procedure / function registry entry: arity bounds, default outputs. -/
structure FnInfo where
  lo : Nat
  hi : Nat
  outputs : List Nat
  isProc : Bool

/-- What the clause parsers take from the expression level. -/
structure Oracle where
  /-- `parse_expr(allow_pattern_predicate)` (cypher.rs:2159). -/
  pe : Bool → P ET
  /-- `parse_primary_expr(false)` (cypher.rs:1926): tree and `recurse`. -/
  pp : P (ET × Bool)
  /-- `parse_map` (cypher.rs:3111), modelled in `Model.parseMap`. -/
  pmap : P ET
  /-- `self.lexer.str[pos..pos(true)]` of a projection, as a name id. -/
  txt : PS → PS → Nat
  /-- `get_functions().get(name, Procedure)` (cypher.rs:1036). -/
  proc : List Nat → Option FnInfo

/-- An oracle that never panics (the expression model satisfies this:
`parseExpr_no_panic`). -/
structure Oracle.Safe (o : Oracle) : Prop where
  pe : ∀ b, NP (o.pe b)
  pp : NP o.pp
  pmap : NP o.pmap

end FalkorParserGrammar.C
