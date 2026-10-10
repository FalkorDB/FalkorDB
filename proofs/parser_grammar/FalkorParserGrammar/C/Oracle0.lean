/-
# A small concrete oracle, for running the clause models on examples
-/
import FalkorParserGrammar.C.Pattern

namespace FalkorParserGrammar.C

/-- An expression is a single literal / parameter / variable, optionally
followed by `.prop` lookups. -/
def atom0 : P ET := fun s =>
  match s.cur with
  | .int i => .ok (leaf (.int i)) s.adv
  | .str t => .ok (leaf (.str t)) s.adv
  | .param p => .ok (leaf (.param p)) s.adv
  | .kw .null => .ok (leaf .null) s.adv
  | .ident v => .ok (leaf (.var v)) s.adv
  | _ => .err

def props0 : Nat → ET → P ET
  | 0, e => pure e
  | n + 1, e => fun s =>
    match s.cur, identOf s.adv.cur with
    | .dot, some p => props0 n (.node (.prop p) [e]) s.adv.adv
    | _, _ => .ok e s

def pe0 : Bool → P ET := fun _ => atom0 >>= props0 8
def pp0 : P (ET × Bool) := fun s =>
  match s.cur with
  | .lparen => .ok (leaf .paren, true) s.adv
  | .lbrack => .ok (leaf .list, true) s.adv
  | _ => (atom0 >>= fun e => pure (e, false)) s
/-- `{}` only. -/
def pmap0 : P ET := fun s =>
  if s.cur = .lbrace ∧ s.adv.cur = .rbrace then .ok (leaf .map) s.adv.adv else .err

def o0 : Oracle := ⟨pe0, pp0, pmap0, fun _ _ => 0, fun f => if f = [77] then some ⟨0, 1, [5], true⟩ else none⟩

def run {α} (x : P α) (ts : List CT) : R α := x (ts, 0)

def R.isOk {α} : R α → Bool | .ok _ _ => true | _ => false
def R.get? {α} : R α → Option α | .ok a _ => some a | _ => none

end FalkorParserGrammar.C
