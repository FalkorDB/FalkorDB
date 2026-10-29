/-
# The pattern-comprehension gate and memo of #3104 (cypher.rs:2818-2912)

#3104 (`4a4f2a72f`) stops the exponential backtracking of
`parse_list_literal_or_comprehension` (W3-parser-9) two ways:

* a **gate**: alternatives 2 and 3 (pattern comprehensions) are only tried
  after `[` when the next token is `(` or a name and `list_has_top_level_pipe`
  finds a `|` before the list's closing `]`, outside nested brackets and
  before any `,` there (`pipeScan`, `gateOK`, in `C.ExprC2`);
* a **memo** `not_pattern_comprehension` of attempts that failed, keyed by
  `(pos, allow_pattern_predicate, forbidden, depth)`: a failed attempt is
  never repeated with the same key (`Oracle.seen`).

What is proved here:

1. **The lookahead is exact on well-bracketed input** (`pipeScan_exact`): the
   single counter over `( [ {` agrees with a kind-matched bracket stack
   (`stackScan`) whenever the stack never sees a mismatched closer — the
   counter *is* the stack depth. On mismatched input the two can differ
   (`pipeScan_mismatch`), which only decides what is tried first.
2. **The gate and memo do not change the parse** (`listLitP_eq_ungated`):
   `listLitP` (merged) and `listLitPU` (before #3104) return the same result
   on every state where the old parser does not run out of model fuel,
   *provided* (a) the expression and map parsers only ever consume
   bracket-balanced token runs with no top-level `|` or `,` (`Oracle.Bal`),
   and (b) the memo only holds attempts that fail (`MemoSound`).
   (a) is what makes a successful pattern comprehension imply an open gate
   (`patCompP_pipe`, through `parse_node_pattern` / `parse_relationship_pattern`).
   **(a) was false for the Rust expression parser before #3041**: a failed
   slice bound was swallowed, so
   `[1][(1..2]` was accepted while consuming an unmatched `(`.
   The resulting counterexample (removed; in 17ccc5412):
   `[(a)-->(b) WHERE [1][(1..2] | b]` was a pattern comprehension before #3104
   and a syntax error after it (CONFIRMED live, see the project header).
   #3041 (`2023af648`) makes a failed bound an error, which removes that
   counterexample: `[1][(1..2]` is refused (`slice_unclosed_paren_refused`) and
   the expression fragment of `Model.lean` consumes balanced runs on every
   input checked (`bounded_bal`). (a) is not proved for the whole Rust
   expression parser (the oracle covers CASE, comprehensions, maps, ... that
   the fragment lacks), so the theorem stays conditional on it. (b) holds when an attempt's outcome is a function of
   the key (`memo_hits_fail`); the key leaves out `anon_counter`, which the
   outcome can read through anonymous names (see the project header).
3. **Failed attempts are linear in the input** (`failed_attempts_linear`):
   along any run, each key fails at most once, so with keys bounded by the
   input length `n`, `MAX_NESTING = 100` and the three
   `ForbiddenPatternComprehension` places, at most `808 * (n + 1)` attempts
   fail. Successful attempts are not memoised; the bound is on the
   re-parsing that made W3-parser-9 exponential.
-/
import FalkorParserGrammar.C.ExprC4

namespace FalkorParserGrammar.C

/-! ## 1. `list_has_top_level_pipe` against a kind-matched bracket stack -/

inductive BK | paren | brack | brace
  deriving DecidableEq, Repr

def openK : CT → Option BK
  | .lparen => some .paren | .lbrack => some .brack | .lbrace => some .brace | _ => none
def closeK : CT → Option BK
  | .rparen => some .paren | .rbrack => some .brack | .rbrace => some .brace | _ => none

/-- The specification: scan with a stack of open bracket *kinds*; `none` on a
closer that does not match the innermost opener. A `|` with an empty stack
is `true`; a `,` there, a closer with nothing open, or the end is `false`. -/
def stackScan : List BK → List CT → Option Bool
  | _, [] => some false
  | st, t :: r =>
    match openK t, closeK t with
    | some k, _ => stackScan (k :: st) r
    | none, some k =>
      match st with
      | [] => some false
      | k' :: st' => if k = k' then stackScan st' r else none
    | none, none =>
      if t = .pipe then (if st = [] then some true else stackScan st r)
      else if t = .comma then (if st = [] then some false else stackScan st r)
      else if t = .eof then some false
      else stackScan st r

/-- **The one-counter lookahead is exact on well-bracketed input**: wherever
the kind-matched scan decides (never meets a mismatched closer), the Rust's
single counter, started at the stack depth, decides the same. -/
theorem pipeScan_exact : ∀ (ts : List CT) (st : List BK) (b : Bool),
    stackScan st ts = some b → pipeScan st.length ts = b
  | [], st, b, h => by simp [stackScan] at h; simp [pipeScan, h]
  | t :: r, st, b, h => by
    have ih := pipeScan_exact r
    cases t
    case lparen | lbrack | lbrace =>
      simp only [stackScan, openK] at h; simp only [pipeScan]
      simpa using ih _ _ h
    case rparen | rbrack | rbrace =>
      simp only [pipeScan]
      cases st with
      | nil => simp [stackScan, openK, closeK] at h; simp [h]
      | cons k st' =>
        simp only [stackScan, openK, closeK] at h
        simp only [List.length_cons, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel]
        split at h
        · exact ih _ _ h
        · simp at h
    case pipe =>
      simp only [stackScan, openK, closeK, ite_true] at h; simp only [pipeScan]
      cases st with
      | nil => simp at h; simp [h]
      | cons k st' => simp at h; simpa using ih _ _ h
    case comma =>
      simp only [stackScan, openK, closeK] at h; simp only [pipeScan]
      cases st with
      | nil => simp at h; simp [h]
      | cons k st' => simp at h; simpa using ih _ _ h
    case eof =>
      simp [stackScan, openK, closeK] at h; simp [pipeScan, h]
    all_goals (simp [stackScan, openK, closeK] at h; simp only [pipeScan]; exact ih _ _ h)

/-- On a top-level scan (`stackScan []`): exactness from depth 0. -/
theorem pipeScan_exact0 (ts : List CT) (b : Bool) (h : stackScan [] ts = some b) :
    pipeScan 0 ts = b := pipeScan_exact ts [] b h

/-- Counting kinds together can say yes where matching them says the list is
malformed: `[([) | 1]` (the Rust test `mismatched_brackets_in_a_list_are_still_rejected`).
Only the order of attempts depends on it; the parse itself still rejects. -/
theorem pipeScan_mismatch :
    pipeScan 0 [.lparen, .lbrack, .rparen, .rbrack, .pipe, .int 1, .rbrack] = true ∧
    stackScan [] [.lparen, .lbrack, .rparen, .rbrack, .pipe, .int 1, .rbrack] = none := by decide

example : pipeScan 0 [.lparen, .ident 1, .rparen, .dash, .dash, .gt, .lparen, .ident 2, .rparen,
    .pipe, .ident 2, .rbrack] = true ∧
    pipeScan 0 [.lparen, .ident 1, .lbrace, .ident 3, .colon, .int 1, .rbrace, .rparen, .rbrack] = false ∧
    pipeScan 0 [.lparen, .int 1, .rparen, .comma, .int 2, .pipe, .rbrack] = false := by decide

/-! ## 2. Consumed runs that the scan sees through -/

/-- `pre` takes the scan from depth `d` to depth `d'` without deciding. -/
def PStep (d d' : Nat) (pre : List CT) : Prop := ∀ rest, pipeScan d (pre ++ rest) = pipeScan d' rest

theorem step_nil (d : Nat) : PStep d d [] := fun _ => rfl

theorem step_append {d e f : Nat} {a b : List CT} (ha : PStep d e a) (hb : PStep e f b) :
    PStep d f (a ++ b) := fun rest => by rw [List.append_assoc, ha, hb]

/-- A token that is no bracket, `|`, `,` or end of input. -/
def isAtom (t : CT) : Bool :=
  match t with
  | .lparen | .lbrack | .lbrace | .rparen | .rbrack | .rbrace | .pipe | .comma | .eof => false
  | _ => true

theorem step_atom {t : CT} (h : isAtom t = true) (d : Nat) : PStep d d [t] := fun rest => by
  cases t <;> simp [isAtom] at h <;> simp [pipeScan]

theorem step_inner {t : CT} (h : t = .pipe ∨ t = .comma) (d : Nat) : PStep (d + 1) (d + 1) [t] :=
  fun rest => by rcases h with rfl | rfl <;> simp [pipeScan]

theorem step_open {t : CT} (h : t = .lparen ∨ t = .lbrack ∨ t = .lbrace) (d : Nat) :
    PStep d (d + 1) [t] := fun rest => by rcases h with rfl | rfl | rfl <;> simp [pipeScan]

theorem step_close {t : CT} (h : t = .rparen ∨ t = .rbrack ∨ t = .rbrace) (d : Nat) :
    PStep (d + 1) d [t] := fun rest => by rcases h with rfl | rfl | rfl <;> simp [pipeScan]

/-- `x` succeeds only by consuming a run taking the scan from `d` to `d'`. -/
def Cn {α} (x : P α) (d d' : Nat) : Prop :=
  ∀ s a s', x s = .ok a s' → ∃ pre, s.1 = pre ++ s'.1 ∧ PStep d d' pre

/-- The same, from states whose current token satisfies `p`. -/
def Cnc {α} (p : CT → Prop) (x : P α) (d d' : Nat) : Prop :=
  ∀ s a s', p s.cur → x s = .ok a s' → ∃ pre, s.1 = pre ++ s'.1 ∧ PStep d d' pre

theorem cn_bind {α β} {x : P α} {f : α → P β} {d e g : Nat} (hx : Cn x d e)
    (hf : ∀ a, Cn (f a) e g) : Cn (x >>= f) d g := by
  intro s b s' h
  peel h => a s1 h1
  obtain ⟨p1, e1, st1⟩ := hx _ _ _ h1
  obtain ⟨p2, e2, st2⟩ := hf a _ _ _ h
  exact ⟨p1 ++ p2, by rw [e1, e2, List.append_assoc], step_append st1 st2⟩

theorem cnc_bind {α β} {p : CT → Prop} {x : P α} {f : α → P β} {d e g : Nat} (hx : Cnc p x d e)
    (hf : ∀ a, Cn (f a) e g) : Cnc p (x >>= f) d g := by
  intro s b s' hp h
  peel h => a s1 h1
  obtain ⟨p1, e1, st1⟩ := hx _ _ _ hp h1
  obtain ⟨p2, e2, st2⟩ := hf a _ _ _ h
  exact ⟨p1 ++ p2, by rw [e1, e2, List.append_assoc], step_append st1 st2⟩

theorem cn_peek_bind {β} {f : CT → P β} {d g : Nat} (hf : ∀ c, Cnc (· = c) (f c) d g) :
    Cn (peek >>= f) d g := by
  intro s b s' h
  simp only [run_bind, peek] at h
  exact hf s.cur s b s' rfl h

theorem cnc_of_cn {α} {p : CT → Prop} {x : P α} {d d' : Nat} (h : Cn x d d') : Cnc p x d d' :=
  fun s a s' _ hx => h s a s' hx

theorem cn_pure {α} (a : α) (d : Nat) : Cn (pure a : P α) d d := by
  intro s b s' h; obtain ⟨-, rfl⟩ := pure_ok h; exact ⟨[], rfl, step_nil d⟩

theorem cn_fail {α} (d d' : Nat) : Cn (fail : P α) d d' := by intro s b s' h; simp at h
theorem cn_fuelP {α} (d d' : Nat) : Cn (fuelP : P α) d d' := by intro s b s' h; simp at h

theorem cn_ite {α} {c : Prop} [Decidable c] {x y : P α} {d d' : Nat} (hx : Cn x d d') (hy : Cn y d d') :
    Cn (if c then x else y) d d' := by split <;> assumption

theorem cn_peek (d : Nat) : Cn peek d d := by
  intro s a s' h; obtain ⟨-, rfl⟩ := peek_ok h; exact ⟨[], rfl, step_nil d⟩

theorem cn_anonFresh (d : Nat) : Cn anonFresh d d := by
  intro s a s' h; simp only [anonFresh, R.ok.injEq] at h; obtain ⟨-, rfl⟩ := h
  exact ⟨[], by simp, step_nil d⟩

theorem cn_tok_gen {t : CT} {d d' : Nat} (he : t ≠ .eof) (hs : PStep d d' [t]) : Cn (tok t) d d' := by
  intro s a s' h
  obtain ⟨hc, rfl⟩ := tok_ok h
  refine ⟨[t], ?_, hs⟩
  obtain ⟨l, n⟩ := s
  cases l with
  | nil => simp [PS.cur] at hc; exact absurd hc.symm he
  | cons x r => simp [PS.cur] at hc; subst hc; rfl

theorem cn_opt_gen {t : CT} {d : Nat} (he : t ≠ .eof) (hs : PStep d d [t]) : Cn (opt t) d d := by
  intro s a s' h
  rcases opt_ok h with ⟨-, hc, rfl⟩ | ⟨-, _, rfl⟩
  · refine ⟨[t], ?_, hs⟩
    obtain ⟨l, n⟩ := s
    cases l with
    | nil => simp [PS.cur] at hc; exact absurd hc.symm he
    | cons x r => simp [PS.cur] at hc; subst hc; rfl
  · exact ⟨[], rfl, step_nil d⟩

theorem atom_ne_eof {t : CT} (h : isAtom t = true) : t ≠ .eof := by rintro rfl; simp [isAtom] at h

theorem cn_tok_atom {t : CT} (h : isAtom t = true) (d : Nat) : Cn (tok t) d d :=
  cn_tok_gen (atom_ne_eof h) (step_atom h d)
theorem cn_opt_atom {t : CT} (h : isAtom t = true) (d : Nat) : Cn (opt t) d d :=
  cn_opt_gen (atom_ne_eof h) (step_atom h d)

theorem identOf_atom {t : CT} {v : Nat} (h : identOf t = some v) : isAtom t = true := by
  cases t <;> simp [identOf] at h <;> rfl

theorem cn_ident (d : Nat) : Cn ident d d := by
  intro s a s' h
  obtain ⟨hv, rfl⟩ := ident_ok h
  obtain ⟨l, n⟩ := s
  cases l with
  | nil => simp [PS.cur, identOf] at hv
  | cons x r =>
    simp [PS.cur] at hv
    exact ⟨[x], rfl, step_atom (identOf_atom hv) d⟩

theorem cn_tryIdent (d : Nat) : Cn tryIdent d d := by
  intro s a s' h
  rcases tryIdent_ok h with ⟨v, -, hv, rfl⟩ | ⟨-, -, rfl⟩
  · obtain ⟨l, n⟩ := s
    cases l with
    | nil => simp [PS.cur, identOf] at hv
    | cons x r =>
      simp [PS.cur] at hv
      exact ⟨[x], rfl, step_atom (identOf_atom hv) d⟩
  · exact ⟨[], rfl, step_nil d⟩

theorem cnc_next_atom {c : CT} (h : isAtom c = true) (d : Nat) : Cnc (· = c) next d d := by
  intro s a s' hp hn
  have := next_ok hn; subst this
  obtain ⟨l, n⟩ := s
  cases l with
  | nil => simp [PS.cur] at hp; subst hp; simp [isAtom] at h
  | cons x r => simp [PS.cur] at hp; subst hp; exact ⟨[x], rfl, step_atom h d⟩

/-- `opt [` followed by a continuation that closes it when it was there. -/
theorem cn_opt_lbrack_bind {β} {f : Bool → P β} {d : Nat}
    (h1 : Cn (f true) (d + 1) d) (h2 : Cn (f false) d d) : Cn (opt .lbrack >>= f) d d := by
  intro s b s' h
  peel h => det s1 h1'
  rcases opt_ok h1' with ⟨rfl, hc, rfl⟩ | ⟨rfl, _, rfl⟩
  · obtain ⟨p, e, st⟩ := h1 _ _ _ h
    obtain ⟨l, n⟩ := s
    cases l with
    | nil => simp [PS.cur] at hc
    | cons x r =>
      simp [PS.cur] at hc; subst hc
      exact ⟨[.lbrack] ++ p, by simp [PS.adv] at e ⊢; exact e,
        step_append (step_open (.inr (.inl rfl)) d) st⟩
  · exact h2 _ _ _ h

/-! ## 3. Patterns consume balanced runs -/

/-- The hypothesis on the expression level: `parse_expr` and `parse_map`
only succeed by consuming a run that is bracket-balanced and has no `|` or `,`
outside its brackets — at every depth. True of a correct expression parser;
false of the Rust one before #3041; no counterexample is known on main, and the
fragment model satisfies it on all inputs checked (`bounded_bal`). -/
structure Oracle.Bal (o : Oracle) : Prop where
  pe : ∀ b d, Cn (o.pe b) d d
  pmap : ∀ d, Cn o.pmap d d

section
variable (o : Oracle) (hb : o.Bal)
include hb

theorem cn_labelsP : ∀ n acc d, Cn (labelsP n acc) d d
  | 0, _, d => cn_fuelP d d
  | n + 1, acc, d => by
    unfold labelsP
    refine cn_peek_bind fun c => ?_
    split
    · rename_i hc; subst hc
      exact cnc_bind (cnc_next_atom rfl d) fun _ =>
        cn_bind (cn_ident d) fun l => cn_labelsP n _ d
    · exact cnc_of_cn (cn_pure _ d)

theorem cn_propsP (d : Nat) : Cn (propsP o) d d := by
  unfold propsP
  refine cn_peek_bind fun c => ?_
  cases c
  case param p => exact cnc_bind (cnc_next_atom rfl d) fun _ => cn_pure _ d
  case lbrace => exact cnc_of_cn (hb.pmap d)
  all_goals exact cnc_of_cn (cn_pure _ d)

theorem cn_aliasOr (a : Option Nat) (d : Nat) : Cn (aliasOr a) d d := by
  cases a
  · exact cn_anonFresh d
  · exact cn_pure _ d

theorem cn_nodeP (F d : Nat) : Cn (nodeP o F) d d := by
  unfold nodeP
  refine cn_bind (cn_tok_gen (by decide) (step_open (.inl rfl) d)) fun _ => ?_
  refine cn_bind (cn_tryIdent _) fun a => ?_
  refine cn_bind (cn_aliasOr o hb a _) fun _ => ?_
  refine cn_bind (cn_labelsP o hb _ _ _) fun _ => ?_
  refine cn_bind (cn_propsP o hb _) fun _ => ?_
  refine cn_bind (cn_tok_gen (by decide) (step_close (.inl rfl) d)) fun _ => ?_
  exact cn_pure _ _

theorem cn_typesP : ∀ n acc d, Cn (typesP n acc) (d + 1) (d + 1)
  | 0, _, d => cn_fuelP _ _
  | n + 1, acc, d => by
    unfold typesP
    refine cn_bind (cn_ident _) fun t => ?_
    refine cn_bind (cn_opt_gen (by decide) (step_inner (.inl rfl) d)) fun p => ?_
    refine cn_bind (cn_opt_atom rfl _) fun c => ?_
    exact cn_ite (cn_typesP n _ d) (cn_pure _ _)

theorem cn_hopP (d : Nat) : Cn hopP d d := by
  unfold hopP
  refine cn_peek_bind fun c => ?_
  cases c
  case int i => exact cnc_bind (cnc_next_atom rfl d) fun _ => cn_pure _ d
  all_goals exact cnc_of_cn (cn_pure _ d)

theorem cn_varLenP (d : Nat) : Cn varLenP d d := by
  unfold varLenP
  refine cn_bind (cn_opt_atom rfl _) fun st => ?_
  refine cn_ite ?_ (cn_pure _ _)
  refine cn_bind (cn_hopP o hb _) fun start => ?_
  refine cn_bind (cn_opt_atom rfl _) fun dd => ?_
  refine cn_ite ?_ ?_
  · exact cn_bind (cn_hopP o hb _) fun _ => cn_pure _ _
  · split <;> exact cn_pure _ _

theorem cn_detailP (F : Nat) (cl : CK) (d : Nat) : Cn (detailP o F cl) (d + 1) d := by
  unfold detailP
  refine cn_bind (cn_tryIdent _) fun a => ?_
  refine cn_bind (cn_aliasOr o hb a _) fun _ => ?_
  refine cn_bind (cn_opt_atom rfl _) fun col => ?_
  refine cn_bind (cn_ite (cn_typesP o hb F [] d) (cn_pure _ _)) fun _ => ?_
  refine cn_bind (cn_varLenP o hb _) fun vl => ?_
  refine cn_ite (cn_fail _ _) ?_
  refine cn_ite (cn_fail _ _) ?_
  refine cn_bind (cn_propsP o hb _) fun _ => ?_
  refine cn_bind (cn_tok_gen (by decide) (step_close (.inr (.inl rfl)) d)) fun _ => ?_
  exact cn_pure _ _

theorem cn_relP (F : Nat) (cl : CK) (src : QNode Nat) (d : Nat) : Cn (relP o F cl src) d d := by
  unfold relP
  refine cn_bind (cn_opt_atom rfl _) fun inc => ?_
  refine cn_bind (cn_tok_atom rfl _) fun _ => ?_
  have rest : ∀ dd : Nat × List Nat × ET × Option (Nat × Option Nat),
      Cn (do
        tok .dash
        let out ← opt .gt
        let dst ← nodeP o F
        let mn := dd.2.2.2.map (·.1)
        let mx := dd.2.2.2.bind (·.2)
        if inc == out then
          if cl = .create then fail
          else pure (QRel.new dd.1 dd.2.1 dd.2.2.1 src dst true mn mx, dst)
        else if inc then pure (QRel.new dd.1 dd.2.1 dd.2.2.1 dst src false mn mx, dst)
        else pure (QRel.new dd.1 dd.2.1 dd.2.2.1 src dst false mn mx, dst)) d d := by
    intro dd
    refine cn_bind (cn_tok_atom rfl _) fun _ => ?_
    refine cn_bind (cn_opt_atom rfl _) fun out => ?_
    refine cn_bind (cn_nodeP o hb F _) fun dst => ?_
    exact cn_ite (cn_ite (cn_fail _ _) (cn_pure _ _)) (cn_ite (cn_pure _ _) (cn_pure _ _))
  refine cn_opt_lbrack_bind ?_ ?_
  · exact cn_bind (cn_detailP o hb F cl d) rest
  · exact cn_bind (cn_bind (cn_anonFresh _) fun _ => cn_pure _ _) rest

theorem cn_pcChain (F : Nat) : ∀ n left g d, Cn (pcChain o F n left g) d d
  | 0, _, _, d => cn_fuelP _ _
  | n + 1, left, g, d => by
    unfold pcChain
    refine cn_peek_bind fun c => cnc_of_cn ?_
    refine cn_ite ?_ (cn_pure _ _)
    exact cn_bind (cn_relP o hb F .match_ left d) fun rr => cn_pcChain F n _ _ d

/-- **A pattern comprehension that parses opens the gate**: given `Oracle.Bal`,
`parse_pattern_comprehension` succeeds only if `list_has_top_level_pipe` holds
where it started. -/
theorem patCompP_pipe (F : Nat) (pv : Option Nat) (s : PS) (t : ET) (s' : PS)
    (h : patCompP o F pv s = .ok t s') : pipeScan 0 s.1 = true := by
  unfold patCompP at h
  peel h => first s1 h1
  peel h => c s2 h2
  obtain ⟨rfl, rfl⟩ := peek_ok h2
  split at h
  · simp at h
  peel h => g s3 h3
  peel h => w s4 h4
  peel h => cond s5 h5
  peel h => u s6 h6
  obtain ⟨hp, -⟩ := tok_ok h6
  obtain ⟨p1, e1, st1⟩ := cn_nodeP o hb F 0 _ _ _ h1
  obtain ⟨p2, e2, st2⟩ := cn_pcChain o hb F F _ _ 0 _ _ _ h3
  obtain ⟨p3, e3, st3⟩ := cn_opt_atom (t := .kw .where_) rfl 0 _ _ _ h4
  obtain ⟨p4, e4, st4⟩ := (cn_ite (c := w = true) (hb.pe false 0) (cn_pure _ 0)) _ _ _ h5
  obtain ⟨l5, n5⟩ := s5
  cases l5 with
  | nil => simp [PS.cur] at hp
  | cons x r =>
    simp [PS.cur] at hp; subst hp
    simp only at e4
    rw [e1, e2, e3, e4]
    have := step_append (step_append (step_append st1 st2) st3) st4 (.pipe :: r)
    simp only [List.append_assoc] at this
    rw [this]; simp [pipeScan]

end

/-! ## 4. The gate and the memo leave the parse unchanged (under `Bal` and `MemoSound`) -/

/-- Both pattern-comprehension attempts from `saved` (after `[`) fail. -/
def PCFails (o : Oracle) (F : Nat) (saved : PS) : Prop :=
  (∀ v, identOf saved.cur = some v → saved.adv.cur = .eq → saved.adv.adv.cur = .lparen →
      ∀ t s', patCompP o F (some v) saved.adv.adv ≠ .ok t s') ∧
  (saved.cur = .lparen → ∀ t s', patCompP o F none saved ≠ .ok t s')

/-- The memo only holds attempts that fail. -/
def MemoSound (o : Oracle) (F : Nat) : Prop := ∀ s, o.seen s = true → PCFails o F s

theorem pipeScan_atom0 {t : CT} (h : isAtom t = true) (r : List CT) : pipeScan 0 (t :: r) = pipeScan 0 r :=
  step_atom h 0 r

/-- **A closed gate means both attempts fail** (given `Oracle.Bal`). -/
theorem gate_closed_fails (o : Oracle) (hb : o.Bal) (F : Nat) (saved : PS)
    (hg : gateOK saved = false) : PCFails o F saved := by
  constructor
  · intro v hv he hl t s' h
    have hp := patCompP_pipe o hb F _ _ _ _ h
    obtain ⟨l, n⟩ := saved
    match l, hv, he with
    | x :: y :: r, hv, he =>
      simp [PS.cur, PS.adv] at hv he hp; subst he
      have : pipeScan 0 (x :: .eq :: r) = true := by
        rw [pipeScan_atom0 (identOf_atom hv), pipeScan_atom0 rfl]; exact hp
      simp [gateOK, PS.cur, hv, this] at hg
    | [], hv, _ => simp [PS.cur, identOf] at hv
    | [_], _, he => simp [PS.cur, PS.adv] at he
  · intro hl t s' h
    have hp := patCompP_pipe o hb F _ _ _ _ h
    simp [gateOK, hl, hp] at hg

theorem attempt_none_or_fuel {α} {x : P α} {s : PS} (hnp : x s ≠ .panic)
    (hno : ∀ a s', x s ≠ .ok a s') : attempt x s = .ok none s ∨ attempt x s = .fuel := by
  unfold attempt
  cases h : x s with
  | ok a s' => exact absurd h (hno a s')
  | err => exact .inl rfl
  | panic => exact absurd h hnp
  | fuel => exact .inr rfl

/-- When both attempts fail, alternatives 2-4 are the list literal (or the
model's fuel ran out inside an attempt). -/
theorem listLitRest_lit (o : Oracle) (hs : o.Safe) (F : Nat) (forb : Bool) (saved : PS)
    (hf : PCFails o F saved) (s0 : PS) :
    listLitRest o F forb saved s0 = .fuel ∨ listLitRest o F forb saved s0 = listLitLit saved s0 := by
  have np := fun pv s => (np_patCompP o hs F pv).run s
  -- alternative 3 and the literal, from `saved`
  have alt3 : (do
      setS saved
      let c3 ← peek
      let r3 ← (if c3 = .lparen then attempt (patCompP o F none) else pure none)
      match r3 with
      | some t => do
        if forb then fail else do
        rejectAggregate t
        pure (t, false)
      | none => do
        setS saved
        let rb ← opt .rbrack
        pure (leaf .list, !rb) : P (ET × Bool)) s0 = .fuel ∨
      (do
      setS saved
      let c3 ← peek
      let r3 ← (if c3 = .lparen then attempt (patCompP o F none) else pure none)
      match r3 with
      | some t => do
        if forb then fail else do
        rejectAggregate t
        pure (t, false)
      | none => do
        setS saved
        let rb ← opt .rbrack
        pure (leaf .list, !rb) : P (ET × Bool)) s0 = listLitLit saved s0 := by
    simp only [run_bind, run_setS, peek]
    by_cases hl : saved.cur = .lparen
    · simp only [hl, if_true]
      rcases attempt_none_or_fuel (np none saved) (hf.2 hl) with h | h
      · rw [h]; exact .inr rfl
      · rw [h]; exact .inl rfl
    · simp only [hl, if_false, run_pure]; exact .inr rfl
  unfold listLitRest
  simp only [run_bind, run_setS, tryIdent]
  cases hv : identOf saved.cur with
  | none => simp only [run_pure, peek]; exact alt3
  | some v =>
    simp only [opt, peek]
    by_cases he : saved.adv.cur = .eq
    · simp only [he, if_true]
      by_cases hl : saved.adv.adv.cur = .lparen
      · simp only [hl]
        rcases attempt_none_or_fuel (np (some v) saved.adv.adv) (hf.1 v hv he hl) with h | h
        · rw [h]; exact alt3
        · rw [h]; exact .inl rfl
      · cases hc : saved.adv.adv.cur <;> simp only [run_pure] <;> first | exact alt3 | exact absurd hc hl
    · simp only [he, if_false]
      cases hc : saved.adv.cur <;> simp only [run_pure] <;> exact alt3

/-- **#3104 does not change the parse** — given that the expression and map
parsers consume balanced runs (`Oracle.Bal`) and that the memo only holds
failing attempts (`MemoSound`): wherever the pre-#3104 parser does not run
out of model fuel, the merged `parse_list_literal_or_comprehension` returns
exactly what it returned (same tree, same `recurse`, same position, same
error). -/
theorem listLitP_eq_ungated (o : Oracle) (hs : o.Safe) (hb : o.Bal) (F : Nat) (hm : MemoSound o F)
    (forb : Bool) (s : PS) (hnf : listLitPU o F forb s ≠ .fuel) :
    listLitP o F forb s = listLitPU o F forb s := by
  have hrest : ∀ s3, listLitRest o F forb s s3 ≠ .fuel →
      (if (gateOK s && !o.seen s) = true then listLitRest o F forb s else listLitLit s) s3 =
        listLitRest o F forb s s3 := by
    intro s3 hnf3
    split
    · rfl
    · rename_i hg
      have hpf : PCFails o F s := by
        cases hgo : gateOK s
        · exact gate_closed_fails o hb F s hgo
        · cases hse : o.seen s
          · simp [hgo, hse] at hg
          · exact hm s hse
      rcases listLitRest_lit o hs F forb s hpf s3 with h | h
      · exact absurd h hnf3
      · exact h.symm
  unfold listLitP listLitPU at *
  simp only [run_bind, run_getS] at *
  cases hta : tryIdent s with
  | ok a s2 =>
    rw [hta] at hnf
    simp only at hnf ⊢
    cases a with
    | none =>
      simp only [run_pure] at hnf ⊢
      exact hrest s2 hnf
    | some v =>
      simp only [optK, opt] at hnf ⊢
      by_cases hin : s2.cur = .kw .in_
      · simp only [hin, if_true]
      · simp only [hin, if_false] at hnf ⊢
        exact hrest s2 hnf
  | _ => rfl

/-! ## 5. History: `Oracle.Bal` failed for the Rust before #3041, a changed parse -/

/-- An expression parser with the pre-#3041 Rust's swallowed slice bound
(cypher.rs:2136-2156 at 8621212fc): `[1][(1..2]` is accepted,
consuming a `(` it never closes. Abstracted: on a name it reads the name, on
anything else it reads up to the next `|`. -/
def peBad : P ET := fun s =>
  match s.cur with
  | .ident v => .ok (leaf (.var v)) s.adv
  | _ => .ok (leaf .null) (s.1.dropWhile (· != .pipe), s.2)

def oBad : Oracle := ⟨fun _ => peBad, fail, fail, fun _ _ => 0, fun _ => none, fun _ => false⟩

/-- `(a)-->(b) WHERE [1][(1..2] | b]`, the tokens after the `[`. -/
def badInput : List CT :=
  [.lparen, .ident 1, .rparen, .dash, .dash, .gt, .lparen, .ident 2, .rparen,
   .kw .where_, .lbrack, .int 1, .rbrack, .lbrack, .lparen, .int 1, .dotdot, .int 2, .rbrack,
   .pipe, .ident 2, .rbrack]

def rootOf : R (ET × Bool) → Option (EOp × Bool)
  | .ok (t, b) _ => some (t.root, b)
  | _ => none

/-! ## 6. The memo: a failed attempt is never repeated, so failures are linear -/

/-- `PatternComprehensionAttempt` (cypher.rs:228-234): `forb` ranges over the
three `ForbiddenPatternComprehension` places (cypher.rs:130-145). -/
structure PCKey where
  pos : Nat
  allow : Bool
  forb : Option (Fin 3)
  depth : Nat
  deriving DecidableEq

/-- One call of `parse_list_literal_or_comprehension` past alternative 1. -/
structure Visit where
  key : PCKey
  /-- `(` or a name, and `list_has_top_level_pipe`. -/
  gate : Bool
  /-- Both attempts would fail. -/
  fails : Bool

/-- Alternatives 2-3 over a run of calls (cypher.rs:2845-2878): an attempt
is made when the gate is open and the memo lacks the key; a failure is
inserted. Returns the memo and the keys of the failed attempts. -/
def memoRun : List PCKey → List Visit → List PCKey × List PCKey
  | memo, [] => (memo, [])
  | memo, v :: vs =>
    if v.gate && !(memo.contains v.key) then
      if v.fails then
        let r := memoRun (v.key :: memo) vs
        (r.1, v.key :: r.2)
      else memoRun memo vs
    else memoRun memo vs

/-- **No key fails twice**, and none that was already in the memo. -/
theorem memoRun_nodup : ∀ (vs : List Visit) (memo : List PCKey),
    (memoRun memo vs).2.Nodup ∧ ∀ k ∈ (memoRun memo vs).2, k ∉ memo
  | [], memo => by simp [memoRun]
  | v :: vs, memo => by
    unfold memoRun
    split
    · rename_i hg
      have hk : v.key ∉ memo := by
        simp at hg; intro hm; exact hg.2 hm
      split
      · obtain ⟨h1, h2⟩ := memoRun_nodup vs (v.key :: memo)
        refine ⟨List.nodup_cons.mpr ⟨fun hm => h2 _ hm (by simp), h1⟩, ?_⟩
        intro k hkm
        simp at hkm
        rcases hkm with rfl | hkm
        · exact hk
        · exact fun hm => h2 k hkm (by simp [hm])
      · exact memoRun_nodup vs memo
    · exact memoRun_nodup vs memo

/-- When an attempt's outcome is a function `f` of its key (`AttemptDet`),
the memo only ever holds failing keys: a memo hit skips an attempt that would
have failed (this is `MemoSound`). -/
theorem memoRun_sound (f : PCKey → Bool) : ∀ (vs : List Visit) (memo : List PCKey),
    (∀ v ∈ vs, v.fails = f v.key) → (∀ k ∈ memo, f k = true) →
    ∀ k ∈ (memoRun memo vs).1, f k = true
  | [], memo, _, hm => by simpa [memoRun] using hm
  | v :: vs, memo, hd, hm => by
    have hd' : ∀ w ∈ vs, w.fails = f w.key := fun w hw => hd w (by simp [hw])
    unfold memoRun
    split
    · split
      · rename_i hfail
        apply memoRun_sound f vs _ hd'
        intro k hk; simp at hk
        rcases hk with rfl | hk
        · rw [← hd v (by simp)]; exact hfail
        · exact hm k hk
      · exact memoRun_sound f vs memo hd' hm
    · exact memoRun_sound f vs memo hd' hm

theorem nodup_length_le {α} [DecidableEq α] : ∀ (l U : List α), l.Nodup → (∀ x ∈ l, x ∈ U) →
    l.length ≤ U.length
  | [], _, _, _ => by simp
  | x :: l, U, hn, hs => by
    have ⟨hx, hl⟩ := List.nodup_cons.mp hn
    have hxU : x ∈ U := hs x (by simp)
    have ih := nodup_length_le l (U.erase x) hl (fun y hy => by
      have hyx : y ≠ x := fun h => hx (h ▸ hy)
      exact (List.mem_erase_of_ne hyx).mpr (hs y (by simp [hy])))
    have := List.length_erase_of_mem hxU
    have hpos : 0 < U.length := List.length_pos_of_mem hxU
    simp; omega

def forbCode : Option (Fin 3) → Nat
  | none => 0
  | some i => i.val + 1

theorem forbCode_lt (f : Option (Fin 3)) : forbCode f < 4 := by
  cases f with
  | none => simp [forbCode]
  | some i => have := i.isLt; simp [forbCode]; omega

theorem forbCode_inj {f g : Option (Fin 3)} (h : forbCode f = forbCode g) : f = g := by
  cases f <;> cases g <;> simp [forbCode] at h ⊢
  exact Fin.ext h

def boolCode (b : Bool) : Nat := if b then 1 else 0

theorem boolCode_lt (b : Bool) : boolCode b < 2 := by cases b <;> simp [boolCode]

theorem boolCode_inj {a b : Bool} (h : boolCode a = boolCode b) : a = b := by
  cases a <;> cases b <;> simp [boolCode] at h ⊢

def PCKey.code (k : PCKey) : Nat :=
  ((k.pos * 2 + boolCode k.allow) * 4 + forbCode k.forb) * 101 + k.depth

theorem PCKey.code_inj {a b : PCKey} (ha : a.depth ≤ 100) (hb : b.depth ≤ 100)
    (h : a.code = b.code) : a = b := by
  obtain ⟨pa, aa, fa, da⟩ := a
  obtain ⟨pb, ab, fb, db⟩ := b
  simp only [PCKey.code] at h ha hb
  have := forbCode_lt fa; have := forbCode_lt fb
  have := boolCode_lt aa; have := boolCode_lt ab
  have hd : da = db := by omega
  have hf : forbCode fa = forbCode fb := by omega
  have hal : boolCode aa = boolCode ab := by omega
  have hp : pa = pb := by omega
  rw [hd, hp, boolCode_inj hal, forbCode_inj hf]

theorem PCKey.code_lt {k : PCKey} {n : Nat} (hp : k.pos ≤ n) (hd : k.depth ≤ 100) :
    k.code < 808 * (n + 1) := by
  simp only [PCKey.code]
  have := forbCode_lt k.forb; have := boolCode_lt k.allow
  have : (k.pos * 2 + boolCode k.allow) * 4 + forbCode k.forb < 8 * (n + 1) := by
    have : k.pos * 2 + boolCode k.allow < 2 * (n + 1) := by omega
    omega
  omega

theorem nodup_map_code : ∀ (l : List PCKey), l.Nodup → (∀ k ∈ l, k.depth ≤ 100) →
    (l.map PCKey.code).Nodup
  | [], _, _ => by simp
  | k :: l, hn, hd => by
    have ⟨hk, hl⟩ := List.nodup_cons.mp hn
    refine List.nodup_cons.mpr ⟨?_, nodup_map_code l hl (fun x hx => hd x (by simp [hx]))⟩
    intro hm
    obtain ⟨k', hk', he⟩ := List.mem_map.mp hm
    have := PCKey.code_inj (hd k' (by simp [hk'])) (hd k (by simp)) he
    exact hk (this ▸ hk')

/-- **Failed attempts are linear in the input** (W3-parser-9): on an input of
`n` bytes, with nesting bounded by `MAX_NESTING = 100` (every key's depth is at
most 100 — `nested` refuses deeper, `proofs/lexer` `nested_depth_bound`), the
attempts that fail in a whole parse number at most `808 · (n + 1)`, however the
enclosing parsers backtrack. -/
theorem failed_attempts_linear (n : Nat) (vs : List Visit)
    (hpos : ∀ v ∈ vs, v.key.pos ≤ n) (hdep : ∀ v ∈ vs, v.key.depth ≤ 100) :
    (memoRun [] vs).2.length ≤ 808 * (n + 1) := by
  obtain ⟨hn, -⟩ := memoRun_nodup vs []
  have hin : ∀ k ∈ (memoRun [] vs).2, ∃ v ∈ vs, v.key = k := by
    have gen : ∀ (vs : List Visit) memo, ∀ k ∈ (memoRun memo vs).2, ∃ v ∈ vs, v.key = k := by
      intro vs
      induction vs with
      | nil => intro memo k hk; simp [memoRun] at hk
      | cons v vs ih =>
        intro memo k hk
        unfold memoRun at hk
        split at hk
        · split at hk
          · simp at hk
            rcases hk with rfl | hk
            · exact ⟨v, by simp, rfl⟩
            · obtain ⟨w, hw, he⟩ := ih _ k hk; exact ⟨w, by simp [hw], he⟩
          · obtain ⟨w, hw, he⟩ := ih _ k hk; exact ⟨w, by simp [hw], he⟩
        · obtain ⟨w, hw, he⟩ := ih _ k hk; exact ⟨w, by simp [hw], he⟩
    exact gen vs []
  have hd : ∀ k ∈ (memoRun [] vs).2, k.depth ≤ 100 := by
    intro k hk; obtain ⟨v, hv, rfl⟩ := hin k hk; exact hdep v hv
  have hm := nodup_map_code _ hn hd
  have := nodup_length_le _ (List.range (808 * (n + 1))) hm (by
    intro c hc
    obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hc
    obtain ⟨v, hv, rfl⟩ := hin k hk
    exact List.mem_range.mpr (PCKey.code_lt (hpos v hv) (hdep v hv)))
  simpa using this

end FalkorParserGrammar.C
