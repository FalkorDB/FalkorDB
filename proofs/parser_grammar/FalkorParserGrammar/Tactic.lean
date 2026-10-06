/- A small helper tactic for the no-panic proof. -/
import Lean

open Lean Elab Tactic Meta

/-- `pe_facts hpe pe`: for every hypothesis `pe a = r` in context, add the
instance of `hpe : ∀ a, P (pe a)` at `r`. -/
elab "pe_facts" hpe:ident pe:ident : tactic => withMainContext do
  let hpeE := (← getLocalDeclFromUserName hpe.getId).toExpr
  let peE := (← getLocalDeclFromUserName pe.getId).toExpr
  let mut n := 0
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    let some (_, lhs, _) := ty.eq? | continue
    unless lhs.isApp && lhs.appFn! == peE do continue
    let a := lhs.appArg!
    let base ← mkAppM' hpeE #[a]
    let bty ← inferType base
    let bty ← whnfR bty
    let bty ← instantiateMVars bty
    let abs ← kabstract bty lhs
    let motive := Lean.mkLambda `x .default (← inferType lhs) abs
    let pf ← mkEqNDRec motive base d.toExpr
    let pty ← inferType pf
    let g ← getMainGoal
    let (_, g') ← (← g.assert (Name.mkSimple s!"hpe{n}") (← instantiateMVars (Expr.headBeta pty)) pf).intro1P
    replaceMainGoal [g']
    n := n + 1


/-- `use_facts t`: `t : ∀ xs, Q (f xs)` (premises allowed). For every
hypothesis `f as = r` in context, add `Q r`, discharging the premises of `t`
by assumption. Used to push "never panics / good tree" facts through the
`match f .. with` splits of the parser. -/
elab "use_facts" t:term : tactic => withMainContext do
  let pf ← Term.elabTerm t none
  Term.synthesizeSyntheticMVarsNoPostponing
  let pf ← instantiateMVars pf
  let pfTy ← instantiateMVars (← inferType pf)
  for d in (← getLCtx) do
    if d.isImplementationDetail then continue
    let ty ← instantiateMVars d.type
    let some (_, lhs, _) := ty.eq? | continue
    let hd := lhs.getAppFn
    unless hd.isConst || hd.isFVar do continue
    let saved ← saveState
    let (xs, _, body) ← forallMetaTelescopeReducing pfTy
    let target := body.appArg!
    if target.getAppFn != hd then
      restoreState saved; continue
    if !(← isDefEq target lhs) then
      restoreState saved; continue
    let mut ok := true
    for x in xs do
      let x ← instantiateMVars x
      if x.isMVar then
        let xty ← instantiateMVars (← inferType x)
        if ← isProp xty then
          match ← findLocalDeclWithType? xty with
          | some fv => x.mvarId!.assign (mkFVar fv)
          | none => ok := false
        else ok := false
    if !ok then
      restoreState saved; continue
    let p ← instantiateMVars (mkAppN pf xs)
    let bodyI ← instantiateMVars body
    let abs ← kabstract bodyI lhs
    let motive := Lean.mkLambda `x .default (← inferType lhs) abs
    let prf ← mkEqNDRec motive p d.toExpr
    let g ← getMainGoal
    let (_, g') ← (← g.assert `hfact (Expr.headBeta (← instantiateMVars (← inferType prf))) prf).intro1P
    replaceMainGoal [g']

/-- `use_goal t`: `t : ∀ xs, Q (f xs)`. For the first subterm `f as` of the
goal, add `Q (f as)`, generalize `f as` everywhere and case on it. -/
elab "use_goal" t:term : tactic => withMainContext do
  let pf ← Term.elabTerm t none
  Term.synthesizeSyntheticMVarsNoPostponing
  let pf ← instantiateMVars pf
  let pfTy ← instantiateMVars (← inferType pf)
  let (_, _, body0) ← forallMetaTelescopeReducing pfTy
  let hd := body0.appArg!.getAppFn
  let goal ← instantiateMVars (← getMainTarget)
  let cands := (goal.foldlM (m := Id) (init := #[]) fun acc e =>
    if e.getAppFn == hd && !e.hasLooseBVars then acc.push e else acc)
  -- `foldlM` over `Expr` visits only direct children; collect all subterms instead
  let mut subs : Array Expr := #[]
  let rec visit (e : Expr) (acc : Array Expr) : Array Expr :=
    let acc := if e.getAppFn == hd && !e.hasLooseBVars then acc.push e else acc
    match e with
    | .app f a => visit a (visit f acc)
    | .lam _ t b _ => visit b (visit t acc)
    | .forallE _ t b _ => visit b (visit t acc)
    | .letE _ t v b _ => visit b (visit v (visit t acc))
    | .mdata _ b => visit b acc
    | .proj _ _ b => visit b acc
    | _ => acc
  subs := visit goal #[]
  let _ := cands
  for e in subs do
    let saved ← saveState
    let (xs, _, body) ← forallMetaTelescopeReducing pfTy
    unless ← isDefEq body.appArg! e do
      restoreState saved; continue
    let mut ok := true
    let mut side : Array MVarId := #[]
    for x in xs do
      let x ← instantiateMVars x
      if x.isMVar then
        let xty ← instantiateMVars (← inferType x)
        if ← isProp xty then
          match ← findLocalDeclWithType? xty with
          | some fv => x.mvarId!.assign (mkFVar fv)
          | none => side := side.push x.mvarId!
        else ok := false
    if !ok then
      restoreState saved; continue
    let p ← instantiateMVars (mkAppN pf xs)
    let g ← getMainGoal
    let (h, g) ← (← g.assert `hfact (← instantiateMVars body) p).intro1P
    let gs ← g.withContext do
      let (_, xsFV, g) ← g.generalizeHyp #[{ expr := e }] #[h]
      g.withContext (g.cases xsFV[0]!)
    replaceMainGoal ((gs.map (·.mvarId)).toList ++ side.toList)
    return
  throwError "use_goal: no matching subterm"
