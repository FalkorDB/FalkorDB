import OpsTraverse.SearchBase
/-
# `enumerate_paths` (graph/src/runtime/functions/algo_procedures.rs)

Iterative DFS with an explicit level stack. `step u` order: the Rust pops from the END of the successor
`Vec`, so a level holds `(succ u).reverse` and pops from its head (`succ` is `node_successors`).
The Rust keeps `levels`, `nodes`, `edges` as three parallel `Vec`s plus the `on_path` hash set; here
they are one stack of frames (`Fr`: node, incoming edge, prefix weight/cost, untried successors),
and `on_path` is membership in the stack's nodes (it is exactly that set: the nodes on a stack are
distinct, `Enum.nodes_nodup`, and `remove` pops the last one). `rec` is `record_found_path` (a parameter,
`R` its result set type; it also sees and returns the bound). Timeout polls are abstracted away.

| here | there |
| --- | --- |
| `Fr`, `ES.stack` | `levels[d]`, `nodes[d]`, `edges[d-1]` for depth `d` |
| `okStep` | `next_weight`/`next_cost` and the three `continue` guards (`is_finite`, `> bound`, `> max_cost`) |
| `stepE` | one iteration of `loop { … }` |
| `runE` | the `loop`, with fuel (the Rust has none; theorems cover finished runs) |
| `enumRec` (`ch`, `visit`) | the same search as a recursion: the semantics `stepE` is proved equal to (`enum_eq_rec`) |
-/
namespace OpsTraverse.Enum
open OpsTraverse.Search

variable {ε F R : Type}

structure Ops (F : Type) where
  zero : F
  add : F → F → F
  fin : F → Bool
  gt : F → F → Bool

/-- The parts of `PathAlgoConfig` and the graph `enumerate_paths` reads. -/
structure EC (ε F : Type) where
  o : Ops F
  succ : Nat → List (Nat × ε)
  source : Nat
  target : Option Nat
  maxLen : Nat
  maxCost : Option F
  wf : ε → F
  cf : ε → F

/-- A found path: edges in order (far endpoint, edge), weight, cost. -/
abbrev Fnd (ε F : Type) := List (Nat × ε) × F × F

structure Fr (ε F : Type) where
  cur : Nat
  inc : Option (Nat × ε)
  pw : F
  pc : F
  rem : Option (List (Nat × ε))

structure ES (ε F R : Type) where
  stack : List (Fr ε F)
  res : R
  bound : F

def pathOf (stack : List (Fr ε F)) : List (Nat × ε) := (stack.filterMap (·.inc)).reverse
def nodesOfS (stack : List (Fr ε F)) : List Nat := stack.map (·.cur)

/-- `Some (next_weight, next_cost)` when the edge passes all three `continue` guards. -/
def okStep (c : EC ε F) (bound pw pc : F) (e : ε) : Option (F × F) :=
  if c.o.fin (c.o.add pw (c.wf e)) = false || c.o.gt (c.o.add pw (c.wf e)) bound then none
  else if c.o.fin (c.o.add pc (c.cf e)) = false ||
      c.maxCost.any (fun m => c.o.gt (c.o.add pc (c.cf e)) m) then none
  else some (c.o.add pw (c.wf e), c.o.add pc (c.cf e))

def stepE (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (st : ES ε F R) : Option (ES ε F R) :=
  match st.stack with
  | [] => none
  | f :: fs =>
    match f.rem with
    | some (x :: r) =>
      let f' : Fr ε F := { f with rem := some r }
      if x.1 ∈ nodesOfS (f' :: fs) then some { st with stack := f' :: fs }
      else
        match okStep c st.bound f.pw f.pc x.2 with
        | none => some { st with stack := f' :: fs }
        | some (nw, nc) =>
          let atT := c.target.all (fun t => x.1 == t)
          let nf : Fr ε F := { cur := x.1, inc := some x, pw := nw, pc := nc, rem := none }
          let rb : R × F := if atT then rec st.res (pathOf (nf :: f' :: fs), nw, nc) st.bound else (st.res, st.bound)
          let expand := !(atT && c.target.isSome) && decide ((f' :: fs).length < c.maxLen)
          let nf' : Fr ε F := { nf with rem := if expand then some (c.succ x.1).reverse else none }
          some { stack := nf' :: f' :: fs, res := rb.1, bound := rb.2 }
    | _ => if fs = [] then none else some { st with stack := fs }

def runE (c : EC ε F) (rec : R → Fnd ε F → F → R × F) : Nat → ES ε F R → Option (ES ε F R)
  | 0, _ => none
  | n + 1, st =>
    match stepE c rec st with
    | none => some st
    | some st' => runE c rec n st'

def root (c : EC ε F) : Fr ε F :=
  { cur := c.source, inc := none, pw := c.o.zero, pc := c.o.zero,
    rem := if 0 < c.maxLen then some (c.succ c.source).reverse else none }

/-- `enumerate_paths` with `record_found_path = rec`, initial results `r0` and bound `b0`. -/
def enumerate (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (r0 : R) (b0 : F) (fuel : Nat) : Option (R × F) :=
  (runE c rec fuel ⟨[root c], r0, b0⟩).map (fun s => (s.res, s.bound))

/-! ## The search as a recursion -/

def visit (c : EC ε F) (rec : R → Fnd ε F → F → R × F)
    (down : List Nat → List (Nat × ε) → F → F → List (Nat × ε) → R × F → R × F)
    (on : List Nat) (pe : List (Nat × ε)) (pw pc : F) (x : Nat × ε) (rs : R × F) : R × F :=
  if x.1 ∈ on then rs else
  match okStep c rs.2 pw pc x.2 with
  | none => rs
  | some (nw, nc) =>
    let atT := c.target.all (fun t => x.1 == t)
    let rs1 : R × F := if atT then rec rs.1 (pe ++ [x], nw, nc) rs.2 else rs
    if !(atT && c.target.isSome) && decide (on.length < c.maxLen) then
      down (x.1 :: on) (pe ++ [x]) nw nc (c.succ x.1).reverse rs1
    else rs1

def ch (c : EC ε F) (rec : R → Fnd ε F → F → R × F) :
    Nat → List Nat → List (Nat × ε) → F → F → List (Nat × ε) → R × F → R × F
  | 0, _, _, _, _, _, rs => rs
  | k + 1, on, pe, pw, pc, rem, rs =>
    rem.foldl (fun rs x => visit c rec (ch c rec k) on pe pw pc x rs) rs

def enumRec (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (r0 : R) (b0 : F) : R × F :=
  ch c rec c.maxLen [c.source] [] c.o.zero c.o.zero (if 0 < c.maxLen then (c.succ c.source).reverse else []) (r0, b0)

def den (c : EC ε F) (rec : R → Fnd ε F → F → R × F) : List (Fr ε F) → R × F → R × F
  | [], rs => rs
  | f :: fs, rs =>
    den c rec fs (ch c rec (c.maxLen + 1 - (f :: fs).length) (nodesOfS (f :: fs)) (pathOf (f :: fs))
      f.pw f.pc (f.rem.getD []) rs)

def Wf (c : EC ε F) : List (Fr ε F) → Prop
  | [] => True
  | f :: fs => Wf c fs ∧ (f.rem.isSome = true → (f :: fs).length ≤ c.maxLen)

theorem ch_nil (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (k : Nat) (on pe pw pc rs) :
    ch c rec k on pe pw pc [] rs = rs := by
  cases k <;> simp [ch]

theorem ch_cons (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (k : Nat) (on pe pw pc x r rs) :
    ch c rec (k + 1) on pe pw pc (x :: r) rs =
      ch c rec (k + 1) on pe pw pc r (visit c rec (ch c rec k) on pe pw pc x rs) := by
  simp [ch]

theorem pathOf_cons (f : Fr ε F) (fs : List (Fr ε F)) (x : Nat × ε) (h : f.inc = some x) :
    pathOf (f :: fs) = pathOf fs ++ [x] := by
  simp [pathOf, h]

theorem step_den (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (st st' : ES ε F R)
    (hw : Wf c st.stack) (h : stepE c rec st = some st') :
    Wf c st'.stack ∧ den c rec st'.stack (st'.res, st'.bound) = den c rec st.stack (st.res, st.bound) := by
  unfold stepE at h
  rcases hs : st.stack with _ | ⟨f, fs⟩
  · rw [hs] at h; simp at h
  · rw [hs] at h
    rw [hs] at hw
    simp only at h
    rcases hr : f.rem with _ | _ | ⟨x, r⟩
    · rw [hr] at h
      simp only at h
      by_cases hfs : fs = []
      · simp [hfs] at h
      · simp only [hfs, ↓reduceIte, Option.some.injEq] at h
        subst h
        refine ⟨hw.1, ?_⟩
        simp only [hs, den, hr, Option.getD_none, ch_nil]
    · rw [hr] at h
      simp only at h
      by_cases hfs : fs = []
      · simp [hfs] at h
      · simp only [hfs, ↓reduceIte, Option.some.injEq] at h
        subst h
        refine ⟨hw.1, ?_⟩
        simp only [hs, den, hr, Option.getD_some, ch_nil]
    · rw [hr] at h
      simp only at h
      have hL : (f :: fs).length ≤ c.maxLen := hw.2 (by simp [hr])
      obtain ⟨k, hk⟩ : ∃ k, c.maxLen + 1 - (f :: fs).length = k + 1 := ⟨c.maxLen - (f :: fs).length, by omega⟩
      have hLs : (f :: fs).length = fs.length + 1 := rfl
      let f' : Fr ε F := { f with rem := some r }
      have hnodes : nodesOfS (f' :: fs) = nodesOfS (f :: fs) := rfl
      have hpath : pathOf (f' :: fs) = pathOf (f :: fs) := rfl
      have hbase : den c rec (f :: fs) (st.res, st.bound) =
          den c rec fs (ch c rec (k + 1) (nodesOfS (f :: fs)) (pathOf (f :: fs)) f.pw f.pc r
            (visit c rec (ch c rec k) (nodesOfS (f :: fs)) (pathOf (f :: fs)) f.pw f.pc x (st.res, st.bound))) := by
        simp only [den, hr, Option.getD_some, hk, ch_cons]
      have hf'den : ∀ rs, den c rec (f' :: fs) rs =
          den c rec fs (ch c rec (k + 1) (nodesOfS (f :: fs)) (pathOf (f :: fs)) f.pw f.pc r rs) := by
        intro rs
        simp only [den, hnodes, hpath, f', Option.getD_some]
        have : c.maxLen + 1 - (f' :: fs).length = k + 1 := hk
        rw [this]
      have hwf' : Wf c (f' :: fs) := ⟨hw.1, fun _ => hL⟩
      by_cases hmem : x.1 ∈ nodesOfS ({ f with rem := some r } :: fs)
      · simp only [hmem, ↓reduceIte, Option.some.injEq] at h
        subst h
        refine ⟨hwf', ?_⟩
        have hv : visit c rec (ch c rec k) (nodesOfS (f :: fs)) (pathOf (f :: fs)) f.pw f.pc x (st.res, st.bound) = (st.res, st.bound) := by
          rw [hnodes] at hmem; simp [visit, hmem]
        simp only [hs]
        rw [hbase, hv]
        exact hf'den _
      · simp only [hmem, ↓reduceIte] at h
        rcases hok : okStep c st.bound f.pw f.pc x.2 with _ | ⟨nw, nc⟩
        · rw [hok] at h
          simp only [Option.some.injEq] at h
          subst h
          refine ⟨hwf', ?_⟩
          have hv : visit c rec (ch c rec k) (nodesOfS (f :: fs)) (pathOf (f :: fs)) f.pw f.pc x (st.res, st.bound) = (st.res, st.bound) := by
            rw [hnodes] at hmem; simp [visit, hmem, hok]
          simp only [hs]
          rw [hbase, hv]
          exact hf'den _
        · rw [hok] at h
          simp only [Option.some.injEq] at h
          subst h
          have hpe : pathOf ({ cur := x.1, inc := some x, pw := nw, pc := nc, rem := none } :: ({ f with rem := some r } :: fs)) = pathOf (f :: fs) ++ [x] := by
            rw [pathOf_cons _ _ x rfl, hpath]
          rw [hpe]
          generalize hrs1 : (if Option.all (fun t => x.fst == t) c.target = true then
                rec st.res (pathOf (f :: fs) ++ [x], nw, nc) st.bound else (st.res, st.bound)) = rs1
          have hv : visit c rec (ch c rec k) (nodesOfS (f :: fs)) (pathOf (f :: fs)) f.pw f.pc x (st.res, st.bound) =
              (if (!((c.target.all (fun t => x.1 == t)) && c.target.isSome) && decide (fs.length + 1 < c.maxLen)) = true
               then ch c rec k (x.1 :: nodesOfS (f :: fs)) (pathOf (f :: fs) ++ [x]) nw nc (c.succ x.1).reverse rs1
               else rs1) := by
            rw [hnodes] at hmem
            simp only [visit, hmem, ↓reduceIte, hok, ← hrs1]
            simp [nodesOfS]
          simp only [List.length_cons, hs]
          have hlen : c.maxLen + 1 - (fs.length + 1 + 1) = k := by simp only [List.length_cons] at hk; omega
          by_cases hexp : (!((c.target.all (fun t => x.1 == t)) && c.target.isSome) && decide (fs.length + 1 < c.maxLen)) = true
          · refine ⟨⟨hwf', fun _ => ?_⟩, ?_⟩
            · have := (decide_eq_true_iff.mp (Bool.and_eq_true_iff.mp hexp).2)
              simp only [List.length_cons]; omega
            · rw [hbase, hv, if_pos hexp, show (rs1.fst, rs1.snd) = rs1 from rfl]
              have hpe2 : pathOf ({ cur := x.1, inc := some x, pw := nw, pc := nc, rem := some (c.succ x.fst).reverse } :: ({ f with rem := some r } :: fs)) = pathOf (f :: fs) ++ [x] := by
                rw [pathOf_cons _ _ x rfl, hpath]
              have key : den c rec ({ cur := x.1, inc := some x, pw := nw, pc := nc, rem := some (c.succ x.fst).reverse } :: ({ f with rem := some r } :: fs)) rs1 =
                  den c rec ({ f with rem := some r } :: fs) (ch c rec k (x.1 :: nodesOfS (f :: fs)) (pathOf (f :: fs) ++ [x]) nw nc (c.succ x.1).reverse rs1) := by
                simp only [den, hpe2, List.length_cons, hlen, Option.getD_some]
                rfl
              rw [key, hf'den, if_pos hexp]
          · rw [hbase, hv, if_neg hexp, show (rs1.fst, rs1.snd) = rs1 from rfl]
            have hexp' : ¬ (!((c.target.all (fun t => x.1 == t)) && c.target.isSome) && decide (fs.length + 1 < c.maxLen)) = true := hexp
            refine ⟨⟨hwf', fun hsome => ?_⟩, ?_⟩
            · simp only [hexp', ↓reduceIte, Option.isSome_none] at hsome; exact absurd hsome (by simp)
            · have hpe2 : pathOf ({ cur := x.1, inc := some x, pw := nw, pc := nc, rem := none } :: ({ f with rem := some r } :: fs)) = pathOf (f :: fs) ++ [x] := by
                rw [pathOf_cons _ _ x rfl, hpath]
              have key : den c rec ({ cur := x.1, inc := some x, pw := nw, pc := nc, rem := none } :: ({ f with rem := some r } :: fs)) rs1 =
                  den c rec ({ f with rem := some r } :: fs) rs1 := by
                simp only [den, Option.getD_none, ch_nil]
              simp only [hexp', ↓reduceIte]
              rw [key, hf'den]
              simp

theorem step_none_den (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (st : ES ε F R)
    (h : stepE c rec st = none) : den c rec st.stack (st.res, st.bound) = (st.res, st.bound) := by
  unfold stepE at h
  rcases hs : st.stack with _ | ⟨f, fs⟩
  · simp [den]
  · rw [hs] at h
    simp only at h
    rcases hr : f.rem with _ | _ | ⟨x, r⟩
    · rw [hr] at h
      simp only at h
      by_cases hfs : fs = []
      · subst hfs; simp [den, hr, ch_nil]
      · simp [hfs] at h
    · rw [hr] at h
      simp only at h
      by_cases hfs : fs = []
      · subst hfs; simp [den, hr, ch_nil]
      · simp [hfs] at h
    · rw [hr] at h
      simp only at h
      split at h
      · simp at h
      · split at h
        · simp at h
        · simp at h

theorem run_den (c : EC ε F) (rec : R → Fnd ε F → F → R × F) :
    ∀ (n : Nat) (st stF : ES ε F R), Wf c st.stack → runE c rec n st = some stF →
      (stF.res, stF.bound) = den c rec st.stack (st.res, st.bound) := by
  intro n
  induction n with
  | zero => intro st stF _ h; simp [runE] at h
  | succ n ih =>
    intro st stF hw h
    simp only [runE] at h
    rcases hq : stepE c rec st with _ | st'
    · rw [hq] at h
      simp only [Option.some.injEq] at h
      subst h
      exact (step_none_den c rec st hq).symm
    · rw [hq] at h
      obtain ⟨hw', hd⟩ := step_den c rec st st' hw hq
      rw [← hd]
      exact ih st' stF hw' h

/-- **PROVEN** (`enumerate_paths`, reduction): the iterative loop with its explicit level stack returns
exactly what the recursive depth-first search `enumRec` returns, for any `record_found_path`. -/
theorem enum_eq_rec (c : EC ε F) (rec : R → Fnd ε F → F → R × F) (r0 : R) (b0 : F) (fuel : Nat) (res : R × F)
    (h : enumerate c rec r0 b0 fuel = some res) : res = enumRec c rec r0 b0 := by
  unfold enumerate at h
  rcases hr : runE c rec fuel ⟨[root c], r0, b0⟩ with _ | stF
  · rw [hr] at h; simp at h
  · rw [hr] at h
    simp only [Option.map_some, Option.some.injEq] at h
    subst h
    have hw : Wf c [root c] := ⟨trivial, fun hs => by
      simp only [root] at hs
      by_cases hm : 0 < c.maxLen
      · show 1 ≤ c.maxLen; omega
      · simp [hm] at hs⟩
    have := run_den c rec fuel _ stF hw hr
    rw [this]
    simp only [den, enumRec, nodesOfS, pathOf, root, List.map_cons, List.map_nil, List.filterMap_cons,
      List.filterMap_nil, List.reverse_nil, List.length_singleton, Nat.add_sub_cancel]
    by_cases hm : 0 < c.maxLen <;> simp [hm]

end OpsTraverse.Enum
