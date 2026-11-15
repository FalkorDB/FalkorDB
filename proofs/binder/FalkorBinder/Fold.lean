/-
# Bind-time constant folding and call rewrites

| here | there |
| --- | --- |
| `Val`, `FD`, `FE`, `insertKV`, `ev` | `Value`, bound `ExprIR` payloads/trees, `OrderMap::insert` (ordermap.rs:76-89), runtime eval of constants / `List` / `Map` (eval.rs:349-359) |
| `nodeToValue`, `exprTreeToValue` | `expr_tree_to_value` / `node_to_value` binder.rs:2448-2477 |
| `collectConstArgs` | `collect_constant_args` binder.rs:2480-2489 |
| `foldCall` | the folding step of `bind_expr_node` binder.rs:2114-2126 |
| `rewriteStruct` | `rewrite_struct_constructor` binder.rs:2498-2553 |
| `rewriteRegex` | `rewrite_compiled_regex` binder.rs:2563-2592 |

* `nodeToValue_sound`: a tree that folds evaluates to the folded value
  (duplicate map keys: last value wins, first position kept, as at runtime);
  `nodeToValue_complete`: every all-constant List/Map tree folds.
* `collectConstArgs_spec`, `foldCall_sound`: folding a pure call on constant
  arguments gives what the runtime call would (when `pure_fn` agrees with the
  runtime function on those arguments — the registration contract).
* `rewriteStruct_*`: positional slots are filled from the literal map (last
  duplicate wins, missing → `Null`), unknown keys bail out; under the
  constructor contract (`struct_fn` on slots = map function on the map) the
  call's value is unchanged (`rewriteStruct_sound`).
* `rewriteRegex_*`: a constant, compilable pattern is moved into the operator;
  value unchanged under the compiled-regex contract.
-/
namespace FalkorBinder.Fold

inductive Val
  | null | bool (b : Bool) | int (i : Int) | str (s : String)
  | list (vs : List Val)
  | map (kvs : List (String × Val))
  | other (tag : Nat)
  deriving Repr, Inhabited

inductive FD
  | const (v : Val) | list | map | func (name : String) | var (n : Nat) | other (tag : Nat)

inductive FE
  | node (d : FD) (cs : List FE)

/-- `OrderMap::insert`: replace in place, else append. -/
def insertKV : List (String × Val) → String → Val → List (String × Val)
  | [], k, v => [(k, v)]
  | (k', v') :: rest, k, v => if k' = k then (k, v) :: rest else (k', v') :: insertKV rest k v

mutual
def nodeToValue : FE → Option Val
  | .node (.const v) _ => some v
  | .node .list cs => (nodeToValueL cs).map Val.list
  | .node .map cs => (mapEntries cs).map fun es => Val.map (es.foldl (fun m kv => insertKV m kv.1 kv.2) [])
  | _ => none
def nodeToValueL : List FE → Option (List Val)
  | [] => some []
  | c :: cs => match nodeToValue c, nodeToValueL cs with
    | some v, some vs => some (v :: vs)
    | _, _ => none
/-- Map children: `Constant(String key)` with the value as child 0. -/
def mapEntries : List FE → Option (List (String × Val))
  | [] => some []
  | .node (.const (.str k)) (v :: _) :: cs => match nodeToValue v, mapEntries cs with
    | some x, some es => some ((k, x) :: es)
    | _, _ => none
  | _ :: _ => none
end

def exprTreeToValue (t : FE) : Option Val := nodeToValue t

/-! ## Runtime evaluation of constant trees (eval.rs) -/

variable (ρ : FE → Option Val)  -- everything that is not a literal (variables, calls, …)

mutual
def ev : FE → Option Val
  | .node (.const v) _ => some v
  | .node .list cs => (evL cs).map Val.list
  | .node .map cs => (evMap cs).map fun es => Val.map (es.foldl (fun m kv => insertKV m kv.1 kv.2) [])
  | e => ρ e
def evL : List FE → Option (List Val)
  | [] => some []
  | c :: cs => match ev c, evL cs with
    | some v, some vs => some (v :: vs)
    | _, _ => none
def evMap : List FE → Option (List (String × Val))
  | [] => some []
  | .node (.const (.str k)) (v :: _) :: cs => match ev v, evMap cs with
    | some x, some es => some ((k, x) :: es)
    | _, _ => none
  | _ :: _ => none
end

mutual
/-- Folding is sound: whatever folds evaluates (at runtime, in any row) to that value. -/
theorem nodeToValue_sound : ∀ e v, nodeToValue e = some v → ev ρ e = some v
  | .node (.const c) _, v, h => by simpa [nodeToValue, ev] using h
  | .node .list cs, v, h => by
    simp only [nodeToValue, Option.map_eq_some_iff] at h
    obtain ⟨vs, h1, rfl⟩ := h
    simp [ev, nodeToValueL_sound cs vs h1]
  | .node .map cs, v, h => by
    simp only [nodeToValue, Option.map_eq_some_iff] at h
    obtain ⟨es, h1, rfl⟩ := h
    simp [ev, mapEntries_sound cs es h1]
  | .node (.func _) _, v, h => by simp [nodeToValue] at h
  | .node (.var _) _, v, h => by simp [nodeToValue] at h
  | .node (.other _) _, v, h => by simp [nodeToValue] at h
theorem nodeToValueL_sound : ∀ l vs, nodeToValueL l = some vs → evL ρ l = some vs
  | [], vs, h => by simpa [nodeToValueL, evL] using h
  | c :: cs, vs, h => by
    simp only [nodeToValueL] at h
    cases h1 : nodeToValue c with
    | none => rw [h1] at h; simp at h
    | some x =>
      cases h2 : nodeToValueL cs with
      | none => rw [h1, h2] at h; simp at h
      | some xs =>
        rw [h1, h2] at h; simp at h; subst h
        simp [evL, nodeToValue_sound c x h1, nodeToValueL_sound cs xs h2]
theorem mapEntries_sound : ∀ l es, mapEntries l = some es → evMap ρ l = some es
  | [], es, h => by simpa [mapEntries, evMap] using h
  | .node (.const (.str k)) (v :: _) :: cs, es, h => by
    simp only [mapEntries] at h
    cases h1 : nodeToValue v with
    | none => rw [h1] at h; simp at h
    | some x =>
      cases h2 : mapEntries cs with
      | none => rw [h1, h2] at h; simp at h
      | some xs =>
        rw [h1, h2] at h; simp at h; subst h
        simp [evMap, nodeToValue_sound v x h1, mapEntries_sound cs xs h2]
  | .node (.const (.str k)) [] :: cs, es, h => by simp [mapEntries] at h
  | .node (.const .null) _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.const (.bool _)) _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.const (.int _)) _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.const (.list _)) _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.const (.map _)) _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.const (.other _)) _ :: cs, es, h => by simp [mapEntries] at h
  | .node .list _ :: cs, es, h => by simp [mapEntries] at h
  | .node .map _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.func _) _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.var _) _ :: cs, es, h => by simp [mapEntries] at h
  | .node (.other _) _ :: cs, es, h => by simp [mapEntries] at h
end

/- Pure literal trees: constants, lists of them, maps with string-key entries. -/
mutual
def pure : FE → Bool
  | .node (.const _) _ => true
  | .node .list cs => pureL cs
  | .node .map cs => pureMap cs
  | _ => false
def pureL : List FE → Bool
  | [] => true
  | c :: cs => pure c && pureL cs
def pureMap : List FE → Bool
  | [] => true
  | .node (.const (.str _)) (v :: _) :: cs => pure v && pureMap cs
  | _ :: _ => false
end

mutual
theorem nodeToValue_complete : ∀ e, pure e = true → (nodeToValue e).isSome
  | .node (.const _) _, _ => rfl
  | .node .list cs, h => by
    have := nodeToValueL_complete cs h
    simp only [nodeToValue]; cases hh : nodeToValueL cs <;> simp_all
  | .node .map cs, h => by
    have := mapEntries_complete cs h
    simp only [nodeToValue]; cases hh : mapEntries cs <;> simp_all
  | .node (.func _) _, h => by simp [pure] at h
  | .node (.var _) _, h => by simp [pure] at h
  | .node (.other _) _, h => by simp [pure] at h
theorem nodeToValueL_complete : ∀ l, pureL l = true → (nodeToValueL l).isSome
  | [], _ => rfl
  | c :: cs, h => by
    simp only [pureL, Bool.and_eq_true] at h
    have h1 := nodeToValue_complete c h.1
    have h2 := nodeToValueL_complete cs h.2
    simp only [nodeToValueL]
    cases hc : nodeToValue c <;> cases hcs : nodeToValueL cs <;> simp_all
theorem mapEntries_complete : ∀ l, pureMap l = true → (mapEntries l).isSome
  | [], _ => rfl
  | .node (.const (.str k)) (v :: _) :: cs, h => by
    simp only [pureMap, Bool.and_eq_true] at h
    have h1 := nodeToValue_complete v h.1
    have h2 := mapEntries_complete cs h.2
    simp only [mapEntries]
    cases hc : nodeToValue v <;> cases hcs : mapEntries cs <;> simp_all
  | .node (.const (.str k)) [] :: cs, h => by simp [pureMap] at h
  | .node (.const .null) _ :: cs, h => by simp [pureMap] at h
  | .node (.const (.bool _)) _ :: cs, h => by simp [pureMap] at h
  | .node (.const (.int _)) _ :: cs, h => by simp [pureMap] at h
  | .node (.const (.list _)) _ :: cs, h => by simp [pureMap] at h
  | .node (.const (.map _)) _ :: cs, h => by simp [pureMap] at h
  | .node (.const (.other _)) _ :: cs, h => by simp [pureMap] at h
  | .node .list _ :: cs, h => by simp [pureMap] at h
  | .node .map _ :: cs, h => by simp [pureMap] at h
  | .node (.func _) _ :: cs, h => by simp [pureMap] at h
  | .node (.var _) _ :: cs, h => by simp [pureMap] at h
  | .node (.other _) _ :: cs, h => by simp [pureMap] at h
end

/-- A duplicate key keeps its first position and its last value, as at runtime. -/
theorem fold_dup_key : nodeToValue (.node .map
    [.node (.const (.str "a")) [.node (.const (.int 1)) []],
     .node (.const (.str "a")) [.node (.const (.int 2)) []]]) = some (.map [("a", .int 2)]) := rfl

/-! ## `collect_constant_args` and the folding step -/

def collectConstArgs : List FE → Option (List Val)
  | [] => some []
  | c :: cs => match exprTreeToValue c, collectConstArgs cs with
    | some v, some vs => some (v :: vs)
    | _, _ => none

theorem collectConstArgs_spec (cs : List FE) (vs : List Val) (h : collectConstArgs cs = some vs) :
    cs.length = vs.length ∧ evL ρ cs = some vs := by
  induction cs generalizing vs with
  | nil => simp [collectConstArgs] at h; subst h; exact ⟨rfl, rfl⟩
  | cons c cs ih =>
    simp only [collectConstArgs] at h
    cases h1 : exprTreeToValue c with
    | none => rw [h1] at h; simp at h
    | some x =>
      cases h2 : collectConstArgs cs with
      | none => rw [h1, h2] at h; simp at h
      | some xs =>
        rw [h1, h2] at h; simp at h; subst h
        obtain ⟨hl, he⟩ := ih xs h2
        refine ⟨by simp [hl], ?_⟩
        simp [evL, nodeToValue_sound ρ c x h1, he]

/-- The fold of `bind_expr_node` (binder.rs:2114-2126): when the call is pure,
its arguments are all constant, their number and types fit, and `pure_fn`
succeeds, the call is replaced by its value. -/
def foldCall (pureFn : Option (List Val → Except String Val)) (arity : Nat) (typeOk : List Val → Bool)
    (cs : List FE) : Option Val :=
  match pureFn with
  | none => none
  | some f => match collectConstArgs cs with
    | none => none
    | some args =>
      if args.length = arity ∧ typeOk args then
        match f args with
        | .ok v => some v
        | .error _ => none
      else none

/-- …and the folded value is the runtime value: the runtime evaluates the
arguments to the same constants and calls the function, which agrees with
`pure_fn` on them (the registration contract `hf`). An erroring `pure_fn`
leaves the call in place, so the error surfaces at runtime only if a row
reaches it. -/
theorem foldCall_sound (f : List Val → Except String Val) (run : List Val → Except String Val)
    (arity : Nat) (typeOk : List Val → Bool) (cs : List FE) (v : Val)
    (hf : ∀ args, f args = .ok v → run args = .ok v)
    (h : foldCall (some f) arity typeOk cs = some v) :
    ∃ args, evL ρ cs = some args ∧ run args = .ok v := by
  simp only [foldCall] at h
  cases hc : collectConstArgs cs with
  | none => rw [hc] at h; simp at h
  | some args =>
    rw [hc] at h
    simp only at h
    split at h
    · cases hfa : f args with
      | error e => rw [hfa] at h; simp at h
      | ok w =>
        rw [hfa] at h; simp at h; subst h
        exact ⟨args, (collectConstArgs_spec ρ cs args hc).2, hf args hfa⟩
    · simp at h

/-! ## `rewrite_struct_constructor` -/

/-- `HashMap` from the literal map's entries (later duplicates replace). -/
def entriesOf : List FE → Option (List (String × FE))
  | [] => some []
  | .node (.const (.str k)) (v :: _) :: cs => (entriesOf cs).map ((k, v) :: ·)
  | _ :: _ => none

def lastLookup (es : List (String × FE)) (k : String) : Option FE :=
  (es.reverse.find? (·.1 == k)).map Prod.snd

/-- The rewritten children: one per slot, `Constant(Null)` if absent. `none`
if the call is left alone (not a single literal map, or unknown keys). -/
def rewriteStruct (slots : List String) (cs : List FE) : Option (List FE) :=
  match cs with
  | [.node .map entries] =>
    match entriesOf entries with
    | none => none
    | some es =>
      if es.all (fun kv => slots.contains kv.1) then
        some (slots.map fun s => (lastLookup es s).getD (.node (.const .null) []))
      else none
  | _ => none

theorem rewriteStruct_slots (slots : List String) (cs out : List FE) (h : rewriteStruct slots cs = some out) :
    out.length = slots.length := by
  unfold rewriteStruct at h
  split at h
  · split at h
    · simp at h
    · split at h
      · simp at h; subst h; simp
      · simp at h
  · simp at h

/-- Unknown keys: no rewrite (the map function reports them). -/
theorem rewriteStruct_unknown (slots : List String) (entries : List FE) (es : List (String × FE))
    (he : entriesOf entries = some es) (k : String) (hk : k ∈ es.map Prod.fst) (hs : k ∉ slots) :
    rewriteStruct slots [.node .map entries] = none := by
  simp only [rewriteStruct, he]
  have : es.all (fun kv => slots.contains kv.1) = false := by
    rw [List.all_eq_false]
    obtain ⟨kv, hkv, rfl⟩ := List.mem_map.1 hk
    exact ⟨kv, hkv, by simpa using hs⟩
  rw [this]; rfl

/-- What the rewrite produces: each slot's (last) entry, `Null` when absent, all
keys known. Under the constructor contract (`struct_fn` applied to these slot
values = the map-taking function applied to the literal map) the call's value
is unchanged. -/
theorem rewriteStruct_shape (slots : List String) (cs out : List FE) (h : rewriteStruct slots cs = some out) :
    ∃ entries es, cs = [.node .map entries] ∧ entriesOf entries = some es ∧
      (∀ kv ∈ es, slots.contains kv.1 = true) ∧
      out = slots.map fun s => (lastLookup es s).getD (.node (.const .null) []) := by
  unfold rewriteStruct at h
  split at h
  · rename_i entries
    split at h
    · simp at h
    · rename_i es hes
      split at h
      · rename_i hall
        simp at h; subst h
        exact ⟨entries, es, rfl, hes, fun kv hkv => by simpa using List.all_eq_true.1 hall kv hkv, rfl⟩
      · simp at h
  · simp at h

/-! ## `rewrite_compiled_regex` -/

inductive RegexKind | matches | matchList | replace
  deriving DecidableEq

def regexKind : String → Option RegexKind
  | "regex_matches" => some .matches
  | "string.matchRegEx" => some .matchList
  | "string.replaceRegEx" => some .replace
  | _ => none

/-- `some (kind, pattern, children without the pattern)` when rewritten. -/
def rewriteRegex (compiles : String → Bool) (name : String) (cs : List FE) :
    Option (RegexKind × String × List FE) :=
  match regexKind name with
  | none => none
  | some k => match cs with
    | c0 :: .node (.const (.str pat)) _ :: rest => if compiles pat then some (k, pat, c0 :: rest) else none
    | _ => none

theorem rewriteRegex_spec (compiles : String → Bool) (name : String) (cs : List FE)
    (k : RegexKind) (pat : String) (cs' : List FE) (h : rewriteRegex compiles name cs = some (k, pat, cs')) :
    regexKind name = some k ∧ compiles pat = true ∧
      ∃ c0 rest pc, cs = c0 :: .node (.const (.str pat)) pc :: rest ∧ cs' = c0 :: rest := by
  unfold rewriteRegex at h
  split at h
  · simp at h
  · rename_i k' hk
    split at h
    · rename_i c0 pat' pc rest
      split at h
      · simp at h; obtain ⟨rfl, rfl, rfl⟩ := h; exact ⟨hk, by assumption, c0, rest, pc, rfl, rfl⟩
      · simp at h
    · simp at h

/-- A pattern that does not compile, or is not a constant string, is left to the runtime. -/
theorem rewriteRegex_bad (compiles : String → Bool) (name : String) (c0 : FE) (pat : String) (pc rest : List FE)
    (h : compiles pat = false) : rewriteRegex compiles name (c0 :: .node (.const (.str pat)) pc :: rest) = none := by
  unfold rewriteRegex; split <;> simp [h]

end FalkorBinder.Fold
