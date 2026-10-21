import Pr3437Review.Chunked

/-!
# PR #3437: chunk loop, chunking, and the three evaluators end to end

| Lean | Rust |
| --- | --- |
| `chunkStep` | `list_loop_chunk` (eval.rs:1459-1550); `useCol` = `self.runtime.is_some() && lp.vector_plan(env).is_some()` (:1470-1472), any value |
| `prRun` | the chunk loops (eval.rs:1350-1359, :1411-1421, :1428-1430) |
| `chunksOf` | `<[Value]>::chunks(LIST_LOOP_CHUNK)` (eval.rs:1350, :1428) |
| `streamLoop` | the streaming buffer (eval.rs:1408-1421) |
| `Source`, `prListComp`, `mainListComp`, `mainSize` | `eval_list_comprehension` (eval.rs:1375-1452) vs *main*'s `ListComprehension` arm (eval.rs:1041-1085) and `size` (functions/list.rs:61-68) |
| `prQuant`, `mainQuant` | `ExprEval::eval_quantifier` (eval.rs:1313-1366) vs *main*'s `Quantifier` arm (eval.rs:1001-1037) |
-/

namespace Pr3437.ListLoop
open FalkorCore FalkorStrList

variable {F : Type}

/-- `list_loop_chunk`: the columnar path when the chunk is long enough and the body allows
it, its result only if it finished; otherwise (or on any error) the per-element path. -/
def chunkStep (b : Body F) (c : Col F) (useCol : List (V F) → Bool) (s : Sink F)
    (chunk : List (V F)) : Except String (Sink F) :=
  if useCol chunk && decide (16 ≤ chunk.length) && (b.hasPred || b.hasProj) then
    match columnar b c s chunk with
    | .ok s' => .ok s'
    | .error _ => perElem b s chunk
  else perElem b s chunk

def prRun (b : Body F) (c : Col F) (useCol : List (V F) → Bool) (s : Sink F)
    (chunks : List (List (V F))) : Except String (Sink F) :=
  chunks.foldlM (chunkStep b c useCol) s

section
variable (b : Body F) (c : Col F) (useCol : List (V F) → Bool)

theorem chunkStep_eq (hd : b.Drops) (hc : c.Sound b.pe b.pj) (s : Sink F) (chunk : List (V F)) :
    chunkStep b c useCol s chunk = ref b.pe b.pj s chunk := by
  unfold chunkStep
  split
  · cases h : columnar b c s chunk with
    | ok s' => exact (columnar_sound b c hd hc s chunk s' h).symm
    | error e => exact perElem_eq b hd s chunk
  · exact perElem_eq b hd s chunk

/-- **Chunked = per-element, for any partition into chunks and any columnar/per-element
choice per chunk.** -/
theorem prRun_eq (hd : b.Drops) (hc : c.Sound b.pe b.pj) :
    ∀ (chunks : List (List (V F))) (s : Sink F), prRun b c useCol s chunks = ref b.pe b.pj s chunks.flatten
  | [], s => rfl
  | ch :: chs, s => by
    simp only [prRun, List.foldlM_cons, List.flatten_cons]
    rw [ref_append, chunkStep_eq b c useCol hd hc]
    cases ref b.pe b.pj s ch with
    | error e => rfl
    | ok s' => exact prRun_eq hd hc chs s'

end

/-! ## Chunking -/

/-- `slice::chunks(n)`. -/
def chunksOf (n : Nat) (l : List α) : List (List α) :=
  if h : 0 < n ∧ l ≠ [] then l.take n :: chunksOf n (l.drop n) else []
termination_by l.length
decreasing_by
  obtain ⟨hn, hl⟩ := h
  have : 0 < l.length := List.length_pos_iff.2 hl
  simp; omega

theorem chunksOf_flatten (n : Nat) (hn : 0 < n) (l : List α) : (chunksOf n l).flatten = l := by
  induction h : l.length using Nat.strongRecOn generalizing l with
  | _ k ih =>
    unfold chunksOf
    by_cases hl : l = []
    · simp [hl]
    · simp only [hn, ne_eq, hl, not_false_eq_true, and_self, dite_true, List.flatten_cons]
      have : (l.drop n).length < k := by
        have : 0 < l.length := List.length_pos_iff.2 hl
        simp; omega
      rw [ih _ this (l.drop n) rfl, List.take_append_drop]

/-- The streaming buffer: push, flush when full, flush the rest at the end (eval.rs:1408-1421). -/
def streamLoop (n : Nat) : List α → List α → List (List α)
  | buf, [] => [buf]
  | buf, x :: xs => if (buf ++ [x]).length = n then (buf ++ [x]) :: streamLoop n [] xs
    else streamLoop n (buf ++ [x]) xs

theorem streamLoop_flatten (n : Nat) : ∀ (l buf : List α), (streamLoop n buf l).flatten = buf ++ l
  | [], buf => by simp [streamLoop]
  | x :: xs, buf => by
    unfold streamLoop
    split
    · simp [streamLoop_flatten n xs []]
    · rw [streamLoop_flatten n xs]; simp

/-! ## End to end -/

def Except.isOk' : Except String α → Bool | .ok _ => true | .error _ => false

/-- What a comprehension's source evaluates to: a streamed list literal / `range`
(`eval_iter_expr`, `none` = empty range), or any other expression (`eval_node`). -/
inductive Source (F : Type) where
  | stream (r : Except String (Option (List (V F))))
  | expr (r : Except String (V F))

def listMismatch (v : V F) : String := "Type mismatch: expected List or Null but was " ++ v.name

/-- *main* eval.rs:1041-1085. -/
def mainListComp (pe pj : V F → Except String (V F)) : Source F → Except String (V F)
  | .stream r => r >>= fun o => match o with
    | none => .ok (.list [])
    | some l => (mainCompLoop pe pj [] l).map .list
  | .expr r => r >>= fun v => match v with
    | .list l => (mainCompLoop pe pj [] l).map .list
    | .null => .ok .null
    | w => .error (listMismatch w)

/-- `size(x)` on what a comprehension returns (functions/list.rs:61-68; a list or null
passes `validate_args_type`, see `ValueOfType`). -/
def mainSize : Except String (V F) → Except String (V F)
  | .ok (.list l) => .ok (.int l.length)
  | .ok .null => .ok .null
  | r => r

def sinkOut (count : Bool) : Sink F → V F
  | .collect acc => if count then .int acc.length else .list acc
  | .tally n => .int n
  | .count .. => .null

/-- `eval_list_comprehension(node, env, _, count)` (eval.rs:1375-1452). -/
def prListComp (b : Body F) (c : Col F) (useCol : List (V F) → Bool) (count : Bool) :
    Source F → Except String (V F)
  | .stream r => r >>= fun o => match o with
    | none => .ok (if count then .int 0 else .list [])
    | some l => (prRun b c useCol (if count then .tally 0 else .collect []) (streamLoop 1024 [] l)).map (sinkOut count)
  | .expr r => r >>= fun v => match v with
    | .list l => (prRun b c useCol (if count then .tally 0 else .collect []) (chunksOf 1024 l)).map (sinkOut count)
    | .null => .ok .null
    | w => .error (listMismatch w)

section
variable (b : Body F) (c : Col F) (useCol : List (V F) → Bool)

theorem run_collect (hd : b.Drops) (hc : c.Sound b.pe b.pj) (chunks : List (List (V F))) :
    (prRun b c useCol (.collect []) chunks).map (sinkOut false) =
      (mainCompLoop b.pe b.pj [] chunks.flatten).map .list := by
  rw [prRun_eq b c useCol hd hc, ref_collect_eq_main]
  cases mainCompLoop b.pe b.pj [] chunks.flatten <;> rfl

theorem run_tally (hd : b.Drops) (hc : c.Sound b.pe b.pj) (chunks : List (List (V F))) :
    (prRun b c useCol (.tally 0) chunks).map (sinkOut true) =
      mainSize ((mainCompLoop b.pe b.pj [] chunks.flatten).map .list) := by
  rw [prRun_eq b c useCol hd hc, ref_tally_eq_collect _ _ _ [] 0]
  cases mainCompLoop b.pe b.pj [] chunks.flatten <;> simp [Except.map, sinkOut, mainSize]

/-- **Claim 1/2 (comprehension).** The PR's list comprehension returns exactly main's:
the same list in the same order, or the same (first) error. -/
theorem comprehension_eq_main (hd : b.Drops) (hc : c.Sound b.pe b.pj) (src : Source F) :
    prListComp b c useCol false src = mainListComp b.pe b.pj src := by
  cases src with
  | stream r =>
    cases r with
    | error e => rfl
    | ok o =>
      cases o with
      | none => rfl
      | some l =>
        simp only [prListComp, mainListComp, bind, Except.bind, Bool.false_eq_true, ite_false]
        rw [run_collect b c useCol hd hc, streamLoop_flatten]; rfl
  | expr r =>
    cases r with
    | error e => rfl
    | ok v =>
      cases v <;> try rfl
      simp only [prListComp, mainListComp, bind, Except.bind, Bool.false_eq_true, ite_false]
      rw [run_collect b c useCol hd hc, chunksOf_flatten 1024 (by decide)]

/-- **Claim 4.** `size(<comprehension>)` counted = `size` of the list main builds (null
source: null; errors, including a projection's, as main). -/
theorem size_eq_main (hd : b.Drops) (hc : c.Sound b.pe b.pj) (src : Source F) :
    prListComp b c useCol true src = mainSize (mainListComp b.pe b.pj src) := by
  cases src with
  | stream r =>
    cases r with
    | error e => rfl
    | ok o =>
      cases o with
      | none => rfl
      | some l =>
        simp only [prListComp, mainListComp, bind, Except.bind, ite_true]
        rw [run_tally b c useCol hd hc, streamLoop_flatten]; rfl
  | expr r =>
    cases r with
    | error e => rfl
    | ok v =>
      cases v <;> try rfl
      simp only [prListComp, mainListComp, bind, Except.bind, ite_true]
      rw [run_tally b c useCol hd hc, chunksOf_flatten 1024 (by decide)]

/-- The counted path evaluates a non-identity projection: when the projection errors on a
passing element (and nothing earlier fails), the counted `size(..)` reports that error. -/
theorem size_projection_error (hd : b.Drops) (hc : c.Sound b.pe b.pj) (l : List (V F)) (e : String)
    (h : mainCompLoop b.pe b.pj [] l = .error e) :
    prListComp b c useCol true (.expr (.ok (.list l))) = .error e := by
  rw [size_eq_main b c useCol hd hc]; simp [mainListComp, bind, Except.bind, h, Except.map, mainSize]

end

/-! ## Quantifiers -/

def kToV : K → V F | .t => .bool true | .f => .bool false | .n => .null

def quantRes (q : QType) (r : Nat × Nat × Nat) : V F := kToV (evalQuantifier q r.1 r.2.1 r.2.2)

/-- *main* eval.rs:1001-1037. -/
def mainQuant (pe : V F → Except String (V F)) (q : QType) (r : Except String (V F)) : Except String (V F) :=
  r >>= fun v => match v with
  | .list l => (mainQuantLoop pe 0 0 0 l).map (quantRes q)
  | .null => .ok .null
  | w => .error ("Type mismatch: expected List but was " ++ w.name)

def countOut (q : QType) : Sink F → V F
  | .count t f n => quantRes q (t, f, n)
  | _ => .null

/-- `ExprEval::eval_quantifier` (eval.rs:1313-1366): a short list runs main's loop
verbatim (:1325-1341); a longer one goes through `ListLoop` with no projection (:1345). -/
def prQuant (b : Body F) (c : Col F) (useCol : List (V F) → Bool) (q : QType) (r : Except String (V F)) :
    Except String (V F) :=
  r >>= fun v => match v with
  | .list l =>
    if l.length < 16 then (mainQuantLoop b.pe 0 0 0 l).map (quantRes q)
    else (prRun b c useCol (.count 0 0 0) (chunksOf 1024 l)).map (countOut q)
  | .null => .ok .null
  | w => .error ("Type mismatch: expected List but was " ++ w.name)

/-- **Claim 1/2 (quantifiers).** `any`/`all`/`none`/`single` give main's answer (Cypher's
three-valued result, `evalQuantifier`) or main's first error. -/
theorem quantifier_eq_main (b : Body F) (c : Col F) (useCol : List (V F) → Bool) (hd : b.Drops)
    (hc : c.Sound b.pe b.pj) (q : QType) (r : Except String (V F)) :
    prQuant b c useCol q r = mainQuant b.pe q r := by
  cases r with
  | error e => rfl
  | ok v =>
    cases v <;> try rfl
    rename_i l
    simp only [prQuant, mainQuant, bind, Except.bind]
    split
    · rfl
    · rw [prRun_eq b c useCol hd hc, chunksOf_flatten 1024 (by decide), ref_count_eq_main]
      cases mainQuantLoop b.pe 0 0 0 l <;> rfl

end Pr3437.ListLoop
