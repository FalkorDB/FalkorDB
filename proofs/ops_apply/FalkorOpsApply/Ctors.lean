/-
Constructors and small helpers of the apply-family operators.

| here | there |
| --- | --- |
| `mergeChildIdx`, `MergeSt.new` | `MergeOp::new` (merge.rs:68) |
| `onceGet`                      | `OnceCell::get_or_init` in `MergeOp::resolve_pattern` (merge.rs:98), `resolve_on_create_set_items` (:103), `resolve_on_match_set_items` (:108) |
| `mergeHashInput`, `mergeHash`  | `MergeOp::compute_merge_pattern_hash` (merge.rs:180) |
| `semiRight`                    | `SemiApplyOp::new` (semi_apply.rs:45) |
| `orBranches`                   | `OrApplyMultiplexerOp::new` (or_apply_multiplexer.rs:46) |
| `forEachBody`                  | `ForEachOp::new` (foreach.rs:57) |
| `storeArg`                     | `UnionOp::store_argument_batch` (union.rs:57) |

A plan node is its child-index list (`runtime.plan.node(idx).child(i)`); `child(i)` out of
range panics (`none`). `num_children() - 1` on 0 children underflows `usize` (debug panic;
release wraps and `child` panics) — also `none`.
-/
namespace Ctors

/-! ## `MergeOp::new` (merge.rs:68) -/

/-- merge.rs:76-80 -/
def mergeChildIdx {I : Type} (children : List I) : Option I :=
  if children.length = 1 then children[0]? else children[1]?

structure MergeSt (C I P S A : Type) where
  child : C
  pending : List A
  mergeChild : I
  pattern : P
  resolvedPattern : Option P
  onCreate : S
  resolvedOnCreate : Option S
  onMatch : S
  resolvedOnMatch : Option S
  isError : Bool
  idx : I

def MergeSt.new {C I P S A : Type} (child : C) (children : List I) (pattern : P) (onCreate onMatch : S)
    (idx : I) : Option (MergeSt C I P S A) :=
  (mergeChildIdx children).map fun m =>
    { child, pending := [], mergeChild := m, pattern, resolvedPattern := none, onCreate,
      resolvedOnCreate := none, onMatch, resolvedOnMatch := none, isError := false, idx }

/-- The MERGE sub-plan is the only child when there is one (no input), else the second
(child 0 is the input stream); all memo cells start empty, no pending rows, no error. -/
theorem mergeNew_spec {C I P S A : Type} (c : C) (children : List I) (p : P) (oc om : S) (i : I)
    (s : MergeSt C I P S A) (h : MergeSt.new c children p oc om i = some s) :
    (children.length = 1 → children[0]? = some s.mergeChild) ∧
    (children.length ≠ 1 → children[1]? = some s.mergeChild) ∧
    s.pending = [] ∧ s.resolvedPattern = none ∧ s.resolvedOnCreate = none ∧
    s.resolvedOnMatch = none ∧ s.isError = false ∧ s.child = c ∧ s.idx = i := by
  unfold MergeSt.new mergeChildIdx at h
  by_cases h1 : children.length = 1
  · simp only [h1, ite_true, Option.map_eq_some_iff] at h
    obtain ⟨m, hm, rfl⟩ := h
    exact ⟨fun _ => hm, fun h' => absurd h1 h', rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  · simp only [h1, ite_false, Option.map_eq_some_iff] at h
    obtain ⟨m, hm, rfl⟩ := h
    exact ⟨fun h' => absurd h' h1, fun _ => hm, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-! ## `OnceCell::get_or_init` memo (merge.rs:98, :103, :108) -/

def onceGet {A : Type} (init : Unit → A) : Option A → A × Option A
  | some a => (a, some a)
  | none => let a := init (); (a, some a)

/-- The resolver runs at most once per operator: the first call stores `init ()`, every
later call returns the stored value, so every call answers `init ()` (resolution is pure). -/
theorem onceGet_spec {A : Type} (init : Unit → A) (c : Option A) (hc : ∀ a, c = some a → a = init ()) :
    (onceGet init c).1 = init () ∧ (onceGet init c).2 = some (init ()) ∧
    onceGet init (onceGet init c).2 = onceGet init c := by
  cases c with
  | none => exact ⟨rfl, rfl, rfl⟩
  | some a => have := hc a rfl; subst this; exact ⟨rfl, rfl, rfl⟩

theorem onceGet_from_new {A : Type} (init : Unit → A) (n : Nat) :
    (Nat.repeat (fun c => (onceGet init c).2) n none) = (if n = 0 then none else some (init ())) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [Nat.repeat, ih]; split <;> rfl

/-! ## `compute_merge_pattern_hash` (merge.rs:180) -/

/-- What gets fed to the `FxHasher`, in order. -/
inductive Tok (V L T : Type) where
  | val (v : V)
  | label (l : L)
  | attrs (m : List (String × Option V))   -- an evaluated `Value::Map` (`none` = Null)
  | types (ts : T)

structure PNode (L E : Type) where
  alias : Nat
  labels : List L
  attrs : E

structure PRel (T E : Type) where
  types : T
  src : Nat
  dst : Nat
  attrs : E

/-- First null-valued key of a map, if any (merge.rs:200-206 / :229-235). -/
def firstNull {V : Type} : List (String × Option V) → Option String
  | [] => none
  | (k, none) :: _ => some k
  | (_, some _) :: ms => firstNull ms

def nodeToks {V L T E : Type} (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (n : PNode L E) : Except String (List (Tok V L T)) :=
  match get n.alias with
  | some v => .ok [.val v]
  | none => do
    let m ← ev n.attrs
    match firstNull m with
    | some k => .error s!"Cannot merge node using null property value for key '{k}'"
    | none => pure (n.labels.map .label ++ [.attrs m])

def relToks {V L T E : Type} (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (r : PRel T E) : Except String (List (Tok V L T)) := do
  let pre : List (Tok V L T) := [.types r.types] ++ ((get r.src).map .val).toList ++ ((get r.dst).map .val).toList
  let m ← ev r.attrs
  match firstNull m with
  | some k => .error s!"Cannot merge relationship using null property value for key '{k}'"
  | none => pure (pre ++ [.attrs m])

def concatM {α β ε : Type} (f : α → Except ε (List β)) : List α → Except ε (List β)
  | [] => .ok []
  | x :: xs => do let a ← f x; let b ← concatM f xs; pure (a ++ b)

def mergeHashInput {V L T E : Type} (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (nodes : List (PNode L E)) (rels : List (PRel T E)) : Except String (List (Tok V L T)) := do
  let a ← concatM (nodeToks get ev) nodes
  let b ← concatM (relToks get ev) rels
  pure (a ++ b)

def mergeHash {V L T E S : Type} (d : S) (h : S → Tok V L T → S) (fin : S → UInt64)
    (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (nodes : List (PNode L E)) (rels : List (PRel T E)) : Except String UInt64 :=
  (mergeHashInput get ev nodes rels).map fun ts => fin (ts.foldl h d)

/-- A bound pattern node contributes only its value (labels/properties are not evaluated). -/
theorem nodeToks_bound {V L T E : Type} (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (n : PNode L E) (v : V) (h : get n.alias = some v) :
    (nodeToks get ev n : Except String (List (Tok V L T))) = .ok [.val v] := by
  simp [nodeToks, h]

/-- An unbound node whose property map holds a null is rejected (C: same error). -/
theorem nodeToks_null {V L T E : Type} (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (n : PNode L E) (m : List (String × Option V)) (k : String) (h : get n.alias = none)
    (hm : ev n.attrs = .ok m) (hk : firstNull m = some k) :
    (nodeToks get ev n : Except String (List (Tok V L T))) =
      .error s!"Cannot merge node using null property value for key '{k}'" := by
  simp [nodeToks, h, hm, hk, bind, Except.bind]

theorem relToks_null {V L T E : Type} (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (r : PRel T E) (m : List (String × Option V)) (k : String)
    (hm : ev r.attrs = .ok m) (hk : firstNull m = some k) :
    (relToks get ev r : Except String (List (Tok V L T))) =
      .error s!"Cannot merge relationship using null property value for key '{k}'" := by
  simp [relToks, hm, hk, bind, Except.bind]

theorem firstNull_none_iff {V : Type} (m : List (String × Option V)) :
    firstNull m = none ↔ ∀ p ∈ m, p.2.isSome := by
  induction m with
  | nil => simp [firstNull]
  | cons p ms ih =>
    obtain ⟨k, v⟩ := p
    cases v <;> simp [firstNull, ih]

/-- The hash is a function of the token stream: two rows producing the same tokens get the
same key (and distinct tokens may collide — FxHash, FINDINGS 6). -/
theorem mergeHash_spec {V L T E S : Type} (d : S) (h : S → Tok V L T → S) (fin : S → UInt64)
    (get : Nat → Option V) (ev : E → Except String (List (String × Option V)))
    (nodes : List (PNode L E)) (rels : List (PRel T E)) :
    mergeHash d h fin get ev nodes rels =
      match mergeHashInput get ev nodes rels with
      | .ok ts => .ok (fin (ts.foldl h d))
      | .error e => .error e := by
  unfold mergeHash; cases mergeHashInput get ev nodes rels <;> rfl

/-! ## `SemiApplyOp::new`, `OrApplyMultiplexerOp::new`, `ForEachOp::new` -/

/-- semi_apply.rs:51: the sub-plan is child 1. -/
def semiRight {I : Type} (children : List I) : Option I := children[1]?

theorem semiRight_spec {I : Type} (a b : I) (rest : List I) : semiRight (a :: b :: rest) = some b := rfl

/-- or_apply_multiplexer.rs:52-57: branches = children `1..=n-1`, in order. -/
def orBranches {I : Type} (children : List I) : Option (List I) :=
  if children.length = 0 then none
  else (List.range' 1 (children.length - 1)).mapM fun i => children[i]?

theorem orBranches_spec {I : Type} (children : List I) (h : children ≠ []) :
    orBranches children = some children.tail := by
  unfold orBranches
  cases children with
  | nil => exact absurd rfl h
  | cons c cs =>
    simp only [List.length_cons, Nat.add_one_ne_zero, ite_false, Nat.add_sub_cancel, List.tail_cons]
    suffices H : ∀ (off : Nat) (l : List I) (pre : List I), pre.length = off →
        (List.range' (off) l.length).mapM (fun i => (pre ++ l)[i]?) = some l by
      have := H 1 cs [c] rfl; simpa using this
    intro off l
    induction l generalizing off with
    | nil => intro pre _; rfl
    | cons x xs ih =>
      intro pre hp
      simp only [List.length_cons, List.range'_succ, List.mapM_cons]
      have h1 : (pre ++ x :: xs)[off]? = some x := by
        rw [List.getElem?_append_right (by omega)]; simp [hp]
      have h2 := ih (off + 1) (pre ++ [x]) (by simp [hp])
      simp only [List.append_assoc, List.singleton_append] at h2
      simp [h1, h2]

/-- foreach.rs:66-67: the body sub-plan is the last child. -/
def forEachBody {I : Type} (children : List I) : Option I :=
  if children.length = 0 then none else children[children.length - 1]?

theorem forEachBody_spec {I : Type} (children : List I) : forEachBody children = children.getLast? := by
  unfold forEachBody
  cases h : children.length with
  | zero => simp [List.eq_nil_of_length_eq_zero h]
  | succ n => simp [List.getLast?_eq_getElem?, h]

/-! ## `UnionOp::store_argument_batch` (union.rs:57) -/

/-- A batch = its rows and optional selection; `into_compacted` = the active rows, densely. -/
def compact {R : Type} (rows : List R) (sel : Option (List Nat)) : List R :=
  match sel with
  | none => rows
  | some idx => idx.filterMap fun i => rows[i]?

def storeArg {R : Type} (_old : Option (List R)) (rows : List R) (sel : Option (List Nat)) : Option (List R) :=
  some (compact rows sel)

/-- The stored argument batch is the active rows, densely, in selection order (row `k` is
input row `sel[k]`), replacing any older one; with no selection it is the batch itself. -/
theorem storeArg_spec {R : Type} (old : Option (List R)) (rows : List R) (sel : List Nat)
    (hs : ∀ i ∈ sel, i < rows.length) :
    storeArg old rows none = some rows ∧
    ∃ out, storeArg old rows (some sel) = some out ∧ out.length = sel.length ∧
      ∀ k (hk : k < sel.length), out[k]? = rows[sel[k]]? := by
  refine ⟨rfl, compact rows (some sel), rfl, ?_⟩
  unfold compact
  induction sel with
  | nil => simp
  | cons i is ih =>
    have hi : rows[i]? = some (rows[i]'(hs i (by simp))) := List.getElem?_eq_getElem _
    obtain ⟨hl, hg⟩ := ih (fun j hj => hs j (List.mem_cons_of_mem _ hj))
    simp only [List.filterMap_cons, hi, List.length_cons]
    refine ⟨by simp [hl], fun k hk => ?_⟩
    cases k with
    | zero => simp
    | succ k => simpa using hg k (by simpa using hk)

end Ctors
