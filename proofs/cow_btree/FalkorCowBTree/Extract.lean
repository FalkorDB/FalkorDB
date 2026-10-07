import FalkorCowBTree.Lookup

/-!
# The generic cursor (`Extract`, #2278) and `heap_bytes`

`RangeIter` became generic over an `Extract` (`cursor.rs:16-53`): `DocExtract` yields the doc
(`range`, `point`), `TupleExtract` the `(key, doc)` pair (`range_tuples`). `next` now reads the key
only when the extract needs it or the leaf is a boundary leaf (`cursor.rs:187-195`). We model that
`next` (`Cursor.nextWith`) and prove it yields `make(keyOf e, docOf e)` for exactly the entries the
old `Cursor.next` (and hence `range_spec`) yields.
-/

namespace CowBTree

/-! ## `Extract` -/

/-- The `Extract` trait (`cursor.rs:16-24`): `NEEDS_KEY` and `make`. -/
structure Extract (α : Type) where
  needsKey : Bool
  make : Nat → Nat → α

/-- `DocExtract` (`cursor.rs:27-39`): `NEEDS_KEY = false`, `make(_key, doc) = doc`. -/
def DocExtract : Extract Nat := ⟨false, fun _ d => d⟩

/-- `TupleExtract` (`cursor.rs:41-53`): `NEEDS_KEY = true`, `make(key, doc) = (key, doc)`. -/
def TupleExtract : Extract (Nat × Nat) := ⟨true, fun k d => (k, d)⟩

/-- The contract `NEEDS_KEY = false` relies on: `make` ignores the key. -/
def Extract.Lawful {α} (ex : Extract α) : Prop := ex.needsKey = false → ∀ k k' d, ex.make k d = ex.make k' d

theorem DocExtract_make (k d : Nat) : DocExtract.make k d = d := rfl
theorem TupleExtract_make (k d : Nat) : TupleExtract.make k d = (k, d) := rfl
theorem DocExtract_lawful : DocExtract.Lawful := fun _ _ _ _ => rfl
theorem TupleExtract_lawful : TupleExtract.Lawful := fun h => by simp [TupleExtract] at h

/-- `Iterator::next` for `RangeIter<.., E>` (`cursor.rs:181-207`). -/
def Cursor.nextWith {α} (ex : Extract α) (cur : Cursor) : Option (α × Cursor) :=
  match hl : cur.leaf with
  | none => none                                                    -- line 183
  | some es =>
    if hp : cur.pos < es.length then
      let e := es[cur.pos]
      let needKey := ex.needsKey || !cur.whole                      -- line 189
      let key := if needKey then keyOf e else 0                     -- line 190
      if !cur.whole && decide (key > cur.hi) then none              -- lines 191-194
      else some (ex.make key (docOf e), { cur with pos := cur.pos + 1 })  -- lines 196-202
    else
      match ha : advanceLeaf cur.stack with                         -- line 204
      | none => none
      | some (stk', es') =>
        Cursor.nextWith ex { stack := stk', leaf := some es', whole := wholeOf es' cur.hi, pos := 0, hi := cur.hi }
termination_by stackWeight cur.stack
decreasing_by exact advanceLeaf_weight _ _ _ ha

/-- **The lazy key read is never wrong**: for a lawful extract, the generic `next` yields
    `make(key, doc)` of exactly the entry the reference cursor yields, and moves to the same state. -/
theorem nextWith_eq {α} (ex : Extract α) (hex : ex.Lawful) : ∀ (w : Nat) (cur : Cursor), stackWeight cur.stack = w →
    cur.nextWith ex = cur.next.map (fun p => (ex.make (keyOf p.1) (docOf p.1), p.2)) := by
  intro w
  induction w using Nat.strongRecOn with
  | ind w ih =>
  intro cur hwt
  match hl : cur.leaf with
  | none =>
    rw [next_none cur hl, Cursor.nextWith]; split
    · rfl
    · rename_i h; rw [hl] at h; simp at h
  | some es =>
    by_cases hp : cur.pos < es.length
    · rw [next_in cur es hl hp, Cursor.nextWith]
      split
      · rename_i h; rw [hl] at h; simp at h
      · rename_i es' h
        rw [hl] at h; simp only [Option.some.injEq] at h; subst h
        simp only [hp, ↓reduceDIte]
        cases hw : cur.whole
        · simp only [Bool.not_false, Bool.or_true, ↓reduceIte, Bool.true_and]
          split <;> rfl
        · simp only [Bool.not_true, Bool.or_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
            Option.map_some]
          cases hk : ex.needsKey
          · simp only [Bool.false_eq_true, ↓reduceIte]
            congr 2
            exact hex hk _ _ _
          · rfl
    · rw [next_adv cur es hl hp, Cursor.nextWith]
      split
      · rename_i h; rw [hl] at h; simp at h
      · rename_i es' h
        rw [hl] at h; simp only [Option.some.injEq] at h; subst h
        simp only [hp, ↓reduceDIte]
        split
        · next ha => simp only [ha, Option.map_none]
        · next stk' es'' ha =>
          simp only [ha]
          exact ih _ (by have := advanceLeaf_weight _ _ _ ha; rw [← hwt]; exact this) _ rfl

/-- Run the generic cursor for at most `n` yields. -/
def Cursor.takeWith {α} (ex : Extract α) : Nat → Cursor → List α
  | 0, _ => []
  | n + 1, cur => match cur.nextWith ex with
    | none => []
    | some (a, cur') => a :: Cursor.takeWith ex n cur'

theorem takeWith_eq {α} (ex : Extract α) (hex : ex.Lawful) : ∀ (n : Nat) (cur : Cursor),
    cur.takeWith ex n = (cur.take n).map (fun e => ex.make (keyOf e) (docOf e))
  | 0, _ => rfl
  | n + 1, cur => by
    simp only [Cursor.takeWith, Cursor.take]
    rw [nextWith_eq ex hex _ cur rfl]
    cases cur.next with
    | none => rfl
    | some p => simp only [Option.map_some]; rw [takeWith_eq ex hex n p.2]; rfl

/-- `RangeIter::new(root, (lo, 0), hi)` collected through extract `ex` (at most `n` results). -/
def rangeWith {α} (ex : Extract α) (h : Nat) (root : Node) (lo hi n : Nat) : List α :=
  (Cursor.new h root lo hi).takeWith ex n

/-- **The generic cursor yields `make(key, doc)` of exactly the entries with key in `[lo, hi]`, in
    `(key, doc)` order**, for every lawful extract. -/
theorem rangeWith_spec {α} (ex : Extract α) (hex : ex.Lawful) (c : Cfg) (h : Nat) (root : Node) (lo hi n : Nat)
    (hw : TreeWF c h root) (hn : (toList h root).length ≤ n) :
    rangeWith ex h root lo hi n =
      ((toList h root).filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi))).map (fun e => ex.make (keyOf e) (docOf e)) := by
  unfold rangeWith
  rw [takeWith_eq ex hex, ← range, range_spec c h root lo hi n hw hn]

/-- **`CowBTree::range` / `point` (`mod.rs:297`, `:307`)** as the Rust yields them: the docs. -/
theorem rangeDocs_spec (c : Cfg) (h : Nat) (root : Node) (lo hi n : Nat) (hw : TreeWF c h root)
    (hn : (toList h root).length ≤ n) :
    rangeWith DocExtract h root lo hi n =
      ((toList h root).filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi))).map docOf :=
  rangeWith_spec DocExtract DocExtract_lawful c h root lo hi n hw hn

/-- **`CowBTree::range_tuples` (`mod.rs:391`)**: the `(key, doc)` pairs of exactly the entries with key
    in `[lo, hi]`, in order. -/
theorem rangeTuples_spec (c : Cfg) (h : Nat) (root : Node) (lo hi n : Nat) (hw : TreeWF c h root)
    (hn : (toList h root).length ≤ n) :
    rangeWith TupleExtract h root lo hi n =
      ((toList h root).filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi))).map (fun e => (keyOf e, docOf e)) :=
  rangeWith_spec TupleExtract TupleExtract_lawful c h root lo hi n hw hn

/-- ...and the pairs are the entries themselves (the tuple cursor loses nothing). -/
theorem rangeTuples_enc (c : Cfg) (h : Nat) (root : Node) (lo hi n : Nat) (hw : TreeWF c h root)
    (hn : (toList h root).length ≤ n) :
    (rangeWith TupleExtract h root lo hi n).map (fun p => enc p.1 p.2) =
      (toList h root).filter (fun e => decide (lo ≤ keyOf e ∧ keyOf e ≤ hi)) := by
  rw [rangeTuples_spec c h root lo hi n hw hn, List.map_map]
  conv => rhs; rw [← List.map_id ((toList h root).filter _)]
  apply List.map_congr_left
  intro e _
  simp only [Function.comp, enc, keyOf, docOf, id]
  exact Nat.div_add_mod' e W

/-! ## `heap_bytes` (`mod.rs:402-424`) -/

/-- What `heap_bytes` reads off each page: a leaf's blob length (`leaf.raw().len()`, fixed by the byte
    encoding — `proofs/index_layer` proves it per format), a branch's two `Vec` capacities (not part of
    the tree model: any value; `>= len`), `size_of::<(u64, u64)>()` and `size_of::<Node>()`. -/
structure Mem where
  raw : List E → Nat
  sepCap : List E → List Node → Nat
  childCap : List E → List Node → Nat
  sepSize : Nat := 16
  nodeSize : Nat

/-- One branch's own bytes (`mod.rs:417-419`). -/
def Mem.branch (m : Mem) (seps : List E) (cs : List Node) : Nat :=
  m.sepCap seps cs * m.sepSize + m.childCap seps cs * m.nodeSize

/-- The nested `walk` of `heap_bytes` (`mod.rs:403-421`), accumulator-passing as in the Rust. -/
def heapWalk (m : Mem) : Nat → Node → Nat → Nat
  | _, .leaf es, acc => acc + m.raw es                                          -- line 410
  | 0, .branch _ _, acc => acc
  | h + 1, .branch seps cs, acc =>
    cs.foldl (fun a ch => heapWalk m h ch a) (acc + m.branch seps cs)          -- lines 417-420

/-- `CowBTree::heap_bytes` (`mod.rs:402`). -/
def heapBytes (m : Mem) (h : Nat) (root : Node) : Nat := heapWalk m h root 0

/-- The closed form: every page contributes its own bytes once. -/
def heapSum (m : Mem) : Nat → Node → Nat
  | _, .leaf es => m.raw es
  | 0, .branch _ _ => 0
  | h + 1, .branch seps cs => m.branch seps cs + (cs.map (heapSum m h)).sum

/-- Leaf part / branch part of the closed form. -/
def leafBytes (m : Mem) : Nat → Node → Nat
  | _, .leaf es => m.raw es
  | 0, .branch _ _ => 0
  | h + 1, .branch _ cs => (cs.map (leafBytes m h)).sum

def branchBytes (m : Mem) : Nat → Node → Nat
  | _, .leaf _ => 0
  | 0, .branch _ _ => 0
  | h + 1, .branch seps cs => m.branch seps cs + (cs.map (branchBytes m h)).sum

theorem foldl_heapWalk (m : Mem) (h : Nat) (ih : ∀ n acc, heapWalk m h n acc = acc + heapSum m h n) :
    ∀ (cs : List Node) (acc : Nat), cs.foldl (fun a ch => heapWalk m h ch a) acc = acc + (cs.map (heapSum m h)).sum
  | [], acc => by simp
  | c :: cs, acc => by
    simp only [List.foldl_cons, List.map_cons, List.sum_cons]
    rw [foldl_heapWalk m h ih cs, ih]; omega

theorem heapWalk_eq (m : Mem) : ∀ (h : Nat) (n : Node) (acc : Nat), heapWalk m h n acc = acc + heapSum m h n
  | _, .leaf es, acc => by simp [heapWalk, heapSum]
  | 0, .branch _ _, acc => rfl
  | h + 1, .branch seps cs, acc => by
    simp only [heapWalk, heapSum]
    rw [foldl_heapWalk m h (heapWalk_eq m h) cs]; omega

/-- **`heap_bytes` counts every page of the tree exactly once.** -/
theorem heapBytes_eq (m : Mem) (h : Nat) (root : Node) : heapBytes m h root = heapSum m h root := by
  simp [heapBytes, heapWalk_eq]

theorem sum_map_add' {α} (f g : α → Nat) : ∀ (l : List α), (l.map (fun x => f x + g x)).sum = (l.map f).sum + (l.map g).sum
  | [] => rfl
  | a :: l => by simp only [List.map_cons, List.sum_cons]; rw [sum_map_add' f g l]; omega

theorem sum_map_mul' {α} (f : α → Nat) (k : Nat) : ∀ (l : List α), (l.map (fun x => f x * k)).sum = (l.map f).sum * k
  | [] => by simp
  | a :: l => by simp only [List.map_cons, List.sum_cons]; rw [sum_map_mul' f k l, Nat.add_mul]

theorem heapSum_split (m : Mem) : ∀ (h : Nat) (n : Node), heapSum m h n = leafBytes m h n + branchBytes m h n
  | _, .leaf es => by simp [heapSum, leafBytes, branchBytes]
  | 0, .branch _ _ => rfl
  | h + 1, .branch seps cs => by
    simp only [heapSum, leafBytes, branchBytes]
    have : (cs.map (heapSum m h)).sum = (cs.map (leafBytes m h)).sum + (cs.map (branchBytes m h)).sum := by
      rw [List.map_congr_left (fun n _ => heapSum_split m h n), sum_map_add']
    omega

/-- With a per-entry blob size (the AoS layout: `raw = count * (8 + DOC_BYTES)`, `index_layer`
    `aosBuild_length`), the leaf part is exactly `entries * stride`. -/
theorem leafBytes_stride (m : Mem) (stride : Nat) (hraw : ∀ es, m.raw es = es.length * stride) :
    ∀ (h : Nat) (n : Node), leafBytes m h n = (toList h n).length * stride
  | _, .leaf es => by simp [leafBytes, toList, hraw]
  | 0, .branch _ _ => by simp [leafBytes, toList]
  | h + 1, .branch _ cs => by
    simp only [leafBytes, toList, List.length_flatten, List.map_map]
    rw [← sum_map_mul']
    congr 1
    apply List.map_congr_left
    intro n _
    simp only [Function.comp]
    rw [leafBytes_stride m stride hraw h n]

/-- **Headline (`heap_bytes`, AoS)**: with AoS leaves `heap_bytes = entries * (8 + DOC_BYTES) + the
    branch vectors' bytes` — so narrowing `DOC_BYTES` from 8 to 4 saves exactly 4 bytes per entry. -/
theorem heapBytes_aos (m : Mem) (D : Nat) (hraw : ∀ es, m.raw es = es.length * (8 + D)) (h : Nat) (root : Node) :
    heapBytes m h root = (toList h root).length * (8 + D) + branchBytes m h root := by
  rw [heapBytes_eq, heapSum_split, leafBytes_stride m (8 + D) hraw]

end CowBTree
