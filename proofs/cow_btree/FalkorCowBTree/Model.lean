/-
# The model: `cow_btree` line by line

Everything here is executable (`#eval`-able) and follows the Rust in
`graph/src/index/falkordb/data_structures/cow_btree/` branch for branch. The
proofs live in the sibling files; this file only *defines*.

## Abstractions (and why they are sound)

* **Entries are one `Nat`.** A Rust entry is `(key: u64, doc: u64)` compared
  lexicographically. We encode it as `key * 2^64 + doc`; `enc_lt_iff` (in
  `Basic.lean`) proves this is an order isomorphism onto `[0, 2^128)`, so every
  `<`/`<=` on tuples in the Rust is exactly `<`/`<=` on the encoding. The
  sentinel `(u64::MAX, u64::MAX)` used by `apply_batch` is `TOP = 2^128 - 1`.
* **Leaf pages are sorted lists.** The three byte encodings (`AosLeaf`,
  `CompactLeaf`, `CompactIndexedLeaf`) are abstracted to the list of entries
  they decode to (`Leaf::to_pairs`). Each Rust fast path (AoS splice, compact
  splice, block-copy merge) computes the *same list* as its slow path
  (decode + edit + `from_pairs`); we model that list once. Encoding fidelity is
  a modelling gap (see REPORT.md) — it is covered by the Rust differential
  tests, not by this proof.
* **`Arc` is a persistent value.** `make_private` always clones before it
  mutates (`node.rs:391`), and leaf buffers are fresh `Arc<[u8]>`s that are
  never written after construction, so from the point of view of any holder of
  an old root every operation is a *pure function* old-tree → new-tree. That is
  what `Snapshot.lean` states; the heap-level frame argument is there too.
* **Heights are ghost parameters.** Rust recursion descends until it meets a
  `Node::Leaf`; we thread the (uniform) height `h` explicitly so Lean sees
  structural recursion. On a tree of uniform depth `h` the two agree.
* **`partition_point` is a count of the true prefix.** Rust binary-searches a
  monotone predicate; on a monotone predicate that equals
  `(l.takeWhile p).length`, which is what we use (`childIndex`, `lowerBound`).
-/

namespace CowBTree

/-- `2^64`: one `u64` field. -/
def W : Nat := 18446744073709551616

/-- A `(key, doc)` entry, encoded as one number (lexicographic order = `Nat` order). A notation, not
    a definition, so `omega` sees plain `Nat`. -/
notation "E" => Nat

/-- Encode `(key, doc)`. -/
def enc (k d : Nat) : E := k * W + d
/-- The key half of an entry. -/
def keyOf (e : E) : Nat := e / W
/-- The doc half of an entry (what `RangeIter` yields). -/
def docOf (e : E) : Nat := e % W
/-- `(u64::MAX, u64::MAX)` — the `child_upper` sentinel of `apply_batch` (`node.rs:202`). -/
def TOP : E := enc (W - 1) (W - 1)

/-- The const-generic parameters and the `const { assert!(..) }` of `mod.rs:113-118`. -/
structure Cfg where
  L : Nat
  B : Nat
  hL : 2 ≤ L ∧ L ≤ 256
  hB : 3 ≤ B

/-- `Node` (`node.rs:14`): a leaf page or a branch (`seps`, `children`, `node.rs:23`). -/
inductive Node where
  | leaf : List E → Node
  | branch : List E → List Node → Node
deriving Repr, Inhabited

/-! ## Vec primitives, as list edits -/

/-- `Vec::insert(i, x)` -/
def insAt {α} (l : List α) (i : Nat) (x : α) : List α := l.take i ++ x :: l.drop i
/-- `Vec::remove(i)` -/
def delAt {α} (l : List α) (i : Nat) : List α := l.take i ++ l.drop (i + 1)
/-- `v[i] = x` -/
def setAt {α} (l : List α) (i : Nat) (x : α) : List α := l.take i ++ x :: l.drop (i + 1)

/-! ## Reading a tree -/

/-- All entries of a node of height `h`, in stored order (`CowBTree::len` walk / `tree_pairs`). -/
def toList : Nat → Node → List E
  | _, .leaf es => es
  | 0, .branch _ _ => []
  | h + 1, .branch _ cs => (cs.map (toList h)).flatten

/-- `Node::min` (`node.rs:155`): first entry of the left spine's leaf. `none` is Rust's panic
    (`leaf.key(0)` on an empty page is an out-of-bounds slice). -/
def minOpt : Nat → Node → Option E
  | _, .leaf es => es.head?
  | 0, .branch _ _ => none
  | h + 1, .branch _ cs => match cs with
    | [] => none
    | c :: _ => minOpt h c

/-- `Branch::child_index` (`node.rs:103`): number of separators `<= x`. -/
def childIndex (seps : List E) (x : E) : Nat := (seps.takeWhile (fun s => decide (s ≤ x))).length

/-! ## Leaf operations (`leaf/mod.rs`) -/

/-- `Leaf::lower_bound_entry` (`leaf/mod.rs:359`): first index with entry `>= x`. -/
def lowerBoundEntry (es : List E) (x : E) : Nat := (es.takeWhile (fun e => decide (e < x))).length

/-- `Leaf::lower_bound` (`leaf/mod.rs:223`): first index with *key* `>= k` (range seek). -/
def lowerBound (es : List E) (k : Nat) : Nat := (es.takeWhile (fun e => decide (keyOf e < k))).length

/-- `LeafInsert` (`leaf/mod.rs:158`). -/
inductive LeafIns where
  | fit (es : List E)
  | split (l : List E) (sep : E) (r : List E)

/-- `Leaf::insert` (`leaf/mod.rs:372`). Both the in-place splice arm (`count < LEAF_MAX`) and the
    rebuild arm produce `take pos ++ x :: drop pos`; only the split decision depends on the count. -/
def leafInsert (c : Cfg) (es : List E) (x : E) : Option LeafIns :=
  let pos := lowerBoundEntry es x
  if es[pos]? = some x then none              -- already present (line 379)
  else
    let ps := insAt es pos x                  -- lines 386-415
    if ps.length ≤ c.L then some (.fit ps)    -- line 416
    else
      let mid := ps.length / 2                -- line 419
      some (.split (ps.take mid) (ps.getD mid 0) (ps.drop mid))

/-- `Leaf::remove` (`leaf/mod.rs:432`): the new page and the underflow flag `new_count < LEAF_MAX / 2`. -/
def leafRemove (c : Cfg) (es : List E) (x : E) : Option (List E × Bool) :=
  let pos := lowerBoundEntry es x
  if es[pos]? = some x then some (delAt es pos, decide (es.length - 1 < c.L / 2))
  else none                                   -- absent (line 439)

/-- `merge_sorted` / `merge_walk` (`leaf/mod.rs:50`, `:97`): two-pointer merge, left-biased on ties,
    dropping an output equal to the previous output. `last` is `out.last()` / `last`. -/
def mergeWalk : Option E → List E → List E → List E
  | _, [], [] => []
  | last, a :: as, [] => if last = some a then mergeWalk last as [] else a :: mergeWalk (some a) as []
  | last, [], b :: bs => if last = some b then mergeWalk last [] bs else b :: mergeWalk (some b) [] bs
  | last, a :: as, b :: bs =>
    if a ≤ b then (if last = some a then mergeWalk last as (b :: bs) else a :: mergeWalk (some a) as (b :: bs))
    else (if last = some b then mergeWalk last (a :: as) bs else b :: mergeWalk (some b) (a :: as) bs)
termination_by _ l r => l.length + r.length

/-- `<[T]>::chunks(n)`. -/
def chunks (n : Nat) (hn : 0 < n) : List α → List (List α)
  | [] => []
  | x :: xs => (x :: xs).take n :: chunks n hn ((x :: xs).drop n)
termination_by l => l.length
decreasing_by simp only [List.length_drop, List.length_cons]; omega

/-- `Leaf::merge_batch` (`leaf/mod.rs:471`). Every fast path (AoS `merge_walk`, compact `merge`,
    `block_copy_merge`) yields the merge-walk list; the slow path re-chunks it. -/
def mergeBatch (c : Cfg) (es batch : List E) : List (List E) :=
  if es.length + batch.length ≤ c.L then [mergeWalk none es batch]
  else chunks c.L (by have := c.hL; omega) (mergeWalk none es batch)

/-! ## Single insert (`node.rs:223`, `mod.rs:169`) -/

/-- `Node::insert_one`. Returns the replacement node and `Some (sep, right)` on a split. -/
def insertOne (c : Cfg) : Nat → Node → E → Node × Option (E × Node)
  | _, .leaf es, x =>
    match leafInsert c es x with
    | none => (.leaf es, none)
    | some (.fit ps) => (.leaf ps, none)
    | some (.split l s r) => (.leaf l, some (s, .leaf r))
  | 0, n@(.branch _ _), _ => (n, none)
  | h + 1, .branch seps cs, x =>
    let i := childIndex seps x                               -- line 245
    match insertOne c h (cs.getD i default) x with           -- line 246
    | (c', none) => (.branch seps (setAt cs i c'), none)     -- line 248
    | (c', some (s, r)) =>
      let seps' := insAt seps i s                            -- line 251
      let cs' := insAt (setAt cs i c') (i + 1) r             -- line 252
      if cs'.length ≤ c.B then (.branch seps' cs', none)     -- line 255
      else
        let mid := cs'.length / 2                            -- line 260
        let ls := seps'.take mid                             -- line 262 (split_off keeps the prefix)
        (.branch ls.dropLast (cs'.take mid),                 -- line 263 (pop)
         some (ls.getLast?.getD 0, .branch (seps'.drop mid) (cs'.drop mid)))

/-- `CowBTree::insert` (`mod.rs:169`). The ghost height grows by one on a root split. -/
def insert (c : Cfg) (h : Nat) (root : Node) (x : E) : Node × Nat :=
  match insertOne c h root x with
  | (n, none) => (n, h)
  | (n, some (s, r)) => (.branch [s] [n, r], h + 1)

/-! ## Single remove (`node.rs:363`, `mod.rs:203`) -/

/-- `Combined` (`node.rs:38`). -/
inductive Combined where
  | one (n : Node)
  | two (l : Node) (sep : E) (r : Node)

/-- `Node::combine` (`node.rs:279`). The AoS byte-concat arm (`aos_combine`, `node.rs:342`) splits at
    `count / 2` entries exactly like the generic arm, so both are this one definition. -/
def combine (c : Cfg) (a : Node) (sep : E) (b : Node) : Combined :=
  match a, b with
  | .leaf p, .leaf q =>
    let ps := p ++ q
    if ps.length ≤ c.L then .one (.leaf ps)
    else
      let mid := ps.length / 2
      .two (.leaf (ps.take mid)) (ps.getD mid 0) (.leaf (ps.drop mid))
  | .branch s1 c1, .branch s2 c2 =>
    let cs := c1 ++ c2
    let ss := s1 ++ sep :: s2
    if cs.length ≤ c.B then .one (.branch ss cs)
    else
      let mid := cs.length / 2
      let ls := ss.take mid
      .two (.branch ls.dropLast (cs.take mid)) (ls.getLast?.getD 0) (.branch (ss.drop mid) (cs.drop mid))
  | _, _ => .one a   -- `unreachable!` (node.rs:334): siblings are the same kind

/-- `Branch::rebalance` (`node.rs:114`) on the branch's `(seps, children)`. -/
def rebalance (c : Cfg) (seps : List E) (cs : List Node) (ci : Nat) : List E × List Node :=
  if cs.length < 2 then (seps, cs)                               -- line 122
  else
    let li := if ci + 1 < cs.length then ci else ci - 1          -- line 126
    let ri := if ci + 1 < cs.length then ci + 1 else ci
    match combine c (cs.getD li default) (seps.getD li 0) (cs.getD ri default) with
    | .one m => (delAt seps li, delAt (setAt cs li m) ri)        -- lines 139-141
    | .two l s r => (setAt seps li s, setAt (setAt cs li l) ri r) -- lines 145-147

/-- `Node::remove_one` (`node.rs:363`): `none` if absent, else the new node and its underflow flag. -/
def removeOne (c : Cfg) : Nat → Node → E → Option (Node × Bool)
  | _, .leaf es, x =>
    match leafRemove c es x with
    | none => none
    | some (es', u) => some (.leaf es', u)
  | 0, .branch _ _, _ => none
  | h + 1, .branch seps cs, x =>
    let i := childIndex seps x
    match removeOne c h (cs.getD i default) x with
    | none => none
    | some (c', u) =>
      let cs1 := setAt cs i c'
      let r := if u then rebalance c seps cs1 i else (seps, cs1)   -- line 377
      some (.branch r.1 r.2, decide (r.2.length < c.B / 2))        -- line 380

/-- The root-collapse loop of `CowBTree::remove` (`mod.rs:210`). -/
def collapse : Nat → Node → Node × Nat
  | h + 1, .branch _ [ch] => collapse h ch
  | h, n => (n, h)

/-- `CowBTree::remove` (`mod.rs:203`). -/
def remove (c : Cfg) (h : Nat) (root : Node) (x : E) : Node × Nat :=
  match removeOne c h root x with
  | none => (root, h)
  | some (n, _) => collapse h n

/-- `CowBTree::is_empty` (`mod.rs:260`). -/
def isEmpty : Node → Bool
  | .leaf es => es.isEmpty
  | .branch _ _ => false

/-! ## Bulk paths (`node.rs:56`, `:90`, `:169`; `mod.rs:142`, `:185`) -/

/-- `pack_branches` (`node.rs:56`): chunks of `BRANCH_MAX`, except a remainder of `BRANCH_MAX + 1`
    is split `BRANCH_MAX - 1` + `2`. Separators are `Node::min` of every non-first child
    (`getD 0` stands in for the panic of `minOpt = none`, tracked separately). `h` is the children's height. -/
def packBranches (c : Cfg) (h : Nat) (rest : List Node) : List Node :=
  if hr : rest = [] then [] else
  let take := if rest.length = c.B + 1 then c.B - 1 else min rest.length c.B
  let chunk := rest.take take
  .branch ((chunk.drop 1).map (fun n => (minOpt h n).getD 0)) chunk :: packBranches c h (rest.drop take)
termination_by rest.length
decreasing_by
  have := c.hB
  have : rest.length ≠ 0 := by simp [hr]
  simp only [List.length_drop]
  split <;> omega

/-- `build_root` (`node.rs:90`): pack until one fragment remains (`fuel` bounds the loop; each round
    of `pack_branches` on `>= 2` fragments strictly shrinks the list, so `fuel = length` suffices).
    Returns the root and its height; the empty list becomes an empty leaf. -/
def buildRoot (c : Cfg) : Nat → Nat → List Node → Node × Nat
  | fuel + 1, h, frags@(_ :: _ :: _) => buildRoot c fuel (h + 1) (packBranches c h frags)
  | _, h, [n] => (n, h)
  | _, h, [] => (.leaf [], h)
  | 0, h, n :: _ => (n, h)   -- fuel exhausted: unreachable (see `buildRoot_fuel`)

/-- The routing sweep of `apply_batch` (`node.rs:194-212`): pair each child with the prefix of the
    remaining batch strictly below its upper separator (`TOP` for the last child). Whatever is left
    after the last child is **dropped** — exactly as in the Rust, where `cursor` simply stops. -/
def route : List E → List Node → List E → List (Node × List E)
  | _, [], _ => []
  | seps, ch :: cs, rest =>
    let upper := seps.head?.getD TOP                          -- lines 198-202
    let sl := rest.takeWhile (fun e => decide (e < upper))    -- lines 204-206
    (ch, sl) :: route seps.tail cs (rest.drop sl.length)

/-- `Node::apply_batch` (`node.rs:169`). Returns the fragments that replace this node (height `h`). -/
def applyBatch (c : Cfg) : Nat → Node → List E → List Node
  | _, .leaf es, batch => (mergeBatch c es batch).map .leaf
  | 0, n@(.branch _ _), _ => [n]
  | h + 1, .branch seps cs, batch =>
    let newChildren := (route seps cs batch).flatMap
      (fun p => if p.2 = [] then [p.1] else applyBatch c h p.1 p.2)   -- lines 208-212
    packBranches c h newChildren                                      -- line 214

/-- `CowBTree::insert_batch` (`mod.rs:185`). -/
def insertBatch (c : Cfg) (h : Nat) (root : Node) (batch : List E) : Node × Nat :=
  if batch = [] then (root, h)
  else
    let frags := applyBatch c h root batch
    buildRoot c frags.length h frags

/-- `CowBTree::from_sorted` (`mod.rs:142`). -/
def fromSorted (c : Cfg) (pairs : List E) : Node × Nat :=
  if pairs = [] then (.leaf [], 0)
  else
    let leaves := (chunks c.L (by have := c.hL; omega) pairs).map .leaf
    buildRoot c leaves.length 0 leaves

/-! ## The range cursor (`cursor.rs`) -/

/-- A cursor frame: the height of the children and the siblings not yet descended. Rust keeps
    `(Arc<Branch>, next)` and tests `next < children.len()`; the observable content is exactly
    `children.drop next`, which is what we store. -/
abbrev Frame := Nat × List Node

/-- `RangeIter` (`cursor.rs:10`). `leaf = none` is exhaustion. -/
structure Cursor where
  stack : List Frame
  leaf : Option (List E)
  whole : Bool
  pos : Nat
  hi : Nat

/-- `set_leaf`'s `whole` (`cursor.rs:74`). -/
def wholeOf (es : List E) (hi : Nat) : Bool :=
  match es.getLast? with
  | none => true
  | some e => decide (keyOf e ≤ hi)

/-- `RangeIter::new` descent (`cursor.rs:45-60`). `lo` is `(lo_key, 0)`. -/
def seek (lo : E) (loKey : Nat) : Nat → Node → List Frame → List Frame × List E × Nat
  | _, .leaf es, stk => (stk, es, lowerBound es loKey)
  | 0, .branch _ _, stk => (stk, [], 0)
  | h + 1, .branch seps cs, stk =>
    let i := childIndex seps lo
    seek lo loKey h (cs.getD i default) ((h, cs.drop (i + 1)) :: stk)

/-- `RangeIter::new` (`cursor.rs:28`). -/
def Cursor.new (h : Nat) (root : Node) (lo hi : Nat) : Cursor :=
  let (stk, es, pos) := seek (enc lo 0) lo h root []
  { stack := stk, leaf := some es, whole := wholeOf es hi, pos := pos, hi := hi }

/-- `descend_left` (`cursor.rs:85`). -/
def descendLeft : Nat → Node → List Frame → List Frame × List E
  | _, .leaf es, stk => (stk, es)
  | 0, .branch _ _, stk => (stk, [])
  | h + 1, .branch _ cs, stk =>
    match cs with
    | [] => (stk, [])
    | ch :: rest => descendLeft h ch ((h, rest) :: stk)

/-- Node count, the termination measure of the cursor loop. -/
def size : Nat → Node → Nat
  | _, .leaf _ => 1
  | 0, .branch _ _ => 1
  | h + 1, .branch _ cs => 1 + (cs.map (size h)).sum

def frameWeight (f : Frame) : Nat := (f.2.map (size f.1)).sum
def stackWeight (s : List Frame) : Nat := (s.map frameWeight).sum

/-- `advance_leaf` (`cursor.rs:105`): pop exhausted frames; descend the next sibling. -/
def advanceLeaf : List Frame → Option (List Frame × List E)
  | [] => none
  | (_, []) :: stk => advanceLeaf stk
  | (h, ch :: rest) :: stk => some (descendLeft h ch ((h, rest) :: stk))

theorem descendLeft_weight (h : Nat) (n : Node) (stk : List Frame) :
    stackWeight (descendLeft h n stk).1 < stackWeight stk + size h n := by
  induction h generalizing n stk with
  | zero => cases n <;> simp [descendLeft, size]
  | succ h ih =>
    cases n with
    | leaf es => simp [descendLeft, size]
    | branch seps cs =>
      cases cs with
      | nil => simp [descendLeft, size]
      | cons ch rest =>
        simp only [descendLeft]
        have := ih ch ((h, rest) :: stk)
        simp [stackWeight, frameWeight, size] at this ⊢
        omega

theorem advanceLeaf_weight (stk stk' : List Frame) (es : List E) :
    advanceLeaf stk = some (stk', es) → stackWeight stk' < stackWeight stk := by
  induction stk with
  | nil => simp [advanceLeaf]
  | cons f stk ih =>
    obtain ⟨h, cs⟩ := f
    cases cs with
    | nil =>
      intro hs; simp only [advanceLeaf] at hs
      have := ih hs; simp [stackWeight, frameWeight] at this ⊢; omega
    | cons ch rest =>
      intro hs; simp only [advanceLeaf, Option.some.injEq] at hs
      have := descendLeft_weight h ch ((h, rest) :: stk)
      rw [hs] at this
      simp [stackWeight, frameWeight] at this ⊢; omega

/-- `Iterator::next` (`cursor.rs:124`): the yielded entry (its doc is `docOf`) and the new cursor. -/
def Cursor.next (cur : Cursor) : Option (E × Cursor) :=
  match hl : cur.leaf with
  | none => none                                                    -- line 126
  | some es =>
    if hp : cur.pos < es.length then
      let e := es[cur.pos]
      if !cur.whole && decide (keyOf e > cur.hi) then none          -- lines 137-139
      else some (e, { cur with pos := cur.pos + 1 })                -- lines 141-142
    else
      match ha : advanceLeaf cur.stack with                         -- line 144
      | none => none
      | some (stk', es') =>
        Cursor.next { stack := stk', leaf := some es', whole := wholeOf es' cur.hi, pos := 0, hi := cur.hi }
termination_by stackWeight cur.stack
decreasing_by exact advanceLeaf_weight _ _ _ ha

/-- Run the cursor for at most `n` yields (the consumer's `collect`, with a bound). -/
def Cursor.take : Nat → Cursor → List E
  | 0, _ => []
  | n + 1, cur => match cur.next with
    | none => []
    | some (e, cur') => e :: Cursor.take n cur'

/-- `CowBTree::range(lo, hi)` collected (at most `n` results). -/
def range (h : Nat) (root : Node) (lo hi n : Nat) : List E := (Cursor.new h root lo hi).take n

end CowBTree
