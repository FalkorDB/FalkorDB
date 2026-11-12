import FalkorCowBTree.Insert
import FalkorCowBTree.Lookup

/-!
# One tree for every key type (#3742): the page trait

Since #3742 the tree layer (`node.rs`, `mod.rs`) is generic over its leaf page
(`page.rs`), so the `u64` tree (`CowBTree<L, B, D>`, the `Leaf` page) and the string tree (`StrTree`,
the `StrLeaf` page) share it. Each step the tree used to do on a leaf itself is now a `Page` method,
and the `u64` `Leaf`'s implementation of each (`leaf/mod.rs:508-614`) is the former inline code:

| Rust (since #3742) | `<Leaf as Page>` impl | model (unchanged) |
|---|---|---|
| `Node::min` leaf arm (`node.rs:240`), `leaf.min()` | `(key(0), doc(0))`, `leaf/mod.rs:528` | `minOpt` |
| `Node::combine` leaf arm (`node.rs:388-393`) | AoS concat (`aos_combine`, `leaf/mod.rs:633`) or rebuild, `leaf/mod.rs:588` | `combine` |
| `Node::remove_batch` leaf arm (`node.rs:464`) | survivor walk, `leaf/mod.rs:567-582` | `removeBatch` / `subtract` |
| `Node::is_underfull` leaf arm (`node.rs:524`) | `count < LEAF_MAX / 2`, `leaf/mod.rs:532` | `isUnderfull` |
| `build_root`, `remove_batch` empty node | `Page::empty`, `Leaf::from_pairs(&[])`, `leaf/mod.rs:513` | `.leaf []` |
| `PageTree::from_sorted` (`mod.rs:199`) | `Page::build`, `chunks(LEAF_MAX)`, `leaf/mod.rs:519` | `fromSorted` |
| `PageTree::heap_bytes` leaf arm (`mod.rs:320`) | `raw().len()`, `leaf/mod.rs:610` | `heapBytes` |

So those definitions model the generic code at the `u64` page as they modelled the code before it.

One step changed shape: a page split may now hand back **several** right pages (a string longer than
a page's byte budget gets a page of its own), so `Node::insert_one` passes a *list* of right siblings
up, inserting them with a loop, and `CowBTree::insert` builds a root over the list. This file models
that code (`insertOneR`, `insertR`) and proves it equal to `insertOne` / `insert` for the `u64` page,
whose split always hands back exactly one right page (`leaf/mod.rs:543-546`). Every theorem about
`insertOne` / `insert` / `insertRet` therefore holds of the code as it is now (`insertR_spec`,
`insertOneR_spec`).
-/

namespace CowBTree

/-- The loop `for (k, x) in xs.enumerate() { v.insert(i + k, x) }` (`node.rs:346-348`): each element
    inserted one past the one before it. -/
def insEach {α} (l : List α) (i : Nat) : List α → List α
  | [] => l
  | x :: xs => insEach (insAt l i x) (i + 1) xs

theorem insEach_single {α} (l : List α) (i : Nat) (x : α) : insEach l i [x] = insAt l i x := rfl

/-- `Node::insert_one` (`node.rs:312`) since #3742: a split hands up the list of right siblings,
    each with the separator before it (`Split { rights }`, `node.rs:32`). The `u64` page's split is
    `<Leaf as Page>::insert` (`leaf/mod.rs:536-549`): `Leaf::insert`'s `LeafInsert::Split` with its one
    right page wrapped in a list. -/
def insertOneR (c : Cfg) : Nat → Node → E → Node × Option (List (E × Node))
  | _, .leaf es, x =>
    match leafInsert c es x with
    | none => (.leaf es, none)
    | some (.fit ps) => (.leaf ps, none)
    | some (.split l s r) => (.leaf l, some [(s, .leaf r)])
  | 0, n@(.branch _ _), _ => (n, none)
  | h + 1, .branch seps cs, x =>
    let i := childIndex seps x                                      -- node.rs:340
    match insertOneR c h (cs.getD i default) x with                 -- node.rs:341
    | (c', none) => (.branch seps (setAt cs i c'), none)            -- node.rs:343
    | (c', some rs) =>
      let seps' := insEach seps i (rs.map Prod.fst)                 -- node.rs:346-347
      let cs' := insEach (setAt cs i c') (i + 1) (rs.map Prod.snd)  -- node.rs:348
      if cs'.length ≤ c.B then (.branch seps' cs', none)            -- node.rs:352
      else
        let mid := cs'.length / 2                                   -- node.rs:357
        let ls := seps'.take mid                                    -- node.rs:359
        (.branch ls.dropLast (cs'.take mid),                        -- node.rs:360
         some [(ls.getLast?.getD 0, .branch (seps'.drop mid) (cs'.drop mid))])

/-- `CowBTree::insert` (`mod.rs:220`) since #3742: a root split stacks the old root and its right
    siblings under a new root (`mod.rs:229-235`): one branch when they fit `BRANCH_MAX`, else
    `build_root` over them. -/
def insertR (c : Cfg) (h : Nat) (root : Node) (x : E) : Node × Nat :=
  match insertOneR c h root x with
  | (n, none) => (n, h)
  | (n, some rs) =>
    let children := n :: rs.map Prod.snd
    if children.length ≤ c.B then (.branch (rs.map Prod.fst) children, h + 1)
    else buildRoot c children.length h children

/-- The new `insert_one` is the old one, with the one right sibling of a split in a list. -/
theorem insertOneR_eq (c : Cfg) : ∀ (h : Nat) (n : Node) (x : E),
    insertOneR c h n x = ((insertOne c h n x).1, (insertOne c h n x).2.map (fun p => [p]))
  | _, .leaf es, x => by
    unfold insertOneR insertOne
    cases leafInsert c es x with
    | none => rfl
    | some r => cases r <;> rfl
  | 0, .branch _ _, _ => by simp [insertOneR, insertOne]
  | h + 1, .branch seps cs, x => by
    unfold insertOneR insertOne
    simp only [insertOneR_eq c h]
    cases insertOne c h (cs.getD (childIndex seps x) default) x with
    | mk c' o =>
      cases o with
      | none => rfl
      | some p =>
        obtain ⟨s, r⟩ := p
        simp [insEach_single]
        split <;> rfl

/-- The new `CowBTree::insert` is the old one: the `u64` page splits into one right page, so a root
    split stacks two children, which always fit (`3 ≤ BRANCH_MAX`, `Cfg.hB`). -/
theorem insertR_eq (c : Cfg) (h : Nat) (root : Node) (x : E) : insertR c h root x = insert c h root x := by
  unfold insertR insert
  rw [insertOneR_eq]
  cases insertOne c h root x with
  | mk n o =>
    cases o with
    | none => rfl
    | some p =>
      obtain ⟨s, r⟩ := p
      have hB := c.hB
      simp
      omega

/-- **`CowBTree::insert` as the code is now**: `insert_spec` for `insertR`. -/
theorem insertR_spec (c : Cfg) (h : Nat) (t : Node) (x : E) (hw : TreeWF c h t) (hx : x < HI) :
    TreeWF c (insertR c h t x).2 (insertR c h t x).1 ∧
    toList (insertR c h t x).2 (insertR c h t x).1 = sins x (toList h t) := by
  rw [insertR_eq]; exact insert_spec c h t x hw hx

/-- **`Node::insert_one` as the code is now**: a split hands up at most one right sibling at the
    `u64` page, and the node part is `insertOne`'s, so `insertOne_spec` and `insertFlag_spec` apply. -/
theorem insertOneR_spec (c : Cfg) (h : Nat) (n : Node) (x : E) :
    (insertOneR c h n x).1 = (insertOne c h n x).1 ∧
    ∀ rs, (insertOneR c h n x).2 = some rs → rs.length = 1 := by
  rw [insertOneR_eq]
  refine ⟨rfl, fun rs hrs => ?_⟩
  cases ho : (insertOne c h n x).2 with
  | none => simp [ho] at hrs
  | some p => simp [ho] at hrs; subst hrs; rfl

end CowBTree
