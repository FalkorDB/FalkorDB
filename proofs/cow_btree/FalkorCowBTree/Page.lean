import FalkorCowBTree.Insert
import FalkorCowBTree.Lookup

/-!
# One tree for every key type (#3738): the page trait

Since #3738 the tree layer (`node.rs`, `mod.rs`) is generic over its leaf page
(`page.rs`), so the `u64` tree (`CowBTree<L, B, D>`, the `Leaf` page) and the string tree (`StrTree`,
the `StrLeaf` page) share it. Each step the tree used to do on a leaf itself is now a `Page` method,
and the `u64` `Leaf`'s implementation of each is the former inline code:

| Rust (since #3738) | `<Leaf as Page>` impl | model (unchanged) |
|---|---|---|
| `Node::min` leaf arm, `leaf.min()` | `(key(0), doc(0))`, `leaf/mod.rs` | `minOpt` |
| `Node::combine` leaf arm (`node.rs`, `combine(..).map(Self::Leaf)`, `page.rs`) | AoS concat (`aos_combine`, `leaf/mod.rs`) or rebuild, `leaf/mod.rs` | `combine` |
| `Node::remove_batch` leaf arm | survivor walk, `leaf/mod.rs` | `removeBatch` / `subtract` |
| `Node::is_underfull` leaf arm | `count < LEAF_MAX / 2`, `leaf/mod.rs` | `isUnderfull` |
| `build_root`, `remove_batch` empty node | `Page::empty`, `Leaf::from_pairs(&[])`, `leaf/mod.rs` | `.leaf []` |
| `PageTree::from_sorted` | `Page::build`, `chunks(LEAF_MAX)`, `leaf/mod.rs` | `fromSorted` |
| `PageTree::heap_bytes` leaf arm | `raw().len()`, `leaf/mod.rs` | `heapBytes` |

So those definitions model the generic code at the `u64` page as they modelled the code before it.

One step changed shape: a page split may now hand back **several** right pages (a string longer than
a page's byte budget gets a page of its own), so `Node::insert_one` passes a *list* of right siblings
up, inserting them with a loop, and `CowBTree::insert` builds a root over the list. This file models
that code (`insertOneR`, `insertR`) and proves it equal to `insertOne` / `insert` for the `u64` page,
whose split always hands back exactly one right page. Every theorem about
`insertOne` / `insert` / `insertRet` therefore holds of the code as it is now (`insertR_spec`,
`insertOneR_spec`).
-/

namespace CowBTree

/-- The loop `for (k, x) in xs.enumerate() { v.insert(i + k, x) }`: each element
    inserted one past the one before it. -/
def insEach {α} (l : List α) (i : Nat) : List α → List α
  | [] => l
  | x :: xs => insEach (insAt l i x) (i + 1) xs

theorem insEach_single {α} (l : List α) (i : Nat) (x : α) : insEach l i [x] = insAt l i x := rfl

/-- `Node::insert_one` since #3738: a split hands up the list of right siblings,
    each with the separator before it (`Split { rights }`, `node.rs`). The `u64` page's split is
    `<Leaf as Page>::insert`: `Leaf::insert`'s `LeafInsert::Split` with its one
    right page wrapped in a list. -/
def insertOneR (c : Cfg) : Nat → Node → E → Node × Option (List (E × Node))
  | _, .leaf es, x =>
    match leafInsert c es x with
    | none => (.leaf es, none)
    | some (.fit ps) => (.leaf ps, none)
    | some (.split l s r) => (.leaf l, some [(s, .leaf r)])
  | 0, n@(.branch _ _), _ => (n, none)
  | h + 1, .branch seps cs, x =>
    let i := childIndex seps x
    match insertOneR c h (cs.getD i default) x with
    | (c', none) => (.branch seps (setAt cs i c'), none)
    | (c', some rs) =>
      let seps' := insEach seps i (rs.map Prod.fst)
      let cs' := insEach (setAt cs i c') (i + 1) (rs.map Prod.snd)
      if cs'.length ≤ c.B then (.branch seps' cs', none)
      else
        let mid := cs'.length / 2
        let ls := seps'.take mid
        (.branch ls.dropLast (cs'.take mid),
         some [(ls.getLast?.getD 0, .branch (seps'.drop mid) (cs'.drop mid))])

/-- `CowBTree::insert` since #3738: a root split stacks the old root and its right
    siblings under a new root: one branch when they fit `BRANCH_MAX`, else
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
