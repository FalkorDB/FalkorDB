/-
# Claim 3: the `'static` guard never outlives the cell it borrows

Merged PR #3436 (25babfe4f), folded from the retired `pr3436_review`. Cites: current main (= PR branch @ e30861ea3, `runtime.rs` re-cited) unless marked. Since 653f467a9 the
guard lives in `HeldRead { guard, _owner }` (`runtime.rs:115-118`): `new` clones `g` into
`_owner` and borrows through it (`:476-491`). The soundness argument therefore rests on
`HeldRead`'s own `Arc`, not on `Runtime`'s field order.

| here | there |
| --- | --- |
| `Alloc`, `H`           | an `Arc<AtomicRefCell<Graph>>` heap allocation: strong count, borrow word (`Cell.B`); `none` = freed |
| `Field.guard a`        | `HeldRead.guard: AtomicRef<'static, Graph>` into the cell at `a` (`runtime.rs:116`); dropping it is `AtomicBorrowRef::drop` — `fetch_sub(1)` on the cell's borrow word (atomic_refcell `lib.rs:273-283`), a write into the allocation |
| `Field.arc a`          | an `Arc` to the cell: `HeldRead._owner` (`:117`), `Runtime.g` (`:161`), or any other holder (`MvccGraph.graph`, the session's clone, the UDF thread-local, …); dropping it decrements the strong count and frees at 0 |
| `heldFields a`         | `HeldRead`'s drop glue: `guard` then `_owner` (declaration order) |
| `runtimeFields`        | `struct Runtime` (`:152-…`): `parameters`, `read_guard: Option<HeldRead>`, `g`, `write`, … |
| `dropFields`           | Rust drop glue for a struct without `Drop` impl: fields in declaration order (Reference, "Destructors"). Neither `Runtime` nor `HeldRead` has a `Drop` impl (grep) |
| `GUse`                 | every mention of the field `g` in `runtime.rs` (`g_uses_never_write`) |

**`g` is never reassigned** (`g_never_reassigned`). `g` is private (`runtime.rs:161`) and
`runtime.rs` declares no child module, so only `runtime.rs` can name it. Since e30861ea3
the field appears there exactly four times (`grep -nE "\.g\b|^\s+g,$"`): the struct
literal in `new` (`:495`, the initialisation) and three reads, all inside `&self` methods —
`graph` `:562` (`self.g.borrow()`), `graph_mut` `:578` (`self.g.borrow_mut()`, after
`assert!(self.write)`; a borrow of the *cell*, not of the field) and `graph_arc` `:588`
(`Arc::clone(&self.g)`). Everyone else gets a `GraphRef`, an `AtomicRefMut<Graph>` or an
owned `Arc` *clone* (`graph_arc_keeps_g`) — none of which can assign the field.
`runtime.rs` has no `&mut self` method at all, and the tree has no `&mut Runtime` /
`let mut runtime` / `let mut rt`. `HeldRead._owner` is written only by its constructor
(`:487-490`). Since 653f467a9 none of this is needed for soundness (`held_survives` holds
whatever happens to `g`); it is what makes `graph()`, `graph_mut()` and `graph_arc()` name
the same cell. The 653f467a9 enumeration (six uses incl. `graph_cell`) is kept as
`pre_e30861e_GUse`.
-/
import RuntimeCore.Cell

namespace RuntimeCore.Drop
open RuntimeCore.Cell

structure Alloc where
  strong : Nat
  b : B
  deriving DecidableEq, Repr

/-- The heap: `none` = freed (or never allocated). -/
abbrev H := Addr → Option Alloc

def H.set (h : H) (a : Addr) (v : Option Alloc) : H := fun x => if x = a then v else h x

theorem H.set_same (h : H) (a : Addr) (v : Option Alloc) : h.set a v a = v := by simp [H.set]

inductive Field where
  | guard (a : Addr)
  | arc (a : Addr)
  | other
  deriving DecidableEq, Repr

/-- `none` = undefined behaviour (a write into a freed allocation, or a double free). -/
def dropField (h : H) : Field → Option H
  | .guard a => match h a with
    | some al => some (h.set a (some { al with b := release al.b }))
    | none => none                                  -- use after free
  | .arc a => match h a with
    | some al => some (h.set a (if al.strong ≤ 1 then none else some { al with strong := al.strong - 1 }))
    | none => none
  | .other => some h

def dropFields (h : H) : List Field → Option H
  | [] => some h
  | f :: fs => match dropField h f with
    | some h' => dropFields h' fs
    | none => none

theorem dropFields_append (h : H) (fs gs : List Field) :
    dropFields h (fs ++ gs) = (dropFields h fs).bind (fun h' => dropFields h' gs) := by
  induction fs generalizing h with
  | nil => rfl
  | cons f fs ih =>
    simp only [List.cons_append, dropFields]
    cases dropField h f with
    | none => rfl
    | some h' => exact ih h'

/-! ### The cell's liveness, as a count of strong references -/

/-- `n` strong references to the cell at `a` are live (`n = 0`: freed). -/
def live (h : H) (a : Addr) (n : Nat) : Prop :=
  (n = 0 ∧ h a = none) ∨ (1 ≤ n ∧ ∃ al, h a = some al ∧ al.strong = n)

/-- Fields that touch only the cell at `a` through an `Arc` (other holders' drops) or not
at all. -/
def OnlyArcs (a : Addr) : List Field → Prop
  | [] => True
  | .arc b :: fs => b = a ∧ OnlyArcs a fs
  | .other :: fs => OnlyArcs a fs
  | .guard _ :: _ => False

def arcs : List Field → Nat
  | [] => 0
  | .arc _ :: fs => arcs fs + 1
  | _ :: fs => arcs fs

/-- Dropping `k ≤ n` of the `n` live `Arc`s is defined and leaves `n - k` live. -/
theorem drop_arcs (a : Addr) : ∀ (fs : List Field) (h : H) (n : Nat), OnlyArcs a fs →
    live h a n → arcs fs ≤ n → ∃ h', dropFields h fs = some h' ∧ live h' a (n - arcs fs)
  | [], h, n, _, hl, _ => ⟨h, rfl, by simpa [arcs] using hl⟩
  | .other :: fs, h, n, ho, hl, hk => by
    simp only [OnlyArcs] at ho
    simpa [dropFields, dropField, arcs] using drop_arcs a fs h n ho hl (by simpa [arcs] using hk)
  | .guard _ :: _, _, _, ho, _, _ => by simp [OnlyArcs] at ho
  | .arc b :: fs, h, n, ho, hl, hk => by
    simp only [OnlyArcs] at ho
    obtain ⟨rfl, ho⟩ := ho
    simp only [arcs] at hk
    rcases hl with ⟨rfl, _⟩ | ⟨hn, al, ha, hs⟩
    · omega
    · by_cases h1 : al.strong ≤ 1
      · have hn1 : n = 1 := by omega
        subst hn1
        have hk0 : arcs fs = 0 := by omega
        have hl' : live (h.set b none) b 0 := Or.inl ⟨rfl, H.set_same _ _ _⟩
        obtain ⟨h', hd, hl''⟩ := drop_arcs b fs (h.set b none) 0 ho hl' (by omega)
        refine ⟨h', ?_, ?_⟩
        · simp only [dropFields, dropField, ha, h1, ite_true]; exact hd
        · simp only [arcs]; rw [hk0] at hl''; simpa using hl''
      · have hl' : live (h.set b (some { al with strong := al.strong - 1 })) b (n - 1) :=
          Or.inr ⟨by omega, _, H.set_same _ _ _, by simp; omega⟩
        obtain ⟨h', hd, hl''⟩ := drop_arcs b fs _ (n - 1) ho hl' (by omega)
        refine ⟨h', ?_, ?_⟩
        · simp only [dropFields, dropField, ha, h1, ite_false]; exact hd
        · simp only [arcs]; have : n - 1 - arcs fs = n - (arcs fs + 1) := by omega
          rw [← this]; exact hl''

/-! ### Since 653f467a9: `HeldRead` owns its cell -/

/-- `HeldRead`'s drop glue: `guard` (declared first, `runtime.rs:116`), then `_owner`. -/
def heldFields (a : Addr) : List Field := [.guard a, .arc a]

/-- **Claim 3 (653f467a9).** The guard never dangles, whatever happens to every *other*
`Arc` of the cell: with `pre` and `post` any drops of the other holders (`Runtime.g`,
`MvccGraph.graph`, the session's clone, …; all of them, in any order, before or after
the `HeldRead` goes), `HeldRead`'s own drop is defined — `_owner` is the one strong
reference no one else can release. -/
theorem held_survives (a : Addr) (pre post : List Field) (hpre : OnlyArcs a pre)
    (hpost : OnlyArcs a post) (h : H) (hl : live h a (arcs pre + arcs post + 1)) :
    ∃ h', dropFields h (pre ++ heldFields a ++ post) = some h' := by
  obtain ⟨h₁, hd₁, hl₁⟩ := drop_arcs a pre h _ hpre hl (by omega)
  have hn : arcs pre + arcs post + 1 - arcs pre = arcs post + 1 := by omega
  rw [hn] at hl₁
  rcases hl₁ with ⟨h0, _⟩ | ⟨_, al, ha, hs⟩
  · omega
  · -- the guard writes into a live allocation, then `_owner` drops one reference
    let h₂ := h₁.set a (some { al with b := release al.b })
    have hl₂ : live h₂ a (arcs post + 1) := Or.inr ⟨by omega, _, H.set_same _ _ _, by simpa using hs⟩
    obtain ⟨h₃, hd₃, -⟩ := drop_arcs a [.arc a] h₂ _ ⟨rfl, trivial⟩ hl₂ (by simp [arcs])
    have hl₃ : live h₃ a (arcs post) := by
      obtain ⟨h₃', hd₃', hl₃'⟩ := drop_arcs a [.arc a] h₂ _ ⟨rfl, trivial⟩ hl₂ (by simp [arcs])
      rw [hd₃] at hd₃'; cases hd₃'; simpa [arcs] using hl₃'
    obtain ⟨h₄, hd₄, -⟩ := drop_arcs a post h₃ _ hpost hl₃ (Nat.le_refl _)
    refine ⟨h₄, ?_⟩
    rw [dropFields_append, dropFields_append, hd₁]
    simp only [Option.bind_some, heldFields]
    have : dropFields h₁ [.guard a, .arc a] = dropFields h₂ [.arc a] := by
      simp [dropFields, dropField, ha, h₂]
    rw [this, hd₃]; exact hd₄

/-- Even if `Runtime.g` were replaced by another graph (the reviewers' scenario: the old
`Arc` dropped while the guard still points into its cell) — modelled as `g`'s `Arc` of
`a` dropped first, every other holder gone too — the `HeldRead` drop is still defined. -/
theorem held_sound_even_if_g_replaced (a : Addr) (h : H) (al : Alloc) (ha : h a = some al)
    (hs : al.strong = 2) :
    ∃ h', dropFields h ([.arc a] ++ heldFields a) = some h' := by
  have := held_survives a [.arc a] [] ⟨rfl, trivial⟩ trivial h
    (Or.inr ⟨by simp [arcs], al, ha, by simp [arcs, hs]⟩)
  simpa using this

/-- What remains load-bearing is the order *inside* `HeldRead`: had `_owner` been declared
before `guard`, dropping the last reference would free the cell under the guard. -/
def heldSwapped (a : Addr) : List Field := [.arc a, .guard a]

theorem held_swapped_unsound :
    let h : H := fun x => if x = 0 then some { strong := 1, b := .shared 1 } else none
    dropFields h (heldSwapped 0) = none := by
  decide

/-- The PR's `Runtime` (653f467a9): `parameters`, `read_guard: Option<HeldRead>`, `g`, … -/
def runtimeFields (rt : Rt) : List Field :=
  [.other] ++ (match rt.guard with | some a => heldFields a | none => []) ++ [.arc rt.g, .other]

/-- `g` declared before `read_guard` — no longer matters. -/
def swappedFields (rt : Rt) : List Field :=
  [.other, .arc rt.g] ++ (match rt.guard with | some a => heldFields a | none => []) ++ [.other]

/-- A read runtime holds two strong references (`g` and `HeldRead._owner`, the clone made
in `new`, `:476`); a write runtime one. -/
def ownRefs (rt : Rt) : Nat := if rt.write then 1 else 2

theorem only_app (a : Addr) : ∀ (xs ys : List Field), OnlyArcs a xs → OnlyArcs a ys →
    OnlyArcs a (xs ++ ys)
  | [], _, _, hy => hy
  | .arc _ :: fs, ys, hx, hy => ⟨hx.1, only_app a fs ys hx.2 hy⟩
  | .other :: fs, ys, hx, hy => only_app a fs ys hx hy
  | .guard _ :: _, _, hx, _ => absurd hx (by simp [OnlyArcs])

theorem arcs_app : ∀ (xs ys : List Field), arcs (xs ++ ys) = arcs xs + arcs ys
  | [], ys => by simp [arcs]
  | .arc _ :: fs, ys => by simp only [List.cons_append, arcs, arcs_app fs ys]; omega
  | .other :: fs, ys => by simp only [List.cons_append, arcs, arcs_app fs ys]
  | .guard _ :: fs, ys => by simp only [List.cons_append, arcs, arcs_app fs ys]

/-- **Claim 3, `Runtime` (653f467a9).** Dropping a `Runtime` is defined in either field
order, interleaved with any drops of the `k` other holders of its cell — the field order
of `Runtime` is no longer part of the argument. -/
theorem drop_sound (rt : Rt) (hr : rt.ok) (pre post : List Field) (hpre : OnlyArcs rt.g pre)
    (hpost : OnlyArcs rt.g post) (h : H) (hl : live h rt.g (arcs pre + arcs post + ownRefs rt)) :
    (∃ h', dropFields h (pre ++ runtimeFields rt ++ post) = some h') ∧
    (∃ h', dropFields h (pre ++ swappedFields rt ++ post) = some h') := by
  unfold Rt.ok at hr
  cases hw : rt.write with
  | true =>
    simp [hw] at hr
    simp only [ownRefs, hw, ite_true] at hl
    have e1 : pre ++ runtimeFields rt ++ post = pre ++ ([.other, .arc rt.g, .other] ++ post) := by
      simp [runtimeFields, hr]
    have e2 : pre ++ swappedFields rt ++ post = pre ++ ([.other, .arc rt.g, .other] ++ post) := by
      simp [swappedFields, hr]
    have hall : OnlyArcs rt.g (pre ++ ([.other, .arc rt.g, .other] ++ post)) :=
      only_app _ _ _ hpre (only_app _ [.other, .arc rt.g, .other] post ⟨rfl, trivial⟩ hpost)
    have hcount : arcs (pre ++ ([.other, .arc rt.g, .other] ++ post)) = arcs pre + arcs post + 1 := by
      rw [arcs_app, arcs_app]; simp [arcs]; omega
    obtain ⟨h', hd, -⟩ := drop_arcs rt.g _ h _ hall hl (by omega)
    exact ⟨⟨h', e1 ▸ hd⟩, ⟨h', e2 ▸ hd⟩⟩
  | false =>
    simp [hw] at hr
    simp only [ownRefs, hw] at hl
    -- both orders: the `g` drop is just one more "other holder" around `heldFields`
    have e1 : pre ++ runtimeFields rt ++ post = (pre ++ [.other]) ++ heldFields rt.g ++ ([.arc rt.g, .other] ++ post) := by
      simp [runtimeFields, hr]
    have e2 : pre ++ swappedFields rt ++ post = (pre ++ [.other, .arc rt.g]) ++ heldFields rt.g ++ ([.other] ++ post) := by
      simp [swappedFields, hr]
    refine ⟨?_, ?_⟩
    · rw [e1]
      apply held_survives
      · exact only_app _ _ _ hpre trivial
      · exact only_app _ [.arc rt.g, .other] post ⟨rfl, trivial⟩ hpost
      · rw [show arcs (pre ++ [Field.other]) + arcs ([Field.arc rt.g, Field.other] ++ post) + 1 =
            arcs pre + arcs post + 2 by rw [arcs_app, arcs_app]; simp [arcs]; omega]
        exact hl
    · rw [e2]
      apply held_survives
      · exact only_app _ _ _ hpre ⟨rfl, trivial⟩
      · exact only_app _ [.other] post trivial hpost
      · rw [show arcs (pre ++ [Field.other, Field.arc rt.g]) + arcs ([Field.other] ++ post) + 1 =
            arcs pre + arcs post + 2 by rw [arcs_app, arcs_app]; simp [arcs]; omega]
        exact hl

/-- After the drop the cell's borrow word is back where it was before `new` (the guard gave
back exactly its one shared borrow), when someone else still holds the cell. -/
theorem drop_restores_borrow (rt : Rt) (hr : rt.ok) (hw : rt.write = false) (h : H) (n s : Nat)
    (ha : h rt.g = some { strong := s + 3, b := .shared (n + 1) }) :
    (dropFields h (runtimeFields rt)).bind (fun h' => h' rt.g) =
      some { strong := s + 1, b := .shared n } := by
  unfold Rt.ok at hr; simp [hw] at hr
  simp [runtimeFields, heldFields, hr, dropFields, dropField, ha, H.set, release]

/-- A `Runtime` value holds addresses into the heap, never into itself: moving it (a
bitwise copy to a new location) leaves every field — and so the drop sequence — unchanged.
Likewise moving `owner` into `HeldRead` after `owner.borrow()` (`:482-489`) moves the
`Arc` pointer, not the cell the guard points into. -/
theorem move_irrelevant (rt : Rt) (moved : Rt) (hm : moved = rt) :
    runtimeFields moved = runtimeFields rt := by rw [hm]

/-! ### `g` is never reassigned (e30861ea3) -/

/-- Every mention of the private field `g` in `runtime.rs` after initialisation. -/
inductive GUse where
  | borrowShared      -- `graph`, `:562`: `self.g.borrow()`
  | borrowMut         -- `graph_mut`, `:578`: `self.g.borrow_mut()` (after `assert!(self.write)`)
  | arcClone          -- `graph_arc`, `:588`: `Arc::clone(&self.g)`
  deriving DecidableEq, Repr

/-- Each use, as it acts on the `Runtime` record: all three are `&self` reads of the field
(a `borrow`/`borrow_mut` acts on the *cell*, modelled in `Cell`/`Protocol`; the clone on
the cell's strong count, `graphArc`), none on which cell `g` names. -/
def GUse.apply (_ : GUse) (rt : Rt) : Rt := rt

/-- The `g` and `read_guard` a runtime ends with are the ones `new` gave it, whatever
uses run. Together with privacy this is "`g` is never reassigned". -/
theorem g_never_reassigned (rt : Rt) (us : List GUse) :
    (us.foldl (fun r u => u.apply r) rt).g = rt.g ∧
    (us.foldl (fun r u => u.apply r) rt).guard = rt.guard := by
  induction us generalizing rt with
  | nil => exact ⟨rfl, rfl⟩
  | cons u us ih => exact ih (u.apply rt)

theorem g_uses_keep_ok (rt : Rt) (hr : rt.ok) (us : List GUse) :
    (us.foldl (fun r u => u.apply r) rt).ok := by
  induction us generalizing rt with
  | nil => exact hr
  | cons u us ih => exact ih (u.apply rt) hr

/-- `Runtime::graph_arc` (`runtime.rs:587-589`): a new strong reference to `g`'s cell,
handed to the caller (graph_core's published version `src/graph_core.rs:263`, the JS UDF
context `udf/js_context.rs:314`). The runtime record is untouched. -/
def graphArc (h : H) (rt : Rt) : Option (Addr × H × Rt) :=
  match h rt.g with
  | some al => some (rt.g, h.set rt.g (some { al with strong := al.strong + 1 }), rt)
  | none => none

/-- **`graph_arc()` cannot replace `g`.** It returns a clone of `g`'s `Arc` (same cell,
one more strong reference) and leaves the runtime's `g` and guard as they were; the
clone's eventual drop is one more "other holder" for `held_survives`. -/
theorem graph_arc_keeps_g (h : H) (rt : Rt) (al : Alloc) (ha : h rt.g = some al) :
    ∃ h', graphArc h rt = some (rt.g, h', rt) ∧ h' rt.g = some { al with strong := al.strong + 1 } ∧
      ∀ x, x ≠ rt.g → h' x = h x := by
  refine ⟨h.set rt.g (some { al with strong := al.strong + 1 }), by simp [graphArc, ha],
    H.set_same _ _ _, fun x hx => by simp [H.set, hx]⟩

/-! ### History: `g`'s uses at 653f467a9 (before e30861ea3) -/

/-- Every mention of the private field `g` in `runtime.rs` after initialisation. -/
inductive pre_e30861e_GUse where
  | borrowShared      -- `graph`, `:557`: `self.g.borrow()`
  | cellRef           -- `graph_cell`, `:566`: `&self.g`
  | resyncRef         -- `resync_published_indexes`, `:588`: `&self.g`
  | deferredRef       -- `commit_deferred_indexes`, `:593`: `&self.g`
  | borrowMutNode     -- `set_pending_node_attr`, `:1467`: `self.g.borrow_mut()`
  | borrowMutRel      -- `set_pending_relationship_attr`, `:1486`: `self.g.borrow_mut()`
  deriving DecidableEq, Repr

/-- Each use, as it acts on the `Runtime` record: all six are `&self` reads of the field
(a `borrow`/`borrow_mut` acts on the *cell*, modelled in `Cell`/`Protocol`, not on which
cell `g` names). -/
def pre_e30861e_GUse.apply (_ : pre_e30861e_GUse) (rt : Rt) : Rt := rt

/-- The `g` and `read_guard` a runtime ends with are the ones `new` gave it, whatever
uses run. Together with privacy (only `runtime.rs` can name `g`; outsiders get
`graph_cell(&self) -> &Arc`) this is "`g` is never reassigned". -/
theorem pre_e30861e_g_never_reassigned (rt : Rt) (us : List pre_e30861e_GUse) :
    (us.foldl (fun r u => u.apply r) rt).g = rt.g ∧
    (us.foldl (fun r u => u.apply r) rt).guard = rt.guard := by
  induction us generalizing rt with
  | nil => exact ⟨rfl, rfl⟩
  | cons u us ih => exact ih (u.apply rt)

theorem pre_e30861e_g_uses_keep_ok (rt : Rt) (hr : rt.ok) (us : List pre_e30861e_GUse) :
    (us.foldl (fun r u => u.apply r) rt).ok := by
  induction us generalizing rt with
  | nil => exact hr
  | cons u us ih => exact ih (u.apply rt) hr

/-! ### History: d7d166e82, before 653f467a9 (`read_guard: Option<AtomicRef>`, `pub g`) -/

end RuntimeCore.Drop
