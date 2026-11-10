/-
# Set-once globals: `graph/src/thread_id.rs` and `graph/src/storage/registry.rs`
(origin/main 3fec7d7c9)

A `OnceLock<T>` is `Option T`: `set` succeeds only on `None`; `get` reads it.
`AtomicBool` is a `Bool` cell (`Relaxed` is enough: single flag, no data published
through it).
-/
namespace SC.Once

def setOnce {α : Type} (c : Option α) (v : α) : Option α × Bool :=
  match c with | none => (some v, true) | some x => (some x, false)

/-! `thread_id.rs` -/

/-- `set_main_thread` (`thread_id.rs:13`): first call wins. -/
def setMainThread (c : Option Nat) (tid : Nat) : Option Nat := (setOnce c tid).1
/-- `is_main_thread` (`:23`): false before registration. -/
def isMainThread (c : Option Nat) (tid : Nat) : Bool := c.any (· == tid)
/-- `set_process_is_child` / `process_is_child` (`:32,39`). -/
def setChild (_old v : Bool) : Bool := v
def isChild (b : Bool) : Bool := b

theorem thread_id_spec (t1 t2 : Nat) (b v : Bool) :
    isMainThread (setMainThread none t1) t1 = true ∧
    setMainThread (setMainThread none t1) t2 = some t1 ∧
    isMainThread none t1 = false ∧
    (t1 ≠ t2 → isMainThread (setMainThread (setMainThread none t1) t2) t2 = false) ∧
    isChild (setChild b v) = v := by
  refine ⟨by simp [isMainThread, setMainThread, setOnce], rfl, rfl, fun h => ?_, rfl⟩
  simp [isMainThread, setMainThread, setOnce]; exact h

/-! `storage/registry.rs` -/

inductive Res (α : Type) | ok (a : α) | panic

/-- `register_index_backend` / `register_attr_backend` (`registry.rs:31,52`):
`assert!` panics on a second registration. -/
def register {α : Type} (c : Option α) (b : α) : Res (Option α) :=
  match setOnce c b with | (c', true) => .ok c' | (_, false) => .panic
/-- `index_backend` / `attr_backend` (`:43,62`): `expect` panics when unset. -/
def get {α : Type} (c : Option α) : Res α := match c with | some b => .ok b | none => .panic

theorem registry_spec {α : Type} (b b' : α) :
    (match register none b with | .ok c => get c = .ok b | .panic => False) ∧
    (match register (some b) b' with | .panic => True | .ok _ => False) ∧
    (match get (none : Option α) with | .panic => True | .ok _ => False) := by
  simp [register, setOnce, get]

end SC.Once
