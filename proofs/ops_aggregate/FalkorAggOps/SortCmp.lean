/-
Sort comparators and key classification (graph/src/runtime/ops/sort.rs).

| here | there |
| --- | --- |
| `cmpKey`            | `OrderedKey::cmp_key` (sort.rs:76) |
| `okEq`/`okPartialCmp`/`okCmp` | `PartialEq`/`PartialOrd`/`Ord for OrderedKey` (sort.rs:90, :99, :107) |
| `rowContent`        | `compare_row_content` (sort.rs:125) |
| `heapCmp`           | `Ord for HeapEntry` (sort.rs:173); `heEq` / `hePartialCmp` (:156, :165) |
| `classifySortKey`   | `classify_sort_key` (sort.rs:203), `classify_numeric(_, false, Pure)` (batch.rs:191) |
| `tiebreak`, `tbCols` | `tiebreak` (sort.rs:237) |

The value comparison is the parameter `cv : V → V → Ordering`; on main (#2912 b033aecc6)
it is `Value::sort_cmp`. Its laws (`Lawful`: reflexive, antisymmetric) are hypotheses
here and are discharged for `sort_cmp` in `Discharge.lean` (`lawful_main`, from
proofs/value_order `Main.scMain_total`). Before #2912 `cv` was `compare_value(..).0`,
for which `Lawful` is false (known #2891; value_order `pre2912_sortCmp_not_antisymmetric`).
-/
namespace FalkorAggOps.SortCmp

variable {V : Type}

/-- Antisymmetry / reflexivity of a comparator. -/
structure Lawful (c : V → V → Ordering) : Prop where
  swap : ∀ a b, c b a = (c a b).swap
  refl : ∀ a, c a a = .eq

structure OKey (V : Type) where
  value : V
  desc : Bool

/-- sort.rs:76 -/
def cmpKey (cv : V → V → Ordering) (a b : OKey V) : Ordering :=
  let o := cv a.value b.value
  if a.desc then o.swap else o

/-- sort.rs:90 / :99 / :107 -/
def okEq (cv : V → V → Ordering) (a b : OKey V) : Bool := cmpKey cv a b == .eq
def okCmp (cv : V → V → Ordering) (a b : OKey V) : Ordering := cmpKey cv a b
def okPartialCmp (cv : V → V → Ordering) (a b : OKey V) : Option Ordering := some (okCmp cv a b)

theorem swap_swap (o : Ordering) : o.swap.swap = o := by cases o <;> rfl

/-- ASC is `compare_value`, DESC is its reverse; `Ord`, `PartialOrd`, `PartialEq` all agree. -/
theorem cmpKey_spec (cv : V → V → Ordering) (a b : OKey V) :
    cmpKey cv a b = (if a.desc then (cv a.value b.value).swap else cv a.value b.value) ∧
    okCmp cv a b = cmpKey cv a b ∧ okPartialCmp cv a b = some (cmpKey cv a b) ∧
    (okEq cv a b = true ↔ cmpKey cv a b = .eq) := by
  refine ⟨rfl, rfl, rfl, ?_⟩
  unfold okEq; cases cmpKey cv a b <;> simp

/-- With a shared direction (one ORDER BY column), `cmp_key` inherits the laws. -/
theorem cmpKey_lawful (cv : V → V → Ordering) (h : Lawful cv) (d : Bool) :
    Lawful (fun x y => cmpKey cv ⟨x, d⟩ ⟨y, d⟩) := by
  constructor
  · intro a b; cases d <;> simp [cmpKey, h.swap a b]
  · intro a; cases d <;> simp [cmpKey, h.refl a, Ordering.swap]

/-! ## Lexicographic slice compare (`[T]::cmp`) -/

def lexCmp (c : V → V → Ordering) : List V → List V → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | a :: as, b :: bs => (c a b).then (lexCmp c as bs)

theorem then_swap (o p : Ordering) : (o.then p).swap = o.swap.then p.swap := by
  cases o <;> cases p <;> rfl

theorem lexCmp_swap (c : V → V → Ordering) (h : ∀ a b, c b a = (c a b).swap) :
    ∀ xs ys, lexCmp c ys xs = (lexCmp c xs ys).swap
  | [], [] => rfl
  | [], _ :: _ => rfl
  | _ :: _, [] => rfl
  | a :: as, b :: bs => by simp [lexCmp, h a b, lexCmp_swap c h as bs, then_swap]

theorem lexCmp_refl (c : V → V → Ordering) (h : ∀ a, c a a = .eq) :
    ∀ xs, lexCmp c xs xs = .eq
  | [] => rfl
  | a :: as => by simp [lexCmp, h a, lexCmp_refl c h as, Ordering.then]

/-! ## `compare_row_content` (sort.rs:125) -/

/-- A row = its slot list (`get_by_id` = `rs[i]?`); `nul` = `Value::Null`. -/
def slotCmp (cv : V → V → Ordering) (nul : V) : Option V → Option V → Ordering
  | some a, some b => cv a b
  | some a, none => cv a nul
  | none, some b => cv nul b
  | none, none => .eq

def rcLoop (cv : V → V → Ordering) (nul : V) (a b : List V) : List Nat → Ordering
  | [] => .eq
  | id :: ids =>
    let o := slotCmp cv nul a[id]? b[id]?
    if o != .eq then o else rcLoop cv nul a b ids

/-- sort.rs:125-143: `for id in 0..max(len a, len b)`, first non-Equal wins. -/
def rowContent (cv : V → V → Ordering) (nul : V) (a b : List V) : Ordering :=
  rcLoop cv nul a b (List.range (max a.length b.length))

/-- The loop is a lexicographic compare of the slot pairs over the index range. -/
theorem rcLoop_eq_fold (cv : V → V → Ordering) (nul : V) (a b : List V) :
    ∀ ids, rcLoop cv nul a b ids =
      ids.foldr (fun id acc => (slotCmp cv nul a[id]? b[id]?).then acc) .eq
  | [] => rfl
  | id :: ids => by
    simp only [rcLoop, List.foldr_cons, rcLoop_eq_fold cv nul a b ids]
    cases slotCmp cv nul a[id]? b[id]? <;> rfl

theorem slotCmp_swap (cv : V → V → Ordering) (h : ∀ a b, cv b a = (cv a b).swap) (nul : V) :
    ∀ x y, slotCmp cv nul y x = (slotCmp cv nul x y).swap
  | some a, some b => h a b
  | some a, none => h a nul
  | none, some b => h nul b
  | none, none => rfl

/-- Antisymmetric: swapping the rows reverses the answer. -/
theorem rowContent_swap (cv : V → V → Ordering) (h : ∀ a b, cv b a = (cv a b).swap)
    (nul : V) (a b : List V) : rowContent cv nul b a = (rowContent cv nul a b).swap := by
  unfold rowContent
  rw [Nat.max_comm b.length a.length, rcLoop_eq_fold, rcLoop_eq_fold]
  generalize List.range (max a.length b.length) = ids
  induction ids with
  | nil => rfl
  | cons id ids ih =>
    simp only [List.foldr_cons]
    rw [ih, slotCmp_swap cv h nul a[id]? b[id]?, then_swap]

/-- Reflexive: a row compares Equal to itself. -/
theorem rowContent_refl (cv : V → V → Ordering) (h : ∀ a, cv a a = .eq) (nul : V) (a : List V) :
    rowContent cv nul a a = .eq := by
  unfold rowContent
  rw [rcLoop_eq_fold]
  induction (List.range (max a.length a.length)) with
  | nil => rfl
  | cons id ids ih =>
    simp only [List.foldr_cons, ih]
    cases a[id]? <;> simp [slotCmp, h, Ordering.then]

/-! ## `HeapEntry` order (sort.rs:156-197) -/

structure HEntry (V : Type) where
  keys : List (OKey V)
  seq : Nat
  row : List V

/-- sort.rs:173 — keys lexicographically, then row content, then `seq`. -/
def heapCmp (cv : V → V → Ordering) (nul : V) (x y : HEntry V) : Ordering :=
  ((lexCmp (cmpKey cv) x.keys y.keys).then (rowContent cv nul x.row y.row)).then
    (compare x.seq y.seq)
def heEq (cv : V → V → Ordering) (nul : V) (x y : HEntry V) : Bool := heapCmp cv nul x y == .eq
def hePartialCmp (cv : V → V → Ordering) (nul : V) (x y : HEntry V) : Option Ordering :=
  some (heapCmp cv nul x y)

theorem then_eq_iff (o p : Ordering) : o.then p = .eq ↔ o = .eq ∧ p = .eq := by
  cases o <;> cases p <;> simp [Ordering.then]

/-- Two heap entries with different arrival `seq` never compare Equal: the heap order is
strict, so the surviving top-k set and its order are determined (no tie left to the heap). -/
theorem heapCmp_seq_strict (cv : V → V → Ordering) (nul : V) (x y : HEntry V) (h : x.seq ≠ y.seq) :
    heapCmp cv nul x y ≠ .eq ∧ heEq cv nul x y = false := by
  have hs : compare x.seq y.seq ≠ .eq := by
    intro he; exact h (Nat.compare_eq_eq.mp he)
  have : heapCmp cv nul x y ≠ .eq := by
    unfold heapCmp; intro hh; rw [then_eq_iff] at hh; exact hs hh.2
  refine ⟨this, ?_⟩
  unfold heEq; cases hc : heapCmp cv nul x y <;> simp_all

/-- Same direction vector on both entries ⇒ the heap order is antisymmetric. -/
theorem heapCmp_swap (cv : V → V → Ordering) (h : Lawful cv) (nul : V) (x y : HEntry V)
    (hd : x.keys.map (·.desc) = y.keys.map (·.desc)) :
    heapCmp cv nul y x = (heapCmp cv nul x y).swap := by
  unfold heapCmp
  have hk : lexCmp (cmpKey cv) y.keys x.keys = (lexCmp (cmpKey cv) x.keys y.keys).swap := by
    generalize x.keys = xs at hd; generalize y.keys = ys at hd
    induction xs generalizing ys with
    | nil => cases ys <;> simp_all [lexCmp]
    | cons a as ih =>
      cases ys with
      | nil => simp at hd
      | cons b bs =>
        simp only [List.map_cons, List.cons.injEq] at hd
        have hab : cmpKey cv b a = (cmpKey cv a b).swap := by
          cases a with | mk av ad => cases b with | mk bv bd =>
            obtain ⟨h1, _⟩ := hd
            simp only at h1; subst h1
            exact (cmpKey_lawful cv h ad).swap av bv
        simp [lexCmp, hab, ih bs hd.2, then_swap]
  rw [hk, rowContent_swap cv h.swap, then_swap, then_swap]
  rw [Std.OrientedOrd.eq_swap (a := y.seq) (b := x.seq)]

theorem heapCmp_refl (cv : V → V → Ordering) (h : Lawful cv) (nul : V) (x : HEntry V) :
    heapCmp cv nul x x = .eq := by
  have hk : lexCmp (cmpKey cv) x.keys x.keys = .eq := by
    generalize x.keys = xs
    induction xs with
    | nil => rfl
    | cons a as ih =>
      have : cmpKey cv a a = .eq := (cmpKey_lawful cv h a.desc).refl a.value
      simp [lexCmp, this, ih, Ordering.then]
  simp [heapCmp, hk, rowContent_refl cv h.refl, Ordering.then]

theorem heEq_partial (cv : V → V → Ordering) (nul : V) (x y : HEntry V) :
    hePartialCmp cv nul x y = some (heapCmp cv nul x y) ∧
    (heEq cv nul x y = true ↔ heapCmp cv nul x y = .eq) := by
  refine ⟨rfl, ?_⟩; unfold heEq; cases heapCmp cv nul x y <;> simp

/-! ## `tiebreak` (sort.rs:237, out of line since 4a4f2a72f) -/

/-- The content loop: `for id in 0..num_columns { combined.compare_rows_at(id, a, b) }`,
first non-Equal wins, else arrival order `a.cmp(&b)`. -/
def tbCols (cols : List (Nat → Nat → Ordering)) (a b : Nat) : Ordering :=
  match cols with
  | c :: cs => if c a b != .eq then c a b else tbCols cs a b
  | [] => compare a b

/-- sort.rs:237-258: the remaining keys (`trees.iter().enumerate().skip(1)`, each
`typed_keys[k].compare_at(a, b)`, reversed when `desc`), then `tbCols`. -/
def tiebreak (keys : List ((Nat → Nat → Ordering) × Bool)) (cols : List (Nat → Nat → Ordering))
    (a b : Nat) : Ordering :=
  match keys with
  | (c, d) :: ks => if c a b != .eq then (if d then (c a b).swap else c a b) else tiebreak ks cols a b
  | [] => tbCols cols a b

theorem tbCols_strict (cols : List (Nat → Nat → Ordering)) (a b : Nat) (h : a ≠ b) :
    tbCols cols a b ≠ .eq := by
  induction cols with
  | nil => simp only [tbCols]; intro he; exact h (Nat.compare_eq_eq.mp he)
  | cons c cs ih =>
    simp only [tbCols]; split
    · rename_i hc; simpa using hc
    · exact ih

/-- Two different rows never tie: the sort order is strict, so the result is
deterministic (no tie left to `sort_unstable_by`). -/
theorem tiebreak_strict (keys : List ((Nat → Nat → Ordering) × Bool))
    (cols : List (Nat → Nat → Ordering)) (a b : Nat) (h : a ≠ b) : tiebreak keys cols a b ≠ .eq := by
  induction keys with
  | nil => exact tbCols_strict cols a b h
  | cons k ks ih =>
    obtain ⟨c, d⟩ := k
    simp only [tiebreak]; split
    · rename_i hc
      have : c a b ≠ .eq := by simpa using hc
      cases d <;> simp only [Bool.false_eq_true, ite_false, ite_true] <;>
        (try exact this) <;> (intro he; apply this; cases hh : c a b <;> simp_all [Ordering.swap])
    · exact ih

theorem tbCols_swap (cols : List (Nat → Nat → Ordering))
    (hs : ∀ c ∈ cols, ∀ a b, c b a = (c a b).swap) (a b : Nat) :
    tbCols cols b a = (tbCols cols a b).swap := by
  induction cols with
  | nil => simp only [tbCols]; exact Std.OrientedOrd.eq_swap
  | cons c cs ih =>
    have hc := hs c (by simp) a b
    have ih' := ih (fun c' hm => hs c' (by simp [hm]))
    simp only [tbCols, hc]
    cases h : c a b <;> simp [Ordering.swap, ih']

/-- With antisymmetric key and column comparators (`compare_at`/`compare_rows_at` are
`sort_cmp`, antisymmetric on main: proofs/value_order `Main.compareAt_eq_scMain`,
`Main.scMain_total`; `Discharge.lawful_main`), `tiebreak` is antisymmetric. -/
theorem tiebreak_swap (keys : List ((Nat → Nat → Ordering) × Bool))
    (cols : List (Nat → Nat → Ordering))
    (hk : ∀ k ∈ keys, ∀ a b, k.1 b a = (k.1 a b).swap)
    (hs : ∀ c ∈ cols, ∀ a b, c b a = (c a b).swap) (a b : Nat) :
    tiebreak keys cols b a = (tiebreak keys cols a b).swap := by
  induction keys with
  | nil => exact tbCols_swap cols hs a b
  | cons k ks ih =>
    obtain ⟨c, d⟩ := k
    have hc : c b a = (c a b).swap := hk (c, d) (by simp) a b
    have ih' := ih (fun k' hm => hk k' (by simp [hm]))
    simp only [tiebreak, hc]
    cases d <;> cases h : c a b <;> simp [Ordering.swap, ih']

end FalkorAggOps.SortCmp
