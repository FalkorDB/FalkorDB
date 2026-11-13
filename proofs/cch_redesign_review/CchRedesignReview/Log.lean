/-!
# The CCH write log (03 §2 "CCH log", §5 "The log")

`CchLog { head: Option<Arc<LogNode>>, pairs: u64 }` with
`LogNode { seq, kind: Pairs(Arc<[(u64,u64)]>) | Overflow, prev }` is a persistent list, newest node at the
head. It is modelled as a `List Node` (head first) plus the separate `pairs` counter, which lives on the
entry (`Model.lean`). A pair is the abstract id of `(min(u,v), max(u,v))`: only equality of pairs matters to
the log's rules.

`covered log s p`: the write of `p` at seq `s` is still in `log`, either in its own node (`seq = s`, `p` in its
pairs) or under an Overflow node at or after `s` (03 §5 "Bound and Overflow": the consumer of an Overflow
rescans everything, so it resolves every write with `seq ≤` its own).
-/
namespace CchReview

/-- An undirected node pair `(min(u,v), max(u,v))` (03 §5 "Writers record pairs"). Self-loops are dropped before
    a pair is logged (`maint.rs:131` @08c) and are not modelled. -/
abbrev Pair := Nat

/-- `LogNode::kind` (03 §2). -/
inductive NKind where
  | pairs (ps : List Pair)
  | overflow
deriving DecidableEq, Repr

/-- `LogNode` without `prev` (the list holds it): its commit's data version and its kind. -/
structure Node where
  seq : Nat
  kind : NKind
deriving DecidableEq, Repr

namespace Node
/-- The pairs a node holds (an Overflow holds none). -/
def ps : Node → List Pair
  | ⟨_, .pairs ps⟩ => ps
  | ⟨_, .overflow⟩ => []

def isOv : Node → Bool
  | ⟨_, .overflow⟩ => true
  | ⟨_, .pairs _⟩ => false

@[simp] theorem ps_pairs (s : Nat) (ps : List Pair) : (Node.mk s (.pairs ps)).ps = ps := rfl
@[simp] theorem ps_ov (s : Nat) : (Node.mk s .overflow).ps = [] := rfl
@[simp] theorem isOv_pairs (s : Nat) (ps : List Pair) : (Node.mk s (.pairs ps)).isOv = false := rfl
@[simp] theorem isOv_ov (s : Nat) : (Node.mk s .overflow).isOv = true := rfl

theorem ps_of_isOv {n : Node} (h : n.isOv = true) : n.ps = [] := by
  rcases n with ⟨s, k⟩; cases k <;> simp_all [isOv, ps]
end Node

/-- The pairs the log holds: 03 §2's `pairs` counter is this sum (`Inv.cnt`). -/
def sumPairs : List Node → Nat
  | [] => 0
  | n :: l => n.ps.length + sumPairs l

def allPairs : List Node → List Pair
  | [] => []
  | n :: l => n.ps ++ allPairs l

def hasOv : List Node → Bool
  | [] => false
  | n :: l => n.isOv || hasOv l

/-- The write of `p` at `s` is still in `log` (its own node, or an Overflow at or after it). -/
def covered (log : List Node) (s : Nat) (p : Pair) : Prop :=
  ∃ n ∈ log, (n.seq = s ∧ p ∈ n.ps) ∨ (n.isOv = true ∧ s ≤ n.seq)

/-- A writer's own nodes in its private copy: those with `seq = fork.version` (03 §5 "Writer rebase"), on top. -/
def own (v : Nat) (l : List Node) : List Node := l.takeWhile (fun n => n.seq == v)

/-- 03 §5 CatchUp: "the published copy's nodes of `seq > fork` relinked on top (walked from the head, stopping
    at the first `seq ≤ fork`)". -/
def relinked (F : Nat) (l : List Node) : List Node := l.takeWhile (fun n => decide (F < n.seq))

instance (l : List Node) (s : Nat) (p : Pair) : Decidable (covered l s p) := by
  unfold covered; exact inferInstance

/-! ## Lemmas -/

@[simp] theorem sumPairs_nil : sumPairs [] = 0 := rfl
@[simp] theorem sumPairs_cons (n : Node) (l : List Node) : sumPairs (n :: l) = n.ps.length + sumPairs l := rfl
@[simp] theorem allPairs_nil : allPairs [] = [] := rfl
@[simp] theorem allPairs_cons (n : Node) (l : List Node) : allPairs (n :: l) = n.ps ++ allPairs l := rfl
@[simp] theorem hasOv_nil : hasOv [] = false := rfl
@[simp] theorem hasOv_cons (n : Node) (l : List Node) : hasOv (n :: l) = (n.isOv || hasOv l) := rfl

theorem sumPairs_append (a b : List Node) : sumPairs (a ++ b) = sumPairs a + sumPairs b := by
  induction a with
  | nil => simp
  | cons n l ih => simp [ih]; omega

theorem allPairs_append (a b : List Node) : allPairs (a ++ b) = allPairs a ++ allPairs b := by
  induction a with
  | nil => simp
  | cons n l ih => simp [ih]

theorem hasOv_append (a b : List Node) : hasOv (a ++ b) = (hasOv a || hasOv b) := by
  induction a with
  | nil => simp
  | cons n l ih => simp [ih, Bool.or_assoc]

theorem hasOv_iff (l : List Node) : hasOv l = true ↔ ∃ n ∈ l, n.isOv = true := by
  induction l with
  | nil => simp
  | cons n l ih => simp [ih]

theorem hasOv_false_iff (l : List Node) : hasOv l = false ↔ ∀ n ∈ l, n.isOv = false := by
  have := hasOv_iff l
  constructor
  · intro h n hn
    cases hv : n.isOv
    · rfl
    · have : hasOv l = true := (hasOv_iff l).2 ⟨n, hn, hv⟩
      simp_all
  · intro h
    cases hh : hasOv l
    · rfl
    · obtain ⟨n, hn, hv⟩ := (hasOv_iff l).1 hh; simp_all

theorem mem_allPairs (l : List Node) (p : Pair) : p ∈ allPairs l ↔ ∃ n ∈ l, p ∈ n.ps := by
  induction l with
  | nil => simp
  | cons n l ih => simp [ih]

theorem covered_mono {a b : List Node} (h : ∀ n ∈ a, n ∈ b) {s : Nat} {p : Pair} (hc : covered a s p) :
    covered b s p := by
  obtain ⟨n, hn, hh⟩ := hc; exact ⟨n, h n hn, hh⟩

theorem covered_append_left {a : List Node} (b : List Node) {s p} (h : covered a s p) : covered (a ++ b) s p :=
  covered_mono (fun _ hn => List.mem_append_left b hn) h

theorem covered_append_right (a : List Node) {b : List Node} {s p} (h : covered b s p) : covered (a ++ b) s p :=
  covered_mono (fun _ hn => List.mem_append_right a hn) h

theorem covered_cons (n : Node) {l : List Node} {s p} (h : covered l s p) : covered (n :: l) s p :=
  covered_mono (fun _ hm => List.mem_cons_of_mem n hm) h

/-- An Overflow at `v` covers every write at or before `v`. -/
theorem covered_ov {l : List Node} {v s : Nat} (p : Pair) (h : ⟨v, .overflow⟩ ∈ l) (hs : s ≤ v) :
    covered l s p := ⟨_, h, Or.inr ⟨rfl, hs⟩⟩

/-- A covered pair in a log without Overflow nodes is in a node of exactly that seq. -/
theorem covered_noOv {l : List Node} {s p} (h : covered l s p) (hov : hasOv l = false) :
    ∃ n ∈ l, n.seq = s ∧ p ∈ n.ps := by
  obtain ⟨n, hn, h1 | ⟨h2, _⟩⟩ := h
  · exact ⟨n, hn, h1⟩
  · have := (hasOv_false_iff l).1 hov n hn; simp_all

/-- `takeWhile` stops exactly at the seam when the head part satisfies the test and the rest's head does not. -/
theorem takeWhile_seam {P : Node → Bool} (a b : List Node) (ha : ∀ n ∈ a, P n = true)
    (hb : ∀ n ∈ b, P n = false) : (a ++ b).takeWhile P = a := by
  induction a with
  | nil => cases b with
    | nil => rfl
    | cons m b => simp [hb m (List.mem_cons_self ..)]
  | cons n a ih =>
    simp only [List.cons_append, List.takeWhile, ha n (List.mem_cons_self ..)]
    rw [ih (fun m hm => ha m (List.mem_cons_of_mem n hm))]

theorem takeWhile_all {P : Node → Bool} (a : List Node) (ha : ∀ n ∈ a, P n = true) : a.takeWhile P = a := by
  have := takeWhile_seam (P := P) a [] ha (by simp); simpa using this

theorem own_seam (v : Nat) (a b : List Node) (ha : ∀ n ∈ a, n.seq = v) (hb : ∀ n ∈ b, n.seq < v) :
    own v (a ++ b) = a := by
  unfold own
  apply takeWhile_seam
  · intro n hn; simp [ha n hn]
  · intro n hn; have := hb n hn; simp; omega

theorem own_all (v : Nat) (a : List Node) (ha : ∀ n ∈ a, n.seq = v) : own v a = a := by
  have := own_seam v a [] ha (by simp); simpa using this

theorem relinked_seam (F : Nat) (a b : List Node) (ha : ∀ n ∈ a, F < n.seq) (hb : ∀ n ∈ b, n.seq ≤ F) :
    relinked F (a ++ b) = a := by
  unfold relinked
  apply takeWhile_seam
  · intro n hn; simp [ha n hn]
  · intro n hn; have := hb n hn; simp; omega

theorem relinked_all (F : Nat) (a : List Node) (ha : ∀ n ∈ a, F < n.seq) : relinked F a = a := by
  have := relinked_seam F a [] ha (by simp); simpa using this

theorem mem_of_mem_takeWhile {P : Node → Bool} {l : List Node} {n : Node} (h : n ∈ l.takeWhile P) : n ∈ l :=
  (List.takeWhile_sublist P).subset h

theorem sumPairs_le_append_left (a b : List Node) : sumPairs a ≤ sumPairs (a ++ b) := by
  rw [sumPairs_append]; omega

end CchReview
