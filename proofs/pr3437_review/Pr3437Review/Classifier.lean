/-!
# PR #3437: which bodies may run columnarly (`collect_body_vars`, `ListLoop::vector_plan`)

| Lean | Rust |
| --- | --- |
| `BK`, `BT` | the `ExprIR` shapes `collect_body_vars` distinguishes, with children |
| `collect`, `collectAll` | `collect_body_vars` (eval.rs:380-405); `collectAll` is `node.children().all(..)` (:403-404), short-circuiting |
| `bad` | "reads the whole row (`NestedPlan`, `Pattern`, `PatternComprehension`), uses `DISTINCT`, aggregates or writes" — anywhere in the tree |
| `vectorPlan` | `ListLoop::vector_plan` (eval.rs:344-371): `predicate.iter().chain(projection.iter()).all(..)` then every free read resolved through `env.value_at` |

**`collect_rejects_iff`**: `collect_body_vars` returns `false` exactly when the tree contains
one of the excluded shapes, at any depth. **`vectorPlan_some`**: a plan exists only for
bodies with none of them, and it broadcasts every variable the bodies read that they do
not bind themselves, with the environment's value (`vectorPlan_broadcasts`).
-/

namespace Pr3437.Classifier

/-- Node kinds as `collect_body_vars` sees them. `func write agg`: a `FuncInvocation`
with `func.write` / `FnType::Aggregation`. -/
inductive BK where
  | var (id : Nat)
  | listComp (id : Nat)
  | quant (id : Nat)
  | reduce (acc it : Nat)
  | nestedPlan | pattern | patternComp | distinct
  | func (write agg : Bool)
  | other

inductive BT where
  | mk (k : BK) (cs : List BT)

/-- The `match node.data()` of eval.rs:385-401: `none` = `return false`. -/
def pre : BK → List Nat → List Nat → Option (List Nat × List Nat)
  | .var id, r, b => some (r ++ [id], b)
  | .listComp id, r, b | .quant id, r, b => some (r, b ++ [id])
  | .reduce a i, r, b => some (r, b ++ [a, i])
  | .nestedPlan, _, _ | .pattern, _, _ | .patternComp, _, _ | .distinct, _, _ => none
  | .func w a, r, b => if w || a then none else some (r, b)
  | .other, r, b => some (r, b)

mutual
def collect : BT → List Nat → List Nat → Bool × List Nat × List Nat
  | .mk k cs, r, b =>
    match pre k r b with
    | none => (false, r, b)
    | some (r', b') => collectAll cs r' b'
def collectAll : List BT → List Nat → List Nat → Bool × List Nat × List Nat
  | [], r, b => (true, r, b)
  | c :: cs, r, b =>
    match collect c r b with
    | (true, r', b') => collectAll cs r' b'
    | (false, r', b') => (false, r', b')
end

def badKind : BK → Bool
  | .nestedPlan | .pattern | .patternComp | .distinct => true
  | .func w a => w || a
  | _ => false

mutual
def bad : BT → Bool
  | .mk k cs => badKind k || badAll cs
def badAll : List BT → Bool
  | [] => false
  | c :: cs => bad c || badAll cs
end

theorem pre_none_iff (k : BK) (r b : List Nat) : pre k r b = none ↔ badKind k = true := by
  cases k with
  | func w a => cases w <;> cases a <;> simp [pre, badKind]
  | _ => simp [pre, badKind]

theorem pre_extends (k : BK) (r b : List Nat) (rb : List Nat × List Nat) (h : pre k r b = some rb) :
    (∃ r₀, rb.1 = r ++ r₀) ∧ (∃ b₀, rb.2 = b ++ b₀) := by
  cases k with
  | func w a =>
    cases w <;> cases a <;> simp [pre] at h <;> subst h <;> exact ⟨⟨[], by simp⟩, ⟨[], by simp⟩⟩
  | var id => simp only [pre, Option.some.injEq] at h; subst h; exact ⟨⟨[id], rfl⟩, ⟨[], by simp⟩⟩
  | listComp id => simp only [pre, Option.some.injEq] at h; subst h; exact ⟨⟨[], by simp⟩, ⟨[id], rfl⟩⟩
  | quant id => simp only [pre, Option.some.injEq] at h; subst h; exact ⟨⟨[], by simp⟩, ⟨[id], rfl⟩⟩
  | reduce a i => simp only [pre, Option.some.injEq] at h; subst h; exact ⟨⟨[], by simp⟩, ⟨[a, i], rfl⟩⟩
  | other => simp only [pre, Option.some.injEq] at h; subst h; exact ⟨⟨[], by simp⟩, ⟨[], by simp⟩⟩
  | _ => simp [pre] at h

mutual
theorem collect_rejects_iff : ∀ (t : BT) (r b : List Nat), (collect t r b).1 = !bad t
  | .mk k cs, r, b => by
    simp only [collect, bad]
    cases hp : pre k r b with
    | none => simp [(pre_none_iff k r b).1 hp]
    | some rb =>
      have : badKind k = false := by
        cases h : badKind k
        · rfl
        · exact absurd ((pre_none_iff k r b).2 h) (by simp [hp])
      simp only [this, Bool.false_or]
      exact collectAll_rejects_iff cs rb.1 rb.2
theorem collectAll_rejects_iff : ∀ (cs : List BT) (r b : List Nat), (collectAll cs r b).1 = !badAll cs
  | [], r, b => rfl
  | c :: cs, r, b => by
    simp only [collectAll, badAll]
    have hc := collect_rejects_iff c r b
    rcases h : collect c r b with ⟨_ | _, r', b'⟩
    · rw [h] at hc; simp only at hc ⊢; simp [hc.symm]
    · rw [h] at hc; simp only at hc ⊢
      have : bad c = false := by cases hb : bad c <;> simp_all
      simp only [this, Bool.false_or]; exact collectAll_rejects_iff cs r' b'
end

/-- **Claim 3.** A body containing a whole-row read, `DISTINCT`, an aggregate or a write —
at any depth — is rejected. -/
theorem collect_rejects (t : BT) (r b : List Nat) (h : bad t = true) : (collect t r b).1 = false := by
  rw [collect_rejects_iff, h]; rfl

-- `collect_body_vars` only appends to `reads` / `bound`.
mutual
theorem collect_extends : ∀ (t : BT) (r b : List Nat),
    (∃ r₁, (collect t r b).2.1 = r ++ r₁) ∧ (∃ b₁, (collect t r b).2.2 = b ++ b₁)
  | .mk k cs, r, b => by
    simp only [collect]
    cases hp : pre k r b with
    | none => exact ⟨⟨[], by simp⟩, ⟨[], by simp⟩⟩
    | some rb =>
      obtain ⟨⟨r₁, hr⟩, ⟨b₁, hb⟩⟩ := collectAll_extends cs rb.1 rb.2
      obtain ⟨⟨r₀, hr0⟩, ⟨b₀, hb0⟩⟩ := pre_extends k r b rb hp
      exact ⟨⟨r₀ ++ r₁, by rw [hr, hr0]; simp⟩, ⟨b₀ ++ b₁, by rw [hb, hb0]; simp⟩⟩
theorem collectAll_extends : ∀ (cs : List BT) (r b : List Nat),
    (∃ r₁, (collectAll cs r b).2.1 = r ++ r₁) ∧ (∃ b₁, (collectAll cs r b).2.2 = b ++ b₁)
  | [], r, b => ⟨⟨[], by simp [collectAll]⟩, ⟨[], by simp [collectAll]⟩⟩
  | c :: cs, r, b => by
    simp only [collectAll]
    obtain ⟨⟨r₁, hr⟩, ⟨b₁, hb⟩⟩ := collect_extends c r b
    rcases h : collect c r b with ⟨_ | _, r', b'⟩
    · rw [h] at hr hb; simp only at hr hb ⊢; exact ⟨⟨r₁, hr⟩, ⟨b₁, hb⟩⟩
    · rw [h] at hr hb; simp only at hr hb ⊢
      obtain ⟨⟨r₂, hr2⟩, ⟨b₂, hb2⟩⟩ := collectAll_extends cs r' b'
      exact ⟨⟨r₁ ++ r₂, by rw [hr2, hr]; simp⟩, ⟨b₁ ++ b₂, by rw [hb2, hb]; simp⟩⟩
end

-- Every variable a node reads.
mutual
def varsOf : BT → List Nat
  | .mk k cs => (match k with | .var id => [id] | _ => []) ++ varsAll cs
def varsAll : List BT → List Nat
  | [] => []
  | c :: cs => varsOf c ++ varsAll cs
end

-- An accepted body records every variable it reads.
mutual
theorem collect_reads : ∀ (t : BT) (r b : List Nat), (collect t r b).1 = true →
    ∀ id ∈ varsOf t, id ∈ (collect t r b).2.1
  | .mk k cs, r, b, hok, id, hid => by
    simp only [collect] at hok ⊢
    cases hp : pre k r b with
    | none => simp [hp] at hok
    | some rb =>
      simp only [hp] at hok ⊢
      simp only [varsOf, List.mem_append] at hid
      rcases hid with hk | hc
      · cases k <;> simp at hk
        subst hk; simp only [pre, Option.some.injEq] at hp; subst hp
        obtain ⟨⟨r₁, hr⟩, _⟩ := collectAll_extends cs (r ++ [id]) b
        rw [hr]; simp
      · exact collectAll_reads cs rb.1 rb.2 hok id hc
theorem collectAll_reads : ∀ (cs : List BT) (r b : List Nat), (collectAll cs r b).1 = true →
    ∀ id ∈ varsAll cs, id ∈ (collectAll cs r b).2.1
  | [], _, _, _, id, hid => by simp [varsAll] at hid
  | c :: cs, r, b, hok, id, hid => by
    simp only [collectAll] at hok ⊢
    rcases h : collect c r b with ⟨_ | _, r', b'⟩
    · rw [h] at hok; simp at hok
    · rw [h] at hok; simp only at hok ⊢
      simp only [varsAll, List.mem_append] at hid
      rcases hid with hc | hcs
      · have := collect_reads c r b (by rw [h]) id hc
        rw [h] at this
        obtain ⟨⟨r₁, hr⟩, _⟩ := collectAll_extends cs r' b'
        rw [hr]; exact List.mem_append_left _ this
      · exact collectAll_reads cs r' b' hok id hcs
end

/-- `ListLoop::vector_plan` (eval.rs:344-371). `reads.sort_unstable(); reads.dedup()` is
`eraseDups` up to order (the plan is a set of bindings; order is irrelevant to the batch). -/
def vectorPlan {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V) :
    Option (List (Nat × V)) :=
  match collectAll bodies [] [varId] with
  | (true, reads, bound) =>
    ((reads.eraseDups).filter fun id => !bound.contains id).mapM fun id => (valueAt id).map (id, ·)
  | (false, _, _) => none

theorem vectorPlan_some {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V)
    (outer : List (Nat × V)) (h : vectorPlan varId bodies valueAt = some outer) :
    badAll bodies = false := by
  unfold vectorPlan at h
  have hc := collectAll_rejects_iff bodies [] [varId]
  rcases hcol : collectAll bodies [] [varId] with ⟨_ | _, reads, bound⟩
  · rw [hcol] at h; simp at h
  · rw [hcol] at hc; cases hb : badAll bodies <;> simp_all

/-- **No excluded shape anywhere in the predicate or the projection** when the columnar path
is taken. -/
theorem vectorPlan_no_bad {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V)
    (outer : List (Nat × V)) (h : vectorPlan varId bodies valueAt = some outer) :
    ∀ t ∈ bodies, bad t = false := by
  have hb := vectorPlan_some varId bodies valueAt outer h
  clear h
  intro t ht
  induction bodies with
  | nil => simp at ht
  | cons c cs ih =>
    simp only [badAll, Bool.or_eq_false_iff] at hb
    rcases List.mem_cons.1 ht with rfl | ht'
    · exact hb.1
    · exact ih hb.2 ht'

theorem mapM_mem {α β : Type} (f : α → Option β) : ∀ (l : List α) (out : List β), l.mapM f = some out →
    ∀ a ∈ l, ∃ b, f a = some b ∧ b ∈ out
  | [], _, _, a, ha => by simp at ha
  | x :: xs, out, h, a, ha => by
    simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨y, hy, ys, hys, rfl⟩ := h
    rcases List.mem_cons.1 ha with rfl | ha'
    · exact ⟨y, hy, by simp⟩
    · obtain ⟨b, hb, hm⟩ := mapM_mem f xs ys hys a ha'
      exact ⟨b, hb, by simp [hm]⟩

/-- **Every free variable the bodies read is broadcast with its environment value**: a
variable read by the predicate or projection is the loop variable, bound inside the body
(nested comprehension / quantifier / `reduce`), or a broadcast column holding
`env.value_at(id)`. -/
theorem vectorPlan_broadcasts {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V)
    (outer : List (Nat × V)) (h : vectorPlan varId bodies valueAt = some outer) :
    ∀ id ∈ varsAll bodies, id ∈ (collectAll bodies [] [varId]).2.2 ∨
      ∃ v, valueAt id = some v ∧ (id, v) ∈ outer := by
  intro id hid
  unfold vectorPlan at h
  have hreads := collectAll_reads bodies [] [varId]
  rcases hcol : collectAll bodies [] [varId] with ⟨_ | _, reads, bound⟩
  · rw [hcol] at h; simp at h
  · rw [hcol] at h hreads
    have hr := hreads rfl id hid
    simp only at h hr ⊢
    by_cases hb : bound.contains id = true
    · exact .inl (by simpa using hb)
    · right
      have hmem : id ∈ (reads.eraseDups).filter fun id => !bound.contains id := by
        rw [List.mem_filter]; exact ⟨List.mem_eraseDups.2 hr, by simpa using hb⟩
      obtain ⟨p, hp, hpo⟩ := mapM_mem _ _ outer h id hmem
      cases hv : valueAt id with
      | none => simp [hv] at hp
      | some v => simp only [hv, Option.map_some, Option.some.injEq] at hp; subst hp; exact ⟨v, rfl, hpo⟩

end Pr3437.Classifier
