import Pr3437Review.ClassifierPre

/-!
# PR #3437 (a739b990f): which bodies may run columnarly (`collect_body_vars`, `vector_plan`)

| Lean | Rust |
| --- | --- |
| `BK`, `BT` | the `ExprIR` shapes `collect_body_vars` distinguishes (shared with `ClassifierPre`) |
| `pre`, `collect`, `collectAll` | `collect_body_vars` (eval.rs:380-402); `collectAll` is `node.children().all(..)` (:400-401), short-circuiting |
| `bad` | a nested `ListComprehension` / `Quantifier` / `Reduce`, a `NestedPlan`, `Pattern`, `PatternComprehension`, `Distinct`, a writing or aggregating function — anywhere in the tree |
| `vectorPlan` | `ListLoop::vector_plan` (eval.rs:344-370): `.all(..)` over the present bodies, then every read other than the loop variable resolved through `env.value_at` (:361) |

**`collect_rejects_iff`**: `collect_body_vars` returns `false` exactly when the tree has a
`bad` node at any depth — nested binders included since a739b990f. **`vectorPlan_broadcasts`**:
under a plan every variable the bodies read is the loop variable or broadcast with its
environment value — no other binding exists in a columnar body. History:
`pre_a739b990f_accepts_nested_binder` (6efd69cd6 let `[x IN l WHERE any(z IN [x] WHERE z > 0)]`
take the columnar path).
-/

namespace Pr3437.Classifier
open Pr3437.ClassifierPre (BK BT)

/-- The `match node.data()` of eval.rs:384-399: `none` = `return false`. -/
def pre : BK → List Nat → Option (List Nat)
  | .var id, r => some (r ++ [id])
  | .listComp _, _ | .quant _, _ | .reduce _ _, _ => none
  | .nestedPlan, _ | .pattern, _ | .patternComp, _ | .distinct, _ => none
  | .func w a, r => if w || a then none else some r
  | .other, r => some r

mutual
def collect : BT → List Nat → Bool × List Nat
  | .mk k cs, r =>
    match pre k r with
    | none => (false, r)
    | some r' => collectAll cs r'
def collectAll : List BT → List Nat → Bool × List Nat
  | [], r => (true, r)
  | c :: cs, r =>
    match collect c r with
    | (true, r') => collectAll cs r'
    | (false, r') => (false, r')
end

def badKind : BK → Bool
  | .listComp _ | .quant _ | .reduce _ _ => true
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

theorem pre_none_iff (k : BK) (r : List Nat) : pre k r = none ↔ badKind k = true := by
  cases k with
  | func w a => cases w <;> cases a <;> simp [pre, badKind]
  | _ => simp [pre, badKind]

theorem pre_extends (k : BK) (r r' : List Nat) (h : pre k r = some r') : ∃ r₀, r' = r ++ r₀ := by
  cases k with
  | func w a => cases w <;> cases a <;> simp [pre] at h <;> subst h <;> exact ⟨[], by simp⟩
  | var id => simp only [pre, Option.some.injEq] at h; subst h; exact ⟨[id], rfl⟩
  | other => simp only [pre, Option.some.injEq] at h; subst h; exact ⟨[], by simp⟩
  | _ => simp [pre] at h

mutual
theorem collect_rejects_iff : ∀ (t : BT) (r : List Nat), (collect t r).1 = !bad t
  | .mk k cs, r => by
    simp only [collect, bad]
    cases hp : pre k r with
    | none => simp [(pre_none_iff k r).1 hp]
    | some r' =>
      have : badKind k = false := by
        cases h : badKind k
        · rfl
        · exact absurd ((pre_none_iff k r).2 h) (by simp [hp])
      simp only [this, Bool.false_or]
      exact collectAll_rejects_iff cs r'
theorem collectAll_rejects_iff : ∀ (cs : List BT) (r : List Nat), (collectAll cs r).1 = !badAll cs
  | [], r => rfl
  | c :: cs, r => by
    simp only [collectAll, badAll]
    have hc := collect_rejects_iff c r
    rcases h : collect c r with ⟨_ | _, r'⟩
    · rw [h] at hc; simp only at hc ⊢; simp [hc.symm]
    · rw [h] at hc; simp only at hc ⊢
      have : bad c = false := by cases hb : bad c <;> simp_all
      simp only [this, Bool.false_or]; exact collectAll_rejects_iff cs r'
end

/-- **Claim 3.** A body with a whole-row read, `DISTINCT`, an aggregate, a write, or (since
a739b990f) a variable binder of its own, at any depth, is rejected. -/
theorem collect_rejects (t : BT) (r : List Nat) (h : bad t = true) : (collect t r).1 = false := by
  rw [collect_rejects_iff, h]; rfl

mutual
theorem collect_extends : ∀ (t : BT) (r : List Nat), ∃ r₁, (collect t r).2 = r ++ r₁
  | .mk k cs, r => by
    simp only [collect]
    cases hp : pre k r with
    | none => exact ⟨[], by simp⟩
    | some r' =>
      obtain ⟨r₁, hr⟩ := collectAll_extends cs r'
      obtain ⟨r₀, hr0⟩ := pre_extends k r r' hp
      exact ⟨r₀ ++ r₁, by rw [hr, hr0]; simp⟩
theorem collectAll_extends : ∀ (cs : List BT) (r : List Nat), ∃ r₁, (collectAll cs r).2 = r ++ r₁
  | [], r => ⟨[], by simp [collectAll]⟩
  | c :: cs, r => by
    simp only [collectAll]
    obtain ⟨r₁, hr⟩ := collect_extends c r
    rcases h : collect c r with ⟨_ | _, r'⟩
    · rw [h] at hr; simp only at hr ⊢; exact ⟨r₁, hr⟩
    · rw [h] at hr; simp only at hr ⊢
      obtain ⟨r₂, hr2⟩ := collectAll_extends cs r'
      exact ⟨r₁ ++ r₂, by rw [hr2, hr]; simp⟩
end

open Pr3437.ClassifierPre (varsOf varsAll) in
mutual
theorem collect_reads : ∀ (t : BT) (r : List Nat), (collect t r).1 = true →
    ∀ id ∈ varsOf t, id ∈ (collect t r).2
  | .mk k cs, r, hok, id, hid => by
    simp only [collect] at hok ⊢
    cases hp : pre k r with
    | none => simp [hp] at hok
    | some r' =>
      simp only [hp] at hok ⊢
      simp only [ClassifierPre.varsOf, List.mem_append] at hid
      rcases hid with hk | hc
      · cases k <;> simp at hk
        subst hk; simp only [pre, Option.some.injEq] at hp; subst hp
        obtain ⟨r₁, hr⟩ := collectAll_extends cs (r ++ [id])
        rw [hr]; simp
      · exact collectAll_reads cs r' hok id hc
theorem collectAll_reads : ∀ (cs : List BT) (r : List Nat), (collectAll cs r).1 = true →
    ∀ id ∈ varsAll cs, id ∈ (collectAll cs r).2
  | [], _, _, id, hid => by simp [ClassifierPre.varsAll] at hid
  | c :: cs, r, hok, id, hid => by
    simp only [collectAll] at hok ⊢
    rcases h : collect c r with ⟨_ | _, r'⟩
    · rw [h] at hok; simp at hok
    · rw [h] at hok; simp only at hok ⊢
      simp only [ClassifierPre.varsAll, List.mem_append] at hid
      rcases hid with hc | hcs
      · have := collect_reads c r (by rw [h]) id hc
        rw [h] at this
        obtain ⟨r₁, hr⟩ := collectAll_extends cs r'
        rw [hr]; exact List.mem_append_left _ this
      · exact collectAll_reads cs r' hok id hcs
end

/-- `ListLoop::vector_plan` (eval.rs:344-370), a739b990f: reads other than the loop
variable, deduplicated (`sort_unstable` + `dedup` is `eraseDups` up to order). -/
def vectorPlan {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V) :
    Option (List (Nat × V)) :=
  match collectAll bodies [] with
  | (true, reads) =>
    ((reads.eraseDups).filter fun id => id != varId).mapM fun id => (valueAt id).map (id, ·)
  | (false, _) => none

theorem vectorPlan_some {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V)
    (outer : List (Nat × V)) (h : vectorPlan varId bodies valueAt = some outer) :
    badAll bodies = false := by
  unfold vectorPlan at h
  have hc := collectAll_rejects_iff bodies []
  rcases hcol : collectAll bodies [] with ⟨_ | _, reads⟩
  · rw [hcol] at h; simp at h
  · rw [hcol] at hc; cases hb : badAll bodies <;> simp_all

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

open Pr3437.ClassifierPre (varsAll mapM_mem) in
/-- **Every variable a columnar body reads is the loop variable or broadcast with its
environment value.** No id-uniqueness premise: since a739b990f nothing else is bound. -/
theorem vectorPlan_broadcasts {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V)
    (outer : List (Nat × V)) (h : vectorPlan varId bodies valueAt = some outer) :
    ∀ id ∈ varsAll bodies, id = varId ∨ ∃ v, valueAt id = some v ∧ (id, v) ∈ outer := by
  intro id hid
  unfold vectorPlan at h
  have hreads := collectAll_reads bodies []
  rcases hcol : collectAll bodies [] with ⟨_ | _, reads⟩
  · rw [hcol] at h; simp at h
  · rw [hcol] at h hreads
    have hr := hreads rfl id hid
    simp only at h hr
    by_cases hb : id = varId
    · exact .inl hb
    · right
      have hmem : id ∈ (reads.eraseDups).filter fun id => id != varId := by
        rw [List.mem_filter]; exact ⟨List.mem_eraseDups.2 hr, by simpa using hb⟩
      obtain ⟨p, hp, hpo⟩ := mapM_mem _ _ outer h id hmem
      cases hv : valueAt id with
      | none => simp [hv] at hp
      | some v => simp only [hv, Option.map_some, Option.some.injEq] at hp; subst hp; exact ⟨v, rfl, hpo⟩

/-- Every pair in a plan is `(id, env.value_at(id))`. -/
theorem vectorPlan_vals {V : Type} (varId : Nat) (bodies : List BT) (valueAt : Nat → Option V)
    (outer : List (Nat × V)) (h : vectorPlan varId bodies valueAt = some outer) :
    ∀ p ∈ outer, valueAt p.1 = some p.2 := by
  intro p hp
  unfold vectorPlan at h
  rcases hcol : collectAll bodies [] with ⟨_ | _, reads⟩
  · rw [hcol] at h; simp at h
  · rw [hcol] at h; simp only at h
    exact vals_of_mapM valueAt _ outer h p hp
where
  vals_of_mapM {V : Type} (valueAt : Nat → Option V) : ∀ (ids : List Nat) (out : List (Nat × V)),
      ids.mapM (fun id => (valueAt id).map (id, ·)) = some out → ∀ p ∈ out, valueAt p.1 = some p.2
    | [], out, h, p, hp => by simp at h; subst h; simp at hp
    | x :: xs, out, h, p, hp => by
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨y, hy, ys, hys, rfl⟩ := h
      rcases List.mem_cons.1 hp with rfl | hp'
      · cases hv : valueAt x with
        | none => simp [hv] at hy
        | some v => simp only [hv, Option.map_some, Option.some.injEq] at hy; subst hy; exact hv
      · exact vals_of_mapM valueAt xs ys hys p hp'

/-! ## History: the 6efd69cd6 classifier accepted nested binders -/

/-- `[x IN l WHERE any(z IN [x] WHERE z > 0)]`'s predicate, as the classifiers see it. -/
def nestedQuant (x z : Nat) : BT :=
  .mk .other [.mk (.quant z) [.mk .other [.mk (.var x) []], .mk .other [.mk (.var z) [], .mk .other []]]]

/-- **History (fixed by a739b990f).** At 6efd69cd6 a body with a nested quantifier took the
columnar path; now it runs per element. -/
theorem pre_a739b990f_accepts_nested_binder :
    (ClassifierPre.collect (nestedQuant 0 1) [] [0]).1 = true ∧
    (collect (nestedQuant 0 1) []).1 = false := by
  decide

end Pr3437.Classifier
