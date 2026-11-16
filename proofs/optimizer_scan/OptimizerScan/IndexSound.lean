import OptimizerScan.Index
namespace OptimizerScan.Index

/-- Property values the index represents faithfully. Since #2975 that is every modelled value:
Null, Int and String as before; a Bool, which has its own TAG key instead of sharing the NUMERIC
field with Int (`IndexCex` C1, fixed); and temporals and lists, which have no main-field key
(since #3076 for temporals), so no `Equal`/`Range` on the main field returns them, and none of them
compares `true` against a good literal. (Before #2975: Null, Int, String only.) -/
def faithV : V → Bool
  | .null | .b _ | .i _ | .s _ | .date _ | .arr _ => true

/-- Literals a range bound (and any comparison) is exact on: non-lossy Int, String. -/
def goodV : V → Bool
  | .i x => !lossy x
  | .s _ => true
  | _ => false

/-- Literals an equality or `IN` list is exact on: `goodV`, and since #2975 also Bool (a TAG key
of its own; `is_exact_index_literal` and `can_be_served` accept it in the default build). -/
def eqV : V → Bool
  | .b _ => true
  | v => goodV v

theorem goodV_eqV (v : V) (h : goodV v = true) : eqV v = true := by
  cases v <;> simp_all [eqV, goodV]

@[simp] theorem V.beq_def (a b : V) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]
@[simp] theorem Fld.beq_def (a b : Fld) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]
@[simp] theorem OptFld.beq_def (a b : Option Fld) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]
@[simp] theorem OptOrd.beq_def (a b : Option Ordering) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]

@[simp] theorem Ord.beq_def (a b : Ordering) : (a == b) = decide (a = b) := by
  by_cases h : a = b <;> simp [h]

@[simp] theorem ordI_lt (x y : Int) : ordI x y = .lt ↔ x < y := by
  unfold ordI; split <;> (try split) <;> simp_all <;> omega
@[simp] theorem ordI_gt (x y : Int) : ordI x y = .gt ↔ y < x := by
  unfold ordI; split <;> (try split) <;> simp_all <;> omega
@[simp] theorem ordI_eq (x y : Int) : ordI x y = .eq ↔ x = y := by
  unfold ordI; split <;> (try split) <;> simp_all <;> omega

def Faithful (n : Node) : Prop := ∀ k, faithV (n.prop k) = true

/-- **PROVEN** (#2975): every node is faithful, so the `Faithful` premise of `utilize_sound` and
its relatives is now always met. -/
theorem faithful_all (n : Node) : Faithful n := fun k => by cases n.prop k <;> rfl

section
variable (idx : Nat → List Nat) (L k : Nat) (n : Node)

theorem sel_eq (c : V) (hc : eqV c = true) (_hv : faithV (n.prop k) = true)
    (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    idxSel idx L (.eq k c) n = Op.eq.holds (n.prop k) c := by
  unfold idxSel bsel
  simp only [hL, decide_true, Bool.true_and]
  cases hp : n.prop k <;>
    cases c <;> simp_all [eqV, goodV, build, valueToNumeric, enc, Op.holds, eq3]

def loHolds (il : Bool) (v c : V) : Bool := (if il then Op.ge else Op.gt).holds v c
def hiHolds (ih : Bool) (v c : V) : Bool := (if ih then Op.le else Op.lt).holds v c

/-- The string trap of `build_string_range_node`: equal string bounds with an exclusive side. -/
def strTrap (lo hi : Option V) (il ih : Bool) : Bool :=
  match lo, hi with
  | some (.s x), some (.s y) => x == y && !(il && ih)
  | _, _ => false

/-- **PROVEN**: a range query on faithful values and literals selects exactly the nodes whose
value satisfies both bounds — the only exception being `strTrap`. -/
theorem sel_range (lo hi : Option V) (il ih : Bool)
    (hlo : lo.all goodV = true) (hhi : hi.all goodV = true) (hne : lo.isSome || hi.isSome)
    (htrap : strTrap lo hi il ih = false)
    (_hv : faithV (n.prop k) = true) (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    idxSel idx L (.range k lo hi il ih) n =
      (lo.all (loHolds il (n.prop k)) && hi.all (hiHolds ih (n.prop k))) := by
  unfold idxSel bsel
  simp only [hL, decide_true, Bool.true_and]
  cases hp : n.prop k <;>
  rcases lo with _ | ⟨_ | _ | x | x | _ | _⟩ <;> rcases hi with _ | ⟨_ | _ | y | y | _ | _⟩ <;>
    simp [goodV] at hlo hhi hne <;>
    cases il <;> cases ih <;>
    simp_all [build, strB, numB, isStrB, valueToNumeric, enc, within, geB, leB, loHolds, hiHolds,
      Op.holds, ord3, strTrap] <;> (repeat' split) <;> (try simp_all) <;> first | omega | (rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq]; omega)

def qsem (q : IQ) (n : Node) : Bool := idxSel idx L (evalIQ q) n

/-- **PROVEN**: `build_op_query` on a faithful literal is exact for every comparison. -/
theorem sel_buildOp (op : Op) (c : V) (hc : goodV c = true) (hv : faithV (n.prop k) = true)
    (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    qsem idx L (buildOp op k (.lit c)) n = op.holds (n.prop k) c := by
  unfold qsem
  cases op
  · exact sel_eq idx L k n c (goodV_eqV c hc) hv hk hL
  all_goals
    simp only [buildOp, evalIQ, Option.map, evalC, evalT]
    rw [sel_range idx L k n _ _ _ _ (by simp [hc]) (by simp [hc]) (by simp)
      (by cases c <;> simp [strTrap]) hv hk hL]
    simp [loHolds, hiHolds]

def goodLitT : T → Bool
  | .lit c => goodV c
  | _ => false

/-- An `IN`-list member / equality literal the index is exact on (`eqV`). -/
def eqLitT : T → Bool
  | .lit c => eqV c
  | _ => false

theorem build_eq_good (c : V) (hc : eqV c = true) (hk : k ∈ idx L) :
    ∃ f, build (idx L) (.eq k c) = some f ∧ ∀ m : Node, faithV (m.prop k) = true →
      f m = Op.eq.holds (m.prop k) c := by
  cases c <;> simp [eqV, goodV] at hc
  all_goals
    refine ⟨_, by simp [build, hk, valueToNumeric]; rfl, ?_⟩
    intro m _
    cases hp : m.prop k <;> simp [enc, Op.holds, eq3]

theorem buildSome_eqs (cs : List V) (hcs : cs.all eqV = true) (hk : k ∈ idx L)
    (hv : faithV (n.prop k) = true) :
    (buildSome (idx L) (cs.map (Q.eq k))).any (· n) = cs.any (Op.eq.holds (n.prop k)) := by
  induction cs with
  | nil => simp [buildSome]
  | cons c cs ih =>
    simp only [List.all_cons, Bool.and_eq_true] at hcs
    obtain ⟨f, hf, hfn⟩ := build_eq_good idx L k c hcs.1 hk
    simp only [List.map_cons, buildSome, hf, List.any_cons, ih hcs.2, hfn n hv]

theorem evalT_good (m : Node) (t : T) (h : goodLitT t = true) : evalT m t = evalC t := by
  cases t <;> simp [goodLitT] at h; rfl

theorem evalT_eqLit (m : Node) (t : T) (h : eqLitT t = true) : evalT m t = evalC t := by
  cases t <;> simp [eqLitT] at h; rfl

theorem eqV_notNull (c : V) (h : eqV c = true) : notNull c = true := by
  cases c <;> simp_all [eqV, goodV, notNull]

/-- **PROVEN**: `n.k IN [c₁,…,cₙ]` on exact literals (non-lossy Int, String, and since #2975 Bool)
is exact through the index (`InList` kept whole since #2383 → `build_query_node`'s `InList` arm,
a RediSearch union of `Equal`s; `build_inList_eq_or` says it is the union the pre-#2383 `Or`
built). -/
theorem sel_inList (ts : List T) (hts : ts.all eqLitT = true) (hv : faithV (n.prop k) = true)
    (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    qsem idx L (.inList k (.list ts)) n = inList (n.prop k) (ts.map (evalT n)) := by
  have hmap : ts.map (evalT n) = ts.map evalC :=
    List.map_congr_left (fun t ht => evalT_eqLit n t (List.all_eq_true.mp hts t ht))
  have hgood : (ts.map evalC).all eqV = true := by
    simp only [List.all_map, List.all_eq_true, Function.comp]
    intro t ht
    have := List.all_eq_true.mp hts t ht
    cases t <;> simp_all [eqLitT, evalC, evalT]
  have hfilt : (ts.map evalC).filter notNull = ts.map evalC :=
    List.filter_eq_self.mpr (fun c hc => eqV_notNull c (List.all_eq_true.mp hgood c hc))
  unfold qsem idxSel bsel
  simp only [evalIQ, listVals, hfilt, build, hL, decide_true, Bool.true_and, hmap]
  rw [← buildSome_map_eq, buildSome_eqs idx L k n _ hgood hk hv]
  rfl

def isStrLitT : Option T → Bool
  | some (.lit (.s _)) => true
  | _ => false

/-- Well-formed ranges: key indexed on `L`, at least one bound, faithful literal bounds, and
string bounds inclusive (so `strTrap` cannot fire). -/
def rangeOK (q : IQ) : Bool :=
  match q with
  | .range k lo hi il ih =>
    decide (k ∈ idx L) && (lo.isSome || hi.isSome) && lo.all goodLitT && hi.all goodLitT &&
      (!isStrLitT lo || il) && (!isStrLitT hi || ih)
  | _ => true

theorem goodLitT_goodV (t : T) (h : goodLitT t = true) : goodV (evalC t) = true := by
  cases t <;> simp_all [goodLitT, evalC, evalT]

theorem qsem_range (k : Nat) (lo hi : Option T) (il ih : Bool)
    (h : rangeOK idx L (.range k lo hi il ih) = true)
    (hv : faithV (n.prop k) = true) (hL : L ∈ n.labels) :
    qsem idx L (.range k lo hi il ih) n =
      ((lo.map evalC).all (loHolds il (n.prop k)) && (hi.map evalC).all (hiHolds ih (n.prop k))) := by
  simp only [rangeOK, Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨⟨⟨hk, hne⟩, hlo⟩, hhi⟩, hsl⟩, hsh⟩ := h
  unfold qsem
  simp only [evalIQ]
  apply sel_range idx L k n _ _ _ _ _ _ _ _ hv hk hL
  · cases lo <;> simp_all [goodLitT_goodV]
  · cases hi <;> simp_all [goodLitT_goodV]
  · cases lo <;> cases hi <;> simp_all
  · rcases lo with _ | ⟨_ | ⟨_ | _ | _ | x | _ | _⟩ | _ | _ | _⟩ <;>
    rcases hi with _ | ⟨_ | ⟨_ | _ | _ | y | _ | _⟩ | _ | _ | _⟩ <;>
      simp_all [strTrap, evalC, evalT, isStrLitT, goodLitT]

/-- **PROVEN**: an `And` of two index queries selects the intersection — including when a
child is null, which nulls the whole intersection (so both sides are `false`). -/
theorem sel_and2 (a b : IQ) : qsem idx L (.and [a, b]) n = (qsem idx L a n && qsem idx L b n) := by
  unfold qsem idxSel bsel
  simp only [evalIQ, evalIQs, build, buildAll]
  cases h1 : build (idx L) (evalIQ a) <;> cases h2 : build (idx L) (evalIQ b) <;>
    by_cases hL : L ∈ n.labels <;> simp [hL]

/-- **PROVEN** (`merge_range_queries` is sound): merging two queries on the scanned node
selects exactly the nodes both select, as long as both are well-formed ranges or not ranges
at all; the merge result is again well-formed. -/
theorem sel_merge (a b : IQ) (ha : rangeOK idx L a = true) (hb : rangeOK idx L b = true)
    (hv : ∀ k, faithV (n.prop k) = true) (hL : L ∈ n.labels) :
    qsem idx L (mergeRange a b) n = (qsem idx L a n && qsem idx L b n) := by
  cases a <;> cases b <;> (try exact sel_and2 idx L n _ _)
  rename_i k lo hi il ih k' lo' hi' il' ih'
  simp only [mergeRange]
  split
  · next hkk =>
    subst hkk
    split
    · exact sel_and2 idx L n _ _
    · next hc =>
      rw [qsem_range idx L n k lo hi il ih ha (hv k) hL, qsem_range idx L n k lo' hi' il' ih' hb (hv k) hL]
      have hc' : ¬ ((lo.isSome ∧ lo'.isSome) ∨ (hi.isSome ∧ hi'.isSome)) := by simpa using hc
      simp only [rangeOK, Bool.and_eq_true, decide_eq_true_eq] at ha hb
      rcases lo with _ | l <;> rcases lo' with _ | l' <;> rcases hi with _ | h <;> rcases hi' with _ | h' <;>
        simp at hc' <;> simp only [ite_true, ite_false, Option.isSome_none, Option.isSome_some,
          Bool.false_eq_true] <;>
        (first
          | (rw [qsem_range idx L n k _ _ _ _ (by simp_all [rangeOK]) (hv k) hL]; simp [Bool.and_comm])
          | simp_all)
  · exact sel_and2 idx L n _ _

/-- The literals a comparison `op` is exact on: `eqV` for `=`, `goodV` for a range. -/
def opLitV : Op → V → Bool
  | .eq, c => eqV c
  | .lt, c | .le, c | .gt, c | .ge, c => goodV c

theorem opLitV_flip (op : Op) (c : V) : opLitV op.flip c = opLitV op c := by cases op <;> rfl

/-- The atom shapes the soundness theorem covers: an equality on an exact literal (`eqV`, Bool
included since #2975), a range comparison on a `goodV` literal, an `IN` list of exact literals,
an unindexable conjunct. -/
def goodA : A → Bool
  | .cmp op (.prop _) (.lit c) => opLitV op c
  | .cmp op (.lit c) (.prop _) => opLitV op c
  | .inn (.prop _) (.list ts) => ts.all eqLitT
  | .opq _ => true
  | _ => false

/-- A strict comparison against a string literal (`n.k > 'B'`, `'B' < n.k`, …). -/
def strictStr : A → Bool
  | .cmp op (.prop _) (.lit (.s _)) => op == .lt || op == .gt
  | .cmp op (.lit (.s _)) (.prop _) => op == .lt || op == .gt
  | _ => false

def isInn : A → Bool
  | .inn _ _ => true
  | _ => false

theorem findLabel_single (k : Nat) :
    findLabel idx [L] k = if k ∈ idx L then some L else none := by
  by_cases h : k ∈ idx L <;> simp [findLabel, List.find?, h]

theorem flip_strict (op : Op) : (op.flip == .lt || op.flip == .gt) = (op == .lt || op == .gt) := by
  cases op <;> rfl

theorem holds_flip' (op : Op) (a b : V) : op.flip.holds a b = op.holds b a :=
  (flip_correct op a b).symm

/-- `build_op_query` on a covered literal: `eqV` for `=`, `goodV` for a range. -/
theorem sel_buildOp_good (op : Op) (c : V) (hc : opLitV op c = true)
    (hv : faithV (n.prop k) = true) (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    qsem idx L (buildOp op k (.lit c)) n = op.holds (n.prop k) c := by
  cases op
  · exact sel_eq idx L k n c (by simpa [opLitV] using hc) hv hk hL
  all_goals exact sel_buildOp idx L k n _ c (by simpa [opLitV] using hc) hv hk hL

theorem cu_buildOp_good (op : Op) (c : V) (hc : opLitV op c = true) :
    canBeServed (evalIQ (buildOp op k (.lit c))) = true := by
  cases op <;> cases c <;>
    simp_all [buildOp, evalIQ, canBeServed, isIndexable, rangeBoundOK, opLitV, goodV, eqV, evalC,
      evalT]

theorem rangeOK_buildOp_good (op : Op) (c : V) (hc : opLitV op c = true)
    (hk : k ∈ idx L) (hs : ((op == .lt || op == .gt) && (match c with | .s _ => true | _ => false)) = false) :
    rangeOK idx L (buildOp op k (.lit c)) = true := by
  cases op <;> cases c <;> simp only [opLitV, goodV, eqV] at hc <;>
    simp_all [buildOp, rangeOK, goodLitT, isStrLitT, goodV]

/-- **PROVEN** (`try_single_filter_scan` is exact on the covered atoms): the pushed query is
on label `L`, selects exactly the nodes satisfying the atom, is a well-formed range when it is
one (given no strict string comparison), and is usable by the runtime (`can_be_served`) unless it
is an `IN`. -/
theorem trySingle_spec (opq : Nat → Node → Bool) (a : A) (L' : Nat) (q : IQ)
    (hg : goodA a = true) (h : trySingle idx [L] a = some (L', q)) :
    L' = L ∧ (∀ n : Node, Faithful n → L ∈ n.labels → qsem idx L q n = evalA opq n a) ∧
      (strictStr a = false → rangeOK idx L q = true) ∧
      (isInn a = false → canBeServed (evalIQ q) = true) ∧
      (∀ ts k, a = .inn (.prop k) (.list ts) → q = .inList k (.list ts)) := by
  match a, hg with
  | .cmp op (.prop k) (.lit c), hg =>
    simp only [goodA] at hg
    simp only [trySingle, hasT, firstP, findLabel_single, Option.bind_eq_bind] at h
    by_cases hk : k ∈ idx L <;> simp [hk] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨rfl, fun n hn hL => ?_, fun hs => ?_, fun _ => cu_buildOp_good k op c hg,
      fun _ _ h => by cases h⟩
    · rw [sel_buildOp_good idx L k n op c hg (hn k) hk hL]; rfl
    · apply rangeOK_buildOp_good idx L k op c hg hk
      cases op <;> cases c <;> simp_all [strictStr]
  | .cmp op (.lit c) (.prop k), hg =>
    simp only [goodA] at hg
    simp only [trySingle, hasT, firstP, findLabel_single, Option.bind_eq_bind] at h
    by_cases hk : k ∈ idx L <;> simp [hk] at h
    obtain ⟨rfl, rfl⟩ := h
    have hg' : opLitV op.flip c = true := by rw [opLitV_flip]; exact hg
    refine ⟨rfl, fun n hn hL => ?_, fun hs => ?_, fun _ => cu_buildOp_good k op.flip c hg',
      fun _ _ h => by cases h⟩
    · rw [sel_buildOp_good idx L k n op.flip c hg' (hn k) hk hL, holds_flip']; rfl
    · apply rangeOK_buildOp_good idx L k op.flip c hg' hk
      cases op <;> cases c <;> simp_all [strictStr, Op.flip]
  | .inn (.prop k) (.list ts), hg =>
    simp only [goodA] at hg
    have hnt : ts.any hasT = false := by
      simp only [List.any_eq_false]
      intro t ht
      have := List.all_eq_true.mp hg t ht
      cases t <;> simp_all [eqLitT, hasT]
    have hnest : nestedList (.list ts) = false := by
      simp only [nestedList, List.any_eq_false]
      intro t ht
      have := List.all_eq_true.mp hg t ht
      rcases t with _ | ⟨_ | _ | _ | _ | _ | _⟩ | _ | _ | _ <;> simp_all [eqLitT, eqV, goodV]
    simp only [trySingle, hasT, hasR, hnt, firstP, findLabel_single, Option.bind_eq_bind,
      anyPropR, hnest] at h
    by_cases hk : k ∈ idx L <;> simp [hk] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨rfl, fun n hn hL => ?_, fun _ => rfl, fun h => by simp [isInn] at h, fun _ _ h => ?_⟩
    · rw [sel_inList idx L k n ts hg (hn k) hk hL]; rfl
    · cases h; rfl
  | .opq i, _ => simp [trySingle] at h

end
end OptimizerScan.Index
