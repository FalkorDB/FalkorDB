import OptimizerScan.Index
namespace OptimizerScan.Index

/-- Property values the index represents faithfully: Null, Int, String. -/
def faithV : V → Bool
  | .null | .i _ | .s _ => true
  | _ => false

/-- Literals the index represents faithfully: non-lossy Int, String. -/
def goodV : V → Bool
  | .i x => !lossy x
  | .s _ => true
  | _ => false

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

section
variable (idx : Nat → List Nat) (L k : Nat) (n : Node)

theorem sel_eq (c : V) (hc : goodV c = true) (hv : faithV (n.prop k) = true)
    (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    idxSel idx L (.eq k c) n = Op.eq.holds (n.prop k) c := by
  unfold idxSel bsel
  simp only [hL, decide_true, Bool.true_and]
  cases hp : n.prop k <;> rw [hp] at hv <;> simp only [faithV] at hv <;> (try contradiction) <;>
    cases c <;> simp_all [goodV, build, valueToNumeric, enc, Op.holds, eq3]

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
    (hv : faithV (n.prop k) = true) (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    idxSel idx L (.range k lo hi il ih) n =
      (lo.all (loHolds il (n.prop k)) && hi.all (hiHolds ih (n.prop k))) := by
  unfold idxSel bsel
  simp only [hL, decide_true, Bool.true_and]
  cases hp : n.prop k <;> rw [hp] at hv <;> simp only [faithV] at hv <;> (try contradiction) <;>
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
  · exact sel_eq idx L k n c hc hv hk hL
  all_goals
    simp only [buildOp, evalIQ, Option.map, evalC, evalT]
    rw [sel_range idx L k n _ _ _ _ (by simp [hc]) (by simp [hc]) (by simp)
      (by cases c <;> simp [strTrap]) hv hk hL]
    simp [loHolds, hiHolds]

def goodLitT : T → Bool
  | .lit c => goodV c
  | _ => false

theorem build_eq_good (c : V) (hc : goodV c = true) (hk : k ∈ idx L) :
    ∃ f, build (idx L) (.eq k c) = some f ∧ ∀ m : Node, faithV (m.prop k) = true →
      f m = Op.eq.holds (m.prop k) c := by
  cases c <;> simp [goodV] at hc
  all_goals
    refine ⟨_, by simp [build, hk, valueToNumeric]; rfl, ?_⟩
    intro m hv
    cases hp : m.prop k <;> rw [hp] at hv <;> simp only [faithV] at hv <;> (try contradiction) <;>
      simp [enc, Op.holds, eq3]

theorem buildSome_eqs (cs : List V) (hcs : cs.all goodV = true) (hk : k ∈ idx L)
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

theorem goodV_isPrim (c : V) (h : goodV c = true) : isPrim c = true := by
  cases c <;> simp_all [goodV, isPrim]

/-- **PROVEN**: `n.k IN [c₁,…,cₙ]` on faithful literals is exact through the index
(`InList` → runtime `Or` of `Equal`s → RediSearch union). -/
theorem sel_inList (ts : List T) (hts : ts.all goodLitT = true) (hv : faithV (n.prop k) = true)
    (hk : k ∈ idx L) (hL : L ∈ n.labels) :
    qsem idx L (.inList k (.list ts)) n = inList (n.prop k) (ts.map (evalT n)) := by
  have hmap : ts.map (evalT n) = ts.map evalC :=
    List.map_congr_left (fun t ht => evalT_good n t (List.all_eq_true.mp hts t ht))
  have hgood : (ts.map evalC).all goodV = true := by
    simp only [List.all_map, List.all_eq_true, Function.comp]
    intro t ht
    have := List.all_eq_true.mp hts t ht
    cases t <;> simp_all [goodLitT, evalC, evalT]
  have hfilt : (ts.map evalC).filter isPrim = ts.map evalC :=
    List.filter_eq_self.mpr (fun c hc => goodV_isPrim c (List.all_eq_true.mp hgood c hc))
  unfold qsem idxSel bsel
  simp only [evalIQ, listVals, hfilt, build, hL, decide_true, Bool.true_and, hmap]
  rw [buildSome_eqs idx L k n _ hgood hk hv]
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

/-- The atom shapes the soundness theorem covers. -/
def goodA : A → Bool
  | .cmp _ (.prop _) (.lit c) => goodV c
  | .cmp _ (.lit c) (.prop _) => goodV c
  | .inn (.prop _) (.list ts) => ts.all goodLitT
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

/-- **PROVEN** (`try_single_filter_scan` is exact on the covered atoms): the pushed query is
on label `L`, selects exactly the nodes satisfying the atom, is a well-formed range when it is
one (given no strict string comparison), and is usable by the runtime unless it is an `IN`. -/
theorem trySingle_spec (opq : Nat → Node → Bool) (a : A) (L' : Nat) (q : IQ)
    (hg : goodA a = true) (h : trySingle idx [L] a = some (L', q)) :
    L' = L ∧ (∀ n : Node, Faithful n → L ∈ n.labels → qsem idx L q n = evalA opq n a) ∧
      (strictStr a = false → rangeOK idx L q = true) ∧
      (isInn a = false → canUtilize (evalIQ q) = true) ∧
      (∀ ts k, a = .inn (.prop k) (.list ts) → q = .inList k (.list ts)) := by
  match a, hg with
  | .cmp op (.prop k) (.lit c), hg =>
    simp only [goodA] at hg
    simp only [trySingle, hasT, firstP, findLabel_single, Option.bind_eq_bind] at h
    by_cases hk : k ∈ idx L <;> simp [hk] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨rfl, fun n hn hL => ?_, fun hs => ?_, fun _ => ?_, fun _ _ h => by cases h⟩
    · rw [sel_buildOp idx L k n op c hg (hn k) hk hL]; rfl
    · cases op <;> cases c <;> simp_all [buildOp, rangeOK, goodLitT, isStrLitT, strictStr]
    · cases op <;> cases c <;> simp_all [buildOp, evalIQ, canUtilize, isIndexable, goodV, evalC, evalT]
  | .cmp op (.lit c) (.prop k), hg =>
    simp only [goodA] at hg
    simp only [trySingle, hasT, firstP, findLabel_single, Option.bind_eq_bind] at h
    by_cases hk : k ∈ idx L <;> simp [hk] at h
    obtain ⟨rfl, rfl⟩ := h
    refine ⟨rfl, fun n hn hL => ?_, fun hs => ?_, fun _ => ?_, fun _ _ h => by cases h⟩
    · rw [sel_buildOp idx L k n op.flip c hg (hn k) hk hL, holds_flip']; rfl
    · cases op <;> cases c <;> simp_all [buildOp, Op.flip, rangeOK, goodLitT, isStrLitT, strictStr]
    · cases op <;> cases c <;>
        simp_all [buildOp, Op.flip, evalIQ, canUtilize, isIndexable, goodV, evalC, evalT]
  | .inn (.prop k) (.list ts), hg =>
    simp only [goodA] at hg
    have hnt : ts.any hasT = false := by
      simp only [List.any_eq_false]
      intro t ht
      have := List.all_eq_true.mp hg t ht
      cases t <;> simp_all [goodLitT, hasT]
    have hnest : nestedList (.list ts) = false := by
      simp only [nestedList, List.any_eq_false]
      intro t ht
      have := List.all_eq_true.mp hg t ht
      rcases t with _ | ⟨_ | _ | _ | _ | _ | _⟩ | _ | _ | _ <;> simp_all [goodLitT, goodV]
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
