/-!
# Function registry and dispatch (functions/mod.rs:419-1160)

`RuntimeFn` dispatch, the `GraphFn` constructors, `Functions::add*` / `set_*_fn` / `iter`,
the global `FUNCTIONS` (`OnceLock`) and the UDF registry (`OnceLock<RwLock<HashMap>>` +
`AtomicU64` version), modelled as sequential state.

Assumptions as structures/hypotheses, not axioms:
* `Lower` — Rust `str::to_lowercase` (idempotent).
* the QuickJS UDF bridge `call_udf_bridge` (FFI) — an abstract function argument.
* concurrency: `RwLock` writes are serialised and `fetch_add` is atomic, so a run of the
  UDF operations is some sequential interleaving; theorems are about sequences of ops.
* `assert!`/`panic!`/`expect` are `none` (`Option`).
-/
namespace ValueMath.Reg

/-- `enum Type` (mod.rs:530). -/
inductive Ty where
  | null | bool | int | float | string | list (t : Ty) | map | node | rel | path | vecf32
  | point | datetime | date | time | duration | any | union (ts : List Ty) | optional (t : Ty)

/-- `Type::union` (mod.rs:560): collect into `Union`. -/
def Ty.mkUnion (ts : List Ty) : Ty := .union ts

theorem mkUnion_eq (ts : List Ty) : Ty.mkUnion ts = .union ts := rfl

mutual
/-- `can_return_entity` (mod.rs:579). -/
def Ty.canReturnEntity : Ty → Bool
  | .node | .rel | .path | .any => true
  | .union ts => anyEntity ts
  | .optional t => t.canReturnEntity
  | _ => false
def anyEntity : List Ty → Bool
  | [] => false
  | t :: ts => t.canReturnEntity || anyEntity ts
end

/-- Entity-capable leaves reachable through `Union`/`Optional` (not through `List`). -/
inductive EntityLeaf : Ty → Prop
  | node : EntityLeaf .node
  | rel : EntityLeaf .rel
  | path : EntityLeaf .path
  | any : EntityLeaf .any
  | union {t ts} : t ∈ ts → EntityLeaf t → EntityLeaf (.union ts)
  | optional {t} : EntityLeaf t → EntityLeaf (.optional t)

mutual
theorem canReturnEntity_sound : (t : Ty) → t.canReturnEntity = true → EntityLeaf t
  | .node, _ => .node | .rel, _ => .rel | .path, _ => .path | .any, _ => .any
  | .optional t, h => .optional (canReturnEntity_sound t (by simpa [Ty.canReturnEntity] using h))
  | .union ts, h => by
    simp only [Ty.canReturnEntity] at h
    obtain ⟨t, ht, he⟩ := anyEntity_sound ts h
    exact .union ht he
  | .null, h | .bool, h | .int, h | .float, h | .string, h | .list _, h | .map, h | .vecf32, h
  | .point, h | .datetime, h | .date, h | .time, h | .duration, h => by
    simp [Ty.canReturnEntity] at h
theorem anyEntity_sound : (ts : List Ty) → anyEntity ts = true → ∃ t ∈ ts, EntityLeaf t
  | [], h => by simp [anyEntity] at h
  | t :: ts, h => by
    simp only [anyEntity, Bool.or_eq_true] at h
    rcases h with h | h
    · exact ⟨t, by simp, canReturnEntity_sound t h⟩
    · obtain ⟨u, hu, he⟩ := anyEntity_sound ts h; exact ⟨u, by simp [hu], he⟩
end

theorem canReturnEntity_complete {t : Ty} (h : EntityLeaf t) : t.canReturnEntity = true := by
  induction h with
  | node | rel | path | any => rfl
  | optional _ ih => simpa [Ty.canReturnEntity] using ih
  | @union t ts hm _ ih =>
    simp only [Ty.canReturnEntity]
    induction ts with
    | nil => simp at hm
    | cons u us iu =>
      simp only [anyEntity, Bool.or_eq_true]
      rcases List.mem_cons.mp hm with rfl | hm'
      · exact Or.inl ih
      · exact Or.inr (iu hm')

/-- `List(Node)` is not an entity type (lists are not looked into). -/
example : (Ty.list .node).canReturnEntity = false := rfl

/-! ## `RuntimeFn` (mod.rs:425-461) -/

/-- Runtime/args/results are abstract. A procedure batch is its row list
(`empty_procedure_batch` = no rows). -/
structure Sig where
  Rt : Type
  V : Type
  Batch : Type
  rows : Batch → Nat
  empty : Batch
  empty_rows : rows empty = 0

variable (S : Sig)

/-- `empty_procedure_batch` (mod.rs:419): `BatchBuilder::new().finish()`. -/
def emptyProcedureBatch : S.Batch := S.empty

theorem emptyProcedureBatch_rows : S.rows (emptyProcedureBatch S) = 0 := S.empty_rows

inductive RuntimeFn where
  | native (f : S.Rt → List S.V → Except String S.V)
  | nativeProcBatch (f : S.Rt → List S.V → Nat → Except String S.Batch)
  | udf (name : String)

/-- `RuntimeFn::call` (mod.rs:436); `bridge` is `call_udf_bridge` (QuickJS, FFI). -/
def RuntimeFn.call (bridge : String → S.Rt → List S.V → Except String S.V) :
    RuntimeFn S → S.Rt → List S.V → Except String S.V
  | .native f, rt, args => f rt args
  | .nativeProcBatch _, _, _ => .error "Procedure runtime function cannot be called as scalar"
  | .udf name, rt, args => bridge name rt args

/-- `RuntimeFn::call_procedure_batch` (mod.rs:451). -/
def RuntimeFn.callProcBatch : RuntimeFn S → S.Rt → List S.V → Nat → Except String S.Batch
  | .nativeProcBatch f, rt, args, y => f rt args y
  | _, _, _, _ => .error "Function is not a procedure runtime function"

theorem call_arms (bridge : String → S.Rt → List S.V → Except String S.V) (rt : S.Rt) (a : List S.V)
    (f : S.Rt → List S.V → Except String S.V) (g : S.Rt → List S.V → Nat → Except String S.Batch)
    (n : String) (y : Nat) :
    (RuntimeFn.native f).call S bridge rt a = f rt a ∧
    (RuntimeFn.udf n).call S bridge rt a = bridge n rt a ∧
    (RuntimeFn.nativeProcBatch g).call S bridge rt a = .error "Procedure runtime function cannot be called as scalar" ∧
    (RuntimeFn.nativeProcBatch g).callProcBatch S rt a y = g rt a y ∧
    (RuntimeFn.native f).callProcBatch S rt a y = .error "Function is not a procedure runtime function" ∧
    (RuntimeFn.udf n : RuntimeFn S).callProcBatch S rt a y = .error "Function is not a procedure runtime function" :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-! ## `FnType`, `GraphFn` (mod.rs:463-898) -/

inductive FnType where
  | function | internal | procedure (cols : List String) | aggregation | udf

/-- `Debug for FnType` (mod.rs:500). -/
def FnType.dbg : FnType → String
  | .function => "Function" | .internal => "Internal" | .procedure _ => "Procedure"
  | .aggregation => "Aggregation" | .udf => "Udf"

/-- `PartialEq for FnType`: same constructor. -/
def FnType.tag : FnType → Nat
  | .function => 0 | .internal => 1 | .procedure _ => 2 | .aggregation => 3 | .udf => 4

theorem dbg_iff_tag (a b : FnType) : a.dbg = b.dbg ↔ a.tag = b.tag := by
  cases a <;> cases b <;> simp [FnType.dbg, FnType.tag]

inductive FnArgs where
  | fixed (ts : List Ty) | varLength (t : Ty)

structure GraphFn where
  name : String
  func : RuntimeFn S
  write : Bool
  nonDet : Bool
  hasPure : Bool
  pureArgs : List Ty
  hasStruct : Bool
  structSlots : List String
  args : FnArgs
  fnType : FnType
  ret : Ty

/-- `GraphFn::new` (mod.rs:715). -/
def GraphFn.new (name : String) (f : S.Rt → List S.V → Except String S.V) (write nonDet : Bool)
    (args : FnArgs) (ft : FnType) (ret : Ty) : GraphFn S :=
  ⟨name, .native f, write, nonDet, false, [], false, [], args, ft, ret⟩

/-- `GraphFn::new_procedure` (mod.rs:740). -/
def GraphFn.newProcedure (name : String) (f : S.Rt → List S.V → Nat → Except String S.Batch)
    (write nonDet : Bool) (args : FnArgs) (ft : FnType) (ret : Ty) : GraphFn S :=
  ⟨name, .nativeProcBatch f, write, nonDet, false, [], false, [], args, ft, ret⟩

/-- `GraphFn::new_udf` (mod.rs:765). -/
def GraphFn.newUdf (name : String) : GraphFn S :=
  ⟨name, .udf name, false, false, false, [], false, [], .varLength .any, .udf, .any⟩

theorem new_fields (n : String) f w d a ft r :
    let g := GraphFn.new S n f w d a ft r
    g.name = n ∧ g.write = w ∧ g.nonDet = d ∧ g.hasPure = false ∧ g.hasStruct = false ∧
      g.fnType = ft := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem newUdf_fields (n : String) :
    let g := GraphFn.newUdf S n
    g.name = n ∧ g.func = .udf n ∧ g.write = false ∧ g.fnType = .udf ∧ g.args = .varLength .any :=
  ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- `GraphFn::is_aggregate` (mod.rs:783). -/
def GraphFn.isAggregate (g : GraphFn S) : Bool := g.fnType.tag = 3

/-- `Debug for GraphFn` (mod.rs:698): `debug_struct` over six fields, non-exhaustive. -/
def GraphFn.dbg (dS : String → String) (dB : Bool → String) (dA : FnArgs → String)
    (dT : Ty → String) (g : GraphFn S) : String :=
  "GraphFn { name: " ++ dS g.name ++ ", write: " ++ dB g.write ++ ", non_deterministic: " ++
  dB g.nonDet ++ ", args_type: " ++ dA g.args ++ ", fn_type: " ++ g.fnType.dbg ++
  ", ret_type: " ++ dT g.ret ++ ", .. }"

/-- The Debug text depends only on the six listed fields (func/pure/struct are hidden). -/
theorem dbg_fields (dS dB dA dT) (g h : GraphFn S) (e1 : g.name = h.name) (e2 : g.write = h.write)
    (e3 : g.nonDet = h.nonDet) (e4 : g.args = h.args) (e5 : g.fnType.dbg = h.fnType.dbg)
    (e6 : g.ret = h.ret) : g.dbg S dS dB dA dT = h.dbg S dS dB dA dT := by
  simp [GraphFn.dbg, e1, e2, e3, e4, e5, e6]

/-- `GraphFn::call_procedure_batch` (mod.rs:894). -/
def GraphFn.callProcBatch (g : GraphFn S) (rt : S.Rt) (args : List S.V) (y : Nat) :
    Except String S.Batch :=
  match g.fnType with
  | .procedure _ => g.func.callProcBatch S rt args y
  | _ => .error s!"Function '{g.name}' is not a procedure"

/-- A procedure registered through `add_procedure` reaches its function. -/
theorem newProcedure_call (n : String) f w d a cols r rt args y :
    (GraphFn.newProcedure S n f w d a (.procedure cols) r).callProcBatch S rt args y = f rt args y := rfl

theorem callProcBatch_not_proc (g : GraphFn S) rt args y (h : g.fnType.tag ≠ 2) :
    g.callProcBatch S rt args y = .error s!"Function '{g.name}' is not a procedure" := by
  unfold GraphFn.callProcBatch; split <;> simp_all [FnType.tag]

/-! ## `Functions` (mod.rs:913-1088) -/

structure Lower where
  lower : String → String
  idem : ∀ s, lower (lower s) = lower s

variable (Lo : Lower)

/-- The `HashMap<String, Arc<GraphFn>>`, as an association list (keys distinct). -/
abbrev Fns := List (String × GraphFn S)

/-- `Functions::new` (mod.rs:918). -/
def Fns.new : Fns S := []

def has (m : Fns S) (k : String) : Bool := m.any (·.1 == k)

def lookup (m : Fns S) (k : String) : Option (GraphFn S) := (m.find? (·.1 == k)).map (·.2)

/-- `Functions::add` (mod.rs:923): assert absent (by lowered name), insert under the lowered
key, keep the original-case `name` in the entry. -/
def add (m : Fns S) (name : String) (f : S.Rt → List S.V → Except String S.V) (w d : Bool) (args : List Ty) (ft : FnType) (r : Ty) : Option (Fns S) :=
  if has S m (Lo.lower name) then none
  else some ((Lo.lower name, GraphFn.new S name f w d (.fixed args) ft r) :: m)

/-- `Functions::add_procedure` (mod.rs:951). -/
def addProcedure (m : Fns S) (name : String) (f : S.Rt → List S.V → Nat → Except String S.Batch) (w d : Bool) (args : List Ty) (ft : FnType) (r : Ty) : Option (Fns S) :=
  if has S m (Lo.lower name) then none
  else some ((Lo.lower name, GraphFn.newProcedure S name f w d (.fixed args) ft r) :: m)

/-- `Functions::add_var_len` (mod.rs:979): the entry's `name` is the LOWERED name. -/
def addVarLen (m : Fns S) (name : String) (f : S.Rt → List S.V → Except String S.V) (w d : Bool) (t : Ty) (ft : FnType) (r : Ty) : Option (Fns S) :=
  if has S m (Lo.lower name) then none
  else some ((Lo.lower name, GraphFn.new S (Lo.lower name) f w d (.varLength t) ft r) :: m)

theorem lookup_cons_self (m : Fns S) (k : String) (g : GraphFn S) : lookup S ((k, g) :: m) k = some g := by
  simp [lookup]

theorem lookup_cons_ne (m : Fns S) (k k' : String) (g : GraphFn S) (h : k ≠ k') :
    lookup S ((k, g) :: m) k' = lookup S m k' := by
  have : (k == k') = false := by simp [h]
  simp [lookup, List.find?, this]

theorem has_iff (m : Fns S) (k : String) : has S m k = true ↔ (lookup S m k).isSome := by
  simp [has, lookup, List.find?_isSome]

/-- `add` then lookup (case-insensitively) finds the new entry; other keys unchanged;
a second `add` of the same name (any case) panics. -/
theorem add_spec (m : Fns S) (name : String) (f : S.Rt → List S.V → Except String S.V) (w d : Bool) (args : List Ty) (ft : FnType) (r : Ty) (h : has S m (Lo.lower name) = false) :
    ∃ m', add S Lo m name f w d args ft r = some m' ∧
      lookup S m' (Lo.lower name) = some (GraphFn.new S name f w d (.fixed args) ft r) ∧
      (∀ k, k ≠ Lo.lower name → lookup S m' k = lookup S m k) ∧
      ∀ name2 (f2 : S.Rt → List S.V → Except String S.V) w2 d2 a2 ft2 r2, Lo.lower name2 = Lo.lower name →
        add S Lo m' name2 f2 w2 d2 a2 ft2 r2 = none := by
  refine ⟨_, by simp [add, h], lookup_cons_self S _ _ _, fun k hk => lookup_cons_ne S _ _ _ _ (Ne.symm hk), ?_⟩
  intro n2 f2 w2 d2 a2 ft2 r2 he
  simp [add, has, he]

theorem addProcedure_spec (m : Fns S) (name : String) (f : S.Rt → List S.V → Nat → Except String S.Batch) (w d : Bool) (args : List Ty) (ft : FnType) (r : Ty) (h : has S m (Lo.lower name) = false) :
    addProcedure S Lo m name f w d args ft r =
      some ((Lo.lower name, GraphFn.newProcedure S name f w d (.fixed args) ft r) :: m) := by
  simp [addProcedure, h]

/-- `add_var_len` stores the lowered name (`add` keeps the original spelling). -/
theorem addVarLen_name (m : Fns S) (name : String) (f : S.Rt → List S.V → Except String S.V) (w d : Bool) (t : Ty) (ft : FnType) (r : Ty) (h : has S m (Lo.lower name) = false) :
    ∃ m', addVarLen S Lo m name f w d t ft r = some m' ∧
      (lookup S m' (Lo.lower name)).map (·.name) = some (Lo.lower name) := by
  refine ⟨(Lo.lower name, GraphFn.new S (Lo.lower name) f w d (.varLength t) ft r) :: m,
    by simp [addVarLen, h], ?_⟩
  rw [lookup_cons_self]; rfl

/-- `set_pure_fn` (mod.rs:1009) / `set_struct_fn` (mod.rs:1031): panic unless registered;
update that entry only. (`Arc::get_mut` cannot fail at init: the registry is the sole owner.) -/
def update (m : Fns S) (k : String) (u : GraphFn S → GraphFn S) : Option (Fns S) :=
  if has S m k then some (m.map fun p => (p.1, if p.1 = k then u p.2 else p.2)) else none

def setPureFn (m : Fns S) (name : String) (args : List Ty) : Option (Fns S) :=
  update S m (Lo.lower name) fun g => { g with hasPure := true, pureArgs := args }

def setStructFn (m : Fns S) (name : String) (slots : List String) : Option (Fns S) :=
  update S m (Lo.lower name) fun g => { g with hasStruct := true, structSlots := slots }

theorem lookup_map_update (m : Fns S) (k k' : String) (u : GraphFn S → GraphFn S) :
    lookup S (m.map fun p => (p.1, if p.1 = k then u p.2 else p.2)) k' =
      if k' = k then (lookup S m k').map u else lookup S m k' := by
  induction m with
  | nil => simp [lookup]
  | cons p ps ih =>
    obtain ⟨a, g⟩ := p
    by_cases h2 : a = k'
    · subst h2
      by_cases h1 : a = k
      · subst h1; simp [lookup]
      · simp [lookup, h1, Ne.symm h1]
    · have hb : (a == k') = false := by simp [h2]
      simp only [lookup, List.map_cons, List.find?, hb] at ih ⊢
      exact ih

theorem update_spec (m m' : Fns S) (k : String) (u : GraphFn S → GraphFn S)
    (h : update S m k u = some m') :
    has S m k = true ∧ (lookup S m' k = (lookup S m k).map u) ∧
      ∀ k', k' ≠ k → lookup S m' k' = lookup S m k' := by
  unfold update at h; split at h
  · rename_i hh; cases h
    exact ⟨hh, by rw [lookup_map_update]; simp, fun k' hk => by rw [lookup_map_update]; simp [hk]⟩
  · cases h

theorem update_none (m : Fns S) (k : String) (u : GraphFn S → GraphFn S) :
    update S m k u = none ↔ has S m k = false := by
  unfold update; split <;> simp_all

theorem setStructFn_spec (m : Fns S) (name : String) (slots : List String) :
    (setStructFn S Lo m name slots = none ↔ has S m (Lo.lower name) = false) ∧
    ∀ m', setStructFn S Lo m name slots = some m' →
      (lookup S m' (Lo.lower name)).map (·.structSlots) = some slots ∧
      ∀ k, k ≠ Lo.lower name → lookup S m' k = lookup S m k := by
  refine ⟨update_none S _ _ _, fun m' h => ?_⟩
  obtain ⟨hh, h1, h2⟩ := update_spec S _ _ _ _ h
  refine ⟨?_, h2⟩
  rw [h1]
  have := (has_iff S m _).mp hh
  cases e : lookup S m (Lo.lower name) with
  | none => simp [e] at this
  | some g => rfl

theorem setPureFn_spec (m : Fns S) (name : String) (args : List Ty) :
    (setPureFn S Lo m name args = none ↔ has S m (Lo.lower name) = false) ∧
    ∀ m', setPureFn S Lo m name args = some m' →
      (lookup S m' (Lo.lower name)).map (·.hasPure) = some true ∧
      ∀ k, k ≠ Lo.lower name → lookup S m' k = lookup S m k := by
  refine ⟨update_none S _ _ _, fun m' h => ?_⟩
  obtain ⟨hh, h1, h2⟩ := update_spec S _ _ _ _ h
  refine ⟨?_, h2⟩
  rw [h1]
  have := (has_iff S m _).mp hh
  cases e : lookup S m (Lo.lower name) with
  | none => simp [e] at this
  | some g => rfl

/-- `Functions::iter` (mod.rs:1074): `functions.values()`. -/
def iter (m : Fns S) : List (GraphFn S) := m.map (·.2)

theorem iter_mem (m : Fns S) (g : GraphFn S) : g ∈ iter S m ↔ ∃ k, (k, g) ∈ m := by
  simp [iter]

/-! ## `init_functions`, `get_functions` (mod.rs:1103, :1164) -/

/-- One module's `register`: a sequence of `add`s (here: names; panics on a duplicate). -/
def registerAll (m : Fns S) : List (String × GraphFn S) → Option (Fns S)
  | [] => some m
  | (n, g) :: rest =>
    if has S m (Lo.lower n) then none else registerAll ((Lo.lower n, g) :: m) rest

/-- If the lowered names are distinct, registration never panics and the result holds
exactly those keys. -/
theorem registerAll_ok (entries : List (String × GraphFn S)) :
    ∀ m : Fns S, ((entries.map (Lo.lower ·.1)) ++ m.map (·.1)).Nodup →
      ∃ m', registerAll S Lo m entries = some m' ∧
        m'.map (·.1) = (entries.map (Lo.lower ·.1)).reverse ++ m.map (·.1) := by
  induction entries with
  | nil => intro m _; exact ⟨m, rfl, by simp⟩
  | cons e es ih =>
    intro m hnd
    obtain ⟨n, g⟩ := e
    simp only [List.map_cons, List.cons_append, List.nodup_cons, List.mem_append] at hnd
    obtain ⟨hn, hnd⟩ := hnd
    have hfree : has S m (Lo.lower n) = false := by
      simp only [has, List.any_eq_false, beq_iff_eq]
      intro p hp he; exact hn (Or.inr (List.mem_map.mpr ⟨p, hp, he⟩))
    have hnd' : (es.map (Lo.lower ·.1) ++ ((Lo.lower n, g) :: m).map (·.1)).Nodup := by
      rw [List.nodup_append] at hnd ⊢
      refine ⟨hnd.1, ?_, ?_⟩
      · simp only [List.map_cons, List.nodup_cons]
        exact ⟨fun h => hn (Or.inr h), hnd.2.1⟩
      · intro a ha b hb
        simp only [List.map_cons, List.mem_cons] at hb
        rcases hb with rfl | hb
        · intro e; subst e; exact hn (Or.inl ha)
        · exact hnd.2.2 a ha b hb
    obtain ⟨m', h1, h2⟩ := ih _ hnd'
    refine ⟨m', by simp [registerAll, hfree, h1], ?_⟩
    rw [h2]; simp

/-- `static FUNCTIONS: OnceLock<Functions>` and `init_functions` (mod.rs:1103):
`FUNCTIONS.set(funcs)` — `Err(funcs)` if already initialised (the first value stays). -/
def initFunctions (cell : Option (Fns S)) (built : Fns S) : Option (Fns S) × Except (Fns S) Unit :=
  match cell with
  | none => (some built, .ok ())
  | some old => (some old, .error built)

theorem initFunctions_once (built1 built2 : Fns S) :
    let (c1, r1) := initFunctions S none built1
    let (c2, r2) := initFunctions S c1 built2
    r1 = .ok () ∧ c2 = some built1 ∧ r2 = .error built2 := ⟨rfl, rfl, rfl⟩

/-- `get_functions` (mod.rs:1164): `expect` — `none` (panic) before init. -/
def getFunctions (cell : Option (Fns S)) : Option (Fns S) := cell

theorem getFunctions_after_init (b : Fns S) : getFunctions S (initFunctions S none b).1 = some b := rfl

/-! ## UDF registry (mod.rs:1091-1161) -/

structure UdfState where
  reg : Option (Fns S)       -- `UDF_FUNCTIONS` (`OnceLock`): `None` until `init_udf_functions`
  version : Nat              -- `UDF_VERSION`

/-- `udf_version` (mod.rs:1099): an atomic load. -/
def udfVersion (st : UdfState S) : Nat := st.version

/-- `init_udf_functions` (mod.rs:1124): set once; later calls are ignored. -/
def initUdf (st : UdfState S) : UdfState S :=
  match st.reg with
  | none => { st with reg := some [] }
  | some _ => st

/-- `register_udf` (mod.rs:1129): insert (replacing) under the lowered name, bump version. -/
def registerUdf (st : UdfState S) (name : String) (g : GraphFn S) : UdfState S :=
  match st.reg with
  | none => st
  | some m => ⟨some ((Lo.lower name, g) :: m.filter (·.1 ≠ Lo.lower name)), st.version + 1⟩

/-- `unregister_udf` (mod.rs:1140). -/
def unregisterUdf (st : UdfState S) (name : String) : UdfState S :=
  match st.reg with
  | none => st
  | some m => ⟨some (m.filter (·.1 ≠ Lo.lower name)), st.version + 1⟩

/-- `flush_udfs` (mod.rs:1148). -/
def flushUdfs (st : UdfState S) : UdfState S :=
  match st.reg with
  | none => st
  | some _ => ⟨some [], st.version + 1⟩

/-- `get_udf_functions` (mod.rs:1157): snapshot of the values, empty before init. -/
def getUdfFunctions (st : UdfState S) : List (GraphFn S) :=
  match st.reg with
  | none => []
  | some m => iter S m

theorem initUdf_idem (st : UdfState S) : initUdf S (initUdf S st) = initUdf S st := by
  unfold initUdf; cases h : st.reg <;> simp [h]

theorem registerUdf_spec (st : UdfState S) (m : Fns S) (h : st.reg = some m) (n : String) (g : GraphFn S) :
    let st' := registerUdf S Lo st n g
    st'.version = st.version + 1 ∧ g ∈ getUdfFunctions S st' ∧
      (st'.reg.bind fun m => lookup S m (Lo.lower n)) = some g := by
  simp [registerUdf, h, getUdfFunctions, iter, lookup]

theorem unregisterUdf_spec (st : UdfState S) (m : Fns S) (h : st.reg = some m) (n : String) :
    let st' := unregisterUdf S Lo st n
    st'.version = st.version + 1 ∧ (st'.reg.bind fun m => lookup S m (Lo.lower n)) = none := by
  simp only [unregisterUdf, h, Option.bind_some, true_and]
  simp only [lookup, Option.map_eq_none_iff, List.find?_eq_none]
  intro p hp; simp at hp; simpa using hp.2

theorem flushUdfs_spec (st : UdfState S) (m : Fns S) (h : st.reg = some m) :
    getUdfFunctions S (flushUdfs S st) = [] ∧ (flushUdfs S st).version = st.version + 1 := by
  simp [flushUdfs, h, getUdfFunctions, iter]

/-- Before `init_udf_functions` every mutation is a no-op and does NOT bump the version. -/
theorem udf_uninit_noop (st : UdfState S) (h : st.reg = none) (n : String) (g : GraphFn S) :
    registerUdf S Lo st n g = st ∧ unregisterUdf S Lo st n = st ∧ flushUdfs S st = st ∧
    getUdfFunctions S st = [] := by
  simp [registerUdf, unregisterUdf, flushUdfs, getUdfFunctions, h]

/-- The version never decreases, and strictly increases on every mutation after init
(plan-cache invalidation relies on this; u64 wrap needs 2^64 mutations). -/
inductive Op where
  | reg (n : String) (g : GraphFn S) | unreg (n : String) | flush

def Op.apply (st : UdfState S) : Op S → UdfState S
  | .reg n g => registerUdf S Lo st n g
  | .unreg n => unregisterUdf S Lo st n
  | .flush => flushUdfs S st

theorem version_mono (st : UdfState S) (o : Op S) :
    st.version ≤ (o.apply S Lo st).version ∧
    (st.reg.isSome → (o.apply S Lo st).version = st.version + 1) := by
  cases o <;> simp only [Op.apply, registerUdf, unregisterUdf, flushUdfs] <;>
    cases st.reg <;> simp

end ValueMath.Reg
