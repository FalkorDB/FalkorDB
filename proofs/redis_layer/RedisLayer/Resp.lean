import RedisLayer.Compact
/-!
# RESP reply streams (`src/reply.rs`)

The encoders do not build reply trees: they make a *sequence* of `RedisModule_Reply*`
calls, some with postponed lengths (`REDISMODULE_POSTPONED_LEN` + `RM_ReplySetArrayLength`).
`Call` is that sequence; `run` is the Redis side of the contract — how a call sequence
becomes reply trees (a fixed-length array closes after its n-th element; a postponed array
closes at the `SetArrayLength` that names its exact element count; anything else is a
protocol error, `none`). This is the only Redis-module fact used, stated as a definition
(AXIOMATISED rows: `RedisModule_ReplyWith*`, `RedisModule_ReplySetArrayLength`).
`resp` is RESP2 framing, so `bytes cs = (run cs).map resp` is the byte stream.

`RedisModule_ReplyWithDouble` in RESP2 is a bulk string of Redis's own double formatting
(`dfmt`, abstract).
-/
namespace RedisLayer.Resp
open RedisLayer.Compact (R)

inductive Call (F : Type) where
  | ll (i : Int)            -- RedisModule_ReplyWithLongLong
  | str (s : String)        -- RedisModule_ReplyWithStringBuffer
  | null                    -- RedisModule_ReplyWithNull
  | arr (n : Nat)           -- RedisModule_ReplyWithArray(n)
  | arrP                    -- RedisModule_ReplyWithArray(REDISMODULE_POSTPONED_LEN)
  | setLen (n : Nat)        -- RedisModule_ReplySetArrayLength(n)
  | dbl (x : F)             -- RedisModule_ReplyWithDouble
  | err (s : String)        -- RedisModule_ReplyWithError

/-- Reply trees: `Compact.R` plus errors. -/
inductive T where
  | r (x : R)
  | err (s : String)

/-- An open array: `some n` fixed length, `none` postponed. -/
abbrev Frame := Option Nat × List R

structure St where
  stack : List Frame
  out : List T

variable {F : Type} (dfmt : F → String)

/-- Add a completed element to the innermost open array (closing it, and possibly its
parents, when full) or to the top-level output. -/
def addLeaf (x : R) : List Frame → List T → St
  | [], out => ⟨[], out ++ [.r x]⟩
  | (some n, acc) :: st, out =>
    if acc.length + 1 = n then addLeaf (.arr (acc ++ [x])) st out
    else ⟨(some n, acc ++ [x]) :: st, out⟩
  | (none, acc) :: st, out => ⟨(none, acc ++ [x]) :: st, out⟩

def step (s : St) : Call F → Option St
  | .ll i => some (addLeaf (.int i) s.stack s.out)
  | .str t => some (addLeaf (.bulk t) s.stack s.out)
  | .null => some (addLeaf .nil s.stack s.out)
  | .dbl x => some (addLeaf (.bulk (dfmt x)) s.stack s.out)
  | .arr 0 => some (addLeaf (.arr []) s.stack s.out)
  | .arr (n+1) => some ⟨(some (n+1), []) :: s.stack, s.out⟩
  | .arrP => some ⟨(none, []) :: s.stack, s.out⟩
  | .setLen n => match s.stack with
    | (none, acc) :: st => if acc.length = n then some (addLeaf (.arr acc) st s.out) else none
    | _ => none
  | .err t => match s.stack with
    | [] => some ⟨[], s.out ++ [.err t]⟩
    | _ => none     -- errors are only ever sent at top level here

def run : List (Call F) → St → Option St
  | [], s => some s
  | c :: cs, s => (step dfmt s c).bind (run cs)

theorem run_append (a b : List (Call F)) (s : St) :
    run dfmt (a ++ b) s = (run dfmt a s).bind (run dfmt b) := by
  induction a generalizing s with
  | nil => simp [run]
  | cons c cs ih =>
    simp only [run, List.cons_append]
    cases step dfmt s c <;> simp [ih]

/-- Leaves added in sequence. -/
def addLeaves : List R → St → St
  | [], s => s
  | x :: xs, s => addLeaves xs (addLeaf x s.stack s.out)

theorem addLeaves_append (xs ys : List R) (s : St) :
    addLeaves ys (addLeaves xs s) = addLeaves (xs ++ ys) s := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih => exact ih _

theorem addLeaves_post (ys acc : List R) (st : List Frame) (out : List T) :
    addLeaves ys ⟨(none, acc) :: st, out⟩ = ⟨(none, acc ++ ys) :: st, out⟩ := by
  induction ys generalizing acc with
  | nil => simp [addLeaves]
  | cons y ys ih => simp [addLeaves, addLeaf, ih]

theorem addLeaves_top (xs : List R) (out : List T) :
    addLeaves xs ⟨[], out⟩ = ⟨[], out ++ xs.map .r⟩ := by
  induction xs generalizing out with
  | nil => simp [addLeaves]
  | cons x xs ih => simp [addLeaves, addLeaf, ih]

/-- A call sequence that adds exactly the leaves `xs` in any context. -/
def Emits (cs : List (Call F)) (xs : List R) : Prop :=
  ∀ s, run dfmt cs s = some (addLeaves xs s)

theorem Emits.nil : Emits dfmt ([] : List (Call F)) [] := fun _ => rfl

theorem Emits.append {a b : List (Call F)} {xs ys : List R} (ha : Emits dfmt a xs)
    (hb : Emits dfmt b ys) : Emits dfmt (a ++ b) (xs ++ ys) := by
  intro s
  rw [run_append, ha s]
  simp only [Option.bind_some]
  rw [hb, addLeaves_append]

theorem Emits.leaf_ll (i : Int) : Emits dfmt [Call.ll (F := F) i] [.int i] := fun _ => rfl
theorem Emits.leaf_str (t : String) : Emits dfmt [Call.str (F := F) t] [.bulk t] := fun _ => rfl
theorem Emits.leaf_null : Emits dfmt [Call.null (F := F)] [.nil] := fun _ => rfl
theorem Emits.leaf_dbl (x : F) : Emits dfmt [Call.dbl x] [.bulk (dfmt x)] := fun _ => rfl

/-- Adding `ys` to an open fixed array that still has room for all of them. -/
theorem addLeaves_frame (n : Nat) (acc ys : List R) (st : List Frame) (out : List T)
    (h : acc.length + ys.length = n) (hy : ys ≠ []) :
    addLeaves ys ⟨(some n, acc) :: st, out⟩ = addLeaf (.arr (acc ++ ys)) st out := by
  induction ys generalizing acc with
  | nil => exact absurd rfl hy
  | cons y ys ih =>
    simp only [addLeaves, addLeaf]
    by_cases hl : acc.length + 1 = n
    · have : ys = [] := by simp at h; cases ys <;> simp_all; omega
      subst this; simp [hl, addLeaves]
    · simp only [hl, ite_false]
      have hy' : ys ≠ [] := by intro e; subst e; simp at h; omega
      rw [ih (acc ++ [y]) (by simp at h ⊢; omega) hy']
      simp

/-- `RedisModule_ReplyWithArray(n)` followed by calls emitting exactly `n` elements
emits one array. -/
theorem Emits.array {cs : List (Call F)} {ys : List R} (h : Emits dfmt cs ys) :
    Emits dfmt (Call.arr ys.length :: cs) [.arr ys] := by
  intro s
  cases ys with
  | nil =>
    simp only [run, step, List.length_nil, Option.bind_some]
    rw [h]; rfl
  | cons y ys =>
    simp only [run, step, List.length_cons, Option.bind_some]
    rw [h]
    simp only [addLeaves]
    rw [show addLeaves ys (addLeaf y ((some (ys.length+1), []) :: s.stack) s.out)
        = addLeaves (y :: ys) ⟨(some (ys.length+1), []) :: s.stack, s.out⟩ from rfl]
    rw [addLeaves_frame _ [] _ _ _ (by simp) (by simp)]
    simp

/-- A postponed array closed by `SetArrayLength(k)` where `k` is the number of elements
emitted in between (`reply.rs:219-228`, `:435-457`, `:505-516`). -/
theorem Emits.postponed {cs : List (Call F)} {ys : List R} (h : Emits dfmt cs ys) :
    Emits dfmt (Call.arrP :: cs ++ [Call.setLen ys.length]) [.arr ys] := by
  intro s
  have key := addLeaves_post ys
  simp only [List.cons_append, run, step, Option.bind_some]
  rw [run_append, h]
  simp only [Option.bind_some]
  rw [key]
  simp [run, step, addLeaves]

/-- If the count passed to `SetArrayLength` is wrong the stream is broken. -/
theorem postponed_wrong_len {cs : List (Call F)} {ys : List R} (h : Emits dfmt cs ys) (k : Nat)
    (hk : k ≠ ys.length) (s : St) :
    run dfmt (Call.arrP :: cs ++ [Call.setLen k]) s = none := by
  have key := addLeaves_post ys
  simp only [List.cons_append, run, step, Option.bind_some]
  rw [run_append, h]
  simp only [Option.bind_some]
  rw [key]
  simp [run, step]; omega

/-- A whole reply: from an empty Redis client buffer, exactly the trees `xs`. -/
theorem Emits.reply {cs : List (Call F)} {xs : List R} (h : Emits dfmt cs xs) :
    run dfmt cs ⟨[], []⟩ = some ⟨[], xs.map .r⟩ := by
  rw [h, addLeaves_top]; simp

/-! ## RESP2 framing -/

def crlf : String := "\r\n"

mutual
def resp : R → String
  | .int i => ":" ++ toString i ++ crlf
  | .bulk s => "$" ++ toString s.utf8ByteSize ++ crlf ++ s ++ crlf
  | .nil => "$-1" ++ crlf
  | .arr xs => "*" ++ toString xs.length ++ crlf ++ respList xs
def respList : List R → String
  | [] => ""
  | x :: xs => resp x ++ respList xs
end

def respT : T → String
  | .r x => resp x
  | .err s => "-" ++ s ++ crlf

/-- The bytes Redis writes to the client for a call sequence (`none`: protocol error). -/
def bytes (cs : List (Call F)) : Option String :=
  (run dfmt cs ⟨[], []⟩).bind fun s =>
    if s.stack = [] then some (String.join (s.out.map respT)) else none

theorem bytes_of_emits {cs : List (Call F)} {x : R} (h : Emits dfmt cs [x]) :
    bytes dfmt cs = some (resp x) := by
  simp [bytes, Emits.reply dfmt h, respT]

end RedisLayer.Resp
