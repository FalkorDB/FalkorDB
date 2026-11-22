/-! # Logical databases (redis_type.rs, #3424)

| here | there |
| --- | --- |
| `ioDb`        | `io_db`: `RedisModule_GetDbIdFromIO` (FFI); its result is the `db` input of the load and save models |
| `selectDb`    | `RedisModule_SelectDb(ctx, d)` (FFI): `ok d` = it returns `REDISMODULE_OK`; on failure the selection is unchanged |
| `restoreDrop` | `RestoreDb::drop`: select the database saved when the guard was made |
| `inDb`        | `in_db`: save the selected db, select `db`, run `f`, restore; `None` when `db` does not exist |
| `forEachLoop`, `forEachDb` | `for_each_db`: select db `0, 1, …` until `SelectDb` fails, running `f` in each one `RedisModule_DbSize` reports non-empty, then restore |

The context's selected database is the state `sel`; a callback `f` is a function of the
database selected when it runs (that is the only part of the context it reads).
`for_each_db`'s `while` loop is modelled with fuel: Redis has finitely many databases
(`databases` in redis.conf), and `forEachDb_spec` holds for any fuel at least that number.
The db counter is an `i32` starting at 0; it stays below that number, far from overflow.

* `inDb_spec` — `in_db` runs `f` in `db` iff `db` exists, and the selection is restored.
* `forEachDb_spec` — `for_each_db` runs `f` exactly in the non-empty databases `0..N`, in
  order, and the selection is restored.
-/
namespace GraphPersist.RedisType

/-- `io_db`: the database of the key an RDB callback runs for (`RedisModule_GetDbIdFromIO`,
FFI; its answer is the input). -/
def ioDb (dbOfIO : Int) : Int := dbOfIO

theorem ioDb_spec (d : Int) : ioDb d = d := rfl

/-- `RedisModule_SelectDb(ctx, d)`: the new selection and whether it returned `REDISMODULE_OK`. -/
def selectDb (ok : Int → Bool) (sel d : Int) : Int × Bool :=
  if ok d then (d, true) else (sel, false)

/-- `RestoreDb::drop`: `RedisModule_SelectDb(self.ctx, self.db)`. -/
def restoreDrop (ok : Int → Bool) (sel saved : Int) : Int := (selectDb ok sel saved).1

/-- The guard restores the database it saved whenever that database exists — always, as it
was the selected one when the guard was made. -/
theorem restoreDrop_spec (ok : Int → Bool) (sel saved : Int) (h : ok saved = true) :
    restoreDrop ok sel saved = saved := by simp [restoreDrop, selectDb, h]

/-- `in_db(ctx, db, f)`: returns `f`'s result (or `None`) and the selection afterwards. -/
def inDb {R : Type} (ok : Int → Bool) (sel db : Int) (f : Int → R) : Option R × Int :=
  let saved := sel                       -- `RedisModule_GetSelectedDb`
  let (sel1, r) := selectDb ok sel db
  if r then (some (f sel1), restoreDrop ok sel1 saved)
  else (none, restoreDrop ok sel1 saved)

theorem inDb_fst {R : Type} (ok : Int → Bool) (sel db : Int) (f : Int → R) :
    (inDb ok sel db f).1 = if ok db then some (f db) else none := by
  by_cases h : ok db <;> simp [inDb, selectDb, h]

/-- **`in_db` runs `f` with `db` selected iff `db` exists, and leaves the selection as it
found it.** -/
theorem inDb_spec {R : Type} (ok : Int → Bool) (sel db : Int) (f : Int → R) (hsel : ok sel = true) :
    inDb ok sel db f = (if ok db then some (f db) else none, sel) := by
  by_cases h : ok db <;> simp [inDb, selectDb, restoreDrop, h, hsel]

/-- `for_each_db`'s loop from db `d`, `cur` selected: the outputs of `f` in order, and the
selection at the end. -/
def forEachLoop {β : Type} (ok nonEmpty : Int → Bool) (f : Int → List β) :
    Nat → Nat → Int → List β × Int
  | 0, _, cur => ([], cur)
  | fuel + 1, d, cur =>
    let (c1, r) := selectDb ok cur d
    if r then
      let (l, c2) := forEachLoop ok nonEmpty f fuel (d + 1) c1
      ((if nonEmpty d then f d else []) ++ l, c2)
    else ([], c1)

/-- `for_each_db(ctx, f)`. -/
def forEachDb {β : Type} (ok nonEmpty : Int → Bool) (f : Int → List β) (fuel : Nat) (sel : Int) :
    List β × Int :=
  let saved := sel
  let (l, c) := forEachLoop ok nonEmpty f fuel 0 sel
  (l, restoreDrop ok c saved)

theorem forEachLoop_spec {β : Type} (ok nonEmpty : Int → Bool) (f : Int → List β) (N : Nat)
    (hN : ∀ d : Int, ok d = true ↔ 0 ≤ d ∧ d < N) :
    ∀ fuel d cur, N ≤ d + fuel →
      (forEachLoop ok nonEmpty f fuel d cur).1 =
        (List.range' d (N - d)).flatMap fun (i : Nat) => if nonEmpty i then f i else [] := by
  intro fuel
  induction fuel with
  | zero => intro d cur h; simp [forEachLoop, show N - d = 0 by omega]
  | succ n ih =>
    intro d cur h
    by_cases hd : d < N
    · have hok : ok d = true := (hN d).2 ⟨by omega, by omega⟩
      have e : N - d = (N - (d + 1)) + 1 := by omega
      simp only [forEachLoop, selectDb, hok, ite_true]
      rw [e, List.range'_succ, List.flatMap_cons, ih (d + 1) (d : Int) (by omega)]
    · have hok : ok d = false := by
        cases hc : ok d
        · rfl
        · have := (hN d).1 hc; omega
      simp [forEachLoop, selectDb, hok, show N - d = 0 by omega]

/-- **`for_each_db` runs `f` in exactly the non-empty databases `0..N` (`N` = the number of
databases), in order, and restores the selection.** -/
theorem forEachDb_spec {β : Type} (ok nonEmpty : Int → Bool) (f : Int → List β) (N fuel : Nat) (sel : Int)
    (hN : ∀ d : Int, ok d = true ↔ 0 ≤ d ∧ d < N) (hfuel : N ≤ fuel) (hsel : ok sel = true) :
    forEachDb ok nonEmpty f fuel sel =
      ((List.range N).flatMap fun (i : Nat) => if nonEmpty i then f i else [], sel) := by
  have h1 := forEachLoop_spec ok nonEmpty f N hN fuel 0 sel (by omega)
  simp only [Nat.sub_zero, ← List.range_eq_range'] at h1
  simp only [forEachDb]
  rw [← h1, restoreDrop_spec ok _ sel hsel]

end GraphPersist.RedisType
