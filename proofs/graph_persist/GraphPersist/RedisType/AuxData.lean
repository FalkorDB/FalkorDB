import GraphPersist.Codec
/-! # `src/redis_type.rs`: aux fields, persistence events, fork hook, `graphmeta` stubs

Source: origin/main `3fec7d7c9`.

| here | there |
| --- | --- |
| `It`, `loadU`, `loadBuf` | the RDB aux stream: `save_unsigned`/`save_string`, `load_unsigned`/`load_string_buffer` |
| `auxSave`               | `graph_aux_save` (:265-284) |
| `auxLoad`               | `graph_aux_load` (:286-335) |
| `metaAuxLoad`           | `graphmeta_aux_load` (:896-907) |
| `Sub`, `onPersistence`  | `on_persistence` (:368-397) |
| `preFork`               | `pre_fork_prepare` (:350-361) |
| `metaFree`              | `graphmeta_free` (:887-894); `graphmeta_rdb_save` is `VKeys.metaSave` since #3161 |

Names and scripts are byte strings; `String::from_utf8_lossy` is the identity on the UTF-8
the module itself writes (a Rust `String`), so it is not modelled further.
-/
namespace GraphPersist.RedisType
open GraphPersist

/-- One item of a module's aux payload. -/
inductive It where
  | u (n : Nat)
  | buf (b : List UInt8)
  deriving DecidableEq, Repr

/-- `load_unsigned`: `Err` on a short read or a non-integer item. -/
def loadU : List It → Option (Nat × List It)
  | .u n :: r => some (n, r)
  | _ => none

/-- `load_string_buffer`. -/
def loadBuf : List It → Option (List UInt8 × List It)
  | .buf b :: r => some (b, r)
  | _ => none

/-- `REDISMODULE_AUX_BEFORE_RDB` / `AFTER_RDB`. -/
inductive When where
  | before
  | after
  deriving DecidableEq, Repr

/-- `strip_trailing_nul` (:259): one trailing NUL. -/
def strip (b : List UInt8) : List UInt8 := stripNul b

/-- `graph_aux_save` (:265): before the keys, the UDF libraries as `count, (name, code)*`,
each NUL-terminated (`save_string_nul`, :241); after the keys, a `0` placeholder. -/
def auxSave (w : When) (libs : List (List UInt8 × List UInt8)) : List It :=
  match w with
  | .before => .u libs.length :: libs.flatMap fun (n, c) => [.buf (nulTerm n), .buf (nulTerm c)]
  | .after => [.u 0]

/-- The library loop of `graph_aux_load` (:295-307). -/
def readLibs : Nat → List It → Option (List (List UInt8 × List UInt8) × List It)
  | 0, r => some ([], r)
  | k + 1, r =>
    match loadBuf r with
    | none => none
    | some (n, r) =>
      match loadBuf r with
      | none => none
      | some (c, r) =>
        match readLibs k r with
        | none => none
        | some (ls, r) => some ((strip n, strip c) :: ls, r)

/-- What `graph_aux_load` leaves behind: the return code, the libraries now registered
(`flush_udfs` then `register_udf` per function, `none` = registry untouched), and whether
it ran `finalize_pending_graphs`. `deser` is `UdfRepo::deserialize` (an input). -/
structure AuxOut (Lib : Type) where
  ret : Nat
  registered : Option (List Lib)
  finalized : Bool
  rest : List It

/-- `graph_aux_load` (:286). -/
def auxLoad {Lib : Type} (deser : List (List UInt8 × List UInt8) → Option (List Lib)) (w : When) (s : List It) :
    AuxOut Lib :=
  match w with
  | .before =>
    match loadU s with
    | none => ⟨1, none, false, s⟩
    | some (count, r) =>
      match readLibs count r with
      | none => ⟨1, none, false, r⟩
      | some (libs, r) =>
        match deser libs with
        | none => ⟨1, none, false, r⟩
        | some loaded => ⟨0, some loaded, false, r⟩
  | .after => ⟨0, none, true, ((loadU s).map Prod.snd).getD s⟩

theorem readLibs_save (libs : List (List UInt8 × List UInt8)) (r : List It) :
    readLibs libs.length ((libs.flatMap fun (n, c) => [It.buf (nulTerm n), It.buf (nulTerm c)]) ++ r) = some (libs, r) := by
  induction libs with
  | nil => rfl
  | cons l ls ih =>
    obtain ⟨n, c⟩ := l
    simp [readLibs, loadBuf, ih, strip, stripNul_nulTerm]

/-- **UDF libraries round-trip through the RDB.** The libraries read back are exactly the
ones saved (names and scripts byte for byte, one NUL stripped each — so a name that itself
ends in NUL survives), the stream is left right after them, and the outcome is whatever
`deserialize` says: on success the registry is flushed and replaced, return 0. -/
theorem aux_roundtrip {Lib : Type} (deser : List (List UInt8 × List UInt8) → Option (List Lib))
    (libs : List (List UInt8 × List UInt8)) (r : List It) :
    auxLoad deser .before (auxSave .before libs ++ r) =
      match deser libs with
      | none => ⟨1, none, false, r⟩
      | some loaded => ⟨0, some loaded, false, r⟩ := by
  simp only [auxLoad, auxSave, List.cons_append, loadU, readLibs_save]

/-- A name ending in NUL keeps it: exactly one NUL is stripped. -/
theorem strip_keeps_inner_nul (b : List UInt8) : strip (nulTerm (b ++ [0])) = b ++ [0] :=
  stripNul_nulTerm _

/-- After the keys: the placeholder is consumed and pending multi-key graphs are finalized. -/
theorem aux_after {Lib : Type} (deser : List (List UInt8 × List UInt8) → Option (List Lib)) (r : List It) :
    auxLoad deser .after (auxSave .after [] ++ r) = ⟨0, none, true, r⟩ := rfl

/-- A truncated library list fails the load (return 1) and registers nothing. -/
theorem aux_truncated {Lib : Type} (deser : List (List UInt8 × List UInt8) → Option (List Lib)) (n : List UInt8) :
    (auxLoad deser .before [.u 1, .buf (nulTerm n)]).ret = 1 ∧
    (auxLoad deser .before [.u 1, .buf (nulTerm n)]).registered = none := by
  simp [auxLoad, loadU, readLibs, loadBuf]

/-- `graphmeta_aux_load` (:896): one placeholder read, finalize after the keys, always 0. -/
def metaAuxLoad (w : When) (s : List It) : Nat × Bool × List It :=
  (0, w == .after, ((loadU s).map Prod.snd).getD s)

theorem metaAuxLoad_spec (w : When) (n : Nat) (r : List It) :
    metaAuxLoad w (.u n :: r) = (0, decide (w = .after), r) := by
  cases w <;> simp [metaAuxLoad, loadU]

/-! ## Persistence events -/

/-- The `REDISMODULE_SUBEVENT_PERSISTENCE_*` cases `on_persistence` distinguishes. -/
inductive Sub where
  | rdbStart       -- BGSAVE fork child
  | syncRdbStart   -- SAVE / DEBUG RELOAD
  | ended
  | failed
  | other
  deriving DecidableEq, Repr

inductive Act where
  | createVKeys
  | deleteVKeys
  | nothing
  deriving DecidableEq, Repr

/-- `on_persistence` (:368). -/
def onPersistence : Sub → Act
  | .syncRdbStart => .createVKeys
  | .rdbStart => .nothing
  | .ended | .failed => .deleteVKeys
  | .other => .nothing

/-- Virtual keys exist only around a synchronous save: a BGSAVE child never allocates them
(so a BGSAVE always writes every graph as a single key), and every end of a save — success
or failure — removes them. -/
theorem onPersistence_spec (s : Sub) :
    (onPersistence s = .createVKeys ↔ s = .syncRdbStart) ∧
    (onPersistence s = .deleteVKeys ↔ s = .ended ∨ s = .failed) := by
  cases s <;> simp [onPersistence]

/-- `pre_fork_prepare` (:350): only on the main thread, wait on every registered graph. -/
def preFork {Gr : Type} (isMain : Bool) (registry : List Gr) : List Gr :=
  if isMain then registry else []

theorem preFork_spec {Gr : Type} (isMain : Bool) (registry : List Gr) :
    (isMain = true → preFork isMain registry = registry) ∧ (isMain = false → preFork isMain registry = []) := by
  cases isMain <;> simp [preFork]

/-- `graphmeta_free` (:886): frees the dummy byte iff the pointer is non-null; returns the
number of frees. -/
def metaFree (p : Option Unit) : Nat := if p.isSome then 1 else 0

theorem metaFree_spec : metaFree none = 0 ∧ metaFree (some ()) = 1 := ⟨rfl, rfl⟩

end GraphPersist.RedisType
