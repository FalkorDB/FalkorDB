/-! # `wrong_type_compat` (redis_type.rs:897)

valkey-module words the error for a key that holds another type "Existing key has wrong
Valkey type" (`verify_type`, valkeymodule-rs `key.rs:678`, reached from `get_value` and
`set_value`); clients have always seen redis-module's "Existing key has wrong Redis type".
Every `get_value`/`set_value` of `GRAPH_TYPE` in `src/commands/` maps its error through
this function.

* `wrongTypeCompat_valkey` — the Valkey wording becomes the Redis one.
* `wrongTypeCompat_str_other`, `wrongTypeCompat_string`, `wrongTypeCompat_other` — every
  other error passes through untouched; only a `ValkeyError::Str` is rewritten, which is
  the arm `verify_type` produces.
* `wrongTypeCompat_idem` — mapping twice is mapping once.
-/
namespace GraphPersist.RedisType

/-- The `ValkeyError` arms `wrong_type_compat` tells apart: `Str` (a static message),
`String` (an owned one), and the rest. -/
inductive VErr | str (s : String) | string (s : String) | other
  deriving DecidableEq, Repr

def wrongTypeValkey : String := "Existing key has wrong Valkey type"
def wrongTypeRedis : String := "Existing key has wrong Redis type"

/-- `wrong_type_compat`. -/
def wrongTypeCompat : VErr → VErr
  | .str s => if s = wrongTypeValkey then .str wrongTypeRedis else .str s
  | e => e

theorem wrongTypeCompat_valkey : wrongTypeCompat (.str wrongTypeValkey) = .str wrongTypeRedis := by
  simp [wrongTypeCompat]
theorem wrongTypeCompat_str_other (s : String) (h : s ≠ wrongTypeValkey) :
    wrongTypeCompat (.str s) = .str s := by
  simp [wrongTypeCompat, h]
theorem wrongTypeCompat_string (s : String) : wrongTypeCompat (.string s) = .string s := rfl
theorem wrongTypeCompat_other : wrongTypeCompat .other = .other := rfl
theorem wrongTypeCompat_idem (e : VErr) : wrongTypeCompat (wrongTypeCompat e) = wrongTypeCompat e := by
  cases e with
  | str s =>
    by_cases h : s = wrongTypeValkey
    · subst h; simp [wrongTypeCompat, wrongTypeValkey, wrongTypeRedis]
    · simp [wrongTypeCompat, h]
  | string s => rfl
  | other => rfl

end GraphPersist.RedisType
