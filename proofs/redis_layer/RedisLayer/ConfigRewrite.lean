import RedisLayer.ModuleInit
import RedisLayer.LoadArgs
/-!
# `CONFIG REWRITE` and the native `graph.*` configs (#3419)

Every integer config in `ModuleInit.table` (`lib.rs:97-116`) is a Redis module numeric
config backed by the same `RedisGILGuard` that `GRAPH.CONFIG SET` (`config_cmd.rs:286-288`)
and the module-load arguments (`config.rs:252-283`) write. `CONFIG REWRITE` therefore saves
`graph.<NAME> <value>` for whatever those paths stored, and on the next start Redis applies
that line through the table's range; a value outside it unloads the module and aborts the
server (#3413).

| here | there |
| --- | --- |
| `nativeLoad` | Redis `numericParseString` (`string2ll`, no flags) + `numericBoundaryCheck` (inclusive) for a module numeric config |
| `rewriteVal` | `rewriteConfigNumericalOption`: the stored `long long` as `%lld` |
| `ModuleInit.table` | `redis_module!` `configurations:` (`lib.rs:97-116`) |
| `Config.validate .vkeyMax` | `GRAPH.CONFIG SET` (`config_cmd.rs:164-172`) |
| `LoadArgs.applyModuleArgs` | the module-load arguments (`config.rs:296`) |

**Fixed by #3419 (`8621212fc`, closes #3413):** `VKEY_MAX_ENTITY_COUNT` was registered with
range `1..1073741824` while both other paths accept every `0..i64::MAX`
(`pre3419_rewrite_refused`: FalkorDB Cloud's `4611686000000000000`, and `0`). Now
`vkey_rewrite_reloads_config` / `vkey_rewrite_reloads_load`: any value either path accepts
is written by `CONFIG REWRITE` as a line the next start accepts, reading back the same value.

**Same class, still open:** the other integer entries keep ranges narrower than what the
load arguments or `GRAPH.CONFIG SET` accept (`rewrite_reload_counterexamples`):
`CACHE_SIZE` (load: any positive; table `≤ 1000`), `THREAD_COUNT` (`≤ 1024`),
`INDEX_WORKER_THREADS` (`≤ 1024`), `NODE_CREATION_BUFFER` (`≤ 2^30`), `JS_HEAP_SIZE` and
`JS_STACK_SIZE` (load and `GRAPH.CONFIG SET`: any `≥ 1 MB`; table `≤ 2^32`).
-/
namespace RedisLayer.ConfigRewrite
open RedisLayer RedisLayer.ModuleInit RedisLayer.LoadArgs

/-- The value Redis applies from a `graph.<name> <v>` line, or `none` (load aborted). -/
def nativeLoadIn (tbl : List (String × Kind)) (name : String) (v : List Nat) : Option Int :=
  match tbl.lookup name with
  | some (.i64 _ lo hi) => (string2ll v).filter (fun x => decide (lo ≤ x ∧ x ≤ hi))
  | _ => none

def nativeLoad := nativeLoadIn table

/-- `%lld` of the stored value. -/
def rewriteVal (x : Int) : List Nat := if x < 0 then 45 :: dec x.natAbs else dec x.toNat

/-- A line written for an in-range non-negative value reads back exactly. -/
theorem nativeLoad_dec (tbl : List (String × Kind)) (name : String) (d lo hi x : Int)
    (ht : tbl.lookup name = some (.i64 d lo hi)) (h0 : 0 ≤ x) (hlo : lo ≤ x) (hhi : x ≤ hi)
    (hm : x ≤ i64Max) : nativeLoadIn tbl name (rewriteVal x) = some x := by
  have hx : ¬ x < 0 := by omega
  have hs := string2ll_dec x.toNat (by omega)
  have hc : (Int.ofNat x.toNat) = x := Int.toNat_of_nonneg h0
  rw [hc] at hs
  simp only [nativeLoadIn, ht, rewriteVal, hx, ite_false, hs]
  simp [Option.filter, hlo, hhi]

/-- A line for a value outside the entry's range is refused. -/
theorem nativeLoad_out (tbl : List (String × Kind)) (name : String) (d lo hi x : Int)
    (ht : tbl.lookup name = some (.i64 d lo hi)) (h0 : 0 ≤ x) (hm : x ≤ i64Max)
    (hout : hi < x) : nativeLoadIn tbl name (rewriteVal x) = none := by
  have hx : ¬ x < 0 := by omega
  have hs := string2ll_dec x.toNat (by omega)
  have hc : (Int.ofNat x.toNat) = x := Int.toNat_of_nonneg h0
  rw [hc] at hs
  simp only [nativeLoadIn, ht, rewriteVal, hx, ite_false, hs]
  simp [Option.filter]; omega

theorem vkey_entry : table.lookup "VKEY_MAX_ENTITY_COUNT" = some (.i64 100000 0 i64Max) := by
  decide

theorem filt_le (o : Option Nat) (m : Nat) (x : Int)
    (h : (o.filter (fun n => decide (n ≤ m))).map Int.ofNat = some x) : x ≤ m := by
  cases o with
  | none => simp at h
  | some n =>
    by_cases hn : n ≤ m
    · simp [Option.filter, hn] at h; omega
    · simp [Option.filter, hn] at h

theorem filt_neg (o : Option Nat) (m : Nat) (x : Int)
    (h : (o.filter (fun n => decide (n ≤ m))).map (fun n => -(Int.ofNat n)) = some x) : x ≤ 0 := by
  cases o with
  | none => simp at h
  | some n =>
    by_cases hn : n ≤ m
    · simp [Option.filter, hn] at h; omega
    · simp [Option.filter, hn] at h

theorem rustParseI64_le (v : List Nat) (x : Int) (h : rustParseI64 v = some x) :
    x ≤ i64Max := by
  unfold rustParseI64 at h
  split at h
  · cases h
  · split at h
    · exact filt_le _ _ _ h
    · split at h
      · have := filt_neg _ _ _ h; unfold i64Max; omega
      · exact filt_le _ _ _ h

theorem vkey_ok_range (v : List Nat) (x : Int) (h : validate .vkeyMax v = .ok x) :
    0 ≤ x ∧ x ≤ i64Max := by
  refine ⟨vkey_ok_nonneg v x h, ?_⟩
  simp only [validate] at h
  split at h
  · rename_i y hy
    split at h
    · cases h
    · cases h; exact rustParseI64_le v _ hy
  · cases h

/-- **`GRAPH.CONFIG SET VKEY_MAX_ENTITY_COUNT v` survives `CONFIG REWRITE`**: the line
written for any accepted value is accepted on the next start and restores that value. -/
theorem vkey_rewrite_reloads_config (v : List Nat) (x : Int) (h : validate .vkeyMax v = .ok x) :
    nativeLoad "VKEY_MAX_ENTITY_COUNT" (rewriteVal x) = some x := by
  obtain ⟨h0, hm⟩ := vkey_ok_range v x h
  exact nativeLoad_dec table _ _ _ _ x vkey_entry h0 h0 hm hm

theorem rstep_runtime (fs : FS) (td tm : Int) (n : N) (v : List Nat) (n' : N) (v' : List Nat)
    (x : Int) (td' tm' : Int) (h : rstep fs td tm n v = .ok (.runtime n' v' x, td', tm')) :
    n' = n ∧ v' = v ∧ validateConfigSet (some n) v = .ok x := by
  unfold rstep at h
  split at h
  · cases h
  · rename_i a ha
    cases h
    exfalso
    revert ha
    cases n <;> simp only [validateLoadOnly, req] <;>
      cases rustParseI64 v <;> simp [Option.filter] <;> (try split) <;> simp
    all_goals (intro he; subst he; rename_i hq; split at hq <;> simp at hq)
  · split at h
    · cases h
    · split at h
      · cases h
      · cases h; exact ⟨rfl, rfl, by assumption⟩

theorem loop_runtime (fs : FS) (td tm : Int) (o w : Bool) (ps : List (List Nat × List Nat))
    (l : List LoadArg) (o' w' : Bool) (h : loop fs td tm o w ps = .ok (l, o', w'))
    (n : N) (v : List Nat) (x : Int) (hm : LoadArg.runtime n v x ∈ l) :
    validateConfigSet (some n) v = .ok x := by
  induction ps generalizing td tm o w l with
  | nil => simp [loop] at h; obtain ⟨rfl, _⟩ := h; simp at hm
  | cons p rest ih =>
    obtain ⟨f, v0⟩ := p
    simp only [loop] at h
    split at h
    · cases h
    · rename_i n0 _
      split at h
      · cases h
      · rename_i a td1 tm1 hs
        split at h
        · cases h
        · rename_i l' o1 w1 hl
          cases h
          rcases List.mem_cons.mp hm with he | hm'
          · subst he
            obtain ⟨rfl, rfl, hv⟩ := rstep_runtime fs td tm n0 v0 n v x td1 tm1 hs
            exact hv
          · exact ih _ _ _ _ _ hl hm'

/-- **A `VKEY_MAX_ENTITY_COUNT` module-load argument survives `CONFIG REWRITE`.** -/
theorem vkey_rewrite_reloads_load (fs : FS) (td tm : Int) (args : List (List Nat))
    (v : List Nat) (x : Int) (hm : LoadArg.runtime .vkeyMax v x ∈ applied (applyModuleArgs fs td tm args)) :
    nativeLoad "VKEY_MAX_ENTITY_COUNT" (rewriteVal x) = some x := by
  unfold applied applyModuleArgs at hm
  split at hm
  · rename_i l hl
    split at hl
    · cases hl
    · split at hl
      · cases hl
      · rename_i vs o w hloop
        split at hl
        · cases hl
        · cases hl
          have := loop_runtime fs td tm false false _ _ o w hloop _ _ _ hm
          simp only [validateConfigSet] at this
          cases hv : validate .vkeyMax v with
          | error e => rw [hv] at this; cases this
          | ok y =>
            rw [hv] at this; cases this
            exact vkey_rewrite_reloads_config v _ hv
  · simp at hm

/-! ## History and the open cases -/

/-- The table before #3419 (`lib.rs:103` at `289c201fe`). -/
def pre3419_table : List (String × Kind) :=
  table.map fun e => if e.1 = "VKEY_MAX_ENTITY_COUNT" then (e.1, .i64 100000 1 1073741824) else e

/-- **Bug before #3419 (#3413)**: `GRAPH.CONFIG SET` accepted `4611686000000000000` (what
FalkorDB Cloud loads with) and `0`, and the rewritten line was refused on the next start. -/
theorem pre3419_rewrite_refused :
    validate .vkeyMax (dec 4611686000000000000) = .ok 4611686000000000000 ∧
    nativeLoadIn pre3419_table "VKEY_MAX_ENTITY_COUNT" (rewriteVal 4611686000000000000) = none ∧
    validate .vkeyMax (b "0") = .ok 0 ∧
    nativeLoadIn pre3419_table "VKEY_MAX_ENTITY_COUNT" (rewriteVal 0) = none := by
  have ht : pre3419_table.lookup "VKEY_MAX_ENTITY_COUNT" = some (.i64 100000 1 1073741824) := by
    decide
  refine ⟨?_, ?_, by decide, ?_⟩
  · simp [validate, rustParseI64_dec 4611686000000000000 (by decide)]
  · have hs := string2ll_dec 4611686000000000000 (by decide)
    simp [nativeLoadIn, ht, rewriteVal, hs, Option.filter]
  · have hs := string2ll_dec 0 (by decide)
    simp [nativeLoadIn, ht, rewriteVal, hs, Option.filter]

/-- **Still open (same class as #3413, not filed)**: values the module-load arguments accept
whose rewritten line the table refuses. `CACHE_SIZE`/`THREAD_COUNT`/`NODE_CREATION_BUFFER`
are load-only (`validateLoadOnly`); `JS_HEAP_SIZE` is also settable by `GRAPH.CONFIG SET`. -/
theorem rewrite_reload_counterexamples (fs : FS) :
    validateLoadOnly fs .cacheSize (dec 1001) = some (.ok (.cacheSize 1001)) ∧
    nativeLoad "CACHE_SIZE" (rewriteVal 1001) = none ∧
    validateLoadOnly fs .threadCount (dec 1025) = some (.ok (.threadCount 1025)) ∧
    nativeLoad "THREAD_COUNT" (rewriteVal 1025) = none ∧
    validateLoadOnly fs .nodeCreationBuffer (dec 1073741825) =
      some (.ok (.nodeCreationBuffer 1073741825)) ∧
    nativeLoad "NODE_CREATION_BUFFER" (rewriteVal 1073741825) = none ∧
    validate .jsHeap (dec 4294967297) = .ok 4294967297 ∧
    nativeLoad "JS_HEAP_SIZE" (rewriteVal 4294967297) = none := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [validateLoadOnly, req, rustParseI64_dec 1001 (by decide), Option.filter]
  · exact nativeLoad_out table "CACHE_SIZE" 25 0 1000 1001 (by decide) (by decide) (by decide) (by decide)
  · simp [validateLoadOnly, req, rustParseI64_dec 1025 (by decide), Option.filter]
  · exact nativeLoad_out table "THREAD_COUNT" 0 0 1024 1025 (by decide) (by decide) (by decide) (by decide)
  · simp [validateLoadOnly, req, rustParseI64_dec 1073741825 (by decide), Option.filter]
  · exact nativeLoad_out table "NODE_CREATION_BUFFER" 16384 0 1073741824 1073741825 (by decide) (by decide) (by decide) (by decide)
  · simp [validate, rustParseI64_dec 4294967297 (by decide), jsMinSize]
  · exact nativeLoad_out table "JS_HEAP_SIZE" 268435456 0 4294967296 4294967297 (by decide) (by decide) (by decide) (by decide)

end RedisLayer.ConfigRewrite
