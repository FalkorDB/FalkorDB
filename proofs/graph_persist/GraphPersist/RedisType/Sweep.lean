import GraphPersist.RedisType.Db
/-! # The stale-key sweeps (redis_type.rs)

| here | there |
| --- | --- |
| `scanKeys`, `Reply`       | `scan_keys_by_type`, one `SCAN … TYPE` reply per step, in the selected database |
| `deleteStaleGraphmeta`    | `delete_stale_graphmeta_keys`: the scan and delete, through `for_each_db` (`Db.forEachDb`) |
| `deleteStaleVirtual`      | `delete_stale_virtual_keys`: the graphmeta sweep, then each `(db, name)` of `meta_keys` deleted through `in_db` (`Db.inDb`) |

`replies d` are the `RedisModule_Call` replies (inputs) of the scan run with database `d`
selected; `delete_key` is a `RedisModule_*` effect, recorded as the `(db, name)` it deletes.

* `scanKeys_spec`, `scanKeys_stops` — what one scan collects.
* `deleteStaleGraphmeta_spec` — the sweep deletes every graphmeta key the scan of each
  non-empty database returns, in that database (#3424: before, only the selected one).
* `deleteStaleVirtual_spec` — plus every recorded meta key, in its own database.
-/
namespace GraphPersist.RedisType

variable {Nm : Type}

/-- One `RedisModule_Call("SCAN", cursor, "TYPE", t)` reply. -/
inductive Reply (Nm : Type) where
  | null
  | notArray
  | short                                        -- an array of length < 2
  | page (last : Bool) (keys : List (Option Nm)) -- `last` = the new cursor is "0"; `none` = non-UTF-8 name

/-- `scan_keys_by_type` over the successive replies. -/
def scanKeys : List (Reply Nm) → List Nm
  | [] => []
  | .null :: _ | .notArray :: _ | .short :: _ => []
  | .page last ks :: rest => ks.filterMap id ++ if last then [] else scanKeys rest

/-- **The scan collects every UTF-8 key name of every page up to the one returning cursor
"0", skips non-UTF-8 names, and stops early on a null, non-array or short reply.** -/
theorem scanKeys_spec (pages : List (List (Option Nm))) (lastKs : List (Option Nm)) (rest : List (Reply Nm)) :
    scanKeys ((pages.map fun ks => Reply.page false ks) ++ Reply.page true lastKs :: rest) =
      (pages ++ [lastKs]).flatMap fun ks => ks.filterMap id := by
  induction pages with
  | nil => simp [scanKeys]
  | cons ks pages ih => simp [scanKeys, ih]

theorem scanKeys_stops (r : Reply Nm) (rest : List (Reply Nm)) (h : r = .null ∨ r = .notArray ∨ r = .short) :
    scanKeys (r :: rest) = [] := by
  rcases h with rfl | rfl | rfl <;> rfl

/-- `delete_stale_graphmeta_keys`: `for_each_db(ctx, || { scan; delete each key })`. Returns
the keys deleted. -/
def deleteStaleGraphmeta (ok nonEmpty : Int → Bool) (fuel : Nat) (sel : Int) (replies : Int → List (Reply Nm)) :
    List (Int × Nm) :=
  (forEachDb ok nonEmpty (fun d => (scanKeys (replies d)).map (d, ·)) fuel sel).1

/-- **The graphmeta sweep deletes, in every non-empty database `0..N`, every key its scan
returns, and nothing else.** -/
theorem deleteStaleGraphmeta_spec (ok nonEmpty : Int → Bool) (N fuel : Nat) (sel : Int)
    (replies : Int → List (Reply Nm)) (hN : ∀ d : Int, ok d = true ↔ 0 ≤ d ∧ d < N) (hfuel : N ≤ fuel)
    (hsel : ok sel = true) (d : Int) (k : Nm) :
    (d, k) ∈ deleteStaleGraphmeta ok nonEmpty fuel sel replies ↔
      0 ≤ d ∧ d < N ∧ nonEmpty d = true ∧ k ∈ scanKeys (replies d) := by
  simp only [deleteStaleGraphmeta, forEachDb_spec ok nonEmpty _ N fuel sel hN hfuel hsel, List.mem_flatMap,
    List.mem_range]
  constructor
  · rintro ⟨i, hi, hk⟩
    split at hk
    · rename_i hne
      obtain ⟨k', hk', he⟩ := List.mem_map.1 hk
      simp only [Prod.mk.injEq] at he; obtain ⟨rfl, rfl⟩ := he
      exact ⟨by omega, by omega, hne, hk'⟩
    · simp at hk
  · rintro ⟨h0, hN', hne, hk⟩
    refine ⟨d.toNat, by omega, ?_⟩
    have hd : ((d.toNat : Nat) : Int) = d := Int.toNat_of_nonneg h0
    rw [hd]; simp only [hne, ite_true]
    exact List.mem_map.2 ⟨k, hk, rfl⟩

/-- `delete_stale_virtual_keys`: the graphmeta sweep, then `mem::take(meta_keys)`, each
`(db, key)` deleted through `in_db(ctx, db, …)`; `meta_keys` is left empty. -/
def deleteStaleVirtual (ok nonEmpty : Int → Bool) (fuel : Nat) (sel : Int) (replies : Int → List (Reply Nm))
    (metaKeys : List (Int × Nm)) : List (Int × Nm) × List (Int × Nm) :=
  (deleteStaleGraphmeta ok nonEmpty fuel sel replies ++
    metaKeys.flatMap (fun p => ((inDb ok sel p.1 fun d => [(d, p.2)]).1).getD []), [])

/-- **The end of a load deletes the stale graphmeta keys and every recorded virtual key, in
the database it was loaded from** (a meta key's database exists: `io_db` read it). -/
theorem deleteStaleVirtual_spec (ok nonEmpty : Int → Bool) (fuel : Nat) (sel : Int)
    (replies : Int → List (Reply Nm)) (metaKeys : List (Int × Nm)) (d : Int) (k : Nm) :
    ((d, k) ∈ (deleteStaleVirtual ok nonEmpty fuel sel replies metaKeys).1 ↔
      (d, k) ∈ deleteStaleGraphmeta ok nonEmpty fuel sel replies ∨ ((d, k) ∈ metaKeys ∧ ok d = true)) ∧
    (deleteStaleVirtual ok nonEmpty fuel sel replies metaKeys).2 = [] := by
  refine ⟨?_, rfl⟩
  simp only [deleteStaleVirtual, List.mem_append, List.mem_flatMap, inDb_fst]
  apply or_congr_right
  constructor
  · rintro ⟨⟨d', k'⟩, hp, hk⟩
    by_cases h : ok d'
    · simp only [h, ite_true, Option.getD_some, List.mem_singleton, Prod.mk.injEq] at hk
      obtain ⟨rfl, rfl⟩ := hk; exact ⟨hp, h⟩
    · simp [h] at hk
  · rintro ⟨hp, h⟩
    exact ⟨(d, k), hp, by simp [h]⟩

end GraphPersist.RedisType
