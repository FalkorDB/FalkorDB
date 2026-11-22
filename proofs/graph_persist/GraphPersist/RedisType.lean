import GraphPersist.RedisType.AuxData
import GraphPersist.RedisType.Uuid
import GraphPersist.RedisType.Db
import GraphPersist.RedisType.VKeys
import GraphPersist.RedisType.Sweep
import GraphPersist.RedisType.Load
import GraphPersist.RedisType.WrongType
/-! `src/redis_type.rs` — see the module headers: `AuxData` (aux fields, persistence
events, fork hook, graphmeta stubs), `Uuid` (virtual-key names), `Db` (`io_db`, `in_db`,
`for_each_db`, `RestoreDb`), `VKeys` (virtual keys on save, `graph_rdb_save`,
`save_key_slice`, keyed by `(db, name)`), `Sweep` (the stale-key sweeps), `Load` (`graph_rdb_load`, placeholders,
`finalize_pending_graphs`), `WrongType` (`wrong_type_compat`). -/
