import GraphPersist.RedisType.AuxData
import GraphPersist.RedisType.Uuid
import GraphPersist.RedisType.VKeys
import GraphPersist.RedisType.Load
/-! `src/redis_type.rs` — see the module headers: `AuxData` (aux fields, persistence
events, fork hook, graphmeta stubs), `Uuid` (virtual-key names), `VKeys` (virtual keys on
save, `graph_rdb_save`, the stale-key sweeps), `Load` (`graph_rdb_load`, placeholders,
`finalize_pending_graphs`). -/
