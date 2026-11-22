/-
# graph_persist: RDB save/load round-trip and constraint enforcement

Lean model of FalkorDB-rs persistence (v19/v20 RDB codec, multi-key virtual-key layout, deleted-id
bitmap, attribute-id filter, `redis_type.rs` save/load) and of UNIQUE/MANDATORY constraint
enforcement (`graph/src/graph/serializers/`, `src/redis_type.rs`, `graph.rs`, `pending.rs`).
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import GraphPersist.Codec
import GraphPersist.Constraint
import GraphPersist.Layout
import GraphPersist.Traits
import GraphPersist.Mvcc
import GraphPersist.Persist
import GraphPersist.RedisType
