import RedisLayer.Allocator
import RedisLayer.Args
import RedisLayer.QueryArgsSpec
import RedisLayer.Config
import RedisLayer.LoadArgs
import RedisLayer.Compact
import RedisLayer.RoQuery
import RedisLayer.Bulk
import RedisLayer.RoundTrip
import RedisLayer.SerialProof
import RedisLayer.SerialCLayout
import RedisLayer.RespVerbose
import RedisLayer.TelemetryLoop
import RedisLayer.TelemetryFreed
import RedisLayer.ModuleInit
import RedisLayer.ConfigRewrite
import RedisLayer.QuerySession
import RedisLayer.GraphCore
import RedisLayer.Commands2
import RedisLayer.SlowLog
import RedisLayer.MultiDb
import RedisLayer.KeyName
import RedisLayer.ErrorText
import RedisLayer.RelLookup
import RedisLayer.EnvBuild

/-
# redis_layer

The Redis-facing layer (`src/`, excluding `redis_type.rs`): argument parsing, `GRAPH.CONFIG`,
reply encoding, the RO_QUERY guard, `GRAPH.BULK`, serializers, telemetry, query sessions and
the smaller command handlers, checked against C on live servers.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
#print axioms RedisLayer.Serial.decField_encFieldC
#print axioms RedisLayer.Serial.decField_encField
#print axioms RedisLayer.Serial.encPh_c_on_iff
