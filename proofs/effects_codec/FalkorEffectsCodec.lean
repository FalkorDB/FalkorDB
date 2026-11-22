/-
# Effects v3 byte codec (`graph/src/effects/**`)

Models the GRAPH.EFFECT wire format: value and record encoders/decoders, payload framing, and the `id_list` link.
See README.md for the theorems, open bugs and gaps; COVERAGE.tsv for the per-fn map.
-/
import FalkorEffectsCodec.Prim
import FalkorEffectsCodec.Value
import FalkorEffectsCodec.Records
import FalkorEffectsCodec.Cursor
import FalkorEffectsCodec.ModRs
import FalkorEffectsCodec.RecShape
import FalkorEffectsCodec.IdLink

open FalkorCodec in
#print axioms decodeV_encV
open FalkorCodec in
#print axioms decodeV_safe
open FalkorCodec in
#print axioms readRec_createNode
open FalkorCodec in
#print axioms readAll_concat
open FalkorCodec in
#print axioms readRecord_encRecord_wf
open FalkorCodec in
#print axioms capTrace_linear
open FalkorCodec in
#print axioms openPayload_maybeCompress
open FalkorCodec in
#print axioms records_concat_idList
