import GraphPersist.Persist.RoundTrip
import GraphPersist.Persist.MultiLoad3
import GraphPersist.Persist.Indexes
/-!

# `src/serializers/{encoder,decoder}/mod.rs`: `decode ∘ encode = id` on the graph state

Source: origin/main `3fec7d7c9`.

The v19 payload of one graph is a sequence of tagged words (`Tok`, from `Codec`): header,
schema, payload directory, then each payload. This file models the encoder
(`encode_graph`, `build_payloads`, `build_multi_key_payloads`) and the three decoders
(`rdb_load_graph`, `load_graph_from_reader` behind `pipe_load_graph`/`vec_load_graph`,
`decode_payloads_into_pending` + `finalize_pending_graph`) and proves, compositionally:

* `RT d toks a` — decoder `d` reads `toks ++ r` back as `a` leaving `r`, **and** on every
  strict prefix of `toks` it runs out of words (`Out.eof`) rather than erring or
  succeeding. `RT_bind` / `RT_rep` compose it, so the whole-graph theorem `load_encode`
  gives both halves at once.
* `load_encode` — `load_graph_from_reader(encode_graph(G))` returns exactly the arguments
  `Graph::restore` is built from for `G` (counts, deleted-id sets, node/edge attribute
  stores, label matrices, relation tensors, adjacency and label matrices, the schema's
  labels/types/attribute dictionary, indexes and constraints).
* `truncated_payload_eof` — every truncated payload (cut at a word boundary) makes the
  decoder ask for one more word. On `vec_load_graph` that read is `BufferedReader::
  from_slice`'s `load_chunk` = `RedisModule_LoadStringBuffer(NULL)` (redis_layer
  `from_slice_past_end`): #2537 for *every* such cut, not just the empty payload.
* `multi_key_tiles` / `multi_key_bound` / `multi_key_matrices` — `build_multi_key_payloads`
  splits each entity kind into consecutive slices that tile `[0, total)` in order, no key
  holds more than `vkey_max` entities, key 0 holds every matrix.
* `pending_any_order` — the multi-key decode accumulates the same pending graph whatever
  order Redis hands the keys over in, and that graph is the single-key one.
* `rebuild_indexes_spec` — one `create_index_sync` per (attribute, field) in `field_order`,
  language/stopwords carried by the first full-text field only.

Sub-codecs proved elsewhere enter as the fields of `Codecs` (laws, not axioms): header and
schema (`redis_layer` `decH_encH`, `decSchema_encSchema`), GraphBLAS matrix/tensor
(AXIOMATISED GraphBLAS serialisation). Scalar values use `Codec.Val` with the `Out`-style
reader `decV` proved here (`decV_rt`); list/vector values are a parameter `VGood`.
-/
