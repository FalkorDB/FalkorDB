/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "effects_bytes.h"
#include "effects_v3.h"
#include "effects_v3_id_list.h"

// groups a query's mutations into v3 records as they arrive
//
// A v3 payload is one record per (opcode, shape), so nothing can be serialized
// until the query stops producing effects: a record states its count and its
// shape ahead of its rows. This is the accumulator that makes that possible -
// entities are filed into their group on arrival, and the records are emitted
// once at the end.
//
// FOUR RULES LIVE HERE, and each is a byte-for-byte divergence if missed. None
// of them is visible in the wire format, which is why the conformance corpus
// cannot arbitrate any of them: a fixture pins what one record looks like, not
// which entities ended up inside it.
//
//   1. ONE RECORD PER (opcode, shape). Two entities of the same shape share a
//      record; two that differ in any component of it do not. Shape is the
//      attribute IDS and never the values, plus the labels for a node and the
//      relationship type for an edge.
//
//   2. GROUPS ARE SORTED BY KEY before emission. A hash table iterates
//      arbitrarily, so the same query could emit its records in a different
//      order on two runs of the same engine, let alone on two engines. That
//      alone defeats the byte-for-byte comparison everything else rests on.
//
//   3. LABEL SETS ARE NORMALISED ASCENDING. Callers keep their own order, so
//      [7,8] and [8,7] arrive for the same set. They are ONE shape, and the
//      ascending form is what makes two engines that agree on the set agree on
//      the bytes.
//
//   4. A NEW ATTRIBUTE IS ANNOUNCED ONCE per payload, not once per group that
//      uses it. This is the only rule that spans groups, so it is the one a
//      per-group implementation satisfies locally and violates globally.
typedef struct EffectsV3Grouping EffectsV3Grouping;

// create an accumulator
EffectsV3Grouping *EffectsV3Grouping_New(void);

// free it and everything it holds
void EffectsV3Grouping_Free
(
	EffectsV3Grouping *g  // accumulator
);

// announce a schema, at most once per (type, id)
//
// emitted ahead of every record that references the ids it introduces, because
// a bulk record carries a bare id and the replica has to have seen the name
void EffectsV3Grouping_AddSchema
(
	EffectsV3Grouping *g,  // accumulator
	SchemaType t,          // node or edge
	int id,                // the id being introduced
	const char *name       // its name
);

// announce an attribute, at most ONCE per payload
//
// v3's ADD_ATTRIBUTE carries no node/relationship discriminator - correctly,
// since C has a single dictionary - so announcing per entity kind would
// introduce the same id twice
void EffectsV3Grouping_AddAttribute
(
	EffectsV3Grouping *g,  // accumulator
	AttributeID id,        // the attribute id
	const char *name       // its name
);

// file a node-shaped entity into its group
//
// 'labels' may arrive in any order; it is normalised before it becomes a key.
// 'values' is n_attrs values in the order 'attr_ids' states, and is copied
void EffectsV3Grouping_AddNode
(
	EffectsV3Grouping *g,        // accumulator
	EffectType opcode,           // CREATE_NODE, UPDATE_NODE, DELETE_NODE,
	                             // SET_LABELS or REMOVE_LABELS
	const LabelID *labels,       // the node's labels, any order
	uint16_t n_labels,           // how many
	uint64_t id,                 // the entity id
	const AttributeID *attr_ids, // attribute ids, ascending; NULL if none
	const SIValue *values,       // one value per attribute id; NULL if none
	uint16_t n_attrs             // how many attributes
);

// file an edge-shaped entity into its group
//
// src and dst are ignored for UPDATE_EDGE, which does not carry endpoints
void EffectsV3Grouping_AddEdge
(
	EffectsV3Grouping *g,        // accumulator
	EffectType opcode,           // CREATE_EDGE, UPDATE_EDGE or DELETE_EDGE
	RelationID relation_id,      // the relationship type
	uint64_t id,                 // the edge id
	uint64_t src,                // source node id
	uint64_t dst,                // destination node id
	const AttributeID *attr_ids, // attribute ids, ascending; NULL if none
	const SIValue *values,       // one value per attribute id; NULL if none
	uint16_t n_attrs             // how many attributes
);

// how many records the accumulator would emit
uint32_t EffectsV3Grouping_RecordCount
(
	const EffectsV3Grouping *g  // accumulator
);

// write every record, schema and attribute announcements first, then the
// groups in key order
void EffectsV3Grouping_Encode
(
	EffectsV3Grouping *g,  // accumulator
	EffectsBytes *out      // sink
);
